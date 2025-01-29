import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import datasets
from torch.utils.data.dataset import Subset

from dataloaders.ed_loader import EdgeDepth
from dataloaders.canon_dp_loader import CanonDualPixel
from dataloaders.canon_dp1169_loader import CanonDualPixel1169
import criteria
from iteration import iterate
from args import parser
import utils.helper as helper

from model.model import GT, Through

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def torch_fix_seed(seed=0, cuda=False):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

criterion_mapping = {
    'l2': criteria.MaskedMSELoss,
    'l1': criteria.MaskedL1Loss,
    'l1l2': criteria.MaskedL1L2Loss,
    'l2c': criteria.UncertaintyL2Loss,
    'l1c': criteria.UncertaintyL1Loss,
    'l1l2c': criteria.UncertaintyL1L2Loss
}

def select_backbone(args, device, conf=False):
    if args.network_variant == 'gt':
        model = GT(args).to(device)
    elif args.network_variant == 'through':
        model = Through(args).to(device)
    elif args.network_variant == 'nlspn':
        from model.model_nlspn import NLSPNModelConf, NLSPNModel
        model = NLSPNModelConf(args).to(device) if conf else NLSPNModel(args).to(device)
    elif args.network_variant == 'costdcnet':
        from model.model_costdcnet import CostDCNetConf, CostDCNet
        model = CostDCNetConf(args).to(device) if conf else CostDCNet(args).to(device)
    else:
        print('Not supported model')
        exit(-1)
    return model

def load_checkpoint(filepath, device):
    if os.path.isfile(filepath):
        print(f"=> loading checkpoint '{filepath}' ... ", end='')
        checkpoint = torch.load(filepath, map_location=device)
        print("Completed.")
        return checkpoint
    else:
        print(f"No model found at '{filepath}'")
        return None

def create_data_loader(dataset, args, shuffle, batch_size=1):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker if args.seed != -1 else None,
        generator=torch.Generator().manual_seed(args.seed) if args.seed != -1 else None
    )

def main():
    args = parser()
    print(args)
    
    if args.data_type in ['cdp', 'cdp1169']:
        import utils.logger_dp as logger_module
        import metrics_dp as metrics
    else:
        import utils.logger as logger_module
        import metrics as metrics

    data_type_mapping = {
        'ed': EdgeDepth,
        'cdp': CanonDualPixel,
        'cdp1169': CanonDualPixel1169
    }
    try:
        Dataset = data_type_mapping[args.data_type]
    except KeyError:
        raise ValueError(f"Unsupported data type: {args.data_type}")

    cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device(f"cuda:{0 if args.gpu < 0 else args.gpu}" if cuda else "cpu")
    print(f"=> using '{device}' for computation.")

    if args.seed != -1:
        torch_fix_seed(args.seed, cuda)

    depth_criterion_class = criterion_mapping.get(args.criterion)
    if depth_criterion_class is None:
        print('Not supported criteria')
        exit(-1)
    depth_criterion = depth_criterion_class()

    checkpoint = None
    is_eval = False

    if args.test or args.test_with_gt:
        test_dataset = Dataset('test', args)
        img_size = test_dataset.__size__()
    else:
        val_dataset = Dataset('val', args)
        img_size = val_dataset.__size__()
    if not args.crop:
        args.val_h = img_size[0]
        args.val_w = img_size[1]

    if args.autoresume or args.bestresume:
        output_directory = helper.get_folder_name(args)
        args.resume = helper.search_checkpoint_latest(output_directory) if args.autoresume else helper.search_checkpoint_best(output_directory)
        print('Resume from:', args.resume)

    if args.evaluate:
        checkpoint = load_checkpoint(args.evaluate, device)
        if checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            is_eval = True
    elif args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        if checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            if args.autoresume or args.bestresume:
                args.start_epoch_bias = args.start_epoch
        else:
            return

    print("=> creating model and optimizer ... ", end='')
    model = select_backbone(args, device, conf=(args.network_model == 'c'))

    if checkpoint:
        model.load_state_dict(checkpoint['model'][0] if type(checkpoint['model']) is tuple else checkpoint['model'], strict=False)
        print("=> checkpoint state loaded.")

    logger = logger_module.logger(args)
    if checkpoint:
        logger.best_result = checkpoint['best_result']
    logger.save_args_txt()
    print("=> logger created.")

    if args.test or args.test_with_gt:
        test_loader = create_data_loader(test_dataset, args, shuffle=False)
        for p in model.parameters():
            p.requires_grad = False
        iterate("test", args, test_loader, model, None, logger, metrics, 0, depth_criterion, device)
        return

    if args.small:
        n_samples = len(val_dataset)
        small_size = int(n_samples * args.small_rate)
        subset_indices = list(range(0, small_size))
        val_dataset = Subset(val_dataset, subset_indices) if subset_indices else torch.utils.data.random_split(val_dataset, [small_size, n_samples - small_size])[0]

    val_loader = create_data_loader(val_dataset, args, shuffle=False)
    print(f"\t==> val_loader size:{len(val_loader)}")

    if args.vis_skip == -1:
        args.vis_skip = int(len(val_dataset) / 8) + 1

    if is_eval:
        for p in model.parameters():
            p.requires_grad = False
        iterate("eval", args, val_loader, model, None, logger, metrics, args.start_epoch - 1, depth_criterion, device)
        return

    model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    if checkpoint and args.optimizer_load:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("completed.")

    if args.gpu < 0:
        model = torch.nn.DataParallel(model)

    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = Dataset('train', args)
        if args.small:
            n_samples = len(train_dataset)
            small_size = int(n_samples * args.small_rate)
            train_dataset = torch.utils.data.random_split(train_dataset, [small_size, n_samples - small_size])[0]
        elif args.train_num > 0:
            n_samples = len(train_dataset)
            subset_indices = list(range(0, args.train_num)) if not args.train_random else None
            train_dataset = Subset(train_dataset, subset_indices) if subset_indices else torch.utils.data.random_split(train_dataset, [args.train_num, n_samples - args.train_num])[0]

        train_loader = create_data_loader(train_dataset, args, shuffle=True, batch_size=args.batch_size)
        print(f"\t==> train_loader size:{len(train_loader)}")

    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print(f"=> starting training epoch {epoch} ..")
        iterate("train", args, train_loader, model, optimizer, logger, metrics, epoch, depth_criterion, device)

        for p in model.parameters():
            p.requires_grad = False
        result, is_best = iterate("val", args, val_loader, model, None, logger, metrics, epoch, depth_criterion, device)
        for p in model.parameters():
            p.requires_grad = True

        save_model = model.module.state_dict() if args.gpu < 0 else model.state_dict()
        helper.save_checkpoint({'epoch': epoch, 'model': save_model, 'best_result': logger.best_result, 'optimizer': optimizer.state_dict(), 'args': args}, is_best, epoch, logger.output_directory)

if __name__ == '__main__':
    main()