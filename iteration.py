import torch
import time
from utils.util import MAX_8BIT, rescale_image
import utils.helper as helper
from dataloaders.canon_dp_loader import CanonDualPixel
from model.wfgs import WFGS

cuda = torch.cuda.is_available()
multi_batch_size = 1

def prepare_batch_data(batch_data, device, args):
    batch_data = {key: val.to(device) for key, val in batch_data.items() if val is not None}
    if args.lowres_cnn:
        batch_data['rgb'] = rescale_image(batch_data['rgb'], args.lowres_scale, 'bilinear')
        batch_data['d'] = rescale_image(batch_data['d'], args.lowres_scale, 'nearest')
    return batch_data

def post_process_batch_data(batch_data, pred, conf, conf_inv, args):
    if args.lowres_cnn:
        up_scale = 1.0 / args.lowres_scale
        batch_data['rgb'] = rescale_image(batch_data['rgb'], up_scale, 'bilinear')
        batch_data['d'] = rescale_image(batch_data['d'], up_scale, 'nearest')
        if isinstance(pred, list):
            for p in range(len(pred)):
                pred[p] = rescale_image(pred[p], up_scale, 'bilinear')
        else:
            pred = rescale_image(pred, up_scale, 'bilinear')
        if conf is not None:
            conf = rescale_image(conf, up_scale, 'bilinear')
            if conf_inv is not None:
                conf_inv = rescale_image(conf_inv, up_scale, 'bilinear')
    if args.post_process and args.data_type == 'cdp':
        batch_data, pred, conf = CanonDualPixel.post_process(batch_data, pred, conf)
    return batch_data, pred, conf, conf_inv

def process_prediction(args, model, batch_data, refine_model):
    if args.network_model == 'n':
        pred = model(batch_data)
        if args.post_refine:
            tmp = batch_data['d']
            batch_data['d'] = pred
            pred = refine_model(batch_data, None)
            batch_data['d'] = tmp
    else:
        pred, conf_inv = model(batch_data)
        conf = 1.0 - conf_inv
        conf = torch.clamp(conf, min=args.confmin, max=args.confmax)
        if args.post_refine:
            tmp = batch_data['d']
            batch_data['d'] = pred
            pred = refine_model(batch_data, conf.to('cpu').detach().numpy().copy())
            batch_data['d'] = tmp
        conf = (conf - args.confmin) / (args.confmax - args.confmin) * MAX_8BIT
    return pred, conf, conf_inv

def compute_loss(args, pred, gt, conf_inv, depth_criterion):
    if args.network_model == 'n':
        return depth_criterion(pred, gt)
    elif args.network_model == 'c':
        return depth_criterion(pred, gt, conf_inv, args.conf_lambda)
    else:
        print('Not supported network model:', args.network_model)
        exit(-1)

def iterate(mode, args, loader, model, optimizer, logger, metrics, epoch, depth_criterion, device):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias
    block_average_meter = metrics.AverageMeter(args)
    block_average_meter.reset(False)
    average_meter = metrics.AverageMeter(args)
    meters = [block_average_meter, average_meter]

    if args.post_refine == 'wfgs':
        refine_model = WFGS(args).to(device)
    else:
        refine_model = None

    assert mode in ["train", "val", "eval", "test"], "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        if mode != 'train': print('Data:', i)
        if args.select_data_num != -1 and i != args.select_data_num: continue
        if cuda and mode != 'train': torch.cuda.synchronize()
        dstart = time.time()
        
        batch_data = prepare_batch_data(batch_data, device, args)
        gt = batch_data['gt'] if args.test_with_gt or mode != 'test' else None
        if cuda and mode != 'train': torch.cuda.synchronize()
        data_time = time.time() - dstart

        pred, conf, conf_inv = None, None, None
        if cuda and mode != 'train': torch.cuda.synchronize()
        start = time.time()

        pred, conf, conf_inv = process_prediction(args, model, batch_data, refine_model)
        batch_data, pred, conf, conf_inv = post_process_batch_data(batch_data, pred, conf, conf_inv, args)
        if cuda and mode != 'train': torch.cuda.synchronize()
        gpu_time = time.time() - start

        loss = 0, 0, 0
        if mode == 'train':
            loss = compute_loss(args, pred, gt, conf_inv, depth_criterion)

            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            loss.backward()
            if i % multi_batch_size == (multi_batch_size-1) or i == (len(loader)-1):
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))

        if isinstance(pred, list):
            pred = pred[-1]

        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = metrics.Result(args)
            if args.test_with_gt or mode != 'test':
                result.evaluate(pred.data, batch_data, conf_inv)
                [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
                if args.eval_each:
                    logger.conditional_save_info(mode, block_average_meter, i)
                    block_average_meter.reset(False)
                else:
                    logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)

        if args.network_model == 'c':
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch, extra=conf, skip=args.vis_skip)
            logger.conditional_save_img(mode, i, batch_data, pred, epoch, extra=conf)
        else:
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch, skip=args.vis_skip)
            logger.conditional_save_img(mode, i, batch_data, pred, epoch, extra=None)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best:
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best