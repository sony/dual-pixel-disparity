import os
import os.path
import glob
import numpy as np
from random import choice
import torch.utils.data as data
import cv2
from dataloaders import transforms
from utils.util import MAX_8BIT, MAX_16BIT, depth_to_phase

def get_globs(split, data_folder):
    globs = {key: os.path.join(data_folder, split, f'{key}/*_{key}.png') for key in ['dedge', 'dgt', 'rgb']}
    return globs

def get_paths_from_globs(globs):
    return {key: sorted(glob.glob(path)) for key, path in globs.items()}

def add_paths_from_globs(paths, globs):
    for key in paths.keys():
        paths[key] += sorted(glob.glob(globs[key]))
    return paths

def print_paths_length(paths):
    print('dedge:{}'.format(len(paths['dedge'])))
    print('dgt:{}'.format(len(paths['dgt'])))
    print('rgb:{}'.format(len(paths['rgb'])))

def validate_paths(paths, split, args):
    if split in ['train', 'val'] or args.test_with_gt:
        if any(len(paths[key]) == 0 for key in paths.keys()):
            print_paths_length(paths)
            raise RuntimeError("Found 0 images.")
        if not all(len(paths[key]) == len(paths['rgb']) for key in paths.keys()):
            print_paths_length(paths)
            raise RuntimeError("The number of images is different.")
    else:
        paths['dgt'] = [None] * len(paths['rgb'])
        if len(paths['dedge']) == 0 or len(paths['rgb']) == 0:
            print_paths_length(paths)
            raise RuntimeError("Found 0 images.")
        if len(paths['dedge']) != len(paths['rgb']):
            print_paths_length(paths)
            raise RuntimeError("The number of images is different.")

def get_paths_and_transform(split, args):
    transform = train_transform if split == 'train' else val_transform
    globs = get_globs(split, args.data_folder)
    paths = get_paths_from_globs(globs)
    print_paths_length(paths)
    validate_paths(paths, split, args)
    return paths, transform

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = cv2.imread(filename, -1)
    if img_file is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))
    if img_file.dtype=='uint16':
        rgb = img_file.astype(dtype=np.float32) / MAX_16BIT # scale pixels to the range [0,1]
    else:
        rgb = img_file.astype(dtype=np.float32) / MAX_8BIT # scale pixels to the range [0,1]       
    rgb = rgb * MAX_8BIT # Scale to 8bit [0, 255]
    return rgb

def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    depth_png = cv2.imread(filename, -1)
    if depth_png is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))

    depth_png = depth_png[:, :, np.newaxis]

    depth = depth_png.astype(np.float) / 256.
    return depth

def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def apply_random_crop(rgb, sparse, target, args, oheight, owidth):
    if args.random_crop:
        h, w = oheight, owidth
        rheight, rwidth = args.random_crop_height, args.random_crop_width
        i, j = np.random.randint(0, h - rheight + 1), np.random.randint(0, w - rwidth + 1)
        crop = lambda x: x[i:i + rheight, j:j + rwidth] if x is not None else None
        rgb, sparse, target = map(crop, [rgb, sparse, target])
    return rgb, sparse, target

def apply_depth_to_phase(sparse, target, args):
    focus_dis = np.random.randint(500, 3000)
    f_stop = choice([1.4, 2.0, 2.8, 4.0, 5.0])
    coc_alpha = np.random.uniform(0.2, 0.4)
    sparse = depth_to_phase(sparse/MAX_8BIT, focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch,
                            args.phase_min, args.phase_max, args.original_depth_max, args.original_depth_min,
                            args.add_phase_noise, args.noise_type, args.noise_sigma_max, args.noise_sigma_gain) * MAX_8BIT
    target = depth_to_phase(target/MAX_8BIT, focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch,
                            args.phase_min, args.phase_max, args.original_depth_max, args.original_depth_min,
                            False) * MAX_8BIT
    return sparse, target

def train_transform(rgb, sparse, target, args, index):
    oheight, owidth = args.val_h, args.val_w
    do_flip = np.random.uniform(0.0, 1.0) < 0.5
    transform_geometric = transforms.Compose([
        transforms.Resize(rgb.shape[:2]),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if rgb is not None:
        rgb = transform_geometric(rgb)
    if sparse is not None:
        sparse = transform_geometric(sparse)
    if target is not None:
        target = transform_geometric(target)
    rgb, sparse, target = apply_random_crop(rgb, sparse, target, args, oheight, owidth)
    if args.depth_to_phase:
        sparse, target = apply_depth_to_phase(sparse, target, args)
    return rgb, sparse, target

def val_transform(rgb, sparse, target, args, index):
    oheight = args.val_h
    owidth = args.val_w

    transform = transforms.Compose([
        transforms.Resize(rgb.shape[:2]),
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)

    if args.depth_to_phase == True:
        focus_dis_min = args.focus_dis_min
        focus_dis_max = args.focus_dis_max
        focus_dis_step = args.focus_dis_step
        focus_dis = (index*focus_dis_step) % focus_dis_max
        focus_dis = np.clip(focus_dis, focus_dis_min, focus_dis_max)        

        f_stop_list = args.f_stop_list
        f_stop = f_stop_list[index % len(f_stop_list)]

        coc_alpha_list = [0.2, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.4]
        coc_alpha = coc_alpha_list[index % len(coc_alpha_list)]


        sparse = depth_to_phase(sparse/MAX_8BIT, focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch,
                                args.phase_min, args.phase_max, args.original_depth_max, args.original_depth_min,
                                args.add_phase_noise, args.noise_type, args.noise_sigma_max) * MAX_8BIT
        target = depth_to_phase(target/MAX_8BIT, focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch,
                                args.phase_min, args.phase_max, args.original_depth_max, args.original_depth_min,
                                False) * MAX_8BIT

    return rgb, sparse, target

def no_transform(rgb, sparse, target, args, index):
    return rgb, sparse, target


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


class EdgeDepth(data.Dataset):
    """A data loader for the EdgeDepth dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None) else None
        sparse = depth_read(self.paths['dedge'][index]) if \
            (self.paths['dedge'][index] is not None) else None
        target = depth_read(self.paths['dgt'][index]) if \
            self.paths['dgt'][index] is not None else None
        return rgb, sparse, target

    def __getitem__(self, index):
        rgb, sparse, target = self.__getraw__(index)
        rgb, sparse, target = self.transform(rgb, sparse, target, self.args, index)

        # rgb, gray = handle_gray(rgb, self.args)
        candidates = {'rgb': rgb, 'd': sparse, 'gt': target}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['rgb'])

    def __size__(self):
        rgb = rgb_read(self.paths['rgb'][0]) if self.paths['rgb'][0] is not None else None
        return rgb.shape[:2] # height, width
