import os
import glob
import numpy as np
import torch.utils.data as data
import cv2
from dataloaders import transforms
from utils.util import MAX_8BIT, MAX_16BIT
import subprocess
import time

def get_globs(split, data_folder):
    globs = {
        'dgt': os.path.join(data_folder, split, 'dgt/*.png'),
        'rgbL': os.path.join(data_folder, split, 'G_matB-png/*.png'),
        'rgbR': os.path.join(data_folder, split, 'G_matA-png/*.png')
    }
    return globs

def get_paths_from_globs(globs):
    paths = {key: sorted(glob.glob(path)) for key, path in globs.items()}
    return paths

def add_paths_from_globs(paths, globs):
    for key in paths:
        paths[key] += sorted(glob.glob(globs[key]))
    return paths

def print_paths_length(paths):
    for key in paths:
        print(f'{key}:{len(paths[key])}')

def get_paths_and_transform(split, args):
    transform = {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }.get(split, None)
    
    if transform is None:
        raise ValueError("Unrecognized split " + str(split))

    globs = get_globs(split, args.data_folder)
    paths = get_paths_from_globs(globs)
    print_paths_length(paths)

    if split in ['train', 'val'] or args.test_with_gt:
        if any(len(paths[key]) == 0 for key in paths):
            print_paths_length(paths)
            raise RuntimeError("Found 0 images.")
        if not all(len(paths[key]) == len(paths['dgt']) for key in paths):
            print_paths_length(paths)
            raise RuntimeError("The number of images is different.")
    else:
        paths['dgt'] = [None] * len(paths['rgbL'])
        if len(paths['rgbL']) == 0 or len(paths['rgbR']) == 0:
            print_paths_length(paths)
            raise RuntimeError("Found 0 images.")
        if len(paths['rgbL']) != len(paths['rgbR']):
            print_paths_length(paths)
            raise RuntimeError("The number of images is different.")

    return paths, transform

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    rgb = cv2.imread(filename, -1)
    if rgb is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))
    assert rgb.shape[:2] == (779, 1169)
    rgb = rgb.astype(np.float32) / (MAX_16BIT if rgb.dtype == 'uint16' else MAX_8BIT)
    rgb *= MAX_8BIT
    return rgb

def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    depth = cv2.imread(filename, -1)
    if depth is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))
    assert depth.shape[:2] == (779, 1169)
    depth = depth[:, :, np.newaxis].astype(np.float32) / MAX_16BIT * MAX_8BIT
    return np.clip(depth, 0, MAX_8BIT)

def train_transform(rgb, phase, target, args):
    oheight, owidth = args.val_h, args.val_w
    do_flip = np.random.uniform(0.0, 1.0) < 0.5

    transform_geometric = transforms.Compose([
        transforms.Resize(rgb.shape[:2]),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])

    if phase is not None:
        phase = transform_geometric(phase)
    if target is not None:
        target = transform_geometric(target)
    if rgb is not None:
        transform_rgb = transforms.Compose([
            transform_geometric
        ])
        rgb = transform_rgb(rgb)

    if args.random_crop:
        h, w = oheight, owidth
        rheight, rwidth = args.random_crop_height, args.random_crop_width
        i, j = np.random.randint(0, h - rheight + 1), np.random.randint(0, w - rwidth + 1)

        if rgb is not None:
            rgb = rgb[i:i + rheight, j:j + rwidth, ...]
        if phase is not None:
            phase = phase[i:i + rheight, j:j + rwidth, ...]
        if target is not None:
            target = target[i:i + rheight, j:j + rwidth, ...]

    return rgb, phase, target

def val_transform(rgb, phase, target, args):
    transform = transforms.Compose([
        transforms.Resize(rgb.shape[:2])
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if phase is not None:
        phase = transform(phase)
    if target is not None:
        target = transform(target)
    return rgb, phase, target

def no_transform(rgb, phase, target, args):
    return rgb, phase, target

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def resize_imgs(rgbL, rgbR, dgt, phase, size):
    if rgbL is not None:
        rgbL = cv2.resize(rgbL, dsize=size, interpolation=cv2.INTER_LINEAR)
    if rgbR is not None:
        rgbR = cv2.resize(rgbR, dsize=size, interpolation=cv2.INTER_LINEAR)
    if dgt is not None:
        dgt = cv2.resize(dgt, dsize=size, interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    if phase is not None:
        phase = cv2.resize(phase, dsize=size, interpolation=cv2.INTER_NEAREST)
    return rgbL, rgbR, dgt, phase

def DPMatching(imgL, imgR, pixShift=21, refArea=19, use_executable=False):
    if use_executable:
        temp_imgL_path = os.path.abspath("temp_imgL.npy")
        temp_imgR_path = os.path.abspath("temp_imgR.npy")
        phase_final_path = os.path.abspath("phase_final.npy")
        
        np.save(temp_imgL_path, imgL)
        np.save(temp_imgR_path, imgR)
        
        try:
            result = subprocess.run([
                "utils/dp_matching", temp_imgL_path, temp_imgR_path,
                "--pixShift", str(pixShift),
                "--refArea", str(refArea),
            ], check=True, capture_output=True, text=True)
            
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("An error occurred while running dp_matching:")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            raise
        
        if not os.path.exists(phase_final_path):
            raise FileNotFoundError("phase_final.npy was not created.")
        
        phase_final = np.load(phase_final_path)
    else:
        from utils.dp_matching import DPMatching
        phase_final = DPMatching(imgL, imgR, pixShift=pixShift, refArea=refArea)

    return phase_final

class CanonDualPixel1169(data.Dataset):
    """A data loader for the Canon dual-pixel dataset"""
    def __init__(self, split, args):
        self.args = args
        self.split = split
        self.paths, self.transform = get_paths_and_transform(split, args)

    def __getraw__(self, index):
        rgbL = rgb_read(self.paths['rgbL'][index]) if self.paths['rgbL'][index] is not None else None
        rgbR = rgb_read(self.paths['rgbR'][index]) if self.paths['rgbR'][index] is not None else None
        dgt = depth_read(self.paths['dgt'][index]) if self.paths['dgt'][index] is not None else None
        return rgbL, rgbR, dgt

    def __getitem__(self, index):
        rgbL, rgbR, dgt = self.__getraw__(index)
        if self.args.lowres_phase:
            pscale = self.args.lowres_pscale
            rgbL, rgbR, _, _ = resize_imgs(rgbL, rgbR, None, None, (int(1169 * pscale), int(779 * pscale)))
        start = time.time()
        phase = DPMatching(rgbL, rgbR, self.args.pix_shift, self.args.ref_area, self.args.use_executable)
        if self.args.lowres_phase:
            rgbL, rgbR, _, phase = resize_imgs(rgbL, rgbR, None, phase / pscale, (1169, 779))
        matching_time = time.time() - start

        mgn = 50
        phase[:, :mgn] = -255
        phase[:mgn, :] = -255
        phase[:, -mgn:] = -255
        phase[-mgn:, :] = -255

        phase = (phase - self.args.phase_min) / (self.args.phase_max - self.args.phase_min)
        phase = np.clip(phase, 0, 1) * MAX_8BIT
        phase = phase[:, :, np.newaxis]
        rgbL, phase, dgt = self.transform(rgbL, phase, dgt, self.args)

        candidates = {'rgb': rgbL, 'd': phase, 'gt': dgt}
        items = {key: to_float_tensor(val) for key, val in candidates.items() if val is not None}
        return items

    def __len__(self):
        return len(self.paths['rgbL'])

    def __size__(self):
        rgbL = rgb_read(self.paths['rgbL'][0]) if self.paths['rgbL'][0] is not None else None
        return rgbL.shape[:2]