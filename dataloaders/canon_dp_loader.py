import os
import os.path
import glob
import numpy as np
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from utils.util import MAX_8BIT, MAX_16BIT
import subprocess
import time

def get_globs(split, data_folder, lowres=False):
    globs = {}
    if lowres:
        globs['dgt'] = os.path.join(data_folder, split, '*_dgt.png')
        globs['rgbL'] = os.path.join(data_folder, split, '*_rgbL.png')
        globs['rgbR'] = os.path.join(data_folder, split, '*_rgbR.png')
    else:
        globs['dgt'] = os.path.join(data_folder, split, '*_D.TIF')
        globs['rgbL'] = os.path.join(data_folder, split, '*_L.jpg')
        globs['rgbR'] = os.path.join(data_folder, split, '*_R.jpg')
    globs['p'] = os.path.join(data_folder, split, '*pedge*.png')
    return globs

def get_paths_from_globs(globs):
    paths = {key: sorted(glob.glob(globs[key])) for key in globs}
    return paths

def add_paths_from_globs(paths, globs):
    for key in globs:
        paths[key] += sorted(glob.glob(globs[key]))
    return paths

def print_paths_length(paths):
    for key in paths:
        print('{}:{}'.format(key, len(paths[key])))

def check_paths(paths, keys):
    for key in keys:
        if len(paths[key]) == 0:
            print_paths_length(paths)
            raise RuntimeError("Found 0 images.")
    if not all(len(paths[key]) == len(paths[keys[0]]) for key in keys):
        print_paths_length(paths)
        raise RuntimeError("The number of images is different.")

def get_paths_and_transform(split, args):
    if split == 'train':
        transform = train_transform
    elif split in ['val', 'test']:
        transform = val_transform
    else:
        raise ValueError("Unrecognized split " + str(split))

    globs = get_globs(split, args.data_folder, args.lowres_input)
    paths = get_paths_from_globs(globs)
    print_paths_length(paths)

    if split in ['train', 'val'] or args.test_with_gt:
        if args.pre_matching:
            check_paths(paths, ['dgt', 'rgbL', 'rgbR', 'p'])
        else:
            paths['p'] = [None] * len(paths['rgbL'])
            check_paths(paths, ['dgt', 'rgbL', 'rgbR'])
    else:
        paths['dgt'] = [None] * len(paths['rgbL'])
        if args.pre_matching:
            check_paths(paths, ['rgbL', 'rgbR', 'p'])
        else:
            paths['p'] = [None] * len(paths['rgbL'])
            check_paths(paths, ['rgbL', 'rgbR'])

    return paths, transform

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    rgb = cv2.imread(filename, -1)
    if rgb is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))
    assert rgb.shape[:2] == (2940, 5180) or rgb.shape[:2] == (735, 1295)
    if rgb.dtype == 'uint16':
        rgb = rgb.astype(dtype=np.float32) / MAX_16BIT
    else:
        rgb = rgb.astype(dtype=np.float32) / MAX_8BIT
    rgb = rgb * MAX_8BIT
    return rgb

def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    depth = cv2.imread(filename, -1)
    if depth is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))
    assert depth.shape[:2] == (2940, 5180) or depth.shape[:2] == (735, 1295)
    valid_mask = depth > 0
    depth = np.clip(MAX_8BIT - depth, 0, MAX_8BIT)
    depth = depth * valid_mask
    depth = depth[:, :, np.newaxis]
    return depth

def phase_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    phase = cv2.imread(filename, -1)
    if phase is None:
        raise FileNotFoundError("{} not found.".format(str(filename)))
    assert phase.shape[:2] == (735, 1295)
    if phase.dtype == 'uint16':
        phase = phase.astype(dtype=np.float32) / MAX_16BIT
    else:
        phase = phase.astype(dtype=np.float32) / MAX_8BIT
    return phase

def train_transform(rgb, phase, target, args):
    oheight = args.val_h
    owidth = args.val_w
    do_flip = np.random.uniform(0.0, 1.0) < 0.5

    transforms_list = [
        transforms.Resize(rgb.shape[:2]),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ]

    transform_geometric = transforms.Compose(transforms_list)

    if phase is not None:
        phase = transform_geometric(phase)
    if target is not None:
        target = transform_geometric(target)
    if rgb is not None:
        transform_rgb = transforms.Compose([transform_geometric])
        rgb = transform_rgb(rgb)

    if args.random_crop:
        h, w = oheight, owidth
        rheight, rwidth = args.random_crop_height, args.random_crop_width
        i = np.random.randint(0, h - rheight + 1)
        j = np.random.randint(0, w - rwidth + 1)
        if rgb is not None:
            rgb = rgb[i:i + rheight, j:j + rwidth, ...]
        if phase is not None:
            phase = phase[i:i + rheight, j:j + rwidth, ...]
        if target is not None:
            target = target[i:i + rheight, j:j + rwidth, ...]

    return rgb, phase, target

def val_transform(rgb, phase, target, args):
    transform = transforms.Compose([transforms.Resize(rgb.shape[:2])])
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

def crop_imgs(rgbL, rgbR, dgt, phase, sz, mgn):
    rgbL = rgbL[mgn[0]: mgn[0] + sz[0], mgn[1]: mgn[1] + sz[1], ...]
    rgbR = rgbR[mgn[0]: mgn[0] + sz[0], mgn[1]: mgn[1] + sz[1], ...]
    dgt = dgt[mgn[0]: mgn[0] + sz[0], mgn[1]: mgn[1] + sz[1], ...]
    phase = phase[mgn[0]: mgn[0] + sz[0], mgn[1]: mgn[1] + sz[1], ...]
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

class CanonDualPixel(data.Dataset):
    """A data loader for the Canon dual-pixel dataset"""
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform

    def __getraw__(self, index):
        rgbL = rgb_read(self.paths['rgbL'][index]) if self.paths['rgbL'][index] is not None else None
        rgbR = rgb_read(self.paths['rgbR'][index]) if self.paths['rgbR'][index] is not None else None
        dgt = depth_read(self.paths['dgt'][index]) if self.paths['dgt'][index] is not None else None
        phase_lowres = phase_read(self.paths['p'][index]) if self.paths['p'][index] is not None else None
        if phase_lowres is not None:
            phase_lowres = phase_lowres * (self.args.phase_max - self.args.phase_min) + self.args.phase_min
        return rgbL, rgbR, dgt, phase_lowres

    def __getitem__(self, index):
        if self.args.select_data_num != -1 and index != self.args.select_data_num:
            return []
        rgbL, rgbR, dgt, phase_lowres = self.__getraw__(index)
        if not self.args.lowres_input:
            rgbL, rgbR, dgt, _ = resize_imgs(rgbL, rgbR, dgt, None, (2590, 1470))
        if self.args.lowres_phase:
            pscale = self.args.lowres_pscale
            rgbL, rgbR, _, _ = resize_imgs(rgbL, rgbR, None, None, (int(2590 * pscale), int(1470 * pscale)))
        if not self.args.pre_matching:
            start = time.time()
            phase_lowres = DPMatching(rgbL, rgbR, self.args.pix_shift, self.args.ref_area, self.args.use_executable)
            phase_lowres = phase_lowres / pscale
        if self.args.lowres_phase:
            rgbL, rgbR, _, phase = resize_imgs(rgbL, rgbR, None, phase_lowres, (2590, 1470))
        else:
            phase = phase_lowres
        if not self.args.lowres_input:
            rgbL, rgbR, dgt, phase = crop_imgs(rgbL, rgbR, dgt, phase, (1296, 2368), (63, 72))
        else:
            rgbL, rgbR, dgt, phase = crop_imgs(rgbL, rgbR, dgt, phase, (648, 1184), (32, 36))
        phase = (phase - self.args.phase_min) / (self.args.phase_max - self.args.phase_min)
        phase = np.clip(phase, 0, 1) * MAX_8BIT
        phase = phase[:, :, np.newaxis]
        rgbL, phase, dgt = self.transform(rgbL, phase, dgt, self.args)
        if self.args.output_lowres_phase:
            phase_lowres = (phase_lowres - self.args.phase_min) / (self.args.phase_max - self.args.phase_min)
            phase_lowres = np.clip(phase_lowres, 0, 1) * MAX_8BIT
        else:
            phase_lowres = None
        candidates = {'rgb': rgbL, 'd': phase, 'd_lowres': phase_lowres, 'gt': dgt}
        items = {key: to_float_tensor(val) for key, val in candidates.items() if val is not None}
        return items

    def __len__(self):
        return len(self.paths['rgbL'])

    def __size__(self):
        rgbL = rgb_read(self.paths['rgbL'][0]) if self.paths['rgbL'][0] is not None else None
        return rgbL.shape[:2]

    def post_process(batch_data, pred, conf):
        def _crop(img, patch_size=111, stride=33):
            h, w = img.shape[2:]
            m = (patch_size - 1) // 2
            mids = (stride - 1) // 2
            rowmax = torch.arange(m + 1, h - m, stride).int()
            colmax = torch.arange(m + 1, w - m, stride).int()
            cropped = img[..., rowmax[0] - mids: rowmax[-1] + mids + 1, colmax[0] - mids: colmax[-1] + mids + 1]
            return cropped

        for k, v in batch_data.items():
            if k != 'd_lowres':
                batch_data[k] = _crop(v)
        if isinstance(pred, list):
            for p in range(len(pred)):
                pred[p] = _crop(pred[p])
        else:
            pred = _crop(pred)
        if conf is not None:
            conf = _crop(conf)
        return batch_data, pred, conf