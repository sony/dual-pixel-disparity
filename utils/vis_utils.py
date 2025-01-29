import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from utils.util import MAX_8BIT, MAX_16BIT, phase_to_depth_normalized
import torch
from scipy.io import savemat

cmap = plt.cm.turbo
cmap2 = plt.cm.nipy_spectral

def validcrop(img):
    ratio = 256/1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h-int(ratio*w):, :]

def preprocess_data(data, max, min, vis_max, vis_min, inv, zero_mask, colormap):
    """
    Preprocess the data for visualization.

    Args:
    data (np.array): Input data to be colorized [0-1].
    max (float): Maximum value of the input data.
    min (float): Minimum value of the input data.
    vis_max (float): Maximum value to be visualized.
    vis_min (float): Minimum value to be visualized.
    inv (bool): Whether to invert the data or not.
    zero_mask (bool): Whether to mask zero values or not.
    colormap (Colormap): Colormap to be used for visualization.

    Returns:
    np.array: Colorized data as a uint8 array.
    """
    mask = data != 0
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    data = data * (max - min) + min
    data = (data - vis_min) / (vis_max - vis_min)
    if inv:
        data = 1 - data
    data = np.clip(data, 0, 1)
    data = MAX_8BIT * colormap(data)[:, :, :3]
    if zero_mask:
        data = data * mask
    return data.astype('uint8')

def data_colorize(data, max=10.0, min=0.0, vis_max=10.0, vis_min=0.0, inv=True, zero_mask=False): 
    """
    Colorize the given data according to the specified range.

    Args:
    data (np.array): Input data to be colorized [0-1].
    max (float): Maximum value of the input data.
    min (float): Minimum value of the input data.
    vis_max (float): Maximum value to be visualized.
    vis_min (float): Minimum value to be visualized.
    inv (bool): Whether to invert the data or not.
    zero_mask (bool): Whether to mask zero values or not.

    Returns:
    np.array: Colorized data as a uint8 array.
    """
    return preprocess_data(data, max, min, vis_max, vis_min, inv, zero_mask, cmap)

def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = MAX_8BIT * cmap2(feature)[:, :, :3]
    return feature.astype('uint8')

def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = MAX_8BIT * mask
    return mask.astype('uint8')

def merge_into_row(args, ele, pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
    def preprocess_depth(x, zero_mask=False):
        y = np.squeeze(x.data.cpu().numpy())
        return data_colorize(y/MAX_8BIT, max=args.depth_max, min=args.depth_min, vis_max=args.vis_depth_max, vis_min=args.vis_depth_min, inv=args.vis_depth_inv, zero_mask=zero_mask)
    def preprocess_phase(x, zero_mask=False):
        y = np.squeeze(x.data.cpu().numpy())
        return data_colorize(y/MAX_8BIT, max=args.phase_max, min=args.phase_min, vis_max=args.vis_phase_max, vis_min=args.vis_phase_min, inv=args.vis_phase_inv, zero_mask=zero_mask)
    def preprocess_phase_to_depth(x, zero_mask=False):
        y = np.squeeze(x.data.cpu().numpy())
        focus_dis = ele['focus_dis'].data.cpu().numpy()
        f_stop = ele['f_stop'].data.cpu().numpy()
        coc_alpha = ele['coc_alpha'].data.cpu().numpy()
        y = phase_to_depth_normalized(np.clip(y/MAX_8BIT, 0, 1), focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch, args.phase_min, args.phase_max, args.depth_min, args.depth_max)
        return data_colorize(y, max=args.depth_max, min=args.depth_min, vis_max=args.vis_depth_max, vis_min=args.vis_depth_min, inv=args.vis_depth_inv, zero_mask=zero_mask)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb[...,::-1])
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    if 'd' in ele:
        if args.data_type=='edp' or 'cdp' in args.data_type:
            img_list.append(preprocess_phase(ele['d'][0, ...], zero_mask=True))
            img_list.append(preprocess_phase(pred[0, ...]))
        else:
            img_list.append(preprocess_depth(ele['d'][0, ...], zero_mask=True))
            img_list.append(preprocess_depth(pred[0, ...]))
    if extrargb is not None:
        img_list.append(preprocess_depth(extrargb[0, ...]))
    if predrgb is not None:
        predrgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        predrgb = np.transpose(predrgb, (1, 2, 0))
        #predrgb = predrgb.astype('uint8')
        img_list.append(predrgb)
    if predg is not None:
        predg = np.squeeze(predg[0, ...].data.cpu().numpy())
        predg = mask_vis(predg)
        predg = np.array(Image.fromarray(predg).convert('RGB'))
        #predg = predg.astype('uint8')
        img_list.append(predg)
    if extra is not None:
        extra = np.squeeze(extra[0, ...].data.cpu().numpy())
        extra = mask_vis(extra)
        extra = np.array(Image.fromarray(extra).convert('RGB'))
        img_list.append(extra)
    if extra2 is not None:
        extra2 = np.squeeze(extra2[0, ...].data.cpu().numpy())
        extra2 = mask_vis(extra2)
        extra2 = np.array(Image.fromarray(extra2).convert('RGB'))
        img_list.append(extra2)
    if 'd' in ele:
        if args.depth_to_phase_vis:
            img_list.append(preprocess_phase_to_depth(pred[0, ...]))
    if 'gt' in ele:
        if args.data_type=='edp':
            img_list.append(preprocess_phase(ele['gt'][0, ...]))
        else:
            img_list.append(preprocess_depth(ele['gt'][0, ...]))
    if 'd' in ele:
        if args.data_type=='edp':
            img_list.append(preprocess_phase_to_depth(pred[0, ...]))
    if 'dgt' in ele:
        img_list.append(preprocess_depth(ele['dgt'][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_mono_image(mono, filename):
    mono = np.squeeze(mono[0, ...].data.cpu().numpy())
    image_to_write = mono.astype('uint8')
    # image_to_write = np.array(Image.fromarray(mono).convert('RGB'))
    cv2.imwrite(filename, image_to_write)

def save_image_torch(rgb, filename, rgb2bgr=True):
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype('uint8')
    if rgb2bgr:
        image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        image_to_write = rgb
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * MAX_16BIT/MAX_8BIT).astype('uint16')
    cv2.imwrite(filename, img)

def save_as_mat(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    savemat(filename, {'array': img})

def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * MAX_16BIT/MAX_8BIT).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_image_generic(args, img, filename, zero_mask, colormap, max, min, vis_max, vis_min, inv):
    img = np.squeeze(img.data.cpu().numpy())
    img = preprocess_data(img/MAX_8BIT, max, min, vis_max, vis_min, inv, zero_mask, colormap)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_depth_as_uint8colored(args, img, filename, zero_mask=False):
    save_image_generic(args, img, filename, zero_mask, cmap, args.depth_max, args.depth_min, args.vis_depth_max, args.vis_depth_min, args.vis_depth_inv)

def save_phase_as_uint8colored(args, img, filename, zero_mask=False):
    save_image_generic(args, img, filename, zero_mask, cmap, args.phase_max, args.phase_min, args.vis_phase_max, args.vis_phase_min, args.vis_phase_inv)

def save_phase_to_depth_as_uint16png(args, phase, filename, focus_dis, f_stop, coc_alpha, zero_mask=False):
    phase = np.squeeze(phase.data.cpu().numpy())
    if isinstance(focus_dis, torch.Tensor):
        focus_dis = focus_dis.data.cpu().numpy()
    if isinstance(f_stop, torch.Tensor):
        f_stop = f_stop.data.cpu().numpy()
    if isinstance(coc_alpha, torch.Tensor):
        coc_alpha = coc_alpha.data.cpu().numpy()
    depth = phase_to_depth_normalized(np.clip(phase/MAX_8BIT, 0, 1), focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch, args.phase_min, args.phase_max, args.depth_min, args.depth_max)
    depth = (depth * MAX_16BIT).astype('uint16')
    cv2.imwrite(filename, depth)

def save_phase_to_depth_as_uint8colored(args, phase, filename, focus_dis, f_stop, coc_alpha, zero_mask=False):
    phase = np.squeeze(phase.data.cpu().numpy())
    if isinstance(focus_dis, torch.Tensor):
        focus_dis = focus_dis.data.cpu().numpy()
    if isinstance(f_stop, torch.Tensor):
        f_stop = f_stop.data.cpu().numpy()
    if isinstance(coc_alpha, torch.Tensor):
        coc_alpha = coc_alpha.data.cpu().numpy()
    depth = phase_to_depth_normalized(np.clip(phase/MAX_8BIT, 0, 1), focus_dis, f_stop, coc_alpha, args.focal_len, args.pixel_pitch, args.phase_min, args.phase_max, args.depth_min, args.depth_max)
    save_image_generic(args, depth, filename, zero_mask, cmap, args.depth_max, args.depth_min, args.vis_depth_max, args.vis_depth_min, args.vis_depth_inv)

def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = np.squeeze(img.data.cpu().numpy())
    if(normalized==False):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if(colored==True):
        img = MAX_8BIT * cmap(img)[:, :, :3]
    else:
        img = MAX_8BIT * img
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_feature_as_uint8colored(img, filename):
    # img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
