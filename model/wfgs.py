import numpy as np
import cv2
import math
from numba import njit, prange
import torch
import torch.nn as nn
from utils.util import MAX_8BIT, MAX_16BIT

# L2_index
def L2_index(p1, p2):
    return np.sum((p1 - p2) ** 2)

# calculate_lut
def calculate_lut(sigmacolor, channels, channelssub, subweight):
    num_levels = int(256 ** 2 * channels + (256 * subweight) ** 2 * channelssub)
    LUT = np.zeros(num_levels)
    inv_sigmacolor = 1.0 / sigmacolor
    scale = 3 if channels == 1 else 1
    for i in range(num_levels):
        LUT[i] = math.exp(-math.sqrt(i * scale) * inv_sigmacolor)
    return LUT

@njit(parallel=True)
def calculate_lut_fast(sigmacolor, channels, channelssub, subweight):
    num_levels = int(256 ** 2 * channels + (256 * subweight) ** 2 * channelssub)
    LUT = np.zeros(num_levels)
    inv_sigmacolor = 1.0 / sigmacolor
    scale = 3 if channels == 1 else 1
    for i in prange(num_levels):
        LUT[i] = math.exp(-math.sqrt(i * scale) * inv_sigmacolor)
    return LUT

# FGS_prepro
class FGS_prepro:
    def __init__(self, sigmacolor=0.0, channels=3, channelssub=0, subweight=1.0):
        self.channelssub = channelssub
        self.subweight = subweight
        if sigmacolor > 0.0:
            self.LUT = calculate_lut_fast(sigmacolor, channels, channelssub, subweight)

# calculate_weight
def calculate_weight(guide, param, guide_add=None):
    w_h = np.zeros(guide.shape[:2], dtype=np.float32)
    w_v = np.zeros(guide.shape[:2], dtype=np.float32)
    if guide_add is None or param.channelssub == 0:
        for y in range(guide.shape[0]):
            for x in range(guide.shape[1] - 1):
                index = sum((guide[y, x, k] - guide[y, x + 1, k]) ** 2 for k in range(guide.shape[2]))
                w_h[y, x] = param.LUT[index]
        for y in range(guide.shape[0] - 1):
            for x in range(guide.shape[1]):
                index = sum((guide[y, x, k] - guide[y + 1, x, k]) ** 2 for k in range(guide.shape[2]))
                w_v[y, x] = param.LUT[index]
    else:
        for y in range(guide.shape[0]):
            for x in range(guide.shape[1] - 1):
                index = sum((guide[y, x, k] - guide[y, x + 1, k]) ** 2 for k in range(guide.shape[2]))
                indexsub = sum(((guide_add[y, x, k] - guide_add[y, x + 1, k]) * param.subweight) ** 2 for k in range(guide_add.shape[2]))
                w_h[y, x] = param.LUT[index + indexsub]
        for y in range(guide.shape[0] - 1):
            for x in range(guide.shape[1]):
                index = sum((guide[y, x, k] - guide[y + 1, x, k]) ** 2 for k in range(guide.shape[2]))
                indexsub = sum(((guide_add[y, x, k] - guide_add[y + 1, x, k]) * param.subweight) ** 2 for k in range(guide_add.shape[2]))
                w_v[y, x] = param.LUT[index + indexsub]
    return w_h, w_v

@njit(parallel=True)
def calculate_weight_fast(guide, channelssub, subweight, LUT, guide_add=None):
    w_h = np.zeros(guide.shape[:2], dtype=np.float32)
    w_v = np.zeros(guide.shape[:2], dtype=np.float32)
    if guide_add is None or channelssub == 0:
        for y in prange(guide.shape[0]):
            for x in prange(guide.shape[1] - 1):
                index = 0
                for k in range(guide.shape[2]):
                    index += (guide[y, x, k] - guide[y, x + 1, k]) ** 2
                w_h[y, x] = LUT[index]
        for y in prange(guide.shape[0] - 1):
            for x in prange(guide.shape[1]):
                index = 0
                for k in prange(guide.shape[2]):
                    index += (guide[y, x, k] - guide[y + 1, x, k]) ** 2
                w_v[y, x] = LUT[index]
    else:
        for y in prange(guide.shape[0]):
            for x in prange(guide.shape[1] - 1):
                index = 0
                indexsub = 0
                for k in prange(guide.shape[2]):
                    index += (guide[y, x, k] - guide[y, x + 1, k]) ** 2
                for k in prange(guide_add.shape[2]):
                    t = (guide_add[y, x, k] - guide_add[y, x + 1, k]) * subweight
                    indexsub += t * t
                w_h[y, x] = LUT[index + indexsub]
        for y in prange(guide.shape[0] - 1):
            for x in prange(guide.shape[1]):
                index = 0
                indexsub = 0
                for k in prange(guide.shape[2]):
                    index += (guide[y, x, k] - guide[y + 1, x, k]) ** 2
                for k in prange(guide_add.shape[2]):
                    t = (guide_add[y, x, k] - guide_add[y + 1, x, k]) * subweight
                    indexsub += t * t
                w_v[y, x] = LUT[index + indexsub]
    return w_h, w_v


# lambda_update
def lambda_update(T, iteration, lambda_):
    return 1.5 * lambda_ * 4.0 ** (T - iteration) / (4.0 ** T - 1.0)

# hor_pass
def hor_pass(w_h, src, lambda_):
    W = src.shape[1]
    H = src.shape[0]
    dst = np.zeros_like(src)
    for y in range(H):
        u = np.zeros(W)
        c_tilde = np.zeros(W)
        f_tilde = np.zeros(W)
        fx0 = src[y, 0]
        c0 = -w_h[y, 0] * lambda_
        c_tilde[0] = c0 / (1.0 - c0)
        f_tilde[0] = fx0 / (1.0 - c0)
        for i in range(1, W):
            ai = -lambda_ * w_h[y, i - 1]
            bi = 1.0 + lambda_ * (w_h[y, i - 1] + w_h[y, i])
            ci = -lambda_ * w_h[y, i]
            fi = src[y, i]
            c_tilde[i] = ci / (bi - c_tilde[i - 1] * ai)
            f_tilde[i] = (fi - f_tilde[i - 1] * ai) / (bi - c_tilde[i - 1] * ai)
        u[W - 1] = f_tilde[W - 1]
        for i in range(W - 2, -1, -1):
            u[i] = f_tilde[i] - c_tilde[i] * u[i + 1]
        dst[y, :] = u
    return dst

@njit(parallel=True)
def hor_pass_fast(w_h, src, lambda_):
    W = src.shape[1]
    H = src.shape[0]
    dst = np.zeros_like(src)
    for y in prange(H):
        u = np.zeros(W)
        c_tilde = np.zeros(W)
        f_tilde = np.zeros(W)
        fx0 = src[y, 0]
        c0 = -w_h[y, 0] * lambda_
        c_tilde[0] = c0 / (1.0 - c0)
        f_tilde[0] = fx0 / (1.0 - c0)
        for i in prange(1, W):
            ai = -lambda_ * w_h[y, i - 1]
            bi = 1.0 + lambda_ * (w_h[y, i - 1] + w_h[y, i])
            ci = -lambda_ * w_h[y, i]
            fi = src[y, i]
            c_tilde[i] = ci / (bi - c_tilde[i - 1] * ai)
            f_tilde[i] = (fi - f_tilde[i - 1] * ai) / (bi - c_tilde[i - 1] * ai)
        u[W - 1] = f_tilde[W - 1]
        for i in range(W - 2, -1, -1):
            u[i] = f_tilde[i] - c_tilde[i] * u[i + 1]
        dst[y, :] = u
    return dst

# ver_pass
def ver_pass(w_v, src, lambda_):
    W = src.shape[1]
    H = src.shape[0]
    dst = np.zeros_like(src)
    for x in range(W):
        u = np.zeros(H)
        c_tilde = np.zeros(H)
        f_tilde = np.zeros(H)
        fx0 = src[0, x]
        c0 = -w_v[0, x] * lambda_
        c_tilde[0] = c0 / (1.0 - c0)
        f_tilde[0] = fx0 / (1.0 - c0)
        for i in range(1, H):
            ai = -lambda_ * w_v[i - 1, x]
            bi = 1.0 + lambda_ * (w_v[i - 1, x] + w_v[i, x])
            ci = -lambda_ * w_v[i, x]
            fi = src[i, x]
            c_tilde[i] = ci / (bi - c_tilde[i - 1] * ai)
            f_tilde[i] = (fi - f_tilde[i - 1] * ai) / (bi - c_tilde[i - 1] * ai)
        u[H - 1] = f_tilde[H - 1]
        for i in range(H - 2, -1, -1):
            u[i] = f_tilde[i] - c_tilde[i] * u[i + 1]
        dst[:, x] = u
    return dst

@njit(parallel=True)
def ver_pass_fast(w_v, src, lambda_):
    W = src.shape[1]
    H = src.shape[0]
    dst = np.zeros_like(src)
    for x in prange(W):
        u = np.zeros(H)
        c_tilde = np.zeros(H)
        f_tilde = np.zeros(H)
        fx0 = src[0, x]
        c0 = -w_v[0, x] * lambda_
        c_tilde[0] = c0 / (1.0 - c0)
        f_tilde[0] = fx0 / (1.0 - c0)
        for i in prange(1, H):
            ai = -lambda_ * w_v[i - 1, x]
            bi = 1.0 + lambda_ * (w_v[i - 1, x] + w_v[i, x])
            ci = -lambda_ * w_v[i, x]
            fi = src[i, x]
            c_tilde[i] = ci / (bi - c_tilde[i - 1] * ai)
            f_tilde[i] = (fi - f_tilde[i - 1] * ai) / (bi - c_tilde[i - 1] * ai)
        u[H - 1] = f_tilde[H - 1]
        for i in range(H - 2, -1, -1):
            u[i] = f_tilde[i] - c_tilde[i] * u[i + 1]
        dst[:, x] = u
    return dst

# FastGlobalSmoother
def FastGlobalSmoother(guide, src, param, lambda_, lambda_attenuation=0.25, itermax=3, guide_add=None, weight=None):
    assert src.dtype == np.float32 and src.ndim <= 3
    src_channels = cv2.split(src)
    dst_channels = []
    if src.shape[1] < 2 or src.shape[0] < 2:
        for k in range(1):
            dst_channels.append(src_channels[k])
    else:
        w_h, w_v = calculate_weight_fast(guide, param.channelssub, param.subweight, param.LUT, guide_add)
        if weight is not None:
            w_h *= weight
            w_v *= weight
        for k in range(1):
            lambda_ = lambda_
            output = src_channels[k].copy()
            for iter in range(itermax):
                input = output.copy()
                tmp = hor_pass_fast(w_h, input, lambda_)
                output = ver_pass_fast(w_v, tmp, lambda_)
                lambda_ *= lambda_attenuation
            dst_channels.append(output)
    return cv2.merge(dst_channels)


def sobel(src):
    ksize = 5
    scale = 1.0 / 10.0
    delta = 0
    ddepth = cv2.CV_32F
    alpha = 1.0

    grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x, alpha=alpha)

    grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_y = cv2.convertScaleAbs(grad_y, alpha=alpha)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad_blur = grad

    return grad_blur

def wmedian(src, guide, mask):
    dst = src.copy()
    indices = np.where(mask == 0)

    for i in range(len(indices[0])):
        x, y = indices[0][i], indices[1][i]
        patch = src[max(0, x-1):min(src.shape[0], x+2), max(0, y-1):min(src.shape[1], y+2)]
        guide_patch = guide[max(0, x-1):min(guide.shape[0], x+2), max(0, y-1):min(guide.shape[1], y+2)]
        weights = np.exp(-np.sum((guide_patch - guide[x, y])**2, axis=-1)/2)
        dst[x, y] = np.median(patch.flatten()[np.argsort(weights.flatten())[::-1]])                

    return dst


@njit(parallel=True)
def wmedian_fast(src, guide, mask, window_size):
    dst = src.copy()
    indices = np.where(mask == 0)

    for i in prange(len(indices[0])):
        x, y = indices[0][i], indices[1][i]
        patch = src[max(0, x-window_size//2):min(src.shape[0], x+window_size//2+1), max(0, y-window_size//2):min(src.shape[1], y+window_size//2+1)]
        guide_patch = guide[max(0, x-window_size//2):min(guide.shape[0], x+window_size//2+1), max(0, y-window_size//2):min(guide.shape[1], y+window_size//2+1)]
        mask_patch = mask[max(0, x-window_size//2):min(mask.shape[0], x+window_size//2+1), max(0, y-window_size//2):min(mask.shape[1], y+window_size//2+1)]        
        weights = np.exp(-np.sum((guide_patch - guide[x, y])**2, axis=-1)/2) * mask_patch
        dst[x, y] = np.median(patch.flatten()[np.argsort(weights.flatten())[::-1]])

    return dst

def calc_mask(src, mask_thr, conf=None, conf_gain=1.0):
    edge_gain = MAX_16BIT / MAX_8BIT
    edge = sobel(src*edge_gain)
    edge = edge.astype(np.float32)
    if conf is not None:
        edge = edge * (1.0 - conf) * 2.0 * conf_gain
    wei = cv2.threshold(edge, mask_thr, 1.0, cv2.THRESH_BINARY_INV)[1]

    return wei


class WFGS(nn.Module):
    def __init__(self, args):
        super(WFGS, self).__init__()
        self.args = args

    def forward(self, input, conf=None):
        guide = input['rgb']
        src = input['d']

        device = guide.device
        guide = guide.to('cpu').detach().numpy().copy()
        src = src.to('cpu').detach().numpy().copy()
        guide = guide.astype(np.int32)

        mask_thr = self.args.wfgs_mask_thr
        lambda_ = self.args.wfgs_lambda
        sigmacolor = self.args.wfgs_sigma
        lambda_attenuation = self.args.wfgs_lambda_att
        iter_ = self.args.wfgs_iter
        preproParamFGS = FGS_prepro(sigmacolor)
        prefill_wsize = self.args.wfgs_prefill_wsize
        norm = MAX_16BIT
        
        d_r = np.zeros(src.shape)
        bs = src.shape[0]
        for b in range(bs):
            guide_b = np.transpose(guide[b, ...], (1, 2, 0))
            src_b = np.transpose(src[b, ...], (1, 2, 0))

            src_b = np.squeeze(src_b)

            if self.args.wfgs_prefill:
                if self.args.wfgs_conf:
                    conf_b = np.squeeze(np.transpose(conf[b, ...], (1, 2, 0)))
                    wei = calc_mask(src_b, mask_thr, conf_b, self.args.wfgs_conf_thr)
                else:
                    wei = calc_mask(src_b, mask_thr)

                src_b = wmedian_fast(src_b, guide_b, wei, prefill_wsize)

            wei = calc_mask(src_b, mask_thr)

            src_b = src_b.astype(np.float32)

            src_b = src_b * wei

            mask = np.ones((wei.shape[0], wei.shape[1]), dtype=np.float32)
            mask = mask * wei

            wei += 1.0E-3
            wei[wei > 1.0] = 1.0

            d_r_b = FastGlobalSmoother(guide_b, src_b, preproParamFGS, lambda_, lambda_attenuation, iter_, None, wei)
            mask_dst = FastGlobalSmoother(guide_b, mask, preproParamFGS, lambda_, lambda_attenuation, iter_, None, wei)

            div = np.ones((wei.shape[0], wei.shape[1]), dtype=np.float32)
            div[mask_dst > 1.0E-6] = mask_dst[mask_dst > 1.0E-6]
            d_r_b = d_r_b / div
            d_r_b = (np.clip(d_r_b / MAX_8BIT, 0, 1) * norm).astype(np.int).astype(np.float)/norm * MAX_8BIT

            d_r_b = d_r_b[..., np.newaxis]
            d_r[b, ...] = np.transpose(d_r_b, (2, 0, 1))

        d_r = torch.from_numpy(d_r.astype(np.float32)).clone().to(device)

        return d_r

