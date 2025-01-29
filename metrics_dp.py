import torch
import numpy as np
from scipy.stats import rankdata
from utils.util import MAX_8BIT

def affine_invariant_1(Y, Target, confidence_map=None, irls_iters=5, eps=1e-3):
    assert Y.shape == Target.shape
    if confidence_map is None:
        confidence_map = np.ones_like(Target)
    
    y, t, conf = Y.ravel(), Target.ravel(), confidence_map.ravel()
    w = np.ones_like(y, float)
    ones = np.ones_like(y, float)
    
    for _ in range(irls_iters):
        w_sqrt = np.sqrt(w * conf)
        WX = w_sqrt[:, None] * np.stack([y, ones], 1)
        Wt = w_sqrt * t
        b = np.linalg.lstsq(WX, Wt, rcond=None)[0]
        affine_y = y * b[0] + b[1]
        residual = np.abs(affine_y - t)
        w = 1 / np.maximum(eps, residual)
    
    ai1 = np.sum(conf * residual) / np.sum(conf)
    return ai1, b

def affine_invariant_2(Y, Target, confidence_map=None, eps=1e-3):
    assert Y.shape == Target.shape
    if confidence_map is None:
        confidence_map = np.ones_like(Target)
    
    y, t, conf = Y.ravel(), Target.ravel(), confidence_map.ravel()
    ones = np.ones_like(y, float)
    X = conf[:, None] * np.stack([y, ones], 1)
    t = conf * t
    b = np.linalg.lstsq(X, t, rcond=None)[0]
    affine_y = y * b[0] + b[1]
    residual_sq = np.minimum(np.square(affine_y - t), np.finfo(np.float32).max)
    ai2 = np.sqrt(np.sum(conf * residual_sq) / np.sum(conf))
    return ai2, b

def spearman_correlation(X, Y, W=None):
    assert X.shape == Y.shape
    if W is None:
        W = np.ones_like(X)
    
    x, y, w = X.ravel(), Y.ravel(), W.ravel()
    
    def _rescale_rank(z):
        return (z - len(z) // 2) / (len(z) // 2)
    
    rx = _rescale_rank(rankdata(x, method='dense'))
    ry = _rescale_rank(rankdata(y, method='dense'))
    
    def E(z):
        return np.sum(w * z) / np.sum(w)
    
    def _pearson_correlation(x, y):
        mu_x, mu_y = E(x), E(y)
        var_x, var_y = E(x * x) - mu_x * mu_x, E(y * y) - mu_y * mu_y
        return (E(x * y) - mu_x * mu_y) / (np.sqrt(var_x * var_y))
    
    return _pearson_correlation(rx, ry)

class Result:
    def __init__(self, args=None):
        self.args = args
        self.ai1 = 0
        self.ai2 = 0
        self.sp = 0
        self.data_time = 0
        self.gpu_time = 0
        self.photometric = 0

    def set_to_worst(self):
        self.ai1 = np.inf
        self.ai2 = np.inf
        self.sp = np.inf
        self.data_time = 0
        self.gpu_time = 0

    def update(self, ai1, ai2, sp, gpu_time, data_time, photometric=0):
        self.ai1 = ai1
        self.ai2 = ai2
        self.sp = sp
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.photometric = photometric

    def evaluate(self, output_phase, batch_data, conf_inv=None, photometric=0):
        output_depth = batch_data['gt'].cpu().numpy() / MAX_8BIT
        output_phase_normalized = torch.clamp(output_phase / MAX_8BIT, 0, 1)
        output_phase_pix = output_phase_normalized * (self.args.phase_max - self.args.phase_min) + self.args.phase_min
        output_phase_pix = output_phase_pix.cpu().numpy()

        self.ai1, b1 = affine_invariant_1(output_phase_pix, output_depth)
        self.ai2, b2 = affine_invariant_2(output_phase_pix, output_depth)
        phase_affine = output_phase_pix * b2[0] + b2[1]
        self.sp = 1 - np.abs(spearman_correlation(phase_affine[0, 0, :, :], output_depth[0, 0, :, :]))
        self.photometric = float(photometric)

class AverageMeter:
    def __init__(self, args=None):
        self.args = args
        self.reset(time_stable=True)

    def reset(self, time_stable):
        self.count = 0.0
        self.sum_ai1 = 0
        self.sum_ai2 = 0
        self.sum_sp = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.time_stable = time_stable
        self.time_stable_counter_init = 10
        self.time_stable_counter = self.time_stable_counter_init

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_ai1 += n * result.ai1
        self.sum_ai2 += n * result.ai2
        self.sum_sp += n * result.sp
        self.sum_data_time += n * data_time
        if self.time_stable and self.time_stable_counter > 0:
            self.time_stable_counter -= 1
        else:
            self.sum_gpu_time += n * gpu_time
        self.sum_photometric += n * result.photometric

    def average(self):
        avg = Result(self.args)
        if self.time_stable:
            if self.count > 0 and self.count - self.time_stable_counter_init > 0:
                avg.update(
                    self.sum_ai1 / self.count, self.sum_ai2 / self.count, self.sum_sp / self.count,
                    self.sum_gpu_time / (self.count - self.time_stable_counter_init),
                    self.sum_data_time / self.count, 
                    self.sum_photometric / self.count
                )
            elif self.count > 0:
                avg.update(
                    self.sum_ai1 / self.count, self.sum_ai2 / self.count, self.sum_sp / self.count,
                    0,
                    self.sum_data_time / self.count,
                    self.sum_photometric / self.count
                )
        elif self.count > 0:
            avg.update(
                self.sum_ai1 / self.count, self.sum_ai2 / self.count, self.sum_sp / self.count,
                self.sum_gpu_time / self.count,
                self.sum_data_time / self.count, 
                self.sum_photometric / self.count
            )
        return avg