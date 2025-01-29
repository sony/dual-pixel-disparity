import torch
import math
import numpy as np
from utils.util import MAX_8BIT

lg_e_10 = math.log(10)

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x."""
    return torch.log(x) / lg_e_10

class Result:
    def __init__(self, args=None):
        self.args = args
        self.reset()

    def reset(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0


    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, 
               delta1, delta2, delta3, gpu_time, data_time, silog):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog

    def evaluate(self, output, batch_data, conf_inv=None):
        target = batch_data['gt']
        output_m, target_m = self._convert_to_meters(output, target)

        valid_mask = target_m > self.args.eval_depth_min
        output_m, target_m = output_m[valid_mask], target_m[valid_mask]

        if len(output_m) == 0: output_m = torch.tensor(0.0)
        if len(target_m) == 0: target_m = torch.tensor(0.0)

        output_mm, target_mm = output_m * 1000, target_m * 1000
        abs_diff = (output_mm - target_mm).abs()

        self._calculate_metrics(output_mm, target_mm, abs_diff)
        self.data_time = 0
        self.gpu_time = 0
        self.silog = self._calculate_silog(output_m, target_m)
        self._calculate_inverse_metrics(output_m, target_m)

    def _convert_to_meters(self, output, target):
        output_m = output / MAX_8BIT * (self.args.depth_max - self.args.depth_min) + self.args.depth_min
        target_m = target / MAX_8BIT * (self.args.depth_max - self.args.depth_min) + self.args.depth_min
        return output_m, target_m

    def _calculate_metrics(self, output_mm, target_mm, abs_diff):
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm)**2).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())

    def _calculate_silog(self, output_m, target_m):
        err_log = torch.log(target_m) - torch.log(output_m)
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        return math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

    def _calculate_inverse_metrics(self, output_m, target_m):
        inv_output_km = (1e-3 * output_m)**(-1)
        inv_target_km = (1e-3 * target_m)**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

class AverageMeter:
    def __init__(self, args=None):
        self.args = args
        self.reset(time_stable=True)

    def reset(self, time_stable):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_silog = 0
        self.time_stable = time_stable
        self.time_stable_counter_init = 10
        self.time_stable_counter = self.time_stable_counter_init

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        if self.time_stable and self.time_stable_counter > 0:
            self.time_stable_counter -= 1
        else:
            self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog

    def average(self):
        avg = Result()
        if self.count > 0:
            gpu_time_avg = self.sum_gpu_time / (self.count - self.time_stable_counter_init) if self.time_stable and self.count > self.time_stable_counter_init else 0
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, gpu_time_avg,
                self.sum_data_time / self.count, self.sum_silog / self.count)
        return avg