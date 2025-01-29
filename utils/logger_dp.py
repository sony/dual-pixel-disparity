import os
import csv
import utils.vis_utils as vis_utils
from metrics_dp import Result
from datetime import datetime, timedelta, timezone
from utils.helper import get_folder_name
from utils.logger import logger as logger_base
JST = timezone(timedelta(hours=+9), 'JST')

class logger(logger_base):
    def set_result(self):
        self.best_result = Result()

    def set_fieldnames(self):
        return [
                'epoch',
                'ai1', 'ai2', 'sp',
                'data_time','gpu_time', 'datetime']

    def conditional_print(self, split, i, epoch, lr, n_set, blk_avg_meter, avg_meter):
        if split != 'train' and (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print('=> output: {}'.format(self.output_directory))
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'AI1={blk_avg.ai1:.4f}({average.ai1:.4f}) '
                'AI2={blk_avg.ai2:.4f}({average.ai2:.4f}) '
                'SP={blk_avg.sp:.4f}({average.sp:.4f}) '
                .format(epoch, i + 1, n_set, lr=lr, blk_avg=blk_avg, average=avg, split=split.capitalize()))
            blk_avg_meter.reset(False)

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        csvfile_name = self.get_csvfile_name(split)
        if csvfile_name:
            self.save_single_txt(self.get_txtfile_name(split), avg, epoch)
            with open(csvfile_name, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow({
                    'epoch': epoch,
                    'ai1': avg.ai1,
                    'ai2': avg.ai2,
                    'sp': avg.sp,
                    'gpu_time': avg.gpu_time,
                    'data_time': avg.data_time,
                    'datetime': datetime.now(JST).strftime('%Y%m%d %H:%M:%S')
                })
        return avg

    def get_csvfile_name(self, split):
        if split == "train":
            return self.train_csv
        elif split == "val":
            return self.val_csv
        elif split == "eval":
            return self.eval_csv
        elif "test" in split and self.args.test_with_gt:
            return self.test_csv
        return None

    def get_txtfile_name(self, split):
        if split == "eval":
            return self.eval_txt
        elif "test" in split and self.args.test_with_gt:
            return self.test_txt
        return None

    def save_single_txt(self, filename, result, epoch):
        if filename:
            with open(filename, 'w') as txtfile:
                txtfile.write(
                    ("rank_metric={}\n" + "epoch={}\n" + 
                     "ai1={:.4f}\n" + "ai2={:.4f}\n" + "sp={:.4f}\n" +
                     "t_gpu={:.4f}").format(self.args.rank_metric, epoch,
                                            result.ai1, result.ai2, result.sp,
                                            result.gpu_time))

    def conditional_save_img(self, mode, i, ele, pred, epoch, extra):
        if ("test" in mode or mode == "eval") and not self.args.skip_image_output:
            data_name = os.path.basename(self.args.data_folder)
            image_folder = os.path.join(self.output_directory, f'test_results_{data_name}')
            os.makedirs(image_folder, exist_ok=True)
            self.save_images(i, ele, pred, image_folder, extra)

    def save_images(self, i, ele, pred, image_folder, extra):
        self.conditional_save_img_each(i, pred, image_folder, 'ppred_mono')
        self.conditional_save_img_each(i, pred, image_folder, 'ppred')
        self.conditional_save_img_each(i, ele['rgb'], image_folder, 'rgb')
        self.conditional_save_img_each(i, ele['d'], image_folder, 'pedge')
        if self.args.output_mono:
            self.conditional_save_img_each(i, ele['d'], image_folder, 'pedge_mono')
        if self.args.output_lowres_phase:
            self.conditional_save_img_each(i, ele['d_lowres'], image_folder, 'pedge_mono_lowres')
            self.conditional_save_img_each(i, ele['d_lowres'], image_folder, 'pedge_lowres')
        if not self.args.test:
            self.conditional_save_img_each(i, ele['gt'], image_folder, 'dgt')
            if self.args.output_mono:
                self.conditional_save_img_each(i, ele['gt'], image_folder, 'dgt_mono')
        if self.args.depth_to_phase_vis:
            self.conditional_save_img_each(i, pred, image_folder, 'pdpred', ele['focus_dis'], ele['f_stop'], ele['coc_alpha'])
        if self.args.network_model == 'c':
            self.conditional_save_img_each(i, extra, image_folder, 'conf')
            self.conditional_save_img_comp_each_extra(i, ele, pred, extra, image_folder, 'comp')
        else:
            self.conditional_save_img_comp_each(i, ele, pred, image_folder, 'comp')

    def conditional_save_img_each(self, i, img, path, name, focus_dis=1000, f_stop=1.4, coc_alpha=0.3):
        filename = os.path.join(path, str(i).zfill(5) + '_' + name + '.png')
        if name in ['rgb']:
            vis_utils.save_image_torch(img, filename, False)
        elif name in ['ppred_mono', 'pedge_mono', 'pedge_mono_lowres', 'dgt_mono']:
            vis_utils.save_depth_as_uint16png_upload(img, filename)
        elif name in ['pedge', 'ppred', 'pedge_lowres']:
            vis_utils.save_phase_as_uint8colored(self.args, img, filename, True)
        elif name == 'pdpred':
            vis_utils.save_phase_to_depth_as_uint8colored(self.args, img, filename, focus_dis, f_stop, coc_alpha, True)
        elif name in ['conf']:
            vis_utils.save_mono_image(img, filename)
        else:
            vis_utils.save_depth_as_uint8colored(self.args, img, filename)

    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
              'AI1={average.ai1:.4f}\n'
              'AI2={average.ai2:.4f}\n'
              'SP={average.sp:.4f}\n'
              't_GPU={time:.3f}'.format(average=avg, time=avg.gpu_time))
        if is_best and mode == "val":
            print("New best model by %s (was %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.old_best_result)))
        elif mode == "val":
            print("(best %s is %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.best_result)))
        print("*\n")
