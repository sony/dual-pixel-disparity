import os
import csv
import utils.vis_utils as vis_utils
from metrics import Result
from datetime import datetime, timedelta, timezone
from utils.helper import get_folder_name
JST = timezone(timedelta(hours=+9), 'JST')

class logger:
    def set_result(self):
        self.best_result = Result()

    def __init__(self, args, prepare=True):
        self.args = args
        output_directory = get_folder_name(args)
        self.output_directory = output_directory
        self.set_result()
        self.best_result.set_to_worst()
        self.fieldnames = self.set_fieldnames()

        if not prepare:
            return
        self._prepare_directories(output_directory)
        self._prepare_csv_files(args)

    def _prepare_directories(self, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.train_csv = os.path.join(output_directory, 'train.csv')
        self.val_csv = os.path.join(output_directory, 'val.csv')
        self.eval_csv = os.path.join(output_directory, 'eval.csv')
        data_name = os.path.basename(self.args.data_folder)
        self.test_csv = os.path.join(output_directory, f'test_{data_name}.csv')
        self.eval_txt = os.path.join(output_directory, 'eval.txt')
        self.test_txt = os.path.join(output_directory, 'test.txt')
        self.best_txt = os.path.join(output_directory, 'best.txt')
        self.args_txt = os.path.join(output_directory, 'args.txt')

    def _prepare_csv_files(self, args):
        if args.resume == '':
            if args.backup_code:
                self._backup_source_code(args)
            self._create_csv_file(self.train_csv)
            self._create_csv_file(self.val_csv)
        if args.test_with_gt:
            self._create_csv_file(self.test_csv)
        if args.evaluate:
            self._create_csv_file(self.eval_csv)

    def _backup_source_code(self, args):
        print("=> creating source code backup ...")
        backup_directory = os.path.join(self.output_directory, "code_backup")
        self.backup_directory = backup_directory
        backup_source_code(args.source_directory, backup_directory)
        print("=> finished creating source code backup.")

    def _create_csv_file(self, csv_path):
        with open(csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def set_fieldnames(self):
        return [
            'epoch', 'rmse', 'mae', 'irmse', 'imae', 'mse', 'absrel', 'lg10',
            'silog', 'squared_rel', 'delta1', 'delta2', 'delta3', 'data_time',
            'gpu_time', 'datetime'
        ]

    def conditional_print(self, split, i, epoch, lr, n_set, blk_avg_meter, avg_meter):
        if split != 'train' and (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print('=> output: {}'.format(self.output_directory))
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
                'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
                'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
                'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
                'silog={blk_avg.silog:.2f}({average.silog:.2f}) '
                'squared_rel={blk_avg.squared_rel:.2f}({average.squared_rel:.2f}) '
                'Delta1={blk_avg.delta1:.3f}({average.delta1:.3f}) '
                'REL={blk_avg.absrel:.3f}({average.absrel:.3f})\n\t'
                'Lg10={blk_avg.lg10:.3f}({average.lg10:.3f}) '
                .format(epoch, i + 1, n_set, lr=lr, blk_avg=blk_avg, average=avg, split=split.capitalize()))
            blk_avg_meter.reset(False)

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        csvfile_name = self._get_csvfile_name(split)
        if csvfile_name:
            self._write_csv(csvfile_name, avg, epoch)
        if split == "eval":
            self.save_single_txt(self.eval_txt, avg, epoch)
        elif "test" in split and self.args.test_with_gt:
            self.save_single_txt(self.test_txt, avg, epoch)
        return avg

    def _get_csvfile_name(self, split):
        if split == "train":
            return self.train_csv
        elif split == "val":
            return self.val_csv
        elif split == "eval":
            return self.eval_csv
        elif "test" in split and self.args.test_with_gt:
            return self.test_csv
        else:
            raise ValueError("wrong split provided to logger")

    def _write_csv(self, csvfile_name, avg, epoch):
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({
                'epoch': epoch,
                'rmse': avg.rmse,
                'mae': avg.mae,
                'irmse': avg.irmse,
                'imae': avg.imae,
                'mse': avg.mse,
                'silog': avg.silog,
                'squared_rel': avg.squared_rel,
                'absrel': avg.absrel,
                'lg10': avg.lg10,
                'delta1': avg.delta1,
                'delta2': avg.delta2,
                'delta3': avg.delta3,
                'gpu_time': avg.gpu_time,
                'data_time': avg.data_time,
                'datetime': datetime.now(JST).strftime('%Y%m%d %H:%M:%S')
            })

    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(
                ("rank_metric={}\n" + "epoch={}\n" + "rmse={:.3f}\n" +
                 "mae={:.3f}\n" + "silog={:.3f}\n" + "squared_rel={:.3f}\n" +
                 "irmse={:.3f}\n" + "imae={:.3f}\n" + "mse={:.3f}\n" +
                 "absrel={:.3f}\n" + "lg10={:.3f}\n" + "delta1={:.3f}\n" +
                 "t_gpu={:.4f}").format(self.args.rank_metric, epoch,
                                        result.rmse, result.mae, result.silog,
                                        result.squared_rel, result.irmse,
                                        result.imae, result.mse, result.absrel,
                                        result.lg10, result.delta1,
                                        result.gpu_time))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def save_args_txt(self):
        with open(self.args_txt, 'w') as txtfile:
            txtfile.write(str(self.args))

    def _get_img_comparison_name(self, mode, epoch, is_best=False):
        if mode == 'eval':
            return self.output_directory + '/comparison_eval.png'
        if mode == 'val':
            if is_best:
                return self.output_directory + '/comparison_best.png'
            else:
                return self.output_directory + '/comparison_' + str(epoch) + '.png'

    def conditional_save_img_comparison(self, mode, i, ele, pred, epoch, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None, skip=100):
        if mode == 'val' or mode == 'eval':
            if i == 0:
                self.img_merge = vis_utils.merge_into_row(self.args, ele, pred, predrgb, predg, extra, extra2, extrargb)
            elif i % skip == 0 and i <= 7 * skip:
                row = vis_utils.merge_into_row(self.args, ele, pred, predrgb, predg, extra, extra2, extrargb)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
                if i == 7 * skip:
                    filename = self._get_img_comparison_name(mode, epoch)
                    if not self.args.skip_conditional_img:
                        vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison_as_best(self, mode, epoch):
        if mode == 'val':
            filename = self._get_img_comparison_name(mode, epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)

    def get_ranking_error(self, result):
        print(vars(result))
        return getattr(result, self.args.rank_metric)

    def rank_conditional_save_best(self, mode, result, epoch):
        is_best = None
        if mode == "val" and not self.args.evaluate:
            error = self.get_ranking_error(result)
            best_error = self.get_ranking_error(self.best_result)
            is_best = error < best_error
            if is_best:
                self.old_best_result = self.best_result
                self.best_result = result
                self.save_best_txt(result, epoch)
        return is_best

    def conditional_save_img(self, mode, i, ele, pred, epoch, extra):
        if ("test" in mode or mode == "eval") and not self.args.skip_image_output:
            image_folder = os.path.join(self.output_directory, mode + "_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            self._save_images(i, ele, pred, image_folder)

    def _save_images(self, i, ele, pred, image_folder):
        self.conditional_save_img_each(i, pred, image_folder, 'dpred_mono')
        self.conditional_save_img_each(i, pred, image_folder, 'dpred')
        self.conditional_save_img_each(i, ele['rgb'], image_folder, 'rgb')
        self.conditional_save_img_each(i, ele['d'], image_folder, 'dedge')
        if not self.args.test:
            self.conditional_save_img_each(i, ele['gt'], image_folder, 'dgt')
        self.conditional_save_img_comp_each(i, ele, pred, image_folder, 'comp')

    def conditional_save_img_each(self, i, img, path, name, focus_dis=None):
        filename = os.path.join(path, str(i).zfill(5) + '_' + name + '.png')
        if name == "rgb":
            vis_utils.save_image_torch(img, filename, False)
        elif name == 'dpred_mono':
            vis_utils.save_depth_as_uint16png_upload(img, filename)
        elif name == 'dedge':
            vis_utils.save_depth_as_uint8colored(self.args, img, filename, True)
        else:
            vis_utils.save_depth_as_uint8colored(self.args, img, filename)

    def conditional_save_img_comp_each(self, i, ele, pred, path, name):
        filename = os.path.join(path, str(i).zfill(5) + '_' + name + '.png')
        img_comp = vis_utils.merge_into_row(self.args, ele, pred)
        vis_utils.save_image(img_comp, filename)

    def conditional_save_img_comp_each_extra(self, i, ele, pred, extra, path, name):
        filename = os.path.join(path, str(i).zfill(5) + '_' + name + '.png')
        img_comp = vis_utils.merge_into_row(self.args, ele, pred, extra=extra)
        vis_utils.save_image(img_comp, filename)        

    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'iRMSE={average.irmse:.3f}\n'
              'iMAE={average.imae:.3f}\n'
              'squared_rel={average.squared_rel}\n'
              'silog={average.silog}\n'
              'Delta1={average.delta1:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
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
