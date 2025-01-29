import os
import shutil
import torch
import csv
import glob
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

def search_checkpoint_latest(dir):
    pths = sorted(glob.glob(os.path.join(dir, 'checkpoint-*.pth.tar')))
    return pths[-1] if pths else ''

def search_checkpoint_best(dir):
    return os.path.join(dir, 'model_best.pth.tar')

ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                       "*build", "*.fuse*", "*_drive_*")

def backup_source_code(source_directory, backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree(source_directory, backup_directory, ignore=ignore_hidden)

def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init
    lr = _adjust_lr_default(lr_init, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def _adjust_lr_default(lr_init, epoch):
    if epoch >= 25:
        return lr_init * 0.01
    if epoch >= 15:
        return lr_init * 0.1
    if epoch >= 10:
        return lr_init * 0.5
    return lr_init

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, f'checkpoint-{epoch}.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    _remove_previous_checkpoint(epoch, output_directory)

def _remove_previous_checkpoint(epoch, output_directory):
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, f'checkpoint-{epoch - 1}.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def get_folder_name(args):
    current_time = datetime.now(JST).strftime('%Y-%m-%d@%H-%M-%S-%f')
    cr = _get_crop_folder_name(args)
    suffix = f".{args.suffix}" if args.suffix else f".time={current_time}"
    return os.path.join(args.result, f'se={args.seed}.nv={args.network_variant}.c={args.criterion}.bs={args.batch_size}.{cr}{suffix}')

def _get_crop_folder_name(args):
    if args.crop_folder_name:
        return args.crop_folder_name
    if args.random_crop:
        return f'rcr={args.random_crop_height}x{args.random_crop_width}'
    return f'cr={args.val_h}x{args.val_w}'