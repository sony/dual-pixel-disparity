import argparse
import criteria

def add_network_args(parser):
    parser.add_argument('-n', '--network-model',
                        type=str,
                        default="n",
                        choices=["n", "c"],
                        help='choose a model: normal(n) or with confidence(c) (default: n)')
    parser.add_argument('-nv', '--network-variant',
                        type=str,
                        default="costdcnet",
                        choices=["gt", "through", "nlspn", "costdcnet"],
                        help='choose a variant of model')

def add_hyperparameter_args(parser):
    parser.add_argument('--seed', '-se',
                        default=-1,
                        type=int,
                        metavar='N',
                        help='seed value. if -1, random seed (default: -1)')
    parser.add_argument('--epochs',
                        default=50,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 50)')
    parser.add_argument('-b', '--batch-size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--lr', '--learning-rate',
                        default=1e-3,
                        type=float,
                        metavar='LR',
                        help='initial learning rate (default 1e-5)')
    parser.add_argument('--weight-decay', '--wd',
                        default=1e-6,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--train-num',
                        default=0,
                        type=int,
                        help='the number of train data, 0 is all (default: 0)')
    parser.add_argument('--train-random',
                        action="store_true",
                        default=False,
                        help='random pickup for training data (default: false)')

def add_loss_function_args(parser):
    parser.add_argument('-c', '--criterion',
                        metavar='LOSS',
                        default='l2',
                        choices=criteria.loss_names,
                        help='loss function: | '.join(criteria.loss_names) +
                        ' (default: l2)')
    parser.add_argument('--rank-metric',
                        type=str,
                        default='mae',
                        help='metrics for which best result is saved')

def add_data_args(parser):
    parser.add_argument('-dt', '--data-type',
                        default='ed',
                        type=str,
                        choices=["ed", "cdp", "cdp1169"],
                        help='The type of source data (ed:EdgeDepth, cdp:CanonDualPixel, cdp:CanonDualPixel, resolution:1169x779).')

def add_paths_args(parser):
    parser.add_argument('--data-folder',
                        default='../data/dataset/train',
                        type=str,
                        metavar='PATH',
                        help='data folder')
    parser.add_argument('--result',
                        default='../data/results/',
                        type=str,
                        metavar='PATH',
                        help='result folder (default: "../data/results/")')
    parser.add_argument('--suffix',
                        default="",
                        type=str,
                        metavar='FN',
                        help='suffix of result folder name (default: none)')
    parser.add_argument('--source-directory',
                        default='.',
                        type=str,
                        metavar='PATH',
                        help='source code directory for backup (default: .)')

def add_augmentation_args(parser):
    parser.add_argument('--random-crop',
                        action="store_true",
                        default=False,
                        help='Random cropping (default: false)')
    parser.add_argument('-he', '--random-crop-height',
                        default=384,
                        type=int,
                        metavar='N',
                        help='random crop height (default: 384)')
    parser.add_argument('-w', '--random-crop-width',
                        default=640,
                        type=int,
                        metavar='N',
                        help='random crop height (default: 640)')
    parser.add_argument('--jitter',
                        type=float,
                        default=0.1,
                        help='color jitter for images')

def add_resume_args(parser):
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start-epoch-bias',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number bias(useful on restarts)')
    parser.add_argument('-ol', '--optimizer-load',
                        action="store_true",
                        default=False,
                        help='load optimizer when resumimg (default: false)')
    parser.add_argument('--autoresume',
                        action="store_true",
                        default=False,
                        help='auto resume from latest checkpoint (default: false)')
    parser.add_argument('--bestresume',
                        action="store_true",
                        default=False,
                        help='auto resume from best checkpoint (default: false)')

def add_evaluation_args(parser):
    parser.add_argument('-e', '--evaluate',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='use existing models for evaluation (default: none)')
    parser.add_argument('--test',
                        action="store_true",
                        default=False,
                        help='save result of test dataset for submission')
    parser.add_argument('--test-with-gt',
                        action="store_true",
                        default=False,
                        help='test with ground truth.')
    parser.add_argument('-p', '--print-freq',                        
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--skip-image-output',
                        action="store_true",
                        default=False,
                        help='skip generating and outputting the result image (default: false)')
    parser.add_argument('--skip-conditional-img',
                        action="store_true",
                        default=True,
                        help='skip generating conditional image (default: true)')
    parser.add_argument('--vis-skip',
                        default=-1,
                        type=int,
                        metavar='N',
                        help='skip of visualize comparison image, automode when set -1 (default: -1)')
    parser.add_argument('--eval-each',
                        action="store_true",
                        default=False,
                        help='evaluation for each image (default: false)')
    parser.add_argument('--depth-max',
                        default=65.535,
                        type=float,
                        metavar='DMAX',
                        help='Max depth (default: 65.535m)')
    parser.add_argument('--depth-min',
                        default=0.0,
                        type=float,
                        metavar='DMIN',
                        help='Min depth (default: 0.0m)')
    parser.add_argument('--vis-depth-max',
                        default=3.0,
                        type=float,
                        metavar='VDMAX',
                        help='Max depth for visualization (default: 3.0m)')
    parser.add_argument('--vis-depth-min',
                        default=0.0,
                        type=float,
                        metavar='VDMIN',
                        help='Min depth for visualization (default: 0.0m)')
    parser.add_argument('--eval-depth-min',
                        default=0.5,
                        type=float,
                        metavar='EDMIN',
                        help='Min depth for evaluation (default: 0.1m)')
    parser.add_argument('--eval-depth-max',
                        default=3.0,
                        type=float,
                        metavar='EDMAX',
                        help='Max depth for evaluation (default: 65.535m)')
    parser.add_argument('--vis-depth-inv',
                        action="store_true",
                        default=True,
                        help='Use inverse depth for visualization (default: true)')

def add_dual_pixel_args(parser):
    parser.add_argument('--phase-max',
                        default=40.0,
                        type=float,
                        metavar='PMAX',
                        help='max phase (default: 10pixel)')
    parser.add_argument('--phase-min',
                        default=-40.0,
                        type=float,
                        metavar='PMIN',
                        help='min phase (default: -10pixel)')
    parser.add_argument('--vis-phase-max',
                        default=10.0,
                        type=float,
                        metavar='VPMAX',
                        help='max phase for visualization (default: 10pixel)')
    parser.add_argument('--vis-phase-min',
                        default=-10.0,
                        type=float,
                        metavar='VPMIN',
                        help='min phase for visualization (default: -10pixel)')
    parser.add_argument('--vis-phase-inv',
                        action="store_true",
                        default=True,
                        help='use inverse phase for visualization (default: true)')
    parser.add_argument('-d2p', '--depth-to-phase',
                        action="store_true",
                        default=False,
                        help='convert depth to phase during training (default: false)')
    parser.add_argument('-d2pvis', '--depth-to-phase-vis',
                        action="store_true",
                        default=False,
                        help='convert depth to phase at visualizing (default: false)')
    parser.add_argument('-apn', '--add-phase-noise',
                        action="store_true",
                        default=False,
                        help='add noise to phase during training (default: false)')
    parser.add_argument('-nt', '--noise-type',
                        default='efmount',
                        type=str,
                        metavar='NT',
                        help='type of noise [efmount / cmount / gauss] (default: efmount)')
    parser.add_argument('-nsg', '--noise-sigma-gain',
                        default=5.0,
                        type=float,
                        metavar='NSM',
                        help='noise sigma gain (default: 5.0)')
    parser.add_argument('-nsm', '--noise-sigma-max',
                        default=5.0,
                        type=float,
                        metavar='NSM',
                        help='noise sigma max (default: 5.0)')
    parser.add_argument('-fl', '--focal-len',
                        default=50.0,
                        type=float,
                        metavar='FL',
                        help='focal length (default: 50.0)')
    parser.add_argument('-pp', '--pixel-pitch',
                        default=0.025316456,
                        type=float,
                        metavar='PP',
                        help='pixel pitch (default: 0.025316456)')
    parser.add_argument('--confmax',
                        default=1.0,
                        type=float,
                        metavar='CMAX',
                        help='maximum value of confidence (default: 1.0)')
    parser.add_argument('--confmin',
                        default=0.0,
                        type=float,
                        metavar='CMIN',
                        help='minimum value of confidence (default: 0.0)')
    parser.add_argument('--conf-lambda',
                        default=10.0,
                        type=float,
                        metavar='CL',
                        help='lambda value of confidence (default: 10.0)')
    parser.add_argument('--conf-ft-scale',
                        default=16,
                        type=int,
                        choices=[1, 2, 4, 8, 16, 32],
                        help='confidence feature scale (default: 2)')
    parser.add_argument('--conf-num-conv',
                        default=2,
                        type=int,
                        help='the number of convolution calclating confidence')
    parser.add_argument('--conf-eval',
                        action="store_true",
                        default=False,
                        help='evaluation confidence map')
    parser.add_argument('--conf-mask',
                        action="store_true",
                        default=False,
                        help='evaluation with confidence mask')
    parser.add_argument('--conf-mask-thr',
                        default=1.0,
                        type=float,
                        help='threthold of confidence mask')
    parser.add_argument('--conf-sparse-num',
                        default=50,
                        type=int,
                        metavar='CSN',
                        help='the number of sparsification of confidence evaluation')
    parser.add_argument('--lowres-input',
                        action="store_true",
                        default=False,
                        help='use low resolution input.')
    parser.add_argument('--lowres-cnn',
                        action="store_true",
                        default=False,
                        help='use low resolution input for CNN.')
    parser.add_argument('--lowres-scale',
                        default=0.5,
                        type=float,
                        help='scale of low resolution input.')
    parser.add_argument('--lowres-phase',
                        action="store_true",
                        default=False,
                        help='use low resolution input at phase detection.')
    parser.add_argument('--lowres-pscale',
                        default=0.5,
                        type=float,
                        help='scale of low resolution input at phase detection.')
    parser.add_argument('-fdmin', '--focus-dis-min',
                        default=500,
                        type=int,
                        help='min value of focus distance for augmentation')
    parser.add_argument('-fdmax','--focus-dis-max',
                        default=3000,
                        type=int,
                        help='min value of focus distance for augmentation')
    parser.add_argument('-fdstep','--focus-dis-step',
                        default=100,
                        type=int,
                        help='skip value of focus distance for augmentation')
    parser.add_argument('-fsl', '--f-stop-list',
                        nargs='*',
                        default=[1.4, 2.0, 2.8, 4.0, 5.0],
                        type=float,                        
                        help='f stop list for augmentation')
    parser.add_argument('--pre-matching',
                        action="store_true",
                        default=False,
                        help='have result data of pre matching')
    parser.add_argument('--post-process',
                        action="store_true",
                        default=False,
                        help='apply postprocess')
    parser.add_argument('--post-refine',
                        default='',
                        choices=['', 'wfgs'],
                        type=str,
                        help='apply post refinement')
    parser.add_argument('--wfgs-lambda',
                        default=80000.0,
                        type=float,
                        help='lambda of WFGS')
    parser.add_argument('--wfgs-lambda-att',
                        default=0.5,
                        type=float,
                        help='sigma of WFGS')
    parser.add_argument('--wfgs-iter',
                        default=3,
                        type=int,
                        help='iteration of WFGS')
    parser.add_argument('--wfgs-sigma',
                        default=16.0,
                        type=float,
                        help='sigma of WFGS')
    parser.add_argument('--wfgs-mask-thr',
                        default=96.0,
                        type=float,
                        help='edge mask threshold of WFGS')
    parser.add_argument('--wfgs-conf',
                        action="store_true",
                        default=False,
                        help='use confidence for WFGS')
    parser.add_argument('--wfgs-conf-thr',
                        default=18.0,
                        type=float,
                        help='confidence threshold of WFGS')
    parser.add_argument('--wfgs-prefill',
                        action="store_true",
                        default=False,
                        help='pre filling of WFGS')
    parser.add_argument('--wfgs-prefill-wsize',
                        default=39,
                        type=int,
                        help='pre filling window size of WFGS')
    parser.add_argument('--pix-shift',
                        default=25,
                        type=int,
                        help='value of pixel shift of matching')
    parser.add_argument('--ref-area',
                        default=13,
                        type=int,
                        help='value of reference area of matching')
    parser.add_argument('--output-lowres-phase',
                        action="store_true",
                        default=False,
                        help='output lowresolution phase image')
    parser.add_argument('--output-mono',
                        action="store_true",
                        default=False,
                        help='output other mono image')
    parser.add_argument('--use-executable',
                        action="store_true",
                        default=False,
                        help='Use executable for DPMatching (default: false)')

def add_debug_args(parser):
    parser.add_argument('--small',
                        action="store_true",
                        default=False,
                        help='use small dataset (default: false)')
    parser.add_argument('--small-rate',
                        default=0.01,
                        type=float,
                        metavar='SR',
                        help='rate of small dataset, use with "small" argument (default: 0.01)')
    parser.add_argument('--backup-code',
                        action="store_true",
                        default=False,
                        help='backup source code when running (default: false)')
    parser.add_argument('--select-data-num',
                        default=-1,
                        type=int,
                        metavar='SD',
                        help='select process data (default: -1)')

def add_other_args(parser):
    parser.add_argument('--crop',
                        action="store_true",
                        default=False,
                        help='crop image')
    parser.add_argument('--val-h',
                        default=384,
                        type=int,
                        metavar='N',
                        help='validation height (default: 384)')
    parser.add_argument('--val-w',
                        default=640,
                        type=int,
                        metavar='N',
                        help='validation width (default: 640)')
    parser.add_argument('--crop-folder-name',
                        default='',
                        type=str,
                        metavar='FN',
                        help='crop size for folder name (default: none)')
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--cpu',
                        action="store_true",
                        default=False,
                        help='run on cpu (default: false)')
    parser.add_argument('--gpu',
                        default=-1,
                        type=int,
                        metavar='N',
                        help='GPU device, if -1, use parallel mode (default: -1)')

def parser():
    parser = argparse.ArgumentParser(description='Depth densification')
    add_network_args(parser)
    add_hyperparameter_args(parser)
    add_loss_function_args(parser)
    add_data_args(parser)
    add_paths_args(parser)
    add_augmentation_args(parser)
    add_resume_args(parser)
    add_evaluation_args(parser)
    add_dual_pixel_args(parser)
    add_debug_args(parser)
    add_other_args(parser)

    args = parser.parse_args()

    if args.data_type == 'ed':
        args.depth_max = 10.0 # 10.0m max
    elif args.data_type == 'cdp' or args.data_type == 'cdp1169':
        args.depth_max = 2.55 # 2.55m max

    if args.depth_to_phase:
        args.original_depth_max = args.depth_max
        args.original_depth_min = args.depth_min
        args.depth_max = args.phase_max
        args.depth_min = args.phase_min
        args.vis_depth_max = args.vis_phase_max
        args.vis_depth_min = args.vis_phase_min
        args.vis_depth_inv = args.vis_phase_inv
        args.eval_depth_min = args.phase_min

    return args