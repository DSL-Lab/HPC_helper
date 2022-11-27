import argparse
import logging
import os
import pdb
import shutil
import random
import sys
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.dist_training import get_ddp_save_flag


def parse_arguments():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description="Running Experiments")
    parser.add_argument('-l', '--log_level', type=str,
                        default='DEBUG', help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="A single line comment for the experiment")
    parser.add_argument('--dp', default=False, action='store_true',
                        help='To use DataParallel distributed learning.')
    parser.add_argument('--ddp', default=False, action='store_true',
                        help='To use DDP distributed learning')
    parser.add_argument('--ddp_gpu_ids', nargs='+', default=None,
                        help="A list of GPU IDs to run distributed learning")
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Training batch size.')
    parser.add_argument('--epoch', default=5, type=int,
                        help='Training epochs.')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Random seed.')

    args = parser.parse_args()

    # add log directory
    if args.dp:
        dist_status = 'dp'
    elif args.ddp:
        dist_status = 'ddp'
    else:
        dist_status = 'single_gpu'

    logdir_nm = dist_status + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(args.comment):
        logdir_nm += '_' + args.comment

    logdir = os.path.join('runs', logdir_nm)
    os.makedirs(logdir, exist_ok=True)

    args.logdir = logdir
    print('Args: \n', args)
    return args


def set_seed_and_logger(seed, logdir, log_level, comment, dist_helper):
    """
    Set up random seed number and global logger.
    """
    # Setup random seed
    if dist_helper.is_ddp:
        seed += dist.get_rank()
    else:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch numerical accuracy flags
    # reference: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False

    # Setup logger
    if dist_helper.is_ddp:
        log_file = os.path.join(logdir, "ddp_rank_{:02d}_".format(dist.get_rank()) + log_level.lower() + ".log")
    else:
        log_file = os.path.join(logdir, log_level.lower() + ".log")
    logger_format = comment + '| %(asctime)s %(message)s'
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format=logger_format,
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[
                            fh,
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger('matplotlib').setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.info('EXPERIMENT BEGIN: ' + comment)
    logging.info('logging into %s', log_file)

    # Setup tensorboard logger
    if get_ddp_save_flag():
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None
    return writer


def backup_code(logdir):
    if get_ddp_save_flag():
        code_path = os.path.join(logdir, 'code')
        dirs_to_save = ['utils']
        os.makedirs(code_path, exist_ok=True)

        # save_name = os.path.join(code_path, 'config.yaml')
        # yaml.dump(dict(config), open(save_name, 'w'), default_flow_style=False)

        os.system('cp ./*py ' + code_path)
        [shutil.copytree(os.path.join('./', this_dir), os.path.join(code_path, this_dir)) for this_dir in dirs_to_save]
