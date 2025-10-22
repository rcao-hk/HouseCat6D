import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from utils.solver import test_func, get_logger
from provider.housecat6d_dataset import HouseCat6DTestDataset
from utils.evaluation_utils import evaluate_housecat

def get_parser():
    parser = argparse.ArgumentParser(
        description="VI-Net")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/housecat.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="REAL275",
                        help="[REAL275 | CAMERA25]")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=0,
                        help="test epoch")
    parser.add_argument("--result_dir",
                        type=str,
                        default="results",
                        help="")
    parser.add_argument("--depth_type", 
                        type=str,
                        default="raw",
                        help="[raw | restored | restored_conf]")
    parser.add_argument("--restored_depth_root",
                        type=str,
                        default="",
                        help="the root path of restored depth maps, only used when depth_type is restored")
    parser.add_argument("--conf_thres",
                        type=float,
                        default=0.1,
                        help="confidence threshold for restored_conf depth type")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dataset = args.dataset
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.depth_type = args.depth_type
    cfg.restored_depth_root = args.restored_depth_root
    cfg.conf_thres = args.conf_thres
    # cfg.log_dir = os.path.join('log', args.dataset)
    cfg.log_dir = cfg.test.log_dir
    cfg.save_path = os.path.join(cfg.log_dir, args.result_dir)
    if not os.path.isdir(cfg.save_path):
        os.makedirs(cfg.save_path)

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+"/test_{}_logger.log".format(cfg.test_epoch))
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    evaluate_housecat(cfg.save_path, logger)