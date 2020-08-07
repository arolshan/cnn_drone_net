import argparse
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import time
import cnn_drone_net_utils as utils
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from cnn_drone_net_consts import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", required=False, default="./output", type=str, help="Output path")
    parser.add_argument("-hl", "--hidden_layers", type=int, default='1', help="Number of hidden layers in Fully connected")
    parser.add_argument("-hs", "--hidden_layer_sizes", nargs='+', default=[2048], help="Hidden Layer Output sizes")
    parser.add_argument("-b", "--batch_size", type=int, default='64', help="Number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default='3', help="Number of epochs")
    parser.add_argument("--cuda_device", type=int, default='0', help="CUDA device id")
    parser.add_argument("--lr", type=float, default='2e-3', help="Learning rate of adam")
    parser.add_argument("--seed", type=int, default='42', help="Download the data")
    parser.add_argument("--dropout", type=float, default='0.1', help="Model dropout probability")
    parser.add_argument("--dl_worker_count", type=int, default='4', help="Number of data loader workers")
    parser.add_argument("--subset", type=int, default='-1', help="Run only on subset of data")
    parser.add_argument("--with_cuda", type=str2bool, nargs='?', const=True, default='True', help="Training with CUDA: true, or false")
    parser.add_argument("--download_data", type=str2bool, nargs='?', const=False, default='True', help="Download the data")
    parser.add_argument("--data_parallel", type=str2bool, nargs='?', const=True, default='True', help="Run batches with data parallel")
    parser.add_argument("--inference", type=str2bool, nargs='?', const=True, default='False', help="Is inference mode, i.e. evaluate last model state without training")
    args = parser.parse_args()

    return args


def create_dirs(args):
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)


def dump_args(args):
    with open(f'{args.output_path}/args.json', 'w') as out:
        json.dump(args.__dict__, out, indent=4)


def setup_logging(args):
    logFormatter = logging.Formatter("[%(asctime)s, %(threadName)s, %(levelname)s] %(message)s")
    logging.basicConfig(level=logging.INFO)
    rootLogger = logging.getLogger()

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(f'{args.output_path}/out_{int(round(time.time() * 1000))}.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)


def download_data_unzip(args):
    if (not args.download_data):
        return

    zipped = f'{DATA_PATH}/{DATA_UAV_COMPRESSED_FILENAME}'
    utils.download_data(DATA_BLOB_UAV_COMPRESSED, out_path=zipped)
    utils.unzip_file(zipped, DATA_PATH)


def train(args):
    # TODO
    pass


def main():
    args = extract_args()
    create_dirs(args)
    dump_args(args)
    setup_logging(args)
    download_data_unzip(args)
    train(args)


if __name__ == "__main__":
    main()
