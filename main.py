import argparse
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import time
import cnn_drone_net_utils
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import time
import torch.nn.functional as F
import sys
import cnn_drone_net_utils
import torchvision
import torchvision.transforms as transforms
import cv2
import io
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from matplotlib.pyplot import figure
from torchvision import models
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
from cnn_drone_net_consts import *
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("-o", "--output_path", required=False, default=DEFAULT_OUT_PATH, type=str, help="Output path")
    parser.add_argument("-hl", "--hidden_layers", type=int, default='1', help="Number of hidden layers in Fully connected")
    parser.add_argument("-hs", "--hidden_layer_sizes", nargs='+', default=[2048], help="Hidden Layer Output sizes")
    parser.add_argument("-b", "--batch_size", type=int, default='8', help="Number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default='1', help="Number of epochs")
    parser.add_argument("--print_rate", type=int, default='10', help="Print every number of steps")
    parser.add_argument("--cuda_device", type=int, default='0', help="CUDA device id")
    parser.add_argument("--lr", type=float, default='2e-3', help="Learning rate of adam")
    parser.add_argument("--seed", type=int, default='42', help="Download the data")
    parser.add_argument("--dropout", type=float, default='0.2', help="Model dropout probability")
    parser.add_argument("--dl_worker_count", type=int, default='4', help="Number of data loader workers")
    parser.add_argument("--subset", type=int, default='-1', help="Run only on subset of data")
    parser.add_argument("--with_cuda", type=str2bool, nargs='?', const=True, default='True', help="Training with CUDA: true, or false")
    parser.add_argument("--download_data", type=str2bool, nargs='?', const=False, default='True', help="Download the data")
    parser.add_argument("--data_parallel", type=str2bool, nargs='?', const=True, default='True', help="Run batches with data parallel")
    parser.add_argument("--inference", type=str2bool, nargs='?', const=True, default='False', help="Is inference mode, i.e. evaluate last model state without training")
    parser.add_argument("--train_set", type=str, default='GoogleEarth', help="Which set to train with. Possible values 'GoogleEarth', 'Satellite'")
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

    download_uav_data = DATA_UAV_COMPRESSED_FILENAME not in os.listdir(DATA_PATH)
    download_ge_data = DATA_GE_COMPRESSED_FILENAME not in os.listdir(DATA_PATH)
    download_sat_data = DATA_SAT_COMPRESSED_FILENAME not in os.listdir(DATA_PATH)

    uav_zipped = f'{UAV_DATA_PATH}/{DATA_UAV_COMPRESSED_FILENAME}'
    ge_zipped = f'{GE_DATA_PATH}/{DATA_GE_COMPRESSED_FILENAME}'
    sat_zipped = f'{SAT_DATA_PATH}/{DATA_SAT_COMPRESSED_FILENAME}'

    if (download_uav_data):
        logging.info(f"Downloading data from {DATA_BLOB_UAV_COMPRESSED} and saving in {uav_zipped}")
        cnn_drone_net_utils.download_data(DATA_BLOB_UAV_COMPRESSED, out_path=uav_zipped)

    if (download_ge_data):
        logging.info(f"Downloading data from {DATA_BLOB_GE_COMPRESSED} and saving in {ge_zipped}")
        cnn_drone_net_utils.download_data(DATA_BLOB_GE_COMPRESSED, out_path=ge_zipped)

    if (download_sat_data):
        logging.info(f"Downloading data from {DATA_BLOB_SAT_COMPRESSED} and saving in {sat_zipped}")
        cnn_drone_net_utils.download_data(DATA_BLOB_SAT_COMPRESSED, out_path=sat_zipped)

    logging.info(f"Unzipping {uav_zipped} to directory {DATA_PATH}")
    cnn_drone_net_utils.unzip_file(uav_zipped, DATA_PATH)

    logging.info(f"Unzipping {ge_zipped} to directory {DATA_PATH}")
    cnn_drone_net_utils.unzip_file(ge_zipped, DATA_PATH)

    logging.info(f"Unzipping {sat_zipped} to directory {DATA_PATH}")
    cnn_drone_net_utils.unzip_file(sat_zipped, DATA_PATH)

    logging.info(f"{DATA_PATH} contents: {', '.join(os.listdir(DATA_PATH))}")


def train(args):
    train_data_dir = GE_DATA_PATH
    if (args.train_set == TRAIN_SET_SAT):
        train_data_dir = GE_DATA_PATH  # Possible values: ['GE_DATA_PATH', 'SAT_DATA_PATH']

    train_loader = cnn_drone_net_utils.load_dataset(train_data_dir, batch_size=args.batch_size)
    logging.info(f'Initialized train set loader from directory {train_data_dir}, classes: {train_loader.dataset.classes}')

    val_data_dir = UAV_DATA_PATH
    val_loader = cnn_drone_net_utils.load_dataset(val_data_dir, batch_size=args.batch_size)
    logging.info(f'Initialized validation set loader from directory {UAV_DATA_PATH}, classes: {val_loader.dataset.classes}')

    logging.info("Sampling training set:")
    examples = iter(train_loader)
    example_data, example_targets = examples.next()

    fig = figure(num=None, figsize=(10, 7.5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(torchvision.utils.make_grid(example_data[i], nrow=1).permute(1, 2, 0))
        plt.title(train_loader.dataset.classes[example_targets[i]])

    plt.savefig(f'{args.output_path}/train_sample.png')
    plt.close()

    logging.info("Sampling validation set:")
    examples = iter(val_loader)
    example_data, example_targets = examples.next()

    fig = figure(num=None, figsize=(10, 7.5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(torchvision.utils.make_grid(example_data[i], nrow=1).permute(1, 2, 0))
        plt.title(train_loader.dataset.classes[example_targets[i]])

    plt.savefig(f'{args.output_path}/validation_sample.png')
    plt.close()

    device = "cpu"
    if (args.with_cuda):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    logging.info(f'Loaded pretrained model: {models.resnet50.__name__}')
    logging.info("Freezing the pretrained model's parameters")
    for param in model.parameters():
        param.requires_grad = False

    logging.info("Initialzing the last fully connected layer of the pretrained model")
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(args.dropout),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    model.to(device)

    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = args.print_rate
    train_losses, val_losses, f1_scores = [], [], []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                f1 = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        true_labels = labels.view(*top_class.shape)
                        equals = top_class == true_labels
                        f1 += f1_score(true_labels.cpu(), top_class.cpu())
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                f1_scores.append(f1 / len(val_loader))
                train_losses.append(running_loss / print_every)
                val_losses.append(val_loss / len(val_loader))
                logging.info(f"Epoch {epoch + 1}/{epochs}.. "
                             f"Train loss: {running_loss / print_every:.3f}.. "
                             f"Validation loss: {val_loss / len(val_loader):.3f}.. "
                             f"Validation accuracy: {accuracy / len(val_loader):.3f} "
                             f"F1 score: {f1 / len(val_loader):.3f} "
                             f"Step: {steps}/{len(train_loader) * epochs}")
                running_loss = 0
                model.train()

    logging.info("Done training model")
    torch.save(model, f'{args.output_path}/model.pth')
    fig = figure(num=None, figsize=(13, 13))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.plot(f1_scores, label='F1 score')
    plt.legend(frameon=False)
    plt.savefig(f'{args.output_path}/out_graph.png')


def main():
    args = extract_args()
    create_dirs(args)
    dump_args(args)
    setup_logging(args)
    download_data_unzip(args)
    train(args)


if __name__ == "__main__":
    main()
