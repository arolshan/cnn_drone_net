import argparse
import json
import logging
import os
import time
import cnn_drone_net_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.pyplot import figure
from sklearn.metrics import f1_score
from torch import nn
from glob import glob
from cnn_drone_net_consts import *

try:
    from types import SimpleNamespace as Namespace
except ImportError:
    # Python 2.x fallback
    from argparse import Namespace


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
    parser.add_argument("-i", "--input_args_dir", required=False, default=None, type=str,
                        help="If this is set, we run multiple experiements based on the JSON files in given directory.")
    parser.add_argument("-o", "--output_path", required=False, default=DEFAULT_OUT_PATH, type=str, help="Output path")
    parser.add_argument("-b", "--batch_size", type=int, default='8', help="Number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default='3', help="Number of epochs")
    parser.add_argument("--print_rate", type=int, default='1', help="Print every number of steps")
    parser.add_argument("--cuda_device", type=int, default='0', help="CUDA device id")
    parser.add_argument("--lr", type=float, default='2e-3', help="Learning rate of adam")
    parser.add_argument("--dropout", type=float, default='0.2', help="Model dropout probability")
    parser.add_argument("--with_cuda", type=str2bool, nargs='?', const=True, default='True', help="Training with CUDA: true, or false")
    parser.add_argument("--download_data", type=str2bool, nargs='?', const=False, default='True', help="Download the data")
    parser.add_argument("--train_set", type=str, default='GoogleEarth', help="Which set to train with. Possible values 'GoogleEarth', 'Satellite', 'UAV'")
    parser.add_argument("--model", type=str, default='Resnet50', help="Which CNN to run. Possible values 'Resnet50', 'VGG16', 'Mobilenet_V2'")
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
    if (args.train_set == TRAIN_SET_UAV):
        train_loader, val_loader = cnn_drone_net_utils.load_split_train_test(UAV_DATA_PATH, batch_size=args.batch_size)
    else:
        if (args.train_set == TRAIN_SET_SAT):
            train_data_dir = GE_DATA_PATH  # Possible values: ['GE_DATA_PATH', 'SAT_DATA_PATH']

        train_loader = cnn_drone_net_utils.load_dataset(train_data_dir, batch_size=args.batch_size)
        logging.info(f'Initialized train set loader from directory {train_data_dir}, classes: {train_loader.dataset.classes}')

        val_data_dir = UAV_DATA_PATH
        val_loader = cnn_drone_net_utils.load_validation_dataset(val_data_dir, batch_size=args.batch_size)
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

    device = torch.device(f"cuda:{args.cuda_device}" if args.with_cuda else "cpu")
    model, optimizer = cnn_drone_net_utils.get_model_optimizer(args)
    criterion = nn.NLLLoss()
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
    plt.savefig(f'{args.output_path}/train_loss.png')
    plt.clf()

    fig = figure(num=None, figsize=(13, 13))
    plt.plot(val_losses, label='Validation loss')
    plt.savefig(f'{args.output_path}/val_loss.png')
    plt.clf()

    fig = figure(num=None, figsize=(13, 13))
    plt.plot(f1_scores, label='F1 score')
    plt.savefig(f'{args.output_path}/f1_score.png')
    plt.clf()

    with open(f'{args.output_path}/losses.json', 'w') as losses_json_file:
        losses = {
            "validation": val_losses,
            "train": train_losses,
            "f1_score": f1_scores
        }
        json.dump(losses, losses_json_file)


def main():
    args_cmd = extract_args()
    if (args_cmd.input_args_dir != None):
        args_list = []
        path = args_cmd.input_args_dir
        print(f'Extracting possible args from directory {path}')
        for json_file_name in glob(f'{args_cmd.input_args_dir}/*.json'):
            print(f'Loading json file {json_file_name}')
            with open(json_file_name, 'r') as json_file:
                args_list += [json.load(json_file, object_hook=lambda d: Namespace(**d))]
    else:
        print(f'Extracting args from command line')
        args_list = [args_cmd]

    print(f'Running on {len(args_list)} possible args')
    for args in args_list:
        create_dirs(args)
        dump_args(args)
        setup_logging(args)
        download_data_unzip(args)
        train(args)

    # save graphs
    print_graphs_results(args_list)


def print_graphs_results(args_list):
    print("Printing graphs")
    val_losses_per_args = []
    train_losses_per_args = []
    f1_scores_per_args = []
    for args in args_list:
        losses_file = f'{args.output_path}/losses.json'
        with open(losses_file, 'r') as json_file:
            dict = json.load(json_file)
            val_losses_per_args += [dict["validation"]]
            train_losses_per_args += [dict["train"]]
            f1_scores_per_args += [dict["f1_score"]]
    i = 0
    fig = figure(num=None, figsize=(30, 15))
    fig.suptitle('Validation loss', fontsize=15)
    num_steps = min(len(v) for v in val_losses_per_args)
    min_val_per_args = {}
    for (val_losses, args) in zip(val_losses_per_args, args_list):
        min_val_per_args[f'Model: {args.model}, Train: {args.train_set}'] = min(val_losses)
        losses = val_losses[:num_steps:1]
        x_new, y_new = cnn_drone_net_utils.interpolate_line(np.arange(0, len(losses)), losses, 0.1)
        plt.plot(x_new, y_new, label=f'Model: {args.model}, Train: {args.train_set}')
        i += 1
    plt.xlabel('step', labelpad=10, fontsize=25)
    plt.ylabel('loss', labelpad=10, fontsize=25, rotation=0)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(loc="upper right", title="Runs", frameon=True, prop={'size': 30})
    plt.savefig(f'output/val_losses.png')
    plt.clf()
    print(min_val_per_args)
    i = 0
    fig = figure(num=None, figsize=(30, 15))
    fig.suptitle('F1 score', fontsize=15)
    num_steps = min(len(v) for v in f1_scores_per_args)
    min_val_per_args = {}
    for (f1_scores, args) in zip(f1_scores_per_args, args_list):
        min_val_per_args[f'Model: {args.model}, Train: {args.train_set}'] = min(f1_scores)
        f1 = f1_scores[:num_steps:3]
        x_new, y_new = cnn_drone_net_utils.interpolate_line(np.arange(0, len(f1)), f1, 0.1)
        plt.plot(x_new, y_new, label=f'Model: {args.model}, Train: {args.train_set}')
        i += 1
    plt.xlabel('steps/3', labelpad=10, fontsize=25)
    plt.ylabel('F1', labelpad=10, fontsize=25, rotation=0)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(loc="lower right", title="Runs", frameon=True, prop={'size': 30})
    plt.savefig(f'output/f1_scores.png')
    plt.clf()
    print("Loss per experiments:")
    print(min_val_per_args)


if __name__ == "__main__":
    main()
