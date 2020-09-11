import gzip
import logging
import os
import shutil
import zipfile
import requests
import torch
import urllib3
import numpy as np
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cnn_drone_net_transforms
from clint.textui import progress
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import interpolate


def unzip_file(file, out_folder):
    assert file.endswith('.zip')
    logger = logging.getLogger()
    logger.info(f'Unzipping file {file} to {out_folder}')
    os.makedirs(os.path.dirname(out_folder), exist_ok=True)
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall(out_folder)
    zip_file.close()


def download_data(url, out_path):
    logger = logging.getLogger()
    logger.info(f'Downloading contents from {url} to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True)
    with open(out_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()


def download_file(url, out_path):
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as out:
        while True:
            data = r.read(4096)
            if not data:
                break
            out.write(data)

    r.release_conn()


def interpolate_line(x, y, new_step=0.1):
    x = np.arange(0, len(y))
    f = interpolate.make_interp_spline(x, y)
    x_new = np.arange(0, len(y), new_step)
    y_new = f(x_new)

    return x_new, y_new

def load_split_train_test(datadir, valid_size=.4, batch_size=64, img_resize=224):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        # cnn_drone_net_transforms.RandomGaussianNoise(0., 1.),
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_loader, test_loader


def load_dataset(datadir, batch_size=64, img_resize=224):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(datadir, transform=data_transforms)
    data_len = len(data)
    indices = list(range(data_len))
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    dataset_loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataset_loader

def load_validation_dataset(datadir, batch_size=64, img_resize=224):
    data_transforms = transforms.Compose([
        # cnn_drone_net_transforms.RandomGaussianNoise(0., 1.),
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(datadir, transform=data_transforms)
    data_len = len(data)
    indices = list(range(data_len))
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    dataset_loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataset_loader


def plot_to_tensorboard(tag, writer, fig, step, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = np.swapaxes(img, 0, 2)
    writer.add_image(tag, img, step)
