import gzip
import logging
import os
import shutil
import zipfile
import requests
import torch
import urllib3
import numpy as np
from clint.textui import progress
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler


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


def load_split_train_test(datadir, valid_size=.2, batch_size=64, img_resize=224):
    train_transforms = transforms.Compose([transforms.Resize(img_resize), transforms.ToTensor(), ])
    test_transforms = transforms.Compose([transforms.Resize(img_resize), transforms.ToTensor(), ])
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
