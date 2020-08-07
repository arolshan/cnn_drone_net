import glob
import gzip
import json
import os
import shutil
import zipfile
import pandas as pd
import torch
import urllib3
import requests
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import List
from pandas import DataFrame
from typing import Dict, Tuple, Sequence
from sklearn.preprocessing import KBinsDiscretizer
from cnn_drone_net_consts import *
from sklearn.cluster import KMeans
from clint.textui import progress


def unzip_file(file, out_folder):
    assert file.endswith('.zip')
    os.makedirs(os.path.dirname(out_folder), exist_ok=True)
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall(out_folder)
    zip_file.close()


def decompress_file(input_path, out_path=None):
    assert input_path.endswith('.gz')
    if (out_path == None):
        out_path = input_path[:-3]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(input_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_data(url, out_path):
    logging.info(f'Downloading contents from {url} to {out_path}')
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
