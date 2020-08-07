import gzip
import logging
import os
import shutil
import zipfile
import requests
import urllib3
from clint.textui import progress


def unzip_file(file, out_folder):
    assert file.endswith('.zip')
    os.makedirs(os.path.dirname(out_folder), exist_ok=True)
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall(out_folder)
    zip_file.close()


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
