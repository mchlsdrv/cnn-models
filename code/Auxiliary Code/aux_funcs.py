import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from os import (
    makedirs,
)
import pathlib
from pathlib import (
    Path,
    WindowsPath,
)
from datetime import datetime


# Global Functions
def sigmoid(X, a, b):
    return 1/(1+np.exp(a*(X-b)))


def shuffle(a, b):
    a_b = np.array(list(zip(a, b)))
    np.random.shuffle(a_b)
    return a_b[:, 0], a_b[:, 1]


def get_image_file_pathes(root_dir_path: pathlib.Path):
    image_file_pathes = []
    for root, dirs, files in os.walk(root_dir_path):
        for file in files:
            image_file_pathes.append(str(Path(root) / file))
    return image_file_pathes


def get_specs():
    specs = f'''
> Specs:
    - {subprocess.check_output(['bash', '-c', 'nvidia-smi -L']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'lscpu |grep "Model name"']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'lscpu | grep "Socket(s):"']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'lscpu | grep "Core(s) per socket"']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'lscpu | grep "Thread(s) per core"']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'lscpu | grep "L3 cache"']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'lscpu | grep MHz']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'cat /proc/meminfo | grep "MemAvailable"']).decode("utf-8")}
    - {subprocess.check_output(['bash', '-c', 'df -h / | awk "{print $4}"']).decode("utf-8")}
    '''
    print(specs)


def get_mem_usage(object):
    mem_in_bytes = sys.getsizeof(object)
    kilo = 1000
    mega = 1000000
    giga = 1000000000
    tera = 1000000000000
    if mem_in_bytes > tera:
        hr_mem_str = f'{mem_in_bytes // tera}.{mem_in_bytes % tera:.0f}[TB]'
    elif mem_in_bytes > giga:
        hr_mem_str = f'{mem_in_bytes // giga}.{mem_in_bytes % giga:.0f}[GB]'
    elif mem_in_bytes > mega:
        hr_mem_str = f'{mem_in_bytes // mega}.{mem_in_bytes % mega:.0f}[MB]'
    elif mem_in_bytes > kilo:
        hr_mem_str = f'{mem_in_bytes // kilo}.{mem_in_bytes % kilo:.0f}[KB]'
    else:
        hr_mem_str = f'{mem_in_bytes}[Bytes]'
    return hr_mem_str


def get_ts():
    def _clean_ts(ts):
        ts = ts[::-1]
        ts = ts[ts.index('.')+1:]
        ts = ts[::-1]
        ts = ts.replace(':', '_')
        return ts
    ts = str(datetime.now())
    return _clean_ts(ts)


def check_dir(dir_path):
    dir_ok = False
    if isinstance(dir_path, Path) or isinstance(dir_path, WindowsPath):
        if not dir_path.is_dir():
            makedirs(dir_path)
        dir_ok = True
        print(f'<info> The \'{dir_path}\' is valid!')
    else:
        print(f'<X> ERROR in check_dir: The path to save_dir ({dir_path}) is not of type \'Path\' but of type {type(dir_path)}!')

    return dir_ok


def get_run_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds - hours * 3600) // 60)
    residual_seconds = int(seconds - hours * 3600 - minutes * 60)
    return f'{hours}:{minutes}:{residual_seconds}'


def get_unbiased_std(std_arr):
    unbised_std = np.sum(std_arr) / (std_arr.shape[0] - 1)
    if np.isinf(unbised_std):
        unbised_std = 0.
    return unbised_std
