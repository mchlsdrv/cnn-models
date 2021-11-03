import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import (
    Path
)

# LOCAL
from aux_funcs import (
    check_dir, 
    get_ts,
)


# Global Functions
def calc_btch_sz(X_sz, prop=.1, min_btch_sz=2, max_btch_sz=128):
    
    btch_sz = int(prop*X_sz)

    if btch_sz < min_btch_sz:
        btch_sz = min_btch_sz
    elif btch_sz > max_btch_sz:
        btch_sz = max_btch_sz

    return btch_sz

def early_stopping(losses, cnt, patience=10, memory=20):
    # Early stopping procedures
    stop_training = False
    if losses.shape[0] > 1 and losses[-1] >= losses[-2]:
        if cnt > patience:
            print(f"<info> Early stopping due to val loss not improving for {patience} epochs!")
            stop_training = True
        else:
            cnt += 1
    elif losses.shape[0] > 2*memory and losses[-memory:].mean() >= losses[-2*memory:-memory].mean():
        if cnt > patience:
            print(f"<info> Early stopping due to val mean (of {memory} iterations) loss not improving for {patience} epochs!")
            stop_training = True
        else:
            cnt += 1
    else:
        cnt = 0
    return stop_training, cnt


def get_cv_x(data, val_prop):
    data = np.array(data)

    n_samples = data.shape[0]
    
    data_idxs = np.arange(n_samples)

    n_val_samples = np.int16(val_prop * n_samples)

    val_cuts = np.arange(0, n_samples, n_val_samples)

    val_idxs_arr = np.array(
        [np.arange(val_start_idx, val_start_idx + n_val_samples) for val_start_idx in val_cuts]
    )

    cv_iters = n_samples // n_val_samples

    cv_val_data_arr = np.array(list(zip(np.arange(cv_iters), val_idxs_arr)))

    cv_data = []
    for cv_idx, val_idxs in cv_val_data_arr:
        cv_data.append((data[np.setdiff1d(data_idxs, val_idxs)], data[val_idxs]))
    print(f'> Running {cv_iters} fold Cross Validation')
    return np.array(cv_data)


def calc_conv_same_padding(input_size, kernel_size, stride):
    return (input_size*(stride-1) + kernel_size - stride) // 2


def calc_conv_out_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1


def pad_with_zeros(x1, x2):
    max_batch_sz = np.max([x1.shape[0], x2.shape[0]])
    max_channels = np.max([x1.shape[1], x2.shape[1]])
    max_widht = np.max([x1.shape[2], x2.shape[2]])
    max_height = np.max([x1.shape[3], x2.shape[3]])

    x1_zp = torch.tensor(np.zeros((max_batch_sz, max_channels, max_widht, max_height), dtype=np.float32))
    x1_zp[:x1.shape[0], :x1.shape[1], :x1.shape[2], :x1.shape[3]] = x1[:, :, :, :]

    x2_zp = torch.tensor(np.zeros((max_batch_sz, max_channels, max_widht, max_height), dtype=np.float32))
    x2_zp[:x2.shape[0], :x2.shape[1], :x2.shape[2], :x2.shape[3]] = x2[:, :, :, :]
    return x1_zp, x2_zp
