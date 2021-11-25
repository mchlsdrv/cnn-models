import os
from pathlib import Path
import io
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2


def get_patch_df(image_file, patch_height, patch_width):
    assert image_file.is_file(), f'No file \'{image_file}\' was found!'

    img = np.expand_dims(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE), axis=-1)
    df = pd.DataFrame(columns=['file', 'image'])
    img_h, img_w, _ = img.shape
    for h in range(0, img_h, patch_height):
        for w in range(0, img_w, patch_width):
            patch = img[h:h+patch_height, w:w+patch_width, :]
            if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                df = df.append(dict(file=str(image_file), image=patch), ignore_index=True)
    return df


def transform_images(images_root_dir, model, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'mean_transform'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:

            # get the patches
            patches_df = get_patch_df(image_file=Path(f'{root}/{file}'), patch_height=patch_height, patch_width=patch_width)

            # get the mean patch transform
            patch_transforms = list()
            for patch in patches_df.loc[:, 'image'].values:
                patch_transforms.append(model(np.expand_dims(patch, axis=0)) if len(patch.shape) < 4 else model(patch))
            patch_transforms = np.array(patch_transforms)
            mean_transform = patch_transforms.mean(axis=0)[0, :]
            df = df.append(dict(file=f'{root}/{file}', mean_transform=mean_transform), ignore_index=True)
    return df

# def transform_images(images_root_dir, model, patch_height, patch_width):
#     df = pd.DataFrame(columns=['file', 'image'])
#     for root, dirs, files in os.walk(images_root_dir):
#         for file in files:
#             df = df.append(get_patch_df(image_file=Path(f'{root}/{file}'), patch_height=patch_height, patch_width=patch_width), ignore_index=True)
#     df.loc[:, 'vector'] = df.loc[:, 'image'].apply(lambda x: model(np.expand_dims(x, axis=0)) if len(x.shape) < 4 else model(x))
#     return df


def get_knn_files(X, files, k):
    # Detect the k nearest neighbors
    nbrs_pred = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    nbrs_files = list()
    for idx, (file, x) in enumerate(zip(files, X)):
        _, nbrs_idxs = nbrs_pred.kneighbors(np.expand_dims(x, axis=0))
        nbrs_files.append(files[nbrs_idxs])
    return nbrs_files


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def get_file_type(file_name: str):
    file_type = None
    if isinstance(file_name, str):
        dot_idx = file_name.find('.')
        if dot_idx > -1:
            file_type = file_name[dot_idx + 1:]
    return file_type


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def launch_tensor_board(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th
