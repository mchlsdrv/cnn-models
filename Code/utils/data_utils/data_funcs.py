import os
import numpy as np
import copy
import pandas as pd
import pathlib
import tensorflow as tf
from functools import partial
from utils.train_utils import train_funcs
from utils.image_utils import image_funcs


def configure_shapes(images, labels, shape):
    images.set_shape(shape)
    labels.set_shape([])
    return images, labels


def get_val_ds(dataset, validation_split):
    dataset = dataset.shuffle(buffer_size=1024)
    ds_size = dataset.cardinality().numpy()
    n_val = int(validation_split * ds_size)

    return dataset.take(n_val)


def rename_files(files_dir_path: pathlib.Path):
    # 1) Rename the files to have consequent name
    idx = 1
    for root, folders, files in os.walk(files_dir_path):
        for file in files:
            file_type = file.split('.')[-1]
            os.rename(f'{root}/{file}', f'{root}/{idx}.{file_type}')
            idx += 1


def get_dataset_from_tiff(data_dir_path, input_image_shape, batch_size, validation_split):
    # 1) Create the global dataset
    # - Rename the fies to have running index as the name
    rename_files(files_dir_path=data_dir_path)

    train_ds = tf.data.Dataset.list_files(str(data_dir_path / '*.tiff'))
    n_samples = train_ds.cardinality().numpy()
    print(f'- Number of train samples: {n_samples}')

    # 2) Split the dataset into train and validation
    val_ds = get_val_ds(dataset=train_ds, validation_split=validation_split)
    n_val = val_ds.cardinality().numpy()

    # 2.1) Create the train dataset
    train_ds = train_ds.map(lambda x: tf.numpy_function(partial(image_funcs.get_crop, image_shape=input_image_shape, get_label=True), [x], (tf.float32, tf.float32)))


    train_ds = train_ds.map(partial(configure_shapes, shape=input_image_shape))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.shuffle(buffer_size=10*n_samples, reshuffle_each_iteration=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()

    # 2.2) Create the validation dataset
    val_ds = val_ds.map(lambda x: tf.numpy_function(partial(image_funcs.get_crop, image_shape=input_image_shape, get_label=True), [x], (tf.float32, tf.float32)))
    val_ds = val_ds.map(partial(configure_shapes, shape=input_image_shape))
    val_ds = val_ds.batch(4)
    val_ds = val_ds.shuffle(buffer_size=1000)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.repeat()

    print(f'- Number of validation samples ({100*validation_split:.2f}%): {n_val}')
    return train_ds, val_ds


class KNNDataLoader(tf.keras.utils.Sequence):
    def __init__(self, knn_image_files: pd.DataFrame, k: int, image_shape: tuple, batch_size: int, crops_per_batch: int, shuffle: bool = True):
        self.n_files = knn_image_files.shape[0]
        self.knn_image_files = copy.deepcopy(knn_image_files)
        self.k = k
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.crops_per_batch = crops_per_batch
        self.shuffle = shuffle

        print(f'''
        - Train files: {self.knn_image_files.shape[0]}
        ''')

    def get_sample(self):
        return self.__getitem__(index=-1)

    def __len__(self):
        return int(np.floor(self.crops_per_batch * self.n_files / self.batch_size))

    def __getitem__(self, index):
        # - Get the indices of the data in the range of current index
        btch_sz = self.batch_size if index > -1 else 1
        btch_idxs = np.random.choice(np.arange(self.n_files), btch_sz, replace=False)
        return self._get_batch(knn_batch_df = self.knn_image_files.loc[btch_idxs])

    def _get_batch(self, knn_batch_df: pd.DataFrame):
        D_btch = []
        N_btch = []
        # I) For each file in the files chosen for it
        for X_file_idx in knn_batch_df.index:
            X, _ = image_funcs.get_crop(
                image_file=knn_batch_df.loc[X_file_idx, 'file'],
                image_shape=self.image_shape,
                get_label=False
            )
            # - Add the original file to the batch
            D_btch.append(X)

            # II) Add each of the neighbors of the original image (first ne)
            # - If the number of the neighbors is greater than 1
            if self.k > 1:
                N_X_files = list(copy.deepcopy(knn_batch_df.loc[X_file_idx, 'neighbors'])[0])
                # - The  image at index 0 is the original image
                N_X_files.pop(0)
                N_X = []
                for N_X_file in N_X_files:
                    ngbr, _ = image_funcs.get_crop(
                        image_file=N_X_file,
                        image_shape=self.image_shape,
                        get_label=False
                    )
                    # - Collect the neighbors of the original image
                    N_X.append(ngbr)
                # - Add the collected neighbors to the neighbors batch
                N_btch.append(N_X)
            else:
                # - If we are interensted only in the closest neighbor
                ngbr, _ = image_funcs.get_crop(
                    image_file=knn_batch_df.loc[X_file_idx, 'neighbors'][0][1],
                    image_shape=self.image_shape,
                    get_label=False
                )
                # - Add the closest neighbor to the neighbors batch
                N_btch.append(ngbr)

        D_btch = np.array(D_btch, dtype=np.float32)
        N_btch = np.array(N_btch, dtype=np.float32)

        if self.shuffle:
            random_idxs = np.arange(D_btch.shape[0])
            np.random.shuffle(random_idxs)
            D_btch = D_btch[random_idxs]
            N_btch = N_btch[random_idxs]
        return tf.convert_to_tensor(D_btch, dtype=tf.float32), tf.convert_to_tensor(N_btch, dtype=tf.float32)


def get_knn_dataset(priors_knn_df: pd.DataFrame, k: int, val_prop: float, image_shape: tuple, train_batch_size: int, val_batch_size: int, crops_per_batch: int, shuffle: int = True):
    train_idxs, val_idxs = train_funcs.get_train_val_idxs(
        n_items=priors_knn_df.shape[0],
        val_prop=val_prop
    )

    train_knn_data_set = KNNDataLoader(
        knn_image_files=priors_knn_df.loc[train_idxs].reset_index(drop=True),
        k=k,
        image_shape=image_shape,
        batch_size=train_batch_size,
        crops_per_batch=crops_per_batch,
        shuffle=shuffle
    )

    val_knn_data_set = KNNDataLoader(
        knn_image_files=priors_knn_df.loc[val_idxs].reset_index(drop=True),
        k=k,
        image_shape=image_shape,
        batch_size=val_batch_size,
        crops_per_batch=crops_per_batch,
        shuffle=shuffle
    )
    return train_knn_data_set, val_knn_data_set
