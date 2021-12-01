import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import types
from configs.general_configs import (
    TRAIN_DATA_DIR_PATH,
    BAR_HEIGHT,
)


def load_image(image_file):
    def _preprocessing_func(image):
        img = image[:-BAR_HEIGHT]
        img = tf.image.random_crop(img, size=128)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
        if img.shape[2] == 3:
            img = tf.image.rgb_to_grayscale(img)
        return img

    # 1) Decode the path
    if isinstance(image_file, bytes):
        image_file = image_file.decode('utf-8')

    # 2) Read the image
    img = cv2.imread(image_file)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    img = _preprocessing_func(image=img)
    img = tf.cast(img, tf.float32)
    # 3) Get the label
    label = tf.strings.split(image_file, "/")[-1]

    label = tf.strings.substr(label, pos=0, len=1)
    label = tf.strings.to_number(label, out_type=tf.float32)
    label = tf.cast(label, tf.float32)

    return img, label


def configure_shapes(images, labels):
    images.set_shape(INPUT_IMAGE_SHAPE)
    labels.set_shape([])
    return images, labels


def get_val_ds(dataset):
    dataset = dataset.shuffle(buffer_size=1000)
    ds_size = dataset.cardinality().numpy()
    n_val = int(VAL_PROP * ds_size)

    return dataset.take(n_val)


def get_dataset_from_tiff(input_image_shape, batch_size, validation_split):

    global INPUT_IMAGE_SHAPE
    INPUT_IMAGE_SHAPE = input_image_shape

    global BATCH_SIZE
    BATCH_SIZE = batch_size

    global VAL_PROP
    VAL_PROP = validation_split

    # 1) Create the global dataset
    train_ds = tf.data.Dataset.list_files(str(TRAIN_DATA_DIR_PATH / '*.tiff'))
    n_samples = train_ds.cardinality().numpy()
    print(f'- Number of train samples: {n_samples}')

    # 2) Split the dataset into train and validation
    val_ds = get_val_ds(dataset=train_ds)
    n_val = val_ds.cardinality().numpy()

    # 2.1) Create the train dataset
    train_ds = train_ds.map(lambda x: tf.numpy_function(load_image, [x], (tf.float32, tf.float32)))


    train_ds = train_ds.map(configure_shapes)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(buffer_size=10*n_samples, reshuffle_each_iteration=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()

    # 2.2) Create the validation dataset
    val_ds = val_ds.map(lambda x: tf.numpy_function(load_image, [x], (tf.float32, tf.float32)))
    val_ds = val_ds.map(configure_shapes)
    val_ds = val_ds.batch(4)
    val_ds = val_ds.shuffle(buffer_size=1000)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.repeat()

    print(f'- Number of validation samples ({100*VAL_PROP:.2f}%): {n_val}')
    return train_ds, val_ds


class KNNDataLoader(tf.keras.utils.Sequence):
    def __init__(self, knn_image_files: pd.DataFrame, preprocessing_func: types.FunctionType, batch_size: int, crops_per_batch: int, val_prop: float = 0.0, shuffle: bool = True):
        self.n_files = knn_image_files.shape[0]
        self.knn_image_files = knn_image_files
        self.preprocessing_func = preprocessing_func
        self.batch_size = batch_size
        self.crops_per_batch = crops_per_batch
        self.val_prop = val_prop
        self.shuffle = shuffle

        print(f'''
        - Train files: {self.knn_image_files.shape[0]}
        ''')

    def __len__(self):
        return int(np.floor(self.crops_per_batch * self.n_files / self.batch_size))

    def __getitem__(self, index):
        # - Get the indices of the data in the range of current index
        btch_idxs = np.random.choice(np.arange(self.n_files), self.batch_size, replace=False)
        # print(btch_idxs.shape)
        return self._get_batch(knn_batch_df = self.knn_image_files.loc[btch_idxs])

    def _get_batch(self, knn_batch_df: pd.DataFrame):
        X_btch= []
        # 1) For each file in the files chosen for it
        for file_idx in knn_batch_df.index:
            btch_item = []
            # 2) Add the original image
            img_file = knn_batch_df.loc[file_idx, 'file']
            x, _ = load_image(image_file=img_file)
            btch_item.append(x)
            # 3) Add each of the neighbors of the original image
            for ngbr_file in knn_batch_df.loc[file_idx, 'neighbors']:
                ngbr, _ = load_image(image_file=ngbr_file)
                btch_item.append(ngbr)
            X_btch.append(btch_item)

        X_btch = np.array(X_btch, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(X_btch)

        return tf.convert_to_tensor(X_btch, dtype=tf.float32)
