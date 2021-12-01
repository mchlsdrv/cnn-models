import os
import types
import pathlib
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
import matplotlib.pyplot as plt
from importlib import reload
# Local imports
os.chdir('C:\\Users\\mchls\\Desktop\\Projects\\Nano-Scout\\Code')

from models.cnn import ConvModel
from data_loaders import data_funcs

DATA_PATH = Path('C:/Users/mchls/Desktop/Projects/Nano-Scout/Data/Test Images - 2 classes/train/10,000x - 89')
DATA_PATH.is_dir()
BATCH_SIZE = 32
INPUT_IMAGE_SHAPE = (128, 128, 1)
CENTRAL_CROP_PROP = .7
CROP_SHAPE = INPUT_IMAGE_SHAPE
BRIGHTNESS_DELTA = 0.1
CONTRAST = (0.4, 0.6)
BAR_HEIGHT = 70
# INPUT_IMAGE_SHAPE = (64, 64, 1)
EPSILON = 1e-7


def preprocessing_func(image):
    img = image[:-BAR_HEIGHT]
    img = tf.image.random_crop(img, INPUT_IMAGE_SHAPE)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
    if img.shape[2] == 3:
        img = tf.image.rgb_to_grayscale(img)
    return img

def load_image(image_file, get_label=True):
    def _preprocessing_func(image):
        img = image[:-BAR_HEIGHT]
        img = tf.image.random_crop(img, INPUT_IMAGE_SHAPE)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
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
    label = None
    if get_label:
        label = tf.strings.split(image_file, "/")[-1]

        label = tf.strings.substr(label, pos=0, len=1)
        label = tf.strings.to_number(label, out_type=tf.float32)
        label = tf.cast(label, tf.float32)

    return img, label


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
            # 3) Add each of the neighbors of the original image (first ne)
            for ngbr_file in knn_batch_df.loc[file_idx, 'neighbors']:
                # ngbr = self.preprocessing_func(cv2.imread(ngbr_file))
                ngbr, _ = load_image(ngbr_file, get_label=False)
                btch_item.append(ngbr)
            X_btch.append(btch_item)

        X_btch = np.array(X_btch, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(X_btch)

        return tf.convert_to_tensor(X_btch, dtype=tf.float32)


class SCANLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-7

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        btch_sz = y_true.shape[0]
        norm = tf.cast(tf.constant(1/btch_sz), y_true.dtype)

        # I) The consistancy
        dot_prod = tf.tensordot(y_true, tf.transpose(y_pred), axes=1)
        diag_mask = tf.cast(tf.eye(btch_sz), dot_prod.dtype)
        btch_dot_prod = tf.math.reduce_sum(dot_prod * diag_mask, 0)
        btch_dot_prod_log = tf.math.log(btch_dot_prod + self.epsilon)
        mean_btch_dot_prod_log = norm * btch_dot_prod_log

        # II) The entropy
        mean_class_prob = norm * tf.reduce_sum(y_pred, 0)
        mean_entropy = tf.math.reduce_sum(mean_class_prob * tf.math.log(mean_class_prob + self.epsilon))

        loss = -mean_btch_dot_prod_log + mean_entropy

        return loss


class ConvModel(keras.Model):
    def __init__(self, net_name: str, input_shape: tuple):
        super().__init__()
        self.net_name = net_name
        self.input_image_shape = input_shape
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(),
            layers.Conv2D(64, 5),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(),
            layers.Conv2D(128, 3, kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(10)
        ])

    def call(self, inputs):
        return self.model(inputs)

    def train_step(self, data):
        # Get the image only (the label is irrelevant)
        X = data
        print(X.shape)
        n_classes = 0
        with tf.GradientTape() as tape:
            consistancy_loss = entropy_loss = 0.0
            x_emds = list()
            # For each sample in the current batch
            # for x in X:
            n_classes = x[0].shape[-1]
            # Add the embeding of the original image
            X_emb = self(X, training=True)
            X_emds.append(X_emb)

            # Get the neighbors of the original image
            N_x = x[1:]

            # For each neighbor of the original image do:
            for ngbr in N_x:
                # Add the embeding of the neighbor to the batch list
                ngbr_emd = self(ngbr, training=True)

                # Calculate the consistency part of the SCAN loss
                dot_prod = tf.cast(tf.tensordot(x_emb, tf.transpose(ngbr_emb), axes=1), dtype=tf.float16)
                consistancy_loss += tf.math.log(dot_prod + EPSILON)

                tf.math.log(mean_class_prob + 1e-7)

            # for x in X:
            #     n_classes = x[0].shape[-1]
            #     # Add the embeding of the original image
            #     x_emb = self(x[0], training=True)
            #     x_emds.append(x_emb)
            #
            #     # Get the neighbors of the original image
            #     N_x = x[1:]
            #
            #     # For each neighbor of the original image do:
            #     for ngbr in N_x:
            #         # Add the embeding of the neighbor to the batch list
            #         ngbr_emd = self(ngbr, training=True)
            #
            #         # Calculate the consistency part of the SCAN loss
            #         dot_prod = tf.cast(tf.tensordot(x_emb, tf.transpose(ngbr_emb), axes=1), dtype=tf.float16)
            #         consistancy_loss += tf.math.log(dot_prod + EPSILON)
            #
            #         tf.math.log(mean_class_prob + 1e-7)
            # Calculate the entropy loss
            mean_class_prob = 1/n_classes * tf.reduce_sum(x_emds, 0)
            entropy = mean_class_prob * tf.math.log(mean_class_prob + 1e-7)
            mean_entropy = tf.math.reduce_mean(entropy)

            loss = -consistancy_loss + entropy_loss

        # Calculate gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return the mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def summary(self):
        return self.model.summary()

    def save(self, save_path):
        self.model.save(save_path)


if __name__ == '__main__':
    # Model with keras.Sequential
    model = ConvModel(net_name='Conv', input_shape=INPUT_IMAGE_SHAPE)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )
    # model.model.summary()

    priors_df = pd.read_pickle('C:/Users/mchls/Desktop/Projects/Nano-Scout/Outputs/priors_knn_df.pkl')

    priors_df.loc[:, 'file'] = priors_df.loc[:, 'file'].apply(lambda file_path: str(DATA_PATH / file_path.split('/')[-1]))
    priors_df.loc[:, 'neighbors'] = priors_df.loc[:, 'neighbors'].apply(lambda file_paths: [str(DATA_PATH / file_path.split('/')[-1]) for file_path in file_paths[0]])
    # priors_df

    knn_seq = KNNDataLoader(
        knn_image_files = priors_df,
        preprocessing_func=preprocessing_func,
        batch_size = 32,
        val_prop = 0.0,
        crops_per_batch=100,
        shuffle = True
    )
    for X_btch in knn_seq:
        print(X_btch.shape)
        fig, axs = plt.subplots(1, 5, figsize=(50, 8))
        for idx, img in enumerate(X_btch[0]):
            axs[idx].imshow(X_btch[0][idx], cmap='gray')
        break
    # I) Data set

    model.fit(
        knn_seq,
        batch_size=32,
        steps_per_epoch=10,
        epochs=10,
    )

    # II) Data Generator
    dg = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=(0.95, 0.95),
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_last',
        validation_split=0.1,
        dtype=tf.float32
    )

    train_dg = dg.flow_from_directory(
        DATA_PATH / 'train',
        target_size=(128, 128),
        batch_size=32,
        color_mode='grayscale',
        class_mode='sparse',
        shuffle=True,
        subset='training'
    )
    model.fit(
        train_dg,
        batch_size=32,
        epochs=10,
        callbacks=callbacks
    )
    np.random.choice(np.arange(20), 10, replace=False)
    d = pd.DataFrame(dict(a=np.arange(10), b=np.arange(10)[::-1]))
    for n in d.loc[[1, 5, 2, 6]].loc[:, 'a']:
        print(n)
    img = tf.convert_to_tensor(cv2.imread(str(DATA_PATH / '1.tiff')))
    img.shape
    tf.image.random_crop(img, (10, 10, 1))  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.img
    y_true = tf.convert_to_tensor(np.arange(1, 6), dtype=tf.float16)
    y_true
    y_pred =tf.convert_to_tensor( np.arange(1, 6)[::-1], dtype=tf.float16)
    y_pred

    dot = tf.tensordot(y_true, tf.transpose(y_pred), axes=1)

    tf.math.log(dot)

    L = [y_true, y_pred]

    mean_class_prob = 1/5 * tf.reduce_sum(L, 0)
    mean_class_prob
    tf.math.log(mean_class_prob + 1e-7)
    entropy = mean_class_prob * tf.math.log(mean_class_prob + 1e-7)
    entropy
    mean_entropy = tf.math.reduce_mean(entropy)
    mean_entropy
