import os
import io
import datetime as dt
import numpy as np
import pathlib
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2


# Local imports
os.getcwd()
os.chdir('C:\\Users\\mchls\\Desktop\\Projects\\Deep-Learning\\TF')

from models.cnn import ConvModel
from callbacks.tensorboard_callbacks import ConvLayerVis
from aux_code import aux_funcs

DATA_PATH = Path('C:/Users/mchls/Desktop/Projects/Nano-Scout/Data/Test Images - 2 classes/train/10,000x - 89')
DATA_PATH.is_dir()
BATCH_SIZE = 32
INPUT_IMAGE_SHAPE = (128, 128, 1)
CROP_SHAPE = INPUT_IMAGE_SHAPE
CENTRAL_CROP_PROP = .7
BRIGHTNESS_DELTA = 0.1
CONTRAST = (0.4, 0.6)


def preprocessing_func(image):
    img = tf.image.central_crop(image, CENTRAL_CROP_PROP)
    if img.shape[2] == 3:
        img = tf.image.rgb_to_grayscale(img)
    return img


def augment(image):
    img = tf.image.random_crop(image, size=CROP_SHAPE)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
    img = tf.image.random_brightness(img, max_delta=BRIGHTNESS_DELTA)  # Equivalent to adjust_brightness() using a delta randomly picked in the interval [-max_delta, max_delta)
    img = tf.image.random_contrast(img, lower=CONTRAST[0], upper=CONTRAST[1])  # Equivalent to adjust_contrast() but uses a contrast_factor randomly picked in the interval [lower, upper).
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img


def load_image(image_file):
    # 1) Decode the path
    image_file = image_file.decode('utf-8')

    # 2) Read the image
    img = cv2.imread(image_file)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    img = preprocessing_func(image=img)
    img = augment(img)
    img = tf.cast(img, tf.float32)
    img.set_shape(INPUT_IMAGE_SHAPE)
    # 3) Get the label
    label = tf.strings.split(image_file, "\\")[-1]
    label = tf.strings.substr(label, pos=0, len=1)
    label = tf.strings.to_number(label, out_type=tf.float32)
    label = tf.cast(label, tf.float32)
    label.set_shape([])
    return img, label


def create_dataset():
    # 1) Rename the files to have consequent name
    idx = 1
    for root, folders, files in os.walk(DATA_PATH / 'train'):
        for file in files:
            os.rename(f'{root}/{file}', f'{root}/{idx}.tiff')
            idx += 1

    idx = 1
    for root, folders, files in os.walk(DATA_PATH / 'test'):
        for file in files:
            os.rename(f'{root}/{file}', f'{root}/{idx}.tiff')
            idx += 1

    # # 2) Group the images with their labels
    # idx = 1
    # for root, folders, files in os.walk(DATA_PATH / 'train'):
    #     for file in files:
    #         label = f'{file[:file.index(".")]}'
    #
    #         # 1) Create a new folder for the images
    #         label_dir = f'{root}/{label}'
    #         # os.mkdir(label_dir)
    #
    #         # 2) Move the original image to the new folder
    #         old_location = f'{root}/{file}'
    #         new_location = f'{label_dir}/{file}'
    #
    #         # 3) Load the image
    #         img, label = load_image(old_location)
    #         print(label)
    #         os.remove(old_location)
    #         for n in range(N_AUGS):
    #             # 1) Augment the image
    #             aug_img, _ = augment(image=img, label=label)
    #             # 2) To save the augmented image we need to reshape it from (w, h, c) to (w, h)
    #             aug_img = aug_img.numpy().reshape(aug_img.shape[:2])
    #             # 3) Conver numpy array to PIL Image
    #             aug_img = Image.fromarray(aug_img)
    #             # 4) Save the image
    #             aug_img.save(f'{label_dir}/{label}_{n}.tiff')

def _fixup_shape(images, labels):
    images.set_shape([128, 128, 1])
    labels.set_shape([])
    return images, labels

if __name__=='__main__':
    # create_dataset()

    train_ds = tf.data.Dataset.list_files(str(DATA_PATH / '*.tiff'))
    train_ds = train_ds.map(lambda x: tf.numpy_function(load_image, [x], (tf.float32, tf.float32)))
    train_ds = train_ds.map(_fixup_shape)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()

    X, y = next(iter(train_ds))
    X.shape.as_list(), y.shape.as_list()
    type(X)
    # Model with keras.Sequential
    model = ConvModel(input_shape=INPUT_IMAGE_SHAPE)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
        # metrics=[tf.keras.metrics.Accuracy()]#'accuracy']
    )
    model.model.summary()
    train_log_dir = f'./logs/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/train_data'
    train_file_writer = tf.summary.create_file_writer(train_log_dir)

    feat_maps_callback = ConvLayerVis(
        X=X,
        figure_configs=dict(rows=8, cols=8, figsize=(15, 15), cmap='gray'),
        file_writer=train_file_writer,
        save_dir = Path('./filters')
    )

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=train_log_dir,
            write_images=True
        ),
        feat_maps_callback
    ]
    # - with dataset
    model.fit(
        train_ds,
        batch_size=32,
        steps_per_epoch=10,
        epochs=10,
        callbacks=callbacks
    )

    # - with data generator
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
        # train_dg,
        batch_size=32,
        # steps_per_epoch=10,
        epochs=10,
        callbacks=callbacks
    )
