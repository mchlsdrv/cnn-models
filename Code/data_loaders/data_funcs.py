import numpy as np
import tensorflow as tf
import cv2

from Code.configs.general_configs import (
    # INPUT_IMAGE_SHAPE,
    TRAIN_DATA_PATH,
    BAR_HEIGHT,
    # BATCH_SIZE,
    # CROP_SHAPE,
    # VAL_PROP,
)


def load_image(image_file):
    def _preprocessing_func(image):
        img = image[:-BAR_HEIGHT]
        img = tf.image.random_crop(img, size=INPUT_IMAGE_SHAPE)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
        if img.shape[2] == 3:
            img = tf.image.rgb_to_grayscale(img)
        return img

    # 1) Decode the path
    image_file = image_file.decode('utf-8')

    # 2) Read the image
    img = cv2.imread(image_file)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    img = _preprocessing_func(image=img)
    img = tf.cast(img, tf.float32)
    # 3) Get the label
    label = tf.strings.split(image_file, "\\")[-1]

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
    train_ds = tf.data.Dataset.list_files(str(TRAIN_DATA_PATH / '*.tiff'))
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
