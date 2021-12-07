import numpy as np
import copy
import types
import tensorflow as tf
import cv2
from functools import partial
from utils import aux_funcs
from image_utils import preprocessing_funcs


def get_crop(image_file, image_shape, get_label=True):

    # 1) Decode the images' path if it is represented in bytes instead of str
    if isinstance(image_file, bytes):
        image_file = image_file.decode('utf-8')

    # 2) Read the image
    img = cv2.imread(image_file)
    # if len(img.shape) < 3:
    #     img = np.expand_dims(img, axis=-1)

    # 3) Apply the preprocessing function
    img = preprocessing_funcs.preprocessing_func(image=img)

    # 4) Randomly crop the image
    img = tf.image.random_crop(img, size=image_shape)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
    img = tf.cast(img, tf.float32)

    # 5) Get the label
    label = None
    if get_label:
        label = tf.strings.split(image_file, "/")[-1]

        label = tf.strings.substr(label, pos=0, len=1)
        label = tf.strings.to_number(label, out_type=tf.float32)
        label = tf.cast(label, tf.float32)

    return img, label
