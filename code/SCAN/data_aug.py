import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (layers, losses, optimizers, callbacks)
import matplotlib.pyplot as plt
import cv2

BASE_PATH = Path('D:\\Projects\\NanoScout\\code\\scan')

AUG_SAVE_PATH = BASE_PATH / 'output/aug'
if not AUG_SAVE_PATH.is_dir():
    os.makedirs(AUG_SAVE_PATH)


if __name__=='__main__':
    img = cv2.imread(str(BASE_PATH / 'img_1.tiff'))
    img = img[:1000, :1000]
    cv2.imwrite(str(AUG_SAVE_PATH / 'original.png'), img)
    plt.imshow(np.array(img))


    # KERAS AUGMENTATION
    resize_and_rescale = keras.Sequential(
        [
            layers.Resizing(1000, 1000),
            # layers.Rescaling(1./255)
        ]
    )
    # 1) Flip
    flip = keras.layers.RandomFlip('horizontal_and_vertical')
    flipped_img = np.array(flip(img))
    plt.imshow(flipped_img)

    # 2) Rotation
    rot = layers.RandomRotation(0.3)
    rot_img = np.array(rot(img))
    plt.imshow(rot_img)
    cv2.imwrite(str(AUG_SAVE_PATH / 'rot_img.png'), rot_img)
    print(rot_img.shape)
    # 3) crop
    crop = layers.RandomCrop(height=150, width=150)
    cropped_img = np.array(resize_and_rescale(crop(img)), dtype=np.int16)
    plt.imshow(cropped_img)
    cv2.imwrite(str(AUG_SAVE_PATH / 'crop_2.png'), cropped_img)
