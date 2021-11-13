import os
import yaml
import io
import copy
import pathlib
from pathlib import Path
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.nn import relu
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.utils import Sequence, Progbar
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile
import types
import functools
import datetime as dt
from importlib import reload

os.getcwd()
project_root_dir = Path('C:\\Users\\mchls\\Desktop\\Projects\\Nano-Scout\\Code')
os.chdir(project_root_dir)

from data_loaders import data_funcs
from models import cnn
from losses import clustering_losses
from augmentations import clustering_augmentations
from callbacks import clustering_callbacks

DEBUG = False
'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Global Settings
INPUT_IMAGE_SHAPE = (128, 128, 1)
TRAIN_DATA_PATH = Path('C:\\Users\\mchls\\Desktop\\Projects\\Data\\antrax\\train\\10,000x - 48')
TRAIN_DATA_PATH.is_dir()

NET_CONFIGS_FILE_PATH = Path('C:\\Users\\mchls\\Desktop\\Projects\\Nano-Scout\\Code\\configs\\resnet_configs.yml')
NET_CONFIGS_FILE_PATH.is_file()

if __name__ == '__main__':
    # 1) Get the data
    reload(data_funcs);
    train_ds, val_ds = data_funcs.get_dataset_from_tiff()

    X_train, y_test = next(iter(train_ds))
    plt.imshow(X_train[0], cmap='gray')

    X_val, y_val = next(iter(val_ds))
    plt.imshow(X_val[0], cmap='gray')

    # 2) Build the model
    reload(cnn);
    reload(clustering_augmentations);
    reload(clustering_losses);
    with NET_CONFIGS_FILE_PATH.open(mode='r') as config_file:
        resnet_configs = yaml.safe_load(config_file)
    resnet_configs
    model = cnn.FeatureExtractionResNet(
        net_configs=resnet_configs,
        augmentations=clustering_augmentations.augmentations,
        similarity_loss=clustering_losses.cosine_similarity_loss
    )

    model.summary()

    # 3) Configure callbacks
    reload(clustering_callbacks);
    callbacks = clustering_callbacks.get_callbacks(
        model=model,
        X=next(iter(val_ds))[0][0]
    )

    # 4) Compile
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # 5) Fit feature extraction model
    N_EPOCHS = 5
    BATCH_SIZE = 32
    N_STEPS_PER_EPOCH = 10
    validation_steps = int(0.1*N_STEPS_PER_EPOCH) if int(0.1*N_STEPS_PER_EPOCH) > 0 else 1

    model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        steps_per_epoch=N_STEPS_PER_EPOCH,
        validation_steps=validation_steps,
        validation_freq=1, # [1, 100, 1500, ...] - validate on these epochs
        shuffle=True,
        callbacks=callbacks
    )
