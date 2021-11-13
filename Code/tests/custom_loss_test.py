import os
import yaml
from pathlib import Path
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

import aux_code.aux_funcs
from models.cnn import ConvModel
from callbacks.tensorboard_callbacks import ConvLayerVis

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow
by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = False
RESNET_CONFIGS_FILE_PATH = Path('./configs/resnet_configs.yml')

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    with RESNET_CONFIGS_FILE_PATH.open(mode='r') as f:
        resnet_configs = yaml.safe_load(f.read())

    # Load the data
    (X, y), (X_test, y_test) = cifar10.load_data()
    X, X_test = X.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
    n, w, h, c = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    n_test, w_test, h_test, c_test = X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]

    print(f'''
Dataset Stats:
    Number of train images: {n}
    Dimensions:
        > Train:
            width = {w}, height = {h}, channels = {c}
        > Test:
            width = {w_test}, height = {h_test}, channels = {c_test}
    ''')

    # Model with keras.Sequential
    model = ConvModel(input_shape=(w, h, c))
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=3e-4), metrics=['accuracy'])

    log_dir = f'./logs/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            write_images=True
        ),
        ConvLayerVis(
            X=X[0],
            input_layer=model.model.input,
            layers=model.model.layers,
            figure_configs=dict(rows=5, cols=5, figsize=(25, 25), cmap='gray'),
            log_dir=f'{log_dir}/train',
            log_interval=3
        )
    ]
    model.fit(
        X,
        y,
        validation_split=0.1,
        batch_size=64,
        epochs=15,
        callbacks=callbacks
    )

    aux_code.aux_funcs.launch_tb(log_dir=log_dir)
    X.shape
    tf.Tensor(X).reshape((-1, 32, 32, 3))
    t = tf.convert_to_tensor(X)
    t.shape

    a = np.arange(10)
    for i in range(20):
        print(a[np.random.randint(0, 10, 100)])
