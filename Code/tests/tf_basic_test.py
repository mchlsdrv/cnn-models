import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow
by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Errors only


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    # dev = tf.

    (X, y), (X_test, y_test) = mnist.load_data()
    w, h = X.shape[1], X.shape[2]
    X, X_test = X.reshape(-1, w*h).astype(np.float32) / 255.0, X_test.reshape(-1, w*h).astype(np.float32) / 255.0

    mdl = keras.Sequential(
        [
            layers.Input(w*h),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(10),
        ]
    )
    mdl.summary()
    mdl.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            'accuracy'
        ],
    )

    mdl.fit(
        X,
        y,
        batch_size=32,
        epochs=10,
        verbose=2
    )

    mdl.evaluate(
        X_test,
        y_test,
        batch_size=32,
        verbose=2
    )
