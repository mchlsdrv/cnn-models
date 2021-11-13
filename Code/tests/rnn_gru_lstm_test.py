import os
from pathlib import WindowsPath
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    regularizers,
    optimizers,
    losses,
    datasets,
    callbacks,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__=='__main__':
    print(tf.config.list_physical_devices('GPU'))

    (X, y), (X_test, y_test) = datasets.mnist.load_data()

    X, X_test = X.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0

    # 1) RNN
    mdl = keras.Sequential()
    mdl.add(keras.Input(shape=(None, 28)))
    mdl.add(layers.SimpleRNN(256, return_sequences=True, activation='relu'))
    mdl.add(layers.SimpleRNN(256, activation='relu'))
    mdl.add(layers.Dense(10))

    # Equivalent to:
    # mdl = keras.Sequential(
    #     [
    #         layers.SimpleRNN(512, activation='relu'),
    #         layers.SimpleRNN(512, activation='relu'),
    #         layers.Dense(10)
    #     ]
    # )
    mdl.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    mdl.summary()

    tensorboard_callback = callbacks.TensorBoard(
        log_dir='D:\\NN\\TF\\logs/',
        histogram_freq=1
    )

    mdl.fit(
        X,
        y,
        batch_size=64,
        epochs=10,
        callbacks=[tensorboard_callback],
    )

    mdl.evaluate(
        X_test,
        y_test,
        batch_size=64,
    )

    # Save weights only - requires the model to be identical to the one from which the weights were saved from
    weights_path = 'D:\\NN\\TF\\weights/'
    mdl.save_weights(weights_path)
    mdl.load_weights(weights_path)
    mdl.fit(X, y, batch_size=64, epochs=5, verbose=2)

    # Save the entire model - may be deployed, as it saves everyting which is esential for the model execution (i.e., architecture, weights, optimizers' state etc.)
    model_path = 'D:\\NN\\TF\\model/'
    mdl.save_model(model_path)
    mdl_2 = keras.load_model(model_path)
    mdl.fit(X, y, batch_size=64, epochs=5, verbose=2)
    # 2) GRU


    # 3) LSTM
    # 3.1) Single Directional

    # 3.2) Bi-Directional
