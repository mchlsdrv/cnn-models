import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow
by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__=='__name__':
    print(tf.config.list_physical_devices('GPU'))
    (X, y), (X_test, y_test) = cifar10.load_data()
    X, X_test = X.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
    w, h, c = X.shape[1], X.shape[2], X.shape[3]
    print(w, h, c)

    # 1) Sequential Method
    model = keras.Sequential(
        [
            keras.Input(shape=(w, h, c)),
            layers.Conv2D(32, 3, padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPool2D(),  # as the default pool_size == (2, 2) we can omit it
            layers.Conv2D(128, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10),
        ]
    )

    model.summary()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    tb_callback = callbacks.TensorBoard(
        log_dir='D:\\NN\\TF\\logs/',
        histogram_freq=1
    )

    model.fit(
        X,
        y,
        batch_size=64,
        epochs=10,
        callbacks=[tb_callback],
        # verbose=2  # wont show the progress bar, but will print the results on each epoch
    )

    model.evaluate(
        X_test,
        y_test,
        batch_size=64
    )

    inp = model.input
    print(inp)
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([model.input], [out]) for out in outputs]

    x = np.random.rand(1, 32, 32, 3)
    outs = [func([x]) for func in functors]
    outs[1][0].shape
    # ----------------
    partial_model = keras.Model(model.inputs, model.layers[1].output)

    x = np.random.rand(1, 32, 32, 3)
    output_train = partial_model([x], training=True)   # runs the model in training mode
    output_test = partial_model([x], training=False)   # runs the model in test mode

    output_test
