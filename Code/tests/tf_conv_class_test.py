import os
import datetime as dt
import numpy as np
import pathlib
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Lambda

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow
by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ConvModel(keras.Model):
    def __init__(self, input_shape):
        super().__init__()
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



def strip_layer_name(layer_full_name, layer_index=''):
    layer_stripped_name = layer_full_name + layer_index
    if find_sub_string(layer_full_name, '_'):
        if find_sub_string(layer_full_name, 'conv'):
            layer_full_name_bw = layer_full_name[::-1]
            layer_stripped_name = layer_full_name[:layer_full_name_bw.index('_')] + layer_index
    return layer_stripped_name


class ConvLayerVis(keras.callbacks.Callback):
    def __init__(self, X, figure_configs, file_writer, save_dir: pathlib.Path = None):
        super().__init__()
        self.X = X
        self.file_writer = file_writer
        self.figure_configs = figure_configs
        self.save_dir = save_dir

    @staticmethod
    def find_sub_string(string: str, sub_string: str):
        return True if string.find(sub_string) > -1 else False

    @staticmethod
    def figure_to_image(figure):
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = tf.image.decode_png(buffer.get_value(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def on_training_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        output_layers = [layer for layer in self.model.model.layers if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]

        # 3) Build partial model
        partial_model = keras.Model(
            inputs=self.model.model.input,
            outputs=output_layers
        )

        # 4) Get the feature maps
        feature_maps = partial_model.predict(self.X)

        # 5) Plot
        rows, cols = self.figure_configs.get('rows'), self.figure_configs.get('cols')
        for feature_map in feature_maps:
            fig, ax = plt.subplots(rows, cols, figsize=figure_configs.get('figsize'))
            for row in range(rows):
                for col in range(cols):
                    ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=figure_configs.get('cmap'))

            plt.show()

            if isinstance(save_dir, pathlib.Path):
                if not save_dir.is_dir():
                    os.makedirs(save_dir)
                    fig.savefig(save_dir / f'./Epoch_{epoch}.png')

            with file_writer.as_default(f' feature map (epoch #{epoch})', figure_to_image(fig), step=epoch):
                tf.summary.image()

    def on_train_end(self, logs=None):
        pass


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    # Load the data
    (X, y), (X_test, y_test) = cifar10.load_data()
    X, X_test = X.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
    w, h, c = X.shape[1], X.shape[2], X.shape[3]
    print(w, h, c)

    # Model with keras.Sequential
    model = ConvModel(input_shape=(w, h, c))
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=3e-4), metrics=['accuracy'])

    train_log_dir = f'./logs/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/train_data'
    train_file_writer = tf.summary.create_file_writer(train_log_dir)

    feat_maps_callback = ConvLayerVis(
        X = X,
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
    model.fit(X, y, batch_size=64, epochs=10, callbacks=callbacks)
    model.evaluate(X_test, y_test, batch_size=64)
    model.model.summary()

    partial_model = keras.Model(
        inputs=model.model.input,
        outputs=[
            model.model.layers[0].output,
            model.model.layers[3].output,
            model.model.layers[6].output
        ]
    )

    x = np.random.randn(1, 32, 32, 3)

    feats = partial_model.predict(x)

    cols = 8
    rows = 8
    for feat in feats:
        fig, ax = plt.subplots(rows, cols, figsize=(15, 15))
        for row in range(rows):
            for col in range(cols):
                ax[row][col].imshow(feat[0, :, :, row+col], cmap='gray')
        plt.show()
