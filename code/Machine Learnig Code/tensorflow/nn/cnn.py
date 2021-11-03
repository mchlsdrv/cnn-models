import os
import pathlib
from pathlib import Path
import datetime as dt
import matplotlib.pyplot as plt
from shutil import copyfile
import types
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.nn import relu
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.datasets import cifar10

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResNet(keras.Model):
    class ResBlock(layers.Layer):
        def __init__(self, filters: tuple, kernel_sizes: tuple, strides: tuple = ((1, 1), (1, 1)), activations: tuple = ('relu', 'relu'), paddings: tuple = ('same', 'same'), dilation_rates: tuple = ((1, 1), (1, 1))):
            super().__init__()

            self.model = keras.Sequential([
            # # I) - Input Layer
            #         layers.Input(shape=input_shape),
            # II) - First conv block
                    layers.Conv2D(filters[0], kernel_sizes[0], strides=strides[0], activation=activations[0], padding=paddings[0], dilation_rate=dilation_rates[0]),
                    layers.BatchNormalization(),
            # III) - Second conv block
                    layers.Conv2D(filters[1], kernel_sizes[1], strides=strides[1], activation=None, padding=paddings[1], dilation_rate=dilation_rates[1]),
                    layers.BatchNormalization(),
            # IV) - Skip connection
                    layers.Conv2D(filters[1], 1, padding='same'),
                ]
            )

            # - Shortcut
            self.skip_connection_layer = layers.Add()
            self.identity_mapping_layer = layers.Conv2D(filters[1], 1, padding='same')

            # - Activation
            self.activation = layers.Activation(activations[1])

        def call(self, inputs, training=False):
            X = self.model(inputs)
            # > Skip connection
            # Depending on the output shape we'd use:
            # - input of the same number of channels if they are equall
            if X.shape[1:] == inputs.shape[1:]:
                X = self.skip_connection_layer(
                    [X, inputs]
                )
            # - perform a 1X1 convolution to increase the number of
            # channels to suit the output of the last Conv layer
            else:
                X = self.skip_connection_layer(
                    [X, self.identity_mapping_layer(inputs)]
                )

            return self.activation(X)

    def __init__(self, net_configs: dict):
        super().__init__()

        self.input_image_shape = net_configs.get('input_image_shape')
        # 1) Input layer
        self.model = keras.Sequential(
            [
                keras.Input(shape=self.input_image_shape),
                layers.Conv2D(**net_configs.get('conv2d_1')),
                layers.Conv2D(**net_configs.get('conv2d_2')),
                layers.MaxPool2D(**net_configs.get('max_pool_2d')),
            ]
        )

        # 2) ResBlocks
        res_blocks_configs = net_configs.get('res_blocks')

        conv2_block_configs = res_blocks_configs.get('conv2_block_configs')
        # conv2_block_configs.get('block_configs')['input_shape'] = (*self.input_image_shape[:-1] , net_configs.get('conv2d_2')['filters'])
        for idx in range(conv2_block_configs.get('n_blocks')):
            # print(idx, self.input_image_shape)
            self.model.add(self.ResBlock(**conv2_block_configs.get('block_configs')))
            # conv2_block_configs.get('block_configs')['input_shape'] = (*self.input_image_shape[:-1], conv2_block_configs.get('filter')[idx])

        conv3_block_configs = res_blocks_configs.get('conv3_block_configs')
        # The number of channels in the input is as the number of channels of the last Conv layer of Conv2 block
        # conv3_block_configs.get('block_configs')['input_shape'] = (*self.input_image_shape[:-1], conv2_block_configs.get('filters')[1])
        for idx in range(conv3_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv3_block_configs.get('block_configs')))
            # conv3_block_configs.get('block_configs')['input_shape'] = (*self.input_image_shape[:-1], conv3_block_configs.get('filters')[idx])

        conv4_block_configs = res_blocks_configs.get('conv4_block_configs')
        # The number of channels in the input is as the number of channels of the last Conv layer of Conv3 block
        # conv4_block_configs.get('block_configs')['input_shape'] = (self.input_image_shape[:-1], conv3_block_configs.get('filters')[1])
        for idx in range(conv4_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv4_block_configs.get('block_configs')))
            # conv4_block_configs.get('block_configs')['input_shape'] = (self.input_image_shape[:-1], conv4_block_configs.get('kernel_sizes')[idx])

        conv5_block_configs = res_blocks_configs.get('conv5_block_configs')
        # The number of channels in the input is as the number of channels of the last Conv layer of Conv4 block
        # conv5_block_configs.get('block_configs')['input_shape'] = (self.input_image_shape[:-1], conv4_block_configs.get('kernel_sizes')[1])
        for idx in range(conv5_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv5_block_configs.get('block_configs')))
            # conv5_block_configs.get('block_configs')['input_shape'] = (self.input_image_shape[:-1], conv5_block_configs.get('kernel_sizes')[idx])

        self.model.add(layers.Conv2D(**net_configs.get('conv2d_3')))

        self.model.add(layers.GlobalAveragePooling2D())

        self.model.add(layers.Dense(**net_configs.get('dense_layer')))

        self.model.add(layers.Dropout(**net_configs.get('dropout_layer')))

        self.model.add(layers.Dense(net_configs.get('number_of_classes')))

    def call(self, inputs, training=False):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()
    # def model(self):
    #     x = keras.Input(shape=self.input_image_shape)
    #     return keras.Model(inputs=[x], outputs=self.call(x))

INPUT_IMAGE_SHAPE = (32, 32, 3)

NET_CONFIGS = dict(
    number_of_classes=10,
    input_image_shape=INPUT_IMAGE_SHAPE,
    conv2d_1=dict(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(1, 1),
        activation='relu',
        padding='same',
    ),
    conv2d_2=dict(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(1, 1),
        activation='relu',
        padding='same',
    ),
    max_pool_2d=dict(
        pool_size=(3, 3),
        strides=(1, 1),
    ),
    res_blocks=dict(
        conv2_block_configs=dict(
            n_blocks=2,
            block_configs=dict(
                filters=(64, 64),
                kernel_sizes=((3, 3), (3, 3)),
                strides=((1, 1), (1, 1)),
                dilation_rates=((1, 1), (1, 1)),
                activations=('relu', 'relu'),
                paddings=('same', 'same')
            )
        ),
        conv3_block_configs=dict(
            n_blocks=2,
            block_configs=dict(
                filters=(64, 64),
                kernel_sizes=((3, 3), (3, 3)),
                strides=((1, 1), (1, 1)),
                dilation_rates=((1, 1), (1, 1)),
                activations=('relu', 'relu'),
                paddings=('same', 'same')
            )
        ),
        conv4_block_configs=dict(
            n_blocks=2,
            block_configs=dict(
                filters=(64, 64),
                kernel_sizes=((3, 3), (3, 3)),
                strides=((1, 1), (1, 1)),
                dilation_rates=((1, 1), (1, 1)),
                activations=('relu', 'relu'),
                paddings=('same', 'same')
            )
        ),
        conv5_block_configs=dict(
            n_blocks=2,
            block_configs=dict(
                filters=(64, 64),
                kernel_sizes=((3, 3), (3, 3)),
                strides=((1, 1), (1, 1)),
                dilation_rates=((1, 1), (1, 1)),
                activations=('relu', 'relu'),
                paddings=('same', 'same')
            )
        )
    ),
    conv2d_3=dict(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(1, 1),
        activation='relu',
        padding='same',
    ),
    dense_layer=dict(
        units=256,
        activation='relu'
    ),
    dropout_layer=dict(
        rate=0.5
    )
)


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def strip_layer_name(layer_full_name, layer_index=''):
    layer_stripped_name = layer_full_name + layer_index
    if find_sub_string(layer_full_name, '_'):
        if find_sub_string(layer_full_name, 'conv'):
            layer_full_name_bw = layer_full_name[::-1]
            layer_stripped_name = layer_full_name[:layer_full_name_bw.index('_')] + layer_index
    return layer_stripped_name


def plot_to_image(figure):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image = tf.image.decode_png(buffer.get_value(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class ConvLayerVis(keras.callbacks.Callback):
    def __init__(self, X, figure_configs, file_writer, save_dir: pathlib.Path = None):
        super().__init__()
        self.X = X
        # self.model = model
        self.file_writer = file_writer
        self.figure_configs = figure_configs
        self.save_dir = save_dir

    def on_training_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        output_layer_tuples = [(idx, layer.output) for idx, layer in enumerate(self.model.layers) if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]
        output_layers = [layer_tuple[1] for layer_tuple in output_layer_tuples]

        # 2) Get the layer names
        conv_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Conv 2D ') for layer_tuple in output_layer_tuples if find_sub_string(layer_tuple[1].name, 'conv2d')]
        max_pool_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Max Pooling 2D') for layer_tuple in output_layer_tuples if find_sub_string(layer_tuple[1].name, 'max_pooling')]
        layer_name_tuples = (conv_layer_name_tuples + max_pool_layer_name_tuples)
        layer_name_tuples.sort(key=lambda x: x[0])
        layer_names = [layer_name_tuple[1] for layer_name_tuple in layer_name_tuples]

        # 3) Build partial model
        partial_model = keras.Model(
            inputs=self.model.model.input,
            outputs=output_layers
        )

        # 4) Get the feature maps
        feature_maps = partial_model.predict(self.X)

        # 5) Plot
        rows, cols = self.figure_configs.get('rows'), self.figure_configs.get('cols')
        for layer_name, feature_map in zip(layer_names, feature_maps):
            fig, ax = plt.subplots(rows, cols, figsize=figure_configs.get('figsize'))
            for row in range(rows):
                for col in range(cols):
                    ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=figure_configs.get('cmap'))

            fig.suptitle(f'{layer_name}')

            plt.show()

            # if isinstance(save_dir, pathlib.Path):
            #     if not save_dir.is_dir():
            #         os.makedirs(save_dir)
            #         fig.savefig(save_dir / f'./{layer_name}.png')

            with file_writer.as_default(f'{layer_name} feature map (epoch #{epoch})', figure_to_image(fig), step=epoch):
                tf.summary.image()

    def on_train_end(self, logs=None):
        pass


def conv_layer_vis(epoch, X, model, figure_configs, file_writer, logs, save_dir: pathlib.Path = None):
    # 1) Get the layers
    output_layer_tuples = [(idx, layer.output) for idx, layer in enumerate(model.layers) if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]
    output_layers = [layer_tuple[1] for layer_tuple in output_layer_tuples]

    # 2) Get the layer names
    conv_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Conv 2D ') for layer_tuple in output_layer_tuples if find_sub_string(layer_tuple[1].name, 'conv2d')]
    max_pool_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Max Pooling 2D') for layer_tuple in output_layer_tuples if find_sub_string(layer_tuple[1].name, 'max_pooling')]
    layer_name_tuples = (conv_layer_name_tuples + max_pool_layer_name_tuples)
    layer_name_tuples.sort(key=lambda x: x[0])
    layer_names = [layer_name_tuple[1] for layer_name_tuple in layer_name_tuples]

    # 3) Build partial model
    partial_model = keras.Model(
        inputs=model.input,
        outputs=output_layers
    )

    # 4) Get the feature maps
    feature_maps = partial_model.predict(X)

    # 5) Plot
    rows, cols = figure_configs.get('rows'), figure_configs.get('cols')
    for layer_name, feature_map in zip(layer_names, feature_maps):
        fig, ax = plt.subplots(rows, cols, figsize=figure_configs.get('figsize'))
        for row in range(rows):
            for col in range(cols):
                ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=figure_configs.get('cmap'))

        fig.suptitle(f'{layer_name}')

        plt.show()

        # if isinstance(save_dir, pathlib.Path):
        #     if not save_dir.is_dir():
        #         os.makedirs(save_dir)
        #         fig.savefig(save_dir / f'./{layer_name}.png')

        with self.file_writer.as_default():
            tf.summary.image(f'{layer_name} feature map (epoch #{epoch})', figure_to_image(fig), step=epoch)



if __name__=='__main__':
    print(tf.config.list_physical_devices('GPU'))

    # Load the data
    (X, y), (X_test, y_test) = cifar10.load_data()
    X, X_test = X.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
    w, h, c = X.shape[1], X.shape[2], X.shape[3]
    print(w, h, c)

    res_net = ResNet(net_configs=NET_CONFIGS)
    res_net.model.input
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

    res_net.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    res_net.fit(
        X,
        y,
        batch_size=64,
        epochs=1,
        callbacks=callbacks
    )

    layer_names = [layer.name for layer in res_net.model.layers]
    layer_names
    res_net.model.layers[0].name
    X = np.random.randn(1, 32, 32, 3)
    for layer in res_net.model.layers:
        print(layer.name)
        print(layer.name.find('conv2d'))
    conv_layer_vis(X = X, model = res_net.model, figure_configs=dict(rows=8, cols=8, figsize=(15, 15), cmap='gray'), save_dir = Path('./filters'))
    # conv_layer_vis(X = X, model = res_net.model, input_layer = res_net.model.input, output_layers=[res_net.model.layers[0].output, res_net.model.layers[1].output, res_net.model.layers[2].output, res_net.model.layers[11].output], figure_configs=dict(rows=8, cols=8, figsize=(15, 15), cmap='gray'), save_dir = Path('./filters'))
