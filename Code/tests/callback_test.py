import os
import io
import threading
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from aux_code import aux_funcs

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


def launch_tb(log_dir):
    os.system(f'tensorboard --logdir={log_dir}')

    th = threading.Thread(
        target=lambda logdir: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    th.start()


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


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def get_file_type(file_name: str):
    file_type = None
    if isinstance(file_name, str):
        dot_idx = file_name.find('.')
        if dot_idx > -1:
            file_type = file_name[dot_idx + 1:]
    return file_type


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class ConvLayerVis(keras.callbacks.Callback):
    def __init__(self, X, input_layer, layers, figure_configs: dict, log_dir: str, log_interval: int):
        super().__init__()
        self.X_test = X
        self.input_layer = input_layer
        self.layers = layers
        n_dims = len(self.X_test.shape)
        assert 2 < n_dims < 5, f'The shape of the test image should be less than 5 and grater than 2, but current shape is {self.X_test.shape}'

        # In case the image is not represented as a tensor - add a dimension to the left for the batch
        if len(self.X_test.shape) < 4:
            self.X_test = np.reshape(self.X_test, (1,) + self.X_test.shape)

        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.figure_configs = figure_configs
        self.log_interval = log_interval

    def on_training_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        if epoch % self.log_interval == 0:
            # 1) Get the layers
            output_layer_tuples = [(idx, layer) for idx, layer in enumerate(self.layers) if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]
            # output_layer_tuples = [(idx, layer) for idx, layer in enumerate(self.model.model.layers) if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]
            output_layers = [layer_tuple[1].output for layer_tuple in output_layer_tuples]

            # 2) Get the layer names
            conv_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Conv 2D ') for layer_tuple in output_layer_tuples if find_sub_string(layer_tuple[1].name, 'conv2d')]
            max_pool_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Max Pooling 2D') for layer_tuple in output_layer_tuples if find_sub_string(layer_tuple[1].name, 'max_pooling2d')]

            layer_name_tuples = (conv_layer_name_tuples + max_pool_layer_name_tuples)
            layer_name_tuples.sort(key=lambda x: x[0])

            layer_names = [layer_name_tuple[1] for layer_name_tuple in layer_name_tuples]

            # 3) Build partial model
            partial_model = keras.Model(
                inputs=self.input_layer,
                # inputs=model.model.input,
                outputs=output_layers
            )

            # 4) Get the feature maps
            feature_maps = partial_model.predict(self.X_test)

            # 5) Plot
            rows, cols = self.figure_configs.get('rows'), self.figure_configs.get('cols')
            for feature_map, layer_name in zip(feature_maps, layer_names):
                fig, ax = plt.subplots(rows, cols, figsize=self.figure_configs.get('figsize'))
                for row in range(rows):
                    for col in range(cols):
                        ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=self.figure_configs.get('cmap'))
                fig.suptitle(f'{layer_name}')

                with self.file_writer.as_default():
                    tf.summary.image(f'{layer_name} Feature Maps', get_image_from_figure(figure=fig), step=epoch)


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

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

    aux_funcs.launch_tb(log_dir=log_dir)
