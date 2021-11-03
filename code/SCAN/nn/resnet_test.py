import os
import copy
import pathlib
from pathlib import Path
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.nn import relu
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.utils import Sequence
import cv2
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


def get_image_file_pathes(root_dir_path: pathlib.Path):
    image_file_pathes = []
    for root, dirs, files in os.walk(root_dir_path):
        for file in files:
            image_file_pathes.append(str(Path(root) / file))
    return image_file_pathes

class MorphDS(Sequence):
    def __init__(self, image_files: list, batch_size: int, val_prop: float = 0.0, central_crop_prop: float = 1.0, shuffle: bool = True):

        self.image_files = image_files
        self.val_prop = val_prop
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.central_crop_prop = central_crop_prop if isinstance(central_crop_prop, float) and 0.0 < central_crop_prop  < 1.0 else 1.0

        self.val_image_files = np.random.choice(self.image_files, int(len(self.image_files) * self.val_prop))
        self.train_image_files = np.setdiff1d(self.image_files, self.val_image_files)
        # self.val_image_files = self.__data_generation(np.setdiff1d(self.image_files, self.train_image_files))
        print(f'''
        Total files: {len(self.image_files)}
        - Train files: {len(self.train_image_files)}
        - Validation files: {len(self.val_image_files)}
        ''')

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # - Get the indices of the data in the range of current index
        train_batch_image_files = self.train_image_files[index * self.batch_size : (index + 1) * self.batch_size]

        return self.load_images(train_batch_image_files)

    def load_images(self, image_files):
        X = []
        for image_path in image_files:

            # - Crop the image in the center and normalize it
            img = tf.image.rgb_to_grayscale(tf.image.central_crop(cv2.imread(image_path), self.central_crop_prop)) / 255

            # - Add the image to the batch list
            X.append(img)

        X = np.array(X, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(X)

        return X

    def get_val_data(self):
        return self.load_images(self.val_image_files)

    def on_epoch_end(self):
        pass


class UnsupervidsedModelTrainer:
    def __init__(self, name, model, optimizer, loss, metric, callbacks):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.val_metric = copy.deepcopy(metric)

        for callback in callbacks:
            callback.set_model(self.model)


    # @tf.function
    def train_step(self, X, X_aug):
        with tf.GradientTape() as tape:
            # - Run the augmented images throught the network to get their latent representation
            X_aug_latent = self.model(X_aug, training=True)

            # - Run the original image throught the network to get the latent representation of
            # the original image as the label
            X_latent = self.model(X, training=False)

            loss = self.loss(X_latent, X_aug_latent)
        grads = tape.gradients(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric.update_state(X_latent, X_aug_latent)

        return loss

    # @tf.function
    def test_step(self, X, X_aug):
        X_latent = self.model(X, training=False)
        X_aug_latent = self.model(X_aug, training=False)

        self.val_metric.update_state(X_latent, X_aug_latent)
        return self.val_metric.result()

    def fit(self, dataset, n_epochs, augs):
        X_val = dataset.get_val_data()
        for epoch in range(n_epochs):
            for btch_idx, X in enumerate(dataset):
                X_aug = augs(X)
                batch_loss = self.train_step(X, X_aug)
            train_metric = self.metric.result()
            self.metric.reset_states()

            # Validation
            for X in X_val:
                X_aug = augs(X)
                _ = self.test_step(X, X_aug)
            val_metric = self.val_metric.result()
            self.val_metric.reset_states()

            print(f'''
            Epoch: {epoch}
                - Train: {train_metric}
                - Validation: {val_metric}
            ''')

    def evaluate(self, X):
        return self.test_step(X)

class ResNet(keras.Model):
    class ResBlock(layers.Layer):
        def __init__(self, filters: tuple, kernel_sizes: tuple, strides: tuple = ((1, 1), (1, 1)), activations: tuple = ('relu', 'relu'), paddings: tuple = ('same', 'same'), dilation_rates: tuple = ((1, 1), (1, 1))):
            super().__init__()

            # I) - First conv block
            self.conv2d_1 = layers.Conv2D(
                filters[0], kernel_sizes[0], strides=strides[0], activation=activations[0], padding=paddings[0], dilation_rate=dilation_rates[0])
            self.batch_norm_1 = layers.BatchNormalization()

            # II) - Second conv block
            self.conv2d_2 = layers.Conv2D(
                filters[1], kernel_sizes[1], strides=strides[1], activation=None, padding=paddings[1], dilation_rate=dilation_rates[1])
            self.batch_norm_2 = layers.BatchNormalization()

            # III) - Skip connection
            self.identity = layers.Conv2D(filters[1], 1, padding='same')
            self.shortcut = layers.Add()

            # IV) - Activation
            self.activation = layers.Activation(activations[1])

        def call(self, inputs, training=False):
            x = self.conv2d_1(inputs)
            x = self.batch_norm_1(x)
            x = self.conv2d_2(x)
            x = self.batch_norm_2(x)

            if x.shape[1:] == inputs.shape[1:]:
                x = self.shortcut([x, inputs])
            else:
                x = self.shortcut([x, self.identity(inputs)])

            return self.activation(x)

    def __init__(self, net_configs: dict):
        super().__init__()
        self.input_image_shape = net_configs.get('input_image_shape')
        # 1) Input layer
        self.input_layer = keras.Input(shape=self.input_image_shape)

        self.conv2d_1 = layers.Conv2D(**net_configs.get('conv2d_1'))

        self.conv2d_2 = layers.Conv2D(**net_configs.get('conv2d_2'))

        self.max_pool2d = layers.MaxPool2D(**net_configs.get('max_pool_2d'))


        # 2) ResBlocks
        res_blocks_configs = net_configs.get('res_blocks')

        conv2_block_configs = res_blocks_configs.get('conv2_block_configs')
        self.conv2_blocks = []
        for idx in range(conv2_block_configs.get('n_blocks')):
            self.conv2_blocks.append(self.ResBlock(**conv2_block_configs.get('block_configs')))

        conv3_block_configs = res_blocks_configs.get('conv3_block_configs')
        self.conv3_blocks = []
        for idx in range(conv3_block_configs.get('n_blocks')):
            self.conv3_blocks.append(self.ResBlock(**conv3_block_configs.get('block_configs')))

        conv4_block_configs = res_blocks_configs.get('conv4_block_configs')
        self.conv4_blocks = []
        for idx in range(conv4_block_configs.get('n_blocks')):
            self.conv4_blocks.append(self.ResBlock(**conv4_block_configs.get('block_configs')))

        conv5_block_configs = res_blocks_configs.get('conv5_block_configs')
        self.conv5_blocks = []
        for idx in range(conv5_block_configs.get('n_blocks')):
            self.conv5_blocks.append(self.ResBlock(**conv5_block_configs.get('block_configs')))

        self.conv2d_3 = layers.Conv2D(**net_configs.get('conv2d_3'))

        self.global_avg_pool = layers.GlobalAveragePooling2D()

        self.dense_layer = layers.Dense(**net_configs.get('dense_layer'))

        self.dropout_layer = layers.Dropout(**net_configs.get('dropout_layer'))

        self.classifier = layers.Dense(net_configs.get('number_of_classes'))

    def call(self, inputs, training=False):
        x = self.conv2d_1(inputs, training=training)

        x = self.conv2d_2(x, training=training)

        x = self.max_pool2d(x)

        for conv2_block in self.conv2_blocks:
            x = conv2_block(x)

        for conv3_block in self.conv3_blocks:
            x = conv3_block(x)

        for conv4_block in self.conv4_blocks:
            x = conv4_block(x)

        for conv5_block in self.conv5_blocks:
            x = conv5_block(x)

        x = self.conv2d_3(x)

        x = self.global_avg_pool(x)

        x = self.dense_layer(x)

        x = self.dropout_layer(x)

        return self.classifier(x)

    def model(self):
        x = keras.Input(shape=self.input_image_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


NET_CONFIGS = dict(
    number_of_classes=10,
    input_image_shape=(762, 718, 1),
    # input_image_shape=(24, 24, 3),
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
            n_blocks=3,
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
            n_blocks=4,
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
            n_blocks=6,
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
            n_blocks=3,
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

def dist_loss(v, w):
    return np.linalg.norm(v - w)

if __name__=='__main__':
    data_root_dir_path = Path('D:\\Projects\\NanoScout\\code\\SCAN\\nn\\data')

    train_ds = MorphDS(
        image_files = get_image_file_pathes(root_dir_path=data_root_dir_path / 'train'),
        batch_size = 32,
        val_prop = .1,
        central_crop_prop = .7,
        shuffle = True
    )
    next(iter(train_ds))[0].shape

    test_ds = MorphDS(
        image_files = get_image_file_pathes(root_dir_path=data_root_dir_path / 'test'),
        batch_size = 32,
        val_prop = .0,
        central_crop_prop = .7,
        shuffle = True
    )
    next(iter(test_ds)).shape

    res_net = ResNet(net_configs=NET_CONFIGS)

    res_net.compile(
        loss=dist_loss,
        # loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
            write_images=True
        )
    ]

    trainer = UnsupervidsedModelTrainer(
        name='ResNetx18',
        model=res_net,
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=dist_loss,
        metric=dist_loss,
        callbacks=callbacks
    )
    # res_net.fit(
    #     train_ds,
    #     epochs=50,
    #     steps_per_epoch=195,
    #     validation_data=test_ds,
    #     validation_steps=3,
    #     callbacks=callbacks
    # )
    augs = tf.keras.Sequential([layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.2)])
    trainer.fit(train_ds, n_epochs=3, augs=augs)
    res_net.model().summary()
