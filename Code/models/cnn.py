import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

    def summary(self):
        return self.model.summary()


# noinspection PyUnusedLocal
class ResNet(keras.Model):
    class ResBlock(keras.Model):
        def __init__(self, filters: tuple, kernel_sizes: tuple, strides: tuple = ((1, 1), (1, 1)), activations: tuple = ('relu', 'relu'), paddings: tuple = ('same', 'same'), dilation_rates: tuple = ((1, 1), (1, 1))):
            super().__init__()

            self.model = keras.Sequential()
            self.build_res_block(filters=filters, kernel_sizes=kernel_sizes, strides=strides, activations=activations, paddings=paddings, dilation_rates=dilation_rates)

            # - Shortcut
            self.skip_connection_layer = layers.Add()
            self.identity_mapping_layer = layers.Conv2D(filters[1], 1, padding='same')

            # - Activation
            self.activation = layers.Activation(activations[1])

        def build_res_block(self, filters: tuple, kernel_sizes: tuple, strides: tuple, activations: tuple, paddings: tuple, dilation_rates: tuple):
            # I) - First conv block
            self.model.add(layers.Conv2D(filters[0], kernel_sizes[0], strides=strides[0], activation=activations[0], padding=paddings[0], dilation_rate=dilation_rates[0]))
            self.model.add(layers.BatchNormalization())
            # II) - Second conv block
            self.model.add(layers.Conv2D(filters[1], kernel_sizes[1], strides=strides[1], activation=None, padding=paddings[1], dilation_rate=dilation_rates[1]))
            self.model.add(layers.BatchNormalization())
            # III) - Output conv layer
            self.model.add(layers.Conv2D(filters[1], 1, padding='same'))

        # noinspection PyUnusedLocal
        def call(self, inputs, training=False):
            X = self.model(inputs)
            # > Skip connection
            # Depending on the output shape we'd use:
            # - input of the same number of channels if they are equal
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
        self.net_configs = net_configs
        self.model = keras.Sequential()
        self.build_net()

    def build_net(self):
        # 1) Input layer
        self.model.add(keras.Input(shape=self.net_configs.get('input_image_shape')))
        self.model.add(layers.Conv2D(**self.net_configs.get('conv2d_1')))
        self.model.add(layers.Conv2D(**self.net_configs.get('conv2d_2')))
        self.model.add(layers.MaxPool2D(**self.net_configs.get('max_pool_2d')))

        # 2) ResBlocks
        res_blocks_configs = self.net_configs.get('res_blocks')

        conv2_block_configs = res_blocks_configs.get('conv2_block_configs')
        for idx in range(conv2_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv2_block_configs.get('block_configs')))

        conv3_block_configs = res_blocks_configs.get('conv3_block_configs')
        for idx in range(conv3_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv3_block_configs.get('block_configs')))

        conv4_block_configs = res_blocks_configs.get('conv4_block_configs')
        for idx in range(conv4_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv4_block_configs.get('block_configs')))

        conv5_block_configs = res_blocks_configs.get('conv5_block_configs')
        for idx in range(conv5_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv5_block_configs.get('block_configs')))

        self.model.add(layers.Conv2D(**self.net_configs.get('conv2d_3')))

        self.model.add(layers.GlobalAveragePooling2D())

        self.model.add(layers.Dense(**self.net_configs.get('dense_layer')))

        self.model.add(layers.Dropout(**self.net_configs.get('dropout_layer')))

        self.model.add(layers.Dense(self.net_configs.get('number_of_classes')))

    def call(self, inputs, training=False):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()


class FeatureExtractionResNet(ResNet):
    def __init__(self, net_configs, augmentations, similarity_loss: tf.keras.losses.Loss):
        super().__init__(net_configs=net_configs)
        print(net_configs)
        self.net_configs = net_configs
        self.augmentations = augmentations
        self.similarity_loss = similarity_loss
        self.model = keras.Sequential()
        self.build_net()

    # noinspection PyCallingNonCallable
    def train_step(self, data):
        # Get the image only (the label is irrelevant)
        X, y = data

        with tf.GradientTape() as tape:

            # Run the original image through the network
            y_pred = self(X, training=True)

            # Run the augmented image through the network
            y_aug_pred = self(self.augmentations(X), training=True)

            # The loss is made of two parts:
            # 1) The cross entropy loss of the crop to its' label (i.e., the image from which it was taken), which extracts features from teh image
            # 2) If the original images' latent representation is clos to the augmented version, which is measured via the cosine closeness loss.
            # This part makes sure that only "strong" features will be extracted
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) + self.similarity_loss(y_pred, y_aug_pred)

        # Calculate gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return the mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
