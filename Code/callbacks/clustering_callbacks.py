import os
import datetime as dt
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from aux_code import aux_funcs


from configs.general_configs import (
    LOG_INTERVAL,
    EARLY_STOPPING_PATIENCE,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    LAYER_VIZ_FIG_SIZE,
    LAYER_VIZ_CMAP,
)


class ConvLayerVis(keras.callbacks.Callback):
    def __init__(self, X, input_layer, layers, figure_configs: dict, log_dir: str, log_interval: int):
        super().__init__()
        self.X_test = X
        self.input_layer = input_layer
        self.layers = layers
        self.log_dir = log_dir

        plt.imshow(self.X_test, cmap='gray')
        test_image_dir = Path(self.log_dir) / 'test_images'
        if not test_image_dir.is_dir():
            os.makedirs(test_image_dir)
        plt.savefig(test_image_dir / 'test_img.png')

        self.tensor_board_th = None
        n_dims = len(self.X_test.shape)
        assert 2 < n_dims < 5, f'The shape of the test image should be less than 5 and grater than 2, but current shape is {self.X_test.shape}'

        # In case the image is not represented as a tensor - add a dimension to the left for the batch
        if len(self.X_test.shape) < 4:
            self.X_test = np.reshape(self.X_test, (1,) + self.X_test.shape)

        self.file_writer = tf.summary.create_file_writer(self.log_dir)
        self.figure_configs = figure_configs
        self.log_interval = log_interval

    def on_training_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        if epoch % self.log_interval == 0:
            print(f'\nSaving conv layer images of epoch #{epoch} to: \'{self.log_dir}\'...')
            # 1) Get the layers
            output_layer_tuples = [(idx, layer) for idx, layer in enumerate(self.layers) if aux_funcs.find_sub_string(layer.name, 'conv2d') or aux_funcs.find_sub_string(layer.name, 'max_pooling2d')]
            output_layers = [layer_tuple[1].output for layer_tuple in output_layer_tuples]

            # 2) Get the layer names
            conv_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Conv 2D ') for layer_tuple in output_layer_tuples if aux_funcs.find_sub_string(layer_tuple[1].name, 'conv2d')]
            max_pool_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Max Pooling 2D') for layer_tuple in output_layer_tuples if aux_funcs.find_sub_string(layer_tuple[1].name, 'max_pooling2d')]

            layer_name_tuples = (conv_layer_name_tuples + max_pool_layer_name_tuples)
            layer_name_tuples.sort(key=lambda x: x[0])

            layer_names = [layer_name_tuple[1] for layer_name_tuple in layer_name_tuples]

            # 3) Build partial model
            partial_model = keras.Model(
                inputs=self.input_layer,
                outputs=output_layers
            )

            # 4) Get the feature maps
            feature_maps = partial_model.predict(self.X_test)

            # 5) Plot
            for feature_map, layer_name in zip(feature_maps, layer_names):
                num_feat_maps = feature_map.shape[3]
                rows = cols = np.ceil(num_feat_maps ** 0.5).astype(np.int8)

                fig, ax = plt.subplots(rows, cols, figsize=self.figure_configs.get('figsize'))

                for row in range(rows):
                    for col in range(cols):
                        feat_map_idx = row + col
                        if feat_map_idx >= num_feat_maps:
                            break
                        ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=self.figure_configs.get('cmap'))
                fig.suptitle(f'{layer_name}')

                with self.file_writer.as_default():
                    tf.summary.image(f'{layer_name} Feature Maps', aux_funcs.get_image_from_figure(figure=fig), step=epoch)

            if self.tensor_board_th is None:
                print(f'Launching a Tensor Board thread on logdir: \'{self.log_dir}\'...')
                self.tensor_board_th = aux_funcs.launch_tensor_board(logdir=self.log_dir)


def get_callbacks(model, X, log_dir):
    callbacks = [
        # -------------------
        # Built-in  callbacks
        # -------------------
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            write_graph=True,
            write_images=True,
            write_steps_per_second=True,
            update_freq='epoch',
            embeddings_freq=LOG_INTERVAL,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=EARLY_STOPPING_PATIENCE,
            mode='auto',
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),

        # -----------------
        # Custom callbacks
        # -----------------
        ConvLayerVis(
            X=X,
            input_layer=model.model.input,
            layers=model.model.layers,
            figure_configs=dict(
                figsize=LAYER_VIZ_FIG_SIZE,
                cmap=LAYER_VIZ_CMAP,
             ),
            log_dir=f'{log_dir}/train',
            log_interval=LOG_INTERVAL
        )
    ]
    return callbacks
