import os
import datetime
import argparse
import yaml
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from aux_code import aux_funcs
from data_loaders import data_funcs
from models import cnn
from losses import clustering_losses
from augmentations import clustering_augmentations
from callbacks import clustering_callbacks
from configs.general_configs import (
    CONFIGS_DIR_PATH,
    TRAIN_DATA_DIR_PATH,
    OUTPUT_DIR_PATH,
)
# from sklearn.neighbors import NearestNeighbors

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='feature_extractor_net', type=str, choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to extract the features from the images')
    parser.add_argument(dest='latent_space_dims', type=int, help='The dimension of the vectors in the latent space which represent the encodeing of each image')
    parser.add_argument(dest='epochs', type=int, help='Number of epochs to train the network')
    parser.add_argument(dest='steps_per_epoch', type=int, help='Number of iterations that will be performed on each epoch')
    parser.add_argument('--crop_size', type=int, choices=[32, 64, 128, 256, 512], default=128, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in each batch')
    parser.add_argument('--validation_split', type=float, default=0.1, help='The proportion of the data to be used for validation (should be in range [0.0, 1.0])')
    parser.add_argument('--validation_steps_proportion', type=float, default=0.5, help='The proportion of validation steps in regards to the training steps (should be in range [0.0, 1.0])')
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(len(tf.config.list_physical_devices('GPU')))], default=0 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--load_feature_extractor_model', type=bool, default=False, help=f'If to load a model or to use the one saved as \'{OUTPUT_DIR_PATH}/model/feature_extractor_model.h5\'')
    parser.add_argument('--load_labeler_model', type=bool, default=False, help=f'If to load a model or to use the one saved as \'{OUTPUT_DIR_PATH}/model/labeler_model.h5\'')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    input_image_shape = (args.crop_size, args.crop_size, 1)

    # 1) Get the data
    train_ds, val_ds = data_funcs.get_dataset_from_tiff(
        input_image_shape=input_image_shape,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )

    # 2) Build the model
    feat_ext_model = None
    if args.load_feature_extractor_model and (Path(OUTPUT_DIR_PATH) / 'models/feature_extractor_model.h5').is_file():
        feat_ext_model = tf.keras.model.load_model(Path(OUTPUT_DIR_PATH) / 'models/feature_extractor_model.h5')
    elif args.feature_extractor_net == 'conv_net':
        feat_ext_model = cnn.ConvModel(input_shape=input_image_shape)
    elif args.feature_extractor_net == 'res_net_x18':
        resnet_configs_file_path = CONFIGS_DIR_PATH / 'res_net_x18_configs.yml'
        with resnet_configs_file_path.open(mode='r') as config_file:
            resnet_configs = yaml.safe_load(config_file)
        resnet_configs['input_image_shape'] = input_image_shape
        resnet_configs['number_of_classes'] = args.latent_space_dims

        feat_ext_model = cnn.FeatureExtractionResNet(
            net_configs=resnet_configs,
            augmentations=clustering_augmentations.augmentations,
            similarity_loss=clustering_losses.cosine_similarity_loss
        )
    elif args.feature_extractor_net == 'res_net_x34':
        resnet_configs_file_path = CONFIGS_DIR_PATH / 'res_net_x34_configs.yml'
        with resnet_configs_file_path.open(mode='r') as config_file:
            resnet_configs = yaml.safe_load(config_file)
        resnet_configs['input_image_shape'] = input_image_shape
        resnet_configs['number_of_classes'] = args.latent_space_dims

        feat_ext_model = cnn.FeatureExtractionResNet(
            net_configs=resnet_configs,
            augmentations=clustering_augmentations.augmentations,
            similarity_loss=clustering_losses.cosine_similarity_loss
        )

    assert feat_ext_model is not None, 'Could not build the model!'

    feat_ext_model.summary()

    # 3) Configure callbacks
    log_dir = Path(f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    callbacks = clustering_callbacks.get_callbacks(
        model=feat_ext_model,
        X=next(iter(val_ds))[0][0],
        log_dir=log_dir
    )

    # 4) Compile
    feat_ext_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # 5) Fit feature extraction model
    validation_steps = int(args.validation_steps_proportion * args.steps_per_epoch) if 0 < int(args.validation_steps_proportion * args.steps_per_epoch) <= args.steps_per_epoch else 1

    feat_ext_model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=1,  # [1, 100, 1500, ...] - validate on these epochs
        shuffle=True,
        callbacks=callbacks
    )

    transforms_df = aux_code.transform_images(images_root_dir=TRAIN_DATA_DIR_PATH, model=feat_ext_model, patch_height=args.crop_size, patch_width=args.crop_size)
    transforms_df.to_pickle(OUTPUT_DIR_PATH / 'transforms_df.plk')

    feat_ext_model.save(OUTPUT_DIR_PATH / 'models/feature_extractor_model.h5')

