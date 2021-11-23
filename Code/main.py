import os
import datetime
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
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
    # LOG_DIR_PATH
)

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN = True

def get_model(purpose, model_type, number_of_classes, crop_size, priors=None):
    input_image_shape = (crop_size, crop_size, 1)
    model = None
    if model_type == 'conv_net':
        model = cnn.ConvModel(input_shape=input_image_shape)
    elif model_type == 'res_net_x18':
        resnet_configs_file_path = CONFIGS_DIR_PATH / 'res_net_x18_configs.yml'
        with resnet_configs_file_path.open(mode='r') as config_file:
            resnet_configs = yaml.safe_load(config_file)
        resnet_configs['input_image_shape'] = input_image_shape
        resnet_configs['number_of_classes'] = number_of_classes
        if purpose == 'feature_extraction':
            model = cnn.FeatureExtractionResNet(
                net_configs=resnet_configs,
                augmentations=clustering_augmentations.augmentations,
                similarity_loss=clustering_losses.cosine_similarity_loss,
            )
        elif purpose == 'labelling':
            model = cnn.LabellingResNet(
                net_configs=resnet_configs,
                priors=priors,
                augmentations=clustering_augmentations.augmentations,
            )

    elif args.feature_extractor_net == 'res_net_x34':
        resnet_configs_file_path = CONFIGS_DIR_PATH / 'res_net_x34_configs.yml'
        with resnet_configs_file_path.open(mode='r') as config_file:
            resnet_configs = yaml.safe_load(config_file)
        resnet_configs['input_image_shape'] = input_image_shape
        resnet_configs['number_of_classes'] = number_of_classes

        if purpose == 'feature_extraction':
            model = cnn.FeatureExtractionResNet(
                net_configs=resnet_configs,
                augmentations=clustering_augmentations.augmentations,
                similarity_loss=clustering_losses.cosine_similarity_loss,
            )
        elif purpose == 'labelling':
            model = cnn.LabellingResNet(
                net_configs=resnet_configs,
                priors=priors,
                augmentations=clustering_augmentations.augmentations,
            )
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='feature_extractor_net', type=str, choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to extract the features from the images')
    parser.add_argument(dest='labeller_net', type=str, choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to label the samples')
    parser.add_argument(dest='latent_space_dims', type=int, help='The dimension of the vectors in the latent space which represent the encodeing of each image')
    parser.add_argument(dest='number_of_classes', type=int, help='The number of label classes')
    parser.add_argument(dest='epochs', type=int, help='Number of epochs to train the network')
    parser.add_argument(dest='steps_per_epoch', type=int, help='Number of iterations that will be performed on each epoch')
    parser.add_argument('--crop_size', type=int, choices=[32, 64, 128, 256, 512], default=128, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in each batch')
    parser.add_argument('--validation_split', type=float, default=0.1, help='The proportion of the data to be used for validation (should be in range [0.0, 1.0])')
    parser.add_argument('--validation_steps_proportion', type=float, default=0.5, help='The proportion of validation steps in regards to the training steps (should be in range [0.0, 1.0])')
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(len(tf.config.list_physical_devices('GPU')))], default=0 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--feature_extractor_checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the feature extraction model')
    parser.add_argument('--no_train_feature_extractor', default=False, action='store_true', help=f'If theres no need to train the feature extractor model')
    parser.add_argument('--load_labeler_checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the labeler model')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_run_dir = OUTPUT_DIR_PATH / f'{ts}'
    if not current_run_dir.is_dir():
        os.makedirs(current_run_dir)

    input_image_shape = (args.crop_size, args.crop_size, 1)

    # 1) Build the feature extractor model
    feat_ext_model = get_model(
        purpose='feature_extraction',
        model_type=args.feature_extractor_net,
        number_of_classes=args.latent_space_dims,
        crop_size=args.crop_size
    )

    assert feat_ext_model is not None, 'Could not build the model!'

    print(feat_ext_model.summary())

    if os.path.exists(args.feature_extractor_checkpoint_dir):
        checkpoint_dir = args.feature_extractor_checkpoint_dir
        latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)

        feat_ext_model.load_weights(latest_cpt)

    if not args.no_train_feature_extractor:
        # 1) Get the data
        train_ds, val_ds = data_funcs.get_dataset_from_tiff(
        input_image_shape=input_image_shape,
        batch_size=args.batch_size,
        validation_split=args.validation_split
        )

        # 3) Configure callbacks
        callbacks = clustering_callbacks.get_callbacks(
            model=feat_ext_model,
            X=next(iter(val_ds))[0][0],
            ts=ts
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

    priors_knn_df = aux_funcs.transform_images(images_root_dir=TRAIN_DATA_DIR_PATH, model=feat_ext_model, patch_height=args.crop_size, patch_width=args.crop_size)

    X = np.array([x[0].numpy() for x in trapriors_knn_df.loc[:, 'vector'].values])
    files = transforms_df.loc[:, 'file'].values

    priors_knn_df.loc[:, 'neighbors'] = aux_funcs.get_knn_files(X=X, files=files, k=10)
    priors_knn_df.to_pickle(current_run_dir / f'priors_knn_df.pkl')

    # 3) Build the feature extractor model
    labeller_model = get_model(
        purpose='labelling',
        model_type=args.labeller_net,
        number_of_classes=args.number_of_classes,
        crop_size=args.crop_size,
        priors=priors_knn_df,
    )


    train_ds, val_ds = data_funcs.get_dataset_from_tiff(
    input_image_shape=input_image_shape,
    batch_size=args.batch_size,
    validation_split=args.validation_split
    )

    # 3) Configure callbacks
    callbacks = clustering_callbacks.get_callbacks(
        model=feat_ext_model,
        X=next(iter(val_ds))[0][0],
        ts=ts
    )

    # 4) Compile
    labeller_model.compile(
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
