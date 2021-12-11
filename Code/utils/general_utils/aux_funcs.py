import os
import yaml
import logging
import threading
import argparse
import pathlib
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from models import cnn
from utils.image_utils import image_funcs
from configs.general_configs import (
    CONFIGS_DIR_PATH,
)


def get_train_val_idxs(n_items, val_prop):
    all_idxs = np.arange(n_items)
    val_idxs = np.random.choice(all_idxs, int(val_prop * n_items), replace=False)
    train_idxs = np.setdiff1d(all_idxs, val_idxs)
    return train_idxs, val_idxs


def check_dir(dir_path: pathlib.Path):
    dir_exists = False
    if isinstance(dir_path, pathlib.Path):
        if not dir_path.is_dir():
            os.makedirs(dir_path)
            dir_exists = True
    return dir_exists


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def get_file_type(file_name: str):
    file_type = None
    if isinstance(file_name, str):
        dot_idx = file_name.find('.')
        if dot_idx > -1:
            file_type = file_name[dot_idx + 1:]
    return file_type


def launch_tensorboard(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th


def get_model(model_name, model_type, number_of_classes, crop_size, augmentations, custom_loss = None, checkpoint_dir: pathlib.Path = None, priors: pd.DataFrame = None, logger: logging.Logger = None):
    input_image_shape = (crop_size, crop_size, 1)
    weights_loaded = False
    model = None
    if model_type == 'conv_net':
        model = cnn.ConvModel(
            model_name=model_name,
            input_shape=input_image_shape
        )

    elif model_type in ['res_net_x18', 'res_net_x34']:
        architecture = model_type.split('_')[-1]
        resnet_configs_file_path = CONFIGS_DIR_PATH / f'res_net_{architecture}_configs.yml'
        with resnet_configs_file_path.open(mode='r') as config_file:
            resnet_configs = yaml.safe_load(config_file)
        resnet_configs['input_image_shape'] = input_image_shape
        resnet_configs['number_of_classes'] = number_of_classes
        if model_name == 'feature_extractor':
            model = cnn.FeatureExtractorResNet(
                model_name=model_name,
                model_configs=resnet_configs,
                augmentations=augmentations,
                similarity_loss=custom_loss,
            )

        elif model_name == 'classifier':
            assert priors is None, f'The \'priors\' can not be None!'
            model = cnn.ClassifierResNet(
                model_name=model_name,
                model_configs=resnet_configs,
                augmentations=augmentations,
                priors=priors
            )

        if checkpoint_dir.is_dir:
            try:
                latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)

                model.load_weights(latest_cpt)
                weights_loaded = True
            except Exception as err:
                if isinstance(logger, logging.Logger):
                    logger.exception(err)
            else:
                if isinstance(logger, logging.Logger):
                    logger.info(f'Weights from \'checkpoint_dir\' were loaded successfully to the \'{model_name}\' model!')

    if isinstance(logger, logging.Logger):
        logger.info(model.summary())

    return model, weights_loaded


def choose_gpu(gpu_id: int = 0, logger: logging.Logger = None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_id > -1:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                physical_gpus = tf.config.list_physical_devices('GPU')
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on: {physical_gpus}
                ====================================================
                ''')
            else:
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on all available devices
                ====================================================
                    ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)


def get_logger(configs_file, save_file):
    logger = None
    try:
        if configs_file.is_file():
            with configs_file.open(mode='r') as f:
                configs = yaml.safe_load(f.read())

                # Assign a valid path to the log file
                configs['handlers']['logfile']['filename'] = str(save_file)
                logging.config.dictConfig(configs)

        logger = logging.getLogger(__name__)
    except Exception as err:
        print(err)

    return logger


def get_priors_knn_df(model, preprocessing_func, k: int, train_data_dir: pathlib.Path, patch_height: int, patch_width: int, patch_optimization: bool, save_dir: pathlib.Path, knn_algorithm: str = 'auto'):
    priors_knn_df = None
    if patch_optimization:
        priors_knn_df = image_funcs.get_patch_transforms(images_root_dir=train_data_dir, model=model, preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width)
        X = np.array([x for x in priors_knn_df.loc[:, 'patch_transform'].values])
    else:
        priors_knn_df = image_funcs.get_mean_image_transforms(images_root_dir=train_data_dir, model=model, preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width)
        X = np.array([x for x in priors_knn_df.loc[:, 'image_mean_transform'].values])
    files = priors_knn_df.loc[:, 'file'].values

    nbrs_distances, nbrs_files = get_knn_files(X=X, files=files, k=k, algorithm=knn_algorithm)

    priors_knn_df.loc[:, 'distances'] = nbrs_distances
    priors_knn_df.loc[:, 'neighbors'] = nbrs_files

    os.makedirs(save_dir, exist_ok=True)
    priors_knn_df.to_pickle(save_dir / f'priors_knn_df.pkl')
    plot_knn(knn=priors_knn_df, save_dir=save_dir)

    return priors_knn_df


def get_arg_parcer():
    parser = argparse.ArgumentParser()

    # FLAGS
    # a) General parameters
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, len(tf.config.list_physical_devices('GPU')))], default=-1 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--crop_size', type=int, default=128, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')

    # b) Feature extractor network
    parser.add_argument('--feature_extractor_model_type', type=str, default='res_net_x18', choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to extract the features from the images')
    parser.add_argument('--feature_extractor_latent_space_dims', type=int, default=256, help='The dimension of the vectors in the latent space which represent the encodeing of each image')
    parser.add_argument('--feature_extractor_train_epochs', type=int, default=100, help='Number of epochs to train the feature extractor network')
    parser.add_argument('--feature_extractor_train_steps_per_epoch', type=int, default=1000, help='Number of iterations that will be performed on each epoch for the featrue extractor network')
    parser.add_argument('--feature_extractor_batch_size', type=int, default=32, help='The number of samples in each batch')
    parser.add_argument('--feature_extractor_validation_split', type=float, default=0.1, help='The proportion of the data to be used for validation in the train process of the feature extractor model ((should be in range [0.0, 1.0])')
    parser.add_argument('--feature_extractor_validation_steps_proportion', type=float, default=0.5, help='The proportion of validation steps in regards to the training steps in the train process of the feature extractor model ((should be in range [0.0, 1.0])')
    parser.add_argument('--feature_extractor_checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the feature extraction model')
    parser.add_argument('--feature_extractor_optimizer_lr', type=float, default=1e-4, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--no_reduce_lr_on_plateau_feature_extractor', default=False, action='store_true', help=f'If not to use the ReduceLROnPlateau callback')
    parser.add_argument('--no_train_feature_extractor', default=False, action='store_true', help=f'If theres no need to train the feature extractor model')

    # c) Priors
    parser.add_argument('--knn_k', type=int, default=5, help=f'Chooses how many Nearest Neighbor to search')
    parser.add_argument('--knn_algorithm', type=str, default='auto', choices=['auto', 'brute', 'kd_tree', 'ball_tree'], help=f'Chooses which K Nearest Neighbor (KNN) algorithm to use for prior mining')
    parser.add_argument('--knn_patch_optimization', default=False, action='store_true', help=f'If to perform the K Nearest Neighbor (KNN) search on the individual patches, or on a mean of the patch transforms')
    parser.add_argument('--knn_df_path', type=str, default='', help=f'The path to the KNN data frame file')

    # d) Classifier network
    parser.add_argument('--classifier_model_type', type=str, default='res_net_x18', choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to classify the samples')
    parser.add_argument('--classifier_number_of_classes', type=int, default=2, help='The number of classes to assign to the samples')
    parser.add_argument('--classifier_train_epochs', type=int, default=100, help='Number of epochs to train the classifier network')
    parser.add_argument('--classifier_train_steps_per_epoch', type=int, default=1000, help='Number of iterations that will be performed on each epoch for the classifier network')
    parser.add_argument('--classifier_batch_size', type=int, default=32, help='The number of samples in each batch')
    parser.add_argument('--classifier_validation_split', type=float, default=0.1, help='The proportion of the data to be used for validation in the train process of the classifier model (should be in range [0.0, 1.0])')
    parser.add_argument('--classifier_validation_steps_proportion', type=float, default=0.5, help='The proportion of validation steps in regards to the training steps in the train process of the classifier model ((should be in range [0.0, 1.0])')
    parser.add_argument('--classifier_checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the labeler model')
    parser.add_argument('--classifier_optimizer_lr', type=float, default=1e-4, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--no_reduce_lr_on_plateau_classifier', default=False, action='store_true', help=f'If not to use the ReduceLROnPlateau callback')
    parser.add_argument('--no_train_classifier', default=False, action='store_true', help=f'If theres no need to train the classifier model')

    return parser
