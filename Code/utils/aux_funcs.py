import os
import yaml
import io
import threading
import argparse
import pathlib
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2
from models import cnn
from losses import clustering_losses
from augmentations import clustering_augmentations
from configs.general_configs import (
    CONFIGS_DIR_PATH,
)


def check_dir(dir_path: pathlib.Path):
    dir_exists = False
    if isinstance(dir_path, pathlib.Path):
        if not dir_path.is_dir():
            os.makedirs(dir_path)
            dir_exists = True
    return dir_exists


def plot_knn(knn: pd.DataFrame, save_dir: pathlib.Path):
    k = len(knn.loc[0, 'neighbors'][0])
    n_files = knn.shape[0]
    plots_dir = save_dir / 'plots'
    check_dir(plots_dir)

    for idx, file in enumerate(knn.loc[:, 'file']):
        fig, axs = plt.subplots(1, k, figsize=(100, 15), facecolor='#c0d6e4');
        file_name = file.split('/')[-1]
        file_name = file_name[:file_name.index('.')]
        distances = knn.loc[idx, 'distances'][0]
        neighbors = knn.loc[idx, 'neighbors'][0]
        for idx, (distance, neighbor) in enumerate(zip(distances, neighbors)):
            axs[idx].imshow(cv2.imread(neighbor))
            neighbor_name = neighbor.split('/')[-1]
            if not idx:
                axs[idx].set(title=f'{neighbor_name} (Original)')
            else:
                axs[idx].set(title=f'{neighbor_name} (Distance = {distance:.1f})')
            axs[idx].title.set_size(70)
        print(f'Saving KNN image - {file_name} ({100 * idx / n_files:.1f}% done)')
        fig.savefig(plots_dir / (file_name + '.png'))

        plt.close(fig)


def get_patch_df(image_file, patch_height, patch_width):
    assert image_file.is_file(), f'No file \'{image_file}\' was found!'

    img = np.expand_dims(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE), axis=-1)
    df = pd.DataFrame(columns=['file', 'image'])
    img_h, img_w, _ = img.shape
    for h in range(0, img_h, patch_height):
        for w in range(0, img_w, patch_width):
            patch = img[h:h+patch_height, w:w+patch_width, :]
            if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                df = df.append(dict(file=str(image_file), image=patch), ignore_index=True)
    return df


def get_mean_image_transforms(images_root_dir, model, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'image_mean_transform'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:

            # get the patches
            patches_df = get_patch_df(image_file=Path(f'{root}/{file}'), patch_height=patch_height, patch_width=patch_width)

            # get the mean patch transform
            patch_transforms = list()
            for patch in patches_df.loc[:, 'image'].values:
                patch_transforms.append(model(np.expand_dims(patch, axis=0)) if len(patch.shape) < 4 else model(patch))
            patch_transforms = np.array(patch_transforms)
            image_mean_transform = patch_transforms.mean(axis=0)[0, :]
            df = df.append(
                {
                    'file': f'{root}/{file}',
                    'image_mean_transform': image_mean_transform
                },
                ignore_index=True
            )
    return df


def get_patch_transforms(images_root_dir, model, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'image'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:
            df = df.append(get_patch_df(image_file=Path(f'{root}/{file}'), patch_height=patch_height, patch_width=patch_width), ignore_index=True)
    df.loc[:, 'patch_transform'] = df.loc[:, 'image'].apply(lambda x: model(np.expand_dims(x, axis=0))[0].numpy() if len(x.shape) < 4 else model(x)[0].numpy())
    df = df.loc[:, ['file', 'patch_transform']]
    return df


def get_knn_files(X, files, k, algorithm='auto'):
    # Detect the k nearest neighbors
    nbrs_pred = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(X)

    nbrs_distances = list()
    nbrs_files = list()
    for idx, (file, x) in enumerate(zip(files, X)):
        distances, nbrs_idxs = nbrs_pred.kneighbors(np.expand_dims(x, axis=0))

        nbrs_distances.append(distances)
        nbrs_files.append(files[nbrs_idxs])

    return nbrs_distances, nbrs_files


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


def launch_tensor_board(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th


def get_model(net_name, model_type, number_of_classes, crop_size, priors=None):
    input_image_shape = (crop_size, crop_size, 1)
    model = None
    if model_type == 'conv_net':
        model = cnn.ConvModel(
            net_name=net_name,
            input_shape=input_image_shape
        )
    elif model_type in ['res_net_x18', 'res_net_x34']:
        architecture = model_type.split('_')[-1]
        resnet_configs_file_path = CONFIGS_DIR_PATH / f'res_net_{architecture}_configs.yml'
        with resnet_configs_file_path.open(mode='r') as config_file:
            resnet_configs = yaml.safe_load(config_file)
        resnet_configs['input_image_shape'] = input_image_shape
        resnet_configs['number_of_classes'] = number_of_classes
        if net_name == 'feature_extractor':
            model = cnn.FeatureExtractorResNet(
                net_name=net_name,
                net_configs=resnet_configs,
                augmentations=clustering_augmentations.augmentations,
                similarity_loss=clustering_losses.cosine_similarity_loss,
            )
        elif net_name == 'classifier':
            assert priors is None, f'The \'priors\' can not be None!'
            model = cnn.ClassifierResNet(
                net_name=net_name,
                net_configs=resnet_configs,
                priors=priors,
                augmentations=clustering_augmentations.augmentations,
            )
    return model


def choose_gpu(gpu_id: int = 0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            physical_gpus = tf.config.list_physical_devices('GPU')
            # upper_line = ''
            print(f'''
            ====================================================
            > Running on: {physical_gpus}
            ====================================================
            ''')
        except RuntimeError as error:
            print(error)


def get_arg_parcer():
    parser = argparse.ArgumentParser()

    # FLAGS
    # a) General parameters
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(len(tf.config.list_physical_devices('GPU')))], default=0 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--crop_size', type=int, default=128, choices=[32, 64, 128, 256, 512], help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')

    # b) Feature extractor network
    parser.add_argument('--feature_extractor_net', type=str, default='res_net_x18', choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to extract the features from the images')
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
    parser.add_argument('--classifier_net', type=str, default='res_net_x18', choices=['conv_net', 'res_net_x18', 'res_net_x34'], help='The network which will be built to classify the samples')
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