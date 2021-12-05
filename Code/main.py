import os
import shutil
import glob
import cv2
import datetime
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Progbar
from utils import aux_funcs
from data_utils import data_funcs
from callbacks import clustering_callbacks
from losses import clustering_losses
from functools import partial
import threading
import logging
import logging.config

from augmentations import clustering_augmentations
from configs.general_configs import (
    CONFIGS_DIR_PATH,
    TRAIN_DATA_DIR_PATH,
    TEST_DATA_DIR_PATH,
    OUTPUT_DIR_PATH,
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
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LEARNING_RATE = 1e-4


if __name__ == '__main__':
    parser = aux_funcs.get_arg_parcer()
    args = parser.parse_args()


    current_run_dir = OUTPUT_DIR_PATH / f'{TS}'
    os.makedirs(current_run_dir, exist_ok=True)

    logger = aux_funcs.get_logger(
        configs_file = CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file = current_run_dir / f'logs.log'
    )

    if isinstance(logger, logging.Logger):
        logger.info(tf.config.list_physical_devices('GPU'))

    aux_funcs.choose_gpu(gpu_id = args.gpu_id)

    input_image_shape = (args.crop_size, args.crop_size, 1)

    priors_knn_df = pd.read_pickle(args.knn_df_path)
    if priors_knn_df is None:
        # - Get the dataset
        if isinstance(logger, logging.Logger):
            logger.info(f'1) Feature Extraction')
            logger.info(f'- Construction the dataset ...')
        train_ds, val_ds = data_funcs.get_dataset_from_tiff(
            input_image_shape=input_image_shape,
            batch_size=args.feature_extractor_batch_size,
            validation_split=args.feature_extractor_validation_split
        )
        # - Train feature extractor
        if isinstance(logger, logging.Logger):
            logger.info(f'- Training the feature extractor model ...')
        feat_ext_model = aux_funcs.train_model(
            model_configs = dict(
                model_name = 'feature_extractor',
                model_type = args.feature_extractor_model_type,
                number_of_classes = args.feature_extractor_latent_space_dims,
                crop_size = args.crop_size,
                augmentations = clustering_augmentations.image_augmentations,
                custom_loss = clustering_losses.cosine_similarity_loss,
                checkpoint_dir = pathlib.Path(args.feature_extractor_checkpoint_dir),
            ),
            data = dict(
                train_dataset = train_ds,
                val_dataset = val_ds,
                X_sample = next(iter(val_ds))[0][0]
            ),
            callback_configs = dict(
                no_reduce_lr_on_plateau = args.no_reduce_lr_on_plateau_feature_extractor
            ),
            compile_configs = dict(
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = keras.optimizers.Adam(learning_rate=args.feature_extractor_optimizer_lr),
                metrics = ['accuracy']
            ),
            fit_configs = dict(
                batch_size = args.feature_extractor_batch_size,
                train_epochs = args.feature_extractor_train_epochs,
                train_steps_per_epoch = args.feature_extractor_train_steps_per_epoch,
                validation_steps_proportion = args.feature_extractor_validation_steps_proportion,
                valdation_freq = 1,  # [1, 100, 1500, ...] - validate on these epochs
                shuffle = True,
            ),
            general_configs = dict(
                time_stamp = TS
            ),
            logger=logger
        )

        if isinstance(logger, logging.Logger):
            logger.info(f'2) KNN Priors Search')
        priors_knn_df = aux_funcs.get_priors_knn_df(
            model=feat_ext_model,
            k=args.knn_k,
            train_data_dir=TRAIN_DATA_DIR_PATH,
            patch_height=args.crop_size,
            patch_width=args.crop_size,
            patch_optimization=args.knn_patch_optimization,
            save_dir=current_run_dir / 'knn_priors',
            knn_algorithm=args.knn_algorithm
        )

    assert priors_knn_df is not None, '\'priors_knn_df\' can\'t be None!'
    if isinstance(logger, logging.Logger):
        logger.info(f'''
> Priors Data Frame:
    - Columns: {priors_knn_df.columns}
    - Shape: {priors_knn_df.shape}
    - Example: {priors_knn_df.loc[0]}
    ''')

    # II) Classification
    if isinstance(logger, logging.Logger):
        logger.info(f'3) Classifier Model Training')
        logger.info(f'- Constructing the KNN dataset ...')
    # 1 - Construct the dataset
    # - Split into train and validation
    train_knn_data_set, val_knn_data_set = data_funcs.get_knn_dataset(
        priors_knn_df=priors_knn_df,
        k=1,
        val_prop=args.classifier_validation_split,
        image_shape=input_image_shape,
        train_batch_size=args.classifier_batch_size,
        val_batch_size=int(args.classifier_batch_size*args.classifier_validation_split),
        crops_per_batch=classifier_train_steps_per_epoch,
        shuffle=True
    )
    # train_idxs, val_idxs = aux_funcs.get_train_val_idxs(
    #     n_items=priors_knn_df.shape[0],
    #     val_prop=args.classifier_validation_split
    # )
    #
    # train_knn_data_set = data_funcs.KNNDataLoader(
    #     knn_image_files=priors_knn_df.loc[train_idxs].reset_index(drop=True),
    #     k=1,
    #     image_shape=input_image_shape,
    #     batch_size=args.classifier_batch_size,
    #     crops_per_batch=args.classifier_train_steps_per_epoch,
    #     shuffle=True
    # )
    #
    # val_knn_data_set = data_funcs.KNNDataLoader(
    #     knn_image_files=priors_knn_df.loc[val_idxs].reset_index(drop=True),
    #     k=1,
    #     image_shape=input_image_shape,
    #     batch_size=2, #int(args.classifier_batch_size*args.classifier_validation_split),
    #     crops_per_batch=args.classifier_train_steps_per_epoch,
    #     shuffle=True
    # )

    if isinstance(logger, logging.Logger):
        logger.info(f'- Training the classifier ...')
    classifier_model = aux_funcs.train_model(
        model_configs = dict(
            model_name = 'classifier',
            model_type = args.classifier_model_type,
            number_of_classes = args.classifier_number_of_classes,
            crop_size = args.crop_size,
            augmentations = clustering_augmentations.image_augmentations,
            checkpoint_dir = pathlib.Path(args.classifier_checkpoint_dir),
        ),
        data = dict(
            train_dataset = train_knn_data_set,
            val_dataset = val_knn_data_set,
            X_sample = val_knn_data_set.get_sample()[0][0],
        ),
        callback_configs = dict(
            no_reduce_lr_on_plateau = args.no_reduce_lr_on_plateau_classifier
        ),
        compile_configs = dict(
            loss = clustering_losses.SCANLoss(), #keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer = keras.optimizers.Adam(learning_rate=args.classifier_optimizer_lr),
            # metrics = ['accuracy']
        ),
        fit_configs = dict(
            batch_size = args.classifier_batch_size,
            train_epochs = args.classifier_train_epochs,
            train_steps_per_epoch = args.classifier_train_steps_per_epoch,
            validation_steps_proportion = args.classifier_validation_steps_proportion,
            valdation_freq = 1,  # [1, 100, 1500, ...] - validate on these epochs
            shuffle = True,
        ),
        general_configs = dict(
            time_stamp = TS
        ),
        logger=logger
    )

    # 4) Classify
    # - Train images
    if isinstance(logger, logging.Logger):
        logger.info(f'4) Classification')
        logger.info(f'- Classifing train data...')
    aux_funcs.classify(
        model=classifier_model,
        images_root_dir=TRAIN_DATA_DIR_PATH,
        patch_height=args.crop_size,
        patch_width=args.crop_size,
        output_dir=OUTPUT_DIR_PATH / f'{TS}/classified/train',
        logger=logger
    )

    # - Test images
    if isinstance(logger, logging.Logger):
        logger.info(f'- Classifing test data...')
    aux_funcs.classify(
        model=classifier_model,
        images_root_dir=TEST_DATA_DIR_PATH,
        patch_height=args.crop_size,
        patch_width=args.crop_size,
        output_dir=OUTPUT_DIR_PATH / f'{TS}/classified/test',
        logger=logger
    )
