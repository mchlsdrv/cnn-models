import os
import datetime
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from aux_code import aux_funcs
from data_loaders import data_funcs
from callbacks import clustering_callbacks
from configs.general_configs import (
    CONFIGS_DIR_PATH,
    TRAIN_DATA_DIR_PATH,
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

    print(tf.config.list_physical_devices('GPU'))

    aux_funcs.choose_gpu(gpu_id = args.gpu_id)

    current_run_dir = OUTPUT_DIR_PATH / f'{TS}'
    if not current_run_dir.is_dir():
        os.makedirs(current_run_dir)

    input_image_shape = (args.crop_size, args.crop_size, 1)

    # 1) Build the feature extractor model
    feat_ext_model = aux_funcs.get_model(
        net_name='feature_extractor',
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
            validation_split=args.feature_extractor_validation_split
        )

        # 3) Configure callbacks
        callbacks = clustering_callbacks.get_callbacks(
            model=feat_ext_model,
            X=next(iter(val_ds))[0][0],
            ts=TS,
            no_reduce_lr_on_plateau=args.no_reduce_lr_on_plateau_feature_extractor
        )

        # 4) Compile
        feat_ext_model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(learning_rate=args.feature_extractor_optimizer_lr),
            metrics=['accuracy']
        )

        # 5) Fit feature extraction model
        validation_steps = int(args.feature_extractor_validation_steps_proportion * args.feature_extractor_train_steps_per_epoch) if 0 < int(args.feature_extractor_validation_steps_proportion * args.feature_extractor_train_steps_per_epoch) <= args.feature_extractor_train_steps_per_epoch else 1
        feat_ext_model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=args.batch_size,
            epochs=args.feature_extractor_train_epochs,
            steps_per_epoch=args.feature_extractor_train_steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=1,  # [1, 100, 1500, ...] - validate on these epochs
            shuffle=True,
            callbacks=callbacks
        )


    if args.knn_patch_optimization:
        priors_knn_df = aux_funcs.get_patch_transforms(images_root_dir=TRAIN_DATA_DIR_PATH, model=feat_ext_model, patch_height=args.crop_size, patch_width=args.crop_size)
        X = np.array([x for x in priors_knn_df.loc[:, 'patch_transform'].values])
    else:
        priors_knn_df = aux_funcs.get_mean_image_transforms(images_root_dir=TRAIN_DATA_DIR_PATH, model=feat_ext_model, patch_height=args.crop_size, patch_width=args.crop_size)
        X = np.array([x for x in priors_knn_df.loc[:, 'image_mean_transform'].values])
    files = priors_knn_df.loc[:, 'file'].values

    nbrs_distances, nbrs_files = aux_funcs.get_knn_files(X=X, files=files, k=args.knn_k, algorithm=args.knn_algorithm)

    priors_knn_df.loc[:, 'distances'] = nbrs_distances
    priors_knn_df.loc[:, 'neighbors'] = nbrs_files

    knn_priors_output_dir = current_run_dir / 'knn_priors'

    if aux_funcs.check_dir(knn_priors_output_dir):
        priors_knn_df.to_pickle(knn_priors_output_dir / f'priors_knn_df.pkl')
        aux_funcs.plot_knn(knn=priors_knn_df, save_dir=knn_priors_output_dir)

    # # 3) Build the feature extractor model
    # labeller_model = aux_funcs.get_model(
    #     net_name='classifier',
    #     model_type=args.labeller_net,
    #     number_of_classes=args.number_of_classes,
    #     crop_size=args.crop_size,
    #     priors=priors_knn_df,
    # )
    #
    #
    # train_ds, val_ds = data_funcs.get_dataset_from_tiff(
    #     input_image_shape=input_image_shape,
    #     batch_size=args.batch_size,
    #     validation_split=args.validation_split
    # )
    #
    # # 3) Configure callbacks
    # callbacks = clustering_callbacks.get_callbacks(
    #     model=feat_ext_model,
    #     X=next(iter(val_ds))[0][0],
    #     ts=ts
    # )
    #
    # # 4) Compile
    # labeller_model.compile(
    #     # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #     metrics=['accuracy']
    # )
    #
    # # 5) Fit feature extraction model
    # validation_steps = int(args.validation_steps_proportion * args.steps_per_epoch) if 0 < int(args.validation_steps_proportion * args.steps_per_epoch) <= args.steps_per_epoch else 1
    # feat_ext_model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     batch_size=args.batch_size,
    #     epochs=args.epochs,
    #     steps_per_epoch=args.steps_per_epoch,
    #     validation_steps=validation_steps,
    #     validation_freq=1,  # [1, 100, 1500, ...] - validate on these epochs
    #     shuffle=True,
    #     callbacks=callbacks
    # )
