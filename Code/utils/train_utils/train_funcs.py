import logging
import numpy as np
from callbacks import clustering_callbacks


def get_train_val_idxs(n_items, val_prop):
    all_idxs = np.arange(n_items)
    val_idxs = np.random.choice(all_idxs, int(val_prop * n_items), replace=False)
    train_idxs = np.setdiff1d(all_idxs, val_idxs)
    return train_idxs, val_idxs


def train_model(model, data: dict, callback_configs: dict, compile_configs: dict, fit_configs: dict, general_configs: dict, logger: logging.Logger = None):

    # 2 - Train model
    # 2.2 Configure callbacks
    callbacks, tensor_board_th = clustering_callbacks.get_callbacks(
        model=model,
        X=data.get('X_sample'),
        ts=general_configs.get('time_stamp'),
        output_dir_path=callback_configs.get('output_dir_path'),
        no_reduce_lr_on_plateau=callback_configs.get('no_reduce_lr_on_plateau')
    )

    # 2.3 Compile model
    model.compile(
        loss=compile_configs.get('loss'),
        optimizer=compile_configs.get('optimizer'),
        metrics=compile_configs.get('metrics')
    )

    # 2.4 Fit model
    validation_steps = int(fit_configs.get('validation_steps_proportion') * fit_configs.get('train_steps_per_epoch')) if 0 < int(fit_configs.get('validation_steps_proportion') * fit_configs.get('train_steps_per_epoch')) <= fit_configs.get('train_steps_per_epoch') else 1
    model.fit(
        data.get('train_dataset'),
        batch_size=fit_configs.get('batch_size'),
        epochs=fit_configs.get('train_epochs'),
        steps_per_epoch=fit_configs.get('train_steps_per_epoch'),
        validation_data=data.get('val_dataset'),
        validation_steps=validation_steps,
        validation_freq=fit_configs.get('valdation_freq'),  # [1, 100, 1500, ...] - validate on these epochs
        shuffle=fit_configs.get('shuffle'),
        callbacks=callbacks
    )
    return model
