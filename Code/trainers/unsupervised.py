import copy
import tensorflow as tf
from tensorflow.keras.utils import Progbar


class UnsupervisedModelTrainer:
    def __init__(self, name, model, optimizer, loss, metric, callbacks):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.val_metric = copy.deepcopy(metric)

        for callback in callbacks:
            callback.set_model(self.model)


    @tf.function
    def train_step(self, X, X_aug):
        with tf.GradientTape() as tape:
            # - Run the augmented images throught the network to get their latent representation
            X_aug_latent = self.model(X_aug, training=True)

            # - Run the original image throught the network to get the latent representation of
            # the original image as the label
            X_latent = self.model(X, training=False)

            loss = self.loss(X_latent, X_aug_latent)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric.update_state(X_latent, X_aug_latent)

        return loss

    @tf.function
    def test_step(self, X, X_aug):
        X_latent = self.model(X, training=False)
        X_aug_latent = self.model(X_aug, training=False)

        self.val_metric.update_state(X_latent, X_aug_latent)
        return self.val_metric.result()

    def fit(self, dataset, n_epochs, augs):
        X_val = dataset.get_val_data()
        for epoch in range(n_epochs):
            print(f'epoch {epoch}/{n_epochs}')
            progress_bar = Progbar()
            for btch_idx, X in enumerate(dataset):
                X_aug = augs(X)
                batch_loss = self.train_step(X, X_aug)
            train_metric = self.metric.result()
            self.metric.reset_states()

            # Validation
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
