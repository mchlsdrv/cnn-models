import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


class UnsupervidsedModelTrainer:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, dataset, n_epochs, loss, optimizer, metric, augs):
        for epoch in range(n_epochs):
            for btch_idx, (x, x_val) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    # - Run the augmented images throught the network to get their latent representation
                    x_aug_latent = self.model(augs(x), training=True)
                    # - Run the original image throught the network to get the latent representation of
                    # the original image as the label
                    x_latent = self.model(x, training=False)

                    loss = loss(x_latent, x_aug_latent)

                grads = tape.gradients(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                metric.update_state(x_latent, x_aug_latent)
            train_metric = metric.result()
            metric.reset_states()

if __name__=='__main__':
    pass
