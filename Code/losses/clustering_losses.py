import tensorflow as tf
from tensorflow import keras
import numpy as np

from configs.general_configs import (
    EPSILON,
)


cosine_similarity_loss = keras.losses.CosineSimilarity(
    axis=-1,
    name='cosine_similarity'
)


class SCANLoss(keras.losses.Loss):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, D, N):
        consistency_loss = entropy_loss = 0.0
        class_probs = list()
        # I) For each sample in the current batch
        for i, X in enumerate(D):
            # - Add the embeding of the original image

            phi_X = self.model(np.expand_dims(X, axis=0), training=True)
            # print(f'phi_X: ', phi_X)
            class_probs.append(phi_X)

            # II) Get the neighbors of the current image
            N_X = N[i]
            # - For each neighbor of the original image do:
            for ngbr in N_X:
                # - Add the embeding of the neighbor to the batch list
                phi_ngbr = self.model(np.expand_dims(ngbr, axis=0), training=True)
                # print(f'phi_ngbr: ', phi_ngbr)

                # - Calculate the consistency part of the SCAN loss
                dot_prod = tf.cast(tf.tensordot(phi_X, tf.transpose(phi_ngbr), axes=1), dtype=tf.float16)
                # print(f'dot_prod: ', dot_prod)
                consistency_loss += tf.math.log(dot_prod + EPSILON)
                # print(f'consistency_loss: ', consistency_loss)

        # III) Calculate the consistency loss
        consistency_loss = 1 / D.shape[0] * consistency_loss

        # IV) Calculate the entropy loss
        mean_class_probs = tf.reduce_mean(class_probs, 0)
        # print(f'mean_class_probs: ', mean_class_probs)
        entropy = mean_class_probs * tf.math.log(mean_class_probs + EPSILON)
        # print(f'entropy: ', entropy)
        entropy_loss = tf.reduce_sum(entropy)
        # print(f'entropy_loss: ', entropy_loss)
        entropy_loss = tf.cast(entropy_loss, consistency_loss.dtype)
        # print('consistency loss:', consistency_loss)
        # print('entropy loss:', entropy_loss)
        # print('> SCAN loss:', -consistency_loss + entropy_loss)
        loss = -consistency_loss + entropy_loss

        return loss
