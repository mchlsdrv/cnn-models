import tensorflow as tf
from tensorflow import keras

cosine_similarity_loss = keras.losses.CosineSimilarity(
    axis=-1,
    name='cosine_similarity'
)


class SCANLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-7

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        btch_sz = y_true.shape[0]
        norm = tf.cast(tf.constant(1/btch_sz), y_true.dtype)

        # I) The consistancy
        dot_prod = tf.tensordot(y_true, tf.transpose(y_pred), axes=1)
        diag_mask = tf.cast(tf.eye(btch_sz), dot_prod.dtype)
        btch_dot_prod = tf.math.reduce_sum(dot_prod * diag_mask, 0)
        btch_dot_prod_log = tf.math.log(btch_dot_prod + self.epsilon)
        mean_btch_dot_prod_log = norm * btch_dot_prod_log

        # II) The entropy
        mean_class_prob = norm * tf.reduce_sum(y_pred, 0)
        mean_entropy = tf.math.reduce_sum(mean_class_prob * tf.math.log(mean_class_prob + self.epsilon))

        loss = -mean_btch_dot_prod_log + mean_entropy

        return loss
