import tensorflow as tf

class WeightedXCE():
    def __init__(self):
        self.name = "weighted_xentropy"

    def __call__(self, y_true, y_pred):
        """
        Args:
          y_true: 1 hot encoded array
          y_pred: Prediction with same shape as y_true
        """
        labels = tf.cast(tf.argmax(y_true, axis=1), dtype=tf.float32)
        class_freq = tf.reduce_sum(y_true, axis=0)+1e-9
        class_weights = 1./class_freq

        effective_weights = tf.reduce_sum(class_weights * y_true, axis = -1)

        unweighted_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        weighted_losses = tf.multiply(unweighted_losses, effective_weights)
        return tf.reduce_mean(weighted_losses)