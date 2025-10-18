import tensorflow as tf

class WeightRegularizationLoss:
  def __init__ (self):
    self.name = "weight_regularization"

  def __call__(self, weights):
    """
    weights: List [tensor]
    """
    loss = tf.constant(0.0)
    n = 0.0
    for weight in weights:
      loss += tf.reduce_sum(tf.square(weight))
      n += tf.reduce_prod(tf.shape(weight))

    loss /= n
    return loss