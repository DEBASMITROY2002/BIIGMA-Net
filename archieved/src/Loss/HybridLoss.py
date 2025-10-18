import tensorflow as tf
from .KernelLoss import KernelLoss
from .WeightedXCE import WeightedXCE
from .WeightRegularizationLoss import WeightRegularizationLoss

class HybridLoss:
    def __init__(self, kernel_loss_factor = 0.5,
                 weight_regularization_loss_factor = 0.5,
                 weighted_xentropy1_factor = 0.5,
                 weighted_xentropy2_factor = 0.5,
                 map_dist_factor = 0.5,
                 kld_factor = 0.5,
                 sigma = 1.0):

        self.kernel_loss_factor = kernel_loss_factor
        self.weight_regularization_loss_factor = weight_regularization_loss_factor
        self.weighted_xentropy1_factor = weighted_xentropy1_factor
        self.weighted_xentropy2_factor = weighted_xentropy2_factor
        self.map_dist_factor = map_dist_factor
        self.kld_factor = kld_factor

        self.name = "hybrid_loss"
        self.kernel_loss = KernelLoss(sigma=sigma)
        self.weight_regularization_loss = WeightRegularizationLoss()
        self.weighted_xentropy = WeightedXCE()
        self.kld = tf.keras.losses.KLDivergence()

    def __call__(self, y_true, y_pred1, y_pred2, feature_blocks, weights, map1, map2, softlogit1, softlogit2):
        kernel_loss = self.kernel_loss(feature_blocks)
        # weight_regularization_loss =0.0# self.weight_regularization_loss(weights) # Temporarily deactivated
        weighted_xentropy1 = self.weighted_xentropy(y_true, y_pred1)
        weighted_xentropy2 = self.weighted_xentropy(y_true, y_pred2)

        # Eucledian Dist l2 norm
        map_dist = tf.math.sqrt(tf.reduce_mean(tf.square(map1-map2)))

        # KLD
        # kld_val = self.kld(softlogit1, softlogit2)

        return kernel_loss ,  weighted_xentropy1, weighted_xentropy2, map_dist #, kld_val