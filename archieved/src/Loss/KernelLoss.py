import tensorflow as tf
from .KernelLoss import KernelLoss
from .WeightedXCE import WeightedXCE


def rbf_kernel_matrix_broadcast(X, Y, sigma):
    """
    Compute the RBF kernel matrix for a tensor using broadcasting.

    Args:
        X: Tensor of shape (B, N, D), where B is the batch size,
           N is the number of samples (flattened spatial dimensions),
           and D is the feature dimension.
        sigma: Bandwidth parameter for the RBF kernel.

    Returns:
        Kernel matrix of shape (B, N, N).
    """
    X_norm = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)  # Shape: (B, N, 1)
    Y_norm = tf.reduce_sum(tf.square(Y), axis=-1, keepdims=True)  # Shape: (B, N, 1)
    pairwise_distances = X_norm - 2 * tf.matmul(X, tf.transpose(Y, perm=[0, 2, 1])) + tf.transpose(Y_norm, perm=[0, 2, 1])  # Shape: (B, N, N)

    kernel_matrix = tf.exp(-pairwise_distances / (2 * sigma**2))
    return kernel_matrix

def mean_point_wise_hsic_score(block1, block2, sigma=1.0):
    """
    Compute the HSIC between two convolutional feature blocks using broadcasting.

    Args:
        block1: Tensor of shape (B, H, H, C), where B is the batch size,
                H x H are spatial dimensions, and C is the number of channels.
        block2: Tensor of shape (B, H, H, C), same dimensions as block1.
        sigma: Bandwidth parameter for the RBF kernel.

    Returns:
        HSIC values of shape (B, H^2), where each element represents the HSIC value for a spatial position.
    """
    B, H, _, C = block1.shape

    # Flatten the spatial dimensions (H x H) into a single dimension for each batch
    block1_flat = tf.reshape(block1, (-1, H * H, C))  # Shape: (B, H^2, C)
    block2_flat = tf.reshape(block2, (-1, H * H, C))  # Shape: (B, H^2, C)

    # Compute RBF kernel matrices
    K1 = rbf_kernel_matrix_broadcast(block1_flat,block1_flat, sigma)  # Shape: (B, H^2, H^2)
    K2 = rbf_kernel_matrix_broadcast(block2_flat,block2_flat, sigma)  # Shape: (B, H^2, H^2)

    # Center the kernel matrices
    n = H * H
    H_center = tf.eye(n) - tf.ones((n, n)) / tf.cast(n, tf.float32)  # Shape: (H^2, H^2)
    # H_center = tf.broadcast_to(H_center, (B, n, n))  # Shape: (B, H^2, H^2)
    H_center = tf.expand_dims(H_center, axis=0)

    K1_centered = tf.matmul(H_center, tf.matmul(K1, H_center))  # Shape: (B, H^2, H^2)
    K2_centered = tf.matmul(H_center, tf.matmul(K2, H_center))  # Shape: (B, H^2, H^2)

    # Compute HSIC values
    hsic_values = tf.reduce_sum(K1_centered * K2_centered, axis=[1, 2]) / tf.cast(n**2, tf.float32)  # Shape: (B,)
    # print(hsic_values)
    return hsic_values


class KernelLoss:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.name = "kernel_loss"

    def __call__(self, feature_blocks):
      """
      feature_blocks : [f1 , f2 , f3, ...] Outputs from diff heads
      return : hsic(f1,f2) + hsic(f2,f3) + hsic(f1,f3) + ... + hisic(fi,fj) for all i != j
      """
      total_loss = tf.constant(0.0)
      for i in range(len(feature_blocks)):
        for j in range(i+1, len(feature_blocks)):
          hsic_value = mean_point_wise_hsic_score(feature_blocks[i], feature_blocks[j], sigma=self.sigma)
          total_loss += tf.reduce_mean(hsic_value)
      return total_loss 