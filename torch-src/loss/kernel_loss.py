import torch
import torch.nn as nn

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
    X_norm = torch.sum(X.pow(2), dim=-1, keepdim=True)  # Shape: (B, N, 1)
    Y_norm = torch.sum(Y.pow(2), dim=-1, keepdim=True)  # Shape: (B, N, 1)
    pairwise_distances = X_norm - 2 * torch.matmul(X, Y.transpose(1, 2)) + Y_norm.transpose(1, 2)  # Shape: (B, N, N)

    kernel_matrix = torch.exp(-pairwise_distances / (2 * sigma**2))
    return kernel_matrix

def mean_point_wise_hsic_score(block1, block2, sigma=1.0):
    """
    Compute the HSIC between two convolutional feature blocks using broadcasting.

    Args:
        block1: Tensor of shape (B, C, H, W), where B is the batch size,
                C is the number of channels, and H x W are spatial dimensions.
        block2: Tensor of shape (B, C, H, W), same dimensions as block1.
        sigma: Bandwidth parameter for the RBF kernel.

    Returns:
        HSIC values of shape (B,), where each element represents the HSIC value for a spatial position.
    """
    # Permute to (B, H, W, C) to match TensorFlow's channel-last format if needed
    block1 = block1.permute(0, 2, 3, 1)
    block2 = block2.permute(0, 2, 3, 1)
    
    B, H, W, C = block1.shape
    device = block1.device

    # Flatten the spatial dimensions (H x W) into a single dimension for each batch
    block1_flat = block1.reshape(B, H * W, C)  # Shape: (B, H*W, C)
    block2_flat = block2.reshape(B, H * W, C)  # Shape: (B, H*W, C)

    # Compute RBF kernel matrices
    K1 = rbf_kernel_matrix_broadcast(block1_flat, block1_flat, sigma)  # Shape: (B, H*W, H*W)
    K2 = rbf_kernel_matrix_broadcast(block2_flat, block2_flat, sigma)  # Shape: (B, H*W, H*W)

    # Center the kernel matrices
    n = H * W
    H_center = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
    H_center = H_center.unsqueeze(0)  # Shape: (1, H*W, H*W)

    K1_centered = torch.matmul(H_center, torch.matmul(K1, H_center))  # Shape: (B, H*W, H*W)
    K2_centered = torch.matmul(H_center, torch.matmul(K2, H_center))  # Shape: (B, H*W, H*W)

    # Compute HSIC values
    hsic_values = torch.sum(K1_centered * K2_centered, dim=[1, 2]) / (n**2)  # Shape: (B,)
    return hsic_values

class KernelLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(KernelLoss, self).__init__()
        self.sigma = sigma
        self.name = "kernel_loss"

    def forward(self, feature_blocks):
      """
      feature_blocks : [f1, f2, f3, ...] Outputs from diff heads
      return : hsic(f1,f2) + hsic(f2,f3) + hsic(f1,f3) + ... + hsic(fi,fj) for all i != j
      """
      if not feature_blocks:
          return 0.0
          
      total_loss = torch.tensor(0.0, device=feature_blocks[0].device)
      for i in range(len(feature_blocks)):
        for j in range(i + 1, len(feature_blocks)):
          hsic_value = mean_point_wise_hsic_score(feature_blocks[i], feature_blocks[j], sigma=self.sigma)
          total_loss += torch.mean(hsic_value)
      return total_loss
