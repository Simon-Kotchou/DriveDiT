"""
Mathematical utility functions.
Pure PyTorch implementations of common math operations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def gaussian_kernel(size: int, sigma: float, device: torch.device = None) -> torch.Tensor:
    """
    Create 2D Gaussian kernel.
    
    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation
        device: Device to place kernel on
    
    Returns:
        2D Gaussian kernel [size, size]
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    # Create 2D kernel
    kernel = g[:, None] * g[None, :]
    return kernel


def soft_clamp(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Soft clamping function using tanh.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Soft-clamped tensor
    """
    scale = (max_val - min_val) / 2
    offset = (max_val + min_val) / 2
    return scale * torch.tanh(x) + offset


def stable_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Numerically stable softmax with temperature.
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        temperature: Temperature parameter
    
    Returns:
        Softmax probabilities
    """
    x = x / temperature
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    return F.softmax(x_shifted, dim=dim)


def log_sum_exp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Numerically stable log-sum-exp.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce
        keepdim: Whether to keep dimensions
    
    Returns:
        Log-sum-exp result
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    return x_max + torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=keepdim))


def gumbel_noise(shape: Tuple[int, ...], device: torch.device = None) -> torch.Tensor:
    """
    Generate Gumbel noise.
    
    Args:
        shape: Shape of noise tensor
        device: Device to place tensor on
    
    Returns:
        Gumbel noise tensor
    """
    uniform = torch.rand(shape, device=device)
    return -torch.log(-torch.log(uniform + 1e-8) + 1e-8)


def gumbel_softmax(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    hard: bool = False, 
    dim: int = -1
) -> torch.Tensor:
    """
    Gumbel-Softmax sampling.
    
    Args:
        logits: Input logits
        temperature: Gumbel temperature
        hard: Whether to use hard sampling
        dim: Dimension to apply softmax
    
    Returns:
        Gumbel-softmax samples
    """
    gumbel = gumbel_noise(logits.shape, device=logits.device)
    y = (logits + gumbel) / temperature
    y_soft = F.softmax(y, dim=dim)
    
    if hard:
        # Straight-through estimator
        index = y_soft.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft


def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        x: First tensor [B, D]
        y: Second tensor [B, D]
    
    Returns:
        Cosine similarity matrix [B, B]
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return torch.mm(x_norm, y_norm.t())


def pairwise_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix.
    
    Args:
        x: First tensor [B, D]
        y: Second tensor [B, D]
    
    Returns:
        Distance matrix [B, B]
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    xy = torch.mm(x, y.t())
    return torch.sqrt(x_norm + y_norm - 2 * xy + 1e-8)


def smoothing_kernel(size: int, smoothing_type: str = 'gaussian') -> torch.Tensor:
    """
    Create smoothing kernel.
    
    Args:
        size: Kernel size
        smoothing_type: Type of smoothing ('gaussian', 'box', 'triangle')
    
    Returns:
        Smoothing kernel
    """
    if smoothing_type == 'gaussian':
        sigma = size / 6.0  # Cover 3 standard deviations
        return gaussian_kernel(size, sigma)
    elif smoothing_type == 'box':
        kernel = torch.ones(size, size) / (size * size)
        return kernel
    elif smoothing_type == 'triangle':
        coords = torch.arange(size, dtype=torch.float32)
        coords = 1 - torch.abs(coords - size // 2) / (size // 2)
        kernel = coords[:, None] * coords[None, :]
        return kernel / kernel.sum()
    else:
        raise ValueError(f"Unknown smoothing type: {smoothing_type}")


def finite_difference_gradient(
    x: torch.Tensor, 
    spacing: float = 1.0, 
    edge_order: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute finite difference gradients.
    
    Args:
        x: Input tensor [B, C, H, W]
        spacing: Grid spacing
        edge_order: Order of edge approximation
    
    Returns:
        Tuple of (grad_y, grad_x)
    """
    # Compute gradients using finite differences
    if edge_order == 1:
        # Forward/backward differences at edges
        grad_y = torch.zeros_like(x)
        grad_x = torch.zeros_like(x)
        
        # Interior points (central difference)
        grad_y[:, :, 1:-1, :] = (x[:, :, 2:, :] - x[:, :, :-2, :]) / (2 * spacing)
        grad_x[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / (2 * spacing)
        
        # Edge points (forward/backward difference)
        grad_y[:, :, 0, :] = (x[:, :, 1, :] - x[:, :, 0, :]) / spacing
        grad_y[:, :, -1, :] = (x[:, :, -1, :] - x[:, :, -2, :]) / spacing
        grad_x[:, :, :, 0] = (x[:, :, :, 1] - x[:, :, :, 0]) / spacing
        grad_x[:, :, :, -1] = (x[:, :, :, -1] - x[:, :, :, -2]) / spacing
        
    else:
        # Simple central difference for now
        grad_y = (torch.roll(x, -1, dims=2) - torch.roll(x, 1, dims=2)) / (2 * spacing)
        grad_x = (torch.roll(x, -1, dims=3) - torch.roll(x, 1, dims=3)) / (2 * spacing)
    
    return grad_y, grad_x


def laplacian(x: torch.Tensor, spacing: float = 1.0) -> torch.Tensor:
    """
    Compute 2D Laplacian using finite differences.
    
    Args:
        x: Input tensor [B, C, H, W]
        spacing: Grid spacing
    
    Returns:
        Laplacian tensor
    """
    # Second derivatives
    laplace_y = (torch.roll(x, -1, dims=2) - 2 * x + torch.roll(x, 1, dims=2)) / (spacing ** 2)
    laplace_x = (torch.roll(x, -1, dims=3) - 2 * x + torch.roll(x, 1, dims=3)) / (spacing ** 2)
    
    return laplace_y + laplace_x


def spectral_norm_power_iteration(
    weight: torch.Tensor, 
    u: torch.Tensor, 
    num_iterations: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Power iteration for spectral normalization.
    
    Args:
        weight: Weight tensor [out_features, in_features]
        u: Left singular vector estimate
        num_iterations: Number of power iterations
    
    Returns:
        Tuple of (new_u, spectral_norm)
    """
    with torch.no_grad():
        for _ in range(num_iterations):
            v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(weight, v), dim=0, eps=1e-12)
        
        sigma = torch.dot(u, torch.mv(weight, v))
    
    return u, sigma


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    """
    Compute trace for batch of matrices.
    
    Args:
        x: Batch of square matrices [B, N, N]
    
    Returns:
        Traces [B]
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def batch_determinant(x: torch.Tensor) -> torch.Tensor:
    """
    Compute determinant for batch of matrices.
    
    Args:
        x: Batch of square matrices [B, N, N]
    
    Returns:
        Determinants [B]
    """
    return torch.det(x)


def matrix_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute matrix square root using SVD.
    
    Args:
        x: Input matrix [..., N, N]
        eps: Small value for numerical stability
    
    Returns:
        Matrix square root
    """
    U, S, V = torch.svd(x)
    S = torch.clamp(S, min=eps)
    return U @ torch.diag_embed(torch.sqrt(S)) @ V.transpose(-2, -1)


def frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor, 
                    mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    Compute Fréchet distance between two multivariate Gaussians.
    
    Args:
        mu1: Mean of first distribution [D]
        sigma1: Covariance of first distribution [D, D]
        mu2: Mean of second distribution [D]
        sigma2: Covariance of second distribution [D, D]
    
    Returns:
        Fréchet distance
    """
    # Mean difference
    mu_diff = mu1 - mu2
    mu_term = torch.dot(mu_diff, mu_diff)
    
    # Trace terms
    trace_term = torch.trace(sigma1) + torch.trace(sigma2)
    
    # Cross term
    sqrt_sigma1 = matrix_sqrt(sigma1)
    cross_term = 2 * torch.trace(matrix_sqrt(sqrt_sigma1 @ sigma2 @ sqrt_sigma1))
    
    return mu_term + trace_term - cross_term


def wasserstein_distance_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Wasserstein distance using sorted samples.
    
    Args:
        x: First sample [N]
        y: Second sample [M]
    
    Returns:
        Wasserstein distance
    """
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    
    # Interpolate to common grid
    n, m = len(x_sorted), len(y_sorted)
    if n != m:
        # Simple approach: pad shorter sequence
        if n < m:
            x_sorted = F.pad(x_sorted, (0, m - n), value=x_sorted[-1])
        else:
            y_sorted = F.pad(y_sorted, (0, n - m), value=y_sorted[-1])
    
    return torch.mean(torch.abs(x_sorted - y_sorted))


def sinkhorn_divergence(
    x: torch.Tensor, 
    y: torch.Tensor, 
    reg: float = 0.1, 
    num_iters: int = 100
) -> torch.Tensor:
    """
    Compute Sinkhorn divergence (regularized optimal transport).
    
    Args:
        x: First point cloud [N, D]
        y: Second point cloud [M, D]
        reg: Regularization parameter
        num_iters: Number of Sinkhorn iterations
    
    Returns:
        Sinkhorn divergence
    """
    # Cost matrix (squared Euclidean distance)
    C = pairwise_distance_matrix(x, y) ** 2
    
    # Initialize dual variables
    u = torch.zeros(len(x), device=x.device)
    v = torch.zeros(len(y), device=y.device)
    
    # Sinkhorn iterations
    for _ in range(num_iters):
        u = reg * torch.log(torch.sum(torch.exp((v[None, :] - C) / reg), dim=1) + 1e-8)
        v = reg * torch.log(torch.sum(torch.exp((u[:, None] - C) / reg), dim=0) + 1e-8)
    
    # Compute transport plan
    P = torch.exp((u[:, None] + v[None, :] - C) / reg)
    
    # Sinkhorn divergence
    return torch.sum(P * C)