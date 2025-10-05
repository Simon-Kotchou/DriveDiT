"""
Rotary Positional Embedding (RoPE) implementation.
Pure mathematical components with explicit tensor operations.
"""

import torch
import torch.nn as nn
from typing import Tuple


def rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor.
    
    Args:
        x: Input tensor [B, T, H, D] - half-rotate every two dims
        sin: Sine component for rotation
        cos: Cosine component for rotation
    
    Returns:
        Rotated tensor with same shape as input
    """
    # Split into pairs for rotation
    x1, x2 = x[..., 0::2], x[..., 1::2]
    
    # Apply rotation: [cos -sin; sin cos] @ [x1; x2]
    rotated = torch.cat([
        x1 * cos - x2 * sin,  # Real part
        x1 * sin + x2 * cos   # Imaginary part
    ], dim=-1)
    
    return rotated


def precompute_rope_freqs(
    dim: int, 
    max_seq_len: int, 
    base: float = 10000.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute sine and cosine frequencies for RoPE.
    
    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base frequency for geometric progression
        device: Device to place tensors on
    
    Returns:
        Tuple of (sin, cos) tensors [max_seq_len, dim//2]
    """
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE"
    
    # Compute frequency for each dimension pair
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    
    # Position indices
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    
    # Outer product to get all position-frequency combinations
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]
    
    return torch.sin(freqs), torch.cos(freqs)


def precompute_rope_3d_freqs(
    dim: int,
    max_time: int,
    max_height: int, 
    max_width: int,
    base: float = 10000.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute 3D RoPE frequencies for (time, height, width) factorization.
    Following V-JEPA-2 approach with 3-axis factorization.
    
    Args:
        dim: Embedding dimension (must be divisible by 6 for 3D)
        max_time: Maximum time sequence length
        max_height: Maximum height
        max_width: Maximum width
        base: Base frequency
        device: Device to place tensors on
    
    Returns:
        Tuple of (sin, cos) tensors [max_time, max_height, max_width, dim]
    """
    assert dim % 6 == 0, "Embedding dimension must be divisible by 6 for 3D RoPE"
    
    dim_per_axis = dim // 3
    
    # Compute frequencies for each axis
    freqs_t = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=torch.float32, device=device) / dim_per_axis))
    freqs_h = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=torch.float32, device=device) / dim_per_axis))
    freqs_w = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=torch.float32, device=device) / dim_per_axis))
    
    # Position grids
    t_pos = torch.arange(max_time, dtype=torch.float32, device=device)
    h_pos = torch.arange(max_height, dtype=torch.float32, device=device)
    w_pos = torch.arange(max_width, dtype=torch.float32, device=device)
    
    # Compute frequencies for each axis
    t_freqs = torch.outer(t_pos, freqs_t)  # [T, dim_per_axis//2]
    h_freqs = torch.outer(h_pos, freqs_h)  # [H, dim_per_axis//2]
    w_freqs = torch.outer(w_pos, freqs_w)  # [W, dim_per_axis//2]
    
    # Broadcast to full 3D grid
    t_freqs = t_freqs[:, None, None, :]  # [T, 1, 1, dim_per_axis//2]
    h_freqs = h_freqs[None, :, None, :]  # [1, H, 1, dim_per_axis//2]
    w_freqs = w_freqs[None, None, :, :]  # [1, 1, W, dim_per_axis//2]
    
    # Expand to full dimensions
    t_freqs = t_freqs.expand(max_time, max_height, max_width, -1)
    h_freqs = h_freqs.expand(max_time, max_height, max_width, -1)
    w_freqs = w_freqs.expand(max_time, max_height, max_width, -1)
    
    # Concatenate along last dimension
    all_freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)
    
    return torch.sin(all_freqs), torch.cos(all_freqs)


class RoPELayer(nn.Module):
    """
    RoPE layer that can be used as a module.
    Pre-computes and caches frequency tables.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Register buffers so they move with the model
        sin, cos = precompute_rope_freqs(dim, max_seq_len, base)
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor [B, T, H, D]
            start_pos: Starting position for KV caching
        
        Returns:
            Rotated tensor
        """
        seq_len = x.size(1)
        end_pos = start_pos + seq_len
        
        # Get frequency slices for this sequence
        sin = self.sin[start_pos:end_pos, :]  # [T, D//2]
        cos = self.cos[start_pos:end_pos, :]  # [T, D//2]
        
        # Expand for batch and head dimensions
        sin = sin[None, :, None, :]  # [1, T, 1, D//2]
        cos = cos[None, :, None, :]  # [1, T, 1, D//2]
        
        return rope(x, sin, cos)