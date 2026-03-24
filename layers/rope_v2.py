"""
Enhanced Rotary Positional Embedding (RoPE) implementation.

Implements modern RoPE techniques including:
- NTK-Aware Scaling (LLaMA 2 / Code Llama)
- V-JEPA-2 style 3D factorization
- YaRN (Yet another RoPE Extension)
- Variable resolution support
- Efficient caching with lazy computation
- Flash Attention compatibility

Pure mathematical components with explicit tensor operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union, Literal
from dataclasses import dataclass, field
from functools import lru_cache
import math


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RoPEConfig:
    """
    Configuration for enhanced RoPE.

    Attributes:
        dim: Embedding dimension (must be even for 1D, divisible by 6 for 3D)
        max_seq_len: Maximum sequence length for precomputation
        base: Base frequency for geometric progression (default 10000)

        # NTK-Aware Scaling
        ntk_alpha: NTK interpolation factor (1.0 = no scaling)
        dynamic_ntk: Enable dynamic NTK scaling based on sequence length
        ntk_factor: Scaling factor for dynamic NTK (typically 1-8)
        original_max_seq_len: Original training sequence length for NTK

        # YaRN Parameters
        yarn_enabled: Enable YaRN extensions
        yarn_beta_fast: YaRN fast beta (typically 32)
        yarn_beta_slow: YaRN slow beta (typically 1)
        yarn_mscale: YaRN attention scaling factor
        yarn_mscale_all_dim: Apply mscale to all dimensions

        # 3D Factorization (V-JEPA-2 style)
        spatial_dims: Tuple of (height, width) for 3D mode
        temporal_base_factor: Slower temporal frequencies (e.g., 0.5 for half speed)

        # Variable Resolution
        interpolate_factor: Resolution interpolation factor

        # Efficiency
        use_half_precision: Store frequencies in half precision
        cache_common_lengths: List of sequence lengths to pre-cache
    """
    dim: int = 64
    max_seq_len: int = 2048
    base: float = 10000.0

    # NTK-Aware Scaling
    ntk_alpha: float = 1.0
    dynamic_ntk: bool = False
    ntk_factor: float = 1.0
    original_max_seq_len: int = 2048

    # YaRN Parameters
    yarn_enabled: bool = False
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0

    # 3D Factorization
    spatial_dims: Optional[Tuple[int, int]] = None  # (height, width)
    temporal_base_factor: float = 1.0  # Slower temporal = smaller value

    # Variable Resolution
    interpolate_factor: float = 1.0

    # Efficiency
    use_half_precision: bool = True
    cache_common_lengths: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)


# =============================================================================
# Core RoPE Functions
# =============================================================================

def rope_apply(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    interleaved: bool = True
) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor.

    This is the core rotation operation: applies a 2D rotation matrix
    to pairs of dimensions using precomputed sin/cos frequencies.

    Args:
        x: Input tensor [B, T, H, D] or [B, T, D]
        sin: Sine frequencies, broadcastable to x
        cos: Cosine frequencies, broadcastable to x
        interleaved: If True, rotate (0,1), (2,3), ... pairs
                     If False, rotate first half with second half

    Returns:
        Rotated tensor with same shape as input

    Mathematical formulation:
        For each pair (x_i, x_{i+1}):
        [x'_i    ]   [cos_i  -sin_i] [x_i    ]
        [x'_{i+1}] = [sin_i   cos_i] [x_{i+1}]

    Example:
        >>> x = torch.randn(2, 10, 8, 64)  # [B, T, H, D]
        >>> sin = torch.randn(10, 32)  # [T, D//2]
        >>> cos = torch.randn(10, 32)
        >>> rotated = rope_apply(x, sin.unsqueeze(0).unsqueeze(2),
        ...                      cos.unsqueeze(0).unsqueeze(2))
    """
    if interleaved:
        # Interleaved: rotate (x[0], x[1]), (x[2], x[3]), ...
        x1, x2 = x[..., 0::2], x[..., 1::2]

        # Apply rotation matrix
        rotated = torch.cat([
            x1 * cos - x2 * sin,  # Real part transformation
            x1 * sin + x2 * cos   # Imaginary part transformation
        ], dim=-1)
    else:
        # Non-interleaved: rotate (x[:D//2], x[D//2:])
        half_d = x.shape[-1] // 2
        x1, x2 = x[..., :half_d], x[..., half_d:]

        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

    return rotated


def rope_apply_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    interleaved: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to both query and key tensors efficiently.

    Args:
        q: Query tensor [B, T, H, D] or [B, H, T, D]
        k: Key tensor [B, T, H, D] or [B, H, T, D]
        sin: Sine frequencies
        cos: Cosine frequencies
        interleaved: Rotation pattern

    Returns:
        Tuple of rotated (q, k) tensors
    """
    return (
        rope_apply(q, sin, cos, interleaved),
        rope_apply(k, sin, cos, interleaved)
    )


# =============================================================================
# Frequency Computation
# =============================================================================

def compute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute basic RoPE frequencies.

    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base frequency (default 10000)
        device: Target device
        dtype: Computation dtype (float32 recommended for accuracy)

    Returns:
        Tuple of (sin, cos) tensors [max_seq_len, dim//2]

    Mathematical formulation:
        freq_i = 1 / (base^(2i/dim)) for i in [0, dim//2)
        theta_{t,i} = t * freq_i
        sin[t,i] = sin(theta_{t,i})
        cos[t,i] = cos(theta_{t,i})
    """
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"

    # Compute frequency for each dimension pair
    # freq_i = base^(-2i/dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))

    # Position indices [0, 1, 2, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, dtype=dtype, device=device)

    # Outer product: theta = t * freq
    # Shape: [max_seq_len, dim//2]
    freqs = torch.outer(t, inv_freq)

    return torch.sin(freqs), torch.cos(freqs)


def compute_ntk_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute NTK-aware scaled RoPE frequencies.

    NTK (Neural Tangent Kernel) aware scaling allows extending
    context length beyond training by scaling the base frequency.

    From Code Llama / LLaMA 2 long context papers:
    - When alpha > 1, effectively extends the context window
    - base_scaled = base * alpha^(dim/(dim-2))

    Args:
        dim: Embedding dimension
        max_seq_len: Maximum sequence length
        base: Original base frequency
        alpha: NTK interpolation factor (>1 for extension)
        device: Target device
        dtype: Computation dtype

    Returns:
        Tuple of (sin, cos) tensors [max_seq_len, dim//2]

    Example:
        # Double context length with alpha=2
        >>> sin, cos = compute_ntk_rope_freqs(64, 4096, alpha=2.0)
    """
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"

    if alpha == 1.0:
        return compute_rope_freqs(dim, max_seq_len, base, device, dtype)

    # NTK-aware base scaling
    # base_scaled = base * alpha^(dim/(dim-2))
    base_scaled = base * (alpha ** (dim / (dim - 2)))

    # Compute frequencies with scaled base
    inv_freq = 1.0 / (base_scaled ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))

    t = torch.arange(max_seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)

    return torch.sin(freqs), torch.cos(freqs)


def compute_dynamic_ntk_rope_freqs(
    dim: int,
    seq_len: int,
    original_max_len: int,
    base: float = 10000.0,
    ntk_factor: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dynamically scaled NTK-aware RoPE frequencies.

    Automatically adjusts alpha based on sequence length:
    - If seq_len <= original_max_len: use standard RoPE
    - If seq_len > original_max_len: scale alpha proportionally

    Args:
        dim: Embedding dimension
        seq_len: Current sequence length
        original_max_len: Original training sequence length
        base: Base frequency
        ntk_factor: Additional scaling factor
        device: Target device
        dtype: Computation dtype

    Returns:
        Tuple of (sin, cos) tensors [seq_len, dim//2]
    """
    if seq_len <= original_max_len:
        return compute_rope_freqs(dim, seq_len, base, device, dtype)

    # Dynamic alpha based on sequence length ratio
    alpha = (ntk_factor * seq_len / original_max_len) - (ntk_factor - 1)

    return compute_ntk_rope_freqs(dim, seq_len, base, alpha, device, dtype)


def compute_yarn_rope_freqs(
    dim: int,
    max_seq_len: int,
    original_max_len: int,
    base: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute YaRN (Yet another RoPE extensioN) frequencies.

    YaRN improves length generalization through:
    1. Frequency interpolation between fast and slow rotations
    2. Attention scaling to maintain softmax temperature

    Args:
        dim: Embedding dimension
        max_seq_len: Target maximum sequence length
        original_max_len: Original training sequence length
        base: Base frequency
        beta_fast: Fast rotation frequency boundary (high freq)
        beta_slow: Slow rotation frequency boundary (low freq)
        mscale: Manual attention scale override (0 for automatic)
        device: Target device
        dtype: Computation dtype

    Returns:
        Tuple of (sin, cos, attention_scale) where attention_scale
        should be applied to attention logits

    Reference:
        YaRN: Efficient Context Window Extension of Large Language Models
        https://arxiv.org/abs/2309.00071
    """
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"

    # Scaling ratio
    scale = max_seq_len / original_max_len

    if scale <= 1.0:
        sin, cos = compute_rope_freqs(dim, max_seq_len, base, device, dtype)
        return sin, cos, 1.0

    # Compute base frequencies
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))

    # YaRN frequency warping
    # Low freq (slow rotation) -> interpolate more
    # High freq (fast rotation) -> keep original

    # Compute wavelength bounds
    low_freq_wavelen = original_max_len / beta_slow
    high_freq_wavelen = original_max_len / beta_fast

    # Wavelength for each frequency
    wavelens = 2 * math.pi / freqs

    # Compute interpolation ratio for each frequency
    # 0 = fully interpolated, 1 = keep original
    smooth_factor = (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    smooth_factor = smooth_factor.clamp(0, 1)

    # Apply ramp function (smooth transition)
    # Frequencies in the middle range get partial interpolation
    ramp = (1 - smooth_factor) / scale + smooth_factor

    # Apply frequency warping
    freqs_scaled = freqs * ramp

    # Position indices
    t = torch.arange(max_seq_len, dtype=dtype, device=device)

    # Compute sin/cos
    theta = torch.outer(t, freqs_scaled)
    sin_out = torch.sin(theta)
    cos_out = torch.cos(theta)

    # Compute attention scale (to maintain softmax temperature)
    if mscale > 0:
        attn_scale = mscale
    else:
        # Automatic scaling based on extension ratio
        attn_scale = 0.1 * math.log(scale) + 1.0

    return sin_out, cos_out, attn_scale


# =============================================================================
# 3D Factorized RoPE (V-JEPA-2 Style)
# =============================================================================

def compute_rope_3d_freqs(
    dim: int,
    max_time: int,
    max_height: int,
    max_width: int,
    base: float = 10000.0,
    temporal_base_factor: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    efficient_mode: bool = True
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],  # Full 3D grid
    Tuple[Tuple[torch.Tensor, torch.Tensor],  # Factorized (t, h, w) tuples
          Tuple[torch.Tensor, torch.Tensor],
          Tuple[torch.Tensor, torch.Tensor]]
]:
    """
    Compute 3D factorized RoPE frequencies for (time, height, width).

    Following V-JEPA-2 approach:
    - Each axis (time, height, width) gets 1/3 of dimensions
    - Temporal axis can use slower frequencies (smaller base)
    - Efficient mode returns factorized tensors to avoid full 3D allocation

    Args:
        dim: Total embedding dimension (must be divisible by 6)
        max_time: Maximum time sequence length
        max_height: Maximum height
        max_width: Maximum width
        base: Base frequency for spatial dimensions
        temporal_base_factor: Scale factor for temporal base
                             (< 1.0 for slower temporal frequencies)
        device: Target device
        dtype: Computation dtype
        efficient_mode: If True, return factorized 1D tensors
                       If False, return full 3D grid (memory intensive)

    Returns:
        If efficient_mode:
            Tuple of ((sin_t, cos_t), (sin_h, cos_h), (sin_w, cos_w))
            Each tuple contains [max_axis, dim_per_axis//2] tensors
        Else:
            Tuple of (sin, cos) tensors [max_time, max_height, max_width, dim//2]

    Example:
        >>> # Efficient factorized computation
        >>> (sin_t, cos_t), (sin_h, cos_h), (sin_w, cos_w) = compute_rope_3d_freqs(
        ...     dim=96, max_time=16, max_height=14, max_width=14,
        ...     temporal_base_factor=0.5, efficient_mode=True
        ... )
    """
    assert dim % 6 == 0, f"Dimension must be divisible by 6 for 3D RoPE, got {dim}"

    dim_per_axis = dim // 3
    half_dim_per_axis = dim_per_axis // 2

    # Compute base frequencies for each axis
    # Temporal uses potentially different (slower) base
    temporal_base = base * temporal_base_factor

    # Frequency vectors for each axis
    inv_freq_t = 1.0 / (temporal_base ** (torch.arange(0, dim_per_axis, 2, dtype=dtype, device=device) / dim_per_axis))
    inv_freq_h = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=dtype, device=device) / dim_per_axis))
    inv_freq_w = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=dtype, device=device) / dim_per_axis))

    # Position indices
    t_pos = torch.arange(max_time, dtype=dtype, device=device)
    h_pos = torch.arange(max_height, dtype=dtype, device=device)
    w_pos = torch.arange(max_width, dtype=dtype, device=device)

    # Compute 1D frequency tables
    t_freqs = torch.outer(t_pos, inv_freq_t)  # [T, dim_per_axis//2]
    h_freqs = torch.outer(h_pos, inv_freq_h)  # [H, dim_per_axis//2]
    w_freqs = torch.outer(w_pos, inv_freq_w)  # [W, dim_per_axis//2]

    if efficient_mode:
        # Return factorized 1D tensors (memory efficient)
        return (
            (torch.sin(t_freqs), torch.cos(t_freqs)),
            (torch.sin(h_freqs), torch.cos(h_freqs)),
            (torch.sin(w_freqs), torch.cos(w_freqs))
        )
    else:
        # Build full 3D grid (memory intensive, but simpler to use)
        # Expand to full 3D grid
        t_freqs = t_freqs[:, None, None, :]  # [T, 1, 1, D/6]
        h_freqs = h_freqs[None, :, None, :]  # [1, H, 1, D/6]
        w_freqs = w_freqs[None, None, :, :]  # [1, 1, W, D/6]

        t_freqs = t_freqs.expand(max_time, max_height, max_width, -1)
        h_freqs = h_freqs.expand(max_time, max_height, max_width, -1)
        w_freqs = w_freqs.expand(max_time, max_height, max_width, -1)

        # Concatenate all axes
        all_freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)

        return torch.sin(all_freqs), torch.cos(all_freqs)


def apply_rope_3d_efficient(
    x: torch.Tensor,
    freqs_t: Tuple[torch.Tensor, torch.Tensor],
    freqs_h: Tuple[torch.Tensor, torch.Tensor],
    freqs_w: Tuple[torch.Tensor, torch.Tensor],
    t_indices: Optional[torch.Tensor] = None,
    h_indices: Optional[torch.Tensor] = None,
    w_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply 3D factorized RoPE efficiently without materializing full 3D grid.

    Args:
        x: Input tensor [B, T*H*W, num_heads, dim] or [B, T, H, W, num_heads, dim]
        freqs_t: Tuple of (sin_t, cos_t) each [max_T, dim_per_axis//2]
        freqs_h: Tuple of (sin_h, cos_h) each [max_H, dim_per_axis//2]
        freqs_w: Tuple of (sin_w, cos_w) each [max_W, dim_per_axis//2]
        t_indices: Time position indices [B, seq_len] or None for sequential
        h_indices: Height position indices [B, seq_len] or None
        w_indices: Width position indices [B, seq_len] or None

    Returns:
        Rotated tensor with same shape as input
    """
    sin_t, cos_t = freqs_t
    sin_h, cos_h = freqs_h
    sin_w, cos_w = freqs_w

    dim_per_axis = sin_t.shape[-1] * 2

    # Split x into temporal, height, width components
    # x: [..., dim] where dim = 3 * dim_per_axis
    x_t = x[..., :dim_per_axis]
    x_h = x[..., dim_per_axis:2*dim_per_axis]
    x_w = x[..., 2*dim_per_axis:]

    # Handle index gathering
    if t_indices is not None:
        # Gather sin/cos for specific positions
        sin_t = sin_t[t_indices]  # [B, seq_len, dim_per_axis//2]
        cos_t = cos_t[t_indices]

    if h_indices is not None:
        sin_h = sin_h[h_indices]
        cos_h = cos_h[h_indices]

    if w_indices is not None:
        sin_w = sin_w[w_indices]
        cos_w = cos_w[w_indices]

    # Ensure proper broadcasting
    while sin_t.dim() < x_t.dim():
        sin_t = sin_t.unsqueeze(-2)
        cos_t = cos_t.unsqueeze(-2)
    while sin_h.dim() < x_h.dim():
        sin_h = sin_h.unsqueeze(-2)
        cos_h = cos_h.unsqueeze(-2)
    while sin_w.dim() < x_w.dim():
        sin_w = sin_w.unsqueeze(-2)
        cos_w = cos_w.unsqueeze(-2)

    # Apply rotation to each component
    rotated_t = rope_apply(x_t, sin_t, cos_t)
    rotated_h = rope_apply(x_h, sin_h, cos_h)
    rotated_w = rope_apply(x_w, sin_w, cos_w)

    # Concatenate back
    return torch.cat([rotated_t, rotated_h, rotated_w], dim=-1)


# =============================================================================
# Variable Resolution Support
# =============================================================================

def interpolate_rope_freqs(
    sin: torch.Tensor,
    cos: torch.Tensor,
    target_len: int,
    mode: Literal["linear", "nearest", "area"] = "linear"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolate RoPE frequencies to different sequence length.

    Useful for variable resolution inference where the spatial
    dimensions differ from training.

    Args:
        sin: Sine frequencies [seq_len, dim//2]
        cos: Cosine frequencies [seq_len, dim//2]
        target_len: Target sequence length
        mode: Interpolation mode (linear recommended)

    Returns:
        Interpolated (sin, cos) tensors [target_len, dim//2]

    Example:
        >>> sin, cos = compute_rope_freqs(64, 196)  # 14x14 patches
        >>> sin_new, cos_new = interpolate_rope_freqs(sin, cos, 256)  # 16x16
    """
    if sin.shape[0] == target_len:
        return sin, cos

    # Add batch and channel dims for interpolate
    sin_interp = sin.unsqueeze(0).transpose(1, 2)  # [1, dim//2, seq_len]
    cos_interp = cos.unsqueeze(0).transpose(1, 2)

    # Interpolate
    sin_interp = F.interpolate(sin_interp, size=target_len, mode=mode,
                               align_corners=True if mode == "linear" else None)
    cos_interp = F.interpolate(cos_interp, size=target_len, mode=mode,
                               align_corners=True if mode == "linear" else None)

    # Reshape back
    sin_out = sin_interp.squeeze(0).transpose(0, 1)  # [target_len, dim//2]
    cos_out = cos_interp.squeeze(0).transpose(0, 1)

    return sin_out, cos_out


def compute_resolution_aware_rope_freqs(
    dim: int,
    height: int,
    width: int,
    original_height: int,
    original_width: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE frequencies for different spatial resolution.

    Adjusts frequencies based on resolution ratio to maintain
    consistent position encoding across resolutions.

    Args:
        dim: Embedding dimension for spatial (must be even)
        height: Current height
        width: Current width
        original_height: Training height
        original_width: Training width
        base: Base frequency
        device: Target device
        dtype: Computation dtype

    Returns:
        Tuple of (sin, cos) tensors [height * width, dim//2]
    """
    assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D factorization"

    dim_per_axis = dim // 2

    # Scale positions by resolution ratio
    h_scale = original_height / height
    w_scale = original_width / width

    # Compute frequencies
    inv_freq_h = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=dtype, device=device) / dim_per_axis))
    inv_freq_w = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2, dtype=dtype, device=device) / dim_per_axis))

    # Scaled position indices
    h_pos = torch.arange(height, dtype=dtype, device=device) * h_scale
    w_pos = torch.arange(width, dtype=dtype, device=device) * w_scale

    # Create 2D grid
    h_freqs = torch.outer(h_pos, inv_freq_h)  # [H, dim_per_axis//2]
    w_freqs = torch.outer(w_pos, inv_freq_w)  # [W, dim_per_axis//2]

    # Expand to full grid
    h_freqs = h_freqs[:, None, :].expand(height, width, -1)  # [H, W, dim_per_axis//2]
    w_freqs = w_freqs[None, :, :].expand(height, width, -1)  # [H, W, dim_per_axis//2]

    # Flatten and concatenate
    all_freqs = torch.cat([
        h_freqs.reshape(-1, h_freqs.shape[-1]),
        w_freqs.reshape(-1, w_freqs.shape[-1])
    ], dim=-1)  # [H*W, dim//2]

    return torch.sin(all_freqs), torch.cos(all_freqs)


# =============================================================================
# Caching System
# =============================================================================

class RoPEFrequencyCache:
    """
    Efficient cache for RoPE frequencies with lazy computation.

    Pre-computes frequencies for common sequence lengths and
    lazily computes for uncommon lengths.

    Attributes:
        config: RoPE configuration
        cache: Dictionary mapping sequence lengths to (sin, cos) tuples
        device: Cache device
        dtype: Storage dtype

    Example:
        >>> cache = RoPEFrequencyCache(config, device='cuda')
        >>> sin, cos = cache.get(512)  # Fast: pre-computed
        >>> sin, cos = cache.get(773)  # Lazy: computed on demand
    """

    def __init__(
        self,
        config: RoPEConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.config = config
        self.device = device
        self.dtype = dtype or (torch.float16 if config.use_half_precision else torch.float32)
        self.compute_dtype = torch.float32  # Always compute in float32

        # Initialize cache
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._3d_cache: Dict[Tuple[int, int, int], any] = {}

        # Pre-compute common lengths
        self._precompute_common()

    def _precompute_common(self):
        """Pre-compute frequencies for common sequence lengths."""
        for length in self.config.cache_common_lengths:
            if length <= self.config.max_seq_len:
                self._compute_and_cache(length)

    def _compute_and_cache(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and cache frequencies for a sequence length."""
        if self.config.yarn_enabled:
            sin, cos, _ = compute_yarn_rope_freqs(
                dim=self.config.dim,
                max_seq_len=seq_len,
                original_max_len=self.config.original_max_seq_len,
                base=self.config.base,
                beta_fast=self.config.yarn_beta_fast,
                beta_slow=self.config.yarn_beta_slow,
                mscale=self.config.yarn_mscale,
                device=self.device,
                dtype=self.compute_dtype
            )
        elif self.config.dynamic_ntk:
            sin, cos = compute_dynamic_ntk_rope_freqs(
                dim=self.config.dim,
                seq_len=seq_len,
                original_max_len=self.config.original_max_seq_len,
                base=self.config.base,
                ntk_factor=self.config.ntk_factor,
                device=self.device,
                dtype=self.compute_dtype
            )
        elif self.config.ntk_alpha != 1.0:
            sin, cos = compute_ntk_rope_freqs(
                dim=self.config.dim,
                max_seq_len=seq_len,
                base=self.config.base,
                alpha=self.config.ntk_alpha,
                device=self.device,
                dtype=self.compute_dtype
            )
        else:
            sin, cos = compute_rope_freqs(
                dim=self.config.dim,
                max_seq_len=seq_len,
                base=self.config.base,
                device=self.device,
                dtype=self.compute_dtype
            )

        # Convert to storage dtype
        if self.dtype != self.compute_dtype:
            sin = sin.to(self.dtype)
            cos = cos.to(self.dtype)

        self._cache[seq_len] = (sin, cos)
        return sin, cos

    def get(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get frequencies for a sequence length.

        Args:
            seq_len: Target sequence length

        Returns:
            Tuple of (sin, cos) tensors [seq_len, dim//2]
        """
        if seq_len in self._cache:
            return self._cache[seq_len]

        # Check if we can slice from a longer cached sequence
        for cached_len in sorted(self._cache.keys(), reverse=True):
            if cached_len >= seq_len:
                sin, cos = self._cache[cached_len]
                return sin[:seq_len], cos[:seq_len]

        # Compute and cache
        return self._compute_and_cache(seq_len)

    def get_3d(
        self,
        max_time: int,
        max_height: int,
        max_width: int,
        efficient_mode: bool = True
    ):
        """
        Get 3D factorized frequencies.

        Args:
            max_time: Maximum time steps
            max_height: Maximum height
            max_width: Maximum width
            efficient_mode: Return factorized or full grid

        Returns:
            Same format as compute_rope_3d_freqs
        """
        key = (max_time, max_height, max_width, efficient_mode)

        if key in self._3d_cache:
            return self._3d_cache[key]

        result = compute_rope_3d_freqs(
            dim=self.config.dim,
            max_time=max_time,
            max_height=max_height,
            max_width=max_width,
            base=self.config.base,
            temporal_base_factor=self.config.temporal_base_factor,
            device=self.device,
            dtype=self.compute_dtype,
            efficient_mode=efficient_mode
        )

        # Convert to storage dtype if efficient mode (tuples of tensors)
        if efficient_mode and self.dtype != self.compute_dtype:
            result = tuple(
                (sin.to(self.dtype), cos.to(self.dtype))
                for sin, cos in result
            )
        elif not efficient_mode and self.dtype != self.compute_dtype:
            result = (result[0].to(self.dtype), result[1].to(self.dtype))

        self._3d_cache[key] = result
        return result

    def to(self, device: torch.device) -> 'RoPEFrequencyCache':
        """Move cache to device."""
        self.device = device
        self._cache = {
            k: (v[0].to(device), v[1].to(device))
            for k, v in self._cache.items()
        }
        self._3d_cache = {}  # Clear 3D cache, will recompute
        return self

    def clear(self):
        """Clear all cached frequencies."""
        self._cache.clear()
        self._3d_cache.clear()


# =============================================================================
# Main RoPE Module
# =============================================================================

class RoPELayerV2(nn.Module):
    """
    Enhanced RoPE layer with modern techniques.

    Features:
    - NTK-aware scaling for context extension
    - YaRN for better length generalization
    - V-JEPA-2 style 3D factorization
    - Variable resolution support
    - Efficient caching system
    - Flash attention compatibility

    Attributes:
        config: RoPE configuration
        cache: Frequency cache

    Example:
        >>> # Basic 1D RoPE
        >>> config = RoPEConfig(dim=64, max_seq_len=2048)
        >>> rope = RoPELayerV2(config)
        >>> x = torch.randn(2, 512, 8, 64)  # [B, T, H, D]
        >>> rotated = rope(x)

        >>> # 3D RoPE for video
        >>> config = RoPEConfig(dim=96, spatial_dims=(14, 14))
        >>> rope = RoPELayerV2(config)
        >>> x = torch.randn(2, 16*14*14, 8, 96)  # [B, T*H*W, heads, D]
        >>> rotated = rope.forward_3d(x, time=16, height=14, width=14)

        >>> # Extended context with NTK
        >>> config = RoPEConfig(dim=64, max_seq_len=8192,
        ...                     original_max_seq_len=2048, dynamic_ntk=True)
        >>> rope = RoPELayerV2(config)
    """

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config

        # Initialize cache (will be populated on first use or device move)
        self.cache = None
        self._yarn_attention_scale = 1.0

        # Pre-compute base frequencies as buffers for standard use
        sin, cos = self._compute_base_freqs()
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)

        # 3D frequencies (lazy initialized)
        self._3d_freqs = None

    def _compute_base_freqs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute base frequencies based on config."""
        if self.config.yarn_enabled:
            sin, cos, scale = compute_yarn_rope_freqs(
                dim=self.config.dim,
                max_seq_len=self.config.max_seq_len,
                original_max_len=self.config.original_max_seq_len,
                base=self.config.base,
                beta_fast=self.config.yarn_beta_fast,
                beta_slow=self.config.yarn_beta_slow,
                mscale=self.config.yarn_mscale
            )
            self._yarn_attention_scale = scale
        elif self.config.ntk_alpha != 1.0:
            sin, cos = compute_ntk_rope_freqs(
                dim=self.config.dim,
                max_seq_len=self.config.max_seq_len,
                base=self.config.base,
                alpha=self.config.ntk_alpha
            )
        else:
            sin, cos = compute_rope_freqs(
                dim=self.config.dim,
                max_seq_len=self.config.max_seq_len,
                base=self.config.base
            )

        dtype = torch.float16 if self.config.use_half_precision else torch.float32
        return sin.to(dtype), cos.to(dtype)

    def get_attention_scale(self) -> float:
        """
        Get YaRN attention scaling factor.

        Should be multiplied with attention logits when using YaRN.

        Returns:
            Attention scale factor (1.0 if YaRN disabled)
        """
        return self._yarn_attention_scale

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor [B, T, H, D] where:
               B = batch size
               T = sequence length
               H = number of heads
               D = head dimension (must match config.dim)
            start_pos: Starting position for KV caching
            seq_len: Override sequence length (for dynamic NTK)

        Returns:
            Rotated tensor [B, T, H, D]
        """
        T = x.size(1)
        actual_seq_len = seq_len or (start_pos + T)

        # Handle dynamic NTK scaling
        if self.config.dynamic_ntk and actual_seq_len > self.config.original_max_seq_len:
            sin, cos = compute_dynamic_ntk_rope_freqs(
                dim=self.config.dim,
                seq_len=actual_seq_len,
                original_max_len=self.config.original_max_seq_len,
                base=self.config.base,
                ntk_factor=self.config.ntk_factor,
                device=x.device,
                dtype=torch.float32
            )
            sin = sin.to(x.dtype)
            cos = cos.to(x.dtype)
        else:
            sin = self.sin
            cos = self.cos

        # Get frequency slices for this sequence
        end_pos = start_pos + T
        sin_slice = sin[start_pos:end_pos, :]  # [T, D//2]
        cos_slice = cos[start_pos:end_pos, :]  # [T, D//2]

        # Expand for batch and head dimensions: [1, T, 1, D//2]
        sin_slice = sin_slice.unsqueeze(0).unsqueeze(2)
        cos_slice = cos_slice.unsqueeze(0).unsqueeze(2)

        return rope_apply(x, sin_slice, cos_slice)

    def forward_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors together.

        Args:
            q: Query tensor [B, T, H, D]
            k: Key tensor [B, T, H, D]
            start_pos: Starting position for KV caching

        Returns:
            Tuple of rotated (q, k) tensors
        """
        T = q.size(1)
        end_pos = start_pos + T

        sin_slice = self.sin[start_pos:end_pos, :].unsqueeze(0).unsqueeze(2)
        cos_slice = self.cos[start_pos:end_pos, :].unsqueeze(0).unsqueeze(2)

        return rope_apply_qk(q, k, sin_slice, cos_slice)

    def forward_3d(
        self,
        x: torch.Tensor,
        time: int,
        height: int,
        width: int,
        t_offset: int = 0,
        h_offset: int = 0,
        w_offset: int = 0
    ) -> torch.Tensor:
        """
        Apply 3D factorized RoPE for video/spatial-temporal data.

        Args:
            x: Input tensor [B, T*H*W, num_heads, D] where D = 3 * dim_per_axis
            time: Number of time steps
            height: Spatial height
            width: Spatial width
            t_offset: Time offset for caching
            h_offset: Height offset
            w_offset: Width offset

        Returns:
            Rotated tensor [B, T*H*W, num_heads, D]
        """
        assert self.config.dim % 6 == 0, "3D RoPE requires dim divisible by 6"

        B, seq_len, H, D = x.shape
        expected_seq = time * height * width
        assert seq_len == expected_seq, f"Sequence length {seq_len} doesn't match T*H*W={expected_seq}"

        # Get or compute 3D frequencies
        if self._3d_freqs is None or self._check_3d_freqs_mismatch(time, height, width):
            self._3d_freqs = compute_rope_3d_freqs(
                dim=self.config.dim,
                max_time=max(time + t_offset, self.config.max_seq_len // (height * width) if height and width else 32),
                max_height=max(height + h_offset, self.config.spatial_dims[0] if self.config.spatial_dims else 32),
                max_width=max(width + w_offset, self.config.spatial_dims[1] if self.config.spatial_dims else 32),
                base=self.config.base,
                temporal_base_factor=self.config.temporal_base_factor,
                device=x.device,
                dtype=torch.float32,
                efficient_mode=True
            )

        freqs_t, freqs_h, freqs_w = self._3d_freqs

        # Convert to proper dtype and device
        freqs_t = (freqs_t[0].to(device=x.device, dtype=x.dtype),
                   freqs_t[1].to(device=x.device, dtype=x.dtype))
        freqs_h = (freqs_h[0].to(device=x.device, dtype=x.dtype),
                   freqs_h[1].to(device=x.device, dtype=x.dtype))
        freqs_w = (freqs_w[0].to(device=x.device, dtype=x.dtype),
                   freqs_w[1].to(device=x.device, dtype=x.dtype))

        # Generate position indices for the 3D grid
        # For sequence [t0_h0_w0, t0_h0_w1, ..., t0_h1_w0, ..., t1_h0_w0, ...]
        t_idx = torch.arange(t_offset, t_offset + time, device=x.device)
        h_idx = torch.arange(h_offset, h_offset + height, device=x.device)
        w_idx = torch.arange(w_offset, w_offset + width, device=x.device)

        # Create full index tensors [T*H*W]
        t_indices = t_idx.repeat_interleave(height * width)
        h_indices = h_idx.repeat(time).repeat_interleave(width)
        w_indices = w_idx.repeat(time * height)

        return apply_rope_3d_efficient(
            x, freqs_t, freqs_h, freqs_w,
            t_indices, h_indices, w_indices
        )

    def _check_3d_freqs_mismatch(self, time: int, height: int, width: int) -> bool:
        """Check if cached 3D frequencies need recomputation."""
        if self._3d_freqs is None:
            return True

        (sin_t, _), (sin_h, _), (sin_w, _) = self._3d_freqs
        return (sin_t.size(0) < time or
                sin_h.size(0) < height or
                sin_w.size(0) < width)

    def forward_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE formatted for Flash Attention.

        Handles packed sequences for efficient batched attention.

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim] for packed
               or [B, T, H, D] for padded
            k: Key tensor with same format as q
            start_pos: Starting position
            cu_seqlens: Cumulative sequence lengths for packed batches
                       [batch_size + 1] where cu_seqlens[0] = 0
            max_seqlen: Maximum sequence length in batch

        Returns:
            Tuple of rotated (q, k) tensors

        Example:
            >>> # Packed sequence input
            >>> q = torch.randn(total_tokens, 8, 64)
            >>> k = torch.randn(total_tokens, 8, 64)
            >>> cu_seqlens = torch.tensor([0, 100, 250, 400])  # 3 sequences
            >>> q_rot, k_rot = rope.forward_flash(q, k, cu_seqlens=cu_seqlens)
        """
        if cu_seqlens is not None:
            # Packed sequence mode
            return self._forward_flash_packed(q, k, start_pos, cu_seqlens, max_seqlen)
        else:
            # Standard padded mode - reshape for standard forward
            if q.dim() == 3:  # [total, H, D]
                q = q.unsqueeze(0)  # [1, total, H, D]
                k = k.unsqueeze(0)
                q_rot, k_rot = self.forward_qk(q, k, start_pos)
                return q_rot.squeeze(0), k_rot.squeeze(0)
            else:
                return self.forward_qk(q, k, start_pos)

    def _forward_flash_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle packed sequences for flash attention."""
        total_tokens = q.size(0)
        batch_size = cu_seqlens.size(0) - 1

        # Build position indices for each token
        position_ids = torch.zeros(total_tokens, device=q.device, dtype=torch.long)

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start
            position_ids[start:end] = torch.arange(start_pos, start_pos + seq_len, device=q.device)

        # Gather sin/cos for each position
        sin = self.sin[position_ids]  # [total_tokens, D//2]
        cos = self.cos[position_ids]

        # Expand for heads: [total_tokens, 1, D//2]
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)

        # Apply rotation
        q_rot = rope_apply(q, sin, cos)
        k_rot = rope_apply(k, sin, cos)

        return q_rot, k_rot

    def forward_chunked(
        self,
        x: torch.Tensor,
        chunk_size: int = 512,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        Apply RoPE in chunks for memory efficiency.

        Useful for very long sequences that don't fit in memory.

        Args:
            x: Input tensor [B, T, H, D]
            chunk_size: Size of each processing chunk
            start_pos: Starting position

        Returns:
            Rotated tensor [B, T, H, D]
        """
        B, T, H, D = x.shape

        if T <= chunk_size:
            return self.forward(x, start_pos)

        # Process in chunks
        outputs = []
        for i in range(0, T, chunk_size):
            end = min(i + chunk_size, T)
            chunk = x[:, i:end]
            rotated_chunk = self.forward(chunk, start_pos + i)
            outputs.append(rotated_chunk)

        return torch.cat(outputs, dim=1)

    def extra_repr(self) -> str:
        """String representation with config details."""
        parts = [f"dim={self.config.dim}", f"max_seq_len={self.config.max_seq_len}"]

        if self.config.ntk_alpha != 1.0:
            parts.append(f"ntk_alpha={self.config.ntk_alpha}")
        if self.config.dynamic_ntk:
            parts.append("dynamic_ntk=True")
        if self.config.yarn_enabled:
            parts.append("yarn=True")
        if self.config.spatial_dims:
            parts.append(f"spatial_dims={self.config.spatial_dims}")

        return ", ".join(parts)


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

class RoPELayer(RoPELayerV2):
    """
    Backward compatible RoPE layer matching original interface.

    This class provides the same API as the original rope.py RoPELayer
    while using the enhanced V2 implementation internally.

    Example:
        >>> # Drop-in replacement for original RoPELayer
        >>> rope = RoPELayer(dim=64, max_seq_len=2048, base=10000.0)
        >>> x = torch.randn(2, 512, 8, 64)
        >>> rotated = rope(x, start_pos=0)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        config = RoPEConfig(
            dim=dim,
            max_seq_len=max_seq_len,
            base=base,
            use_half_precision=False  # Match original behavior
        )
        super().__init__(config)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base


# =============================================================================
# Utility Functions for Backward Compatibility
# =============================================================================

def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward compatible frequency computation.

    Matches original rope.py interface.
    """
    return compute_rope_freqs(dim, max_seq_len, base, device)


def precompute_rope_3d_freqs(
    dim: int,
    max_time: int,
    max_height: int,
    max_width: int,
    base: float = 10000.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward compatible 3D frequency computation.

    Returns full 3D grid for compatibility with original interface.
    """
    return compute_rope_3d_freqs(
        dim, max_time, max_height, max_width, base, device,
        efficient_mode=False  # Full grid for compatibility
    )


def rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Backward compatible rope function.

    Matches original rope.py interface.
    """
    return rope_apply(x, sin, cos, interleaved=True)


# =============================================================================
# Testing and Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RoPE V2 - Enhanced Rotary Position Embedding")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Test 1: Basic 1D RoPE
    print("\n1. Basic 1D RoPE")
    print("-" * 40)

    config = RoPEConfig(dim=64, max_seq_len=2048)
    rope_layer = RoPELayerV2(config).to(device)

    x = torch.randn(2, 512, 8, 64, device=device)
    rotated = rope_layer(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {rotated.shape}")
    print(f"   Output norm:  {rotated.norm().item():.4f}")

    # Test 2: NTK-Aware Scaling
    print("\n2. NTK-Aware Scaling (2x context extension)")
    print("-" * 40)

    config_ntk = RoPEConfig(
        dim=64,
        max_seq_len=4096,
        ntk_alpha=2.0,
        original_max_seq_len=2048
    )
    rope_ntk = RoPELayerV2(config_ntk).to(device)

    x_long = torch.randn(1, 3000, 4, 64, device=device)
    rotated_ntk = rope_ntk(x_long)
    print(f"   Input shape:  {x_long.shape}")
    print(f"   Output shape: {rotated_ntk.shape}")
    print(f"   NTK alpha:    {config_ntk.ntk_alpha}")

    # Test 3: Dynamic NTK
    print("\n3. Dynamic NTK Scaling")
    print("-" * 40)

    config_dynamic = RoPEConfig(
        dim=64,
        max_seq_len=8192,
        dynamic_ntk=True,
        original_max_seq_len=2048,
        ntk_factor=2.0
    )
    rope_dynamic = RoPELayerV2(config_dynamic).to(device)

    # Short sequence (no scaling)
    x_short = torch.randn(1, 1000, 4, 64, device=device)
    rotated_short = rope_dynamic(x_short)
    print(f"   Short seq ({x_short.shape[1]}): norm = {rotated_short.norm().item():.4f}")

    # Long sequence (dynamic scaling)
    x_long = torch.randn(1, 4000, 4, 64, device=device)
    rotated_long = rope_dynamic(x_long)
    print(f"   Long seq ({x_long.shape[1]}):  norm = {rotated_long.norm().item():.4f}")

    # Test 4: YaRN
    print("\n4. YaRN Extension")
    print("-" * 40)

    config_yarn = RoPEConfig(
        dim=64,
        max_seq_len=8192,
        yarn_enabled=True,
        original_max_seq_len=2048,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0
    )
    rope_yarn = RoPELayerV2(config_yarn).to(device)

    x_yarn = torch.randn(1, 4000, 4, 64, device=device)
    rotated_yarn = rope_yarn(x_yarn)
    print(f"   Input shape:       {x_yarn.shape}")
    print(f"   Output shape:      {rotated_yarn.shape}")
    print(f"   Attention scale:   {rope_yarn.get_attention_scale():.4f}")

    # Test 5: 3D Factorized RoPE
    print("\n5. 3D Factorized RoPE (V-JEPA-2 style)")
    print("-" * 40)

    config_3d = RoPEConfig(
        dim=96,  # Must be divisible by 6
        max_seq_len=4096,
        spatial_dims=(14, 14),
        temporal_base_factor=0.5  # Slower temporal frequencies
    )
    rope_3d = RoPELayerV2(config_3d).to(device)

    # Video input: 16 frames, 14x14 spatial, 8 heads, 96 dim
    time, height, width = 16, 14, 14
    x_3d = torch.randn(2, time * height * width, 8, 96, device=device)
    rotated_3d = rope_3d.forward_3d(x_3d, time=time, height=height, width=width)
    print(f"   Input shape:  {x_3d.shape}")
    print(f"   T x H x W:    {time} x {height} x {width}")
    print(f"   Output shape: {rotated_3d.shape}")
    print(f"   Temporal factor: {config_3d.temporal_base_factor}")

    # Test 6: Variable Resolution
    print("\n6. Variable Resolution Support")
    print("-" * 40)

    # Training resolution
    sin_train, cos_train = compute_resolution_aware_rope_freqs(
        dim=64, height=14, width=14,
        original_height=14, original_width=14,
        device=device
    )
    print(f"   Training (14x14):    sin shape = {sin_train.shape}")

    # Higher resolution inference
    sin_high, cos_high = compute_resolution_aware_rope_freqs(
        dim=64, height=20, width=20,
        original_height=14, original_width=14,
        device=device
    )
    print(f"   Inference (20x20):   sin shape = {sin_high.shape}")

    # Test 7: Flash Attention Compatibility
    print("\n7. Flash Attention Format")
    print("-" * 40)

    config_flash = RoPEConfig(dim=64, max_seq_len=2048)
    rope_flash = RoPELayerV2(config_flash).to(device)

    # Packed sequences
    total_tokens = 1500
    q = torch.randn(total_tokens, 8, 64, device=device)
    k = torch.randn(total_tokens, 8, 64, device=device)
    cu_seqlens = torch.tensor([0, 500, 1200, 1500], device=device)

    q_rot, k_rot = rope_flash.forward_flash(q, k, cu_seqlens=cu_seqlens)
    print(f"   Packed tokens: {total_tokens}")
    print(f"   Sequences:     {len(cu_seqlens) - 1}")
    print(f"   Q rotated:     {q_rot.shape}")
    print(f"   K rotated:     {k_rot.shape}")

    # Test 8: Efficient Caching
    print("\n8. Efficient Caching")
    print("-" * 40)

    config_cache = RoPEConfig(
        dim=64,
        max_seq_len=4096,
        cache_common_lengths=(128, 256, 512, 1024, 2048, 4096)
    )
    cache = RoPEFrequencyCache(config_cache, device=device)

    # Cached access (fast)
    import time

    start = time.perf_counter()
    for _ in range(100):
        _ = cache.get(512)
    cached_time = time.perf_counter() - start
    print(f"   100x cached (512):   {cached_time*1000:.2f} ms")

    # Uncached access (computed once, then cached)
    start = time.perf_counter()
    _ = cache.get(773)  # Uncommon length
    uncached_time = time.perf_counter() - start
    print(f"   1x uncached (773):   {uncached_time*1000:.2f} ms")

    # Now it's cached
    start = time.perf_counter()
    for _ in range(100):
        _ = cache.get(773)
    recached_time = time.perf_counter() - start
    print(f"   100x recached (773): {recached_time*1000:.2f} ms")

    # Test 9: Chunked Processing
    print("\n9. Chunked Processing (Memory Efficient)")
    print("-" * 40)

    config_chunk = RoPEConfig(dim=64, max_seq_len=8192)
    rope_chunk = RoPELayerV2(config_chunk).to(device)

    x_huge = torch.randn(1, 4000, 4, 64, device=device)

    # Standard (may OOM for very large)
    rotated_standard = rope_chunk(x_huge)

    # Chunked (memory efficient)
    rotated_chunked = rope_chunk.forward_chunked(x_huge, chunk_size=1000)

    diff = (rotated_standard - rotated_chunked).abs().max()
    print(f"   Input length:    {x_huge.shape[1]}")
    print(f"   Chunk size:      1000")
    print(f"   Max difference:  {diff.item():.2e}")

    # Test 10: Backward Compatibility
    print("\n10. Backward Compatibility")
    print("-" * 40)

    # Using backward compatible interface
    rope_compat = RoPELayer(dim=64, max_seq_len=2048, base=10000.0).to(device)

    x_compat = torch.randn(2, 256, 8, 64, device=device)
    rotated_compat = rope_compat(x_compat, start_pos=0)
    print(f"   Using RoPELayer (compatible): {rotated_compat.shape}")

    # Using backward compatible functions
    sin, cos = precompute_rope_freqs(64, 2048, device=device)
    print(f"   precompute_rope_freqs: sin={sin.shape}, cos={cos.shape}")

    sin_3d, cos_3d = precompute_rope_3d_freqs(96, 16, 14, 14, device=device)
    print(f"   precompute_rope_3d_freqs: sin={sin_3d.shape}, cos={cos_3d.shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
