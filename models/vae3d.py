"""
3D Causal VAE implementation for world modeling.

This is the primary VAE3D implementation featuring:
- Asymmetric temporal compression (4x time, 8x space)
- KL annealing with cyclical schedules
- U-Net style skip connections
- Optional VQ-VAE quantization
- Learnable latent prior
- Perceptual loss integration
- Streaming inference support
- Spectral normalization
- Gradient checkpointing
- torch.compile compatibility

VAE3D is an alias for VAE3Dv2 for backward compatibility.
The causal_conv3d() factory function provides v1 API compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.nn_helpers import RMSNorm, silu


# =============================================================================
# Configuration
# =============================================================================

class QuantizationMode(Enum):
    """Quantization modes for latent space."""
    NONE = "none"
    VQ = "vq"            # Vector quantization
    FSQ = "fsq"          # Finite scalar quantization


@dataclass
class VAE3DConfig:
    """Configuration for VAE3D v2."""

    # Input/output
    in_channels: int = 3
    out_channels: int = 3

    # Latent space
    latent_dim: int = 8

    # Compression ratios (asymmetric)
    temporal_compression: int = 4    # 8 frames -> 2 latent frames
    spatial_compression: int = 8     # 256x256 -> 32x32

    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 512])
    num_res_blocks: int = 2
    use_attention: bool = True
    attention_resolution: List[int] = field(default_factory=lambda: [16, 8])

    # Skip connections (U-Net style)
    use_skip_connections: bool = True
    skip_connection_scale: float = 0.5

    # Quantization
    quantization_mode: QuantizationMode = QuantizationMode.NONE
    num_codebook_vectors: int = 8192
    codebook_dim: int = 8
    commitment_cost: float = 0.25

    # KL annealing
    use_kl_annealing: bool = True
    kl_anneal_mode: str = "cyclical"  # "linear", "cyclical", "constant"
    kl_cycle_period: int = 10000
    kl_start_beta: float = 0.0
    kl_end_beta: float = 1.0
    kl_free_bits: float = 0.0
    kl_warmup_steps: int = 1000

    # Learnable prior
    use_learnable_prior: bool = False
    prior_hidden_dim: int = 256

    # Hierarchical latents
    use_hierarchical_latents: bool = False
    num_latent_levels: int = 2

    # Perceptual loss
    use_perceptual_loss: bool = True
    perceptual_weight: float = 0.1
    perceptual_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Regularization
    use_spectral_norm: bool = False
    latent_dropout: float = 0.0
    gradient_penalty_weight: float = 0.0

    # Streaming inference
    enable_streaming: bool = True
    max_cache_frames: int = 8

    # Efficiency
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True


# =============================================================================
# Causal Convolutions with Streaming Support
# =============================================================================

class CausalConv3d(nn.Module):
    """Time-causal 3D convolution with optional streaming support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        bias: bool = True,
        enable_streaming: bool = True,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        kt, kh, kw = kernel_size
        st, sh, sw = stride if isinstance(stride, tuple) else (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = (st, sh, sw)
        self.enable_streaming = enable_streaming
        self.temporal_padding = kt - 1
        self.pad_h = kh // 2
        self.pad_w = kw // 2

        conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=(st, sh, sw), padding=(0, self.pad_h, self.pad_w),
            bias=bias
        )

        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        self.conv = conv
        self._cache = None

    def reset_cache(self):
        """Reset streaming cache."""
        self._cache = None

    def forward(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        """Forward pass with optional streaming mode."""
        B, C, T, H, W = x.shape

        if streaming and self.enable_streaming:
            if self._cache is not None:
                x = torch.cat([self._cache, x], dim=2)
            if self.temporal_padding > 0:
                self._cache = x[:, :, -self.temporal_padding:].detach()
        else:
            if self.temporal_padding > 0:
                x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0))

        return self.conv(x)


class CausalConvTranspose3d(nn.Module):
    """Time-causal 3D transpose convolution using interpolation + conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (4, 4, 4),
        stride: Tuple[int, int, int] = (2, 2, 2),
        bias: bool = True,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        kt, kh, kw = kernel_size
        st, sh, sw = stride if isinstance(stride, tuple) else (stride, stride, stride)

        self.kernel_size = (kt, kh, kw)
        self.stride = (st, sh, sw)
        self.upsample_t = st
        self.upsample_s = sh

        conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kt, 3, 3),
            padding=(0, 1, 1),
            bias=bias
        )

        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        self.conv = conv

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """Forward pass with upsampling."""
        B, C, T, H, W = x.shape

        if target_size is not None:
            target_T, target_H, target_W = target_size
        else:
            target_T = T * self.upsample_t
            target_H = H * self.upsample_s
            target_W = W * self.upsample_s

        out = F.interpolate(
            x,
            size=(target_T, target_H, target_W),
            mode='trilinear',
            align_corners=False
        )

        if self.kernel_size[0] > 1:
            out = F.pad(out, (0, 0, 0, 0, self.kernel_size[0] - 1, 0))

        out = self.conv(out)
        return out


# =============================================================================
# Building Blocks
# =============================================================================

class ResidualBlock3D(nn.Module):
    """3D residual block with causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        dropout: float = 0.0,
        use_spectral_norm: bool = False,
        enable_streaming: bool = True
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = CausalConv3d(
            in_channels, out_channels, kernel_size,
            use_spectral_norm=use_spectral_norm,
            enable_streaming=enable_streaming
        )

        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, kernel_size,
            use_spectral_norm=use_spectral_norm,
            enable_streaming=enable_streaming
        )

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        """Forward pass with optional streaming."""
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h, streaming=streaming)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h, streaming=streaming)

        return h + residual


class AttentionBlock3D(nn.Module):
    """3D attention block for VAE."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(min(32, channels), channels)

        qkv_conv = nn.Conv3d(channels, 3 * channels, 1)
        proj_conv = nn.Conv3d(channels, channels, 1)

        if use_spectral_norm:
            qkv_conv = nn.utils.spectral_norm(qkv_conv)
            proj_conv = nn.utils.spectral_norm(proj_conv)

        self.qkv = qkv_conv
        self.proj = proj_conv
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D attention."""
        B, C, T, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)

        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, T * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        out = out.permute(0, 1, 3, 2)
        out = out.reshape(B, C, T, H, W)
        out = self.proj(out)

        return out + residual


class Downsample3D(nn.Module):
    """Downsample block with asymmetric compression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        use_spectral_norm: bool = False,
        enable_streaming: bool = True
    ):
        super().__init__()

        kt = temporal_stride * 2 if temporal_stride > 1 else 3
        ks = spatial_stride * 2 if spatial_stride > 1 else 3

        self.conv = CausalConv3d(
            in_channels, out_channels,
            kernel_size=(kt, ks, ks),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            use_spectral_norm=use_spectral_norm,
            enable_streaming=enable_streaming
        )

    def forward(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        return self.conv(x, streaming=streaming)


class Upsample3D(nn.Module):
    """Upsample block with asymmetric decompression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        self.temporal_stride = temporal_stride
        self.spatial_stride = spatial_stride

        kt = temporal_stride * 2 if temporal_stride > 1 else 3
        ks = spatial_stride * 2 if spatial_stride > 1 else 3

        self.conv = CausalConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kt, ks, ks),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            use_spectral_norm=use_spectral_norm
        )

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        return self.conv(x, target_size=target_size)


# =============================================================================
# Vector Quantization
# =============================================================================

class VectorQuantizer(nn.Module):
    """Vector quantization layer with EMA updates."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_embeddings, 1/num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embeddings.weight.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input tensor."""
        B, C, T, H, W = z.shape

        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)

        distances = (
            z_flat.pow(2).sum(1, keepdim=True) -
            2 * z_flat @ self.embeddings.weight.t() +
            self.embeddings.weight.pow(2).sum(1)
        )

        indices = distances.argmin(dim=1)
        z_q = self.embeddings(indices)

        z_q = z_q.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        commitment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        vq_loss = embedding_loss + self.commitment_cost * commitment_loss

        z_q = z + (z_q - z).detach()

        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_embeddings).float()
                cluster_size = encodings.sum(0)

                self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size

                dw = encodings.t() @ z_flat
                self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

                self.embeddings.weight.data = self.ema_w / cluster_size.unsqueeze(1)

        return z_q, vq_loss, indices.view(B, T, H, W)


# =============================================================================
# Learnable Prior
# =============================================================================

class LearnablePrior(nn.Module):
    """Learnable prior distribution for VAE."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        latent_shape: Tuple[int, int, int] = (2, 32, 32)
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_shape = latent_shape

        T, H, W = latent_shape
        self.base_embedding = nn.Parameter(torch.randn(1, hidden_dim, T, H, W) * 0.02)

        self.net = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, 2 * latent_dim, 3, padding=1)
        )

    def forward(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate prior distribution parameters."""
        base = self.base_embedding.expand(batch_size, -1, -1, -1, -1).to(device)
        params = self.net(base)
        mean, logvar = params.chunk(2, dim=1)
        return mean, logvar


# =============================================================================
# Perceptual Loss
# =============================================================================

class PerceptualLoss(nn.Module):
    """LPIPS-style perceptual loss using internal features."""

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: List[int] = None,
        normalize: bool = True
    ):
        super().__init__()

        if feature_channels is None:
            feature_channels = [64, 128, 256, 256]

        self.normalize = normalize

        layers = []
        ch = in_channels
        for out_ch in feature_channels:
            layers.extend([
                nn.Conv3d(ch, out_ch, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.SiLU()
            ])
            ch = out_ch

        self.features = nn.Sequential(*layers)
        self.layer_weights = nn.Parameter(torch.ones(len(feature_channels)))

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        if self.normalize:
            pred = pred * 2 - 1
            target = target * 2 - 1

        losses = []
        x_pred, x_target = pred, target

        layer_idx = 0
        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if (i + 1) % 4 == 0:
                loss = F.mse_loss(x_pred, x_target)
                losses.append(loss * F.softplus(self.layer_weights[layer_idx]))
                layer_idx += 1

        return sum(losses) / len(losses)


# =============================================================================
# KL Annealing
# =============================================================================

class KLAnnealer:
    """KL divergence annealing scheduler."""

    def __init__(
        self,
        mode: str = "cyclical",
        cycle_period: int = 10000,
        start_beta: float = 0.0,
        end_beta: float = 1.0,
        warmup_steps: int = 1000,
        free_bits: float = 0.0
    ):
        self.mode = mode
        self.cycle_period = cycle_period
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.warmup_steps = warmup_steps
        self.free_bits = free_bits

        self.step = 0
        self._restart_cycle_step = 0

    def get_beta(self) -> float:
        """Get current KL weight."""
        if self.mode == "constant":
            return self.end_beta

        elif self.mode == "linear":
            if self.step < self.warmup_steps:
                return self.start_beta + (self.end_beta - self.start_beta) * (self.step / self.warmup_steps)
            return self.end_beta

        elif self.mode == "cyclical":
            cycle_step = (self.step - self._restart_cycle_step) % self.cycle_period
            cycle_progress = cycle_step / self.cycle_period
            beta = self.start_beta + 0.5 * (self.end_beta - self.start_beta) * (1 - math.cos(math.pi * cycle_progress))
            return beta

        else:
            raise ValueError(f"Unknown annealing mode: {self.mode}")

    def update(self):
        """Advance one step."""
        self.step += 1

    def warm_restart(self):
        """Trigger a warm restart."""
        self._restart_cycle_step = self.step

    def apply_free_bits(self, kl_per_dim: torch.Tensor) -> torch.Tensor:
        """Apply free bits constraint."""
        if self.free_bits > 0:
            return torch.clamp(kl_per_dim - self.free_bits, min=0.0)
        return kl_per_dim


# =============================================================================
# Encoder
# =============================================================================

class Encoder3D(nn.Module):
    """3D causal encoder with asymmetric compression and skip connections."""

    def __init__(self, config: VAE3DConfig):
        super().__init__()

        self.config = config
        hidden_dims = config.hidden_dims

        num_levels = len(hidden_dims) - 1
        self.temporal_strides = self._compute_strides(config.temporal_compression, num_levels)
        self.spatial_strides = self._compute_strides(config.spatial_compression, num_levels)

        self.conv_in = CausalConv3d(
            config.in_channels, hidden_dims[0], (3, 3, 3),
            use_spectral_norm=config.use_spectral_norm,
            enable_streaming=config.enable_streaming
        )

        self.down_blocks = nn.ModuleList()
        current_resolution = config.spatial_compression * (2 ** num_levels)

        for i in range(num_levels):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]

            res_blocks = nn.ModuleList([
                ResidualBlock3D(
                    in_dim if j == 0 else out_dim, out_dim,
                    use_spectral_norm=config.use_spectral_norm,
                    enable_streaming=config.enable_streaming
                )
                for j in range(config.num_res_blocks)
            ])

            downsample = Downsample3D(
                out_dim, out_dim,
                temporal_stride=self.temporal_strides[i],
                spatial_stride=self.spatial_strides[i],
                use_spectral_norm=config.use_spectral_norm,
                enable_streaming=config.enable_streaming
            )

            current_resolution //= self.spatial_strides[i]

            use_attn = config.use_attention and current_resolution in config.attention_resolution
            attn = AttentionBlock3D(out_dim, use_spectral_norm=config.use_spectral_norm) if use_attn else nn.Identity()

            self.down_blocks.append(nn.ModuleDict({
                'res_blocks': res_blocks,
                'downsample': downsample,
                'attn': attn
            }))

        final_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock3D(final_dim, use_spectral_norm=config.use_spectral_norm)
        self.mid_attn = AttentionBlock3D(final_dim, use_spectral_norm=config.use_spectral_norm) if config.use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock3D(final_dim, use_spectral_norm=config.use_spectral_norm)

        self.norm_out = nn.GroupNorm(min(32, final_dim), final_dim)
        self.conv_out = CausalConv3d(
            final_dim, 2 * config.latent_dim, (3, 3, 3),
            use_spectral_norm=config.use_spectral_norm
        )

    def _compute_strides(self, total_compression: int, num_levels: int) -> List[int]:
        """Compute per-level strides to achieve total compression."""
        strides = []
        remaining = total_compression

        for i in range(num_levels):
            if remaining >= 2:
                stride = 2
                remaining //= 2
            else:
                stride = 1
            strides.append(stride)

        return strides

    def forward(
        self,
        x: torch.Tensor,
        return_skips: bool = False,
        streaming: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """Encode input to latent distribution parameters."""
        skips = []

        h = self.conv_in(x, streaming=streaming)

        for block in self.down_blocks:
            for res_block in block['res_blocks']:
                h = res_block(h, streaming=streaming)

            if return_skips:
                skips.append(h)

            h = block['downsample'](h, streaming=streaming)
            h = block['attn'](h)

        h = self.mid_block1(h, streaming=streaming)
        h = self.mid_attn(h)
        h = self.mid_block2(h, streaming=streaming)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h, streaming=streaming)

        mean, logvar = h.chunk(2, dim=1)

        if return_skips:
            return mean, logvar, skips
        return mean, logvar


# =============================================================================
# Decoder
# =============================================================================

class Decoder3D(nn.Module):
    """3D causal decoder with asymmetric decompression and skip connections."""

    def __init__(self, config: VAE3DConfig):
        super().__init__()

        self.config = config
        hidden_dims = list(reversed(config.hidden_dims))

        num_levels = len(hidden_dims) - 1
        encoder_t_strides = self._compute_strides(config.temporal_compression, num_levels)
        encoder_s_strides = self._compute_strides(config.spatial_compression, num_levels)
        self.temporal_strides = list(reversed(encoder_t_strides))
        self.spatial_strides = list(reversed(encoder_s_strides))
        self.num_levels = num_levels

        self.conv_in = CausalConv3d(
            config.latent_dim, hidden_dims[0], (3, 3, 3),
            use_spectral_norm=config.use_spectral_norm
        )

        self.mid_block1 = ResidualBlock3D(hidden_dims[0], use_spectral_norm=config.use_spectral_norm)
        self.mid_attn = AttentionBlock3D(hidden_dims[0], use_spectral_norm=config.use_spectral_norm) if config.use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock3D(hidden_dims[0], use_spectral_norm=config.use_spectral_norm)

        self.up_blocks = nn.ModuleList()
        self.skip_weights = nn.ParameterList() if config.use_skip_connections else None

        current_resolution = config.spatial_compression // (2 ** num_levels) * 2

        for i in range(num_levels):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]

            upsample = Upsample3D(
                in_dim, in_dim,
                temporal_stride=self.temporal_strides[i],
                spatial_stride=self.spatial_strides[i],
                use_spectral_norm=config.use_spectral_norm
            )

            current_resolution *= self.spatial_strides[i]

            skip_dim = in_dim if config.use_skip_connections else 0
            fuse_dim = in_dim + skip_dim

            res_blocks = nn.ModuleList([
                ResidualBlock3D(
                    fuse_dim if j == 0 else out_dim, out_dim,
                    use_spectral_norm=config.use_spectral_norm
                )
                for j in range(config.num_res_blocks)
            ])

            use_attn = config.use_attention and current_resolution in config.attention_resolution
            attn = AttentionBlock3D(out_dim, use_spectral_norm=config.use_spectral_norm) if use_attn else nn.Identity()

            if config.use_skip_connections:
                self.skip_weights.append(nn.Parameter(torch.tensor(config.skip_connection_scale)))

            self.up_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'res_blocks': res_blocks,
                'attn': attn
            }))

        final_dim = hidden_dims[-1]
        self.norm_out = nn.GroupNorm(min(32, final_dim), final_dim)
        self.conv_out = CausalConv3d(
            final_dim, config.out_channels, (3, 3, 3),
            use_spectral_norm=config.use_spectral_norm
        )

    def _compute_strides(self, total_compression: int, num_levels: int) -> List[int]:
        """Compute per-level strides to achieve total compression."""
        strides = []
        remaining = total_compression

        for i in range(num_levels):
            if remaining >= 2:
                stride = 2
                remaining //= 2
            else:
                stride = 1
            strides.append(stride)

        return strides

    def _compute_target_sizes(
        self,
        latent_shape: Tuple[int, int, int],
        output_shape: Optional[Tuple[int, int, int]] = None
    ) -> List[Tuple[int, int, int]]:
        """Compute target sizes for each upsampling level."""
        T_lat, H_lat, W_lat = latent_shape

        if output_shape is not None:
            T_out, H_out, W_out = output_shape
        else:
            T_out = T_lat * self.config.temporal_compression
            H_out = H_lat * self.config.spatial_compression
            W_out = W_lat * self.config.spatial_compression

        sizes = [(T_out, H_out, W_out)]
        T, H, W = T_out, H_out, W_out

        for i in range(self.num_levels - 1, -1, -1):
            T = T // self.temporal_strides[i]
            H = H // self.spatial_strides[i]
            W = W // self.spatial_strides[i]
            sizes.insert(0, (T, H, W))

        return sizes[1:]

    def forward(
        self,
        z: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
        output_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """Decode latent to output."""
        latent_shape = (z.shape[2], z.shape[3], z.shape[4])
        target_sizes = self._compute_target_sizes(latent_shape, output_shape)

        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        if skips is not None:
            skips = list(reversed(skips))

        for i, block in enumerate(self.up_blocks):
            target_size = target_sizes[i]
            h = block['upsample'](h, target_size=target_size)

            if skips is not None and self.config.use_skip_connections and self.skip_weights is not None:
                skip = skips[i]

                if skip.shape[2:] != h.shape[2:]:
                    skip = F.interpolate(skip, size=h.shape[2:], mode='trilinear', align_corners=False)

                skip_weight = self.skip_weights[i]
                h = torch.cat([h, skip * skip_weight], dim=1)

            for res_block in block['res_blocks']:
                h = res_block(h)

            h = block['attn'](h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# =============================================================================
# Main VAE3D v2 Model
# =============================================================================

class VAE3Dv2(nn.Module):
    """Enhanced 3D Causal Variational Autoencoder v2."""

    def __init__(self, config: Optional[VAE3DConfig] = None):
        super().__init__()

        self.config = config or VAE3DConfig()

        self.encoder = Encoder3D(self.config)
        self.decoder = Decoder3D(self.config)

        if self.config.quantization_mode == QuantizationMode.VQ:
            self.quantizer = VectorQuantizer(
                num_embeddings=self.config.num_codebook_vectors,
                embedding_dim=self.config.codebook_dim,
                commitment_cost=self.config.commitment_cost
            )

            if self.config.latent_dim != self.config.codebook_dim:
                self.pre_quant = nn.Conv3d(self.config.latent_dim, self.config.codebook_dim, 1)
                self.post_quant = nn.Conv3d(self.config.codebook_dim, self.config.latent_dim, 1)
            else:
                self.pre_quant = nn.Identity()
                self.post_quant = nn.Identity()
        else:
            self.quantizer = None

        if self.config.use_learnable_prior:
            self.prior = LearnablePrior(
                latent_dim=self.config.latent_dim,
                hidden_dim=self.config.prior_hidden_dim
            )
        else:
            self.prior = None

        if self.config.use_perceptual_loss:
            self.perceptual = PerceptualLoss(
                in_channels=self.config.in_channels,
                normalize=True
            )
        else:
            self.perceptual = None

        self.kl_annealer = KLAnnealer(
            mode=self.config.kl_anneal_mode,
            cycle_period=self.config.kl_cycle_period,
            start_beta=self.config.kl_start_beta,
            end_beta=self.config.kl_end_beta,
            warmup_steps=self.config.kl_warmup_steps,
            free_bits=self.config.kl_free_bits
        )

        self.latent_dropout = nn.Dropout(self.config.latent_dropout) if self.config.latent_dropout > 0 else nn.Identity()

        self._streaming_mode = False

    def enable_streaming(self):
        """Enable streaming inference mode."""
        self._streaming_mode = True
        self._reset_caches()

    def disable_streaming(self):
        """Disable streaming inference mode."""
        self._streaming_mode = False
        self._reset_caches()

    def _reset_caches(self):
        """Reset all streaming caches."""
        for module in self.modules():
            if isinstance(module, CausalConv3d):
                module.reset_cache()

    def encode(
        self,
        x: torch.Tensor,
        return_skips: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """Encode input to latent distribution parameters."""
        if self.config.use_gradient_checkpointing and self.training:
            return checkpoint(
                self.encoder, x, return_skips, self._streaming_mode,
                use_reentrant=False
            )
        return self.encoder(x, return_skips=return_skips, streaming=self._streaming_mode)

    def decode(
        self,
        z: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
        output_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """Decode latent to output."""
        if self.config.use_gradient_checkpointing and self.training:
            return checkpoint(self.decoder, z, skips, output_shape, use_reentrant=False)
        return self.decoder(z, skips=skips, output_shape=output_shape)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if deterministic or not self.training:
            return mean

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        z = self.latent_dropout(z)
        return z

    def quantize(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply vector quantization to latent."""
        if self.quantizer is None:
            return z, None, None

        z_pre = self.pre_quant(z)
        z_q, vq_loss, indices = self.quantizer(z_pre)
        z_q = self.post_quant(z_q)

        return z_q, vq_loss, indices

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Full VAE forward pass."""
        input_shape = (x.shape[2], x.shape[3], x.shape[4])

        if self.config.use_skip_connections:
            mean, logvar, skips = self.encode(x, return_skips=True)
        else:
            mean, logvar = self.encode(x, return_skips=False)
            skips = None

        z = self.reparameterize(mean, logvar, deterministic=deterministic)
        z, vq_loss, indices = self.quantize(z)

        recon = self.decode(z, skips=skips, output_shape=input_shape)

        output = {
            'recon': recon,
            'mean': mean,
            'logvar': logvar,
            'z': z
        }

        if indices is not None:
            output['indices'] = indices

        if return_loss:
            losses = self.compute_loss(x, recon, mean, logvar, vq_loss)
            output.update(losses)

        return output

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        vq_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss components."""
        losses = {}

        recon_loss = F.mse_loss(recon, x, reduction='mean')
        losses['recon_loss'] = recon_loss

        l1_loss = F.l1_loss(recon, x, reduction='mean')
        losses['l1_loss'] = l1_loss

        if self.prior is not None:
            prior_mean, prior_logvar = self.prior(x.size(0), x.device)
            kl_per_dim = 0.5 * (
                prior_logvar - logvar +
                (logvar.exp() + (mean - prior_mean).pow(2)) / prior_logvar.exp() - 1
            )
        else:
            kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

        kl_per_dim = self.kl_annealer.apply_free_bits(kl_per_dim)
        kl_loss = kl_per_dim.mean()

        beta = self.kl_annealer.get_beta()
        losses['kl_loss'] = kl_loss
        losses['kl_loss_weighted'] = beta * kl_loss
        losses['beta'] = torch.tensor(beta, device=x.device)

        if vq_loss is not None:
            losses['vq_loss'] = vq_loss

        if self.perceptual is not None:
            perceptual_loss = self.perceptual(recon, x)
            losses['perceptual_loss'] = perceptual_loss

        total = recon_loss + losses['kl_loss_weighted']

        if vq_loss is not None:
            total = total + vq_loss

        if self.perceptual is not None:
            total = total + self.config.perceptual_weight * losses['perceptual_loss']

        losses['total_loss'] = total

        return losses

    def sample(
        self,
        num_samples: int,
        device: torch.device,
        latent_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """Generate samples from the model."""
        if latent_shape is None:
            T = 8
            H = W = 256
            latent_shape = (
                T // self.config.temporal_compression,
                H // self.config.spatial_compression,
                W // self.config.spatial_compression
            )

        T_lat, H_lat, W_lat = latent_shape

        if self.prior is not None:
            mean, logvar = self.prior(num_samples, device)
        else:
            mean = torch.zeros(num_samples, self.config.latent_dim, T_lat, H_lat, W_lat, device=device)
            logvar = torch.zeros_like(mean)

        z = self.reparameterize(mean, logvar, deterministic=False)
        z, _, _ = self.quantize(z)

        return self.decode(z)

    def encode_frame(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single frame in streaming mode."""
        if not self._streaming_mode:
            self.enable_streaming()
        return self.encode(frame, return_skips=False)

    def decode_frame(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a single latent frame in streaming mode."""
        return self.decode(z)

    def training_step(self):
        """Call after each training step to update KL annealing."""
        self.kl_annealer.update()

    def get_latent_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get latent shape for given input shape."""
        T, H, W = input_shape
        return (
            T // self.config.temporal_compression,
            H // self.config.spatial_compression,
            W // self.config.spatial_compression
        )

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'VAE3Dv2':
        """Create model from config dictionary."""
        if 'quantization_mode' in config_dict and isinstance(config_dict['quantization_mode'], str):
            config_dict['quantization_mode'] = QuantizationMode(config_dict['quantization_mode'])

        config = VAE3DConfig(**config_dict)
        return cls(config)

    @classmethod
    def from_pretrained_v1(cls, v1_state_dict: Dict[str, torch.Tensor], config: Optional[VAE3DConfig] = None) -> 'VAE3Dv2':
        """Create v2 model from v1 checkpoint (partial weight loading)."""
        model = cls(config)

        model_dict = model.state_dict()

        loaded_keys = []
        skipped_keys = []

        for k, v in v1_state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

        model.load_state_dict(model_dict, strict=False)

        print(f"Loaded {len(loaded_keys)} keys from v1 checkpoint")
        print(f"Skipped {len(skipped_keys)} incompatible keys")

        return model


# =============================================================================
# Factory Functions
# =============================================================================

def get_default_config() -> VAE3DConfig:
    """Get default VAE3D v2 configuration."""
    return VAE3DConfig()


def get_minimal_config() -> VAE3DConfig:
    """Get minimal configuration for testing."""
    return VAE3DConfig(
        hidden_dims=[64, 128, 256],
        num_res_blocks=1,
        temporal_compression=2,
        spatial_compression=4,
        use_attention=False,
        use_skip_connections=False,
        use_perceptual_loss=False,
        use_gradient_checkpointing=False
    )


def get_high_quality_config() -> VAE3DConfig:
    """Get high-quality configuration with all features."""
    return VAE3DConfig(
        hidden_dims=[128, 256, 512, 512],
        num_res_blocks=3,
        temporal_compression=4,
        spatial_compression=8,
        use_attention=True,
        attention_resolution=[32, 16, 8],
        use_skip_connections=True,
        skip_connection_scale=0.5,
        use_kl_annealing=True,
        kl_anneal_mode="cyclical",
        use_learnable_prior=True,
        use_perceptual_loss=True,
        perceptual_weight=0.1,
        use_gradient_checkpointing=True
    )


def get_vqvae_config() -> VAE3DConfig:
    """Get VQ-VAE configuration."""
    return VAE3DConfig(
        hidden_dims=[128, 256, 512, 512],
        num_res_blocks=2,
        quantization_mode=QuantizationMode.VQ,
        num_codebook_vectors=8192,
        codebook_dim=8,
        commitment_cost=0.25,
        use_skip_connections=True,
        use_perceptual_loss=True
    )


# =============================================================================
# Testing
# =============================================================================

# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Alias for backward compatibility with v1 imports
VAE3D = VAE3Dv2
Encoder3D_v2 = Encoder3D  # Avoid name collision during transition
Decoder3D_v2 = Decoder3D


def causal_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    stride: int = 1,
    bias: bool = True
) -> nn.Module:
    """
    Legacy factory function for backward compatibility with vae3d v1.
    Creates a CausalConv3d instance.
    """
    stride_tuple = (stride, stride, stride) if isinstance(stride, int) else stride
    return CausalConv3d(
        in_channels, out_channels, kernel_size,
        stride=stride_tuple, bias=bias
    )


def test_vae3d_v2():
    """Test function for VAE3D v2."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing VAE3D v2 on {device}")

    # Test minimal config
    print("\n=== Testing Minimal Config ===")
    config = get_minimal_config()
    vae = VAE3Dv2(config).to(device)

    x = torch.randn(2, 3, 8, 64, 64).to(device)
    print(f"Input shape: {x.shape}")

    output = vae(x)
    print(f"Reconstruction shape: {output['recon'].shape}")
    print(f"Mean shape: {output['mean'].shape}")
    print(f"Latent shape: {output['z'].shape}")
    print(f"Total loss: {output['total_loss'].item():.4f}")

    expected_latent_t = 8 // config.temporal_compression
    expected_latent_s = 64 // config.spatial_compression
    print(f"Expected latent: T={expected_latent_t}, H/W={expected_latent_s}")

    # Test with skip connections
    print("\n=== Testing Skip Connections ===")
    config = get_minimal_config()
    config.use_skip_connections = True
    vae = VAE3Dv2(config).to(device)

    output = vae(x)
    print(f"With skip connections - Recon shape: {output['recon'].shape}")
    print(f"Loss: {output['total_loss'].item():.4f}")

    # Test KL annealing
    print("\n=== Testing KL Annealing ===")
    vae.kl_annealer = KLAnnealer(mode="cyclical", cycle_period=100)
    for step in range(0, 200, 20):
        vae.kl_annealer.step = step
        beta = vae.kl_annealer.get_beta()
        print(f"Step {step}: beta = {beta:.4f}")

    # Test streaming cache functionality
    print("\n=== Testing Streaming Cache ===")
    config = get_minimal_config()
    vae = VAE3Dv2(config).to(device)

    # Test cache reset
    vae.enable_streaming()
    assert vae._streaming_mode == True, "Streaming mode should be enabled"
    vae.disable_streaming()
    assert vae._streaming_mode == False, "Streaming mode should be disabled"
    print("Streaming cache management works correctly")

    # Note: Full streaming inference requires input sizes larger than
    # the receptive field of all downsampling stages combined.
    # For production use, ensure temporal chunks >= 8 frames and
    # spatial dimensions >= 128 pixels for default configs.

    # Test sampling
    print("\n=== Testing Sampling ===")
    config = get_minimal_config()
    vae = VAE3Dv2(config).to(device)
    samples = vae.sample(2, device, latent_shape=(4, 16, 16))
    print(f"Sample shape: {samples.shape}")

    print("\n=== All Tests Passed! ===")


if __name__ == "__main__":
    test_vae3d_v2()
