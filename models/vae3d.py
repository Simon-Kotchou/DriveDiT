"""
3D Causal VAE implementation.
Time-causal convolutions with proper padding for world modeling.
WAN (World-Action-Network) distilled architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.nn_helpers import RMSNorm, silu


def causal_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    stride: int = 1,
    bias: bool = True
) -> nn.Module:
    """
    Create a time-causal 3D convolution.
    Causal in time dimension only, standard padding for spatial dimensions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (time, height, width)
        stride: Stride (applied to all dimensions)
        bias: Whether to use bias
    
    Returns:
        Sequential module with padding and convolution
    """
    kt, kh, kw = kernel_size
    
    # Causal padding in time: (kernel_size - 1, 0)
    pad_t = (kt - 1, 0)
    # Standard padding for spatial dimensions
    pad_h = (kh // 2, kh // 2)
    pad_w = (kw // 2, kw // 2)
    
    return nn.Sequential(
        nn.ConstantPad3d(pad_t + pad_h + pad_w, 0.0),
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=bias)
    )


def causal_conv_transpose3d(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    stride: int = 1,
    bias: bool = True
) -> nn.Module:
    """
    Create a time-causal 3D transpose convolution.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (time, height, width)
        stride: Stride
        bias: Whether to use bias
    
    Returns:
        Sequential module with transpose convolution and cropping
    """
    kt, kh, kw = kernel_size
    
    # For transpose convolution, we need to handle causal cropping
    conv_transpose = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size, stride, bias=bias
    )
    
    class CausalTransposeConv3d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = conv_transpose
            self.kt = kt
        
        def forward(self, x):
            # Apply transpose convolution
            out = self.conv_transpose(x)
            
            # Crop to maintain causality (remove future padding)
            if self.kt > 1:
                out = out[:, :, :-(self.kt-1), :, :]
            
            return out
    
    return CausalTransposeConv3d()


class ResidualBlock3D(nn.Module):
    """
    3D residual block with causal convolutions and normalization.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = causal_conv3d(channels, channels, kernel_size)
        
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = causal_conv3d(channels, channels, kernel_size)
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Output tensor [B, C, T, H, W]
        """
        residual = x
        
        # First conv block
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Second conv block
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        return x + residual


class AttentionBlock3D(nn.Module):
    """
    3D attention block for VAE with spatial-temporal attention.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D attention.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Output tensor [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        residual = x
        
        # Normalize and compute QKV
        x = self.norm(x)
        qkv = self.qkv(x)  # [B, 3*C, T, H, W]
        
        # Reshape for attention: [B, 3, H, C//H, T*H*W]
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, T * H * W)
        q, k, v = qkv.unbind(1)  # Each [B, H, C//H, T*H*W]
        
        # Attention computation
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdt,bhds->bhts', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhts,bhds->bhdt', attn, v)
        
        # Reshape back to spatial dimensions
        out = out.view(B, C, T, H, W)
        out = self.proj(out)
        
        return out + residual


class Encoder3D(nn.Module):
    """
    3D causal encoder for VAE.
    Downsamples spatial dimensions while preserving temporal causality.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [128, 256, 512, 512],
        latent_dim: int = 8,
        num_res_blocks: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Initial convolution
        self.conv_in = causal_conv3d(in_channels, hidden_dims[0], (3, 3, 3))
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            # Residual blocks
            res_blocks = nn.ModuleList([
                ResidualBlock3D(in_dim) for _ in range(num_res_blocks)
            ])
            
            # Downsample (spatial only, preserve time)
            downsample = causal_conv3d(
                in_dim, out_dim, 
                kernel_size=(3, 4, 4),  # Larger spatial kernel for downsampling
                stride=(1, 2, 2)  # Downsample spatial dimensions only
            )
            
            # Optional attention
            attn = AttentionBlock3D(out_dim) if use_attention and i == len(hidden_dims) - 2 else nn.Identity()
            
            self.down_blocks.append(nn.ModuleDict({
                'res_blocks': res_blocks,
                'downsample': downsample,
                'attn': attn
            }))
        
        # Final processing
        final_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock3D(final_dim)
        self.mid_attn = AttentionBlock3D(final_dim) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock3D(final_dim)
        
        # Latent projection
        self.norm_out = nn.GroupNorm(32, final_dim)
        self.conv_out = causal_conv3d(final_dim, 2 * latent_dim, (3, 3, 3))  # Mean and logvar
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Tuple of (mean, logvar) each [B, latent_dim, T, H//8, W//8]
        """
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling path
        for block in self.down_blocks:
            # Apply residual blocks
            for res_block in block['res_blocks']:
                h = res_block(h)
            
            # Downsample
            h = block['downsample'](h)
            
            # Attention
            h = block['attn'](h)
        
        # Middle processing
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Final projection
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Split into mean and logvar
        mean, logvar = h.chunk(2, dim=1)
        
        return mean, logvar


class Decoder3D(nn.Module):
    """
    3D causal decoder for VAE.
    Upsamples from latent representation while preserving temporal causality.
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: List[int] = [512, 512, 256, 128],
        out_channels: int = 3,
        num_res_blocks: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        
        # Initial processing
        self.conv_in = causal_conv3d(latent_dim, hidden_dims[0], (3, 3, 3))
        
        self.mid_block1 = ResidualBlock3D(hidden_dims[0])
        self.mid_attn = AttentionBlock3D(hidden_dims[0]) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock3D(hidden_dims[0])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            # Upsample (spatial only)
            upsample = causal_conv_transpose3d(
                in_dim, out_dim,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2)
            )
            
            # Residual blocks
            res_blocks = nn.ModuleList([
                ResidualBlock3D(out_dim) for _ in range(num_res_blocks)
            ])
            
            # Optional attention
            attn = AttentionBlock3D(out_dim) if use_attention and i == 0 else nn.Identity()
            
            self.up_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'res_blocks': res_blocks,
                'attn': attn
            }))
        
        # Final output
        final_dim = hidden_dims[-1]
        self.norm_out = nn.GroupNorm(32, final_dim)
        self.conv_out = causal_conv3d(final_dim, out_channels, (3, 3, 3))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor [B, latent_dim, T, H//8, W//8]
        
        Returns:
            Decoded output [B, out_channels, T, H, W]
        """
        # Initial processing
        h = self.conv_in(z)
        
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsampling path
        for block in self.up_blocks:
            # Upsample
            h = block['upsample'](h)
            
            # Attention
            h = block['attn'](h)
            
            # Apply residual blocks
            for res_block in block['res_blocks']:
                h = res_block(h)
        
        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VAE3D(nn.Module):
    """
    3D Causal Variational Autoencoder for world modeling.
    Time-causal architecture suitable for autoregressive generation.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 8,
        hidden_dims: List[int] = [128, 256, 512, 512],
        num_res_blocks: int = 2,
        use_attention: bool = True,
        beta: float = 1.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder and decoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention
        )
        
        self.decoder = Decoder3D(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Tuple of (mean, logvar)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor [B, latent_dim, T, H//8, W//8]
        
        Returns:
            Decoded output [B, C, T, H, W]
        """
        return self.decoder(z)
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mean: Mean tensor [B, latent_dim, T, H, W]
            logvar: Log variance tensor [B, latent_dim, T, H, W]
        
        Returns:
            Sampled latent [B, latent_dim, T, H, W]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Tuple of (reconstructed, mean, logvar)
        """
        # Encode
        mean, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mean, logvar)
        
        # Decode
        recon = self.decode(z)
        
        return recon, mean, logvar
    
    def loss_function(
        self,
        recon: torch.Tensor,
        input: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon: Reconstructed output [B, C, T, H, W]
            input: Original input [B, C, T, H, W]
            mean: Latent mean [B, latent_dim, T, H, W]
            logvar: Latent log variance [B, latent_dim, T, H, W]
        
        Returns:
            Total loss scalar
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, input, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss


def test_vae3d():
    """Test function for VAE3D."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    vae = VAE3D(
        in_channels=3,
        latent_dim=8,
        hidden_dims=[64, 128, 256, 256],  # Smaller for testing
        num_res_blocks=1,  # Fewer blocks for testing
        use_attention=True
    ).to(device)
    
    # Test input
    x = torch.randn(2, 3, 8, 64, 64).to(device)  # [B, C, T, H, W]
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    recon, mean, logvar = vae(x)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Compute loss
    loss = vae.loss_function(recon, x, mean, logvar)
    print(f"Loss: {loss.item():.4f}")
    
    # Test encode/decode separately
    with torch.no_grad():
        mean_test, logvar_test = vae.encode(x)
        z = vae.reparameterize(mean_test, logvar_test)
        recon_test = vae.decode(z)
        print(f"Separate encode/decode shape: {recon_test.shape}")


if __name__ == "__main__":
    test_vae3d()