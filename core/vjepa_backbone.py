"""
V-JEPA 2.1 Backbone for REPA (Representation Alignment).

Provides frozen V-JEPA 2.1 features for aligning DiT hidden states.
V-JEPA 2.1 gives DINOv3-quality dense features + native video temporal understanding.

References:
- V-JEPA 2.1: arxiv:2603.14482
- REPA: 17.5× training speedup via representation alignment (ICLR 2025 Oral)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from einops import rearrange
import math
import warnings


@dataclass
class VJEPAConfig:
    """Configuration for V-JEPA backbone."""
    # Model architecture (ViT-L/16 defaults)
    embed_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    tubelet_size: int = 2  # Temporal patch size

    # Input specifications
    num_frames: int = 16
    image_size: int = 224

    # Feature extraction
    extract_layers: List[int] = field(default_factory=lambda: [18, 19, 20, 21, 22, 23])
    pool_type: str = "mean"  # "mean", "cls", "all"

    # Memory/performance
    use_fp16: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self):
        self.d_head = self.embed_dim // self.num_heads
        self.mlp_dim = int(self.embed_dim * self.mlp_ratio)

        # Compute patch counts
        self.num_spatial_patches = (self.image_size // self.patch_size) ** 2
        self.num_temporal_patches = self.num_frames // self.tubelet_size
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video with tubelets."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_spatial_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] video tensor

        Returns:
            [B, num_patches, embed_dim] patch embeddings
        """
        B = x.shape[0]
        x = self.proj(x)  # [B, E, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, E]
        return x


class VJEPAAttentionBlock(nn.Module):
    """Pre-norm attention block for V-JEPA encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Self-attention
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))

        return x


class VJEPAEncoder(nn.Module):
    """V-JEPA Vision Transformer encoder with intermediate feature extraction."""

    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            image_size=config.image_size,
            patch_size=config.patch_size,
            tubelet_size=config.tubelet_size,
            embed_dim=config.embed_dim
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.embed_dim)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VJEPAAttentionBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T, H, W] video tensor
            return_intermediate: Return features from extract_layers

        Returns:
            Dict with 'final', 'pooled', 'patch_features', 'intermediate'
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Interpolate position embedding if needed
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self._interpolate_pos_embed(x.shape[1])
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Collect intermediate features
        intermediate = {}

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if return_intermediate and idx in self.config.extract_layers:
                intermediate[f'layer_{idx}'] = x[:, 1:]  # Exclude CLS

        # Final norm
        x = self.norm(x)

        # Extract outputs
        cls_token = x[:, 0]
        patch_features = x[:, 1:]

        return {
            'final': cls_token,
            'pooled': patch_features.mean(dim=1),
            'patch_features': patch_features,
            'intermediate': intermediate
        }

    def _interpolate_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Interpolate position embeddings for different sequence lengths."""
        pos_embed = self.pos_embed
        if num_patches == pos_embed.shape[1]:
            return pos_embed

        # Simple linear interpolation
        pos_embed = F.interpolate(
            pos_embed.transpose(1, 2),
            size=num_patches,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        return pos_embed


class VJEPABackbone(nn.Module):
    """
    Frozen V-JEPA backbone for REPA feature extraction.

    Provides dense video features for aligning DiT hidden states.
    All parameters are frozen - only used for feature extraction.
    """

    def __init__(self, config: Optional[VJEPAConfig] = None):
        super().__init__()
        self.config = config or VJEPAConfig()

        # Build encoder
        self.encoder = VJEPAEncoder(self.config)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Normalization constants (ImageNet)
        self.register_buffer('pixel_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[VJEPAConfig] = None,
        device: Optional[torch.device] = None
    ) -> 'VJEPABackbone':
        """Load pretrained V-JEPA model."""
        if config is None:
            # Default to ViT-L config
            config = VJEPAConfig()

        backbone = cls(config=config)

        # Try to load weights
        try:
            if model_name_or_path.startswith('facebook/') or model_name_or_path.startswith('meta/'):
                # HuggingFace model hub
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(
                    repo_id=model_name_or_path.replace('meta/', 'facebook/'),
                    filename="pytorch_model.bin"
                )
                state_dict = torch.load(checkpoint_path, map_location='cpu')
            else:
                # Local path
                state_dict = torch.load(model_name_or_path, map_location='cpu')

            # Load state dict (handle various formats)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            backbone.encoder.load_state_dict(state_dict, strict=False)

        except Exception as e:
            warnings.warn(f"Could not load pretrained weights: {e}. Using random initialization.")

        if device is not None:
            backbone = backbone.to(device)

        return backbone

    def preprocess(self, video: torch.Tensor) -> torch.Tensor:
        """Preprocess video for V-JEPA."""
        B, T, C, H, W = video.shape if video.dim() == 5 else (video.shape[0], *video.shape[1:])

        # Ensure BCTHW format
        if video.dim() == 5 and video.shape[1] != 3:
            video = video.transpose(1, 2)  # BTCHW -> BCTHW

        # Sample/resize if needed
        target_frames = self.config.num_frames
        target_size = self.config.image_size

        if T != target_frames:
            indices = torch.linspace(0, T - 1, target_frames).long()
            video = video[:, :, indices]

        if H != target_size or W != target_size:
            video = rearrange(video, 'b c t h w -> (b t) c h w')
            video = F.interpolate(video, size=(target_size, target_size), mode='bilinear', align_corners=False)
            video = rearrange(video, '(b t) c h w -> b c t h w', b=B, t=target_frames)

        # Normalize
        if video.max() > 1.0:
            video = video / 255.0
        video = (video - self.pixel_mean) / self.pixel_std

        return video

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        preprocess: bool = True,
        return_all_layers: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from video.

        Args:
            video: [B, T, C, H, W] or [B, C, T, H, W] input video
            preprocess: Apply preprocessing
            return_all_layers: Return features from all extract_layers

        Returns:
            Dict with 'final', 'pooled', 'patch_features', 'intermediate'
        """
        self.encoder.eval()

        if preprocess:
            video = self.preprocess(video)

        if self.config.use_fp16 and video.dtype != torch.float16:
            video = video.half()

        return self.encoder(video, return_intermediate=return_all_layers)

    def get_feature_dim(self) -> int:
        return self.config.embed_dim

    def train(self, mode: bool = True):
        """Keep encoder frozen."""
        self.encoder.eval()
        return self


def create_vjepa_backbone(
    model_size: str = 'large',
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    **kwargs
) -> VJEPABackbone:
    """Factory function to create V-JEPA backbone."""
    configs = {
        'base': VJEPAConfig(embed_dim=768, num_layers=12, num_heads=12, extract_layers=[6, 7, 8, 9, 10, 11]),
        'large': VJEPAConfig(embed_dim=1024, num_layers=24, num_heads=16, extract_layers=[18, 19, 20, 21, 22, 23]),
        'huge': VJEPAConfig(embed_dim=1280, num_layers=32, num_heads=16, extract_layers=[26, 27, 28, 29, 30, 31])
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    config = configs[model_size]
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if pretrained:
        model_names = {
            'base': 'facebook/vjepa-vit-b',
            'large': 'facebook/vjepa-2.1-vit-l',
            'huge': 'facebook/vjepa-vit-h'
        }
        try:
            backbone = VJEPABackbone.from_pretrained(model_names[model_size], config=config, device=device)
        except Exception:
            warnings.warn(f"Could not load pretrained weights for {model_size}. Using random initialization.")
            backbone = VJEPABackbone(config=config)
            if device:
                backbone = backbone.to(device)
    else:
        backbone = VJEPABackbone(config=config)
        if device:
            backbone = backbone.to(device)

    return backbone


if __name__ == "__main__":
    print("Testing V-JEPA Backbone...")

    config = VJEPAConfig(
        embed_dim=512, num_layers=6, num_heads=8,
        extract_layers=[3, 4, 5], num_frames=8, image_size=112
    )

    backbone = VJEPABackbone(config=config)

    B, T, C, H, W = 2, 8, 3, 112, 112
    video = torch.randn(B, T, C, H, W)

    features = backbone(video)

    print(f"Final features: {features['final'].shape}")
    print(f"Pooled features: {features['pooled'].shape}")
    print(f"Patch features: {features['patch_features'].shape}")
    print(f"Intermediate layers: {list(features['intermediate'].keys())}")

    print("\nV-JEPA Backbone test passed!")
