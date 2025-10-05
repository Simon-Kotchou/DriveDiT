"""
Data preprocessing utilities for DriveDiT.
Handles video, latent, and control signal preprocessing.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
from pathlib import Path


class VideoPreprocessor:
    """Video preprocessing utilities."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        mean: List[float] = None,
        std: List[float] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize video preprocessor.
        
        Args:
            target_size: Target spatial size (H, W)
            normalize: Whether to normalize frames
            mean: Normalization mean
            std: Normalization std
            dtype: Target data type
        """
        self.target_size = target_size
        self.normalize = normalize
        self.dtype = dtype
        
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        else:
            self.mean = mean
        
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std
    
    def preprocess_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a batch of video frames.
        
        Args:
            frames: Video frames [B, T, C, H, W] or [T, C, H, W]
            
        Returns:
            Preprocessed frames
        """
        # Ensure correct data type
        frames = frames.to(dtype=self.dtype)
        
        # Ensure values are in [0, 1] range
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        # Resize if needed
        if frames.shape[-2:] != self.target_size:
            frames = self._resize_video(frames)
        
        # Normalize if requested
        if self.normalize:
            frames = self._normalize_video(frames)
        
        return frames
    
    def _resize_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize video frames."""
        original_shape = frames.shape
        
        if frames.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)
        elif frames.dim() == 4:  # [T, C, H, W]
            T, C, H, W = frames.shape
            frames_flat = frames
        else:
            raise ValueError(f"Unsupported frame dimensions: {frames.dim()}")
        
        # Resize
        resized_flat = F.interpolate(
            frames_flat, 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Reshape back
        if frames.dim() == 5:
            return resized_flat.view(B, T, C, *self.target_size)
        else:
            return resized_flat
    
    def _normalize_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize video frames."""
        mean = torch.tensor(self.mean, dtype=frames.dtype, device=frames.device)
        std = torch.tensor(self.std, dtype=frames.dtype, device=frames.device)
        
        # Reshape for broadcasting
        if frames.dim() == 5:  # [B, T, C, H, W]
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
        elif frames.dim() == 4:  # [T, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        
        return (frames - mean) / std
    
    def denormalize(self, frames: torch.Tensor) -> torch.Tensor:
        """Denormalize video frames."""
        if not self.normalize:
            return frames
        
        mean = torch.tensor(self.mean, dtype=frames.dtype, device=frames.device)
        std = torch.tensor(self.std, dtype=frames.dtype, device=frames.device)
        
        # Reshape for broadcasting
        if frames.dim() == 5:  # [B, T, C, H, W]
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
        elif frames.dim() == 4:  # [T, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        
        return frames * std + mean
    
    def get_frame_statistics(self, frames: torch.Tensor) -> Dict[str, float]:
        """Get frame statistics."""
        return {
            'mean': frames.mean().item(),
            'std': frames.std().item(),
            'min': frames.min().item(),
            'max': frames.max().item(),
            'shape': list(frames.shape)
        }


class LatentPreprocessor:
    """Latent representation preprocessing utilities."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        spatial_size: Tuple[int, int] = (32, 32),
        normalize_latents: bool = True,
        clamp_range: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize latent preprocessor.
        
        Args:
            latent_dim: Latent feature dimension
            spatial_size: Spatial size of latent maps
            normalize_latents: Whether to normalize latent values
            clamp_range: Range to clamp latent values
        """
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.normalize_latents = normalize_latents
        self.clamp_range = clamp_range
    
    def preprocess_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Preprocess latent representations.
        
        Args:
            latents: Latent tensors [B, T, C, H, W] or [B, C, H, W]
            
        Returns:
            Preprocessed latents
        """
        # Resize spatial dimensions if needed
        if latents.shape[-2:] != self.spatial_size:
            latents = self._resize_latents(latents)
        
        # Normalize if requested
        if self.normalize_latents:
            latents = self._normalize_latents(latents)
        
        # Clamp if requested
        if self.clamp_range is not None:
            latents = torch.clamp(latents, *self.clamp_range)
        
        return latents
    
    def _resize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Resize latent spatial dimensions."""
        if latents.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = latents.shape
            latents_flat = latents.view(B * T, C, H, W)
            resized_flat = F.interpolate(latents_flat, size=self.spatial_size, mode='bilinear', align_corners=False)
            return resized_flat.view(B, T, C, *self.spatial_size)
        elif latents.dim() == 4:  # [B, C, H, W]
            return F.interpolate(latents, size=self.spatial_size, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported latent dimensions: {latents.dim()}")
    
    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents to unit variance."""
        # Compute statistics across spatial dimensions
        dims = [-2, -1]  # Spatial dimensions
        
        mean = latents.mean(dim=dims, keepdim=True)
        std = latents.std(dim=dims, keepdim=True)
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)
        
        return (latents - mean) / std
    
    def tokenize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert latents to token sequences.
        
        Args:
            latents: Latent tensors [B, T, C, H, W]
            
        Returns:
            Token sequences [B, T, C*H*W]
        """
        if latents.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = latents.shape
            return latents.view(B, T, C * H * W)
        elif latents.dim() == 4:  # [B, C, H, W]
            B, C, H, W = latents.shape
            return latents.view(B, C * H * W)
        else:
            raise ValueError(f"Unsupported latent dimensions: {latents.dim()}")
    
    def detokenize_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert token sequences back to latents.
        
        Args:
            tokens: Token sequences [B, T, C*H*W] or [B, C*H*W]
            
        Returns:
            Latent tensors [B, T, C, H, W] or [B, C, H, W]
        """
        C, H, W = self.latent_dim, *self.spatial_size
        
        if tokens.dim() == 3:  # [B, T, C*H*W]
            B, T, _ = tokens.shape
            return tokens.view(B, T, C, H, W)
        elif tokens.dim() == 2:  # [B, C*H*W]
            B, _ = tokens.shape
            return tokens.view(B, C, H, W)
        else:
            raise ValueError(f"Unsupported token dimensions: {tokens.dim()}")
    
    def compute_latent_statistics(self, latents: torch.Tensor) -> Dict[str, float]:
        """Compute latent representation statistics."""
        return {
            'mean': latents.mean().item(),
            'std': latents.std().item(),
            'min': latents.min().item(),
            'max': latents.max().item(),
            'norm': latents.norm().item(),
            'sparsity': (latents.abs() < 1e-6).float().mean().item()
        }


class ControlPreprocessor:
    """Control signal preprocessing utilities."""
    
    def __init__(
        self,
        control_dim: int = 4,
        normalize_controls: bool = True,
        control_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        smooth_controls: bool = True,
        smooth_window: int = 3
    ):
        """
        Initialize control preprocessor.
        
        Args:
            control_dim: Control signal dimension
            normalize_controls: Whether to normalize control signals
            control_ranges: Range for each control dimension
            smooth_controls: Whether to smooth control signals
            smooth_window: Smoothing window size
        """
        self.control_dim = control_dim
        self.normalize_controls = normalize_controls
        self.smooth_controls = smooth_controls
        self.smooth_window = smooth_window
        
        if control_ranges is None:
            # Default ranges for [steering, acceleration, goal_x, goal_y]
            self.control_ranges = {
                0: (-1.0, 1.0),   # Steering
                1: (-1.0, 1.0),   # Acceleration
                2: (-10.0, 10.0), # Goal X
                3: (-10.0, 10.0)  # Goal Y
            }
        else:
            self.control_ranges = control_ranges
    
    def preprocess_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Preprocess control signals.
        
        Args:
            controls: Control signals [B, T, C] or [T, C]
            
        Returns:
            Preprocessed controls
        """
        # Smooth controls if requested
        if self.smooth_controls:
            controls = self._smooth_controls(controls)
        
        # Normalize if requested
        if self.normalize_controls:
            controls = self._normalize_controls(controls)
        
        return controls
    
    def _smooth_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to control signals."""
        if self.smooth_window <= 1:
            return controls
        
        # Use 1D convolution for smoothing
        kernel = torch.ones(1, 1, self.smooth_window, device=controls.device) / self.smooth_window
        
        if controls.dim() == 3:  # [B, T, C]
            B, T, C = controls.shape
            controls_reshaped = controls.permute(0, 2, 1)  # [B, C, T]
            
            smoothed = []
            for c in range(C):
                channel = controls_reshaped[:, c:c+1, :].unsqueeze(1)  # [B, 1, 1, T]
                smoothed_channel = F.conv2d(
                    channel, 
                    kernel.unsqueeze(0), 
                    padding=(0, self.smooth_window // 2)
                )
                smoothed.append(smoothed_channel.squeeze(1).squeeze(1))  # [B, T]
            
            smoothed = torch.stack(smoothed, dim=2)  # [B, T, C]
            
        elif controls.dim() == 2:  # [T, C]
            T, C = controls.shape
            controls = controls.unsqueeze(0)  # [1, T, C]
            smoothed = self._smooth_controls(controls)
            smoothed = smoothed.squeeze(0)  # [T, C]
        
        else:
            raise ValueError(f"Unsupported control dimensions: {controls.dim()}")
        
        return smoothed
    
    def _normalize_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """Normalize control signals to [-1, 1] range."""
        normalized = controls.clone()
        
        for dim_idx, (min_val, max_val) in self.control_ranges.items():
            if dim_idx < controls.shape[-1]:
                # Normalize to [-1, 1]
                channel_data = controls[..., dim_idx]
                normalized_channel = 2 * (channel_data - min_val) / (max_val - min_val) - 1
                normalized[..., dim_idx] = torch.clamp(normalized_channel, -1.0, 1.0)
        
        return normalized
    
    def denormalize_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """Denormalize control signals from [-1, 1] range."""
        if not self.normalize_controls:
            return controls
        
        denormalized = controls.clone()
        
        for dim_idx, (min_val, max_val) in self.control_ranges.items():
            if dim_idx < controls.shape[-1]:
                # Denormalize from [-1, 1]
                channel_data = controls[..., dim_idx]
                denormalized_channel = (channel_data + 1) * (max_val - min_val) / 2 + min_val
                denormalized[..., dim_idx] = denormalized_channel
        
        return denormalized
    
    def compute_control_statistics(self, controls: torch.Tensor) -> Dict[str, Any]:
        """Compute control signal statistics."""
        stats = {}
        
        for dim_idx in range(controls.shape[-1]):
            channel_data = controls[..., dim_idx]
            stats[f'dim_{dim_idx}'] = {
                'mean': channel_data.mean().item(),
                'std': channel_data.std().item(),
                'min': channel_data.min().item(),
                'max': channel_data.max().item()
            }
        
        return stats
    
    def interpolate_controls(self, controls: torch.Tensor, target_length: int) -> torch.Tensor:
        """Interpolate control signals to target length."""
        if controls.dim() == 2:  # [T, C]
            controls = controls.unsqueeze(0)  # [1, T, C]
            squeeze = True
        else:
            squeeze = False
        
        # Interpolate using 1D interpolation
        B, T, C = controls.shape
        controls_transposed = controls.permute(0, 2, 1)  # [B, C, T]
        
        interpolated = F.interpolate(
            controls_transposed.unsqueeze(-1),  # [B, C, T, 1]
            size=(target_length, 1),
            mode='bilinear',
            align_corners=False
        ).squeeze(-1)  # [B, C, target_length]
        
        interpolated = interpolated.permute(0, 2, 1)  # [B, target_length, C]
        
        if squeeze:
            interpolated = interpolated.squeeze(0)  # [target_length, C]
        
        return interpolated
    
    def add_control_noise(self, controls: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add noise to control signals for regularization."""
        noise = torch.randn_like(controls) * noise_std
        return controls + noise


class DataPreprocessingPipeline:
    """Complete data preprocessing pipeline."""
    
    def __init__(
        self,
        video_preprocessor: Optional[VideoPreprocessor] = None,
        latent_preprocessor: Optional[LatentPreprocessor] = None,
        control_preprocessor: Optional[ControlPreprocessor] = None
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            video_preprocessor: Video preprocessing component
            latent_preprocessor: Latent preprocessing component
            control_preprocessor: Control preprocessing component
        """
        self.video_preprocessor = video_preprocessor or VideoPreprocessor()
        self.latent_preprocessor = latent_preprocessor or LatentPreprocessor()
        self.control_preprocessor = control_preprocessor or ControlPreprocessor()
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply preprocessing to a batch."""
        processed_batch = {}
        
        # Process video frames
        if 'frames' in batch:
            processed_batch['frames'] = self.video_preprocessor.preprocess_batch(batch['frames'])
        
        # Process latents
        if 'latents' in batch:
            processed_batch['latents'] = self.latent_preprocessor.preprocess_latents(batch['latents'])
        
        # Process controls
        if 'controls' in batch:
            processed_batch['controls'] = self.control_preprocessor.preprocess_controls(batch['controls'])
        
        # Copy other fields
        for key, value in batch.items():
            if key not in ['frames', 'latents', 'controls']:
                processed_batch[key] = value
        
        return processed_batch
    
    def get_preprocessor_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration of all preprocessors."""
        return {
            'video': {
                'target_size': self.video_preprocessor.target_size,
                'normalize': self.video_preprocessor.normalize,
                'mean': self.video_preprocessor.mean,
                'std': self.video_preprocessor.std
            },
            'latent': {
                'latent_dim': self.latent_preprocessor.latent_dim,
                'spatial_size': self.latent_preprocessor.spatial_size,
                'normalize_latents': self.latent_preprocessor.normalize_latents
            },
            'control': {
                'control_dim': self.control_preprocessor.control_dim,
                'normalize_controls': self.control_preprocessor.normalize_controls,
                'control_ranges': self.control_preprocessor.control_ranges
            }
        }