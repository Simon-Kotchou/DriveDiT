"""
Video and frame transformation utilities with zero dependencies.
Implements data augmentation and preprocessing transforms.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import random
import math
import cv2
import numpy as np


class FrameTransforms:
    """Individual frame transformation utilities."""
    
    @staticmethod
    def normalize(frame: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> torch.Tensor:
        """Normalize frame with mean and std."""
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        mean = torch.tensor(mean, dtype=frame.dtype, device=frame.device).view(-1, 1, 1)
        std = torch.tensor(std, dtype=frame.dtype, device=frame.device).view(-1, 1, 1)
        
        return (frame - mean) / std
    
    @staticmethod
    def denormalize(frame: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> torch.Tensor:
        """Denormalize frame."""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        mean = torch.tensor(mean, dtype=frame.dtype, device=frame.device).view(-1, 1, 1)
        std = torch.tensor(std, dtype=frame.dtype, device=frame.device).view(-1, 1, 1)
        
        return frame * std + mean
    
    @staticmethod
    def resize(frame: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear') -> torch.Tensor:
        """Resize frame to target size."""
        if frame.dim() == 3:  # [C, H, W]
            frame = frame.unsqueeze(0)  # [1, C, H, W]
            squeeze = True
        else:
            squeeze = False
        
        resized = F.interpolate(frame, size=size, mode=mode, align_corners=False)
        
        if squeeze:
            resized = resized.squeeze(0)
        
        return resized
    
    @staticmethod
    def crop(frame: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
        """Crop frame."""
        if frame.dim() == 3:  # [C, H, W]
            return frame[:, top:top+height, left:left+width]
        elif frame.dim() == 4:  # [B, C, H, W]
            return frame[:, :, top:top+height, left:left+width]
        else:
            raise ValueError(f"Unsupported frame dimensions: {frame.dim()}")
    
    @staticmethod
    def center_crop(frame: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Center crop frame."""
        h, w = frame.shape[-2:]
        target_h, target_w = size
        
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        
        return FrameTransforms.crop(frame, top, left, target_h, target_w)
    
    @staticmethod
    def random_crop(frame: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Random crop frame."""
        h, w = frame.shape[-2:]
        target_h, target_w = size
        
        if h < target_h or w < target_w:
            # Pad if too small
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            frame = F.pad(frame, (0, pad_w, 0, pad_h))
            h, w = frame.shape[-2:]
        
        top = random.randint(0, h - target_h)
        left = random.randint(0, w - target_w)
        
        return FrameTransforms.crop(frame, top, left, target_h, target_w)
    
    @staticmethod
    def horizontal_flip(frame: torch.Tensor) -> torch.Tensor:
        """Horizontally flip frame."""
        return torch.flip(frame, dims=[-1])
    
    @staticmethod
    def vertical_flip(frame: torch.Tensor) -> torch.Tensor:
        """Vertically flip frame."""
        return torch.flip(frame, dims=[-2])
    
    @staticmethod
    def rotate(frame: torch.Tensor, angle: float, mode: str = 'bilinear') -> torch.Tensor:
        """Rotate frame by angle (in degrees)."""
        # Convert to radians
        angle_rad = math.radians(angle)
        
        # Create rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # For simplicity, use affine transformation
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=frame.dtype, device=frame.device)
        theta = theta.unsqueeze(0)  # [1, 2, 3]
        
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        grid = F.affine_grid(theta, frame.size(), align_corners=False)
        rotated = F.grid_sample(frame, grid, mode=mode, align_corners=False)
        
        if squeeze:
            rotated = rotated.squeeze(0)
        
        return rotated
    
    @staticmethod
    def brightness(frame: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust brightness."""
        return torch.clamp(frame * factor, 0.0, 1.0)
    
    @staticmethod
    def contrast(frame: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust contrast."""
        mean = frame.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((frame - mean) * factor + mean, 0.0, 1.0)
    
    @staticmethod
    def saturation(frame: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust saturation."""
        if frame.shape[-3] != 3:
            return frame  # Only works for RGB
        
        # Convert to grayscale
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=frame.dtype, device=frame.device)
        weights = weights.view(-1, 1, 1)
        gray = (frame * weights).sum(dim=-3, keepdim=True)
        gray = gray.expand_as(frame)
        
        return torch.clamp((frame - gray) * factor + gray, 0.0, 1.0)
    
    @staticmethod
    def hue(frame: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust hue (simplified RGB implementation)."""
        if frame.shape[-3] != 3:
            return frame
        
        # Simple hue shift in RGB space (not perfect but zero-dependency)
        r, g, b = frame.unbind(dim=-3)
        
        # Rotate colors
        factor = factor % 1.0
        if factor < 1/3:
            # R -> G shift
            alpha = factor * 3
            new_r = r * (1 - alpha) + g * alpha
            new_g = g * (1 - alpha) + b * alpha
            new_b = b * (1 - alpha) + r * alpha
        elif factor < 2/3:
            # G -> B shift
            alpha = (factor - 1/3) * 3
            new_r = g * (1 - alpha) + b * alpha
            new_g = b * (1 - alpha) + r * alpha
            new_b = r * (1 - alpha) + g * alpha
        else:
            # B -> R shift
            alpha = (factor - 2/3) * 3
            new_r = b * (1 - alpha) + r * alpha
            new_g = r * (1 - alpha) + g * alpha
            new_b = g * (1 - alpha) + b * alpha
        
        return torch.clamp(torch.stack([new_r, new_g, new_b], dim=-3), 0.0, 1.0)
    
    @staticmethod
    def gaussian_noise(frame: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(frame) * std
        return torch.clamp(frame + noise, 0.0, 1.0)


class VideoTransforms:
    """Video sequence transformation utilities."""
    
    @staticmethod
    def temporal_crop(video: torch.Tensor, length: int, start: Optional[int] = None) -> torch.Tensor:
        """Crop video temporally."""
        T = video.shape[0] if video.dim() == 4 else video.shape[1]
        
        if start is None:
            start = random.randint(0, max(0, T - length))
        
        if video.dim() == 4:  # [T, C, H, W]
            return video[start:start+length]
        elif video.dim() == 5:  # [B, T, C, H, W]
            return video[:, start:start+length]
        else:
            raise ValueError(f"Unsupported video dimensions: {video.dim()}")
    
    @staticmethod
    def temporal_subsample(video: torch.Tensor, factor: int) -> torch.Tensor:
        """Subsample video temporally."""
        if video.dim() == 4:  # [T, C, H, W]
            return video[::factor]
        elif video.dim() == 5:  # [B, T, C, H, W]
            return video[:, ::factor]
        else:
            raise ValueError(f"Unsupported video dimensions: {video.dim()}")
    
    @staticmethod
    def temporal_flip(video: torch.Tensor) -> torch.Tensor:
        """Flip video temporally (reverse)."""
        if video.dim() == 4:  # [T, C, H, W]
            return torch.flip(video, dims=[0])
        elif video.dim() == 5:  # [B, T, C, H, W]
            return torch.flip(video, dims=[1])
        else:
            raise ValueError(f"Unsupported video dimensions: {video.dim()}")
    
    @staticmethod
    def apply_frame_transform(video: torch.Tensor, transform_fn: Callable, **kwargs) -> torch.Tensor:
        """Apply frame transform to all frames in video."""
        if video.dim() == 4:  # [T, C, H, W]
            return torch.stack([transform_fn(frame, **kwargs) for frame in video])
        elif video.dim() == 5:  # [B, T, C, H, W]
            B, T = video.shape[:2]
            video_flat = video.view(B * T, *video.shape[2:])
            transformed_flat = torch.stack([transform_fn(frame, **kwargs) for frame in video_flat])
            return transformed_flat.view(B, T, *transformed_flat.shape[1:])
        else:
            raise ValueError(f"Unsupported video dimensions: {video.dim()}")
    
    @staticmethod
    def temporal_jitter(video: torch.Tensor, max_jitter: int = 2) -> torch.Tensor:
        """Apply temporal jitter to video frames."""
        if max_jitter <= 0:
            return video
        
        T = video.shape[0] if video.dim() == 4 else video.shape[1]
        
        # Create jittered indices
        jitter_offsets = [random.randint(-max_jitter, max_jitter) for _ in range(T)]
        indices = []
        
        for i, offset in enumerate(jitter_offsets):
            new_idx = max(0, min(T - 1, i + offset))
            indices.append(new_idx)
        
        indices = torch.tensor(indices, dtype=torch.long, device=video.device)
        
        if video.dim() == 4:  # [T, C, H, W]
            return video[indices]
        elif video.dim() == 5:  # [B, T, C, H, W]
            return video[:, indices]
        else:
            raise ValueError(f"Unsupported video dimensions: {video.dim()}")


class AugmentationPipeline:
    """Composable augmentation pipeline for videos."""
    
    def __init__(self, transforms: List[Dict[str, Any]], p: float = 0.5):
        """
        Initialize augmentation pipeline.
        
        Args:
            transforms: List of transform specifications
            p: Probability of applying the pipeline
        """
        self.transforms = transforms
        self.p = p
    
    def __call__(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply augmentation pipeline."""
        if random.random() > self.p:
            return video
        
        for transform_spec in self.transforms:
            transform_name = transform_spec['name']
            transform_p = transform_spec.get('p', 1.0)
            transform_params = transform_spec.get('params', {})
            
            if random.random() <= transform_p:
                video = self._apply_transform(video, transform_name, transform_params)
        
        return video
    
    def _apply_transform(self, video: torch.Tensor, transform_name: str, params: Dict[str, Any]) -> torch.Tensor:
        """Apply a single transform."""
        if transform_name == 'horizontal_flip':
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.horizontal_flip)
        
        elif transform_name == 'vertical_flip':
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.vertical_flip)
        
        elif transform_name == 'rotation':
            angle_range = params.get('angle_range', (-10, 10))
            angle = random.uniform(*angle_range)
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.rotate, angle=angle)
        
        elif transform_name == 'brightness':
            factor_range = params.get('factor_range', (0.8, 1.2))
            factor = random.uniform(*factor_range)
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.brightness, factor=factor)
        
        elif transform_name == 'contrast':
            factor_range = params.get('factor_range', (0.8, 1.2))
            factor = random.uniform(*factor_range)
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.contrast, factor=factor)
        
        elif transform_name == 'saturation':
            factor_range = params.get('factor_range', (0.8, 1.2))
            factor = random.uniform(*factor_range)
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.saturation, factor=factor)
        
        elif transform_name == 'hue':
            factor_range = params.get('factor_range', (-0.1, 0.1))
            factor = random.uniform(*factor_range)
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.hue, factor=factor)
        
        elif transform_name == 'gaussian_noise':
            std_range = params.get('std_range', (0.0, 0.02))
            std = random.uniform(*std_range)
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.gaussian_noise, std=std)
        
        elif transform_name == 'temporal_crop':
            length = params.get('length', video.shape[0] if video.dim() == 4 else video.shape[1])
            return VideoTransforms.temporal_crop(video, length)
        
        elif transform_name == 'temporal_subsample':
            factor = params.get('factor', 2)
            return VideoTransforms.temporal_subsample(video, factor)
        
        elif transform_name == 'temporal_flip':
            return VideoTransforms.temporal_flip(video)
        
        elif transform_name == 'temporal_jitter':
            max_jitter = params.get('max_jitter', 2)
            return VideoTransforms.temporal_jitter(video, max_jitter)
        
        elif transform_name == 'random_crop':
            size = params.get('size', (224, 224))
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.random_crop, size=size)
        
        elif transform_name == 'center_crop':
            size = params.get('size', (224, 224))
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.center_crop, size=size)
        
        elif transform_name == 'resize':
            size = params.get('size', (256, 256))
            mode = params.get('mode', 'bilinear')
            return VideoTransforms.apply_frame_transform(video, FrameTransforms.resize, size=size, mode=mode)
        
        else:
            raise ValueError(f"Unknown transform: {transform_name}")
    
    @classmethod
    def create_training_pipeline(cls, image_size: Tuple[int, int] = (256, 256)) -> 'AugmentationPipeline':
        """Create a standard training augmentation pipeline."""
        transforms = [
            {
                'name': 'horizontal_flip',
                'p': 0.5
            },
            {
                'name': 'brightness',
                'p': 0.3,
                'params': {'factor_range': (0.9, 1.1)}
            },
            {
                'name': 'contrast',
                'p': 0.3,
                'params': {'factor_range': (0.9, 1.1)}
            },
            {
                'name': 'saturation',
                'p': 0.2,
                'params': {'factor_range': (0.9, 1.1)}
            },
            {
                'name': 'gaussian_noise',
                'p': 0.1,
                'params': {'std_range': (0.0, 0.01)}
            },
            {
                'name': 'temporal_jitter',
                'p': 0.3,
                'params': {'max_jitter': 1}
            }
        ]
        
        return cls(transforms, p=0.8)
    
    @classmethod
    def create_validation_pipeline(cls, image_size: Tuple[int, int] = (256, 256)) -> 'AugmentationPipeline':
        """Create a minimal validation pipeline."""
        transforms = [
            {
                'name': 'center_crop',
                'p': 1.0,
                'params': {'size': image_size}
            }
        ]
        
        return cls(transforms, p=1.0)