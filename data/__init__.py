"""
Data loading and preprocessing utilities for DriveDiT.
Zero-dependency data pipeline components.

- File-based: video loading with memory-mapped caching (pipeline.py, video_loader.py)
- Enfusion: dc-1.0 telemetry preprocessing (enfusion_preprocessing.py, legacy)

dc-2.0 capture sessions are read by data/dc2/reader.py (docs/capture_contract/dc-2.0.md). The dc-1.0 session loader,
the Rust loaders and the streaming loader were deleted in DD-12; they are kept in tag archive/dc1-rust.
"""

from .video_loader import VideoDataset, VideoLoader, FrameBuffer
from .transforms import VideoTransforms, FrameTransforms, AugmentationPipeline
from .preprocessing import VideoPreprocessor, LatentPreprocessor, ControlPreprocessor
from .collate import VideoCollator, LatentCollator, custom_collate_fn
from .sampler import TemporalSampler, BalancedSampler, ChunkedSampler

# Enfusion capture data support (dc-1.0 telemetry, legacy)
from .enfusion_preprocessing import (
    EnfusionTelemetryConfig,
    EnfusionTelemetryParser,
    EnfusionControlNormalizer,
    EnfusionFrameProcessor,
    EnfusionDepthProcessor,
    EnfusionSceneParser,
    EnfusionQualityFilter,
    EnfusionAnchorDetector,
    create_enfusion_preprocessors
)

__all__ = [
    # Core video utilities
    'VideoDataset', 'VideoLoader', 'FrameBuffer',
    'VideoTransforms', 'FrameTransforms', 'AugmentationPipeline',
    'VideoPreprocessor', 'LatentPreprocessor', 'ControlPreprocessor',
    'VideoCollator', 'LatentCollator', 'custom_collate_fn',
    'TemporalSampler', 'BalancedSampler', 'ChunkedSampler',
    # Enfusion preprocessing
    'EnfusionTelemetryConfig',
    'EnfusionTelemetryParser',
    'EnfusionControlNormalizer',
    'EnfusionFrameProcessor',
    'EnfusionDepthProcessor',
    'EnfusionSceneParser',
    'EnfusionQualityFilter',
    'EnfusionAnchorDetector',
    'create_enfusion_preprocessors',
]
