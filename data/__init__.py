"""
Data loading and preprocessing utilities for DriveDiT.
Zero-dependency data pipeline components.
"""

from .video_loader import VideoDataset, VideoLoader, FrameBuffer
from .transforms import VideoTransforms, FrameTransforms, AugmentationPipeline
from .preprocessing import VideoPreprocessor, LatentPreprocessor, ControlPreprocessor
from .collate import VideoCollator, LatentCollator, custom_collate_fn
from .sampler import TemporalSampler, BalancedSampler, ChunkedSampler

# Enfusion capture data support
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
from .enfusion_loader import (
    EnfusionDatasetConfig,
    EnfusionSession,
    EnfusionDataset,
    EnfusionCollator,
    EnfusionDataLoader,
    EnfusionToDriveDiTAdapter,
    create_enfusion_dataloaders
)

# High-performance Rust-based data loading (optional)
try:
    from .rust_loader import (
        RustDataLoader,
        RustVideoDecoder,
        RustTelemetryParser,
        RustFrameBuffer,
        check_rust_available
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Hybrid Python/Rust loader with automatic fallback
from .hybrid_loader import (
    HybridDataLoader,
    HybridLoaderConfig,
    create_hybrid_dataloaders,
    get_best_loader
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
    # Enfusion data loading
    'EnfusionDatasetConfig',
    'EnfusionSession',
    'EnfusionDataset',
    'EnfusionCollator',
    'EnfusionDataLoader',
    'EnfusionToDriveDiTAdapter',
    'create_enfusion_dataloaders',
    # Hybrid loader (Python/Rust)
    'HybridDataLoader',
    'HybridLoaderConfig',
    'create_hybrid_dataloaders',
    'get_best_loader',
    'RUST_AVAILABLE',
]

# Conditionally export Rust loader components
if RUST_AVAILABLE:
    __all__.extend([
        'RustDataLoader',
        'RustVideoDecoder',
        'RustTelemetryParser',
        'RustFrameBuffer',
        'check_rust_available',
    ])