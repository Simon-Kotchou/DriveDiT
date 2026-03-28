"""
Data loading and preprocessing utilities for DriveDiT.
Zero-dependency data pipeline components.

Supports multiple loading modes:
- File-based: Traditional video/image loading with caching
- Enfusion: Session-based loading from Enfusion game captures
- Streaming: Real-time loading via Rust drivedit_streaming module
- Hybrid: Automatic fallback between streaming and file-based
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

# Streaming data support (real-time Enfusion capture via Rust)
from .pipeline import create_streaming_loader, is_streaming_available
try:
    from .streaming import (
        StreamingDataLoader,
        StreamingEnfusionDataset,
        StreamingConfig,
        check_streaming_available,
    )
    from .streaming_utils import (
        TelemetryMapping,
        telemetry_to_control_tensor,
        normalize_frames,
        batch_telemetry_to_controls,
        SequenceWindower,
        StreamingBatcher,
        validate_batch,
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

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
    # Streaming data loading (real-time Enfusion capture)
    'create_streaming_loader',
    'is_streaming_available',
    'STREAMING_AVAILABLE',
    # Hybrid loader (Python/Rust)
    'HybridDataLoader',
    'HybridLoaderConfig',
    'create_hybrid_dataloaders',
    'get_best_loader',
    'RUST_AVAILABLE',
]

# Conditionally export streaming components
if STREAMING_AVAILABLE:
    __all__.extend([
        'StreamingDataLoader',
        'StreamingEnfusionDataset',
        'StreamingConfig',
        'check_streaming_available',
        # Streaming utilities
        'TelemetryMapping',
        'telemetry_to_control_tensor',
        'normalize_frames',
        'batch_telemetry_to_controls',
        'SequenceWindower',
        'StreamingBatcher',
        'validate_batch',
    ])

# Conditionally export Rust loader components
if RUST_AVAILABLE:
    __all__.extend([
        'RustDataLoader',
        'RustVideoDecoder',
        'RustTelemetryParser',
        'RustFrameBuffer',
        'check_rust_available',
    ])