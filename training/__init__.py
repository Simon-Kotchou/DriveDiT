"""
Training pipelines for DriveDiT.
Zero-dependency training implementations with Self-Forcing and flow matching.
"""

from .self_forcing import SelfForcingTrainer, create_simple_model
from .self_forcing_plus import (
    RollingKVCache,
    CurriculumScheduler,
    SelfForcingPlusConfig,
    FutureAnchorEncoder,
    ExtendedControlEncoder,
    ControlConditioner,
    EMAModel,
    UncertaintyWeighting,
    PerLayerGradientClipper,
    SelfForcingPlusTrainer
)
from .unified_trainer import UnifiedTrainer
from .distill import DistillationTrainer
from .distributed import MemoryMonitor, CheckpointManager, DistributedManager
from .losses import (
    ReconstructionLoss,
    TemporalConsistencyLoss,
    FlowMatchingLoss,
    UnifiedLoss
)

# Enhanced losses v2
from .losses_v2 import (
    LossType,
    LossConfig,
    LatentMSELoss,
    TeacherStudentFlowLoss,
    InfoNCELoss,
    ConditionalFlowMatchingLoss,
    ScaleInvariantDepthLoss,
    LPIPSLoss,
    TemporalConsistencyLoss as TemporalConsistencyLossV2
)

# CUDA-optimized training utilities
from .cuda_optimized import (
    CUDAOptimizedConfig,
    AsyncPrefetcher,
    CUDADataPipeline,
    GradientAccumulator,
    MemoryOptimizer,
    OptimizedTrainer,
    create_optimized_optimizer
)

# Fused CUDA kernels
from .kernels.fused_normalize import FusedNormalize, FusedRMSNorm, fused_normalize_kernel
from .kernels.fused_augment import FusedAugmentation, VideoAugmentation, fused_augment_kernel

__all__ = [
    # Self-Forcing v1
    'SelfForcingTrainer',
    'create_simple_model',
    # Self-Forcing++ (v2)
    'RollingKVCache',
    'CurriculumScheduler',
    'SelfForcingPlusConfig',
    'FutureAnchorEncoder',
    'ExtendedControlEncoder',
    'ControlConditioner',
    'EMAModel',
    'UncertaintyWeighting',
    'PerLayerGradientClipper',
    'SelfForcingPlusTrainer',
    # Unified Trainer
    'UnifiedTrainer',
    # Distillation
    'DistillationTrainer',
    # Distributed
    'MemoryMonitor',
    'CheckpointManager',
    'DistributedManager',
    # Losses v1
    'ReconstructionLoss',
    'TemporalConsistencyLoss',
    'FlowMatchingLoss',
    'UnifiedLoss',
    # Losses v2 (enhanced)
    'LossType',
    'LossConfig',
    'LatentMSELoss',
    'TeacherStudentFlowLoss',
    'InfoNCELoss',
    'ConditionalFlowMatchingLoss',
    'ScaleInvariantDepthLoss',
    'LPIPSLoss',
    'TemporalConsistencyLossV2',
    # CUDA optimized
    'CUDAOptimizedConfig',
    'AsyncPrefetcher',
    'CUDADataPipeline',
    'GradientAccumulator',
    'MemoryOptimizer',
    'OptimizedTrainer',
    'create_optimized_optimizer',
    # Fused kernels
    'FusedNormalize',
    'FusedRMSNorm',
    'fused_normalize_kernel',
    'FusedAugmentation',
    'VideoAugmentation',
    'fused_augment_kernel',
]
