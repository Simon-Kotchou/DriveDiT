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
    # Losses
    'ReconstructionLoss',
    'TemporalConsistencyLoss',
    'FlowMatchingLoss',
    'UnifiedLoss',
]
