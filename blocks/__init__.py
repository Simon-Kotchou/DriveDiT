"""
Composite building blocks for DriveDiT.
DiT blocks, flow matching components, and other composite modules.
"""

from .dit_block import DiTBlock, DiTBlockWithCrossAttention, MemoryEfficientDiTBlock
from .flow_matching import (
    FlowMatchingConfig,
    TimestepEmbedding,
    FlowMatchingBlock,
    FlowPredictor,
    RectifiedFlowSampler,
    FlowMatchingLoss,
    FlowMatchingTrainer
)

# Backward compatibility aliases
FlowMatchingSampler = RectifiedFlowSampler
FlowLoss = FlowMatchingLoss

__all__ = [
    # DiT blocks
    'DiTBlock',
    'DiTBlockWithCrossAttention',
    'MemoryEfficientDiTBlock',
    # Flow matching
    'FlowMatchingConfig',
    'TimestepEmbedding',
    'FlowMatchingBlock',
    'FlowPredictor',
    'RectifiedFlowSampler',
    'FlowMatchingLoss',
    'FlowMatchingTrainer',
    # Aliases for backward compatibility
    'FlowMatchingSampler',
    'FlowLoss',
]
