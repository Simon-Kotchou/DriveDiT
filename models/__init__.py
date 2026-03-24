"""
Model architectures for DriveDiT.
Zero-dependency implementations of world modeling components.
"""

from .dit_student import DiTStudent, MemoryBank, TokenEmbedding
from .dit_teacher import DiTTeacher, TeacherEmbedding
from .vae3d import VAE3D, Encoder3D, Decoder3D, ResidualBlock3D, AttentionBlock3D
from .vae3d_v2 import (
    VAE3Dv2,
    VAE3DConfig,
    KLAnnealer,
    VectorQuantizer,
    LearnablePrior,
    CausalConv3d,
    CausalConvTranspose3d
)
from .world_model import (
    WorldModel,
    PatchEmbed,
    ControlEncoder,
    DepthEncoder,
    MemorySystem,
    JEPAPredictor,
    FlowMatchingPredictor,
    SelfForcingScheduler
)

# Advanced conditioning modules
from .conditioning import (
    ConditioningConfig,
    CameraEncoder,
    RoadTopologyEncoder,
    BoundingBox3DEncoder,
    EgoStateEncoder,
    TemporalEnvironmentalEncoder,
    AdaLNModulation,
    ConditioningIntegration,
    create_conditioning_module
)

__all__ = [
    # DiT Student
    'DiTStudent',
    'MemoryBank',
    'TokenEmbedding',
    # DiT Teacher
    'DiTTeacher',
    'TeacherEmbedding',
    # VAE3D v1
    'VAE3D',
    'Encoder3D',
    'Decoder3D',
    'ResidualBlock3D',
    'AttentionBlock3D',
    # VAE3D v2 (enhanced)
    'VAE3Dv2',
    'VAE3DConfig',
    'KLAnnealer',
    'VectorQuantizer',
    'LearnablePrior',
    'CausalConv3d',
    'CausalConvTranspose3d',
    # World Model
    'WorldModel',
    'PatchEmbed',
    'ControlEncoder',
    'DepthEncoder',
    'MemorySystem',
    'JEPAPredictor',
    'FlowMatchingPredictor',
    'SelfForcingScheduler',
    # Conditioning modules
    'ConditioningConfig',
    'CameraEncoder',
    'RoadTopologyEncoder',
    'BoundingBox3DEncoder',
    'EgoStateEncoder',
    'TemporalEnvironmentalEncoder',
    'AdaLNModulation',
    'ConditioningIntegration',
    'create_conditioning_module',
]
