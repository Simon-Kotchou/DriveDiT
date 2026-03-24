"""
Model architectures for DriveDiT.
Zero-dependency implementations of world modeling components.
"""

from .dit_student import DiTStudent, MemoryBank, TokenEmbedding
from .dit_teacher import DiTTeacher, TeacherEmbedding
# VAE3D (v2 is now the default, with v1 backward compatibility aliases)
from .vae3d import (
    VAE3D,              # Alias for VAE3Dv2
    VAE3Dv2,            # Full v2 implementation
    VAE3DConfig,
    Encoder3D,
    Decoder3D,
    ResidualBlock3D,
    AttentionBlock3D,
    KLAnnealer,
    VectorQuantizer,
    LearnablePrior,
    CausalConv3d,
    CausalConvTranspose3d,
    causal_conv3d,      # Factory function for backward compatibility
    # Config factories
    get_default_config,
    get_minimal_config,
    get_high_quality_config,
    get_vqvae_config,
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
    # VAE3D (v2 with v1 compatibility)
    'VAE3D',           # Alias for VAE3Dv2
    'VAE3Dv2',         # Explicit v2 reference
    'Encoder3D',
    'Decoder3D',
    'ResidualBlock3D',
    'AttentionBlock3D',
    'VAE3DConfig',
    'KLAnnealer',
    'VectorQuantizer',
    'LearnablePrior',
    'CausalConv3d',
    'CausalConvTranspose3d',
    'causal_conv3d',
    'get_default_config',
    'get_minimal_config',
    'get_high_quality_config',
    'get_vqvae_config',
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
