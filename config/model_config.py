"""
Model configuration classes.
Configuration for VAE, DiT student/teacher, and composite models.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from .base_config import BaseConfig


@dataclass
class VAEConfig(BaseConfig):
    """Configuration for 3D Causal VAE."""
    
    # Architecture
    in_channels: int = 3
    latent_dim: int = 8
    hidden_dims: List[int] = None
    num_res_blocks: int = 2
    use_attention: bool = True
    
    # Training
    beta: float = 1.0
    
    # Performance
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 512, 512]
        super().__post_init__()
    
    def validate(self):
        """Validate VAE configuration."""
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.num_res_blocks >= 0, "num_res_blocks must be non-negative"
        assert 0.0 <= self.beta <= 10.0, "beta should be in [0, 10]"
        assert len(self.hidden_dims) > 0, "hidden_dims cannot be empty"


@dataclass
class DiTConfig(BaseConfig):
    """Configuration for DiT (Diffusion Transformer) models."""
    
    # Architecture
    latent_dim: int = 8
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: Optional[int] = None
    mlp_type: str = 'swiglu'
    
    # Attention
    use_rope: bool = True
    use_flash_attention: bool = False
    causal: bool = True  # True for student, False for teacher
    
    # Memory and context
    max_seq_len: int = 2048
    context_length: int = 8
    use_memory: bool = True
    memory_size: int = 1024
    
    # Conditioning
    cond_dim: Optional[int] = None
    time_embed_dim: int = 256
    
    # Performance
    dropout: float = 0.0
    bias: bool = False
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Diffusion (for teacher)
    num_diffusion_steps: int = 1000
    
    def __post_init__(self):
        if self.d_ff is None:
            if self.mlp_type in ['swiglu', 'geglu']:
                self.d_ff = int(8 * self.d_model / 3)
                self.d_ff = ((self.d_ff + 7) // 8) * 8  # Round to multiple of 8
            else:
                self.d_ff = 4 * self.d_model
        super().__post_init__()
    
    def validate(self):
        """Validate DiT configuration."""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        assert self.mlp_type in ['standard', 'swiglu', 'geglu'], f"Invalid mlp_type: {self.mlp_type}"


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration combining all components."""
    
    # Model components
    vae: VAEConfig = None
    student: DiTConfig = None
    teacher: DiTConfig = None
    
    # Global settings
    device: str = 'cuda'
    seed: int = 42
    
    # Input/output dimensions
    image_size: Tuple[int, int] = (512, 512)
    sequence_length: int = 32
    control_dim: int = 4  # steer, accel, goal_x, goal_y
    
    # Flow matching
    flow_steps: int = 4
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    
    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.vae is None:
            self.vae = VAEConfig()
        
        if self.student is None:
            self.student = DiTConfig(causal=True)
        
        if self.teacher is None:
            self.teacher = DiTConfig(causal=False)
        
        # Ensure consistency between components
        self.student.latent_dim = self.vae.latent_dim
        self.teacher.latent_dim = self.vae.latent_dim
        
        super().__post_init__()
    
    def validate(self):
        """Validate complete model configuration."""
        assert self.image_size[0] > 0 and self.image_size[1] > 0, "Invalid image_size"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert self.control_dim >= 0, "control_dim must be non-negative"
        assert self.flow_steps > 0, "flow_steps must be positive"
        assert 0 < self.sigma_min < self.sigma_max, "Invalid sigma values"
        
        # Validate sub-configurations
        self.vae.validate()
        self.student.validate()
        self.teacher.validate()
        
        # Check consistency
        assert self.student.latent_dim == self.vae.latent_dim, "Student and VAE latent_dim mismatch"
        assert self.teacher.latent_dim == self.vae.latent_dim, "Teacher and VAE latent_dim mismatch"


# Predefined configurations for different use cases
def get_small_config() -> ModelConfig:
    """Get configuration for small/testing model."""
    return ModelConfig(
        vae=VAEConfig(
            latent_dim=4,
            hidden_dims=[64, 128, 256],
            num_res_blocks=1
        ),
        student=DiTConfig(
            latent_dim=4,
            d_model=256,
            n_layers=6,
            n_heads=8,
            max_seq_len=128
        ),
        teacher=DiTConfig(
            latent_dim=4,
            d_model=256,
            n_layers=6,
            n_heads=8,
            max_seq_len=128,
            causal=False
        ),
        image_size=(128, 128),
        sequence_length=16
    )


def get_medium_config() -> ModelConfig:
    """Get configuration for medium-sized model."""
    return ModelConfig(
        vae=VAEConfig(
            latent_dim=8,
            hidden_dims=[128, 256, 512],
            num_res_blocks=2
        ),
        student=DiTConfig(
            latent_dim=8,
            d_model=512,
            n_layers=12,
            n_heads=8,
            max_seq_len=512
        ),
        teacher=DiTConfig(
            latent_dim=8,
            d_model=512,
            n_layers=12,
            n_heads=8,
            max_seq_len=512,
            causal=False
        ),
        image_size=(256, 256),
        sequence_length=32
    )


def get_large_config() -> ModelConfig:
    """Get configuration for large production model."""
    return ModelConfig(
        vae=VAEConfig(
            latent_dim=8,
            hidden_dims=[128, 256, 512, 512],
            num_res_blocks=2,
            mixed_precision=True
        ),
        student=DiTConfig(
            latent_dim=8,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            max_seq_len=2048,
            mixed_precision=True,
            gradient_checkpointing=True
        ),
        teacher=DiTConfig(
            latent_dim=8,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            max_seq_len=2048,
            causal=False,
            mixed_precision=True,
            gradient_checkpointing=True
        ),
        image_size=(512, 512),
        sequence_length=64
    )