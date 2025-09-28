"""
Unified configuration system for DriveDiT.
Single configuration class that controls all aspects of the model and training.
Replaces scattered config classes with a clean, unified system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum


class ComponentType(Enum):
    """Types of optional components."""
    CONTROL = "control"
    DEPTH = "depth" 
    MEMORY = "memory"
    JEPA = "jepa"
    FLOW_MATCHING = "flow_matching"


@dataclass
class DriveDiTConfig:
    """Unified configuration for DriveDiT model and training."""
    
    # Model architecture
    model_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: int = 4
    
    # Input/output specifications
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    sequence_length: int = 16
    vae_latent_dim: int = 8
    
    # Enabled components (modular design)
    enabled_components: List[ComponentType] = field(default_factory=lambda: [
        ComponentType.CONTROL,
        ComponentType.MEMORY,
        ComponentType.FLOW_MATCHING
    ])
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Self-forcing and curriculum learning (integrated comma.ai insights)
    enable_curriculum: bool = True
    initial_sequence_length: int = 4
    final_sequence_length: int = 32
    curriculum_warmup_steps: int = 10000
    
    initial_self_forcing_ratio: float = 0.0
    final_self_forcing_ratio: float = 0.8
    self_forcing_warmup_steps: int = 5000
    self_forcing_schedule: str = "cosine"  # "linear", "cosine", "exponential"
    max_self_forcing_ratio: float = 0.8
    
    # Flow matching configuration
    enable_flow_matching: bool = True
    flow_matching_weight: float = 1.0
    num_flow_steps: int = 4
    
    # Control signals (if enabled)
    control_input_dim: int = 6  # [steer, accel, brake, goal_x, goal_y, speed]
    control_hidden_dim: int = 256
    control_num_layers: int = 2
    control_dropout: float = 0.1
    
    # Depth processing (if enabled)
    depth_max_depth: float = 100.0
    depth_channels: int = 1
    
    # Memory system (if enabled)
    memory_dim: int = 512
    object_memory_size: int = 64
    spatial_memory_size: int = 256
    memory_update_rate: float = 0.1
    
    # JEPA configuration (if enabled)
    jepa_target_length: int = 4
    jepa_prediction_head_dim: int = 512
    jepa_temperature: float = 0.1
    
    # Training stability
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    checkpoint_every: int = 1000
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # Memory management
    max_memory_gb: float = 16.0
    memory_cleanup_threshold: float = 0.8
    
    # Data pipeline
    video_chunk_duration: int = 300  # seconds
    video_overlap_duration: int = 30  # seconds
    use_scene_detection: bool = True
    min_quality_score: float = 0.7
    
    # Loss weights
    reconstruction_weight: float = 1.0
    temporal_consistency_weight: float = 0.2
    jepa_weight: float = 0.1
    perceptual_weight: float = 0.1
    l1_weight: float = 0.1
    
    def get_enabled_components(self) -> List[ComponentType]:
        """Get list of enabled components."""
        return self.enabled_components
    
    def is_component_enabled(self, component: ComponentType) -> bool:
        """Check if a specific component is enabled."""
        return component in self.enabled_components
    
    def enable_component(self, component: ComponentType):
        """Enable a component."""
        if component not in self.enabled_components:
            self.enabled_components.append(component)
    
    def disable_component(self, component: ComponentType):
        """Disable a component."""
        if component in self.enabled_components:
            self.enabled_components.remove(component)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            'model_dim': self.model_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'sequence_length': self.sequence_length,
            'vae_latent_dim': self.vae_latent_dim,
            'enabled_components': self.enabled_components
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_steps': self.max_steps,
            'warmup_steps': self.warmup_steps,
            'enable_curriculum': self.enable_curriculum,
            'curriculum_warmup_steps': self.curriculum_warmup_steps,
            'initial_self_forcing_ratio': self.initial_self_forcing_ratio,
            'final_self_forcing_ratio': self.final_self_forcing_ratio,
            'self_forcing_warmup_steps': self.self_forcing_warmup_steps,
            'enable_flow_matching': self.enable_flow_matching,
            'flow_matching_weight': self.flow_matching_weight,
            'gradient_clip_norm': self.gradient_clip_norm,
            'mixed_precision': self.mixed_precision
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data pipeline configuration."""
        return {
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'image_size': self.image_size,
            'video_chunk_duration': self.video_chunk_duration,
            'video_overlap_duration': self.video_overlap_duration,
            'use_scene_detection': self.use_scene_detection,
            'min_quality_score': self.min_quality_score
        }


def get_minimal_config() -> DriveDiTConfig:
    """Get minimal configuration for development/testing."""
    return DriveDiTConfig(
        # Small model for fast iteration
        model_dim=256,
        num_layers=6,
        num_heads=4,
        sequence_length=8,
        image_size=128,
        
        # Basic training
        batch_size=4,
        max_steps=10000,
        
        # Enable only core components
        enabled_components=[ComponentType.CONTROL],
        
        # Simple curriculum
        enable_curriculum=True,
        initial_sequence_length=2,
        final_sequence_length=8,
        curriculum_warmup_steps=1000,
        
        # Basic flow matching
        enable_flow_matching=True,
        num_flow_steps=2,
        
        # Memory management
        max_memory_gb=4.0
    )


def get_research_config() -> DriveDiTConfig:
    """Get full research configuration with all components."""
    return DriveDiTConfig(
        # Full model
        model_dim=768,
        num_layers=24,
        num_heads=12,
        sequence_length=32,
        image_size=256,
        
        # Research training
        batch_size=8,
        max_steps=200000,
        learning_rate=5e-5,
        
        # All components enabled
        enabled_components=[
            ComponentType.CONTROL,
            ComponentType.DEPTH,
            ComponentType.MEMORY,
            ComponentType.JEPA,
            ComponentType.FLOW_MATCHING
        ],
        
        # Full curriculum
        enable_curriculum=True,
        initial_sequence_length=4,
        final_sequence_length=32,
        curriculum_warmup_steps=20000,
        
        # Advanced self-forcing
        initial_self_forcing_ratio=0.0,
        final_self_forcing_ratio=0.9,
        self_forcing_warmup_steps=10000,
        
        # Flow matching
        enable_flow_matching=True,
        num_flow_steps=4,
        flow_matching_weight=1.0,
        
        # Memory management
        max_memory_gb=32.0
    )


def get_production_config() -> DriveDiTConfig:
    """Get production configuration optimized for large-scale training."""
    return DriveDiTConfig(
        # Production model
        model_dim=1024,
        num_layers=32,
        num_heads=16,
        sequence_length=64,
        image_size=512,
        
        # Large-scale training
        batch_size=16,
        max_steps=1000000,
        learning_rate=1e-4,
        
        # Select components for production
        enabled_components=[
            ComponentType.CONTROL,
            ComponentType.MEMORY,
            ComponentType.FLOW_MATCHING
        ],
        
        # Production curriculum
        enable_curriculum=True,
        initial_sequence_length=8,
        final_sequence_length=64,
        curriculum_warmup_steps=50000,
        
        # Production self-forcing
        initial_self_forcing_ratio=0.1,
        final_self_forcing_ratio=0.95,
        self_forcing_warmup_steps=25000,
        
        # Production flow matching
        enable_flow_matching=True,
        num_flow_steps=4,
        
        # Distributed training
        distributed=True,
        
        # Large memory
        max_memory_gb=128.0,
        
        # Production data pipeline
        video_chunk_duration=600,  # 10 minutes
        min_quality_score=0.8
    )


def get_video_only_config() -> DriveDiTConfig:
    """Get configuration for video-only training (no control signals)."""
    return DriveDiTConfig(
        # Video-focused model
        model_dim=512,
        num_layers=16,
        num_heads=8,
        sequence_length=24,
        
        # No control components
        enabled_components=[
            ComponentType.MEMORY,
            ComponentType.JEPA,
            ComponentType.FLOW_MATCHING
        ],
        
        # Video-focused training
        batch_size=6,
        max_steps=150000,
        
        # Strong temporal modeling
        temporal_consistency_weight=0.5,
        jepa_weight=0.3,
        
        # Flow matching emphasis
        enable_flow_matching=True,
        flow_matching_weight=2.0
    )


def get_flow_matching_config() -> DriveDiTConfig:
    """Get configuration focused on flow matching experiments."""
    return DriveDiTConfig(
        # Moderate model size
        model_dim=512,
        num_layers=12,
        num_heads=8,
        
        # Flow matching focus
        enabled_components=[ComponentType.FLOW_MATCHING],
        enable_flow_matching=True,
        flow_matching_weight=5.0,
        num_flow_steps=8,
        
        # Simpler curriculum for flow matching
        enable_curriculum=True,
        initial_sequence_length=2,
        final_sequence_length=12,
        
        # Flow matching specific training
        batch_size=8,
        learning_rate=2e-4,
        max_steps=50000
    )


# Configuration registry for easy access
CONFIG_REGISTRY = {
    'minimal': get_minimal_config,
    'research': get_research_config,
    'production': get_production_config,
    'video_only': get_video_only_config,
    'flow_matching': get_flow_matching_config
}


def get_config(config_name: str) -> DriveDiTConfig:
    """Get configuration by name."""
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    
    return CONFIG_REGISTRY[config_name]()


def list_configs() -> List[str]:
    """List available configuration names."""
    return list(CONFIG_REGISTRY.keys())


if __name__ == "__main__":
    # Test configurations
    print("Testing DriveDiT configurations...")
    
    for config_name in list_configs():
        print(f"\n=== {config_name.upper()} CONFIG ===")
        config = get_config(config_name)
        
        print(f"Model: {config.model_dim}d, {config.num_layers}L, {config.num_heads}H")
        print(f"Training: {config.batch_size}bs, {config.max_steps} steps")
        print(f"Components: {[c.value for c in config.enabled_components]}")
        print(f"Curriculum: {config.enable_curriculum}")
        print(f"Flow Matching: {config.enable_flow_matching}")
        print(f"Self-forcing: {config.initial_self_forcing_ratio} -> {config.final_self_forcing_ratio}")
        print(f"Memory: {config.max_memory_gb}GB")
    
    # Test component enabling/disabling
    print("\n=== COMPONENT MANAGEMENT ===")
    config = get_minimal_config()
    print(f"Initial components: {[c.value for c in config.enabled_components]}")
    
    config.enable_component(ComponentType.DEPTH)
    print(f"After enabling depth: {[c.value for c in config.enabled_components]}")
    
    config.disable_component(ComponentType.CONTROL)
    print(f"After disabling control: {[c.value for c in config.enabled_components]}")
    
    print(f"Is JEPA enabled? {config.is_component_enabled(ComponentType.JEPA)}")
    print(f"Is MEMORY enabled? {config.is_component_enabled(ComponentType.MEMORY)}")
    
    print("\nConfiguration system test completed successfully!")