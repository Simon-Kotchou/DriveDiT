"""
Self-Forcing++ Training Implementation for DriveDiT.

Components:
1. Rolling KV Cache - Sliding window KV management
2. Curriculum Learning Scheduler - Progressive training
3. Future Anchor Conditioning - Goal state conditioning (comma.ai)

Based on Self-Forcing++ paper and comma.ai insights for extended
sequence generation and stable training.

References:
- Self-Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
- Self-Forcing++: Extended sequence generation with rolling KV cache
- comma.ai: Curriculum learning and future anchor conditioning
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field


class RollingKVCache:
    """
    Rolling KV cache for efficient long-sequence generation.

    Unlike discrete anchor frames, this cache slides over the sequence,
    maintaining a fixed-size window of past KV pairs.

    Key features:
    - Automatic truncation when exceeding max length
    - Gradient detachment at configurable intervals
    - Memory-efficient storage with optional CPU offloading

    Args:
        max_length: Maximum cache length before truncation
        truncate_to: Length to truncate to when exceeding max
        detach_interval: Detach gradients every N steps
        num_layers: Number of transformer layers
        device: Device for cache storage
    """

    def __init__(
        self,
        max_length: int = 512,
        truncate_to: int = 256,
        detach_interval: int = 8,
        num_layers: int = 12,
        device: torch.device = None
    ):
        self.max_length = max_length
        self.truncate_to = truncate_to
        self.detach_interval = detach_interval
        self.num_layers = num_layers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cache storage: dict[layer_idx] -> {'k': tensor, 'v': tensor}
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # Tracking
        self.step_count = 0
        self.total_length = 0

    def reset(self):
        """Reset cache for new sequence."""
        self.cache = {}
        self.step_count = 0
        self.total_length = 0

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV pairs and return full KV for attention.

        Args:
            layer_idx: Index of transformer layer
            new_k: New key tensor [B, T_new, H, D]
            new_v: New value tensor [B, T_new, H, D]

        Returns:
            Tuple of (full_k, full_v) including cached history
        """
        self.step_count += 1

        # Detach gradients periodically to prevent accumulation
        should_detach = (self.step_count % self.detach_interval == 0)

        if layer_idx not in self.cache:
            # Initialize cache for this layer
            self.cache[layer_idx] = {
                'k': new_k.detach() if should_detach else new_k,
                'v': new_v.detach() if should_detach else new_v
            }
            self.total_length = new_k.size(1)
        else:
            # Concatenate with existing cache
            cached_k = self.cache[layer_idx]['k']
            cached_v = self.cache[layer_idx]['v']

            if should_detach:
                cached_k = cached_k.detach()
                cached_v = cached_v.detach()
                new_k = new_k.detach()
                new_v = new_v.detach()

            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)

            # Truncate if exceeding max length
            if full_k.size(1) > self.max_length:
                # Keep most recent entries
                truncate_from = full_k.size(1) - self.truncate_to
                full_k = full_k[:, truncate_from:].detach()
                full_v = full_v[:, truncate_from:].detach()

            self.cache[layer_idx] = {'k': full_k, 'v': full_v}
            self.total_length = full_k.size(1)

        return self.cache[layer_idx]['k'], self.cache[layer_idx]['v']

    def get_cache(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached KV for a specific layer."""
        return self.cache.get(layer_idx, None)

    def get_length(self) -> int:
        """Get current cache length."""
        return self.total_length

    def get_memory_usage(self) -> float:
        """Get memory usage in GB."""
        total_bytes = 0
        for layer_cache in self.cache.values():
            for tensor in layer_cache.values():
                total_bytes += tensor.element_size() * tensor.numel()
        return total_bytes / (1024 ** 3)


# =============================================================================
# Component 2: Curriculum Learning Configuration
# =============================================================================

@dataclass
class SelfForcingPlusConfig:
    """Configuration for Self-Forcing++ training."""

    # Model dimensions
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12

    # Sequence configuration
    initial_sequence_length: int = 8
    final_sequence_length: int = 64
    max_sequence_length: int = 128  # Maximum for very long rollouts
    context_frames: int = 4  # Number of initial context frames

    # Rolling KV Cache
    kv_cache_max_length: int = 512  # Maximum KV cache length before truncation
    kv_cache_truncate_to: int = 256  # Truncate to this length when exceeding max
    kv_cache_detach_interval: int = 8  # Detach gradients every N steps

    # Curriculum Learning
    curriculum_warmup_steps: int = 10000
    sequence_curriculum_steps: int = 20000  # Steps to grow from initial to final length

    # Self-Forcing Ratio Schedule
    initial_self_forcing_ratio: float = 0.0  # Start with all ground truth
    final_self_forcing_ratio: float = 1.0  # End with all self-generated
    self_forcing_warmup_steps: int = 15000
    self_forcing_schedule: str = "cosine"  # "linear", "cosine", "exponential"

    # Future Anchor Conditioning (comma.ai) - placeholder for next component
    enable_future_anchors: bool = True
    future_anchor_horizons: List[float] = field(default_factory=lambda: [2.0, 4.0, 6.0])
    future_anchor_dim: int = 64
    fps: float = 10.0

    # Control Signal - placeholder for next component
    control_dim: int = 6
    control_hidden_dim: int = 256

    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    mixed_precision: bool = True
    max_memory_gb: float = 16.0


# =============================================================================
# Component 2: Curriculum Learning Scheduler
# =============================================================================

class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive training.

    Based on comma.ai insights for stable world model training:
    - Sequence length: Gradually increases from short to long sequences
    - Self-forcing ratio: Progressively relies more on model predictions
    - Loss weights: Adjusts loss contributions over training

    Args:
        config: SelfForcingPlusConfig with curriculum parameters
    """

    def __init__(self, config: SelfForcingPlusConfig):
        self.config = config
        self.current_step = 0

    def update(self, step: int):
        """Update scheduler with current training step."""
        self.current_step = step

    def get_sequence_length(self) -> int:
        """Get current target sequence length."""
        if self.current_step >= self.config.sequence_curriculum_steps:
            return self.config.final_sequence_length

        progress = self.current_step / self.config.sequence_curriculum_steps

        # Use smooth step function for gradual transition
        smooth_progress = self._smooth_step(progress)

        length = int(
            self.config.initial_sequence_length +
            smooth_progress * (self.config.final_sequence_length - self.config.initial_sequence_length)
        )

        # Ensure within bounds
        return max(self.config.initial_sequence_length, min(length, self.config.final_sequence_length))

    def get_self_forcing_ratio(self) -> float:
        """
        Get current self-forcing ratio.

        Returns value in [initial_ratio, final_ratio] based on schedule.
        - 0.0 = use all ground truth frames
        - 1.0 = use all self-generated frames
        """
        if self.current_step >= self.config.self_forcing_warmup_steps:
            return self.config.final_self_forcing_ratio

        progress = self.current_step / self.config.self_forcing_warmup_steps

        if self.config.self_forcing_schedule == "linear":
            ratio = progress
        elif self.config.self_forcing_schedule == "cosine":
            # Smooth cosine schedule
            ratio = 0.5 * (1 - math.cos(math.pi * progress))
        elif self.config.self_forcing_schedule == "exponential":
            # Slower ramp-up initially
            ratio = 1 - math.exp(-3 * progress)
        else:
            ratio = progress

        return (
            self.config.initial_self_forcing_ratio +
            ratio * (self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio)
        )

    def get_curriculum_weights(self) -> Dict[str, float]:
        """
        Get curriculum-adjusted loss weights.

        Some losses are introduced gradually as training progresses.
        """
        progress = min(1.0, self.current_step / self.config.curriculum_warmup_steps)

        weights = {
            'reconstruction': 1.0,  # Always full weight
            'temporal_consistency': progress,  # Add temporal loss gradually
            'flow_matching': max(0.1, progress),  # Start with small weight
            'future_anchor': max(0, progress - 0.3) / 0.7 if progress > 0.3 else 0,  # After 30%
            'control_prediction': progress,
            'kv_regularization': 0.01
        }

        return weights

    def _smooth_step(self, x: float) -> float:
        """Smooth step function for gradual transitions (Hermite interpolation)."""
        x = max(0, min(1, x))
        return x * x * (3 - 2 * x)

    def get_stats(self) -> Dict[str, Any]:
        """Get current curriculum statistics."""
        return {
            'step': self.current_step,
            'sequence_length': self.get_sequence_length(),
            'self_forcing_ratio': self.get_self_forcing_ratio(),
            'curriculum_weights': self.get_curriculum_weights()
        }


# =============================================================================
# Component 3: Future Anchor Conditioning (comma.ai)
# =============================================================================

class FutureAnchorEncoder(nn.Module):
    """
    Encodes future goal states for conditioning.

    Based on comma.ai's future anchor methodology:
    - Prevents drift by conditioning on goal states
    - Multiple horizons (2s, 4s, 6s) for multi-scale planning
    - Includes position, heading, and velocity information

    This enables the model to "know where it's going" which prevents
    the common issue of trajectory drift in long autoregressive rollouts.

    Args:
        config: SelfForcingPlusConfig with anchor parameters
    """

    def __init__(self, config: SelfForcingPlusConfig):
        super().__init__()
        self.config = config
        self.horizons = config.future_anchor_horizons
        self.anchor_dim = config.future_anchor_dim

        # Each horizon encodes: [x, y, heading, speed, heading_rate]
        self.anchor_input_dim = 5

        # Per-horizon encoders
        self.horizon_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.anchor_input_dim, config.future_anchor_dim),
                nn.SiLU(),
                nn.Linear(config.future_anchor_dim, config.future_anchor_dim),
                nn.LayerNorm(config.future_anchor_dim)
            )
            for _ in self.horizons
        ])

        # Fusion layer for multi-horizon features
        self.fusion = nn.Sequential(
            nn.Linear(len(self.horizons) * config.future_anchor_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
            nn.LayerNorm(config.model_dim)
        )

    def forward(
        self,
        future_states: torch.Tensor,
        current_time: int = 0
    ) -> torch.Tensor:
        """
        Encode future anchor states.

        Args:
            future_states: [B, T, 5] tensor of [x, y, heading, speed, heading_rate]
            current_time: Current timestep in sequence

        Returns:
            [B, D] anchor embedding for conditioning
        """
        B, T, _ = future_states.shape
        device = future_states.device

        horizon_features = []

        for i, horizon_sec in enumerate(self.horizons):
            # Convert horizon to frame index
            horizon_frames = int(horizon_sec * self.config.fps)
            target_idx = min(current_time + horizon_frames, T - 1)

            # Extract future state at this horizon
            future_state = future_states[:, target_idx]  # [B, 5]

            # Compute relative state (relative to current position)
            current_state = future_states[:, min(current_time, T - 1)]
            relative_state = future_state - current_state

            # Encode
            horizon_feat = self.horizon_encoders[i](relative_state)
            horizon_features.append(horizon_feat)

        # Concatenate and fuse
        combined = torch.cat(horizon_features, dim=-1)  # [B, num_horizons * anchor_dim]
        anchor_embedding = self.fusion(combined)  # [B, model_dim]

        return anchor_embedding

    def get_anchor_indices(self, current_time: int, max_time: int) -> List[int]:
        """Get frame indices for each anchor horizon."""
        indices = []
        for horizon_sec in self.horizons:
            horizon_frames = int(horizon_sec * self.config.fps)
            target_idx = min(current_time + horizon_frames, max_time - 1)
            indices.append(target_idx)
        return indices


# =============================================================================
# Factory Functions
# =============================================================================

def get_default_config() -> SelfForcingPlusConfig:
    """Get default Self-Forcing++ configuration."""
    return SelfForcingPlusConfig()


def get_minimal_config() -> SelfForcingPlusConfig:
    """Get minimal configuration for testing/development."""
    return SelfForcingPlusConfig(
        initial_sequence_length=4,
        final_sequence_length=16,
        kv_cache_max_length=128,
        kv_cache_truncate_to=64,
        curriculum_warmup_steps=1000,
        sequence_curriculum_steps=2000,
        self_forcing_warmup_steps=2000
    )


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Self-Forcing++ Components Test")
    print("=" * 60)

    # Test 1: Rolling KV Cache
    print("\n1. Rolling KV Cache Test")
    cache = RollingKVCache(max_length=32, truncate_to=16, detach_interval=4)

    for i in range(10):
        k = torch.randn(2, 4, 8, 64)
        v = torch.randn(2, 4, 8, 64)
        full_k, full_v = cache.update(layer_idx=0, new_k=k, new_v=v)
        print(f"  Step {i}: cache length = {cache.get_length()}")

    print(f"  Memory usage: {cache.get_memory_usage():.6f} GB")

    # Test 2: Curriculum Scheduler
    print("\n2. Curriculum Scheduler Test")
    config = get_minimal_config()
    scheduler = CurriculumScheduler(config)

    test_steps = [0, 500, 1000, 1500, 2000]
    for step in test_steps:
        scheduler.update(step)
        stats = scheduler.get_stats()
        print(f"  Step {step}: seq_len={stats['sequence_length']}, "
              f"sf_ratio={stats['self_forcing_ratio']:.3f}")

    # Test 3: Future Anchor Encoder
    print("\n3. Future Anchor Encoder Test")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anchor_encoder = FutureAnchorEncoder(config).to(device)

    ego_states = torch.randn(2, 64, 5, device=device)  # [B, T, 5]
    anchor_emb = anchor_encoder(ego_states, current_time=0)
    print(f"  Ego states shape: {ego_states.shape}")
    print(f"  Anchor embedding shape: {anchor_emb.shape}")
    print(f"  Anchor indices for t=0: {anchor_encoder.get_anchor_indices(0, 64)}")

    print("\n" + "=" * 60)
    print("All component tests completed!")
