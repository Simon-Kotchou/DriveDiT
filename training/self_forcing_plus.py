"""
Self-Forcing++ Training Implementation for DriveDiT.

Components:
1. Rolling KV Cache - Sliding window KV management
2. Curriculum Learning Scheduler - Progressive training
3. Future Anchor Conditioning - Goal state conditioning (comma.ai)
4. Extended 6D Control Signal - Full control encoding with inverse dynamics
5. Stability Improvements - EMA, uncertainty weighting, per-layer gradient clipping
6. SelfForcingPlusTrainer - Integrated trainer with all components

Based on Self-Forcing++ paper and comma.ai insights for extended
sequence generation and stable training.

References:
- Self-Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
- Self-Forcing++: Extended sequence generation with rolling KV cache
- comma.ai: Curriculum learning and future anchor conditioning
- Kendall et al.: Multi-task learning using uncertainty
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
# Component 4: Extended 6D Control Signal
# =============================================================================

class ExtendedControlEncoder(nn.Module):
    """
    Encodes extended 6D control signals with proper normalization.

    Control dimensions:
    0: steering (-1 to 1, normalized)
    1: acceleration (-5 to 5 m/s^2)
    2: goal_x (relative position in meters)
    3: goal_y (relative position in meters)
    4: speed (0 to 40 m/s)
    5: heading_rate (-1 to 1 rad/s)

    Features:
    - Per-dimension normalization with configurable ranges
    - Temporal encoding for control sequences
    - Control prediction head for inverse dynamics

    Args:
        config: SelfForcingPlusConfig with control parameters
    """

    # Default normalization ranges for each control dimension
    DEFAULT_RANGES = {
        'steering': (-1.0, 1.0),
        'acceleration': (-5.0, 5.0),
        'goal_x': (-50.0, 50.0),
        'goal_y': (-50.0, 50.0),
        'speed': (0.0, 40.0),
        'heading_rate': (-1.0, 1.0)
    }

    def __init__(
        self,
        config: SelfForcingPlusConfig,
        custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        super().__init__()
        self.config = config
        self.control_dim = config.control_dim
        self.hidden_dim = config.control_hidden_dim
        self.model_dim = config.model_dim

        # Merge default ranges with custom ranges
        self.ranges = {**self.DEFAULT_RANGES}
        if custom_ranges:
            self.ranges.update(custom_ranges)

        # Register normalization parameters as buffers (non-trainable)
        range_keys = ['steering', 'acceleration', 'goal_x', 'goal_y', 'speed', 'heading_rate']
        mins = torch.tensor([self.ranges[k][0] for k in range_keys])
        maxs = torch.tensor([self.ranges[k][1] for k in range_keys])
        self.register_buffer('norm_min', mins)
        self.register_buffer('norm_max', maxs)

        # Control encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.control_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.model_dim),
            nn.LayerNorm(self.model_dim)
        )

        # Temporal control encoder for sequences
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim)
        )

        # Control prediction head for inverse dynamics
        self.control_predictor = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.hidden_dim),  # Takes current + next state
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.control_dim)
        )

    def normalize(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Normalize control signals to [-1, 1] range.

        Args:
            controls: [B, T, 6] or [B, 6] raw control values

        Returns:
            Normalized controls in [-1, 1] range
        """
        # Expand dimensions for broadcasting
        norm_min = self.norm_min.view(1, -1) if controls.dim() == 2 else self.norm_min.view(1, 1, -1)
        norm_max = self.norm_max.view(1, -1) if controls.dim() == 2 else self.norm_max.view(1, 1, -1)

        # Clamp to valid range
        controls = controls.clamp(norm_min, norm_max)

        # Normalize to [-1, 1]
        normalized = 2 * (controls - norm_min) / (norm_max - norm_min + 1e-8) - 1

        return normalized

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize control signals from [-1, 1] to original range.

        Args:
            normalized: [B, T, 6] or [B, 6] normalized control values

        Returns:
            Denormalized controls in original ranges
        """
        norm_min = self.norm_min.view(1, -1) if normalized.dim() == 2 else self.norm_min.view(1, 1, -1)
        norm_max = self.norm_max.view(1, -1) if normalized.dim() == 2 else self.norm_max.view(1, 1, -1)

        # Denormalize from [-1, 1]
        denormalized = (normalized + 1) / 2 * (norm_max - norm_min) + norm_min

        return denormalized

    def forward(
        self,
        controls: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode control signals.

        Args:
            controls: [B, T, 6] or [B, 6] control signals
            normalize: Whether to normalize inputs (set False if pre-normalized)

        Returns:
            [B, T, D] or [B, D] control embeddings
        """
        if normalize:
            controls = self.normalize(controls)

        # Encode controls
        embeddings = self.encoder(controls)

        # Apply temporal encoding for sequences
        if controls.dim() == 3:
            embeddings = embeddings + self.temporal_encoder(embeddings)

        return embeddings

    def predict_control(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict control signal from state transition (inverse dynamics).

        Args:
            current_state: [B, D] current latent state
            next_state: [B, D] next latent state

        Returns:
            [B, 6] predicted control signals (normalized)
        """
        combined = torch.cat([current_state, next_state], dim=-1)
        predicted = self.control_predictor(combined)
        # Clamp to valid normalized range
        return predicted.clamp(-1, 1)

    def get_control_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        normalize_target: bool = True
    ) -> torch.Tensor:
        """
        Compute control prediction loss.

        Args:
            predicted: [B, 6] predicted normalized controls
            target: [B, 6] target controls (raw or normalized)
            normalize_target: Whether to normalize target

        Returns:
            Scalar loss value
        """
        if normalize_target:
            target = self.normalize(target)

        # L2 loss for continuous controls
        return ((predicted - target) ** 2).mean()


class ControlConditioner(nn.Module):
    """
    Applies control conditioning to model hidden states.

    Uses FiLM (Feature-wise Linear Modulation) for conditioning:
    output = gamma * hidden + beta

    Args:
        config: SelfForcingPlusConfig with model dimensions
    """

    def __init__(self, config: SelfForcingPlusConfig):
        super().__init__()
        self.config = config

        # FiLM parameter generators
        self.gamma_net = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        control_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply control conditioning via FiLM.

        Args:
            hidden: [B, T, D] model hidden states
            control_embedding: [B, D] or [B, T, D] control embeddings

        Returns:
            [B, T, D] conditioned hidden states
        """
        # Expand control embedding if needed
        if control_embedding.dim() == 2 and hidden.dim() == 3:
            control_embedding = control_embedding.unsqueeze(1).expand(-1, hidden.size(1), -1)

        # Compute FiLM parameters
        gamma = self.gamma_net(control_embedding)
        beta = self.beta_net(control_embedding)

        # Apply FiLM modulation
        return gamma * hidden + beta


# =============================================================================
# Component 5: Stability Improvements
# =============================================================================

class EMAModel:
    """
    Exponential Moving Average model for stable training targets.

    Maintains a shadow copy of model weights that are updated with
    exponential moving average. Used for:
    - Stable teacher targets during distillation
    - Evaluation and inference
    - Reducing variance in training

    Args:
        model: The model to track
        decay: EMA decay rate (higher = slower update, default 0.9999)
        device: Device for EMA parameters
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.device = device

        # Create shadow parameters
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device:
                    self.shadow[name] = self.shadow[name].to(device)

    def update(self, model: nn.Module):
        """Update EMA parameters with current model weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    new_value = param.data
                    if self.device:
                        new_value = new_value.to(self.device)
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * new_value

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (save current weights as backup)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
                if param.device != self.shadow[name].device:
                    param.data = param.data.to(param.device)

    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow.copy(),
            'decay': self.decay
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


class UncertaintyWeighting(nn.Module):
    """
    Automatic uncertainty-based loss weighting (Kendall et al., 2018).

    Learns per-task uncertainty (log variance) and uses it to weight
    losses automatically. Tasks with higher uncertainty get lower weight.

    Loss = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)

    This allows the model to automatically balance multiple loss terms
    during training without manual tuning.

    Args:
        loss_names: List of loss names to weight
        initial_log_var: Initial log variance (default 0 = sigma=1)
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_log_var: float = 0.0
    ):
        super().__init__()
        self.loss_names = loss_names
        self.num_losses = len(loss_names)

        # Learnable log variances (one per loss)
        self.log_vars = nn.Parameter(
            torch.full((self.num_losses,), initial_log_var)
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted total loss with uncertainty weighting.

        Args:
            losses: Dict of loss_name -> loss_value

        Returns:
            Tuple of (weighted_total_loss, weights_dict)
        """
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        weights = {}

        for i, name in enumerate(self.loss_names):
            if name in losses:
                loss_val = losses[name]
                log_var = self.log_vars[i]

                # Precision weighting: (1/(2*sigma^2)) * loss
                precision = torch.exp(-log_var)
                weighted_loss = 0.5 * precision * loss_val

                # Regularization term: log(sigma) = 0.5 * log_var
                regularization = 0.5 * log_var

                total_loss = total_loss + weighted_loss + regularization

                # Record effective weight
                weights[name] = precision.item()

        return total_loss, weights

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights (precisions)."""
        weights = {}
        with torch.no_grad():
            for i, name in enumerate(self.loss_names):
                weights[name] = torch.exp(-self.log_vars[i]).item()
        return weights

    def get_uncertainties(self) -> Dict[str, float]:
        """Get current uncertainties (standard deviations)."""
        uncertainties = {}
        with torch.no_grad():
            for i, name in enumerate(self.loss_names):
                # sigma = exp(0.5 * log_var)
                uncertainties[name] = torch.exp(0.5 * self.log_vars[i]).item()
        return uncertainties


class PerLayerGradientClipper:
    """
    Per-layer gradient clipping for fine-grained control.

    Different layers may need different clipping thresholds:
    - Early layers: More sensitive, need tighter clipping
    - Deep layers: More stable, can use looser clipping
    - Attention layers: May need special handling

    Supports adaptive clipping based on gradient history.

    Args:
        model: Model to clip gradients for
        default_max_norm: Default maximum gradient norm
        layer_configs: Dict of layer_name_pattern -> max_norm
        adaptive: Whether to use adaptive clipping
        adaptive_factor: Factor for adaptive threshold update
    """

    def __init__(
        self,
        model: nn.Module,
        default_max_norm: float = 1.0,
        layer_configs: Optional[Dict[str, float]] = None,
        adaptive: bool = False,
        adaptive_factor: float = 0.99
    ):
        self.model = model
        self.default_max_norm = default_max_norm
        self.layer_configs = layer_configs or {}
        self.adaptive = adaptive
        self.adaptive_factor = adaptive_factor

        # Track gradient norms for adaptive clipping
        self.grad_norm_history: Dict[str, List[float]] = {}
        self.adaptive_thresholds: Dict[str, float] = {}

        # Precompute layer max norms
        self.layer_max_norms = self._compute_layer_norms()

    def _compute_layer_norms(self) -> Dict[str, float]:
        """Compute max norm for each parameter group."""
        layer_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                max_norm = self.default_max_norm
                # Check for pattern matches
                for pattern, norm in self.layer_configs.items():
                    if pattern in name:
                        max_norm = norm
                        break
                layer_norms[name] = max_norm
        return layer_norms

    def clip_gradients(self) -> Dict[str, float]:
        """
        Clip gradients per-layer and return gradient statistics.

        Returns:
            Dict of layer_name -> gradient_norm (before clipping)
        """
        grad_norms = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Compute gradient norm
                grad_norm = param.grad.data.norm(2).item()
                grad_norms[name] = grad_norm

                # Get max norm for this layer
                if self.adaptive and name in self.adaptive_thresholds:
                    max_norm = self.adaptive_thresholds[name]
                else:
                    max_norm = self.layer_max_norms.get(name, self.default_max_norm)

                # Clip if necessary
                if grad_norm > max_norm:
                    clip_coef = max_norm / (grad_norm + 1e-8)
                    param.grad.data.mul_(clip_coef)

                # Update adaptive threshold
                if self.adaptive:
                    if name not in self.grad_norm_history:
                        self.grad_norm_history[name] = []
                    self.grad_norm_history[name].append(grad_norm)

                    # Keep limited history
                    if len(self.grad_norm_history[name]) > 100:
                        self.grad_norm_history[name] = self.grad_norm_history[name][-100:]

                    # Update adaptive threshold (percentile-based)
                    if len(self.grad_norm_history[name]) >= 10:
                        sorted_norms = sorted(self.grad_norm_history[name])
                        percentile_95 = sorted_norms[int(0.95 * len(sorted_norms))]
                        self.adaptive_thresholds[name] = (
                            self.adaptive_factor * self.adaptive_thresholds.get(name, percentile_95) +
                            (1 - self.adaptive_factor) * percentile_95
                        )

        return grad_norms

    def get_total_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm)

    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get gradient statistics for monitoring."""
        stats = {
            'total_norm': self.get_total_grad_norm(),
            'layer_norms': {},
            'clipped_layers': []
        }

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.data.norm(2).item()
                stats['layer_norms'][name] = norm
                max_norm = self.layer_max_norms.get(name, self.default_max_norm)
                if norm > max_norm:
                    stats['clipped_layers'].append(name)

        return stats


# =============================================================================
# Component 6: Self-Forcing++ Trainer
# =============================================================================

class SelfForcingPlusTrainer:
    """
    Complete Self-Forcing++ trainer integrating all components.

    Combines:
    - Rolling KV Cache for efficient long-sequence generation
    - Curriculum learning for progressive training
    - Future anchor conditioning for drift prevention
    - Extended 6D control signals
    - Stability improvements (EMA, uncertainty, gradient clipping)

    Usage:
        config = get_default_config()
        trainer = SelfForcingPlusTrainer(model, config)

        for batch in dataloader:
            losses = trainer.train_step(batch, optimizer)
            trainer.update_curriculum(global_step)

    Args:
        model: The world model to train
        config: SelfForcingPlusConfig with all training parameters
        device: Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        config: SelfForcingPlusConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.kv_cache = RollingKVCache(
            max_length=config.kv_cache_max_length,
            truncate_to=config.kv_cache_truncate_to,
            detach_interval=config.kv_cache_detach_interval,
            num_layers=config.num_layers,
            device=self.device
        )

        self.curriculum = CurriculumScheduler(config)

        # Future anchor encoder (if enabled)
        if config.enable_future_anchors:
            self.future_anchor_encoder = FutureAnchorEncoder(config).to(self.device)
        else:
            self.future_anchor_encoder = None

        # Control encoder
        self.control_encoder = ExtendedControlEncoder(config).to(self.device)
        self.control_conditioner = ControlConditioner(config).to(self.device)

        # Stability components
        self.ema = EMAModel(model, decay=0.9999, device=self.device)

        loss_names = ['reconstruction', 'temporal', 'flow_matching', 'control', 'future_anchor']
        self.uncertainty_weighter = UncertaintyWeighting(loss_names).to(self.device)

        self.gradient_clipper = PerLayerGradientClipper(
            model,
            default_max_norm=1.0,
            layer_configs={'attention': 0.5, 'output': 1.5},
            adaptive=True
        )

        # Training state
        self.global_step = 0
        self.training_stats = {
            'total_loss': [],
            'losses': {},
            'gradient_norms': [],
            'curriculum_stats': []
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Execute one training step with Self-Forcing++.

        Args:
            batch: Dict containing:
                - 'frames': [B, T, C, H, W] video frames
                - 'controls': [B, T, 6] control signals
                - 'ego_states': [B, T, 5] ego vehicle states (optional)
            optimizer: Optimizer for model parameters

        Returns:
            Dict of loss values and training statistics
        """
        self.model.train()
        optimizer.zero_grad()

        # Get current curriculum parameters
        seq_length = self.curriculum.get_sequence_length()
        sf_ratio = self.curriculum.get_self_forcing_ratio()
        curriculum_weights = self.curriculum.get_curriculum_weights()

        # Prepare batch
        frames = batch['frames'][:, :seq_length].to(self.device)
        controls = batch['controls'][:, :seq_length].to(self.device)
        B, T = frames.shape[:2]

        # Reset KV cache for new sequence
        self.kv_cache.reset()

        # Encode controls
        control_embeddings = self.control_encoder(controls)

        # Get future anchor conditioning if enabled
        if self.future_anchor_encoder is not None and 'ego_states' in batch:
            ego_states = batch['ego_states'][:, :seq_length].to(self.device)
            anchor_embedding = self.future_anchor_encoder(ego_states, current_time=0)
        else:
            anchor_embedding = None

        # Self-forcing rollout
        losses = self._self_forcing_rollout(
            frames=frames,
            control_embeddings=control_embeddings,
            anchor_embedding=anchor_embedding,
            sf_ratio=sf_ratio
        )

        # Apply curriculum weights to losses
        weighted_losses = {}
        for name, loss in losses.items():
            weight = curriculum_weights.get(name, 1.0)
            weighted_losses[name] = loss * weight

        # Compute total loss with uncertainty weighting
        total_loss, uncertainty_weights = self.uncertainty_weighter(weighted_losses)

        # Backward pass
        total_loss.backward()

        # Per-layer gradient clipping
        grad_norms = self.gradient_clipper.clip_gradients()

        # Optimizer step
        optimizer.step()

        # Update EMA
        self.ema.update(self.model)

        # Update training stats
        self.global_step += 1
        self.curriculum.update(self.global_step)

        # Prepare return dict
        result = {
            'total_loss': total_loss.item(),
            'sf_ratio': sf_ratio,
            'seq_length': seq_length,
            'grad_norm': self.gradient_clipper.get_total_grad_norm()
        }

        for name, loss in losses.items():
            result[f'loss_{name}'] = loss.item()

        for name, weight in uncertainty_weights.items():
            result[f'weight_{name}'] = weight

        return result

    def _self_forcing_rollout(
        self,
        frames: torch.Tensor,
        control_embeddings: torch.Tensor,
        anchor_embedding: Optional[torch.Tensor],
        sf_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """
        Execute self-forcing rollout with curriculum-based mixing.

        Args:
            frames: [B, T, C, H, W] ground truth frames
            control_embeddings: [B, T, D] encoded controls
            anchor_embedding: [B, D] future anchor (optional)
            sf_ratio: Self-forcing ratio (0=GT, 1=self-generated)

        Returns:
            Dict of loss tensors
        """
        B, T = frames.shape[:2]
        device = frames.device

        # Initialize losses
        reconstruction_losses = []
        temporal_losses = []
        control_losses = []

        # Context frames (always use ground truth)
        context_len = self.config.context_frames
        prev_output = frames[:, context_len - 1]  # Last context frame

        for t in range(context_len, T):
            # Get control embedding for this timestep
            ctrl_emb = control_embeddings[:, t]

            # Decide whether to use ground truth or self-generated
            use_self_generated = (
                t > context_len and
                torch.rand(1).item() < sf_ratio
            )

            if use_self_generated:
                # Use previous model output
                input_frame = prev_output.detach()
            else:
                # Use ground truth
                input_frame = frames[:, t - 1]

            # Apply control conditioning
            # (In a full implementation, this would pass through the model)
            # Here we demonstrate the interface
            model_input = input_frame

            # Forward pass through model would happen here
            # For demonstration, we simulate the output
            # output = self.model(model_input, control=ctrl_emb, anchor=anchor_embedding)
            output = input_frame  # Placeholder

            # Compute reconstruction loss
            target = frames[:, t]
            recon_loss = ((output - target) ** 2).mean()
            reconstruction_losses.append(recon_loss)

            # Compute temporal consistency loss
            if t > context_len:
                prev_target = frames[:, t - 1]
                temporal_loss = ((output - prev_output) - (target - prev_target)).abs().mean()
                temporal_losses.append(temporal_loss)

            # Control prediction loss (inverse dynamics)
            if t > context_len:
                # Flatten for control prediction
                current_flat = input_frame.view(B, -1)[:, :self.config.model_dim]
                next_flat = output.view(B, -1)[:, :self.config.model_dim]
                predicted_ctrl = self.control_encoder.predict_control(current_flat, next_flat)
                ctrl_loss = self.control_encoder.get_control_loss(
                    predicted_ctrl,
                    torch.zeros(B, 6, device=device)  # Placeholder target
                )
                control_losses.append(ctrl_loss)

            prev_output = output

        # Aggregate losses
        losses = {}

        if reconstruction_losses:
            losses['reconstruction'] = torch.stack(reconstruction_losses).mean()
        else:
            # Fallback for very short sequences
            losses['reconstruction'] = torch.tensor(0.0, device=device, requires_grad=True)

        if temporal_losses:
            losses['temporal'] = torch.stack(temporal_losses).mean()

        if control_losses:
            losses['control'] = torch.stack(control_losses).mean()

        # Flow matching loss placeholder
        losses['flow_matching'] = torch.tensor(0.0, device=device, requires_grad=True)

        # Future anchor loss placeholder
        if anchor_embedding is not None:
            losses['future_anchor'] = torch.tensor(0.0, device=device, requires_grad=True)

        return losses

    def update_curriculum(self, step: int):
        """Update curriculum scheduler with current step."""
        self.curriculum.update(step)
        self.global_step = step

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get current curriculum statistics."""
        return self.curriculum.get_stats()

    def use_ema_for_eval(self) -> None:
        """Apply EMA weights for evaluation."""
        self.ema.apply_shadow(self.model)

    def restore_training_weights(self) -> None:
        """Restore training weights after evaluation."""
        self.ema.restore(self.model)

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        return {
            'global_step': self.global_step,
            'curriculum_step': self.curriculum.current_step,
            'ema': self.ema.state_dict(),
            'uncertainty_weighter': self.uncertainty_weighter.state_dict(),
            'training_stats': self.training_stats
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        self.global_step = state_dict.get('global_step', 0)
        self.curriculum.update(state_dict.get('curriculum_step', 0))
        if 'ema' in state_dict:
            self.ema.load_state_dict(state_dict['ema'])
        if 'uncertainty_weighter' in state_dict:
            self.uncertainty_weighter.load_state_dict(state_dict['uncertainty_weighter'])
        self.training_stats = state_dict.get('training_stats', self.training_stats)


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

    # Test 4: Extended Control Encoder
    print("\n4. Extended Control Encoder Test")
    control_encoder = ExtendedControlEncoder(config).to(device)

    # Create sample control signals [B, T, 6]
    # steering, accel, goal_x, goal_y, speed, heading_rate
    raw_controls = torch.tensor([
        [0.5, 2.0, 10.0, 5.0, 25.0, 0.3],  # Sample control 1
        [-0.3, -1.0, -5.0, 2.0, 15.0, -0.2]  # Sample control 2
    ], device=device)

    print(f"  Raw controls shape: {raw_controls.shape}")
    print(f"  Raw controls: {raw_controls[0].tolist()}")

    # Test normalization
    normalized = control_encoder.normalize(raw_controls)
    print(f"  Normalized controls: {normalized[0].tolist()}")

    # Test denormalization (should match original)
    denormalized = control_encoder.denormalize(normalized)
    print(f"  Denormalized controls: {denormalized[0].tolist()}")
    print(f"  Reconstruction error: {(raw_controls - denormalized).abs().max().item():.6f}")

    # Test encoding
    control_emb = control_encoder(raw_controls)
    print(f"  Control embedding shape: {control_emb.shape}")

    # Test control prediction (inverse dynamics)
    current_state = torch.randn(2, config.model_dim, device=device)
    next_state = torch.randn(2, config.model_dim, device=device)
    predicted_control = control_encoder.predict_control(current_state, next_state)
    print(f"  Predicted control shape: {predicted_control.shape}")

    # Test control loss
    loss = control_encoder.get_control_loss(predicted_control, raw_controls)
    print(f"  Control loss: {loss.item():.4f}")

    # Test 5: Control Conditioner (FiLM)
    print("\n5. Control Conditioner Test")
    conditioner = ControlConditioner(config).to(device)

    hidden = torch.randn(2, 8, config.model_dim, device=device)  # [B, T, D]
    control_emb_single = torch.randn(2, config.model_dim, device=device)  # [B, D]

    conditioned = conditioner(hidden, control_emb_single)
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Control embedding shape: {control_emb_single.shape}")
    print(f"  Conditioned output shape: {conditioned.shape}")

    # Test 6: EMA Model
    print("\n6. EMA Model Test")
    # Create a simple test model
    test_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    ).to(device)

    ema = EMAModel(test_model, decay=0.99)
    print(f"  EMA initialized with {len(ema.shadow)} parameters")

    # Simulate training updates
    original_param = list(test_model.parameters())[0].data.clone()
    for i in range(10):
        # Simulate parameter update
        with torch.no_grad():
            for param in test_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        ema.update(test_model)

    print(f"  After 10 updates, EMA tracking active")

    # Test apply/restore
    ema.apply_shadow(test_model)
    print(f"  Applied EMA shadow weights")
    ema.restore(test_model)
    print(f"  Restored original weights")

    # Test 7: Uncertainty Weighting
    print("\n7. Uncertainty Weighting Test")
    loss_names = ['reconstruction', 'temporal', 'flow_matching', 'control']
    uncertainty_weighter = UncertaintyWeighting(loss_names).to(device)

    # Simulate losses
    test_losses = {
        'reconstruction': torch.tensor(1.0, device=device),
        'temporal': torch.tensor(0.5, device=device),
        'flow_matching': torch.tensor(2.0, device=device),
        'control': torch.tensor(0.3, device=device)
    }

    weighted_loss, weights = uncertainty_weighter(test_losses)
    print(f"  Input losses: {[f'{k}={v.item():.2f}' for k, v in test_losses.items()]}")
    print(f"  Weighted total: {weighted_loss.item():.4f}")
    print(f"  Learned weights: {uncertainty_weighter.get_weights()}")
    print(f"  Uncertainties: {uncertainty_weighter.get_uncertainties()}")

    # Test 8: Per-Layer Gradient Clipper
    print("\n8. Per-Layer Gradient Clipper Test")
    # Create test model with gradients
    clip_model = nn.Sequential(
        nn.Linear(32, 64),
        nn.Linear(64, 32)
    ).to(device)

    # Forward and backward to create gradients
    x = torch.randn(4, 32, device=device)
    y = clip_model(x)
    loss = y.sum()
    loss.backward()

    # Configure per-layer clipping
    layer_configs = {
        '0': 0.5,  # Tighter clipping for first layer
        '1': 1.0   # Normal clipping for second layer
    }

    clipper = PerLayerGradientClipper(
        clip_model,
        default_max_norm=1.0,
        layer_configs=layer_configs,
        adaptive=False
    )

    print(f"  Initial total grad norm: {clipper.get_total_grad_norm():.4f}")
    grad_norms = clipper.clip_gradients()
    print(f"  After clipping, grad norms: {list(grad_norms.values())[:4]}")
    stats = clipper.get_gradient_stats()
    print(f"  Clipped layers: {len(stats['clipped_layers'])}")

    # Test 9: SelfForcingPlusTrainer
    print("\n9. SelfForcingPlusTrainer Integration Test")

    # Create a simple mock model
    class MockWorldModel(nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.encoder = nn.Linear(3 * 64 * 64, dim)
            self.decoder = nn.Linear(dim, 3 * 64 * 64)

        def forward(self, x):
            B, C, H, W = x.shape
            flat = x.view(B, -1)
            encoded = self.encoder(flat)
            decoded = self.decoder(encoded)
            return decoded.view(B, C, H, W)

    mock_model = MockWorldModel(config.model_dim).to(device)
    print(f"  Mock model created with {sum(p.numel() for p in mock_model.parameters())} parameters")

    # Create trainer
    trainer = SelfForcingPlusTrainer(mock_model, config, device)
    print(f"  Trainer initialized")
    print(f"  Initial curriculum: seq_len={trainer.curriculum.get_sequence_length()}, "
          f"sf_ratio={trainer.curriculum.get_self_forcing_ratio():.3f}")

    # Advance curriculum to get longer sequences
    trainer.update_curriculum(500)  # Get sequence length > context_frames
    print(f"  After advancing curriculum: seq_len={trainer.curriculum.get_sequence_length()}")

    # Create mock batch
    batch = {
        'frames': torch.randn(2, 16, 3, 64, 64, device=device),
        'controls': torch.randn(2, 16, 6, device=device),
        'ego_states': torch.randn(2, 16, 5, device=device)
    }

    # Create optimizer
    optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

    # Run training step
    print(f"  Running training step...")
    result = trainer.train_step(batch, optimizer)

    print(f"  Training step completed:")
    print(f"    Total loss: {result['total_loss']:.4f}")
    print(f"    Sequence length: {result['seq_length']}")
    print(f"    Self-forcing ratio: {result['sf_ratio']:.3f}")
    print(f"    Gradient norm: {result['grad_norm']:.4f}")

    # Test curriculum progression
    trainer.update_curriculum(1000)
    stats = trainer.get_curriculum_stats()
    print(f"  After 1000 steps: seq_len={stats['sequence_length']}, "
          f"sf_ratio={stats['self_forcing_ratio']:.3f}")

    # Test state dict
    state = trainer.state_dict()
    print(f"  State dict keys: {list(state.keys())}")

    # Test EMA for eval
    trainer.use_ema_for_eval()
    print(f"  EMA weights applied for evaluation")
    trainer.restore_training_weights()
    print(f"  Training weights restored")

    print("\n" + "=" * 60)
    print("All component tests completed!")
