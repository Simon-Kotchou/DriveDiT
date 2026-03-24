"""
Unified training pipeline integrating all methodologies:
- Self-forcing training with curriculum learning
- Flow matching and distillation
- Modular components (control, JEPA, depth, memory)
- Large-scale distributed training capabilities
- REPA (Representation Alignment) with HASTE early-stopping
- C-JEPA (Contextual Joint Embedding Predictive Architecture) for object-level prediction
- MoE (Mixture of Experts) load balancing and monitoring

This replaces the scattered training approaches with a single, cohesive system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Protocol
from abc import ABC, abstractmethod
import math
import random
import time
import logging
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.losses import UnifiedLoss
from training.distributed import MemoryMonitor, CheckpointManager, DistributedManager
from config.config import DriveDiTConfig, ComponentType
from models.world_model import WorldModel


# =============================================================================
# Training Callbacks Protocol and Implementations
# =============================================================================

class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_train_step_end(
        self,
        trainer: 'UnifiedTrainer',
        step: int,
        losses: Dict[str, float],
        metrics: Dict[str, Any]
    ) -> None:
        """Called at the end of each training step."""
        ...

    def on_epoch_end(
        self,
        trainer: 'UnifiedTrainer',
        epoch: int,
        metrics: Dict[str, Any]
    ) -> None:
        """Called at the end of each epoch."""
        ...


class CallbackEvent(Enum):
    """Types of callback events."""
    REPA_DISABLED = "repa_disabled"
    EXPERT_IMBALANCE = "expert_imbalance"
    CURRICULUM_MILESTONE = "curriculum_milestone"
    VALIDATION_IMPROVEMENT = "validation_improvement"


@dataclass
class OnREPADisabledCallback:
    """Callback triggered when HASTE stops REPA training.

    This callback is fired when the representation alignment loss
    has converged sufficiently that continuing REPA training provides
    diminishing returns (HASTE = Halt Alignment STrategy Early).
    """
    on_disabled: Optional[Callable[[int, float, float], None]] = None

    def __call__(self, step: int, final_loss: float, threshold: float) -> None:
        """Execute callback when REPA is disabled.

        Args:
            step: Training step when REPA was disabled
            final_loss: Final REPA loss value when disabled
            threshold: Threshold that triggered the disable
        """
        if self.on_disabled is not None:
            self.on_disabled(step, final_loss, threshold)
        logging.info(
            f"REPA disabled at step {step}: "
            f"loss={final_loss:.6f} < threshold={threshold:.6f}"
        )


@dataclass
class OnExpertImbalanceCallback:
    """Callback triggered when MoE expert utilization is unbalanced.

    Monitors expert load and fires when some experts are significantly
    under or over-utilized, which can indicate training issues.
    """
    imbalance_threshold: float = 0.3
    on_imbalance: Optional[Callable[[int, Dict[str, float]], None]] = None

    def __call__(self, step: int, expert_loads: Dict[str, float]) -> bool:
        """Check for imbalance and execute callback if detected.

        Args:
            step: Current training step
            expert_loads: Dictionary mapping expert names to utilization ratios

        Returns:
            True if imbalance was detected
        """
        if not expert_loads:
            return False

        loads = list(expert_loads.values())
        max_load = max(loads)
        min_load = min(loads)

        imbalance = max_load - min_load

        if imbalance > self.imbalance_threshold:
            if self.on_imbalance is not None:
                self.on_imbalance(step, expert_loads)
            logging.warning(
                f"Expert imbalance detected at step {step}: "
                f"max={max_load:.3f}, min={min_load:.3f}, delta={imbalance:.3f}"
            )
            return True
        return False


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class REPAConfig:
    """Configuration for REPA (Representation Alignment) training.

    REPA aligns the model's latent representations with a pre-trained
    encoder (e.g., V-JEPA 2.1) to improve representation quality.
    """
    enabled: bool = False
    backbone_type: str = "vjepa_2.1"  # Pre-trained backbone for alignment
    alignment_dim: int = 768  # Dimension for alignment projections
    loss_weight: float = 0.5
    temperature: float = 0.07  # InfoNCE temperature
    projection_hidden_dim: int = 2048
    projection_output_dim: int = 256

    # HASTE (Halt Alignment STrategy Early) configuration
    haste_enabled: bool = True
    haste_warmup_steps: int = 5000  # Steps before HASTE can trigger
    haste_window_size: int = 1000  # Window for computing loss plateau
    haste_threshold: float = 0.01  # Loss improvement threshold
    haste_patience: int = 3  # Number of windows without improvement


@dataclass
class CJEPAConfig:
    """Configuration for C-JEPA (Contextual JEPA) object-level prediction.

    C-JEPA extends JEPA to perform object-level contrastive prediction,
    enabling better object permanence and tracking in world models.
    """
    enabled: bool = False
    num_object_queries: int = 64  # Number of object slots
    object_dim: int = 256
    loss_weight: float = 0.3
    contrastive_temperature: float = 0.1
    prediction_horizon: int = 4  # Frames ahead to predict
    use_slot_attention: bool = True
    slot_attention_iters: int = 3

    # Object-level masking
    mask_ratio: float = 0.75
    mask_strategy: str = "object"  # "object", "random", "structured"


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts monitoring and load balancing.

    MoE layers can suffer from load imbalance where some experts are
    used much more than others. This configuration controls monitoring
    and balancing strategies.
    """
    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2  # Number of experts per token
    load_balance_weight: float = 0.01  # Auxiliary loss weight
    router_z_loss_weight: float = 0.001  # Router z-loss for stability
    capacity_factor: float = 1.25  # Expert capacity factor

    # Monitoring
    log_expert_utilization: bool = True
    utilization_log_frequency: int = 100  # Steps between logs

    # Balancing strategies
    use_auxiliary_loss: bool = True
    use_entropy_regularization: bool = True


@dataclass
class ValidationConfig:
    """Configuration for validation and evaluation."""
    enabled: bool = True
    frequency: int = 1000  # Steps between validations
    num_batches: int = 10  # Batches per validation

    # Closed-loop evaluation
    closed_loop_enabled: bool = False
    closed_loop_horizon: int = 30  # Frames for closed-loop rollout

    # Physics metrics
    compute_physics_metrics: bool = False

    # Driving-specific metrics
    compute_driving_metrics: bool = False


@dataclass
class TrainingConfig:
    """Unified training configuration."""
    # Basic training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000

    # Self-forcing curriculum (integrated comma.ai insights)
    enable_curriculum: bool = True
    initial_sequence_length: int = 4
    final_sequence_length: int = 32
    curriculum_warmup_steps: int = 10000

    initial_self_forcing_ratio: float = 0.0
    final_self_forcing_ratio: float = 0.8
    self_forcing_warmup_steps: int = 5000

    # Flow matching
    enable_flow_matching: bool = True
    flow_matching_weight: float = 1.0
    num_flow_steps: int = 4

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

    # New component configurations
    repa: REPAConfig = field(default_factory=REPAConfig)
    cjepa: CJEPAConfig = field(default_factory=CJEPAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Loss weights (configurable)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'reconstruction': 1.0,
        'flow_matching': 1.0,
        'temporal_consistency': 0.2,
        'jepa': 0.1,
        'repa': 0.5,
        'cjepa': 0.3,
        'moe_balance': 0.01,
    })


# =============================================================================
# Helper Classes
# =============================================================================

class CurriculumScheduler:
    """Manages curriculum learning schedule."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.step = 0

    def update(self, step: int) -> None:
        """Update current training step."""
        self.step = step

    def get_sequence_length(self) -> int:
        """Get current sequence length."""
        if not self.config.enable_curriculum:
            return self.config.final_sequence_length

        progress = min(1.0, self.step / self.config.curriculum_warmup_steps)
        progress = self._smooth_schedule(progress)

        length = (
            self.config.initial_sequence_length +
            progress * (self.config.final_sequence_length - self.config.initial_sequence_length)
        )
        return int(length)

    def get_self_forcing_ratio(self) -> float:
        """Get current self-forcing ratio."""
        progress = min(1.0, self.step / self.config.self_forcing_warmup_steps)
        progress = self._smooth_schedule(progress)

        ratio = (
            self.config.initial_self_forcing_ratio +
            progress * (self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio)
        )
        return ratio

    def _smooth_schedule(self, progress: float) -> float:
        """Apply smooth scheduling (cosine)."""
        return 0.5 * (1 - math.cos(math.pi * progress))


class HASTEController:
    """Controls HASTE (Halt Alignment STrategy Early) for REPA.

    HASTE monitors REPA loss and automatically disables it when
    the model has learned good representations and continuing
    alignment provides diminishing returns.
    """

    def __init__(self, config: REPAConfig):
        self.config = config
        self.loss_history: List[float] = []
        self.is_active = config.enabled and config.haste_enabled
        self.repa_disabled = False
        self.disable_step: Optional[int] = None
        self.callbacks: List[OnREPADisabledCallback] = []

    def add_callback(self, callback: OnREPADisabledCallback) -> None:
        """Add a callback to be triggered when REPA is disabled."""
        self.callbacks.append(callback)

    def update(self, step: int, repa_loss: float) -> bool:
        """Update HASTE state and check if REPA should be disabled.

        Args:
            step: Current training step
            repa_loss: Current REPA loss value

        Returns:
            True if REPA is still active, False if disabled
        """
        if not self.is_active or self.repa_disabled:
            return not self.repa_disabled

        # Wait for warmup
        if step < self.config.haste_warmup_steps:
            return True

        self.loss_history.append(repa_loss)

        # Keep only recent history
        window_size = self.config.haste_window_size
        if len(self.loss_history) > window_size * (self.config.haste_patience + 1):
            self.loss_history = self.loss_history[-window_size * (self.config.haste_patience + 1):]

        # Check for plateau
        if len(self.loss_history) >= window_size * 2:
            should_disable = self._check_plateau()

            if should_disable:
                self.repa_disabled = True
                self.disable_step = step

                # Trigger callbacks
                final_loss = sum(self.loss_history[-window_size:]) / window_size
                for callback in self.callbacks:
                    callback(step, final_loss, self.config.haste_threshold)

                return False

        return True

    def _check_plateau(self) -> bool:
        """Check if REPA loss has plateaued."""
        window_size = self.config.haste_window_size
        patience = self.config.haste_patience
        threshold = self.config.haste_threshold

        if len(self.loss_history) < window_size * (patience + 1):
            return False

        # Compare recent windows
        windows = []
        for i in range(patience + 1):
            start = -(i + 1) * window_size
            end = -i * window_size if i > 0 else None
            window_avg = sum(self.loss_history[start:end]) / window_size
            windows.append(window_avg)

        # Check if improvement is below threshold across all windows
        for i in range(len(windows) - 1):
            improvement = windows[i + 1] - windows[i]  # Older - newer
            if abs(improvement) > threshold:
                return False

        return True

    def should_compute_repa(self) -> bool:
        """Check if REPA loss should be computed."""
        return self.config.enabled and not self.repa_disabled


# =============================================================================
# REPA (Representation Alignment) Components
# =============================================================================

class REPAProjector(nn.Module):
    """Projector network for REPA alignment.

    Projects model representations to alignment space for comparison
    with the pre-trained backbone representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 256
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project representations to alignment space.

        Args:
            x: Input representations [B, T, D] or [B, D]

        Returns:
            Projected representations with same batch dimensions
        """
        original_shape = x.shape

        # Flatten for BatchNorm
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            x = self.projector(x)
            x = x.reshape(B, T, -1)
        else:
            x = self.projector(x)

        return x


class REPABackbone(nn.Module):
    """Placeholder backbone for REPA alignment.

    In production, this would load a pre-trained V-JEPA 2.1 or similar
    encoder. For now, provides a simple encoder that can be replaced.
    """

    def __init__(self, config: REPAConfig):
        super().__init__()
        self.config = config

        # Simple encoder (replace with actual V-JEPA 2.1 in production)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.alignment_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Freeze backbone (pre-trained)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to representations.

        Args:
            frames: Input frames [B, T, C, H, W] or [B, C, H, W]

        Returns:
            Encoded representations
        """
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            frames = frames.reshape(B * T, C, H, W)
            features = self.encoder(frames)
            features = features.reshape(B, T, -1)
        else:
            features = self.encoder(frames)

        return features


# =============================================================================
# C-JEPA (Contextual JEPA) Components
# =============================================================================

class SlotAttention(nn.Module):
    """Slot Attention module for object-centric representations.

    Iteratively refines object slot representations through competition.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        # Initialize slot parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Attention components
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.k_proj = nn.Linear(input_dim, slot_dim)
        self.v_proj = nn.Linear(input_dim, slot_dim)
        self.q_proj = nn.Linear(slot_dim, slot_dim)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim)
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply slot attention to inputs.

        Args:
            inputs: Input features [B, N, D]

        Returns:
            Slot representations [B, num_slots, slot_dim]
        """
        B, N, _ = inputs.shape

        # Initialize slots
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
            B, self.num_slots, self.slot_dim, device=inputs.device
        )

        # Normalize input
        inputs = self.norm_input(inputs)

        # Project keys and values
        k = self.k_proj(inputs)  # [B, N, slot_dim]
        v = self.v_proj(inputs)  # [B, N, slot_dim]

        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            q = self.q_proj(slots)  # [B, num_slots, slot_dim]

            # Scale dot-product attention
            scale = self.slot_dim ** -0.5
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, num_slots, N]
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots

            # Normalize attention weights
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)

            # Aggregate
            updates = torch.bmm(attn, v)  # [B, num_slots, slot_dim]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class CJEPAPredictor(nn.Module):
    """C-JEPA predictor for object-level future prediction."""

    def __init__(self, config: CJEPAConfig, model_dim: int):
        super().__init__()
        self.config = config

        # Slot attention for object extraction
        if config.use_slot_attention:
            self.slot_attention = SlotAttention(
                num_slots=config.num_object_queries,
                slot_dim=config.object_dim,
                input_dim=model_dim,
                num_iterations=config.slot_attention_iters
            )
        else:
            # Simple pooling alternative
            self.slot_attention = None
            self.query_embed = nn.Parameter(
                torch.randn(config.num_object_queries, config.object_dim)
            )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.object_dim, config.object_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.object_dim * 2, config.object_dim * config.prediction_horizon)
        )

        # Projection for contrastive loss
        self.projection = nn.Linear(config.object_dim, config.object_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_slots: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Extract object slots and predict future states.

        Args:
            hidden_states: Model hidden states [B, T, D]
            return_slots: Whether to return slot representations

        Returns:
            Dictionary with predictions and optionally slots
        """
        B, T, D = hidden_states.shape

        # Extract object slots
        if self.slot_attention is not None:
            slots = self.slot_attention(hidden_states)  # [B, num_slots, slot_dim]
        else:
            # Simple attention with learned queries
            queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
            attn = torch.bmm(queries, hidden_states.transpose(1, 2))
            attn = F.softmax(attn / (D ** 0.5), dim=-1)
            slots = torch.bmm(attn, hidden_states)

        # Predict future object states
        predictions = self.predictor(slots)  # [B, num_slots, slot_dim * horizon]
        predictions = predictions.reshape(
            B,
            self.config.num_object_queries,
            self.config.prediction_horizon,
            self.config.object_dim
        )

        # Project for contrastive loss
        projected = self.projection(slots)

        outputs = {
            'predictions': predictions,
            'projected': projected,
        }

        if return_slots:
            outputs['slots'] = slots

        return outputs


# =============================================================================
# MoE Monitoring
# =============================================================================

class MoEMonitor:
    """Monitor for Mixture of Experts load balancing.

    Tracks expert utilization and computes auxiliary losses
    for balanced routing.
    """

    def __init__(self, config: MoEConfig):
        self.config = config
        self.expert_counts: Dict[str, List[int]] = {}
        self.routing_history: List[torch.Tensor] = []
        self.callbacks: List[OnExpertImbalanceCallback] = []

    def add_callback(self, callback: OnExpertImbalanceCallback) -> None:
        """Add callback for expert imbalance detection."""
        self.callbacks.append(callback)

    def update(
        self,
        router_logits: torch.Tensor,
        layer_name: str = "default"
    ) -> Dict[str, float]:
        """Update expert usage statistics.

        Args:
            router_logits: Router output logits [B, N, num_experts]
            layer_name: Name of the MoE layer

        Returns:
            Dictionary of utilization metrics
        """
        if not self.config.enabled:
            return {}

        # Get routing decisions
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.config.top_k, dim=-1
        )

        # Count expert usage
        expert_usage = torch.zeros(self.config.num_experts, device=router_logits.device)
        for k in range(self.config.top_k):
            indices = top_k_indices[..., k].flatten()
            expert_usage.scatter_add_(
                0,
                indices,
                torch.ones_like(indices, dtype=torch.float)
            )

        # Normalize
        total_tokens = router_logits.shape[0] * router_logits.shape[1]
        expert_usage = expert_usage / total_tokens

        # Store history
        if layer_name not in self.expert_counts:
            self.expert_counts[layer_name] = []
        self.expert_counts[layer_name].append(expert_usage.tolist())

        # Keep recent history
        if len(self.expert_counts[layer_name]) > 100:
            self.expert_counts[layer_name] = self.expert_counts[layer_name][-100:]

        # Compute metrics
        metrics = {
            f'{layer_name}_expert_usage_std': expert_usage.std().item(),
            f'{layer_name}_expert_usage_max': expert_usage.max().item(),
            f'{layer_name}_expert_usage_min': expert_usage.min().item(),
        }

        return metrics

    def compute_load_balance_loss(
        self,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss.

        Args:
            router_logits: Router output logits [B, N, num_experts]

        Returns:
            Load balancing loss
        """
        if not self.config.use_auxiliary_loss:
            return torch.tensor(0.0, device=router_logits.device)

        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Expert fraction: mean probability per expert
        expert_frac = routing_probs.mean(dim=[0, 1])  # [num_experts]

        # Ideal uniform distribution
        ideal_frac = 1.0 / self.config.num_experts

        # Load balance loss: encourage uniform distribution
        loss = self.config.num_experts * (expert_frac * expert_frac).sum()

        return loss * self.config.load_balance_weight

    def compute_router_z_loss(
        self,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute router z-loss for stability.

        Args:
            router_logits: Router output logits [B, N, num_experts]

        Returns:
            Z-loss for router stability
        """
        # Z-loss: penalize large logits to prevent router collapse
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return z_loss * self.config.router_z_loss_weight

    def get_expert_utilization(self) -> Dict[str, float]:
        """Get current expert utilization statistics."""
        utilization = {}

        for layer_name, counts in self.expert_counts.items():
            if counts:
                recent_counts = counts[-10:]  # Last 10 updates
                avg_counts = [
                    sum(c[i] for c in recent_counts) / len(recent_counts)
                    for i in range(self.config.num_experts)
                ]
                for i, count in enumerate(avg_counts):
                    utilization[f'{layer_name}_expert_{i}'] = count

        return utilization

    def check_imbalance(self, step: int) -> bool:
        """Check for expert imbalance and trigger callbacks.

        Args:
            step: Current training step

        Returns:
            True if imbalance was detected
        """
        utilization = self.get_expert_utilization()

        for callback in self.callbacks:
            if callback(step, utilization):
                return True

        return False


# =============================================================================
# Unified Trainer
# =============================================================================

class UnifiedTrainer:
    """
    Unified trainer that combines all training methodologies:
    - Self-forcing with curriculum learning (comma.ai insights)
    - Flow matching and distillation
    - Distributed training
    - Memory management
    - REPA with HASTE early-stopping
    - C-JEPA object-level prediction
    - MoE load balancing
    """

    def __init__(
        self,
        model: WorldModel,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        """Initialize the unified trainer.

        Args:
            model: WorldModel instance to train
            config: TrainingConfig with all settings
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Training components
        self.curriculum = CurriculumScheduler(config)
        self.loss_fn = UnifiedLoss(config)
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)

        # Distributed training
        self.distributed_manager = DistributedManager() if config.distributed else None

        # Checkpoint management
        self.checkpoint_manager = CheckpointManager("./checkpoints")

        # Create internal component configs (used throughout trainer)
        self._repa_config = REPAConfig() if self.config.is_component_enabled(ComponentType.REPA) else None
        self._cjepa_config = CJEPAConfig() if self.config.is_component_enabled(ComponentType.CAUSAL_JEPA) else None
        self._moe_config = MoEConfig() if self.config.is_component_enabled(ComponentType.MOE) else None

        # Initialize REPA components
        self._init_repa_components()

        # Initialize C-JEPA components
        self._init_cjepa_components()

        # Initialize MoE monitoring
        self._init_moe_monitoring()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Metrics tracking
        self.metrics: Dict[str, List[float]] = {
            'train_loss': [],
            'sequence_length': [],
            'self_forcing_ratio': [],
            'memory_usage': [],
            'repa_loss': [],
            'cjepa_loss': [],
            'moe_balance_loss': [],
        }

        # Callbacks
        self.callbacks: List[TrainingCallback] = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        self.logger.info(
            f"Initialized UnifiedTrainer with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        self._log_component_status()

    def _init_repa_components(self) -> None:
        """Initialize REPA (Representation Alignment) components."""
        if self._repa_config is not None:
            # REPA backbone (frozen pre-trained encoder)
            self.repa_backbone = REPABackbone(self._repa_config).to(self.device)

            # Projector for model representations
            self.repa_projector = REPAProjector(
                input_dim=self.config.model_dim,
                hidden_dim=self._repa_config.projection_hidden_dim,
                output_dim=self._repa_config.projection_output_dim
            ).to(self.device)

            # HASTE controller
            self.haste_controller = HASTEController(self._repa_config)

            # Add default callback
            self.haste_controller.add_callback(OnREPADisabledCallback())

            self.logger.info("REPA components initialized with HASTE controller")
        else:
            self.repa_backbone = None
            self.repa_projector = None
            self.haste_controller = None

    def _init_cjepa_components(self) -> None:
        """Initialize C-JEPA (Contextual JEPA) components."""
        if self._cjepa_config is not None:
            self.cjepa_predictor = CJEPAPredictor(
                self._cjepa_config,
                self.config.model_dim
            ).to(self.device)

            self.logger.info(
                f"C-JEPA initialized with {self._cjepa_config.num_object_queries} object slots"
            )
        else:
            self.cjepa_predictor = None

    def _init_moe_monitoring(self) -> None:
        """Initialize MoE monitoring."""
        if self._moe_config is not None:
            self.moe_monitor = MoEMonitor(self._moe_config)

            # Add default callback
            self.moe_monitor.add_callback(OnExpertImbalanceCallback())

            self.logger.info("MoE monitoring initialized")
        else:
            self.moe_monitor = None

    def _log_component_status(self) -> None:
        """Log status of all training components."""
        status = []
        status.append(f"REPA: {'enabled' if self.config.is_component_enabled(ComponentType.REPA) else 'disabled'}")
        status.append(f"C-JEPA: {'enabled' if self.config.is_component_enabled(ComponentType.CAUSAL_JEPA) else 'disabled'}")
        status.append(f"MoE: {'enabled' if self.config.is_component_enabled(ComponentType.MOE) else 'disabled'}")
        status.append(f"Flow Matching: {'enabled' if self.config.enable_flow_matching else 'disabled'}")
        status.append(f"Curriculum: {'enabled' if self.config.enable_curriculum else 'disabled'}")

        self.logger.info(f"Component status: {', '.join(status)}")

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)

    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            # Update curriculum
            self.curriculum.update(self.global_step)

            # Adapt batch to current curriculum
            batch = self._adapt_batch_sequence_length(batch)

            # Training step
            losses = self.train_step(batch, optimizer)
            epoch_losses.append(losses['total'])

            # Trigger callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_train_step_end'):
                    callback.on_train_step_end(
                        self,
                        self.global_step,
                        losses,
                        self.get_current_metrics()
                    )

            # Logging
            if self.global_step % 100 == 0:
                self._log_metrics(losses)

            # Checkpointing
            if self.global_step % self.config.checkpoint_every == 0:
                self._save_checkpoint(losses['total'], optimizer, scheduler)

            # Validation
            if (self.config.validation.enabled and
                self.global_step % self.config.validation.frequency == 0 and
                self.global_step > 0):
                # Note: validation dataloader should be passed separately in production
                pass

            # Memory management
            if self.memory_monitor.should_cleanup():
                self._cleanup_memory()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            self.global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        self.epoch += 1

        # Trigger epoch callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(self, self.epoch, self.get_current_metrics())

        return avg_loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step with unified methodology.

        Args:
            batch: Batch of training data
            optimizer: Optimizer instance

        Returns:
            Dictionary of loss values
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get current curriculum parameters
        seq_len = self.curriculum.get_sequence_length()
        sf_ratio = self.curriculum.get_self_forcing_ratio()

        optimizer.zero_grad()

        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                predictions = self._forward_pass(batch, sf_ratio)
                losses = self._compute_losses(predictions, batch, sf_ratio)

            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard training
            predictions = self._forward_pass(batch, sf_ratio)
            losses = self._compute_losses(predictions, batch, sf_ratio)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            optimizer.step()

        # Update metrics
        self._update_metrics(losses, seq_len, sf_ratio)

        # Check MoE balance (if applicable)
        if self.moe_monitor is not None and self._moe_config is not None:
            if self.global_step % self._moe_config.utilization_log_frequency == 0:
                self.moe_monitor.check_imbalance(self.global_step)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    def _forward_pass(
        self,
        batch: Dict[str, torch.Tensor],
        sf_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with self-forcing.

        Args:
            batch: Batch of training data
            sf_ratio: Self-forcing ratio

        Returns:
            Model predictions
        """
        frames = batch['frames']  # [B, T, C, H, W]
        controls = batch.get('controls')  # [B, T, control_dim] (optional)
        depth = batch.get('depth')  # [B, T, 1, H, W] (optional)

        B, T = frames.shape[:2]

        # Use self-forcing mode for training
        outputs = self.model(
            frames=frames,
            controls=controls,
            depth=depth,
            mode="self_forcing",
            self_forcing_ratio=sf_ratio
        )

        return outputs

    def _compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        sf_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses using unified loss function.

        Args:
            predictions: Model predictions
            batch: Input batch
            sf_ratio: Self-forcing ratio

        Returns:
            Dictionary of losses including total loss
        """
        # Prepare targets
        targets = {
            'frames': batch['frames'],
            'controls': batch.get('controls'),
            'depth': batch.get('depth')
        }

        # Add flow matching targets if enabled
        if self.config.enable_flow_matching and 'predictions' in predictions:
            targets['flow_target'] = self._compute_flow_target(predictions, targets)

        # Compute base losses
        losses = self.loss_fn(predictions, targets, self.global_step)

        # Compute REPA loss
        if self.haste_controller is not None and self.haste_controller.should_compute_repa():
            repa_loss = self.compute_repa_loss(
                predictions.get('hidden_states'),
                batch['frames']
            )
            losses['repa'] = repa_loss * self.config.loss_weights.get('repa', 0.5)

            # Update HASTE controller
            self.haste_controller.update(self.global_step, repa_loss.item())

        # Compute C-JEPA loss
        if self.cjepa_predictor is not None and 'hidden_states' in predictions:
            cjepa_loss = self.compute_cjepa_loss(
                predictions['hidden_states'],
                batch['frames']
            )
            losses['cjepa'] = cjepa_loss * self.config.loss_weights.get('cjepa', 0.3)

        # Compute MoE balance loss
        if self.moe_monitor is not None and 'router_logits' in predictions:
            moe_loss = self.log_moe_metrics(predictions['router_logits'])
            losses['moe_balance'] = moe_loss

        # Recompute total loss with all components
        total_loss = torch.tensor(0.0, device=batch['frames'].device)
        for key, loss in losses.items():
            if key != 'total' and torch.is_tensor(loss):
                total_loss = total_loss + loss

        losses['total'] = total_loss

        return losses

    def compute_repa_loss(
        self,
        hidden_states: Optional[torch.Tensor],
        frames: torch.Tensor
    ) -> torch.Tensor:
        """Compute REPA alignment loss with V-JEPA 2.1 backbone.

        REPA aligns the model's representations with a pre-trained encoder
        to improve representation quality and transfer.

        Args:
            hidden_states: Model hidden states [B, T, D]
            frames: Input frames [B, T, C, H, W]

        Returns:
            REPA alignment loss
        """
        if hidden_states is None or self.repa_backbone is None:
            return torch.tensor(0.0, device=frames.device)

        # Get target representations from frozen backbone
        with torch.no_grad():
            target_features = self.repa_backbone(frames)  # [B, T, alignment_dim]

        # Project model representations
        projected_features = self.repa_projector(hidden_states)  # [B, T, projection_dim]

        # Compute InfoNCE loss for alignment
        B, T = projected_features.shape[:2]

        # Flatten for batch processing
        proj_flat = projected_features.reshape(B * T, -1)
        target_flat = target_features.reshape(B * T, -1)

        # Normalize for cosine similarity
        proj_norm = F.normalize(proj_flat, dim=-1)
        target_norm = F.normalize(target_flat, dim=-1)

        # Compute similarity matrix
        temperature = self._repa_config.temperature if self._repa_config else 0.07
        similarity = torch.matmul(proj_norm, target_norm.T) / temperature

        # InfoNCE loss: diagonal elements are positives
        labels = torch.arange(B * T, device=frames.device)
        loss = F.cross_entropy(similarity, labels)

        return loss

    def compute_cjepa_loss(
        self,
        hidden_states: torch.Tensor,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """Compute C-JEPA loss for object-level prediction.

        C-JEPA predicts future object states in a latent space,
        enabling better object permanence and tracking.

        Args:
            hidden_states: Model hidden states [B, T, D]
            frames: Input frames [B, T, C, H, W]

        Returns:
            C-JEPA contrastive loss
        """
        if self.cjepa_predictor is None:
            return torch.tensor(0.0, device=frames.device)

        B, T, D = hidden_states.shape
        horizon = self._cjepa_config.prediction_horizon if self._cjepa_config else 4

        if T <= horizon:
            return torch.tensor(0.0, device=frames.device)

        # Split into context and target
        context_states = hidden_states[:, :-horizon]
        target_states = hidden_states[:, horizon:]

        # Get object slots from context
        cjepa_outputs = self.cjepa_predictor(context_states, return_slots=True)
        predictions = cjepa_outputs['predictions']  # [B, num_slots, horizon, slot_dim]
        context_slots = cjepa_outputs['slots']  # [B, num_slots, slot_dim]

        # Get target slots
        with torch.no_grad():
            target_outputs = self.cjepa_predictor(target_states, return_slots=True)
            target_slots = target_outputs['slots']  # [B, num_slots, slot_dim]

        # Compute contrastive loss between predicted and actual future slots
        # Use the last prediction step
        pred_final = predictions[:, :, -1, :]  # [B, num_slots, slot_dim]

        # Normalize
        pred_norm = F.normalize(pred_final, dim=-1)
        target_norm = F.normalize(target_slots, dim=-1)

        # Compute similarity
        similarity = torch.bmm(pred_norm, target_norm.transpose(1, 2))
        cjepa_temp = self._cjepa_config.contrastive_temperature if self._cjepa_config else 0.1
        similarity = similarity / cjepa_temp

        # Contrastive loss: each slot should match its corresponding target
        num_queries = self._cjepa_config.num_object_queries if self._cjepa_config else 64
        labels = torch.arange(
            num_queries,
            device=frames.device
        ).unsqueeze(0).expand(B, -1)

        loss = F.cross_entropy(
            similarity.reshape(-1, num_queries),
            labels.reshape(-1)
        )

        return loss

    def log_moe_metrics(
        self,
        router_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Log MoE expert utilization statistics.

        Args:
            router_logits: Router output logits [B, N, num_experts]

        Returns:
            MoE auxiliary loss
        """
        if self.moe_monitor is None or router_logits is None:
            return torch.tensor(0.0, device=self.device)

        # Update monitor
        metrics = self.moe_monitor.update(router_logits)

        # Compute auxiliary losses
        balance_loss = self.moe_monitor.compute_load_balance_loss(router_logits)
        z_loss = self.moe_monitor.compute_router_z_loss(router_logits)

        total_moe_loss = balance_loss + z_loss

        # Log metrics periodically
        log_freq = self._moe_config.utilization_log_frequency if self._moe_config else 100
        if self.global_step % log_freq == 0:
            utilization = self.moe_monitor.get_expert_utilization()
            self.logger.info(f"MoE utilization at step {self.global_step}: {utilization}")

        return total_moe_loss

    def check_haste_condition(self) -> bool:
        """Check whether HASTE has disabled REPA.

        Returns:
            True if REPA is still active, False if disabled by HASTE
        """
        if self.haste_controller is None:
            return False

        return not self.haste_controller.repa_disabled

    def _compute_flow_target(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute flow matching target."""
        if 'predictions' in predictions and 'frames' in targets:
            pred_frames = predictions['predictions']
            target_frames = targets['frames']

            # Handle shape mismatch: predictions may only contain target portion
            # (e.g., from self-forcing where only second half is predicted)
            pred_T = pred_frames.size(1)
            target_T = target_frames.size(1)

            if pred_T != target_T:
                # Slice target frames to match prediction length (from the end)
                target_frames = target_frames[:, -pred_T:]

            # Simple flow target: difference between prediction and target
            return target_frames - pred_frames

        return torch.zeros_like(predictions.get('predictions', targets['frames']))

    def _adapt_batch_sequence_length(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Adapt batch to current curriculum sequence length."""
        target_length = self.curriculum.get_sequence_length()
        adapted_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                current_length = value.size(1)
                if current_length > target_length:
                    # Random crop
                    start_idx = random.randint(0, current_length - target_length)
                    adapted_batch[key] = value[:, start_idx:start_idx + target_length]
                elif current_length < target_length:
                    # Repeat last element
                    repeats = target_length - current_length
                    last_elements = value[:, -1:].repeat(1, repeats, *([1] * (value.dim() - 2)))
                    adapted_batch[key] = torch.cat([value, last_elements], dim=1)
                else:
                    adapted_batch[key] = value
            else:
                adapted_batch[key] = value

        return adapted_batch

    def _update_metrics(
        self,
        losses: Dict[str, torch.Tensor],
        seq_len: int,
        sf_ratio: float
    ) -> None:
        """Update training metrics."""
        self.metrics['train_loss'].append(
            losses['total'].item() if torch.is_tensor(losses['total']) else losses['total']
        )
        self.metrics['sequence_length'].append(seq_len)
        self.metrics['self_forcing_ratio'].append(sf_ratio)

        # Component-specific metrics
        if 'repa' in losses:
            self.metrics['repa_loss'].append(
                losses['repa'].item() if torch.is_tensor(losses['repa']) else losses['repa']
            )
        if 'cjepa' in losses:
            self.metrics['cjepa_loss'].append(
                losses['cjepa'].item() if torch.is_tensor(losses['cjepa']) else losses['cjepa']
            )
        if 'moe_balance' in losses:
            self.metrics['moe_balance_loss'].append(
                losses['moe_balance'].item() if torch.is_tensor(losses['moe_balance']) else losses['moe_balance']
            )

        # Memory usage
        memory_stats = self.memory_monitor.update()
        self.metrics['memory_usage'].append(memory_stats['usage_percent'])

        # Keep only recent history
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-500:]

    def _log_metrics(self, losses: Dict[str, float]) -> None:
        """Log current metrics."""
        seq_len = self.curriculum.get_sequence_length()
        sf_ratio = self.curriculum.get_self_forcing_ratio()
        memory_stats = self.memory_monitor.update()

        log_parts = [
            f"Step {self.global_step}",
            f"Loss={losses['total']:.4f}",
            f"SeqLen={seq_len}",
            f"SF_Ratio={sf_ratio:.3f}",
            f"Memory={memory_stats['usage_percent']:.1f}%"
        ]

        # Add HASTE status
        if self.haste_controller is not None:
            status = "disabled" if self.haste_controller.repa_disabled else "active"
            log_parts.append(f"REPA={status}")

        self.logger.info(", ".join(log_parts))

        # Detailed loss breakdown
        if len(losses) > 1:
            loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses.items() if k != 'total'])
            self.logger.info(f"  Losses: {loss_str}")

    def _save_checkpoint(
        self,
        current_loss: float,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> None:
        """Save training checkpoint."""
        is_best = current_loss < self.best_loss
        if is_best:
            self.best_loss = current_loss

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=current_loss,
            metrics=self.get_current_metrics(),
            is_best=is_best
        )

        if is_best:
            self.logger.info(f"New best model saved: {checkpoint_path}")

    def _cleanup_memory(self) -> None:
        """Clean up memory when needed."""
        freed_gb = self.memory_monitor.cleanup()
        if freed_gb > 0:
            self.logger.info(f"Freed {freed_gb:.2f}GB of memory")

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        metrics = {}

        for key, values in self.metrics.items():
            if values:
                metrics[f'avg_{key}'] = sum(values[-100:]) / len(values[-100:])
                metrics[f'current_{key}'] = values[-1]

        metrics['global_step'] = self.global_step
        metrics['epoch'] = self.epoch
        metrics['best_loss'] = self.best_loss

        # Add HASTE status
        if self.haste_controller is not None:
            metrics['repa_active'] = not self.haste_controller.repa_disabled
            if self.haste_controller.disable_step is not None:
                metrics['repa_disabled_at_step'] = self.haste_controller.disable_step

        return metrics

    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Load training checkpoint."""
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            self.model, optimizer, scheduler, checkpoint_path
        )

        if checkpoint_info:
            self.global_step = checkpoint_info.get('step', 0)
            self.epoch = checkpoint_info.get('epoch', 0)
            self.best_loss = checkpoint_info.get('loss', float('inf'))

            self.logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")

        return checkpoint_info

    def evaluate(
        self,
        dataloader,
        compute_closed_loop: bool = False,
        compute_physics: bool = False,
        compute_driving: bool = False
    ) -> Dict[str, float]:
        """Evaluate model on validation data.

        Args:
            dataloader: Validation data loader
            compute_closed_loop: Whether to run closed-loop evaluation
            compute_physics: Whether to compute physics metrics
            compute_driving: Whether to compute driving-specific metrics

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_metrics: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch = self._adapt_batch_sequence_length(batch)

                predictions = self.model(
                    frames=batch['frames'],
                    controls=batch.get('controls'),
                    depth=batch.get('depth'),
                    mode="train"
                )

                losses = self._compute_losses(predictions, batch, 0.0)
                total_loss += losses['total'].item()
                num_batches += 1

                # Collect individual loss metrics
                for key, value in losses.items():
                    if key != 'total' and torch.is_tensor(value):
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value.item())

        # Compute averages
        result = {'eval_loss': total_loss / num_batches if num_batches > 0 else 0.0}

        for key, values in all_metrics.items():
            result[f'eval_{key}'] = sum(values) / len(values) if values else 0.0

        # Closed-loop evaluation
        if compute_closed_loop and self.config.validation.closed_loop_enabled:
            closed_loop_metrics = self._evaluate_closed_loop(dataloader)
            result.update(closed_loop_metrics)

        # Physics metrics
        if compute_physics and self.config.validation.compute_physics_metrics:
            physics_metrics = self._compute_physics_metrics(dataloader)
            result.update(physics_metrics)

        # Driving metrics
        if compute_driving and self.config.validation.compute_driving_metrics:
            driving_metrics = self._compute_driving_metrics(dataloader)
            result.update(driving_metrics)

        self.model.train()

        return result

    def _evaluate_closed_loop(
        self,
        dataloader
    ) -> Dict[str, float]:
        """Run closed-loop evaluation.

        Generates sequences autoregressively and compares with ground truth.
        """
        horizon = self.config.validation.closed_loop_horizon
        total_mse = 0.0
        total_temporal_consistency = 0.0
        num_samples = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            frames = batch['frames']
            B, T = frames.shape[:2]

            if T <= horizon:
                continue

            # Use first half as context
            context_length = T // 2
            context_frames = frames[:, :context_length]
            target_frames = frames[:, context_length:context_length + horizon]

            # Autoregressive generation
            outputs = self.model(
                frames=context_frames,
                controls=batch.get('controls', None),
                depth=batch.get('depth', None),
                mode="inference",
                num_steps=min(horizon, target_frames.shape[1])
            )

            generated = outputs.get('generated_frames', outputs.get('predictions'))

            if generated is not None:
                # Compute MSE
                min_len = min(generated.shape[1], target_frames.shape[1])
                mse = F.mse_loss(
                    generated[:, :min_len],
                    target_frames[:, :min_len]
                ).item()
                total_mse += mse

                # Temporal consistency
                if min_len > 1:
                    gen_diff = generated[:, 1:min_len] - generated[:, :min_len-1]
                    target_diff = target_frames[:, 1:min_len] - target_frames[:, :min_len-1]
                    temp_consistency = F.mse_loss(gen_diff, target_diff).item()
                    total_temporal_consistency += temp_consistency

                num_samples += 1

            # Limit evaluation batches
            if num_samples >= self.config.validation.num_batches:
                break

        return {
            'closed_loop_mse': total_mse / max(num_samples, 1),
            'closed_loop_temporal_consistency': total_temporal_consistency / max(num_samples, 1),
        }

    def _compute_physics_metrics(
        self,
        dataloader
    ) -> Dict[str, float]:
        """Compute physics-based metrics."""
        # Import physics metrics
        try:
            from evaluation.metrics import PhysicsMetrics
        except ImportError:
            return {}

        trajectory_smoothness_values = []
        speed_consistency_values = []

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Extract positions from controls if available
            if 'positions' in batch:
                positions = batch['positions']

                smoothness = PhysicsMetrics.trajectory_smoothness(positions)
                trajectory_smoothness_values.append(smoothness.item())

                consistency = PhysicsMetrics.speed_consistency(positions)
                speed_consistency_values.append(consistency.item())

        return {
            'physics_trajectory_smoothness': (
                sum(trajectory_smoothness_values) / len(trajectory_smoothness_values)
                if trajectory_smoothness_values else 0.0
            ),
            'physics_speed_consistency': (
                sum(speed_consistency_values) / len(speed_consistency_values)
                if speed_consistency_values else 0.0
            ),
        }

    def _compute_driving_metrics(
        self,
        dataloader
    ) -> Dict[str, float]:
        """Compute driving-specific metrics."""
        try:
            from evaluation.metrics import ControlMetrics
        except ImportError:
            return {}

        steering_accuracy_values = []
        acceleration_accuracy_values = []
        control_smoothness_values = []

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if 'controls' not in batch:
                continue

            controls = batch['controls']

            # Get model predictions
            predictions = self.model(
                frames=batch['frames'],
                controls=controls,
                depth=batch.get('depth'),
                mode="train"
            )

            # Assume model can predict controls
            if 'predicted_controls' in predictions:
                pred_controls = predictions['predicted_controls']

                steering_acc = ControlMetrics.steering_accuracy(pred_controls, controls)
                steering_accuracy_values.append(steering_acc.item())

                accel_acc = ControlMetrics.acceleration_accuracy(pred_controls, controls)
                acceleration_accuracy_values.append(accel_acc.item())

            smoothness = ControlMetrics.control_smoothness(controls)
            control_smoothness_values.append(smoothness.item())

        return {
            'driving_steering_accuracy': (
                sum(steering_accuracy_values) / len(steering_accuracy_values)
                if steering_accuracy_values else 0.0
            ),
            'driving_acceleration_accuracy': (
                sum(acceleration_accuracy_values) / len(acceleration_accuracy_values)
                if acceleration_accuracy_values else 0.0
            ),
            'driving_control_smoothness': (
                sum(control_smoothness_values) / len(control_smoothness_values)
                if control_smoothness_values else 0.0
            ),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_unified_trainer(
    model_config: DriveDiTConfig,
    training_config: TrainingConfig,
    device: str = "cuda"
) -> UnifiedTrainer:
    """Create unified trainer with world model.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        device: Device to train on

    Returns:
        Configured UnifiedTrainer instance
    """
    # Create world model
    world_model = WorldModel(model_config)

    # Create trainer
    trainer = UnifiedTrainer(world_model, training_config, device)

    return trainer


def create_training_config(
    preset: str = "default",
    **overrides
) -> TrainingConfig:
    """Create training configuration from preset.

    Args:
        preset: Configuration preset name
        **overrides: Override specific settings

    Returns:
        TrainingConfig instance
    """
    if preset == "minimal":
        config = TrainingConfig(
            batch_size=2,
            max_steps=1000,
            enable_curriculum=False,
            repa=REPAConfig(enabled=False),
            cjepa=CJEPAConfig(enabled=False),
            moe=MoEConfig(enabled=False),
        )
    elif preset == "research":
        config = TrainingConfig(
            batch_size=8,
            max_steps=100000,
            enable_curriculum=True,
            repa=REPAConfig(enabled=True),
            cjepa=CJEPAConfig(enabled=True),
            moe=MoEConfig(enabled=False),
        )
    elif preset == "full":
        config = TrainingConfig(
            batch_size=16,
            max_steps=500000,
            enable_curriculum=True,
            repa=REPAConfig(enabled=True, haste_enabled=True),
            cjepa=CJEPAConfig(enabled=True, use_slot_attention=True),
            moe=MoEConfig(enabled=True, num_experts=8),
        )
    else:
        config = TrainingConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Test unified trainer with all components
    from config.config import get_minimal_config

    print("Testing Unified Trainer with all components...")

    # Create configurations
    model_config = get_minimal_config()
    training_config = create_training_config(
        preset="research",
        batch_size=2,
        max_steps=100,
        enable_curriculum=True,
        enable_flow_matching=True
    )

    # Enable test components
    training_config.repa.enabled = True
    training_config.cjepa.enabled = True

    print(f"\nConfiguration:")
    print(f"  REPA enabled: {training_config.repa.enabled}")
    print(f"  C-JEPA enabled: {training_config.cjepa.enabled}")
    print(f"  MoE enabled: {training_config.moe.enabled}")
    print(f"  HASTE enabled: {training_config.repa.haste_enabled}")

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = create_unified_trainer(model_config, training_config, device)

    # Create test batch
    batch = {
        'frames': torch.randn(2, 8, 3, 128, 128, device=device),
        'controls': torch.randn(2, 8, 6, device=device) if model_config.is_component_enabled(ComponentType.CONTROL) else None
    }

    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(trainer.model.parameters()) +
        (list(trainer.repa_projector.parameters()) if trainer.repa_projector else []) +
        (list(trainer.cjepa_predictor.parameters()) if trainer.cjepa_predictor else []),
        lr=training_config.learning_rate
    )

    # Test training step
    print("\nRunning training step...")
    losses = trainer.train_step(batch, optimizer)

    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value:.6f}")

    # Test metrics
    print("\nMetrics:")
    metrics = trainer.get_current_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Test HASTE status
    print(f"\nHASTE Status:")
    print(f"  REPA active: {trainer.check_haste_condition()}")

    print("\nUnified trainer test completed successfully!")

