"""
REPA Loss with HASTE (Holistic Alignment with Stage-wise Termination).

Implements Representation Alignment (REPA) loss for aligning DiT hidden states
with V-JEPA 2.1 features.

Key insight from HASTE: REPA accelerates early training but can plateau later
due to capacity mismatch. The teacher's lower-dimensional embeddings become a
"straitjacket" once the student starts modeling the full joint distribution.

References:
- REPA: arxiv:2410.06940 (17.5× training speedup, ICLR 2025 Oral)
- HASTE: Early-stop alignment at 40% of training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import math
import warnings


@dataclass
class REPAConfig:
    """Configuration for REPA loss with HASTE."""
    # Dimension matching
    dit_hidden_dim: int = 512
    vjepa_feature_dim: int = 1024
    projection_hidden_dim: Optional[int] = None

    # Layer alignment
    dit_layers_to_align: List[int] = field(default_factory=lambda: [6, 8, 10])
    vjepa_layers_to_use: List[int] = field(default_factory=lambda: [18, 20, 22])

    # HASTE configuration
    enable_haste: bool = True
    haste_termination_ratio: float = 0.4  # Stop at 40% of training
    haste_warmup_ratio: float = 0.05
    haste_schedule: str = "cosine"

    # Loss weighting
    base_repa_weight: float = 1.0
    per_layer_weights: Optional[List[float]] = None
    repa_relative_weight: float = 0.1

    # Alignment method
    alignment_type: str = "cosine"
    normalize_features: bool = True

    # Training tracking
    total_training_steps: int = 100000

    def __post_init__(self):
        if self.projection_hidden_dim is None:
            self.projection_hidden_dim = (self.dit_hidden_dim + self.vjepa_feature_dim) // 2
        if self.per_layer_weights is None:
            self.per_layer_weights = [1.0] * len(self.dit_layers_to_align)


class ProjectionMLP(nn.Module):
    """MLP to project DiT hidden states to V-JEPA feature dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(output_dim)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.norm(x)
        return x


class HASTEScheduler:
    """
    HASTE (Holistic Alignment with Stage-wise Termination) scheduler.

    Schedule:
    1. Warmup phase: Gradually increase REPA weight from 0 to base_weight
    2. Active phase: Full REPA weight
    3. Termination phase: REPA disabled (weight = 0)
    """

    def __init__(self, config: REPAConfig):
        self.config = config
        self.current_step = 0

        total_steps = config.total_training_steps
        self.warmup_end = int(total_steps * config.haste_warmup_ratio)
        self.termination_start = int(total_steps * config.haste_termination_ratio)

    def get_weight(self, step: Optional[int] = None) -> float:
        if step is not None:
            self.current_step = step

        if not self.config.enable_haste:
            return 1.0

        step = self.current_step
        total_steps = self.config.total_training_steps

        # Phase 1: Warmup
        if step < self.warmup_end:
            progress = step / self.warmup_end
            if self.config.haste_schedule == "linear":
                return progress
            elif self.config.haste_schedule == "cosine":
                return 0.5 * (1 - math.cos(math.pi * progress))
            else:
                return 0.0 if progress < 0.5 else 1.0

        # Phase 3: After termination
        if step >= self.termination_start:
            return 0.0

        # Phase 2: Active alignment
        return 1.0

    def update(self, step: int):
        self.current_step = step

    def is_active(self) -> bool:
        return self.get_weight() > 0.0

    def get_phase(self) -> str:
        if self.current_step < self.warmup_end:
            return "warmup"
        elif self.current_step < self.termination_start:
            return "active"
        else:
            return "terminated"

    def get_progress_info(self) -> Dict[str, Any]:
        return {
            'current_step': self.current_step,
            'warmup_end': self.warmup_end,
            'termination_start': self.termination_start,
            'total_steps': self.config.total_training_steps,
            'current_weight': self.get_weight(),
            'current_phase': self.get_phase(),
            'is_active': self.is_active()
        }


class REPALoss(nn.Module):
    """
    REPA Loss: Representation Alignment with V-JEPA teacher.

    Usage:
        repa_loss = REPALoss(config)

        # In training loop:
        dit_features = model.get_intermediate_features(x)
        vjepa_features = vjepa_backbone(video)['intermediate']

        loss_dict = repa_loss(
            dit_features=dit_features,
            vjepa_features=vjepa_features,
            step=current_step
        )

        total_loss = reconstruction_loss + loss_dict['total_repa_loss']
    """

    def __init__(self, config: REPAConfig):
        super().__init__()
        self.config = config

        # Create projection layers
        self.projections = nn.ModuleDict()
        for i, (dit_layer, vjepa_layer) in enumerate(
            zip(config.dit_layers_to_align, config.vjepa_layers_to_use)
        ):
            self.projections[f'proj_{dit_layer}_to_{vjepa_layer}'] = ProjectionMLP(
                input_dim=config.dit_hidden_dim,
                hidden_dim=config.projection_hidden_dim,
                output_dim=config.vjepa_feature_dim
            )

        # HASTE scheduler
        self.haste_scheduler = HASTEScheduler(config)

        # Loss tracking
        self.loss_history: Dict[str, List[float]] = {
            'total': [],
            'per_layer': {f'layer_{l}': [] for l in config.dit_layers_to_align}
        }

    def _compute_alignment_loss(
        self,
        dit_features: torch.Tensor,
        vjepa_features: torch.Tensor,
        alignment_type: str = "cosine"
    ) -> torch.Tensor:
        """Compute alignment loss between projected DiT and V-JEPA features."""
        # Handle sequence length mismatch
        if dit_features.shape[1] != vjepa_features.shape[1]:
            target_len = min(dit_features.shape[1], vjepa_features.shape[1])

            if dit_features.shape[1] > target_len:
                dit_features = dit_features.transpose(1, 2)
                dit_features = F.adaptive_avg_pool1d(dit_features, target_len)
                dit_features = dit_features.transpose(1, 2)

            if vjepa_features.shape[1] > target_len:
                vjepa_features = vjepa_features.transpose(1, 2)
                vjepa_features = F.adaptive_avg_pool1d(vjepa_features, target_len)
                vjepa_features = vjepa_features.transpose(1, 2)

        # Normalize
        if self.config.normalize_features:
            dit_features = F.normalize(dit_features, dim=-1)
            vjepa_features = F.normalize(vjepa_features, dim=-1)

        # Compute loss
        if alignment_type == "cosine":
            cos_sim = (dit_features * vjepa_features).sum(dim=-1)
            loss = (1 - cos_sim).mean()
        elif alignment_type == "mse":
            loss = F.mse_loss(dit_features, vjepa_features)
        elif alignment_type == "smooth_l1":
            loss = F.smooth_l1_loss(dit_features, vjepa_features)
        else:
            raise ValueError(f"Unknown alignment type: {alignment_type}")

        return loss

    def forward(
        self,
        dit_features: Dict[str, torch.Tensor],
        vjepa_features: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        return_per_layer: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute REPA loss.

        Args:
            dit_features: Dict mapping layer names to [B, N, D_dit] tensors
            vjepa_features: Dict mapping layer names to [B, M, D_vjepa] tensors
            step: Current training step (for HASTE scheduling)
            return_per_layer: Return per-layer losses

        Returns:
            Dict with 'total_repa_loss', 'repa_weight', 'per_layer_losses', etc.
        """
        if step is not None:
            self.haste_scheduler.update(step)

        haste_weight = self.haste_scheduler.get_weight()

        # Early return if REPA is terminated
        if haste_weight == 0.0:
            device = next(iter(dit_features.values())).device
            return {
                'total_repa_loss': torch.tensor(0.0, device=device),
                'repa_weight': 0.0,
                'haste_phase': self.haste_scheduler.get_phase(),
                'per_layer_losses': {} if return_per_layer else None
            }

        # Compute per-layer losses
        layer_losses = {}
        total_loss = 0.0

        for i, (dit_layer, vjepa_layer) in enumerate(
            zip(self.config.dit_layers_to_align, self.config.vjepa_layers_to_use)
        ):
            dit_key = f'layer_{dit_layer}'
            vjepa_key = f'layer_{vjepa_layer}'

            if dit_key not in dit_features:
                warnings.warn(f"DiT layer {dit_key} not found")
                continue
            if vjepa_key not in vjepa_features:
                warnings.warn(f"V-JEPA layer {vjepa_key} not found")
                continue

            dit_feat = dit_features[dit_key]
            vjepa_feat = vjepa_features[vjepa_key]

            # Project DiT features
            proj_key = f'proj_{dit_layer}_to_{vjepa_layer}'
            projected_dit = self.projections[proj_key](dit_feat)

            # Compute alignment loss
            layer_loss = self._compute_alignment_loss(
                projected_dit,
                vjepa_feat.detach(),
                self.config.alignment_type
            )

            # Apply per-layer weight
            layer_weight = self.config.per_layer_weights[i]
            weighted_layer_loss = layer_loss * layer_weight

            layer_losses[dit_key] = weighted_layer_loss
            total_loss = total_loss + weighted_layer_loss

        # Apply HASTE weight and base weight
        total_loss = total_loss * haste_weight * self.config.base_repa_weight
        final_loss = total_loss * self.config.repa_relative_weight

        # Update history
        self.loss_history['total'].append(final_loss.item())

        result = {
            'total_repa_loss': final_loss,
            'raw_repa_loss': total_loss,
            'repa_weight': haste_weight,
            'haste_phase': self.haste_scheduler.get_phase()
        }

        if return_per_layer:
            result['per_layer_losses'] = layer_losses

        return result

    def get_loss_stats(self) -> Dict[str, float]:
        """Get statistics about recent losses."""
        stats = {}
        if self.loss_history['total']:
            recent = self.loss_history['total'][-100:]
            stats['repa_mean'] = sum(recent) / len(recent)
            stats['repa_current'] = recent[-1]
        stats.update(self.haste_scheduler.get_progress_info())
        return stats


class REPAIntegration:
    """Helper class for clean pipeline integration."""

    def __init__(self, vjepa_backbone: nn.Module, repa_config: REPAConfig):
        self.vjepa_backbone = vjepa_backbone
        self.repa_loss = REPALoss(repa_config)
        self.config = repa_config

    def to(self, device: torch.device) -> 'REPAIntegration':
        self.repa_loss = self.repa_loss.to(device)
        self.vjepa_backbone = self.vjepa_backbone.to(device)
        return self

    def compute_repa_loss(
        self,
        video: torch.Tensor,
        dit_intermediate_features: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """Compute REPA loss for a training batch."""
        with torch.no_grad():
            vjepa_features = self.vjepa_backbone(video, return_all_layers=True)

        return self.repa_loss(
            dit_features=dit_intermediate_features,
            vjepa_features=vjepa_features['intermediate'],
            step=step
        )

    def is_active(self) -> bool:
        return self.repa_loss.haste_scheduler.is_active()

    def get_trainable_parameters(self):
        return self.repa_loss.projections.parameters()


def create_repa_loss(
    dit_hidden_dim: int,
    vjepa_feature_dim: int = 1024,
    dit_layers: List[int] = None,
    vjepa_layers: List[int] = None,
    total_training_steps: int = 100000,
    haste_termination_ratio: float = 0.4,
    **kwargs
) -> REPALoss:
    """Factory function to create REPA loss."""
    if dit_layers is None:
        dit_layers = [6, 8, 10]
    if vjepa_layers is None:
        vjepa_layers = [18, 20, 22]

    config = REPAConfig(
        dit_hidden_dim=dit_hidden_dim,
        vjepa_feature_dim=vjepa_feature_dim,
        dit_layers_to_align=dit_layers,
        vjepa_layers_to_use=vjepa_layers,
        total_training_steps=total_training_steps,
        haste_termination_ratio=haste_termination_ratio
    )

    return REPALoss(config)


if __name__ == "__main__":
    print("Testing REPA Loss with HASTE...")

    config = REPAConfig(
        dit_hidden_dim=512,
        vjepa_feature_dim=1024,
        dit_layers_to_align=[4, 5, 6],
        vjepa_layers_to_use=[3, 4, 5],
        total_training_steps=10000,
        haste_termination_ratio=0.4,
        haste_warmup_ratio=0.1
    )

    repa_loss = REPALoss(config)

    B, N_dit, N_vjepa = 2, 64, 128
    dit_features = {
        'layer_4': torch.randn(B, N_dit, 512),
        'layer_5': torch.randn(B, N_dit, 512),
        'layer_6': torch.randn(B, N_dit, 512)
    }
    vjepa_features = {
        'layer_3': torch.randn(B, N_vjepa, 1024),
        'layer_4': torch.randn(B, N_vjepa, 1024),
        'layer_5': torch.randn(B, N_vjepa, 1024)
    }

    print("\n=== HASTE Schedule Test ===")
    test_steps = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000]

    for step in test_steps:
        output = repa_loss(dit_features, vjepa_features, step=step)
        print(f"Step {step:5d}: weight={output['repa_weight']:.3f}, "
              f"phase={output['haste_phase']:10s}, "
              f"loss={output['total_repa_loss'].item():.6f}")

    print("\nREPA Loss test passed!")
