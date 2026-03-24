"""
C-JEPA: Causal JEPA with Object-Level Masking.

Upgrades from standard JEPA by:
- Using object-trajectory masking (mask entire object trajectories, not random patches)
- Predicting inter-object interactions
- Supporting latent causal intervention formulation
- Enabling counterfactual prediction ("what if object X wasn't there?")

Key insight: Objects in driving scenes have causal relationships - masking random patches
breaks these relationships. C-JEPA masks object trajectories to learn causal structure.

References:
- C-JEPA paper: Object-level masking for causal understanding
- 20% improvement on counterfactual reasoning benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from einops import rearrange
import math


@dataclass
class CausalJEPAConfig:
    """Configuration for C-JEPA."""
    # Object slots
    max_objects: int = 32
    slot_dim: int = 256
    num_slot_iterations: int = 3
    slot_mlp_hidden_dim: int = 512

    # Input
    input_dim: int = 256

    # Attention
    num_heads: int = 8
    dropout: float = 0.1

    # Interactions
    interaction_dim: int = 128
    num_interaction_heads: int = 4
    max_interaction_distance: float = 50.0

    # Trajectory prediction
    predictor_dim: int = 512
    predictor_depth: int = 4
    prediction_horizon: int = 8

    # Masking
    min_objects_masked: int = 2
    max_objects_masked: int = 8
    mask_ratio: float = 0.75

    # Counterfactual
    enable_counterfactual: bool = True
    num_counterfactuals: int = 4

    # Loss weights
    slot_reconstruction_weight: float = 1.0
    trajectory_prediction_weight: float = 0.5
    interaction_weight: float = 0.3
    contrastive_weight: float = 0.2
    counterfactual_weight: float = 0.2


def get_default_causal_jepa_config() -> CausalJEPAConfig:
    return CausalJEPAConfig()


def get_minimal_causal_jepa_config() -> CausalJEPAConfig:
    return CausalJEPAConfig(
        max_objects=16, slot_dim=128, num_slot_iterations=2,
        slot_mlp_hidden_dim=256, predictor_dim=256, predictor_depth=2,
        prediction_horizon=4
    )


class SlotAttention(nn.Module):
    """Slot Attention mechanism for object discovery."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Projections
        self.project_k = nn.Linear(input_dim, slot_dim)
        self.project_v = nn.Linear(input_dim, slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.scale = slot_dim ** -0.5

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, N, D] input features

        Returns:
            slots: [B, num_slots, slot_dim]
            attn_weights: [B, num_slots, N] attention weights
        """
        B, N, D = inputs.shape

        # Initialize slots
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            B, self.num_slots, self.slot_dim, device=inputs.device
        )

        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Attention
            attn = torch.einsum('bnd,bsd->bsn', k, q) * self.scale  # [B, num_slots, N]
            attn = attn.softmax(dim=1)  # Softmax over slots

            # Weighted sum
            updates = torch.einsum('bsn,bnd->bsd', attn, v)

            # GRU update
            slots = self.gru(
                updates.flatten(0, 1),
                slots_prev.flatten(0, 1)
            ).view(B, self.num_slots, self.slot_dim)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class ObjectSlotEncoder(nn.Module):
    """Encode video features into object slots."""

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        max_objects: int,
        num_iterations: int = 3,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.max_objects = max_objects

        # Slot attention
        self.slot_attention = SlotAttention(
            num_slots=max_objects,
            slot_dim=slot_dim,
            input_dim=input_dim,
            num_iterations=num_iterations,
            hidden_dim=hidden_dim
        )

        # Position predictor
        self.position_head = nn.Linear(slot_dim, 4)  # x, y, w, h

        # Existence predictor
        self.existence_head = nn.Linear(slot_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, T, N, D] video features
            masks: Optional [B, T, num_objects, H, W] segmentation masks

        Returns:
            Dict with slots, positions, existence_probs
        """
        B, T, N, D = features.shape

        # Process each frame
        all_slots = []
        all_positions = []
        all_existence = []

        for t in range(T):
            frame_features = features[:, t]  # [B, N, D]
            slots, attn = self.slot_attention(frame_features)  # [B, max_objects, slot_dim]

            positions = self.position_head(slots)  # [B, max_objects, 4]
            existence = torch.sigmoid(self.existence_head(slots).squeeze(-1))  # [B, max_objects]

            all_slots.append(slots)
            all_positions.append(positions)
            all_existence.append(existence)

        slots = torch.stack(all_slots, dim=1)  # [B, T, max_objects, slot_dim]
        positions = torch.stack(all_positions, dim=1)  # [B, T, max_objects, 4]
        existence_probs = torch.stack(all_existence, dim=1)  # [B, T, max_objects]

        return {
            'slots': slots,
            'positions': positions,
            'existence_probs': existence_probs
        }


class InteractionEncoder(nn.Module):
    """Encode pairwise object interactions."""

    def __init__(
        self,
        slot_dim: int,
        interaction_dim: int,
        num_heads: int = 4,
        max_distance: float = 50.0
    ):
        super().__init__()
        self.interaction_dim = interaction_dim
        self.max_distance = max_distance

        # Pairwise interaction network
        self.interaction_net = nn.Sequential(
            nn.Linear(slot_dim * 2 + 4, interaction_dim * 2),
            nn.ReLU(),
            nn.Linear(interaction_dim * 2, interaction_dim)
        )

        # Interaction type classifier
        self.interaction_type = nn.Linear(interaction_dim, 8)  # 8 interaction types

        # Causal influence estimator
        self.causal_influence = nn.Linear(interaction_dim, 1)

    def forward(
        self,
        slots: torch.Tensor,
        positions: torch.Tensor,
        existence_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            slots: [B, num_objects, slot_dim]
            positions: [B, num_objects, 4]
            existence_mask: [B, num_objects] binary mask

        Returns:
            Dict with interaction_features, interaction_types, causal_influence
        """
        B, N, D = slots.shape

        # Compute pairwise features
        slots_i = slots.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        slots_j = slots.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]

        # Relative positions
        pos_i = positions.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 4]
        pos_j = positions.unsqueeze(1).expand(-1, N, -1, -1)
        rel_pos = pos_i[:, :, :, :2] - pos_j[:, :, :, :2]  # [B, N, N, 2]

        # Distance
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B, N, N, 1]
        distance_norm = distance / self.max_distance

        # Concatenate features
        pair_features = torch.cat([
            slots_i, slots_j,
            rel_pos, distance_norm, distance_norm.clamp(max=1.0)
        ], dim=-1)

        # Compute interactions
        interaction_features = self.interaction_net(pair_features)  # [B, N, N, interaction_dim]

        # Mask non-existent objects
        mask = existence_mask.unsqueeze(2) * existence_mask.unsqueeze(1)
        interaction_features = interaction_features * mask.unsqueeze(-1)

        # Interaction types and causal influence
        interaction_types = self.interaction_type(interaction_features)
        causal_influence = torch.sigmoid(self.causal_influence(interaction_features).squeeze(-1))

        return {
            'interaction_features': interaction_features,
            'interaction_types': interaction_types,
            'causal_influence': causal_influence * mask
        }


class TrajectoryPredictor(nn.Module):
    """Predict future object trajectories."""

    def __init__(
        self,
        slot_dim: int,
        predictor_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        prediction_horizon: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.prediction_horizon = prediction_horizon

        # Input projection
        self.input_proj = nn.Linear(slot_dim, predictor_dim)

        # Positional encoding for future steps
        self.pos_encoding = nn.Parameter(torch.randn(1, prediction_horizon, predictor_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projections
        self.output_proj = nn.Linear(predictor_dim, slot_dim)
        self.position_head = nn.Linear(predictor_dim, 4)
        self.uncertainty_head = nn.Linear(predictor_dim, slot_dim)

    def forward(
        self,
        context_slots: torch.Tensor,
        interaction_features: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context_slots: [B, T_ctx, num_objects, slot_dim]
            interaction_features: Optional [B, num_objects, num_objects, interaction_dim]
            num_steps: Number of future steps to predict

        Returns:
            Dict with predicted_slots, predicted_positions, uncertainty
        """
        num_steps = num_steps or self.prediction_horizon
        B, T_ctx, num_objects, slot_dim = context_slots.shape

        # Project context
        context_flat = rearrange(context_slots, 'b t n d -> (b n) t d')
        context_proj = self.input_proj(context_flat)

        # Query for future predictions
        query = self.pos_encoding[:, :num_steps].expand(B * num_objects, -1, -1)

        # Generate causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(num_steps).to(context_proj.device)

        # Predict
        predicted = self.transformer_decoder(query, context_proj, tgt_mask=tgt_mask)

        # Project outputs
        predicted_slots = self.output_proj(predicted)
        predicted_slots = rearrange(predicted_slots, '(b n) t d -> b t n d', b=B, n=num_objects)

        predicted_positions = self.position_head(predicted)
        predicted_positions = rearrange(predicted_positions, '(b n) t d -> b t n d', b=B, n=num_objects)

        uncertainty = F.softplus(self.uncertainty_head(predicted))
        uncertainty = rearrange(uncertainty, '(b n) t d -> b t n d', b=B, n=num_objects)

        return {
            'predicted_slots': predicted_slots,
            'predicted_positions': predicted_positions,
            'uncertainty': uncertainty
        }


class CausalJEPAPredictor(nn.Module):
    """
    Causal JEPA Predictor with object-level masking.

    Masks entire object trajectories (not random patches) to learn
    causal relationships between objects in driving scenes.
    """

    def __init__(self, config: Optional[CausalJEPAConfig] = None):
        super().__init__()
        self.config = config or get_default_causal_jepa_config()

        # Object slot encoder
        self.slot_encoder = ObjectSlotEncoder(
            input_dim=self.config.input_dim,
            slot_dim=self.config.slot_dim,
            max_objects=self.config.max_objects,
            num_iterations=self.config.num_slot_iterations,
            hidden_dim=self.config.slot_mlp_hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        )

        # Interaction encoder
        self.interaction_encoder = InteractionEncoder(
            slot_dim=self.config.slot_dim,
            interaction_dim=self.config.interaction_dim,
            num_heads=self.config.num_interaction_heads,
            max_distance=self.config.max_interaction_distance
        )

        # Trajectory predictor
        self.trajectory_predictor = TrajectoryPredictor(
            slot_dim=self.config.slot_dim,
            predictor_dim=self.config.predictor_dim,
            num_layers=self.config.predictor_depth,
            num_heads=self.config.num_heads,
            prediction_horizon=self.config.prediction_horizon,
            dropout=self.config.dropout
        )

        # Counterfactual predictor
        if self.config.enable_counterfactual:
            self.counterfactual_encoder = nn.Sequential(
                nn.Linear(self.config.slot_dim * 2, self.config.predictor_dim),
                nn.ReLU(),
                nn.Linear(self.config.predictor_dim, self.config.slot_dim)
            )

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.config.slot_dim) * 0.02)

    def forward(
        self,
        features: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        object_mask_indices: Optional[torch.Tensor] = None,
        return_interactions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of C-JEPA predictor.

        Args:
            features: [B, T, N, D] video features
            masks: Optional object segmentation masks
            object_mask_indices: Indices of objects to mask
            return_interactions: Compute interaction features

        Returns:
            Dict with slots, predictions, interactions, etc.
        """
        B, T, N, D = features.shape

        # Encode object slots
        slot_output = self.slot_encoder(features, masks)
        slots = slot_output['slots']
        existence_probs = slot_output['existence_probs']
        positions = slot_output['positions']

        # Generate mask indices if not provided
        if object_mask_indices is None:
            object_mask_indices = self._sample_mask_indices(B, existence_probs)

        # Apply trajectory masking
        masked_slots, mask_targets = self._apply_trajectory_mask(slots, object_mask_indices)

        # Compute interactions
        interactions = None
        if return_interactions:
            avg_slots = slots.mean(dim=1)
            avg_positions = positions.mean(dim=1)
            avg_existence = (existence_probs.mean(dim=1) > 0.5).float()
            interactions = self.interaction_encoder(avg_slots, avg_positions, avg_existence)

        # Predict masked trajectories
        predictions = self.trajectory_predictor(
            masked_slots,
            interaction_features=interactions['interaction_features'] if interactions else None
        )

        return {
            'slots': slots,
            'masked_slots': masked_slots,
            'predicted_slots': predictions['predicted_slots'],
            'predictions': predictions,
            'mask_targets': mask_targets,
            'object_mask_indices': object_mask_indices,
            'interactions': interactions,
            'existence_probs': existence_probs,
            'positions': positions
        }

    def predict_counterfactual(
        self,
        features: torch.Tensor,
        removed_object_indices: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict what would happen if specified objects weren't there."""
        if not self.config.enable_counterfactual:
            raise ValueError("Counterfactual prediction is disabled")

        B, T, N, D = features.shape

        # Get original predictions
        original_output = self.forward(features, masks, return_interactions=True)
        original_slots = original_output['slots']

        # Create counterfactual slots (zero out removed objects)
        counterfactual_slots = original_slots.clone()
        batch_indices = torch.arange(B, device=features.device).unsqueeze(1)
        counterfactual_slots[batch_indices, :, removed_object_indices] = 0.0

        # Predict counterfactual trajectories
        counterfactual_predictions = self.trajectory_predictor(counterfactual_slots)

        # Estimate causal effect
        causal_effect = (
            original_output['predictions']['predicted_slots'] -
            counterfactual_predictions['predicted_slots']
        ).pow(2).mean(dim=(1, 2, 3))

        return {
            'original_predictions': original_output['predictions'],
            'counterfactual_predictions': counterfactual_predictions,
            'causal_effect': causal_effect,
            'removed_objects': removed_object_indices
        }

    def _sample_mask_indices(self, batch_size: int, existence_probs: torch.Tensor) -> torch.Tensor:
        """Sample object indices to mask."""
        device = existence_probs.device
        avg_existence = existence_probs.mean(dim=1)
        max_masked = self.config.max_objects_masked

        mask_indices = []
        for b in range(batch_size):
            probs = avg_existence[b]
            n_mask = min(
                torch.randint(self.config.min_objects_masked, max_masked + 1, (1,)).item(),
                (probs > 0.5).sum().item()
            )
            n_mask = max(n_mask, self.config.min_objects_masked)

            if probs.sum() > 0:
                indices = torch.multinomial(probs + 1e-8, min(n_mask, len(probs)), replacement=False)
            else:
                indices = torch.arange(min(n_mask, self.config.max_objects), device=device)

            padded = torch.zeros(max_masked, dtype=torch.long, device=device)
            padded[:len(indices)] = indices
            mask_indices.append(padded)

        return torch.stack(mask_indices)

    def _apply_trajectory_mask(
        self,
        slots: torch.Tensor,
        mask_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply trajectory masking to object slots."""
        B, T, num_objects, slot_dim = slots.shape
        device = slots.device

        mask_targets = slots.clone()
        masked_slots = slots.clone()

        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        for t in range(T):
            masked_slots[batch_indices, t, mask_indices] = self.mask_token.expand(
                B, -1, slot_dim
            )[:, :mask_indices.shape[1]]

        return masked_slots, mask_targets


class CausalJEPALoss(nn.Module):
    """Loss functions for C-JEPA training."""

    def __init__(self, config: Optional[CausalJEPAConfig] = None):
        super().__init__()
        self.config = config or get_default_causal_jepa_config()
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all C-JEPA losses."""
        losses = {}

        # Slot prediction loss
        if 'predicted_slots' in outputs and 'mask_targets' in outputs:
            losses['slot_prediction'] = self._slot_prediction_loss(
                outputs['predicted_slots'],
                outputs['mask_targets'],
                outputs['object_mask_indices']
            ) * self.config.slot_reconstruction_weight

        # Trajectory loss
        if 'predictions' in outputs:
            losses['trajectory'] = self._trajectory_loss(
                outputs['predictions'],
                outputs.get('mask_targets')
            ) * self.config.trajectory_prediction_weight

        # Contrastive loss
        if 'slots' in outputs and 'predicted_slots' in outputs:
            losses['contrastive'] = self._contrastive_loss(
                outputs['slots'],
                outputs['predicted_slots']
            ) * self.config.contrastive_weight

        # Counterfactual loss
        if 'counterfactual_predictions' in outputs:
            losses['counterfactual'] = self._counterfactual_loss(
                outputs
            ) * self.config.counterfactual_weight

        losses['total'] = sum(losses.values())
        return losses

    def _slot_prediction_loss(
        self,
        predicted_slots: torch.Tensor,
        target_slots: torch.Tensor,
        mask_indices: torch.Tensor
    ) -> torch.Tensor:
        """Masked slot reconstruction loss."""
        B, T, num_objects, slot_dim = target_slots.shape
        device = predicted_slots.device

        T_pred = predicted_slots.shape[1]
        target_aligned = target_slots[:, -T_pred:]

        mask = torch.zeros(B, num_objects, device=device)
        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        mask[batch_indices, mask_indices] = 1.0
        mask = mask.unsqueeze(1).unsqueeze(-1)

        diff = (predicted_slots - target_aligned) ** 2
        masked_diff = diff * mask

        return masked_diff.sum() / (mask.sum() * slot_dim + 1e-8)

    def _trajectory_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Trajectory prediction loss."""
        predicted_positions = predictions['predicted_positions']
        uncertainty = predictions['uncertainty']

        if targets is None:
            smoothness = (predicted_positions[:, 1:] - predicted_positions[:, :-1]).pow(2).mean()
            uncertainty_reg = uncertainty.mean()
            return smoothness + 0.1 * uncertainty_reg

        diff = (predictions['predicted_slots'] - targets) ** 2
        nll = diff / (2 * uncertainty.pow(2) + 1e-8) + 0.5 * torch.log(uncertainty.pow(2) + 1e-8)
        return nll.mean()

    def _contrastive_loss(
        self,
        slots: torch.Tensor,
        predicted_slots: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        B, T, num_objects, slot_dim = slots.shape
        T_pred = predicted_slots.shape[1]

        positive_slots = slots[:, -T_pred:]

        pred_flat = rearrange(predicted_slots, 'b t n d -> (b t n) d')
        pos_flat = rearrange(positive_slots, 'b t n d -> (b t n) d')

        pred_norm = F.normalize(pred_flat, dim=-1)
        pos_norm = F.normalize(pos_flat, dim=-1)

        sim = torch.mm(pred_norm, pos_norm.t()) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)

        return F.cross_entropy(sim, labels)

    def _counterfactual_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Counterfactual consistency loss."""
        original_pred = outputs['original_predictions']['predicted_slots']
        cf_pred = outputs['counterfactual_predictions']['predicted_slots']
        causal_effect = outputs['causal_effect']

        difference = (original_pred - cf_pred).pow(2).mean(dim=(1, 2, 3))
        consistency = (difference - causal_effect).pow(2).mean()
        magnitude_reg = difference.pow(2).mean()

        return consistency + 0.1 * magnitude_reg


def create_causal_jepa_predictor(config: Optional[CausalJEPAConfig] = None) -> CausalJEPAPredictor:
    """Factory function to create C-JEPA predictor."""
    return CausalJEPAPredictor(config)


if __name__ == "__main__":
    print("Testing C-JEPA (Causal JEPA)...")

    config = get_minimal_causal_jepa_config()
    predictor = CausalJEPAPredictor(config)
    loss_fn = CausalJEPALoss(config)

    B, T, N, D = 2, 8, 64, config.input_dim
    features = torch.randn(B, T, N, D)

    print(f"Input shape: {features.shape}")

    output = predictor(features)
    print(f"Predicted slots: {output['predicted_slots'].shape}")

    losses = loss_fn(output)
    print(f"Losses: { {k: v.item() for k, v in losses.items()} }")

    # Test counterfactual
    removed = torch.tensor([[0, 1], [2, 3]])
    cf_output = predictor.predict_counterfactual(features, removed)
    print(f"Causal effect: {cf_output['causal_effect']}")

    print("\nC-JEPA test passed!")
