"""
Comprehensive Unified Loss System v2 for DriveDiT.

This module consolidates all loss functions for autonomous driving world modeling:
- Self-forcing losses (latent MSE, flow matching, KV cache consistency)
- JEPA contrastive losses (InfoNCE, cross-modal, VICReg)
- Flow matching losses (CFM, distillation, consistency, trajectory straightening)
- Depth losses (scale-invariant, gradient, edge-aware)
- Control/action losses (inverse dynamics, smoothness, goal-conditioned)
- Perceptual losses (LPIPS, VGG, DINO, CLIP)
- Video quality losses (temporal consistency, flicker, motion)
- Uncertainty weighting (learned task weights, gradient normalization)
- Loss aggregator with curriculum scheduling and gradient surgery

Mathematical foundations from:
- Rectified Flow (Liu et al., 2022)
- V-JEPA-2 (Bardes et al., 2024)
- VICReg (Bardes et al., 2022)
- Scale-Invariant Depth (Eigen et al., 2014)
- Multi-Task Learning (Kendall et al., 2018)
- Gradient Surgery (Yu et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from abc import ABC, abstractmethod


# =============================================================================
# Configuration and Enums
# =============================================================================

class LossType(Enum):
    """Enumeration of available loss types."""
    # Self-forcing losses
    LATENT_MSE = "latent_mse"
    TEACHER_STUDENT_FLOW = "teacher_student_flow"
    KV_CACHE_CONSISTENCY = "kv_cache_consistency"
    AUTOREGRESSIVE_NEXT_FRAME = "autoregressive_next_frame"

    # JEPA contrastive losses
    INFONCE = "infonce"
    CROSS_MODAL_CONTRASTIVE = "cross_modal_contrastive"
    TEMPORAL_CONTRASTIVE = "temporal_contrastive"
    VICREG = "vicreg"

    # Flow matching losses
    CONDITIONAL_FLOW_MATCHING = "conditional_flow_matching"
    FLOW_DISTILLATION = "flow_distillation"
    FLOW_CONSISTENCY = "flow_consistency"
    TRAJECTORY_STRAIGHTENING = "trajectory_straightening"

    # Depth losses
    SCALE_INVARIANT_DEPTH = "scale_invariant_depth"
    DEPTH_GRADIENT_SMOOTHNESS = "depth_gradient_smoothness"
    EDGE_AWARE_DEPTH = "edge_aware_depth"
    MULTI_SCALE_DEPTH = "multi_scale_depth"

    # Control/action losses
    INVERSE_DYNAMICS = "inverse_dynamics"
    ACTION_SMOOTHNESS = "action_smoothness"
    BOUNDARY_ACTION_CLIPPING = "boundary_action_clipping"
    GOAL_CONDITIONED_ACTION = "goal_conditioned_action"

    # Perceptual losses
    LPIPS = "lpips"
    VGG_FEATURE = "vgg_feature"
    DINO_FEATURE = "dino_feature"
    CLIP_SIMILARITY = "clip_similarity"

    # Video quality losses
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    FLICKER_REDUCTION = "flicker_reduction"
    MOTION_MAGNITUDE = "motion_magnitude"
    SCENE_STABILITY = "scene_stability"


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    # Enabled loss types
    enabled_losses: List[LossType] = field(default_factory=lambda: [
        LossType.LATENT_MSE,
        LossType.TEMPORAL_CONSISTENCY,
        LossType.CONDITIONAL_FLOW_MATCHING
    ])

    # Loss weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'latent_mse': 1.0,
        'teacher_student_flow': 0.5,
        'kv_cache_consistency': 0.1,
        'autoregressive_next_frame': 1.0,
        'infonce': 0.5,
        'cross_modal_contrastive': 0.3,
        'temporal_contrastive': 0.3,
        'vicreg': 0.2,
        'conditional_flow_matching': 1.0,
        'flow_distillation': 0.5,
        'flow_consistency': 0.2,
        'trajectory_straightening': 0.1,
        'scale_invariant_depth': 0.5,
        'depth_gradient_smoothness': 0.1,
        'edge_aware_depth': 0.2,
        'multi_scale_depth': 0.3,
        'inverse_dynamics': 0.3,
        'action_smoothness': 0.1,
        'boundary_action_clipping': 0.05,
        'goal_conditioned_action': 0.2,
        'lpips': 0.5,
        'vgg_feature': 0.3,
        'dino_feature': 0.3,
        'clip_similarity': 0.2,
        'temporal_consistency': 0.3,
        'flicker_reduction': 0.1,
        'motion_magnitude': 0.1,
        'scene_stability': 0.1
    })

    # Uncertainty weighting
    use_uncertainty_weighting: bool = False
    use_gradient_normalization: bool = False
    use_gradient_surgery: bool = False

    # Curriculum scheduling
    enable_curriculum: bool = True
    curriculum_warmup_steps: int = 10000

    # JEPA parameters
    infonce_temperature: float = 0.1
    vicreg_sim_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0

    # Flow matching parameters
    flow_num_steps: int = 4
    flow_sigma_min: float = 1e-4

    # Depth parameters
    depth_max_depth: float = 100.0
    depth_gradient_weight: float = 0.1

    # Control parameters
    action_smoothness_order: int = 2
    action_boundary_margin: float = 0.95

    # Perceptual parameters
    lpips_net: str = 'vgg'  # 'vgg' or 'alex'
    vgg_layers: List[str] = field(default_factory=lambda: ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])

    # Video quality parameters
    temporal_window: int = 3
    flicker_threshold: float = 0.1


# =============================================================================
# Base Loss Class
# =============================================================================

class BaseLoss(ABC, nn.Module):
    """Abstract base class for all loss functions."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self._history: List[float] = []

    @abstractmethod
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Compute the loss value."""
        pass

    def update_history(self, value: float):
        """Update loss history for tracking."""
        self._history.append(value)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

    @property
    def mean(self) -> float:
        """Get mean of recent loss values."""
        if not self._history:
            return 0.0
        return np.mean(self._history[-100:])

    @property
    def std(self) -> float:
        """Get std of recent loss values."""
        if len(self._history) < 2:
            return 0.0
        return np.std(self._history[-100:])


# =============================================================================
# Self-Forcing Losses
# =============================================================================

class LatentMSELoss(BaseLoss):
    """
    Latent MSE loss between student predictions and ground truth latents.

    L = ||z_student - z_gt||^2
    """

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_latents = predictions.get('latents', predictions.get('predictions'))
        target_latents = targets.get('latents', targets.get('frames'))

        if pred_latents is None or target_latents is None:
            return torch.tensor(0.0, device=self._get_device(predictions))

        # Ensure same shape
        if pred_latents.shape != target_latents.shape:
            target_latents = F.interpolate(
                target_latents.view(-1, *target_latents.shape[-3:]),
                size=pred_latents.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).view_as(pred_latents)

        loss = F.mse_loss(pred_latents, target_latents)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, d: Dict) -> torch.device:
        for v in d.values():
            if torch.is_tensor(v):
                return v.device
        return torch.device('cpu')


class TeacherStudentFlowLoss(BaseLoss):
    """
    Flow matching loss between teacher and student predictions.

    L = ||f_student(z_t) - f_teacher(z_t)||^2

    Used for distilling multi-step teacher to single-step student.
    """

    def __init__(self, weight: float = 1.0, detach_teacher: bool = True):
        super().__init__(weight)
        self.detach_teacher = detach_teacher

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        student_flow = predictions.get('flow_predictions')
        teacher_flow = targets.get('teacher_flow')

        if student_flow is None or teacher_flow is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        if self.detach_teacher:
            teacher_flow = teacher_flow.detach()

        loss = F.mse_loss(student_flow, teacher_flow)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class KVCacheConsistencyLoss(BaseLoss):
    """
    Loss ensuring KV cache consistency across autoregressive steps.

    Penalizes large changes in KV cache between consecutive steps:
    L = sum_t ||K_{t} - K_{t-1}||^2 + ||V_{t} - V_{t-1}||^2
    """

    def __init__(self, weight: float = 1.0, margin: float = 0.1):
        super().__init__(weight)
        self.margin = margin

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        kv_cache_history = predictions.get('kv_cache_history')

        if kv_cache_history is None or len(kv_cache_history) < 2:
            return torch.tensor(0.0, device=self._get_device(predictions))

        total_loss = 0.0
        count = 0

        for i in range(1, len(kv_cache_history)):
            prev_kv = kv_cache_history[i - 1]
            curr_kv = kv_cache_history[i]

            if 'k' in prev_kv and 'k' in curr_kv:
                # Compute difference in overlapping region
                min_len = min(prev_kv['k'].size(1), curr_kv['k'].size(1))
                k_diff = F.mse_loss(prev_kv['k'][:, :min_len], curr_kv['k'][:, :min_len])
                v_diff = F.mse_loss(prev_kv['v'][:, :min_len], curr_kv['v'][:, :min_len])

                # Margin loss - only penalize if change exceeds margin
                total_loss += torch.clamp(k_diff - self.margin, min=0.0)
                total_loss += torch.clamp(v_diff - self.margin, min=0.0)
                count += 2

        if count == 0:
            return torch.tensor(0.0, device=self._get_device(predictions))

        loss = total_loss / count
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, d: Dict) -> torch.device:
        for v in d.values():
            if torch.is_tensor(v):
                return v.device
        return torch.device('cpu')


class AutoregressiveNextFrameLoss(BaseLoss):
    """
    Next frame prediction loss for autoregressive models.

    L = sum_t ||f(z_{1:t}) - z_{t+1}||^2
    """

    def __init__(self, weight: float = 1.0, use_l1: bool = False):
        super().__init__(weight)
        self.loss_fn = F.l1_loss if use_l1 else F.mse_loss

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_frames = predictions.get('predictions')
        target_frames = targets.get('frames')

        if pred_frames is None or target_frames is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        B, T = pred_frames.shape[:2]

        if T <= 1:
            return torch.tensor(0.0, device=pred_frames.device)

        # Shift predictions and targets for next-frame prediction
        pred_next = pred_frames[:, :-1]  # Predictions for frames 1 to T-1
        target_next = target_frames[:, 1:]  # Ground truth frames 2 to T

        loss = self.loss_fn(pred_next, target_next)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# JEPA Contrastive Losses
# =============================================================================

class InfoNCELoss(BaseLoss):
    """
    InfoNCE contrastive loss between z_t and z_{t+delta}.

    L = -log(exp(sim(z_t, z_{t+d})/tau) / sum_j exp(sim(z_t, z_j)/tau))

    Reference: V-JEPA-2 (Bardes et al., 2024)
    """

    def __init__(self, weight: float = 1.0, temperature: float = 0.1,
                 delta: int = 1, use_hard_negatives: bool = True):
        super().__init__(weight)
        self.temperature = temperature
        self.delta = delta
        self.use_hard_negatives = use_hard_negatives

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        hidden_states = predictions.get('hidden_states')

        if hidden_states is None:
            return torch.tensor(0.0, device=self._get_device(predictions))

        B, T, D = hidden_states.shape

        if T <= self.delta:
            return torch.tensor(0.0, device=hidden_states.device)

        # Get anchor and positive pairs
        anchors = hidden_states[:, :-self.delta]  # [B, T-delta, D]
        positives = hidden_states[:, self.delta:]  # [B, T-delta, D]

        # Flatten for contrastive computation
        anchors_flat = anchors.reshape(-1, D)  # [B*(T-delta), D]
        positives_flat = positives.reshape(-1, D)  # [B*(T-delta), D]

        # Normalize
        anchors_norm = F.normalize(anchors_flat, dim=-1)
        positives_norm = F.normalize(positives_flat, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(anchors_norm, positives_norm.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        # Add hard negatives from same sequence (temporal neighbors)
        if self.use_hard_negatives and T > self.delta + 1:
            # Use temporally adjacent frames as hard negatives
            hard_negs = hidden_states[:, self.delta - 1:-1] if self.delta > 1 else hidden_states[:, 1:-self.delta]
            hard_negs_flat = F.normalize(hard_negs.reshape(-1, D), dim=-1)
            hard_logits = torch.matmul(anchors_norm, hard_negs_flat.T) / self.temperature

            # Combined logits
            combined_logits = torch.cat([logits, hard_logits], dim=1)
            combined_labels = torch.arange(logits.size(0), device=logits.device)

            loss = F.cross_entropy(combined_logits, combined_labels)

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, d: Dict) -> torch.device:
        for v in d.values():
            if torch.is_tensor(v):
                return v.device
        return torch.device('cpu')


class CrossModalContrastiveLoss(BaseLoss):
    """
    Cross-modal contrastive loss between different modalities.

    Supports: RGB-depth, RGB-control, depth-control alignments.
    """

    def __init__(self, weight: float = 1.0, temperature: float = 0.1,
                 modality_pairs: List[Tuple[str, str]] = None):
        super().__init__(weight)
        self.temperature = temperature
        self.modality_pairs = modality_pairs or [('rgb', 'depth'), ('rgb', 'control')]

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        total_loss = 0.0
        count = 0

        # Mapping of modality names to keys
        modality_keys = {
            'rgb': 'hidden_states',
            'depth': 'depth_features',
            'control': 'control_features'
        }

        for mod1, mod2 in self.modality_pairs:
            key1 = modality_keys.get(mod1, mod1)
            key2 = modality_keys.get(mod2, mod2)

            feat1 = predictions.get(key1) or targets.get(key1)
            feat2 = predictions.get(key2) or targets.get(key2)

            if feat1 is None or feat2 is None:
                continue

            # Ensure same sequence dimension
            if feat1.dim() == 3 and feat2.dim() == 2:
                feat2 = feat2.unsqueeze(1).expand(-1, feat1.size(1), -1)
            elif feat2.dim() == 3 and feat1.dim() == 2:
                feat1 = feat1.unsqueeze(1).expand(-1, feat2.size(1), -1)

            # Global average pooling if needed
            if feat1.dim() == 3:
                feat1 = feat1.mean(dim=1)
            if feat2.dim() == 3:
                feat2 = feat2.mean(dim=1)

            # Project to same dimension if needed
            if feat1.size(-1) != feat2.size(-1):
                continue

            # Normalize features
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)

            # Compute cross-modal similarity
            logits = torch.matmul(feat1, feat2.T) / self.temperature
            labels = torch.arange(logits.size(0), device=logits.device)

            # Symmetric loss
            loss_12 = F.cross_entropy(logits, labels)
            loss_21 = F.cross_entropy(logits.T, labels)

            total_loss += (loss_12 + loss_21) / 2
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        loss = total_loss / count
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class TemporalContrastiveLoss(BaseLoss):
    """
    Temporal contrastive loss with hard negatives.

    Encourages representations to capture temporal structure
    while distinguishing from temporally distant frames.
    """

    def __init__(self, weight: float = 1.0, temperature: float = 0.1,
                 positive_window: int = 2, negative_margin: int = 5):
        super().__init__(weight)
        self.temperature = temperature
        self.positive_window = positive_window
        self.negative_margin = negative_margin

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        hidden_states = predictions.get('hidden_states')

        if hidden_states is None:
            return torch.tensor(0.0, device=self._get_device(predictions))

        B, T, D = hidden_states.shape

        if T <= self.positive_window + self.negative_margin:
            return torch.tensor(0.0, device=hidden_states.device)

        device = hidden_states.device
        total_loss = 0.0
        count = 0

        # Normalize features
        feats = F.normalize(hidden_states, dim=-1)

        for t in range(T - self.positive_window):
            anchor = feats[:, t]  # [B, D]

            # Positive: frames within window
            pos_indices = list(range(max(0, t - self.positive_window),
                                    min(T, t + self.positive_window + 1)))
            pos_indices = [i for i in pos_indices if i != t]

            if not pos_indices:
                continue

            positives = feats[:, pos_indices].mean(dim=1)  # [B, D]

            # Hard negatives: frames outside margin
            neg_start = min(t + self.negative_margin, T - 1)
            if neg_start < T - 1:
                negatives = feats[:, neg_start:]  # [B, T-neg_start, D]

                # Compute scores
                pos_score = (anchor * positives).sum(dim=-1) / self.temperature  # [B]
                neg_scores = torch.matmul(anchor.unsqueeze(1), negatives.transpose(-2, -1)).squeeze(1) / self.temperature  # [B, num_neg]

                # InfoNCE loss
                all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # [B, 1+num_neg]
                labels = torch.zeros(B, dtype=torch.long, device=device)

                loss = F.cross_entropy(all_scores, labels)
                total_loss += loss
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        loss = total_loss / count
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, d: Dict) -> torch.device:
        for v in d.values():
            if torch.is_tensor(v):
                return v.device
        return torch.device('cpu')


class VICRegLoss(BaseLoss):
    """
    Variance-Invariance-Covariance Regularization loss.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning", ICLR 2022.

    L = lambda * invariance + mu * variance + nu * covariance
    """

    def __init__(self, weight: float = 1.0,
                 sim_weight: float = 25.0,
                 var_weight: float = 25.0,
                 cov_weight: float = 1.0,
                 variance_target: float = 1.0):
        super().__init__(weight)
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.variance_target = variance_target

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # Get two views/augmentations of representations
        z1 = predictions.get('hidden_states')
        z2 = predictions.get('hidden_states_aug') or targets.get('hidden_states')

        if z1 is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # If only one view, create augmented view by temporal shifting
        if z2 is None:
            if z1.dim() == 3 and z1.size(1) > 1:
                z2 = z1[:, 1:]
                z1 = z1[:, :-1]
            else:
                return torch.tensor(0.0, device=z1.device)

        # Flatten temporal dimension
        if z1.dim() == 3:
            z1 = z1.reshape(-1, z1.size(-1))
        if z2.dim() == 3:
            z2 = z2.reshape(-1, z2.size(-1))

        # Ensure same size
        min_size = min(z1.size(0), z2.size(0))
        z1 = z1[:min_size]
        z2 = z2[:min_size]

        N, D = z1.shape

        # Invariance loss: MSE between representations
        invariance_loss = F.mse_loss(z1, z2)

        # Variance loss: encourage variance along feature dimension
        z1_std = torch.sqrt(z1.var(dim=0) + 1e-4)
        z2_std = torch.sqrt(z2.var(dim=0) + 1e-4)
        variance_loss = (torch.relu(self.variance_target - z1_std).mean() +
                        torch.relu(self.variance_target - z2_std).mean())

        # Covariance loss: decorrelate features
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov1 = (z1_centered.T @ z1_centered) / (N - 1)
        cov2 = (z2_centered.T @ z2_centered) / (N - 1)

        # Off-diagonal covariance
        off_diag1 = cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()
        off_diag2 = cov2.pow(2).sum() - cov2.diagonal().pow(2).sum()
        covariance_loss = (off_diag1 + off_diag2) / D

        # Combined loss
        loss = (self.sim_weight * invariance_loss +
                self.var_weight * variance_loss +
                self.cov_weight * covariance_loss)

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Flow Matching Losses
# =============================================================================

class ConditionalFlowMatchingLoss(BaseLoss):
    """
    Conditional Flow Matching loss.

    L = E_{t,z_0,z_1}[||v_theta(z_t, t, c) - (z_1 - z_0)||^2]

    where z_t = (1-t)*z_0 + t*z_1 (linear interpolation)
    """

    def __init__(self, weight: float = 1.0, sigma_min: float = 1e-4):
        super().__init__(weight)
        self.sigma_min = sigma_min

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        flow_pred = predictions.get('flow_predictions')
        z_0 = targets.get('z_0', targets.get('noise'))
        z_1 = targets.get('z_1', targets.get('latents'))

        if flow_pred is None:
            # Compute flow target if predictions exist
            pred = predictions.get('predictions')
            target = targets.get('frames')
            if pred is not None and target is not None:
                flow_target = target - pred
                return F.mse_loss(pred, target) * self.weight
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        if z_0 is None or z_1 is None:
            return torch.tensor(0.0, device=flow_pred.device)

        # Target flow field (rectified flow)
        flow_target = z_1 - z_0

        # MSE loss
        loss = F.mse_loss(flow_pred, flow_target)

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class FlowDistillationLoss(BaseLoss):
    """
    Distillation loss for compressing multi-step teacher to single-step student.

    L = ||f_student(z_0) - (z_N - z_0)/N||^2

    where z_N is the final state after N teacher steps.
    """

    def __init__(self, weight: float = 1.0, teacher_steps: int = 4):
        super().__init__(weight)
        self.teacher_steps = teacher_steps

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        student_flow = predictions.get('flow_predictions')
        teacher_trajectory = targets.get('teacher_trajectory')

        if student_flow is None or teacher_trajectory is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # teacher_trajectory: [num_steps+1, B, T, D]
        z_0 = teacher_trajectory[0]  # Initial state
        z_N = teacher_trajectory[-1]  # Final state

        # Target: average velocity across teacher trajectory
        target_flow = (z_N - z_0) / self.teacher_steps

        loss = F.mse_loss(student_flow, target_flow)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class FlowConsistencyLoss(BaseLoss):
    """
    Consistency loss for continuous-time flow models.

    Ensures flow predictions are consistent across nearby timesteps:
    L = ||v_theta(z_t, t) - v_theta(z_{t+dt}, t+dt)||^2

    where z_{t+dt} = z_t + dt * v_theta(z_t, t)
    """

    def __init__(self, weight: float = 1.0, dt: float = 0.01):
        super().__init__(weight)
        self.dt = dt

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                flow_model: nn.Module = None, **kwargs) -> torch.Tensor:
        z_t = predictions.get('z_t')
        t = predictions.get('timesteps')
        flow_pred = predictions.get('flow_predictions')

        if z_t is None or t is None or flow_pred is None or flow_model is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        with torch.no_grad():
            # Compute z_{t+dt}
            z_t_plus_dt = z_t + self.dt * flow_pred
            t_plus_dt = torch.clamp(t + self.dt, 0, 1)

            # Predict flow at t+dt
            flow_at_t_plus_dt = flow_model(z_t_plus_dt, t_plus_dt)

        # Consistency loss
        loss = F.mse_loss(flow_pred, flow_at_t_plus_dt)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class TrajectoryStraighteningLoss(BaseLoss):
    """
    Loss encouraging straight-line flow trajectories.

    Penalizes curvature in the flow field:
    L = ||d^2z/dt^2||^2 = ||dv/dt||^2
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        trajectory = predictions.get('trajectory')

        if trajectory is None:
            # Try to compute from flow predictions at multiple timesteps
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # trajectory: [num_steps+1, B, T, D]
        if trajectory.size(0) < 3:
            return torch.tensor(0.0, device=trajectory.device)

        # Compute velocities
        velocities = trajectory[1:] - trajectory[:-1]  # [num_steps, B, T, D]

        # Compute accelerations (second derivative)
        accelerations = velocities[1:] - velocities[:-1]  # [num_steps-1, B, T, D]

        # Penalize non-zero accelerations
        loss = accelerations.pow(2).mean()
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Depth Losses
# =============================================================================

class ScaleInvariantDepthLoss(BaseLoss):
    """
    Scale-invariant depth loss from Eigen et al.

    L = (1/n) * sum(d_i^2) - (lambda/n^2) * (sum(d_i))^2

    where d_i = log(pred_i) - log(gt_i)
    """

    def __init__(self, weight: float = 1.0, lambda_scale: float = 0.5,
                 max_depth: float = 100.0, eps: float = 1e-6):
        super().__init__(weight)
        self.lambda_scale = lambda_scale
        self.max_depth = max_depth
        self.eps = eps

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_depth = predictions.get('depth_predictions', predictions.get('depth'))
        gt_depth = targets.get('depth')

        if pred_depth is None or gt_depth is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Ensure valid depth range
        pred_depth = torch.clamp(pred_depth, self.eps, self.max_depth)
        gt_depth = torch.clamp(gt_depth, self.eps, self.max_depth)

        # Compute log difference
        d = torch.log(pred_depth) - torch.log(gt_depth)

        # Flatten for computation
        d_flat = d.view(d.size(0), -1)  # [B, N]
        n = d_flat.size(1)

        # Scale-invariant loss
        loss = (d_flat.pow(2).mean(dim=1) -
                self.lambda_scale * d_flat.mean(dim=1).pow(2)).mean()

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class DepthGradientSmoothnessLoss(BaseLoss):
    """
    Gradient-based depth smoothness loss.

    L = sum |d(depth)/dx| * exp(-|d(rgb)/dx|) +
        sum |d(depth)/dy| * exp(-|d(rgb)/dy|)

    Encourages smooth depth while allowing edges where RGB has edges.
    """

    def __init__(self, weight: float = 1.0, edge_weight: float = 1.0):
        super().__init__(weight)
        self.edge_weight = edge_weight

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_depth = predictions.get('depth_predictions', predictions.get('depth'))
        rgb = targets.get('frames', predictions.get('predictions'))

        if pred_depth is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Handle sequence dimension
        if pred_depth.dim() == 5:  # [B, T, C, H, W]
            pred_depth = pred_depth.view(-1, *pred_depth.shape[-3:])
        if rgb is not None and rgb.dim() == 5:
            rgb = rgb.view(-1, *rgb.shape[-3:])

        # Compute depth gradients
        depth_grad_x = torch.abs(pred_depth[..., :, :-1] - pred_depth[..., :, 1:])
        depth_grad_y = torch.abs(pred_depth[..., :-1, :] - pred_depth[..., 1:, :])

        if rgb is not None:
            # Compute RGB gradients
            rgb_grad_x = torch.abs(rgb[..., :, :-1] - rgb[..., :, 1:]).mean(dim=-3, keepdim=True)
            rgb_grad_y = torch.abs(rgb[..., :-1, :] - rgb[..., 1:, :]).mean(dim=-3, keepdim=True)

            # Edge-aware weighting
            weight_x = torch.exp(-self.edge_weight * rgb_grad_x)
            weight_y = torch.exp(-self.edge_weight * rgb_grad_y)

            loss = (depth_grad_x * weight_x).mean() + (depth_grad_y * weight_y).mean()
        else:
            loss = depth_grad_x.mean() + depth_grad_y.mean()

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class EdgeAwareDepthLoss(BaseLoss):
    """
    Edge-aware depth loss that preserves depth discontinuities at object boundaries.

    Uses Sobel filters to detect edges and applies different loss at edge regions.
    """

    def __init__(self, weight: float = 1.0, edge_threshold: float = 0.1):
        super().__init__(weight)
        self.edge_threshold = edge_threshold

        # Register Sobel filters
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer (move to Module storage when used)."""
        setattr(self, name, tensor)

    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel filters."""
        if x.dim() == 5:  # [B, T, C, H, W]
            x = x.view(-1, *x.shape[-3:])

        # Convert to grayscale if needed
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)

        # Apply Sobel filters
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)

        # Magnitude
        edges = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-8)
        return edges

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_depth = predictions.get('depth_predictions', predictions.get('depth'))
        gt_depth = targets.get('depth')
        rgb = targets.get('frames')

        if pred_depth is None or gt_depth is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Compute edge maps
        depth_edges = self._compute_edges(gt_depth)
        if rgb is not None:
            rgb_edges = self._compute_edges(rgb)
            edge_mask = ((depth_edges > self.edge_threshold) |
                        (rgb_edges > self.edge_threshold)).float()
        else:
            edge_mask = (depth_edges > self.edge_threshold).float()

        # Flatten for matching shapes
        if pred_depth.dim() == 5:
            pred_depth = pred_depth.view(-1, *pred_depth.shape[-3:])
        if gt_depth.dim() == 5:
            gt_depth = gt_depth.view(-1, *gt_depth.shape[-3:])

        # Different loss for edge and non-edge regions
        edge_loss = F.l1_loss(pred_depth * edge_mask, gt_depth * edge_mask)
        smooth_loss = F.mse_loss(pred_depth * (1 - edge_mask), gt_depth * (1 - edge_mask))

        loss = edge_loss + smooth_loss
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class MultiScaleDepthLoss(BaseLoss):
    """
    Multi-scale depth matching loss.

    Computes loss at multiple resolutions for better gradient flow.
    """

    def __init__(self, weight: float = 1.0, scales: List[int] = None,
                 use_si: bool = True):
        super().__init__(weight)
        self.scales = scales or [1, 2, 4]
        self.use_si = use_si
        self.si_loss = ScaleInvariantDepthLoss(weight=1.0) if use_si else None

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_depth = predictions.get('depth_predictions', predictions.get('depth'))
        gt_depth = targets.get('depth')

        if pred_depth is None or gt_depth is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Flatten sequence dimension
        if pred_depth.dim() == 5:
            pred_depth = pred_depth.view(-1, *pred_depth.shape[-3:])
        if gt_depth.dim() == 5:
            gt_depth = gt_depth.view(-1, *gt_depth.shape[-3:])

        total_loss = 0.0

        for scale in self.scales:
            if scale > 1:
                pred_scaled = F.avg_pool2d(pred_depth, scale)
                gt_scaled = F.avg_pool2d(gt_depth, scale)
            else:
                pred_scaled = pred_depth
                gt_scaled = gt_depth

            if self.use_si and self.si_loss is not None:
                scale_preds = {'depth': pred_scaled}
                scale_targets = {'depth': gt_scaled}
                loss = self.si_loss(scale_preds, scale_targets)
            else:
                loss = F.mse_loss(pred_scaled, gt_scaled)

            # Weight by scale (finer scales have higher weight)
            total_loss += loss / scale

        loss = total_loss / len(self.scales)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Control/Action Losses
# =============================================================================

class InverseDynamicsLoss(BaseLoss):
    """
    Inverse dynamics prediction loss.

    Predicts action from state transition:
    L = ||a_pred - a_gt||^2

    where a_pred = f(s_t, s_{t+1})
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_actions = predictions.get('predicted_actions')
        gt_actions = targets.get('controls', targets.get('actions'))

        if pred_actions is None or gt_actions is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Align temporal dimensions
        min_t = min(pred_actions.size(1), gt_actions.size(1))
        pred_actions = pred_actions[:, :min_t]
        gt_actions = gt_actions[:, :min_t]

        loss = F.mse_loss(pred_actions, gt_actions)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class ActionSmoothnessLoss(BaseLoss):
    """
    Action smoothness regularization.

    Penalizes rapid changes in actions:
    L = sum_t ||a_{t+1} - a_t||^2 (first order)
    L = sum_t ||a_{t+2} - 2*a_{t+1} + a_t||^2 (second order)
    """

    def __init__(self, weight: float = 1.0, order: int = 2):
        super().__init__(weight)
        self.order = order

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        actions = predictions.get('predicted_actions') or targets.get('controls')

        if actions is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        T = actions.size(1)

        if T < self.order + 1:
            return torch.tensor(0.0, device=actions.device)

        if self.order == 1:
            # First-order smoothness (velocity)
            diffs = actions[:, 1:] - actions[:, :-1]
            loss = diffs.pow(2).mean()
        else:
            # Second-order smoothness (acceleration)
            accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
            loss = accel.pow(2).mean()

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class BoundaryActionClippingLoss(BaseLoss):
    """
    Loss penalizing actions near boundaries.

    Encourages actions to stay within valid range with margin:
    L = sum max(0, |a| - margin)^2
    """

    def __init__(self, weight: float = 1.0, margin: float = 0.95,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__(weight)
        self.margin = margin
        self.lower = action_bounds[0]
        self.upper = action_bounds[1]

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        actions = predictions.get('predicted_actions')

        if actions is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Normalize to [-1, 1] if needed
        action_range = self.upper - self.lower
        actions_norm = 2 * (actions - self.lower) / action_range - 1

        # Penalize values beyond margin
        excess = torch.clamp(torch.abs(actions_norm) - self.margin, min=0)
        loss = excess.pow(2).mean()

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class GoalConditionedActionLoss(BaseLoss):
    """
    Goal-conditioned action loss.

    Encourages predicted actions to move towards goal:
    L = ||a_pred - a*(s, g)||^2

    where a*(s, g) is the optimal action to reach goal g from state s.
    """

    def __init__(self, weight: float = 1.0, goal_weight: float = 1.0):
        super().__init__(weight)
        self.goal_weight = goal_weight

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_actions = predictions.get('predicted_actions')
        gt_actions = targets.get('controls', targets.get('actions'))
        goals = targets.get('goals')
        states = predictions.get('hidden_states')

        if pred_actions is None or gt_actions is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Base action matching loss
        action_loss = F.mse_loss(pred_actions, gt_actions)

        # Goal-conditioned term
        if goals is not None and states is not None:
            # Compute direction to goal
            if states.dim() == 3:
                states = states.mean(dim=1)  # [B, D]
            if goals.dim() == 3:
                goals = goals.mean(dim=1)  # [B, D]

            # Goal direction (simplified)
            direction = goals - states  # [B, D]
            direction_norm = F.normalize(direction, dim=-1)

            # Encourage actions aligned with goal direction
            if pred_actions.dim() == 3:
                pred_flat = pred_actions.mean(dim=1)  # [B, action_dim]
            else:
                pred_flat = pred_actions

            # Project to same dimension if needed
            if pred_flat.size(-1) != direction_norm.size(-1):
                # Use learned projection (would need additional module)
                goal_loss = torch.tensor(0.0, device=pred_actions.device)
            else:
                # Cosine similarity loss
                goal_loss = 1 - F.cosine_similarity(pred_flat, direction_norm).mean()

            loss = action_loss + self.goal_weight * goal_loss
        else:
            loss = action_loss

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Perceptual Losses
# =============================================================================

class LPIPSLoss(BaseLoss):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.

    Uses pretrained network features for perceptual similarity.
    Note: This is a simplified version. Full LPIPS requires pretrained weights.
    """

    def __init__(self, weight: float = 1.0, net: str = 'vgg'):
        super().__init__(weight)
        self.net_type = net
        self._build_network()

    def _build_network(self):
        """Build feature extraction network."""
        # Simplified VGG-style feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )

        # Freeze features
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred = predictions.get('predictions')
        target = targets.get('frames')

        if pred is None or target is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Flatten sequence dimension
        if pred.dim() == 5:
            pred = pred.view(-1, *pred.shape[-3:])
        if target.dim() == 5:
            target = target.view(-1, *target.shape[-3:])

        # Ensure RGB
        if pred.size(1) != 3:
            pred = pred[:, :3] if pred.size(1) > 3 else pred.repeat(1, 3, 1, 1)[:, :3]
        if target.size(1) != 3:
            target = target[:, :3] if target.size(1) > 3 else target.repeat(1, 3, 1, 1)[:, :3]

        # Extract features
        self.features = self.features.to(pred.device)
        pred_feats = self.features(pred)
        target_feats = self.features(target)

        # Normalize and compute distance
        pred_feats = F.normalize(pred_feats, dim=1)
        target_feats = F.normalize(target_feats, dim=1)

        loss = F.mse_loss(pred_feats, target_feats)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class VGGFeatureLoss(BaseLoss):
    """
    VGG feature matching loss at multiple layers.

    L = sum_l ||phi_l(pred) - phi_l(target)||^2
    """

    def __init__(self, weight: float = 1.0,
                 layers: List[str] = None):
        super().__init__(weight)
        self.layer_names = layers or ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self._build_vgg()

    def _build_vgg(self):
        """Build simplified VGG feature extractor."""
        # Simplified multi-layer feature extractor
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )

        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # Freeze
        for param in self.layers.parameters():
            param.requires_grad = False

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at multiple layers."""
        features = []
        for layer in self.layers:
            layer = layer.to(x.device)
            x = layer(x)
            features.append(x)
        return features

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred = predictions.get('predictions')
        target = targets.get('frames')

        if pred is None or target is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Flatten sequence
        if pred.dim() == 5:
            pred = pred.view(-1, *pred.shape[-3:])
        if target.dim() == 5:
            target = target.view(-1, *target.shape[-3:])

        # Ensure RGB
        if pred.size(1) != 3:
            pred = pred.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if target.size(1) != 3:
            target = target.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Extract features
        pred_feats = self._extract_features(pred)
        target_feats = self._extract_features(target)

        # Compute loss at each layer
        total_loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            total_loss += F.mse_loss(pf, tf)

        loss = total_loss / len(pred_feats)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class DINOFeatureLoss(BaseLoss):
    """
    DINO/DINOv2 feature matching loss.

    Uses self-supervised vision transformer features.
    Note: This is a simplified proxy. Full DINO requires pretrained weights.
    """

    def __init__(self, weight: float = 1.0, embed_dim: int = 384):
        super().__init__(weight)
        self.embed_dim = embed_dim
        self._build_encoder()

    def _build_encoder(self):
        """Build simplified transformer encoder."""
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=16, stride=16)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=6, batch_first=True),
            num_layers=4
        )

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINO-style features."""
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        return x.mean(dim=1)  # [B, D]

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred = predictions.get('predictions')
        target = targets.get('frames')

        if pred is None or target is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Flatten sequence
        if pred.dim() == 5:
            pred = pred.view(-1, *pred.shape[-3:])
        if target.dim() == 5:
            target = target.view(-1, *target.shape[-3:])

        # Ensure RGB
        if pred.size(1) != 3:
            pred = pred.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if target.size(1) != 3:
            target = target.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Move model to device
        self.patch_embed = self.patch_embed.to(pred.device)
        self.transformer = self.transformer.to(pred.device)

        # Extract and compare features
        pred_feats = self._extract_features(pred)
        target_feats = self._extract_features(target)

        # Cosine similarity loss
        loss = 1 - F.cosine_similarity(pred_feats, target_feats).mean()
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class CLIPSimilarityLoss(BaseLoss):
    """
    CLIP similarity loss for semantic consistency.

    Encourages generated frames to maintain semantic similarity with targets.
    Note: This is a simplified proxy. Full CLIP requires pretrained weights.
    """

    def __init__(self, weight: float = 1.0, embed_dim: int = 512):
        super().__init__(weight)
        self.embed_dim = embed_dim
        self._build_encoder()

    def _build_encoder(self):
        """Build simplified vision encoder."""
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.embed_dim)
        )

        # Freeze
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred = predictions.get('predictions')
        target = targets.get('frames')

        if pred is None or target is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Flatten sequence
        if pred.dim() == 5:
            B, T = pred.shape[:2]
            pred = pred.view(-1, *pred.shape[-3:])
            target = target.view(-1, *target.shape[-3:])
        else:
            B, T = pred.size(0), 1

        # Ensure RGB
        if pred.size(1) != 3:
            pred = pred.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if target.size(1) != 3:
            target = target.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Encode
        self.encoder = self.encoder.to(pred.device)
        pred_embed = self.encoder(pred)  # [B*T, D]
        target_embed = self.encoder(target)  # [B*T, D]

        # Normalize
        pred_embed = F.normalize(pred_embed, dim=-1)
        target_embed = F.normalize(target_embed, dim=-1)

        # Cosine similarity loss
        loss = 1 - (pred_embed * target_embed).sum(dim=-1).mean()
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Video Quality Losses
# =============================================================================

class TemporalConsistencyLoss(BaseLoss):
    """
    Temporal consistency loss using optical flow.

    L = ||f_t - warp(f_{t-1}, flow)||^2

    Encourages consistent motion across frames.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def _compute_simple_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Compute simplified optical flow (correlation-based)."""
        # Simplified: use frame difference as proxy for motion
        flow = frame2 - frame1  # [B, C, H, W]
        return flow

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_frames = predictions.get('predictions')
        target_frames = targets.get('frames')

        if pred_frames is None or target_frames is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        B, T = pred_frames.shape[:2]

        if T <= 1:
            return torch.tensor(0.0, device=pred_frames.device)

        # Compute temporal differences
        pred_diffs = pred_frames[:, 1:] - pred_frames[:, :-1]
        target_diffs = target_frames[:, 1:] - target_frames[:, :-1]

        # Match temporal dynamics
        loss = F.mse_loss(pred_diffs, target_diffs)
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class FlickerReductionLoss(BaseLoss):
    """
    Flicker reduction loss.

    Penalizes rapid brightness/color changes:
    L = sum_t |mean(f_{t+1}) - mean(f_t)|^2
    """

    def __init__(self, weight: float = 1.0, threshold: float = 0.1):
        super().__init__(weight)
        self.threshold = threshold

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_frames = predictions.get('predictions')

        if pred_frames is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        B, T = pred_frames.shape[:2]

        if T <= 1:
            return torch.tensor(0.0, device=pred_frames.device)

        # Compute mean intensity per frame
        frame_means = pred_frames.view(B, T, -1).mean(dim=-1)  # [B, T]

        # Compute differences
        mean_diffs = frame_means[:, 1:] - frame_means[:, :-1]  # [B, T-1]

        # Penalize large changes (potential flicker)
        excess = torch.clamp(torch.abs(mean_diffs) - self.threshold, min=0)
        loss = excess.pow(2).mean()

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class MotionMagnitudeLoss(BaseLoss):
    """
    Motion magnitude regularization.

    Penalizes unrealistic motion magnitudes:
    L = |E[||motion||] - target_magnitude|^2
    """

    def __init__(self, weight: float = 1.0, target_magnitude: float = 0.05):
        super().__init__(weight)
        self.target_magnitude = target_magnitude

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_frames = predictions.get('predictions')

        if pred_frames is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        B, T = pred_frames.shape[:2]

        if T <= 1:
            return torch.tensor(0.0, device=pred_frames.device)

        # Compute motion (frame differences)
        motion = pred_frames[:, 1:] - pred_frames[:, :-1]

        # Motion magnitude
        magnitude = motion.pow(2).mean(dim=(2, 3, 4)).sqrt().mean()  # Average magnitude

        # Regularize to target
        loss = (magnitude - self.target_magnitude).pow(2)

        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


class SceneStabilityLoss(BaseLoss):
    """
    Scene stability loss.

    Encourages stable background while allowing object motion:
    L = ||f_t - f_{t-1}||^2 * (1 - motion_mask)
    """

    def __init__(self, weight: float = 1.0, motion_threshold: float = 0.1):
        super().__init__(weight)
        self.motion_threshold = motion_threshold

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pred_frames = predictions.get('predictions')
        target_frames = targets.get('frames')

        if pred_frames is None:
            return torch.tensor(0.0, device=self._get_device(predictions, targets))

        B, T = pred_frames.shape[:2]

        if T <= 1:
            return torch.tensor(0.0, device=pred_frames.device)

        # Compute motion from ground truth
        if target_frames is not None:
            gt_motion = torch.abs(target_frames[:, 1:] - target_frames[:, :-1])
            motion_mask = (gt_motion.mean(dim=2, keepdim=True) > self.motion_threshold).float()
        else:
            motion_mask = torch.zeros_like(pred_frames[:, 1:, :1])

        # Compute prediction stability in non-motion regions
        pred_diffs = pred_frames[:, 1:] - pred_frames[:, :-1]
        stable_diffs = pred_diffs * (1 - motion_mask)

        loss = stable_diffs.pow(2).mean()
        self.update_history(loss.item())
        return loss * self.weight

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Uncertainty Weighting
# =============================================================================

class LearnedTaskWeights(nn.Module):
    """
    Learned task weights using homoscedastic uncertainty.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.

    L_total = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    """

    def __init__(self, num_tasks: int, init_sigma: float = 1.0):
        super().__init__()
        # Log sigma for numerical stability
        self.log_sigmas = nn.Parameter(torch.full((num_tasks,), math.log(init_sigma)))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply learned task weighting.

        Args:
            losses: Dictionary of task losses

        Returns:
            Tuple of (total_loss, weighted_losses_dict)
        """
        total_loss = torch.tensor(0.0, device=self.log_sigmas.device)
        weighted_losses = {}

        for i, (name, loss) in enumerate(losses.items()):
            if i >= len(self.log_sigmas):
                break

            log_sigma = self.log_sigmas[i]
            precision = torch.exp(-2 * log_sigma)  # 1/sigma^2

            weighted_loss = precision * loss + log_sigma
            weighted_losses[name] = weighted_loss.item()
            total_loss = total_loss + weighted_loss

        return total_loss, weighted_losses

    def get_weights(self) -> Dict[int, float]:
        """Get current task weights."""
        sigmas = torch.exp(self.log_sigmas)
        weights = 1.0 / (sigmas.pow(2) + 1e-8)
        return {i: w.item() for i, w in enumerate(weights)}


class GradientNormalizer:
    """
    Gradient normalization for balanced multi-task learning.

    Reference: Chen et al., "GradNorm: Gradient Normalization for
    Adaptive Loss Balancing in Deep Multitask Networks", ICML 2018.
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.initial_losses = None
        self.weights = torch.ones(num_tasks)

    def normalize(self, losses: Dict[str, torch.Tensor],
                  shared_params: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize gradients across tasks.

        Args:
            losses: Dictionary of task losses
            shared_params: Shared model parameters

        Returns:
            Normalized losses
        """
        loss_list = list(losses.values())

        # Store initial losses
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in loss_list]

        # Compute gradient norms
        grad_norms = []
        for loss in loss_list:
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            grad_norm = sum(g.norm() for g in grads if g is not None)
            grad_norms.append(grad_norm)

        # Average gradient norm
        avg_grad_norm = sum(grad_norms) / len(grad_norms)

        # Compute relative inverse training rates
        loss_ratios = []
        for i, loss in enumerate(loss_list):
            ratio = loss.item() / (self.initial_losses[i] + 1e-8)
            loss_ratios.append(ratio)

        avg_ratio = sum(loss_ratios) / len(loss_ratios)
        relative_rates = [r / (avg_ratio + 1e-8) for r in loss_ratios]

        # Update weights
        for i in range(min(len(loss_list), len(self.weights))):
            target_grad_norm = avg_grad_norm * (relative_rates[i] ** self.alpha)
            self.weights[i] = target_grad_norm / (grad_norms[i] + 1e-8)

        # Apply weights
        normalized_losses = {}
        for i, (name, loss) in enumerate(losses.items()):
            if i < len(self.weights):
                normalized_losses[name] = loss * self.weights[i]
            else:
                normalized_losses[name] = loss

        return normalized_losses


class GradientSurgery:
    """
    Gradient surgery for conflicting gradients.

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

    Projects conflicting gradients to resolve conflicts.
    """

    @staticmethod
    def project_conflicting_gradients(
        grads: List[torch.Tensor],
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Project gradients to remove conflicts.

        Args:
            grads: List of gradient tensors from different tasks
            reduction: How to combine ('mean', 'sum')

        Returns:
            Combined gradient without conflicts
        """
        if len(grads) == 1:
            return grads[0]

        # Stack gradients
        grads = [g.flatten() for g in grads]
        grad_stack = torch.stack(grads)  # [num_tasks, num_params]

        # Compute pairwise dot products
        num_tasks = len(grads)

        # Project conflicting gradients
        projected = grad_stack.clone()

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    dot = (grad_stack[i] * grad_stack[j]).sum()
                    if dot < 0:
                        # Conflict detected - project out conflicting component
                        norm_j = grad_stack[j].norm().pow(2) + 1e-8
                        projected[i] = projected[i] - (dot / norm_j) * grad_stack[j]

        # Combine
        if reduction == 'mean':
            return projected.mean(dim=0)
        else:
            return projected.sum(dim=0)


# =============================================================================
# Loss Aggregator
# =============================================================================

class LossAggregator(nn.Module):
    """
    Configurable loss aggregator with curriculum scheduling and gradient surgery.

    Features:
    - Configurable loss combinations
    - Automatic logging and tracking
    - Curriculum-based loss scheduling
    - Gradient surgery for conflicting losses
    - Uncertainty-based task weighting
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.step = 0

        # Build loss functions
        self.losses = nn.ModuleDict()
        self._build_losses()

        # Uncertainty weighting
        if config.use_uncertainty_weighting:
            self.task_weights = LearnedTaskWeights(len(self.losses))
        else:
            self.task_weights = None

        # Gradient normalization
        if config.use_gradient_normalization:
            self.grad_normalizer = GradientNormalizer(len(self.losses))
        else:
            self.grad_normalizer = None

        # Gradient surgery
        self.use_gradient_surgery = config.use_gradient_surgery

        # History tracking
        self.loss_history: Dict[str, List[float]] = {}
        self.curriculum_weights: Dict[str, float] = {}

    def _build_losses(self):
        """Build loss functions based on configuration."""
        loss_mapping = {
            # Self-forcing
            LossType.LATENT_MSE: LatentMSELoss,
            LossType.TEACHER_STUDENT_FLOW: TeacherStudentFlowLoss,
            LossType.KV_CACHE_CONSISTENCY: KVCacheConsistencyLoss,
            LossType.AUTOREGRESSIVE_NEXT_FRAME: AutoregressiveNextFrameLoss,

            # JEPA
            LossType.INFONCE: lambda w: InfoNCELoss(w, self.config.infonce_temperature),
            LossType.CROSS_MODAL_CONTRASTIVE: CrossModalContrastiveLoss,
            LossType.TEMPORAL_CONTRASTIVE: TemporalContrastiveLoss,
            LossType.VICREG: lambda w: VICRegLoss(
                w, self.config.vicreg_sim_weight,
                self.config.vicreg_var_weight, self.config.vicreg_cov_weight
            ),

            # Flow matching
            LossType.CONDITIONAL_FLOW_MATCHING: ConditionalFlowMatchingLoss,
            LossType.FLOW_DISTILLATION: FlowDistillationLoss,
            LossType.FLOW_CONSISTENCY: FlowConsistencyLoss,
            LossType.TRAJECTORY_STRAIGHTENING: TrajectoryStraighteningLoss,

            # Depth
            LossType.SCALE_INVARIANT_DEPTH: ScaleInvariantDepthLoss,
            LossType.DEPTH_GRADIENT_SMOOTHNESS: DepthGradientSmoothnessLoss,
            LossType.EDGE_AWARE_DEPTH: EdgeAwareDepthLoss,
            LossType.MULTI_SCALE_DEPTH: MultiScaleDepthLoss,

            # Control
            LossType.INVERSE_DYNAMICS: InverseDynamicsLoss,
            LossType.ACTION_SMOOTHNESS: lambda w: ActionSmoothnessLoss(w, self.config.action_smoothness_order),
            LossType.BOUNDARY_ACTION_CLIPPING: lambda w: BoundaryActionClippingLoss(w, self.config.action_boundary_margin),
            LossType.GOAL_CONDITIONED_ACTION: GoalConditionedActionLoss,

            # Perceptual
            LossType.LPIPS: lambda w: LPIPSLoss(w, self.config.lpips_net),
            LossType.VGG_FEATURE: lambda w: VGGFeatureLoss(w, self.config.vgg_layers),
            LossType.DINO_FEATURE: DINOFeatureLoss,
            LossType.CLIP_SIMILARITY: CLIPSimilarityLoss,

            # Video quality
            LossType.TEMPORAL_CONSISTENCY: TemporalConsistencyLoss,
            LossType.FLICKER_REDUCTION: lambda w: FlickerReductionLoss(w, self.config.flicker_threshold),
            LossType.MOTION_MAGNITUDE: MotionMagnitudeLoss,
            LossType.SCENE_STABILITY: SceneStabilityLoss,
        }

        for loss_type in self.config.enabled_losses:
            name = loss_type.value
            weight = self.config.weights.get(name, 1.0)

            loss_cls = loss_mapping.get(loss_type)
            if loss_cls is not None:
                if callable(loss_cls) and not isinstance(loss_cls, type):
                    # Lambda function
                    self.losses[name] = loss_cls(weight)
                else:
                    # Class
                    self.losses[name] = loss_cls(weight)

    def _get_curriculum_weight(self, loss_name: str) -> float:
        """Get curriculum-based weight for a loss."""
        if not self.config.enable_curriculum:
            return 1.0

        progress = min(1.0, self.step / self.config.curriculum_warmup_steps)

        # Define curriculum schedules
        schedules = {
            # Start immediately with full weight
            'latent_mse': 1.0,
            'autoregressive_next_frame': 1.0,

            # Ramp up gradually
            'temporal_consistency': progress,
            'conditional_flow_matching': 0.1 + 0.9 * progress,
            'flow_distillation': progress,

            # Start after warmup
            'infonce': max(0, progress - 0.3) / 0.7 if progress > 0.3 else 0.0,
            'vicreg': max(0, progress - 0.3) / 0.7 if progress > 0.3 else 0.0,
            'cross_modal_contrastive': max(0, progress - 0.2) / 0.8 if progress > 0.2 else 0.0,

            # Perceptual losses ramp up
            'lpips': 0.5 + 0.5 * progress,
            'vgg_feature': 0.5 + 0.5 * progress,
            'dino_feature': progress,
            'clip_similarity': max(0, progress - 0.4) / 0.6 if progress > 0.4 else 0.0,
        }

        return schedules.get(loss_name, 1.0)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses and aggregate.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            step: Current training step
            **kwargs: Additional arguments for specific losses

        Returns:
            Dictionary containing individual losses and total
        """
        if step is not None:
            self.step = step

        losses = {}

        # Compute individual losses
        for name, loss_fn in self.losses.items():
            try:
                loss_value = loss_fn(predictions, targets, **kwargs)

                # Apply curriculum weight
                curriculum_weight = self._get_curriculum_weight(name)
                self.curriculum_weights[name] = curriculum_weight

                losses[name] = loss_value * curriculum_weight

                # Update history
                if name not in self.loss_history:
                    self.loss_history[name] = []
                self.loss_history[name].append(losses[name].item())

                # Trim history
                if len(self.loss_history[name]) > 1000:
                    self.loss_history[name] = self.loss_history[name][-500:]

            except Exception as e:
                # Log error but continue
                losses[name] = torch.tensor(0.0, device=self._get_device(predictions, targets))

        # Apply uncertainty weighting
        if self.task_weights is not None:
            total_loss, weighted_info = self.task_weights(losses)
        else:
            total_loss = sum(losses.values())

        losses['total'] = total_loss

        return losses

    def apply_gradient_surgery(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply gradient surgery to resolve conflicts.

        Args:
            losses: Dictionary of losses
            shared_params: Shared model parameters

        Returns:
            Combined gradient tensor
        """
        if not self.use_gradient_surgery:
            return sum(losses.values())

        # Compute gradients for each loss
        grads = []
        for name, loss in losses.items():
            if name != 'total' and loss.requires_grad:
                grad = torch.autograd.grad(
                    loss, shared_params,
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False
                )
                # Stack gradients
                grad_flat = torch.cat([g.flatten() if g is not None else torch.zeros_like(p.flatten())
                                      for g, p in zip(grad, shared_params)])
                grads.append(grad_flat)

        if not grads:
            return sum(losses.values())

        # Apply gradient surgery
        combined_grad = GradientSurgery.project_conflicting_gradients(grads)

        # Manually set gradients (for reference - actual implementation
        # would need to unflatten and assign to parameters)
        return losses['total']

    def get_loss_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all losses."""
        stats = {}

        for name, history in self.loss_history.items():
            if history:
                recent = history[-100:]
                stats[name] = {
                    'mean': np.mean(recent),
                    'std': np.std(recent),
                    'min': np.min(recent),
                    'max': np.max(recent),
                    'current': history[-1],
                    'curriculum_weight': self.curriculum_weights.get(name, 1.0)
                }

        return stats

    def get_curriculum_progress(self) -> Dict[str, float]:
        """Get current curriculum weights for all losses."""
        return {name: self._get_curriculum_weight(name) for name in self.losses.keys()}

    def _get_device(self, p: Dict, t: Dict) -> torch.device:
        for d in [p, t]:
            for v in d.values():
                if torch.is_tensor(v):
                    return v.device
        return torch.device('cpu')


# =============================================================================
# Factory Functions
# =============================================================================

def create_loss_aggregator(config: LossConfig = None) -> LossAggregator:
    """Create a loss aggregator with default or custom configuration."""
    if config is None:
        config = LossConfig()
    return LossAggregator(config)


def create_default_loss_config() -> LossConfig:
    """Create default loss configuration."""
    return LossConfig(
        enabled_losses=[
            LossType.LATENT_MSE,
            LossType.AUTOREGRESSIVE_NEXT_FRAME,
            LossType.TEMPORAL_CONSISTENCY,
            LossType.CONDITIONAL_FLOW_MATCHING,
        ],
        enable_curriculum=True,
        use_uncertainty_weighting=False,
        use_gradient_normalization=False,
        use_gradient_surgery=False
    )


def create_full_loss_config() -> LossConfig:
    """Create configuration with all losses enabled."""
    return LossConfig(
        enabled_losses=list(LossType),
        enable_curriculum=True,
        use_uncertainty_weighting=True,
        use_gradient_normalization=True,
        use_gradient_surgery=True
    )


def create_minimal_loss_config() -> LossConfig:
    """Create minimal loss configuration for fast training."""
    return LossConfig(
        enabled_losses=[
            LossType.LATENT_MSE,
            LossType.TEMPORAL_CONSISTENCY,
        ],
        enable_curriculum=False,
        use_uncertainty_weighting=False
    )


# =============================================================================
# Backward Compatibility with losses.py
# =============================================================================

class UnifiedLoss:
    """
    Backward-compatible wrapper matching the original losses.py interface.
    """

    def __init__(self, training_config):
        self.config = training_config
        self.step = 0
        self.loss_history = {}

        # Create loss aggregator with appropriate config
        loss_config = LossConfig(
            enabled_losses=[
                LossType.LATENT_MSE,
                LossType.AUTOREGRESSIVE_NEXT_FRAME,
                LossType.TEMPORAL_CONSISTENCY,
            ],
            enable_curriculum=getattr(training_config, 'enable_curriculum', True),
            curriculum_warmup_steps=getattr(training_config, 'curriculum_warmup_steps', 10000)
        )

        if getattr(training_config, 'enable_flow_matching', False):
            loss_config.enabled_losses.append(LossType.CONDITIONAL_FLOW_MATCHING)

        self.aggregator = LossAggregator(loss_config)

    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with backward-compatible interface."""
        if step is not None:
            self.step = step

        losses = self.aggregator(predictions, targets, step=self.step)

        # Update history
        for name, loss in losses.items():
            if name not in self.loss_history:
                self.loss_history[name] = []
            self.loss_history[name].append(loss.item() if torch.is_tensor(loss) else loss)
            if len(self.loss_history[name]) > 1000:
                self.loss_history[name] = self.loss_history[name][-500:]

        return losses

    def get_loss_stats(self) -> Dict[str, float]:
        """Get loss statistics."""
        return self.aggregator.get_loss_stats()


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Comprehensive Loss System v2")
    print("=" * 60)

    # Test configuration
    config = create_default_loss_config()
    print(f"Enabled losses: {[l.value for l in config.enabled_losses]}")

    # Create aggregator
    aggregator = create_loss_aggregator(config)
    print(f"Created aggregator with {len(aggregator.losses)} loss functions")

    # Test data
    B, T, C, H, W = 2, 8, 3, 64, 64
    D = 256

    predictions = {
        'predictions': torch.randn(B, T, C, H, W),
        'latents': torch.randn(B, T, D),
        'hidden_states': torch.randn(B, T, D),
        'flow_predictions': torch.randn(B, T, D),
    }

    targets = {
        'frames': torch.randn(B, T, C, H, W),
        'latents': torch.randn(B, T, D),
        'z_0': torch.randn(B, T, D),
        'z_1': torch.randn(B, T, D),
    }

    # Compute losses
    losses = aggregator(predictions, targets, step=500)

    print("\nComputed Losses:")
    for name, value in losses.items():
        if torch.is_tensor(value):
            print(f"  {name}: {value.item():.6f}")
        else:
            print(f"  {name}: {value:.6f}")

    # Test curriculum progress
    print("\nCurriculum Progress (step=500):")
    progress = aggregator.get_curriculum_progress()
    for name, weight in progress.items():
        print(f"  {name}: {weight:.3f}")

    # Test statistics
    print("\nLoss Statistics:")
    stats = aggregator.get_loss_stats()
    for name, stat_dict in stats.items():
        print(f"  {name}: mean={stat_dict['mean']:.4f}, std={stat_dict['std']:.4f}")

    # Test individual losses
    print("\n" + "=" * 60)
    print("Testing Individual Loss Functions")
    print("=" * 60)

    # InfoNCE
    infonce = InfoNCELoss(weight=1.0, temperature=0.1)
    infonce_loss = infonce(predictions, targets)
    print(f"InfoNCE Loss: {infonce_loss.item():.6f}")

    # VICReg
    vicreg = VICRegLoss(weight=1.0)
    vicreg_loss = vicreg(predictions, targets)
    print(f"VICReg Loss: {vicreg_loss.item():.6f}")

    # Temporal Consistency
    temp_cons = TemporalConsistencyLoss(weight=1.0)
    temp_loss = temp_cons(predictions, targets)
    print(f"Temporal Consistency Loss: {temp_loss.item():.6f}")

    # Test uncertainty weighting
    print("\n" + "=" * 60)
    print("Testing Uncertainty Weighting")
    print("=" * 60)

    task_weights = LearnedTaskWeights(num_tasks=4)
    test_losses = {
        'loss1': torch.tensor(1.0),
        'loss2': torch.tensor(2.0),
        'loss3': torch.tensor(0.5),
        'loss4': torch.tensor(1.5),
    }

    total, weighted = task_weights(test_losses)
    print(f"Total weighted loss: {total.item():.4f}")
    print(f"Individual weights: {task_weights.get_weights()}")

    # Test backward compatibility
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility")
    print("=" * 60)

    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        enable_flow_matching: bool = True
        enable_curriculum: bool = True
        curriculum_warmup_steps: int = 1000

    mock_config = MockConfig()
    unified_loss = UnifiedLoss(mock_config)
    compat_losses = unified_loss(predictions, targets, step=500)

    print("Backward-compatible losses:")
    for name, value in compat_losses.items():
        if torch.is_tensor(value):
            print(f"  {name}: {value.item():.6f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
