"""
Unified loss system consolidating all loss functions:
- Flow matching and rectified flow losses
- Reconstruction losses (L1, L2, perceptual)
- Temporal consistency losses
- JEPA contrastive losses
- All other training losses

This replaces the scattered loss implementations with a single, comprehensive system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np


class UnifiedLoss:
    """
    Unified loss function that handles all loss computations for the training pipeline.
    Consolidates flow matching, reconstruction, temporal, and other losses.
    """
    
    def __init__(self, training_config):
        self.config = training_config
        self.step = 0
        self.loss_history = {}
        
        # Loss weights (can be made configurable)
        self.weights = {
            'reconstruction': 1.0,
            'flow_matching': 1.0 if getattr(training_config, 'enable_flow_matching', False) else 0.0,
            'temporal_consistency': 0.2,
            'jepa': 0.1,
            'perceptual': 0.1,
            'l1': 0.1,
        }
    
    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the current training step.
        
        Args:
            predictions: Model predictions dictionary
            targets: Target values dictionary  
            step: Current training step
        
        Returns:
            Dictionary of computed losses
        """
        if step is not None:
            self.step = step
        
        losses = {}
        device = next(iter(predictions.values())).device
        
        # Core reconstruction loss
        if 'predictions' in predictions and 'frames' in targets:
            losses.update(self._reconstruction_losses(
                predictions['predictions'], 
                targets['frames']
            ))
        
        # Flow matching loss
        if getattr(self.config, 'enable_flow_matching', False) and 'flow_predictions' in predictions:
            losses['flow_matching'] = self._flow_matching_loss(
                predictions, targets
            )
        
        # Temporal consistency
        if 'predictions' in predictions and 'frames' in targets:
            losses['temporal_consistency'] = self._temporal_consistency_loss(
                predictions['predictions'],
                targets['frames']
            )
        
        # JEPA contrastive loss
        if 'jepa_predictions' in predictions:
            losses['jepa'] = self._jepa_loss(predictions, targets)
        
        # Apply curriculum weighting
        if getattr(self.config, 'enable_curriculum', False):
            losses = self._apply_curriculum_weights(losses)
        
        # Compute total loss
        total_loss = self._compute_total_loss(losses)
        losses['total'] = total_loss
        
        # Update history
        self._update_history(losses)
        
        return losses
    
    def _reconstruction_losses(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction losses."""
        losses = {}
        
        # L2 loss (MSE)
        l2_loss = F.mse_loss(pred_frames, target_frames)
        losses['reconstruction'] = l2_loss * self.weights['reconstruction']
        
        # L1 loss
        l1_loss = F.l1_loss(pred_frames, target_frames)
        losses['l1'] = l1_loss * self.weights['l1']
        
        # Perceptual loss (simplified - grayscale comparison)
        pred_gray = pred_frames.mean(dim=2, keepdim=True)
        target_gray = target_frames.mean(dim=2, keepdim=True)
        perceptual_loss = F.l1_loss(pred_gray, target_gray)
        losses['perceptual'] = perceptual_loss * self.weights['perceptual']
        
        return losses
    
    def _flow_matching_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Flow matching loss: ||v_θ(z_t, t) - (z_1 - z_0)||²
        """
        if 'flow_predictions' not in predictions:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        flow_pred = predictions['flow_predictions']
        
        # If we have explicit flow target
        if 'flow_target' in targets:
            flow_target = targets['flow_target']
            loss = F.mse_loss(flow_pred, flow_target)
        else:
            # Compute flow target from predictions and targets
            if 'predictions' in predictions and 'frames' in targets:
                pred_frames = predictions['predictions']
                target_frames = targets['frames']
                flow_target = target_frames - pred_frames
                loss = F.mse_loss(flow_pred, flow_target)
            else:
                loss = torch.tensor(0.0, device=flow_pred.device)
        
        return loss * self.weights['flow_matching']
    
    def _temporal_consistency_loss(
        self,
        pred_sequence: torch.Tensor,
        target_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Temporal consistency loss."""
        B, T = pred_sequence.shape[:2]
        
        if T <= 1:
            return torch.tensor(0.0, device=pred_sequence.device)
        
        # Frame-to-frame differences
        pred_diffs = pred_sequence[:, 1:] - pred_sequence[:, :-1]
        target_diffs = target_sequence[:, 1:] - target_sequence[:, :-1]
        
        temporal_loss = F.mse_loss(pred_diffs, target_diffs)
        
        return temporal_loss * self.weights['temporal_consistency']
    
    def _jepa_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """JEPA contrastive loss."""
        if 'jepa_predictions' not in predictions:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        jepa_pred = predictions['jepa_predictions']  # [B, target_length, D]
        
        # For JEPA, we need to create positive and negative pairs
        # Simplified implementation - in practice would be more sophisticated
        B, target_len, D = jepa_pred.shape
        
        # Create targets (future representations)
        if 'hidden_states' in predictions:
            hidden_states = predictions['hidden_states']  # [B, T, D]
            if hidden_states.size(1) > target_len:
                # Use future representations as targets
                jepa_targets = hidden_states[:, -target_len:]
            else:
                # Fallback: use last representation repeated
                jepa_targets = hidden_states[:, -1:].repeat(1, target_len, 1)
        else:
            # No targets available
            return torch.tensor(0.0, device=jepa_pred.device)
        
        # Contrastive loss (simplified)
        # Positive pairs: predicted and actual future representations
        pos_loss = F.mse_loss(jepa_pred, jepa_targets)
        
        # Negative pairs: predicted vs shuffled targets
        shuffled_targets = jepa_targets[torch.randperm(B)]
        neg_loss = F.mse_loss(jepa_pred, shuffled_targets)
        
        # Contrastive: minimize positive, maximize negative (within margin)
        margin = 1.0
        contrastive_loss = pos_loss + torch.clamp(margin - neg_loss, min=0.0)
        
        return contrastive_loss * self.weights['jepa']
    
    def _apply_curriculum_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply curriculum learning to loss weights."""
        if not getattr(self.config, 'enable_curriculum', False):
            return losses
        
        # Get curriculum progress
        progress = min(1.0, self.step / getattr(self.config, 'curriculum_warmup_steps', 10000))
        
        # Curriculum scheduling for different loss types
        curriculum_weights = {
            'reconstruction': 1.0,  # Always full weight
            'l1': 1.0,
            'perceptual': 1.0,
            'flow_matching': 0.1 + 0.9 * progress,  # Ramp up flow matching
            'temporal_consistency': progress,  # Add temporal loss gradually
            'jepa': max(0, progress - 0.3) * (1/0.7),  # Only after 30% progress
        }
        
        # Apply curriculum weights
        weighted_losses = {}
        for key, loss in losses.items():
            weight = curriculum_weights.get(key, 1.0)
            weighted_losses[key] = loss * weight
        
        return weighted_losses
    
    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total weighted loss."""
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        for key, loss in losses.items():
            if key != 'total':  # Avoid recursive addition
                total = total + loss
        
        return total
    
    def _update_history(self, losses: Dict[str, torch.Tensor]):
        """Update loss history for monitoring."""
        for key, loss in losses.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            
            value = loss.item() if torch.is_tensor(loss) else loss
            self.loss_history[key].append(value)
            
            # Keep only recent history
            if len(self.loss_history[key]) > 1000:
                self.loss_history[key] = self.loss_history[key][-500:]
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Get loss statistics."""
        stats = {}
        
        for key, history in self.loss_history.items():
            if history:
                recent_history = history[-100:]  # Last 100 steps
                stats[f'{key}_mean'] = np.mean(recent_history)
                stats[f'{key}_std'] = np.std(recent_history)
                stats[f'{key}_current'] = history[-1]
        
        return stats


class FlowMatchingLoss:
    """Specialized flow matching loss functions."""
    
    @staticmethod
    def rectified_flow_loss(
        flow_pred: torch.Tensor,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rectified flow loss for straight-line interpolation.
        Flow target is simply (z_1 - z_0) for all timesteps.
        """
        target_velocity = z_1 - z_0  # Constant velocity
        return F.mse_loss(flow_pred, target_velocity)
    
    @staticmethod
    def velocity_matching_loss(
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        weighting: str = "uniform"
    ) -> torch.Tensor:
        """
        Velocity matching with different weighting schemes.
        
        Args:
            pred_velocity: [B, T, D] predicted velocity field
            target_velocity: [B, T, D] target velocity field
            timesteps: [B] timesteps in [0, 1]
            weighting: "uniform", "snr", "truncated_snr"
        """
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        
        if timesteps is not None and weighting != "uniform":
            if weighting == "snr":
                # Signal-to-noise ratio weighting
                snr = 1.0 / (timesteps + 1e-8)
                weights = snr.view(-1, 1, 1)
            elif weighting == "truncated_snr":
                # Truncated SNR (clip very large weights)
                snr = torch.clamp(1.0 / (timesteps + 1e-8), 0, 1000)
                weights = snr.view(-1, 1, 1)
            else:
                weights = 1.0
            
            loss = loss * weights
        
        return loss.mean()
    
    @staticmethod
    def distillation_loss(
        student_flow: torch.Tensor,
        teacher_trajectory: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Distillation loss for compressing multi-step teacher to single-step student.
        
        Args:
            student_flow: [B, T, D] student predicted velocity
            teacher_trajectory: [num_steps+1, B, T, D] teacher trajectory
            dt: teacher timestep size
        """
        # Compute teacher velocity from trajectory
        z_start = teacher_trajectory[0]  # [B, T, D]
        z_end = teacher_trajectory[-1]   # [B, T, D]
        
        # Total displacement over trajectory
        teacher_velocity = (z_end - z_start) / (len(teacher_trajectory) - 1)
        
        return F.mse_loss(student_flow, teacher_velocity)


class ReconstructionLoss:
    """Specialized reconstruction loss functions."""
    
    @staticmethod
    def multi_scale_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        scales: List[int] = [1, 2, 4]
    ) -> torch.Tensor:
        """Multi-scale reconstruction loss."""
        total_loss = 0.0
        
        for scale in scales:
            if scale > 1:
                # Downsample
                pred_scaled = F.avg_pool2d(pred.view(-1, *pred.shape[-3:]), scale)
                target_scaled = F.avg_pool2d(target.view(-1, *target.shape[-3:]), scale)
            else:
                pred_scaled = pred.view(-1, *pred.shape[-3:])
                target_scaled = target.view(-1, *target.shape[-3:])
            
            loss = F.mse_loss(pred_scaled, target_scaled)
            total_loss += loss / scale  # Weight by scale
        
        return total_loss
    
    @staticmethod
    def gradient_loss(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Gradient-based loss for edge preservation."""
        # Compute gradients
        pred_grad_x = pred[..., :-1, :] - pred[..., 1:, :]
        pred_grad_y = pred[..., :, :-1] - pred[..., :, 1:]
        
        target_grad_x = target[..., :-1, :] - target[..., 1:, :]
        target_grad_y = target[..., :, :-1] - target[..., :, 1:]
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class TemporalConsistencyLoss:
    """Temporal consistency loss for video generation."""
    
    @staticmethod
    def frame_difference_loss(
        pred_sequence: torch.Tensor,
        target_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Frame-to-frame difference consistency."""
        if pred_sequence.shape[1] <= 1:
            return torch.tensor(0.0, device=pred_sequence.device)
        
        pred_diffs = pred_sequence[:, 1:] - pred_sequence[:, :-1]
        target_diffs = target_sequence[:, 1:] - target_sequence[:, :-1]
        
        return F.mse_loss(pred_diffs, target_diffs)
    
    @staticmethod
    def smoothness_loss(frames: torch.Tensor) -> torch.Tensor:
        """Second-order temporal smoothness."""
        if frames.shape[1] < 3:
            return torch.tensor(0.0, device=frames.device)
        
        # Second-order differences (acceleration)
        second_diffs = frames[:, 2:] - 2 * frames[:, 1:-1] + frames[:, :-2]
        return second_diffs.abs().mean()


def create_unified_loss(training_config) -> UnifiedLoss:
    """Create unified loss function."""
    return UnifiedLoss(training_config)


if __name__ == "__main__":
    # Test unified loss system
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        enable_flow_matching: bool = True
        enable_curriculum: bool = True
        curriculum_warmup_steps: int = 1000
    
    config = MockConfig()
    loss_fn = create_unified_loss(config)
    
    # Test data
    B, T, C, H, W = 2, 8, 3, 64, 64
    D = 128
    
    predictions = {
        'predictions': torch.randn(B, T, C, H, W),
        'flow_predictions': torch.randn(B, T, D),
        'jepa_predictions': torch.randn(B, 4, D),
        'hidden_states': torch.randn(B, T, D)
    }
    
    targets = {
        'frames': torch.randn(B, T, C, H, W),
        'flow_target': torch.randn(B, T, D)
    }
    
    # Compute losses
    losses = loss_fn(predictions, targets, step=500)
    
    print("Unified Loss Test Results:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Test loss statistics
    stats = loss_fn.get_loss_stats()
    print(f"\nLoss Statistics: {stats}")
    
    # Test specialized losses
    print("\nSpecialized Loss Tests:")
    
    # Flow matching
    z_0 = torch.randn(B, T, D)
    z_1 = torch.randn(B, T, D)
    flow_pred = torch.randn(B, T, D)
    
    rect_loss = FlowMatchingLoss.rectified_flow_loss(flow_pred, z_0, z_1)
    print(f"  Rectified Flow Loss: {rect_loss.item():.6f}")
    
    # Multi-scale reconstruction
    multi_loss = ReconstructionLoss.multi_scale_loss(
        predictions['predictions'], targets['frames']
    )
    print(f"  Multi-scale Loss: {multi_loss.item():.6f}")
    
    # Temporal consistency
    temp_loss = TemporalConsistencyLoss.frame_difference_loss(
        predictions['predictions'], targets['frames']
    )
    print(f"  Temporal Consistency Loss: {temp_loss.item():.6f}")
    
    print("\nUnified loss system test completed successfully!")