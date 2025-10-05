"""
Comprehensive evaluation metrics for DriveDiT models.
Zero-dependency implementations of video and driving-specific metrics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np


class VideoMetrics:
    """Video quality and reconstruction metrics."""
    
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean Squared Error."""
        error = (pred - target) ** 2
        
        if mask is not None:
            error = error * mask
            return error.sum() / mask.sum().clamp(min=1)
        else:
            return error.mean()
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean Absolute Error."""
        error = torch.abs(pred - target)
        
        if mask is not None:
            error = error * mask
            return error.sum() / mask.sum().clamp(min=1)
        else:
            return error.mean()
    
    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """Peak Signal-to-Noise Ratio."""
        mse_val = VideoMetrics.mse(pred, target)
        return 20 * torch.log10(max_val) - 10 * torch.log10(mse_val.clamp(min=1e-8))
    
    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
        """Structural Similarity Index (simplified implementation)."""
        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        
        # 2D Gaussian kernel
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
        
        # Ensure input dimensions
        if pred.dim() == 4:  # [B, C, H, W]
            pred_flat = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_flat = target.view(-1, 1, target.shape[-2], target.shape[-1])
        else:
            pred_flat = pred
            target_flat = target
        
        # Compute local means
        mu1 = F.conv2d(pred_flat, kernel, padding=window_size//2)
        mu2 = F.conv2d(target_flat, kernel, padding=window_size//2)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(pred_flat**2, kernel, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target_flat**2, kernel, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred_flat * target_flat, kernel, padding=window_size//2) - mu1_mu2
        
        # SSIM constants
        C1 = (0.01 * max_val)**2
        C2 = (0.03 * max_val)**2
        
        # SSIM computation
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator.clamp(min=1e-8)
        return ssim_map.mean()
    
    @staticmethod
    def lpips_proxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """LPIPS proxy using VGG-like features (simplified)."""
        # Simple perceptual loss using gradients as feature proxy
        def extract_features(x):
            # Sobel edge detection as feature proxy
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
            
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            if x.dim() == 4 and x.shape[1] == 3:  # RGB
                # Convert to grayscale
                gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            else:
                gray = x.mean(dim=1, keepdim=True) if x.dim() == 4 else x
            
            edges_x = F.conv2d(gray, sobel_x, padding=1)
            edges_y = F.conv2d(gray, sobel_y, padding=1)
            
            return torch.cat([edges_x, edges_y], dim=1)
        
        feat_pred = extract_features(pred)
        feat_target = extract_features(target)
        
        return F.mse_loss(feat_pred, feat_target)


class LatentMetrics:
    """Metrics for latent space representations."""
    
    @staticmethod
    def latent_mse(pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
        """MSE in latent space."""
        return F.mse_loss(pred_latents, target_latents)
    
    @staticmethod
    def latent_cosine_similarity(pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in latent space."""
        pred_flat = pred_latents.view(pred_latents.shape[0], -1)
        target_flat = target_latents.view(target_latents.shape[0], -1)
        
        return F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
    
    @staticmethod
    def latent_diversity(latents: torch.Tensor) -> torch.Tensor:
        """Measure diversity in latent representations."""
        # Compute pairwise distances
        latents_flat = latents.view(latents.shape[0], -1)
        
        # Pairwise L2 distances
        dists = torch.cdist(latents_flat, latents_flat, p=2)
        
        # Remove diagonal (self-distances)
        mask = ~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
        
        return dists[mask].mean()
    
    @staticmethod
    def latent_smoothness(latents: torch.Tensor) -> torch.Tensor:
        """Measure temporal smoothness in latent sequences."""
        if latents.dim() == 5:  # [B, T, C, H, W]
            # Temporal differences
            temporal_diff = latents[:, 1:] - latents[:, :-1]
            return temporal_diff.norm(dim=(2, 3, 4)).mean()
        elif latents.dim() == 3:  # [B, T, D]
            temporal_diff = latents[:, 1:] - latents[:, :-1]
            return temporal_diff.norm(dim=2).mean()
        else:
            return torch.tensor(0.0, device=latents.device)
    
    @staticmethod
    def latent_reconstruction_error(latents: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Reconstruction error in latent space."""
        return F.mse_loss(latents, reconstructed)


class TemporalConsistencyMetrics:
    """Metrics for temporal consistency in video sequences."""
    
    @staticmethod
    def temporal_mse(frames: torch.Tensor) -> torch.Tensor:
        """Temporal MSE between consecutive frames."""
        if frames.dim() == 5:  # [B, T, C, H, W]
            frame_diff = frames[:, 1:] - frames[:, :-1]
            return frame_diff.pow(2).mean()
        elif frames.dim() == 4:  # [T, C, H, W]
            frame_diff = frames[1:] - frames[:-1]
            return frame_diff.pow(2).mean()
        else:
            return torch.tensor(0.0, device=frames.device)
    
    @staticmethod
    def optical_flow_consistency(frames: torch.Tensor) -> torch.Tensor:
        """Simplified optical flow consistency metric."""
        if frames.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)
        else:
            frames_flat = frames
        
        # Simple gradient-based flow proxy
        grad_x = frames_flat[:, :, :, 1:] - frames_flat[:, :, :, :-1]
        grad_y = frames_flat[:, :, 1:, :] - frames_flat[:, :, :-1, :]
        
        # Temporal flow (simplified)
        if frames.dim() == 5:
            temporal_grad = frames[:, 1:] - frames[:, :-1]
            flow_consistency = torch.var(temporal_grad.view(B, T-1, -1), dim=2).mean()
        else:
            temporal_grad = frames[1:] - frames[:-1]
            flow_consistency = torch.var(temporal_grad.view(T-1, -1), dim=1).mean()
        
        return flow_consistency
    
    @staticmethod
    def frame_warping_error(frames: torch.Tensor, flows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Frame warping error using optical flow."""
        if flows is None:
            # Compute simple displacement
            flows = TemporalConsistencyMetrics._compute_simple_flow(frames)
        
        # Warp frames using flow
        warped_frames = TemporalConsistencyMetrics._warp_frames(frames[:-1], flows)
        target_frames = frames[1:]
        
        return F.mse_loss(warped_frames, target_frames)
    
    @staticmethod
    def _compute_simple_flow(frames: torch.Tensor) -> torch.Tensor:
        """Compute simple optical flow approximation."""
        # Simple intensity-based flow estimation
        frame_diff = frames[1:] - frames[:-1]
        
        # Gradient computation
        grad_x = torch.gradient(frames[:-1], dim=-1)[0]
        grad_y = torch.gradient(frames[:-1], dim=-2)[0]
        
        # Lucas-Kanade-like approximation
        epsilon = 1e-6
        flow_x = -frame_diff * grad_x / (grad_x**2 + epsilon)
        flow_y = -frame_diff * grad_y / (grad_y**2 + epsilon)
        
        return torch.stack([flow_x, flow_y], dim=-1)
    
    @staticmethod
    def _warp_frames(frames: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """Warp frames using flow vectors."""
        B, T, C, H, W = frames.shape
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=frames.device),
            torch.arange(W, device=frames.device),
            indexing='ij'
        )
        
        grid = torch.stack([x_coords, y_coords], dim=-1).float()
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
        
        # Add flow to grid
        warped_grid = grid + flows[..., :2]  # Use only x, y components
        
        # Normalize to [-1, 1]
        warped_grid[..., 0] = 2 * warped_grid[..., 0] / (W - 1) - 1
        warped_grid[..., 1] = 2 * warped_grid[..., 1] / (H - 1) - 1
        
        # Sample using grid
        frames_flat = frames.view(B * T, C, H, W)
        warped_grid_flat = warped_grid.view(B * T, H, W, 2)
        
        warped_flat = F.grid_sample(frames_flat, warped_grid_flat, align_corners=False)
        
        return warped_flat.view(B, T, C, H, W)


class ControlMetrics:
    """Metrics for control signal accuracy and consistency."""
    
    @staticmethod
    def control_mse(pred_controls: torch.Tensor, target_controls: torch.Tensor) -> torch.Tensor:
        """MSE for control signals."""
        return F.mse_loss(pred_controls, target_controls)
    
    @staticmethod
    def control_mae(pred_controls: torch.Tensor, target_controls: torch.Tensor) -> torch.Tensor:
        """MAE for control signals."""
        return F.l1_loss(pred_controls, target_controls)
    
    @staticmethod
    def steering_accuracy(pred_controls: torch.Tensor, target_controls: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Steering angle accuracy within threshold."""
        steering_pred = pred_controls[..., 0]  # Assume first dimension is steering
        steering_target = target_controls[..., 0]
        
        error = torch.abs(steering_pred - steering_target)
        accuracy = (error < threshold).float().mean()
        
        return accuracy
    
    @staticmethod
    def acceleration_accuracy(pred_controls: torch.Tensor, target_controls: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Acceleration accuracy within threshold."""
        accel_pred = pred_controls[..., 1]  # Assume second dimension is acceleration
        accel_target = target_controls[..., 1]
        
        error = torch.abs(accel_pred - accel_target)
        accuracy = (error < threshold).float().mean()
        
        return accuracy
    
    @staticmethod
    def control_smoothness(controls: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness of control signals."""
        if controls.dim() == 3:  # [B, T, C]
            control_diff = controls[:, 1:] - controls[:, :-1]
            return control_diff.norm(dim=2).mean()
        elif controls.dim() == 2:  # [T, C]
            control_diff = controls[1:] - controls[:-1]
            return control_diff.norm(dim=1).mean()
        else:
            return torch.tensor(0.0, device=controls.device)


class DepthMetrics:
    """Metrics for depth estimation and consistency."""
    
    @staticmethod
    def depth_mse(pred_depth: torch.Tensor, target_depth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MSE for depth maps."""
        error = (pred_depth - target_depth)**2
        
        if mask is not None:
            error = error * mask
            return error.sum() / mask.sum().clamp(min=1)
        else:
            return error.mean()
    
    @staticmethod
    def depth_mae(pred_depth: torch.Tensor, target_depth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MAE for depth maps."""
        error = torch.abs(pred_depth - target_depth)
        
        if mask is not None:
            error = error * mask
            return error.sum() / mask.sum().clamp(min=1)
        else:
            return error.mean()
    
    @staticmethod
    def depth_relative_error(pred_depth: torch.Tensor, target_depth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Relative depth error."""
        relative_error = torch.abs(pred_depth - target_depth) / (target_depth.clamp(min=1e-6))
        
        if mask is not None:
            relative_error = relative_error * mask
            return relative_error.sum() / mask.sum().clamp(min=1)
        else:
            return relative_error.mean()
    
    @staticmethod
    def depth_gradient_consistency(depth: torch.Tensor) -> torch.Tensor:
        """Consistency of depth gradients (smoothness)."""
        grad_x = torch.abs(depth[..., :, 1:] - depth[..., :, :-1])
        grad_y = torch.abs(depth[..., 1:, :] - depth[..., :-1, :])
        
        return grad_x.mean() + grad_y.mean()


class PhysicsMetrics:
    """Physics-based metrics for autonomous driving scenarios."""
    
    @staticmethod
    def trajectory_smoothness(positions: torch.Tensor) -> torch.Tensor:
        """Smoothness of trajectory (acceleration variance)."""
        # positions: [T, 2] or [B, T, 2] (x, y coordinates)
        if positions.dim() == 3:  # [B, T, 2]
            velocity = positions[:, 1:] - positions[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            return torch.var(acceleration.view(-1, 2), dim=0).mean()
        else:  # [T, 2]
            velocity = positions[1:] - positions[:-1]
            acceleration = velocity[1:] - velocity[:-1]
            return torch.var(acceleration, dim=0).mean()
    
    @staticmethod
    def speed_consistency(positions: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Consistency of speed over time."""
        if positions.dim() == 3:  # [B, T, 2]
            velocity = (positions[:, 1:] - positions[:, :-1]) / dt
            speed = torch.norm(velocity, dim=2)
            return torch.var(speed, dim=1).mean()
        else:  # [T, 2]
            velocity = (positions[1:] - positions[:-1]) / dt
            speed = torch.norm(velocity, dim=1)
            return torch.var(speed)
    
    @staticmethod
    def turning_radius_validity(positions: torch.Tensor, min_radius: float = 1.0) -> torch.Tensor:
        """Check if turning radius is physically valid."""
        if positions.dim() == 3:
            B, T = positions.shape[:2]
            valid_ratios = []
            
            for b in range(B):
                ratio = PhysicsMetrics._compute_turning_validity(positions[b], min_radius)
                valid_ratios.append(ratio)
            
            return torch.tensor(valid_ratios, device=positions.device).mean()
        else:
            return PhysicsMetrics._compute_turning_validity(positions, min_radius)
    
    @staticmethod
    def _compute_turning_validity(positions: torch.Tensor, min_radius: float) -> torch.Tensor:
        """Compute turning radius validity for single trajectory."""
        if positions.shape[0] < 3:
            return torch.tensor(1.0, device=positions.device)
        
        # Compute curvature using three consecutive points
        valid_count = 0
        total_count = 0
        
        for i in range(1, positions.shape[0] - 1):
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # Compute curvature
            v1 = p2 - p1
            v2 = p3 - p2
            
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            norm_product = torch.norm(v1) * torch.norm(v2)
            
            if norm_product > 1e-6:
                sin_theta = torch.abs(cross) / norm_product
                if sin_theta > 1e-6:
                    radius = torch.norm(v1) / (2 * sin_theta)
                    if radius >= min_radius:
                        valid_count += 1
                else:
                    valid_count += 1  # Straight line is valid
            
            total_count += 1
        
        return torch.tensor(valid_count / max(total_count, 1), device=positions.device)


class PerceptualMetrics:
    """High-level perceptual metrics for generated content."""
    
    @staticmethod
    def feature_matching_distance(pred_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Distance between feature representations."""
        pred_flat = pred_features.view(pred_features.shape[0], -1)
        target_flat = target_features.view(target_features.shape[0], -1)
        
        return F.mse_loss(pred_flat, target_flat)
    
    @staticmethod
    def frechet_distance(pred_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Simplified Fréchet distance between feature distributions."""
        # Compute means
        mu_pred = pred_features.mean(dim=0)
        mu_target = target_features.mean(dim=0)
        
        # Compute covariances
        sigma_pred = torch.cov(pred_features.T)
        sigma_target = torch.cov(target_features.T)
        
        # Fréchet distance computation
        diff = mu_pred - mu_target
        
        # Simplified version (ignoring covariance matrix square root)
        trace_term = torch.trace(sigma_pred + sigma_target)
        
        return torch.norm(diff)**2 + trace_term
    
    @staticmethod
    def inception_score(features: torch.Tensor, epsilon: float = 1e-16) -> torch.Tensor:
        """Simplified Inception Score using feature distributions."""
        # Compute softmax probabilities
        p_yx = F.softmax(features, dim=1)
        
        # Marginal distribution
        p_y = p_yx.mean(dim=0, keepdim=True)
        
        # KL divergence
        kl_div = p_yx * (torch.log(p_yx + epsilon) - torch.log(p_y + epsilon))
        kl_div = kl_div.sum(dim=1).mean()
        
        return torch.exp(kl_div)


def compute_all_metrics(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    pred_controls: Optional[torch.Tensor] = None,
    target_controls: Optional[torch.Tensor] = None,
    pred_depth: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    # Video metrics
    metrics['mse'] = VideoMetrics.mse(pred_frames, target_frames)
    metrics['mae'] = VideoMetrics.mae(pred_frames, target_frames)
    metrics['psnr'] = VideoMetrics.psnr(pred_frames, target_frames)
    metrics['ssim'] = VideoMetrics.ssim(pred_frames, target_frames)
    metrics['lpips_proxy'] = VideoMetrics.lpips_proxy(pred_frames, target_frames)
    
    # Temporal consistency
    metrics['temporal_mse'] = TemporalConsistencyMetrics.temporal_mse(pred_frames)
    metrics['optical_flow_consistency'] = TemporalConsistencyMetrics.optical_flow_consistency(pred_frames)
    
    # Control metrics
    if pred_controls is not None and target_controls is not None:
        metrics['control_mse'] = ControlMetrics.control_mse(pred_controls, target_controls)
        metrics['control_mae'] = ControlMetrics.control_mae(pred_controls, target_controls)
        metrics['steering_accuracy'] = ControlMetrics.steering_accuracy(pred_controls, target_controls)
        metrics['acceleration_accuracy'] = ControlMetrics.acceleration_accuracy(pred_controls, target_controls)
        metrics['control_smoothness'] = ControlMetrics.control_smoothness(pred_controls)
    
    # Depth metrics
    if pred_depth is not None and target_depth is not None:
        metrics['depth_mse'] = DepthMetrics.depth_mse(pred_depth, target_depth)
        metrics['depth_mae'] = DepthMetrics.depth_mae(pred_depth, target_depth)
        metrics['depth_relative_error'] = DepthMetrics.depth_relative_error(pred_depth, target_depth)
        metrics['depth_gradient_consistency'] = DepthMetrics.depth_gradient_consistency(pred_depth)
    
    # Physics metrics
    if positions is not None:
        metrics['trajectory_smoothness'] = PhysicsMetrics.trajectory_smoothness(positions)
        metrics['speed_consistency'] = PhysicsMetrics.speed_consistency(positions)
        metrics['turning_radius_validity'] = PhysicsMetrics.turning_radius_validity(positions)
    
    return metrics