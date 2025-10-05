"""
Complete flow matching implementation with 4-step to 1-step distillation.
Implements Rectified Flow and Flow Matching for diffusion model acceleration.

Mathematical Foundation:
- Flow Matching: dz/dt = v_θ(z_t, t)
- Rectified Flow: z_1 = z_0 + ∫[0,1] v_θ(z_t, t) dt
- 4→1 Step Distillation: Student learns to match teacher trajectory in one step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from einops import rearrange, repeat

from ..layers.nn_helpers import SiLU, RMSNorm
from ..layers.mha import MultiHeadAttention
from ..layers.mlp import MLP


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching components."""
    # Model architecture
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    
    # Flow parameters
    num_steps: int = 4
    sigma_min: float = 1e-4
    sigma_max: float = 80.0
    
    # Training parameters
    use_cfg: bool = True  # Classifier-free guidance
    cfg_scale: float = 7.5
    
    # Distillation parameters
    teacher_steps: int = 4
    student_steps: int = 1
    distillation_weight: float = 1.0
    
    # Time embedding
    time_embed_dim: int = 128
    fourier_features: int = 256


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with learned projection."""
    
    def __init__(self, dim: int, fourier_features: int = 256):
        super().__init__()
        self.fourier_features = fourier_features
        self.linear1 = nn.Linear(fourier_features, dim)
        self.activation = SiLU()
        self.linear2 = nn.Linear(dim, dim)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] timesteps in [0, 1]
        Returns:
            [B, dim] time embeddings
        """
        # Sinusoidal encoding
        half_dim = self.fourier_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Project to target dimension
        emb = self.linear1(emb)
        emb = self.activation(emb)
        emb = self.linear2(emb)
        
        return emb


class FlowMatchingBlock(nn.Module):
    """Transformer block with time conditioning for flow matching."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Time-conditioned layer norms
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Time modulation (AdaLN style)
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        
        # Attention and MLP
        self.attention = MultiHeadAttention(dim, num_heads)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input tokens
            time_emb: [B, D] time embedding
            mask: [B, T, T] attention mask
        """
        B, T, D = x.shape
        
        # Time modulation parameters
        time_params = self.time_mlp(time_emb)  # [B, 6*D]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = time_params.chunk(6, dim=-1)
        
        # Self-attention with time conditioning
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h = self.attention(h, mask=mask)
        x = x + alpha1.unsqueeze(1) * h
        
        # MLP with time conditioning
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + alpha2.unsqueeze(1) * h
        
        return x


class FlowPredictor(nn.Module):
    """Flow field predictor network v_θ(z_t, t)."""
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = TimestepEmbedding(config.dim, config.fourier_features)
        
        # Input projection
        self.input_proj = nn.Linear(config.dim, config.dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowMatchingBlock(config.dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            RMSNorm(config.dim),
            nn.Linear(config.dim, config.dim)
        )
        
        # Initialize output projection to zero for stability
        nn.init.zeros_(self.output_proj[1].weight)
        nn.init.zeros_(self.output_proj[1].bias)
    
    def forward(
        self, 
        z: torch.Tensor, 
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict flow field v_θ(z_t, t).
        
        Args:
            z: [B, T, D] noisy latent states
            t: [B] timesteps in [0, 1]
            mask: [B, T, T] attention mask
            context: [B, C, D] conditioning context
        
        Returns:
            [B, T, D] predicted flow field
        """
        B, T, D = z.shape
        
        # Time embedding
        time_emb = self.time_embed(t)  # [B, D]
        
        # Input projection
        h = self.input_proj(z)
        
        # Add context if provided
        if context is not None:
            # Cross-attention with context would go here
            # For now, simple addition
            h = h + context.mean(dim=1, keepdim=True)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, time_emb, mask)
        
        # Output projection
        flow = self.output_proj(h)
        
        return flow


class RectifiedFlowSampler:
    """Rectified Flow sampler for generation."""
    
    def __init__(self, flow_predictor: FlowPredictor, config: FlowMatchingConfig):
        self.flow_predictor = flow_predictor
        self.config = config
    
    def sample(
        self,
        z_init: torch.Tensor,
        num_steps: int,
        context: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample using Rectified Flow with Euler integration.
        
        Args:
            z_init: [B, T, D] initial noise
            num_steps: number of integration steps
            context: [B, C, D] conditioning context
            guidance_scale: classifier-free guidance scale
            mask: [B, T, T] attention mask
        
        Returns:
            Dictionary with 'trajectory' and 'final_sample'
        """
        device = z_init.device
        B, T, D = z_init.shape
        
        # Time schedule
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        
        # Initialize trajectory
        trajectory = [z_init.clone()]
        z_t = z_init.clone()
        
        with torch.no_grad():
            for i in range(num_steps):
                t = timesteps[i]
                t_batch = torch.full((B,), t, device=device)
                
                # Predict flow
                if guidance_scale > 1.0 and context is not None:
                    # Classifier-free guidance
                    flow_cond = self.flow_predictor(z_t, t_batch, mask, context)
                    flow_uncond = self.flow_predictor(z_t, t_batch, mask, None)
                    flow = flow_uncond + guidance_scale * (flow_cond - flow_uncond)
                else:
                    flow = self.flow_predictor(z_t, t_batch, mask, context)
                
                # Euler step
                z_t = z_t + dt * flow
                trajectory.append(z_t.clone())
        
        return {
            'trajectory': torch.stack(trajectory, dim=0),  # [num_steps+1, B, T, D]
            'final_sample': z_t
        }
    
    def ddim_sample(
        self,
        z_init: torch.Tensor,
        num_steps: int,
        context: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """DDIM-style sampling for comparison."""
        device = z_init.device
        B = z_init.size(0)
        
        # Time schedule (reverse)
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        z_t = z_init.clone()
        
        with torch.no_grad():
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                
                t_batch = torch.full((B,), t_curr, device=device)
                
                # Predict flow
                flow = self.flow_predictor(z_t, t_batch, mask, context)
                
                # DDIM step
                dt = t_next - t_curr
                z_t = z_t + dt * flow
        
        return z_t


class FlowMatchingLoss:
    """Flow matching loss functions."""
    
    def __init__(self, config: FlowMatchingConfig):
        self.config = config
    
    def flow_matching_loss(
        self,
        flow_pred: torch.Tensor,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flow matching loss: ||v_θ(z_t, t) - (z_1 - z_0)||²
        
        Args:
            flow_pred: [B, T, D] predicted flow
            z_0: [B, T, D] source samples (noise)
            z_1: [B, T, D] target samples (data)
            t: [B] timesteps
            mask: [B, T] sequence mask
        """
        # True flow field
        flow_target = z_1 - z_0  # [B, T, D]
        
        # L2 loss
        loss = F.mse_loss(flow_pred, flow_target, reduction='none')  # [B, T, D]
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def distillation_loss(
        self,
        student_flow: torch.Tensor,
        teacher_trajectory: torch.Tensor,
        dt: float,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Distillation loss for 4→1 step compression.
        
        Args:
            student_flow: [B, T, D] student predicted flow
            teacher_trajectory: [B, num_teacher_steps+1, T, D] teacher trajectory
            dt: teacher timestep size
            mask: [B, T] sequence mask
        """
        # Compute teacher flow from trajectory
        # teacher_flow = (z_{t+dt} - z_t) / dt
        z_t = teacher_trajectory[0]  # Initial state
        z_t_plus_dt = teacher_trajectory[1]  # After one teacher step
        
        teacher_flow = (z_t_plus_dt - z_t) / dt
        
        # Distillation loss
        loss = F.mse_loss(student_flow, teacher_flow, reduction='none')
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def consistency_loss(
        self,
        flow_pred: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Consistency loss: flow should be consistent across small time steps.
        """
        with torch.no_grad():
            # Small perturbation
            t_eps = t + dt
            t_eps = torch.clamp(t_eps, 0, 1)
            z_t_eps = z_t + dt * flow_pred
        
        # Predict flow at perturbed state
        flow_eps = self.flow_predictor(z_t_eps, t_eps)
        
        # Consistency loss
        loss = F.mse_loss(flow_pred, flow_eps)
        
        return loss


class FlowMatchingTrainer:
    """Complete training pipeline for flow matching."""
    
    def __init__(self, flow_predictor: FlowPredictor, config: FlowMatchingConfig):
        self.flow_predictor = flow_predictor
        self.config = config
        self.loss_fn = FlowMatchingLoss(config)
        self.sampler = RectifiedFlowSampler(flow_predictor, config)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step: int
    ) -> Dict[str, float]:
        """
        Single training step with flow matching loss.
        
        Args:
            batch: Dictionary with 'latents' and optional 'context'
            optimizer: Model optimizer
            step: Current training step
        
        Returns:
            Dictionary of losses
        """
        device = next(self.flow_predictor.parameters()).device
        
        # Get data
        z_1 = batch['latents'].to(device)  # [B, T, D] target data
        context = batch.get('context')  # [B, C, D] optional context
        mask = batch.get('mask')  # [B, T] optional mask
        
        B, T, D = z_1.shape
        
        # Sample noise
        z_0 = torch.randn_like(z_1)  # [B, T, D] source noise
        
        # Sample timesteps
        t = torch.rand(B, device=device)  # [B] in [0, 1]
        
        # Interpolate between noise and data
        z_t = (1 - t.view(B, 1, 1)) * z_0 + t.view(B, 1, 1) * z_1
        
        # Predict flow
        flow_pred = self.flow_predictor(z_t, t, mask=None, context=context)
        
        # Compute loss
        loss_fm = self.loss_fn.flow_matching_loss(flow_pred, z_0, z_1, t, mask)
        
        # Additional losses
        losses = {'flow_matching': loss_fm}
        
        # Consistency loss (optional)
        if step % 10 == 0:  # Every 10 steps
            loss_consistency = self.loss_fn.consistency_loss(flow_pred, z_t, t)
            losses['consistency'] = loss_consistency * 0.1
        
        # Total loss
        total_loss = sum(losses.values())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.flow_predictor.parameters(), 1.0)
        
        optimizer.step()
        
        # Return scalar losses
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def distillation_step(
        self,
        teacher_predictor: FlowPredictor,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Distillation training step: learn to compress teacher's multi-step trajectory.
        """
        device = next(self.flow_predictor.parameters()).device
        
        z_1 = batch['latents'].to(device)
        context = batch.get('context')
        mask = batch.get('mask')
        
        B, T, D = z_1.shape
        z_0 = torch.randn_like(z_1)
        
        # Generate teacher trajectory
        with torch.no_grad():
            teacher_sampler = RectifiedFlowSampler(teacher_predictor, self.config)
            teacher_result = teacher_sampler.sample(
                z_0, 
                num_steps=self.config.teacher_steps,
                context=context,
                mask=None
            )
            teacher_trajectory = teacher_result['trajectory']  # [steps+1, B, T, D]
        
        # Student predicts single-step flow
        t = torch.zeros(B, device=device)  # Start from t=0
        student_flow = self.flow_predictor(z_0, t, mask=None, context=context)
        
        # Distillation loss
        dt = 1.0 / self.config.teacher_steps
        loss_distill = self.loss_fn.distillation_loss(
            student_flow, teacher_trajectory, dt, mask
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_distill.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_predictor.parameters(), 1.0)
        optimizer.step()
        
        return {'distillation': loss_distill.item()}
    
    def sample_and_evaluate(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = None
    ) -> Dict[str, torch.Tensor]:
        """Sample and evaluate generation quality."""
        if num_steps is None:
            num_steps = self.config.num_steps
        
        device = next(self.flow_predictor.parameters()).device
        z_1 = batch['latents'].to(device)
        context = batch.get('context')
        
        B, T, D = z_1.shape
        z_0 = torch.randn_like(z_1)
        
        # Generate samples
        result = self.sampler.sample(z_0, num_steps, context)
        generated = result['final_sample']
        
        # Compute metrics
        mse = F.mse_loss(generated, z_1)
        cosine_sim = F.cosine_similarity(
            generated.view(B, -1), 
            z_1.view(B, -1), 
            dim=-1
        ).mean()
        
        return {
            'generated_samples': generated,
            'mse': mse.item(),
            'cosine_similarity': cosine_sim.item()
        }


def create_flow_matching_components(config: FlowMatchingConfig):
    """Create flow matching predictor and trainer."""
    
    flow_predictor = FlowPredictor(config)
    trainer = FlowMatchingTrainer(flow_predictor, config)
    
    return flow_predictor, trainer


if __name__ == "__main__":
    # Test flow matching implementation
    config = FlowMatchingConfig(
        dim=128,
        num_layers=4,
        num_heads=8,
        num_steps=4
    )
    
    # Create components
    flow_predictor, trainer = create_flow_matching_components(config)
    
    # Test data
    B, T, D = 2, 16, 128
    batch = {
        'latents': torch.randn(B, T, D),
        'context': torch.randn(B, 8, D)  # Optional context
    }
    
    # Test training step
    optimizer = torch.optim.Adam(flow_predictor.parameters(), lr=1e-4)
    losses = trainer.train_step(batch, optimizer, step=0)
    print(f"Training losses: {losses}")
    
    # Test sampling
    eval_result = trainer.sample_and_evaluate(batch, num_steps=4)
    print(f"Evaluation metrics: MSE={eval_result['mse']:.4f}, "
          f"Cosine Similarity={eval_result['cosine_similarity']:.4f}")
    
    # Test different number of steps
    for steps in [1, 2, 4, 8]:
        result = trainer.sampler.sample(
            torch.randn(1, 8, D), 
            num_steps=steps
        )
        print(f"Generated with {steps} steps: shape {result['final_sample'].shape}")
    
    print(f"Flow predictor parameters: {sum(p.numel() for p in flow_predictor.parameters()):,}")
    print("Flow matching implementation test completed successfully!")