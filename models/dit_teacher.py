"""
DiT Teacher Model - Bidirectional World Model
Non-causal diffusion transformer for teacher distillation.
Processes entire sequences bidirectionally for high-quality flow predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blocks.dit_block import DiTBlock
from layers.rope import RoPELayer
from layers.nn_helpers import AdaLN, RMSNorm, PositionalEncoding
from blocks.flow_match import FlowMatchingSampler, FlowLoss


class TeacherEmbedding(nn.Module):
    """
    Embedding layer for teacher model with noise level conditioning.
    """
    
    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        max_seq_len: int = 2048,
        time_embed_dim: int = 256
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim
        
        # Latent token projection
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Time/noise level embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Context embedding (for conditioning)
        self.context_proj = nn.Linear(latent_dim, d_model)
        
        # Position embedding
        self.pos_embed = PositionalEncoding(d_model, max_seq_len)
        
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: Timestep values [B] or [B, 1]
        
        Returns:
            Timestep embeddings [B, time_embed_dim]
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
        
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.time_embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        
        return emb
    
    def forward(
        self,
        latent_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        add_pos_embed: bool = True
    ) -> torch.Tensor:
        """
        Embed tokens with noise level conditioning.
        
        Args:
            latent_tokens: Latent tokens [B, T, latent_dim]
            timesteps: Noise level timesteps [B]
            context: Optional context tokens [B, T_ctx, latent_dim]
            add_pos_embed: Whether to add positional embeddings
        
        Returns:
            Embedded tokens [B, T, d_model]
        """
        # Project latent tokens
        x = self.latent_proj(latent_tokens)
        
        # Get timestep embeddings
        time_emb = self.get_timestep_embedding(timesteps)
        time_emb = self.time_embed(time_emb)  # [B, d_model]
        
        # Add timestep conditioning
        x = x + time_emb.unsqueeze(1)  # Broadcast to [B, T, d_model]
        
        # Add context if provided
        if context is not None:
            context_emb = self.context_proj(context)
            # For simplicity, add mean context embedding
            x = x + context_emb.mean(dim=1, keepdim=True)
        
        # Add positional embeddings
        if add_pos_embed:
            x = self.pos_embed(x)
        
        return x


class DiTTeacher(nn.Module):
    """
    DiT Teacher model for bidirectional world modeling.
    Non-causal transformer that processes entire sequences for distillation.
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        d_ff: Optional[int] = None,
        mlp_type: str = 'swiglu',
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        time_embed_dim: int = 256,
        use_rope: bool = True,
        bias: bool = False,
        num_diffusion_steps: int = 1000
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        
        # Token embedding with noise conditioning
        self.embed = TeacherEmbedding(
            latent_dim=latent_dim,
            d_model=d_model,
            max_seq_len=max_seq_len,
            time_embed_dim=time_embed_dim
        )
        
        # RoPE if enabled
        if use_rope:
            self.rope_layer = RoPELayer(
                dim=d_model // n_heads,
                max_seq_len=max_seq_len
            )
        else:
            self.rope_layer = None
        
        # Transformer blocks (bidirectional)
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                mlp_type=mlp_type,
                dropout=dropout,
                causal=False,  # Bidirectional for teacher
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                cond_dim=d_model,  # Conditioning dimension
                bias=bias
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(d_model)
        
        # Output heads
        self.flow_head = nn.Linear(d_model, latent_dim, bias=False)  # Flow prediction
        self.eps_head = nn.Linear(d_model, latent_dim, bias=False)   # Noise prediction (optional)
        
        # Flow matching sampler for inference
        self.flow_sampler = FlowMatchingSampler(
            num_steps=4,  # 4-step distillation target
            sigma_min=0.002,
            sigma_max=80.0
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        return_eps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through DiT teacher.
        
        Args:
            noisy_latents: Noisy latent tokens [B, T, latent_dim]
            timesteps: Diffusion timesteps [B]
            context: Optional context for conditioning [B, T_ctx, latent_dim]  
            conditioning: Optional conditioning vector [B, d_model]
            return_eps: Whether to return noise prediction as well
        
        Returns:
            Flow prediction [B, T, latent_dim] or tuple with eps prediction
        """
        B, T, D = noisy_latents.shape
        
        # Embed tokens with noise level conditioning
        x = self.embed(
            latent_tokens=noisy_latents,
            timesteps=timesteps,
            context=context
        )
        
        # Forward through transformer blocks
        for block in self.blocks:
            x, _ = block(
                x=x,
                cond=conditioning  # Pass conditioning to AdaLN
            )
        
        # Final layer norm  
        x = self.norm(x)
        
        # Predict flow
        flow_pred = self.flow_head(x)
        
        if return_eps:
            # Also predict noise (for compatibility with other diffusion models)
            eps_pred = self.eps_head(x)
            return flow_pred, eps_pred
        
        return flow_pred
    
    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to clean latents according to diffusion schedule.
        
        Args:
            clean_latents: Clean latent tokens [B, T, latent_dim]
            noise: Noise to add [B, T, latent_dim]
            timesteps: Timestep indices [B]
        
        Returns:
            Noisy latents [B, T, latent_dim]
        """
        # Get noise schedule values
        sigmas = self.flow_sampler.sigmas
        alpha_t = sigmas[timesteps].view(-1, 1, 1)  # [B, 1, 1]
        
        # Add noise: x_t = x_0 + sigma_t * epsilon
        noisy_latents = clean_latents + alpha_t * noise
        
        return noisy_latents
    
    def compute_loss(
        self,
        clean_latents: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        loss_type: str = 'flow_matching'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for teacher model.
        
        Args:
            clean_latents: Clean latent sequence [B, T, latent_dim]
            context: Optional context [B, T_ctx, latent_dim]
            conditioning: Optional conditioning [B, d_model]
            loss_type: Type of loss ('flow_matching', 'diffusion')
        
        Returns:
            Dictionary of losses
        """
        B, T, D = clean_latents.shape
        device = clean_latents.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_diffusion_steps, 
            (B,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(clean_latents)
        
        # Add noise to latents
        noisy_latents = self.add_noise(clean_latents, noise, timesteps)
        
        if loss_type == 'flow_matching':
            # Flow matching loss
            flow_pred = self.forward(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                context=context,
                conditioning=conditioning
            )
            
            # Target is the noise itself for flow matching
            target_flow = noise
            flow_loss = F.mse_loss(flow_pred, target_flow)
            
            return {'flow_loss': flow_loss, 'total_loss': flow_loss}
            
        elif loss_type == 'diffusion':
            # Standard diffusion loss (noise prediction)
            flow_pred, eps_pred = self.forward(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                context=context,
                conditioning=conditioning,
                return_eps=True
            )
            
            # Noise prediction loss
            eps_loss = F.mse_loss(eps_pred, noise)
            
            # Optional flow loss
            flow_loss = F.mse_loss(flow_pred, noise)
            
            total_loss = eps_loss + 0.1 * flow_loss
            
            return {
                'eps_loss': eps_loss,
                'flow_loss': flow_loss,
                'total_loss': total_loss
            }
        
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def sample(
        self,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Sample from the teacher model using DDPM sampling.
        
        Args:
            shape: Shape of samples to generate (B, T, latent_dim)
            context: Optional context
            conditioning: Optional conditioning
            num_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            return_intermediates: Whether to return intermediate steps
        
        Returns:
            Generated samples [B, T, latent_dim]
        """
        device = next(self.parameters()).device
        B, T, D = shape
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.num_diffusion_steps - 1, 0, num_steps, device=device
        ).long()
        
        intermediates = [x] if return_intermediates else []
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            
            # Predict flow/noise
            with torch.no_grad():
                flow_pred = self.forward(
                    noisy_latents=x,
                    timesteps=t_batch,
                    context=context,
                    conditioning=conditioning
                )
            
            # Apply classifier-free guidance if enabled
            if guidance_scale != 1.0 and context is not None:
                # Unconditional prediction
                flow_uncond = self.forward(
                    noisy_latents=x,
                    timesteps=t_batch,
                    context=None,
                    conditioning=None
                )
                
                # Guided prediction
                flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)
            
            # DDPM step (simplified)
            if i < len(timesteps) - 1:
                alpha = 0.98  # Simplified noise schedule
                x = alpha * x + (1 - alpha) * flow_pred
            else:
                x = flow_pred
            
            if return_intermediates:
                intermediates.append(x)
        
        return intermediates if return_intermediates else x
    
    def distill_to_student(
        self,
        student_model: nn.Module,
        clean_latents: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_distill_steps: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Distill teacher knowledge to student model using flow matching.
        
        Args:
            student_model: Student model to distill to
            clean_latents: Clean latent sequence [B, T, latent_dim]
            context: Optional context
            num_distill_steps: Number of distillation steps
        
        Returns:
            Dictionary of distillation losses
        """
        B, T, D = clean_latents.shape
        device = clean_latents.device
        
        # Sample timesteps for distillation
        timesteps = torch.randint(
            0, self.num_diffusion_steps, 
            (B,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(clean_latents)
        noisy_latents = self.add_noise(clean_latents, noise, timesteps)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_flow = self.forward(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                context=context
            )
        
        # Student forward pass
        student_flow, _ = student_model(noisy_latents)
        
        # Distillation loss
        distill_loss = F.mse_loss(student_flow, teacher_flow)
        
        return {'distill_loss': distill_loss}


def test_dit_teacher():
    """Test function for DiT Teacher."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DiTTeacher(
        latent_dim=8,
        d_model=512,  # Smaller for testing
        n_layers=6,   # Fewer layers for testing
        n_heads=8,
        max_seq_len=128,
        use_rope=True,
        num_diffusion_steps=100  # Fewer steps for testing
    ).to(device)
    
    # Test input
    B, T, D = 2, 32, 8
    clean_latents = torch.randn(B, T, D).to(device)
    timesteps = torch.randint(0, 100, (B,)).to(device)
    noise = torch.randn_like(clean_latents)
    
    print(f"Input shape: {clean_latents.shape}")
    
    # Add noise
    noisy_latents = model.add_noise(clean_latents, noise, timesteps)
    print(f"Noisy latents shape: {noisy_latents.shape}")
    
    # Test forward pass
    flow_pred = model(noisy_latents, timesteps)
    print(f"Flow prediction shape: {flow_pred.shape}")
    
    # Test loss computation
    losses = model.compute_loss(clean_latents)
    print(f"Loss: {losses['total_loss'].item():.4f}")
    
    # Test sampling
    samples = model.sample(
        shape=(1, 16, 8),
        num_steps=10  # Fewer steps for testing
    )
    print(f"Sample shape: {samples.shape}")


if __name__ == "__main__":
    test_dit_teacher()