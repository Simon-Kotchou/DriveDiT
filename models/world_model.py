"""
Unified world model integrating all components:
- Self-forcing training methodology
- Flow matching and distillation
- Modular components (control, depth, memory, JEPA)
- Clean integration with unified training pipeline

This replaces the scattered model implementations with a single, cohesive world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
import sys
import os
from einops import rearrange, repeat

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DriveDiTConfig, ComponentType
from layers.mha import MultiHeadAttention
from layers.mlp import MLP
from layers.nn_helpers import RMSNorm
from blocks.dit_block import DiTBlock

# Use torch's built-in SiLU
SiLU = nn.SiLU


class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class ControlEncoder(nn.Module):
    """Encode control signals for conditioning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        return self.encoder(controls)


class DepthEncoder(nn.Module):
    """Encode depth information."""
    
    def __init__(self, input_dim: int, output_dim: int, max_depth: float = 100.0):
        super().__init__()
        self.max_depth = max_depth
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(2)
        )
    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # Normalize depth
        depth_norm = torch.clamp(depth / self.max_depth, 0, 1)
        x = self.encoder(depth_norm)
        x = x.transpose(-2, -1)  # [B, 16*16, output_dim]
        return x


class MemorySystem(nn.Module):
    """Simple memory system for object and spatial permanence."""
    
    def __init__(self, dim: int, max_objects: int = 64, max_spatial: int = 256):
        super().__init__()
        self.dim = dim
        self.max_objects = max_objects
        self.max_spatial = max_spatial
        
        # Memory banks
        self.object_memory = nn.Parameter(torch.randn(max_objects, dim))
        self.spatial_memory = nn.Parameter(torch.randn(max_spatial, dim))
        
        # Memory update networks
        self.object_update = nn.Linear(dim, dim)
        self.spatial_update = nn.Linear(dim, dim)
        
        # Memory retrieval
        self.memory_retrieval = MultiHeadAttention(dim, 8)
    
    def forward(self, hidden_states: torch.Tensor, step: int = 0) -> torch.Tensor:
        """Update and retrieve memory."""
        B, T, D = hidden_states.shape
        
        # Combine object and spatial memory
        memory_tokens = torch.cat([
            self.object_memory.unsqueeze(0).repeat(B, 1, 1),
            self.spatial_memory.unsqueeze(0).repeat(B, 1, 1)
        ], dim=1)  # [B, max_objects + max_spatial, dim]
        
        # Simple memory retrieval (cross-attention)
        query = hidden_states.mean(dim=1, keepdim=True)  # [B, 1, D]
        retrieved_memory = self.memory_retrieval(query, kv=memory_tokens)  # [B, 1, D]
        
        return retrieved_memory.repeat(1, T, 1)  # [B, T, D]
    
    def get_memory_tokens(self, batch_size: int) -> torch.Tensor:
        """Get memory tokens for conditioning."""
        memory_tokens = torch.cat([self.object_memory, self.spatial_memory], dim=0)
        return memory_tokens.unsqueeze(0).repeat(batch_size, 1, 1)


class JEPAPredictor(nn.Module):
    """V-JEPA style predictor for future frame representations."""
    
    def __init__(self, input_dim: int, target_length: int, prediction_head_dim: int):
        super().__init__()
        self.target_length = target_length
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, prediction_head_dim),
            nn.ReLU(),
            nn.Linear(prediction_head_dim, input_dim * target_length)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict future representations."""
        B, T, D = hidden_states.shape
        
        # Use last few tokens for prediction
        context = hidden_states[:, -4:].mean(dim=1)  # Average last 4 tokens
        
        predictions = self.predictor(context)
        predictions = predictions.view(B, self.target_length, D)
        
        return predictions


class FlowMatchingPredictor(nn.Module):
    """Flow matching predictor for diffusion acceleration."""
    
    def __init__(self, dim: int, num_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        
        self.flow_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            SiLU(),
            nn.Linear(dim * 2, dim),
            SiLU(),
            nn.Linear(dim, dim),
            nn.Tanh()  # Bounded flow predictions
        )
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict flow field v_θ(z_t, t)."""
        # Simple implementation - in full version would include time embedding
        flow = self.flow_net(z)
        return flow
    
    def sample(self, z_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Sample using Euler integration."""
        dt = 1.0 / num_steps
        z_t = z_init.clone()
        
        for step in range(num_steps):
            t = torch.full((z_t.size(0),), step * dt, device=z_t.device)
            flow = self.forward(z_t, t)
            z_t = z_t + dt * flow
        
        return z_t


class SelfForcingScheduler:
    """Scheduler for self-forcing ratio during training."""
    
    def __init__(self, config: DriveDiTConfig):
        self.config = config
        self.step = 0
        
    def update(self, step: int):
        """Update internal step counter."""
        self.step = step
    
    def get_ratio(self) -> float:
        """Get current self-forcing ratio."""
        if self.config.self_forcing_schedule == "linear":
            progress = min(1.0, self.step / self.config.self_forcing_warmup_steps)
            return self.config.initial_self_forcing_ratio + progress * (
                self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio
            )
        elif self.config.self_forcing_schedule == "cosine":
            progress = min(1.0, self.step / self.config.self_forcing_warmup_steps)
            cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return self.config.initial_self_forcing_ratio + cos_progress * (
                self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio
            )
        else:  # exponential
            decay = 0.999 ** self.step
            return self.config.final_self_forcing_ratio + (
                self.config.initial_self_forcing_ratio - self.config.final_self_forcing_ratio
            ) * decay


class WorldModel(nn.Module):
    """
    Unified world model combining all methodologies:
    - Self-forcing training with curriculum learning
    - Flow matching and distillation
    - Modular components (control, depth, memory, JEPA)
    """
    
    def __init__(self, config: DriveDiTConfig):
        super().__init__()
        self.config = config
        self.enabled_components = config.get_enabled_components()
        
        # Calculate max sequence length based on config
        patches_per_frame = (config.image_size // config.patch_size) ** 2
        max_seq_len = patches_per_frame * config.final_sequence_length + 512  # Extra for context

        # Core transformer backbone
        self.backbone = nn.ModuleList([
            DiTBlock(
                d_model=config.model_dim,
                n_heads=config.num_heads,
                d_ff=config.model_dim * config.mlp_ratio,
                causal=True,
                max_seq_len=max_seq_len
            )
            for _ in range(config.num_layers)
        ])
        
        # Input encoders
        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=config.model_dim
        )

        # Note: RoPE is handled internally by DiTBlock when use_rope=True

        # Optional components based on config
        self._init_optional_components()
        
        # Output heads - each token predicts its own patch latent
        self.frame_head = nn.Linear(config.model_dim, config.vae_latent_dim)
        
        # Self-forcing scheduler
        self.self_forcing_scheduler = SelfForcingScheduler(config)
        
        # Cache for autoregressive generation
        self.past_kvs = None
        
    def _init_optional_components(self):
        """Initialize components based on configuration."""
        config = self.config
        
        # Control encoder
        if ComponentType.CONTROL in self.enabled_components:
            self.control_encoder = ControlEncoder(
                input_dim=config.control_input_dim,
                hidden_dim=config.control_hidden_dim,
                output_dim=config.model_dim,
                num_layers=config.control_num_layers,
                dropout=config.control_dropout
            )
        else:
            self.control_encoder = None
        
        # Depth encoder
        if ComponentType.DEPTH in self.enabled_components:
            self.depth_encoder = DepthEncoder(
                input_dim=config.depth_channels,
                output_dim=config.model_dim,
                max_depth=config.depth_max_depth
            )
        else:
            self.depth_encoder = None
        
        # Memory system
        if ComponentType.MEMORY in self.enabled_components:
            self.memory_system = MemorySystem(
                dim=config.memory_dim,
                max_objects=config.object_memory_size,
                max_spatial=config.spatial_memory_size
            )
        else:
            self.memory_system = None
        
        # Flow matching predictor
        if ComponentType.FLOW_MATCHING in self.enabled_components:
            self.flow_predictor = FlowMatchingPredictor(
                dim=config.model_dim,
                num_steps=config.num_flow_steps
            )
        else:
            self.flow_predictor = None
        
        # JEPA predictor
        if ComponentType.JEPA in self.enabled_components:
            self.jepa_predictor = JEPAPredictor(
                input_dim=config.model_dim,
                target_length=config.jepa_target_length,
                prediction_head_dim=config.jepa_prediction_head_dim
            )
        else:
            self.jepa_predictor = None
    
    def forward(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mode: str = "train",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with different modes.
        
        Args:
            frames: [B, T, C, H, W] input video frames
            controls: [B, T, control_dim] control signals (optional)
            depth: [B, T, 1, H, W] depth maps (optional)
            mode: "train", "inference", or "self_forcing"
        """
        if mode == "train":
            return self._forward_train(frames, controls, depth, **kwargs)
        elif mode == "inference":
            return self._forward_inference(frames, controls, depth, **kwargs)
        elif mode == "self_forcing":
            return self._forward_self_forcing(frames, controls, depth, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _forward_train(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Standard training forward pass."""
        B, T, C, H, W = frames.shape
        
        # Patch embedding
        frame_tokens = self.patch_embed(frames.reshape(-1, C, H, W))
        frame_tokens = rearrange(frame_tokens, '(b t) n d -> b (t n) d', b=B, t=T)
        
        # Add optional modalities
        context_tokens = []
        
        if self.control_encoder is not None and controls is not None:
            control_tokens = self.control_encoder(controls.reshape(-1, controls.size(-1)))
            control_tokens = rearrange(control_tokens, '(b t) d -> b t d', b=B, t=T)
            context_tokens.append(control_tokens)
        
        if self.depth_encoder is not None and depth is not None:
            depth_tokens = self.depth_encoder(depth.reshape(-1, *depth.shape[-3:]))
            depth_tokens = rearrange(depth_tokens, '(b t) n d -> b t n d', b=B, t=T)
            depth_tokens = rearrange(depth_tokens, 'b t n d -> b (t n) d')
            context_tokens.append(depth_tokens)
        
        # Combine context
        if context_tokens:
            context = torch.cat(context_tokens, dim=1)
            all_tokens = torch.cat([frame_tokens, context], dim=1)
        else:
            all_tokens = frame_tokens
        
        # Transformer forward (DiTBlock handles RoPE internally)
        hidden_states = all_tokens
        for layer in self.backbone:
            hidden_states, _ = layer(hidden_states)
        
        # Extract frame tokens
        num_frame_tokens = frame_tokens.size(1)
        frame_hidden = hidden_states[:, :num_frame_tokens]
        
        # Predictions
        outputs = {}
        
        # Frame prediction
        frame_pred = self.frame_head(frame_hidden)
        outputs['predictions'] = self._decode_frame_tokens(frame_pred, B, T)
        outputs['hidden_states'] = frame_hidden
        
        # Optional component predictions
        if self.jepa_predictor is not None:
            jepa_pred = self.jepa_predictor(frame_hidden)
            outputs['jepa_predictions'] = jepa_pred
        
        if self.flow_predictor is not None:
            # Generate random time steps for flow matching
            t = torch.rand(B, device=frames.device)
            flow_pred = self.flow_predictor(frame_hidden.mean(dim=1), t)
            outputs['flow_predictions'] = flow_pred
        
        return outputs
    
    def _forward_inference(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive inference for generation."""
        B, T, C, H, W = frames.shape
        
        generated_frames = []
        current_frames = frames
        
        with torch.no_grad():
            for step in range(num_steps):
                # Standard forward pass
                outputs = self._forward_train(current_frames, controls, depth)
                
                # Get next frame
                next_frame = outputs['predictions'][:, -1:]  # Last predicted frame
                generated_frames.append(next_frame)
                
                # Update context window
                current_frames = torch.cat([current_frames[:, 1:], next_frame], dim=1)
        
        return {
            'generated_frames': torch.cat(generated_frames, dim=1),
            'final_states': outputs['hidden_states']
        }
    
    def _forward_self_forcing(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        self_forcing_ratio: Optional[float] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Self-forcing training: mix ground truth and model predictions.
        """
        B, T, C, H, W = frames.shape
        
        # Get self-forcing ratio
        if self_forcing_ratio is None:
            sf_ratio = self.self_forcing_scheduler.get_ratio()
        else:
            sf_ratio = self_forcing_ratio
        
        # Split into context and target
        context_length = T // 2
        context_frames = frames[:, :context_length]
        target_frames = frames[:, context_length:]

        # Slice controls and depth to match context
        context_controls = controls[:, :context_length] if controls is not None else None
        context_depth = depth[:, :context_length] if depth is not None else None

        # Process context normally
        context_outputs = self._forward_train(context_frames, context_controls, context_depth)
        
        # Autoregressive generation with self-forcing
        predictions = []
        current_frame = context_frames[:, -1:]  # Last context frame
        
        for t in range(target_frames.size(1)):
            # Decide whether to use ground truth or prediction
            use_gt = torch.rand(1).item() > sf_ratio
            
            if use_gt and t < target_frames.size(1):
                # Use ground truth
                input_frame = target_frames[:, t:t+1]
            else:
                # Use model prediction
                input_frame = current_frame
            
            # Create input sequence (context + current)
            input_sequence = torch.cat([context_frames, input_frame], dim=1)
            seq_len = input_sequence.size(1)

            # Slice controls and depth to match input sequence length
            seq_controls = controls[:, :seq_len] if controls is not None else None
            seq_depth = depth[:, :seq_len] if depth is not None else None

            # Forward pass
            outputs = self._forward_train(input_sequence, seq_controls, seq_depth)
            
            # Get prediction for next frame
            frame_pred = outputs['predictions'][:, -1:]  # Last frame
            predictions.append(frame_pred)
            
            # Update current frame for next iteration
            current_frame = frame_pred
        
        return {
            'predictions': torch.cat(predictions, dim=1),
            'targets': target_frames,
            'self_forcing_ratio': sf_ratio,
            'context_length': context_length,
            'hidden_states': context_outputs['hidden_states']
        }
    
    def _decode_frame_tokens(self, frame_tokens: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Decode frame tokens to actual frames."""
        # Reshape frame tokens back to spatial format
        # Input: [B, T * num_patches², vae_latent_dim]
        # Output: [B, T, vae_latent_dim, num_patches, num_patches]
        num_patches = self.config.image_size // self.config.patch_size

        # Reshape to [B, T, num_patches, num_patches, vae_latent_dim]
        frames = rearrange(
            frame_tokens,
            'b (t h w) c -> b t c h w',
            t=T,
            h=num_patches,
            w=num_patches,
            c=self.config.vae_latent_dim
        )
        
        # Upsample to original resolution (simplified)
        frames = F.interpolate(
            frames.reshape(-1, *frames.shape[-3:]),
            size=(self.config.image_size, self.config.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to RGB if needed
        if self.config.vae_latent_dim != 3:
            frames = F.conv2d(
                frames,
                torch.ones(3, self.config.vae_latent_dim, 1, 1, device=frames.device),
                bias=None
            )
        
        frames = torch.sigmoid(frames)  # Normalize to [0, 1]
        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=B, t=T)
        
        return frames
    
    def reset_cache(self):
        """Reset cache for new sequence."""
        self.past_kvs = None
    
    def update_self_forcing_step(self, step: int):
        """Update self-forcing scheduler."""
        self.self_forcing_scheduler.update(step)


def create_world_model(config: DriveDiTConfig) -> WorldModel:
    """Create world model with configuration."""
    return WorldModel(config)


if __name__ == "__main__":
    # Test unified world model
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import get_minimal_config

    # Use minimal config for faster testing
    config = get_minimal_config()
    model = create_world_model(config)

    # Test inputs (smaller dimensions for faster testing)
    B, T = 2, 4
    C, H, W = config.in_channels, config.image_size, config.image_size
    frames = torch.randn(B, T, C, H, W)
    controls = torch.randn(B, T, config.control_input_dim) if config.is_component_enabled(ComponentType.CONTROL) else None
    depth = torch.randn(B, T, config.depth_channels, H, W) if config.is_component_enabled(ComponentType.DEPTH) else None

    # Test different modes
    print("Testing training mode...")
    train_outputs = model(frames, controls, depth, mode="train")
    print(f"Training outputs: {list(train_outputs.keys())}")

    print("Testing self-forcing mode...")
    sf_outputs = model(frames, controls, depth, mode="self_forcing", self_forcing_ratio=0.5)
    print(f"Self-forcing outputs: {list(sf_outputs.keys())}")

    print("Testing inference mode...")
    context_frames = frames[:, :2]  # Use first 2 frames as context
    context_controls = controls[:, :2] if controls is not None else None
    context_depth = depth[:, :2] if depth is not None else None
    inf_outputs = model(context_frames, context_controls, context_depth, mode="inference", num_steps=2)
    print(f"Inference outputs: {list(inf_outputs.keys())}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Unified world model test completed successfully!")