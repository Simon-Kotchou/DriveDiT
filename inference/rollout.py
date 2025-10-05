"""
Real-time inference rollout for DriveDiT world modeling.
Single-GPU streaming inference with KV caching and memory management.
Pure mathematical components with explicit tensor operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time
from collections import deque, defaultdict
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit_student import DiTStudent
from models.vae3d import VAE3D
from blocks.flow_match import FlowMatchingSampler


class MemoryBank:
    """
    Memory bank for storing spatial and object information.
    Implements efficient memory management for long sequences.
    """
    
    def __init__(
        self,
        d_model: int = 1024,
        max_spatial_memory: int = 2048,
        max_object_memory: int = 512,
        decay_rate: float = 0.95
    ):
        self.d_model = d_model
        self.max_spatial_memory = max_spatial_memory
        self.max_object_memory = max_object_memory
        self.decay_rate = decay_rate
        
        # Memory storage
        self.spatial_memory = deque(maxlen=max_spatial_memory)
        self.object_memory = deque(maxlen=max_object_memory)
        self.memory_scores = deque(maxlen=max_spatial_memory)
        
        # Frame tracking
        self.frame_count = 0
        
    def update(self, frame: torch.Tensor, frame_idx: int):
        """
        Update memory with new frame information.
        
        Args:
            frame: Current frame [B, 3, H, W]
            frame_idx: Frame index in sequence
        """
        B, C, H, W = frame.shape
        
        # Extract spatial features (simplified)
        spatial_features = F.adaptive_avg_pool2d(frame, (8, 8)).flatten(1)  # [B, 3*64]
        
        # Add to spatial memory
        self.spatial_memory.append(spatial_features.detach())
        
        # Compute importance score based on frame variance
        frame_var = frame.var(dim=(2, 3)).mean()
        self.memory_scores.append(frame_var.item())
        
        # Object detection (simplified - use frame gradients as proxy)
        grad_x = frame[:, :, :, 1:] - frame[:, :, :, :-1]
        grad_y = frame[:, :, 1:, :] - frame[:, :, :-1, :]
        edge_strength = grad_x.abs().mean() + grad_y.abs().mean()
        
        if edge_strength > 0.1:  # Threshold for "interesting" objects
            object_features = F.adaptive_avg_pool2d(frame, (4, 4)).flatten(1)
            self.object_memory.append(object_features.detach())
        
        self.frame_count += 1
    
    def get_memory_tokens(self, top_k: int = 64) -> torch.Tensor:
        """
        Retrieve top-k most relevant memory tokens.
        
        Args:
            top_k: Number of top memory tokens to retrieve
        
        Returns:
            Memory tokens [1, top_k, d_model]
        """
        if not self.spatial_memory:
            return torch.zeros(1, 0, self.d_model)
        
        # Convert memory to tensor
        spatial_mem = torch.stack(list(self.spatial_memory), dim=0)  # [T, B, features]
        
        # Get importance scores
        scores = torch.tensor(list(self.memory_scores))
        
        # Apply decay based on recency
        decay_weights = torch.exp(-0.1 * torch.arange(len(scores), 0, -1, dtype=torch.float32))
        weighted_scores = scores * decay_weights
        
        # Select top-k
        if len(weighted_scores) <= top_k:
            selected_memory = spatial_mem
        else:
            _, top_indices = torch.topk(weighted_scores, k=top_k)
            top_indices = torch.sort(top_indices)[0]  # Keep temporal order
            selected_memory = spatial_mem[top_indices]
        
        # Project to model dimension (simplified projection)
        if selected_memory.numel() > 0:
            memory_tokens = F.linear(
                selected_memory.view(-1, selected_memory.size(-1)),
                torch.randn(self.d_model, selected_memory.size(-1)) * 0.02
            )
            return memory_tokens.unsqueeze(0)  # [1, selected_k, d_model]
        else:
            return torch.zeros(1, 0, self.d_model)
    
    def clear(self):
        """Clear all memory."""
        self.spatial_memory.clear()
        self.object_memory.clear()
        self.memory_scores.clear()
        self.frame_count = 0


class InferenceConfig:
    """Configuration for streaming inference."""
    
    def __init__(
        self,
        max_sequence_length: int = 300,
        context_window: int = 8,
        memory_limit: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_kv_cache: bool = True,
        mixed_precision: bool = True,
        chunk_size: int = 4,
        memory_offload_freq: int = 20
    ):
        self.max_sequence_length = max_sequence_length
        self.context_window = context_window
        self.memory_limit = memory_limit
        self.temperature = temperature
        self.top_p = top_p
        self.use_kv_cache = use_kv_cache
        self.mixed_precision = mixed_precision
        self.chunk_size = chunk_size
        self.memory_offload_freq = memory_offload_freq


class StreamingRollout:
    """
    Streaming rollout engine for real-time world modeling inference.
    Implements efficient autoregressive generation with memory management.
    """
    
    def __init__(
        self,
        world_model: DiTStudent,
        vae_model: VAE3D,
        config: InferenceConfig,
        device: str = 'cuda'
    ):
        self.world_model = world_model.eval()
        self.vae_model = vae_model.eval()
        self.config = config
        self.device = device
        
        # Initialize memory bank
        self.memory_bank = MemoryBank(
            d_model=world_model.d_model,
            max_spatial_memory=config.memory_limit
        )
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # KV cache storage
        self.kv_cache = None
        self.cache_position = 0
        
        # Move models to device
        self.world_model.to(device)
        self.vae_model.to(device)
        
        # Optimize for inference
        if config.mixed_precision:
            self.world_model = self.world_model.half()
            self.vae_model = self.vae_model.half()
    
    def reset_state(self):
        """Reset all state for new sequence generation."""
        self.memory_bank.clear()
        self.kv_cache = None
        self.cache_position = 0
        self.frame_times.clear()
        self.memory_usage.clear()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def generate_sequence(
        self,
        context_frames: torch.Tensor,
        control_sequence: torch.Tensor,
        max_new_frames: int,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate video sequence autoregressively.
        
        Args:
            context_frames: Initial frames [B, T_ctx, 3, H, W]
            control_sequence: Control inputs [B, T_total, 4] (steer, accel, goal_x, goal_y)
            max_new_frames: Maximum number of new frames to generate
            return_intermediates: Whether to return intermediate states
        
        Returns:
            Dictionary with generated sequence and metadata
        """
        B, T_ctx, C, H, W = context_frames.shape
        device = context_frames.device
        
        # Reset state for new generation
        self.reset_state()
        
        # Initialize with context
        self._initialize_context(context_frames)
        
        # Generation loop
        generated_frames = []
        current_window = context_frames
        metadata = {
            'frame_times': [],
            'memory_usage': [],
            'cache_sizes': []
        }
        
        for t in range(max_new_frames):
            frame_start = time.time()
            
            # Get control for current step
            control_idx = min(t + T_ctx, control_sequence.size(1) - 1)
            current_control = control_sequence[:, control_idx:control_idx+1]  # [B, 1, 4]
            
            # Generate next frame
            next_frame = self._generate_next_frame(current_window, current_control)
            generated_frames.append(next_frame)
            
            # Update sliding window
            current_window = torch.cat([
                current_window[:, 1:],  # Remove oldest frame
                next_frame.unsqueeze(1)  # Add new frame
            ], dim=1)
            
            # Keep window size manageable
            if current_window.size(1) > self.config.context_window:
                current_window = current_window[:, -self.config.context_window:]
            
            # Update memory bank
            self.memory_bank.update(next_frame, T_ctx + t)
            
            # Performance tracking
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated(device) / 1e6
                self.memory_usage.append(memory_mb)
            
            metadata['frame_times'].append(frame_time)
            metadata['memory_usage'].append(memory_mb if torch.cuda.is_available() else 0)
            metadata['cache_sizes'].append(self._get_cache_size())
            
            # Memory management
            if t % self.config.memory_offload_freq == 0:
                self._manage_memory()
        
        # Combine results
        if generated_frames:
            generated_tensor = torch.cat(generated_frames, dim=1)  # [B, T_gen, 3, H, W]
            full_sequence = torch.cat([context_frames, generated_tensor], dim=1)
        else:
            generated_tensor = torch.empty(B, 0, C, H, W, device=device)
            full_sequence = context_frames
        
        result = {
            'context_frames': context_frames,
            'generated_frames': generated_tensor,
            'full_sequence': full_sequence,
            'metadata': metadata,
            'performance': {
                'avg_fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0.0,
                'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0.0,
                'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0.0
            }
        }
        
        return result
    
    def _initialize_context(self, context_frames: torch.Tensor):
        """Initialize model state with context frames."""
        B, T_ctx, C, H, W = context_frames.shape
        
        # Add context frames to memory bank
        for t in range(T_ctx):
            frame = context_frames[:, t]  # [B, 3, H, W]
            self.memory_bank.update(frame, t)
        
        # Optionally warm up KV cache with context
        if self.config.use_kv_cache and T_ctx > 1:
            self._warm_up_cache(context_frames)
    
    def _warm_up_cache(self, context_frames: torch.Tensor):
        """Warm up KV cache with context sequence."""
        B, T_ctx, C, H, W = context_frames.shape
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Encode context frames
            context_reshaped = context_frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            mean, logvar = self.vae_model.encode(context_reshaped)
            context_latents = self.vae_model.reparameterize(mean, logvar)
            context_latents = context_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            
            # Prepare tokens for world model
            for t in range(1, T_ctx):
                prev_latents = context_latents[:, :t]
                tokens = self._prepare_tokens(prev_latents, None)
                
                # Forward through world model to build cache
                self.world_model(tokens, use_cache=True)
    
    def _generate_next_frame(
        self,
        prev_frames: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate single next frame given previous frames and control.
        
        Args:
            prev_frames: Previous frames [B, T, 3, H, W]
            control: Control input [B, 1, 4]
        
        Returns:
            Next frame [B, 3, H, W]
        """
        B, T, C, H, W = prev_frames.shape
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Encode frames to latents
            frames_reshaped = prev_frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            mean, logvar = self.vae_model.encode(frames_reshaped)
            latents = self.vae_model.reparameterize(mean, logvar)
            latents = latents.permute(0, 2, 1, 3, 4)  # [B, T, C_lat, H_lat, W_lat]
            
            # Prepare tokens for world model
            tokens = self._prepare_tokens(latents, control)
            
            # Predict next latent
            next_latent_logits, self.kv_cache = self.world_model(
                tokens=tokens,
                kv_cache=self.kv_cache,
                use_cache=self.config.use_kv_cache
            )
            
            # Sample next latent (taking last prediction)
            next_latent = self._sample_latent(next_latent_logits[:, -1:])  # [B, 1, latent_dim]
            
            # Reshape for VAE decoding
            C_lat, H_lat, W_lat = latents.shape[2:]
            next_latent_reshaped = next_latent.view(B, 1, C_lat, H_lat, W_lat)
            next_latent_reshaped = next_latent_reshaped.permute(0, 2, 1, 3, 4)  # [B, C_lat, 1, H_lat, W_lat]
            
            # Decode to frame
            next_frame_decoded = self.vae_model.decode(next_latent_reshaped)
            next_frame = next_frame_decoded.permute(0, 2, 1, 3, 4).squeeze(1)  # [B, 3, H, W]
        
        return next_frame
    
    def _prepare_tokens(
        self,
        latents: torch.Tensor,
        control: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Prepare input tokens for world model.
        
        Args:
            latents: Latent representations [B, T, C, H, W]
            control: Control signals [B, 1, 4] or None
        
        Returns:
            Input tokens [B, seq_len, d_model]
        """
        B, T, C, H, W = latents.shape
        
        # Flatten latents to tokens
        latent_tokens = latents.view(B, T, -1)  # [B, T, C*H*W]
        
        # Add control tokens if provided
        if control is not None:
            # Expand control to match latent token dimension
            control_expanded = control.expand(-1, T, -1)  # [B, T, 4]
            
            # Simple concatenation (in practice, use learned embeddings)
            if latent_tokens.size(-1) >= 4:
                # Replace last 4 dims with control
                latent_tokens = torch.cat([
                    latent_tokens[:, :, :-4],
                    control_expanded
                ], dim=-1)
            else:
                # Concatenate if latent dim is small
                latent_tokens = torch.cat([latent_tokens, control_expanded], dim=-1)
        
        # Add memory context
        memory_tokens = self.memory_bank.get_memory_tokens(top_k=32)
        if memory_tokens.numel() > 0:
            # Broadcast memory across batch dimension
            memory_tokens = memory_tokens.expand(B, -1, -1)
            
            # Concatenate with latent tokens
            latent_tokens = torch.cat([memory_tokens, latent_tokens], dim=1)
        
        return latent_tokens
    
    def _sample_latent(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next latent from model predictions.
        
        Args:
            logits: Model output logits [B, 1, latent_dim]
        
        Returns:
            Sampled latent [B, 1, latent_dim]
        """
        if self.config.temperature == 1.0:
            return logits
        
        # Apply temperature
        scaled_logits = logits / self.config.temperature
        
        # Add noise for sampling
        noise = torch.randn_like(scaled_logits) * 0.1
        return scaled_logits + noise
    
    def _get_cache_size(self) -> int:
        """Get current KV cache size in number of elements."""
        if self.kv_cache is None:
            return 0
        
        total_size = 0
        for cache_layer in self.kv_cache:
            if cache_layer is not None:
                for key, tensor in cache_layer.items():
                    total_size += tensor.numel()
        
        return total_size
    
    def _manage_memory(self):
        """Manage GPU memory by pruning old cache entries."""
        if self.kv_cache is None:
            return
        
        # Prune old cache entries if cache is too large
        max_cache_size = 1000000  # Adjust based on GPU memory
        current_size = self._get_cache_size()
        
        if current_size > max_cache_size:
            # Prune oldest cache entries (simplified)
            for layer_cache in self.kv_cache:
                if layer_cache is not None:
                    for key in layer_cache:
                        # Keep only recent entries
                        max_seq_len = 128
                        if layer_cache[key].size(1) > max_seq_len:
                            layer_cache[key] = layer_cache[key][:, -max_seq_len:]
        
        # Clear unused GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RolloutEvaluator:
    """Evaluate rollout performance and quality."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def evaluate_sequence(
        self,
        generated_frames: torch.Tensor,
        ground_truth_frames: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate generated sequence against ground truth.
        
        Args:
            generated_frames: Generated frames [B, T, 3, H, W]
            ground_truth_frames: Ground truth frames [B, T, 3, H, W]
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # MSE loss
        mse_loss = F.mse_loss(generated_frames, ground_truth_frames)
        metrics['mse'] = mse_loss.item()
        
        # LPIPS (simplified with L1 gradient loss)
        pred_grad = self._compute_gradients(generated_frames)
        gt_grad = self._compute_gradients(ground_truth_frames)
        lpips_proxy = F.l1_loss(pred_grad, gt_grad)
        metrics['lpips_proxy'] = lpips_proxy.item()
        
        # Temporal consistency
        if generated_frames.size(1) > 1:
            pred_diff = generated_frames[:, 1:] - generated_frames[:, :-1]
            gt_diff = ground_truth_frames[:, 1:] - ground_truth_frames[:, :-1]
            temporal_loss = F.mse_loss(pred_diff, gt_diff)
            metrics['temporal_consistency'] = temporal_loss.item()
        
        return metrics
    
    def _compute_gradients(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradients as perceptual features."""
        grad_x = frames[:, :, :, :, 1:] - frames[:, :, :, :, :-1]
        grad_y = frames[:, :, :, 1:, :] - frames[:, :, :, :-1, :]
        
        # Pad to maintain size
        grad_x = F.pad(grad_x, (0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        return grad_x + grad_y


def test_streaming_rollout():
    """Test function for streaming rollout."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    world_model = DiTStudent(
        latent_dim=8,
        d_model=256,
        n_layers=4,
        n_heads=8,
        use_memory=True
    ).to(device)
    
    vae_model = VAE3D(
        in_channels=3,
        latent_dim=8,
        hidden_dims=[32, 64, 128]
    ).to(device)
    
    # Create config
    config = InferenceConfig(
        max_sequence_length=60,
        context_window=8,
        temperature=0.8,
        use_kv_cache=True,
        mixed_precision=False  # Disable for testing
    )
    
    # Create rollout engine
    rollout = StreamingRollout(world_model, vae_model, config, device)
    
    # Test data
    B, T_ctx, C, H, W = 1, 4, 3, 64, 64
    context_frames = torch.randn(B, T_ctx, C, H, W).to(device)
    control_sequence = torch.randn(B, 20, 4).to(device)
    
    print("Testing streaming rollout...")
    print(f"Context shape: {context_frames.shape}")
    print(f"Control shape: {control_sequence.shape}")
    
    # Generate sequence
    result = rollout.generate_sequence(
        context_frames=context_frames,
        control_sequence=control_sequence,
        max_new_frames=16
    )
    
    print(f"Generated frames shape: {result['generated_frames'].shape}")
    print(f"Full sequence shape: {result['full_sequence'].shape}")
    print(f"Performance: {result['performance']['avg_fps']:.2f} FPS")
    print(f"Peak memory: {result['performance']['peak_memory_mb']:.1f} MB")
    
    # Test evaluation
    evaluator = RolloutEvaluator()
    if result['generated_frames'].numel() > 0:
        # Create dummy ground truth
        ground_truth = torch.randn_like(result['generated_frames'])
        metrics = evaluator.evaluate_sequence(result['generated_frames'], ground_truth)
        print(f"Evaluation metrics: {metrics}")
    
    print("Streaming rollout test completed!")


if __name__ == "__main__":
    test_streaming_rollout()