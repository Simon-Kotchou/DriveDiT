"""
Core components with transformers and SAM2 integrations.
Zero-dependency implementations inspired by official architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
from einops import rearrange, repeat

from ..config.modular_config import DriveDiTConfig


class RoPE3D(nn.Module):
    """3D Rotary Position Embedding for video transformers."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, temperature: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        
        # Pre-compute frequency matrix
        inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 3D position embeddings."""
        # Time positions
        t_pos = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        # Height positions  
        h_pos = torch.arange(height, device=self.inv_freq.device, dtype=torch.float)
        # Width positions
        w_pos = torch.arange(width, device=self.inv_freq.device, dtype=torch.float)
        
        # Factorized 3D embeddings (time, height, width)
        t_emb = torch.outer(t_pos, self.inv_freq)
        h_emb = torch.outer(h_pos, self.inv_freq) 
        w_emb = torch.outer(w_pos, self.inv_freq)
        
        # Combine embeddings
        t_sin, t_cos = t_emb.sin(), t_emb.cos()
        h_sin, h_cos = h_emb.sin(), h_emb.cos()
        w_sin, w_cos = w_emb.sin(), w_emb.cos()
        
        # Create full 3D position matrix [seq_len, height, width, dim]
        sin_emb = torch.zeros(seq_len, height, width, self.dim, device=self.inv_freq.device)
        cos_emb = torch.zeros(seq_len, height, width, self.dim, device=self.inv_freq.device)
        
        # Distribute dimensions across t, h, w
        dim_third = self.dim // 3
        
        # Time embeddings
        sin_emb[:, :, :, :dim_third] = t_sin[:, None, None, :]
        cos_emb[:, :, :, :dim_third] = t_cos[:, None, None, :]
        
        # Height embeddings  
        sin_emb[:, :, :, dim_third:2*dim_third] = h_sin[None, :, None, :]
        cos_emb[:, :, :, dim_third:2*dim_third] = h_cos[None, :, None, :]
        
        # Width embeddings
        sin_emb[:, :, :, 2*dim_third:] = w_sin[None, None, :, :]
        cos_emb[:, :, :, 2*dim_third:] = w_cos[None, None, :, :]
        
        return sin_emb, cos_emb
    
    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        # q, k: [batch, seq_len, num_heads, head_dim]
        # sin, cos: [seq_len, height, width, head_dim]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        
        # Apply rotation
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        
        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking and RoPE."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        causal: bool = False,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE3D(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        kv: Optional[torch.Tensor] = None,
        past_kv: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: [B, T, D] input tensor
            kv: [B, T_kv, D] optional separate key-value tensor for cross-attention
            past_kv: cached key-value pairs
            use_cache: whether to return cache for next iteration
            attn_mask: [T, T] attention mask
        """
        B, T, D = x.shape
        
        # Generate Q, K, V
        if kv is not None:
            # Cross-attention
            q = F.linear(x, self.qkv.weight[:D], self.qkv.bias[:D] if self.qkv.bias is not None else None)
            kv_proj = F.linear(kv, self.qkv.weight[D:], self.qkv.bias[D:] if self.qkv.bias is not None else None)
            k, v = kv_proj.chunk(2, dim=-1)
        else:
            # Self-attention
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if past_kv is not None:
            past_k, past_v = past_kv['k'], past_kv['v']
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Apply RoPE if enabled
        if self.use_rope and kv is None:  # Only for self-attention
            seq_len, height, width = T, int(T**0.5), int(T**0.5)  # Assume square spatial
            sin_emb, cos_emb = self.rope(seq_len, height, width)
            # Flatten spatial dimensions for attention
            sin_flat = sin_emb.view(seq_len * height * width, -1)[:T]
            cos_flat = cos_emb.view(seq_len * height * width, -1)[:T]
            q, k = self.rope.apply_rope(q, k, sin_flat, cos_flat)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if self.causal:
            # Create causal mask
            T_q, T_k = q.size(2), k.size(2)
            causal_mask = torch.triu(torch.ones(T_q, T_k, device=q.device), diagonal=T_k-T_q+1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Softmax and apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.proj(out)
        
        # Prepare cache
        cache = None
        if use_cache:
            cache = {'k': k.detach(), 'v': v.detach()}
        
        return out, cache


class TransformerBlock(nn.Module):
    """Transformer block with optional cross-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        causal: bool = False,
        cross_attention: bool = False,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()
        self.cross_attention = cross_attention
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            causal=causal,
            dropout=dropout,
            use_rope=use_rope
        )
        
        # Cross-attention (optional)
        if cross_attention:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = MultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                causal=False,  # Cross-attention is not causal
                dropout=dropout,
                use_rope=False  # Don't use RoPE for cross-attention
            )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        past_kv: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with optional cross-attention and caching."""
        
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, cache = self.self_attn(
            norm_x, 
            past_kv=past_kv, 
            use_cache=use_cache, 
            attn_mask=attn_mask
        )
        x = x + attn_out
        
        # Cross-attention (if enabled and context provided)
        if self.cross_attention and context is not None:
            norm_x_cross = self.norm_cross(x)
            cross_out, _ = self.cross_attn(norm_x_cross, kv=context)
            x = x + cross_out
        
        # MLP
        norm_x_mlp = self.norm2(x)
        mlp_out = self.mlp(norm_x_mlp)
        x = x + mlp_out
        
        return x, cache


class CausalVideoTransformer(nn.Module):
    """Causal video transformer for world modeling."""
    
    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        max_seq_len: int = 512,
        cross_attention_layers: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Determine which layers have cross-attention
        if cross_attention_layers is None:
            cross_attention_layers = list(range(num_layers // 2, num_layers))  # Second half
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                causal=True,
                cross_attention=(i in cross_attention_layers),
                dropout=dropout,
                use_rope=True
            )
            for i in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass through causal transformer.
        
        Args:
            x: [B, T, D] input sequence
            context: [B, T_ctx, D] optional context for cross-attention
            past_kvs: cached key-value pairs from previous forward passes
            use_cache: whether to return KV cache for next iteration
        """
        if past_kvs is None:
            past_kvs = [None] * self.num_layers
        
        new_kvs = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(
                x,
                context=context,
                past_kv=past_kvs[i],
                use_cache=use_cache
            )
            
            if use_cache:
                new_kvs.append(layer_cache)
        
        x = self.norm(x)
        return x, new_kvs


class SAM2InspiredTracker(nn.Module):
    """SAM2-inspired object tracking for memory management."""
    
    def __init__(self, dim: int = 256, max_objects: int = 32):
        super().__init__()
        self.dim = dim
        self.max_objects = max_objects
        
        # Object encoder (simplified SAM2-style)
        self.object_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 64, dim)
        )
        
        # Object memory
        self.object_memory = nn.Parameter(torch.randn(max_objects, dim))
        self.memory_attention = MultiHeadAttention(dim, num_heads=8, causal=False, use_rope=False)
        
        # Object state tracking
        self.object_states = {}  # Track object persistence
        
    def encode_objects(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract object features from frames."""
        B, T, C, H, W = frames.shape
        
        # Process frames
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        object_features = self.object_encoder(frames_flat)
        object_features = rearrange(object_features, '(b t) d -> b t d', b=B, t=T)
        
        return object_features
    
    def update_memory(self, object_features: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Update object memory with new observations."""
        B, T, D = object_features.shape
        
        # Expand memory for batch
        memory = self.object_memory.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention to update memory
        updated_memory, _ = self.memory_attention(
            memory, 
            kv=object_features.view(B, -1, D)
        )
        
        return updated_memory
    
    def get_memory_tokens(self, batch_size: int) -> torch.Tensor:
        """Get current memory tokens for attention."""
        return self.object_memory.unsqueeze(0).expand(batch_size, -1, -1)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization for conditioning."""
    
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Linear(condition_dim, dim)
        self.shift = nn.Linear(condition_dim, dim)
        
        # Initialize to identity
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization."""
        x_norm = self.norm(x)
        scale = self.scale(condition)
        shift = self.shift(condition)
        
        # Handle broadcasting for different shapes
        if x_norm.dim() == 3 and scale.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return x_norm * scale + shift


class FlowMatchingPredictor(nn.Module):
    """Flow matching predictor for diffusion distillation."""
    
    def __init__(self, dim: int, num_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Flow prediction network
        self.flow_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),  # Input + time embedding
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.Tanh()  # Bound flow predictions
        )
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict flow at time t."""
        # Embed time
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # Concatenate input and time
        z_t = torch.cat([z, t_embed], dim=-1)
        
        # Predict flow
        flow = self.flow_net(z_t)
        return flow
    
    def sample(self, z_init: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Sample using flow matching (Euler method)."""
        if num_steps is None:
            num_steps = self.num_steps
        
        z = z_init
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((z.size(0),), i * dt, device=z.device)
            flow = self.forward(z, t)
            z = z + dt * flow
        
        return z


def create_attention_mask(seq_len: int, causal: bool = True) -> torch.Tensor:
    """Create attention mask for transformer."""
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.float().masked_fill(mask, float('-inf'))
    else:
        return torch.zeros(seq_len, seq_len)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test RoPE
    rope = RoPE3D(64)
    sin, cos = rope(16, 8, 8)
    print(f"RoPE embeddings shape: {sin.shape}")
    
    # Test attention
    mha = MultiHeadAttention(512, num_heads=8, causal=True, use_rope=True)
    x = torch.randn(2, 16, 512)
    out, cache = mha(x, use_cache=True)
    print(f"Attention output shape: {out.shape}")
    
    # Test transformer
    transformer = CausalVideoTransformer(
        dim=512,
        num_layers=6,
        num_heads=8,
        cross_attention_layers=[3, 4, 5]
    )
    
    x = torch.randn(2, 16, 512)
    context = torch.randn(2, 8, 512)
    out, kvs = transformer(x, context=context, use_cache=True)
    print(f"Transformer output shape: {out.shape}")
    print(f"Number of parameters: {count_parameters(transformer):,}")
    
    # Test SAM2 tracker
    tracker = SAM2InspiredTracker(dim=256, max_objects=16)
    frames = torch.randn(2, 4, 3, 64, 64)
    obj_features = tracker.encode_objects(frames)
    memory = tracker.update_memory(obj_features, 0)
    print(f"Object memory shape: {memory.shape}")
    
    print("All components tested successfully!")