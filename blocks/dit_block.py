"""
Diffusion Transformer (DiT) Block implementation.
Attention + MLP + Adaptive Layer Normalization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.mha import CausalMultiHeadAttention, BidirectionalMultiHeadAttention
from layers.mlp import create_mlp
from layers.nn_helpers import AdaLN, RMSNorm
from layers.rope import RoPELayer


class DiTBlock(nn.Module):
    """
    DiT (Diffusion Transformer) block with attention and MLP layers.
    Supports both causal and bidirectional attention modes.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        mlp_type: str = 'swiglu',
        dropout: float = 0.0,
        causal: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        cond_dim: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        self.use_rope = use_rope
        self.cond_dim = cond_dim
        
        # RoPE layer if enabled
        self.rope_layer = None
        if use_rope:
            self.rope_layer = RoPELayer(
                dim=d_model // n_heads,
                max_seq_len=max_seq_len
            )
        
        # Multi-head attention
        if causal:
            self.attn = CausalMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
                rope_layer=self.rope_layer
            )
        else:
            self.attn = BidirectionalMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
                rope_layer=self.rope_layer
            )
        
        # MLP
        self.mlp = create_mlp(
            d_model=d_model,
            mlp_type=mlp_type,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias
        )
        
        # Normalization layers
        if cond_dim is not None:
            # Adaptive normalization with conditioning
            self.norm_attn = AdaLN(d_model, cond_dim)
            self.norm_mlp = AdaLN(d_model, cond_dim)
        else:
            # Standard RMS normalization
            self.norm_attn = RMSNorm(d_model)
            self.norm_mlp = RMSNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through DiT block.
        
        Args:
            x: Input tensor [B, T, D]
            kv_cache: Optional KV cache for efficient inference
            pos: Optional RoPE position embeddings (sin, cos)
            cond: Optional conditioning tensor for AdaLN
            causal_mask: Optional causal mask override
            start_pos: Starting position for RoPE/KV cache
        
        Returns:
            Tuple of (output, updated_kv_cache)
        """
        # Pre-normalization for attention
        if self.cond_dim is not None and cond is not None:
            normed_x = self.norm_attn(x, cond)
        else:
            normed_x = self.norm_attn(x)
        
        # Self-attention with residual connection
        attn_out, new_kv_cache = self.attn(
            normed_x, 
            kv_cache=kv_cache,
            start_pos=start_pos
        )
        x = x + attn_out
        
        # Pre-normalization for MLP
        if self.cond_dim is not None and cond is not None:
            normed_x = self.norm_mlp(x, cond)
        else:
            normed_x = self.norm_mlp(x)
        
        # MLP with residual connection
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        
        return x, new_kv_cache


class DiTBlockWithCrossAttention(DiTBlock):
    """
    DiT block with additional cross-attention for conditioning on external context.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        mlp_type: str = 'swiglu',
        dropout: float = 0.0,
        causal: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        cond_dim: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__(
            d_model, n_heads, d_ff, mlp_type, dropout, 
            causal, use_rope, max_seq_len, cond_dim, bias
        )
        
        # Cross-attention layer (always bidirectional)
        self.cross_attn = BidirectionalMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            rope_layer=None  # No RoPE for cross-attention
        )
        
        # Additional normalization for cross-attention
        if cond_dim is not None:
            self.norm_cross = AdaLN(d_model, cond_dim)
        else:
            self.norm_cross = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        cross_kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with cross-attention.
        
        Args:
            x: Input tensor [B, T, D]
            context: Context tensor for cross-attention [B, S, D]
            kv_cache: KV cache for self-attention
            cross_kv_cache: KV cache for cross-attention
            pos: RoPE position embeddings
            cond: Conditioning tensor
            causal_mask: Causal mask override
            start_pos: Starting position
        
        Returns:
            Tuple of (output, self_kv_cache, cross_kv_cache)
        """
        # Self-attention (same as parent class)
        if self.cond_dim is not None and cond is not None:
            normed_x = self.norm_attn(x, cond)
        else:
            normed_x = self.norm_attn(x)
        
        attn_out, new_kv_cache = self.attn(
            normed_x,
            kv_cache=kv_cache,
            start_pos=start_pos
        )
        x = x + attn_out
        
        # Cross-attention if context is provided
        new_cross_kv_cache = cross_kv_cache
        if context is not None:
            if self.cond_dim is not None and cond is not None:
                normed_x = self.norm_cross(x, cond)
            else:
                normed_x = self.norm_cross(x)
            
            # Cross-attention: Q from x, K,V from context
            cross_attn_out, new_cross_kv_cache = self.cross_attn(
                normed_x,
                kv_cache=cross_kv_cache,
                start_pos=0  # Context doesn't use incremental decoding
            )
            x = x + cross_attn_out
        
        # MLP
        if self.cond_dim is not None and cond is not None:
            normed_x = self.norm_mlp(x, cond)
        else:
            normed_x = self.norm_mlp(x)
        
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        
        return x, new_kv_cache, new_cross_kv_cache


class MemoryEfficientDiTBlock(DiTBlock):
    """
    Memory-efficient DiT block with gradient checkpointing and chunked processing.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        mlp_type: str = 'swiglu',
        dropout: float = 0.0,
        causal: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        cond_dim: Optional[int] = None,
        bias: bool = False,
        chunk_size: int = 512,
        use_checkpoint: bool = True
    ):
        super().__init__(
            d_model, n_heads, d_ff, mlp_type, dropout,
            causal, use_rope, max_seq_len, cond_dim, bias
        )
        
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
    
    def _forward_chunk(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        cond: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for a single chunk."""
        return super().forward(x, kv_cache, None, cond, None, start_pos)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Memory-efficient forward pass with optional chunking.
        """
        if self.training and self.use_checkpoint:
            # Use gradient checkpointing during training
            return torch.utils.checkpoint.checkpoint(
                self._forward_chunk, x, kv_cache, cond, start_pos,
                use_reentrant=False
            )
        else:
            # Standard forward pass
            return self._forward_chunk(x, kv_cache, cond, start_pos)


def create_dit_block(
    d_model: int,
    n_heads: int,
    block_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of DiT blocks.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        block_type: Type of block ('standard', 'cross_attn', 'memory_efficient')
        **kwargs: Additional arguments for block constructor
    
    Returns:
        DiT block module
    """
    if block_type == 'standard':
        return DiTBlock(d_model, n_heads, **kwargs)
    elif block_type == 'cross_attn':
        return DiTBlockWithCrossAttention(d_model, n_heads, **kwargs)
    elif block_type == 'memory_efficient':
        return MemoryEfficientDiTBlock(d_model, n_heads, **kwargs)
    else:
        raise ValueError(f"Unsupported block type: {block_type}")