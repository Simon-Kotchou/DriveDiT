"""
Multi-Head Attention implementation using einsum operations.
Pure mathematical components with explicit tensor operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


def mha(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> torch.Tensor:
    """
    Multi-head attention using einsum operations.
    
    Args:
        q: Query tensor [B, T, H, D]
        k: Key tensor [B, S, H, D] 
        v: Value tensor [B, S, H, D]
        mask: Attention mask [B, H, T, S] or broadcastable. 1 for valid, 0 for masked
        dropout_p: Dropout probability
        training: Training mode flag
    
    Returns:
        Attention output [B, T, H, D]
    """
    d = q.size(-1)
    
    # Compute attention scores: Q @ K^T
    scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(d)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
    
    # Softmax in float32 for numerical stability
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    
    # Apply dropout
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    
    # Apply attention to values: P @ V
    output = torch.einsum('bhts,bshd->bthd', attn_weights, v)
    
    return output


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to place mask on
    
    Returns:
        Causal mask [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def create_block_causal_mask(
    block_size: int, 
    num_blocks: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Create block-wise causal mask for efficient attention.
    
    Args:
        block_size: Size of each attention block
        num_blocks: Number of blocks
        device: Device to place mask on
    
    Returns:
        Block causal mask
    """
    seq_len = block_size * num_blocks
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(num_blocks):
        start_i = i * block_size
        end_i = (i + 1) * block_size
        
        for j in range(i + 1):  # Can attend to current and previous blocks
            start_j = j * block_size
            end_j = (j + 1) * block_size
            
            if i == j:  # Current block - use causal mask
                mask[start_i:end_i, start_j:end_j] = torch.tril(torch.ones(block_size, block_size))
            else:  # Previous blocks - full attention
                mask[start_i:end_i, start_j:end_j] = 1.0
    
    return mask.unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with RoPE support and KV caching.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.rope_layer = rope_layer
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * 12))  # Scaled for depth
        
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [B, T, D]
            kv_cache: Dict with 'k' and 'v' cached tensors
            mask: Attention mask
            start_pos: Starting position for RoPE
        
        Returns:
            Tuple of (output, new_kv_cache)
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, T, D]
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head)  # [B, T, H, D_head]
        k = k.view(B, T, self.n_heads, self.d_head)  # [B, T, H, D_head]
        v = v.view(B, T, self.n_heads, self.d_head)  # [B, T, H, D_head]
        
        # Apply RoPE if available
        if self.rope_layer is not None:
            q = self.rope_layer(q, start_pos)
            k = self.rope_layer(k, start_pos)
        
        # Handle KV caching
        new_kv_cache = None
        if kv_cache is not None:
            # Concatenate with cached K, V
            k = torch.cat([kv_cache['k'], k], dim=1)
            v = torch.cat([kv_cache['v'], v], dim=1)
            
            # Update cache with new K, V (detached to prevent gradient accumulation)
            new_kv_cache = {
                'k': k.detach(),
                'v': v.detach()
            }
        
        # Apply attention
        output = mha(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training)
        
        # Reshape and project
        output = output.view(B, T, D)  # [B, T, D]
        output = self.proj(output)
        
        return output, new_kv_cache


class CausalMultiHeadAttention(MultiHeadAttention):
    """
    Causal Multi-Head Attention with automatic causal masking.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with automatic causal masking.
        
        Args:
            x: Input tensor [B, T, D]
            kv_cache: Dict with 'k' and 'v' cached tensors
            start_pos: Starting position for RoPE
        
        Returns:
            Tuple of (output, new_kv_cache)
        """
        T = x.size(1)
        
        # Create causal mask if not using KV cache (for training)
        mask = None
        if kv_cache is None and T > 1:
            mask = create_causal_mask(T, x.device)
        
        return super().forward(x, kv_cache, mask, start_pos)


class BidirectionalMultiHeadAttention(MultiHeadAttention):
    """
    Bidirectional Multi-Head Attention (no causal masking).
    Used for teacher models.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass without causal masking.
        """
        return super().forward(x, kv_cache, None, start_pos)