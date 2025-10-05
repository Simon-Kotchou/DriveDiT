"""
Neural network helper functions and primitives.
SiLU, RMSNorm, fused operations, and other building blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Swish) activation function: x * sigmoid(x).
    
    Args:
        x: Input tensor
    
    Returns:
        SiLU activated tensor
    """
    return x * torch.sigmoid(x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation using tanh approximation (faster than erf).
    
    Args:
        x: Input tensor
    
    Returns:
        GELU activated tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable and efficient than LayerNorm for transformer models.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
        
        Returns:
            Normalized tensor with same shape
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.
    """
    
    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor [..., dim]
        
        Returns:
            Normalized tensor
        """
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization for DiT blocks.
    Modulates scale and shift based on conditioning.
    """
    
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm = RMSNorm(dim, eps)
        
        # Conditioning projection for scale and shift
        self.cond_proj = nn.Linear(cond_dim, 2 * dim, bias=True)
        
        # Initialize to identity transform
        with torch.no_grad():
            self.cond_proj.weight.zero_()
            self.cond_proj.bias.zero_()
            # Set bias to [1, 0, 1, 0, ...] for identity scale and zero shift
            self.cond_proj.bias[0::2] = 1.0
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor [B, T, dim]
            cond: Conditioning tensor [B, cond_dim] or [B, T, cond_dim]
        
        Returns:
            Modulated tensor [B, T, dim]
        """
        # Normalize input
        normed = self.norm(x)
        
        # Get conditioning parameters
        cond_params = self.cond_proj(cond)  # [B, 2*dim] or [B, T, 2*dim]
        
        # Split into scale and shift
        if cond_params.dim() == 2:  # [B, 2*dim]
            scale, shift = cond_params.chunk(2, dim=1)  # Each [B, dim]
            scale = scale.unsqueeze(1)  # [B, 1, dim]
            shift = shift.unsqueeze(1)  # [B, 1, dim]
        else:  # [B, T, 2*dim]
            scale, shift = cond_params.chunk(2, dim=2)  # Each [B, T, dim]
        
        # Apply modulation
        return normed * (1 + scale) + shift


class FusedMLP(nn.Module):
    """
    Fused MLP operations for efficiency.
    Combines multiple linear operations into single kernel calls.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        activation: str = 'silu',
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Single fused linear layer for gate and up projections
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        
        if activation == 'silu':
            self.activation = F.silu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.gate_up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02 / math.sqrt(2 * 12))
        
        if self.gate_up_proj.bias is not None:
            nn.init.zeros_(self.gate_up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused MLP forward pass.
        
        Args:
            x: Input tensor [B, T, d_model]
        
        Returns:
            Output tensor [B, T, d_model]
        """
        # Single linear operation for both gate and up projection
        gate_up = self.gate_up_proj(x)  # [B, T, 2*d_ff]
        
        # Split into gate and up components
        gate, up = gate_up.chunk(2, dim=-1)  # Each [B, T, d_ff]
        
        # Apply activation to gate and multiply with up
        hidden = self.activation(gate) * up
        
        # Down projection
        return self.down_proj(hidden)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention with optional flash attention optimization.
    """
    
    def __init__(self, dropout: float = 0.0, use_flash: bool = False):
        super().__init__()
        self.dropout = dropout
        self.use_flash = use_flash
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Scaled dot-product attention.
        
        Args:
            q: Query tensor [B, H, T, D]
            k: Key tensor [B, H, S, D]
            v: Value tensor [B, H, S, D]
            mask: Attention mask [B, H, T, S] or broadcastable
        
        Returns:
            Attention output [B, H, T, D]
        """
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's flash attention if available
            return F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(mask is None)  # Assume causal if no mask provided
            )
        else:
            # Manual implementation
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            
            if self.dropout > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            
            return torch.matmul(attn_weights, v)


def fused_add_norm(
    x: torch.Tensor, 
    residual: torch.Tensor, 
    norm: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual addition and normalization.
    
    Args:
        x: Input tensor
        residual: Residual tensor to add
        norm: Normalization module
    
    Returns:
        Tuple of (normalized_output, new_residual)
    """
    # Add residual
    residual = residual + x
    
    # Normalize
    normalized = norm(residual)
    
    return normalized, residual


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor, 
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key tensors.
    
    Args:
        q: Query tensor [B, H, T, D]
        k: Key tensor [B, H, T, D] 
        cos: Cosine tensor [T, D//2] or [B, T, D//2]
        sin: Sine tensor [T, D//2] or [B, T, D//2]
        position_ids: Position indices [B, T] (optional)
    
    Returns:
        Tuple of rotated (q, k) tensors
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    
    # Ensure cos/sin have correct dimensions
    if cos.dim() == 2:  # [T, D//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
    elif cos.dim() == 3:  # [B, T, D//2]
        cos = cos.unsqueeze(1)  # [B, 1, T, D//2]
        sin = sin.unsqueeze(1)  # [B, 1, T, D//2]
    
    # Apply rotary embedding
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]