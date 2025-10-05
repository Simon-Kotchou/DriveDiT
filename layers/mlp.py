"""
Multi-Layer Perceptron (MLP) implementations.
Feed-forward networks with various activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MLP(nn.Module):
    """
    Standard MLP with configurable activation function.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: str = 'silu',
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  # Standard transformer ratio
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Linear layers
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        
        # Activation function
        if activation == 'silu':
            self.activation = F.silu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = F.silu  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02 / math.sqrt(2 * 12))  # Scaled for depth
        
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            Output tensor [B, T, D]
        """
        # First linear layer + activation
        hidden = self.activation(self.w1(x))
        
        # Apply dropout
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        
        # Second linear layer
        output = self.w2(hidden)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SiLU(x @ W1) * (x @ W2).
    More parameter efficient than standard MLP with better performance.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        
        if d_ff is None:
            # SwiGLU typically uses 2/3 of standard FFN size for same parameter count
            d_ff = int(8 * d_model / 3)
            # Round to nearest multiple of 8 for efficiency
            d_ff = ((d_ff + 7) // 8) * 8
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Two parallel linear layers for gating
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # Value
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)  # Output projection
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02 / math.sqrt(2 * 12))
        
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
        if self.w3.bias is not None:
            nn.init.zeros_(self.w3.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            Output tensor [B, T, D]
        """
        # Compute gate and value in parallel
        gate = F.silu(self.w1(x))  # [B, T, d_ff]
        value = self.w2(x)         # [B, T, d_ff]
        
        # Element-wise multiplication (gating)
        hidden = gate * value
        
        # Apply dropout
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        
        # Output projection
        output = self.w3(hidden)
        
        return output


class GeGLU(nn.Module):
    """
    GeGLU activation function: GELU(x @ W1) * (x @ W2).
    Alternative to SwiGLU using GELU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 7) // 8) * 8
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02 / math.sqrt(2 * 12))
        
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
        if self.w3.bias is not None:
            nn.init.zeros_(self.w3.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GeGLU.
        """
        gate = F.gelu(self.w1(x))
        value = self.w2(x)
        hidden = gate * value
        
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        
        return self.w3(hidden)


class ConditionalMLP(nn.Module):
    """
    MLP with conditional input (e.g., for DiT blocks with conditioning).
    """
    
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        d_ff: Optional[int] = None,
        activation: str = 'silu',
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_cond = d_cond
        self.d_ff = d_ff
        
        # Main MLP
        self.mlp = MLP(d_model, d_ff, activation, dropout, bias)
        
        # Conditioning projection
        self.cond_proj = nn.Linear(d_cond, d_model, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize conditioning projection."""
        nn.init.normal_(self.cond_proj.weight, std=0.02)
        if self.cond_proj.bias is not None:
            nn.init.zeros_(self.cond_proj.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning.
        
        Args:
            x: Input tensor [B, T, D]
            cond: Conditioning tensor [B, T, D_cond] or [B, D_cond]
        
        Returns:
            Output tensor [B, T, D]
        """
        # Project conditioning
        cond_proj = self.cond_proj(cond)
        
        # Add conditioning to input
        if cond_proj.dim() == 2:  # [B, D_cond] -> [B, 1, D]
            cond_proj = cond_proj.unsqueeze(1)
        
        conditioned_input = x + cond_proj
        
        # Apply MLP
        return self.mlp(conditioned_input)


def create_mlp(
    d_model: int,
    mlp_type: str = 'standard',
    d_ff: Optional[int] = None,
    activation: str = 'silu',
    dropout: float = 0.0,
    bias: bool = False
) -> nn.Module:
    """
    Factory function to create different types of MLPs.
    
    Args:
        d_model: Model dimension
        mlp_type: Type of MLP ('standard', 'swiglu', 'geglu')
        d_ff: Feed-forward dimension
        activation: Activation function (for standard MLP)
        dropout: Dropout probability
        bias: Whether to use bias
    
    Returns:
        MLP module
    """
    if mlp_type == 'standard':
        return MLP(d_model, d_ff, activation, dropout, bias)
    elif mlp_type == 'swiglu':
        return SwiGLU(d_model, d_ff, dropout, bias)
    elif mlp_type == 'geglu':
        return GeGLU(d_model, d_ff, dropout, bias)
    else:
        raise ValueError(f"Unsupported MLP type: {mlp_type}")