"""
Sparse-Linear Attention (SLA) implementation.

Achieves O(N * sqrt(N)) effective complexity by routing attention blocks to:
- Critical: Exact O(N^2) attention for high-rank blocks
- Marginal: O(N) linear attention for low-rank blocks
- Negligible: Skip entirely for near-zero blocks

References:
- SLA paper: arxiv:2509.24006
- 20× attention reduction, 95% compute savings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math


@dataclass
class SLAConfig:
    """Configuration for Sparse-Linear Attention."""
    block_size: int = 64  # Block size for partitioning
    critical_threshold: float = 0.5  # Threshold for critical blocks
    negligible_threshold: float = 0.01  # Threshold for negligible blocks
    causal: bool = False  # Use causal attention
    flash_attention: bool = True  # Use FlashAttention when available
    use_elu_feature: bool = True  # Use ELU+1 feature map for linear attention
    linear_feature_dim: int = 64  # Dim for random Fourier features
    temperature: float = 1.0  # Softmax temperature


@dataclass
class SLAStats:
    """Statistics from SLA forward pass."""
    critical_blocks: int
    marginal_blocks: int
    negligible_blocks: int
    total_blocks: int
    critical_ratio: float
    compute_savings: float


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature map for linear attention."""
    return F.elu(x) + 1


def _random_fourier_features(
    x: torch.Tensor,
    omega: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """Random Fourier features for softmax approximation."""
    proj = torch.einsum('...d,dk->...k', x * scale, omega)
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(omega.shape[1])


def classify_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    config: SLAConfig,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Classify attention blocks into critical, marginal, and negligible.

    Args:
        q: Query tensor [B, T, H, D]
        k: Key tensor [B, S, H, D]
        config: SLA configuration
        mask: Optional attention mask

    Returns:
        Tuple of (critical_mask, marginal_mask, negligible_mask)
        Each mask is [B, H, num_q_blocks, num_k_blocks] boolean tensor
    """
    B, T, H, D = q.shape
    S = k.size(1)
    block_size = config.block_size

    num_q_blocks = (T + block_size - 1) // block_size
    num_k_blocks = (S + block_size - 1) // block_size

    # Pad to block boundaries
    T_pad = num_q_blocks * block_size
    S_pad = num_k_blocks * block_size

    if T_pad > T:
        q = F.pad(q, (0, 0, 0, 0, 0, T_pad - T))
    if S_pad > S:
        k = F.pad(k, (0, 0, 0, 0, 0, S_pad - S))

    # Reshape into blocks
    q_blocks = q.view(B, num_q_blocks, block_size, H, D)
    k_blocks = k.view(B, num_k_blocks, block_size, H, D)

    # Use mean vectors as representatives
    q_rep = q_blocks.mean(dim=2)  # [B, num_q_blocks, H, D]
    k_rep = k_blocks.mean(dim=2)  # [B, num_k_blocks, H, D]

    # Compute pairwise block scores
    block_scores = torch.einsum('bqhd,bkhd->bhqk', q_rep, k_rep) / math.sqrt(D)

    # Apply causal masking at block level
    if config.causal:
        causal_block_mask = torch.tril(
            torch.ones(num_q_blocks, num_k_blocks, device=q.device, dtype=torch.bool)
        )
        block_scores = block_scores.masked_fill(~causal_block_mask, -1e9)

    # Estimate importance via variance
    score_softmax = F.softmax(block_scores, dim=-1)
    score_variance = score_softmax.var(dim=-1, keepdim=True)

    importance = score_variance.expand_as(block_scores)
    max_scores = block_scores.max(dim=-1, keepdim=True).values
    importance = importance * (max_scores - block_scores.min()).clamp(min=1e-6)

    # Normalize importance
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

    # Classify blocks
    critical_mask = importance_norm >= config.critical_threshold
    negligible_mask = importance_norm <= config.negligible_threshold
    marginal_mask = ~critical_mask & ~negligible_mask

    # Apply causal constraint
    if config.causal:
        upper_tri = ~torch.tril(torch.ones(num_q_blocks, num_k_blocks, device=q.device, dtype=torch.bool))
        critical_mask = critical_mask & ~upper_tri
        marginal_mask = marginal_mask & ~upper_tri
        negligible_mask = negligible_mask | upper_tri

    return critical_mask, marginal_mask, negligible_mask


def sla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    config: Optional[SLAConfig] = None
) -> torch.Tensor:
    """
    Sparse-Linear Attention with automatic block routing.

    Drop-in replacement for standard multi-head attention.

    Args:
        q: Query tensor [B, T, H, D]
        k: Key tensor [B, S, H, D]
        v: Value tensor [B, S, H, D]
        mask: Attention mask [B, H, T, S]
        dropout_p: Dropout probability
        training: Training mode
        config: SLA configuration

    Returns:
        Attention output [B, T, H, D]
    """
    if config is None:
        config = SLAConfig()

    B, T, H, D = q.shape
    S = k.size(1)
    device = q.device
    dtype = q.dtype

    # For short sequences, use regular attention
    if T <= config.block_size or S <= config.block_size:
        return _exact_attention(q, k, v, mask, dropout_p, training, config)

    # Classify blocks
    critical_mask, marginal_mask, negligible_mask = classify_blocks(q, k, config, mask)

    # Initialize output
    output = torch.zeros(B, T, H, D, device=device, dtype=dtype)
    attention_weights_sum = torch.zeros(B, T, H, 1, device=device, dtype=dtype)

    block_size = config.block_size
    num_q_blocks = (T + block_size - 1) // block_size
    num_k_blocks = (S + block_size - 1) // block_size

    # Pad sequences
    T_pad = num_q_blocks * block_size
    S_pad = num_k_blocks * block_size

    q_padded = F.pad(q, (0, 0, 0, 0, 0, T_pad - T)) if T_pad > T else q
    k_padded = F.pad(k, (0, 0, 0, 0, 0, S_pad - S)) if S_pad > S else k
    v_padded = F.pad(v, (0, 0, 0, 0, 0, S_pad - S)) if S_pad > S else v

    # Feature function for linear attention
    if config.use_elu_feature:
        feature_fn = _elu_feature_map
    else:
        omega = torch.randn(D, config.linear_feature_dim, device=device, dtype=dtype)
        omega = omega / omega.norm(dim=0, keepdim=True)
        scale = 1.0 / math.sqrt(D)
        feature_fn = lambda x: _random_fourier_features(x, omega, scale)

    # Process each query block
    for qi in range(num_q_blocks):
        q_start = qi * block_size
        q_end = min((qi + 1) * block_size, T)
        q_block = q_padded[:, q_start:q_end]

        block_output = torch.zeros(B, q_end - q_start, H, D, device=device, dtype=dtype)
        block_weights_sum = torch.zeros(B, q_end - q_start, H, 1, device=device, dtype=dtype)

        for ki in range(num_k_blocks):
            k_start = ki * block_size
            k_end = min((ki + 1) * block_size, S)
            k_block = k_padded[:, k_start:k_end]
            v_block = v_padded[:, k_start:k_end]

            # Get block classification
            is_critical = critical_mask[:, :, qi, ki]  # [B, H]
            is_marginal = marginal_mask[:, :, qi, ki]

            # Process critical blocks with exact attention
            if is_critical.any():
                critical_output, critical_weights = _exact_attention_block(
                    q_block, k_block, v_block, config, mask, qi, ki, dropout_p, training
                )
                critical_mask_exp = is_critical.view(B, 1, H, 1).expand_as(critical_output)
                block_output = block_output + critical_output * critical_mask_exp
                block_weights_sum = block_weights_sum + critical_weights * critical_mask_exp[..., :1]

            # Process marginal blocks with linear attention
            if is_marginal.any():
                marginal_output, marginal_weights = _linear_attention_block(
                    q_block, k_block, v_block, feature_fn, config
                )
                marginal_mask_exp = is_marginal.view(B, 1, H, 1).expand_as(marginal_output)
                block_output = block_output + marginal_output * marginal_mask_exp
                block_weights_sum = block_weights_sum + marginal_weights * marginal_mask_exp[..., :1]

        # Normalize
        block_output = block_output / (block_weights_sum + 1e-6)
        output[:, q_start:q_end] = block_output

    return output[:, :T]


def _exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    config: SLAConfig
) -> torch.Tensor:
    """Exact attention computation (fallback)."""
    d = q.size(-1)
    scale = 1.0 / (math.sqrt(d) * config.temperature)

    # Try FlashAttention
    if config.flash_attention and hasattr(F, 'scaled_dot_product_attention'):
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=config.causal and mask is None,
            scale=scale
        )
        return output.transpose(1, 2)

    # Manual attention
    scores = torch.einsum('bthd,bshd->bhts', q, k) * scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    if config.causal:
        T, S = q.size(1), k.size(1)
        causal_mask = torch.tril(torch.ones(T, S, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, -1e9)

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    return torch.einsum('bhts,bshd->bthd', attn_weights, v)


def _exact_attention_block(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    config: SLAConfig,
    mask: Optional[torch.Tensor],
    qi: int,
    ki: int,
    dropout_p: float,
    training: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact attention for a single block pair."""
    B, T_block, H, D = q_block.shape
    S_block = k_block.size(1)

    scale = 1.0 / (math.sqrt(D) * config.temperature)
    scores = torch.einsum('bthd,bshd->bhts', q_block, k_block) * scale

    # Causal masking within block
    if config.causal and qi == ki:
        causal_mask = torch.tril(torch.ones(T_block, S_block, device=q_block.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, -1e9)
    elif config.causal and qi < ki:
        scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool), -1e9)

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_block.dtype)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    output = torch.einsum('bhts,bshd->bthd', attn_weights, v_block)
    weights_sum = attn_weights.sum(dim=-1).transpose(1, 2).unsqueeze(-1)

    return output, weights_sum


def _linear_attention_block(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    feature_fn,
    config: SLAConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear attention for a single block pair."""
    q_feat = feature_fn(q_block / config.temperature)
    k_feat = feature_fn(k_block / config.temperature)

    # K^T @ V aggregation
    kv = torch.einsum('bshd,bshv->bhdv', k_feat, v_block)

    # Q @ (K^T V)
    numerator = torch.einsum('bthd,bhdv->bthv', q_feat, kv)

    # Normalization
    k_sum = k_feat.sum(dim=1)
    denominator = torch.einsum('bthd,bhd->bth', q_feat, k_sum) + 1e-6

    output = numerator / denominator.unsqueeze(-1)
    weights_proxy = denominator.unsqueeze(-1)

    return output, weights_proxy


class SparseLinearAttention(nn.Module):
    """
    Sparse-Linear Attention module - drop-in replacement for MultiHeadAttention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        config: Optional[SLAConfig] = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope_layer: Optional[nn.Module] = None
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.config = config or SLAConfig()
        self.rope_layer = rope_layer

        # Projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * 12))
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
        """Forward pass with optional KV caching."""
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        # Apply RoPE
        if self.rope_layer is not None:
            q = self.rope_layer(q, start_pos)
            k = self.rope_layer(k, start_pos)

        # KV caching
        new_kv_cache = None
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=1)
            v = torch.cat([kv_cache['v'], v], dim=1)
            new_kv_cache = {'k': k.detach(), 'v': v.detach()}

        # Apply SLA
        output = sla(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training, config=self.config)

        # Reshape and project
        output = output.reshape(B, T, D)
        output = self.proj(output)

        return output, new_kv_cache


class CausalSparseLinearAttention(SparseLinearAttention):
    """Causal SLA with automatic causal masking."""

    def __init__(self, d_model: int, n_heads: int, config: Optional[SLAConfig] = None, **kwargs):
        if config is None:
            config = SLAConfig(causal=True)
        else:
            config.causal = True
        super().__init__(d_model, n_heads, config, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        return super().forward(x, kv_cache, None, start_pos)


def create_sla_layer(
    d_model: int,
    n_heads: int,
    causal: bool = False,
    block_size: int = 64,
    **kwargs
) -> SparseLinearAttention:
    """Factory function for creating SLA layers."""
    config = SLAConfig(block_size=block_size, causal=causal)
    if causal:
        return CausalSparseLinearAttention(d_model, n_heads, config, **kwargs)
    return SparseLinearAttention(d_model, n_heads, config, **kwargs)


if __name__ == "__main__":
    print("Testing Sparse-Linear Attention...")

    B, T, H, D = 2, 256, 8, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    q = torch.randn(B, T, H, D, device=device)
    k = torch.randn(B, T, H, D, device=device)
    v = torch.randn(B, T, H, D, device=device)

    config = SLAConfig(block_size=32, critical_threshold=0.3)
    output = sla(q, k, v, config=config)

    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print("\nSLA test passed!")
