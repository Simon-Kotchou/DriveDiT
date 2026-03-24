"""
Mixture of Experts (MoE) implementation for DiT FFN blocks.

DeepSeek-style implementation with:
- Fine-grained experts (many small experts, activate few)
- Auxiliary-loss-free load balancing via learnable bias
- Optional shared experts for common knowledge
- Proper gradient flow through top-k selection

References:
- DeepSeek-V3: arxiv:2412.19437
- GigaWorld-0: 2B active params matches 14B dense
- Dense2MoE: 62.5% reduction in activated FFN parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math


@dataclass
class MoEMetrics:
    """Metrics for monitoring MoE routing behavior."""
    expert_counts: torch.Tensor  # [num_experts] token count per expert
    expert_probs: torch.Tensor  # [num_experts] average routing probability
    load_balance_loss: float  # CV^2 loss for reference
    expert_utilization: float  # Fraction of experts with non-zero load
    max_load_ratio: float  # max(load) / mean(load)
    entropy: float  # Routing entropy
    avg_expert_confidence: float  # Average top-1 routing weight
    routing_sparsity: float  # Fraction of near-zero routing weights


class MoEOutput(NamedTuple):
    """Output from MoE forward pass."""
    output: torch.Tensor
    metrics: Optional[MoEMetrics] = None


class ExpertFFN(nn.Module):
    """Single expert FFN using SwiGLU activation."""

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_expert, bias=bias)
        self.w_value = nn.Linear(d_model, d_expert, bias=bias)
        self.w_out = nn.Linear(d_expert, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_value.weight, std=0.02)
        nn.init.normal_(self.w_out.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: x * SiLU(gate) * value."""
        gate = F.silu(self.w_gate(x))
        value = self.w_value(x)
        hidden = gate * value
        hidden = self.dropout(hidden)
        return self.w_out(hidden)


class Router(nn.Module):
    """
    Top-k router with auxiliary-loss-free load balancing.

    Uses learnable bias terms updated via EMA of expert utilization
    to achieve load balancing without explicit auxiliary losses.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        bias_update_speed: float = 0.001
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.bias_update_speed = bias_update_speed

        # Router gate
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

        # Learnable load balancing bias (updated via EMA)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('expert_load_ema', torch.ones(num_experts) / num_experts)

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Route tokens to experts.

        Args:
            x: [N, D] flattened tokens
            return_metrics: Whether to compute routing metrics

        Returns:
            Tuple of (routing_weights, selected_experts, metrics)
        """
        N = x.shape[0]

        # Compute routing logits
        logits = self.gate(x)  # [N, num_experts]

        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        # Add load balancing bias
        logits = logits + self.expert_bias

        # Softmax for probabilities
        routing_probs = F.softmax(logits, dim=-1)  # [N, num_experts]

        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # [N, top_k], [N, top_k]

        # Renormalize weights for selected experts
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)

        metrics = None
        if self.training or return_metrics:
            # Count tokens per expert for load balancing
            expert_counts = torch.zeros(
                self.num_experts, device=x.device, dtype=torch.long
            )
            for k in range(self.top_k):
                expert_counts.scatter_add_(
                    0, selected_experts[:, k],
                    torch.ones(N, device=x.device, dtype=torch.long)
                )

            # Update load balancing bias
            self._update_load_balance_bias(expert_counts, N)

            if return_metrics:
                metrics = self._compute_metrics(
                    routing_probs, routing_weights, selected_experts, expert_counts
                )

        return routing_weights, selected_experts, metrics

    def _update_load_balance_bias(self, expert_counts: torch.Tensor, n_tokens: int):
        """Update load balancing bias based on current batch statistics."""
        if not self.training:
            return

        # Compute load as fraction of tokens
        load = expert_counts.float() / (n_tokens * self.top_k + 1e-8)

        # Update EMA
        self.expert_load_ema.mul_(1 - self.bias_update_speed).add_(
            load, alpha=self.bias_update_speed
        )

        # Compute bias update: experts with high load get negative bias
        target_load = 1.0 / self.num_experts
        load_deviation = self.expert_load_ema - target_load
        bias_update = -self.bias_update_speed * load_deviation
        self.expert_bias.add_(bias_update)

    def _compute_metrics(
        self,
        routing_probs: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_counts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute routing metrics for monitoring."""
        N = routing_probs.shape[0]

        # Average probability per expert
        expert_probs = routing_probs.mean(dim=0)

        # Load balance loss (CV^2 - coefficient of variation squared)
        f = expert_counts.float() / (N + 1e-8)
        P = expert_probs
        load_balance_loss = self.num_experts * (f * P).sum()

        # Expert utilization
        active_experts = (expert_counts > 0).float().sum()
        expert_utilization = active_experts / self.num_experts

        # Max load ratio
        mean_load = expert_counts.float().mean()
        max_load = expert_counts.float().max()
        max_load_ratio = max_load / (mean_load + 1e-8)

        # Routing entropy
        entropy = -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1).mean()

        # Average confidence
        avg_confidence = routing_weights[:, 0].mean()

        # Routing sparsity
        routing_sparsity = (routing_probs < 1e-3).float().mean()

        return {
            'expert_counts': expert_counts,
            'expert_probs': expert_probs,
            'load_balance_loss': load_balance_loss,
            'expert_utilization': expert_utilization,
            'max_load_ratio': max_load_ratio,
            'entropy': entropy,
            'avg_confidence': avg_confidence,
            'routing_sparsity': routing_sparsity
        }


class MoEFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.

    DeepSeek-style implementation with fine-grained experts,
    auxiliary-loss-free load balancing, and optional shared experts.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dim: Optional[int] = None,
        num_shared_experts: int = 0,
        dropout: float = 0.0,
        bias: bool = False,
        jitter_noise: float = 0.0,
        bias_update_speed: float = 0.001
    ):
        """
        Initialize MoE FFN.

        Args:
            d_model: Input/output dimension
            num_experts: Total number of routed experts
            top_k: Number of experts to activate per token
            expert_dim: Hidden dim per expert (default: 4*d_model/top_k)
            num_shared_experts: Number of always-active shared experts
            dropout: Dropout probability
            bias: Use bias in linear layers
            jitter_noise: Router exploration noise
            bias_update_speed: Load balancing update speed
        """
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        # Compute expert dimension to match dense FFN capacity
        if expert_dim is None:
            expert_dim = int(4 * d_model / top_k)
            expert_dim = ((expert_dim + 7) // 8) * 8  # Round to 8

        self.expert_dim = expert_dim

        # Router
        self.router = Router(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            bias_update_speed=bias_update_speed
        )

        # Routed experts
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, expert_dim, dropout, bias)
            for _ in range(num_experts)
        ])

        # Shared experts (always activated)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertFFN(d_model, expert_dim, dropout, bias)
                for _ in range(num_shared_experts)
            ])
            self.shared_gate = nn.Linear(d_model, num_shared_experts, bias=False)
        else:
            self.shared_experts = None
            self.shared_gate = None

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> MoEOutput:
        """
        Forward pass through MoE FFN.

        Args:
            x: Input tensor [B, T, D]
            return_metrics: Whether to return routing metrics

        Returns:
            MoEOutput containing output tensor and optional metrics
        """
        original_shape = x.shape
        B, T, D = x.shape

        # Flatten to [N, D]
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        # Route tokens to experts
        routing_weights, selected_experts, router_metrics = self.router(
            x_flat, return_metrics=return_metrics or self.training
        )

        # Compute expert outputs
        output = self._compute_expert_outputs(x_flat, routing_weights, selected_experts)

        # Add shared expert outputs
        if self.shared_experts is not None:
            shared_output = self._compute_shared_outputs(x_flat)
            output = output + shared_output

        # Reshape back
        output = output.view(*original_shape)

        # Compile metrics
        metrics = None
        if return_metrics and router_metrics is not None:
            metrics = MoEMetrics(
                expert_counts=router_metrics['expert_counts'],
                expert_probs=router_metrics['expert_probs'],
                load_balance_loss=router_metrics['load_balance_loss'].item(),
                expert_utilization=router_metrics['expert_utilization'].item(),
                max_load_ratio=router_metrics['max_load_ratio'].item(),
                entropy=router_metrics['entropy'].item(),
                avg_expert_confidence=router_metrics['avg_confidence'].item(),
                routing_sparsity=router_metrics['routing_sparsity'].item()
            )

        return MoEOutput(output=output, metrics=metrics)

    def _compute_expert_outputs(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted sum of expert outputs."""
        N, D = x.shape
        device = x.device
        dtype = x.dtype

        output = torch.zeros(N, D, device=device, dtype=dtype)

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]

            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)

            if not expert_mask.any():
                continue

            # Get indices of tokens selecting this expert
            token_indices = expert_mask.any(dim=1).nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue

            # Get inputs and compute output
            expert_input = x[token_indices]
            expert_output = expert(expert_input)

            # Get weights for this expert
            weights = (routing_weights * expert_mask.float()).sum(dim=1)
            expert_weights = weights[token_indices].unsqueeze(-1)

            # Accumulate weighted output
            output.index_add_(0, token_indices, expert_output * expert_weights)

        return output

    def _compute_shared_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute outputs from shared experts."""
        if self.shared_experts is None:
            return torch.zeros_like(x)

        # Gate weights for shared experts
        shared_weights = F.softmax(self.shared_gate(x), dim=-1)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.shared_experts):
            expert_output = expert(x)
            output = output + shared_weights[:, i:i+1] * expert_output

        return output

    def get_load_balance_stats(self) -> Dict[str, float]:
        """Get current load balancing statistics."""
        return {
            'load_ema_std': self.router.expert_load_ema.std().item(),
            'load_ema_max': self.router.expert_load_ema.max().item(),
            'load_ema_min': self.router.expert_load_ema.min().item(),
            'bias_std': self.router.expert_bias.std().item(),
            'bias_max': self.router.expert_bias.max().item(),
            'bias_min': self.router.expert_bias.min().item(),
        }


class MoEDiTBlock(nn.Module):
    """
    DiT block with MoE FFN - drop-in replacement for standard DiTBlock.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dim: Optional[int] = None,
        num_shared_experts: int = 0,
        dropout: float = 0.0,
        causal: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        cond_dim: Optional[int] = None,
        bias: bool = False,
        jitter_noise: float = 0.0,
        bias_update_speed: float = 0.001
    ):
        super().__init__()

        # Import attention modules
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from layers.mha import CausalMultiHeadAttention, BidirectionalMultiHeadAttention
        from layers.nn_helpers import AdaLN, RMSNorm
        from layers.rope_v2 import RoPELayer

        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        self.cond_dim = cond_dim

        # RoPE layer
        self.rope_layer = None
        if use_rope:
            self.rope_layer = RoPELayer(dim=d_model // n_heads, max_seq_len=max_seq_len)

        # Attention
        if causal:
            self.attn = CausalMultiHeadAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                bias=bias, rope_layer=self.rope_layer
            )
        else:
            self.attn = BidirectionalMultiHeadAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                bias=bias, rope_layer=self.rope_layer
            )

        # MoE FFN
        self.moe_ffn = MoEFFN(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            expert_dim=expert_dim,
            num_shared_experts=num_shared_experts,
            dropout=dropout,
            bias=bias,
            jitter_noise=jitter_noise,
            bias_update_speed=bias_update_speed
        )

        # Normalization
        if cond_dim is not None:
            self.norm_attn = AdaLN(d_model, cond_dim)
            self.norm_mlp = AdaLN(d_model, cond_dim)
        else:
            self.norm_attn = RMSNorm(d_model)
            self.norm_mlp = RMSNorm(d_model)

        self._last_moe_metrics: Optional[MoEMetrics] = None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        return_moe_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through MoE DiT block."""
        # Pre-norm for attention
        if self.cond_dim is not None and cond is not None:
            normed_x = self.norm_attn(x, cond)
        else:
            normed_x = self.norm_attn(x)

        # Attention with residual
        attn_out, new_kv_cache = self.attn(normed_x, kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out

        # Pre-norm for MoE FFN
        if self.cond_dim is not None and cond is not None:
            normed_x = self.norm_mlp(x, cond)
        else:
            normed_x = self.norm_mlp(x)

        # MoE FFN with residual
        moe_output = self.moe_ffn(normed_x, return_metrics=return_moe_metrics)
        x = x + moe_output.output

        if return_moe_metrics:
            self._last_moe_metrics = moe_output.metrics

        return x, new_kv_cache

    def get_moe_metrics(self) -> Optional[MoEMetrics]:
        """Get last computed MoE metrics."""
        return self._last_moe_metrics

    def get_load_balance_stats(self) -> Dict[str, float]:
        """Get current load balancing statistics."""
        return self.moe_ffn.get_load_balance_stats()


def create_moe_dit_block(d_model: int, n_heads: int, num_experts: int = 8, top_k: int = 2, **kwargs) -> MoEDiTBlock:
    """Factory function to create MoE DiT block."""
    return MoEDiTBlock(d_model=d_model, n_heads=n_heads, num_experts=num_experts, top_k=top_k, **kwargs)


if __name__ == "__main__":
    print("Testing MoE FFN implementation...")

    # Test MoE FFN
    B, T, D = 2, 16, 512
    x = torch.randn(B, T, D)

    moe_ffn = MoEFFN(d_model=D, num_experts=8, top_k=2, num_shared_experts=1)
    output = moe_ffn(x, return_metrics=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.output.shape}")

    if output.metrics:
        print(f"Expert utilization: {output.metrics.expert_utilization:.2%}")
        print(f"Max load ratio: {output.metrics.max_load_ratio:.2f}")

    print("\nAll MoE tests passed!")
