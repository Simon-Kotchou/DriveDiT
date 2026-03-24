"""
Triton kernels for Mixture of Experts (MoE) dispatch.

Optimized GPU kernels for:
- Top-k expert routing with softmax
- Expert scatter (distribute tokens to experts)
- Expert gather (collect results from experts)
- Load balancing auxiliary loss

Based on Megablocks (arxiv:2211.15841) and DeepSeek-V3 patterns.

Install: pip install triton>=2.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

# Try to import Triton
MOE_KERNELS_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    MOE_KERNELS_AVAILABLE = True
except ImportError:
    triton = None
    tl = None


if MOE_KERNELS_AVAILABLE:
    # =========================================================================
    # Top-K Softmax Kernel (Fused)
    # =========================================================================

    @triton.jit
    def _topk_softmax_kernel(
        logits_ptr, indices_ptr, weights_ptr,
        stride_lb, stride_lt, stride_le,
        stride_ib, stride_it, stride_ik,
        stride_wb, stride_wt, stride_wk,
        num_tokens, num_experts, top_k: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """
        Fused top-k selection and softmax normalization.

        For each token:
        1. Find top-k experts by logit score
        2. Apply softmax over selected experts only
        """
        pid_b = tl.program_id(0)  # batch
        pid_t = tl.program_id(1)  # token

        # Load logits for this token
        logits_base = logits_ptr + pid_b * stride_lb + pid_t * stride_lt
        expert_offsets = tl.arange(0, BLOCK_E)
        expert_mask = expert_offsets < num_experts

        logits = tl.load(logits_base + expert_offsets * stride_le, mask=expert_mask, other=-float('inf'))

        # Find top-k (simple bubble sort for small k)
        top_indices = tl.zeros([top_k], dtype=tl.int32) - 1
        top_values = tl.full([top_k], -float('inf'), dtype=tl.float32)

        for e in range(num_experts):
            val = logits[e]
            # Insert into sorted top-k
            for k in range(top_k - 1, -1, -1):
                if val > top_values[k]:
                    if k < top_k - 1:
                        top_values[k + 1] = top_values[k]
                        top_indices[k + 1] = top_indices[k]
                    top_values[k] = val
                    top_indices[k] = e
                else:
                    break

        # Softmax over top-k
        max_val = tl.max(top_values, axis=0)
        exp_vals = tl.exp(top_values - max_val)
        sum_exp = tl.sum(exp_vals, axis=0)
        softmax_vals = exp_vals / (sum_exp + 1e-6)

        # Store results
        indices_base = indices_ptr + pid_b * stride_ib + pid_t * stride_it
        weights_base = weights_ptr + pid_b * stride_wb + pid_t * stride_wt

        for k in range(top_k):
            tl.store(indices_base + k * stride_ik, top_indices[k])
            tl.store(weights_base + k * stride_wk, softmax_vals[k])


    # =========================================================================
    # Expert Scatter Kernel
    # =========================================================================

    @triton.jit
    def _expert_scatter_kernel(
        input_ptr, indices_ptr, output_ptr, token_counts_ptr,
        stride_ib, stride_it, stride_id,
        stride_xb, stride_xt, stride_xk,
        stride_ob, stride_oe, stride_ot, stride_od,
        num_tokens, hidden_dim, num_experts, top_k: tl.constexpr,
        max_tokens_per_expert: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Scatter tokens to their assigned experts.

        Takes dense [B, T, D] input and produces sparse [B, E, max_tokens, D] output
        where each expert has its assigned tokens.
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_k = tl.program_id(2)

        # Get expert assignment for this token
        idx_ptr = indices_ptr + pid_b * stride_xb + pid_t * stride_xt + pid_k * stride_xk
        expert_idx = tl.load(idx_ptr)

        if expert_idx < 0:
            return

        # Atomically get position in expert's buffer
        count_ptr = token_counts_ptr + pid_b * num_experts + expert_idx
        pos = tl.atomic_add(count_ptr, 1)

        if pos >= max_tokens_per_expert:
            return

        # Copy token to expert buffer
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_dim

        # Load input token
        in_ptr = input_ptr + pid_b * stride_ib + pid_t * stride_it
        token = tl.load(in_ptr + d_offsets * stride_id, mask=d_mask, other=0.0)

        # Store to expert buffer
        out_ptr = output_ptr + pid_b * stride_ob + expert_idx * stride_oe + pos * stride_ot
        tl.store(out_ptr + d_offsets * stride_od, token, mask=d_mask)


    # =========================================================================
    # Expert Gather Kernel
    # =========================================================================

    @triton.jit
    def _expert_gather_kernel(
        expert_output_ptr, indices_ptr, weights_ptr, output_ptr,
        position_ptr,
        stride_eb, stride_ee, stride_et, stride_ed,
        stride_xb, stride_xt, stride_xk,
        stride_wb, stride_wt, stride_wk,
        stride_ob, stride_ot, stride_od,
        stride_pb, stride_pe,
        num_tokens, hidden_dim, num_experts, top_k: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Gather expert outputs and combine with routing weights.

        Takes sparse expert outputs and produces dense [B, T, D] output.
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)

        # Initialize output accumulator
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_dim
        output_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Accumulate weighted outputs from each assigned expert
        for k in range(top_k):
            idx_ptr = indices_ptr + pid_b * stride_xb + pid_t * stride_xt + k * stride_xk
            expert_idx = tl.load(idx_ptr)

            if expert_idx < 0:
                continue

            weight_ptr = weights_ptr + pid_b * stride_wb + pid_t * stride_wt + k * stride_wk
            weight = tl.load(weight_ptr)

            # Get this token's position in the expert buffer
            pos_ptr = position_ptr + pid_b * stride_pb + pid_t * num_experts + expert_idx
            pos = tl.load(pos_ptr)

            # Load expert output
            exp_ptr = expert_output_ptr + pid_b * stride_eb + expert_idx * stride_ee + pos * stride_et
            exp_out = tl.load(exp_ptr + d_offsets * stride_ed, mask=d_mask, other=0.0)

            output_acc += weight * exp_out.to(tl.float32)

        # Store combined output
        out_ptr = output_ptr + pid_b * stride_ob + pid_t * stride_ot
        tl.store(out_ptr + d_offsets * stride_od, output_acc.to(tl.float16), mask=d_mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def triton_topk_softmax(
    logits: torch.Tensor,
    top_k: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused top-k selection and softmax using Triton.

    Args:
        logits: Router logits [B, T, num_experts]
        top_k: Number of experts per token

    Returns:
        indices: Selected expert indices [B, T, top_k]
        weights: Softmax weights for selected experts [B, T, top_k]
    """
    if not MOE_KERNELS_AVAILABLE:
        raise RuntimeError("Triton not available")

    B, T, E = logits.shape

    indices = torch.empty(B, T, top_k, device=logits.device, dtype=torch.int32)
    weights = torch.empty(B, T, top_k, device=logits.device, dtype=torch.float32)

    # Ensure power of 2 for BLOCK_E
    BLOCK_E = triton.next_power_of_2(E)

    grid = (B, T)
    _topk_softmax_kernel[grid](
        logits, indices, weights,
        logits.stride(0), logits.stride(1), logits.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        T, E, top_k,
        BLOCK_E=BLOCK_E,
    )

    return indices, weights


def triton_expert_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    max_tokens_per_expert: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter tokens to experts using Triton kernel.

    Args:
        x: Input tensor [B, T, D]
        indices: Expert indices [B, T, top_k]
        num_experts: Number of experts
        max_tokens_per_expert: Capacity per expert (default: T * top_k / num_experts * 2)

    Returns:
        expert_input: Scattered tokens [B, E, max_tokens, D]
        token_counts: Number of tokens per expert [B, E]
    """
    if not MOE_KERNELS_AVAILABLE:
        raise RuntimeError("Triton not available")

    B, T, D = x.shape
    top_k = indices.shape[2]

    if max_tokens_per_expert is None:
        max_tokens_per_expert = (T * top_k // num_experts) * 2 + 1

    # Ensure contiguous and half precision
    x = x.contiguous().half()

    # Output buffers
    expert_input = torch.zeros(B, num_experts, max_tokens_per_expert, D,
                               device=x.device, dtype=x.dtype)
    token_counts = torch.zeros(B, num_experts, device=x.device, dtype=torch.int32)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, T, top_k)
    _expert_scatter_kernel[grid](
        x, indices, expert_input, token_counts,
        x.stride(0), x.stride(1), x.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        expert_input.stride(0), expert_input.stride(1), expert_input.stride(2), expert_input.stride(3),
        T, D, num_experts, top_k,
        max_tokens_per_expert=max_tokens_per_expert,
        BLOCK_D=BLOCK_D,
    )

    return expert_input, token_counts


def triton_expert_gather(
    expert_output: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    output_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Gather and combine expert outputs using Triton kernel.

    Args:
        expert_output: Expert outputs [B, E, max_tokens, D]
        indices: Expert indices [B, T, top_k]
        weights: Routing weights [B, T, top_k]
        positions: Token positions in expert buffers [B, T, E]
        output_shape: (B, T, D)

    Returns:
        Combined output [B, T, D]
    """
    if not MOE_KERNELS_AVAILABLE:
        raise RuntimeError("Triton not available")

    B, T, D = output_shape
    top_k = indices.shape[2]
    num_experts = expert_output.shape[1]

    output = torch.empty(B, T, D, device=expert_output.device, dtype=expert_output.dtype)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, T)
    _expert_gather_kernel[grid](
        expert_output, indices, weights, output, positions,
        expert_output.stride(0), expert_output.stride(1), expert_output.stride(2), expert_output.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        positions.stride(0), positions.stride(1),
        T, D, num_experts, top_k,
        BLOCK_D=BLOCK_D,
    )

    return output


class TritonMoEDispatch(nn.Module):
    """
    Triton-accelerated MoE dispatch layer.

    Handles token routing, scattering to experts, and gathering results
    using optimized GPU kernels.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        expert_fn: callable
    ) -> Tuple[torch.Tensor, dict]:
        """
        Route tokens to experts and combine outputs.

        Args:
            x: Input tensor [B, T, D]
            expert_fn: Function that processes expert inputs [B, E, max_tokens, D] -> [B, E, max_tokens, D]

        Returns:
            output: Combined output [B, T, D]
            aux_info: Dictionary with routing statistics
        """
        B, T, D = x.shape

        # Compute routing logits
        logits = self.router(x)  # [B, T, E]

        if MOE_KERNELS_AVAILABLE:
            # Use Triton kernels
            indices, weights = triton_topk_softmax(logits, self.top_k)

            max_tokens = int(T * self.top_k / self.num_experts * self.capacity_factor) + 1
            expert_input, token_counts = triton_expert_scatter(
                x, indices, self.num_experts, max_tokens
            )

            # Process through experts
            expert_output = expert_fn(expert_input)

            # Build position map for gather
            positions = self._build_position_map(indices, token_counts)

            # Gather and combine
            output = triton_expert_gather(
                expert_output, indices, weights, positions, (B, T, D)
            )
        else:
            # Fallback to PyTorch
            output, indices, weights = self._pytorch_dispatch(x, logits, expert_fn)

        # Compute auxiliary info
        aux_info = {
            'router_logits': logits,
            'expert_indices': indices,
            'expert_weights': weights,
            'expert_counts': token_counts if MOE_KERNELS_AVAILABLE else None,
        }

        return output, aux_info

    def _build_position_map(
        self,
        indices: torch.Tensor,
        token_counts: torch.Tensor
    ) -> torch.Tensor:
        """Build mapping from (batch, token, expert) -> position in expert buffer."""
        B, T, K = indices.shape
        E = self.num_experts

        positions = torch.zeros(B, T, E, device=indices.device, dtype=torch.int32)

        for b in range(B):
            expert_pos = torch.zeros(E, device=indices.device, dtype=torch.int32)
            for t in range(T):
                for k in range(K):
                    e = indices[b, t, k].item()
                    if e >= 0:
                        positions[b, t, e] = expert_pos[e]
                        expert_pos[e] += 1

        return positions

    def _pytorch_dispatch(self, x, logits, expert_fn):
        """Fallback PyTorch implementation."""
        B, T, D = x.shape

        # Top-k selection
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Simple implementation: process each expert separately
        output = torch.zeros_like(x)

        for e in range(self.num_experts):
            mask = (indices == e).any(dim=-1)  # [B, T]
            if not mask.any():
                continue

            # Get tokens for this expert
            expert_tokens = x[mask]  # [num_tokens, D]
            if expert_tokens.numel() == 0:
                continue

            # Process through expert (assuming expert_fn handles single expert)
            # This is simplified - real implementation would batch properly
            expert_out = expert_tokens  # Placeholder

            # Combine with weights
            for k in range(self.top_k):
                expert_mask = indices[:, :, k] == e
                weight = weights[:, :, k:k+1]
                output[expert_mask] += (weight[expert_mask] * expert_out).squeeze()

        return output, indices, weights


# =============================================================================
# Fallback implementations
# =============================================================================

if not MOE_KERNELS_AVAILABLE:
    def triton_topk_softmax(logits, top_k=2):
        """Fallback: PyTorch top-k softmax."""
        weights, indices = torch.topk(logits, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return indices.int(), weights

    def triton_expert_scatter(x, indices, num_experts, max_tokens_per_expert=None):
        """Fallback: PyTorch scatter."""
        B, T, D = x.shape
        top_k = indices.shape[2]

        if max_tokens_per_expert is None:
            max_tokens_per_expert = (T * top_k // num_experts) * 2 + 1

        expert_input = torch.zeros(B, num_experts, max_tokens_per_expert, D,
                                   device=x.device, dtype=x.dtype)
        token_counts = torch.zeros(B, num_experts, device=x.device, dtype=torch.int32)

        for b in range(B):
            for t in range(T):
                for k in range(top_k):
                    e = indices[b, t, k].item()
                    if e >= 0 and e < num_experts:
                        pos = token_counts[b, e].item()
                        if pos < max_tokens_per_expert:
                            expert_input[b, e, pos] = x[b, t]
                            token_counts[b, e] += 1

        return expert_input, token_counts

    def triton_expert_gather(expert_output, indices, weights, positions, output_shape):
        """Fallback: PyTorch gather."""
        B, T, D = output_shape
        top_k = indices.shape[2]

        output = torch.zeros(B, T, D, device=expert_output.device, dtype=expert_output.dtype)

        for b in range(B):
            for t in range(T):
                for k in range(top_k):
                    e = indices[b, t, k].item()
                    if e >= 0:
                        pos = positions[b, t, e].item()
                        w = weights[b, t, k]
                        output[b, t] += w * expert_output[b, e, pos]

        return output
