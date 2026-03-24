"""
Fused Triton kernels for common operations.

Optimized kernels for:
- RMSNorm (fused normalization)
- SiLU * gate (fused activation)
- Add + RMSNorm (fused residual + norm)
- Rotary Position Embedding (RoPE)

These fused operations reduce memory bandwidth and improve throughput.

Install: pip install triton>=2.1.0
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

# Try to import Triton
FUSED_OPS_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    FUSED_OPS_AVAILABLE = True
except ImportError:
    triton = None
    tl = None


if FUSED_OPS_AVAILABLE:
    # =========================================================================
    # Fused RMSNorm Kernel
    # =========================================================================

    @triton.jit
    def _rmsnorm_kernel(
        X_ptr, W_ptr, Out_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_ob, stride_ot, stride_od,
        num_tokens, hidden_dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused RMSNorm: out = x / rms(x) * weight
        where rms(x) = sqrt(mean(x^2) + eps)
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)

        # Load input
        x_base = X_ptr + pid_b * stride_xb + pid_t * stride_xt
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_dim

        x = tl.load(x_base + d_offsets * stride_xd, mask=d_mask, other=0.0)
        x_fp32 = x.to(tl.float32)

        # Compute RMS
        mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / hidden_dim
        rms = tl.sqrt(mean_sq + eps)

        # Normalize
        x_norm = x_fp32 / rms

        # Load weight and apply
        w = tl.load(W_ptr + d_offsets, mask=d_mask, other=1.0)
        out = x_norm * w.to(tl.float32)

        # Store output
        out_base = Out_ptr + pid_b * stride_ob + pid_t * stride_ot
        tl.store(out_base + d_offsets * stride_od, out.to(tl.float16), mask=d_mask)


    # =========================================================================
    # Fused SiLU * Gate Kernel (for MLP)
    # =========================================================================

    @triton.jit
    def _silu_multiply_kernel(
        X_ptr, Gate_ptr, Out_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_gb, stride_gt, stride_gd,
        stride_ob, stride_ot, stride_od,
        num_tokens, hidden_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused SiLU activation with element-wise multiply.
        out = silu(x) * gate = (x * sigmoid(x)) * gate
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_dim

        # Load inputs
        x_base = X_ptr + pid_b * stride_xb + pid_t * stride_xt
        g_base = Gate_ptr + pid_b * stride_gb + pid_t * stride_gt

        x = tl.load(x_base + d_offsets * stride_xd, mask=d_mask, other=0.0)
        gate = tl.load(g_base + d_offsets * stride_gd, mask=d_mask, other=0.0)

        # Compute SiLU: x * sigmoid(x)
        x_fp32 = x.to(tl.float32)
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x_fp32))
        silu_x = x_fp32 * sigmoid_x

        # Multiply with gate
        out = silu_x * gate.to(tl.float32)

        # Store
        out_base = Out_ptr + pid_b * stride_ob + pid_t * stride_ot
        tl.store(out_base + d_offsets * stride_od, out.to(tl.float16), mask=d_mask)


    # =========================================================================
    # Fused Add + RMSNorm Kernel (for residual connections)
    # =========================================================================

    @triton.jit
    def _add_rmsnorm_kernel(
        X_ptr, Residual_ptr, W_ptr, Out_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_rb, stride_rt, stride_rd,
        stride_ob, stride_ot, stride_od,
        num_tokens, hidden_dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused residual add + RMSNorm.
        out = rmsnorm(x + residual) * weight
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_dim

        # Load inputs
        x_base = X_ptr + pid_b * stride_xb + pid_t * stride_xt
        r_base = Residual_ptr + pid_b * stride_rb + pid_t * stride_rt

        x = tl.load(x_base + d_offsets * stride_xd, mask=d_mask, other=0.0)
        residual = tl.load(r_base + d_offsets * stride_rd, mask=d_mask, other=0.0)

        # Add
        combined = x.to(tl.float32) + residual.to(tl.float32)

        # RMSNorm
        mean_sq = tl.sum(combined * combined, axis=0) / hidden_dim
        rms = tl.sqrt(mean_sq + eps)
        normalized = combined / rms

        # Apply weight
        w = tl.load(W_ptr + d_offsets, mask=d_mask, other=1.0)
        out = normalized * w.to(tl.float32)

        # Store
        out_base = Out_ptr + pid_b * stride_ob + pid_t * stride_ot
        tl.store(out_base + d_offsets * stride_od, out.to(tl.float16), mask=d_mask)


    # =========================================================================
    # Fused Rotary Embedding Kernel
    # =========================================================================

    @triton.jit
    def _rotary_embedding_kernel(
        Q_ptr, K_ptr, Cos_ptr, Sin_ptr, Q_out_ptr, K_out_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_kb, stride_kt, stride_kh, stride_kd,
        stride_ct, stride_cd,
        stride_qob, stride_qot, stride_qoh, stride_qod,
        stride_kob, stride_kot, stride_koh, stride_kod,
        seq_len, num_heads, head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused rotary position embedding for Q and K.

        Applies rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        to pairs of dimensions.
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_h = tl.program_id(2)

        half_d = head_dim // 2
        d_offsets = tl.arange(0, BLOCK_D // 2)
        d_mask = d_offsets < half_d

        # Load cos/sin for this position
        cos_base = Cos_ptr + pid_t * stride_ct
        sin_base = Sin_ptr + pid_t * stride_ct
        cos_vals = tl.load(cos_base + d_offsets * stride_cd, mask=d_mask, other=1.0)
        sin_vals = tl.load(sin_base + d_offsets * stride_cd, mask=d_mask, other=0.0)

        # Process Q
        q_base = Q_ptr + pid_b * stride_qb + pid_t * stride_qt + pid_h * stride_qh
        q1 = tl.load(q_base + (d_offsets * 2) * stride_qd, mask=d_mask, other=0.0)
        q2 = tl.load(q_base + (d_offsets * 2 + 1) * stride_qd, mask=d_mask, other=0.0)

        q1_out = q1.to(tl.float32) * cos_vals.to(tl.float32) - q2.to(tl.float32) * sin_vals.to(tl.float32)
        q2_out = q1.to(tl.float32) * sin_vals.to(tl.float32) + q2.to(tl.float32) * cos_vals.to(tl.float32)

        qo_base = Q_out_ptr + pid_b * stride_qob + pid_t * stride_qot + pid_h * stride_qoh
        tl.store(qo_base + (d_offsets * 2) * stride_qod, q1_out.to(tl.float16), mask=d_mask)
        tl.store(qo_base + (d_offsets * 2 + 1) * stride_qod, q2_out.to(tl.float16), mask=d_mask)

        # Process K
        k_base = K_ptr + pid_b * stride_kb + pid_t * stride_kt + pid_h * stride_kh
        k1 = tl.load(k_base + (d_offsets * 2) * stride_kd, mask=d_mask, other=0.0)
        k2 = tl.load(k_base + (d_offsets * 2 + 1) * stride_kd, mask=d_mask, other=0.0)

        k1_out = k1.to(tl.float32) * cos_vals.to(tl.float32) - k2.to(tl.float32) * sin_vals.to(tl.float32)
        k2_out = k1.to(tl.float32) * sin_vals.to(tl.float32) + k2.to(tl.float32) * cos_vals.to(tl.float32)

        ko_base = K_out_ptr + pid_b * stride_kob + pid_t * stride_kot + pid_h * stride_koh
        tl.store(ko_base + (d_offsets * 2) * stride_kod, k1_out.to(tl.float16), mask=d_mask)
        tl.store(ko_base + (d_offsets * 2 + 1) * stride_kod, k2_out.to(tl.float16), mask=d_mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused RMSNorm using Triton kernel.

    Args:
        x: Input tensor [B, T, D] or [B, D]
        weight: Scale weights [D]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    if not FUSED_OPS_AVAILABLE:
        # Fallback to PyTorch
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return x_norm * weight

    orig_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(1)

    B, T, D = x.shape
    x = x.contiguous().half()

    out = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, T)
    _rmsnorm_kernel[grid](
        x, weight, out,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        T, D, eps,
        BLOCK_D=BLOCK_D,
    )

    if len(orig_shape) == 2:
        out = out.squeeze(1)

    return out


def fused_silu_multiply(
    x: torch.Tensor,
    gate: torch.Tensor
) -> torch.Tensor:
    """
    Fused SiLU activation with element-wise multiply.

    Computes: silu(x) * gate = (x * sigmoid(x)) * gate

    Args:
        x: Input tensor [B, T, D]
        gate: Gate tensor [B, T, D]

    Returns:
        Output tensor [B, T, D]
    """
    if not FUSED_OPS_AVAILABLE:
        # Fallback to PyTorch
        return torch.nn.functional.silu(x) * gate

    B, T, D = x.shape
    x = x.contiguous().half()
    gate = gate.contiguous().half()

    out = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, T)
    _silu_multiply_kernel[grid](
        x, gate, out,
        x.stride(0), x.stride(1), x.stride(2),
        gate.stride(0), gate.stride(1), gate.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        T, D,
        BLOCK_D=BLOCK_D,
    )

    return out


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused residual add + RMSNorm.

    Computes: rmsnorm(x + residual) * weight

    Args:
        x: Input tensor [B, T, D]
        residual: Residual tensor [B, T, D]
        weight: Scale weights [D]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [B, T, D]
    """
    if not FUSED_OPS_AVAILABLE:
        # Fallback to PyTorch
        combined = x + residual
        variance = combined.float().pow(2).mean(-1, keepdim=True)
        x_norm = combined * torch.rsqrt(variance + eps)
        return x_norm * weight

    B, T, D = x.shape
    x = x.contiguous().half()
    residual = residual.contiguous().half()

    out = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, T)
    _add_rmsnorm_kernel[grid](
        x, residual, weight, out,
        x.stride(0), x.stride(1), x.stride(2),
        residual.stride(0), residual.stride(1), residual.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        T, D, eps,
        BLOCK_D=BLOCK_D,
    )

    return out


def fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused rotary position embedding for Q and K.

    Args:
        q: Query tensor [B, T, H, D]
        k: Key tensor [B, T, H, D]
        cos: Cosine values [T, D//2]
        sin: Sine values [T, D//2]

    Returns:
        q_rot: Rotated queries [B, T, H, D]
        k_rot: Rotated keys [B, T, H, D]
    """
    if not FUSED_OPS_AVAILABLE:
        # Fallback to PyTorch
        def rotate_half(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([-x2, x1], dim=-1).flatten(-2)

        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D//2]
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Interleave cos/sin
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

    B, T, H, D = q.shape
    q = q.contiguous().half()
    k = k.contiguous().half()
    cos = cos.contiguous()
    sin = sin.contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, T, H)
    _rotary_embedding_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        T, H, D,
        BLOCK_D=BLOCK_D,
    )

    return q_out, k_out


# =============================================================================
# Fused Module Classes
# =============================================================================

class FusedRMSNorm(nn.Module):
    """RMSNorm using fused Triton kernel."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


class FusedSiLUMLP(nn.Module):
    """MLP with fused SiLU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        hidden = fused_silu_multiply(gate, up)
        return self.down_proj(hidden)


# =============================================================================
# Fallbacks when Triton not available
# =============================================================================

if not FUSED_OPS_AVAILABLE:
    def fused_rmsnorm(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return (x_norm * weight).to(x.dtype)

    def fused_silu_multiply(x, gate):
        return torch.nn.functional.silu(x) * gate

    def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
        combined = x + residual
        variance = combined.float().pow(2).mean(-1, keepdim=True)
        x_norm = combined * torch.rsqrt(variance + eps)
        return (x_norm * weight).to(x.dtype)

    def fused_rotary_embedding(q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([-x2, x1], dim=-1).flatten(-2)

        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

    class FusedRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + self.eps)
            return (x_norm * self.weight).to(x.dtype)

    class FusedSiLUMLP(nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
