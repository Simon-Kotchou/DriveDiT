"""
Custom Triton kernels for DriveDiT GPU acceleration.

Provides optimized kernels for:
- SLA (Sparse-Linear Attention) block routing
- MoE (Mixture of Experts) expert dispatch
- Fused normalization and activation operations

Install Triton: pip install triton>=2.1.0
"""

from .sla_kernels import (
    triton_linear_attention,
    triton_block_routing,
    TritonSLA,
    SLA_KERNELS_AVAILABLE,
)
from .moe_kernels import (
    triton_expert_scatter,
    triton_expert_gather,
    triton_topk_softmax,
    TritonMoEDispatch,
    MOE_KERNELS_AVAILABLE,
)
from .fused_ops import (
    fused_rmsnorm,
    fused_silu_multiply,
    fused_add_rmsnorm,
    fused_rotary_embedding,
    FUSED_OPS_AVAILABLE,
)

__all__ = [
    # SLA kernels
    'triton_linear_attention',
    'triton_block_routing',
    'TritonSLA',
    'SLA_KERNELS_AVAILABLE',
    # MoE kernels
    'triton_expert_scatter',
    'triton_expert_gather',
    'triton_topk_softmax',
    'TritonMoEDispatch',
    'MOE_KERNELS_AVAILABLE',
    # Fused ops
    'fused_rmsnorm',
    'fused_silu_multiply',
    'fused_add_rmsnorm',
    'fused_rotary_embedding',
    'FUSED_OPS_AVAILABLE',
]
