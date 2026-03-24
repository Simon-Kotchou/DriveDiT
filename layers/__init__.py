"""
Core layer implementations for DriveDiT.
Mathematical primitives with explicit tensor operations.
"""

from .rope_v2 import (
    # Backward compatible exports (previously from rope.py)
    RoPELayer, rope, precompute_rope_freqs, precompute_rope_3d_freqs,
    # Enhanced v2 exports
    RoPEConfig,
    RoPELayerV2,
    RoPEFrequencyCache,
    rope_apply,
    rope_apply_qk,
    compute_rope_freqs,
    compute_rope_3d_freqs,
    apply_rope_3d_efficient
)
from .mha import (
    mha,
    MultiHeadAttention,
    CausalMultiHeadAttention,
    BidirectionalMultiHeadAttention,
    create_causal_mask,
    create_block_causal_mask
)
from .mlp import MLP, SwiGLU, GeGLU, ConditionalMLP, create_mlp
from .nn_helpers import (
    RMSNorm,
    LayerNorm,
    AdaLN,
    FusedMLP,
    silu,
    gelu_tanh,
    fused_add_norm,
    apply_rotary_pos_emb,
    PositionalEncoding
)

__all__ = [
    # RoPE (unified - v2 with backward compat)
    'RoPELayer',
    'rope',
    'precompute_rope_freqs',
    'precompute_rope_3d_freqs',
    'RoPEConfig',
    'RoPELayerV2',
    'RoPEFrequencyCache',
    'rope_apply',
    'rope_apply_qk',
    'compute_rope_freqs',
    'compute_rope_3d_freqs',
    'apply_rope_3d_efficient',
    # Attention
    'mha',
    'MultiHeadAttention',
    'CausalMultiHeadAttention',
    'BidirectionalMultiHeadAttention',
    'create_causal_mask',
    'create_block_causal_mask',
    # MLP
    'MLP',
    'SwiGLU',
    'GeGLU',
    'ConditionalMLP',
    'create_mlp',
    # Helpers
    'RMSNorm',
    'LayerNorm',
    'AdaLN',
    'FusedMLP',
    'silu',
    'gelu_tanh',
    'fused_add_norm',
    'apply_rotary_pos_emb',
    'PositionalEncoding',
]
