"""
Unit tests for layer components (rope.py, mha.py, mlp.py, nn_helpers.py).
Tests mathematical correctness and tensor operations.
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple
import math

# Import layers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from layers.rope import RoPELayer, precompute_rope_freqs, precompute_rope_3d_freqs, rope
from layers.mha import MultiHeadAttention, CausalMultiHeadAttention, BidirectionalMultiHeadAttention, mha, create_causal_mask
from layers.mlp import MLP, SwiGLU, GeGLU, create_mlp
from layers.nn_helpers import RMSNorm, LayerNorm, AdaLN, silu, gelu_tanh


class TestRoPE:
    """Test Rotary Positional Embedding implementation."""
    
    def test_rope_basic_functionality(self, device):
        """Test basic RoPE rotation."""
        B, T, H, D = 2, 4, 8, 16
        x = torch.randn(B, T, H, D, device=device)
        
        # Precompute frequencies
        sin, cos = precompute_rope_freqs(D, T, device=device)
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
        
        # Apply RoPE
        rotated = rope(x, sin, cos)
        
        # Check output shape
        assert rotated.shape == x.shape
        
        # Check that rotation preserves norm (approximately)
        original_norm = x.norm(dim=-1)
        rotated_norm = rotated.norm(dim=-1)
        assert torch.allclose(original_norm, rotated_norm, atol=1e-5)
    
    def test_rope_layer(self, device):
        """Test RoPE layer wrapper."""
        d_head = 64
        max_seq_len = 128
        
        rope_layer = RoPELayer(d_head, max_seq_len).to(device)
        
        B, T, H = 2, 10, 8
        x = torch.randn(B, T, H, d_head, device=device)
        
        # Apply RoPE
        rotated = rope_layer(x)
        
        assert rotated.shape == x.shape
        assert not torch.equal(rotated, x)  # Should be different
    
    def test_rope_3d_frequencies(self, device):
        """Test 3D RoPE frequency computation."""
        dim = 192  # Must be divisible by 6
        max_time, max_height, max_width = 8, 16, 16
        
        sin, cos = precompute_rope_3d_freqs(
            dim, max_time, max_height, max_width, device=device
        )
        
        expected_shape = (max_time, max_height, max_width, dim)
        assert sin.shape == expected_shape
        assert cos.shape == expected_shape
        
        # Check that sin^2 + cos^2 ≈ 1
        sin_cos_norm = sin**2 + cos**2
        assert torch.allclose(sin_cos_norm, torch.ones_like(sin_cos_norm), atol=1e-5)
    
    def test_rope_invariances(self, device):
        """Test RoPE mathematical properties."""
        d_head = 32
        rope_layer = RoPELayer(d_head, 16).to(device)
        
        # Test position shift invariance
        x1 = torch.randn(1, 4, 1, d_head, device=device)
        x2 = torch.randn(1, 4, 1, d_head, device=device)
        
        # Apply RoPE to concatenated sequence
        x_concat = torch.cat([x1, x2], dim=1)
        rotated_concat = rope_layer(x_concat)
        
        # Apply RoPE to second part with offset
        rotated_x2_offset = rope_layer(x2, start_pos=4)
        
        # Check that offset application gives same result
        assert torch.allclose(
            rotated_concat[:, 4:8], 
            rotated_x2_offset, 
            atol=1e-5
        )


class TestMultiHeadAttention:
    """Test Multi-Head Attention implementations."""
    
    def test_mha_basic_functionality(self, device):
        """Test basic multi-head attention."""
        B, T, H, D = 2, 8, 4, 16
        
        q = torch.randn(B, T, H, D, device=device)
        k = torch.randn(B, T, H, D, device=device)
        v = torch.randn(B, T, H, D, device=device)
        
        output = mha(q, k, v)
        
        assert output.shape == (B, T, H, D)
        assert torch.isfinite(output).all()
    
    def test_mha_with_mask(self, device):
        """Test attention with causal mask."""
        B, T, H, D = 2, 8, 4, 16
        
        q = torch.randn(B, T, H, D, device=device)
        k = torch.randn(B, T, H, D, device=device)
        v = torch.randn(B, T, H, D, device=device)
        
        # Create causal mask
        mask = create_causal_mask(T, device)
        
        output = mha(q, k, v, mask=mask)
        
        assert output.shape == (B, T, H, D)
        assert torch.isfinite(output).all()
    
    def test_multihead_attention_layer(self, device):
        """Test MultiHeadAttention layer."""
        d_model = 256
        n_heads = 8
        seq_len = 16
        batch_size = 2
        
        mha_layer = MultiHeadAttention(d_model, n_heads).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output, _ = mha_layer(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_causal_attention(self, device):
        """Test causal attention implementation."""
        d_model = 128
        n_heads = 4
        seq_len = 8
        batch_size = 1
        
        causal_mha = CausalMultiHeadAttention(d_model, n_heads).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output, _ = causal_mha(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_kv_caching(self, device):
        """Test KV caching functionality."""
        d_model = 128
        n_heads = 4
        seq_len = 4
        batch_size = 1
        
        mha_layer = MultiHeadAttention(d_model, n_heads).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # First forward pass
        output1, kv_cache = mha_layer(x, use_cache=True)
        
        # Second forward pass with cache
        x_new = torch.randn(batch_size, 1, d_model, device=device)
        output2, new_kv_cache = mha_layer(x_new, kv_cache=kv_cache, use_cache=True)
        
        assert output1.shape == x.shape
        assert output2.shape == x_new.shape
        assert kv_cache is not None
        assert new_kv_cache is not None
    
    def test_attention_weights_sum_to_one(self, device):
        """Test that attention weights sum to one."""
        B, T, H, D = 1, 4, 1, 8
        
        q = torch.randn(B, T, H, D, device=device)
        k = torch.randn(B, T, H, D, device=device)
        v = torch.randn(B, T, H, D, device=device)
        
        # Compute attention weights manually
        scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(D)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Check that weights sum to 1 along last dimension
        weight_sums = attn_weights.sum(dim=-1)
        expected = torch.ones_like(weight_sums)
        assert torch.allclose(weight_sums, expected, atol=1e-6)


class TestMLP:
    """Test Multi-Layer Perceptron implementations."""
    
    def test_standard_mlp(self, device):
        """Test standard MLP implementation."""
        d_model = 256
        d_ff = 1024
        batch_size = 2
        seq_len = 8
        
        mlp = MLP(d_model, d_ff, activation='silu').to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        output = mlp(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_swiglu_mlp(self, device):
        """Test SwiGLU MLP implementation."""
        d_model = 256
        batch_size = 2
        seq_len = 8
        
        mlp = SwiGLU(d_model).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        output = mlp(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_geglu_mlp(self, device):
        """Test GeGLU MLP implementation."""
        d_model = 256
        batch_size = 2
        seq_len = 8
        
        mlp = GeGLU(d_model).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        output = mlp(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_mlp_factory(self, device):
        """Test MLP factory function."""
        d_model = 128
        
        # Test different MLP types
        for mlp_type in ['standard', 'swiglu', 'geglu']:
            mlp = create_mlp(d_model, mlp_type=mlp_type).to(device)
            
            x = torch.randn(1, 4, d_model, device=device)
            output = mlp(x)
            
            assert output.shape == x.shape
            assert torch.isfinite(output).all()
    
    def test_mlp_different_activations(self, device):
        """Test MLP with different activation functions."""
        d_model = 128
        
        activations = ['silu', 'gelu', 'relu', 'swish']
        
        for activation in activations:
            mlp = MLP(d_model, activation=activation).to(device)
            
            x = torch.randn(1, 4, d_model, device=device)
            output = mlp(x)
            
            assert output.shape == x.shape
            assert torch.isfinite(output).all()


class TestNeuralNetworkHelpers:
    """Test neural network helper functions and layers."""
    
    def test_silu_activation(self, device):
        """Test SiLU activation function."""
        x = torch.randn(4, 8, device=device)
        
        # Test custom implementation
        output_custom = silu(x)
        
        # Test against PyTorch implementation
        output_pytorch = F.silu(x)
        
        assert torch.allclose(output_custom, output_pytorch, atol=1e-6)
    
    def test_gelu_tanh_activation(self, device):
        """Test GELU with tanh approximation."""
        x = torch.randn(4, 8, device=device)
        
        output = gelu_tanh(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        
        # Test against PyTorch GELU (should be close but not exact)
        output_pytorch = F.gelu(x)
        assert torch.allclose(output, output_pytorch, atol=1e-2)
    
    def test_rms_norm(self, device):
        """Test RMS normalization."""
        dim = 256
        batch_size = 2
        seq_len = 8
        
        rms_norm = RMSNorm(dim).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        output = rms_norm(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        
        # Check that RMS is approximately 1
        rms_values = torch.sqrt(torch.mean(output**2, dim=-1))
        expected_rms = torch.ones_like(rms_values)
        assert torch.allclose(rms_values, expected_rms, atol=1e-3)
    
    def test_layer_norm(self, device):
        """Test Layer normalization."""
        dim = 256
        batch_size = 2
        seq_len = 8
        
        layer_norm = LayerNorm(dim).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        output = layer_norm(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        
        # Check that mean is approximately 0 and std is approximately 1
        means = torch.mean(output, dim=-1)
        stds = torch.std(output, dim=-1)
        
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-3)
    
    def test_ada_ln(self, device):
        """Test Adaptive Layer Normalization."""
        dim = 256
        cond_dim = 128
        batch_size = 2
        seq_len = 8
        
        ada_ln = AdaLN(dim, cond_dim).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        cond = torch.randn(batch_size, cond_dim, device=device)
        
        output = ada_ln(x, cond)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_ada_ln_with_sequence_conditioning(self, device):
        """Test AdaLN with sequence-level conditioning."""
        dim = 256
        cond_dim = 128
        batch_size = 2
        seq_len = 8
        
        ada_ln = AdaLN(dim, cond_dim).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        cond = torch.randn(batch_size, seq_len, cond_dim, device=device)
        
        output = ada_ln(x, cond)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_normalization_backward_pass(self, device):
        """Test that normalization layers work with gradients."""
        dim = 128
        
        for norm_layer in [RMSNorm(dim), LayerNorm(dim)]:
            norm_layer = norm_layer.to(device)
            
            x = torch.randn(2, 4, dim, device=device, requires_grad=True)
            output = norm_layer(x)
            
            # Compute a simple loss
            loss = output.sum()
            loss.backward()
            
            # Check that gradients exist
            assert x.grad is not None
            assert torch.isfinite(x.grad).all()


# Performance and edge case tests
class TestLayerEdgeCases:
    """Test edge cases and performance characteristics."""
    
    def test_attention_with_zero_sequence_length(self, device):
        """Test attention with empty sequences."""
        d_model = 128
        n_heads = 4
        
        mha_layer = MultiHeadAttention(d_model, n_heads).to(device)
        
        # Empty sequence
        x = torch.empty(1, 0, d_model, device=device)
        
        output, _ = mha_layer(x)
        
        assert output.shape == (1, 0, d_model)
    
    def test_mlp_with_zero_hidden_dim(self, device):
        """Test that MLP handles edge cases gracefully."""
        d_model = 128
        d_ff = 0  # Edge case
        
        # Should raise an error or handle gracefully
        try:
            mlp = MLP(d_model, d_ff).to(device)
            x = torch.randn(1, 4, d_model, device=device)
            output = mlp(x)
            # If it doesn't raise an error, check the output
            assert output.shape == x.shape
        except (ValueError, RuntimeError):
            # Expected behavior for invalid configuration
            pass
    
    def test_normalization_with_very_small_values(self, device):
        """Test normalization with very small input values."""
        dim = 64
        
        rms_norm = RMSNorm(dim, eps=1e-8).to(device)
        
        # Very small values
        x = torch.ones(1, 4, dim, device=device) * 1e-10
        output = rms_norm(x)
        
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()
    
    def test_large_sequence_length_memory(self, device):
        """Test memory usage with large sequence lengths."""
        if device.type == 'cpu':
            pytest.skip("Skipping memory test on CPU")
        
        d_model = 256
        n_heads = 8
        large_seq_len = 2048
        
        mha_layer = MultiHeadAttention(d_model, n_heads).to(device)
        
        # This should not cause OOM on reasonable GPUs
        x = torch.randn(1, large_seq_len, d_model, device=device)
        
        try:
            output, _ = mha_layer(x)
            assert output.shape == x.shape
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient GPU memory for large sequence test")
            else:
                raise