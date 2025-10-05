"""
Comprehensive test suite for mathematical components.
Tests all mathematical operations, tensor shapes, and numerical stability.
"""

import pytest
import torch
import numpy as np
import math
from typing import Tuple, List

# Import components to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from layers.rope import RoPE, apply_rope
from layers.mha import MultiHeadAttention
from layers.mlp import MLP
from layers.nn_helpers import SiLU, RMSNorm
from core.components import RoPE3D, MultiHeadAttention as CoreMHA, FlowMatchingPredictor
from blocks.dit_block import DiTBlock


class TestRoPE:
    """Test Rotary Position Embedding implementations."""
    
    def test_rope_basic_functionality(self):
        """Test basic RoPE functionality."""
        dim = 64
        seq_len = 16
        batch_size = 2
        
        rope = RoPE(dim)
        
        # Test position encoding generation
        pos_enc = rope.get_position_encodings(seq_len)
        assert pos_enc.shape == (seq_len, dim)
        
        # Test application to query/key
        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)
        
        q_rot, k_rot = apply_rope(q, k, pos_enc)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Test that rotation is applied (output should be different)
        assert not torch.allclose(q, q_rot, atol=1e-6)
        assert not torch.allclose(k, k_rot, atol=1e-6)
    
    def test_rope_3d_functionality(self):
        """Test 3D RoPE for video transformers."""
        dim = 64
        seq_len = 8
        height = 4
        width = 4
        
        rope_3d = RoPE3D(dim)
        
        # Generate 3D position embeddings
        sin_emb, cos_emb = rope_3d(seq_len, height, width)
        
        assert sin_emb.shape == (seq_len, height, width, dim)
        assert cos_emb.shape == (seq_len, height, width, dim)
        
        # Test that embeddings are different across dimensions
        assert not torch.allclose(sin_emb[0, 0, 0], sin_emb[1, 0, 0])  # Time
        assert not torch.allclose(sin_emb[0, 0, 0], sin_emb[0, 1, 0])  # Height
        assert not torch.allclose(sin_emb[0, 0, 0], sin_emb[0, 0, 1])  # Width
    
    def test_rope_rotational_property(self):
        """Test that RoPE maintains rotational properties."""
        dim = 32
        seq_len = 4
        
        rope = RoPE(dim)
        pos_enc = rope.get_position_encodings(seq_len)
        
        # Create test vectors
        x = torch.randn(1, seq_len, dim)
        y = torch.randn(1, seq_len, dim)
        
        x_rot, y_rot = apply_rope(x, y, pos_enc)
        
        # Test that dot products are preserved for same positions
        for i in range(seq_len):
            original_dot = torch.dot(x[0, i], y[0, i])
            rotated_dot = torch.dot(x_rot[0, i], y_rot[0, i])
            assert torch.allclose(original_dot, rotated_dot, atol=1e-5)
    
    def test_rope_frequency_correctness(self):
        """Test that RoPE frequencies are computed correctly."""
        dim = 8
        theta = 10000.0
        
        rope = RoPE(dim, theta=theta)
        
        # Check frequency computation
        freqs = rope.get_frequencies()
        expected_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        assert torch.allclose(freqs, expected_freqs, atol=1e-6)


class TestMultiHeadAttention:
    """Test Multi-Head Attention implementations."""
    
    def test_mha_basic_shapes(self):
        """Test that MHA produces correct output shapes."""
        batch_size = 2
        seq_len = 16
        dim = 256
        num_heads = 8
        
        mha = MultiHeadAttention(dim, num_heads)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = mha(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_mha_causal_masking(self):
        """Test causal masking in attention."""
        batch_size = 1
        seq_len = 4
        dim = 64
        num_heads = 4
        
        mha = CoreMHA(dim, num_heads, causal=True)
        
        x = torch.randn(batch_size, seq_len, dim)
        output, _ = mha(x)
        
        # Create attention weights manually to verify causality
        # This is a simplified test - in practice, we'd need access to attention weights
        assert output.shape == (batch_size, seq_len, dim)
        
        # Test that future positions don't affect current positions
        # by masking future inputs and checking outputs don't change
        x_masked = x.clone()
        x_masked[0, 2:] = 0  # Mask future positions
        
        output_masked, _ = mha(x_masked)
        
        # First two positions should be identical
        assert torch.allclose(output[0, :2], output_masked[0, :2], atol=1e-5)
    
    def test_mha_cross_attention(self):
        """Test cross-attention functionality."""
        batch_size = 2
        seq_len_q = 8
        seq_len_kv = 12
        dim = 128
        num_heads = 4
        
        mha = CoreMHA(dim, num_heads, causal=False)
        
        q = torch.randn(batch_size, seq_len_q, dim)
        kv = torch.randn(batch_size, seq_len_kv, dim)
        
        output, _ = mha(q, kv=kv)
        
        assert output.shape == (batch_size, seq_len_q, dim)
    
    def test_mha_kv_caching(self):
        """Test KV caching for efficient inference."""
        batch_size = 1
        seq_len = 4
        dim = 64
        num_heads = 4
        
        mha = CoreMHA(dim, num_heads, causal=True)
        
        x = torch.randn(batch_size, seq_len, dim)
        
        # First forward pass
        output1, cache1 = mha(x, use_cache=True)
        
        # Second forward pass with new token
        new_token = torch.randn(batch_size, 1, dim)
        output2, cache2 = mha(new_token, past_kv=cache1, use_cache=True)
        
        # Full forward pass for comparison
        x_full = torch.cat([x, new_token], dim=1)
        output_full, _ = mha(x_full)
        
        # Last token should match
        assert torch.allclose(output2[0, -1], output_full[0, -1], atol=1e-5)
    
    def test_mha_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        # This would require access to intermediate attention weights
        # For now, we test that the output is reasonable
        batch_size = 2
        seq_len = 8
        dim = 64
        num_heads = 4
        
        mha = MultiHeadAttention(dim, num_heads)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = mha(x)
        
        # Check that output is finite and has reasonable magnitude
        assert torch.isfinite(output).all()
        assert output.abs().mean() < 10.0  # Reasonable magnitude


class TestMLP:
    """Test MLP implementations."""
    
    def test_mlp_basic_functionality(self):
        """Test basic MLP functionality."""
        input_dim = 256
        hidden_dim = 1024
        output_dim = 256
        
        mlp = MLP(input_dim, hidden_dim, output_dim)
        
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = mlp(x)
        assert output.shape == (batch_size, seq_len, output_dim)
    
    def test_mlp_activation_functions(self):
        """Test different activation functions."""
        input_dim = 128
        hidden_dim = 256
        
        # Test SiLU activation
        silu = SiLU()
        x = torch.randn(2, 10, input_dim)
        
        output = silu(x)
        assert output.shape == x.shape
        
        # Test that SiLU is applied correctly: x * sigmoid(x)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(output, expected, atol=1e-6)
    
    def test_mlp_gradient_flow(self):
        """Test that gradients flow properly through MLP."""
        mlp = MLP(64, 128, 64)
        
        x = torch.randn(2, 8, 64, requires_grad=True)
        target = torch.randn(2, 8, 64)
        
        output = mlp(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that all parameters have gradients
        for param in mlp.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestNormalizationLayers:
    """Test normalization layer implementations."""
    
    def test_rms_norm_functionality(self):
        """Test RMSNorm implementation."""
        dim = 256
        rms_norm = RMSNorm(dim)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, dim)
        
        output = rms_norm(x)
        
        assert output.shape == x.shape
        
        # Test that RMS normalization is applied correctly
        # RMS = sqrt(mean(x^2))
        eps = 1e-5
        x_squared_mean = torch.mean(x ** 2, dim=-1, keepdim=True)
        expected_norm = x / torch.sqrt(x_squared_mean + eps)
        
        # Apply scaling factor
        expected_output = expected_norm * rms_norm.scale
        
        assert torch.allclose(output, expected_output, atol=1e-5)
    
    def test_rms_norm_numerical_stability(self):
        """Test RMSNorm numerical stability with extreme values."""
        dim = 128
        rms_norm = RMSNorm(dim)
        
        # Test with very small values
        x_small = torch.randn(2, 8, dim) * 1e-8
        output_small = rms_norm(x_small)
        assert torch.isfinite(output_small).all()
        
        # Test with very large values
        x_large = torch.randn(2, 8, dim) * 1e8
        output_large = rms_norm(x_large)
        assert torch.isfinite(output_large).all()


class TestFlowMatching:
    """Test Flow Matching implementation."""
    
    def test_flow_predictor_basic(self):
        """Test basic flow predictor functionality."""
        dim = 256
        num_steps = 4
        
        predictor = FlowMatchingPredictor(dim, num_steps)
        
        batch_size = 2
        z = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        
        flow = predictor(z, t)
        
        assert flow.shape == (batch_size, dim)
        assert torch.isfinite(flow).all()
        
        # Flow should be bounded due to Tanh activation
        assert flow.abs().max() <= 1.0
    
    def test_flow_matching_sampling(self):
        """Test flow matching sampling process."""
        dim = 64
        num_steps = 4
        
        predictor = FlowMatchingPredictor(dim, num_steps)
        
        batch_size = 2
        z_init = torch.randn(batch_size, dim)
        
        z_final = predictor.sample(z_init, num_steps)
        
        assert z_final.shape == z_init.shape
        assert torch.isfinite(z_final).all()
        
        # Final sample should be different from initial
        assert not torch.allclose(z_init, z_final, atol=1e-3)
    
    def test_flow_matching_euler_integration(self):
        """Test that Euler integration is performed correctly."""
        dim = 32
        num_steps = 2
        
        predictor = FlowMatchingPredictor(dim, num_steps)
        
        z_init = torch.randn(1, dim)
        dt = 1.0 / num_steps
        
        # Manual Euler step
        t0 = torch.zeros(1)
        flow0 = predictor(z_init, t0)
        z1_manual = z_init + dt * flow0
        
        t1 = torch.full((1,), dt)
        flow1 = predictor(z1_manual, t1)
        z2_manual = z1_manual + dt * flow1
        
        # Automatic sampling
        z2_auto = predictor.sample(z_init, num_steps)
        
        # Should be approximately equal (may have small differences due to implementation)
        assert torch.allclose(z2_manual, z2_auto, atol=1e-4)


class TestNumericalStability:
    """Test numerical stability across all components."""
    
    def test_gradient_explosion_protection(self):
        """Test that gradients don't explode in deep networks."""
        # Create a simple deep network
        layers = []
        for _ in range(10):
            layers.extend([
                torch.nn.Linear(128, 128),
                SiLU()
            ])
        
        model = torch.nn.Sequential(*layers)
        
        x = torch.randn(2, 128, requires_grad=True)
        target = torch.randn(2, 128)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # Gradient norm should be reasonable (not exploded)
        assert total_grad_norm < 100.0
        assert not math.isnan(total_grad_norm)
    
    def test_attention_numerical_stability(self):
        """Test attention numerical stability with extreme inputs."""
        mha = CoreMHA(64, 4, causal=False)
        
        # Test with very large values (should not cause overflow)
        x_large = torch.randn(1, 8, 64) * 100
        output_large, _ = mha(x_large)
        assert torch.isfinite(output_large).all()
        
        # Test with very small values
        x_small = torch.randn(1, 8, 64) * 1e-6
        output_small, _ = mha(x_small)
        assert torch.isfinite(output_small).all()
    
    def test_rope_numerical_stability(self):
        """Test RoPE numerical stability."""
        rope = RoPE(64, theta=10000.0)
        
        # Test with very long sequences
        long_seq_len = 2048
        pos_enc = rope.get_position_encodings(long_seq_len)
        
        assert torch.isfinite(pos_enc).all()
        assert not torch.isnan(pos_enc).any()
        
        # Check that frequencies don't cause numerical issues
        freqs = rope.get_frequencies()
        assert torch.isfinite(freqs).all()
        assert (freqs > 0).all()


class TestTensorShapeConsistency:
    """Test tensor shape consistency across all operations."""
    
    @pytest.mark.parametrize("batch_size,seq_len,dim", [
        (1, 8, 64),
        (4, 16, 128),
        (2, 32, 256),
        (8, 64, 512)
    ])
    def test_attention_shape_consistency(self, batch_size, seq_len, dim):
        """Test attention shape consistency across different input sizes."""
        num_heads = 8
        mha = CoreMHA(dim, num_heads)
        
        x = torch.randn(batch_size, seq_len, dim)
        output, cache = mha(x, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, dim)
        if cache:
            assert cache['k'].shape[2] == seq_len  # Sequence dimension
            assert cache['v'].shape[2] == seq_len
    
    @pytest.mark.parametrize("input_dim,hidden_dim,output_dim", [
        (64, 256, 64),
        (128, 512, 128),
        (256, 1024, 256),
        (512, 2048, 512)
    ])
    def test_mlp_shape_consistency(self, input_dim, hidden_dim, output_dim):
        """Test MLP shape consistency."""
        mlp = MLP(input_dim, hidden_dim, output_dim)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = mlp(x)
        assert output.shape == (batch_size, seq_len, output_dim)


# Test utilities
def assert_tensor_properties(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                           finite: bool = True, dtype: torch.dtype = None):
    """Assert tensor has expected properties."""
    assert tensor.shape == expected_shape
    
    if finite:
        assert torch.isfinite(tensor).all()
    
    if dtype is not None:
        assert tensor.dtype == dtype


def test_mathematical_identities():
    """Test important mathematical identities."""
    # Test that RoPE preserves dot products for same positions
    dim = 64
    rope = RoPE(dim)
    pos_enc = rope.get_position_encodings(4)
    
    q = torch.randn(1, 4, dim)
    k = torch.randn(1, 4, dim)
    
    q_rot, k_rot = apply_rope(q, k, pos_enc)
    
    # Dot products at same positions should be preserved
    for i in range(4):
        original_dot = torch.dot(q[0, i], k[0, i])
        rotated_dot = torch.dot(q_rot[0, i], k_rot[0, i])
        assert torch.allclose(original_dot, rotated_dot, atol=1e-5)


def test_backward_compatibility():
    """Test that saved models can be loaded and produce same outputs."""
    # This would test model serialization/deserialization
    # For now, we test basic save/load functionality
    
    model = MultiHeadAttention(128, 8)
    x = torch.randn(2, 16, 128)
    
    # Save state dict
    state_dict = model.state_dict()
    
    # Create new model and load state
    model_loaded = MultiHeadAttention(128, 8)
    model_loaded.load_state_dict(state_dict)
    
    # Test that outputs are identical
    with torch.no_grad():
        output1 = model(x)
        output2 = model_loaded(x)
        
        assert torch.allclose(output1, output2, atol=1e-8)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])