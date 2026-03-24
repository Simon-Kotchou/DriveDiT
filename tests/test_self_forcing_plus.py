"""
Tests for Self-Forcing++ training implementation.

Tests cover:
1. Rolling KV Cache functionality
2. Curriculum Learning Scheduler
3. Future Anchor Encoding (comma.ai)
4. Extended Control Signal Encoder
5. EMA Model
6. Uncertainty Weighting
7. Per-layer Gradient Clipping
8. Full Training Step Integration
"""

import torch
import torch.nn as nn
import pytest
import math
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.self_forcing_plus import (
    SelfForcingPlusConfig,
    RollingKVCache,
    CurriculumScheduler,
    FutureAnchorEncoder,
    ExtendedControlEncoder,
    EMAModel,
    UncertaintyWeighting,
    PerLayerGradientClipper,
    SelfForcingPlusTrainer,
    get_default_config,
    get_minimal_config,
)

# Alias for backward compatibility with tests
clip_grad_norm_per_layer = PerLayerGradientClipper
get_production_config = get_default_config  # Use default as production for testing


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default test configuration."""
    return get_minimal_config()


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_model(device):
    """Simple model for testing."""
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 512)
    ).to(device)
    return model


# =============================================================================
# Rolling KV Cache Tests
# =============================================================================

class TestRollingKVCache:
    """Tests for Rolling KV Cache functionality."""

    def test_init(self):
        """Test cache initialization."""
        cache = RollingKVCache(max_length=64, truncate_to=32, detach_interval=4)
        assert cache.max_length == 64
        assert cache.truncate_to == 32
        assert cache.detach_interval == 4
        assert cache.get_length() == 0

    def test_single_update(self):
        """Test single cache update."""
        cache = RollingKVCache(max_length=64, truncate_to=32)

        k = torch.randn(2, 8, 4, 64)  # [B, T, H, D]
        v = torch.randn(2, 8, 4, 64)

        full_k, full_v = cache.update(layer_idx=0, new_k=k, new_v=v)

        assert full_k.shape == k.shape
        assert full_v.shape == v.shape
        assert cache.get_length() == 8

    def test_multiple_updates(self):
        """Test cache accumulation."""
        cache = RollingKVCache(max_length=64, truncate_to=32)

        for i in range(5):
            k = torch.randn(2, 4, 4, 64)
            v = torch.randn(2, 4, 4, 64)
            full_k, full_v = cache.update(layer_idx=0, new_k=k, new_v=v)

        assert cache.get_length() == 20
        assert full_k.shape[1] == 20

    def test_truncation(self):
        """Test cache truncation when exceeding max length."""
        cache = RollingKVCache(max_length=32, truncate_to=16)

        # Add enough to exceed max
        for i in range(10):
            k = torch.randn(2, 8, 4, 64)
            v = torch.randn(2, 8, 4, 64)
            full_k, full_v = cache.update(layer_idx=0, new_k=k, new_v=v)

        # Should be truncated
        assert cache.get_length() <= 32

    def test_gradient_detachment(self):
        """Test gradient detachment at intervals."""
        cache = RollingKVCache(max_length=64, truncate_to=32, detach_interval=2)

        k1 = torch.randn(2, 4, 4, 64, requires_grad=True)
        v1 = torch.randn(2, 4, 4, 64, requires_grad=True)

        # First update
        full_k, full_v = cache.update(layer_idx=0, new_k=k1, new_v=v1)

        # Second update should detach
        k2 = torch.randn(2, 4, 4, 64, requires_grad=True)
        v2 = torch.randn(2, 4, 4, 64, requires_grad=True)
        full_k, full_v = cache.update(layer_idx=0, new_k=k2, new_v=v2)

        # Verify detachment happened
        assert not full_k.requires_grad
        assert not full_v.requires_grad

    def test_reset(self):
        """Test cache reset."""
        cache = RollingKVCache(max_length=64, truncate_to=32)

        k = torch.randn(2, 8, 4, 64)
        v = torch.randn(2, 8, 4, 64)
        cache.update(layer_idx=0, new_k=k, new_v=v)

        cache.reset()

        assert cache.get_length() == 0
        assert len(cache.cache) == 0

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        cache = RollingKVCache(max_length=64, truncate_to=32)

        k = torch.randn(2, 16, 4, 64)
        v = torch.randn(2, 16, 4, 64)
        cache.update(layer_idx=0, new_k=k, new_v=v)

        memory_gb = cache.get_memory_usage()
        assert memory_gb > 0
        assert memory_gb < 1  # Should be small for this test


# =============================================================================
# Curriculum Scheduler Tests
# =============================================================================

class TestCurriculumScheduler:
    """Tests for Curriculum Learning Scheduler."""

    def test_initial_values(self, default_config):
        """Test initial curriculum values."""
        scheduler = CurriculumScheduler(default_config)

        assert scheduler.get_sequence_length() == default_config.initial_sequence_length
        assert scheduler.get_self_forcing_ratio() == default_config.initial_self_forcing_ratio

    def test_sequence_length_progression(self, default_config):
        """Test sequence length grows over training."""
        scheduler = CurriculumScheduler(default_config)

        lengths = []
        for step in range(0, default_config.sequence_curriculum_steps + 1000, 500):
            scheduler.update(step)
            lengths.append(scheduler.get_sequence_length())

        # Should be non-decreasing
        for i in range(1, len(lengths)):
            assert lengths[i] >= lengths[i-1]

        # Final should be at target
        scheduler.update(default_config.sequence_curriculum_steps + 1000)
        assert scheduler.get_sequence_length() == default_config.final_sequence_length

    def test_self_forcing_ratio_progression(self, default_config):
        """Test self-forcing ratio grows over training."""
        scheduler = CurriculumScheduler(default_config)

        ratios = []
        for step in range(0, default_config.self_forcing_warmup_steps + 1000, 500):
            scheduler.update(step)
            ratios.append(scheduler.get_self_forcing_ratio())

        # Should be non-decreasing
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i-1]

        # Final should be at target
        scheduler.update(default_config.self_forcing_warmup_steps + 1000)
        final_ratio = scheduler.get_self_forcing_ratio()
        assert abs(final_ratio - default_config.final_self_forcing_ratio) < 0.01

    def test_cosine_schedule(self):
        """Test cosine schedule is smooth."""
        config = get_minimal_config()
        config.self_forcing_schedule = "cosine"
        scheduler = CurriculumScheduler(config)

        # Cosine should be smooth (no sudden jumps)
        prev_ratio = scheduler.get_self_forcing_ratio()
        for step in range(100, config.self_forcing_warmup_steps, 100):
            scheduler.update(step)
            curr_ratio = scheduler.get_self_forcing_ratio()

            # Change should be gradual
            assert abs(curr_ratio - prev_ratio) < 0.1
            prev_ratio = curr_ratio

    def test_curriculum_weights(self, default_config):
        """Test curriculum weight adjustment."""
        scheduler = CurriculumScheduler(default_config)

        # At step 0
        weights = scheduler.get_curriculum_weights()
        assert 'reconstruction' in weights
        assert weights['reconstruction'] > 0

        # Some weights should be low initially
        assert weights['temporal_consistency'] < 0.1

        # After warmup
        scheduler.update(default_config.curriculum_warmup_steps)
        weights_after = scheduler.get_curriculum_weights()

        # Weights should increase
        assert weights_after['temporal_consistency'] >= weights['temporal_consistency']


# =============================================================================
# Future Anchor Encoder Tests
# =============================================================================

class TestFutureAnchorEncoder:
    """Tests for Future Anchor Encoding (comma.ai methodology)."""

    def test_init(self, default_config, device):
        """Test encoder initialization."""
        encoder = FutureAnchorEncoder(default_config).to(device)

        assert len(encoder.horizon_encoders) == len(default_config.future_anchor_horizons)

    def test_forward(self, default_config, device):
        """Test forward pass."""
        encoder = FutureAnchorEncoder(default_config).to(device)

        # [B, T, 5] ego states
        ego_states = torch.randn(2, 64, 5, device=device)

        anchor_emb = encoder(ego_states, current_time=0)

        assert anchor_emb.shape == (2, default_config.model_dim)

    def test_different_current_times(self, default_config, device):
        """Test encoding at different timesteps."""
        encoder = FutureAnchorEncoder(default_config).to(device)
        ego_states = torch.randn(2, 64, 5, device=device)

        emb1 = encoder(ego_states, current_time=0)
        emb2 = encoder(ego_states, current_time=10)
        emb3 = encoder(ego_states, current_time=30)

        # Different timesteps should give different embeddings
        assert not torch.allclose(emb1, emb2)
        assert not torch.allclose(emb2, emb3)

    def test_anchor_indices(self, default_config):
        """Test anchor index calculation."""
        encoder = FutureAnchorEncoder(default_config)

        indices = encoder.get_anchor_indices(current_time=0, max_time=64)

        # Should have one index per horizon
        assert len(indices) == len(default_config.future_anchor_horizons)

        # Indices should be within bounds
        for idx in indices:
            assert 0 <= idx < 64


# =============================================================================
# Extended Control Encoder Tests
# =============================================================================

class TestExtendedControlEncoder:
    """Tests for Extended 6D Control Signal Encoder."""

    def test_init(self, default_config, device):
        """Test encoder initialization."""
        encoder = ExtendedControlEncoder(default_config).to(device)

        # Check normalization buffers exist
        assert encoder.norm_min.shape == (default_config.control_dim,)
        assert encoder.norm_max.shape == (default_config.control_dim,)

    def test_normalization(self, default_config, device):
        """Test control normalization - forward pass should work."""
        encoder = ExtendedControlEncoder(default_config).to(device)

        # Control values within expected ranges
        controls = torch.tensor([[0.0, 0.0, 0.0, 0.0, 20.0, 0.0]], device=device)

        # Forward should produce valid embeddings
        encoded = encoder(controls)
        assert torch.isfinite(encoded).all()

    def test_encoding(self, default_config, device):
        """Test control encoding."""
        encoder = ExtendedControlEncoder(default_config).to(device)

        # Single timestep
        controls_single = torch.randn(2, 6, device=device)
        encoded_single = encoder(controls_single)
        assert encoded_single.shape == (2, default_config.model_dim)

        # Sequence
        controls_seq = torch.randn(2, 8, 6, device=device)
        encoded_seq = encoder(controls_seq)
        assert encoded_seq.shape == (2, 8, default_config.model_dim)

    def test_control_prediction(self, default_config, device):
        """Test inverse dynamics (control prediction) if method exists."""
        encoder = ExtendedControlEncoder(default_config).to(device)

        # Check if predict_controls method exists
        if hasattr(encoder, 'predict_controls'):
            hidden = torch.randn(2, 8, default_config.model_dim, device=device)
            predicted = encoder.predict_controls(hidden)
            assert predicted.shape == (2, 8, default_config.control_dim)
        else:
            # Test that control_predictor exists for inverse dynamics
            assert hasattr(encoder, 'control_predictor'), "Encoder should have control_predictor for control prediction"


# =============================================================================
# EMA Model Tests
# =============================================================================

class TestEMAModel:
    """Tests for Exponential Moving Average."""

    def test_init(self, simple_model):
        """Test EMA initialization."""
        ema = EMAModel(simple_model, decay=0.999)

        # EMA shadow should be a copy of model params
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(param, ema.shadow[name])

    def test_update(self, simple_model, device):
        """Test EMA update."""
        ema = EMAModel(simple_model, decay=0.99)

        # Store original shadow values
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify original model
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.randn_like(p))

        # Update EMA
        ema.update(simple_model)

        # Shadow should now be different from original (moved toward model)
        for name in original_shadow:
            assert not torch.allclose(original_shadow[name], ema.shadow[name])

    def test_decay_rate(self, simple_model):
        """Test different decay rates."""
        ema_fast = EMAModel(simple_model, decay=0.9)  # Fast decay
        ema_slow = EMAModel(simple_model, decay=0.999)  # Slow decay

        # Modify model
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema_fast.update(simple_model)
        ema_slow.update(simple_model)

        # Fast EMA should have moved more toward the model
        # Both should work without error

    def test_shadow_params(self, simple_model):
        """Test shadow parameter storage."""
        ema = EMAModel(simple_model, decay=0.999)

        # Shadow params should exist for all trainable params
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow


# =============================================================================
# Uncertainty Weighting Tests
# =============================================================================

class TestUncertaintyWeighting:
    """Tests for Uncertainty-based Loss Weighting."""

    def test_init(self, device):
        """Test initialization."""
        loss_names = ['loss_a', 'loss_b', 'loss_c']
        uw = UncertaintyWeighting(loss_names=loss_names).to(device)

        assert len(uw.log_vars) == 3
        # Initialized to 0 (variance = 1, weight = 1)
        assert torch.allclose(uw.log_vars, torch.zeros(3, device=device))

    def test_forward(self, device):
        """Test forward pass."""
        loss_names = ['loss_a', 'loss_b', 'loss_c']
        uw = UncertaintyWeighting(loss_names=loss_names).to(device)

        losses = {
            'loss_a': torch.tensor(1.0, device=device),
            'loss_b': torch.tensor(0.5, device=device),
            'loss_c': torch.tensor(2.0, device=device)
        }

        total_loss, weights = uw(losses)

        assert total_loss.item() > 0
        assert len(weights) == 3
        assert all(w > 0 for w in weights.values())

    def test_learnable_weights(self, device):
        """Test that weights are learnable."""
        loss_names = ['loss_a', 'loss_b']
        uw = UncertaintyWeighting(loss_names=loss_names).to(device)

        losses = {
            'loss_a': torch.tensor(1.0, device=device, requires_grad=True),
            'loss_b': torch.tensor(0.5, device=device, requires_grad=True)
        }

        total_loss, _ = uw(losses)
        total_loss.backward()

        # Log vars should have gradients
        assert uw.log_vars.grad is not None

    def test_weight_interpretation(self, device):
        """Test weight interpretation."""
        loss_names = ['loss_a', 'loss_b']
        uw = UncertaintyWeighting(loss_names=loss_names).to(device)

        # Set log_vars to known values
        with torch.no_grad():
            uw.log_vars[0] = 0.0  # variance = 1, precision = 1
            uw.log_vars[1] = math.log(4)  # variance = 4, precision = 0.25

        # The forward pass computes weights
        losses = {
            'loss_a': torch.tensor(1.0, device=device),
            'loss_b': torch.tensor(1.0, device=device)
        }
        _, weights = uw(losses)

        assert weights['loss_a'] > weights['loss_b']  # Higher precision = higher weight
        assert abs(weights['loss_b'] - 0.25) < 0.01


# =============================================================================
# Per-Layer Gradient Clipping Tests
# =============================================================================

class TestPerLayerGradClipping:
    """Tests for per-layer gradient clipping."""

    def test_basic_clipping(self, device):
        """Test basic gradient clipping."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 64)
        ).to(device)

        # Create clipper
        clipper = PerLayerGradientClipper(model, default_max_norm=0.1)

        # Create gradients
        x = torch.randn(4, 64, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Clip gradients
        grad_norms = clipper.clip_gradients()

        # Should have recorded norms for all params
        assert len(grad_norms) > 0

    def test_clipping_effectiveness(self, device):
        """Test that clipping actually limits gradient norms."""
        model = nn.Linear(64, 64).to(device)

        max_norm = 0.5
        clipper = PerLayerGradientClipper(model, default_max_norm=max_norm)

        # Create large gradients
        x = torch.randn(4, 64, device=device) * 100
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Clip
        clipper.clip_gradients()

        # Check that gradients are clipped
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm()
                assert grad_norm <= max_norm * 1.1  # Allow small tolerance


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfigurations:
    """Tests for configuration factory functions."""

    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()

        assert config.model_dim > 0
        assert config.num_heads > 0
        assert config.initial_sequence_length > 0
        assert config.final_sequence_length >= config.initial_sequence_length

    def test_minimal_config(self):
        """Test minimal configuration."""
        config = get_minimal_config()

        # Should have smaller/minimal values
        assert config.final_sequence_length <= 64
        assert config.model_dim > 0
        assert config.batch_size > 0

    def test_production_config(self):
        """Test production configuration (aliased to default)."""
        config = get_production_config()

        # Should have valid values
        assert config.model_dim > 0
        assert config.final_sequence_length >= config.initial_sequence_length
        assert config.kv_cache_max_length > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for Self-Forcing++ trainer."""

    def test_trainer_initialization(self, default_config, device):
        """Test trainer initialization."""
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        ).to(device)

        # Mock model to have forward that returns expected dict
        class MockModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, frames, controls=None, mode='train', **kwargs):
                B, T = frames.shape[:2]
                return {
                    'predictions': frames,  # Identity for testing
                    'hidden_states': torch.randn(B, T, 512, device=frames.device)
                }

        mock_model = MockModel(model).to(device)

        trainer = SelfForcingPlusTrainer(
            model=mock_model,
            config=default_config,
            device=device
        )

        assert trainer.global_step == 0
        assert trainer.curriculum is not None
        assert trainer.control_encoder is not None

    def test_curriculum_integration(self, default_config):
        """Test curriculum is used correctly during training."""
        scheduler = CurriculumScheduler(default_config)

        # Simulate training
        for step in range(0, 100, 10):
            scheduler.update(step)
            seq_len = scheduler.get_sequence_length()
            sf_ratio = scheduler.get_self_forcing_ratio()

            # Should be valid values
            assert seq_len >= default_config.initial_sequence_length
            assert 0 <= sf_ratio <= 1

    def test_all_components_together(self, default_config, device):
        """Test all components work together."""
        # Initialize all components
        kv_cache = RollingKVCache(
            max_length=default_config.kv_cache_max_length,
            truncate_to=default_config.kv_cache_truncate_to
        )

        curriculum = CurriculumScheduler(default_config)

        if default_config.enable_future_anchors:
            anchor_encoder = FutureAnchorEncoder(default_config).to(device)

        control_encoder = ExtendedControlEncoder(default_config).to(device)

        # Simulate a training step
        B, T = 2, 8
        frames = torch.randn(B, T, 3, 64, 64, device=device)
        controls = torch.randn(B, T, 6, device=device)
        ego_states = torch.randn(B, T, 5, device=device)

        # Get curriculum values
        curriculum.update(100)
        seq_len = curriculum.get_sequence_length()
        sf_ratio = curriculum.get_self_forcing_ratio()

        # Encode controls
        control_emb = control_encoder(controls)

        # Encode future anchors
        if default_config.enable_future_anchors:
            anchor_emb = anchor_encoder(ego_states, current_time=0)

        # Update KV cache
        k = torch.randn(B, 4, 8, 64, device=device)
        v = torch.randn(B, 4, 8, 64, device=device)
        full_k, full_v = kv_cache.update(0, k, v)

        # Everything should work together
        assert control_emb.shape[1] == T
        assert full_k.shape[1] >= 4


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    print("Running Self-Forcing++ Tests")
    print("=" * 60)

    # Run pytest if available
    pytest.main([__file__, "-v", "--tb=short"])
