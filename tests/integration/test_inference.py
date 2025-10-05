"""
Integration tests for inference and rollout components.
Tests complete inference pipelines and performance characteristics.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time

# Import inference modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference.rollout import StreamingRollout, MemoryBank, InferenceConfig, RolloutEvaluator
from models.vae3d import VAE3D
from models.dit_student import DiTStudent
from config.model_config import get_small_config


class TestMemoryBank:
    """Test memory bank functionality in inference context."""
    
    def test_memory_bank_initialization(self, device):
        """Test memory bank initialization."""
        d_model = 256
        max_memory = 1000
        
        memory_bank = MemoryBank(
            d_model=d_model,
            max_spatial_memory=max_memory
        )
        
        assert memory_bank.d_model == d_model
        assert memory_bank.max_spatial_memory == max_memory
        assert len(memory_bank.spatial_memory) == 0
        assert len(memory_bank.object_memory) == 0
    
    def test_memory_bank_update(self, device):
        """Test memory bank update functionality."""
        d_model = 256
        memory_bank = MemoryBank(d_model=d_model)
        
        # Add some frames
        for i in range(10):
            frame = torch.randn(2, 3, 32, 32, device=device)
            memory_bank.update(frame, frame_idx=i)
        
        assert len(memory_bank.spatial_memory) == 10
        assert len(memory_bank.memory_scores) == 10
        assert memory_bank.frame_count == 10
    
    def test_memory_bank_retrieval(self, device):
        """Test memory bank retrieval functionality."""
        d_model = 256
        memory_bank = MemoryBank(d_model=d_model)
        
        # Add frames with varying importance
        for i in range(20):
            # Create frames with different variance (importance)
            if i % 5 == 0:
                frame = torch.randn(1, 3, 32, 32, device=device) * 2  # Higher variance
            else:
                frame = torch.randn(1, 3, 32, 32, device=device) * 0.5  # Lower variance
            
            memory_bank.update(frame, frame_idx=i)
        
        # Retrieve memories
        memory_tokens = memory_bank.get_memory_tokens(top_k=10)
        
        assert memory_tokens.shape[0] == 1  # Batch dimension
        assert memory_tokens.shape[1] <= 10  # Top-k
        assert memory_tokens.shape[2] == d_model  # Feature dimension
        assert torch.isfinite(memory_tokens).all()
    
    def test_memory_bank_capacity_limit(self, device):
        """Test memory bank capacity limits."""
        max_memory = 50
        memory_bank = MemoryBank(max_spatial_memory=max_memory)
        
        # Add more frames than capacity
        for i in range(100):
            frame = torch.randn(1, 3, 16, 16, device=device)
            memory_bank.update(frame, frame_idx=i)
        
        # Should not exceed capacity
        assert len(memory_bank.spatial_memory) <= max_memory
        assert len(memory_bank.memory_scores) <= max_memory
    
    def test_memory_bank_clear(self, device):
        """Test memory bank clearing."""
        memory_bank = MemoryBank()
        
        # Add some data
        for i in range(5):
            frame = torch.randn(1, 3, 16, 16, device=device)
            memory_bank.update(frame, frame_idx=i)
        
        assert len(memory_bank.spatial_memory) > 0
        
        # Clear
        memory_bank.clear()
        
        assert len(memory_bank.spatial_memory) == 0
        assert len(memory_bank.object_memory) == 0
        assert len(memory_bank.memory_scores) == 0
        assert memory_bank.frame_count == 0


class TestInferenceConfig:
    """Test inference configuration."""
    
    def test_inference_config_defaults(self):
        """Test default inference configuration."""
        config = InferenceConfig()
        
        assert config.max_sequence_length > 0
        assert config.context_window > 0
        assert config.temperature > 0
        assert isinstance(config.use_kv_cache, bool)
        assert isinstance(config.mixed_precision, bool)
    
    def test_inference_config_custom(self):
        """Test custom inference configuration."""
        config = InferenceConfig(
            max_sequence_length=500,
            context_window=16,
            temperature=0.8,
            use_kv_cache=False
        )
        
        assert config.max_sequence_length == 500
        assert config.context_window == 16
        assert config.temperature == 0.8
        assert config.use_kv_cache == False


class TestStreamingRollout:
    """Test streaming rollout functionality."""
    
    @pytest.fixture
    def rollout_setup(self, device):
        """Setup for rollout tests."""
        model_config = get_small_config()
        
        # Create models with very small sizes for testing
        world_model = DiTStudent(
            latent_dim=model_config.student.latent_dim,
            d_model=64,  # Very small
            n_layers=2,
            n_heads=4,
            max_seq_len=64
        ).to(device)
        
        vae_model = VAE3D(
            in_channels=model_config.vae.in_channels,
            latent_dim=model_config.vae.latent_dim,
            hidden_dims=[16, 32],  # Very small
            num_res_blocks=1
        ).to(device)
        
        config = InferenceConfig(
            max_sequence_length=32,
            context_window=4,
            temperature=0.8,
            use_kv_cache=True,
            mixed_precision=False  # Disable for testing stability
        )
        
        rollout = StreamingRollout(world_model, vae_model, config, device)
        
        return rollout, world_model, vae_model, config
    
    def test_streaming_rollout_initialization(self, rollout_setup):
        """Test streaming rollout initialization."""
        rollout, world_model, vae_model, config = rollout_setup
        
        assert rollout.world_model is not None
        assert rollout.vae_model is not None
        assert rollout.config == config
        assert rollout.memory_bank is not None
        assert isinstance(rollout.frame_times, type(rollout.frame_times))
    
    def test_streaming_rollout_reset_state(self, rollout_setup):
        """Test state reset functionality."""
        rollout, _, _, _ = rollout_setup
        
        # Add some state
        rollout.frame_times.append(0.1)
        rollout.memory_usage.append(100.0)
        rollout.kv_cache = {'dummy': 'cache'}
        
        # Reset
        rollout.reset_state()
        
        assert len(rollout.frame_times) == 0
        assert len(rollout.memory_usage) == 0
        assert rollout.kv_cache is None
        assert rollout.cache_position == 0
    
    def test_streaming_rollout_generate_sequence(self, rollout_setup, device):
        """Test sequence generation."""
        rollout, _, _, _ = rollout_setup
        
        # Test data
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 8, 4, device=device)
        
        # Generate sequence
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=4,
                return_intermediates=False
            )
        
        # Check results
        assert 'context_frames' in result
        assert 'generated_frames' in result
        assert 'full_sequence' in result
        assert 'metadata' in result
        assert 'performance' in result
        
        # Check shapes
        assert result['context_frames'].shape == context_frames.shape
        assert result['generated_frames'].shape[0] == B
        assert result['generated_frames'].shape[1] == 4  # max_new_frames
        assert result['full_sequence'].shape[1] == T_ctx + 4
        
        # Check performance metrics
        assert 'avg_fps' in result['performance']
        assert 'peak_memory_mb' in result['performance']
        assert result['performance']['avg_fps'] >= 0
    
    def test_streaming_rollout_with_kv_cache(self, rollout_setup, device):
        """Test rollout with KV caching."""
        rollout, _, _, _ = rollout_setup
        rollout.config.use_kv_cache = True
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 6, 4, device=device)
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=3
            )
        
        # Should complete successfully with caching
        assert result['generated_frames'].shape[1] == 3
        assert torch.isfinite(result['generated_frames']).all()
    
    def test_streaming_rollout_without_kv_cache(self, rollout_setup, device):
        """Test rollout without KV caching."""
        rollout, _, _, _ = rollout_setup
        rollout.config.use_kv_cache = False
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 6, 4, device=device)
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=3
            )
        
        # Should complete successfully without caching
        assert result['generated_frames'].shape[1] == 3
        assert torch.isfinite(result['generated_frames']).all()
    
    def test_streaming_rollout_temperature_sampling(self, rollout_setup, device):
        """Test temperature sampling effects."""
        rollout, _, _, _ = rollout_setup
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 6, 4, device=device)
        
        # Test different temperatures
        temperatures = [0.1, 1.0, 2.0]
        results = []
        
        for temp in temperatures:
            rollout.config.temperature = temp
            
            with torch.no_grad():
                result = rollout.generate_sequence(
                    context_frames=context_frames,
                    control_sequence=control_sequence,
                    max_new_frames=2
                )
            
            results.append(result['generated_frames'])
        
        # Different temperatures should give different results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not torch.allclose(results[i], results[j], atol=1e-3)
    
    def test_streaming_rollout_memory_management(self, rollout_setup, device):
        """Test memory management during rollout."""
        rollout, _, _, _ = rollout_setup
        rollout.config.memory_offload_freq = 2  # Frequent memory management
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 10, 4, device=device)
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=6  # Trigger memory management
            )
        
        # Should complete successfully
        assert result['generated_frames'].shape[1] == 6
        assert torch.isfinite(result['generated_frames']).all()
    
    def test_streaming_rollout_empty_generation(self, rollout_setup, device):
        """Test rollout with zero new frames."""
        rollout, _, _, _ = rollout_setup
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 2, 4, device=device)
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=0
            )
        
        # Should handle empty generation gracefully
        assert result['generated_frames'].shape[1] == 0
        assert result['full_sequence'].shape == context_frames.shape


class TestRolloutEvaluator:
    """Test rollout evaluation functionality."""
    
    def test_rollout_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RolloutEvaluator()
        
        assert hasattr(evaluator, 'metrics')
        assert len(evaluator.metrics) == 0
    
    def test_rollout_evaluator_sequence_evaluation(self, device):
        """Test sequence evaluation."""
        evaluator = RolloutEvaluator()
        
        B, T, C, H, W = 2, 4, 3, 32, 32
        generated_frames = torch.randn(B, T, C, H, W, device=device)
        ground_truth_frames = torch.randn(B, T, C, H, W, device=device)
        
        metrics = evaluator.evaluate_sequence(generated_frames, ground_truth_frames)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'lpips_proxy' in metrics
        
        # Check metric values
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, float)
            assert metric_value >= 0
            assert not torch.isnan(torch.tensor(metric_value))
    
    def test_rollout_evaluator_temporal_consistency(self, device):
        """Test temporal consistency evaluation."""
        evaluator = RolloutEvaluator()
        
        B, T, C, H, W = 1, 6, 3, 16, 16
        
        # Create temporally consistent sequences
        base_frame = torch.randn(B, 1, C, H, W, device=device)
        noise_scale = 0.1
        
        generated_frames = []
        ground_truth_frames = []
        
        for t in range(T):
            # Add small temporal variations
            gen_frame = base_frame + torch.randn_like(base_frame) * noise_scale
            gt_frame = base_frame + torch.randn_like(base_frame) * noise_scale * 0.5
            
            generated_frames.append(gen_frame)
            ground_truth_frames.append(gt_frame)
        
        generated_seq = torch.cat(generated_frames, dim=1)
        gt_seq = torch.cat(ground_truth_frames, dim=1)
        
        metrics = evaluator.evaluate_sequence(generated_seq, gt_seq)
        
        assert 'temporal_consistency' in metrics
        assert metrics['temporal_consistency'] >= 0
    
    def test_rollout_evaluator_gradient_computation(self, device):
        """Test gradient-based evaluation."""
        evaluator = RolloutEvaluator()
        
        # Create frames with known gradients
        B, T, C, H, W = 1, 2, 3, 16, 16
        
        # Create a simple gradient pattern
        x_coords = torch.linspace(0, 1, W, device=device)
        y_coords = torch.linspace(0, 1, H, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Linear gradient in x direction
        gradient_frame = X.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, T, C, 1, 1)
        
        # Test gradient computation
        gradients = evaluator._compute_gradients(gradient_frame)
        
        assert gradients.shape == gradient_frame.shape
        assert torch.isfinite(gradients).all()
        
        # X gradients should be roughly constant
        grad_x_mean = gradients[:, :, :, :, 1:].mean()  # Exclude boundary
        assert abs(grad_x_mean) > 0.01  # Should have non-zero gradient


class TestInferenceIntegration:
    """Test complete inference pipeline integration."""
    
    def test_vae_dit_inference_pipeline(self, device):
        """Test complete VAE + DiT inference pipeline."""
        config = get_small_config()
        
        # Create models
        vae = VAE3D(
            in_channels=config.vae.in_channels,
            latent_dim=config.vae.latent_dim,
            hidden_dims=[16, 32],
            num_res_blocks=1
        ).to(device)
        
        world_model = DiTStudent(
            latent_dim=config.student.latent_dim,
            d_model=64,
            n_layers=2,
            n_heads=4
        ).to(device)
        
        # Test pipeline
        B, T, C, H, W = 1, 3, 3, 16, 16
        input_frames = torch.randn(B, T, C, H, W, device=device)
        
        with torch.no_grad():
            # 1. Encode frames
            frames_reshaped = input_frames.permute(0, 2, 1, 3, 4)
            mean, logvar = vae.encode(frames_reshaped)
            latents = vae.reparameterize(mean, logvar)
            latents = latents.permute(0, 2, 1, 3, 4)
            
            # 2. Process with DiT
            B_lat, T_lat, C_lat, H_lat, W_lat = latents.shape
            tokens = latents.view(B_lat, T_lat, -1)
            processed_tokens, _ = world_model(tokens)
            processed_latents = processed_tokens.view(B_lat, T_lat, C_lat, H_lat, W_lat)
            
            # 3. Decode back
            processed_reshaped = processed_latents.permute(0, 2, 1, 3, 4)
            output_frames = vae.decode(processed_reshaped)
            output_frames = output_frames.permute(0, 2, 1, 3, 4)
        
        # Check pipeline
        assert output_frames.shape == input_frames.shape
        assert torch.isfinite(output_frames).all()
    
    def test_inference_with_control_signals(self, rollout_setup, device):
        """Test inference with control signal integration."""
        rollout, _, _, _ = rollout_setup
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        
        # Create structured control signals
        num_future_frames = 4
        control_sequence = torch.zeros(B, T_ctx + num_future_frames, 4, device=device)
        
        # Add some control variation
        control_sequence[:, :, 0] = torch.sin(torch.linspace(0, 3.14, T_ctx + num_future_frames))  # Steering
        control_sequence[:, :, 1] = 0.5  # Constant acceleration
        control_sequence[:, :, 2:] = torch.randn(B, T_ctx + num_future_frames, 2, device=device) * 0.1  # Goal positions
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=num_future_frames
            )
        
        # Should complete with control integration
        assert result['generated_frames'].shape[1] == num_future_frames
        assert torch.isfinite(result['generated_frames']).all()
    
    def test_inference_performance_tracking(self, rollout_setup, device):
        """Test performance tracking during inference."""
        rollout, _, _, _ = rollout_setup
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 8, 4, device=device)
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=4
            )
        
        # Check performance metadata
        metadata = result['metadata']
        performance = result['performance']
        
        assert 'frame_times' in metadata
        assert 'memory_usage' in metadata
        assert 'cache_sizes' in metadata
        
        assert len(metadata['frame_times']) == 4  # One per generated frame
        assert len(metadata['memory_usage']) == 4
        assert len(metadata['cache_sizes']) == 4
        
        assert 'avg_fps' in performance
        assert 'peak_memory_mb' in performance
        assert 'avg_memory_mb' in performance
        
        # Check reasonable values
        assert performance['avg_fps'] > 0
        assert performance['peak_memory_mb'] >= 0
        assert performance['avg_memory_mb'] >= 0


# Performance and stress tests
class TestInferencePerformance:
    """Test inference performance characteristics."""
    
    @pytest.mark.performance
    def test_rollout_speed(self, rollout_setup, device, performance_timer):
        """Test rollout generation speed."""
        rollout, _, _, _ = rollout_setup
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 8, 4, device=device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(2):
                result = rollout.generate_sequence(
                    context_frames=context_frames,
                    control_sequence=control_sequence,
                    max_new_frames=2
                )
        
        # Benchmark
        num_iterations = 5
        performance_timer.start()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                result = rollout.generate_sequence(
                    context_frames=context_frames,
                    control_sequence=control_sequence,
                    max_new_frames=2
                )
        
        elapsed_time = performance_timer.stop()
        avg_time = elapsed_time / num_iterations
        
        # Should be reasonably fast
        assert avg_time < 1.0, f"Rollout too slow: {avg_time:.3f}s per sequence"
    
    @pytest.mark.performance
    def test_memory_efficiency(self, rollout_setup, device):
        """Test memory efficiency during long sequences."""
        if device.type == 'cpu':
            pytest.skip("Memory test requires GPU")
        
        rollout, _, _, _ = rollout_setup
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 20, 4, device=device)
        
        # Measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Generate longer sequence
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=15  # Longer sequence
            )
        
        # Check memory growth
        final_memory = torch.cuda.memory_allocated(device)
        memory_growth = final_memory - initial_memory
        
        # Should not grow excessively
        max_allowed_growth = 100 * 1024 * 1024  # 100MB
        assert memory_growth < max_allowed_growth, f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB"
        
        # Check that sequence was generated successfully
        assert result['generated_frames'].shape[1] == 15
        assert torch.isfinite(result['generated_frames']).all()
    
    @pytest.mark.slow
    def test_long_sequence_generation(self, rollout_setup, device):
        """Test generation of very long sequences."""
        rollout, _, _, _ = rollout_setup
        rollout.config.max_sequence_length = 100  # Allow longer sequences
        
        B, T_ctx, C, H, W = 1, 2, 3, 16, 16
        context_frames = torch.randn(B, T_ctx, C, H, W, device=device)
        control_sequence = torch.randn(B, 50, 4, device=device)
        
        with torch.no_grad():
            result = rollout.generate_sequence(
                context_frames=context_frames,
                control_sequence=control_sequence,
                max_new_frames=30  # Long sequence
            )
        
        # Should complete successfully
        assert result['generated_frames'].shape[1] == 30
        assert torch.isfinite(result['generated_frames']).all()
        
        # Performance should still be reasonable
        assert result['performance']['avg_fps'] > 0.1  # At least 0.1 FPS