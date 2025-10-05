"""
Unit tests for utility functions (tensor_utils, model_utils, math_utils).
Tests helper functions and mathematical operations.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Dict
import tempfile
import json

# Import utils
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.tensor_utils import (
    safe_cat, safe_stack, pad_sequence, chunk_tensor, flatten_dict_tensors,
    unflatten_dict_tensors, tensor_memory_usage, move_to_device, detach_tensors,
    interpolate_tensors, normalize_tensor
)
from utils.model_utils import (
    count_parameters, freeze_parameters, get_device, set_seed, initialize_weights,
    clip_gradients, get_gradient_norm, save_checkpoint, load_checkpoint,
    exponential_moving_average, get_model_size_mb
)
from utils.math_utils import (
    gaussian_kernel, soft_clamp, stable_softmax, log_sum_exp, gumbel_noise,
    gumbel_softmax, cosine_similarity_matrix, pairwise_distance_matrix,
    finite_difference_gradient, laplacian, frechet_distance, wasserstein_distance_1d
)


class TestTensorUtils:
    """Test tensor utility functions."""
    
    def test_safe_cat(self, device):
        """Test safe tensor concatenation."""
        # Normal case
        tensors = [
            torch.randn(2, 4, device=device),
            torch.randn(3, 4, device=device),
            torch.randn(1, 4, device=device)
        ]
        
        result = safe_cat(tensors, dim=0)
        assert result.shape == (6, 4)
        assert result.device == device
        
        # Empty list
        empty_result = safe_cat([])
        assert empty_result.numel() == 0
        
        # List with None values
        tensors_with_none = [tensors[0], None, tensors[1]]
        result_filtered = safe_cat(tensors_with_none, dim=0)
        assert result_filtered.shape == (5, 4)
    
    def test_safe_stack(self, device):
        """Test safe tensor stacking."""
        # Normal case
        tensors = [
            torch.randn(2, 4, device=device),
            torch.randn(2, 4, device=device),
            torch.randn(2, 4, device=device)
        ]
        
        result = safe_stack(tensors, dim=0)
        assert result.shape == (3, 2, 4)
        assert result.device == device
        
        # Empty list
        empty_result = safe_stack([])
        assert empty_result.numel() == 0
    
    def test_pad_sequence(self, device):
        """Test sequence padding functionality."""
        # Variable length sequences
        sequences = [
            torch.randn(3, 8, device=device),
            torch.randn(5, 8, device=device),
            torch.randn(2, 8, device=device)
        ]
        
        # Pad to max length
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        assert padded.shape == (3, 5, 8)  # max length is 5
        assert padded.device == device
        
        # Check padding values
        assert torch.all(padded[2, 2:, :] == 0.0)  # Third sequence padded
        
        # Pad with custom max length
        padded_custom = pad_sequence(sequences, batch_first=True, max_length=7)
        assert padded_custom.shape == (3, 7, 8)
    
    def test_chunk_tensor(self, device):
        """Test tensor chunking."""
        tensor = torch.randn(20, 8, device=device)
        
        # Basic chunking
        chunks = chunk_tensor(tensor, chunk_size=6, dim=0)
        assert len(chunks) == 4  # 20 / 6 = 3.33, so 4 chunks
        assert chunks[0].shape == (6, 8)
        assert chunks[-1].shape == (2, 8)  # Last chunk smaller
        
        # Chunking with overlap
        chunks_overlap = chunk_tensor(tensor, chunk_size=6, dim=0, overlap=2)
        assert len(chunks_overlap) >= len(chunks)  # More chunks due to overlap
    
    def test_flatten_unflatten_dict_tensors(self, device):
        """Test dictionary flattening and unflattening."""
        nested_dict = {
            'model': {
                'encoder': {
                    'weight': torch.randn(4, 8, device=device),
                    'bias': torch.randn(4, device=device)
                },
                'decoder': {
                    'weight': torch.randn(8, 4, device=device)
                }
            },
            'optimizer': {
                'lr': 0.001
            }
        }
        
        # Flatten
        flattened = flatten_dict_tensors(nested_dict)
        expected_keys = [
            'model.encoder.weight',
            'model.encoder.bias', 
            'model.decoder.weight',
            'optimizer.lr'
        ]
        
        for key in expected_keys:
            assert key in flattened
        
        # Unflatten
        unflattened = unflatten_dict_tensors(flattened)
        assert 'model' in unflattened
        assert 'encoder' in unflattened['model']
        assert 'weight' in unflattened['model']['encoder']
    
    def test_tensor_memory_usage(self, device):
        """Test memory usage calculation."""
        tensor = torch.randn(100, 100, dtype=torch.float32, device=device)
        
        memory_mb = tensor_memory_usage(tensor)
        expected_mb = (100 * 100 * 4) / (1024 * 1024)  # 4 bytes per float32
        
        assert abs(memory_mb - expected_mb) < 1e-6
    
    def test_move_to_device(self, device):
        """Test moving objects to device."""
        # Test with tensor
        tensor = torch.randn(4, 8)
        moved_tensor = move_to_device(tensor, device)
        assert moved_tensor.device == device
        
        # Test with nested structure
        nested_obj = {
            'tensor1': torch.randn(2, 4),
            'list': [torch.randn(3, 3), torch.randn(1, 5)],
            'tuple': (torch.randn(2, 2), 'string'),
            'scalar': 42
        }
        
        moved_obj = move_to_device(nested_obj, device)
        assert moved_obj['tensor1'].device == device
        assert moved_obj['list'][0].device == device
        assert moved_obj['tuple'][0].device == device
        assert moved_obj['scalar'] == 42  # Non-tensor preserved
    
    def test_detach_tensors(self, device):
        """Test tensor detachment."""
        # Create tensors with gradients
        tensor = torch.randn(4, 8, device=device, requires_grad=True)
        loss = tensor.sum()
        loss.backward()
        
        assert tensor.grad is not None
        
        # Detach
        detached = detach_tensors(tensor)
        assert not detached.requires_grad
        assert detached.grad is None
    
    def test_interpolate_tensors(self, device):
        """Test tensor interpolation."""
        tensor1 = torch.zeros(4, 8, device=device)
        tensor2 = torch.ones(4, 8, device=device)
        
        # Interpolate at alpha=0.5
        interpolated = interpolate_tensors(tensor1, tensor2, 0.5)
        expected = torch.full((4, 8), 0.5, device=device)
        
        assert torch.allclose(interpolated, expected)
        
        # Test boundary cases
        assert torch.allclose(interpolate_tensors(tensor1, tensor2, 0.0), tensor1)
        assert torch.allclose(interpolate_tensors(tensor1, tensor2, 1.0), tensor2)
    
    def test_normalize_tensor(self, device):
        """Test tensor normalization."""
        tensor = torch.randn(4, 8, device=device) * 10  # Scale up
        
        # Normalize along last dimension
        normalized = normalize_tensor(tensor, dim=-1)
        
        # Check that norms are approximately 1
        norms = torch.norm(normalized, dim=-1)
        expected_norms = torch.ones(4, device=device)
        assert torch.allclose(norms, expected_norms, atol=1e-6)


class TestModelUtils:
    """Test model utility functions."""
    
    def test_count_parameters(self, device):
        """Test parameter counting."""
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.Linear(20, 5)    # 20*5 + 5 = 105 params
        ).to(device)
        
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        assert total_params == 325  # 220 + 105
        assert trainable_params == 325  # All parameters trainable by default
    
    def test_freeze_parameters(self, device):
        """Test parameter freezing."""
        model = nn.Linear(10, 5).to(device)
        
        # Initially all parameters should be trainable
        assert all(p.requires_grad for p in model.parameters())
        
        # Freeze parameters
        freeze_parameters(model, freeze=True)
        assert not any(p.requires_grad for p in model.parameters())
        
        # Unfreeze parameters
        freeze_parameters(model, freeze=False)
        assert all(p.requires_grad for p in model.parameters())
    
    def test_get_device(self, device):
        """Test device detection."""
        model = nn.Linear(10, 5).to(device)
        detected_device = get_device(model)
        assert detected_device == device
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        
        set_seed(42)  # Reset seed
        
        # Should get same numbers
        torch_rand2 = torch.rand(5)
        
        assert torch.allclose(torch_rand1, torch_rand2)
    
    def test_initialize_weights(self, device):
        """Test weight initialization."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 5)
        ).to(device)
        
        # Save original weights
        original_weights = [p.clone() for p in model.parameters()]
        
        # Initialize with different method
        initialize_weights(model, init_type='xavier', gain=0.5)
        
        # Weights should be different
        new_weights = list(model.parameters())
        for orig, new in zip(original_weights, new_weights):
            assert not torch.allclose(orig, new, atol=1e-6)
    
    def test_gradient_operations(self, device):
        """Test gradient clipping and norm calculation."""
        model = nn.Linear(10, 5).to(device)
        
        # Create some loss
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 5, device=device)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        
        # Get gradient norm
        grad_norm = get_gradient_norm(model)
        assert grad_norm > 0
        assert torch.isfinite(torch.tensor(grad_norm))
        
        # Clip gradients
        clipped_norm = clip_gradients(model, max_norm=1.0)
        assert clipped_norm > 0
        
        # Check that gradients are clipped
        new_grad_norm = get_gradient_norm(model)
        assert new_grad_norm <= 1.0 + 1e-6  # Allow for small numerical errors
    
    def test_checkpoint_save_load(self, device, temp_dir):
        """Test checkpoint saving and loading."""
        model = nn.Linear(10, 5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few steps to get some state
        for _ in range(3):
            x = torch.randn(8, 10, device=device)
            y = torch.randn(8, 5, device=device)
            loss = nn.MSELoss()(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=loss.item(),
            filepath=str(checkpoint_path),
            custom_data="test"
        )
        
        assert checkpoint_path.exists()
        
        # Create new model and optimizer
        new_model = nn.Linear(10, 5).to(device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        # Load checkpoint
        checkpoint = load_checkpoint(
            filepath=str(checkpoint_path),
            model=new_model,
            optimizer=new_optimizer,
            device=device
        )
        
        assert checkpoint['epoch'] == 5
        assert checkpoint['custom_data'] == "test"
        
        # Models should have same parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_exponential_moving_average(self, device):
        """Test exponential moving average."""
        model = nn.Linear(10, 5).to(device)
        ema_model = nn.Linear(10, 5).to(device)
        
        # Initialize EMA model with different weights
        with torch.no_grad():
            for p in ema_model.parameters():
                p.fill_(0.0)
        
        # Update EMA
        exponential_moving_average(model, ema_model, decay=0.9)
        
        # EMA model weights should be updated
        for ema_param in ema_model.parameters():
            assert not torch.allclose(ema_param, torch.zeros_like(ema_param))
    
    def test_get_model_size_mb(self, device):
        """Test model size calculation."""
        model = nn.Linear(100, 50).to(device)  # 100*50 + 50 = 5050 float32 params
        
        size_mb = get_model_size_mb(model)
        expected_mb = (5050 * 4) / (1024 * 1024)  # 4 bytes per float32
        
        assert abs(size_mb - expected_mb) < 1e-6


class TestMathUtils:
    """Test mathematical utility functions."""
    
    def test_gaussian_kernel(self, device):
        """Test Gaussian kernel generation."""
        size = 5
        sigma = 1.0
        
        kernel = gaussian_kernel(size, sigma, device)
        
        assert kernel.shape == (size, size)
        assert kernel.device == device
        
        # Check that kernel sums to approximately 1
        assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
        
        # Check symmetry
        assert torch.allclose(kernel, kernel.t())
        
        # Center should be maximum
        center = size // 2
        assert kernel[center, center] == kernel.max()
    
    def test_soft_clamp(self, device):
        """Test soft clamping function."""
        x = torch.linspace(-10, 10, 100, device=device)
        
        clamped = soft_clamp(x, min_val=-2.0, max_val=2.0)
        
        # Check that output is approximately in range
        assert clamped.min() >= -2.1  # Allow small tolerance
        assert clamped.max() <= 2.1
        
        # Check that function is monotonic
        diff = clamped[1:] - clamped[:-1]
        assert torch.all(diff >= 0)  # Should be non-decreasing
    
    def test_stable_softmax(self, device):
        """Test numerically stable softmax."""
        # Test with large numbers that could cause overflow
        x = torch.tensor([1000, 1001, 1002], dtype=torch.float32, device=device)
        
        stable_probs = stable_softmax(x, temperature=1.0)
        pytorch_probs = torch.softmax(x, dim=0)
        
        # Should be close to PyTorch implementation
        assert torch.allclose(stable_probs, pytorch_probs, atol=1e-6)
        
        # Check that probabilities sum to 1
        assert torch.allclose(stable_probs.sum(), torch.tensor(1.0))
        
        # Test with temperature
        hot_probs = stable_softmax(x, temperature=0.1)
        cold_probs = stable_softmax(x, temperature=10.0)
        
        # Lower temperature should be more peaked
        assert hot_probs.max() > stable_probs.max()
        # Higher temperature should be more uniform
        assert cold_probs.max() < stable_probs.max()
    
    def test_log_sum_exp(self, device):
        """Test log-sum-exp computation."""
        x = torch.randn(10, 5, device=device)
        
        lse = log_sum_exp(x, dim=1)
        
        # Compare with manual computation
        manual_lse = torch.log(torch.sum(torch.exp(x), dim=1))
        
        assert torch.allclose(lse, manual_lse, atol=1e-6)
        
        # Test with large numbers
        large_x = torch.tensor([[1000, 1001], [500, 502]], dtype=torch.float32, device=device)
        lse_large = log_sum_exp(large_x, dim=1)
        
        assert torch.isfinite(lse_large).all()
    
    def test_gumbel_noise(self, device):
        """Test Gumbel noise generation."""
        shape = (100, 50)
        noise = gumbel_noise(shape, device)
        
        assert noise.shape == shape
        assert noise.device == device
        assert torch.isfinite(noise).all()
        
        # Check statistical properties (approximately)
        # Gumbel distribution has mean ≈ 0.577 (Euler's constant)
        mean = noise.mean()
        assert 0.4 < mean < 0.8  # Rough check
    
    def test_gumbel_softmax(self, device):
        """Test Gumbel-Softmax sampling."""
        logits = torch.randn(10, 5, device=device)
        
        # Soft sampling
        soft_samples = gumbel_softmax(logits, temperature=1.0, hard=False)
        
        assert soft_samples.shape == logits.shape
        assert torch.allclose(soft_samples.sum(dim=1), torch.ones(10, device=device))
        
        # Hard sampling
        hard_samples = gumbel_softmax(logits, temperature=1.0, hard=True)
        
        assert hard_samples.shape == logits.shape
        # Each row should be one-hot
        assert torch.allclose(hard_samples.sum(dim=1), torch.ones(10, device=device))
        
        # Check that hard samples are indeed one-hot (approximately)
        max_vals, _ = hard_samples.max(dim=1)
        assert torch.allclose(max_vals, torch.ones(10, device=device))
    
    def test_cosine_similarity_matrix(self, device):
        """Test cosine similarity matrix computation."""
        x = torch.randn(8, 16, device=device)
        y = torch.randn(10, 16, device=device)
        
        sim_matrix = cosine_similarity_matrix(x, y)
        
        assert sim_matrix.shape == (8, 10)
        assert sim_matrix.device == device
        
        # Values should be in [-1, 1]
        assert sim_matrix.min() >= -1.0 - 1e-6
        assert sim_matrix.max() <= 1.0 + 1e-6
        
        # Test self-similarity (should be identity for normalized vectors)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        self_sim = cosine_similarity_matrix(x_norm, x_norm)
        
        # Diagonal should be 1
        diagonal = torch.diag(self_sim)
        assert torch.allclose(diagonal, torch.ones(8, device=device), atol=1e-6)
    
    def test_pairwise_distance_matrix(self, device):
        """Test pairwise distance matrix computation."""
        x = torch.randn(6, 8, device=device)
        y = torch.randn(4, 8, device=device)
        
        dist_matrix = pairwise_distance_matrix(x, y)
        
        assert dist_matrix.shape == (6, 4)
        assert dist_matrix.device == device
        assert torch.all(dist_matrix >= 0)  # Distances are non-negative
        
        # Test self-distance (should be zero)
        self_dist = pairwise_distance_matrix(x, x)
        diagonal = torch.diag(self_dist)
        assert torch.allclose(diagonal, torch.zeros(6, device=device), atol=1e-6)
    
    def test_finite_difference_gradient(self, device):
        """Test finite difference gradient computation."""
        # Create a simple 2D function: f(x,y) = x^2 + y^2
        H, W = 32, 32
        x_coords = torch.linspace(-2, 2, W, device=device)
        y_coords = torch.linspace(-2, 2, H, device=device)
        
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        f = X**2 + Y**2  # f(x,y) = x^2 + y^2
        f = f.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        grad_y, grad_x = finite_difference_gradient(f)
        
        assert grad_y.shape == f.shape
        assert grad_x.shape == f.shape
        
        # Analytical gradients: df/dx = 2x, df/dy = 2y
        expected_grad_x = 2 * X
        expected_grad_y = 2 * Y
        
        # Check central regions (edges have different finite difference schemes)
        center_slice = slice(5, -5)
        assert torch.allclose(
            grad_x[0, 0, center_slice, center_slice],
            expected_grad_x[center_slice, center_slice],
            atol=1e-3
        )
    
    def test_laplacian(self, device):
        """Test Laplacian computation."""
        # Create a 2D Gaussian
        H, W = 32, 32
        x_coords = torch.linspace(-2, 2, W, device=device)
        y_coords = torch.linspace(-2, 2, H, device=device)
        
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        gaussian = torch.exp(-(X**2 + Y**2))
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        lap = laplacian(gaussian)
        
        assert lap.shape == gaussian.shape
        assert torch.isfinite(lap).all()
        
        # For a Gaussian, Laplacian should be negative at the center
        center_y, center_x = H // 2, W // 2
        assert lap[0, 0, center_y, center_x] < 0


# Integration tests for utility combinations
class TestUtilityIntegration:
    """Test combinations of utility functions."""
    
    def test_tensor_model_utils_integration(self, device, temp_dir):
        """Test tensor and model utils working together."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ).to(device)
        
        # Count parameters
        param_count = count_parameters(model)
        assert param_count > 0
        
        # Create some tensor data
        tensors = [torch.randn(8, 10, device=device) for _ in range(5)]
        batched_data = safe_stack(tensors)
        
        # Forward pass
        output = model(batched_data)
        
        # Check tensor properties
        assert tensor_memory_usage(output) > 0
        assert torch.isfinite(output).all()
        
        # Save model
        checkpoint_path = temp_dir / "integration_test.pt"
        save_checkpoint(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            epoch=1,
            loss=0.5,
            filepath=str(checkpoint_path)
        )
        
        assert checkpoint_path.exists()
    
    def test_math_tensor_utils_integration(self, device):
        """Test math and tensor utils working together."""
        # Create some data
        data = torch.randn(100, 64, device=device)
        
        # Normalize
        normalized_data = normalize_tensor(data, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity_matrix(normalized_data, normalized_data)
        
        # Apply soft clamping
        clamped_sim = soft_clamp(sim_matrix, -0.8, 0.8)
        
        # Apply stable softmax
        attention_weights = stable_softmax(clamped_sim, dim=-1)
        
        # Check final result
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(100, device=device))
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)
    
    @pytest.mark.slow
    def test_memory_efficiency(self, device):
        """Test memory efficiency of utility functions."""
        if device.type == 'cpu':
            pytest.skip("Memory efficiency test requires GPU")
        
        # Create large tensors
        large_tensors = [torch.randn(1000, 1000, device=device) for _ in range(10)]
        
        # Measure initial memory
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Use chunking to process large tensors
        chunked_results = []
        for tensor in large_tensors:
            chunks = chunk_tensor(tensor, chunk_size=200, dim=0)
            chunk_results = []
            
            for chunk in chunks:
                # Some processing
                processed = normalize_tensor(chunk, dim=-1)
                chunk_results.append(processed)
            
            # Concatenate results
            result = safe_cat(chunk_results, dim=0)
            chunked_results.append(result)
        
        # Clean up
        del large_tensors, chunked_results
        torch.cuda.empty_cache()
        
        # Memory should not have grown too much
        final_memory = torch.cuda.memory_allocated(device)
        memory_growth = final_memory - initial_memory
        
        # Allow some memory growth but not excessive
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth