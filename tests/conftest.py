"""
Pytest configuration and fixtures for DriveDiT tests.
Common fixtures and test utilities.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Dict, Any
import tempfile
import shutil
from pathlib import Path

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import ModelConfig, get_small_config, get_medium_config
from models import VAE3D, DiTStudent, DiTTeacher
from utils import set_seed


@pytest.fixture(scope="session")
def device():
    """Test device (CPU for CI compatibility)."""
    return torch.device('cpu')  # Use CPU for stable testing


@pytest.fixture(scope="session")
def seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def setup_seed(seed):
    """Automatically set seed for all tests."""
    set_seed(seed)


@pytest.fixture
def small_config():
    """Small model configuration for testing."""
    return get_small_config()


@pytest.fixture
def medium_config():
    """Medium model configuration for testing."""
    return get_medium_config()


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_video_data(device):
    """Sample video data for testing."""
    B, T, C, H, W = 2, 8, 3, 64, 64
    frames = torch.randn(B, T, C, H, W, device=device)
    return frames


@pytest.fixture
def sample_control_data(device):
    """Sample control data for testing."""
    B, T, control_dim = 2, 8, 4
    controls = torch.randn(B, T, control_dim, device=device)
    return controls


@pytest.fixture
def sample_latent_data(device):
    """Sample latent data for testing."""
    B, T, C, H, W = 2, 8, 8, 8, 8  # Latent dimensions
    latents = torch.randn(B, T, C, H, W, device=device)
    return latents


@pytest.fixture
def vae_model(small_config, device):
    """VAE model for testing."""
    model = VAE3D(
        in_channels=small_config.vae.in_channels,
        latent_dim=small_config.vae.latent_dim,
        hidden_dims=small_config.vae.hidden_dims,
        num_res_blocks=small_config.vae.num_res_blocks,
        use_attention=small_config.vae.use_attention
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def dit_student_model(small_config, device):
    """DiT student model for testing."""
    model = DiTStudent(
        latent_dim=small_config.student.latent_dim,
        d_model=small_config.student.d_model,
        n_layers=small_config.student.n_layers,
        n_heads=small_config.student.n_heads,
        max_seq_len=small_config.student.max_seq_len,
        use_rope=small_config.student.use_rope,
        use_memory=small_config.student.use_memory
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def dit_teacher_model(small_config, device):
    """DiT teacher model for testing."""
    model = DiTTeacher(
        latent_dim=small_config.teacher.latent_dim,
        d_model=small_config.teacher.d_model,
        n_layers=small_config.teacher.n_layers,
        n_heads=small_config.teacher.n_heads,
        max_seq_len=small_config.teacher.max_seq_len,
        num_diffusion_steps=100  # Smaller for testing
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def sample_tensor_2d(device):
    """Sample 2D tensor for testing."""
    return torch.randn(4, 8, device=device)


@pytest.fixture
def sample_tensor_3d(device):
    """Sample 3D tensor for testing."""
    return torch.randn(2, 4, 8, device=device)


@pytest.fixture
def sample_tensor_4d(device):
    """Sample 4D tensor for testing."""
    return torch.randn(2, 3, 16, 16, device=device)


@pytest.fixture
def sample_tensor_5d(device):
    """Sample 5D tensor for testing."""
    return torch.randn(1, 3, 8, 16, 16, device=device)


# Test data generators
def generate_test_cases(param_sets):
    """Generate parameterized test cases."""
    return pytest.mark.parametrize("params", param_sets)


# Common test utilities
class TestUtils:
    """Utility class for common test operations."""
    
    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype):
        """Assert tensor has expected dtype."""
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    @staticmethod
    def assert_tensor_device(tensor: torch.Tensor, expected_device: torch.device):
        """Assert tensor is on expected device."""
        assert tensor.device == expected_device, f"Expected device {expected_device}, got {tensor.device}"
    
    @staticmethod
    def assert_tensor_finite(tensor: torch.Tensor):
        """Assert tensor contains only finite values."""
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
    
    @staticmethod
    def assert_tensor_not_nan(tensor: torch.Tensor):
        """Assert tensor contains no NaN values."""
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    
    @staticmethod
    def assert_tensor_not_inf(tensor: torch.Tensor):
        """Assert tensor contains no infinite values."""
        assert not torch.isinf(tensor).any(), "Tensor contains infinite values"
    
    @staticmethod
    def assert_model_output_valid(model_output, expected_shape=None):
        """Assert model output is valid."""
        if isinstance(model_output, torch.Tensor):
            TestUtils.assert_tensor_finite(model_output)
            if expected_shape:
                TestUtils.assert_tensor_shape(model_output, expected_shape)
        elif isinstance(model_output, (tuple, list)):
            for output in model_output:
                if isinstance(output, torch.Tensor):
                    TestUtils.assert_tensor_finite(output)
        elif isinstance(model_output, dict):
            for key, output in model_output.items():
                if isinstance(output, torch.Tensor):
                    TestUtils.assert_tensor_finite(output)


@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils


# Performance measurement utilities
@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed_time()
        
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Memory measurement utilities
@pytest.fixture
def memory_profiler():
    """Memory profiler for performance tests."""
    
    class MemoryProfiler:
        def __init__(self, device):
            self.device = device
        
        def get_memory_usage(self):
            """Get current memory usage in MB."""
            if self.device.type == 'cuda':
                return torch.cuda.memory_allocated(self.device) / 1024 / 1024
            else:
                # For CPU, we can't easily measure memory usage
                return 0.0
        
        def reset_peak_memory(self):
            """Reset peak memory statistics."""
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
        
        def get_peak_memory(self):
            """Get peak memory usage in MB."""
            if self.device.type == 'cuda':
                return torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            else:
                return 0.0
    
    return MemoryProfiler


# Skip conditions for different test types
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

skip_if_no_memory = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3,
    reason="Insufficient GPU memory (< 8GB)"
)

# Test categories
unit_test = pytest.mark.unit
integration_test = pytest.mark.integration
performance_test = pytest.mark.performance
slow_test = pytest.mark.slow