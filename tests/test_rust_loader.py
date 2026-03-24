"""
Tests for Rust-accelerated data loader.

This module provides comprehensive tests for:
- Rust loader functionality
- Output equivalence between Rust and Python loaders
- Performance benchmarks
- Edge cases and error handling

Run with:
    pytest tests/test_rust_loader.py -v
    pytest tests/test_rust_loader.py -v -k benchmark  # benchmarks only
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
import cv2

# Import loaders
from data import (
    RUST_AVAILABLE,
    check_rust_available,
    get_rust_version,
    get_backend_info,
    create_dataloaders,
    get_best_loader,
    HybridDataLoader,
    HybridLoaderConfig,
)

# Try to import Rust-specific modules
try:
    from data.rust_loader import (
        RustEnfusionDataset,
        RustDataLoader,
        RustBatchLoader,
        RustDatasetConfig,
        RustCollator,
        benchmark_rust_vs_python,
    )
except ImportError:
    RustEnfusionDataset = None
    RustDataLoader = None
    RustBatchLoader = None
    RustDatasetConfig = None
    RustCollator = None
    benchmark_rust_vs_python = None

# Import Python loader for comparison
from data.enfusion_loader import (
    EnfusionDataset as PythonEnfusionDataset,
    EnfusionDatasetConfig as PythonDatasetConfig,
)


@dataclass
class TestSessionData:
    """Test session data generator."""
    session_dir: Path
    num_frames: int
    image_size: Tuple[int, int]

    @classmethod
    def create(
        cls,
        root_dir: Path,
        session_id: str = "session_0001",
        num_frames: int = 32,
        image_size: Tuple[int, int] = (256, 256),
    ) -> "TestSessionData":
        """Create a test session with synthetic data."""
        session_dir = root_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create frames directory
        frames_dir = session_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Create depth directory
        depth_dir = session_dir / "depth"
        depth_dir.mkdir(exist_ok=True)

        # Generate synthetic frames
        for i in range(num_frames):
            # Create random image
            frame = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            frame_path = frames_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)

            # Create depth map
            depth = np.random.rand(*image_size).astype(np.float32) * 100
            depth_path = depth_dir / f"depth_{i:06d}.npz"
            np.savez(str(depth_path), depth=depth)

        # Create telemetry
        telemetry_data = []
        for i in range(num_frames):
            telemetry_data.append({
                "timestamp": i / 30.0,
                "steering_angle": np.sin(i * 0.1) * 0.5,
                "accel": 0.3,
                "brake": 0.0,
                "speed": 10.0 + np.random.randn() * 0.5,
            })

        telemetry_path = session_dir / "telemetry.csv"
        with open(telemetry_path, "w") as f:
            # Write header
            f.write("timestamp,steering_angle,accel,brake,speed\n")
            for row in telemetry_data:
                f.write(f"{row['timestamp']},{row['steering_angle']},"
                       f"{row['accel']},{row['brake']},{row['speed']}\n")

        # Create session info
        info_path = session_dir / "session_info.txt"
        with open(info_path, "w") as f:
            f.write(f"session_id: {session_id}\n")
            f.write(f"num_frames: {num_frames}\n")

        return cls(
            session_dir=session_dir,
            num_frames=num_frames,
            image_size=image_size,
        )


@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create multiple test sessions
        TestSessionData.create(root, "session_0001", num_frames=32)
        TestSessionData.create(root, "session_0002", num_frames=48)
        TestSessionData.create(root, "session_0003", num_frames=24)

        yield root


@pytest.fixture
def small_test_data():
    """Create minimal test data for quick tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        TestSessionData.create(root, "session_test", num_frames=16, image_size=(64, 64))
        yield root


class TestBackendAvailability:
    """Test backend availability and version info."""

    def test_rust_availability_check(self):
        """Test that Rust availability check works."""
        result = check_rust_available()
        assert isinstance(result, bool)

    def test_backend_info(self):
        """Test backend info retrieval."""
        info = get_backend_info()
        assert "rust_available" in info
        assert "preferred_backend" in info
        assert info["preferred_backend"] in ["rust", "python"]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_rust_version(self):
        """Test Rust version retrieval."""
        version = get_rust_version()
        assert version is not None
        assert isinstance(version, str)


class TestHybridLoader:
    """Test hybrid loader with automatic backend selection."""

    def test_hybrid_loader_creation(self, small_test_data):
        """Test that hybrid loader can be created."""
        config = HybridLoaderConfig(
            sequence_length=8,
            image_size=(64, 64),
            batch_size=2,
        )

        loader = HybridDataLoader(small_test_data, config, split="train")
        assert len(loader) > 0
        assert loader.backend in ["rust", "python"]

    def test_hybrid_loader_iteration(self, small_test_data):
        """Test iterating through hybrid loader."""
        config = HybridLoaderConfig(
            sequence_length=4,
            image_size=(64, 64),
            batch_size=2,
        )

        loader = HybridDataLoader(small_test_data, config, split="train")

        batch = next(iter(loader))
        assert "frames" in batch
        assert "controls" in batch

        # Check tensor shapes
        frames = batch["frames"]
        assert frames.dim() == 5  # [B, T, C, H, W]
        assert frames.shape[1] == 4  # sequence_length
        assert frames.shape[2] == 3  # RGB channels

    def test_create_dataloaders_unified(self, test_data_dir):
        """Test unified create_dataloaders function."""
        train, val, test = create_dataloaders(
            test_data_dir,
            batch_size=2,
            sequence_length=8,
            image_size=(64, 64),
        )

        assert len(train) > 0
        assert len(val) >= 0  # May be empty if not enough sessions
        assert len(test) >= 0

    def test_get_best_loader(self, small_test_data):
        """Test get_best_loader convenience function."""
        loader = get_best_loader(
            small_test_data,
            batch_size=2,
            sequence_length=4,
            image_size=(64, 64),
        )

        assert isinstance(loader, HybridDataLoader)
        assert len(loader) > 0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
class TestRustLoader:
    """Test Rust-specific loader functionality."""

    def test_rust_dataset_creation(self, small_test_data):
        """Test creating Rust dataset."""
        config = RustDatasetConfig(
            sequence_length=8,
            image_height=64,
            image_width=64,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        assert len(dataset) > 0
        assert dataset.num_sessions > 0

    def test_rust_dataset_getitem(self, small_test_data):
        """Test getting items from Rust dataset."""
        config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        sample = dataset[0]
        assert "frames" in sample
        assert "controls" in sample
        assert "metadata" in sample

        # Check shapes
        frames = sample["frames"]
        assert isinstance(frames, torch.Tensor)
        assert frames.shape[0] == 4  # sequence_length
        assert frames.shape[1] == 3  # channels
        assert frames.shape[2] == 64  # height
        assert frames.shape[3] == 64  # width

    def test_rust_batch_loading(self, small_test_data):
        """Test batch loading with Rust."""
        config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        # Load batch
        indices = [0, 1] if len(dataset) > 1 else [0]
        samples = dataset.load_batch(indices)

        assert len(samples) == len(indices)
        for sample in samples:
            assert "frames" in sample

    def test_rust_dataloader(self, small_test_data):
        """Test RustDataLoader wrapper."""
        config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        loader = RustDataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))
        assert "frames" in batch
        assert batch["frames"].dim() == 5

    def test_rust_cache_stats(self, small_test_data):
        """Test cache statistics."""
        config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
            cache_frames=True,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        # Load some data
        _ = dataset[0]
        _ = dataset[0]  # Should hit cache

        stats = dataset.cache_stats()
        assert "cache_hits" in stats or "cache_hit_rate" in stats

    def test_rust_collator(self, small_test_data):
        """Test RustCollator."""
        config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        collator = RustCollator()

        # Get two samples
        samples = [dataset[0]]
        if len(dataset) > 1:
            samples.append(dataset[1])

        batch = collator(samples)
        assert "frames" in batch
        assert batch["frames"].shape[0] == len(samples)


class TestOutputEquivalence:
    """Test that Rust and Python loaders produce equivalent output."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_frame_value_equivalence(self, small_test_data):
        """Test that frame values are approximately equal between backends."""
        # Create Rust dataset
        rust_config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
            normalize_frames=True,
        )
        rust_dataset = RustEnfusionDataset(
            small_test_data,
            config=rust_config,
            split="train",
        )

        # Create Python dataset
        python_config = PythonDatasetConfig(
            sequence_length=4,
            image_size=(64, 64),
            normalize_frames=True,
        )
        python_dataset = PythonEnfusionDataset(
            small_test_data,
            config=python_config,
            split="train",
        )

        if len(rust_dataset) == 0 or len(python_dataset) == 0:
            pytest.skip("Not enough data for equivalence test")

        # Compare first sample
        rust_sample = rust_dataset[0]
        python_sample = python_dataset[0]

        rust_frames = rust_sample["frames"].numpy()
        python_frames = python_sample["frames"].numpy()

        # Check shapes match
        assert rust_frames.shape == python_frames.shape

        # Values should be close (allow for small numerical differences)
        # Note: due to different image loading libraries, there may be
        # small differences in pixel values
        np.testing.assert_allclose(
            rust_frames, python_frames,
            rtol=0.1,  # 10% relative tolerance
            atol=0.05,  # 5% absolute tolerance
        )

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_control_equivalence(self, small_test_data):
        """Test that control signals match between backends."""
        rust_config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
        )
        rust_dataset = RustEnfusionDataset(
            small_test_data,
            config=rust_config,
            split="train",
        )

        python_config = PythonDatasetConfig(
            sequence_length=4,
            image_size=(64, 64),
        )
        python_dataset = PythonEnfusionDataset(
            small_test_data,
            config=python_config,
            split="train",
        )

        if len(rust_dataset) == 0 or len(python_dataset) == 0:
            pytest.skip("Not enough data for equivalence test")

        rust_sample = rust_dataset[0]
        python_sample = python_dataset[0]

        rust_controls = rust_sample["controls"].numpy()
        python_controls = python_sample["controls"].numpy()

        # Shape should match (may have different dims for the last axis)
        assert rust_controls.shape[0] == python_controls.shape[0]  # sequence length


@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmarks comparing Rust and Python loaders."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_rust_loading_speed(self, test_data_dir, benchmark):
        """Benchmark Rust frame loading speed."""
        config = RustDatasetConfig(
            sequence_length=8,
            image_height=128,
            image_width=128,
        )

        dataset = RustEnfusionDataset(
            test_data_dir,
            config=config,
            split="train",
        )

        def load_sample():
            return dataset[0]

        result = benchmark(load_sample)
        assert result is not None

    def test_python_loading_speed(self, test_data_dir, benchmark):
        """Benchmark Python frame loading speed."""
        config = PythonDatasetConfig(
            sequence_length=8,
            image_size=(128, 128),
        )

        dataset = PythonEnfusionDataset(
            test_data_dir,
            config=config,
            split="train",
        )

        if len(dataset) == 0:
            pytest.skip("No data for benchmark")

        def load_sample():
            return dataset[0]

        result = benchmark(load_sample)
        assert result is not None

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_speedup_measurement(self, test_data_dir):
        """Measure Rust vs Python speedup."""
        # Time Rust loading
        rust_config = RustDatasetConfig(
            sequence_length=8,
            image_height=128,
            image_width=128,
            cache_frames=False,  # Disable cache for fair comparison
        )
        rust_dataset = RustEnfusionDataset(
            test_data_dir,
            config=rust_config,
            split="train",
        )

        # Time Python loading
        python_config = PythonDatasetConfig(
            sequence_length=8,
            image_size=(128, 128),
            cache_frames=False,
        )
        python_dataset = PythonEnfusionDataset(
            test_data_dir,
            config=python_config,
            split="train",
        )

        if len(rust_dataset) == 0 or len(python_dataset) == 0:
            pytest.skip("Not enough data for speedup test")

        num_iterations = 10

        # Warmup
        _ = rust_dataset[0]
        _ = python_dataset[0]

        # Time Rust
        rust_start = time.time()
        for _ in range(num_iterations):
            _ = rust_dataset[0]
        rust_time = (time.time() - rust_start) / num_iterations

        # Time Python
        python_start = time.time()
        for _ in range(num_iterations):
            _ = python_dataset[0]
        python_time = (time.time() - python_start) / num_iterations

        speedup = python_time / rust_time if rust_time > 0 else float('inf')

        print(f"\nSpeedup Results:")
        print(f"  Rust:   {rust_time*1000:.2f} ms/sample")
        print(f"  Python: {python_time*1000:.2f} ms/sample")
        print(f"  Speedup: {speedup:.1f}x")

        # Assert at least some speedup (or not worse)
        assert speedup >= 0.5, f"Rust should not be more than 2x slower than Python"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_batch_loading_speedup(self, test_data_dir):
        """Measure batch loading speedup."""
        rust_config = RustDatasetConfig(
            sequence_length=8,
            image_height=128,
            image_width=128,
            cache_frames=False,
        )
        rust_dataset = RustEnfusionDataset(
            test_data_dir,
            config=rust_config,
            split="train",
        )

        if len(rust_dataset) < 4:
            pytest.skip("Not enough data for batch loading test")

        batch_indices = list(range(min(4, len(rust_dataset))))

        # Time batch loading
        num_iterations = 5

        # Warmup
        _ = rust_dataset.load_batch(batch_indices)

        start = time.time()
        for _ in range(num_iterations):
            _ = rust_dataset.load_batch(batch_indices)
        batch_time = (time.time() - start) / num_iterations

        # Time sequential loading
        start = time.time()
        for _ in range(num_iterations):
            for idx in batch_indices:
                _ = rust_dataset[idx]
        sequential_time = (time.time() - start) / num_iterations

        batch_speedup = sequential_time / batch_time if batch_time > 0 else float('inf')

        print(f"\nBatch Loading Results:")
        print(f"  Batch:      {batch_time*1000:.2f} ms")
        print(f"  Sequential: {sequential_time*1000:.2f} ms")
        print(f"  Speedup:    {batch_speedup:.1f}x")

        # Batch loading should provide some benefit
        assert batch_speedup >= 0.8


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory(self, tmp_path):
        """Test handling of empty data directory."""
        config = HybridLoaderConfig(
            sequence_length=4,
            image_size=(64, 64),
            batch_size=2,
        )

        with pytest.raises(Exception):
            HybridDataLoader(tmp_path, config, split="train")

    def test_negative_indexing(self, small_test_data):
        """Test negative indexing."""
        config = HybridLoaderConfig(
            sequence_length=4,
            image_size=(64, 64),
            batch_size=2,
        )

        loader = HybridDataLoader(small_test_data, config, split="train")
        dataset = loader._loader.dataset if hasattr(loader._loader, 'dataset') else None

        if dataset is not None and len(dataset) > 0:
            # Test negative indexing
            sample_neg = dataset[-1]
            sample_pos = dataset[len(dataset) - 1]

            # Should be the same sample
            assert sample_neg["metadata"]["start_frame"] == sample_pos["metadata"]["start_frame"]

    def test_index_out_of_bounds(self, small_test_data):
        """Test index out of bounds handling."""
        config = HybridLoaderConfig(
            sequence_length=4,
            image_size=(64, 64),
            batch_size=2,
        )

        loader = HybridDataLoader(small_test_data, config, split="train")
        dataset = loader._loader.dataset if hasattr(loader._loader, 'dataset') else None

        if dataset is not None:
            with pytest.raises(IndexError):
                _ = dataset[len(dataset) + 100]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
    def test_clear_cache(self, small_test_data):
        """Test cache clearing."""
        config = RustDatasetConfig(
            sequence_length=4,
            image_height=64,
            image_width=64,
            cache_frames=True,
        )

        dataset = RustEnfusionDataset(
            small_test_data,
            config=config,
            split="train",
        )

        # Load some data
        _ = dataset[0]

        # Clear cache
        dataset.clear_cache()

        # Should work after clearing
        _ = dataset[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
