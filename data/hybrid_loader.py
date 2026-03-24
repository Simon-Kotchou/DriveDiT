"""
Hybrid data loader with Rust acceleration and Python fallback.

This module provides a unified API that transparently uses the Rust
backend when available, falling back to pure Python implementation
when Rust is not installed.

Features:
- Automatic Rust/Python backend selection
- Optional Go service integration for prefetching
- Seamless integration with existing training code
- Performance monitoring and logging

Usage:
    from data.hybrid_loader import HybridDataLoader, create_hybrid_dataloaders

    # Automatically uses Rust if available, Python otherwise
    loader = HybridDataLoader(
        data_root="/path/to/data",
        batch_size=8,
        sequence_length=16,
    )

    for batch in loader:
        frames = batch['frames']
        controls = batch['controls']
"""

import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable
from dataclasses import dataclass, field
import warnings
import time
import threading
import queue
import logging

# Import backends
from .rust_loader import (
    RUST_AVAILABLE,
    RustEnfusionDataset,
    RustDataLoader,
    RustBatchLoader,
    RustDatasetConfig,
    RustCollator,
)

from .enfusion_loader import (
    EnfusionDataset as PythonEnfusionDataset,
    EnfusionDatasetConfig as PythonDatasetConfig,
    EnfusionCollator as PythonCollator,
    EnfusionDataLoader as PythonDataLoader,
)

# Logger
logger = logging.getLogger(__name__)


@dataclass
class HybridLoaderConfig:
    """Configuration for hybrid data loader."""

    # Common settings
    sequence_length: int = 16
    frame_skip: int = 1
    image_size: Tuple[int, int] = (256, 256)
    load_depth: bool = True
    load_controls: bool = True
    control_dim: int = 6

    # Loading settings
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Caching settings
    cache_frames: bool = True
    max_cache_mb: float = 1000.0

    # Backend settings
    prefer_rust: bool = True
    use_batch_loader: bool = True  # Use Rust batch loader if available

    # Go prefetch service (optional)
    go_prefetch_enabled: bool = False
    go_prefetch_address: str = "localhost:50051"
    go_prefetch_buffer_size: int = 16

    # Performance monitoring
    track_performance: bool = True
    log_interval: int = 100

    def to_rust_config(self) -> RustDatasetConfig:
        """Convert to Rust dataset config."""
        return RustDatasetConfig(
            sequence_length=self.sequence_length,
            frame_skip=self.frame_skip,
            image_height=self.image_size[0],
            image_width=self.image_size[1],
            load_depth=self.load_depth,
            load_controls=self.load_controls,
            control_dim=self.control_dim,
            cache_frames=self.cache_frames,
            max_cache_mb=self.max_cache_mb,
            num_workers=self.num_workers,
        )

    def to_python_config(self) -> PythonDatasetConfig:
        """Convert to Python dataset config."""
        return PythonDatasetConfig(
            sequence_length=self.sequence_length,
            frame_skip=self.frame_skip,
            image_size=self.image_size,
            load_depth=self.load_depth,
            load_controls=self.load_controls,
            control_dim=self.control_dim,
            cache_frames=self.cache_frames,
            max_cache_memory_mb=self.max_cache_mb,
        )


class PerformanceTracker:
    """Track data loading performance metrics."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.load_times: List[float] = []
        self.batch_count = 0
        self.total_samples = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_batch(self, load_time: float, batch_size: int):
        """Record a batch loading event."""
        with self._lock:
            self.load_times.append(load_time)
            self.batch_count += 1
            self.total_samples += batch_size

            if self.batch_count % self.log_interval == 0:
                self._log_stats()

    def _log_stats(self):
        """Log performance statistics."""
        if not self.load_times:
            return

        avg_time = np.mean(self.load_times[-self.log_interval:])
        total_time = time.time() - self.start_time
        samples_per_sec = self.total_samples / total_time if total_time > 0 else 0

        logger.info(
            f"Batch {self.batch_count}: "
            f"avg_load_time={avg_time*1000:.1f}ms, "
            f"samples/sec={samples_per_sec:.1f}"
        )

    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        with self._lock:
            if not self.load_times:
                return {}

            total_time = time.time() - self.start_time
            return {
                "batch_count": self.batch_count,
                "total_samples": self.total_samples,
                "total_time_sec": total_time,
                "avg_load_time_ms": np.mean(self.load_times) * 1000,
                "samples_per_sec": self.total_samples / total_time if total_time > 0 else 0,
            }

    def reset(self):
        """Reset statistics."""
        with self._lock:
            self.load_times = []
            self.batch_count = 0
            self.total_samples = 0
            self.start_time = time.time()


class GoPrefetchClient:
    """
    Optional Go service client for prefetching.

    The Go service provides additional prefetching capabilities
    using gRPC for communication. This is optional and the loader
    works without it.
    """

    def __init__(
        self,
        address: str = "localhost:50051",
        buffer_size: int = 16,
    ):
        self.address = address
        self.buffer_size = buffer_size
        self._connected = False
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._worker_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Attempt to connect to Go prefetch service."""
        try:
            # Try to import gRPC and connect
            # This is optional - if not available, we fall back to local prefetching
            import grpc

            # In production, this would use generated gRPC stubs
            # For now, we just check if we can create a channel
            channel = grpc.insecure_channel(self.address)
            # Test connection
            try:
                grpc.channel_ready_future(channel).result(timeout=1.0)
                self._connected = True
                logger.info(f"Connected to Go prefetch service at {self.address}")
                return True
            except grpc.FutureTimeoutError:
                logger.warning(f"Go prefetch service not available at {self.address}")
                return False
        except ImportError:
            logger.warning("gRPC not available, Go prefetch service disabled")
            return False

    def prefetch(self, indices: List[int]) -> None:
        """Request prefetching of samples."""
        if not self._connected:
            return

        # In production, this would send prefetch request to Go service
        # For now, we just log the request
        logger.debug(f"Prefetch request: {len(indices)} samples")

    def get_prefetched(self) -> Optional[Dict[str, Any]]:
        """Get a prefetched sample if available."""
        if not self._connected:
            return None

        try:
            return self._prefetch_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        """Close connection to Go service."""
        self._connected = False


class HybridDataLoader:
    """
    Hybrid data loader that automatically uses the best available backend.

    Priority:
    1. Rust backend (if available and prefer_rust=True)
    2. Go prefetch service (if enabled and available)
    3. Pure Python fallback

    This provides seamless integration with existing training code
    while offering significant performance improvements when Rust is available.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        config: Optional[HybridLoaderConfig] = None,
        split: str = "train",
        session_ids: Optional[List[str]] = None,
    ):
        """
        Initialize hybrid data loader.

        Args:
            data_root: Root directory containing session folders
            config: Loader configuration
            split: Dataset split ('train', 'val', 'test')
            session_ids: Optional list of specific session IDs
        """
        self.data_root = Path(data_root)
        self.config = config or HybridLoaderConfig()
        self.split = split
        self.session_ids = session_ids

        # Determine backend
        self._use_rust = RUST_AVAILABLE and self.config.prefer_rust
        self._backend_name = "rust" if self._use_rust else "python"

        # Initialize performance tracker
        self._performance_tracker = None
        if self.config.track_performance:
            self._performance_tracker = PerformanceTracker(self.config.log_interval)

        # Initialize Go prefetch client (optional)
        self._go_client = None
        if self.config.go_prefetch_enabled:
            self._go_client = GoPrefetchClient(
                self.config.go_prefetch_address,
                self.config.go_prefetch_buffer_size,
            )
            self._go_client.connect()

        # Create underlying loader
        self._loader = self._create_loader()

        logger.info(
            f"HybridDataLoader initialized: backend={self._backend_name}, "
            f"split={split}, batch_size={self.config.batch_size}"
        )

    def _create_loader(self):
        """Create the underlying data loader based on available backend."""
        if self._use_rust:
            return self._create_rust_loader()
        else:
            return self._create_python_loader()

    def _create_rust_loader(self):
        """Create Rust-backed loader."""
        rust_config = self.config.to_rust_config()

        dataset = RustEnfusionDataset(
            self.data_root,
            config=rust_config,
            split=self.split,
            session_ids=self.session_ids,
        )

        if self.config.use_batch_loader:
            return RustBatchLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                drop_last=self.config.drop_last,
            )
        else:
            return RustDataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                num_workers=0,  # Rust handles parallelism
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last,
            )

    def _create_python_loader(self):
        """Create Python-backed loader."""
        python_config = self.config.to_python_config()

        dataset = PythonEnfusionDataset(
            self.data_root,
            config=python_config,
            split=self.split,
            session_ids=self.session_ids,
        )

        return PythonDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            prefetch_factor=self.config.prefetch_factor,
        )

    @property
    def backend(self) -> str:
        """Get the current backend name."""
        return self._backend_name

    @property
    def is_rust(self) -> bool:
        """Check if using Rust backend."""
        return self._use_rust

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches."""
        for batch in self._loader:
            if self._performance_tracker:
                start_time = time.time()

            # Yield the batch
            yield batch

            if self._performance_tracker:
                load_time = time.time() - start_time
                batch_size = batch.get('frames', batch.get('controls')).shape[0]
                self._performance_tracker.record_batch(load_time, batch_size)

    def __len__(self) -> int:
        return len(self._loader)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self._performance_tracker:
            return self._performance_tracker.get_stats()
        return {}

    def reset_performance_stats(self):
        """Reset performance statistics."""
        if self._performance_tracker:
            self._performance_tracker.reset()

    def cache_stats(self) -> Dict[str, float]:
        """Get cache statistics (if available)."""
        if hasattr(self._loader, 'cache_stats'):
            return self._loader.cache_stats()
        if hasattr(self._loader, 'dataset') and hasattr(self._loader.dataset, 'cache_stats'):
            return self._loader.dataset.cache_stats()
        return {}

    def clear_cache(self):
        """Clear the data cache."""
        if hasattr(self._loader, 'clear_cache'):
            self._loader.clear_cache()
        elif hasattr(self._loader, 'dataset') and hasattr(self._loader.dataset, 'clear_cache'):
            self._loader.dataset.clear_cache()

    def close(self):
        """Clean up resources."""
        if self._go_client:
            self._go_client.close()


def create_hybrid_dataloaders(
    data_root: Union[str, Path],
    config: Optional[HybridLoaderConfig] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[HybridDataLoader, HybridDataLoader, HybridDataLoader]:
    """
    Create train/val/test data loaders with hybrid backend.

    Args:
        data_root: Root data directory
        config: Loader configuration
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or HybridLoaderConfig()
    data_root = Path(data_root)

    # Check for existing splits
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if train_dir.exists() and val_dir.exists():
        # Use existing split directories
        train_loader = HybridDataLoader(train_dir, config, split="train")

        # Disable shuffle for validation/test
        val_config = HybridLoaderConfig(
            **{k: v for k, v in config.__dict__.items() if k != 'shuffle'}
        )
        val_config.shuffle = False

        val_loader = HybridDataLoader(val_dir, val_config, split="val")
        test_loader = HybridDataLoader(
            test_dir if test_dir.exists() else val_dir,
            val_config,
            split="test"
        )
    else:
        # Create random split
        all_sessions = sorted(p.name for p in data_root.glob("session_*"))
        np.random.shuffle(all_sessions)

        n_sessions = len(all_sessions)
        n_train = int(n_sessions * train_ratio)
        n_val = int(n_sessions * val_ratio)

        train_sessions = all_sessions[:n_train]
        val_sessions = all_sessions[n_train:n_train + n_val]
        test_sessions = all_sessions[n_train + n_val:]

        train_loader = HybridDataLoader(
            data_root, config, split="train", session_ids=train_sessions
        )

        val_config = HybridLoaderConfig(
            **{k: v for k, v in config.__dict__.items() if k != 'shuffle'}
        )
        val_config.shuffle = False

        val_loader = HybridDataLoader(
            data_root, val_config, split="val", session_ids=val_sessions
        )
        test_loader = HybridDataLoader(
            data_root, val_config, split="test", session_ids=test_sessions
        )

    logger.info(
        f"Created hybrid dataloaders: "
        f"train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader


def get_best_loader(
    data_root: Union[str, Path],
    batch_size: int = 8,
    sequence_length: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    **kwargs
) -> HybridDataLoader:
    """
    Convenience function to get the best available loader.

    This automatically uses Rust when available, Python otherwise.

    Args:
        data_root: Root data directory
        batch_size: Batch size
        sequence_length: Number of frames per sequence
        image_size: Target image size (H, W)
        **kwargs: Additional config options

    Returns:
        HybridDataLoader configured with best available backend
    """
    config = HybridLoaderConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        image_size=image_size,
        **kwargs
    )

    return HybridDataLoader(data_root, config)


if __name__ == "__main__":
    # Test hybrid loader
    logging.basicConfig(level=logging.INFO)

    print(f"Rust available: {RUST_AVAILABLE}")
    print(f"Backend: {'Rust' if RUST_AVAILABLE else 'Python'}")

    # Create test config
    config = HybridLoaderConfig(
        sequence_length=8,
        image_size=(128, 128),
        batch_size=2,
        track_performance=True,
    )

    print(f"Config: {config}")
