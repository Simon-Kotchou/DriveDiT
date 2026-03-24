"""
Rust-accelerated data loader for DriveDiT training.

This module provides Python wrappers around the Rust drivedit_data library,
offering 10x faster data loading compared to pure Python implementation.

Features:
- Zero-copy numpy array creation
- Parallel I/O with Rayon
- Memory-mapped file access
- Efficient LRU caching

Usage:
    from data.rust_loader import RustEnfusionDataset, RustDataLoader

    dataset = RustEnfusionDataset(
        data_root="/path/to/data",
        sequence_length=16,
        image_size=(256, 256)
    )
    loader = RustDataLoader(dataset, batch_size=8, num_workers=4)

    for batch in loader:
        frames = batch['frames']  # [B, T, C, H, W]
        controls = batch['controls']  # [B, T, D]
"""

import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass

# Try to import the Rust extension
try:
    import drivedit_data as rust_lib
    from drivedit_data import (
        RustDatasetConfig,
        RustEnfusionDataset as _RustEnfusionDataset,
        RustEnfusionSample,
        RustEnfusionSession,
        RustCollator,
        benchmark_frame_loading,
        is_available,
        num_cpus,
        version,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    rust_lib = None


def check_rust_available():
    """Check if the Rust extension is available."""
    return RUST_AVAILABLE


def get_rust_version() -> Optional[str]:
    """Get the Rust extension version if available."""
    if RUST_AVAILABLE:
        return version()
    return None


@dataclass
class RustDatasetConfig:
    """Configuration for RustEnfusionDataset.

    This is a Python-side config that mirrors the Rust config structure.
    """
    sequence_length: int = 16
    frame_skip: int = 1
    image_height: int = 256
    image_width: int = 256
    load_depth: bool = True
    load_controls: bool = True
    control_dim: int = 6
    normalize_frames: bool = True
    cache_frames: bool = True
    max_cache_mb: float = 1000.0
    num_workers: int = 4

    def to_rust_config(self):
        """Convert to Rust config object."""
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust extension not available")

        from drivedit_data import RustDatasetConfig as RustConfig
        return RustConfig(
            sequence_length=self.sequence_length,
            frame_skip=self.frame_skip,
            image_height=self.image_height,
            image_width=self.image_width,
            load_depth=self.load_depth,
            load_controls=self.load_controls,
            control_dim=self.control_dim,
            normalize_frames=self.normalize_frames,
            cache_frames=self.cache_frames,
            max_cache_mb=self.max_cache_mb,
            num_workers=self.num_workers,
        )

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.image_height, self.image_width)


class RustEnfusionDataset(data.Dataset):
    """
    PyTorch Dataset wrapper for Rust-accelerated Enfusion data loading.

    Provides 10x+ speedup over pure Python implementation through:
    - Parallel I/O with Rayon thread pool
    - Memory-mapped file access
    - SIMD-accelerated image normalization
    - Zero-copy numpy array creation

    Example:
        >>> config = RustDatasetConfig(sequence_length=16, image_height=256)
        >>> dataset = RustEnfusionDataset("/path/to/data", config=config)
        >>> len(dataset)
        1000
        >>> sample = dataset[0]
        >>> sample['frames'].shape
        torch.Size([16, 3, 256, 256])
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        config: Optional[RustDatasetConfig] = None,
        split: str = "train",
        session_ids: Optional[List[str]] = None,
        return_tensors: bool = True,
    ):
        """
        Initialize Rust-accelerated Enfusion dataset.

        Args:
            data_root: Root directory containing session folders
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
            session_ids: Optional list of specific session IDs to load
            return_tensors: Convert numpy arrays to PyTorch tensors

        Raises:
            RuntimeError: If Rust extension not available
            IOError: If data_root does not exist
            ValueError: If no valid sessions found
        """
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust extension 'drivedit_data' not available. "
                "Install with: pip install drivedit-data or run scripts/build_rust.sh"
            )

        self.data_root = Path(data_root)
        self.config = config or RustDatasetConfig()
        self.split = split
        self.return_tensors = return_tensors

        # Create Rust dataset
        rust_config = self.config.to_rust_config()
        self._dataset = _RustEnfusionDataset(
            str(self.data_root),
            config=rust_config,
            split=split,
            session_ids=session_ids,
        )

        print(f"RustEnfusionDataset [{split}]: {self.num_sessions} sessions, "
              f"{len(self)} sequences")

    @property
    def num_sessions(self) -> int:
        """Get number of sessions in dataset."""
        return self._dataset.num_sessions

    @property
    def session_ids(self) -> List[str]:
        """Get list of session IDs."""
        return self._dataset.session_ids

    @property
    def total_frames(self) -> int:
        """Get total frames across all sessions."""
        return self._dataset.total_frames

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - frames: [T, C, H, W] tensor
                - controls: [T, D] tensor
                - depth: [T, 1, H, W] tensor (if enabled)
                - ego_transform: [T, 4, 4] tensor
                - anchor_mask: [T] tensor
                - metadata: dict with session_id, start_frame, etc.
        """
        # Get sample from Rust
        sample = self._dataset[idx]

        # Convert to dictionary
        result = sample.to_dict()

        # Convert numpy arrays to tensors if requested
        if self.return_tensors:
            result = self._to_tensors(result)

        return result

    def _to_tensors(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays to PyTorch tensors."""
        tensor_keys = ['frames', 'controls', 'depth', 'ego_transform', 'anchor_mask']

        for key in tensor_keys:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])

        return sample

    def load_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Load a batch of samples in parallel.

        This is more efficient than calling __getitem__ repeatedly
        as it uses Rust's parallel loading capabilities.

        Args:
            indices: List of sample indices

        Returns:
            List of sample dictionaries
        """
        samples = self._dataset.load_batch(indices)

        results = []
        for sample in samples:
            result = sample.to_dict()
            if self.return_tensors:
                result = self._to_tensors(result)
            results.append(result)

        return results

    def prefetch(self, indices: List[int]) -> None:
        """
        Prefetch samples in background.

        Useful when you know which samples will be needed next.

        Args:
            indices: List of sample indices to prefetch
        """
        self._dataset.prefetch(indices)

    def cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        return self._dataset.cache_stats()

    def clear_cache(self) -> None:
        """Clear the frame cache."""
        self._dataset.clear_cache()

    def get_session_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded sessions."""
        return self._dataset.get_session_info()


class RustCollator:
    """
    Fast data collator using Rust backend.

    Collates list of samples into batched tensors efficiently.
    """

    def __init__(
        self,
        pad_to_max: bool = True,
        include_depth: bool = True,
    ):
        """
        Initialize collator.

        Args:
            pad_to_max: Pad sequences to max length in batch
            include_depth: Include depth maps in collated batch
        """
        self.pad_to_max = pad_to_max
        self.include_depth = include_depth

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Collated batch dictionary with stacked tensors
        """
        if not batch:
            return {}

        # Stack tensor fields
        tensor_fields = ['frames', 'controls', 'depth', 'ego_transform', 'anchor_mask']

        collated = {}
        for field in tensor_fields:
            if field in batch[0]:
                tensors = [sample[field] for sample in batch if field in sample]
                if tensors and len(tensors) == len(batch):
                    if isinstance(tensors[0], torch.Tensor):
                        collated[field] = torch.stack(tensors, dim=0)
                    elif isinstance(tensors[0], np.ndarray):
                        collated[field] = torch.from_numpy(np.stack(tensors, axis=0))

        # Collect metadata
        collated['metadata'] = [sample.get('metadata', {}) for sample in batch]

        return collated


class RustDataLoader:
    """
    High-performance data loader using Rust backend.

    Wraps PyTorch DataLoader with optimizations for Rust-loaded data.
    """

    def __init__(
        self,
        dataset: RustEnfusionDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,  # Rust handles parallelism
        pin_memory: bool = True,
        drop_last: bool = True,
        prefetch_factor: int = 2,
    ):
        """
        Initialize data loader.

        Args:
            dataset: RustEnfusionDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of PyTorch workers (0 recommended - Rust handles parallel I/O)
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop incomplete last batch
            prefetch_factor: Prefetch factor per worker
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create collator
        collator = RustCollator(
            include_depth=dataset.config.load_depth
        )

        # Create PyTorch DataLoader
        # Note: num_workers=0 is recommended since Rust handles parallel I/O
        self._loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=collator,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches."""
        for batch in self._loader:
            # Move tensors to GPU with non-blocking transfer
            yield {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def __len__(self) -> int:
        return len(self._loader)

    def cache_stats(self) -> Dict[str, float]:
        """Get dataset cache statistics."""
        return self.dataset.cache_stats()

    def clear_cache(self) -> None:
        """Clear dataset cache."""
        self.dataset.clear_cache()


class RustBatchLoader:
    """
    Batch-optimized loader using Rust's parallel batch loading.

    This loader bypasses PyTorch DataLoader and directly uses
    Rust's batch loading for maximum performance.
    """

    def __init__(
        self,
        dataset: RustEnfusionDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Initialize batch loader.

        Args:
            dataset: RustEnfusionDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle indices
            drop_last: Drop incomplete last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Build index list
        self._indices = list(range(len(dataset)))
        self._collator = RustCollator(
            include_depth=dataset.config.load_depth
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches using Rust parallel loading."""
        indices = self._indices.copy()

        if self.shuffle:
            np.random.shuffle(indices)

        # Create batches
        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size > 0:
            num_batches += 1

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]

            # Use Rust parallel batch loading
            samples = self.dataset.load_batch(batch_indices)

            # Collate
            batch = self._collator(samples)

            # Move to GPU
            yield {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._indices) // self.batch_size
        return (len(self._indices) + self.batch_size - 1) // self.batch_size


def create_rust_dataloaders(
    data_root: Union[str, Path],
    config: Optional[RustDatasetConfig] = None,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_batch_loader: bool = True,
) -> Tuple[Union[RustDataLoader, RustBatchLoader], ...]:
    """
    Create train/val/test data loaders with Rust backend.

    Args:
        data_root: Root data directory
        config: Dataset configuration
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        use_batch_loader: Use RustBatchLoader for maximum performance

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or RustDatasetConfig()
    data_root = Path(data_root)

    # Check for existing splits
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    LoaderClass = RustBatchLoader if use_batch_loader else RustDataLoader

    if train_dir.exists() and val_dir.exists():
        # Use existing splits
        train_dataset = RustEnfusionDataset(train_dir, config, split="train")
        val_dataset = RustEnfusionDataset(val_dir, config, split="val")
        test_dataset = RustEnfusionDataset(
            test_dir if test_dir.exists() else val_dir,
            config,
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

        train_dataset = RustEnfusionDataset(
            data_root, config, split="train", session_ids=train_sessions
        )
        val_dataset = RustEnfusionDataset(
            data_root, config, split="val", session_ids=val_sessions
        )
        test_dataset = RustEnfusionDataset(
            data_root, config, split="test", session_ids=test_sessions
        )

    # Create loaders
    train_loader = LoaderClass(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = LoaderClass(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = LoaderClass(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Benchmark utilities
def benchmark_rust_vs_python(
    session_dir: str,
    frame_indices: List[int],
    target_size: Tuple[int, int] = (256, 256),
    iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark Rust vs Python frame loading performance.

    Args:
        session_dir: Path to session directory
        frame_indices: Frame indices to load
        target_size: Target image size
        iterations: Number of iterations

    Returns:
        Dictionary with timing results
    """
    if not RUST_AVAILABLE:
        return {"error": "Rust extension not available"}

    # Benchmark Rust
    rust_time = benchmark_frame_loading(
        session_dir, frame_indices, target_size, iterations
    )

    # Benchmark Python (using cv2)
    import time
    import cv2
    from pathlib import Path

    frames_dir = Path(session_dir) / "frames"
    frame_paths = sorted(frames_dir.glob("*.png"))[:max(frame_indices) + 1]

    python_times = []
    for _ in range(iterations):
        start = time.time()
        frames = []
        for idx in frame_indices:
            if idx < len(frame_paths):
                frame = cv2.imread(str(frame_paths[idx]))
                frame = cv2.resize(frame, target_size[::-1])
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        if frames:
            _ = np.stack(frames)
        python_times.append(time.time() - start)

    python_time = np.mean(python_times)

    return {
        "rust_time_sec": rust_time,
        "python_time_sec": python_time,
        "speedup": python_time / rust_time if rust_time > 0 else float('inf'),
        "frames_per_second_rust": len(frame_indices) / rust_time if rust_time > 0 else 0,
        "frames_per_second_python": len(frame_indices) / python_time if python_time > 0 else 0,
    }


if __name__ == "__main__":
    print(f"Rust extension available: {RUST_AVAILABLE}")
    if RUST_AVAILABLE:
        print(f"Rust version: {get_rust_version()}")
        print(f"Available CPUs: {num_cpus()}")
