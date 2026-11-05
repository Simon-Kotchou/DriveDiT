"""
Efficient data pipeline for large-scale video processing.
Zero-dependency implementation using only torch, einops, and cv2.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import mmap
import struct
from typing import Dict, List, Tuple, Optional, Iterator
from einops import rearrange
import gc
import os


class VideoMemoryMap:
    """Memory-mapped video storage for efficient random access."""
    
    def __init__(self, video_path: str, cache_dir: str = None):
        self.video_path = Path(video_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.video_path.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Generate cache files
        self.index_path = self.cache_dir / f"{self.video_path.stem}.idx"
        self.data_path = self.cache_dir / f"{self.video_path.stem}.dat"
        
        if not self.index_path.exists() or not self.data_path.exists():
            self._build_cache()
        
        self._load_index()
        self._open_data_file()
    
    def _build_cache(self):
        """Build memory-mapped cache from video file."""
        print(f"Building cache for {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Write metadata and frame offsets
        with open(self.index_path, 'wb') as idx_file:
            # Header: frame_count, fps, width, height
            idx_file.write(struct.pack('IFII', frame_count, fps, width, height))
            
            with open(self.data_path, 'wb') as data_file:
                offset = 0
                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB and compress
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_bytes = cv2.imencode('.jpg', frame_rgb, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                    
                    # Write frame offset and size to index
                    idx_file.write(struct.pack('QI', offset, len(frame_bytes)))
                    
                    # Write frame data
                    data_file.write(frame_bytes)
                    offset += len(frame_bytes)
                    
                    if i % 1000 == 0:
                        print(f"Processed {i}/{frame_count} frames")
        
        cap.release()
        print(f"Cache built: {frame_count} frames")
    
    def _load_index(self):
        """Load frame index from cache."""
        with open(self.index_path, 'rb') as f:
            header = struct.unpack('IFII', f.read(16))
            self.frame_count, self.fps, self.width, self.height = header
            
            self.frame_offsets = []
            self.frame_sizes = []
            
            for _ in range(self.frame_count):
                offset, size = struct.unpack('QI', f.read(12))
                self.frame_offsets.append(offset)
                self.frame_sizes.append(size)
    
    def _open_data_file(self):
        """Open memory-mapped data file."""
        self.data_file = open(self.data_path, 'rb')
        self.mmap_data = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get_frame(self, idx: int) -> np.ndarray:
        """Get frame by index with zero-copy access."""
        if idx >= self.frame_count:
            raise IndexError(f"Frame {idx} >= {self.frame_count}")
        
        offset = self.frame_offsets[idx]
        size = self.frame_sizes[idx]
        
        # Direct memory access
        frame_bytes = self.mmap_data[offset:offset + size]
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_batch(self, indices: List[int]) -> torch.Tensor:
        """Get batch of frames efficiently."""
        frames = []
        for idx in indices:
            frame = self.get_frame(idx)
            frames.append(frame)
        
        # Stack and convert to tensor
        batch = np.stack(frames, axis=0)  # [B, H, W, 3]
        tensor = torch.from_numpy(batch).float() / 255.0
        return rearrange(tensor, 'b h w c -> b c h w')
    
    def __len__(self):
        return self.frame_count
    
    def __del__(self):
        if hasattr(self, 'mmap_data'):
            self.mmap_data.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()


class DrivingDataset(Dataset):
    """Efficient dataset for driving video sequences."""
    
    def __init__(
        self,
        video_paths: List[str],
        sequence_length: int = 16,
        image_size: Tuple[int, int] = (256, 256),
        cache_dir: str = None,
        stride: int = 1,
        augment: bool = True
    ):
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.stride = stride
        self.augment = augment
        
        # Initialize memory-mapped videos
        self.videos = []
        self.video_lengths = []
        self.cumulative_lengths = [0]
        
        for video_path in video_paths:
            video_mmap = VideoMemoryMap(video_path, cache_dir)
            self.videos.append(video_mmap)
            
            # Calculate valid sequence starts
            valid_length = max(0, len(video_mmap) - sequence_length + 1)
            self.video_lengths.append(valid_length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_length)
        
        self.total_sequences = self.cumulative_lengths[-1]
        print(f"Dataset: {len(video_paths)} videos, {self.total_sequences} sequences")
    
    def __len__(self):
        return self.total_sequences
    
    def _get_video_and_start(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (video_idx, start_frame)."""
        for i, cumsum in enumerate(self.cumulative_lengths[1:]):
            if idx < cumsum:
                video_idx = i
                start_frame = idx - self.cumulative_lengths[i]
                return video_idx, start_frame
        raise IndexError(f"Index {idx} out of range")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, start_frame = self._get_video_and_start(idx)
        video = self.videos[video_idx]
        
        # Get sequence indices
        frame_indices = list(range(
            start_frame, 
            start_frame + self.sequence_length * self.stride, 
            self.stride
        ))
        
        # Load frames efficiently
        frames = video.get_batch(frame_indices)  # [T, C, H, W]
        
        # Resize if needed
        if frames.shape[-2:] != self.image_size:
            frames = F.interpolate(
                frames, 
                size=self.image_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Data augmentation
        if self.augment and torch.rand(1) > 0.5:
            frames = self._augment_sequence(frames)
        
        return {
            'frames': frames,  # [T, C, H, W]
            'video_idx': video_idx,
            'start_frame': start_frame
        }
    
    def _augment_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Temporal-consistent augmentation."""
        # Random brightness/contrast (consistent across time)
        if torch.rand(1) > 0.7:
            brightness = 0.8 + 0.4 * torch.rand(1)
            contrast = 0.8 + 0.4 * torch.rand(1)
            frames = torch.clamp(frames * contrast + brightness - 1, 0, 1)
        
        # Random horizontal flip (consistent across time)
        if torch.rand(1) > 0.5:
            frames = torch.flip(frames, dims=[-1])
        
        return frames


class StreamingDataLoader:
    """Memory-efficient streaming data loader for large datasets."""
    
    def __init__(
        self,
        dataset: DrivingDataset,
        batch_size: int = 8,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        
        # Create DataLoader with memory optimizations
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=True,
            drop_last=True
        )
    
    def __iter__(self):
        for batch in self.dataloader:
            # Move to device efficiently
            yield {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
    
    def __len__(self):
        return len(self.dataloader)


class DatasetSplitter:
    """Efficient train/val/test splitting for video datasets."""
    
    @staticmethod
    def split_by_video(
        video_paths: List[str], 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split videos ensuring no temporal leakage."""
        np.random.shuffle(video_paths)
        
        n_videos = len(video_paths)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        
        train_videos = video_paths[:n_train]
        val_videos = video_paths[n_train:n_train + n_val]
        test_videos = video_paths[n_train + n_val:]
        
        return train_videos, val_videos, test_videos
    
    @staticmethod
    def create_datasets(
        video_paths: List[str],
        sequence_length: int = 16,
        image_size: Tuple[int, int] = (256, 256),
        cache_dir: str = None,
        **kwargs
    ) -> Tuple[DrivingDataset, DrivingDataset, DrivingDataset]:
        """Create train/val/test datasets."""
        train_paths, val_paths, test_paths = DatasetSplitter.split_by_video(video_paths)
        
        train_dataset = DrivingDataset(
            train_paths, sequence_length, image_size, cache_dir, augment=True, **kwargs
        )
        val_dataset = DrivingDataset(
            val_paths, sequence_length, image_size, cache_dir, augment=False, **kwargs
        )
        test_dataset = DrivingDataset(
            test_paths, sequence_length, image_size, cache_dir, augment=False, **kwargs
        )
        
        return train_dataset, val_dataset, test_dataset


class MemoryMonitor:
    """Monitor GPU memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
    
    def update(self):
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0
    
    def cleanup(self):
        """Force memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __repr__(self):
        return f"Memory: {self.current_memory:.2f}GB (peak: {self.peak_memory:.2f}GB)"


def create_efficient_pipeline(
    video_dir: str,
    batch_size: int = 8,
    sequence_length: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    cache_dir: str = None,
    num_workers: int = 4
) -> Tuple[StreamingDataLoader, StreamingDataLoader, StreamingDataLoader]:
    """Create complete data pipeline for training."""
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_paths = []
    
    for ext in video_extensions:
        video_paths.extend(list(Path(video_dir).rglob(f'*{ext}')))
    
    video_paths = [str(p) for p in video_paths]
    print(f"Found {len(video_paths)} videos")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = DatasetSplitter.create_datasets(
        video_paths, sequence_length, image_size, cache_dir
    )
    
    # Create data loaders
    train_loader = StreamingDataLoader(train_dataset, batch_size, num_workers)
    val_loader = StreamingDataLoader(val_dataset, batch_size, num_workers)
    test_loader = StreamingDataLoader(test_dataset, batch_size, num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    video_dir = "/path/to/driving/videos"
    cache_dir = "/path/to/cache"
    
    train_loader, val_loader, test_loader = create_efficient_pipeline(
        video_dir=video_dir,
        batch_size=16,
        sequence_length=16,
        image_size=(256, 256),
        cache_dir=cache_dir,
        num_workers=8
    )
    
    # Memory monitoring
    memory_monitor = MemoryMonitor()
    
    # Test iteration
    for i, batch in enumerate(train_loader):
        frames = batch['frames']  # [B, T, C, H, W]
        print(f"Batch {i}: {frames.shape}")
        
        memory_monitor.update()
        print(memory_monitor)
        
        if i >= 5:  # Test first few batches
            break
    
    memory_monitor.cleanup()