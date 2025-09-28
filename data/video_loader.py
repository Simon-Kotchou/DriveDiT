"""
Video data loading utilities with zero dependencies.
Handles video frame sequences and temporal data loading.
"""

import torch
import torch.utils.data as data
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import random
import cv2
import numpy as np
from collections import deque
import threading
import queue


class FrameBuffer:
    """Memory-efficient frame buffer for video sequences."""
    
    def __init__(self, max_frames: int = 1000, max_memory_mb: float = 500.0):
        self.max_frames = max_frames
        self.max_memory_mb = max_memory_mb
        self.buffer = {}
        self.access_times = {}
        self.current_memory = 0.0
        self.access_counter = 0
        self._lock = threading.Lock()
    
    def _estimate_frame_memory(self, frame: torch.Tensor) -> float:
        """Estimate memory usage of a frame in MB."""
        return frame.numel() * frame.element_size() / (1024 * 1024)
    
    def _evict_lru(self) -> None:
        """Evict least recently used frames."""
        if not self.buffer:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        for key in sorted_keys:
            if len(self.buffer) <= self.max_frames // 2 and self.current_memory <= self.max_memory_mb * 0.8:
                break
            
            frame = self.buffer.pop(key)
            self.current_memory -= self._estimate_frame_memory(frame)
            del self.access_times[key]
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get frame from buffer."""
        with self._lock:
            if key in self.buffer:
                self.access_times[key] = self.access_counter
                self.access_counter += 1
                return self.buffer[key]
            return None
    
    def put(self, key: str, frame: torch.Tensor) -> None:
        """Add frame to buffer."""
        with self._lock:
            frame_memory = self._estimate_frame_memory(frame)
            
            # Check if we need to evict
            if (len(self.buffer) >= self.max_frames or 
                self.current_memory + frame_memory > self.max_memory_mb):
                self._evict_lru()
            
            self.buffer[key] = frame.clone()
            self.current_memory += frame_memory
            self.access_times[key] = self.access_counter
            self.access_counter += 1
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()
            self.access_times.clear()
            self.current_memory = 0.0
            self.access_counter = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                'num_frames': len(self.buffer),
                'memory_mb': self.current_memory,
                'memory_utilization': self.current_memory / self.max_memory_mb,
                'capacity_utilization': len(self.buffer) / self.max_frames
            }


class VideoDataset(data.Dataset):
    """Zero-dependency video dataset for loading frame sequences."""
    
    def __init__(
        self,
        data_root: Union[str, Path],
        sequence_length: int = 16,
        frame_skip: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        split: str = 'train',
        cache_frames: bool = True,
        load_controls: bool = True,
        load_depth: bool = False,
        supported_formats: List[str] = None
    ):
        """
        Initialize video dataset.
        
        Args:
            data_root: Root directory containing video data
            sequence_length: Number of frames per sequence
            frame_skip: Skip frames for temporal subsampling
            image_size: Target image size (H, W)
            split: Dataset split ('train', 'val', 'test')
            cache_frames: Whether to cache loaded frames
            load_controls: Whether to load control signals
            load_depth: Whether to load depth information
            supported_formats: List of supported image formats
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.split = split
        self.load_controls = load_controls
        self.load_depth = load_depth
        
        if supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        else:
            self.supported_formats = supported_formats
        
        # Initialize frame cache
        self.cache_frames = cache_frames
        if cache_frames:
            self.frame_buffer = FrameBuffer()
        
        # Load dataset metadata
        self.sequences = self._load_sequences()
        self.sequence_indices = self._build_sequence_indices()
        
        print(f"Loaded {len(self.sequence_indices)} sequences for {split} split")
    
    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load sequence metadata from dataset."""
        sequences = []
        
        # Look for metadata file
        metadata_path = self.data_root / f"{self.split}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            sequences = metadata.get('sequences', [])
        else:
            # Fallback: scan directories
            sequences = self._scan_directories()
        
        return sequences
    
    def _scan_directories(self) -> List[Dict[str, Any]]:
        """Scan directories for video sequences."""
        sequences = []
        
        # Look for split-specific subdirectory
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            split_dir = self.data_root
        
        # Find all subdirectories that contain images
        for seq_dir in split_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            
            # Find image files
            frame_files = []
            for ext in self.supported_formats:
                frame_files.extend(list(seq_dir.glob(f"*{ext}")))
            
            if len(frame_files) >= self.sequence_length:
                frame_files.sort()  # Ensure temporal order
                
                sequence = {
                    'sequence_id': seq_dir.name,
                    'frame_paths': [str(f) for f in frame_files],
                    'num_frames': len(frame_files)
                }
                
                # Look for control signals
                if self.load_controls:
                    control_path = seq_dir / 'controls.json'
                    if control_path.exists():
                        sequence['control_path'] = str(control_path)
                
                # Look for depth data
                if self.load_depth:
                    depth_dir = seq_dir / 'depth'
                    if depth_dir.exists():
                        depth_files = list(depth_dir.glob("*.png"))
                        if len(depth_files) == len(frame_files):
                            sequence['depth_paths'] = [str(f) for f in sorted(depth_files)]
                
                sequences.append(sequence)
        
        return sequences
    
    def _build_sequence_indices(self) -> List[Tuple[int, int]]:
        """Build indices for valid sequences."""
        indices = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            num_frames = sequence['num_frames']
            max_start = num_frames - (self.sequence_length - 1) * self.frame_skip - 1
            
            # Create multiple starting points for each sequence
            for start_idx in range(0, max_start + 1, self.sequence_length // 2):
                indices.append((seq_idx, start_idx))
        
        return indices
    
    def _load_frame(self, frame_path: str) -> torch.Tensor:
        """Load a single frame from disk."""
        # Check cache first
        if self.cache_frames:
            cached_frame = self.frame_buffer.get(frame_path)
            if cached_frame is not None:
                return cached_frame
        
        # Load from disk using OpenCV
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load frame: {frame_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if frame.shape[:2] != self.image_size:
            frame = cv2.resize(frame, self.image_size[::-1])  # OpenCV uses (W, H)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Cache the frame
        if self.cache_frames:
            self.frame_buffer.put(frame_path, frame_tensor)
        
        return frame_tensor
    
    def _load_controls(self, control_path: str, start_idx: int) -> torch.Tensor:
        """Load control signals for sequence."""
        with open(control_path, 'r') as f:
            controls_data = json.load(f)
        
        controls = controls_data.get('controls', [])
        if not controls:
            # Return dummy controls if none available
            return torch.zeros(self.sequence_length, 4)
        
        # Extract control sequence
        control_sequence = []
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * self.frame_skip
            if frame_idx < len(controls):
                control = controls[frame_idx]
                # Expect [steering, acceleration, goal_x, goal_y]
                control_vec = [
                    control.get('steering', 0.0),
                    control.get('acceleration', 0.0),
                    control.get('goal_x', 0.0),
                    control.get('goal_y', 0.0)
                ]
            else:
                control_vec = [0.0, 0.0, 0.0, 0.0]
            
            control_sequence.append(control_vec)
        
        return torch.tensor(control_sequence, dtype=torch.float32)
    
    def _load_depth(self, depth_paths: List[str], start_idx: int) -> torch.Tensor:
        """Load depth maps for sequence."""
        depth_sequence = []
        
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * self.frame_skip
            if frame_idx < len(depth_paths):
                depth_path = depth_paths[frame_idx]
                
                # Load depth map (assuming 16-bit PNG)
                depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_map is None:
                    depth_map = np.zeros(self.image_size, dtype=np.uint16)
                
                # Resize if needed
                if depth_map.shape != self.image_size:
                    depth_map = cv2.resize(depth_map, self.image_size[::-1])
                
                # Convert to float and normalize
                depth_tensor = torch.from_numpy(depth_map).float() / 65535.0
                depth_sequence.append(depth_tensor.unsqueeze(0))  # Add channel dimension
            else:
                # Dummy depth map
                depth_sequence.append(torch.zeros(1, *self.image_size))
        
        return torch.stack(depth_sequence, dim=0)  # [T, 1, H, W]
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a video sequence."""
        seq_idx, start_idx = self.sequence_indices[idx]
        sequence = self.sequences[seq_idx]
        
        # Load frame sequence
        frames = []
        frame_paths = sequence['frame_paths']
        
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * self.frame_skip
            if frame_idx < len(frame_paths):
                frame = self._load_frame(frame_paths[frame_idx])
            else:
                # Repeat last frame if sequence is too short
                frame = self._load_frame(frame_paths[-1])
            frames.append(frame)
        
        frames_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
        
        # Prepare return dictionary
        sample = {
            'frames': frames_tensor,
            'sequence_id': sequence['sequence_id'],
            'start_idx': start_idx
        }
        
        # Load control signals if requested
        if self.load_controls and 'control_path' in sequence:
            controls = self._load_controls(sequence['control_path'], start_idx)
            sample['controls'] = controls
        elif self.load_controls:
            # Dummy controls
            sample['controls'] = torch.zeros(self.sequence_length, 4)
        
        # Load depth maps if requested
        if self.load_depth and 'depth_paths' in sequence:
            depth = self._load_depth(sequence['depth_paths'], start_idx)
            sample['depth'] = depth
        
        return sample
    
    def get_sequence_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sequence."""
        seq_idx, start_idx = self.sequence_indices[idx]
        sequence = self.sequences[seq_idx]
        
        return {
            'sequence_id': sequence['sequence_id'],
            'start_idx': start_idx,
            'num_frames': sequence['num_frames'],
            'sequence_length': self.sequence_length,
            'frame_skip': self.frame_skip
        }


class VideoLoader:
    """Wrapper around DataLoader with video-specific functionality."""
    
    def __init__(
        self,
        dataset: VideoDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        prefetch_factor: int = 2
    ):
        """
        Initialize video data loader.
        
        Args:
            dataset: Video dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            prefetch_factor: Number of batches to prefetch per worker
        """
        self.dataset = dataset
        
        # Import collate function
        from .collate import custom_collate_fn
        
        self.dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate_fn,
            persistent_workers=num_workers > 0
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get frame cache statistics."""
        if hasattr(self.dataset, 'frame_buffer'):
            return self.dataset.frame_buffer.stats()
        return {}
    
    def clear_cache(self) -> None:
        """Clear frame cache."""
        if hasattr(self.dataset, 'frame_buffer'):
            self.dataset.frame_buffer.clear()