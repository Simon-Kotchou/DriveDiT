"""
Enfusion capture data loader for DriveDiT training.
PyTorch Dataset implementation for loading Enfusion session data.
Zero-dependency implementation using only torch, numpy, and cv2.

Enfusion Session Structure:
    session_XXXX/
    ├── telemetry.csv          # Vehicle state at 5Hz
    ├── frames/                # PNG frames (if captured)
    │   ├── frame_000001.png
    │   └── ...
    ├── capture.enfcap         # Binary format (future)
    ├── depth/                 # Depth maps (if captured)
    │   ├── depth_000001.npz
    │   └── ...
    ├── scene/                 # Scene graphs (if captured)
    │   ├── scene_000001.json
    │   └── ...
    └── session_info.txt       # Metadata

Output Format (for DriveDiT):
    {
        'frames': torch.Tensor,      # [T, C, H, W] float32 [0,1]
        'controls': torch.Tensor,    # [T, 6] float32 [-1,1]
        'depth': torch.Tensor,       # [T, 1, H, W] float32 (optional)
        'ego_transform': torch.Tensor, # [T, 4, 4] float32
        'scene_entities': list,      # List of entity dicts
        'anchor_mask': torch.Tensor, # [T] bool
        'metadata': dict
    }
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass, field
import warnings
import random

from .enfusion_preprocessing import (
    EnfusionTelemetryConfig,
    EnfusionTelemetryParser,
    EnfusionControlNormalizer,
    EnfusionFrameProcessor,
    EnfusionDepthProcessor,
    EnfusionSceneParser,
    EnfusionQualityFilter,
    EnfusionAnchorDetector,
    create_enfusion_preprocessors
)


@dataclass
class EnfusionDatasetConfig:
    """Configuration for Enfusion dataset."""

    # Sequence configuration
    sequence_length: int = 16
    frame_skip: int = 1
    image_size: Tuple[int, int] = (256, 256)

    # Data loading options
    load_depth: bool = True
    load_scene: bool = True
    load_controls: bool = True

    # Control signal dimension
    control_dim: int = 6

    # Preprocessing options
    normalize_frames: bool = True
    normalize_controls: bool = True

    # Quality filtering
    min_quality_score: float = 0.5
    filter_sessions: bool = True

    # Caching options
    cache_frames: bool = True
    cache_telemetry: bool = True
    max_cache_memory_mb: float = 1000.0

    # Depth options
    max_depth: float = 100.0
    depth_format: str = "npz"

    # Telemetry configuration
    telemetry_config: EnfusionTelemetryConfig = field(
        default_factory=EnfusionTelemetryConfig
    )

    # Augmentation
    augment: bool = True
    horizontal_flip_prob: float = 0.5
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)


class EnfusionSession:
    """Represents a single Enfusion capture session."""

    def __init__(
        self,
        session_dir: Union[str, Path],
        config: EnfusionDatasetConfig
    ):
        """
        Initialize session.

        Args:
            session_dir: Path to session directory
            config: Dataset configuration
        """
        self.session_dir = Path(session_dir)
        self.config = config

        # Validate session structure
        self._validate_session()

        # Load session metadata
        self.metadata = self._load_metadata()

        # Initialize frame paths
        self.frame_paths = self._get_frame_paths()
        self.depth_paths = self._get_depth_paths() if config.load_depth else []
        self.scene_paths = self._get_scene_paths() if config.load_scene else []

        # Initialize preprocessors
        self.telemetry_parser = EnfusionTelemetryParser(config.telemetry_config)
        self.control_normalizer = EnfusionControlNormalizer(config.telemetry_config)
        self.frame_processor = EnfusionFrameProcessor(
            target_size=config.image_size,
            normalize=config.normalize_frames
        )
        self.depth_processor = EnfusionDepthProcessor(
            target_size=config.image_size,
            max_depth=config.max_depth,
            depth_format=config.depth_format
        ) if config.load_depth else None
        self.scene_parser = EnfusionSceneParser() if config.load_scene else None
        self.anchor_detector = EnfusionAnchorDetector()

        # Load and cache telemetry
        self._telemetry = None
        self._controls = None
        self._ego_transforms = None

    def _validate_session(self):
        """Validate session directory structure."""
        if not self.session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {self.session_dir}")

        # Check for required files
        telemetry_path = self.session_dir / "telemetry.csv"
        frames_dir = self.session_dir / "frames"

        if not telemetry_path.exists():
            warnings.warn(f"Telemetry not found: {telemetry_path}")

        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load session metadata."""
        metadata = {
            'session_id': self.session_dir.name,
            'path': str(self.session_dir)
        }

        # Try to load session_info.txt
        info_path = self.session_dir / "session_info.txt"
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip()] = value.strip()
            except Exception:
                pass

        # Try to load JSON metadata
        json_path = self.session_dir / "metadata.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    metadata.update(json.load(f))
            except Exception:
                pass

        return metadata

    def _get_frame_paths(self) -> List[Path]:
        """Get sorted list of frame paths."""
        frames_dir = self.session_dir / "frames"
        if not frames_dir.exists():
            return []

        # Support multiple image formats
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        frame_paths = []
        for pattern in patterns:
            frame_paths.extend(frames_dir.glob(pattern))

        return sorted(frame_paths)

    def _get_depth_paths(self) -> List[Path]:
        """Get sorted list of depth map paths."""
        depth_dir = self.session_dir / "depth"
        if not depth_dir.exists():
            return []

        patterns = ["*.npz", "*.png", "*.exr"]
        depth_paths = []
        for pattern in patterns:
            depth_paths.extend(depth_dir.glob(pattern))

        return sorted(depth_paths)

    def _get_scene_paths(self) -> List[Path]:
        """Get sorted list of scene graph paths."""
        scene_dir = self.session_dir / "scene"
        if not scene_dir.exists():
            return []

        return sorted(scene_dir.glob("*.json"))

    @property
    def num_frames(self) -> int:
        """Get number of frames in session."""
        return len(self.frame_paths)

    @property
    def telemetry(self) -> Dict[str, np.ndarray]:
        """Get telemetry data (cached)."""
        if self._telemetry is None:
            telemetry_path = self.session_dir / "telemetry.csv"
            if telemetry_path.exists():
                self._telemetry = self.telemetry_parser.parse_csv(telemetry_path)
                # Interpolate to frame rate
                self._telemetry = self.telemetry_parser.interpolate_to_frame_rate(
                    self._telemetry, self.num_frames
                )
            else:
                # Create dummy telemetry
                self._telemetry = self._create_dummy_telemetry()

        return self._telemetry

    @property
    def controls(self) -> torch.Tensor:
        """Get normalized control signals (cached)."""
        if self._controls is None:
            self._controls = self.control_normalizer.normalize_controls(
                self.telemetry, self.config.control_dim
            )
        return self._controls

    @property
    def ego_transforms(self) -> torch.Tensor:
        """Get ego vehicle transforms (cached)."""
        if self._ego_transforms is None:
            position = self.telemetry.get('position', np.zeros((self.num_frames, 3)))
            rotation = self.telemetry.get('rotation', np.zeros((self.num_frames, 3)))
            transforms = self.telemetry_parser.compute_ego_transform(position, rotation)
            self._ego_transforms = torch.from_numpy(transforms)
        return self._ego_transforms

    def _create_dummy_telemetry(self) -> Dict[str, np.ndarray]:
        """Create dummy telemetry when none exists."""
        N = self.num_frames
        return {
            'timestamp': np.linspace(0, N / 30.0, N).astype(np.float32),
            'position': np.zeros((N, 3), dtype=np.float32),
            'rotation': np.zeros((N, 3), dtype=np.float32),
            'velocity': np.zeros((N, 3), dtype=np.float32),
            'steering': np.zeros(N, dtype=np.float32),
            'throttle': np.zeros(N, dtype=np.float32),
            'brake': np.zeros(N, dtype=np.float32),
            'speed': np.zeros(N, dtype=np.float32)
        }

    def get_valid_start_indices(self, sequence_length: int, frame_skip: int = 1) -> List[int]:
        """Get valid starting indices for sequences."""
        total_frames_needed = (sequence_length - 1) * frame_skip + 1
        max_start = self.num_frames - total_frames_needed

        if max_start < 0:
            return []

        return list(range(max_start + 1))


class EnfusionDataset(Dataset):
    """
    PyTorch Dataset for Enfusion capture data.

    Loads Enfusion session data and converts to DriveDiT training format.
    Supports multi-session loading, quality filtering, and data augmentation.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        config: Optional[EnfusionDatasetConfig] = None,
        split: str = "train",
        session_ids: Optional[List[str]] = None
    ):
        """
        Initialize Enfusion dataset.

        Args:
            data_root: Root directory containing session folders
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
            session_ids: Optional list of specific session IDs to load
        """
        self.data_root = Path(data_root)
        self.config = config or EnfusionDatasetConfig()
        self.split = split

        # Discover and load sessions
        self.sessions = self._discover_sessions(session_ids)

        # Build sequence index
        self.sequence_indices = self._build_sequence_indices()

        # Quality filter
        if self.config.filter_sessions:
            self._filter_by_quality()

        print(f"EnfusionDataset [{split}]: {len(self.sessions)} sessions, "
              f"{len(self.sequence_indices)} sequences")

    def _discover_sessions(
        self,
        session_ids: Optional[List[str]] = None
    ) -> List[EnfusionSession]:
        """Discover and load sessions."""
        sessions = []

        # Find all session directories
        if session_ids is not None:
            # Load specific sessions
            session_dirs = [self.data_root / sid for sid in session_ids]
        else:
            # Find all session_ prefixed directories
            session_dirs = sorted(self.data_root.glob("session_*"))

            # Also check for split-specific directories
            split_dir = self.data_root / self.split
            if split_dir.exists():
                session_dirs.extend(sorted(split_dir.glob("session_*")))

        for session_dir in session_dirs:
            if not session_dir.is_dir():
                continue

            try:
                session = EnfusionSession(session_dir, self.config)
                if session.num_frames >= self.config.sequence_length:
                    sessions.append(session)
            except (FileNotFoundError, Exception) as e:
                warnings.warn(f"Skipping session {session_dir}: {e}")
                continue

        return sessions

    def _build_sequence_indices(self) -> List[Tuple[int, int]]:
        """
        Build index of valid sequences.

        Returns:
            List of (session_idx, start_frame) tuples
        """
        indices = []

        for session_idx, session in enumerate(self.sessions):
            valid_starts = session.get_valid_start_indices(
                self.config.sequence_length,
                self.config.frame_skip
            )

            for start_frame in valid_starts:
                indices.append((session_idx, start_frame))

        return indices

    def _filter_by_quality(self):
        """Filter sessions by quality score."""
        quality_filter = EnfusionQualityFilter(self.config.telemetry_config)

        high_quality_sessions = []
        high_quality_indices = []

        for session_idx, session in enumerate(self.sessions):
            quality = quality_filter.compute_quality_score(
                session.telemetry,
                session.frame_paths
            )

            if quality >= self.config.min_quality_score:
                new_session_idx = len(high_quality_sessions)
                high_quality_sessions.append(session)

                # Remap indices
                for idx, (s_idx, start_frame) in enumerate(self.sequence_indices):
                    if s_idx == session_idx:
                        high_quality_indices.append((new_session_idx, start_frame))

        self.sessions = high_quality_sessions
        self.sequence_indices = high_quality_indices

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - frames: [T, C, H, W] float32
                - controls: [T, 6] float32
                - depth: [T, 1, H, W] float32 (if enabled)
                - ego_transform: [T, 4, 4] float32
                - scene_entities: List of entity dicts
                - anchor_mask: [T] bool
                - metadata: dict
        """
        session_idx, start_frame = self.sequence_indices[idx]
        session = self.sessions[session_idx]

        # Compute frame indices
        frame_indices = [
            start_frame + i * self.config.frame_skip
            for i in range(self.config.sequence_length)
        ]

        # Load frames
        frames = self._load_frames(session, frame_indices)

        # Load controls
        controls = self._load_controls(session, frame_indices)

        # Compute anchor mask
        anchor_mask = session.anchor_detector.detect_anchors(controls)

        # Build sample
        sample = {
            'frames': frames,
            'controls': controls,
            'anchor_mask': anchor_mask,
            'metadata': {
                'session_id': session.metadata['session_id'],
                'start_frame': start_frame,
                'frame_indices': frame_indices
            }
        }

        # Load ego transforms
        sample['ego_transform'] = session.ego_transforms[frame_indices]

        # Load depth if enabled
        if self.config.load_depth and session.depth_paths:
            depth = self._load_depth(session, frame_indices)
            sample['depth'] = depth

        # Load scene entities if enabled
        if self.config.load_scene and session.scene_paths:
            scene_entities = self._load_scenes(session, frame_indices)
            sample['scene_entities'] = scene_entities

        # Apply augmentation
        if self.config.augment and self.split == 'train':
            sample = self._apply_augmentation(sample)

        return sample

    def _load_frames(
        self,
        session: EnfusionSession,
        frame_indices: List[int]
    ) -> torch.Tensor:
        """Load frames for the given indices."""
        frame_paths = [session.frame_paths[i] for i in frame_indices if i < len(session.frame_paths)]

        # Pad with last frame if needed
        while len(frame_paths) < len(frame_indices):
            frame_paths.append(session.frame_paths[-1])

        return session.frame_processor.load_frames(frame_paths)

    def _load_controls(
        self,
        session: EnfusionSession,
        frame_indices: List[int]
    ) -> torch.Tensor:
        """Load controls for the given indices."""
        controls = session.controls

        # Extract indices (handle out of bounds)
        valid_indices = [min(i, len(controls) - 1) for i in frame_indices]

        return controls[valid_indices]

    def _load_depth(
        self,
        session: EnfusionSession,
        frame_indices: List[int]
    ) -> torch.Tensor:
        """Load depth maps for the given indices."""
        depth_paths = []
        for i in frame_indices:
            if i < len(session.depth_paths):
                depth_paths.append(session.depth_paths[i])
            elif session.depth_paths:
                depth_paths.append(session.depth_paths[-1])
            else:
                depth_paths.append(None)

        depths = []
        for path in depth_paths:
            if path is not None:
                try:
                    depth = session.depth_processor.load_depth(path)
                    depths.append(depth)
                except Exception:
                    depths.append(torch.zeros(1, *self.config.image_size))
            else:
                depths.append(torch.zeros(1, *self.config.image_size))

        return torch.stack(depths, dim=0)

    def _load_scenes(
        self,
        session: EnfusionSession,
        frame_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Load scene graphs for the given indices."""
        scenes = []
        for i in frame_indices:
            if i < len(session.scene_paths):
                try:
                    scene = session.scene_parser.parse_scene(session.scene_paths[i])
                    scenes.append(scene)
                except Exception:
                    scenes.append({'entities': [], 'timestamp': 0.0})
            else:
                scenes.append({'entities': [], 'timestamp': 0.0})

        return scenes

    def _apply_augmentation(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply data augmentation."""
        # Horizontal flip
        if random.random() < self.config.horizontal_flip_prob:
            sample = self._horizontal_flip(sample)

        # Brightness/contrast
        if random.random() < 0.3:
            brightness = random.uniform(*self.config.brightness_range)
            sample['frames'] = torch.clamp(sample['frames'] * brightness, 0, 1)

        if random.random() < 0.3:
            contrast = random.uniform(*self.config.contrast_range)
            mean = sample['frames'].mean(dim=(-2, -1), keepdim=True)
            sample['frames'] = torch.clamp((sample['frames'] - mean) * contrast + mean, 0, 1)

        return sample

    def _horizontal_flip(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply horizontal flip augmentation."""
        # Flip frames
        sample['frames'] = torch.flip(sample['frames'], dims=[-1])

        # Flip depth if present
        if 'depth' in sample:
            sample['depth'] = torch.flip(sample['depth'], dims=[-1])

        # Invert steering
        sample['controls'][:, 0] = -sample['controls'][:, 0]

        # Flip lateral velocity
        if sample['controls'].shape[-1] > 4:
            sample['controls'][:, 4] = -sample['controls'][:, 4]

        return sample

    def get_session_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded sessions."""
        return [
            {
                'session_id': session.metadata['session_id'],
                'num_frames': session.num_frames,
                'path': str(session.session_dir)
            }
            for session in self.sessions
        ]


class EnfusionCollator:
    """
    Collator for EnfusionDataset batches.

    Handles variable-length sequences, scene entity lists, and metadata.
    """

    def __init__(
        self,
        pad_to_max: bool = True,
        max_sequence_length: Optional[int] = None,
        include_scene_entities: bool = False
    ):
        """
        Initialize collator.

        Args:
            pad_to_max: Pad sequences to max length in batch
            max_sequence_length: Maximum sequence length
            include_scene_entities: Whether to collate scene entities
        """
        self.pad_to_max = pad_to_max
        self.max_sequence_length = max_sequence_length
        self.include_scene_entities = include_scene_entities

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of samples."""
        if not batch:
            return {}

        collated = {}

        # Collate tensor fields
        tensor_fields = ['frames', 'controls', 'depth', 'ego_transform', 'anchor_mask']

        for field in tensor_fields:
            if field in batch[0]:
                tensors = [sample[field] for sample in batch]
                collated[field] = torch.stack(tensors, dim=0)

        # Collate metadata
        collated['metadata'] = [sample['metadata'] for sample in batch]

        # Collate scene entities if requested
        if self.include_scene_entities and 'scene_entities' in batch[0]:
            collated['scene_entities'] = [sample['scene_entities'] for sample in batch]

        return collated


class EnfusionDataLoader:
    """
    Data loader wrapper for EnfusionDataset.

    Provides efficient batching, prefetching, and GPU transfer.
    """

    def __init__(
        self,
        dataset: EnfusionDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        prefetch_factor: int = 2,
        include_scene_entities: bool = False
    ):
        """
        Initialize data loader.

        Args:
            dataset: EnfusionDataset instance
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop incomplete batches
            prefetch_factor: Prefetch factor per worker
            include_scene_entities: Include scene entities in batches
        """
        self.dataset = dataset

        collator = EnfusionCollator(include_scene_entities=include_scene_entities)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            collate_fn=collator,
            persistent_workers=num_workers > 0
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.dataloader:
            # Move tensors to GPU
            yield {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def __len__(self) -> int:
        return len(self.dataloader)


def create_enfusion_dataloaders(
    data_root: Union[str, Path],
    config: Optional[EnfusionDatasetConfig] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[EnfusionDataLoader, EnfusionDataLoader, EnfusionDataLoader]:
    """
    Create train/val/test data loaders.

    Args:
        data_root: Root data directory
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)
    config = config or EnfusionDatasetConfig()

    # Check for split directories
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if train_dir.exists() and val_dir.exists():
        # Use existing split
        train_dataset = EnfusionDataset(train_dir, config, split="train")
        val_dataset = EnfusionDataset(val_dir, config, split="val")
        test_dataset = EnfusionDataset(
            test_dir if test_dir.exists() else val_dir,
            config,
            split="test"
        )
    else:
        # Create random split
        all_sessions = sorted(data_root.glob("session_*"))
        random.shuffle(all_sessions)

        n_sessions = len(all_sessions)
        n_train = int(n_sessions * train_ratio)
        n_val = int(n_sessions * val_ratio)

        train_sessions = [s.name for s in all_sessions[:n_train]]
        val_sessions = [s.name for s in all_sessions[n_train:n_train + n_val]]
        test_sessions = [s.name for s in all_sessions[n_train + n_val:]]

        # Create train config with augmentation
        train_config = EnfusionDatasetConfig(
            **{k: v for k, v in config.__dict__.items() if k != 'augment'}
        )
        train_config.augment = True

        # Create val/test config without augmentation
        eval_config = EnfusionDatasetConfig(
            **{k: v for k, v in config.__dict__.items() if k != 'augment'}
        )
        eval_config.augment = False

        train_dataset = EnfusionDataset(data_root, train_config, "train", train_sessions)
        val_dataset = EnfusionDataset(data_root, eval_config, "val", val_sessions)
        test_dataset = EnfusionDataset(data_root, eval_config, "test", test_sessions)

    # Create loaders
    train_loader = EnfusionDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = EnfusionDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = EnfusionDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


# Integration with DriveDiT pipeline
class EnfusionToDriveDiTAdapter:
    """
    Adapter to convert Enfusion data format to DriveDiT expected format.

    Ensures compatibility with existing DriveDiT training pipeline.
    """

    def __init__(
        self,
        target_control_dim: int = 6,
        include_depth: bool = True,
        include_ego_transform: bool = True
    ):
        """
        Initialize adapter.

        Args:
            target_control_dim: Target control dimension
            include_depth: Include depth in output
            include_ego_transform: Include ego transforms
        """
        self.target_control_dim = target_control_dim
        self.include_depth = include_depth
        self.include_ego_transform = include_ego_transform

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert Enfusion batch to DriveDiT format.

        Args:
            batch: Enfusion batch dictionary

        Returns:
            DriveDiT compatible batch
        """
        drivedit_batch = {}

        # Frames: Enfusion [B, T, C, H, W] -> DriveDiT [B, T, C, H, W]
        # Already in correct format
        drivedit_batch['frames'] = batch['frames']

        # Controls: Ensure correct dimension
        controls = batch['controls']
        if controls.shape[-1] < self.target_control_dim:
            # Pad with zeros
            padding = torch.zeros(
                *controls.shape[:-1],
                self.target_control_dim - controls.shape[-1],
                device=controls.device,
                dtype=controls.dtype
            )
            controls = torch.cat([controls, padding], dim=-1)
        elif controls.shape[-1] > self.target_control_dim:
            # Truncate
            controls = controls[..., :self.target_control_dim]

        drivedit_batch['controls'] = controls

        # Depth
        if self.include_depth and 'depth' in batch:
            drivedit_batch['depth'] = batch['depth']

        # Ego transforms
        if self.include_ego_transform and 'ego_transform' in batch:
            drivedit_batch['ego_transform'] = batch['ego_transform']

        # Anchor mask
        if 'anchor_mask' in batch:
            drivedit_batch['anchor_mask'] = batch['anchor_mask']

        return drivedit_batch


if __name__ == "__main__":
    # Test dataset components
    print("Testing EnfusionDataset components...")

    # Create test configuration
    config = EnfusionDatasetConfig(
        sequence_length=8,
        image_size=(128, 128),
        load_depth=True,
        load_scene=True,
        augment=False
    )

    print(f"Config: {config.sequence_length} frames, {config.image_size} size")

    # Test adapter
    adapter = EnfusionToDriveDiTAdapter()
    print("Adapter created successfully")

    # Create synthetic batch for testing
    batch_size = 2
    seq_len = config.sequence_length
    img_size = config.image_size

    fake_batch = {
        'frames': torch.randn(batch_size, seq_len, 3, *img_size),
        'controls': torch.randn(batch_size, seq_len, 6),
        'depth': torch.randn(batch_size, seq_len, 1, *img_size),
        'ego_transform': torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1),
        'anchor_mask': torch.zeros(batch_size, seq_len, dtype=torch.bool),
        'metadata': [{'session_id': f'test_{i}'} for i in range(batch_size)]
    }

    # Test adapter
    drivedit_batch = adapter(fake_batch)
    print(f"Adapted batch keys: {list(drivedit_batch.keys())}")
    print(f"Frames shape: {drivedit_batch['frames'].shape}")
    print(f"Controls shape: {drivedit_batch['controls'].shape}")

    # Test collator
    collator = EnfusionCollator()
    samples = [
        {
            'frames': torch.randn(seq_len, 3, *img_size),
            'controls': torch.randn(seq_len, 6),
            'anchor_mask': torch.zeros(seq_len, dtype=torch.bool),
            'metadata': {'session_id': f'test_{i}'}
        }
        for i in range(batch_size)
    ]

    collated = collator(samples)
    print(f"Collated batch keys: {list(collated.keys())}")
    print(f"Collated frames shape: {collated['frames'].shape}")

    print("\nEnfusionDataset test completed successfully!")
