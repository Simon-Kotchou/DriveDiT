"""
Enfusion capture data preprocessing utilities for DriveDiT.
Handles telemetry parsing, control signal normalization, and data quality filtering.
Zero-dependency implementation using only torch, numpy, and cv2.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class EnfusionTelemetryConfig:
    """Configuration for Enfusion telemetry preprocessing."""

    # Telemetry columns expected in CSV
    timestamp_col: str = "timestamp"
    position_cols: Tuple[str, str, str] = ("pos_x", "pos_y", "pos_z")
    rotation_cols: Tuple[str, str, str] = ("rot_pitch", "rot_yaw", "rot_roll")
    velocity_cols: Tuple[str, str, str] = ("vel_x", "vel_y", "vel_z")

    # Control signal columns
    throttle_col: str = "throttle"
    brake_col: str = "brake"
    steering_col: str = "steering"

    # Extended control signals
    gear_col: str = "gear"
    rpm_col: str = "rpm"
    speed_col: str = "speed"

    # Telemetry rate (Hz)
    telemetry_rate: float = 5.0
    frame_rate: float = 30.0

    # Control signal normalization ranges
    steering_range: Tuple[float, float] = (-1.0, 1.0)
    throttle_range: Tuple[float, float] = (0.0, 1.0)
    brake_range: Tuple[float, float] = (0.0, 1.0)
    speed_range: Tuple[float, float] = (0.0, 50.0)  # m/s

    # Quality filtering thresholds
    min_speed_variation: float = 0.1  # Minimum speed std to consider "dynamic"
    max_stationary_ratio: float = 0.8  # Max ratio of frames where vehicle is stationary
    min_steering_variation: float = 0.01  # Minimum steering variation

    # Outlier detection
    max_position_jump: float = 10.0  # Max position change between frames (m)
    max_rotation_jump: float = 90.0  # Max rotation change (degrees)


class EnfusionTelemetryParser:
    """Parser for Enfusion telemetry CSV files."""

    def __init__(self, config: Optional[EnfusionTelemetryConfig] = None):
        """
        Initialize telemetry parser.

        Args:
            config: Telemetry configuration
        """
        self.config = config or EnfusionTelemetryConfig()

    def parse_csv(self, csv_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Parse telemetry CSV file.

        Args:
            csv_path: Path to telemetry CSV

        Returns:
            Dictionary of telemetry arrays
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {csv_path}")

        data = {
            'timestamp': [],
            'position': [],
            'rotation': [],
            'velocity': [],
            'throttle': [],
            'brake': [],
            'steering': [],
            'speed': [],
            'gear': [],
            'rpm': []
        }

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse timestamp
                    data['timestamp'].append(float(row.get(self.config.timestamp_col, 0)))

                    # Parse position
                    pos = [
                        float(row.get(col, 0)) for col in self.config.position_cols
                    ]
                    data['position'].append(pos)

                    # Parse rotation
                    rot = [
                        float(row.get(col, 0)) for col in self.config.rotation_cols
                    ]
                    data['rotation'].append(rot)

                    # Parse velocity
                    vel = [
                        float(row.get(col, 0)) for col in self.config.velocity_cols
                    ]
                    data['velocity'].append(vel)

                    # Parse control signals
                    data['throttle'].append(float(row.get(self.config.throttle_col, 0)))
                    data['brake'].append(float(row.get(self.config.brake_col, 0)))
                    data['steering'].append(float(row.get(self.config.steering_col, 0)))
                    data['speed'].append(float(row.get(self.config.speed_col, 0)))
                    data['gear'].append(int(row.get(self.config.gear_col, 0)))
                    data['rpm'].append(float(row.get(self.config.rpm_col, 0)))

                except (ValueError, TypeError) as e:
                    warnings.warn(f"Skipping malformed row: {e}")
                    continue

        # Convert to numpy arrays
        result = {}
        for key, values in data.items():
            if values:
                if key in ['position', 'rotation', 'velocity']:
                    result[key] = np.array(values, dtype=np.float32)
                elif key == 'gear':
                    result[key] = np.array(values, dtype=np.int32)
                else:
                    result[key] = np.array(values, dtype=np.float32)
            else:
                result[key] = np.array([])

        return result

    def compute_ego_transform(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """
        Compute ego vehicle 4x4 transformation matrices.

        Args:
            position: Position array [N, 3]
            rotation: Rotation array [N, 3] (pitch, yaw, roll in degrees)

        Returns:
            Transform matrices [N, 4, 4]
        """
        N = position.shape[0]
        transforms = np.zeros((N, 4, 4), dtype=np.float32)

        for i in range(N):
            # Convert rotation from degrees to radians
            pitch, yaw, roll = np.radians(rotation[i])

            # Create rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])

            Ry = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])

            Rz = np.array([
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1]
            ])

            # Combined rotation (yaw-pitch-roll order)
            R = Ry @ Rx @ Rz

            # Build 4x4 transform
            transforms[i, :3, :3] = R
            transforms[i, :3, 3] = position[i]
            transforms[i, 3, 3] = 1.0

        return transforms

    def interpolate_to_frame_rate(
        self,
        telemetry: Dict[str, np.ndarray],
        num_frames: int
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate telemetry to match frame rate.

        Args:
            telemetry: Parsed telemetry data
            num_frames: Number of video frames

        Returns:
            Interpolated telemetry
        """
        if not telemetry.get('timestamp', []).size:
            return telemetry

        # Original timestamps
        orig_timestamps = telemetry['timestamp']
        if len(orig_timestamps) < 2:
            return telemetry

        # Target timestamps (assuming constant frame rate)
        duration = orig_timestamps[-1] - orig_timestamps[0]
        target_timestamps = np.linspace(orig_timestamps[0], orig_timestamps[-1], num_frames)

        interpolated = {'timestamp': target_timestamps}

        for key, values in telemetry.items():
            if key == 'timestamp':
                continue

            if values.size == 0:
                interpolated[key] = np.zeros((num_frames,) + values.shape[1:], dtype=values.dtype)
                continue

            if values.ndim == 1:
                # 1D interpolation
                interpolated[key] = np.interp(target_timestamps, orig_timestamps, values).astype(values.dtype)
            elif values.ndim == 2:
                # 2D interpolation (per-column)
                result = np.zeros((num_frames, values.shape[1]), dtype=values.dtype)
                for col in range(values.shape[1]):
                    result[:, col] = np.interp(target_timestamps, orig_timestamps, values[:, col])
                interpolated[key] = result
            else:
                # For higher dimensions, just repeat
                interpolated[key] = values

        return interpolated


class EnfusionControlNormalizer:
    """Normalizer for Enfusion control signals."""

    def __init__(self, config: Optional[EnfusionTelemetryConfig] = None):
        """
        Initialize control normalizer.

        Args:
            config: Telemetry configuration
        """
        self.config = config or EnfusionTelemetryConfig()

    def normalize_controls(
        self,
        telemetry: Dict[str, np.ndarray],
        output_dim: int = 6
    ) -> torch.Tensor:
        """
        Normalize control signals to standard DriveDiT format.

        Args:
            telemetry: Parsed telemetry data
            output_dim: Output control dimension (default 6)

        Returns:
            Normalized control tensor [T, output_dim]
        """
        # Get number of timesteps
        T = len(telemetry.get('timestamp', []))
        if T == 0:
            return torch.zeros(1, output_dim, dtype=torch.float32)

        controls = torch.zeros(T, output_dim, dtype=torch.float32)

        # Channel 0: Steering (normalized to [-1, 1])
        if 'steering' in telemetry and telemetry['steering'].size:
            steering = telemetry['steering']
            steering_norm = self._normalize_to_range(
                steering,
                self.config.steering_range,
                (-1.0, 1.0)
            )
            controls[:, 0] = torch.from_numpy(steering_norm)

        # Channel 1: Acceleration (throttle - brake, normalized to [-1, 1])
        throttle = telemetry.get('throttle', np.zeros(T))
        brake = telemetry.get('brake', np.zeros(T))
        if throttle.size and brake.size:
            accel = throttle - brake  # Combined acceleration
            controls[:, 1] = torch.from_numpy(accel.astype(np.float32))

        # Channel 2: Brake (normalized to [0, 1])
        if 'brake' in telemetry and telemetry['brake'].size:
            brake_norm = self._normalize_to_range(
                brake,
                self.config.brake_range,
                (0.0, 1.0)
            )
            controls[:, 2] = torch.from_numpy(brake_norm)

        # Channel 3: Forward velocity (goal_x proxy)
        # Use velocity or speed as proxy for goal direction
        if 'velocity' in telemetry and telemetry['velocity'].size:
            # Use forward velocity component
            vel_forward = telemetry['velocity'][:, 0]  # Assuming x is forward
            vel_norm = self._normalize_to_range(
                vel_forward,
                (-self.config.speed_range[1], self.config.speed_range[1]),
                (-1.0, 1.0)
            )
            controls[:, 3] = torch.from_numpy(vel_norm)
        elif 'speed' in telemetry and telemetry['speed'].size:
            speed_norm = self._normalize_to_range(
                telemetry['speed'],
                self.config.speed_range,
                (0.0, 1.0)
            )
            controls[:, 3] = torch.from_numpy(speed_norm)

        # Channel 4: Lateral velocity (goal_y proxy)
        if 'velocity' in telemetry and telemetry['velocity'].size:
            vel_lateral = telemetry['velocity'][:, 1]  # Assuming y is lateral
            vel_norm = self._normalize_to_range(
                vel_lateral,
                (-self.config.speed_range[1] * 0.5, self.config.speed_range[1] * 0.5),
                (-1.0, 1.0)
            )
            controls[:, 4] = torch.from_numpy(vel_norm)

        # Channel 5: Speed (normalized to [0, 1])
        if 'speed' in telemetry and telemetry['speed'].size:
            speed_norm = self._normalize_to_range(
                telemetry['speed'],
                self.config.speed_range,
                (0.0, 1.0)
            )
            controls[:, 5] = torch.from_numpy(speed_norm)

        return controls

    def _normalize_to_range(
        self,
        values: np.ndarray,
        input_range: Tuple[float, float],
        output_range: Tuple[float, float]
    ) -> np.ndarray:
        """Normalize values from input range to output range."""
        in_min, in_max = input_range
        out_min, out_max = output_range

        # Clip to input range
        clipped = np.clip(values, in_min, in_max)

        # Normalize to [0, 1]
        if in_max - in_min > 1e-8:
            normalized = (clipped - in_min) / (in_max - in_min)
        else:
            normalized = np.zeros_like(clipped)

        # Scale to output range
        scaled = normalized * (out_max - out_min) + out_min

        return scaled.astype(np.float32)

    def denormalize_controls(
        self,
        controls: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Denormalize controls back to original ranges.

        Args:
            controls: Normalized control tensor [T, 6] or [B, T, 6]

        Returns:
            Dictionary of denormalized control signals
        """
        result = {}

        # Handle batch dimension
        if controls.dim() == 3:
            is_batched = True
        else:
            is_batched = False
            controls = controls.unsqueeze(0)

        # Steering
        result['steering'] = controls[..., 0]  # Already in [-1, 1]

        # Acceleration
        result['acceleration'] = controls[..., 1]

        # Brake
        result['brake'] = controls[..., 2]

        # Forward velocity
        speed_range = self.config.speed_range
        result['velocity_forward'] = controls[..., 3] * speed_range[1]

        # Lateral velocity
        result['velocity_lateral'] = controls[..., 4] * (speed_range[1] * 0.5)

        # Speed
        result['speed'] = controls[..., 5] * speed_range[1]

        if not is_batched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result


class EnfusionFrameProcessor:
    """Processor for Enfusion captured frames."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        """
        Initialize frame processor.

        Args:
            target_size: Target image size (H, W)
            normalize: Whether to normalize frames
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def load_frame(self, frame_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess a single frame.

        Args:
            frame_path: Path to frame image

        Returns:
            Preprocessed frame tensor [C, H, W]
        """
        frame_path = Path(frame_path)

        # Load with OpenCV
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        if frame.shape[:2] != self.target_size:
            frame = cv2.resize(frame, self.target_size[::-1])  # cv2 uses (W, H)

        # Convert to tensor [H, W, C] -> [C, H, W]
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)

        # Normalize
        if self.normalize:
            mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)
            frame_tensor = (frame_tensor - mean) / std

        return frame_tensor

    def load_frames(
        self,
        frame_paths: List[Union[str, Path]],
        max_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Load multiple frames as a sequence.

        Args:
            frame_paths: List of frame paths
            max_frames: Maximum number of frames to load

        Returns:
            Frame tensor [T, C, H, W]
        """
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        frames = []
        for path in frame_paths:
            try:
                frame = self.load_frame(path)
                frames.append(frame)
            except (ValueError, Exception) as e:
                warnings.warn(f"Skipping frame {path}: {e}")
                continue

        if not frames:
            raise ValueError("No frames could be loaded")

        return torch.stack(frames, dim=0)

    def denormalize(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Denormalize frames for visualization.

        Args:
            frames: Normalized frame tensor

        Returns:
            Denormalized frames in [0, 1]
        """
        if not self.normalize:
            return frames

        mean = torch.tensor(self.mean, dtype=frames.dtype, device=frames.device)
        std = torch.tensor(self.std, dtype=frames.dtype, device=frames.device)

        # Handle different dimensions
        if frames.dim() == 4:  # [T, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif frames.dim() == 5:  # [B, T, C, H, W]
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
        else:  # [C, H, W]
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return torch.clamp(frames * std + mean, 0.0, 1.0)


class EnfusionDepthProcessor:
    """Processor for Enfusion depth maps."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        max_depth: float = 100.0,
        depth_format: str = "npz"  # "npz", "png16", "exr"
    ):
        """
        Initialize depth processor.

        Args:
            target_size: Target depth map size
            max_depth: Maximum depth value for normalization
            depth_format: Depth map format
        """
        self.target_size = target_size
        self.max_depth = max_depth
        self.depth_format = depth_format

    def load_depth(self, depth_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess a single depth map.

        Args:
            depth_path: Path to depth file

        Returns:
            Depth tensor [1, H, W] normalized to [0, 1]
        """
        depth_path = Path(depth_path)

        if self.depth_format == "npz":
            data = np.load(depth_path)
            depth = data['depth'] if 'depth' in data else data[data.files[0]]
        elif self.depth_format == "png16":
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 65535.0 * self.max_depth
        else:
            raise ValueError(f"Unsupported depth format: {self.depth_format}")

        if depth is None:
            raise ValueError(f"Failed to load depth: {depth_path}")

        # Ensure 2D
        if depth.ndim == 3:
            depth = depth[..., 0]

        # Resize
        if depth.shape != self.target_size:
            depth = cv2.resize(depth, self.target_size[::-1])

        # Normalize to [0, 1]
        depth_norm = np.clip(depth / self.max_depth, 0.0, 1.0)

        # Convert to tensor [1, H, W]
        depth_tensor = torch.from_numpy(depth_norm).float().unsqueeze(0)

        return depth_tensor

    def load_depth_sequence(
        self,
        depth_paths: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        Load multiple depth maps as sequence.

        Args:
            depth_paths: List of depth paths

        Returns:
            Depth tensor [T, 1, H, W]
        """
        depths = []
        for path in depth_paths:
            try:
                depth = self.load_depth(path)
                depths.append(depth)
            except (ValueError, Exception) as e:
                warnings.warn(f"Skipping depth {path}: {e}")
                # Use zero depth as fallback
                depths.append(torch.zeros(1, *self.target_size))

        return torch.stack(depths, dim=0)

    def denormalize(self, depth: torch.Tensor) -> torch.Tensor:
        """Denormalize depth to meters."""
        return depth * self.max_depth


class EnfusionSceneParser:
    """Parser for Enfusion scene graph JSON files."""

    def __init__(self, max_entities: int = 64):
        """
        Initialize scene parser.

        Args:
            max_entities: Maximum number of entities to track
        """
        self.max_entities = max_entities

    def parse_scene(self, scene_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse scene graph JSON file.

        Args:
            scene_path: Path to scene JSON

        Returns:
            Parsed scene data
        """
        scene_path = Path(scene_path)

        with open(scene_path, 'r') as f:
            scene_data = json.load(f)

        entities = []

        for entity in scene_data.get('entities', [])[:self.max_entities]:
            parsed_entity = {
                'id': entity.get('id', ''),
                'class': entity.get('class', 'unknown'),
                'position': entity.get('position', [0, 0, 0]),
                'rotation': entity.get('rotation', [0, 0, 0]),
                'velocity': entity.get('velocity', [0, 0, 0]),
                'bbox': entity.get('bbox', [0, 0, 0, 0]),  # 2D bbox if available
                'bbox_3d': entity.get('bbox_3d', [0, 0, 0, 0, 0, 0]),  # 3D bbox
                'visible': entity.get('visible', True),
                'distance': entity.get('distance', 0.0)
            }
            entities.append(parsed_entity)

        return {
            'entities': entities,
            'timestamp': scene_data.get('timestamp', 0.0),
            'weather': scene_data.get('weather', 'clear'),
            'time_of_day': scene_data.get('time_of_day', 12.0)
        }

    def parse_scene_sequence(
        self,
        scene_paths: List[Union[str, Path]]
    ) -> List[Dict[str, Any]]:
        """
        Parse sequence of scene graphs.

        Args:
            scene_paths: List of scene JSON paths

        Returns:
            List of parsed scene data
        """
        scenes = []
        for path in scene_paths:
            try:
                scene = self.parse_scene(path)
                scenes.append(scene)
            except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
                warnings.warn(f"Skipping scene {path}: {e}")
                scenes.append({'entities': [], 'timestamp': 0.0})

        return scenes


class EnfusionQualityFilter:
    """Quality filter for Enfusion capture sessions."""

    def __init__(self, config: Optional[EnfusionTelemetryConfig] = None):
        """
        Initialize quality filter.

        Args:
            config: Telemetry configuration
        """
        self.config = config or EnfusionTelemetryConfig()

    def compute_quality_score(
        self,
        telemetry: Dict[str, np.ndarray],
        frame_paths: List[Path]
    ) -> float:
        """
        Compute quality score for a capture session.

        Args:
            telemetry: Parsed telemetry data
            frame_paths: List of frame paths

        Returns:
            Quality score in [0, 1]
        """
        scores = []

        # Frame completeness
        expected_frames = self._estimate_expected_frames(telemetry)
        frame_score = min(len(frame_paths) / max(expected_frames, 1), 1.0)
        scores.append(frame_score)

        # Speed variation (dynamic content)
        if 'speed' in telemetry and telemetry['speed'].size:
            speed_std = np.std(telemetry['speed'])
            speed_score = min(speed_std / self.config.min_speed_variation, 1.0)
            scores.append(speed_score)

        # Steering variation
        if 'steering' in telemetry and telemetry['steering'].size:
            steering_std = np.std(telemetry['steering'])
            steering_score = min(steering_std / self.config.min_steering_variation, 1.0)
            scores.append(steering_score)

        # Outlier detection (penalize sessions with position jumps)
        outlier_score = self._compute_outlier_score(telemetry)
        scores.append(outlier_score)

        # Stationary ratio
        if 'speed' in telemetry and telemetry['speed'].size:
            stationary_ratio = np.mean(telemetry['speed'] < 0.5)
            stationary_score = 1.0 - min(stationary_ratio / self.config.max_stationary_ratio, 1.0)
            scores.append(stationary_score)

        return np.mean(scores) if scores else 0.0

    def _estimate_expected_frames(self, telemetry: Dict[str, np.ndarray]) -> int:
        """Estimate expected number of frames based on telemetry duration."""
        if 'timestamp' not in telemetry or not telemetry['timestamp'].size:
            return 0

        duration = telemetry['timestamp'][-1] - telemetry['timestamp'][0]
        return int(duration * self.config.frame_rate)

    def _compute_outlier_score(self, telemetry: Dict[str, np.ndarray]) -> float:
        """Compute score based on data continuity (penalize outliers)."""
        if 'position' not in telemetry or telemetry['position'].size < 2:
            return 1.0

        positions = telemetry['position']
        position_diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Count large jumps
        outliers = position_diffs > self.config.max_position_jump
        outlier_ratio = np.mean(outliers)

        return 1.0 - min(outlier_ratio * 10, 1.0)  # Penalize heavily

    def filter_sessions(
        self,
        session_dirs: List[Path],
        min_quality: float = 0.5
    ) -> List[Tuple[Path, float]]:
        """
        Filter capture sessions by quality.

        Args:
            session_dirs: List of session directories
            min_quality: Minimum quality threshold

        Returns:
            List of (session_path, quality_score) tuples
        """
        filtered = []
        parser = EnfusionTelemetryParser(self.config)

        for session_dir in session_dirs:
            try:
                # Load telemetry
                telemetry_path = session_dir / "telemetry.csv"
                if not telemetry_path.exists():
                    continue

                telemetry = parser.parse_csv(telemetry_path)

                # Get frame paths
                frames_dir = session_dir / "frames"
                if frames_dir.exists():
                    frame_paths = sorted(frames_dir.glob("*.png"))
                else:
                    frame_paths = []

                # Compute quality
                quality = self.compute_quality_score(telemetry, frame_paths)

                if quality >= min_quality:
                    filtered.append((session_dir, quality))

            except Exception as e:
                warnings.warn(f"Error processing session {session_dir}: {e}")
                continue

        # Sort by quality (descending)
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered


class EnfusionAnchorDetector:
    """Detector for anchor frames (keyframes) in Enfusion sequences."""

    def __init__(
        self,
        min_steering_change: float = 0.1,
        min_speed_change: float = 2.0,
        min_interval: int = 10
    ):
        """
        Initialize anchor detector.

        Args:
            min_steering_change: Minimum steering change to trigger anchor
            min_speed_change: Minimum speed change to trigger anchor
            min_interval: Minimum frames between anchors
        """
        self.min_steering_change = min_steering_change
        self.min_speed_change = min_speed_change
        self.min_interval = min_interval

    def detect_anchors(
        self,
        controls: torch.Tensor,
        include_first_last: bool = True
    ) -> torch.Tensor:
        """
        Detect anchor frames in control sequence.

        Args:
            controls: Control tensor [T, D]
            include_first_last: Include first and last frames as anchors

        Returns:
            Boolean mask [T] indicating anchor frames
        """
        T = controls.shape[0]
        anchors = torch.zeros(T, dtype=torch.bool)

        if include_first_last:
            anchors[0] = True
            anchors[-1] = True

        # Compute control changes
        steering_diff = torch.abs(controls[1:, 0] - controls[:-1, 0])
        speed_diff = torch.abs(controls[1:, 5] - controls[:-1, 5]) if controls.shape[1] > 5 else torch.zeros(T-1)

        # Detect significant changes
        last_anchor = 0
        for i in range(1, T):
            if i - last_anchor < self.min_interval:
                continue

            if i < T:
                if (steering_diff[i-1] > self.min_steering_change or
                    speed_diff[i-1] > self.min_speed_change):
                    anchors[i] = True
                    last_anchor = i

        return anchors


# Factory function for creating preprocessors
def create_enfusion_preprocessors(
    config: Optional[EnfusionTelemetryConfig] = None,
    target_size: Tuple[int, int] = (256, 256),
    max_depth: float = 100.0
) -> Dict[str, Any]:
    """
    Create all Enfusion preprocessing components.

    Args:
        config: Telemetry configuration
        target_size: Target image size
        max_depth: Maximum depth value

    Returns:
        Dictionary of preprocessor instances
    """
    config = config or EnfusionTelemetryConfig()

    return {
        'telemetry_parser': EnfusionTelemetryParser(config),
        'control_normalizer': EnfusionControlNormalizer(config),
        'frame_processor': EnfusionFrameProcessor(target_size=target_size),
        'depth_processor': EnfusionDepthProcessor(target_size=target_size, max_depth=max_depth),
        'scene_parser': EnfusionSceneParser(),
        'quality_filter': EnfusionQualityFilter(config),
        'anchor_detector': EnfusionAnchorDetector()
    }


if __name__ == "__main__":
    # Test preprocessing components
    print("Testing Enfusion preprocessing components...")

    # Create config
    config = EnfusionTelemetryConfig()
    print(f"Telemetry rate: {config.telemetry_rate} Hz")
    print(f"Frame rate: {config.frame_rate} Hz")

    # Create preprocessors
    preprocessors = create_enfusion_preprocessors(config)
    print(f"Created {len(preprocessors)} preprocessor components")

    # Test control normalizer with synthetic data
    normalizer = preprocessors['control_normalizer']
    fake_telemetry = {
        'timestamp': np.linspace(0, 10, 50),
        'steering': np.sin(np.linspace(0, 2*np.pi, 50)).astype(np.float32),
        'throttle': np.abs(np.sin(np.linspace(0, np.pi, 50))).astype(np.float32),
        'brake': np.zeros(50, dtype=np.float32),
        'velocity': np.random.randn(50, 3).astype(np.float32) * 10,
        'speed': np.abs(np.sin(np.linspace(0, np.pi, 50)) * 20).astype(np.float32)
    }

    controls = normalizer.normalize_controls(fake_telemetry)
    print(f"Normalized controls shape: {controls.shape}")
    print(f"Control ranges: min={controls.min().item():.3f}, max={controls.max().item():.3f}")

    # Test anchor detection
    anchor_detector = preprocessors['anchor_detector']
    anchors = anchor_detector.detect_anchors(controls)
    print(f"Detected {anchors.sum().item()} anchor frames out of {len(anchors)}")

    print("\nEnfusion preprocessing test completed successfully!")
