"""
Streaming utilities for DriveDiT real-time data pipeline.

Provides helper functions for:
- Telemetry dict to control tensor conversion
- Frame normalization (uint8 -> float16)
- Sequence windowing for training

These utilities are designed to work with the Rust drivedit_streaming module
but can also be used standalone for testing and development.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class TelemetryMapping:
    """
    Mapping configuration for Enfusion telemetry to DriveDiT control tensor.

    DriveDiT expects 6D control: [steering, accel, goal_x, goal_y, speed, heading_rate]
    Enfusion provides: steering, throttle, brake, speed, position, rotation vectors

    This mapping handles the conversion between formats.
    """
    # Output indices in 6D control tensor
    STEERING_IDX: int = 0
    ACCEL_IDX: int = 1
    GOAL_X_IDX: int = 2
    GOAL_Y_IDX: int = 3
    SPEED_IDX: int = 4
    HEADING_RATE_IDX: int = 5

    # Normalization ranges (matching CLAUDE.md spec)
    steering_range: Tuple[float, float] = (-1.0, 1.0)
    accel_range: Tuple[float, float] = (-5.0, 5.0)  # m/s^2
    goal_range: Tuple[float, float] = (-50.0, 50.0)  # meters
    speed_range: Tuple[float, float] = (0.0, 40.0)  # m/s
    heading_rate_range: Tuple[float, float] = (-1.0, 1.0)  # rad/s


def telemetry_to_control_tensor(
    telemetry: Dict[str, Any],
    mapping: Optional[TelemetryMapping] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert Enfusion telemetry dict to DriveDiT 6D control tensor.

    Args:
        telemetry: Dict from Rust streaming or Enfusion loader containing:
            - steering: float [-1, 1]
            - throttle: float [0, 1]
            - brake: float [0, 1]
            - speed_kmh: float (km/h)
            - pos: [x, y, z] world position
            - forward: [x, y, z] forward vector
            Optional:
            - acceleration: float (m/s^2)
            - goal_x, goal_y: relative goal position
            - heading_rate: rad/s
        mapping: TelemetryMapping configuration
        dtype: Output tensor dtype
        device: Output tensor device

    Returns:
        torch.Tensor: [6] control tensor normalized to DriveDiT ranges
    """
    mapping = mapping or TelemetryMapping()

    control = torch.zeros(6, dtype=dtype, device=device)

    # Steering: direct passthrough (already [-1, 1])
    steering = telemetry.get('steering', 0.0)
    if isinstance(steering, (list, np.ndarray)):
        steering = float(steering[0]) if len(steering) > 0 else 0.0
    control[mapping.STEERING_IDX] = _clamp_normalize(
        float(steering), mapping.steering_range
    )

    # Acceleration: compute from throttle/brake or use provided
    if 'acceleration' in telemetry:
        accel = float(telemetry['acceleration'])
    else:
        # Compute from throttle/brake: throttle is positive, brake is negative
        throttle = float(telemetry.get('throttle', 0.0))
        brake = float(telemetry.get('brake', 0.0))
        # Map [0,1] throttle to [0, max_accel], [0,1] brake to [-max_decel, 0]
        accel = throttle * mapping.accel_range[1] - brake * abs(mapping.accel_range[0])
    control[mapping.ACCEL_IDX] = _clamp_normalize(accel, mapping.accel_range)

    # Goal position: relative to vehicle (if provided)
    goal_x = float(telemetry.get('goal_x', 0.0))
    goal_y = float(telemetry.get('goal_y', 0.0))
    control[mapping.GOAL_X_IDX] = _clamp_normalize(goal_x, mapping.goal_range)
    control[mapping.GOAL_Y_IDX] = _clamp_normalize(goal_y, mapping.goal_range)

    # Speed: convert km/h to m/s if needed
    speed = float(telemetry.get('speed_kmh', telemetry.get('speed', 0.0)))
    if speed > mapping.speed_range[1] * 1.5:  # Likely km/h
        speed = speed / 3.6  # Convert to m/s
    control[mapping.SPEED_IDX] = _clamp_normalize(speed, mapping.speed_range)

    # Heading rate: use provided or compute from rotation delta
    heading_rate = float(telemetry.get('heading_rate', 0.0))
    control[mapping.HEADING_RATE_IDX] = _clamp_normalize(
        heading_rate, mapping.heading_rate_range
    )

    return control


def _clamp_normalize(value: float, range_: Tuple[float, float]) -> float:
    """Clamp value to range and normalize to [-1, 1]."""
    min_val, max_val = range_
    clamped = max(min_val, min(max_val, value))
    # Normalize to [-1, 1] for symmetric ranges or [0, 1] for positive ranges
    if min_val >= 0:
        # Positive range: normalize to [0, 1]
        return clamped / max_val if max_val != 0 else 0.0
    else:
        # Symmetric range: normalize to [-1, 1]
        return 2.0 * (clamped - min_val) / (max_val - min_val) - 1.0


def normalize_frames(
    pixels: Union[np.ndarray, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    channel_order: str = 'hwc'
) -> torch.Tensor:
    """
    Normalize frame pixels from uint8 [0, 255] to float [0, 1].

    Args:
        pixels: Input pixels as numpy array or tensor
            - Shape [H, W, C] or [T, H, W, C] (HWC format from Rust)
            - Shape [C, H, W] or [T, C, H, W] (CHW format)
            - dtype: uint8 or float
        dtype: Output tensor dtype (float16 recommended for training)
        device: Output tensor device
        channel_order: 'hwc' or 'chw' - input channel order

    Returns:
        torch.Tensor: Normalized frames in [T, C, H, W] or [C, H, W] format
            - dtype: float16 or float32
            - range: [0, 1]
    """
    # Convert to tensor if numpy
    if isinstance(pixels, np.ndarray):
        tensor = torch.from_numpy(pixels)
    else:
        tensor = pixels

    # Handle uint8 normalization
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype) / 255.0
    elif tensor.dtype in (torch.float32, torch.float64):
        tensor = tensor.to(dtype)
        # If values are in [0, 255], normalize
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        tensor = tensor.to(dtype)

    # Rearrange channels if needed (HWC -> CHW)
    if channel_order == 'hwc':
        if tensor.dim() == 3:  # [H, W, C]
            tensor = tensor.permute(2, 0, 1)  # -> [C, H, W]
        elif tensor.dim() == 4:  # [T, H, W, C]
            tensor = tensor.permute(0, 3, 1, 2)  # -> [T, C, H, W]

    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def batch_telemetry_to_controls(
    telemetry_list: List[Dict[str, Any]],
    mapping: Optional[TelemetryMapping] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a batch of telemetry dicts to control tensor.

    Args:
        telemetry_list: List of telemetry dicts
        mapping: TelemetryMapping configuration
        dtype: Output tensor dtype
        device: Output tensor device

    Returns:
        torch.Tensor: [T, 6] control tensor
    """
    controls = [
        telemetry_to_control_tensor(t, mapping, dtype, device)
        for t in telemetry_list
    ]
    return torch.stack(controls, dim=0)


class SequenceWindower:
    """
    Helper for extracting training sequences from streaming data.

    Maintains a sliding window buffer and yields complete sequences
    when enough frames have accumulated.
    """

    def __init__(
        self,
        sequence_length: int = 16,
        stride: int = 1,
        min_overlap: int = 0
    ):
        """
        Initialize sequence windower.

        Args:
            sequence_length: Number of frames per training sequence
            stride: Stride between consecutive sequences (1 = maximum overlap)
            min_overlap: Minimum overlap between sequences (for context)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.min_overlap = min_overlap

        # Frame buffer
        self._frames: List[torch.Tensor] = []
        self._controls: List[torch.Tensor] = []
        self._telemetry: List[Dict[str, Any]] = []
        self._frame_ids: List[int] = []

        # Position tracking
        self._next_yield_start = 0

    def add_frame(
        self,
        frame: torch.Tensor,
        control: torch.Tensor,
        telemetry: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Add a frame to the buffer and potentially yield a sequence.

        Args:
            frame: Single frame tensor [C, H, W]
            control: Control tensor [6]
            telemetry: Raw telemetry dict (optional, for debugging)
            frame_id: Frame ID for tracking (optional)

        Returns:
            Dict with 'frames' and 'controls' if sequence is ready, else None
        """
        self._frames.append(frame)
        self._controls.append(control)
        self._telemetry.append(telemetry or {})
        self._frame_ids.append(frame_id if frame_id is not None else len(self._frames) - 1)

        # Check if we can yield a sequence
        buffer_len = len(self._frames)
        if buffer_len >= self._next_yield_start + self.sequence_length:
            return self._extract_sequence(self._next_yield_start)

        return None

    def _extract_sequence(self, start_idx: int) -> Dict[str, torch.Tensor]:
        """Extract a sequence starting at the given index."""
        end_idx = start_idx + self.sequence_length

        frames = torch.stack(self._frames[start_idx:end_idx], dim=0)
        controls = torch.stack(self._controls[start_idx:end_idx], dim=0)
        frame_ids = self._frame_ids[start_idx:end_idx]

        # Update next yield position
        self._next_yield_start = start_idx + self.stride

        # Trim buffer to prevent unbounded growth
        # Keep at least min_overlap frames before next yield start
        trim_to = max(0, self._next_yield_start - self.min_overlap)
        if trim_to > 0:
            self._frames = self._frames[trim_to:]
            self._controls = self._controls[trim_to:]
            self._telemetry = self._telemetry[trim_to:]
            self._frame_ids = self._frame_ids[trim_to:]
            self._next_yield_start -= trim_to

        return {
            'frames': frames,
            'controls': controls,
            'frame_ids': frame_ids,
        }

    def flush(self) -> List[Dict[str, torch.Tensor]]:
        """
        Flush remaining complete sequences from buffer.

        Call this at the end of a session to get any remaining sequences.

        Returns:
            List of sequence dicts
        """
        sequences = []
        while len(self._frames) >= self._next_yield_start + self.sequence_length:
            seq = self._extract_sequence(self._next_yield_start)
            sequences.append(seq)
        return sequences

    def clear(self):
        """Clear the buffer completely."""
        self._frames.clear()
        self._controls.clear()
        self._telemetry.clear()
        self._frame_ids.clear()
        self._next_yield_start = 0

    @property
    def buffer_size(self) -> int:
        """Current number of frames in buffer."""
        return len(self._frames)

    @property
    def pending_frames(self) -> int:
        """Frames needed until next sequence is ready."""
        return max(0, self._next_yield_start + self.sequence_length - len(self._frames))


class StreamingBatcher:
    """
    Batches sequences from multiple streaming sources.

    Accumulates sequences and yields complete batches for training.
    """

    def __init__(
        self,
        batch_size: int = 8,
        drop_incomplete: bool = True
    ):
        """
        Initialize streaming batcher.

        Args:
            batch_size: Number of sequences per batch
            drop_incomplete: Whether to drop incomplete final batch
        """
        self.batch_size = batch_size
        self.drop_incomplete = drop_incomplete
        self._buffer: List[Dict[str, torch.Tensor]] = []

    def add_sequence(
        self,
        sequence: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Add a sequence and potentially yield a batch.

        Args:
            sequence: Dict with 'frames' [T, C, H, W] and 'controls' [T, 6]

        Returns:
            Batched dict if batch is ready, else None
        """
        self._buffer.append(sequence)

        if len(self._buffer) >= self.batch_size:
            return self._make_batch()

        return None

    def _make_batch(self) -> Dict[str, torch.Tensor]:
        """Create a batch from buffered sequences."""
        batch_seqs = self._buffer[:self.batch_size]
        self._buffer = self._buffer[self.batch_size:]

        # Stack tensors
        frames = torch.stack([s['frames'] for s in batch_seqs], dim=0)
        controls = torch.stack([s['controls'] for s in batch_seqs], dim=0)

        return {
            'frames': frames,  # [B, T, C, H, W]
            'controls': controls,  # [B, T, 6]
        }

    def flush(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Flush remaining sequences as a batch.

        Returns:
            Final batch if not dropping incomplete and buffer non-empty
        """
        if self._buffer and not self.drop_incomplete:
            batch = self._make_batch()
            self._buffer.clear()
            return batch
        return None

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        """Current number of sequences in buffer."""
        return len(self._buffer)


def validate_batch(
    batch: Dict[str, torch.Tensor],
    expected_sequence_length: int = 16,
    expected_image_size: Tuple[int, int] = (256, 256),
    check_nan: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate a training batch for common issues.

    Args:
        batch: Training batch dict
        expected_sequence_length: Expected T dimension
        expected_image_size: Expected (H, W)
        check_nan: Whether to check for NaN/Inf values

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Check required keys
    if 'frames' not in batch:
        errors.append("Missing 'frames' key")
    if 'controls' not in batch:
        errors.append("Missing 'controls' key")

    if errors:
        return False, errors

    frames = batch['frames']
    controls = batch['controls']

    # Check frames shape: [B, T, C, H, W]
    if frames.dim() != 5:
        errors.append(f"Frames should be 5D [B,T,C,H,W], got {frames.dim()}D")
    else:
        B, T, C, H, W = frames.shape
        if T != expected_sequence_length:
            errors.append(f"Sequence length {T} != expected {expected_sequence_length}")
        if C != 3:
            errors.append(f"Channels {C} != expected 3")
        if (H, W) != expected_image_size:
            errors.append(f"Image size ({H},{W}) != expected {expected_image_size}")

    # Check controls shape: [B, T, 6]
    if controls.dim() != 3:
        errors.append(f"Controls should be 3D [B,T,6], got {controls.dim()}D")
    elif controls.shape[-1] != 6:
        errors.append(f"Control dim {controls.shape[-1]} != expected 6")

    # Check value ranges
    if frames.min() < 0 or frames.max() > 1:
        errors.append(f"Frames out of [0,1] range: [{frames.min():.3f}, {frames.max():.3f}]")

    if controls.min() < -1.1 or controls.max() > 1.1:
        errors.append(f"Controls out of [-1,1] range: [{controls.min():.3f}, {controls.max():.3f}]")

    # Check for NaN/Inf
    if check_nan:
        if torch.isnan(frames).any():
            errors.append("NaN values in frames")
        if torch.isinf(frames).any():
            errors.append("Inf values in frames")
        if torch.isnan(controls).any():
            errors.append("NaN values in controls")
        if torch.isinf(controls).any():
            errors.append("Inf values in controls")

    return len(errors) == 0, errors


if __name__ == "__main__":
    # Test utilities
    print("Testing streaming utilities...")

    # Test telemetry conversion
    telemetry = {
        'steering': 0.5,
        'throttle': 0.8,
        'brake': 0.0,
        'speed_kmh': 60.0,
        'pos': [100.0, 50.0, 0.0],
        'forward': [1.0, 0.0, 0.0],
    }

    control = telemetry_to_control_tensor(telemetry)
    print(f"Control tensor shape: {control.shape}")
    print(f"Control values: {control}")

    # Test frame normalization
    fake_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    normalized = normalize_frames(fake_frame, dtype=torch.float16)
    print(f"Normalized frame shape: {normalized.shape}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test sequence windower
    windower = SequenceWindower(sequence_length=4, stride=2)

    for i in range(10):
        frame = torch.randn(3, 64, 64)
        ctrl = torch.randn(6)
        result = windower.add_frame(frame, ctrl, frame_id=i)
        if result is not None:
            print(f"Sequence ready at frame {i}: frames={result['frames'].shape}")

    remaining = windower.flush()
    print(f"Flushed {len(remaining)} remaining sequences")

    # Test batch validation
    batch = {
        'frames': torch.rand(8, 16, 3, 256, 256),
        'controls': torch.randn(8, 16, 6).clamp(-1, 1)
    }
    is_valid, errors = validate_batch(batch)
    print(f"Batch valid: {is_valid}, errors: {errors}")

    print("\nStreaming utilities test completed!")
