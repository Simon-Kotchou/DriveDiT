"""
Streaming data pipeline for DriveDiT real-time training.

Provides PyTorch-compatible streaming data loading from Enfusion game capture.
Works with the Rust drivedit_streaming module for high-performance, zero-copy
data transfer from the game engine to the training pipeline.

Architecture:
    Enfusion (BMP @ 5Hz) -> Rust Watcher -> Ring Buffer -> Python Iterator -> PyTorch

Features:
- Zero-copy frame transfer via numpy views
- GIL-release during Rust operations for parallel Python execution
- Graceful fallback to file-based loading when Rust module unavailable
- Sequence windowing for variable-length training sequences
- Backpressure handling when training falls behind capture rate

Usage:
    from data.streaming import StreamingDataLoader

    loader = StreamingDataLoader(
        watch_path="$profile:DrivingData/session_12345",
        batch_size=8,
        sequence_length=16,
        buffer_size=256
    )

    for batch in loader:
        frames = batch['frames']  # [B, T, C, H, W] torch.Tensor
        controls = batch['controls']  # [B, T, 6] torch.Tensor
        loss = trainer.train_step(batch, optimizer)
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
from abc import ABC, abstractmethod

from .streaming_utils import (
    TelemetryMapping,
    telemetry_to_control_tensor,
    normalize_frames,
    batch_telemetry_to_controls,
    SequenceWindower,
    StreamingBatcher,
    validate_batch,
)

# Try to import the Rust streaming extension
try:
    import drivedit_streaming as rust_streaming
    RUST_STREAMING_AVAILABLE = True
except ImportError:
    RUST_STREAMING_AVAILABLE = False
    rust_streaming = None


def check_streaming_available() -> bool:
    """Check if the Rust streaming extension is available."""
    return RUST_STREAMING_AVAILABLE


@dataclass
class StreamingConfig:
    """Configuration for streaming data loader."""

    # Sequence parameters
    sequence_length: int = 16
    stride: int = 8  # Stride between sequences (for overlap)

    # Image parameters
    image_size: Tuple[int, int] = (256, 256)
    normalize: bool = True

    # Buffer parameters
    buffer_size: int = 256  # Ring buffer capacity (frames)
    prefetch_batches: int = 2  # Batches to prefetch

    # Control signal parameters
    control_dim: int = 6
    telemetry_mapping: TelemetryMapping = field(default_factory=TelemetryMapping)

    # Performance tuning
    use_float16: bool = True  # Use half precision for frames
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    timeout_ms: int = 5000  # Timeout waiting for frames

    # Fallback behavior
    fallback_to_file: bool = True  # Fall back to file-based loading
    session_dir: Optional[str] = None  # Directory for file-based fallback


class StreamingSource(ABC):
    """Abstract base class for streaming data sources."""

    @abstractmethod
    def start(self) -> None:
        """Start the streaming source."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the streaming source."""
        pass

    @abstractmethod
    def get_frame(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Get the next frame from the source.

        Returns:
            Dict with 'pixels', 'telemetry', 'frame_id', 'timestamp_ms'
            or None if timeout/end of stream
        """
        pass

    @abstractmethod
    def get_batch(
        self,
        count: int,
        timeout_ms: int = 5000
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get a batch of frames from the source.

        Returns:
            List of frame dicts or None if timeout
        """
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the source is running."""
        pass

    @property
    @abstractmethod
    def buffer_fill(self) -> float:
        """Get buffer fill ratio [0, 1]."""
        pass


class RustStreamingSource(StreamingSource):
    """
    Streaming source using Rust drivedit_streaming module.

    High-performance implementation with:
    - Kernel-level file watching (inotify/kqueue)
    - SIMD-optimized BMP decoding
    - Lock-free ring buffer
    - Zero-copy numpy array creation
    """

    def __init__(
        self,
        watch_path: str,
        config: StreamingConfig
    ):
        """
        Initialize Rust streaming source.

        Args:
            watch_path: Path to watch for new frames
                - Can be absolute path
                - Can use $profile: prefix for Enfusion profile directory
            config: Streaming configuration
        """
        if not RUST_STREAMING_AVAILABLE:
            raise RuntimeError(
                "Rust streaming extension 'drivedit_streaming' not available. "
                "Build with: cd streaming && maturin develop --release"
            )

        self.watch_path = watch_path
        self.config = config

        # Create Rust streamer
        self._streamer = rust_streaming.FrameStreamer(
            watch_path=watch_path,
            buffer_size=config.buffer_size,
            image_width=config.image_size[1],
            image_height=config.image_size[0],
        )

        self._running = False

    def start(self) -> None:
        """Start watching for frames."""
        self._streamer.start()
        self._running = True

    def stop(self) -> None:
        """Stop watching for frames."""
        self._streamer.stop()
        self._running = False

    def get_frame(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Get the next frame with telemetry."""
        if not self._running:
            return None

        try:
            # This releases the GIL while waiting
            frame = self._streamer.get_frame(timeout_ms=timeout_ms)
            if frame is None:
                return None

            return {
                'pixels': frame.pixels,  # numpy array [H, W, 3] uint8
                'telemetry': frame.telemetry,  # dict
                'frame_id': frame.frame_id,
                'timestamp_ms': frame.timestamp_ms,
            }
        except Exception as e:
            warnings.warn(f"Error getting frame: {e}")
            return None

    def get_batch(
        self,
        count: int,
        timeout_ms: int = 5000
    ) -> Optional[List[Dict[str, Any]]]:
        """Get a batch of frames."""
        if not self._running:
            return None

        try:
            # Rust handles batch collection efficiently
            frames = self._streamer.get_batch(count=count, timeout_ms=timeout_ms)
            if frames is None:
                return None

            return [
                {
                    'pixels': f.pixels,
                    'telemetry': f.telemetry,
                    'frame_id': f.frame_id,
                    'timestamp_ms': f.timestamp_ms,
                }
                for f in frames
            ]
        except Exception as e:
            warnings.warn(f"Error getting batch: {e}")
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def buffer_fill(self) -> float:
        return self._streamer.buffer_fill if self._running else 0.0


class FileStreamingSource(StreamingSource):
    """
    Fallback streaming source using file-based loading.

    Used when Rust module is unavailable. Monitors a session directory
    for new frames and simulates streaming behavior.
    """

    def __init__(
        self,
        session_dir: str,
        config: StreamingConfig
    ):
        """
        Initialize file-based streaming source.

        Args:
            session_dir: Path to session directory with frames/
            config: Streaming configuration
        """
        self.session_dir = Path(session_dir)
        self.config = config
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.buffer_size)
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track processed frames
        self._processed_frames: set = set()
        self._frame_counter = 0

    def start(self) -> None:
        """Start watching for new frames."""
        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        self._running = True

    def stop(self) -> None:
        """Stop watching for frames."""
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2.0)
        self._running = False

    def _watch_loop(self) -> None:
        """Background thread that watches for new frames."""
        import cv2

        frames_dir = self.session_dir / "frames"
        telemetry_path = self.session_dir / "telemetry.csv"

        # Load telemetry if available
        telemetry_data = self._load_telemetry(telemetry_path)

        while not self._stop_event.is_set():
            # Find new frames
            if frames_dir.exists():
                frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.bmp"))

                for frame_path in frame_files:
                    if frame_path.name in self._processed_frames:
                        continue

                    if self._stop_event.is_set():
                        break

                    try:
                        # Load and process frame
                        frame = cv2.imread(str(frame_path))
                        if frame is None:
                            continue

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Resize if needed
                        target_h, target_w = self.config.image_size
                        if frame.shape[:2] != (target_h, target_w):
                            frame = cv2.resize(frame, (target_w, target_h))

                        # Get telemetry for this frame
                        frame_idx = self._frame_counter
                        telemetry = {}
                        if telemetry_data and frame_idx < len(telemetry_data):
                            telemetry = telemetry_data[frame_idx]

                        frame_dict = {
                            'pixels': frame,  # [H, W, 3] uint8
                            'telemetry': telemetry,
                            'frame_id': self._frame_counter,
                            'timestamp_ms': time.time() * 1000,
                        }

                        # Add to queue (blocks if full - backpressure)
                        try:
                            self._frame_queue.put(frame_dict, timeout=0.1)
                            self._processed_frames.add(frame_path.name)
                            self._frame_counter += 1
                        except queue.Full:
                            pass  # Drop frame if queue full

                    except Exception as e:
                        warnings.warn(f"Error loading frame {frame_path}: {e}")

            # Small sleep to avoid busy-waiting
            self._stop_event.wait(0.05)

    def _load_telemetry(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load telemetry CSV file."""
        if not csv_path.exists():
            return []

        import csv
        telemetry = []

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    telemetry.append({
                        'steering': float(row.get('steering', 0)),
                        'throttle': float(row.get('throttle', 0)),
                        'brake': float(row.get('brake', 0)),
                        'speed_kmh': float(row.get('speed', 0)),
                        'speed': float(row.get('speed', 0)),
                    })
        except Exception as e:
            warnings.warn(f"Error loading telemetry: {e}")

        return telemetry

    def get_frame(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Get the next frame."""
        try:
            return self._frame_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None

    def get_batch(
        self,
        count: int,
        timeout_ms: int = 5000
    ) -> Optional[List[Dict[str, Any]]]:
        """Get a batch of frames."""
        frames = []
        deadline = time.time() + (timeout_ms / 1000.0)

        while len(frames) < count and time.time() < deadline:
            remaining = max(0.001, deadline - time.time())
            frame = self.get_frame(timeout_ms=int(remaining * 1000))
            if frame is not None:
                frames.append(frame)

        return frames if frames else None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def buffer_fill(self) -> float:
        return self._frame_queue.qsize() / self.config.buffer_size


class StreamingEnfusionDataset(data.IterableDataset):
    """
    PyTorch IterableDataset for streaming Enfusion capture data.

    Yields batched sequences in real-time as frames arrive from the game.
    Works with either Rust streaming module or file-based fallback.

    Note: This is an IterableDataset, not a map-style Dataset, because
    streaming data has no fixed length and samples arrive in order.
    """

    def __init__(
        self,
        watch_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ):
        """
        Initialize streaming dataset.

        Args:
            watch_path: Path to watch for new frames (Rust streaming)
            session_dir: Session directory for file-based fallback
            config: Streaming configuration
            transform: Optional transform to apply to batches
        """
        super().__init__()

        self.config = config or StreamingConfig()
        self.transform = transform

        # Initialize source
        if watch_path and RUST_STREAMING_AVAILABLE:
            self._source: StreamingSource = RustStreamingSource(watch_path, self.config)
            self._mode = "rust"
        elif session_dir or self.config.session_dir:
            dir_path = session_dir or self.config.session_dir
            self._source = FileStreamingSource(dir_path, self.config)
            self._mode = "file"
        elif watch_path and not RUST_STREAMING_AVAILABLE:
            if self.config.fallback_to_file:
                # Try to convert watch_path to session_dir
                warnings.warn(
                    "Rust streaming unavailable, falling back to file-based loading. "
                    "Install drivedit_streaming for better performance."
                )
                self._source = FileStreamingSource(watch_path, self.config)
                self._mode = "file_fallback"
            else:
                raise RuntimeError(
                    "Rust streaming unavailable and fallback disabled. "
                    "Set config.fallback_to_file=True or install drivedit_streaming."
                )
        else:
            raise ValueError(
                "Must provide either watch_path (for Rust streaming) "
                "or session_dir (for file-based loading)"
            )

        # Sequence windower for building training sequences
        self._windower = SequenceWindower(
            sequence_length=self.config.sequence_length,
            stride=self.config.stride,
        )

        # Batcher for accumulating sequences into batches
        self._batcher: Optional[StreamingBatcher] = None  # Set externally if needed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over streaming data.

        Yields:
            Dict containing:
                - frames: [T, C, H, W] tensor
                - controls: [T, 6] tensor
        """
        # Start the source
        self._source.start()
        self._windower.clear()

        dtype = torch.float16 if self.config.use_float16 else torch.float32

        try:
            while self._source.is_running:
                # Get next frame
                frame_data = self._source.get_frame(timeout_ms=self.config.timeout_ms)

                if frame_data is None:
                    # Timeout - check if source is still running
                    continue

                # Process frame
                pixels = frame_data['pixels']
                telemetry = frame_data['telemetry']
                frame_id = frame_data.get('frame_id', 0)

                # Normalize pixels: [H, W, C] uint8 -> [C, H, W] float
                frame_tensor = normalize_frames(
                    pixels,
                    dtype=dtype,
                    channel_order='hwc'
                )

                # Convert telemetry to control tensor
                control_tensor = telemetry_to_control_tensor(
                    telemetry,
                    mapping=self.config.telemetry_mapping,
                    dtype=torch.float32,
                )

                # Add to windower
                sequence = self._windower.add_frame(
                    frame_tensor,
                    control_tensor,
                    telemetry=telemetry,
                    frame_id=frame_id
                )

                if sequence is not None:
                    # Apply transform if provided
                    if self.transform is not None:
                        sequence = self.transform(sequence)

                    yield sequence

        finally:
            # Cleanup
            self._source.stop()

            # Flush remaining sequences
            for sequence in self._windower.flush():
                if self.transform is not None:
                    sequence = self.transform(sequence)
                yield sequence

    @property
    def mode(self) -> str:
        """Get the current streaming mode."""
        return self._mode

    @property
    def buffer_fill(self) -> float:
        """Get the current buffer fill ratio."""
        return self._source.buffer_fill


class StreamingDataLoader:
    """
    High-level streaming data loader for DriveDiT training.

    Wraps StreamingEnfusionDataset with batching, GPU transfer,
    and monitoring capabilities.

    Example:
        loader = StreamingDataLoader(
            watch_path="$profile:DrivingData/session_12345",
            batch_size=8,
            sequence_length=16
        )

        for batch in loader:
            frames = batch['frames']  # [B, T, C, H, W]
            controls = batch['controls']  # [B, T, 6]
            loss = trainer.train_step(batch, optimizer)
    """

    def __init__(
        self,
        watch_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        batch_size: int = 8,
        sequence_length: int = 16,
        buffer_size: int = 256,
        image_size: Tuple[int, int] = (256, 256),
        use_float16: bool = True,
        pin_memory: bool = True,
        device: Optional[torch.device] = None,
        config: Optional[StreamingConfig] = None,
        transform: Optional[Callable] = None,
        drop_incomplete: bool = True,
    ):
        """
        Initialize streaming data loader.

        Args:
            watch_path: Path to watch for new frames (Rust streaming)
            session_dir: Session directory for file-based fallback
            batch_size: Number of sequences per batch
            sequence_length: Frames per sequence
            buffer_size: Ring buffer capacity
            image_size: Target image size (H, W)
            use_float16: Use half precision for frames
            pin_memory: Pin memory for faster GPU transfer
            device: Target device for tensors (default: cuda if available)
            config: Override all settings with StreamingConfig
            transform: Optional transform for batches
            drop_incomplete: Drop incomplete final batch
        """
        # Build config
        if config is None:
            config = StreamingConfig(
                sequence_length=sequence_length,
                image_size=image_size,
                buffer_size=buffer_size,
                use_float16=use_float16,
                pin_memory=pin_memory,
            )

        self.config = config
        self.batch_size = batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.transform = transform
        self.drop_incomplete = drop_incomplete

        # Create dataset
        self._dataset = StreamingEnfusionDataset(
            watch_path=watch_path,
            session_dir=session_dir,
            config=config,
            transform=None,  # We apply transform after batching
        )

        # Batcher
        self._batcher = StreamingBatcher(
            batch_size=batch_size,
            drop_incomplete=drop_incomplete,
        )

        # Metrics
        self._batches_yielded = 0
        self._frames_processed = 0
        self._start_time: Optional[float] = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over batched streaming data.

        Yields:
            Dict containing:
                - frames: [B, T, C, H, W] tensor on target device
                - controls: [B, T, 6] tensor on target device
        """
        self._start_time = time.time()
        self._batches_yielded = 0
        self._frames_processed = 0
        self._batcher.clear()

        for sequence in self._dataset:
            self._frames_processed += sequence['frames'].shape[0]

            # Add to batcher
            batch = self._batcher.add_sequence(sequence)

            if batch is not None:
                # Apply transform if provided
                if self.transform is not None:
                    batch = self.transform(batch)

                # Transfer to device
                batch = self._to_device(batch)

                self._batches_yielded += 1
                yield batch

        # Flush remaining
        final_batch = self._batcher.flush()
        if final_batch is not None:
            if self.transform is not None:
                final_batch = self.transform(final_batch)
            final_batch = self._to_device(final_batch)
            self._batches_yielded += 1
            yield final_batch

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transfer batch to target device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if self.config.pin_memory and self.device.type == 'cuda':
                    # Use non-blocking transfer with pinned memory
                    if not value.is_pinned():
                        value = value.pin_memory()
                    result[key] = value.to(self.device, non_blocking=True)
                else:
                    result[key] = value.to(self.device)
            else:
                result[key] = value
        return result

    @property
    def mode(self) -> str:
        """Get the streaming mode (rust/file/file_fallback)."""
        return self._dataset.mode

    @property
    def buffer_fill(self) -> float:
        """Get the current buffer fill ratio [0, 1]."""
        return self._dataset.buffer_fill

    @property
    def batches_yielded(self) -> int:
        """Get number of batches yielded since iteration started."""
        return self._batches_yielded

    @property
    def frames_per_second(self) -> float:
        """Get average frames processed per second."""
        if self._start_time is None or self._frames_processed == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._frames_processed / elapsed if elapsed > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            'mode': self.mode,
            'buffer_fill': self.buffer_fill,
            'batches_yielded': self._batches_yielded,
            'frames_processed': self._frames_processed,
            'fps': self.frames_per_second,
            'rust_available': RUST_STREAMING_AVAILABLE,
        }


def create_streaming_loader(
    watch_path: Optional[str] = None,
    session_dir: Optional[str] = None,
    batch_size: int = 8,
    sequence_length: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    **kwargs
) -> StreamingDataLoader:
    """
    Factory function to create a streaming data loader.

    Automatically selects the best available streaming backend.

    Args:
        watch_path: Path to watch for new frames
        session_dir: Session directory for file-based loading
        batch_size: Batch size
        sequence_length: Sequence length
        image_size: Target image size
        **kwargs: Additional arguments for StreamingDataLoader

    Returns:
        StreamingDataLoader instance
    """
    return StreamingDataLoader(
        watch_path=watch_path,
        session_dir=session_dir,
        batch_size=batch_size,
        sequence_length=sequence_length,
        image_size=image_size,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing streaming data pipeline...")
    print(f"Rust streaming available: {RUST_STREAMING_AVAILABLE}")

    # Test with mock data (file-based fallback)
    import tempfile
    import os

    # Create mock session directory
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir)
        frames_dir = session_dir / "frames"
        frames_dir.mkdir()

        # Create mock frames
        import cv2
        for i in range(20):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), frame)

        # Create mock telemetry
        with open(session_dir / "telemetry.csv", 'w') as f:
            f.write("steering,throttle,brake,speed\n")
            for i in range(20):
                f.write(f"{0.1*i:.3f},{0.5:.3f},{0.0:.3f},{30.0:.3f}\n")

        print(f"Created mock session at {session_dir}")

        # Create streaming loader
        config = StreamingConfig(
            sequence_length=4,
            stride=2,
            image_size=(64, 64),
            buffer_size=16,
            timeout_ms=2000,
        )

        loader = StreamingDataLoader(
            session_dir=str(session_dir),
            batch_size=2,
            config=config,
        )

        print(f"Loader mode: {loader.mode}")

        # Iterate (with timeout)
        batch_count = 0
        max_batches = 3

        for batch in loader:
            frames = batch['frames']
            controls = batch['controls']
            print(f"Batch {batch_count}: frames={frames.shape}, controls={controls.shape}")

            # Validate batch
            is_valid, errors = validate_batch(
                batch,
                expected_sequence_length=config.sequence_length,
                expected_image_size=config.image_size,
            )
            print(f"  Valid: {is_valid}, Errors: {errors}")

            batch_count += 1
            if batch_count >= max_batches:
                break

        print(f"\nStats: {loader.stats()}")
        print("\nStreaming pipeline test completed!")
