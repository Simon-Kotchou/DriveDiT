# drivedit-py

High-performance Python bindings for the DriveDiT data pipeline, providing both batch and streaming APIs for autonomous driving world model training.

## Installation

```bash
# From source with maturin
pip install maturin
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/drivedit-*.whl
```

## Quick Start

### Batch Loading

Load data from stored datasets for training:

```python
import drivedit

# Load an entire dataset
dataset = drivedit.EnfusionDataset("/path/to/data")
print(f"Total samples: {len(dataset)}")

for frames, controls, depth in dataset:
    # frames: [T, C, H, W] float32
    # controls: [T, 4] float32
    # depth: [T, 1, H, W] float32 or None
    pass

# Load a single session
session = drivedit.EnfusionSession("/path/to/session")
frames = session.get_frames(0)  # [T, C, H, W]
controls = session.get_controls(0)  # [T, 4]
```

### Streaming

Real-time ingestion from live Enfusion game capture:

```python
from drivedit.streaming import StreamingLoader

loader = StreamingLoader("/path/to/live", buffer_size=256)
loader.start()

for batch in loader:
    frames = batch["frames"]      # [B, H, W, 3] uint8
    telemetry = batch["telemetry"] # [B, 20] float32
    # Training loop...

loader.stop()
```

### Configuration

```python
from drivedit.batch import DatasetConfig, LoaderConfig

# Custom dataset configuration
config = DatasetConfig(
    sequence_length=16,
    image_height=256,
    image_width=512,
    control_dim=4,
)

# Loader configuration
loader_config = LoaderConfig.auto_detect()

dataset = drivedit.EnfusionDataset(
    "/path/to/data",
    config=config,
    loader_config=loader_config,
)
```

## Module Structure

- `drivedit` - Top-level module with convenience re-exports
- `drivedit.batch` - Batch loading APIs for stored datasets
- `drivedit.streaming` - Real-time streaming APIs for live capture
- `drivedit.types` - Shared type wrappers (Telemetry, Frame, etc.)

## Utility Functions

```python
import drivedit
import numpy as np

# Convert between HWC and CHW formats
hwc = np.random.randint(0, 256, (8, 360, 640, 3), dtype=np.uint8)
chw = drivedit.hwc_to_chw(hwc)  # [8, 3, 360, 640] float32

# Normalize frames
normalized = drivedit.normalize_frames(hwc)  # float32 [0, 1]
```

## Performance

- Zero-copy numpy array creation where possible
- GIL release during blocking I/O operations
- Parallel frame loading with rayon
- Memory-mapped file access for large datasets
- Lock-free ring buffer for streaming

## Requirements

- Python >= 3.10
- NumPy >= 1.24
- Rust toolchain (for building from source)

## License

MIT
