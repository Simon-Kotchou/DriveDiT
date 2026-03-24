# drivedit-data

High-performance Rust-based data loading for DriveDiT.

## Features

- Fast video decoding with parallel frame loading
- Efficient telemetry parsing (CSV and binary ENFCAP formats)
- Zero-copy frame buffer management
- Python bindings via PyO3

## Building

```bash
# Install maturin
pip install maturin

# Build and install in development mode
maturin develop --release
```

## Usage

```python
from drivedit_data import RustDataLoader, RustVideoDecoder

# Load video frames
decoder = RustVideoDecoder("video.mp4")
frames = decoder.decode_range(0, 100)

# Parse telemetry
from drivedit_data import RustTelemetryParser
parser = RustTelemetryParser("telemetry.csv")
data = parser.parse()
```
