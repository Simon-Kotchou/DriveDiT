# DriveDiT: Zero-Dependency Diffusion Transformer for Autonomous Driving

A minimal, production-ready implementation of a diffusion transformer for autonomous driving world modeling. Built with zero external dependencies beyond PyTorch, einops, and OpenCV.

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run development environment
docker-compose up drivedit-dev

# For training
docker-compose up drivedit-train

# For inference server
docker-compose up drivedit-inference
```

### Local Installation

```bash
# Install minimal dependencies
pip install torch>=2.2.0 einops>=0.7.0 opencv-python==4.10.0 torchmetrics

# Development tools (optional)
pip install jupyter black ruff

# Clone and run
git clone <repository>
cd drivedit
python training/unified_trainer.py
```

### Rust Data Pipeline (Optional)

```bash
cd rust/drivedit-data
pip install maturin
maturin develop --release
```

## Architecture

```
drivedit/
├── layers/              # Mathematical primitives
│   ├── rope.py                # Rotary positional embedding
│   ├── rope_v2.py             # Extended RoPE with 3D support
│   ├── mha.py                 # Multi-head attention (causal/bidirectional)
│   ├── mlp.py                 # Feed-forward networks
│   └── nn_helpers.py          # SiLU, RMSNorm, fused operations
│
├── blocks/              # Composite components
│   ├── dit_block.py           # Attention + MLP + AdaLN
│   └── flow_matching.py       # Complete flow matching implementation
│
├── core/                # Advanced components
│   ├── base.py                # Core base classes and utilities
│   └── components.py          # RoPE3D, SAM2 tracker, flow predictor
│
├── models/              # Complete architectures
│   ├── world_model.py         # Unified world model with all components
│   ├── vae3d.py               # 3D causal VAE (WAN distilled)
│   ├── vae3d_v2.py            # Extended VAE with improvements
│   ├── dit_student.py         # Causal world model (student)
│   ├── dit_teacher.py         # Bidirectional teacher
│   └── conditioning.py        # Conditioning modules
│
├── config/              # Configuration system
│   ├── config.py              # Unified DriveDiTConfig
│   ├── base_config.py         # Base configuration classes
│   └── model_config.py        # Model-specific configurations
│
├── data/                # Data pipeline
│   ├── pipeline.py            # Memory-mapped video processing
│   ├── video_chunking.py      # Intelligent video chunking
│   ├── large_scale_processing.py  # 100k+ hour dataset processing
│   ├── enfusion_loader.py     # Enfusion ENFCAP format loader
│   ├── rust_loader.py         # Rust-accelerated data loading
│   ├── hybrid_loader.py       # Multi-source hybrid loader
│   ├── transforms.py          # Video augmentations
│   └── preprocessing.py       # Data preprocessing utilities
│
├── training/            # Training loops
│   ├── unified_trainer.py     # Complete unified training pipeline
│   ├── self_forcing.py        # Self-forcing training
│   ├── self_forcing_plus.py   # Self-Forcing++ (extended generation)
│   ├── distill.py             # Flow-matching distillation
│   ├── distributed.py         # Multi-GPU training support
│   ├── losses.py              # Loss functions
│   ├── losses_v2.py           # Extended loss implementations
│   ├── cuda_optimized.py      # CUDA-optimized training
│   └── kernels/               # Custom CUDA kernels
│       ├── fused_augment.py   # Fused augmentation kernels
│       └── fused_normalize.py # Fused normalization kernels
│
├── inference/           # Generation
│   ├── rollout.py             # Single-GPU streaming inference
│   ├── pipeline.py            # Inference pipeline
│   ├── server.py              # Inference server
│   └── optimization.py        # Inference optimization utilities
│
├── evaluation/          # Metrics and benchmarks
│   ├── metrics.py             # FVD, LPIPS, depth metrics
│   ├── evaluators.py          # Evaluation utilities
│   ├── benchmarks.py          # Performance benchmarks
│   └── visualization.py       # Result visualization
│
├── tests/               # Comprehensive test suite
│   ├── test_math_components.py    # Mathematical component tests
│   ├── test_training_pipeline.py  # Training pipeline tests
│   ├── test_self_forcing_plus.py  # Self-Forcing++ tests
│   ├── test_rust_loader.py        # Rust data loader tests
│   └── conftest.py                # Test fixtures
│
├── rust/                # Rust data pipeline
│   └── drivedit-data/         # PyO3-based data loader
│       ├── src/               # Rust source code
│       ├── python/            # Python bindings
│       └── benches/           # Performance benchmarks
│
├── DriveDiT_DataCapture/    # Arma Reforger mod (synthetic data)
│   ├── Scripts/Game/DataCapture/  # Enfusion capture scripts
│   ├── Prefabs/               # Vehicle and camera prefabs
│   ├── Configs/               # Mod configurations
│   └── README.md              # Mod documentation
│
├── src/enfusion/        # Enfusion script library
│   ├── datacapture/           # Data capture components
│   │   ├── core/              # Orchestrator, buffers, serializers
│   │   ├── modules/           # Telemetry, depth, road, scene modules
│   │   ├── SCR_MLDataCollector.c
│   │   ├── SCR_AnchorFrameSelector.c
│   │   ├── SCR_BinarySerializer.c
│   │   ├── SCR_MultiCameraRig.c
│   │   └── ...
│   └── navigation/            # Road extraction utilities
│       ├── extractors/        # Road topology extractors
│       └── visualizers/       # Debug visualizers
│
├── scripts/             # Utility scripts
│   ├── train.py               # Training entry point
│   └── process_data.py        # Data processing script
│
└── docs/                # Documentation
    └── REFERENCES.md          # Mathematical foundations
```

## Core Features

### Zero-Dependency Design
- **Pure PyTorch**: Only `torch >= 2.2`, `einops`, and `opencv`
- **Mathematical transparency**: Every operation implemented explicitly
- **No external model libraries**: Self-contained implementations

### Self-Forcing++ Training
Extended generation from 5 seconds to 4+ minutes using:
- **Rolling KV Cache**: Sliding window with auto-truncation and gradient detachment
- **Curriculum Scheduler**: Progressive sequence length (8→64) and self-forcing ratio (0→1)
- **Future Anchor Conditioning**: Goal states at 2s, 4s, 6s horizons
- **Extended 6D Control**: steering, accel, goal_x, goal_y, speed, heading_rate
- **Stability Improvements**: EMA, uncertainty weighting, per-layer gradient clipping

```python
from training.self_forcing_plus import SelfForcingPlusTrainer, get_default_config

config = get_default_config()
trainer = SelfForcingPlusTrainer(model, config)

for batch in dataloader:
    losses = trainer.train_step(batch, optimizer)
    # Curriculum automatically advances
```

### Unified World Model
Modular architecture with configurable components:
```python
from config.config import get_research_config, ComponentType
from models.world_model import WorldModel

config = get_research_config()
config.enable_component(ComponentType.CONTROL)
config.enable_component(ComponentType.MEMORY)
config.disable_component(ComponentType.DEPTH)

model = WorldModel(config)
```

### Rust Data Pipeline
High-performance data loading with PyO3 bindings:
```python
from data.rust_loader import RustVideoLoader

loader = RustVideoLoader(
    data_dir="/path/to/videos",
    batch_size=4,
    num_workers=8
)
```

### Enfusion Integration (Synthetic Data)
Arma Reforger mod for generating synthetic driving data:
- **Multi-camera capture**: Configurable camera rigs
- **Anchor frame selection**: For Self-Forcing++ training
- **Binary serialization**: ENFCAP format for efficient storage
- **Depth raycasting**: Ground-truth depth maps
- **Scene enumeration**: Entity tracking and labeling

## Mathematical Foundations

### Diffusion Transformer (DiT)
```python
# AdaLN-Zero conditioning with self-forcing
x = x + α₁ · Attention(AdaLN(x, c)) + α₂ · MLP(AdaLN(x, c))
```

### Flow Matching (4→1 step distillation)
```python
z_{t+1} = z_t - Δσ_t · f_θ(z_t, ctx)
Loss = ||f_θ(z_t) - (z_t - z_{t+1}^teacher)/Δσ_t||²
```

### Rotary Positional Embedding (3D extension)
```python
# Factorized (time, height, width) embeddings
def rope(x, sin, cos):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

See [docs/REFERENCES.md](docs/REFERENCES.md) for complete mathematical formulations.

## Running the Code

### Training

```bash
# Unified training pipeline
python training/unified_trainer.py

# Self-Forcing++ training
python training/self_forcing_plus.py

# Flow-matching distillation
python training/distill.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Mathematical component tests
python tests/test_math_components.py

# Training pipeline tests
python tests/test_training_pipeline.py

# Self-Forcing++ tests
python tests/test_self_forcing_plus.py

# Rust loader tests
python tests/test_rust_loader.py
```

### Models

```bash
# Test individual models
python models/dit_student.py
python models/vae3d.py
python models/world_model.py
```

### Inference

```bash
# Real-time rollout
python inference/rollout.py

# Start inference server
python -m inference.server --checkpoint checkpoints/best_model.pt
```

### Data Processing

```bash
# Large-scale data processing
python data/large_scale_processing.py

# Process videos to memory-mapped format
python scripts/process_data.py \
  --input_dir /path/to/videos \
  --output_dir /data/processed \
  --num_workers 8
```

## Tensor Contracts

### Model Inputs/Outputs
- **RGB frames**: `[B, T, 3, H, W]` (float16, normalized 0-1)
- **Control signals**: `[B, 6]` (steering, accel, goal_x, goal_y, speed, heading_rate)
- **Ego states**: `[B, T, 5]` (x, y, heading, speed, heading_rate)
- **Latent representations**: `[B, T, C, H//8, W//8]`
- **Attention tokens**: `[B, T, D]` where D=model_dim
- **KV cache**: `{'k': [B, past_T, H, D], 'v': [B, past_T, H, D]}`

### Control Signal Normalization
- **steering**: [-1, 1] (normalized)
- **acceleration**: [-5, 5] m/s²
- **goal_x/goal_y**: [-50, 50] m (relative position)
- **speed**: [0, 40] m/s
- **heading_rate**: [-1, 1] rad/s

## Performance Optimization

### Compilation and Acceleration
- `torch.compile(model, mode='reduce-overhead')` for 2x+ speedup
- Mixed precision training/inference
- Efficient einsum operations
- Custom CUDA kernels for fused operations

### Memory Optimization
- Chunked VAE processing for large sequences
- Gradient checkpointing for deep networks
- Rolling KV-cache with auto-truncation
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`

### Benchmarks
- **16GB GPU**: Batch size 4, sequence length 32, 256px resolution
- **Memory mapping**: 10x reduction in RAM usage for large datasets
- **Rust loader**: 3x faster data loading compared to pure Python

## Configuration

### Model Sizes
```python
# Compact model (research)
model_dim=256, num_layers=8, num_heads=8

# Production model (deployment)
model_dim=512, num_layers=12, num_heads=16

# Large model (best quality)
model_dim=768, num_layers=24, num_heads=24
```

### Training Presets
```python
from config.config import get_minimal_config, get_research_config, get_production_config

config = get_research_config()  # Balanced for experimentation
config = get_minimal_config()   # Fast iteration
config = get_production_config() # Full quality
```

## Enfusion Data Capture

The `DriveDiT_DataCapture/` directory contains an Arma Reforger mod for synthetic data generation:

### Components
- **SCR_MLDataCollector**: Main telemetry capture (CSV/binary)
- **SCR_AnchorFrameSelector**: Anchor frame selection for Self-Forcing++
- **SCR_BinarySerializer**: High-performance ENFCAP format
- **SCR_MultiCameraRig**: Configurable multi-view capture
- **SCR_DepthRaycaster**: Ground-truth depth generation
- **SCR_SceneEnumerator**: Entity tracking and labeling

### Anchor Frame Triggers
- `PERIODIC`: Regular interval anchors (default: every 30 seconds)
- `ROAD_CHANGE`: Road type transitions
- `JUNCTION`: Intersection approach/exit
- `STOP/START`: Vehicle stop and restart events
- `STEERING`: Large steering angle changes
- `SPEED_CHANGE`: Significant acceleration/deceleration

### Binary Format (ENFCAP v1)
- **Header**: 64 bytes (magic, version, frame count, timestamp, flags)
- **Index Table**: Frame offsets for O(1) random access
- **Frame Records**: Variable-length binary with ego transform, vehicle state, scene entities

## Development

### Code Structure
- **Pure mathematics**: Explicit tensor operations with shape comments
- **Modular design**: Swappable components for research
- **Type hints**: Full typing for IDE support
- **Comprehensive tests**: Unit, integration, and performance tests

### Testing
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# All tests with coverage
python -m pytest tests/ --tb=short -v
```

## Documentation

- [Mathematical Foundations](docs/REFERENCES.md): Paper references and formulations
- [CLAUDE.md](CLAUDE.md): Development instructions for Claude Code
- [DriveDiT_DataCapture/README.md](DriveDiT_DataCapture/README.md): Enfusion mod documentation

## Acknowledgments

Built on insights from:
- **DiT**: Scalable Diffusion Models with Transformers
- **Self-Forcing**: Bridging Train-Test Gap in Video Generation
- **Self-Forcing++**: Extended stable video generation
- **V-JEPA**: Video Joint Embedding Predictive Architecture
- **Flow Matching**: Continuous Normalizing Flows
- **RoPE**: Rotary Position Embedding
- **comma.ai**: Extended control and curriculum learning

## License

See [LICENSE](LICENSE) for details.

---

**Zero dependencies. Maximum performance. Production ready.**
