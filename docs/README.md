# DriveDiT: Zero-Dependency Diffusion Transformer for Autonomous Driving

**A comprehensive, zero-dependency PyTorch implementation of a diffusion transformer for autonomous driving world modeling.**

## 🚀 Quick Start

```python
import torch
from drivedit import DiTStudent, VAE3D, StreamingRollout

# Load models
student = DiTStudent(latent_dim=64, d_model=512, n_layers=12, n_heads=8)
vae = VAE3D(in_channels=3, latent_dim=64)

# Create rollout for inference
rollout = StreamingRollout(student, vae, device='cuda')

# Generate video sequence
context_frames = torch.randn(1, 4, 3, 256, 256)
control_sequence = torch.randn(1, 12, 4)

result = rollout.generate_sequence(
    context_frames=context_frames,
    control_sequence=control_sequence,
    max_new_frames=8
)
```

## 📋 Features

- **Zero Dependencies**: Only PyTorch, einops, and OpenCV
- **Pure Mathematics**: Explicit tensor operations, no external model libraries
- **Comprehensive Testing**: 100+ unit and integration tests
- **Production Ready**: Memory-efficient inference, streaming support
- **Evaluation Framework**: Extensive metrics and benchmarking tools
- **Modular Design**: Easy to extend and customize

## 🏗️ Architecture

### Core Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Layers** | RoPE, Multi-Head Attention, MLP, Layer Norms | `layers/` |
| **Blocks** | DiT Block, Flow Matching Sampler | `blocks/` |
| **Models** | VAE3D, DiT Student/Teacher | `models/` |
| **Training** | Self-forcing, Distillation | `training/` |
| **Inference** | Streaming Rollout, Memory Bank | `inference/` |

### Mathematical Foundations

#### Rotary Positional Embedding (3-Axis)
```python
def rope_3d(x, sin, cos):
    # x: [B, T, H, W, D] where D is divisible by 6
    # Factorized rotation for time, height, width
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

#### Flow Matching Distillation
```
z_{i+1} = z_i - Δσ_i * f_θ(z_i, ctx)
Loss = ||f_θ(z_i) - (z_i - z_{i+1}^teacher)/Δσ_i||²
```

#### Self-Forcing Training
- **Teacher**: Bidirectional attention over complete sequences
- **Student**: Causal autoregressive generation using own predictions
- **Distillation**: 4-step → 1-step flow matching compression

## 🔧 Installation

```bash
# Minimal dependencies
pip install torch>=2.2 einops opencv-python==4.10.0

# Development tools
pip install pytest black ruff jupyter
```

## 📊 Data Pipeline

### Video Loading
```python
from drivedit.data import VideoDataset, VideoLoader

dataset = VideoDataset(
    data_root='./data/driving_videos',
    sequence_length=16,
    image_size=(256, 256),
    load_controls=True
)

loader = VideoLoader(dataset, batch_size=8, num_workers=4)
```

### Preprocessing
```python
from drivedit.data import VideoPreprocessor, AugmentationPipeline

preprocessor = VideoPreprocessor(target_size=(256, 256))
augmentation = AugmentationPipeline.create_training_pipeline()

# Process batch
processed_frames = preprocessor.preprocess_batch(frames)
augmented_frames = augmentation(processed_frames)
```

## 🏋️ Training

### Self-Forcing Training
```python
from drivedit.training import SelfForcingTrainer

trainer = SelfForcingTrainer(
    student_model=student,
    vae_model=vae,
    optimizer=optimizer,
    device='cuda'
)

# Train epoch
losses = trainer.train_epoch(dataloader)
```

### Teacher-Student Distillation
```python
from drivedit.training import DistillationTrainer

trainer = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    vae_model=vae,
    optimizer=optimizer
)

losses = trainer.train_epoch(dataloader)
```

## 🚀 Inference

### Streaming Rollout
```python
from drivedit.inference import StreamingRollout, InferenceConfig

config = InferenceConfig(
    max_sequence_length=64,
    context_window=8,
    use_kv_cache=True,
    mixed_precision=True
)

rollout = StreamingRollout(student, vae, config, device='cuda')

# Real-time generation
for frame_batch in video_stream:
    result = rollout.generate_sequence(
        context_frames=frame_batch,
        max_new_frames=1,
        temperature=0.8
    )
```

### Memory Bank
```python
from drivedit.models import MemoryBank

memory = MemoryBank(d_model=512, max_spatial_memory=1000)

# Update with new observations
memory.update(spatial_tokens, importance_scores)

# Retrieve relevant memories
memory_tokens = memory.get_memory_tokens(top_k=32)
```

## 📈 Evaluation

### Comprehensive Metrics
```python
from drivedit.evaluation import SequenceEvaluator, compute_all_metrics

evaluator = SequenceEvaluator(student, vae, device='cuda')

results = evaluator.evaluate_sequence(
    context_frames=context,
    target_frames=targets,
    controls=controls
)

# Results include: MSE, PSNR, SSIM, temporal consistency, etc.
```

### Performance Benchmarking
```python
from drivedit.evaluation import PerformanceBenchmark, WorldModelBenchmark

# Performance metrics
perf_benchmark = PerformanceBenchmark(student, device='cuda')
perf_results = perf_benchmark.run_performance_suite()

# Comprehensive evaluation
world_benchmark = WorldModelBenchmark(student, vae, device='cuda')
world_results = world_benchmark.run_full_benchmark(test_data)
```

## 🔧 Configuration

### Model Configuration
```python
from drivedit.config import ModelConfig, VAEConfig, StudentConfig

config = ModelConfig(
    vae=VAEConfig(
        in_channels=3,
        latent_dim=64,
        hidden_dims=[64, 128, 256, 512]
    ),
    student=StudentConfig(
        latent_dim=64,
        d_model=512,
        n_layers=12,
        n_heads=8,
        max_seq_len=256
    )
)

# Save/load configuration
config.save_json('config.json')
loaded_config = ModelConfig.from_json('config.json')
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/ -m performance

# With coverage
pytest tests/ --cov=drivedit --cov-report=html
```

## 📁 Project Structure

```
drivedit/
├── layers/              # Mathematical primitives
│   ├── rope.py         # Rotary positional embedding
│   ├── mha.py          # Multi-head attention
│   ├── mlp.py          # Feed-forward networks
│   └── nn_helpers.py   # Normalization, activations
├── blocks/             # Composite components
│   ├── dit_block.py    # DiT transformer block
│   └── flow_match.py   # Flow matching sampler
├── models/             # Complete architectures
│   ├── vae3d.py        # 3D causal VAE
│   ├── dit_student.py  # Causal world model
│   └── dit_teacher.py  # Bidirectional teacher
├── training/           # Training algorithms
│   ├── self_forcing.py # Autoregressive training
│   ├── distill.py      # Teacher-student distillation
│   └── losses.py       # Loss functions
├── inference/          # Generation and rollout
│   └── rollout.py      # Streaming inference
├── data/              # Data pipeline
│   ├── video_loader.py # Video dataset loading
│   ├── transforms.py   # Data augmentation
│   ├── preprocessing.py # Data preprocessing
│   ├── collate.py      # Batch collation
│   └── sampler.py      # Data sampling
├── evaluation/        # Metrics and benchmarks
│   ├── metrics.py      # Evaluation metrics
│   ├── benchmarks.py   # Performance benchmarks
│   ├── evaluators.py   # High-level evaluators
│   └── visualization.py # Result visualization
├── utils/             # Utility functions
│   ├── tensor_utils.py # Tensor operations
│   ├── model_utils.py  # Model utilities
│   └── math_utils.py   # Mathematical functions
├── config/            # Configuration management
│   ├── base_config.py  # Base configuration
│   └── model_config.py # Model configurations
├── tests/             # Comprehensive test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── conftest.py    # Test configuration
└── docs/              # Documentation
    ├── README.md       # This file
    ├── EXAMPLES.md     # Usage examples
    └── API.md          # API reference
```

## 🎯 Use Cases

### Autonomous Driving Simulation
- **World Model**: Predict future driving scenarios
- **Control Planning**: Test control strategies in simulation
- **Safety Validation**: Evaluate edge cases and failure modes

### Research Applications
- **Temporal Modeling**: Study long-range dependencies in video
- **Representation Learning**: Analyze learned driving representations
- **Generative Modeling**: Explore controllable video generation

### Production Deployment
- **Real-time Inference**: Streaming prediction for autonomous vehicles
- **Edge Deployment**: Memory-efficient models for embedded systems
- **Scalable Training**: Distributed training on large datasets

## 🔬 Technical Details

### Memory Efficiency
- **KV Caching**: Efficient attention computation for long sequences
- **Memory Bank**: Selective retention of important spatial information
- **Gradient Checkpointing**: Trade compute for memory during training

### Performance Optimizations
- **Mixed Precision**: FP16 training and inference
- **Torch Compile**: JIT compilation for 2x+ speedup
- **Batched Operations**: Vectorized computation across sequences

### Mathematical Purity
- **Explicit Tensors**: All operations use raw PyTorch tensors
- **No Hidden APIs**: Complete mathematical transparency
- **Reproducible**: Deterministic operations with proper seeding

## 📚 Citation

```bibtex
@software{drivedit2024,
  title={DriveDiT: Zero-Dependency Diffusion Transformer for Autonomous Driving},
  author={DriveDiT Contributors},
  year={2024},
  url={https://github.com/your-repo/drivedit}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: See `docs/` directory for detailed guides