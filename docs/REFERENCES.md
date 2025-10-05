# Mathematical Foundations and Paper References

## Core Architecture Papers

### Diffusion Transformers (DiT)
- **Scalable Diffusion Models with Transformers** (Peebles & Xie, 2023)
  - DiT architecture: `AdaLN-Zero` conditioning blocks
  - Mathematical formulation: `x = x + α₁ · Attention(AdaLN(x, c)) + α₂ · MLP(AdaLN(x, c))`
  - Scaling laws for parameter count vs. generation quality

### Flow Matching & Rectified Flow
- **Flow Matching for Generative Modeling** (Lipman et al., 2023)
  - Continuous normalizing flows: `dz/dt = v_θ(z(t), t)`
  - Training objective: `L = E[||v_θ(z_t, t) - (z₁ - z₀)||²]`
  
- **Flow Straight and Fast** (Liu et al., 2023)
  - Rectified flow distillation for 1-step generation
  - 4→1 step distillation: `z_{t+1} = z_t - Δσ_t · f_θ(z_t, ctx)`

### Rotary Position Embedding (RoPE)
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
  - 2D rotation matrices: `R_θ = [[cos θ, -sin θ], [sin θ, cos θ]]`
  - Multi-axis extension for 3D: factorized (time, height, width) embeddings

### Causal World Models
- **World Models** (Ha & Schmidhuber, 2018)
  - Vision-action-memory architecture
  - Latent dynamics: `z_{t+1} = f(z_t, a_t) + ε`

- **Dreamer v3** (Hafner et al., 2023)
  - World model objective: `L = L_reconstruction + L_dynamics + L_representation`
  - KL regularization for stable latent dynamics

## Video Understanding & Generation

### Video Transformers
- **Video Vision Transformer (ViViT)** (Arnab et al., 2021)
  - Factorized spatio-temporal attention
  - Tubelet tokenization: `(T×H×W×C) → (N×D)` where `N = THW/P³`

- **V-JEPA** (Bardes et al., 2024)
  - Self-supervised video representation learning
  - Masked prediction in latent space: `L = ||f_θ(z_t) - z_{t+Δ}||²`

### 3D Causal Convolutions
- **WaveNet** (van den Oord et al., 2016)
  - Causal dilated convolutions for sequential modeling
  - Padding scheme: `pad_t = (kernel_size - 1, 0)` for causality

## Memory and Context Management

### Memory-Augmented Networks
- **Neural Turing Machines** (Graves et al., 2014)
  - External memory with read/write attention
  - Memory update: `M_t = M_{t-1} + w_t^w ⊗ a_t - E_t ⊙ M_{t-1}`

- **Transformer-XL** (Dai et al., 2019)
  - Segment-level recurrence with relative positional encoding
  - Memory reuse across segments for long sequences

## Autonomous Driving Specific

### End-to-End Driving
- **PlanT** (Renz et al., 2022)
  - Planning with transformers in latent space
  - Cost function: `J = Σ_t [c_collision(s_t) + c_comfort(a_t) + c_progress(s_t)]`

- **UniAD** (Hu et al., 2023)
  - Unified autonomous driving with multi-task transformers
  - Joint perception-prediction-planning objective

### Self-Forcing Training
- **Self-Forcing** (Zhang et al., 2024)
  - Scheduled sampling with curriculum learning
  - Mixing ratio: `λ_t = max(0, 1 - t/T_max)` for teacher forcing decay

## Mathematical Formulations

### Core Training Loss
```
L_total = λ₁·L_reconstruction + λ₂·L_flow + λ₃·L_jepa + λ₄·L_consistency

Where:
- L_reconstruction = ||VAE_decode(z_pred) - x_gt||²
- L_flow = ||f_θ(z_t) - (z_t - z_{t+1})/Δσ||²  
- L_jepa = ||MLP(z_t) - z_{t+Δ}||²
- L_consistency = KL(Student(x) || Teacher(x))
```

### Attention Mechanism
```python
# Multi-head attention with RoPE
Q, K, V = Linear(x).chunk(3, dim=-1)
Q_rot = RoPE(Q, pos_emb)
K_rot = RoPE(K, pos_emb)
Attention = softmax(Q_rot @ K_rot^T / √d_k) @ V
```

### 3D VAE Latent Space
```
Encoder: RGB(B,T,3,H,W) → z(B,T,C,H//8,W//8)
Decoder: z(B,T,C,H//8,W//8) → RGB(B,T,3,H,W)
Causal constraint: z_t depends only on {z_τ | τ ≤ t}
```

### Flow Matching Distillation
```
# Teacher (4-step) → Student (1-step)
z₁ = z₀ + ∫₀¹ f_teacher(z_t, t) dt
z₁_student = z₀ + f_student(z₀, 0)
L_distill = ||f_student(z₀) - ∫₀¹ f_teacher(z_t, t) dt||²
```

## Implementation References

### Efficient Transformers
- **Flash Attention** (Dao et al., 2022)
  - Memory-efficient attention: O(N) memory vs O(N²)
  - Tiling strategy for GPU memory hierarchy

- **Ring Attention** (Liu et al., 2023)
  - Distributed attention across devices
  - Communication-optimal partitioning

### Quantization & Acceleration
- **BitsAndBytes** (Dettmers et al., 2022)
  - 8-bit quantization for large models
  - Dynamic scaling for stable training

- **torch.compile** (PyTorch 2.0)
  - Graph optimization and kernel fusion
  - 2x+ speedup for transformer workloads

## Datasets and Benchmarks

### Autonomous Driving Datasets
- **nuScenes** (Caesar et al., 2020): 1000 scenes, 1.4M camera images
- **Waymo Open Dataset** (Sun et al., 2020): 1000 sequences, 9.9M LiDAR frames  
- **CARLA** (Dosovitskiy et al., 2017): Synthetic urban driving simulator

### Video Generation Benchmarks
- **Kinetics-700** (Carreira et al., 2019): 700K video clips, 700 action classes
- **Something-Something v2** (Goyal et al., 2017): 220K videos, temporal reasoning

## Hardware Optimization

### CUDA Kernels
- **Triton** (Tillet et al., 2019): Python-like GPU kernel programming
- **CuDNN** optimized operations for convolutions and attention
- **NCCL** for multi-GPU communication primitives

### Memory Management
```python
# Efficient memory allocation
torch.cuda.empty_cache()  # Clear unused memory
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"  # Fragment management
```