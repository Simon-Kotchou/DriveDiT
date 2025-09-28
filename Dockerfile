# Multi-stage production Dockerfile for DriveDiT
# Optimized for minimal dependencies and efficient training

# ============================
# Base stage with CUDA runtime
# ============================
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash drivedit
USER drivedit
WORKDIR /home/drivedit

# ============================
# Development stage
# ============================
FROM base as development

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --user --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=drivedit:drivedit . ./drivedit/
WORKDIR /home/drivedit/drivedit

# Set Python path
ENV PYTHONPATH=/home/drivedit/drivedit:$PYTHONPATH

# Development entrypoint
CMD ["python3", "-m", "training.self_forcing"]

# ============================
# Production training stage
# ============================
FROM base as production

# Install only essential dependencies
RUN python3 -m pip install --user --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    einops==0.7.0 \
    opencv-python==4.10.0.82 \
    numpy==1.24.3

# Copy only necessary source files
COPY --chown=drivedit:drivedit layers/ ./drivedit/layers/
COPY --chown=drivedit:drivedit blocks/ ./drivedit/blocks/
COPY --chown=drivedit:drivedit models/ ./drivedit/models/
COPY --chown=drivedit:drivedit training/ ./drivedit/training/
COPY --chown=drivedit:drivedit data/ ./drivedit/data/
COPY --chown=drivedit:drivedit utils/ ./drivedit/utils/
COPY --chown=drivedit:drivedit __init__.py ./drivedit/
COPY --chown=drivedit:drivedit CLAUDE.md ./drivedit/

WORKDIR /home/drivedit/drivedit

# Set Python path
ENV PYTHONPATH=/home/drivedit/drivedit:$PYTHONPATH

# Create directories for data and checkpoints
RUN mkdir -p /home/drivedit/data /home/drivedit/checkpoints /home/drivedit/logs

# Production entrypoint
ENTRYPOINT ["python3", "-m"]
CMD ["training.self_forcing"]

# ============================
# Inference stage (minimal)
# ============================
FROM base as inference

# Install minimal dependencies for inference
RUN python3 -m pip install --user --no-cache-dir \
    torch==2.2.1 \
    einops==0.7.0 \
    opencv-python==4.10.0.82 \
    numpy==1.24.3

# Copy only inference components
COPY --chown=drivedit:drivedit layers/ ./drivedit/layers/
COPY --chown=drivedit:drivedit blocks/ ./drivedit/blocks/
COPY --chown=drivedit:drivedit models/ ./drivedit/models/
COPY --chown=drivedit:drivedit inference/ ./drivedit/inference/
COPY --chown=drivedit:drivedit utils/ ./drivedit/utils/
COPY --chown=drivedit:drivedit __init__.py ./drivedit/

WORKDIR /home/drivedit/drivedit

# Set Python path
ENV PYTHONPATH=/home/drivedit/drivedit:$PYTHONPATH

# Expose port for inference server
EXPOSE 8000

# Inference entrypoint
CMD ["python3", "-m", "inference.server"]

# ============================
# Data processing stage
# ============================
FROM base as data-processor

# Install dependencies for data processing
RUN python3 -m pip install --user --no-cache-dir \
    torch==2.2.1 \
    einops==0.7.0 \
    opencv-python==4.10.0.82 \
    numpy==1.24.3 \
    tqdm==4.65.0

# Copy data processing components
COPY --chown=drivedit:drivedit data/ ./drivedit/data/
COPY --chown=drivedit:drivedit utils/ ./drivedit/utils/
COPY --chown=drivedit:drivedit __init__.py ./drivedit/

WORKDIR /home/drivedit/drivedit

# Set Python path
ENV PYTHONPATH=/home/drivedit/drivedit:$PYTHONPATH

# Create data directories
RUN mkdir -p /home/drivedit/raw_data /home/drivedit/processed_data /home/drivedit/cache

# Data processing entrypoint
CMD ["python3", "-m", "data.pipeline"]