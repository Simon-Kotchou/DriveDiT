"""
Fused data augmentation Triton kernels.
Combines flip, normalize, brightness, and contrast in a single GPU pass.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Try to import triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_flip_normalize_kernel(
        X_ptr,
        Y_ptr,
        Mean_ptr,
        Std_ptr,
        batch_size,
        channels,
        height,
        width,
        flip_flags_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused horizontal flip + channel-wise normalization.

        Each program handles one pixel position across all batches and channels.
        """
        # Get pixel index
        pid = tl.program_id(0)
        total_pixels = height * width

        if pid >= total_pixels:
            return

        # Convert to h, w coordinates
        h = pid // width
        w = pid % width

        # Process each batch and channel
        for b in range(batch_size):
            # Check if this batch should be flipped
            flip = tl.load(flip_flags_ptr + b)

            # Compute source w coordinate
            src_w = (width - 1 - w) if flip else w

            for c in range(channels):
                # Load mean and std for this channel
                mean = tl.load(Mean_ptr + c)
                std = tl.load(Std_ptr + c)

                # Compute input index: [b, c, h, src_w]
                in_idx = b * channels * height * width + c * height * width + h * width + src_w

                # Load and normalize
                x = tl.load(X_ptr + in_idx).to(tl.float32)
                y = (x - mean) / (std + 1e-8)

                # Compute output index: [b, c, h, w]
                out_idx = b * channels * height * width + c * height * width + h * width + w
                tl.store(Y_ptr + out_idx, y)

    @triton.jit
    def _fused_augment_kernel(
        X_ptr,
        Y_ptr,
        brightness_ptr,
        contrast_ptr,
        batch_size,
        channels,
        height,
        width,
        flip_flags_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused augmentation: flip + brightness + contrast.

        Formula: y = contrast * (x + brightness - 0.5) + 0.5
        Applies per-batch augmentation parameters.
        """
        # Get linear pixel index
        pid = tl.program_id(0)
        num_elements_per_batch = channels * height * width

        # Determine batch and local index
        b = pid // num_elements_per_batch
        local_idx = pid % num_elements_per_batch

        if b >= batch_size:
            return

        # Decode local index to c, h, w
        c = local_idx // (height * width)
        hw_idx = local_idx % (height * width)
        h = hw_idx // width
        w = hw_idx % width

        # Check flip flag for this batch
        flip = tl.load(flip_flags_ptr + b)

        # Compute source coordinates
        src_w = (width - 1 - w) if flip else w

        # Load augmentation parameters for this batch
        brightness = tl.load(brightness_ptr + b)
        contrast = tl.load(contrast_ptr + b)

        # Compute input index
        in_idx = b * num_elements_per_batch + c * height * width + h * width + src_w

        # Load input value
        x = tl.load(X_ptr + in_idx).to(tl.float32)

        # Apply augmentation: contrast * (x + brightness - 0.5) + 0.5
        y = contrast * (x + brightness - 0.5) + 0.5

        # Clamp to [0, 1]
        y = tl.maximum(tl.minimum(y, 1.0), 0.0)

        # Store result
        out_idx = pid
        tl.store(Y_ptr + out_idx, y)

    @triton.jit
    def _fused_normalize_augment_kernel(
        X_ptr,
        Y_ptr,
        Mean_ptr,
        Std_ptr,
        brightness_ptr,
        contrast_ptr,
        batch_size,
        channels,
        height,
        width,
        flip_flags_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fully fused: flip + normalize + brightness + contrast.

        Combines all augmentation operations in a single kernel pass.
        """
        pid = tl.program_id(0)
        num_elements_per_batch = channels * height * width
        total_elements = batch_size * num_elements_per_batch

        # Process BLOCK_SIZE elements per thread
        for i in range(BLOCK_SIZE):
            idx = pid * BLOCK_SIZE + i
            if idx >= total_elements:
                return

            # Decode index
            b = idx // num_elements_per_batch
            local_idx = idx % num_elements_per_batch
            c = local_idx // (height * width)
            hw_idx = local_idx % (height * width)
            h = hw_idx // width
            w = hw_idx % width

            # Get flip flag
            flip = tl.load(flip_flags_ptr + b)
            src_w = (width - 1 - w) if flip else w

            # Load parameters
            mean = tl.load(Mean_ptr + c)
            std = tl.load(Std_ptr + c)
            brightness = tl.load(brightness_ptr + b)
            contrast = tl.load(contrast_ptr + b)

            # Compute source index
            src_idx = b * num_elements_per_batch + c * height * width + h * width + src_w

            # Load and process
            x = tl.load(X_ptr + src_idx).to(tl.float32)

            # Normalize
            x_norm = (x - mean) / (std + 1e-8)

            # Apply brightness and contrast
            y = contrast * (x_norm + brightness)

            # Store
            tl.store(Y_ptr + idx, y)


def fused_augment_kernel(
    x: torch.Tensor,
    flip_flags: torch.Tensor,
    brightness: torch.Tensor,
    contrast: torch.Tensor,
) -> torch.Tensor:
    """
    Apply fused augmentation using Triton kernel.

    Args:
        x: Input tensor [B, C, H, W]
        flip_flags: Boolean tensor [B] for horizontal flip
        brightness: Per-batch brightness adjustment [B]
        contrast: Per-batch contrast adjustment [B]

    Returns:
        Augmented tensor [B, C, H, W]
    """
    if not TRITON_AVAILABLE or not x.is_cuda:
        return _torch_augment(x, flip_flags, brightness, contrast)

    B, C, H, W = x.shape
    y = torch.empty_like(x)

    # Total elements
    total_elements = B * C * H * W

    # Convert flip flags to int8 for kernel
    flip_int = flip_flags.to(torch.int8)

    # Launch kernel
    grid = (total_elements,)
    _fused_augment_kernel[grid](
        x, y,
        brightness, contrast,
        B, C, H, W,
        flip_int,
        BLOCK_SIZE=1,
    )

    return y


def _torch_augment(
    x: torch.Tensor,
    flip_flags: torch.Tensor,
    brightness: torch.Tensor,
    contrast: torch.Tensor,
) -> torch.Tensor:
    """PyTorch fallback for augmentation."""
    B, C, H, W = x.shape

    # Create output
    y = x.clone()

    # Apply flips
    for b in range(B):
        if flip_flags[b]:
            y[b] = torch.flip(y[b], dims=[-1])

    # Apply brightness and contrast
    brightness = brightness.view(B, 1, 1, 1)
    contrast = contrast.view(B, 1, 1, 1)

    y = contrast * (y + brightness - 0.5) + 0.5
    y = torch.clamp(y, 0, 1)

    return y


class FusedAugmentation(nn.Module):
    """
    GPU-accelerated fused augmentation module.

    Combines multiple augmentation operations into a single efficient kernel:
    - Horizontal flip (temporal consistent)
    - Brightness adjustment
    - Contrast adjustment
    - Optional normalization

    For video data, applies consistent augmentation across all frames.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (-0.1, 0.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        normalize: bool = False,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize fused augmentation.

        Args:
            flip_prob: Probability of horizontal flip
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            normalize: Whether to apply normalization
            mean: Channel-wise mean for normalization
            std: Channel-wise std for normalization
        """
        super().__init__()
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.normalize = normalize

        # Register normalization parameters as buffers
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        flip_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply fused augmentation.

        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W] for video
            deterministic: If True, skip random augmentation
            flip_override: Override flip flags for video temporal consistency

        Returns:
            Augmented tensor with same shape
        """
        is_video = x.dim() == 5

        if is_video:
            B, T, C, H, W = x.shape
            # Flatten batch and time for processing
            x = x.view(B * T, C, H, W)
            # Generate consistent augmentation params per batch
            flip_flags, brightness, contrast = self._sample_params(B, x.device, deterministic)
            # Repeat for each frame
            flip_flags = flip_flags.repeat_interleave(T)
            brightness = brightness.repeat_interleave(T)
            contrast = contrast.repeat_interleave(T)
        else:
            B = x.shape[0]
            flip_flags, brightness, contrast = self._sample_params(B, x.device, deterministic)

        # Override flip if provided
        if flip_override is not None:
            if is_video:
                flip_flags = flip_override.repeat_interleave(T)
            else:
                flip_flags = flip_override

        # Apply fused augmentation
        if TRITON_AVAILABLE and x.is_cuda:
            y = self._triton_augment(x, flip_flags, brightness, contrast)
        else:
            y = self._torch_augment(x, flip_flags, brightness, contrast)

        # Apply normalization if enabled
        if self.normalize:
            y = (y - self.mean) / self.std

        if is_video:
            y = y.view(B, T, C, H, W)

        return y

    def _sample_params(
        self,
        batch_size: int,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample augmentation parameters."""
        if deterministic:
            flip_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
            brightness = torch.zeros(batch_size, dtype=torch.float32, device=device)
            contrast = torch.ones(batch_size, dtype=torch.float32, device=device)
        else:
            flip_flags = torch.rand(batch_size, device=device) < self.flip_prob
            brightness = (
                torch.rand(batch_size, device=device) *
                (self.brightness_range[1] - self.brightness_range[0]) +
                self.brightness_range[0]
            )
            contrast = (
                torch.rand(batch_size, device=device) *
                (self.contrast_range[1] - self.contrast_range[0]) +
                self.contrast_range[0]
            )

        return flip_flags, brightness, contrast

    def _triton_augment(
        self,
        x: torch.Tensor,
        flip_flags: torch.Tensor,
        brightness: torch.Tensor,
        contrast: torch.Tensor,
    ) -> torch.Tensor:
        """Apply augmentation using Triton kernel."""
        return fused_augment_kernel(x, flip_flags, brightness, contrast)

    def _torch_augment(
        self,
        x: torch.Tensor,
        flip_flags: torch.Tensor,
        brightness: torch.Tensor,
        contrast: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch fallback augmentation."""
        y = x.clone()

        # Apply per-sample augmentation
        for i in range(x.shape[0]):
            if flip_flags[i]:
                y[i] = torch.flip(y[i], dims=[-1])

        # Vectorized brightness/contrast
        brightness = brightness.view(-1, 1, 1, 1)
        contrast = contrast.view(-1, 1, 1, 1)

        y = contrast * (y + brightness - 0.5) + 0.5
        y = torch.clamp(y, 0, 1)

        return y


class VideoAugmentation(nn.Module):
    """
    Temporally-consistent video augmentation.

    Ensures all frames in a sequence receive identical augmentation
    for training stability.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (-0.1, 0.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        color_jitter_prob: float = 0.3,
        color_range: Tuple[float, float] = (0.9, 1.1),
    ):
        super().__init__()
        self.fused_aug = FusedAugmentation(
            flip_prob=flip_prob,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
        )
        self.color_jitter_prob = color_jitter_prob
        self.color_range = color_range

    def forward(
        self,
        frames: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Apply temporally-consistent augmentation.

        Args:
            frames: Video frames [B, T, C, H, W]
            deterministic: Skip random augmentation

        Returns:
            Augmented frames [B, T, C, H, W]
        """
        B, T, C, H, W = frames.shape

        # Sample consistent flip per batch
        if deterministic:
            flip_override = torch.zeros(B, dtype=torch.bool, device=frames.device)
        else:
            flip_override = torch.rand(B, device=frames.device) < self.fused_aug.flip_prob

        # Apply fused augmentation
        frames = self.fused_aug(frames, deterministic=deterministic, flip_override=flip_override)

        # Optional color jitter (per-channel scaling, consistent across time)
        if not deterministic and self.color_jitter_prob > 0:
            if torch.rand(1).item() < self.color_jitter_prob:
                color_scale = (
                    torch.rand(B, C, 1, 1, device=frames.device) *
                    (self.color_range[1] - self.color_range[0]) +
                    self.color_range[0]
                )
                # Apply consistent scale across time
                frames = frames * color_scale.unsqueeze(1)
                frames = torch.clamp(frames, 0, 1)

        return frames


if __name__ == "__main__":
    print(f"Triton available: {TRITON_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test image augmentation
    print("\nTesting image augmentation...")
    x = torch.rand(8, 3, 256, 256, device=device)
    aug = FusedAugmentation().to(device)
    y = aug(x)
    print(f"Image input: {x.shape}, output: {y.shape}")

    # Test video augmentation
    print("\nTesting video augmentation...")
    video = torch.rand(4, 16, 3, 256, 256, device=device)
    video_aug = VideoAugmentation().to(device)
    video_out = video_aug(video)
    print(f"Video input: {video.shape}, output: {video_out.shape}")

    # Performance test
    if torch.cuda.is_available():
        print("\nPerformance test...")
        x = torch.rand(32, 3, 512, 512, device='cuda')

        # Warmup
        for _ in range(10):
            _ = aug(x)
        torch.cuda.synchronize()

        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(100):
            y = aug(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"Fused augmentation: {elapsed*1000/100:.2f}ms per batch")

    print("\nFused augmentation test completed!")
