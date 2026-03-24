"""
Self-Forcing++ Training Implementation for DriveDiT.

Component 1: Rolling KV Cache

Based on Self-Forcing++ paper insights for extended sequence generation.
Instead of discrete anchor frames, uses a sliding window over the sequence
maintaining a fixed-size window of past KV pairs.

References:
- Self-Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
- Self-Forcing++: Extended sequence generation with rolling KV cache
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class RollingKVCache:
    """
    Rolling KV cache for efficient long-sequence generation.

    Unlike discrete anchor frames, this cache slides over the sequence,
    maintaining a fixed-size window of past KV pairs.

    Key features:
    - Automatic truncation when exceeding max length
    - Gradient detachment at configurable intervals
    - Memory-efficient storage with optional CPU offloading

    Args:
        max_length: Maximum cache length before truncation
        truncate_to: Length to truncate to when exceeding max
        detach_interval: Detach gradients every N steps
        num_layers: Number of transformer layers
        device: Device for cache storage
    """

    def __init__(
        self,
        max_length: int = 512,
        truncate_to: int = 256,
        detach_interval: int = 8,
        num_layers: int = 12,
        device: torch.device = None
    ):
        self.max_length = max_length
        self.truncate_to = truncate_to
        self.detach_interval = detach_interval
        self.num_layers = num_layers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cache storage: dict[layer_idx] -> {'k': tensor, 'v': tensor}
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # Tracking
        self.step_count = 0
        self.total_length = 0

    def reset(self):
        """Reset cache for new sequence."""
        self.cache = {}
        self.step_count = 0
        self.total_length = 0

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV pairs and return full KV for attention.

        Args:
            layer_idx: Index of transformer layer
            new_k: New key tensor [B, T_new, H, D]
            new_v: New value tensor [B, T_new, H, D]

        Returns:
            Tuple of (full_k, full_v) including cached history
        """
        self.step_count += 1

        # Detach gradients periodically to prevent accumulation
        should_detach = (self.step_count % self.detach_interval == 0)

        if layer_idx not in self.cache:
            # Initialize cache for this layer
            self.cache[layer_idx] = {
                'k': new_k.detach() if should_detach else new_k,
                'v': new_v.detach() if should_detach else new_v
            }
            self.total_length = new_k.size(1)
        else:
            # Concatenate with existing cache
            cached_k = self.cache[layer_idx]['k']
            cached_v = self.cache[layer_idx]['v']

            if should_detach:
                cached_k = cached_k.detach()
                cached_v = cached_v.detach()
                new_k = new_k.detach()
                new_v = new_v.detach()

            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)

            # Truncate if exceeding max length
            if full_k.size(1) > self.max_length:
                # Keep most recent entries
                truncate_from = full_k.size(1) - self.truncate_to
                full_k = full_k[:, truncate_from:].detach()
                full_v = full_v[:, truncate_from:].detach()

            self.cache[layer_idx] = {'k': full_k, 'v': full_v}
            self.total_length = full_k.size(1)

        return self.cache[layer_idx]['k'], self.cache[layer_idx]['v']

    def get_cache(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached KV for a specific layer."""
        return self.cache.get(layer_idx, None)

    def get_length(self) -> int:
        """Get current cache length."""
        return self.total_length

    def get_memory_usage(self) -> float:
        """Get memory usage in GB."""
        total_bytes = 0
        for layer_cache in self.cache.values():
            for tensor in layer_cache.values():
                total_bytes += tensor.element_size() * tensor.numel()
        return total_bytes / (1024 ** 3)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Rolling KV Cache Test")
    print("=" * 60)

    cache = RollingKVCache(max_length=32, truncate_to=16, detach_interval=4)

    for i in range(10):
        k = torch.randn(2, 4, 8, 64)  # [B, T, H, D]
        v = torch.randn(2, 4, 8, 64)

        full_k, full_v = cache.update(layer_idx=0, new_k=k, new_v=v)
        print(f"Step {i}: cache length = {cache.get_length()}, shape = {full_k.shape}")

    print(f"\nMemory usage: {cache.get_memory_usage():.6f} GB")
    print("Rolling KV Cache test completed!")
