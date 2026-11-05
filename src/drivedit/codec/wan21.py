"""Wan2.1 VAE run one frame at a time: ``wan21-vae``, the generic video-VAE control of the bake-off.

The Wan2.1 causal 3D VAE (``diffusers.AutoencoderKLWan``) from ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers`` (subfolder ``vae``),
Apache-2.0. Natively it compresses 4 frames into one latent after a spatial-only first frame. Here every frame is fed
as its own one-frame clip, so it always takes the first-frame path: one latent per frame and no temporal compression,
which removes the 4x compression that delta-2026-11 §2 suspects hurt Wan2.2 in CADRE's driving table. The control
compares a generic VAE latent against the frozen-SSL CADRE latent at the same token count.

**Latent space.** The mode of the encoder posterior, ``[N, 16, H/8, W/8]``, standardised per channel with the VAE
config's ``latents_mean`` and ``latents_std``: ``(z - mean) / std``, as diffusers' ``WanPipeline`` does before its
transformer. At 288x512 that is ``[16, 36, 64]``: 144 tokens per frame at DiT patch 4 (delta-2026-11 §2). Decoding
inverts the standardisation and decodes the one-frame clip; the normalisation runs in float32.

**Licence.** Apache-2.0 (model card and ``LICENSE.txt`` of the Wan2.1 repository).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

# The defining module, not the lazy top-level `diffusers` namespace, whose type stubs resolve to placeholders.
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from drivedit.codec.base import CODECS, Codec, FrozenCodec, fetch_weights

if TYPE_CHECKING:
    from drivedit.codec.base import CodecInfo

__all__ = ["Wan21Codec", "load"]

SUBFOLDER: Final = "vae"
SPATIAL_SCALE: Final = 8


class Wan21Codec(FrozenCodec):
    """The Wan2.1 VAE on one-frame clips; ``vae`` is on ``device``, in ``dtype`` and in eval mode."""

    def __init__(
        self, info: CodecInfo, vae: AutoencoderKLWan, *, device: torch.device | str, dtype: torch.dtype
    ) -> None:
        super().__init__(info, device=device, dtype=dtype)
        config = vae.config
        if config.get("patch_size") is not None:
            raise ValueError(f"{info.codec_id}: a patchified Wan VAE (Wan2.2) is not the Wan2.1 control")
        self._vae = vae
        self._channels = int(config["z_dim"])
        self._scale = 2 ** (len(config["dim_mult"]) - 1)
        if self._scale != SPATIAL_SCALE:
            raise ValueError(f"{info.codec_id}: expected an {SPATIAL_SCALE}x spatial VAE, got {self._scale}x")
        self._mean = torch.tensor(config["latents_mean"], dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self._std = torch.tensor(config["latents_std"], dtype=torch.float32, device=self.device).view(1, -1, 1, 1)

    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]:
        if height <= 0 or width <= 0 or height % self._scale or width % self._scale:
            raise ValueError(f"{self.codec_id} needs a frame size divisible by {self._scale}, got {height}x{width}")
        return self._channels, height // self._scale, width // self._scale

    def image_size(self, height: int, width: int) -> tuple[int, int]:
        return height * self._scale, width * self._scale

    def _encode_unit(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self._vae.encode(x[:, :, None]).latent_dist.mode()[:, :, 0]  # clip [N, 3, T=1, H, W]
        return (z.to(torch.float32) - self._mean) / self._std

    def _decode_unit(self, z: torch.Tensor) -> torch.Tensor:
        z = (z.to(torch.float32) * self._std + self._mean).to(self.dtype)
        decoded: torch.Tensor = self._vae.decode(z[:, :, None]).sample[:, :, 0]
        return decoded


def load(codec_id: str, *, device: torch.device, dtype: torch.dtype) -> Codec:
    """Build ``wan21-vae`` from the pinned Wan2.1-T2V-1.3B-Diffusers snapshot."""
    info = CODECS[codec_id]
    if info.module != "wan21":
        raise KeyError(f"{codec_id} is not the Wan2.1 codec")
    snapshot = fetch_weights(info.weights)
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder=SUBFOLDER, torch_dtype=dtype)
    return Wan21Codec(info, vae.to(device).eval(), device=device, dtype=dtype)
