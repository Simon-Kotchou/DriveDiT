"""FLUX.2 VAE per frame: ``flux2-vae``, the licence-clean arm of the bake-off (delta-2026-11 §2).

The FLUX.2 image autoencoder (``diffusers.AutoencoderKLFlux2``), as distributed in
``black-forest-labs/FLUX.2-klein-4B`` (subfolder ``vae``) under Apache-2.0. It is an image model, so each frame is
encoded on its own: there is no frame-rate mismatch at 5 Hz.

**Latent space.** FLUX.2's own diffusion latent, built exactly as diffusers' ``Flux2KleinPipeline`` builds it
(``_encode_vae_image`` and the decode tail of ``__call__``):

1. the mode of the encoder posterior, ``[N, 32, H/8, W/8]``;
2. 2x2 space-to-channel patching (``_patchify_latents``), ``[N, 128, H/16, W/16]``, channel ``4c + 2dy + dx``;
3. standardisation by the VAE's frozen batch-norm statistics, ``(z - running_mean) / sqrt(running_var + eps)``.

At 288x512 that is ``[128, 18, 32]``: 144 tokens per frame at DiT patch 2 (delta-2026-11 §2). Decoding inverts the
three steps and runs the full decoder. The normalisation runs in float32 whatever the compute dtype.

**Licence.** Apache-2.0 (the FLUX.2-klein-4B model card and ``LICENSE.md``). The same VAE config names FLUX.2-dev as its
origin, but the FLUX.2-dev repository carries the FLUX non-commercial licence; this arm reads only the klein-4B copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

# The defining module, not the lazy top-level `diffusers` namespace, whose type stubs resolve to placeholders.
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from drivedit.codec.base import CODECS, Codec, FrozenCodec, fetch_weights

if TYPE_CHECKING:
    from drivedit.codec.base import CodecInfo

__all__ = ["Flux2Codec", "load", "patchify", "unpatchify"]

SUBFOLDER: Final = "vae"
PATCH: Final = 2


def patchify(z: torch.Tensor) -> torch.Tensor:
    """``[N, C, h, w]`` -> ``[N, 4C, h/2, w/2]``, channel ``4c + 2dy + dx`` (diffusers ``_patchify_latents``)."""
    n, c, h, w = z.shape
    z = z.view(n, c, h // PATCH, PATCH, w // PATCH, PATCH).permute(0, 1, 3, 5, 2, 4)
    return z.reshape(n, c * PATCH * PATCH, h // PATCH, w // PATCH)


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    """The inverse of ``patchify``: ``[N, 4C, h, w]`` -> ``[N, C, 2h, 2w]``."""
    n, c, h, w = z.shape
    z = z.reshape(n, c // (PATCH * PATCH), PATCH, PATCH, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(n, c // (PATCH * PATCH), h * PATCH, w * PATCH)


class Flux2Codec(FrozenCodec):
    """The FLUX.2 VAE on one frame at a time; ``vae`` is on ``device``, in ``dtype`` and in eval mode."""

    def __init__(
        self, info: CodecInfo, vae: AutoencoderKLFlux2, *, device: torch.device | str, dtype: torch.dtype
    ) -> None:
        super().__init__(info, device=device, dtype=dtype)
        config = vae.config
        if tuple(config["patch_size"]) != (PATCH, PATCH):
            raise ValueError(f"{info.codec_id}: expected a {PATCH}x{PATCH} latent patch, got {config['patch_size']}")
        self._vae = vae
        self._channels = int(config["latent_channels"]) * PATCH * PATCH
        # The VAE downsamples by 2 per encoder block after the first; the patching adds one more factor of 2.
        self._scale = 2 ** (len(config["block_out_channels"]) - 1) * PATCH
        mean, variance = vae.bn.running_mean, vae.bn.running_var
        if mean is None or variance is None:
            raise ValueError(f"{info.codec_id}: the VAE's batch norm carries no running statistics")
        self._bn_mean = mean.detach().to(device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        variance = variance.detach().to(device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        self._bn_std = torch.sqrt(variance + float(config["batch_norm_eps"]))

    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]:
        if height <= 0 or width <= 0 or height % self._scale or width % self._scale:
            raise ValueError(f"{self.codec_id} needs a frame size divisible by {self._scale}, got {height}x{width}")
        return self._channels, height // self._scale, width // self._scale

    def image_size(self, height: int, width: int) -> tuple[int, int]:
        return height * self._scale, width * self._scale

    def _encode_unit(self, x: torch.Tensor) -> torch.Tensor:
        z = self._vae.encode(x).latent_dist.mode()
        return (patchify(z).to(torch.float32) - self._bn_mean) / self._bn_std

    def _decode_unit(self, z: torch.Tensor) -> torch.Tensor:
        z = unpatchify(z.to(torch.float32) * self._bn_std + self._bn_mean).to(self.dtype)
        decoded: torch.Tensor = self._vae.decode(z).sample
        return decoded


def load(codec_id: str, *, device: torch.device, dtype: torch.dtype) -> Codec:
    """Build ``flux2-vae`` from the pinned FLUX.2-klein-4B snapshot."""
    info = CODECS[codec_id]
    if info.module != "flux2":
        raise KeyError(f"{codec_id} is not the FLUX.2 codec")
    snapshot = fetch_weights(info.weights)
    vae = AutoencoderKLFlux2.from_pretrained(snapshot, subfolder=SUBFOLDER, torch_dtype=dtype)
    return Flux2Codec(info, vae.to(device).eval(), device=device, dtype=dtype)
