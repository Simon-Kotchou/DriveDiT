"""The FLUX.2 and Wan2.1 arms on small random VAEs (CPU, no download): conformance, latent space and determinism.

The latent-space tests compare each arm against diffusers' own pipeline code (FLUX.2's copied verbatim), so the
stored latents are exactly the space FLUX.2 / Wan2.1 diffusion models work in. The real architectures' shapes at
288x512 are checked on meta-device models built from the pinned configs (no weights, no memory).
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from drivedit.codec.base import CODECS, Codec, FrozenCodec, images_to_unit, unit_to_images
from drivedit.codec.flux2 import Flux2Codec, patchify, unpatchify
from drivedit.codec.wan21 import Wan21Codec
from tests.drivedit.codec.helpers import frames

# vae/config.json of black-forest-labs/FLUX.2-klein-4B@e7b7dc27 and Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a (the
# fields that set the latent shape; the rest are the diffusers defaults).
FLUX2_CONFIG = {
    "block_out_channels": (128, 256, 512, 512),
    "latent_channels": 32,
    "patch_size": (2, 2),
    "batch_norm_eps": 0.0001,
}
WAN21_CONFIG = {"base_dim": 96, "z_dim": 16, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2}


# Verbatim from diffusers 0.42.1, Flux2Pipeline._patchify_latents / _unpatchify_latents (pipelines/flux2), the
# reference for the FLUX.2 latent layout. The pipeline module itself is not imported: it needs transformers.
def _reference_patchify(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)


def _reference_unpatchify(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)


def test_real_architectures_give_the_documented_latent_shapes() -> None:
    with torch.device("meta"):
        flux2 = Flux2Codec(CODECS["flux2-vae"], AutoencoderKLFlux2(**FLUX2_CONFIG), device="meta", dtype=torch.float32)
        wan21 = Wan21Codec(CODECS["wan21-vae"], AutoencoderKLWan(**WAN21_CONFIG), device="meta", dtype=torch.float32)
    assert flux2.latent_shape(288, 512) == (128, 18, 32)  # 144 tokens at patch 2
    assert wan21.latent_shape(288, 512) == (16, 36, 64)  # 144 tokens at patch 4


@pytest.fixture(params=["flux2", "wan21"])
def arm(request: pytest.FixtureRequest) -> FrozenCodec:
    codec: FrozenCodec = request.getfixturevalue(f"tiny_{request.param}")
    return codec


def test_conformance(arm: FrozenCodec) -> None:
    assert isinstance(arm, Codec)
    assert arm.weights is CODECS[arm.codec_id].weights
    assert arm.device == torch.device("cpu")
    images = frames(3)
    z = arm.encode(images)
    assert z.shape == (3, *arm.latent_shape(288, 512))
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()
    out = arm.decode(z)
    assert out.shape == images.shape
    assert out.dtype == torch.uint8


def test_encode_is_deterministic_and_per_frame(arm: FrozenCodec) -> None:
    images = frames(3, seed=7)
    first = arm.encode(images)
    assert torch.equal(first, arm.encode(images))
    alone = arm.encode(images[1:2])
    torch.testing.assert_close(alone, first[1:2], rtol=1e-5, atol=1e-5)


def test_unsupported_sizes_raise(arm: FrozenCodec) -> None:
    with pytest.raises(ValueError, match="divisible"):
        arm.latent_shape(290, 512)


def test_flux2_patchify_matches_diffusers() -> None:
    z = torch.randn(2, 4, 6, 10)
    reference = _reference_patchify(z)
    assert torch.equal(patchify(z), reference)
    assert torch.equal(unpatchify(reference), _reference_unpatchify(reference))
    assert torch.equal(unpatchify(patchify(z)), z)


def test_flux2_latents_are_the_pipelines_latents(tiny_flux2: Flux2Codec) -> None:
    images = frames(2, seed=3)
    vae = tiny_flux2._vae
    with torch.no_grad():
        mode = vae.encode(images_to_unit(images, torch.float32)).latent_dist.mode()
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        expected = (_reference_patchify(mode) - bn_mean) / bn_std
        z = tiny_flux2.encode(images)
        torch.testing.assert_close(z, expected, rtol=1e-5, atol=1e-5)
        decoded = vae.decode(_reference_unpatchify(z * bn_std + bn_mean)).sample
    assert torch.equal(tiny_flux2.decode(z), unit_to_images(decoded))


def test_wan21_latents_are_the_pipelines_latents(tiny_wan21: Wan21Codec) -> None:
    images = frames(2, seed=4)
    vae = tiny_wan21._vae
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    with torch.no_grad():
        mode = vae.encode(images_to_unit(images, torch.float32)[:, :, None]).latent_dist.mode()
        assert mode.shape[2] == 1  # a one-frame clip gives one latent frame
        expected = (mode[:, :, 0] - mean) / std
        z = tiny_wan21.encode(images)
        torch.testing.assert_close(z, expected, rtol=1e-5, atol=1e-5)
        decoded = vae.decode((z * std + mean)[:, :, None]).sample[:, :, 0]
    assert torch.equal(tiny_wan21.decode(z), unit_to_images(decoded))
