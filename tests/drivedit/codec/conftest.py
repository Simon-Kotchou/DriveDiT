"""Fixtures for the codec tests: the pooling test codec and small random FLUX.2 / Wan2.1 VAEs.

Nothing here downloads: the VAEs are built from small configs with seeded random weights, at the real VAEs' depth (so
the same downsampling). Tests that need the real weights live in test_pretrained.py and skip without them.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from drivedit.codec.base import CODECS
from drivedit.codec.flux2 import Flux2Codec
from drivedit.codec.wan21 import Wan21Codec
from tests.drivedit.codec.helpers import POOL_INFO, PoolCodec


@pytest.fixture
def pool_codec() -> PoolCodec:
    return PoolCodec(POOL_INFO, device="cpu", dtype=torch.float32)


@pytest.fixture
def tiny_flux2() -> Flux2Codec:
    """A small random ``AutoencoderKLFlux2`` (4 blocks: 8x down, 16x after patching) with non-trivial BN stats."""
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 16, 16), latent_channels=4, norm_num_groups=4)
    generator = torch.Generator().manual_seed(1)
    channels = vae.bn.num_features
    with torch.no_grad():
        vae.bn.running_mean.copy_(torch.randn(channels, generator=generator) * 0.3)
        vae.bn.running_var.copy_(torch.rand(channels, generator=generator) * 2 + 0.5)
    return Flux2Codec(CODECS["flux2-vae"], vae.eval(), device="cpu", dtype=torch.float32)


@pytest.fixture
def tiny_wan21() -> Wan21Codec:
    """A small random ``AutoencoderKLWan`` (dim_mult of 4 stages: 8x down) with non-trivial latent statistics."""
    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=8,
        z_dim=4,
        num_res_blocks=1,
        latents_mean=[0.5, -0.25, 0.1, -1.0],
        latents_std=[2.0, 0.5, 1.5, 3.0],
    )
    return Wan21Codec(CODECS["wan21-vae"], vae.eval(), device="cpu", dtype=torch.float32)
