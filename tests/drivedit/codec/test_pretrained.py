"""Every registered codec with its real pinned weights on the GPU: conformance, determinism and a reconstruction floor.

Local only. Each case skips without CUDA, without the weights already in the Hugging Face cache, and for CADRE
without a verified DINOv3 checkout (``DRIVEDIT_DINOV3_DIR`` or the default directory). Nothing here downloads weights or
code. CI has CPU torch and no weights.

The PSNR floors only check that the weights loaded into the right places: a wrong mapping or normalisation decodes
noise (below 12 dB on these frames). They are not a quality measurement; tools/codec/measure_codecs.py measures that,
and reconstruction never selects a codec (MODELING_PLAN P2).
"""

from __future__ import annotations

import pytest
import torch

from drivedit.codec.base import CODECS, Codec, codec_ids, get_codec, weights_cached
from drivedit.codec.dinov3 import DinoV3SourceError, dinov3_dir, verify_checkout
from tests.drivedit.codec.helpers import frames

PSNR_FLOOR_DB = 18.0


def _skip_reason(codec_id: str) -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    if not weights_cached(codec_id):
        return f"{CODECS[codec_id].weights.repo_id} is not in the Hugging Face cache"
    if CODECS[codec_id].module == "cadre":
        try:
            verify_checkout(dinov3_dir())
        except DinoV3SourceError as error:
            return f"no verified DINOv3 checkout: {error}"
    return None


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.double() - b.double()).square().mean().item()
    return 10.0 * torch.log10(torch.tensor(255.0**2 / max(mse, 1e-12))).item()


@pytest.fixture(scope="module", params=codec_ids())
def pretrained(request: pytest.FixtureRequest) -> Codec:
    reason = _skip_reason(request.param)
    if reason is not None:
        pytest.skip(reason)
    return get_codec(request.param, device="cuda", dtype=torch.bfloat16)


def test_pretrained_conformance(pretrained: Codec) -> None:
    assert isinstance(pretrained, Codec)
    images = frames(2, seed=11).to(pretrained.device)
    z = pretrained.encode(images)
    assert z.shape == (2, *pretrained.latent_shape(288, 512))
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()
    out = pretrained.decode(z)
    assert out.shape == images.shape
    assert out.dtype == torch.uint8
    assert _psnr(out, images) > PSNR_FLOOR_DB


def test_pretrained_is_deterministic(pretrained: Codec) -> None:
    images = frames(2, seed=12).to(pretrained.device)
    z = pretrained.encode(images)
    assert torch.equal(z, pretrained.encode(images))
    assert torch.equal(pretrained.decode(z), pretrained.decode(z))
