"""The CADRE arm on CPU: the vendored code's isolation, and the CadreCodec wrapper on a tiny random CADRE.

The tiny CADRE is the vendored architecture at a small width and depth, with a stand-in for DINOv3 (a strided
convolution to DINOv3's 768 channels), so the wrapper's shapes, standardisation, seeded decoding and clip decoding run
in CI. The real codec, with the real DINOv3 and weights, is exercised in test_pretrained.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from drivedit.codec._vendor.cadre.model_autoencoder import CADRE, CADRE_MODELS, DINO_DIM, PATCH, CADREConfig
from drivedit.codec.base import CODECS, Codec
from drivedit.codec.cadre import MAX_CLIP_FRAMES, CadreCodec
from tests.drivedit.codec.helpers import frames

SRC = Path(__file__).resolve().parents[3] / "src"


def test_the_vendored_code_imports_nothing_by_a_global_name() -> None:
    """A foreign top-level `util` (with `util.hub`) is loaded first and must not be picked up."""
    code = (
        "import sys, types\n"
        "util = types.ModuleType('util'); util.__path__ = []; hub = types.ModuleType('util.hub')\n"
        "sys.modules.update({'util': util, 'util.hub': hub, 'util.attention': types.ModuleType('util.attention')})\n"
        "from drivedit.codec._vendor.cadre import model_autoencoder as m\n"
        "assert m.hub.__name__ == 'drivedit.codec._vendor.cadre.util.hub', m.hub\n"
        "assert m.SelfAttention.__module__ == 'drivedit.codec._vendor.cadre.util.attention'\n"
        "assert 'model_autoencoder' not in sys.modules\n"
        "assert not hasattr(m, 'load_dinov3') and not hasattr(m, 'DINOV3_HUB')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(SRC)},  # inherit TMPDIR, CUDA_VISIBLE_DEVICES, HF_HUB_OFFLINE, ...
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_registered_cells_have_the_documented_grids() -> None:
    assert CADRE_MODELS["cadre_1x16x16"].latent_grid == (18, 32)
    assert CADRE_MODELS["cadre_1x48x48"].latent_grid == (6, 11)


class StandInDino(nn.Module):
    """Not DINOv3: a strided convolution to its 768 channels with DINOv3's ``get_intermediate_layers`` signature."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, DINO_DIM, PATCH, stride=PATCH)

    def get_intermediate_layers(
        self, x: torch.Tensor, *, n: tuple[int, ...], norm: bool, reshape: bool
    ) -> list[torch.Tensor]:
        assert reshape
        features = self.proj(x)
        return [features * (1 + i / 10) for i in range(len(n))]


@pytest.fixture(params=["cadre-1x16x16", "cadre-1x48x48"])
def tiny_cadre(request: pytest.FixtureRequest) -> CadreCodec:
    cell = CADRE_MODELS[request.param.replace("-", "_")]
    config = CADREConfig(
        **{**cell.model_dump(), "dim": 32, "depth": 2, "tail_depth": 1, "num_heads": 2, "latent_dim": 8}
    )
    torch.manual_seed(0)
    model = CADRE(config, StandInDino()).eval()
    model.latent_mean, model.latent_std = 0.25, 2.0
    return CadreCodec(CODECS[request.param], model, device="cpu", dtype=torch.float32)


def test_conformance(tiny_cadre: CadreCodec) -> None:
    assert isinstance(tiny_cadre, Codec)
    grid = (18, 32) if tiny_cadre.codec_id == "cadre-1x16x16" else (6, 11)
    assert tiny_cadre.latent_shape(288, 512) == (8, *grid)
    images = frames(2)
    z = tiny_cadre.encode(images)
    assert z.shape == (2, 8, *grid)
    assert z.dtype == torch.float32
    out = tiny_cadre.decode(z)
    assert out.shape == images.shape
    assert out.dtype == torch.uint8
    with pytest.raises(ValueError, match="288x512 frames only"):
        tiny_cadre.latent_shape(144, 256)


def test_encode_is_the_standardised_cadre_latent(tiny_cadre: CadreCodec) -> None:
    images = frames(3, seed=5)
    model = tiny_cadre._model
    with torch.no_grad():
        raw = model.encode(model.preprocess(images[:, None]))[:, 0]
    torch.testing.assert_close(tiny_cadre.encode(images), (raw - 0.25) / 2.0)
    torch.testing.assert_close(tiny_cadre.encode(images[1:2]), tiny_cadre.encode(images)[1:2])  # per frame


def test_decoding_is_seeded(tiny_cadre: CadreCodec) -> None:
    z = tiny_cadre.encode(frames(2, seed=6))
    first = tiny_cadre.decode(z)
    assert torch.equal(first, tiny_cadre.decode(z))
    tiny_cadre.decode_seed = 1
    assert not torch.equal(first, tiny_cadre.decode(z))


def test_decode_clip(tiny_cadre: CadreCodec) -> None:
    z = tiny_cadre.encode(frames(3, seed=7))
    clip = tiny_cadre.decode_clip(z)
    assert clip.shape == (3, 3, 288, 512)
    assert clip.dtype == torch.uint8
    with pytest.raises(ValueError, match=f"1-{MAX_CLIP_FRAMES} frames"):
        tiny_cadre.decode_clip(z.repeat(MAX_CLIP_FRAMES, 1, 1, 1)[: MAX_CLIP_FRAMES + 1])
