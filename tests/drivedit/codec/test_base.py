"""drivedit.codec.base: the registry, the uint8 conventions and the input checks every arm inherits."""

from __future__ import annotations

import importlib
import re

import pytest
import torch

from drivedit.codec import base
from drivedit.codec.base import CODECS, Codec, codec_ids, get_codec, images_to_unit, unit_to_images
from tests.drivedit.codec.helpers import POOL_INFO, PoolCodec, frames

BAKE_OFF = ("cadre-1x16x16", "cadre-1x48x48", "flux2-vae", "wan21-vae")


def test_registry_holds_the_bake_off_arms() -> None:
    assert codec_ids() == BAKE_OFF
    for codec_id, info in CODECS.items():
        assert info.codec_id == codec_id
        assert re.fullmatch(r"[0-9a-f]{40}", info.weights.revision), "weights pinned to a full commit sha"
        assert info.weights.files
        assert info.weights.licence
        assert importlib.util.find_spec(f"drivedit.codec.{info.module}") is not None


@pytest.mark.parametrize("module", sorted({info.module for info in CODECS.values()}))
def test_arm_loaders_refuse_foreign_ids(module: str) -> None:
    arm = importlib.import_module(f"drivedit.codec.{module}")
    foreign = next(codec_id for codec_id, info in CODECS.items() if info.module != module)
    with pytest.raises(KeyError):
        arm.load(foreign, device=torch.device("cpu"), dtype=torch.float32)


def test_get_codec_rejects_unknown_ids() -> None:
    with pytest.raises(KeyError, match="unknown codec"):
        get_codec("vae3d", device="cpu", dtype=torch.float32)


def test_uint8_round_trip_is_exact() -> None:
    images = torch.arange(256, dtype=torch.uint8).view(1, 1, 16, 16).expand(1, 3, 16, 16)
    unit = images_to_unit(images, torch.float32)
    assert unit.min() == -1.0
    assert unit.max() == 1.0
    assert torch.equal(unit_to_images(unit), images)


def test_unit_to_images_rounds_and_clamps() -> None:
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0, 2 / 255 - 1]).view(1, 1, 1, 6)
    assert unit_to_images(x).flatten().tolist() == [0, 0, 128, 255, 255, 1]


def test_pool_codec_conforms(pool_codec: PoolCodec) -> None:
    assert isinstance(pool_codec, Codec)
    images = frames(2)
    z = pool_codec.encode(images)
    assert z.shape == (2, *pool_codec.latent_shape(288, 512))
    assert z.dtype == torch.float32
    out = pool_codec.decode(z)
    assert out.shape == images.shape
    assert out.dtype == torch.uint8


def test_encode_rejects_non_uint8_and_bad_shapes(pool_codec: PoolCodec) -> None:
    images = frames(1)
    with pytest.raises(TypeError, match="uint8"):
        pool_codec.encode(images.float())
    with pytest.raises(ValueError, match=r"\[N>0, 3, H, W\]"):
        pool_codec.encode(images[0])
    with pytest.raises(ValueError, match=r"\[N>0, 3, H, W\]"):
        pool_codec.encode(images[:, :2])
    with pytest.raises(ValueError, match="multiples of 16"):
        pool_codec.encode(images[:, :, :280])


def test_decode_rejects_bad_latents(pool_codec: PoolCodec) -> None:
    z = pool_codec.encode(frames(1))
    with pytest.raises(TypeError, match="floating point"):
        pool_codec.decode(z.to(torch.int32))
    with pytest.raises(ValueError, match="latents must be"):
        pool_codec.decode(z[:, :2])


def test_inputs_on_another_device_raise() -> None:
    codec = PoolCodec(POOL_INFO, device="meta", dtype=torch.float32)
    with pytest.raises(ValueError, match="move it first"):
        codec.encode(frames(1))
    with pytest.raises(ValueError, match="move it first"):
        codec.decode(torch.zeros(1, 3, 18, 32))


def test_compute_dtype_must_be_floating() -> None:
    with pytest.raises(ValueError, match="floating point"):
        PoolCodec(POOL_INFO, device="cpu", dtype=torch.int8)


def test_latents_are_cast_to_the_compute_dtype() -> None:
    codec = PoolCodec(POOL_INFO, device="cpu", dtype=torch.bfloat16)
    z = codec.encode(frames(1))
    assert z.dtype == torch.float32  # encode always returns float32
    assert codec.decode(z.to(torch.float16)).dtype == torch.uint8


def test_resolve_device_keeps_explicit_devices() -> None:
    assert base.resolve_device("cpu") == torch.device("cpu")
    assert base.resolve_device(torch.device("meta")) == torch.device("meta")
