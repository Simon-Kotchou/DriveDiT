"""Shared helpers of the codec tests: a pooling test codec, synthetic frames, and a fake features_v1 shard writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from drivedit.codec.base import CodecInfo, FrozenCodec, WeightSource
from drivedit.codec.encode import IMAGE_PATTERN, IMAGE_SIZE, SHARD_SCHEMA
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

POOL_INFO = CodecInfo(
    "test-pool",
    "none",
    WeightSource(repo_id="none/none", revision="0" * 40, files=(), licence="n/a", code="tests/drivedit/codec"),
)


class PoolCodec(FrozenCodec):
    """A test double, not a codec: the latent is the 16x16 average of the [-1, 1] image, decoded by repetition."""

    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]:
        if height % 16 or width % 16:
            raise ValueError(f"PoolCodec needs multiples of 16, got {height}x{width}")
        return 3, height // 16, width // 16

    def image_size(self, height: int, width: int) -> tuple[int, int]:
        return height * 16, width * 16

    def _encode_unit(self, x: torch.Tensor) -> torch.Tensor:
        return functional.avg_pool2d(x, 16)

    def _decode_unit(self, z: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(z, scale_factor=16, mode="nearest")


def frames(n: int, *, seed: int = 0, size: tuple[int, int] = IMAGE_SIZE) -> torch.Tensor:
    """``[n, 3, H, W]`` uint8 test frames: smooth gradients plus a seeded block pattern, distinct per frame."""
    height, width = size
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    out = np.empty((n, 3, height, width), dtype=np.uint8)
    for i in range(n):
        blocks = rng.integers(0, 256, size=(3, height // 32 + 1, width // 32 + 1)).repeat(32, 1).repeat(32, 2)
        base = np.stack([yy, xx, (yy + xx) / 2]) * 255
        out[i] = (0.6 * base + 0.4 * blocks[:, :height, :width]).astype(np.uint8)
    return torch.from_numpy(out)


@dataclass(frozen=True)
class FakeShard:
    """A fake features_v1 shard: ``shard_dir`` = ``<root>/<session_id>/features_v1``, images in ``deriv_dir``."""

    shard_dir: Path
    deriv_dir: Path
    slot_id: np.ndarray


def write_shard(
    root: Path,
    slot_id: Sequence[int],
    *,
    session_id: str = "arland_T0L_20990101T000000Z_wb_i00_test",
    map_id: str = "arland",
    images: Callable[[int], np.ndarray] | None = None,
) -> FakeShard:
    """Write the parts of a features_v1 shard (DRIVEDIT_INTERFACES §2) that the pre-encoder reads.

    ``images(slot)`` gives each slot's ``[288, 512, 3]`` uint8 RGB image (default: ``frames`` with seed ``slot``).
    """
    session = root / session_id
    shard_dir = session / "features_v1"
    deriv_dir = session / "_harvest" / "deriv"
    shard_dir.mkdir(parents=True)
    deriv_dir.mkdir(parents=True)
    slots = np.asarray(slot_id, dtype=np.int64)
    for slot in sorted({int(s) for s in slots if s >= 0}):
        rgb = images(slot) if images is not None else frames(1, seed=slot)[0].permute(1, 2, 0).numpy()
        ok = cv2.imwrite(
            str(deriv_dir / IMAGE_PATTERN.format(slot=slot)), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        assert ok
    np.save(shard_dir / "slot_id.npy", slots)
    np.save(shard_dir / "step.npy", np.arange(slots.size, dtype=np.int64) * 6)
    meta = {
        "schema": SHARD_SCHEMA,
        "session_id": session_id,
        "features_version": "features_v1",
        "contract_version": "dc-2.0",
        "build_key": "test-build",
        "map_id": map_id,
        "split": "train",
        "decision_hz": 5,
        "n_steps": int(slots.size),
        "episodes": [{"episode_id": 0, "start": 0, "stop": int(slots.size), "mode": "E"}],
        "segments": [[0, int(slots.size)]],
        "image_source": {"deriv_dir": str(deriv_dir), "pattern": IMAGE_PATTERN},
        "sha256": {},
    }
    (shard_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return FakeShard(shard_dir=shard_dir, deriv_dir=deriv_dir, slot_id=slots)
