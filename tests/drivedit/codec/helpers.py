"""Shared helpers of the codec tests: a pooling test codec, synthetic frames, and a fake features_v1 shard writer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch.nn import functional

from drivedit.codec.base import CodecInfo, FrozenCodec, WeightSource
from drivedit.codec.encode import IMAGE_PATTERN, IMAGE_SIZE, SHARD_SCHEMA

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


# The top-level keys of meta.json as drivedit.data (DD-11) writes them: origin/feat/features-v1-data at e6ca886,
# src/drivedit/data/build.py (`_meta`, then `session_features` and `_write_files`). The real Arland T0L shard of
# 2026-11-05 has exactly these keys.
DD11_META_KEYS = (
    "arrays", "build_key", "builder_version", "categories", "channels", "contract_version", "counts", "decision_hz",
    "decision_index_columns", "episodes", "excluded", "features_version", "future_pose_horizon",
    "hindsight_anchor_steps", "image", "map_id", "missing", "n_steps", "physics_steps_per_decision", "pin_id",
    "reader_version", "schema", "segments", "session_id", "source", "split", "step_dt_s", "trial_id",
)  # fmt: skip
# DD-11's image block (`_image_source`): the harvest derivative directory, the file pattern, and the derivative
# parameters from _harvest/result.json, as in the real shard.
DERIVATIVE = {
    "box": [390, 0, 1058, 595],
    "codec": "jpeg_q95",
    "interpolation": "area",
    "library": "opencv 4.10.0",
    "quality": 95,
    "wh": [512, 288],
}


@dataclass(frozen=True)
class FakeShard:
    """A fake features_v1 shard: ``shard_dir`` = ``<root>/<session_id>/features_v1``, images in ``deriv_dir``."""

    shard_dir: Path
    deriv_dir: Path
    slot_id: np.ndarray


def _array_entry(path: Path, array: np.ndarray) -> dict[str, object]:
    """One ``arrays`` entry as DD-11's ``_write_files`` records it."""
    np.save(path, array)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"dtype": str(array.dtype), "file": path.name, "sha256": digest, "shape": list(array.shape)}


def write_shard(
    root: Path,
    slot_id: Sequence[int],
    *,
    session_id: str = "arland_T0L_20990101T000000Z_wb_i00_test",
    map_id: str = "arland",
    images: Callable[[int], np.ndarray] | None = None,
    with_images: bool = True,
) -> FakeShard:
    """A features_v1 shard laid out as DD-11 writes it, holding the arrays the pre-encoder reads (slot_id, step).

    ``images(slot)`` gives each slot's ``[288, 512, 3]`` uint8 RGB image (default: ``frames`` with seed ``slot``).
    ``with_images=False`` writes a shard without derivatives: ``image`` null and the reason in ``missing.image``.
    """
    session = root / session_id
    shard_dir = session / "features_v1"
    deriv_dir = session / "_harvest" / "deriv"
    shard_dir.mkdir(parents=True)
    slots = np.asarray(slot_id, dtype=np.int64)
    n = int(slots.size)
    missing: dict[str, str] = {"road_context": "not built by the reader"}
    image: dict[str, object] | None = None
    if with_images:
        deriv_dir.mkdir(parents=True)
        for slot in sorted({int(s) for s in slots if s >= 0}):
            rgb = images(slot) if images is not None else frames(1, seed=slot)[0].permute(1, 2, 0).numpy()
            path = deriv_dir / IMAGE_PATTERN.format(slot=slot)
            assert cv2.imwrite(str(path), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        image = {"deriv_dir": str(deriv_dir.resolve()), "pattern": IMAGE_PATTERN, "derivative": DERIVATIVE}
    else:
        missing["image"] = (
            "no harvest derivatives (_harvest/deriv) in the session: slot_id is written, the images are not"
        )
    arrays = {
        "slot_id": _array_entry(shard_dir / "slot_id.npy", slots),
        "step": _array_entry(shard_dir / "step.npy", np.arange(n, dtype=np.int64) * 6),
    }
    meta = {
        "schema": SHARD_SCHEMA,
        "builder_version": "drivedit.data.build/1",
        "reader_version": "dc2-reader/1",
        "session_id": session_id,
        "features_version": "features_v1",
        "contract_version": "dc-2.0",
        "build_key": "0" * 64,
        "pin_id": "1.8.0.13",
        "map_id": map_id,
        "trial_id": "T0L",
        "split": "train",
        "decision_hz": 5.0,
        "physics_steps_per_decision": 6,
        "step_dt_s": 1 / 30,
        "n_steps": n,
        "future_pose_horizon": 30,
        "hindsight_anchor_steps": [10, 20, 30],
        "episodes": [
            {"episode_id": 0, "mode": "E", "route_id": "test_route", "split": "train", "start": 0, "stop": n,
             "termination_reason": "route_complete"}
        ],
        "segments": [{"episode_id": 0, "start": 0, "stop": n}],
        "decision_index_columns": [],
        "categories": {"contact_class": []},
        "counts": {"decision_grid_points_skipped": 0, "segment_steps": n},
        "source": {"session_dir": str(session.resolve())},
        "image": image,
        "missing": missing,
        "excluded": {"eval_only": "not written"},
        "channels": {},
        "arrays": arrays,
    }  # fmt: skip
    assert tuple(sorted(meta)) == DD11_META_KEYS
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return FakeShard(shard_dir=shard_dir, deriv_dir=deriv_dir, slot_id=slots)
