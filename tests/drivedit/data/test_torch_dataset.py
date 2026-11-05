"""drivedit.data.torch.LatentWindowDataset over the golden shard and a synthetic latents directory (section 3):
normalisation, alignment with slot_id and the channels, the access checks (C24, offsets, targets), build-scoped
pooling, non-finite inputs and DataLoader collation.

Needs torch: listed in tests/conftest.py TORCH_TEST_MODULES (the rest of tests/drivedit/data is torch-free)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from drivedit.data.shard import LatentStats, Shard, WindowIndex
from drivedit.data.spec import AccessError, FeatureSpec
from drivedit.data.torch import LatentWindowDataset

from .conftest import write_latents

CODEC = "toy-2x3x4"
STATS = LatentStats(mean=(1.0, 2.0), std=(2.0, 4.0))


def _rewrite(shard_dir: Path, name: str, array: np.ndarray) -> None:
    """Replace one array of a shard copy and keep its meta.json sha256 true."""
    path = shard_dir / f"{name}.npy"
    np.save(path, array)
    meta = json.loads((shard_dir / "meta.json").read_text("utf-8"))
    meta["arrays"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (shard_dir / "meta.json").write_text(json.dumps(meta), "utf-8")


def test_items_are_aligned_and_normalised(shard_copy: Path, spec: FeatureSpec) -> None:
    shard = Shard(shard_copy)
    raw = write_latents(shard, CODEC)
    data = LatentWindowDataset(WindowIndex([shard], 6), CODEC, stats=STATS, spec=spec)
    assert len(data) == len(WindowIndex([shard], 6))
    window = data.index[3]
    item = data[3]
    assert set(item) == {"latents", "valid", "action_cmd", "state", "future_pose", "contact_events", "masks"}
    assert item["latents"].shape == (6, 2, 3, 4)
    assert item["latents"].dtype == torch.float32
    rows = raw[window.start : window.stop].astype(np.float32)
    want = (rows - np.array([1.0, 2.0])[:, None, None]) / np.array([2.0, 4.0])[:, None, None]
    np.testing.assert_allclose(item["latents"].numpy(), want, rtol=1e-6)
    assert bool(item["valid"].all())
    for name in ("action_cmd", "state", "contact_events"):
        np.testing.assert_array_equal(item[name].numpy(), shard.array(name)[window.start : window.stop])
    np.testing.assert_array_equal(
        item["future_pose"].numpy(), shard.array("future_pose")[window.start : window.stop]
    )  # NaN kept: no target
    assert item["masks"].dtype == torch.uint8
    assert item["future_pose"].shape == (6, 30, 3)


def test_a_row_the_codec_did_not_encode(shard_copy: Path, spec: FeatureSpec) -> None:
    shard = Shard(shard_copy)
    write_latents(shard, CODEC)
    root = shard.path.parent / "latents" / CODEC
    valid = np.load(root / "valid.npy")
    valid[12] = False
    np.save(root / "valid.npy", valid)
    data = LatentWindowDataset(WindowIndex([shard], 4), CODEC, stats=STATS, spec=spec)
    item = next(data[i] for i in range(len(data)) if data.index[i].start == 10)
    assert item["valid"].tolist() == [True, True, False, True]
    assert float(item["latents"][2].abs().sum()) == 0.0


@pytest.mark.parametrize(
    ("channels", "aux", "role"),
    [
        (("hindsight_anchors",), (), "world_model"),  # C24: future-derived
        (("eval_only",), (), "world_model"),  # C24: absolute pose, map and session identity
        (("future_pose",), (), "world_model"),  # an aux target is never an input
        (("action_label",), (), "world_model"),
        (("masks",), (), "world_model"),
        (("action_cmd",), (), "policy"),  # history only for the policy: never aligned at t
        (("state",), ("state",), "world_model"),  # state is not a world-model target
        ((), ("eval_only",), "world_model"),
    ],
)
def test_access_is_enforced(
    shard_copy: Path, spec: FeatureSpec, channels: tuple[str, ...], aux: tuple[str, ...], role: str
) -> None:
    shard = Shard(shard_copy)
    write_latents(shard, CODEC)
    with pytest.raises(AccessError):
        LatentWindowDataset(
            WindowIndex([shard], 4), CODEC, stats=STATS, channels=channels, aux=aux, role=role, spec=spec
        )


def test_build_scoped_channels_are_not_pooled_across_builds(
    shard_copy: Path, golden_shard_dir: Path, spec: FeatureSpec, tmp: Path
) -> None:
    other = tmp / "other" / "arland_T0b_20261025T091512Z_wb_i00_0000" / "features_v1"
    shutil.copytree(golden_shard_dir, other)
    meta = json.loads((other / "meta.json").read_text("utf-8"))
    meta["session_id"] = other.parent.name
    meta["build_key"] = "0" * 64
    (other / "meta.json").write_text(json.dumps(meta), "utf-8")
    a, b = Shard(shard_copy), Shard(other)
    write_latents(a, CODEC)
    write_latents(b, CODEC)
    index = WindowIndex([a, b], 4)
    pooled = LatentWindowDataset(index, CODEC, stats=STATS, spec=spec)
    assert len(pooled) == len(index)
    with pytest.raises(ValueError, match="build_key"):
        LatentWindowDataset(index, CODEC, stats=STATS, channels=("action_cmd", "action_exec"), spec=spec)
    single = LatentWindowDataset(WindowIndex([a], 4), CODEC, stats=STATS, channels=("action_exec",), spec=spec)
    assert single[0]["action_exec"].shape == (4, 6, 4)


def test_non_finite_inputs_raise(shard_copy: Path, spec: FeatureSpec) -> None:
    state = np.array(np.load(shard_copy / "state.npy"))
    state[20, 3] = np.nan
    _rewrite(shard_copy, "state", state)
    shard = Shard(shard_copy)
    write_latents(shard, CODEC)
    data = LatentWindowDataset(WindowIndex([shard], 4), CODEC, stats=STATS, spec=spec)
    bad = [i for i in range(len(data)) if data.index[i].start <= 20 < data.index[i].stop]
    with pytest.raises(ValueError, match="state is not finite"):
        data[bad[0]]
    good = next(i for i in range(len(data)) if i not in bad)
    assert torch.isfinite(data[good]["state"]).all()


def test_dataloader_collates(shard_copy: Path, spec: FeatureSpec) -> None:
    shard = Shard(shard_copy)
    write_latents(shard, CODEC)
    data = LatentWindowDataset(WindowIndex([shard], 8, stride=4), CODEC, stats=STATS, spec=spec)
    batch = next(iter(DataLoader(data, batch_size=3, shuffle=False)))
    assert batch["latents"].shape == (3, 8, 2, 3, 4)
    assert batch["action_cmd"].shape == (3, 8, 3)
    assert batch["masks"].shape == (3, 8, 5)
