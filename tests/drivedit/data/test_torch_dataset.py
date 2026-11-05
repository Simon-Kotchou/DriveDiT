"""drivedit.data.torch.LatentWindowDataset over the golden shard and a synthetic latents directory (section 3):
normalisation, alignment with slot_id and the channels, the access checks (C24, offsets, targets), build-scoped
pooling, non-finite inputs and DataLoader collation.

Needs torch: listed in tests/conftest.py TORCH_TEST_MODULES (the rest of tests/drivedit/data is torch-free)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from drivedit.data.shard import LatentStats, Shard, ShardError, WindowIndex
from drivedit.data.spec import AccessError, FeatureSpec
from drivedit.data.torch import LatentWindowDataset

from .conftest import rewrite_array, write_latents

CODEC = "toy-2x3x4"
STATS = LatentStats(mean=(1.0, 2.0), std=(2.0, 4.0))


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


def test_a_row_the_codec_did_not_encode_is_refused(shard_copy: Path, spec: FeatureSpec) -> None:
    """valid.npy is exactly slot_id >= 0 (section 3), so no window frame can lack its latent."""
    shard = Shard(shard_copy)
    write_latents(shard, CODEC)
    root = shard.path.parent / "latents" / CODEC
    valid = np.load(root / "valid.npy")
    valid[12] = False
    np.save(root / "valid.npy", valid)
    with pytest.raises(ShardError, match="slot_id"):
        LatentWindowDataset(WindowIndex([shard], 4), CODEC, stats=STATS, spec=spec)


def test_action_exec_padding_is_nan_only(shard_copy: Path, spec: FeatureSpec) -> None:
    """Empty action_exec entries (NaN past P_t+1) pass; an entry of infinities or a partly empty one is refused
    (review round 1)."""
    exec_ = np.array(np.load(shard_copy / "action_exec.npy"))
    t = 20
    assert np.isfinite(exec_[t, 0]).all()
    for bad in (np.full(4, np.inf), np.array([np.nan, 0.5, 0.0, 0.1])):
        mutated = exec_.copy()
        mutated[t, 0] = bad
        rewrite_array(shard_copy, "action_exec", mutated)
        shard = Shard(shard_copy)
        if not (shard.path.parent / "latents" / CODEC).exists():
            write_latents(shard, CODEC)
        data = LatentWindowDataset(WindowIndex([shard], 4), CODEC, stats=STATS, channels=("action_exec",), spec=spec)
        hits = [i for i in range(len(data)) if data.index[i].start <= t < data.index[i].stop]
        with pytest.raises(ValueError, match="action_exec is not finite"):
            data[hits[0]]
    mutated = exec_.copy()
    mutated[t, 0] = np.nan  # a whole empty entry, as past P_t+1
    rewrite_array(shard_copy, "action_exec", mutated)
    data = LatentWindowDataset(
        WindowIndex([Shard(shard_copy)], 4), CODEC, stats=STATS, channels=("action_exec",), spec=spec
    )
    item = next(data[i] for i in range(len(data)) if data.index[i].start == t)
    assert torch.isnan(item["action_exec"][0, 0]).all()


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
    rewrite_array(shard_copy, "state", state)
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
