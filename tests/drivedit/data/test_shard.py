"""drivedit.data.shard: read-only memmaps and sha256 verification, the latents directory of section 3, and the
deterministic window index."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drivedit.data.shard import LatentStats, Shard, ShardError, WindowIndex

from .conftest import write_latents


def test_arrays_are_read_only_memmaps(golden_shard_dir: Path) -> None:
    shard = Shard(golden_shard_dir)
    array = shard.array("action_cmd")
    assert isinstance(array, np.memmap)
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        array[0, 0] = 1.0
    assert shard.names == tuple(sorted(shard.meta["arrays"]))


def test_a_missing_tensor_names_its_reason(golden_shard_dir: Path) -> None:
    with pytest.raises(ShardError, match="road_class"):
        Shard(golden_shard_dir).array("road_context")


def test_sha256_verification(shard_copy: Path) -> None:
    path = shard_copy / "state.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ShardError, match="sha256"):
        Shard(shard_copy)
    assert Shard(shard_copy, verify=False).n_steps == 68


def test_ranges_are_checked(shard_copy: Path) -> None:
    meta = json.loads((shard_copy / "meta.json").read_text("utf-8"))
    meta["segments"][0]["stop"] = meta["episodes"][0]["stop"] + 1  # across the episode end
    (shard_copy / "meta.json").write_text(json.dumps(meta), "utf-8")
    with pytest.raises(ShardError, match="segment"):
        Shard(shard_copy)


def test_window_index_is_deterministic(golden_shard_dir: Path, shard_copy: Path) -> None:
    meta = json.loads((shard_copy / "meta.json").read_text("utf-8"))
    meta["session_id"] = "arland_T0b_20261025T091512Z_wb_i00_0000"  # another session id sorts first
    (shard_copy / "meta.json").write_text(json.dumps(meta), "utf-8")
    a, b = Shard(golden_shard_dir), Shard(shard_copy)
    first = [(w.shard.session_id, w.start) for w in WindowIndex([a, b], 8, stride=3)]
    second = [(w.shard.session_id, w.start) for w in WindowIndex([b, a], 8, stride=3)]
    assert first == second
    assert first[0][0] == meta["session_id"]
    assert [s for _, s in first[: len(first) // 2]] == [10, 13, 16, 19, 22, 25, 28, 39, 42, 45, 48, 51, 54, 57]
    index = WindowIndex([a, b], 8, stride=3)
    picks = [(w.shard.session_id, w.start) for w in index.sample(5, np.random.default_rng(7))]
    assert picks == [(w.shard.session_id, w.start) for w in index.sample(5, np.random.default_rng(7))]
    assert len(set(picks)) == 5
    with pytest.raises(ValueError, match="distinct"):
        index.sample(len(index) + 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match="twice"):
        WindowIndex([a, Shard(golden_shard_dir)], 8)
    window = index[0]
    assert window.take("action_cmd").shape == (8, 3)


def test_latents_open_and_align(shard_copy: Path) -> None:
    shard = Shard(shard_copy)
    latents = write_latents(shard, "toy-2x3x4")
    opened = shard.latents("toy-2x3x4")
    assert opened.latent_shape == (2, 3, 4)
    np.testing.assert_array_equal(opened.latents, latents)
    restored: Any = pickle.loads(pickle.dumps(opened))  # a spawned worker reopens the memmaps
    assert isinstance(restored.latents, np.memmap)
    np.testing.assert_array_equal(restored.valid, opened.valid)
    with pytest.raises(ShardError, match="codec id"):
        shard.latents("../escape")


def test_latents_refuse_misalignment(shard_copy: Path) -> None:
    shard = Shard(shard_copy)
    write_latents(shard, "toy-2x3x4")
    root = shard.path.parent / "latents" / "toy-2x3x4"
    valid = np.ones(shard.n_steps, dtype=bool)  # also rows without an image (slot_id -1)
    np.save(root / "valid.npy", valid)
    with pytest.raises(ShardError, match="slot_id"):
        shard.latents("toy-2x3x4")
    np.save(root / "valid.npy", np.asarray(shard.array("slot_id")) >= 0)
    np.save(root / "latents.npy", np.zeros((shard.n_steps, 2, 3, 4), dtype=np.float32))
    with pytest.raises(ShardError, match="float16"):
        shard.latents("toy-2x3x4")


def test_latent_stats() -> None:
    stats = LatentStats(mean=(1.0, -1.0), std=(2.0, 0.5))
    x = np.ones((4, 2, 3, 3), dtype=np.float16)
    y = stats.normalise(x)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y[:, 0], 0.0)
    np.testing.assert_allclose(y[:, 1], 4.0)
    with pytest.raises(ValueError, match="positive"):
        LatentStats(mean=(0.0,), std=(0.0,))
    with pytest.raises(ValueError, match="channels"):
        stats.normalise(np.ones((1, 3, 2, 2)))
