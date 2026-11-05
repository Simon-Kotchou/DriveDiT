"""drivedit.data.shard: read-only memmaps and sha256 verification, the latents directory of section 3, and the
deterministic window index."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drivedit.data.shard import LatentStats, Shard, ShardError, WindowIndex

from .conftest import rewrite_array, write_latents


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
    valid = np.asarray(shard.array("slot_id")) >= 0
    valid[12] = False  # a slot the codec did not encode: an encoder failure, not a gap (section 3)
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


def test_segment_steps_need_an_image(shard_copy: Path) -> None:
    """Review round 1 (DD-20): a step without an image (slot_id -1, decision_index.obs_valid 0) never lies inside a
    segment, so no window frame lacks an observation."""
    slot = np.array(np.load(shard_copy / "slot_id.npy"))
    assert (slot[10:36] >= 0).all()
    slot[15] = -1
    rewrite_array(shard_copy, "slot_id", slot)
    with pytest.raises(ShardError, match="no image"):
        Shard(shard_copy)


def test_segments_follow_obs_valid(golden_shard_dir: Path) -> None:
    shard = Shard(golden_shard_dir)
    obs_valid = shard.array("decision_index")[:, shard.meta["decision_index_columns"].index("obs_valid")]
    for seg in shard.segments:
        assert (obs_valid[seg.start : seg.stop] == 1).all()
        assert (shard.array("slot_id")[seg.start : seg.stop] >= 0).all()


def test_symlinked_shard_files_are_refused(shard_copy: Path, tmp: Path) -> None:
    """Review (DD-20 round 2): every file a shard is read from is its own regular file, directly inside the shard.
    Integrity, not a sealed-data rule."""
    elsewhere = tmp / "elsewhere"
    elsewhere.mkdir()
    (shard_copy / "meta.json").rename(elsewhere / "meta.json")
    (shard_copy / "meta.json").symlink_to(elsewhere / "meta.json")
    with pytest.raises(ShardError, match="is a symlink"):
        Shard(shard_copy)
    (shard_copy / "meta.json").unlink()
    (elsewhere / "meta.json").rename(shard_copy / "meta.json")
    # a symlink that stays inside the shard: only the symlink rule refuses it (same bytes, so the sha256 passes)
    (shard_copy / "state.npy").rename(shard_copy / "state_bytes.npy")
    (shard_copy / "state.npy").symlink_to(shard_copy / "state_bytes.npy")
    with pytest.raises(ShardError, match="is a symlink"):
        Shard(shard_copy)
    with pytest.raises(ShardError, match="is a symlink"):
        Shard(shard_copy, verify=False).array("state")
    (shard_copy / "state.npy").unlink()
    (shard_copy / "state_bytes.npy").rename(shard_copy / "state.npy")
    assert Shard(shard_copy).array("state").shape == (68, 4)
    meta = json.loads((shard_copy / "meta.json").read_text("utf-8"))
    shutil.copy(shard_copy / "style.npy", shard_copy.parent / "style.npy")
    meta["arrays"]["style"]["file"] = "../style.npy"  # a regular file outside the shard
    (shard_copy / "meta.json").write_text(json.dumps(meta), "utf-8")
    with pytest.raises(ShardError, match="resolves outside"):
        Shard(shard_copy)


def test_symlinked_latents_are_refused(shard_copy: Path, tmp: Path) -> None:
    shard = Shard(shard_copy)
    write_latents(shard, "toy-2x3x4")
    root = shard.path.parent / "latents"
    codec = root / "toy-2x3x4"
    shard.latents("toy-2x3x4")
    (codec / "latents.npy").rename(codec / "latents_bytes.npy")
    (codec / "latents.npy").symlink_to(codec / "latents_bytes.npy")  # inside the codec directory
    with pytest.raises(ShardError, match="is a symlink"):
        shard.latents("toy-2x3x4")
    (codec / "latents.npy").unlink()
    (codec / "latents_bytes.npy").rename(codec / "latents.npy")
    codec.rename(tmp / "toy_store")
    codec.symlink_to(tmp / "toy_store")  # the codec directory itself
    with pytest.raises(ShardError, match="is a symlink"):
        shard.latents("toy-2x3x4")
    codec.unlink()
    root.rename(tmp / "latents_store")
    (tmp / "latents_store" / "toy-2x3x4").symlink_to(tmp / "toy_store")
    root.symlink_to(tmp / "latents_store")  # the latents directory
    with pytest.raises(ShardError, match="is a symlink"):
        shard.latents("toy-2x3x4")
