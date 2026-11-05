"""drivedit.codec.encode: features_v1 shards in DD-11's layout through the pooling test codec (DRIVEDIT_INTERFACES §3).

``DRIVEDIT_TEST_SHARD`` may name a real DD-11 shard (``<root>/<session_id>/features_v1``); the test that reads it
only reads, and skips without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from drivedit.codec import encode as encode_module
from drivedit.codec.encode import (
    IMAGE_PATTERN,
    LATENTS_SCHEMA,
    LatentsLockedError,
    encode_session,
    main,
    read_shard,
)
from tests.drivedit.codec.helpers import DD11_META_KEYS, POOL_INFO, PoolCodec, write_shard

SLOTS = [-1, 0, 1, 2, -1, 3, 4, 5, 6, -1]
REAL_SHARD = os.environ.get("DRIVEDIT_TEST_SHARD")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(out: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return meta


def _assert_consistent(out: Path) -> None:
    """The published arrays are the ones meta.json describes."""
    meta = _meta(out)
    assert meta["sha256"] == {name: _sha256(out / name) for name in ("latents.npy", "valid.npy")}
    latents = np.load(out / "latents.npy").astype(np.float64)
    rows = latents[np.load(out / "valid.npy")]
    np.testing.assert_allclose(meta["mean"], rows.mean(axis=(0, 2, 3)), rtol=1e-9, atol=1e-12)
    assert meta["max_abs"] == np.abs(rows).max()


def test_writes_the_section_3_layout(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    out = encode_session(shard.shard_dir, pool_codec, batch=3)
    assert out == shard.shard_dir.parent / "latents" / "test-pool"
    assert sorted(p.name for p in out.iterdir()) == ["latents.npy", "meta.json", "valid.npy"]
    assert [p.name for p in out.parent.iterdir()] == ["test-pool"], "no lock or staging directory is left"

    latents = np.load(out / "latents.npy")
    valid = np.load(out / "valid.npy")
    assert latents.shape == (len(SLOTS), 3, 18, 32)
    assert latents.dtype == np.float16
    assert valid.dtype == np.bool_
    assert valid.tolist() == [s >= 0 for s in SLOTS]
    assert not latents[~valid].any(), "rows without an image are zeros"
    assert np.abs(latents[valid]).sum(axis=(1, 2, 3)).min() > 0

    meta = _meta(out)
    shard_meta = json.loads((shard.shard_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == LATENTS_SCHEMA
    assert meta["codec_id"] == "test-pool"
    assert meta["weights"]["revision"] == "0" * 40
    assert meta["input_size"] == [288, 512]
    assert meta["latent_shape"] == [3, 18, 32]
    assert meta["dtype"] == "float16"
    assert meta["compute_dtype"] == "float32"
    assert meta["n_steps"] == len(SLOTS)
    assert meta["n_valid"] == int(valid.sum())
    assert meta["session_id"] == shard_meta["session_id"] == shard.shard_dir.parent.name
    assert meta["build_key"] == shard_meta["build_key"]
    assert meta["image"] == shard_meta["image"]
    assert meta["aligned_to"] == {"file": "slot_id.npy", "sha256": shard_meta["arrays"]["slot_id"]["sha256"]}
    assert meta["encoded_at"].endswith("+00:00")
    _assert_consistent(out)


def test_meta_stats_are_the_stored_latents_stats(tmp_path: Path, pool_codec: PoolCodec) -> None:
    out = encode_session(write_shard(tmp_path, SLOTS).shard_dir, pool_codec, batch=4)
    latents = np.load(out / "latents.npy").astype(np.float64)
    rows = latents[np.load(out / "valid.npy")]
    meta = _meta(out)
    np.testing.assert_allclose(meta["mean"], rows.mean(axis=(0, 2, 3)), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(meta["std"], rows.std(axis=(0, 2, 3)), rtol=1e-6, atol=1e-9)
    assert meta["max_abs"] == np.abs(rows).max()
    assert len(meta["mean"]) == len(meta["std"]) == 3


def test_rows_are_the_codec_latents_of_their_slot(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    latents = np.load(encode_session(shard.shard_dir, pool_codec, batch=2) / "latents.npy")
    for t, slot in enumerate(SLOTS):
        if slot < 0:
            continue
        bgr = cv2.imread(str(shard.deriv_dir / IMAGE_PATTERN.format(slot=slot)))
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)))[None]
        expected = pool_codec.encode(rgb).to(torch.float16).numpy()[0]
        assert np.array_equal(latents[t], expected), f"step {t} (slot {slot})"


def test_images_are_read_as_rgb(tmp_path: Path, pool_codec: PoolCodec) -> None:
    red = np.zeros((288, 512, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    shard = write_shard(tmp_path, [0], images=lambda _slot: red)
    latents = np.load(encode_session(shard.shard_dir, pool_codec, batch=1) / "latents.npy")
    means = latents[0].astype(np.float64).mean(axis=(1, 2))
    assert means[0] > 0.95  # red channel near +1
    assert means[1] < -0.95
    assert means[2] < -0.95


def test_encode_is_deterministic_and_batch_invariant(tmp_path: Path, pool_codec: PoolCodec) -> None:
    hashes = []
    for i, batch in enumerate((1, 3, 16)):
        shard = write_shard(tmp_path / str(i), SLOTS)
        hashes.append(_sha256(encode_session(shard.shard_dir, pool_codec, batch=batch) / "latents.npy"))
    assert len(set(hashes)) == 1


def test_the_image_block_is_read_as_dd11_writes_it(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    images = read_shard(shard.shard_dir)
    assert images.deriv_dir == shard.deriv_dir.resolve()
    assert images.meta["image"]["pattern"] == IMAGE_PATTERN
    assert set(images.meta["image"]) == {"deriv_dir", "pattern", "derivative"}


def test_a_shard_without_images_raises_with_the_reason(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS, with_images=False)
    with pytest.raises(ValueError, match=r"no images \(no harvest derivatives"):
        encode_session(shard.shard_dir, pool_codec, batch=2)
    assert not (shard.shard_dir.parent / "latents").exists()


def test_a_changed_slot_id_is_refused(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    np.save(shard.shard_dir / "slot_id.npy", np.asarray([*SLOTS[:-1], 7], dtype=np.int64))
    with pytest.raises(ValueError, match=r"sha256 differs from meta\.json arrays\.slot_id"):
        encode_session(shard.shard_dir, pool_codec, batch=2)


def test_slot_ids_must_match_n_steps(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    meta_path = shard.shard_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "n_steps": len(SLOTS) + 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="n_steps"):
        encode_session(shard.shard_dir, pool_codec, batch=2)


@pytest.mark.parametrize(
    ("slots", "message"),
    [([-1, -1], "no step has an image"), ([0, -2], "below -1")],
)
def test_bad_slot_ids_raise(tmp_path: Path, pool_codec: PoolCodec, slots: list[int], message: str) -> None:
    shard = write_shard(tmp_path, slots)
    with pytest.raises(ValueError, match=message):
        encode_session(shard.shard_dir, pool_codec, batch=2)


def test_not_a_shard_raises(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    meta_path = shard.shard_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "schema": "something/else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        encode_session(shard.shard_dir, pool_codec, batch=2)
    meta_path.unlink()
    with pytest.raises(FileNotFoundError, match="not a features_v1 shard"):
        encode_session(shard.shard_dir, pool_codec, batch=2)


def test_a_missing_image_raises_and_leaves_nothing(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    (shard.deriv_dir / IMAGE_PATTERN.format(slot=4)).unlink()
    with pytest.raises(FileNotFoundError, match=r"f_00000004\.jpg"):
        encode_session(shard.shard_dir, pool_codec, batch=2)
    assert list((shard.shard_dir.parent / "latents").iterdir()) == [], "no output, staging directory or lock"


def test_a_wrong_size_image_raises(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, [0, 1])
    cv2.imwrite(str(shard.deriv_dir / IMAGE_PATTERN.format(slot=1)), np.zeros((360, 640, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="640x360"):
        encode_session(shard.shard_dir, pool_codec, batch=2)


def test_sealed_shards_need_sealed_eval(tmp_path: Path, pool_codec: PoolCodec) -> None:
    everon = write_shard(tmp_path / "a", SLOTS, map_id="everon")
    heldout = write_shard(tmp_path / "heldout", SLOTS)
    for shard in (everon, heldout):
        with pytest.raises(PermissionError, match="sealed_eval"):
            encode_session(shard.shard_dir, pool_codec, batch=4)
        assert encode_session(shard.shard_dir, pool_codec, batch=4, sealed_eval=True).is_dir()


# --- one writer per destination ---------------------------------------------------------------------------------


class ScaledCodec(PoolCodec):
    """PoolCodec times ``scale``; with ``gate``, every encode first signals ``entered`` and waits for ``release``."""

    def __init__(
        self, scale: float, gate: tuple[threading.Event, threading.Event] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(POOL_INFO, device="cpu", dtype=torch.float32, **kwargs)
        self.scale = scale
        self.gate = gate

    def _encode_unit(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is not None:
            entered, release = self.gate
            entered.set()
            if not release.wait(timeout=30):
                raise TimeoutError("the test never released the encode")
        return super()._encode_unit(x) * self.scale


def test_existing_output_needs_overwrite_and_is_replaced_consistently(tmp_path: Path) -> None:
    shard = write_shard(tmp_path, SLOTS)
    out = encode_session(shard.shard_dir, ScaledCodec(1.0), batch=4)
    first = np.load(out / "latents.npy")
    (out / "stale").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        encode_session(shard.shard_dir, ScaledCodec(2.0), batch=4)
    assert np.array_equal(np.load(out / "latents.npy"), first)
    encode_session(shard.shard_dir, ScaledCodec(2.0), batch=4, overwrite=True)
    assert not (out / "stale").exists()
    second = np.load(out / "latents.npy").astype(np.float64)
    np.testing.assert_allclose(second, 2 * first.astype(np.float64), rtol=1e-3, atol=1e-4)  # float16 rounding
    _assert_consistent(out)
    assert [p.name for p in out.parent.iterdir()] == ["test-pool"], "no .old, staging directory or lock is left"


def test_a_concurrent_encode_is_refused_while_the_first_one_runs(tmp_path: Path) -> None:
    shard = write_shard(tmp_path, SLOTS)
    entered, release = threading.Event(), threading.Event()
    result: dict[str, Any] = {}

    def first() -> None:
        try:
            result["out"] = encode_session(shard.shard_dir, ScaledCodec(2.0, (entered, release)), batch=4)
        except BaseException as error:  # reported by the assertions below
            result["error"] = error

    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert entered.wait(timeout=30), "the first encode never started"
        for overwrite in (False, True):
            with pytest.raises(LatentsLockedError, match=r"another encode .*\"pid\""):
                encode_session(shard.shard_dir, ScaledCodec(1.0), batch=4, overwrite=overwrite)
    finally:
        release.set()
        thread.join(timeout=60)
    assert "error" not in result, result.get("error")
    out = result["out"]
    _assert_consistent(out)
    reference = encode_session(write_shard(tmp_path / "ref", SLOTS).shard_dir, ScaledCodec(1.0), batch=4)
    published = np.load(out / "latents.npy").astype(np.float64)
    expected = 2 * np.load(reference / "latents.npy").astype(np.float64)
    np.testing.assert_allclose(published, expected, rtol=1e-3, atol=1e-4)  # the first writer's latents won
    assert [p.name for p in out.parent.iterdir()] == ["test-pool"]


def test_a_left_lock_refuses_without_touching_the_output(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    out = encode_session(shard.shard_dir, pool_codec, batch=4)
    digest = _sha256(out / "latents.npy")
    lock = out.parent / ".test-pool.lock"
    lock.write_text('{"pid": 1, "started": "2026-01-01T00:00:00+00:00"}\n', encoding="utf-8")
    with pytest.raises(LatentsLockedError, match="delete the lock file"):
        encode_session(shard.shard_dir, pool_codec, batch=4, overwrite=True)
    assert lock.is_file(), "a refused call never removes another writer's lock"
    assert _sha256(out / "latents.npy") == digest


def test_a_failed_overwrite_keeps_the_old_output(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    out = encode_session(shard.shard_dir, pool_codec, batch=4)
    digest = _sha256(out / "latents.npy")
    (shard.deriv_dir / IMAGE_PATTERN.format(slot=5)).unlink()
    with pytest.raises(FileNotFoundError):
        encode_session(shard.shard_dir, pool_codec, batch=4, overwrite=True)
    assert _sha256(out / "latents.npy") == digest
    _assert_consistent(out)
    assert [p.name for p in out.parent.iterdir()] == ["test-pool"]


# --- CLI and a real shard --------------------------------------------------------------------------------------


def test_cli(
    tmp_path: Path, pool_codec: PoolCodec, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shard = write_shard(tmp_path, SLOTS)
    calls = []

    def fake_get_codec(codec_id: str, *, device: str, dtype: torch.dtype) -> PoolCodec:
        calls.append((codec_id, device, dtype))
        return pool_codec

    monkeypatch.setattr(encode_module, "get_codec", fake_get_codec)
    argv = [str(shard.shard_dir), "--codec", "wan21-vae", "--batch", "4", "--device", "cpu", "--dtype", "bfloat16"]
    assert main(argv) == 0
    assert calls == [("wan21-vae", "cpu", torch.bfloat16)]
    out = shard.shard_dir.parent / "latents" / "test-pool"
    assert capsys.readouterr().out.strip() == str(out)
    assert (out / "meta.json").is_file()


@pytest.mark.skipif(not REAL_SHARD, reason="DRIVEDIT_TEST_SHARD (a real DD-11 shard) is not set")
def test_a_real_dd11_shard_reads() -> None:
    assert REAL_SHARD is not None
    shard = read_shard(REAL_SHARD)
    assert tuple(sorted(shard.meta)) == DD11_META_KEYS, "the fixture's layout is DD-11's"
    assert shard.slot_id.shape == (shard.meta["n_steps"],)
    first = int(shard.slot_id[shard.slot_id >= 0][0])
    assert (shard.deriv_dir / IMAGE_PATTERN.format(slot=first)).is_file()
