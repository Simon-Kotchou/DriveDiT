"""drivedit.codec.encode: a fake features_v1 shard through the pooling test codec, against DRIVEDIT_INTERFACES §3."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import torch

from drivedit.codec import encode as encode_module
from drivedit.codec.encode import IMAGE_PATTERN, LATENTS_SCHEMA, encode_session, main
from tests.drivedit.codec.helpers import write_shard

if TYPE_CHECKING:
    from pathlib import Path

    from tests.drivedit.codec.helpers import PoolCodec

SLOTS = [-1, 0, 1, 2, -1, 3, 4, 5, 6, -1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_writes_the_section_3_layout(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    out = encode_session(shard.shard_dir, pool_codec, batch=3)
    assert out == shard.shard_dir.parent / "latents" / "test-pool"
    assert sorted(p.name for p in out.iterdir()) == ["latents.npy", "meta.json", "valid.npy"]
    assert not (out.parent / ".test-pool.partial").exists()

    latents = np.load(out / "latents.npy")
    valid = np.load(out / "valid.npy")
    assert latents.shape == (len(SLOTS), 3, 18, 32)
    assert latents.dtype == np.float16
    assert valid.dtype == np.bool_
    assert valid.tolist() == [s >= 0 for s in SLOTS]
    assert not latents[~valid].any(), "rows without an image are zeros"
    assert np.abs(latents[valid]).sum(axis=(1, 2, 3)).min() > 0

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == LATENTS_SCHEMA
    assert meta["codec_id"] == "test-pool"
    assert meta["weights"]["revision"] == "0" * 40
    assert meta["input_size"] == [288, 512]
    assert meta["latent_shape"] == [3, 18, 32]
    assert meta["dtype"] == "float16"
    assert meta["n_steps"] == len(SLOTS)
    assert meta["n_valid"] == int(valid.sum())
    assert meta["session_id"] == shard.shard_dir.parent.name
    assert meta["aligned_to"] == {"file": "slot_id.npy", "sha256": _sha256(shard.shard_dir / "slot_id.npy")}
    assert meta["sha256"] == {name: _sha256(out / name) for name in ("latents.npy", "valid.npy")}
    assert meta["encoded_at"].endswith("+00:00")


def test_meta_stats_are_the_stored_latents_stats(tmp_path: Path, pool_codec: PoolCodec) -> None:
    out = encode_session(write_shard(tmp_path, SLOTS).shard_dir, pool_codec, batch=4)
    latents = np.load(out / "latents.npy").astype(np.float64)
    valid = np.load(out / "valid.npy")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    rows = latents[valid]
    np.testing.assert_allclose(meta["mean"], rows.mean(axis=(0, 2, 3)), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(meta["std"], rows.std(axis=(0, 2, 3)), rtol=1e-6, atol=1e-9)
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


def test_existing_output_needs_overwrite(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    out = encode_session(shard.shard_dir, pool_codec, batch=4)
    (out / "stale").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        encode_session(shard.shard_dir, pool_codec, batch=4)
    encode_session(shard.shard_dir, pool_codec, batch=4, overwrite=True)
    assert not (out / "stale").exists()


def test_a_missing_image_raises_and_leaves_nothing(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    (shard.deriv_dir / IMAGE_PATTERN.format(slot=4)).unlink()
    with pytest.raises(FileNotFoundError, match=r"f_00000004\.jpg"):
        encode_session(shard.shard_dir, pool_codec, batch=2)
    assert not (shard.shard_dir.parent / "latents" / "test-pool").exists()
    assert not (shard.shard_dir.parent / "latents" / ".test-pool.partial").exists()


def test_a_wrong_size_image_raises(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, [0, 1])
    cv2.imwrite(str(shard.deriv_dir / IMAGE_PATTERN.format(slot=1)), np.zeros((360, 640, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="640x360"):
        encode_session(shard.shard_dir, pool_codec, batch=2)


@pytest.mark.parametrize(
    ("slots", "message"),
    [([-1, -1], "no step has an image"), ([0, -2], "below -1")],
)
def test_bad_slot_ids_raise(tmp_path: Path, pool_codec: PoolCodec, slots: list[int], message: str) -> None:
    shard = write_shard(tmp_path, slots)
    with pytest.raises(ValueError, match=message):
        encode_session(shard.shard_dir, pool_codec, batch=2)


def test_slot_ids_must_match_n_steps(tmp_path: Path, pool_codec: PoolCodec) -> None:
    shard = write_shard(tmp_path, SLOTS)
    np.save(shard.shard_dir / "slot_id.npy", np.asarray(SLOTS[:-1], dtype=np.int64))
    with pytest.raises(ValueError, match="n_steps"):
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


def test_sealed_shards_need_sealed_eval(tmp_path: Path, pool_codec: PoolCodec) -> None:
    everon = write_shard(tmp_path / "a", SLOTS, map_id="everon")
    heldout = write_shard(tmp_path / "heldout", SLOTS)
    for shard in (everon, heldout):
        with pytest.raises(PermissionError, match="sealed_eval"):
            encode_session(shard.shard_dir, pool_codec, batch=4)
        assert encode_session(shard.shard_dir, pool_codec, batch=4, sealed_eval=True).is_dir()


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
