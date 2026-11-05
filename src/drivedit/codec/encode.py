"""The latent pre-encoder: one features_v1 shard's frames through a frozen codec, written once (DRIVEDIT_INTERFACES §3).

``encode_session(shard_dir, codec, batch=...)`` reads ``<root>/<session_id>/features_v1/`` (§2) and writes
``<root>/<session_id>/latents/<codec_id>/``:

- ``latents.npy``: ``[T, C, h, w]`` float16, row t the latent of step t's image ``deriv_dir/f_{slot:08d}.jpg`` with
  ``slot = slot_id[t]``; rows with ``slot_id == -1`` are zeros.
- ``valid.npy``: ``[T]`` bool, ``slot_id >= 0``.
- ``meta.json``: ``schema "drivedit.latents/1"``, the codec id and its pinned weight source, the input size (288, 512),
  the latent shape, the per-channel ``mean``/``std`` of the stored float16 latents over the valid rows of this session
  (population statistics over every row and grid position), the dtype and the encode timestamp. These statistics are
  provenance only: train-time normalisation is computed over the training split by ``drivedit.train``.

The images are the harvest derivatives (512x288 JPEG q95, features_v1 ``image`` rule R1), decoded with OpenCV and
converted BGR -> RGB; an image of another size raises. A step whose image is missing raises: nothing is skipped
silently. The output is written to a sibling ``.<codec_id>.partial`` directory and renamed into place at the end, so
an interrupted run never leaves a half-written ``latents/<codec_id>``.

Like the dc-2.0 reader, shards of Everon (``map_id`` everon) or under a ``heldout/`` or ``SEALED`` directory are refused
unless ``sealed_eval=True`` (contract section 3.2).

CLI::

    python -m drivedit.codec.encode <root>/<session_id>/features_v1 --codec cadre-1x16x16 --batch 16 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
import torch

from drivedit import __version__
from drivedit.codec.base import Codec, codec_ids, get_codec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["IMAGE_SIZE", "LATENTS_SCHEMA", "SHARD_SCHEMA", "encode_session", "main"]

SHARD_SCHEMA: Final = "drivedit.features_v1-shard/1"
LATENTS_SCHEMA: Final = "drivedit.latents/1"
IMAGE_PATTERN: Final = "f_{slot:08d}.jpg"
IMAGE_SIZE: Final = (288, 512)  # features_v1 tensors[image].shape [T, 3, 288, 512]
SEALED_PATH_PARTS: Final = ("heldout", "sealed")  # data/dc2/reader.py SEALED_PATH_PARTS
SEALED_MAPS: Final = ("everon",)
DTYPES: Final = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_shard_meta(shard_dir: Path) -> dict[str, Any]:
    path = shard_dir / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: not a features_v1 shard")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"{path}: schema {meta.get('schema')!r}, expected {SHARD_SCHEMA!r}")
    return meta


def _check_not_sealed(shard_dir: Path, meta: dict[str, Any], sealed_eval: bool) -> None:
    if sealed_eval:
        return
    parts = [part.lower() for part in shard_dir.resolve().parts]
    if any(part in SEALED_PATH_PARTS for part in parts):
        raise PermissionError(
            f"{shard_dir} lies under a heldout/ or SEALED directory; encode it only with "
            "sealed_eval=True (contract section 3.2)"
        )
    if str(meta.get("map_id", "")).lower() in SEALED_MAPS:
        raise PermissionError(
            f"{shard_dir} is an Everon shard; encode it only with sealed_eval=True (contract section 3.2)"
        )


def _image_dir(shard_dir: Path, meta: dict[str, Any]) -> Path:
    """``deriv_dir`` of the shard's ``image_source``; a relative path is taken from the shard directory."""
    source = meta.get("image_source")
    if not isinstance(source, dict) or "deriv_dir" not in source or "pattern" not in source:
        raise ValueError(f"{shard_dir}/meta.json: image_source must hold deriv_dir and pattern")
    if source["pattern"] != IMAGE_PATTERN:
        raise ValueError(f"{shard_dir}/meta.json: image pattern {source['pattern']!r}, expected {IMAGE_PATTERN!r}")
    deriv_dir = Path(source["deriv_dir"])
    deriv_dir = deriv_dir if deriv_dir.is_absolute() else shard_dir / deriv_dir
    if not deriv_dir.is_dir():
        raise FileNotFoundError(f"image directory {deriv_dir} does not exist")
    return deriv_dir


def _read_image(path: Path) -> NDArray[np.uint8]:
    """One derivative JPEG as ``[3, 288, 512]`` uint8 RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    if bgr.shape != (*IMAGE_SIZE, 3):
        raise ValueError(f"{path}: image is {bgr.shape[1]}x{bgr.shape[0]}, expected {IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}")
    rgb: NDArray[np.uint8] = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
    return rgb


def _read_slots(shard_dir: Path, n_steps: int) -> NDArray[np.int64]:
    path = shard_dir / "slot_id.npy"
    slot_id = np.load(path)
    if slot_id.dtype != np.int64 or slot_id.shape != (n_steps,):
        raise ValueError(f"{path}: expected [{n_steps}] int64 (n_steps), got {slot_id.shape} {slot_id.dtype}")
    if np.any(slot_id < -1):
        raise ValueError(f"{path}: slot ids below -1")
    if not np.any(slot_id >= 0):
        raise ValueError(f"{path}: no step has an image")
    slots: NDArray[np.int64] = slot_id
    return slots


def _encode_rows(
    codec: Codec, deriv_dir: Path, slot_id: NDArray[np.int64], latents: NDArray[np.float16], batch: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fill the valid rows of ``latents`` ``[T, C, h, w]``; returns the per-channel mean and std of what was stored."""
    rows = np.flatnonzero(slot_id >= 0)
    channels, height, width = latents.shape[1:]
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    for start in range(0, rows.size, batch):
        chunk = rows[start : start + batch]
        images = np.stack([_read_image(deriv_dir / IMAGE_PATTERN.format(slot=int(slot_id[t]))) for t in chunk])
        z = codec.encode(torch.from_numpy(images).to(codec.device)).to(torch.float16).cpu().numpy()
        if not np.isfinite(z).all():
            raise FloatingPointError(f"{codec.codec_id}: non-finite latents for steps {chunk[0]}..{chunk[-1]}")
        latents[chunk] = z
        wide = z.astype(np.float64)
        total += wide.sum(axis=(0, 2, 3))
        total_sq += np.square(wide).sum(axis=(0, 2, 3))
    count = rows.size * height * width
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
    return mean, std


def encode_session(
    shard_dir: Path | str,
    codec: Codec,
    *,
    batch: int,
    overwrite: bool = False,
    sealed_eval: bool = False,
) -> Path:
    """Encode every image of the shard at ``shard_dir`` with ``codec``; returns ``<root>/<session_id>/latents/<id>``.

    ``batch`` frames go through the codec at a time. An existing output raises ``FileExistsError`` unless
    ``overwrite``, which replaces it.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    shard_dir = Path(shard_dir)
    meta = _read_shard_meta(shard_dir)
    _check_not_sealed(shard_dir, meta, sealed_eval)
    n_steps = int(meta["n_steps"])
    slot_id = _read_slots(shard_dir, n_steps)
    deriv_dir = _image_dir(shard_dir, meta)
    latent_shape = codec.latent_shape(*IMAGE_SIZE)

    out_dir = shard_dir.parent / "latents" / codec.codec_id
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} exists; pass overwrite=True to replace it")
    partial = out_dir.with_name(f".{codec.codec_id}.partial")
    if partial.exists():
        shutil.rmtree(partial)  # left by an interrupted run of this function
    partial.mkdir(parents=True)

    try:
        valid = slot_id >= 0
        latents = np.lib.format.open_memmap(
            partial / "latents.npy", mode="w+", dtype=np.float16, shape=(n_steps, *latent_shape)
        )
        latents[~valid] = 0
        mean, std = _encode_rows(codec, deriv_dir, slot_id, latents, batch)
        latents.flush()
        del latents
        np.save(partial / "valid.npy", valid)

        weights = codec.weights
        out_meta = {
            "schema": LATENTS_SCHEMA,
            "codec_id": codec.codec_id,
            "weights": {
                "repo_id": weights.repo_id,
                "revision": weights.revision,
                "files": list(weights.files),
                "licence": weights.licence,
                "code": weights.code,
            },
            "session_id": meta.get("session_id"),
            "build_key": meta.get("build_key"),
            "n_steps": n_steps,
            "n_valid": int(valid.sum()),
            "input_size": list(IMAGE_SIZE),
            "latent_shape": list(latent_shape),
            "dtype": "float16",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "stats": "per channel over the valid rows and all grid positions of the stored float16 latents; population",
            "compute_dtype": str(codec.dtype).removeprefix("torch."),
            "aligned_to": {"file": "slot_id.npy", "sha256": _sha256(shard_dir / "slot_id.npy")},
            "sha256": {name: _sha256(partial / name) for name in ("latents.npy", "valid.npy")},
            "software": {"drivedit": __version__, "torch": torch.__version__},
            "encoded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (partial / "meta.json").write_text(json.dumps(out_meta, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if out_dir.exists():
        shutil.rmtree(out_dir)  # overwrite=True, checked above
    partial.rename(out_dir)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m drivedit.codec.encode``: encode one features_v1 shard with one codec."""
    parser = argparse.ArgumentParser(
        prog="python -m drivedit.codec.encode",
        description="Encode one features_v1 shard's images into latents/<codec_id>/ (DRIVEDIT_INTERFACES section 3).",
    )
    parser.add_argument("shard_dir", type=Path, help="a <root>/<session_id>/features_v1 directory")
    parser.add_argument("--codec", required=True, choices=codec_ids())
    parser.add_argument("--batch", type=int, default=16, help="frames per codec call (default 16)")
    parser.add_argument("--device", default="cuda", help="torch device (default cuda)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32", help="compute dtype (default float32)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing latents/<codec_id>")
    parser.add_argument("--sealed-eval", action="store_true", help="allow Everon / heldout / SEALED shards")
    args = parser.parse_args(argv)
    codec = get_codec(args.codec, device=args.device, dtype=DTYPES[args.dtype])
    out_dir = encode_session(
        args.shard_dir, codec, batch=args.batch, overwrite=args.overwrite, sealed_eval=args.sealed_eval
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
