"""dd-train's data: the features_v1 latent windows of the training split, normalised with that split's statistics.

The dataset is DD-11's ``LatentWindowDataset`` (DRIVEDIT_INTERFACES section 4). It is resolved when ``dd-train``
runs, not when this module is imported, so drivedit.train builds and tests without it; a missing data layer raises.

``open_training_windows`` is dd-train's only data entry point, and the only place where the training-data rules are
enforced. The lower-level APIs it composes (DD-11's ``WindowIndex`` and ``LatentWindowDataset``, ``channel_stats``,
``Trainer``) trust their caller. ``check_training_sessions`` runs before any array is read:

- Everon, ``heldout/`` and ``SEALED`` data are refused outright, including through symlinks: every shard and latent
  resource is resolved first. Training code has no sealed evaluation path.
- Only Arland ``train`` sessions fit the model and its normalisation statistics, and VAL-C sessions are held out
  (dc-2.0 section 3.2).
- Each shard's decision rate must equal the model's.

Normalisation statistics are then computed over these sessions only (DRIVEDIT_INTERFACES section 3).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import stat
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from drivedit.train.config import DataConfig, RunConfig

#: The (map_id, split) pairs that may fit a learned part, normalisation statistics included. dc-2.0 section 3.2:
#: Arland ``train`` fits everything, except its hash-chosen VAL-C sessions (``is_val_c``); ``val_l``, ``a0`` and
#: ``a_stress`` are evaluation references; Kolguyev (``k_val``, ``k_rep``) never trains; Everon is sealed. The route
#: split tags are drivedit_nav ``MAP_SPLITS``.
TRAINING_SPLITS = {"arland": ("train",)}
#: Path parts and map ids of the sealed test map (dc-2.0 section 3.2; the dc-2.0 reader's and drivedit_nav's rule).
SEALED_PATH_PARTS = ("heldout", "sealed")
SEALED_MAP_IDS = ("everon",)
#: The metadata file of a shard folder and of a latents folder, and the files the latents reader opens (DD-11 Latents).
META_NAME = "meta.json"
LATENTS_FILES = (META_NAME, "latents.npy", "valid.npy")


class SealedDataError(PermissionError):
    """Everon, a ``heldout/`` path or a ``SEALED`` directory met by training code, which never reads them."""


class TrainingDataError(ValueError):
    """A session that may not train this model: an evaluation split, a non-training map or another decision rate."""


def is_val_c(session_id: str) -> bool:
    """dc-2.0 section 3.2: VAL-C is the 10% of Arland ``train`` sessions with ``int(sha256(session_id)[:8], 16) mod 10
    == 0``, the comparison of the validator's C02 (``tools/validator/checks/files.py``)."""
    return int(hashlib.sha256(session_id.encode("ascii")).hexdigest()[:8], 16) % 10 == 0


def _refuse_sealed(path: Path) -> None:
    """``SealedDataError`` if ``path``, as given or resolved with every symlink followed, has a heldout / SEALED part.
    Only the path is resolved; nothing is opened."""
    resolved = Path(os.path.realpath(path))
    if {p.lower() for p in (*path.parts, *resolved.parts)} & set(SEALED_PATH_PARTS):
        raise SealedDataError(f"{path} (resolved {resolved}): heldout/ or SEALED data; training code never reads it")


def _refuse_sealed_resources(shard_dir: Path, latents_dir: Path) -> None:
    """Apply the sealed path rule to both directories, then to every entry in them. Listing a directory reads no
    file."""
    for directory in (shard_dir, latents_dir):
        _refuse_sealed(directory)
    for directory in (shard_dir, latents_dir):
        if directory.is_dir():
            for entry in sorted(directory.iterdir()):
                _refuse_sealed(entry)


def _plain_name(name: object, where: Path) -> str:
    """``name`` if it is a plain file name: a non-empty string with no separator, no ``..``, no NUL or other control
    character, and not absolute."""
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
        or Path(name).is_absolute()
    ):
        raise TrainingDataError(f"{where} references {name!r}; only a plain file name in its own folder is allowed")
    return name


def _resource(folder: Path, name: object, where: Path) -> Path:
    """The file ``name`` in the resolved ``folder``, checked before anything reads it.

    Only the path is resolved and ``lstat``-ed; nothing is opened. The checks:
    - a plain name (``_plain_name``; ``where`` is the file that names it);
    - the sealed path rule;
    - it resolves directly inside ``folder``;
    - it is a regular file, not a symlink;
    - it has exactly one hard link, since a second link could alias a file under ``heldout/`` or ``SEALED`` whatever
      path it is opened by.
    """
    path = folder / _plain_name(name, where)
    _refuse_sealed(path)
    if path.resolve().parent != folder:
        raise TrainingDataError(f"{path} resolves to {path.resolve()}, outside {folder}")
    try:
        st = path.lstat()
    except FileNotFoundError:
        raise TrainingDataError(f"{path} is missing") from None
    if not stat.S_ISREG(st.st_mode):
        raise TrainingDataError(f"{path} is not a regular file (a symlink or another kind of file)")
    if st.st_nlink != 1:
        raise TrainingDataError(
            f"{path} has {st.st_nlink} hard links; a training file has one, so it cannot alias sealed data"
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    meta = json.loads(path.read_text("utf-8"))
    if not isinstance(meta, dict):
        raise TrainingDataError(f"{path} is not an object")
    return meta


def _check_latents_identity(latents_meta: dict[str, Any], meta: dict[str, Any], codec_id: str, where: Path) -> None:
    """The latents folder belongs to the approved shard: an encoder output copied into the wrong session is refused.

    DD-19's pre-encoder writes into the latents meta.json the shard's ``session_id``, and in ``aligned_to.sha256``
    the sha256 of the shard's ``slot_id.npy``, which it checks against the shard meta's ``arrays.slot_id.sha256``.
    Both must match this shard. ``codec_id``, ``build_key``, ``map_id`` and ``split``, where the latents meta has
    them, must agree too. An Everon session, map or split in the latents meta raises ``SealedDataError``.
    """
    sid = latents_meta.get("session_id")
    map_id, split = latents_meta.get("map_id"), latents_meta.get("split")
    if str(sid).lower().startswith("everon") or str(map_id).lower() in SEALED_MAP_IDS:
        raise SealedDataError(f"{where}: latents of session {sid!r}, map {map_id!r}: the sealed test map")
    if str(split).lower().startswith("everon"):
        raise SealedDataError(f"{where}: latents of split {split!r}: the sealed test map")
    if sid != meta["session_id"]:
        raise TrainingDataError(f"{where}: latents of session {sid!r}, not of this shard's {meta['session_id']!r}")
    slot = meta.get("arrays", {}).get("slot_id")
    shard_slots = slot.get("sha256") if isinstance(slot, dict) else None
    aligned = latents_meta.get("aligned_to")
    latents_slots = aligned.get("sha256") if isinstance(aligned, dict) else None
    if shard_slots is None or latents_slots != shard_slots:
        raise TrainingDataError(
            f"{where}: aligned_to.sha256 {latents_slots!r} is not this shard's arrays.slot_id.sha256 {shard_slots!r}; "
            "the latents were encoded from another shard or build"
        )
    for key, want in (("codec_id", codec_id), ("build_key", meta.get("build_key")), ("map_id", meta.get("map_id"))):
        if key in latents_meta and want is not None and latents_meta[key] != want:
            raise TrainingDataError(f"{where}: {key} {latents_meta[key]!r}, but this shard's is {want!r}")
    if "split" in latents_meta and split != meta.get("split"):
        raise TrainingDataError(f"{where}: split {split!r}, but this shard's is {meta.get('split')!r}")


def _check_referenced_files(shard_dir: Path, latents_dir: Path, meta: dict[str, Any], codec_id: str) -> None:
    """Every file the shard and latents readers open, and every file their meta.json names, as a ``_resource``; and
    the latents' identity (``_check_latents_identity``), all before a latent array is read.

    The shard reader opens ``meta["arrays"][name]["file"]`` relative to the resolved shard folder (an absolute name
    would win in that path join). The latents reader opens ``LATENTS_FILES`` in the latents folder. The latents
    meta.json names its own files in ``sha256`` and the shard's ``aligned_to.file`` (DD-19's pre-encoder).
    """
    arrays = meta.get("arrays", {})
    if not isinstance(arrays, dict):
        raise TrainingDataError(f"{shard_dir}/meta.json: arrays is not an object")
    for info in arrays.values():
        _resource(shard_dir, info.get("file") if isinstance(info, dict) else None, shard_dir / META_NAME)
    if not latents_dir.exists():
        return  # the latents reader raises on a missing folder
    latents_real = latents_dir.resolve()
    for name in LATENTS_FILES:
        _resource(latents_real, name, latents_real)
    latents_meta = _read_json(latents_real / META_NAME)
    _check_latents_identity(latents_meta, meta, codec_id, latents_real / META_NAME)
    sha256, aligned = latents_meta.get("sha256", {}), latents_meta.get("aligned_to", {})
    for name in sha256 if isinstance(sha256, dict) else ():
        _resource(latents_real, name, latents_real / META_NAME)
    if isinstance(aligned, dict) and "file" in aligned:
        _resource(shard_dir, aligned["file"], latents_real / META_NAME)


def _check_meta(session: str, meta: dict[str, Any], decision_hz: float) -> None:
    sid, map_id, split, hz = meta.get("session_id"), meta.get("map_id"), meta.get("split"), meta.get("decision_hz")
    if not isinstance(sid, str) or not sid.isascii() or sid != session:
        raise TrainingDataError(f"{session}: meta.json names session {sid!r}; a shard lies in its own session's folder")
    terrain = str(meta.get("terrain", "")).lower()
    if sid.lower().startswith("everon") or str(map_id).lower() in SEALED_MAP_IDS or terrain == "eden":
        raise SealedDataError(f"{session}: map_id {map_id!r}, terrain {terrain!r} is the sealed test map")
    if str(split).lower().startswith("everon"):
        raise SealedDataError(f"{session}: split {split!r} is the sealed test map")
    if split not in TRAINING_SPLITS.get(str(map_id), ()):
        raise TrainingDataError(
            f"{session}: split {split!r} of map {map_id!r} never trains or enters normalisation statistics; "
            f"only {TRAINING_SPLITS} do (dc-2.0 section 3.2)"
        )
    if is_val_c(sid):
        raise TrainingDataError(
            f"{session}: meta split {split!r} disagrees with dc-2.0 section 3.2, which makes this session VAL-C "
            "(int(sha256(session_id)[:8], 16) mod 10 == 0); VAL-C never trains or enters normalisation statistics"
        )
    if not isinstance(hz, int | float) or not math.isclose(hz, decision_hz):
        raise TrainingDataError(f"{session}: decision_hz {hz!r}, but the model runs at {decision_hz} Hz")


def check_training_sessions(data: DataConfig, decision_hz: float) -> list[Path]:
    """The resolved ``features_v1`` shard folders of ``data.sessions``, each checked before any of its arrays is read.

    The checks cover the resources DD-11's reader opens. That reader resolves the shard folder, reads ``meta.json``
    and the arrays named in it there, and finds the latents beside the resolved folder, in
    ``<resolved shard>/../latents/<codec_id>/``.

    1. An ``everon`` session id raises ``SealedDataError``. So does a ``heldout`` / ``SEALED`` part, as given or
       resolved, in the shard folder, that latents folder or any file in them. This happens before any of them is
       opened.
    2. ``meta.json`` alone. Its ``session_id`` must be the folder's. An Everon ``map_id``, ``terrain`` or split
       raises ``SealedDataError``. ``TrainingDataError`` is raised for a (map_id, split) outside
       ``TRAINING_SPLITS``, for a VAL-C session (``is_val_c``; defence in depth, since DD-11 writes the effective
       split), and for a ``decision_hz`` other than the model's.
    3. Every file that a reader opens, or that meta.json or the latents meta.json names, is checked as a ``_resource``:
       a plain name, inside its own resolved folder, a regular file with one hard link. Anything else raises
       ``TrainingDataError`` (``_check_referenced_files``).
    4. The latents meta.json must name this shard: its ``session_id`` and ``aligned_to.sha256`` match the shard's
       (``_check_latents_identity``), so latents copied from another session, possibly Everon, are refused.
    """
    dirs = []
    for session in data.sessions:
        if session.lower().startswith("everon"):
            raise SealedDataError(f"{session}: an Everon session; training code never reads it")
        configured = Path(data.root) / session / "features_v1"
        _refuse_sealed(configured)
        shard_dir = configured.resolve()  # what the shard reader opens ...
        latents_dir = shard_dir.parent / "latents" / data.codec_id  # ... and where it finds the latents
        _refuse_sealed_resources(shard_dir, latents_dir)
        meta = _read_json(_resource(shard_dir, META_NAME, shard_dir))
        _check_meta(session, meta, decision_hz)
        _check_referenced_files(shard_dir, latents_dir, meta, data.codec_id)
        dirs.append(shard_dir)
    return dirs


def channel_stats(chunks: Iterable[npt.NDArray[Any]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-channel mean and population std over every element of latent chunks ``[n, C, h, w]`` (float64 sums)."""
    total = sq = None
    count = 0
    for chunk in chunks:
        x = np.asarray(chunk, dtype=np.float64)
        s, s2 = x.sum(axis=(0, 2, 3)), np.square(x).sum(axis=(0, 2, 3))
        total, sq = (s, s2) if total is None or sq is None else (total + s, sq + s2)
        count += x.shape[0] * x.shape[2] * x.shape[3]
    if total is None or sq is None or count == 0:
        raise ValueError("no valid latent rows to compute statistics from")
    mean = total / count
    std = np.sqrt(np.maximum(sq / count - np.square(mean), 0.0))
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def _module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as err:
        raise RuntimeError(f"dd-train needs {name} (DD-11, the features_v1 data layer)") from err


def _attr(module: ModuleType, name: str) -> Any:
    if not hasattr(module, name):
        raise RuntimeError(f"{module.__name__} has no {name} (DD-11, the features_v1 data layer)")
    return getattr(module, name)


def open_training_windows(cfg: RunConfig, *, chunk_rows: int = 4096) -> Dataset[Any]:
    """dd-train's only data entry point: the training windows of ``cfg.data``.

    Each session gives ``window``-frame windows: the latents of ``codec_id``, normalised by per-channel statistics of
    the valid rows of these sessions; the conditioning channels; and the targets of the configured aux heads.
    ``check_training_sessions`` runs first. Training data built any other way skips those rules.

    The guard defends against accidental contamination: wrong paths, symlinks, hard links, mislabelled or
    hash-held-out splits, latents copied from another session, and sealed maps. It does not defend against a local
    user deliberately changing files during or between the checks (TOCTOU), which is out of scope for local
    single-user runs.
    """
    if cfg.data is None:
        raise ValueError("RunConfig.data is required to train from shards")
    shard_dirs = check_training_sessions(cfg.data, cfg.model.decision_hz)
    shard_mod, torch_mod = _module("drivedit.data.shard"), _module("drivedit.data.torch")
    shard_cls, index_cls, stats_cls = (_attr(shard_mod, n) for n in ("Shard", "WindowIndex", "LatentStats"))
    dataset_cls = _attr(torch_mod, "LatentWindowDataset")

    codec_id = cfg.data.codec_id
    shards = [shard_cls(path) for path in shard_dirs]

    def valid_rows() -> Iterable[npt.NDArray[Any]]:
        for shard in shards:
            latents = shard.latents(codec_id)
            for start in range(0, len(latents.valid), chunk_rows):
                valid = np.asarray(latents.valid[start : start + chunk_rows], dtype=np.bool_)
                yield np.asarray(latents.latents[start : start + chunk_rows])[valid]

    mean, std = channel_stats(valid_rows())
    dataset: Dataset[Any] = dataset_cls(
        index_cls(shards, length=cfg.model.window, stride=cfg.data.stride),
        codec_id,
        stats=stats_cls(mean=mean, std=std),
        channels=("action_cmd", "state"),
        aux=cfg.model.aux_heads,  # each head is named as the features_v1 target it predicts
    )
    return dataset
