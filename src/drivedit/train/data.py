"""dd-train's data: the features_v1 latent windows of the training split, normalised with that split's statistics.

The dataset is DD-11's ``LatentWindowDataset`` (DRIVEDIT_INTERFACES section 4). It is resolved when ``dd-train``
runs, not when this module is imported, so drivedit.train builds and tests without it; a missing data layer raises.

Every session is checked before any of its arrays is read (``check_training_sessions``): Everon, ``heldout/`` and
``SEALED`` paths are refused outright (training code has no sealed evaluation path), only Arland ``train`` sessions
fit the model and its normalisation statistics (dc-2.0 section 3.2), and each shard's decision rate must equal the
model's. Normalisation statistics are then computed over these sessions only (DRIVEDIT_INTERFACES section 3).
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from drivedit.train.config import DataConfig, RunConfig

#: The (map_id, split) pairs that may fit a learned part, normalisation statistics included. dc-2.0 section 3.2:
#: Arland ``train`` fits everything; ``val_c`` (hash-chosen ``train`` sessions, DD-6's manifests), ``val_l``, ``a0``
#: and ``a_stress`` are evaluation references; Kolguyev (``k_val``, ``k_rep``) never trains; Everon is sealed. The
#: route split tags are drivedit_nav ``MAP_SPLITS``.
TRAINING_SPLITS = {"arland": ("train",)}
#: Path parts and map ids of the sealed test map (dc-2.0 section 3.2; the dc-2.0 reader's and drivedit_nav's rule).
SEALED_PATH_PARTS = ("heldout", "sealed")
SEALED_MAP_IDS = ("everon",)


class SealedDataError(PermissionError):
    """Everon, a ``heldout/`` path or a ``SEALED`` directory met by training code, which never reads them."""


class TrainingDataError(ValueError):
    """A session that may not train this model: an evaluation split, a non-training map or another decision rate."""


def check_training_sessions(data: DataConfig, decision_hz: float) -> list[Path]:
    """The ``features_v1`` shard directories of ``data.sessions``, each checked before any of its arrays is read.

    First the session id and the path (resolved and as given): an ``everon`` session id or a ``heldout`` / ``SEALED``
    path part raises ``SealedDataError`` before the filesystem is touched under it. Then ``meta.json`` alone:
    ``map_id`` ``everon`` or an ``everon_*`` split raises ``SealedDataError``; a (map_id, split) outside
    ``TRAINING_SPLITS`` or a ``decision_hz`` other than the model's (``decision_hz``) raises ``TrainingDataError``.
    """
    dirs = []
    for session in data.sessions:
        path = Path(data.root) / session / "features_v1"
        parts = {p.lower() for p in (*path.parts, *Path(os.path.realpath(path)).parts)}
        if session.lower().startswith("everon") or parts & set(SEALED_PATH_PARTS):
            raise SealedDataError(f"{path}: Everon, heldout/ or SEALED data; training code never reads it")
        meta = json.loads((path / "meta.json").read_text("utf-8"))
        if not isinstance(meta, dict):
            raise TrainingDataError(f"{path}/meta.json is not an object")
        map_id, split, hz = meta.get("map_id"), meta.get("split"), meta.get("decision_hz")
        if map_id in SEALED_MAP_IDS or str(split).startswith("everon"):
            raise SealedDataError(f"{path}: map_id {map_id!r}, split {split!r} is the sealed test map")
        if split not in TRAINING_SPLITS.get(str(map_id), ()):
            raise TrainingDataError(
                f"{session}: split {split!r} of map {map_id!r} never trains or enters normalisation statistics; "
                f"only {TRAINING_SPLITS} do (dc-2.0 section 3.2)"
            )
        if not isinstance(hz, int | float) or not math.isclose(hz, decision_hz):
            raise TrainingDataError(f"{session}: decision_hz {hz!r}, but the model runs at {decision_hz} Hz")
        dirs.append(path)
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
    """The training windows of ``cfg.data``: ``window``-frame windows of every session, with the latents of
    ``codec_id`` normalised by per-channel statistics of the valid rows of these sessions, the conditioning channels
    and the targets of the configured aux heads. ``check_training_sessions`` runs first."""
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
