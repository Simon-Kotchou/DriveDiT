"""dd-train's data: the features_v1 latent windows of the training split, normalised with that split's statistics.

The dataset is DD-11's ``LatentWindowDataset`` (DRIVEDIT_INTERFACES section 4). It is resolved when ``dd-train``
runs, not when this module is imported, so drivedit.train builds and tests without it; a missing data layer raises.
Normalisation statistics are computed here over the training sessions only (section 3: never from validation or test
sessions).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from drivedit.train.config import RunConfig


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
    and the targets of the configured aux heads."""
    if cfg.data is None:
        raise ValueError("RunConfig.data is required to train from shards")
    shard_mod, torch_mod = _module("drivedit.data.shard"), _module("drivedit.data.torch")
    shard_cls, index_cls, stats_cls = (_attr(shard_mod, n) for n in ("Shard", "WindowIndex", "LatentStats"))
    dataset_cls = _attr(torch_mod, "LatentWindowDataset")

    root, codec_id = Path(cfg.data.root), cfg.data.codec_id
    shards = [shard_cls(root / session / "features_v1") for session in cfg.data.sessions]

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
