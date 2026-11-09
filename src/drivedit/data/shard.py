"""Read-only features_v1 shards, their pre-encoded latents, and the episode-aware window index (numpy only).

    shard = Shard("<DATA_ROOT>/datasets/features_v1/<session_id>/features_v1")   # sha256 verified
    shard.array("action_cmd")             # [T, 3] float32, a read-only memmap
    shard.segments                         # usable decision ranges [start, stop), never across an episode
    index = WindowIndex([shard], length=10)
    index[0]                               # Window(shard, start, length)
    index.sample(64, np.random.default_rng(0))

A shard is the directory ``drivedit.data.build`` writes (DRIVEDIT_INTERFACES section 2). Its arrays are opened as
read-only memmaps; ``Shard(path)`` checks every array file against the sha256 in ``meta.json`` first (``verify``), and
that every segment step is wm-valid, after the warm-up and has an image (slot_id >= 0). Every file a shard or its
latents are read from must be a regular file, not a symlink, directly inside its own directory, and the latents
directory must not be a symlink.
``Shard.latents(codec_id)`` opens ``<root>/<session_id>/latents/<codec_id>/`` (section 3), the latents of
``drivedit.codec``'s pre-encoder, aligned to ``slot_id.npy``.

``WindowIndex(shards, length, stride)`` lists every ``(shard, start)`` whose decision range
``[start, start + length)`` lies inside one segment, so a window never crosses an episode, a reset or a mask gap.
The order is deterministic: shards by session_id, then segments, then starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SHARD_SCHEMA = "drivedit.features_v1-shard/1"
LATENTS_SCHEMA = "drivedit.latents/1"
FEATURES_VERSION = "features_v1"
META_NAME = "meta.json"
CODEC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class ShardError(ValueError):
    """A shard or latents directory is missing, inconsistent with its meta.json, or fails its sha256."""


@dataclass(frozen=True)
class Episode:
    """One episode's decision range ``[start, stop)`` in the shard."""

    episode_id: int
    start: int
    stop: int
    mode: str


@dataclass(frozen=True)
class Segment:
    """A usable decision range ``[start, stop)``: wm-valid, after the warm-up, inside one episode, on consecutive
    grid points (meta.json segments)."""

    episode_id: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, directory: Path) -> Path:
    """``path`` when it is a regular file, not a symlink, that resolves directly inside ``directory`` (a resolved
    shard or latents directory); ShardError otherwise. Integrity, not a sealed-data rule: every file a shard or its
    latents are read from is the shard's own."""
    if path.is_symlink():
        raise ShardError(f"{path} is a symlink; a shard reads only its own regular files")
    if not path.is_file():
        raise ShardError(f"{path} not found or not a regular file")
    if path.resolve().parent != directory:
        raise ShardError(f"{path} resolves outside {directory}")
    return path


def _read_meta(path: Path, schema: str) -> dict[str, Any]:
    path = _regular_file(path, path.parent.resolve())
    meta = json.loads(path.read_text("utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != schema:
        raise ShardError(f"{path}: schema {meta.get('schema') if isinstance(meta, dict) else None!r} is not {schema}")
    return meta


class Shard:
    """One session's features_v1 shard. ``meta`` is the parsed meta.json; ``array(name)`` a read-only memmap."""

    def __init__(self, path: str | Path, *, verify: bool = True) -> None:
        self.path = Path(path).resolve()
        self.meta = _read_meta(self.path / META_NAME, SHARD_SCHEMA)
        if self.meta.get("features_version") != FEATURES_VERSION:
            raise ShardError(f"{self.path}: features_version {self.meta.get('features_version')!r}")
        self.session_id = str(self.meta["session_id"])
        self.n_steps = int(self.meta["n_steps"])
        self.decision_hz = float(self.meta["decision_hz"])
        self.build_key = None if self.meta.get("build_key") is None else str(self.meta["build_key"])
        self.episodes = tuple(
            Episode(int(e["episode_id"]), int(e["start"]), int(e["stop"]), str(e["mode"]))
            for e in self.meta["episodes"]
        )
        self.segments = tuple(
            Segment(int(s["episode_id"]), int(s["start"]), int(s["stop"])) for s in self.meta["segments"]
        )
        self._check_ranges()
        self._arrays: dict[str, npt.NDArray[Any]] = {}
        if verify:
            self.verify()
        self._check_segments()

    def __repr__(self) -> str:
        return f"Shard({self.session_id}, n_steps={self.n_steps})"

    def __getstate__(self) -> dict[str, Any]:
        # a pickled shard (a spawned DataLoader worker) reopens its memmaps instead of copying their data
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    @property
    def names(self) -> tuple[str, ...]:
        """The array files of the shard (tensors, slot_id, step, decision_index), sorted."""
        return tuple(sorted(self.meta["arrays"]))

    def has(self, name: str) -> bool:
        return name in self.meta["arrays"]

    def array(self, name: str) -> npt.NDArray[Any]:
        """The array ``<name>.npy`` as a read-only memmap; its shape and dtype must match meta.json."""
        if name not in self._arrays:
            info = self.meta["arrays"].get(name)
            if info is None:
                missing = self.meta.get("missing", {})
                why = f": {missing[name]}" if name in missing else ""
                raise ShardError(f"{self.path} has no array {name!r}{why}")
            array = np.load(_regular_file(self.path / str(info["file"]), self.path), mmap_mode="r", allow_pickle=False)
            if list(array.shape) != list(info["shape"]) or str(array.dtype) != str(info["dtype"]):
                raise ShardError(
                    f"{self.path}/{info['file']}: shape {array.shape} {array.dtype}, meta.json says "
                    f"{info['shape']} {info['dtype']}"
                )
            self._arrays[name] = array
        return self._arrays[name]

    def verify(self) -> None:
        """Check every array file against the sha256 in meta.json; ShardError names the first mismatch."""
        for name, info in sorted(self.meta["arrays"].items()):
            path = _regular_file(self.path / str(info["file"]), self.path)
            if _sha256(path) != info["sha256"]:
                raise ShardError(f"{path}: sha256 differs from meta.json ({name})")

    def image_path(self, slot: int) -> Path:
        """The harvest derivative of an image slot (``deriv_dir/f_{slot:08d}.jpg``)."""
        image = self.meta.get("image")
        if image is None:
            raise ShardError(f"{self.path}: no image source ({self.meta.get('missing', {}).get('image')})")
        if slot < 0:
            raise ShardError(f"slot {slot}: no image (slot_id -1)")
        return Path(str(image["deriv_dir"])) / str(image["pattern"]).format(slot=slot)

    def latents(self, codec_id: str) -> Latents:
        """The latents of ``codec_id`` for this shard (``<root>/<session_id>/latents/<codec_id>/``)."""
        return Latents(self, codec_id)

    def _check_segments(self) -> None:
        """Every segment step is wm-valid, after the warm-up and has an image (slot_id >= 0): the world model sees an
        observation at every frame of a window (features_v1 wm_valid: 0 at decision_index.obs_valid 0)."""
        names = list(self.meta["channels"]["masks"])
        masks = self.array("masks")
        slot = self.array("slot_id")
        for seg in self.segments:
            rows = masks[seg.start : seg.stop]
            ok = (
                (rows[:, names.index("wm_valid")] == 1).all()
                and (rows[:, names.index("warmup")] == 0).all()
                and (slot[seg.start : seg.stop] >= 0).all()
            )
            if not ok:
                raise ShardError(f"{self.path}: segment {seg} holds a step that is not wm-valid or has no image")

    def _check_ranges(self) -> None:
        bounds = {e.episode_id: e for e in self.episodes}
        last = 0
        for seg in self.segments:
            ep = bounds.get(seg.episode_id)
            if ep is None or not (ep.start <= seg.start < seg.stop <= ep.stop) or seg.start < last:
                raise ShardError(f"{self.path}: segment {seg} is not an ordered range inside its episode")
            last = seg.stop
        if any(not (0 <= e.start < e.stop <= self.n_steps) for e in self.episodes):
            raise ShardError(f"{self.path}: an episode lies outside [0, {self.n_steps})")


@dataclass(frozen=True)
class LatentStats:
    """Per-channel latent normalisation (mean, std over C). drivedit.train computes it over the training split only,
    never from validation or test sessions (DRIVEDIT_INTERFACES section 3)."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std) or not self.mean:
            raise ValueError("mean and std need the same, non-zero number of channels")
        if not all(math.isfinite(m) for m in self.mean) or not all(math.isfinite(s) and s > 0 for s in self.std):
            raise ValueError("mean must be finite and std finite and positive")

    def normalise(self, latents: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """``(x - mean) / std`` per channel of ``[..., C, h, w]`` latents, as float32."""
        if latents.shape[-3] != len(self.mean):
            raise ValueError(f"latents have {latents.shape[-3]} channels, the stats {len(self.mean)}")
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        return ((latents.astype(np.float32) - mean) / std).astype(np.float32)


class Latents:
    """``latents.npy`` ``[T, C, h, w]`` float16 and ``valid.npy`` ``[T]`` bool of one codec, as read-only memmaps.
    Their meta.json names the shard's session_id and, in ``aligned_to``, the sha256 of its slot_id.npy; T equals the
    shard's n_steps, and valid is exactly ``slot_id >= 0``."""

    def __init__(self, shard: Shard, codec_id: str) -> None:
        if not CODEC_ID_RE.match(codec_id):
            raise ShardError(f"codec id {codec_id!r} is not a codec id")
        self.shard = shard
        self.codec_id = codec_id
        self.path = shard.path.parent / "latents" / codec_id
        for d in (self.path.parent, self.path):
            if d.is_symlink():
                raise ShardError(f"{d} is a symlink; latents are read only from the shard's own directory")
        if not self.path.is_dir():
            raise ShardError(f"{self.path} not found")
        self.path = self.path.resolve()
        self.meta = _read_meta(self.path / META_NAME, LATENTS_SCHEMA)
        if self.meta.get("codec_id") != codec_id:
            raise ShardError(f"{self.path}: meta.json codec_id {self.meta.get('codec_id')!r} is not {codec_id}")
        # latents copied from another session can match its shape and valid mask: they must name this session and
        # the exact slot_id.npy they were encoded against (drivedit.codec's encoder records both), checked before any
        # array is read
        if self.meta.get("session_id") != shard.session_id:
            raise ShardError(
                f"{self.path}: latents of session {self.meta.get('session_id')!r}, the shard is {shard.session_id}"
            )
        aligned = self.meta.get("aligned_to")
        want = shard.meta["arrays"]["slot_id"]["sha256"]
        if not isinstance(aligned, dict) or aligned.get("file") != "slot_id.npy" or aligned.get("sha256") != want:
            raise ShardError(
                f"{self.path}: aligned_to {aligned!r} is not this shard's slot_id.npy (sha256 {str(want)[:12]})"
            )
        self._open()
        shape = tuple(int(v) for v in self.meta["latent_shape"])
        if self.latents.dtype != np.float16 or self.latents.shape != (shard.n_steps, *shape):
            raise ShardError(
                f"{self.path}/latents.npy: {self.latents.shape} {self.latents.dtype}, expected "
                f"{(shard.n_steps, *shape)} float16"
            )
        if self.valid.dtype != np.bool_ or self.valid.shape != (shard.n_steps,):
            raise ShardError(f"{self.path}/valid.npy: {self.valid.shape} {self.valid.dtype}, expected [T] bool")
        # section 3: rows with slot_id -1 are zeros and valid.npy marks them, so valid is exactly slot_id >= 0. A slot
        # the codec could not encode is an encoder failure, never a gap inside a segment.
        if not np.array_equal(np.asarray(self.valid), np.asarray(shard.array("slot_id")) >= 0):
            raise ShardError(f"{self.path}: valid.npy differs from slot_id >= 0; not aligned to slot_id.npy")

    def _open(self) -> None:
        self.latents = np.load(_regular_file(self.path / "latents.npy", self.path), mmap_mode="r", allow_pickle=False)
        self.valid = np.load(_regular_file(self.path / "valid.npy", self.path), mmap_mode="r", allow_pickle=False)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        del state["latents"], state["valid"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        c, h, w = (int(v) for v in self.latents.shape[1:])
        return c, h, w


@dataclass(frozen=True)
class Window:
    """Decision steps ``[start, start + length)`` of one shard, inside one segment."""

    shard: Shard
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length

    def take(self, name: str) -> npt.NDArray[Any]:
        """The window's rows of one shard array, ``[length, ...]`` (a copy)."""
        return np.array(self.shard.array(name)[self.start : self.stop])


class WindowIndex:
    """Every ``(shard, start)`` such that ``[start, start + length)`` lies inside one segment, every ``stride``-th
    start of each segment. Shards are ordered by session_id, so the order does not depend on how they were listed."""

    def __init__(self, shards: Sequence[Shard], length: int, stride: int = 1) -> None:
        if length < 1 or stride < 1:
            raise ValueError("length and stride must be >= 1")
        ordered = sorted(shards, key=lambda s: s.session_id)
        ids = [s.session_id for s in ordered]
        if len(set(ids)) != len(ids):
            raise ValueError("a session appears twice in the index")
        rates = {s.decision_hz for s in ordered}
        if len(rates) > 1:
            raise ValueError(f"shards at different decision rates {sorted(rates)}: one window, one time base")
        self.shards = tuple(ordered)
        self.length = length
        self.stride = stride
        which: list[int] = []
        starts: list[int] = []
        for i, shard in enumerate(self.shards):
            for seg in shard.segments:
                first = list(range(seg.start, seg.stop - length + 1, stride))
                which += [i] * len(first)
                starts += first
        self._shard = np.array(which, dtype=np.int64)
        self._start = np.array(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._start)

    def __getitem__(self, i: int) -> Window:
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        return Window(self.shards[int(self._shard[i])], int(self._start[i]), self.length)

    def __iter__(self) -> Iterator[Window]:
        for i in range(len(self)):
            yield self[i]

    def sample(self, n: int, rng: np.random.Generator) -> list[Window]:
        """n distinct windows drawn uniformly by ``rng`` (no replacement)."""
        if not 0 <= n <= len(self):
            raise ValueError(f"cannot draw {n} distinct windows from {len(self)}")
        return [self[int(i)] for i in rng.choice(len(self), size=n, replace=False)]
