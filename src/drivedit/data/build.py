"""Build the features_v1 shard of one harvested dc-2.0 session (numpy only).

    python -m drivedit.data.build --session DIR [--session DIR ...] --out ROOT [--routes-dir DIR]
        [--vehicle-card FILE] [--include-eval-only] [--include-privileged] [--overwrite]

``build_session(session_dir, out_root, spec=spec)`` reads the session through the dc-2.0 reader
(``data/dc2/reader.py``, loaded by file path as the tools load it; strict, fiducials decoded) and writes
``<out_root>/<session_id>/features_v1/`` (DRIVEDIT_INTERFACES section 2):

- ``<tensor>.npy``: one ``[T, ...]`` array per features_v1 tensor the session provides, T = n_steps decision steps
  (one per decision callback C_t of the session, warm-up and inter-episode steps included; ``masks`` and
  ``meta.json`` segments say which are usable). Every channel is aligned at t, unshifted: the model applies the
  conditioning shift itself (section 1). float32, except ``masks`` (uint8). NaN marks a value the session does not
  have at t (no telemetry row, the no-command sentinel, an empty action_exec entry past P_t+1, a future pose outside
  the episode); nothing is filled in.
- ``slot_id.npy`` ``[T] int64``: the image slot of obs_t (``deriv_dir/f_{slot:08d}.jpg``) when obs_t is valid
  (decision_index.obs_valid 1), else -1. ``step.npy`` ``[T] int64``: P_t.
- ``decision_index.npy`` ``[T, 9] float64``: the derived decision_index of the streams JSON (decision_step, hl_seq,
  interval_steps, obs_slot_id, obs_captured_render_frame, obs_last_physics_step, obs_age_steps, obs_age_ms,
  obs_valid; NaN = empty), the realised image age per decision step that the features_v1 image rule asks the builder
  to record.
- ``meta.json``: provenance, episodes, segments, the image source, the sha256 of every array, and ``missing``: the
  features_v1 tensors this build cannot produce, with the reason.

Tensor semantics (features_v1 ``tensors[].rule``, dc-2.0 sections 6, 9.5.1, 12 and 13):

- ``state``, ``action_cmd``, ``action_label``, ``action_exec``: the reader's decision samples. state_t is the
  telemetry_export row of C_t; action_cmd_t is ctl.hl_* of the row after C_t (the applied command over
  (P_t, P_t+1]); action_label_t is ctl.exp_*_now of C_t; action_exec_t the llc output and input_post of steps
  P_t + 1 .. P_t + N, NaN past P_t+1.
- ``nav``: ``drivedit_nav.nav_sequence`` over every telemetry row of each route episode in row order (pos_x, pos_z
  and telemetry_export.heading_rad), progress carried from row to row, read at the decision steps: the contract's
  per-telemetry-row nav view (dc-2.0 section 9.8, streams JSON derived.nav). The route is
  ``<routes_dir>/<route_id>.json`` and must hash to the episode's ``route_sha256`` (or, for the one allow-listed route
  version of the DD-7 defect, to its 8191-character prefix form: ``PREFIX_HASH_ROUTES``, ``_logged_route_hash``).
  Steps outside route episodes have no nav (NaN).
- ``style``: the episode's style_v1 row, on every decision step of the episode.
- ``masks``: the reader's masks, then nav_valid 0 (more than 30 m off the route, or no pose) clears policy_valid and
  wm_valid (features_v1 nav rule).
- ``future_pose``: ego-frame pose deltas (forward, right, heading change; ``drivedit_nav.to_ego``) from C_t to
  C_t+k, k = 1 .. H, H = ``FUTURE_POSE_S`` x decision_hz. ``hindsight_anchors``: the same deltas at +2/+4/+6 s
  (k = 2, 4, 6 x decision_hz), x and y only. A delta exists only when both steps belong to the same episode, the grid
  points are consecutive (C_t+k is k grid points after C_t) and no ego teleport or respawn lies in (P_t, P_t+k].
- ``contact_events``: an ego ``collision`` row (any src) whose payload onset_step lies in the decision interval
  (P_t, P_t+1] sets contact_event 1 and contact_class to the index of its payload ``other`` in the streams JSON
  enum (``meta.json`` categories; -1 without contact); with several, the earliest onset names the class.

Not written by default (features_v1 classes): ``eval_only`` (``include_eval_only``) and the teacher_input
``privileged_route`` (``include_privileged``). The C24 forbidden names stay out of every model input through
``FeatureSpec.access``.

Output never goes inside the session directory or its captures tree, nor to the C: drive. An existing shard is
replaced only with ``overwrite=True`` (written beside it first, then swapped).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from drivedit.data.spec import FeatureSpec, load_spec, repo_root

SHARD_SCHEMA = "drivedit.features_v1-shard/1"
BUILDER_VERSION = "drivedit.data.build/1"
SHARD_DIRNAME = "features_v1"
META_NAME = "meta.json"
# future_pose spans +1 .. +H decision steps (H is set by the builder, features_v1). 6 s is the longest hindsight
# anchor, so the plan target covers the anchor horizon (comma's policy trains on rollouts of 0-7 s, MODELING_PLAN 6).
FUTURE_POSE_S = 6.0
ANCHOR_S = (2, 4, 6)  # features_v1 hindsight_anchors: +2/+4/+6 s
DERIV_DIR = Path("_harvest") / "deriv"
IMAGE_PATTERN = "f_{slot:08d}.jpg"
ROUTE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")  # route_schema.json route_id
SUBSTRING_MAX_CHARS = 8191  # 1.8.0.13 string.Substring bound (DC_TextFile.SUBSTRING_MAX_CHARS, DD-7)
# The route versions whose sessions logged a truncated route_sha256 (DD-7, fixed for new sessions by PR #17, commit
# cc31f07): DC_Route.Load hashed a string.Substring copy of the route file, and Substring returns at most 8191
# characters on 1.8.0.13, so episodes on a longer route logged the sha256 of its first 8191 bytes. route_id -> the
# full route_sha256 (drivedit_nav Route.sha256, the sha256 of the canonical JSON) of the version that was deployed.
# Only an unedited file of a listed version is accepted against its prefix hash (_logged_route_hash).
#   arland_t0_loop: the file of addon commit 78e539b (session.json addon.git_sha of the first T0L session,
#   2026-11-05, git_dirty false), one 39,692-character canonical line plus LF; its prefix hash is 6370fc78f426...
PREFIX_HASH_ROUTES = {
    "arland_t0_loop": "77838c1346f2c7a6b18be78570d9f8cb560ee10a0e69718b28eb0f028cc0f66f",
}
FORBIDDEN_ROOTS = ("/mnt/c/",)
# Written by default, in features_v1 order; image is referenced (slot_id + deriv_dir), never copied.
DEFAULT_TENSORS = (
    "state",
    "action_cmd",
    "action_label",
    "action_exec",
    "nav",
    "style",
    "masks",
    "hindsight_anchors",
    "future_pose",
    "road_context",
    "contact_events",
)
# eval_only channels with a per-decision-step computation from the reader; the rest are listed as missing.
EVAL_ONLY_BUILT = (
    "pos_x",
    "pos_y",
    "pos_z",
    "heading_world_rad",
    "vel_world_x",
    "vel_world_y",
    "vel_world_z",
    "route_progress_m",
)
EVAL_ONLY_SESSION_CONSTANTS = ("map_id", "terrain", "split", "session_id", "pin_id", "build_key")
MISSING_REASONS = {
    "road_context": (
        "road_class comes from the harvester's road view (streams JSON derived.road: E-08b road matching, "
        "DD-6/DD-7), which the reader does not build and the session does not carry; road_width_m, lane_lat_err_m and "
        "lane_head_err_rad are in telemetry, but a partial [T,4] tensor is not written"
    ),
    "privileged_route": (
        "path_kappa_* need a route-curvature estimator that features_v1 does not specify (source 'route geometry'), "
        "and lead_gap_m is reserved for traffic (T2)"
    ),
}
EVAL_ONLY_MISSING = {
    "distance_total_m": (
        "'telemetry (integrated)' does not specify the integration (3D or horizontal, from poses or speed)"
    ),
    "region": "comes from the harvester's road view (derived.road.region, splits_v2), which the reader does not build",
    "exp_now_60hz": "a per-callback diagnostic (every ctl row), not a per-decision-step channel",
}

FloatArray = npt.NDArray[np.float64]


class BuildError(ValueError):
    """The session cannot be built into a features_v1 shard (a missing or inconsistent input)."""


# ------------------------------------------------------------------------------------------------
# The repository's reader and nav_v1, loaded by file path (as tools/ load them)
# ------------------------------------------------------------------------------------------------
def _load_by_path(name: str, relpath: str) -> ModuleType:
    path = repo_root() / relpath
    if not path.is_file():
        raise BuildError(f"{path} not found: drivedit.data.build runs from a repository checkout until DD-17")
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise BuildError(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@cache
def reader_module() -> ModuleType:
    """``data/dc2/reader.py`` (the dc-2.0 reader)."""
    return _load_by_path("dc2_reader", "data/dc2/reader.py")


@cache
def nav_module() -> ModuleType:
    """``drivedit_nav/nav_v1.py``: the one nav function of the harvester, the builder and the bridge (C22)."""
    return _load_by_path("drivedit_nav_v1", "drivedit_nav/nav_v1.py")


def default_routes_dir() -> Path:
    """The capture addon's route files, ``addons/DataCapture/Configs/routes`` (the harvester's default)."""
    return repo_root() / "addons" / "DataCapture" / "Configs" / "routes"


def open_dc2_session(session_dir: str | Path, *, vehicle_card: str | Path | None = None) -> Any:
    """The reader's Session of a sealed dc-2.0 session: strict (any C01/C02/C05/C07/C34 problem raises), never on
    the sealed evaluation path, fiducials decoded (obs_t needs the captured render frame). ``vehicle_card`` overrides
    the reader's default card (``tools/ident/cards/<pin_id>/<vehicle_card_id>.json``)."""
    reader = reader_module()
    card = None if vehicle_card is None else str(vehicle_card)
    return reader.open_session(str(session_dir), strict=True, decode_images=True, vehicle_card=card)


# ------------------------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------------------------
def build_session(
    session_dir: str | Path,
    out_root: str | Path,
    *,
    spec: FeatureSpec,
    include_privileged: bool = False,
    include_eval_only: bool = False,
    routes_dir: str | Path | None = None,
    vehicle_card: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Read one session and write its shard; returns ``<out_root>/<session_id>/features_v1``."""
    session = open_dc2_session(session_dir, vehicle_card=vehicle_card)
    return write_shard(
        session,
        out_root,
        spec=spec,
        include_privileged=include_privileged,
        include_eval_only=include_eval_only,
        routes_dir=routes_dir,
        overwrite=overwrite,
    )


def write_shard(
    session: Any,
    out_root: str | Path,
    *,
    spec: FeatureSpec,
    include_privileged: bool = False,
    include_eval_only: bool = False,
    routes_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the shard of a Session that ``open_dc2_session`` returned (the build without the read). The output
    paths are resolved and checked (``_shard_dir``) before anything is computed or written."""
    session_real = Path(str(session.path)).resolve()
    out_dir = _shard_dir(Path(out_root), session_real, str(session.session_id))
    if out_dir.exists() and not overwrite:
        raise BuildError(f"{out_dir} exists; pass overwrite=True (--overwrite) to replace it")
    arrays, meta = session_features(
        session,
        spec=spec,
        routes_dir=Path(routes_dir) if routes_dir is not None else default_routes_dir(),
        include_privileged=include_privileged,
        include_eval_only=include_eval_only,
    )
    tmp = out_dir.with_name(f".{SHARD_DIRNAME}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _write_files(tmp, arrays, meta)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out_dir.exists():
        old = out_dir.with_name(f".{SHARD_DIRNAME}.old-{os.getpid()}")
        out_dir.rename(old)
        tmp.rename(out_dir)
        shutil.rmtree(old)
    else:
        tmp.rename(out_dir)
    return out_dir


def _write_files(directory: Path, arrays: dict[str, npt.NDArray[Any]], meta: dict[str, Any]) -> None:
    """Each array as ``<name>.npy`` and meta.json with their shapes, dtypes and sha256 (``arrays``)."""
    files: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        files[name] = {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    meta["arrays"] = files
    (directory / META_NAME).write_text(json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")


# ------------------------------------------------------------------------------------------------
# The tensors
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Timeline:
    """The decision steps of a session: P_t, grid points, episode ids, poses and the ego teleport steps."""

    steps: npt.NDArray[np.int64]
    grid: npt.NDArray[np.int64]
    episode: npt.NDArray[np.int64]
    poses: FloatArray
    tele_steps: tuple[int, ...]
    decision_hz: float
    n_exec: int
    horizon: int

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def session_features(
    session: Any,
    *,
    spec: FeatureSpec,
    routes_dir: Path,
    include_privileged: bool = False,
    include_eval_only: bool = False,
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """(arrays by file stem, meta.json without its ``arrays`` block) of one Session."""
    tl = _timeline(session, spec)
    episodes = _episodes(session, tl.episode)
    routes = _routes(session, episodes, routes_dir)
    arrays, nav_valid, progress, contact_counts = _model_tensors(session, spec, tl, episodes, routes)
    missing = {"road_context": MISSING_REASONS["road_context"]}
    excluded: dict[str, str] = {}
    if include_eval_only:
        arrays["eval_only"] = _eval_only(session, tl.steps, progress)
    else:
        excluded["eval_only"] = "eval_only channels are never inputs or targets; written only with include_eval_only"
    if include_privileged:
        missing["privileged_route"] = MISSING_REASONS["privileged_route"]
    else:
        excluded["privileged_route"] = "teacher_input (privileged); written only with include_privileged"
    image, image_missing = _index_arrays(session, arrays, tl.steps)
    if image is None:
        missing["image"] = image_missing
    segments = _segments(session, arrays["masks"], spec, tl.steps)
    meta = _meta(session, spec, tl, episodes=episodes, routes=routes, segments=segments)
    meta["image"] = image
    meta["missing"] = missing
    meta["excluded"] = excluded
    meta["channels"] = {name: list(spec.tensor(name).channel_names) for name in arrays if name in spec.tensor_names}
    if include_eval_only:
        meta["channels"]["eval_only"] = list(EVAL_ONLY_BUILT)
    mask_names = spec.tensor("masks").channel_names
    meta["counts"].update(
        {
            "wm_valid": int(arrays["masks"][:, mask_names.index("wm_valid")].sum()),
            "policy_valid": int(arrays["masks"][:, mask_names.index("policy_valid")].sum()),
            "nav_invalid_in_route_episodes": int((nav_valid == 0).sum()),
            "nonfinite_inputs_in_segments": _nonfinite_in_segments(arrays, segments),
            **contact_counts,
        }
    )
    if include_eval_only:
        meta["eval_only"] = {
            "channels": list(EVAL_ONLY_BUILT),
            "missing": EVAL_ONLY_MISSING,
            "session_constants_in_meta": list(EVAL_ONLY_SESSION_CONSTANTS),
        }
    return arrays, meta


def _timeline(session: Any, spec: FeatureSpec) -> _Timeline:
    decisions: list[dict[str, Any]] = list(session.decisions)
    if not decisions:
        raise BuildError(f"{session.path}: no decision callbacks")
    decision_hz = float(session.clocks["decision_hz"])
    n_exec = int(session.clocks["N"])
    if spec.steps_per_decision(decision_hz) != n_exec:
        raise BuildError(
            f"{session.path}: physics_steps_per_decision {n_exec} at {decision_hz} Hz, features_v1 has "
            f"{spec.steps_per_decision(decision_hz)}"
        )
    if session.vehicle_card is None:
        raise BuildError(
            f"{session.path}: no vehicle card, so state.dir_state cannot be read (TBD-7); pass vehicle_card"
        )
    steps = np.array([int(d["decision_step"]) for d in decisions], dtype=np.int64)
    return _Timeline(
        steps=steps,
        grid=np.array([int(d["grid_point"]) for d in decisions], dtype=np.int64),
        episode=np.array([_int_or(d["episode_id"], -1) for d in decisions], dtype=np.int64),
        poses=_poses(session, steps),
        tele_steps=tuple(_teleport_steps(session)),
        decision_hz=decision_hz,
        n_exec=n_exec,
        horizon=round(FUTURE_POSE_S * decision_hz),
    )


def _model_tensors(
    session: Any,
    spec: FeatureSpec,
    tl: _Timeline,
    episodes: dict[int, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> tuple[dict[str, npt.NDArray[Any]], npt.NDArray[np.int8], FloatArray, dict[str, int]]:
    """(the default tensors, nav_valid, progress_m, contact counts); every shape checked against features_v1."""
    decisions: list[dict[str, Any]] = list(session.decisions)
    arrays: dict[str, npt.NDArray[Any]] = {
        "state": _rows(decisions, "state", 4),
        "action_cmd": _rows(decisions, "action_cmd", 3),
        "action_label": _rows(decisions, "action_label", 3),
        "action_exec": _exec(decisions, tl.n_exec),
    }
    nav, nav_valid, progress = _nav(spec, session, tl, episodes, routes)
    arrays["nav"] = nav
    arrays["style"] = _style(spec, session, tl.episode)
    arrays["masks"] = _masks(spec, decisions, nav_valid)
    arrays["hindsight_anchors"] = (
        _ego_deltas(tl, [round(s * tl.decision_hz) for s in ANCHOR_S])[:, :, :2]
        .reshape(tl.n_steps, 2 * len(ANCHOR_S))
        .astype(np.float32)
    )
    arrays["future_pose"] = _ego_deltas(tl, list(range(1, tl.horizon + 1))).astype(np.float32)
    arrays["contact_events"], contact_counts = _contacts(session, tl.steps, _contact_categories(session))
    for name, array in arrays.items():
        want = spec.shape(name, tl.n_steps, decision_hz=tl.decision_hz, horizon=tl.horizon)
        if array.shape != want:
            raise BuildError(f"{name}: built shape {array.shape}, features_v1 {want}")
    return arrays, nav_valid, progress, contact_counts


def _index_arrays(
    session: Any, arrays: dict[str, npt.NDArray[Any]], steps: npt.NDArray[np.int64]
) -> tuple[dict[str, Any] | None, str]:
    """Adds slot_id, step and decision_index to arrays; returns the image source (or None and the reason). Every
    obs slot must have its derivative when the session has derivatives."""
    decisions: list[dict[str, Any]] = list(session.decisions)
    slot_id = np.array([int(d["obs_slot_id"]) if d["obs_valid"] == 1 else -1 for d in decisions], dtype=np.int64)
    image, reason = _image_source(session)
    if image is not None:
        deriv = Path(image["deriv_dir"])
        absent = [int(s) for s in slot_id if s >= 0 and not (deriv / _image_name(int(s))).is_file()]
        if absent:
            raise BuildError(f"{session.path}: {len(absent)} obs image derivative(s) missing, first slot {absent[0]}")
    arrays["slot_id"] = slot_id
    arrays["step"] = steps
    columns = _decision_index_columns(session)
    arrays["decision_index"] = np.array([[_float(d[c]) for c in columns] for d in decisions], dtype=np.float64)
    return image, reason


def _decision_index_columns(session: Any) -> list[str]:
    """The streams JSON derived.decision_index field names (decision_index.npy columns)."""
    return [str(f["name"]) for f in session.schema.raw["derived"]["decision_index"]["fields"]]


def _meta(
    session: Any,
    spec: FeatureSpec,
    tl: _Timeline,
    *,
    episodes: dict[int, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    segments: list[dict[str, int]],
) -> dict[str, Any]:
    """meta.json without image, missing, excluded, channels and arrays."""
    splits = sorted({r["split"] for r in routes.values()})
    if len(splits) > 1:
        raise BuildError(f"{session.path}: episodes on routes of splits {splits}; a shard holds one split")
    return {
        "schema": SHARD_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "reader_version": str(reader_module().READER_VERSION),
        "session_id": str(session.session_id),
        "features_version": spec.features_version,
        "contract_version": spec.contract_version,
        "build_key": session.build_key(),
        "pin_id": session.meta_get("pin.pin_id"),
        "map_id": str(session.map_id),
        "trial_id": session.trial_id,
        "split": splits[0] if splits else None,
        "decision_hz": tl.decision_hz,
        "physics_steps_per_decision": tl.n_exec,
        "step_dt_s": float(session.clocks["step_dt_s"]),
        "n_steps": tl.n_steps,
        "future_pose_horizon": tl.horizon,
        "hindsight_anchor_steps": [round(s * tl.decision_hz) for s in ANCHOR_S],
        "episodes": [
            {k: e[k] for k in ("episode_id", "start", "stop", "mode", "route_id", "split", "termination_reason")}
            for e in episodes.values()
        ],
        "segments": segments,
        "decision_index_columns": _decision_index_columns(session),
        "categories": {"contact_class": _contact_categories(session)},
        "counts": {
            "decision_grid_points_skipped": int(session.decision_grid_skipped),
            "segment_steps": sum(s["stop"] - s["start"] for s in segments),
        },
        "source": {
            "session_dir": str(Path(str(session.path)).resolve()),
            "features_v1_sha256": spec.sha256,
            "streams_sha256": str(session.schema.sha256),
            "routes": {
                rid: {"sha256": r["sha256"], "file": r["file"], "logged_route_sha256": r["logged"]}
                for rid, r in sorted(routes.items())
            },
            "vehicle_card_id": session.meta_get("control.vehicle_card_id"),
            "vehicle_card_sha": session.meta_get("control.vehicle_card_sha"),
        },
    }


def _int_or(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _rows(decisions: Sequence[dict[str, Any]], key: str, width: int) -> npt.NDArray[np.float32]:
    """[T, width] of a per-decision list (None, or a list with None entries) -> float32, NaN for None."""
    out = np.full((len(decisions), width), np.nan, dtype=np.float32)
    for t, d in enumerate(decisions):
        values = d[key]
        if values is None:
            continue
        if len(values) != width:
            raise BuildError(f"decision step {d['decision_step']}: {key} has {len(values)} values, not {width}")
        out[t] = [_float(v) for v in values]
    return out


def _exec(decisions: Sequence[dict[str, Any]], n_exec: int) -> npt.NDArray[np.float32]:
    """action_exec [T, N, 4]: the reader's per-step entries; NaN past P_t+1 (empty entries)."""
    out = np.full((len(decisions), n_exec, 4), np.nan, dtype=np.float32)
    for t, d in enumerate(decisions):
        rows = d["action_exec"]
        if len(rows) != n_exec:
            raise BuildError(f"decision step {d['decision_step']}: {len(rows)} action_exec entries, not {n_exec}")
        for k, row in enumerate(rows):
            if row is not None:
                out[t, k] = [_float(v) for v in row]
    return out


def _episodes(session: Any, episode: npt.NDArray[np.int64]) -> dict[int, dict[str, Any]]:
    """{episode_id: row} of the committed episodes rows with their decision range [start, stop)."""
    ep = session.streams["episodes"]
    out: dict[int, dict[str, Any]] = {}
    for i in range(len(ep)):
        row = ep.row(i)
        e = int(row["episode_id"])
        where = np.flatnonzero(episode == e)
        if where.size == 0:
            continue
        start, stop = int(where[0]), int(where[-1]) + 1
        if stop - start != where.size:
            raise BuildError(f"{session.path}: the decision steps of episode {e} are not contiguous")
        out[e] = {
            "episode_id": e,
            "start": start,
            "stop": stop,
            "mode": row["mode"],
            "route_id": row["route_id"] or None,
            "route_sha256": row["route_sha256"] or None,
            "termination_reason": row["termination_reason"],
            "split": None,
        }
    unknown = sorted({int(e) for e in episode if e >= 0} - set(out))
    if unknown:
        raise BuildError(f"{session.path}: decision steps of episode(s) {unknown} without an episodes row")
    return out


def _routes(session: Any, episodes: dict[int, dict[str, Any]], routes_dir: Path) -> dict[str, dict[str, Any]]:
    """{route_id: {route, sha256, split, file, path, logged}} of the route episodes; checks the map and each
    episode's route_sha256 (``_logged_route_hash``; ``logged`` lists the distinct checks), and sets each episode's
    split from its route."""
    nav = nav_module()
    out: dict[str, dict[str, Any]] = {}
    for e in episodes.values():
        rid = e["route_id"]
        if rid is None:
            continue
        if not ROUTE_ID_RE.match(rid):
            raise BuildError(f"{session.path}: episode {e['episode_id']} route_id {rid!r} is not a route id")
        if rid not in out:
            path = routes_dir / f"{rid}.json"
            if not path.is_file():
                raise BuildError(f"{session.path}: no route file {path} for episode {e['episode_id']}")
            route = nav.load_route(str(path), sealed_eval=False)
            if route.map_id != session.map_id:
                raise BuildError(f"route {rid} is on map {route.map_id}, the session on {session.map_id}")
            out[rid] = {
                "route": route,
                "sha256": str(route.sha256),
                "split": str(route.split),
                "file": _repo_relative(path),
                "path": path,
                "logged": [],
            }
        logged = e["route_sha256"]
        if logged is not None:
            check = _logged_route_hash(logged, out[rid]["route"], out[rid]["path"])
            if check is None:
                raise BuildError(
                    f"{session.path}: episode {e['episode_id']} names route sha256 {logged[:12]}, "
                    f"{out[rid]['file']} has {out[rid]['sha256'][:12]}"
                )
            if check not in out[rid]["logged"]:
                out[rid]["logged"].append(check)
        e["split"] = out[rid]["split"]
    return out


def _repo_relative(path: Path) -> str:
    """A path inside the repository as repo-relative text (stable across checkouts), else absolute."""
    real = path.resolve()
    return str(real.relative_to(repo_root())) if real.is_relative_to(repo_root()) else str(real)


def _logged_route_hash(logged: str, route: Any, path: Path) -> dict[str, str] | None:
    """How an episode's route_sha256 identifies the route file, as ``{"logged", "form", "verified_against"}``, or
    None when it names another route.

    - ``exact``: the logged value is the file's full route_sha256 (drivedit_nav Route.sha256).
    - ``prefix_8191``: the DD-7 defect (``PREFIX_HASH_ROUTES``). Accepted only when the route is allow-listed, the
      current file's full route_sha256 equals the listed deployed version (so any edit anywhere in the route, its
      split included, is refused), and the logged value is the sha256 of that file's first 8191 bytes, the bytes the
      engine hashed. ``verified_against`` is the full hash the file was checked against.
    """
    if logged == route.sha256:
        return {"logged": logged, "form": "exact", "verified_against": str(route.sha256)}
    deployed = PREFIX_HASH_ROUTES.get(str(route.route_id))
    if deployed is None or route.sha256 != deployed:
        return None
    raw = path.read_bytes()
    if len(raw) > SUBSTRING_MAX_CHARS and logged == hashlib.sha256(raw[:SUBSTRING_MAX_CHARS]).hexdigest():
        return {"logged": logged, "form": "prefix_8191", "verified_against": deployed}
    return None


def _poses(session: Any, steps: npt.NDArray[np.int64]) -> FloatArray:
    """[T, 3] (pos_x, pos_z, heading_rad) of the telemetry row at P_t; NaN without one."""
    tel = session.streams["telemetry"]
    index = tel.index()
    heading = session.telemetry_export["heading_rad"]
    out = np.full((len(steps), 3), np.nan, dtype=np.float64)
    for t, p in enumerate(steps):
        i = index.get(int(p))
        if i is not None:
            out[t] = [_float(tel.data["pos_x"][i]), _float(tel.data["pos_z"][i]), float(heading[i])]
    return out


def _teleport_steps(session: Any) -> list[int]:
    """Sorted physics steps of the teleport and respawn rows of the ego or of no vehicle (the reader's rule)."""
    slots = set(reader_module().ego_slots(session)) | {-1}
    ev = session.streams["events"]
    return sorted(
        int(ev.data["physics_step"][i])
        for i in range(len(ev))
        if ev.data["kind"][i] in ("teleport", "respawn") and ev.data["veh_slot_id"][i] in slots
    )


def episode_nav(route: Any, rows: Sequence[tuple[int, tuple[float, float, float]]]) -> dict[int, dict[str, Any]]:
    """nav_v1 over one episode's telemetry rows ``(physics_step, (pos_x, pos_z, heading_rad))`` in row order, the
    progress carried from row to row and starting at 0 m on the first row: ``{physics_step: nav_v1 result}``.

    This is the cadence of the contract's nav view (dc-2.0 section 9.8 and streams JSON ``derived.nav``: one row per
    telemetry row of an episode with a route, the harvester's nav_fill). Ordered progress never decreases, so a pose
    between two decision callbacks can move it: evaluating only the decision poses would give other nav values."""
    nav = nav_module()
    return {step: r for (step, _), r in zip(rows, nav.nav_sequence(route, [p for _, p in rows]), strict=True)}


def _nav(
    spec: FeatureSpec,
    session: Any,
    tl: _Timeline,
    episodes: dict[int, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8], FloatArray]:
    """(nav [T,10], nav_valid [T] (1, 0, or -1 outside route episodes), progress_m [T]): ``episode_nav`` over every
    telemetry row of each route episode, read at the decision steps (the telemetry row of C_t, P_t). A decision step
    of a route episode without its telemetry row has no nav (nav_valid 0)."""
    nav = nav_module()
    names = spec.tensor("nav").channel_names
    if tuple(nav.NAV_CHANNELS) != names:
        raise BuildError(f"drivedit_nav NAV_CHANNELS {nav.NAV_CHANNELS} differ from features_v1 nav {names}")
    tel = session.streams["telemetry"]
    heading = session.telemetry_export["heading_rad"]
    out = np.full((tl.n_steps, len(names)), np.nan, dtype=np.float32)
    valid = np.full(tl.n_steps, -1, dtype=np.int8)
    progress = np.full(tl.n_steps, np.nan, dtype=np.float64)
    for e in episodes.values():
        if e["route_id"] is None:
            continue
        rows = []
        for i in range(len(tel)):
            if tel.data["episode_id"][i] != e["episode_id"]:
                continue
            pose = (_float(tel.data["pos_x"][i]), _float(tel.data["pos_z"][i]), float(heading[i]))
            if not all(math.isfinite(v) for v in pose):
                raise BuildError(f"{session.path}: telemetry row {tel.data['physics_step'][i]} has no finite pose")
            rows.append((int(tel.data["physics_step"][i]), pose))
        results = episode_nav(routes[e["route_id"]]["route"], rows)
        valid[e["start"] : e["stop"]] = 0
        for t in range(e["start"], e["stop"]):
            r = results.get(int(tl.steps[t]))
            if r is not None:
                out[t] = nav.nav_vector(r)
                valid[t] = int(r["nav_valid"])
                progress[t] = float(r["progress_m"])
    return out, valid, progress


def _style(spec: FeatureSpec, session: Any, episode: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """style [T,3]: the episode's style_v1 values (episodes.style_*) on its decision steps."""
    names = spec.tensor("style").channel_names
    sources = [c.source.split(".", 1)[1] for c in spec.tensor("style").channels]  # episodes.style_*
    ep = session.streams["episodes"]
    out = np.full((len(episode), len(names)), np.nan, dtype=np.float32)
    for i in range(len(ep)):
        e = int(ep.data["episode_id"][i])
        out[episode == e] = [_float(ep.data[col][i]) for col in sources]
    return out


def _masks(
    spec: FeatureSpec, decisions: Sequence[dict[str, Any]], nav_valid: npt.NDArray[np.int8]
) -> npt.NDArray[np.uint8]:
    """masks [T,5] uint8: the reader's masks by features_v1 channel name; nav_valid 0 clears policy_valid and
    wm_valid."""
    names = spec.tensor("masks").channel_names
    out = np.array([[int(d["masks"][n]) for n in names] for d in decisions], dtype=np.uint8)
    off = nav_valid == 0
    out[off, names.index("policy_valid")] = 0
    out[off, names.index("wm_valid")] = 0
    return out


def _segments(
    session: Any, masks: npt.NDArray[np.uint8], spec: FeatureSpec, steps: npt.NDArray[np.int64]
) -> list[dict[str, int]]:
    """The reader's segments (maximal runs of wm-valid, non-warm-up steps of one episode on consecutive grid
    points) over the final masks, as decision ranges [start, stop)."""
    reader = reader_module()
    decisions: list[dict[str, Any]] = list(session.decisions)
    names = spec.tensor("masks").channel_names
    wm = names.index("wm_valid")
    masked = [dict(d, masks=dict(d["masks"], wm_valid=int(masks[t, wm]))) for t, d in enumerate(decisions)]
    position = {int(p): t for t, p in enumerate(steps)}
    out = []
    for seg in reader.segments_of(masked):
        start = position[int(seg["first_decision_step"])]
        stop = position[int(seg["last_decision_step"])] + 1
        if stop - start != int(seg["n_steps"]):
            raise BuildError(f"segment at step {seg['first_decision_step']} is not a contiguous decision range")
        out.append({"episode_id": int(seg["episode_id"]), "start": start, "stop": stop})
    return out


def _ego_deltas(tl: _Timeline, offsets: Sequence[int]) -> FloatArray:
    """[T, len(offsets), 3] ego-frame (forward, right, heading change) of C_t+k seen from C_t; NaN unless both are of
    one episode, k grid points apart, with poses, and no ego teleport or respawn lies in (P_t, P_t+k]."""
    nav = nav_module()
    poses, episode, grid, steps, tele_steps = tl.poses, tl.episode, tl.grid, tl.steps, tl.tele_steps
    n_steps = tl.n_steps
    out = np.full((n_steps, len(offsets), 3), np.nan, dtype=np.float64)
    for t in range(n_steps):
        if episode[t] < 0 or not np.isfinite(poses[t]).all():
            continue
        x, z, h = (float(v) for v in poses[t])
        origin = nav.Pose(x, z, h)
        for j, k in enumerate(offsets):
            u = t + k
            if u >= n_steps or episode[u] != episode[t] or grid[u] - grid[t] != k or not np.isfinite(poses[u]).all():
                continue
            first = bisect.bisect_right(tele_steps, int(steps[t]))
            if first < len(tele_steps) and tele_steps[first] <= int(steps[u]):
                continue
            fwd, right = nav.to_ego(origin, float(poses[u, 0]), float(poses[u, 1]))
            out[t, j] = [fwd, right, _wrap_pi(float(poses[u, 2]) - h)]
    return out


def _wrap_pi(angle: float) -> float:
    """angle wrapped to (-pi, pi]."""
    a = math.remainder(angle, 2.0 * math.pi)
    return math.pi if a == -math.pi else a


def _contact_categories(session: Any) -> list[str]:
    """The collision payload 'other' enum of the streams JSON (contact_class indices)."""
    for key in session.schema.payloads["collision"]:
        if key["key"] == "other":
            return [str(v) for v in key["enum"]]
    raise BuildError("the streams JSON collision payload has no 'other' key")


def _contacts(
    session: Any, steps: npt.NDArray[np.int64], categories: Sequence[str]
) -> tuple[npt.NDArray[np.float32], dict[str, int]]:
    """contact_events [T,2] and counts: ego collision rows by onset, in the decision interval (P_t, P_t+1]."""
    out = np.zeros((len(steps), 2), dtype=np.float32)
    out[:, 1] = -1.0
    rows = [c for c in session.crash_events if c["kind"] == "collision" and c["ego"]]
    bad = [c for c in rows if c["problems"]]
    if bad:
        raise BuildError(
            f"{session.path}: ego collision row {bad[0]['event_seq']} has payload problems {bad[0]['problems']} (C36)"
        )
    outside = 0
    for c in sorted(rows, key=lambda c: (int(c["step"]), int(c["event_seq"]))):
        t = bisect.bisect_left(steps.tolist(), int(c["step"])) - 1
        if t < 0 or t + 1 >= len(steps):
            outside += 1
            continue
        if out[t, 0] == 0.0:
            out[t, 1] = float(list(categories).index(str(c["values"]["other"])))
        out[t, 0] = 1.0
    return out, {"ego_collision_rows": len(rows), "ego_collision_rows_outside_decision_intervals": outside}


def _decision_index(decisions: Sequence[dict[str, Any]], columns: Sequence[str]) -> FloatArray:
    """decision_index [T, len(columns)] float64 (integers, exact; NaN = empty)."""
    return np.array([[_float(d[c]) for c in columns] for d in decisions], dtype=np.float64)


def _eval_only(session: Any, steps: npt.NDArray[np.int64], progress: FloatArray) -> npt.NDArray[np.float32]:
    """eval_only [T, 8] float32 in EVAL_ONLY_BUILT order: the absolute pose, world heading and world velocity
    (vel_local rotated by the logged right/up/fwd axes) of the telemetry row at P_t, and nav progress_m."""
    tel = session.streams["telemetry"]
    index = tel.index()
    heading = session.telemetry_export["heading_rad"]
    out = np.full((len(steps), len(EVAL_ONLY_BUILT)), np.nan, dtype=np.float64)
    for t, p in enumerate(steps):
        i = index.get(int(p))
        if i is None:
            continue
        col = tel.data
        axes = np.array(
            [[_float(col[f"{a}_{c}"][i]) for c in "xyz"] for a in ("right", "up", "fwd")],
            dtype=np.float64,
        )
        vel_local = np.array([_float(col[f"vel_local_{c}"][i]) for c in "xyz"], dtype=np.float64)
        vel_world = vel_local @ axes
        out[t, :7] = [
            _float(col["pos_x"][i]),
            _float(col["pos_y"][i]),
            _float(col["pos_z"][i]),
            float(heading[i]),
            *vel_world.tolist(),
        ]
    out[:, 7] = progress
    return out.astype(np.float32)


def _image_name(slot: int) -> str:
    return IMAGE_PATTERN.format(slot=slot)


def _image_source(session: Any) -> tuple[dict[str, Any] | None, str]:
    """(meta.json image block, '') when the harvest derivatives exist, else (None, the reason)."""
    deriv = Path(str(session.path)) / DERIV_DIR
    if not deriv.is_dir():
        return None, f"no harvest derivatives ({DERIV_DIR}) in the session: slot_id is written, the images are not"
    block: dict[str, Any] = {"deriv_dir": str(deriv.resolve()), "pattern": IMAGE_PATTERN}
    result = Path(str(session.path)) / "_harvest" / "result.json"
    if result.is_file():
        derivative = json.loads(result.read_text("utf-8")).get("derivative")
        if isinstance(derivative, dict):
            block["derivative"] = derivative
    return block, ""


def _nonfinite_in_segments(arrays: dict[str, npt.NDArray[Any]], segments: Sequence[dict[str, int]]) -> dict[str, int]:
    """Decision steps inside segments whose world-model conditioning (action_cmd, state) is not finite."""
    out = {}
    for name in ("action_cmd", "state"):
        a = arrays[name]
        out[name] = sum(int((~np.isfinite(a[s["start"] : s["stop"]]).all(axis=1)).sum()) for s in segments)
    return out


def _shard_dir(out_root: Path, session_real: Path, session_id: str) -> Path:
    """``<out_root>/<session_id>/features_v1`` under its resolved parent, checked before anything is written.

    Every path the build creates, renames or removes lies in that parent (the staging directories ``.features_v1.*``
    are siblings of the shard), so the checks run on resolved paths, symlinks followed: the out root, the parent
    ``<out_root>/<session_id>`` and the destination must each lie outside the C: drive, the session directory and its
    captures tree. A symlinked destination directory is refused: the shard is replaced by renames inside its parent.
    """
    if not session_id or "/" in session_id or session_id in (".", ".."):
        raise BuildError(f"session id {session_id!r} is not a directory name")
    root = out_root.resolve()
    parent = (root / session_id).resolve()
    dest = parent / SHARD_DIRNAME
    if dest.is_symlink():
        raise BuildError(f"refusing {dest}: the shard directory is a symlink")
    protected = [session_real] + [p for p in session_real.parents if p.name == "captures"]
    for path in (root, parent, dest.resolve()):
        if any((str(path) + "/").startswith(r) for r in FORBIDDEN_ROOTS):
            raise BuildError(f"refusing {out_root}: {path} is on the C: drive (no writes to C:)")
        for p in protected:
            if path == p or path.is_relative_to(p):
                raise BuildError(f"refusing {out_root}: {path} lies inside {p} (sessions and captures are read-only)")
    return dest


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def summary(shard_dir: Path) -> dict[str, Any]:
    """The meta.json fields a build report prints."""
    meta = json.loads((shard_dir / META_NAME).read_text("utf-8"))
    keys = ("session_id", "map_id", "split", "build_key", "decision_hz", "n_steps", "future_pose_horizon")
    out: dict[str, Any] = {k: meta[k] for k in keys}
    out["episodes"] = len(meta["episodes"])
    out["segments"] = len(meta["segments"])
    out["arrays"] = sorted(meta["arrays"])
    out["missing"] = sorted(meta["missing"])
    out["counts"] = meta["counts"]
    out["shard"] = str(shard_dir)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m drivedit.data.build", description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", action="append", required=True, help="dc-2.0 session directory; repeatable")
    ap.add_argument("--out", required=True, help="dataset root: writes <out>/<session_id>/features_v1/")
    ap.add_argument("--routes-dir", default=None, help=f"route files (default {default_routes_dir()})")
    ap.add_argument("--vehicle-card", default=None, help="vehicle card JSON (default: the reader's card lookup)")
    ap.add_argument("--include-eval-only", action="store_true", help="also write eval_only.npy")
    ap.add_argument("--include-privileged", action="store_true", help="also build privileged_route (teacher_input)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing shard")
    args = ap.parse_args(argv)
    spec = load_spec()
    for session_dir in args.session:
        shard = build_session(
            session_dir,
            args.out,
            spec=spec,
            include_privileged=args.include_privileged,
            include_eval_only=args.include_eval_only,
            routes_dir=args.routes_dir,
            vehicle_card=args.vehicle_card,
            overwrite=args.overwrite,
        )
        sys.stdout.write(json.dumps(summary(shard), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
