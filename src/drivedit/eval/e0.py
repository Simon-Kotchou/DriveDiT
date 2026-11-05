"""E0: does ``action_cmd`` improve next-step (Δ yaw rate, Δ speed) prediction over ``state`` alone? (numpy only)

The first rung of the evaluation ladder (MODELING_PLAN §5 E0; research delta-2026-11 §3.3 "A day-1 data check" and
§6 R0). A world model learns only the action responses its data contains (P1; AIM Thm 3.1: the dataset's
I(a; s' | s) bounds any model's controllability), so before any GPU time we ask the telemetry whether the commands
carry information about the next state that the current state does not.

Two closed-form ridge regressions predict the next-step change of each target from the decision step's features:

- **state**: ``state`` at t (features_v1 ``[speed_mps, yaw_rate_rps, acc_long_achieved_mps2, dir_state]``) and at the
  ``history`` earlier steps of the same segment;
- **state + action**: the same plus ``action_cmd`` at t (``[kappa_target, a_target, dir_cmd]``) and the kinematic
  commanded yaw rate ``speed_mps x kappa_target`` (ψ̇ ≈ v·κ, delta §5.2 M3; a linear model cannot form the product).

Targets: ``d_yaw_rate_rps = yaw_rate_rps[t+1] - yaw_rate_rps[t]`` and ``d_speed_mps = speed_mps[t+1] - speed_mps[t]``.
The contract offsets make this a clean test: ``state[t]`` is the telemetry of decision callback C_t, which holds no
step of ``action_cmd[t]``; the command applies over (P_t, P_t+1], and ``state[t+1]`` is the state after it
(features_v1 ``offsets``, DRIVEDIT_INTERFACES §1).

**History.** The baseline should see what the world model sees besides the action: its context window of 8 frames
(MODELING_PLAN §3, window 8-10), so ``history`` defaults to 7 (``state[t-7..t]``; delta §3.3 also specifies "state
history"). It matters: in an expert session the command is the expert's response to its own state trajectory, and one
state sample hides the trend. Measured on the first T0L session (2026-11-05): with ``history=0`` adding ``a_target``
gains 0.43 in Δ speed R², with any history from 1 to 9 it gains 0.06-0.08 and loses in one block, while the Δ yaw-rate
gain stays 0.59-0.64 throughout.

Out-of-fold R² comes from cross-validation that never crosses a segment (``drivedit.eval.linear``): one fold per
session when there are several, else purged blocked folds. The **gain** is R²(state + action) - R²(state). It is
**shown** only when it is positive in every fold. Separately, **can_show** says whether the data could show a gain at
all: not when the target is constant, when every command channel that drives it is constant (constant speed with a
constant ``a_target``), or when those channels are numerically a function of the state features. Per-speed-bin results
(``drivedit.eval.strata``) follow delta §6 R0's gate, "positive for κ *and* a in every speed bin"; inside a bin R² is
relative to that bin's spread, so read it with the bin's ``target_std`` and RMSEs.

CLI (reads dc-2.0 sessions through ``data/dc2/reader.py``, loaded by file path as the repo's tools load it; prints the
report as JSON on stdout and writes nothing):

    uv run python -m drivedit.eval.e0 --session DIR [--session DIR ...] [--vehicle-card CARD] [--history H]
        [--folds K] [--lam L]

Rows are the transitions (t, t+1) inside the reader's segments (wm-valid, non-warm-up decision steps of one episode
on consecutive grid points). Two or more sessions are cross-validated by held-out session. The reader refuses Everon
(sealed) and dc-1.0 sessions, and this CLI never asks it for the sealed path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from drivedit.eval.linear import (
    FloatArray,
    IntArray,
    Segment,
    as_matrix,
    check_segments,
    compact_segments,
    constant_columns,
    cross_val_predict,
    r2_score,
    segment_folds,
)
from drivedit.eval.strata import SPEED_BIN_NAMES, speed_bin

# features_v1 channel names (docs/capture_contract/features_v1.json tensors state and action_cmd), verbatim.
STATE_CHANNELS = ("speed_mps", "yaw_rate_rps", "acc_long_achieved_mps2", "dir_state")
ACTION_CHANNELS = ("kappa_target", "a_target", "dir_cmd")
ACTION_FEATURES = (*ACTION_CHANNELS, "speed_mps*kappa_target")
TARGETS = ("d_yaw_rate_rps", "d_speed_mps")
# The command channels that drive each target: curvature through llc_v1 steering, acceleration and direction through
# its throttle and brake (DC_LLCv1.c header).
TARGET_ACTIONS = {"d_yaw_rate_rps": ("kappa_target",), "d_speed_mps": ("a_target", "dir_cmd")}
_SPEED, _YAW = 0, 1
_KAPPA = 0
# An action channel predicted from state with an out-of-fold R² at or above this is a function of the state up to
# numerical error: the regression cannot separate its effect (a tolerance, not a modelling threshold).
_DETERMINED_R2 = 1.0 - 1e-6


@dataclass(frozen=True)
class ActionDiagnostic:
    """One command channel that drives a target: its spread, and how well the state features predict it."""

    channel: str
    std: float
    constant: bool
    r2_from_state: float


@dataclass(frozen=True)
class SpeedBinResult:
    """Both regressions' out-of-fold fit inside one speed bin: R² relative to the bin's own spread (NaN when the bin
    has < 2 rows or a constant target), and RMSE in the target's unit (NaN for an empty bin)."""

    speed_bin: str
    n: int
    target_std: float
    rmse_state: float
    rmse_state_action: float
    r2_state: float
    r2_state_action: float
    gain: float


@dataclass(frozen=True)
class TargetResult:
    """E0 for one target. ``gain`` is pooled over all out-of-fold rows; ``fold_gains`` per fold."""

    target: str
    target_std: float
    r2_state: float
    r2_state_action: float
    gain: float
    fold_gains: tuple[float, ...]
    can_show: bool
    shown: bool
    reasons: tuple[str, ...]
    actions: tuple[ActionDiagnostic, ...]
    speed_bins: tuple[SpeedBinResult, ...]


@dataclass(frozen=True)
class E0Report:
    """The E0 result: one ``TargetResult`` per target in ``TARGETS``, and how it was computed."""

    n_rows: int
    n_segments: int
    cv: str
    n_folds: int
    gap: int
    history: int
    lam: float
    state_features: tuple[str, ...]
    action_features: tuple[str, ...]
    targets: tuple[TargetResult, ...]

    def target(self, name: str) -> TargetResult:
        for result in self.targets:
            if result.target == name:
                return result
        raise KeyError(name)

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict; NaN becomes null."""
        out: dict[str, Any] = _jsonable(asdict(self))
        return out


def e0_action_gain(
    state: npt.ArrayLike,
    action_cmd: npt.ArrayLike,
    next_state: npt.ArrayLike,
    *,
    segments: Sequence[Segment],
    groups: Sequence[Hashable] | None = None,
    history: int = 7,
    n_folds: int = 5,
    lam: float = 1.0,
) -> E0Report:
    """E0 over transition rows: ``state`` ``[N, 4]``, ``action_cmd`` ``[N, 3]`` and ``next_state`` ``[N, 4]`` (the
    state at the next decision step of the same segment), all features_v1 channels in contract order.

    ``segments`` are ``[start, stop)`` row ranges of consecutive decision steps; rows outside them are not used, and
    the first ``history`` rows of each segment only serve as history. ``groups`` labels each segment with its session:
    with two or more sessions the folds hold out one session each, else ``n_folds`` purged blocks. ``lam`` is the
    ridge penalty on standardised coefficients (a numerical stabiliser at thousands of rows, not a tuned
    regulariser).
    """
    st = _channels(state, "state", len(STATE_CHANNELS))
    ac = _channels(action_cmd, "action_cmd", len(ACTION_CHANNELS))
    nx = _channels(next_state, "next_state", len(STATE_CHANNELS))
    n_all = st.shape[0]
    if ac.shape[0] != n_all or nx.shape[0] != n_all:
        raise ValueError(f"state, action_cmd and next_state have {n_all}, {ac.shape[0]} and {nx.shape[0]} rows")
    if history < 0:
        raise ValueError(f"history must be >= 0, got {history}")
    segs = check_segments(segments, n_all)
    # rows that have their full history inside the segment; the segments shrink by `history` at the start
    usable = [(start + history, stop) for start, stop in segs if stop - start > history]
    kept_groups = None if groups is None else [g for g, (s, e) in zip(groups, segs, strict=True) if e - s > history]
    if not usable:
        raise ValueError(f"no segment is longer than history={history}")
    rows, compact = compact_segments(usable)
    gap = history + 1  # features span t-history .. t, the target t+1: no shared raw sample across the purge
    folds = segment_folds(compact, n_folds=n_folds, gap=gap, groups=kept_groups)

    x_state = np.concatenate([st[rows - lag] for lag in range(history, -1, -1)], axis=1)
    x_action = np.concatenate([ac[rows], (st[rows, _SPEED] * ac[rows, _KAPPA])[:, None]], axis=1)
    x_full = np.concatenate([x_state, x_action], axis=1)
    y = np.stack([nx[rows, _YAW] - st[rows, _YAW], nx[rows, _SPEED] - st[rows, _SPEED]], axis=1)

    # clipped to the training range: bounded physical channels with rare contact spikes (linear.fit_ridge)
    pred_state = cross_val_predict(x_state, y, folds, lam=lam, clip=True)
    pred_full = cross_val_predict(x_full, y, folds, lam=lam, clip=True)
    r2_state = r2_score(y, pred_state)
    r2_full = r2_score(y, pred_full)
    fold_r2 = [(r2_score(y[f.test], pred_state[f.test]), r2_score(y[f.test], pred_full[f.test])) for f in folds.folds]
    bins = speed_bin(st[rows, _SPEED])
    action_r2 = r2_score(ac[rows], cross_val_predict(x_state, ac[rows], folds, lam=lam, clip=True))
    action_const = constant_columns(ac[rows])

    results = []
    for k, target in enumerate(TARGETS):
        diags = tuple(
            ActionDiagnostic(
                channel=ch,
                std=float(ac[rows, j].std()),
                constant=bool(action_const[j]),
                r2_from_state=float(action_r2[j]),
            )
            for j, ch in enumerate(ACTION_CHANNELS)
            if ch in TARGET_ACTIONS[target]
        )
        fold_gains = tuple(float(full[k] - base[k]) for base, full in fold_r2)
        gain = float(r2_full[k] - r2_state[k])
        reasons = _cannot_show(target, y[:, k], diags)
        can_show = not reasons
        shown = can_show and math.isfinite(gain) and gain > 0.0 and all(g > 0.0 for g in fold_gains)
        results.append(
            TargetResult(
                target=target,
                target_std=float(y[:, k].std()),
                r2_state=float(r2_state[k]),
                r2_state_action=float(r2_full[k]),
                gain=gain,
                fold_gains=fold_gains,
                can_show=can_show,
                shown=shown,
                reasons=reasons,
                actions=diags,
                speed_bins=_speed_bins(y[:, k], pred_state[:, k], pred_full[:, k], bins),
            )
        )
    lag_names = [f"{ch}[t-{lag}]" if lag else f"{ch}[t]" for lag in range(history, -1, -1) for ch in STATE_CHANNELS]
    return E0Report(
        n_rows=int(rows.size),
        n_segments=len(usable),
        cv=folds.kind,
        n_folds=len(folds.folds),
        gap=folds.gap,
        history=history,
        lam=lam,
        state_features=tuple(lag_names),
        action_features=ACTION_FEATURES,
        targets=tuple(results),
    )


def _channels(a: npt.ArrayLike, name: str, width: int) -> FloatArray:
    m = as_matrix(a, name)
    if m.shape[1] != width:
        raise ValueError(f"{name} must be [N, {width}], got {m.shape}")
    return m


def _cannot_show(target: str, y: FloatArray, diags: tuple[ActionDiagnostic, ...]) -> tuple[str, ...]:
    """Why the data cannot show an action gain for ``target`` (empty when it can)."""
    if constant_columns(y[:, None])[0]:
        return (f"{target} is constant: there is no next-step change to explain",)
    varying = [d for d in diags if not d.constant]
    if not varying:
        names = ", ".join(d.channel for d in diags)
        return (f"{names} constant in every row: the data holds no command variation that drives {target}",)
    if all(d.r2_from_state >= _DETERMINED_R2 for d in varying):
        names = ", ".join(d.channel for d in varying)
        why = f"a function of the state features (R² >= {_DETERMINED_R2}), so its effect cannot be separated"
        return (f"{names}: {why}",)
    return ()


def _speed_bins(
    y: FloatArray, pred_state: FloatArray, pred_full: FloatArray, bins: IntArray
) -> tuple[SpeedBinResult, ...]:
    out = []
    for b, name in enumerate(SPEED_BIN_NAMES):
        sel = bins == b
        n = int(sel.sum())
        base = float(r2_score(y[sel], pred_state[sel])[0])
        full = float(r2_score(y[sel], pred_full[sel])[0])
        out.append(
            SpeedBinResult(
                speed_bin=name,
                n=n,
                target_std=float(y[sel].std()) if n else math.nan,
                rmse_state=_rmse(y[sel], pred_state[sel]),
                rmse_state_action=_rmse(y[sel], pred_full[sel]),
                r2_state=base,
                r2_state_action=full,
                gain=full - base,
            )
        )
    return tuple(out)


def _rmse(y: FloatArray, pred: FloatArray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2))) if y.size else math.nan


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


# --------------------------------------------------------------------------------------------------------------------
# dc-2.0 sessions -> transition rows (the CLI)
# --------------------------------------------------------------------------------------------------------------------
READER_PATH = Path(__file__).resolve().parents[3] / "data" / "dc2" / "reader.py"


@dataclass(frozen=True)
class SessionTransitions:
    """The (t, t+1) transitions inside one session's segments: ``state`` ``[N, 4]``, ``action_cmd`` ``[N, 3]``,
    ``next_state`` ``[N, 4]`` (features_v1 channels), and their ``[start, stop)`` row ranges, one per segment."""

    session_id: str
    state: FloatArray
    action_cmd: FloatArray
    next_state: FloatArray
    segments: tuple[Segment, ...]


def load_reader() -> ModuleType:
    """``data/dc2/reader.py`` of the source tree, loaded by file path the way the repo's tools load it."""
    if not READER_PATH.is_file():
        raise FileNotFoundError(
            f"{READER_PATH} not found: the E0 CLI reads sessions through the repo's dc-2.0 reader, so run it from a "
            "source checkout"
        )
    spec = importlib.util.spec_from_file_location("dc2_reader", READER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {READER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_transitions(session_dir: str | Path, *, vehicle_card: str | Path | None = None) -> SessionTransitions:
    """Read one dc-2.0 session (strict, images decoded: segments need valid observations) into transition rows.

    ``vehicle_card`` overrides the reader's default card lookup (``tools/ident/cards/<pin_id>/<card_id>.json``), which
    ``dir_state`` needs. Raises when a segment step lacks a state or a command.
    """
    reader = load_reader()
    card = None if vehicle_card is None else str(vehicle_card)
    session = reader.open_session(str(session_dir), vehicle_card=card)
    decisions: list[dict[str, Any]] = list(session.decisions)
    index = {int(d["decision_step"]): i for i, d in enumerate(decisions)}
    state: list[list[float]] = []
    action: list[list[float]] = []
    nxt: list[list[float]] = []
    segments: list[Segment] = []
    for seg in session.segments:
        first, last = index[int(seg["first_decision_step"])], index[int(seg["last_decision_step"])]
        start = len(state)
        for t in range(first, last):
            state.append(_values(decisions[t], "state", 4))
            action.append(_values(decisions[t], "action_cmd", 3))
            nxt.append(_values(decisions[t + 1], "state", 4))
        if len(state) > start:
            segments.append((start, len(state)))
    if not segments:
        raise ValueError(f"{session_dir}: no segment has two consecutive decision steps")
    return SessionTransitions(
        session_id=str(session.session_id),
        state=np.asarray(state, dtype=np.float64),
        action_cmd=np.asarray(action, dtype=np.float64),
        next_state=np.asarray(nxt, dtype=np.float64),
        segments=tuple(segments),
    )


def _values(decision: dict[str, Any], key: str, width: int) -> list[float]:
    values = decision[key]
    if values is None or len(values) != width or any(v is None for v in values):
        hint = " (dir_state needs the vehicle card: pass --vehicle-card)" if key == "state" else ""
        raise ValueError(f"decision step {decision['decision_step']}: {key} is {values!r}{hint}")
    return [float(v) for v in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m drivedit.eval.e0", description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", action="append", required=True, help="dc-2.0 session directory (repeatable)")
    parser.add_argument("--vehicle-card", help="vehicle card JSON (default: the reader's lookup by pin and card id)")
    parser.add_argument("--history", type=int, default=7, help="earlier state steps in the features (default 7)")
    parser.add_argument("--folds", type=int, default=5, help="blocked folds with one session (default 5)")
    parser.add_argument("--lam", type=float, default=1.0, help="ridge penalty on standardised coefficients")
    args = parser.parse_args(argv)
    parts = [session_transitions(path, vehicle_card=args.vehicle_card) for path in args.session]
    offsets = np.cumsum([0] + [p.state.shape[0] for p in parts[:-1]])
    segments = [(s + int(o), e + int(o)) for p, o in zip(parts, offsets, strict=True) for s, e in p.segments]
    groups = [p.session_id for p in parts for _ in p.segments]
    report = e0_action_gain(
        np.concatenate([p.state for p in parts]),
        np.concatenate([p.action_cmd for p in parts]),
        np.concatenate([p.next_state for p in parts]),
        segments=segments,
        groups=groups,
        history=args.history,
        n_folds=args.folds,
        lam=args.lam,
    )
    out = {"sessions": [p.session_id for p in parts], **report.to_json()}
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
