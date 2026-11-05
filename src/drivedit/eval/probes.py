"""Probes of what a world model represents and uses: linear state probes and the explained action information
(numpy only).

- ``state_probe_r2``: a linear ridge probe from features (a world-model layer's activations, latents, any ``[N, D]``
  per decision step) to engine state, scored by blocked cross-validation that never crosses a segment. Probes recover
  game state with R² 0.97-0.98 in small world models (drive-physics-wfm-2026-11 §6 and TL;DR 7(iii); "Probing world
  models", arXiv 2603.21546), so per-layer probe R² of ``state`` (speed, yaw rate) is a cheap health metric. Privileged
  state grades and never conditions (MODELING_PLAN P5).
- ``explained_action_information``: AIM's explained action information (EAI), the model's prediction loss with the
  action replaced by its learned null minus the loss with the true action (delta-2026-11 §5.1 item 2 and §5.2 M1).
  EAI ≈ 0 means the model ignores the action, however the video looks. The losses come from the caller: one extra
  forward pass with the null action per evaluation window. The gate is delta §5.2's bar, "EAI > 0 in every speed bin":
  every unit (all steps, and each bucket) must have a positive mean and a positive lower 95% bound, with a Student-t
  critical value over its independent clips, and at least ``min_clusters`` clips; otherwise the gate is "fail" or
  "insufficient", never "pass".
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from drivedit.eval.linear import (
    Segment,
    as_matrix,
    check_segments,
    compact_segments,
    cross_val_predict,
    r2_score,
    segment_folds,
)

# Fewest independent clips for a unit (the whole set, or one bucket) of the EAI gate: gap-4 §8.1 proposes "at least 50
# clips per reported cell; otherwise mark the cell n/a".
MIN_CLUSTERS = 50
_COVERAGE = 0.95


@dataclass(frozen=True)
class StateProbe:
    """Out-of-fold R² of a linear probe per target channel, pooled (``r2``) and per fold (``fold_r2``)."""

    names: tuple[str, ...]
    r2: tuple[float, ...]
    fold_r2: tuple[tuple[float, ...], ...]
    n: int
    cv: str
    gap: int
    lam: float

    def r2_of(self, name: str) -> float:
        return self.r2[self.names.index(name)]


def state_probe_r2(
    features: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    segments: Sequence[Segment],
    names: Sequence[str] | None = None,
    groups: Sequence[Hashable] | None = None,
    n_folds: int = 5,
    gap: int = 0,
    lam: float = 1.0,
) -> StateProbe:
    """Linear ridge probe ``features`` ``[N, D]`` -> ``targets`` ``[N, K]`` (for example features_v1 ``state``
    channels at the same decision step), scored out of fold.

    ``segments`` are the ``[start, stop)`` row ranges to use; ``groups`` labels each segment with its session (two or
    more sessions: held-out-session folds), else ``n_folds`` blocks purged by ``gap`` rows. Set ``gap`` to the number
    of decision steps the features look back (a world model's context window), so no training row shares a frame with
    a test row. Wide features (D > N) are solved in the dual; ``lam`` is the penalty on standardised coefficients.
    """
    x = as_matrix(features, "features")
    y = as_matrix(targets, "targets")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"features have {x.shape[0]} rows, targets {y.shape[0]}")
    segs = check_segments(segments, x.shape[0])
    channel_names = tuple(names) if names is not None else tuple(f"target_{k}" for k in range(y.shape[1]))
    if len(channel_names) != y.shape[1]:
        raise ValueError(f"{len(channel_names)} names for {y.shape[1]} target channels")
    rows, compact = compact_segments(segs)
    folds = segment_folds(compact, n_folds=n_folds, gap=gap, groups=groups)
    xs, ys = x[rows], y[rows]
    pred = cross_val_predict(xs, ys, folds, lam=lam)
    fold_r2 = tuple(tuple(float(v) for v in r2_score(ys[f.test], pred[f.test])) for f in folds.folds)
    return StateProbe(
        names=channel_names,
        r2=tuple(float(v) for v in r2_score(ys, pred)),
        fold_r2=fold_r2,
        n=int(rows.size),
        cv=folds.kind,
        gap=folds.gap,
        lam=lam,
    )


@dataclass(frozen=True)
class EaiUnit:
    """EAI over one unit of the gate (all steps, or one bucket such as a speed bin or a κ cell).

    ``n`` steps in ``n_clusters`` independent clips; ``eai`` the mean, ``se`` its standard error over clips, ``t_crit``
    the two-sided 95% Student-t critical value with ``n_clusters - 1`` degrees of freedom and ``lower_95 = eai - t_crit
    x se``. ``status`` is "insufficient" below the gate's ``min_clusters`` (a required bucket without rows included),
    else "pass" when ``eai > 0`` and ``lower_95 > 0``, else "fail".
    """

    bucket: str
    n: int
    n_clusters: int
    eai: float
    se: float
    t_crit: float
    lower_95: float
    status: str


@dataclass(frozen=True)
class ExplainedActionInformation:
    """AIM's explained action information, ``loss(null action) - loss(true action)``, over all steps (``overall``)
    and per bucket, with the gate verdict.

    ``gate`` is "pass" only when every unit passes; "fail" when any unit with enough clips fails; else "insufficient".
    ``reasons`` names every unit that does not pass and why.
    """

    metric: str
    overall: EaiUnit
    buckets: tuple[EaiUnit, ...]
    min_clusters: int
    gate: str
    reasons: tuple[str, ...]
    defined: bool

    @property
    def value(self) -> float:
        return self.overall.eai


def explained_action_information(
    loss_null: npt.ArrayLike,
    loss_true: npt.ArrayLike,
    *,
    buckets: Sequence[Hashable] | None = None,
    required_buckets: Sequence[Hashable] = (),
    clusters: Sequence[Hashable] | None = None,
    min_clusters: int = MIN_CLUSTERS,
) -> ExplainedActionInformation:
    """EAI from per-step losses ``[N]`` of the same model on the same held-out steps: with the action replaced by the
    null token (``loss_null``) and with the true ``action_cmd`` (``loss_true``).

    ``buckets`` labels each step for the per-bucket gate (``drivedit.eval.strata.speed_bin``, a κ cell); every bucket
    that occurs is part of the gate, and so is every label in ``required_buckets`` (one without rows is
    "insufficient", so an unvisited speed bin cannot pass). ``clusters`` labels each step's independent unit (its clip
    or rollout): neighbouring steps of one clip are correlated, so the standard error and the degrees of freedom come
    from clips. Without clusters every step counts as an independent clip and the bound is optimistic.
    """
    if min_clusters < 2:
        raise ValueError(f"min_clusters must be >= 2 (a standard error needs two clips), got {min_clusters}")
    null = np.asarray(loss_null, dtype=np.float64).ravel()
    true = np.asarray(loss_true, dtype=np.float64).ravel()
    if null.shape != true.shape or null.size == 0:
        raise ValueError(f"loss_null and loss_true must be equal non-empty [N], got {null.size} and {true.size}")
    if not (np.all(np.isfinite(null)) and np.all(np.isfinite(true))):
        raise ValueError("losses must be finite")
    diff = null - true
    cluster = _codes(clusters, diff.size, "clusters") if clusters is not None else np.arange(diff.size)
    overall = _unit("all", diff, cluster, min_clusters)
    units = []
    if buckets is not None:
        labels = list(buckets)
        bucket = _codes(labels, diff.size, "buckets")
        for code, label in enumerate(dict.fromkeys(labels)):
            sel = bucket == code
            units.append(_unit(str(label), diff[sel], cluster[sel], min_clusters))
        seen = set(labels)
        units += [
            _unit(str(label), diff[:0], cluster[:0], min_clusters) for label in required_buckets if label not in seen
        ]
    elif required_buckets:
        raise ValueError("required_buckets needs bucket labels for the steps")
    reasons = [_why(u, min_clusters) for u in (overall, *units) if u.status != "pass"]
    statuses = {u.status for u in (overall, *units)}
    gate = "fail" if "fail" in statuses else "insufficient" if "insufficient" in statuses else "pass"
    return ExplainedActionInformation(
        metric="explained_action_information",
        overall=overall,
        buckets=tuple(units),
        min_clusters=min_clusters,
        gate=gate,
        reasons=tuple(reasons),
        defined=True,
    )


def _unit(name: str, values: npt.NDArray[np.float64], cluster: npt.NDArray[np.int64], min_clusters: int) -> EaiUnit:
    nan = math.nan
    if values.size == 0:
        return EaiUnit(name, 0, 0, nan, nan, nan, nan, "insufficient")
    mean, se, k = _clustered_mean(values, cluster)
    t_crit = t_critical(k - 1) if k >= 2 else nan
    lower = mean - t_crit * se if k >= 2 else nan
    passes = mean > 0.0 and lower > 0.0
    status = "insufficient" if k < min_clusters else "pass" if passes else "fail"
    return EaiUnit(name, int(values.size), k, mean, se, t_crit, lower, status)


def _why(unit: EaiUnit, min_clusters: int) -> str:
    if unit.status == "insufficient":
        return f"{unit.bucket}: {unit.n_clusters} clip(s), fewer than the {min_clusters} required"
    return f"{unit.bucket}: EAI {unit.eai:.4g}, lower 95% bound {unit.lower_95:.4g} is not above 0"


def _codes(labels: Sequence[Hashable], n: int, name: str) -> npt.NDArray[np.int64]:
    """Integer codes of ``labels`` in order of first appearance."""
    if len(labels) != n:
        raise ValueError(f"{len(labels)} {name} labels for {n} steps")
    keys = {key: i for i, key in enumerate(dict.fromkeys(labels))}
    return np.asarray([keys[key] for key in labels], dtype=np.int64)


def _clustered_mean(values: npt.NDArray[np.float64], cluster: npt.NDArray[np.int64]) -> tuple[float, float, int]:
    """The mean of ``values``, its standard error over clusters (cluster-weighted by size) and the cluster count."""
    _, inverse, counts = np.unique(cluster, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=values)
    k = counts.size
    mean = float(values.mean())
    if k < 2:
        return mean, math.nan, int(k)
    # cluster-robust variance of a mean: sum over clusters of (cluster sum - size * mean)^2 / n^2, small-k corrected
    resid = sums - counts * mean
    var = float((resid**2).sum()) / values.size**2 * k / (k - 1)
    return mean, math.sqrt(var), int(k)


def t_critical(df: int, coverage: float = _COVERAGE) -> float:
    """The two-sided Student-t critical value: ``t`` with P(|T| <= t) = ``coverage`` for ``df`` degrees of freedom.

    Exact for integer ``df``: bisection on the closed-form distribution of Abramowitz & Stegun, *Handbook of
    Mathematical Functions*, 26.7.3 (no scipy). For example 12.706 at df 1, 2.228 at df 10, 2.010 at df 49.
    """
    if df < 1 or not 0.0 < coverage < 1.0:
        raise ValueError(f"need df >= 1 and 0 < coverage < 1, got {df} and {coverage}")
    lo, hi = 0.0, 1.0
    while _t_coverage(hi, df) < coverage:
        lo, hi = hi, 2.0 * hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _t_coverage(mid, df) < coverage:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _t_coverage(t: float, df: int) -> float:
    """A(t | df) = P(|T| <= t), Abramowitz & Stegun 26.7.3, with theta = atan(t / sqrt(df))."""
    theta = math.atan(t / math.sqrt(df))
    s, c2 = math.sin(theta), math.cos(theta) ** 2
    if df % 2:  # odd: (2/pi) [theta + sin(theta) (cos + 2/3 cos^3 + ... + (2.4...(df-3))/(1.3...(df-2)) cos^(df-2))]
        series, term = 0.0, math.cos(theta)
        if df > 1:
            series = term
            for k in range(1, (df - 3) // 2 + 1):
                term *= c2 * (2 * k) / (2 * k + 1)
                series += term
        return 2.0 / math.pi * (theta + s * series)
    # even: sin(theta) [1 + 1/2 cos^2 + (1.3)/(2.4) cos^4 + ... + (1.3...(df-3))/(2.4...(df-2)) cos^(df-2)]
    series, term = 1.0, 1.0
    for k in range(1, (df - 2) // 2 + 1):
        term *= c2 * (2 * k - 1) / (2 * k)
        series += term
    return s * series
