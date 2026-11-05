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
  forward pass with the null action per evaluation window.
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

# Two-sided 95% normal quantile, for the lower confidence bound of a mean.
_Z95 = 1.959963984540054


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
class EaiBucket:
    """EAI inside one bucket (a speed bin, a κ cell): mean, standard error, its lower 95% bound and ``n`` steps in
    ``n_clusters`` independent units."""

    bucket: str
    n: int
    n_clusters: int
    eai: float
    se: float
    lower_95: float


@dataclass(frozen=True)
class ExplainedActionInformation:
    """AIM's explained action information, ``loss(null action) - loss(true action)``, over all steps and per bucket.

    ``positive_in_every_bucket`` is delta §5.2's pass bar ("EAI > 0 in every speed bin"), read as a lower 95% bound
    above 0 in every bucket that has at least two independent units.
    """

    metric: str
    n: int
    n_clusters: int
    eai: float
    se: float
    lower_95: float
    buckets: tuple[EaiBucket, ...]
    positive_in_every_bucket: bool
    defined: bool

    @property
    def value(self) -> float:
        return self.eai


def explained_action_information(
    loss_null: npt.ArrayLike,
    loss_true: npt.ArrayLike,
    *,
    buckets: Sequence[Hashable] | None = None,
    clusters: Sequence[Hashable] | None = None,
) -> ExplainedActionInformation:
    """EAI from per-step losses ``[N]`` of the same model on the same held-out steps: with the action replaced by the
    null token (``loss_null``) and with the true ``action_cmd`` (``loss_true``).

    ``buckets`` labels each step for the per-bucket report (``drivedit.eval.strata.speed_bin``, a κ cell).
    ``clusters`` labels each step's independent unit (its clip or rollout): neighbouring steps of one clip are
    correlated, so the standard error is computed over cluster means; without clusters every step counts as
    independent and the error is optimistic.
    """
    null = np.asarray(loss_null, dtype=np.float64).ravel()
    true = np.asarray(loss_true, dtype=np.float64).ravel()
    if null.shape != true.shape or null.size == 0:
        raise ValueError(f"loss_null and loss_true must be equal non-empty [N], got {null.size} and {true.size}")
    if not (np.all(np.isfinite(null)) and np.all(np.isfinite(true))):
        raise ValueError("losses must be finite")
    diff = null - true
    cluster = _codes(clusters, diff.size, "clusters") if clusters is not None else np.arange(diff.size)
    bucket_labels: Sequence[Hashable] = buckets if buckets is not None else ["all"] * diff.size
    bucket = _codes(bucket_labels, diff.size, "buckets")
    mean, se, n_clusters = _clustered_mean(diff, cluster)
    per_bucket = []
    for code, label in enumerate(dict.fromkeys(bucket_labels)):
        sel = bucket == code
        b_mean, b_se, b_clusters = _clustered_mean(diff[sel], cluster[sel])
        per_bucket.append(
            EaiBucket(
                bucket=str(label),
                n=int(sel.sum()),
                n_clusters=b_clusters,
                eai=b_mean,
                se=b_se,
                lower_95=b_mean - _Z95 * b_se,
            )
        )
    scored = [b for b in per_bucket if b.n_clusters >= 2]
    return ExplainedActionInformation(
        metric="explained_action_information",
        n=int(diff.size),
        n_clusters=n_clusters,
        eai=mean,
        se=se,
        lower_95=mean - _Z95 * se,
        buckets=tuple(per_bucket) if buckets is not None else (),
        positive_in_every_bucket=bool(scored) and all(b.lower_95 > 0.0 for b in scored),
        defined=True,
    )


def _codes(labels: Sequence[Hashable], n: int, name: str) -> npt.NDArray[np.int64]:
    """Integer codes of ``labels`` in order of first appearance."""
    if len(labels) != n:
        raise ValueError(f"{len(labels)} {name} labels for {n} steps")
    keys = {key: i for i, key in enumerate(dict.fromkeys(labels))}
    return np.asarray([keys[key] for key in labels], dtype=np.int64)


def _clustered_mean(values: npt.NDArray[np.float64], cluster: npt.NDArray[np.int64]) -> tuple[float, float, int]:
    """The mean of ``values`` and its standard error over cluster means (cluster-weighted by size)."""
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
