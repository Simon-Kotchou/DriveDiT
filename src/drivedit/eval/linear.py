"""Closed-form ridge regression, R², and cross-validation folds that never cross a segment (numpy only).

Shared by the E0 telemetry check (``drivedit.eval.e0``) and the linear state probes (``drivedit.eval.probes``).

Rows are decision steps. A *segment* is a ``[start, stop)`` range of rows on consecutive decision steps of one episode:
the features_v1 shard ``meta.json`` segments, or the dc-2.0 reader's ``Session.segments`` mapped to rows. Features
that look back and targets that look ahead are built inside one segment, and the folds here respect segments:

- **held-out groups** (sessions): one fold per group, used when the rows come from two or more groups;
- **blocked**: the rows, in segment order, are cut into ``n_folds`` contiguous blocks. Training drops every row within
  ``gap`` rows of a test row of the same segment, so with ``gap`` = the span of the feature and target windows no
  training row shares a raw sample with a test row (purged blocked CV for autocorrelated series; Bergmeir & Benitez,
  "On the use of cross-validation for time series predictor evaluation", Inf. Sci. 2012).

Folds are never shuffled: neighbouring decision steps are strongly autocorrelated, so a random split would score
interpolation, not prediction.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Segment = tuple[int, int]

# A column whose spread is below this fraction of its magnitude is constant up to float rounding.
_CONSTANT_RTOL = 1e-12


@dataclass(frozen=True)
class RidgeFit:
    """A fitted ridge map ``y = x @ coef + intercept`` in the units of the inputs.

    ``coef`` is ``[D, K]`` and ``intercept`` ``[K]``. Columns that were constant in the fitting rows get a zero
    coefficient (``live`` is False for them). With ``clip`` set, ``predict`` first clips each input column to the
    fitting rows' range ``[x_min, x_max]``, so the map never extrapolates beyond its support (see ``fit_ridge``).
    """

    coef: FloatArray
    intercept: FloatArray
    live: npt.NDArray[np.bool_]
    x_min: FloatArray
    x_max: FloatArray
    clip: bool

    def predict(self, x: npt.ArrayLike) -> FloatArray:
        """``[N, D]`` -> ``[N, K]``."""
        xa = as_matrix(x, "x")
        if xa.shape[1] != self.coef.shape[0]:
            raise ValueError(f"x has {xa.shape[1]} columns, the fit has {self.coef.shape[0]}")
        if self.clip:
            xa = np.clip(xa, self.x_min, self.x_max)
        out: FloatArray = xa @ self.coef + self.intercept
        return out


@dataclass(frozen=True)
class Fold:
    """Row indices of one cross-validation fold."""

    train: IntArray
    test: IntArray


@dataclass(frozen=True)
class Folds:
    """The folds of one cross-validation and how they were made: ``kind`` is ``"group"`` or ``"blocked"``."""

    kind: str
    folds: tuple[Fold, ...]
    gap: int


def as_matrix(x: npt.ArrayLike, name: str) -> FloatArray:
    """``x`` as a finite float64 ``[N, D]`` array (a 1-D input becomes one column)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be [N] or [N, D], got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} has non-finite values")
    return a


def constant_columns(x: FloatArray) -> npt.NDArray[np.bool_]:
    """``[D]`` True where a column of ``x`` ``[N, D]`` is constant up to float rounding."""
    spread = np.ptp(x, axis=0)
    scale = np.maximum(np.max(np.abs(x), axis=0), 1.0)
    out: npt.NDArray[np.bool_] = spread <= _CONSTANT_RTOL * scale
    return out


def fit_ridge(x: npt.ArrayLike, y: npt.ArrayLike, *, lam: float, clip: bool = False) -> RidgeFit:
    """Closed-form ridge regression of ``y`` ``[N, K]`` on ``x`` ``[N, D]``.

    The columns of ``x`` are standardised over the fitting rows and ``y`` is centred, so the unpenalised intercept is
    the mean and ``lam`` penalises the squared standardised coefficients against the ``N`` rows' squared error.
    Constant columns are dropped (zero coefficient). Solves the primal ``(ZᵀZ + λI)β = Zᵀy`` when ``D <= N`` and the
    dual ``β = Zᵀ(ZZᵀ + λI)⁻¹y`` when ``D > N`` (wide probe features); both are the same estimator.

    ``clip`` makes the fit clip inputs to the fitting range when it predicts. Use it for bounded physical features
    with rare spikes, where one sample would otherwise dominate (E0: in the first T0L session one terrain contact
    gives ``acc_long_achieved_mps2`` = -85 m/s² against a range of about ±6, and unclipped it alone predicted
    Δ speed -15 m/s and turned the Δ speed R² from 0.2 to -4). Not for features that are linear in a drifting
    quantity (probes), where clipping each column would distort a valid extrapolation.
    """
    if not lam > 0.0:
        raise ValueError(f"lam must be > 0, got {lam}")
    xa = as_matrix(x, "x")
    ya = as_matrix(y, "y")
    n = xa.shape[0]
    if ya.shape[0] != n:
        raise ValueError(f"x has {n} rows, y has {ya.shape[0]}")
    if n < 2:
        raise ValueError(f"ridge needs at least 2 rows, got {n}")
    live = ~constant_columns(xa)
    x_mean = xa.mean(axis=0)
    y_mean = ya.mean(axis=0)
    coef = np.zeros((xa.shape[1], ya.shape[1]), dtype=np.float64)
    if np.any(live):
        std = xa[:, live].std(axis=0)
        z = (xa[:, live] - x_mean[live]) / std
        yc = ya - y_mean
        d = z.shape[1]
        if d <= n:
            beta = np.linalg.solve(z.T @ z + lam * np.eye(d), z.T @ yc)
        else:
            beta = z.T @ np.linalg.solve(z @ z.T + lam * np.eye(n), yc)
        coef[live] = beta / std[:, None]
    intercept = y_mean - x_mean @ coef
    return RidgeFit(coef=coef, intercept=intercept, live=live, x_min=xa.min(axis=0), x_max=xa.max(axis=0), clip=clip)


def r2_score(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> FloatArray:
    """Per-column coefficient of determination ``1 - SS_res / SS_tot`` of ``[N, K]`` arrays.

    NaN for a column whose true values are constant (R² is undefined there) or with fewer than 2 rows. Not clipped:
    a prediction worse than the column mean scores below 0.
    """
    yt = as_matrix(y_true, "y_true")
    yp = as_matrix(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} differ in shape")
    out = np.full(yt.shape[1], np.nan)
    if yt.shape[0] < 2:
        return out
    const = constant_columns(yt)
    ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum(axis=0)
    ss_res = ((yt - yp) ** 2).sum(axis=0)
    ok = ~const & (ss_tot > 0.0)
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def check_segments(segments: Sequence[Segment], n_rows: int) -> tuple[Segment, ...]:
    """The segments as int pairs, checked: non-empty ``[start, stop)`` ranges inside ``[0, n_rows)`` that do not
    overlap. Order is kept."""
    out: list[Segment] = []
    for seg in segments:
        start, stop = int(seg[0]), int(seg[1])
        if not 0 <= start < stop <= n_rows:
            raise ValueError(f"segment {seg} is not a non-empty range inside [0, {n_rows})")
        out.append((start, stop))
    if not out:
        raise ValueError("no segments")
    ordered = sorted(out)
    for (_, stop), (start, _) in itertools.pairwise(ordered):
        if start < stop:
            raise ValueError(f"segments overlap at row {start}")
    return tuple(out)


def segment_rows(segments: Sequence[Segment]) -> IntArray:
    """Every row of the segments, in segment order."""
    return np.concatenate([np.arange(start, stop, dtype=np.int64) for start, stop in segments])


def compact_segments(segments: Sequence[Segment]) -> tuple[IntArray, tuple[Segment, ...]]:
    """The rows of ``segments`` in segment order, and the same segments as ranges over that compact order (the
    folds index the compact rows)."""
    rows = segment_rows(segments)
    stops = np.cumsum([stop - start for start, stop in segments])
    starts = np.concatenate([[0], stops[:-1]])
    return rows, tuple((int(a), int(b)) for a, b in zip(starts, stops, strict=True))


def segment_folds(
    segments: Sequence[Segment],
    *,
    n_folds: int,
    gap: int,
    groups: Sequence[Hashable] | None = None,
) -> Folds:
    """Cross-validation folds over the rows of ``segments`` that never cross a segment.

    ``groups`` labels each segment with its held-out unit (a session id). With two or more distinct labels the folds
    hold out one group each (``kind`` "group"; ``n_folds`` and ``gap`` are unused there because groups share no rows).
    Otherwise ``n_folds`` contiguous blocks (``kind`` "blocked"), purged by ``gap`` rows on each side of the test
    block inside its segments. Raises when a fold would have no test or fewer than 2 training rows.
    """
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    segs = tuple((int(s), int(e)) for s, e in segments)
    if not segs:
        raise ValueError("no segments")
    if groups is not None:
        if len(groups) != len(segs):
            raise ValueError(f"{len(groups)} group labels for {len(segs)} segments")
        labels = list(dict.fromkeys(groups))
        if len(labels) >= 2:
            folds = []
            for label in labels:
                test = segment_rows([s for s, g in zip(segs, groups, strict=True) if g == label])
                train = segment_rows([s for s, g in zip(segs, groups, strict=True) if g != label])
                folds.append(_checked_fold(train, test, label))
            return Folds(kind="group", folds=tuple(folds), gap=0)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    rows = segment_rows(segs)
    seg_id = np.concatenate([np.full(e - s, i, dtype=np.int64) for i, (s, e) in enumerate(segs)])
    n = rows.size
    if n < n_folds:
        raise ValueError(f"{n} rows cannot make {n_folds} blocked folds")
    bounds = np.round(np.linspace(0, n, n_folds + 1)).astype(np.int64)
    folds = []
    for k in range(n_folds):
        in_test = np.zeros(n, dtype=bool)
        in_test[bounds[k] : bounds[k + 1]] = True
        keep = ~in_test
        for i in np.unique(seg_id[in_test]):
            test_rows = rows[in_test & (seg_id == i)]
            near = (seg_id == i) & (rows >= test_rows.min() - gap) & (rows <= test_rows.max() + gap)
            keep &= ~near
        folds.append(_checked_fold(rows[keep], rows[in_test], f"block {k}"))
    return Folds(kind="blocked", folds=tuple(folds), gap=gap)


def _checked_fold(train: IntArray, test: IntArray, name: object) -> Fold:
    if test.size == 0 or train.size < 2:
        raise ValueError(f"fold {name!r} has {test.size} test and {train.size} training rows; use fewer folds")
    return Fold(train=train, test=test)


def cross_val_predict(x: FloatArray, y: FloatArray, folds: Folds, *, lam: float, clip: bool = False) -> FloatArray:
    """Out-of-fold ridge predictions ``[N, K]`` (NaN on rows that are in no test set); ``clip`` as in
    ``fit_ridge``."""
    out = np.full((x.shape[0], y.shape[1]), np.nan)
    for fold in folds.folds:
        fit = fit_ridge(x[fold.train], y[fold.train], lam=lam, clip=clip)
        out[fold.test] = fit.predict(x[fold.test])
    return out
