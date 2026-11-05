"""Controllability metrics on (commanded, realised) pairs: the yaw-gain test through llc_v1, the response slope,
opposite-action sign accuracy and the brake test (numpy only).

Every metric takes arrays of what was commanded and what was realised, and does not know where they came from. For
real data the realised values are engine telemetry; for a world-model rollout they are an estimate from the generated
frames (the evaluation IDM of ``drivedit.eval.idm``, or a geometric estimator; delta-2026-11 §5.2 M3). So one function
grades both, and ``grade`` pairs a model result with the real-data result of the same metric, the ceiling: a model
number is never reported without what real data achieves (gap-4 §3.1 and §8.1, "the reconstruction is the achievable
ceiling"; causal-memory-2026-11 §7 exit gates, "within margin of decoded-GT").

Sources: the yaw gain is delta §5.2 M3 (ψ̇ ≈ v·κ after the controller's response; gain 1 = faithful, 0 = ignores);
the response slope is ActionRoPE's motion slope (causal-memory §5.1 and §5.4: every feature-conditioned baseline
under-responds, slope < 1); the sign test is ForgeWM's opposite-action test (causal-memory §2.1, §5.5); the brake test
measures ego deceleration tracking only: whether a commanded deceleration slows the ego, and by how much
(causal-memory §5.4 and §5.5). It is *not* DIAMOND's brake test of causal confusion, whether the *other* traffic
stops; that needs the other agents' tracks and is not measured here.

Ratios (``grade``, the brake tracking, and ARR in ``drivedit.eval.idm``) share one rule, ``stable_ratio``: a ratio
whose denominator is below a stated floor is undefined, with a reason, never a huge number.

Alignment (features_v1 ``offsets``): ``action_cmd[t]`` applies over the decision interval (P_t, P_t+1], so the realised
response to it is read at the end of that interval, ``state[t+1]``. Every metric here takes the commanded value at t
and the realised value of the interval it governs, both indexed by t.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from drivedit.eval.linear import FloatArray


class MetricResult(Protocol):
    """What ``grade`` needs from a metric result: its name, headline value and whether it is defined."""

    @property
    def metric(self) -> str: ...

    @property
    def value(self) -> float: ...

    @property
    def defined(self) -> bool: ...


def stable_ratio(numerator: float, denominator: float, *, floor: float) -> tuple[float, str]:
    """``numerator / denominator``, or NaN with the reason when either is not finite or ``|denominator| < floor``
    (``floor`` > 0). Without a floor a near-zero ceiling turns noise into ratios of 1e16 (review round 1)."""
    if not floor > 0.0:
        raise ValueError(f"floor must be > 0, got {floor}")
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return math.nan, "undefined: a side is not finite"
    if abs(denominator) < floor:
        return math.nan, f"undefined: |denominator| {abs(denominator):.3g} is below the floor {floor:.3g}"
    return numerator / denominator, ""


@dataclass(frozen=True)
class Graded:
    """One metric on a world model next to the same metric on real data (the ceiling).

    ``ratio`` = model / real, NaN with a ``reason`` when either side is undefined or ``|real| < min_real``.
    """

    metric: str
    real: float
    model: float
    ratio: float
    reason: str


def grade(real: MetricResult, model: MetricResult, *, min_real: float) -> Graded:
    """Pair a model result with the real-data result of the same metric. ``min_real`` (> 0) is the smallest real
    value a ratio is taken against, in the metric's unit: a ceiling near 0 means real data shows no response, and a
    ratio to it means nothing. There is no default; the caller states it per metric."""
    if real.metric != model.metric:
        raise ValueError(f"cannot grade {model.metric} against {real.metric}")
    r = real.value if real.defined else math.nan
    m = model.value if model.defined else math.nan
    ratio, reason = stable_ratio(m, r, floor=min_real)
    undefined = [side for side, result in (("real", real), ("model", model)) if not result.defined]
    if undefined:
        reason = f"undefined: the {' and '.join(undefined)} result is undefined"
    return Graded(metric=real.metric, real=r, model=m, ratio=ratio, reason=reason)


@dataclass(frozen=True)
class LlcResponse:
    """The response of llc_v1 plus the vehicle from the curvature target to the realised curvature: a dead time, a
    first-order lag and a gain, ``κ(t) = gain · lag_τ(κ_target(t - dead_time))``.

    This is the structure family ``tools/ident/fit_steer_kappa.py`` fits on the raw engine path (pure delay, or delay +
    first order) and the vehicle card stores per bin. The full llc path (κ target -> κ realised) is **not identified
    yet**: ``tools/ident/cards/1.8.0.13/S105.json`` has ``actuator.full_llc_path.structure`` null ("kappa target ->
    kappa realised needs llc_v1 sessions", CC-08, DD-7-ident). So there is no default here: callers pass the
    identified values (delta §5.2 M3: identify the response once from real N-mode pulses). A constant offset would not
    change any slope, so the model has none.

    The response has state: the lag's output and the commands still inside the dead time. Counterfactual arms branch
    from one shared state, so pass the context's real commands before the branch (``prehistory``) and, when known,
    the realised curvature at its start (``initial``). Without them each arm starts settled at its own command and
    skips the transient it commands (review round 1: a faithful 2 s response with dead time 0.4 s and lag 0.5 s
    then scored 0.605 instead of 1).
    """

    dead_time_s: float
    tau_s: float
    gain: float

    def __post_init__(self) -> None:
        if not (self.dead_time_s >= 0.0 and self.tau_s >= 0.0 and math.isfinite(self.gain) and self.gain != 0.0):
            raise ValueError(f"invalid llc response {self}")

    def realised_kappa(
        self,
        kappa_target: npt.ArrayLike,
        *,
        step_s: float,
        prehistory: npt.ArrayLike | None = None,
        initial: npt.ArrayLike | None = None,
    ) -> FloatArray:
        """The realised curvature at the end of each decision interval, ``[..., T]`` like ``kappa_target``.

        ``kappa_target[..., t]`` is held over interval t (zero-order hold of ``action_cmd``; ``step_s`` is the
        decision period, 1 / features_v1 ``offsets.decision_hz``). The response is integrated exactly over each
        interval, including a dead time that is not a multiple of ``step_s``.

        ``prehistory`` ``[..., P]`` (or ``[P]``, shared by every window) is the commands of the P intervals before the
        window: they are integrated first and fill the dead time, so arms that share them branch from one state.
        ``initial`` (``[...]`` or a scalar) is the realised curvature at the start of the integration, the start of
        the prehistory if there is one. Before the first integrated interval the first command is taken to have been
        held; without ``initial`` the response starts settled at it. For an exact branch the prehistory must cover the
        dead time, and either reach back far enough to settle the lag or come with ``initial``.
        """
        if not step_s > 0.0:
            raise ValueError(f"step_s must be > 0, got {step_s}")
        window = np.asarray(kappa_target, dtype=np.float64)
        if window.ndim == 0 or window.shape[-1] == 0 or not np.all(np.isfinite(window)):
            raise ValueError("kappa_target must be a finite [..., T] array with T >= 1")
        n_pre = 0
        u = window
        if prehistory is not None:
            pre = np.asarray(prehistory, dtype=np.float64)
            if pre.ndim == 0 or not np.all(np.isfinite(pre)):
                raise ValueError("prehistory must be a finite [..., P] or [P] array")
            n_pre = pre.shape[-1]
            u = np.concatenate([np.broadcast_to(pre, (*window.shape[:-1], n_pre)), window], axis=-1)
        y0 = self.gain * u[..., 0]
        if initial is not None:
            y0 = np.broadcast_to(np.asarray(initial, dtype=np.float64), window.shape[:-1]).copy()
            if not np.all(np.isfinite(y0)):
                raise ValueError("initial must be finite")
        # dead time = m whole intervals + `part` seconds (the 1e-9 absorbs float error at whole multiples)
        q = self.dead_time_s / step_s
        m = math.floor(q + 1e-9)
        part = max(0.0, (q - m) * step_s)
        early = _delayed(u, m + 1)  # input over the first `part` seconds of interval t
        late = _delayed(u, m)  # input over the rest of it
        if self.tau_s == 0.0:
            pure: FloatArray = self.gain * late[..., n_pre:]
            return pure
        e_full = math.exp(-step_s / self.tau_s)
        e_late = math.exp(-(step_s - part) / self.tau_s)
        out = np.empty_like(u)
        y = y0
        for t in range(u.shape[-1]):
            y = y * e_full + self.gain * (early[..., t] * (e_late - e_full) + late[..., t] * (1.0 - e_late))
            out[..., t] = y
        return out[..., n_pre:]


def _delayed(u: FloatArray, steps: int) -> FloatArray:
    """``u`` delayed by ``steps`` intervals along the last axis, the first value held before the start."""
    if steps == 0:
        return u
    n = u.shape[-1]
    idx = np.clip(np.arange(n) - steps, 0, None)
    out: FloatArray = u[..., idx]
    return out


@dataclass(frozen=True)
class Slope:
    """A least-squares response slope of realised on commanded values.

    ``slope`` 1 = faithful, 0 = ignores the command, < 1 under-responds, > 1 exaggerates. ``r2`` is the fit's share of
    the realised variation (within groups when there are groups). ``defined`` is False, with a ``reason``, when the
    commanded values do not vary (within groups): no slope can be measured then.
    """

    metric: str
    slope: float
    intercept: float
    r2: float
    n: int
    n_groups: int
    defined: bool
    reason: str

    @property
    def value(self) -> float:
        return self.slope


def response_slope(
    commanded: npt.ArrayLike,
    realised: npt.ArrayLike,
    *,
    groups: Sequence[Hashable] | npt.ArrayLike | None = None,
    metric: str = "response_slope",
) -> Slope:
    """Slope of ``realised`` on ``commanded`` (any equal shapes, flattened; e.g. a commanded deviation and the
    realised deviation of the response).

    ``groups`` (same shape, or one label per element) makes it the within-group slope: commanded and realised values
    are centred inside each group first, so for counterfactual arms that share a context (and a step) everything the
    arms have in common cancels and only the response to the command difference is measured. Without groups it is
    the ordinary slope with an intercept.
    """
    c = np.asarray(commanded, dtype=np.float64).ravel()
    r = np.asarray(realised, dtype=np.float64).ravel()
    if c.shape != r.shape or c.size == 0:
        raise ValueError(f"commanded and realised must be equal non-empty shapes, got {c.size} and {r.size} values")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        raise ValueError("commanded and realised must be finite")
    if groups is None:
        labels = np.zeros(c.size, dtype=np.int64)
    else:
        g = np.asarray(groups).ravel()
        if g.size != c.size:
            raise ValueError(f"{g.size} group labels for {c.size} values")
        _, labels = np.unique(g, return_inverse=True)
    n_groups = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_groups).astype(np.float64)
    cd = c - (np.bincount(labels, weights=c, minlength=n_groups) / counts)[labels]
    rd = r - (np.bincount(labels, weights=r, minlength=n_groups) / counts)[labels]
    sxx = float(cd @ cd)
    scale = max(float(np.max(np.abs(c))), 1e-300)
    if sxx <= (1e-12 * scale) ** 2 * c.size:
        where = " within groups" if groups is not None else ""
        return Slope(
            metric, math.nan, math.nan, math.nan, int(c.size), n_groups, False, f"commanded is constant{where}"
        )
    slope = float(cd @ rd) / sxx
    syy = float(rd @ rd)
    r2 = 1.0 - float(((rd - slope * cd) ** 2).sum()) / syy if syy > 0.0 else math.nan
    intercept = float(r.mean() - slope * c.mean()) if groups is None else math.nan
    return Slope(metric, slope, intercept, r2, int(c.size), n_groups, True, "")


def yaw_gain(
    kappa_target: npt.ArrayLike,
    speed_mps: npt.ArrayLike,
    yaw_rate_rps: npt.ArrayLike,
    *,
    response: LlcResponse,
    step_s: float,
    contexts: Sequence[Hashable] | npt.ArrayLike | None = None,
    prehistory_kappa: npt.ArrayLike | None = None,
    initial_kappa: npt.ArrayLike | None = None,
    settle_steps: int = 0,
) -> Slope:
    """The yaw-gain test (delta §5.2 M3): the slope of the realised yaw rate against the commanded one,
    ψ̇_cmd = v · κ_resp with κ_resp = ``response.realised_kappa(kappa_target)``.

    All three arrays are ``[T]`` (one window) or ``[N, T]`` (N windows), per decision step t:

    - ``kappa_target``: ``action_cmd.kappa_target`` (1/m, right-positive), held over interval t;
    - ``speed_mps``, ``yaw_rate_rps``: the realised speed (m/s) and yaw rate (rad/s, right-positive) at the end of
      interval t: ``state[t+1]`` for real data, the estimate from generated frame t+1 for a rollout.

    ``contexts`` (``[N]``): windows with the same label are arms of one counterfactual set (the same context under
    different κ offsets, delta §5.2 M3); the gain is then the within-(context, step) slope. Arms branch from their
    context's state: pass the context's real ``kappa_target`` before the branch as ``prehistory_kappa`` (``[N, P]``,
    or ``[P]`` for all) and, when known, the realised curvature at its start as ``initial_kappa`` (``[N]`` or a
    scalar; ``LlcResponse.realised_kappa``). Without a prehistory the reference starts each window settled at its
    first command, which is only right for windows that start in steady state; ``settle_steps`` leading steps of each
    window can then be left out.
    """
    k = np.atleast_2d(np.asarray(kappa_target, dtype=np.float64))
    v = np.atleast_2d(np.asarray(speed_mps, dtype=np.float64))
    w = np.atleast_2d(np.asarray(yaw_rate_rps, dtype=np.float64))
    if not (k.shape == v.shape == w.shape) or k.ndim != 2:
        raise ValueError(
            f"kappa_target, speed_mps and yaw_rate_rps must be equal [T] or [N, T], got "
            f"{k.shape}, {v.shape} and {w.shape}"
        )
    if not 0 <= settle_steps < k.shape[1]:
        raise ValueError(f"settle_steps must be in [0, {k.shape[1]}), got {settle_steps}")
    commanded = v * response.realised_kappa(k, step_s=step_s, prehistory=prehistory_kappa, initial=initial_kappa)
    groups = None
    if contexts is not None:
        ctx = np.asarray(contexts).ravel()
        if ctx.size != k.shape[0]:
            raise ValueError(f"{ctx.size} contexts for {k.shape[0]} windows")
        _, ctx_id = np.unique(ctx, return_inverse=True)
        groups = (ctx_id[:, None] * k.shape[1] + np.arange(k.shape[1])[None, :])[:, settle_steps:]
    return response_slope(commanded[:, settle_steps:], w[:, settle_steps:], groups=groups, metric="yaw_gain")


@dataclass(frozen=True)
class SignAccuracy:
    """Opposite-action sign test: the share of pairs whose realised difference has the commanded sign.

    ``chance`` is 0.5: a model whose response is noise unrelated to the command scores about that, and one that does
    not respond at all scores 0 (a difference within the dead band counts as wrong).
    """

    metric: str
    accuracy: float
    n: int
    n_correct: int
    chance: float
    defined: bool
    reason: str

    @property
    def value(self) -> float:
        return self.accuracy


def sign_accuracy(commanded: npt.ArrayLike, realised: npt.ArrayLike, *, deadband: float = 0.0) -> SignAccuracy:
    """ForgeWM's opposite-action sign test on ``[N, 2]`` pairs (arm a, arm b).

    Each row is two arms of one context whose commands differ (for example κ offsets +δ and -δ, or brake vs
    accelerate, held for the same time) and the realised response of each arm (for example the mean yaw rate or the
    heading change at +2 s, or the speed change). A pair is correct when ``sign(realised_a - realised_b) ==
    sign(commanded_a - commanded_b)`` and ``|realised_a - realised_b| > deadband``. For real data the pairs are matched
    real windows (similar initial state, commands of opposite deviation); the caller forms them.
    """
    c = np.asarray(commanded, dtype=np.float64)
    r = np.asarray(realised, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or r.shape != c.shape:
        raise ValueError(f"commanded and realised must both be [N, 2], got {c.shape} and {r.shape}")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        raise ValueError("commanded and realised must be finite")
    if deadband < 0.0:
        raise ValueError(f"deadband must be >= 0, got {deadband}")
    dc = c[:, 0] - c[:, 1]
    if np.any(dc == 0.0):
        raise ValueError(f"{int((dc == 0.0).sum())} pair(s) have equal commands: not an opposite-action pair")
    n = int(c.shape[0])
    if n == 0:
        return SignAccuracy("sign_accuracy", math.nan, 0, 0, 0.5, False, "no pairs")
    dr = r[:, 0] - r[:, 1]
    correct = (np.sign(dr) == np.sign(dc)) & (np.abs(dr) > deadband)
    n_correct = int(correct.sum())
    return SignAccuracy("sign_accuracy", n_correct / n, n, n_correct, 0.5, True, "")


@dataclass(frozen=True)
class BrakeTest:
    """Ego deceleration tracking: commanded decelerations and what the ego achieved over the same intervals. Not
    DIAMOND's test of whether other traffic stops (module docstring).

    ``tracking`` = mean achieved / mean commanded acceleration over the steps with ``a_target < 0`` (1 = the commanded
    deceleration is achieved, 0 = no deceleration, < 0 = the car speeds up); undefined, with a ``reason``, when the
    mean commanded deceleration is below ``min_decel_mps2``. ``decel_fraction`` is the share of those steps that
    decelerate at all.
    """

    metric: str
    n_brake: int
    tracking: float
    decel_fraction: float
    mean_commanded_mps2: float
    mean_achieved_mps2: float
    defined: bool
    reason: str

    @property
    def value(self) -> float:
        return self.tracking


def brake_test(a_target: npt.ArrayLike, achieved_mps2: npt.ArrayLike, *, min_decel_mps2: float) -> BrakeTest:
    """Ego deceleration tracking over decision steps (any equal shapes): ``a_target`` is ``action_cmd.a_target``
    (m/s²) held over interval t, ``achieved_mps2`` the ego's mean longitudinal acceleration over the same interval,
    ``(v[t+1] - v[t]) / step_s`` from telemetry or from the speeds a rollout's estimator reads off frames t and t+1.

    ``min_decel_mps2`` (> 0) is the floor of the tracking ratio's denominator, for example the vehicle card's
    ``llc_v1.coast_band_mps2``, below which llc_v1 neither brakes nor throttles (DC_LLCv1.c). There is no default."""
    a = np.asarray(a_target, dtype=np.float64).ravel()
    got = np.asarray(achieved_mps2, dtype=np.float64).ravel()
    if a.shape != got.shape:
        raise ValueError(f"a_target and achieved_mps2 differ in size: {a.size} and {got.size}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(got))):
        raise ValueError("a_target and achieved_mps2 must be finite")
    brake = a < 0.0
    n = int(brake.sum())
    if n == 0:
        nan = math.nan
        return BrakeTest("brake_test", 0, nan, nan, nan, nan, False, "no step commands a deceleration (a_target < 0)")
    mean_cmd = float(a[brake].mean())
    mean_got = float(got[brake].mean())
    tracking, reason = stable_ratio(mean_got, mean_cmd, floor=min_decel_mps2)
    return BrakeTest(
        metric="brake_test",
        n_brake=n,
        tracking=tracking,
        decel_fraction=float((got[brake] < 0.0).mean()),
        mean_commanded_mps2=mean_cmd,
        mean_achieved_mps2=mean_got,
        defined=not reason,
        reason=reason,
    )
