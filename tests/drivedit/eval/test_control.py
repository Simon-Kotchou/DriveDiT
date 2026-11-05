"""Controllability metrics (drivedit.eval.control) on synthetic (commanded, realised) data with known answers:
perfect, none, shuffled, and the counterfactual-arms case."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drivedit.eval.control import (
    LlcResponse,
    brake_test,
    grade,
    response_slope,
    sign_accuracy,
    stable_ratio,
    yaw_gain,
)

STEP_S = 0.2  # features_v1 offsets: decision_hz 5
# A response model for the tests only: the real llc_v1 full-path response is not identified yet (card S105
# actuator.full_llc_path.structure is null), so these values are illustrative, not the vehicle's.
LAGGED = LlcResponse(dead_time_s=0.07, tau_s=0.25, gain=0.95)


def _brute_force(response: LlcResponse, u: np.ndarray, fine: int = 400, y0: float | None = None) -> np.ndarray:
    """The response integrated on a fine grid, sampled at the end of each decision interval; it starts at ``y0``
    (default: settled at the first command, held before the start)."""
    dt = STEP_S / fine
    t = (np.arange(u.size * fine) + 1) * dt  # end of each fine step
    held = u[np.clip(np.floor((t - dt - response.dead_time_s) / STEP_S + 1e-9).astype(int), 0, None)]
    y, out = (response.gain * u[0] if y0 is None else y0), []
    decay = math.exp(-dt / response.tau_s) if response.tau_s else 0.0
    for i, w in enumerate(held):
        y = response.gain * w + (y - response.gain * w) * decay
        if (i + 1) % fine == 0:
            out.append(y)
    return np.asarray(out)


@pytest.mark.parametrize(
    "response",
    [
        LlcResponse(dead_time_s=0.0, tau_s=0.0, gain=1.0),
        LlcResponse(dead_time_s=0.2, tau_s=0.0, gain=1.0),
        LlcResponse(dead_time_s=0.0, tau_s=0.3, gain=0.8),
        LAGGED,
        LlcResponse(dead_time_s=0.45, tau_s=0.1, gain=1.2),
    ],
)
def test_llc_response_is_the_exact_zero_order_hold_response(response: LlcResponse) -> None:
    u = np.random.default_rng(0).normal(0.0, 0.02, size=25)
    np.testing.assert_allclose(response.realised_kappa(u, step_s=STEP_S), _brute_force(response, u), atol=1e-12)


def test_prehistory_and_initial_continue_one_response() -> None:
    """Splitting a command sequence into prehistory + window gives the tail of the whole sequence's response, and an
    initial condition sets where the lag starts."""
    rng = np.random.default_rng(10)
    u = rng.normal(0.0, 0.02, size=30)
    whole = LAGGED.realised_kappa(u, step_s=STEP_S)
    np.testing.assert_allclose(LAGGED.realised_kappa(u[12:], step_s=STEP_S, prehistory=u[:12]), whole[12:], atol=1e-15)
    shared = LAGGED.realised_kappa(np.stack([u[12:], -u[12:]]), step_s=STEP_S, prehistory=u[:12])
    np.testing.assert_allclose(shared[0], whole[12:], atol=1e-15)  # a [P] prehistory is shared by every window
    start = LAGGED.realised_kappa(np.zeros(5), step_s=STEP_S, prehistory=np.zeros(1), initial=0.1)
    np.testing.assert_allclose(start, _brute_force(LAGGED, np.zeros(6), y0=0.1)[1:], atol=1e-12)


def test_llc_response_shapes_and_dead_time_shift() -> None:
    u = np.arange(10.0)
    one_step = LlcResponse(dead_time_s=0.2, tau_s=0.0, gain=1.0).realised_kappa(u, step_s=STEP_S)
    np.testing.assert_allclose(one_step, [0.0, 0.0, *u[1:-1]])  # the first value is held before the start
    batch = LAGGED.realised_kappa(np.stack([u, -u]), step_s=STEP_S)
    np.testing.assert_allclose(batch[1], -batch[0])
    with pytest.raises(ValueError, match="invalid llc response"):
        LlcResponse(dead_time_s=-0.1, tau_s=0.0, gain=1.0)


def test_response_slope_perfect_scaled_none_and_constant() -> None:
    rng = np.random.default_rng(1)
    c = rng.normal(size=500)
    perfect = response_slope(c, 2.0 * c + 1.0)
    assert perfect.slope == pytest.approx(2.0)
    assert perfect.intercept == pytest.approx(1.0)
    assert perfect.r2 == pytest.approx(1.0)
    none = response_slope(c, rng.normal(size=500))
    assert abs(none.slope) < 0.15
    constant = response_slope(np.ones(10), np.arange(10.0))
    assert not constant.defined
    assert "constant" in constant.reason


def test_within_group_slope_cancels_what_arms_share() -> None:
    rng = np.random.default_rng(2)
    contexts = np.repeat(np.arange(40), 5)
    offsets = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 40)
    base = rng.normal(0.0, 5.0, size=40)[contexts]
    commanded = base + offsets  # arms of one context share a large commanded part ...
    realised = 3.0 * base + 0.8 * offsets  # ... to which the realised value responds differently
    assert response_slope(commanded, realised, groups=contexts).slope == pytest.approx(0.8)
    assert response_slope(commanded, realised).slope > 2.5  # without groups the shared part dominates


def _windows(seed: int, n: int = 30, t: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kappa = np.cumsum(rng.normal(0.0, 0.004, size=(n, t)), axis=1)
    speed = rng.uniform(4.0, 12.0, size=(n, 1)) + np.cumsum(rng.normal(0.0, 0.05, size=(n, t)), axis=1)
    return kappa, speed


def test_yaw_gain_perfect_under_response_none_and_shuffled() -> None:
    kappa, speed = _windows(3)
    faithful = speed * LAGGED.realised_kappa(kappa, step_s=STEP_S)
    perfect = yaw_gain(kappa, speed, faithful, response=LAGGED, step_s=STEP_S, settle_steps=3)
    assert perfect.metric == "yaw_gain"
    assert perfect.slope == pytest.approx(1.0)
    assert perfect.r2 == pytest.approx(1.0)
    half = yaw_gain(kappa, speed, 0.5 * faithful, response=LAGGED, step_s=STEP_S)
    assert half.slope == pytest.approx(0.5)
    rng = np.random.default_rng(4)
    ignores = yaw_gain(kappa, speed, rng.normal(0.0, faithful.std(), faithful.shape), response=LAGGED, step_s=STEP_S)
    assert abs(ignores.slope) < 0.1
    shuffled = yaw_gain(kappa, speed, faithful[rng.permutation(len(faithful))], response=LAGGED, step_s=STEP_S)
    assert abs(shuffled.slope) < 0.3


def test_yaw_gain_needs_the_response_model() -> None:
    """Graded without the llc lag, a faithful response under-scores: the reference must be the controller's."""
    kappa, speed = _windows(5)
    faithful = speed * LAGGED.realised_kappa(kappa, step_s=STEP_S)
    instant = LlcResponse(dead_time_s=0.0, tau_s=0.0, gain=1.0)
    assert yaw_gain(kappa, speed, faithful, response=instant, step_s=STEP_S).r2 < 0.99


def test_yaw_gain_on_counterfactual_arms() -> None:
    """Five κ offsets per context (delta §5.2 M3): the model follows 0.8 of each offset on top of a context-driven
    yaw; the within-(context, step) gain is 0.8."""
    rng = np.random.default_rng(6)
    n_ctx, t = 12, 20
    contexts = np.repeat(np.arange(n_ctx), 5)
    offsets = np.repeat((np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], n_ctx) * 0.01)[:, None], t, axis=1)
    kappa = rng.normal(0.0, 0.02, size=(n_ctx, 1))[contexts] + offsets
    speed = np.full((n_ctx * 5, t), 10.0)
    context_yaw = rng.normal(0.0, 0.3, size=(n_ctx, t))[contexts]  # what the context alone makes the car do
    realised = context_yaw + 0.8 * speed * LAGGED.realised_kappa(offsets, step_s=STEP_S)
    gain = yaw_gain(kappa, speed, realised, response=LAGGED, step_s=STEP_S, contexts=contexts)
    assert gain.slope == pytest.approx(0.8)


def test_short_opposite_arms_from_a_shared_state_score_one() -> None:
    """Review round 1: an exact first-order-plus-dead-time responder (dead time 0.4 s, lag 0.5 s), driving straight
    and then branching into κ = ±δ held for 2 s, scored 0.605 because each arm's reference started settled at its own
    command. With the context's prehistory it scores 1."""
    responder = LlcResponse(dead_time_s=0.4, tau_s=0.5, gain=1.0)
    pre = np.zeros(15)  # 3 s of straight driving before the branch
    arms = np.array([-0.02, -0.01, 0.01, 0.02])
    kappa = np.repeat(arms[:, None], 10, axis=1)  # 2 s per arm
    speed = np.full_like(kappa, 10.0)
    realised = np.stack([10.0 * _brute_force(responder, np.concatenate([pre, np.full(10, a)]))[15:] for a in arms])
    contexts = [0, 0, 0, 0]
    gain = yaw_gain(kappa, speed, realised, response=responder, step_s=STEP_S, contexts=contexts, prehistory_kappa=pre)
    assert gain.slope == pytest.approx(1.0, abs=1e-6)
    assert gain.r2 == pytest.approx(1.0, abs=1e-6)
    # the dead time's worth of prehistory with the lag's state at its start is enough as well
    short = yaw_gain(
        kappa,
        speed,
        realised,
        response=responder,
        step_s=STEP_S,
        contexts=contexts,
        prehistory_kappa=pre[-2:],
        initial_kappa=0.0,
    )
    assert short.slope == pytest.approx(1.0, abs=1e-6)
    biased = yaw_gain(kappa, speed, realised, response=responder, step_s=STEP_S, contexts=contexts)
    assert biased.slope == pytest.approx(0.605, abs=0.01)  # the round-1 behaviour, kept as the counter-example


def test_sign_accuracy_perfect_none_and_shuffled() -> None:
    rng = np.random.default_rng(7)
    delta = rng.uniform(0.005, 0.02, size=400)
    commanded = np.stack([delta, -delta], axis=1)
    response = 10.0 * commanded
    assert sign_accuracy(commanded, response).accuracy == 1.0
    frozen = sign_accuracy(commanded, np.zeros_like(commanded))  # no response at all: every pair fails
    assert frozen.accuracy == 0.0
    noise = sign_accuracy(commanded, rng.normal(size=commanded.shape))
    assert noise.accuracy == pytest.approx(0.5, abs=0.08)
    flipped = rng.random(400) < 0.5
    shuffled = np.where(flipped[:, None], response[:, ::-1], response)
    assert sign_accuracy(commanded, shuffled).accuracy == pytest.approx(0.5, abs=0.08)
    with pytest.raises(ValueError, match="equal commands"):
        sign_accuracy(np.zeros((3, 2)), np.ones((3, 2)))


def test_sign_accuracy_deadband() -> None:
    commanded = np.array([[1.0, -1.0], [1.0, -1.0]])
    realised = np.array([[0.01, 0.0], [1.0, 0.0]])
    assert sign_accuracy(commanded, realised, deadband=0.1).n_correct == 1


def test_brake_test_perfect_none_and_no_brakes() -> None:
    rng = np.random.default_rng(8)
    a = rng.uniform(-3.0, 1.5, size=300)
    perfect = brake_test(a, a, min_decel_mps2=0.05)
    assert perfect.tracking == pytest.approx(1.0)
    assert perfect.decel_fraction == 1.0
    assert perfect.n_brake == int((a < 0).sum())
    coasting = brake_test(a, np.zeros_like(a), min_decel_mps2=0.05)
    assert coasting.tracking == 0.0
    assert coasting.decel_fraction == 0.0
    # a response unrelated to the command achieves the overall mean acceleration on the brake steps
    shuffled = brake_test(a, a[rng.permutation(a.size)], min_decel_mps2=0.05)
    assert shuffled.tracking == pytest.approx(a.mean() / a[a < 0].mean(), abs=0.1)
    cruise = brake_test(np.full(50, 0.3), np.zeros(50), min_decel_mps2=0.05)
    assert not cruise.defined
    assert cruise.n_brake == 0


def test_ratios_near_a_zero_denominator_are_undefined() -> None:
    """Review round 1: a commanded mean deceleration or a real ceiling near 0 gave ratios up to 1e16."""
    feather = brake_test([-1e-17, -1e-17], [-0.1, -0.2], min_decel_mps2=0.05)
    assert not feather.defined
    assert np.isnan(feather.tracking)
    assert "below the floor" in feather.reason
    flat = response_slope(np.arange(10.0), 1e-16 * np.arange(10.0))  # real data that barely responds
    model = response_slope(np.arange(10.0), np.arange(10.0))
    graded = grade(flat, model, min_real=0.05)
    assert np.isnan(graded.ratio)
    assert "below the floor" in graded.reason
    assert stable_ratio(1.0, 4.0, floor=0.1) == (0.25, "")
    with pytest.raises(ValueError, match="floor"):
        stable_ratio(1.0, 4.0, floor=0.0)


def test_grade_pairs_a_model_with_its_real_ceiling() -> None:
    kappa, speed = _windows(9)
    faithful = speed * LAGGED.realised_kappa(kappa, step_s=STEP_S)
    real = yaw_gain(kappa, speed, faithful, response=LAGGED, step_s=STEP_S)
    model = yaw_gain(kappa, speed, 0.6 * faithful, response=LAGGED, step_s=STEP_S)
    graded = grade(real, model, min_real=0.05)
    assert graded.metric == "yaw_gain"
    assert graded.ratio == pytest.approx(0.6)
    assert graded.reason == ""
    with pytest.raises(ValueError, match="cannot grade"):
        grade(real, brake_test([-1.0], [-1.0], min_decel_mps2=0.05), min_real=0.05)
    no_brakes = brake_test([1.0], [0.0], min_decel_mps2=0.05)
    undefined = grade(no_brakes, brake_test([-1.0], [-1.0], min_decel_mps2=0.05), min_real=0.05)
    assert math.isnan(undefined.ratio)
    assert undefined.reason == "undefined: the real result is undefined"
