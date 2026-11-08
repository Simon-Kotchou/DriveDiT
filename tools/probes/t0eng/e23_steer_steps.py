#!/usr/bin/env python3
"""CC-08 steering actuator structure from the S2 steering steps of probe e23 (DD-14b, DC_ProbeSteerSteps).

    python3 -I -B tools/probes/t0eng/e23_steer_steps.py <e23 play_dir> [--e22 <e22 play_dir>] [--min-r2 0.9]
                                                        [--max-cv 0.2] [--min-events 6]

<play_dir> is one $profile:DC_Probes/e23/<launch_id>/play_NN directory (run_header.txt, e23_steps.csv,
e23_events.csv, e23_phases.csv, run_footer.txt); --e22 is the e22 play whose CC-07 step 0 validates the wheel-angle
channel. Nothing is written.

MIGRATION_1_9.md 6.1 CC-08 asks for steps 0 -> +-0.5 and 0 -> +1.0 at 10/30/60 km/h, 50 repetitions each with random
phase, a held-out PRBS, competing structures per speed and amplitude bin for the raw engine path (cmd -> steer_input
-> wheel_angle) and the full llc path (kappa target -> kappa_realised), the llc limiter rule and the 10 Hz ZOH
tracking. This analysis covers the raw engine path; what it cannot cover is INCOMPLETE, never PASS:
  coverage       the spec grid (SPEC_SPEEDS x SPEC_STEPS x SPEC_REPS, encoded here, not read from the run) against the
                 ok events in each measured-speed bin, and a PRBS segment at every spec speed: PASS only when all of
                 it is there; else INCOMPLETE with the counts (the default e23 grid is PREVIEW coverage)
  llc_path       always INCOMPLETE: the llc path (llc_v1 in the loop), its limiter rule and the 10 Hz ZOH kappa
                 tracking are not measured by e23 (DD-7 checks)
  evidence       the e23 run footer complete (and the e22 footer with --e22), else INCOMPLETE
  input_<v>kmh_<A>, wheel_<v>kmh_<A>   the raw engine path per spec bin (PARTIAL evidence for CC-08):
                 input = cmd_steer -> input_post_steer; wheel = cmd_steer -> delta_eff of the channel validated by
                 CC-07 step 0 (--e22; the steering0 signal with its validated scale), INCOMPLETE without one.
                 PASS: the selected structure's held-out R2 >= --min-r2 and every identified parameter stable
                 (CV <= --max-cv); FAIL: R2 or a CV out of bounds; INCOMPLETE: too few events or doublets, a
                 parameter that is not resolved
  prbs_<path>_<v>kmh   the held-out PRBS at each spec speed predicted by the bin model of that speed: only a PRBS
                 that completed its whole sequence (e23_phases.csv prbs_result ok, prbs_bits = prbs_expected, the
                 segment covering every bit); R2 >= --min-r2 PASS, below FAIL, none or no model INCOMPLETE
  aborts         a phase that did not end ok (aborted_heading, overspeed, aborted_area, spin, skipped) withholds
                 every bin and PRBS of its speed (INCOMPLETE), even when the footer is complete
  steer_sign     the kappa plateaus have the command's sign (+1 on 1.8.0.13 and 1.9.0.37)
  INFO           cmd -> kappa per bin (labelled PARTIAL: the command to the realised curvature, not the llc path),
                 the rate limit vs speed, the centering on release, the L/R angle ratio, the script-setter limit
Structures (discrete in engine steps of dt; a row that covers n merged steps is n steps of the model with the row's
command, its output compared at the last of them; target = g x u[step - d]):
  M1 delay           y = target
  M2 + first order   y1 += (1 - a) (target - y1), a = exp(-dt / tau)
  M3 + rate limit    the M2 step clamped to +-R dt (R = q x |g| x the bin's step size per second)
  M4 + relaxation    M3 then a second first order with tau2 = sigma / v (tyre relaxation length sigma)
Identifiability: every command change needs readbacks in its first TRANSITION_ROWS rows, and its first readback
must come from a dispatch covering exactly one engine step: after a merged dispatch delays d and d + 1 leave the
same recorded endpoint, so such a bin and its PRBS are INCOMPLETE ("delay not identifiable under merged
dispatches") and CC-07 (a)'s model exception is unavailable. The real runs so far report clock.merged_fixed_steps 0,
so the rule should cost no real data. The PRBS is replayed through the producer's bit-boundary state machine
(applied_matches): any other dispatch sequence is not a held-out set.
Windows: the scored rows run from an event's onset to its end; the rows before are unscored history (the model state
starts at the measured output of the row before the onset, the D_MAX engine steps of commands before feed the delay),
so no row is scored twice. Every output is measured from its run's hold baseline (command 0 at speed). An event
counts in a bin only when its mean measured speed is within SPEED_BIN_SHARE of the bin speed.
Split: whole doublets (+A then -A), alternating train / held-out, so both sets hold both signs.
Gain: closed form for the linear structures; for a clamped M3 / M4 the gain is optimised against the full nonlinear
simulation (golden section around the closed-form start) wherever the parameters are refined.
Fit: a grid over d (0..D_MAX) and tau; joint grids of tau with R (M3) or sigma (M4); coordinate descent; golden-section
refinement at the best delay and at +-1 step. Selection: the lowest held-out MSE; within 2% of it (or the noise
floor) the lowest BIC; a first-order stage that does not improve the fit beyond the noise floor is dropped (tau 0).
Parameter stability: the selected structure fitted to each event alone; CV = stdev / |mean| of g (unresolved when a
rate limit covers the whole on-time in more than 1/3 of the events), of tau, R and sigma; a stage active in some
events and inactive in others (tau < dt / ln 100, no rate limit, no relaxation) is unresolved, never stable; the delay
is stable when constant or its CV <= --max-cv. Without a complete footer every bin, PRBS and steer_sign result is
withheld (INCOMPLETE with its observed tag). input_model (CC-07 a's exception) needs every input bin and the held-out
PRBS at every bin speed to PASS and no aborted phase at those speeds; its scope is the validated cells only (each bin
speed +-15% with the largest amplitude fitted there), no gap filling, no extrapolation to standstill.
Exit codes: 0 the analysis ran (the items carry the verdicts), 1 an input, identity or usage error.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402
import steer_common as sc  # noqa: E402

D_MAX = 8
TAU_GRID_S = [0.0] + [0.008 * (1.5 / 0.008) ** (i / 13.0) for i in range(14)]
Q_GRID = [None] + [0.3 * (60.0 / 0.3) ** (i / 13.0) for i in range(14)]     # rate limit: step sizes per second
SIGMA_GRID_M = [0.0] + [0.05 * (5.0 / 0.05) ** (i / 9.0) for i in range(10)]
TIE_SHARE = 0.02
NOISE_FLOOR_SHARE = 1e-4      # a held-out MSE below this share of the output variance (R2 >= 0.9999) is a tie
STRUCTURES = ("M1", "M2", "M3", "M4")
STRUCTURE_NAMES = {"M1": "delay", "M2": "delay+first order", "M3": "delay+first order+rate limit",
                   "M4": "delay+first order+rate limit+relaxation"}
PARAM_COUNT = {"M1": 2, "M2": 3, "M3": 4, "M4": 5}
RAW_PATHS = ("input", "wheel")
EVENT_COLUMNS = ("event_idx", "phase", "target_kmh", "amplitude", "run_idx", "jitter_steps", "onset_step",
                 "release_step", "end_step", "onset_kmh", "result")
# The CC-08 grid (MIGRATION_1_9.md 6.1), independent of what the run was configured to do
SPEC_SPEEDS = (10.0, 30.0, 60.0)
SPEC_STEPS = ((0.5, (1, -1)), (1.0, (1,)))
SPEC_REPS = 50
SPEED_BIN_SHARE = 0.15
MIN_DOUBLETS = 2              # per side of the split
RESOLVED_SHARE = 2.0 / 3.0
TAU_RESOLUTION = math.log(100.0)
# Rows after every command change that must carry a readback: the delay search (D_MAX) + 2 (a missing first
# readback would hide a delay)
TRANSITION_ROWS = D_MAX + 2
# DC_ProbePrbs taps per order (DC_ProbeSequences.c)
PRBS_TAPS = {5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7)}
STATE_TEXT = {True: "PASS", False: "FAIL", None: "INCOMPLETE"}


# ---------------------------------------------------------------------------------------------------------------
# models


class Params:
    """One candidate: delay d (engine steps), gain g, tau (s), q (rate limit in step sizes per second, None = none),
    sigma (relaxation length m, 0 = none)."""

    __slots__ = ("d", "g", "tau", "q", "sigma")

    def __init__(self, d=0, g=1.0, tau=0.0, q=None, sigma=0.0):
        self.d, self.g, self.tau, self.q, self.sigma = d, g, tau, q, sigma

    def copy(self):
        return Params(self.d, self.g, self.tau, self.q, self.sigma)


def clamped(structure, params):
    return structure in ("M3", "M4") and params.q is not None


def simulate(window, params, dt, structure, with_input=True, with_state=True):
    """Output of a structure at the scored rows of one window: u_e (the commands per engine step, the history
    first), offset (where the scored steps begin), n (engine steps per scored row), y0 (the measured output before
    the first scored row, the initial state of every stage), v, step_size. with_input / with_state: the forced and
    the free response alone (their sum is the output of the linear structures; the closed-form gain uses them)."""
    u_e = window["u_e"]
    a = math.exp(-dt / params.tau) if params.tau > 0 else 0.0
    rate = None
    if clamped(structure, params):
        rate = params.q * abs(params.g) * window["step_size"] * dt
    b = 0.0
    if structure == "M4" and params.sigma > 0 and window["v"] > 0.1:
        b = math.exp(-dt * window["v"] / params.sigma)
    y0 = window["y0"] if with_state else 0.0
    y1 = y0
    y2 = y0
    y = y0
    out = []
    e = window["offset"]
    g = params.g
    d = params.d
    for n in window["n"]:
        for _ in range(n):
            j = e - d
            command = u_e[j] if with_input and j >= 0 else 0.0
            target = g * command
            if structure == "M1":
                y = target
            else:
                step = (1.0 - a) * (target - y1)
                if rate is not None:
                    step = max(-rate, min(rate, step))
                y1 += step
                y = y1
                if b > 0:
                    y2 = b * y2 + (1.0 - b) * y1
                    y = y2
            e += 1
        out.append(y)
    return out


def sse(windows, params, dt, structure):
    total = 0.0
    n = 0
    for w in windows:
        for ys, ym in zip(simulate(w, params, dt, structure), w["y"]):
            if ym is not None:
                total += (ys - ym) ** 2
                n += 1
    return total, n


def closed_gain(windows, params, dt, structure):
    """Least-squares gain of the linear decomposition: g = sum(A (y - B)) / sum(A A) from the unit-gain forced
    response A and the free response B (exact for M1 / M2 and an unclamped M3 / M4)"""
    unit = params.copy()
    unit.g = 1.0
    num = 0.0
    den = 0.0
    for w in windows:
        forced = simulate(w, unit, dt, structure, with_state=False)
        free = simulate(w, unit, dt, structure, with_input=False)
        for fa, fb, ym in zip(forced, free, w["y"]):
            if ym is not None:
                num += fa * (ym - fb)
                den += fa * fa
    return num / den if den > 0 else 0.0


def score(windows, params, dt, structure, exact=False):
    """SSE of a candidate with its gain (set on params). Linear structures: the closed form (exact). A clamped M3 /
    M4 does not superpose, so the closed form is only the start: the gain is then optimised on the full nonlinear
    simulation, by successive parabolic steps in the grid stages and by a golden-section search (exact) in the
    refinement"""
    params.g = closed_gain(windows, params, dt, structure)
    if not clamped(structure, params):
        return sse(windows, params, dt, structure)[0]

    def evaluate(g):
        p = params.copy()
        p.g = g
        return sse(windows, p, dt, structure)[0]

    g = params.g
    value = evaluate(g)
    if exact:
        lo, hi = (0.25 * g, 2.5 * g) if abs(g) > 1e-9 else (-1.0, 1.0)
        best_g, best = golden(evaluate, min(lo, hi), max(lo, hi), steps=18)
        if best < value:
            g, value = best_g, best
    for _ in range(3):
        h = 0.15 * abs(g) if abs(g) > 1e-9 else 0.05
        lo_v, hi_v = evaluate(g - h), evaluate(g + h)
        curvature = lo_v + hi_v - 2.0 * value
        if curvature <= 0:
            trial = g - h if lo_v < hi_v else g + h
        else:
            trial = g - h * (hi_v - lo_v) / (2.0 * curvature)
        trial_v = evaluate(trial)
        for cand, cand_v in ((g - h, lo_v), (g + h, hi_v), (trial, trial_v)):
            if cand_v < value:
                g, value = cand, cand_v
    params.g = g
    return value


def neighbours(grid, value):
    """The grid values around value (continuous refinement bounds)"""
    values = sorted(v for v in grid if v is not None)
    lower = max([v for v in values if v < value] or [values[0]])
    upper = min([v for v in values if v > value] or [values[-1]])
    return lower, upper


def golden(evaluate, lo, hi, steps=14):
    """Minimum of evaluate on [lo, hi] (golden-section search); (x, value)"""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc, fd = evaluate(c), evaluate(d)
    for _ in range(steps):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = evaluate(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = evaluate(d)
    return (c, fc) if fc <= fd else (d, fd)


def fit_structure(windows, dt, structure, start=None, joint=True):
    """Grid / coordinate-descent fit of one structure, then a continuous refinement of tau, the rate limit and
    sigma between their grid neighbours with the exact gain (so a richer structure never wins by the grid or by a
    superposition shortcut); returns (Params, SSE, n). joint: the two-parameter grids of M3 / M4 (the bin fits;
    the per-event fits start next to the bin optimum and skip them)."""
    best = None
    best_sse = None

    def consider(p, exact=False):
        nonlocal best, best_sse
        p = p.copy()
        value = score(windows, p, dt, structure, exact)
        if best_sse is None or value < best_sse - 1e-15:
            best, best_sse = p, value

    if start is None:
        taus = [0.0] if structure == "M1" else TAU_GRID_S
        for d in range(D_MAX + 1):
            for tau in taus:
                consider(Params(d=d, tau=tau))
    else:
        consider(start)
        for d in range(D_MAX + 1):
            p = best.copy()
            p.d = d
            consider(p)
        if structure != "M1":
            for tau in TAU_GRID_S:
                p = best.copy()
                p.tau = tau
                consider(p)
    # joint grids of the two parameters that trade off against each other (a lag can mimic a rate limit or a
    # relaxation); an added stage takes over part of the dead time, so they also try up to 3 steps less delay
    if structure == "M3" and joint:
        base = best.copy()
        for d in range(max(0, base.d - 3), base.d + 1):
            for tau in TAU_GRID_S:
                for q in Q_GRID:
                    p = base.copy()
                    p.d, p.tau, p.q = d, tau, q
                    consider(p)
    if structure == "M4" and joint:
        base = best.copy()
        for d in range(max(0, base.d - 3), base.d + 1):
            for tau in TAU_GRID_S:
                for sigma in SIGMA_GRID_M:
                    p = base.copy()
                    p.d, p.tau, p.sigma = d, tau, sigma
                    consider(p)
    if structure in ("M3", "M4"):
        for _ in range(2):
            for q in Q_GRID:
                p = best.copy()
                p.q = q
                consider(p)
            if structure == "M4":
                for sigma in SIGMA_GRID_M:
                    p = best.copy()
                    p.sigma = sigma
                    consider(p)
            for tau in TAU_GRID_S:
                p = best.copy()
                p.tau = tau
                consider(p)
            for d in range(D_MAX + 1):
                p = best.copy()
                p.d = d
                consider(p)

    def refine_from(params):
        """Rounds of golden-section refinement of the continuous parameters from params (exact gain); (params,
        SSE)"""
        current = params.copy()
        current_sse = score(windows, current, dt, structure, exact=True)
        names = []
        if structure != "M1":
            names.append(("tau", TAU_GRID_S))
        if clamped(structure, current):
            names.append(("q", Q_GRID))
        if structure == "M4" and current.sigma > 0:
            names.append(("sigma", SIGMA_GRID_M))
        for _ in range(3):
            for name, grid in names:
                lo, hi = neighbours(grid, getattr(current, name))
                if hi <= lo:
                    continue

                def evaluate(x, name=name):
                    p = current.copy()
                    setattr(p, name, x)
                    return score(windows, p, dt, structure, exact=True)

                x, value = golden(evaluate, lo, hi, steps=12)
                if value < current_sse:
                    setattr(current, name, x)
                    current_sse = score(windows, current, dt, structure, exact=True)
        return current, current_sse

    refined, refined_sse = refine_from(best)
    for d in (refined.d - 1, refined.d + 1):
        if d < 0 or d > D_MAX or structure == "M1":
            continue
        trial = refined.copy()
        trial.d = d
        candidate, candidate_sse = refine_from(trial)
        if candidate_sse < refined_sse:
            refined, refined_sse = candidate, candidate_sse
    if refined_sse < best_sse - 1e-15:
        best, best_sse = refined, refined_sse
    total, n = sse(windows, best, dt, structure)
    return best, total, n


def r2(windows, params, dt, structure):
    total, n = sse(windows, params, dt, structure)
    values = [y for w in windows for y in w["y"] if y is not None]
    if not values:
        return None
    m = sum(values) / len(values)
    sst = sum((y - m) ** 2 for y in values)
    if sst <= 0:
        return None
    return 1.0 - total / sst


def bic(total, n, structure, floor_mse=0.0):
    """n ln(SSE / n) + k ln n, the SSE floored at n x floor_mse (errors below the noise floor are equal, so the
    parameter count decides between candidates that both reach it)"""
    if n <= 0:
        return None
    total = max(total, n * floor_mse)
    if total <= 0:
        return float("-inf")
    return n * math.log(total / n) + PARAM_COUNT[structure] * math.log(n)


def variance(windows):
    values = [y for w in windows for y in w["y"] if y is not None]
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((y - m) ** 2 for y in values) / len(values)


def split_doublets(windows):
    """(train, test, train doublets, test doublets): whole doublets alternately, in the order they ran (both signs
    on both sides)"""
    order = []
    for w in windows:
        if w["doublet"] not in order:
            order.append(w["doublet"])
    train_keys = set(order[0::2])
    train = [w for w in windows if w["doublet"] in train_keys]
    test = [w for w in windows if w["doublet"] not in train_keys]
    return train, test, len(order[0::2]), len(order[1::2])


def fit_bin(windows, dt):
    """All four structures on the training doublets, held-out on the others; the selection and its numbers."""
    train, test, n_train, n_test = split_doublets(windows)
    floor_mse = NOISE_FLOOR_SHARE * variance(train)
    result = {"structures": {}, "floor_mse": floor_mse, "train_doublets": n_train, "test_doublets": n_test,
              "selected": None}
    if n_train < MIN_DOUBLETS or n_test < MIN_DOUBLETS:
        return result
    start = None
    for structure in STRUCTURES:
        params, total, n = fit_structure(train, dt, structure, start if structure in ("M3", "M4") else None)
        test_sse, test_n = sse(test, params, dt, structure)
        result["structures"][structure] = {
            "params": params, "train_sse": total, "train_n": n, "bic": bic(total, n, structure, floor_mse),
            "test_mse": test_sse / test_n if test_n else None, "test_r2": r2(test, params, dt, structure)}
        if structure in ("M2", "M3"):
            start = params
    candidates = [(s, v) for s, v in result["structures"].items() if v["test_mse"] is not None]
    if not candidates:
        return result
    best_mse = min(v["test_mse"] for _, v in candidates)
    close = [(s, v) for s, v in candidates if v["test_mse"] <= best_mse * (1.0 + TIE_SHARE) + floor_mse]
    close.sort(key=lambda item: (item[1]["bic"] if item[1]["bic"] is not None else float("inf")))
    selected = close[0][0]
    # a selected structure whose extra stage is inactive is the simpler one with the same parameters
    params = result["structures"][selected]["params"]
    if selected == "M4" and params.sigma <= 0:
        selected = "M3"
        result["structures"]["M3"] = dict(result["structures"]["M4"], demoted_from="M4")
    if selected == "M3" and params.q is None:
        selected = "M2"
        result["structures"]["M2"] = dict(result["structures"]["M3"], demoted_from="M3")
    # a first-order stage that does not change the training fit beyond the noise floor is absent (tau 0): it is not
    # identified, so it must not be reported as a parameter
    params = result["structures"][selected]["params"]
    if selected != "M1" and params.tau > 0:
        trial = params.copy()
        trial.tau = 0.0
        trial_sse = score(train, trial, dt, selected, exact=True)
        info = result["structures"][selected]
        if trial_sse <= info["train_sse"] + info["train_n"] * floor_mse:
            test_sse, test_n = sse(test, trial, dt, selected)
            result["structures"][selected] = dict(info, params=trial, train_sse=trial_sse, tau_dropped=True,
                                                  test_mse=test_sse / test_n if test_n else None,
                                                  test_r2=r2(test, trial, dt, selected))
    result["selected"] = selected
    return result


def per_event_params(windows, dt, structure, start):
    """The selected structure fitted to every window alone (from the bin fit: delay, tau and the other stages
    searched again, the gain exact)"""
    return [fit_structure([w], dt, structure, start, joint=False)[0] for w in windows]


def cv(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    if abs(m) < 1e-12:
        return None
    return sc.stdev(values) / abs(m)


def reaches_target(window, params, dt, structure):
    """The simulated output comes within 10% of its target g x amplitude before the release (else a rate limit
    covers the whole on-time and the gain is not identified)"""
    rel = window.get("release_index")
    if rel is None or rel < 1:
        return False
    target = params.g * window["amplitude"]
    span = abs(target - window["y0"])
    if span <= 1e-12:
        return False
    sim = simulate(window, params, dt, structure)
    return any(abs(v - target) <= 0.1 * span for v in sim[:rel])


def stability(selected, params, per_event, windows, dt, step_size, max_cv):
    """Every identified parameter of the selected structure: {name: (state, text)} with state True (stable), False
    (CV over max_cv) or None (not resolved)"""
    out = {}
    gain_cv = cv([p.g for p in per_event])
    out["g"] = (None if gain_cv is None else gain_cv <= max_cv, "g CV %s" % sc.fmt(gain_cv, 3))
    if clamped(selected, params):
        reached = sum(1 for w, p in zip(windows, per_event) if reaches_target(w, p, dt, selected))
        if reached < RESOLVED_SHARE * len(per_event):
            out["g"] = (None, "g unresolved (the output reaches the target before the release in %d of %d events: "
                              "the rate limit hides the gain)" % (reached, len(per_event)))
    delays = [p.d for p in per_event]
    mean_d = sum(delays) / len(delays)
    if max(delays) == min(delays):
        out["d"] = (True, "d %d in every event" % delays[0])
    else:
        delay_cv = sc.stdev(delays) / mean_d if mean_d > 0 else None
        out["d"] = (delay_cv is not None and delay_cv <= max_cv, "d %d..%d (CV %s)" % (
            min(delays), max(delays), sc.fmt(delay_cv, 3)))
    if selected != "M1":
        resolution = dt / TAU_RESOLUTION
        active = [p.tau for p in per_event if p.tau >= resolution]
        if not active and params.tau < resolution:
            out["tau"] = (True, "tau < dt / ln 100 = %s s in the bin and every event (no first-order stage)" % sc.fmt(
                resolution, 3))
        else:
            out["tau"] = stage_stability("tau", [p.tau if p.tau >= resolution else None for p in per_event],
                                         max_cv)
    if selected in ("M3", "M4"):
        out["R"] = stage_stability("R", [p.q * abs(p.g) * step_size if p.q is not None else None
                                         for p in per_event], max_cv)
    if selected == "M4":
        out["sigma"] = stage_stability("sigma", [p.sigma if p.sigma > 0 else None for p in per_event], max_cv)
    return out


def stage_stability(name, values, max_cv):
    """(state, text) of a model stage's parameter over the events (None = the stage is inactive in that event): a
    stage present in some repetitions and absent in others is unresolved (None), never stable; present in all: its
    CV"""
    present = [v for v in values if v is not None]
    if len(present) < len(values):
        return None, "%s unresolved (the stage is active in %d of %d events)" % (name, len(present), len(values))
    value = cv(present)
    return (None if value is None else value <= max_cv), "%s CV %s" % (name, sc.fmt(value, 3))


# ---------------------------------------------------------------------------------------------------------------
# data


def path_value(step, path, channel, scale_rad):
    if path == "input":
        return step.f.get("input_post_steer")
    if path == "kappa":
        return step.kappa
    return sc.channel_angle(step, channel, scale_rad)


def run_baselines(steps, path, channel, scale_rad):
    """{(phase, run): the median output over the run's hold segment} (command 0, at speed: the at-rest level that
    every window of the run is measured from, so toe and self-steer offsets cancel)"""
    values = {}
    for step in steps:
        if step.segment != "hold":
            continue
        value = path_value(step, path, channel, scale_rad)
        if value is not None:
            values.setdefault((step.phase, step.i.get("run_idx")), []).append(value)
    return {key: sc.median(v) for key, v in values.items()}


def make_window(steps, first, last, path, channel, scale_rad, baselines, dt):
    """Scored rows steps[first..last]; the row before is the initial state, the commands of up to D_MAX engine steps
    before feed the delay. None without a baseline, an initial value or enough values."""
    if first < 1:
        return None
    rows = steps[first:last + 1]
    key = (rows[0].phase, rows[0].i.get("run_idx"))
    base = baselines.get(key)
    prev = steps[first - 1]
    if base is None or (prev.phase, prev.i.get("run_idx")) != key:
        return None
    y0 = path_value(prev, path, channel, scale_rad)
    if y0 is None:
        return None
    hist = []
    k = first - 1
    while k >= 0 and len(hist) < D_MAX:
        row = steps[k]
        if (row.phase, row.i.get("run_idx")) != key:
            break
        hist[0:0] = [row.f.get("cmd_steer") or 0.0] * sc.covered_steps(row, dt)
        k -= 1
    hist = hist[-D_MAX:]
    u, n, y = [], [], []
    for step in rows:
        cmd = step.f.get("cmd_steer")
        val = path_value(step, path, channel, scale_rad)
        u.append(cmd if cmd is not None else 0.0)
        n.append(sc.covered_steps(step, dt))
        y.append(None if val is None else val - base)
    if sum(1 for v in y if v is not None) < 0.8 * len(y):
        return None
    # every command change needs its readbacks: the first TRANSITION_ROWS rows after it identify the delay
    # and its first readback must come from a dispatch of exactly one engine step: after a merged dispatch delays d
    # and d + 1 give the same recorded endpoint (the runs so far had clock.merged_fixed_steps 0, so this costs none)
    gaps = 0
    merged = 0
    previous = hist[-1] if hist else (prev.f.get("cmd_steer") or 0.0)
    for k, cmd in enumerate(u):
        if abs(cmd - previous) > 1e-9:
            gaps += sum(1 for v in y[k:k + TRANSITION_ROWS] if v is None)
            merged += int(n[k] != 1)
        previous = cmd
    u_e = list(hist)
    for cmd, steps_n in zip(u, n):
        u_e.extend([cmd] * steps_n)
    speeds = [s.f.get("vel_l_z") for s in rows if s.f.get("vel_l_z") is not None]
    return {"u": u, "n": n, "y": y, "y0": y0 - base, "u_e": u_e, "offset": len(hist), "rows": rows,
            "v": max(sc.mean(speeds) or 0.0, 0.1), "gaps": gaps, "merged": merged}


def on_speed(kmh, target):
    """The planned speed when the measured mean speed is within SPEED_BIN_SHARE of it, else None (bins are keyed
    by the planned speed of the phase, validated by the measured one; the spec speeds are the CC-08 bins, other
    planned speeds only widen the input model's validated cells)"""
    if kmh is None or target is None or target <= 0:
        return None
    return round(target, 3) if abs(kmh - target) <= SPEED_BIN_SHARE * target else None


def event_windows(steps, events, path, channel, scale_rad, dt):
    """({(planned speed, |amplitude|): [window]}, {reason: count of events left out}, {key: (events left out for a
    missing readback near a command change or no usable signal, events whose command change is first observed
    after a merged dispatch)}): the ok events, each scored from its onset to its end, binned by their phase's
    planned speed when the measured mean speed of the scored rows is within SPEED_BIN_SHARE of it. A bin with either
    kind of event cannot be certified (analyse_bin: INCOMPLETE)"""
    by_step = {}
    for idx, step in enumerate(steps):
        number = step.i.get("physics_step")
        if number is not None:
            by_step[number] = idx
    baselines = run_baselines(steps, path, channel, scale_rad)
    bins = {}
    dropped = {}
    gaps = {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for event in events:
        if event.get("result") != "ok":
            drop("result_" + (event.get("result") or "none"))
            continue
        onset = probe_common.to_int(event.get("onset_step"))
        end = probe_common.to_int(event.get("end_step"))
        amplitude = probe_common.to_float(event.get("amplitude"))
        index = probe_common.to_int(event.get("event_idx"))
        if onset is None or end is None or amplitude is None or index is None or onset not in by_step \
                or end not in by_step or by_step[end] <= by_step[onset]:
            drop("incomplete_row")
            continue
        scored = steps[by_step[onset]:by_step[end] + 1]
        speed_kmh = sc.mean([s.f.get("speed_kmh") for s in scored if s.f.get("speed_kmh") is not None])
        spec = on_speed(speed_kmh, probe_common.to_float(event.get("target_kmh")))
        if spec is None:
            drop("off_speed")
            continue
        key = (spec, round(abs(amplitude), 3))
        window = make_window(steps, by_step[onset], by_step[end], path, channel, scale_rad, baselines, dt)
        if window is None or window["gaps"]:
            drop("no_signal" if window is None else "transition_gap")
            missing, merged = gaps.get(key, (0, 0))
            gaps[key] = (missing + 1, merged)
            continue
        if window["merged"]:
            missing, merged = gaps.get(key, (0, 0))
            gaps[key] = (missing, merged + 1)
        release = probe_common.to_int(event.get("release_step"))
        window.update({"step_size": abs(amplitude), "amplitude": amplitude, "speed_kmh": speed_kmh,
                       "release_index": by_step[release] - by_step[onset] if release in by_step else None,
                       "doublet": (event.get("phase"), event.get("run_idx"), index // 2)})
        bins.setdefault(key, []).append(window)
    return bins, dropped, gaps


def aborted_speeds(play):
    """{phase: (planned speed, result)} of the e23 phases that did not end ok"""
    out = {}
    targets = {}
    if play.has("e23_phases.csv"):
        for row in play.csv("e23_phases", ("phase", "target_kmh")):
            targets[row.get("phase", "")] = probe_common.to_float(row.get("target_kmh"))
    for name, result in sc.aborted_phases(play, "e23").items():
        target = targets.get(name)
        out[name] = (round(target, 3) if target is not None else None, result)
    return out


def prbs_status(play):
    """{phase: (bits, expected, bit_steps, result)} from e23_phases.csv (the probe's PRBS completion record)"""
    out = {}
    if not play.has("e23_phases.csv"):
        return out
    for row in play.csv("e23_phases", ("phase",)):
        out[row.get("phase", "")] = (probe_common.to_int(row.get("prbs_bits")),
                                     probe_common.to_int(row.get("prbs_expected")),
                                     probe_common.to_int(row.get("prbs_bit_steps")), row.get("prbs_result", ""))
    return out


def prbs_sequence(play):
    """The planned PRBS bits of one period (DC_ProbePrbs with params m_iPrbsOrder / m_iPrbsSeed), or None"""
    order = probe_common.to_int(play.param("m_iPrbsOrder"))
    seed = probe_common.to_int(play.param("m_iPrbsSeed"))
    if order not in PRBS_TAPS or seed is None or seed & ((1 << order) - 1) == 0:
        return None
    shifts = [order - tap for tap in PRBS_TAPS[order]]
    state = seed & ((1 << order) - 1)
    bits = []
    for _ in range((1 << order) - 1):
        bits.append(state & 1)
        feedback = 0
        for shift in shifts:
            feedback ^= (state >> shift) & 1
        state = (state >> 1) | (feedback << (order - 1))
    return bits


def applied_matches(rows, dt, bits, bit_steps):
    """(True, '') when the PRBS segment's dispatches replay the producer exactly (DC_ProbeSteerSteps: a bit is drawn
    at a dispatch boundary when no bit is open, applied for the covered steps of each dispatch, and closed once it
    has bit_steps engine steps): every dispatch's cmd_steer is the open planned bit's command (+A for 1, -A for 0,
    one amplitude), and exactly one period completes with nothing after it; else (False, why)"""
    if not rows or not bits:
        return False, "no PRBS rows or no planned bits"
    amplitude = None
    index = -1
    open_bit = False
    applied = 0
    completed = 0
    for k, s in enumerate(rows):
        cmd = s.f.get("cmd_steer")
        if cmd is None or abs(cmd) < 1e-9:
            return False, "dispatch %d without a non-zero command" % k
        if amplitude is None:
            amplitude = abs(cmd)
        elif abs(abs(cmd) - amplitude) > 1e-4:
            return False, "dispatch %d at amplitude %s, not %s" % (k, sc.fmt(abs(cmd)), sc.fmt(amplitude))
        if not open_bit:
            if index + 1 >= len(bits):
                return False, "dispatch %d after the period completed" % k
            index += 1
            open_bit = True
            applied = 0
        if (1 if cmd > 0 else 0) != bits[index]:
            return False, "dispatch %d applies %+d, planned bit %d is %d (a bit was skipped, added or reordered)" % (
                k, 1 if cmd > 0 else -1, index, bits[index])
        applied += sc.covered_steps(s, dt)
        if applied >= bit_steps:
            open_bit = False
            completed += 1
    if completed != len(bits) or open_bit:
        return False, "%d of %d planned bits completed" % (completed, len(bits))
    return True, ""


def prbs_windows(steps, path, channel, scale_rad, dt, status, planned=None):
    """{planned speed: window} of the PRBS segments that completed their whole sequence: the phase's prbs_result ok
    with bits == expected, the commands re-derived from cmd_steer equal to the planned bits (applied_matches;
    planned: prbs_sequence), every readback present near each bit change (make_window gaps) and the measured mean speed
    within the bin share. An aborted, truncated, distorted or partly unobserved PRBS is not a held-out set"""
    out = {}
    index = {id(s): i for i, s in enumerate(steps)}
    baselines = run_baselines(steps, path, channel, scale_rad)
    for phase, run, segment, rows in sc.segments(steps):
        if segment != "prbs" or len(rows) < 20:
            continue
        bits, expected, bit_steps, result = status.get(phase, (None, None, None, ""))
        if result != "ok" or not expected or bits != expected or not bit_steps:
            continue
        if sum(sc.covered_steps(s, dt) for s in rows) < expected * bit_steps:
            continue
        if planned is None or len(planned) != expected or not applied_matches(rows, dt, planned, bit_steps)[0]:
            continue
        begin = index.get(id(rows[0]), 0)
        end = index.get(id(rows[-1]), begin)
        window = make_window(steps, begin, end, path, channel, scale_rad, baselines, dt)
        if window is None or window["gaps"] or window["merged"]:
            continue
        speed_kmh = sc.mean([s.f.get("speed_kmh") for s in rows if s.f.get("speed_kmh") is not None])
        spec = on_speed(speed_kmh, rows[-1].f.get("seg_target"))
        if spec is None:
            continue
        amps = [abs(s.f.get("cmd_steer") or 0.0) for s in rows]
        window["step_size"] = max(amps) if amps else 1.0
        out[spec] = window
    return out


def plateau(window):
    """Mean output over the last 3 rows before the release, minus the output before the onset (None without a
    release index)."""
    rel = window.get("release_index")
    if rel is None or rel < 3:
        return None
    tail = [v for v in window["y"][rel - 3:rel] if v is not None]
    return sum(tail) / len(tail) - window["y0"] if tail else None


def centering(window):
    """Seconds from the release until the output is back within 10% of the plateau of its pre-onset value, and the
    remaining share of the plateau at the window end."""
    p = plateau(window)
    rel = window.get("release_index")
    if p is None or rel is None or abs(p) < 1e-9:
        return None, None
    t10 = None
    elapsed = 0.0
    for k in range(rel, len(window["y"])):
        val = window["y"][k]
        if val is not None and abs(val - window["y0"]) <= 0.1 * abs(p):
            t10 = elapsed
            break
        elapsed += window["rows"][k].f.get("timeslice_s") or 0.0
    last = [v for v in window["y"][-3:] if v is not None]
    residual = (sum(last) / len(last) - window["y0"]) / p if last else None
    return t10, residual


# ---------------------------------------------------------------------------------------------------------------
# analysis


def analyse_bin(windows, dt, args, step_size, gaps=(0, 0)):
    """{'state', 'selected', 'params', 'rate', 'detail', 'windows'} of one bin; gaps: (its events left out for a
    missing readback near a command change or no usable signal, its events whose command change is first observed
    after a merged dispatch). Either makes the bin INCOMPLETE whatever the fit (the fit is still reported)"""
    out = {"state": None, "selected": None, "params": None, "rate": None, "windows": windows}
    missing, merged = gaps
    if missing:
        out["detail"] = "%d events miss readbacks within %d rows of a command change or their initial state " \
                        "(the delay and dynamics are not observed)" % (missing, TRANSITION_ROWS)
        return out
    signs = sorted(set(1 if w["amplitude"] > 0 else -1 for w in windows))
    if len(windows) < args.min_events:
        out["detail"] = "%d events (< %d)" % (len(windows), args.min_events)
        return out
    fit = fit_bin(windows, dt)
    selected = fit["selected"]
    if selected is None:
        out["detail"] = "%d events in %d + %d doublets (needs %d on each side of the split)" % (
            len(windows), fit["train_doublets"], fit["test_doublets"], MIN_DOUBLETS)
        return out
    info = fit["structures"][selected]
    params = info["params"]
    per_event = per_event_params(windows, dt, selected, params)
    stable = stability(selected, params, per_event, windows, dt, step_size, args.max_cv)
    r2_ok = info["test_r2"] is not None and info["test_r2"] >= args.min_r2
    parts = [r2_ok] + [s for s, _ in stable.values()]
    state = sc.combine(parts)
    rate = params.q * abs(params.g) * step_size if clamped(selected, params) else None
    others = ", ".join("%s R2 %s BIC %s" % (s, sc.fmt(v["test_r2"], 4), sc.fmt(v["bic"], 6))
                       for s, v in fit["structures"].items())
    if merged:
        parts = parts + [None]
        state = None
    out.update({"state": state, "selected": selected, "params": params, "rate": rate, "test_r2": info["test_r2"],
                "per_event": per_event})
    out["detail"] = ("%s (%s): held-out R2 %s (>= %g) on %d doublets (trained on %d), d %d steps, g %s, tau %s s, "
                     "rate limit %s /s, sigma %s m; %s (<= %g); %d events, signs %s; others: %s" % (
                         selected, STRUCTURE_NAMES[selected], sc.fmt(info["test_r2"], 4), args.min_r2,
                         fit["test_doublets"], fit["train_doublets"], params.d, sc.fmt(params.g, 4),
                         sc.fmt(params.tau, 3), sc.fmt(rate, 4) if rate is not None else "none",
                         sc.fmt(params.sigma, 3) if selected == "M4" else "-",
                         "; ".join(text for _, text in stable.values()), args.max_cv, len(windows),
                         "/".join("%+d" % s for s in signs), others)) + sc.observed(parts)
    if merged:
        out["detail"] = ("delay not identifiable under merged dispatches (%d events with a command change first "
                         "observed after a dispatch of more than one engine step): " % merged) + out["detail"]
    return out


def channel_for(args, play, verdict):
    """(channel, scale_rad): the channel CC-07 step 0 validates on the e22 play (complete footer), with the signal
    scale; (None, None) otherwise"""
    if not args.e22:
        verdict.info("channel", "no --e22: the wheel-angle channel is not validated (the wheel bins are INCOMPLETE)")
        return None, None
    import e21_selfsteer  # noqa: E402  (step 0 lives with CC-07)
    load = probe_common.Play(args.e22)
    selection = e21_selfsteer.step0_selection(load, args, extra_play=play)
    verdict.info("channel", "step 0 (e22 %s): %s" % (load.dir, selection.get("detail")))
    ok, detail = sc.play_evidence(load, "e22")
    if not ok:
        verdict.info("channel_evidence", detail + ": the channel is not used")
        return None, None
    return selection.get("selected"), selection.get("scale_rad")


def prbs_check(results, path, speed, dt, args):
    window = results[(path, "prbs")].get(speed)
    if window is None:
        return None, "no completed PRBS at %g km/h (e23_phases prbs_result ok with every bit, the commands re-derived " \
                     "from cmd_steer replaying the planned bits, every readback near a bit change, every bit change " \
                     "observed on a single-step dispatch; measured speed within %d%%)" % (speed, SPEED_BIN_SHARE * 100)
    models = [(abs(amplitude - window["step_size"]), amplitude, results[(path, (speed, amplitude))])
              for amplitude, _ in SPEC_STEPS
              if (path, (speed, amplitude)) in results and results[(path, (speed, amplitude))]["selected"]]
    if not models:
        return None, "no bin model at %g km/h to predict the PRBS with" % speed
    models.sort(key=lambda m: m[0])
    _, amplitude, res = models[0]
    scored = dict(window)
    scored["step_size"] = amplitude       # the bin's rate limit in absolute units
    value = r2([scored], res["params"], dt, res["selected"])
    state = None if value is None else value >= args.min_r2
    return state, "held-out PRBS (amplitude %s, %d rows) predicted by the %g km/h x %g model %s: R2 %s (>= %g)" % (
        sc.fmt(window["step_size"], 3), len(window["rows"]), speed, amplitude, res["selected"], sc.fmt(value, 4),
        args.min_r2)


def analyse(play, args, verdict):
    plays = [("e23", play)]
    if args.e22:
        plays.append(("e22", probe_common.Play(args.e22)))
    sc.check_identity(plays)
    steps = sc.load_steps(play, "e23")
    events = play.csv("e23_events", EVENT_COLUMNS)
    dt = sc.step_dt(play)
    if dt is None:
        raise probe_common.InputError("%s: no physics step in the run header (dt_s, clock.step_dt_s)" % play.dir)
    states = [sc.play_evidence(p, name) for name, p in plays]
    verdict.check("evidence", sc.combine([s for s, _ in states]), "; ".join(d for _, d in states))
    channel, scale_rad = channel_for(args, play, verdict)
    status = prbs_status(play)
    aborted = aborted_speeds(play)
    verdict.info("phases", "not ok (their speeds are withheld): %s" % (
        ", ".join("%s %s" % (name, result) for name, (speed, result) in sorted(aborted.items())) or "none"))

    results = {}
    dropped_all = {}
    for path in RAW_PATHS + ("kappa",):
        if path == "wheel" and channel is None:
            continue
        bins, dropped, gaps = event_windows(steps, events, path, channel, scale_rad, dt)
        if path == "input":
            dropped_all = dropped
            input_bins = bins
            input_gaps = gaps
        for speed in SPEC_SPEEDS:
            for amplitude, _ in SPEC_STEPS:
                results[(path, (speed, amplitude))] = analyse_bin(bins.get((speed, amplitude), []), dt, args,
                                                                  amplitude, gaps.get((speed, amplitude), (0, 0)))
        results[(path, "prbs")] = prbs_windows(steps, path, channel, scale_rad, dt, status, prbs_sequence(play))

    # coverage against the spec grid (the input path's binning; the counts per sign)
    coverage_ok = True
    parts = []
    for speed in SPEC_SPEEDS:
        for amplitude, signs in SPEC_STEPS:
            windows = input_bins.get((speed, amplitude), [])
            for sign in signs:
                count = sum(1 for w in windows if (w["amplitude"] > 0) == (sign > 0))
                coverage_ok = coverage_ok and count >= SPEC_REPS
                parts.append("%g km/h %+g: %d" % (speed, sign * amplitude, count))
        has_prbs = speed in results[("input", "prbs")]
        coverage_ok = coverage_ok and has_prbs
        parts.append("%g km/h PRBS: %s" % (speed, "yes" if has_prbs else "no"))
    left_out = ", ".join("%s %d" % item for item in sorted(dropped_all.items())) or "none"
    verdict.check("coverage", True if coverage_ok else None, "%s against the spec grid (>= %d ok events per step "
                                                             "and sign, PRBS at every speed): %s; events left out: %s"
                  % ("FULL" if coverage_ok else "PREVIEW", SPEC_REPS, "; ".join(parts), left_out))
    verdict.check("llc_path", None, "not measured by e23: the full llc path (kappa target -> kappa_realised with "
                                    "llc_v1 in the loop), the llc limiter rule (identity or slower than the engine "
                                    "limit) and the 10 Hz ZOH kappa tracking are DD-7 checks; CC-08 stays "
                                    "INCOMPLETE until they exist")

    raw_states = []
    for path in RAW_PATHS:
        for speed in SPEC_SPEEDS:
            for amplitude, _ in SPEC_STEPS:
                name = "%s_%gkmh_%g" % (path, speed, amplitude)
                if path == "wheel" and channel is None:
                    verdict.check(name, None, "no wheel-angle channel validated by CC-07 step 0 (--e22)")
                    raw_states.append(None)
                    continue
                res = results[(path, (speed, amplitude))]
                verdict.check(name, res["state"], "PARTIAL (raw engine path): " + res["detail"])
                raw_states.append(res["state"])
        for speed in SPEC_SPEEDS:
            name = "prbs_%s_%gkmh" % (path, speed)
            if path == "wheel" and channel is None:
                verdict.check(name, None, "no validated wheel-angle channel")
                raw_states.append(None)
                continue
            state, text = prbs_check(results, path, speed, dt, args)
            verdict.check(name, state, text)
            raw_states.append(state)
    extra = sorted(key for key in input_bins if key[0] not in SPEC_SPEEDS and len(input_bins[key]) >= args.min_events)
    for key in extra:
        res = analyse_bin(input_bins[key], dt, args, key[1], input_gaps.get(key, (0, 0)))
        verdict.info("input_%gkmh_%g" % key, "outside the CC-08 grid (a validated cell of CC-07 (a)'s input model "
                                             "only): %s, %s" % (STATE_TEXT[res["state"]], res["detail"]))
    for speed in SPEC_SPEEDS:
        for amplitude, _ in SPEC_STEPS:
            res = results[("kappa", (speed, amplitude))]
            verdict.info("cmd_kappa_%gkmh_%g" % (speed, amplitude), "PARTIAL (cmd -> kappa_realised, not the llc "
                                                                   "path; not gated): %s, %s" % (
                                                                       STATE_TEXT[res["state"]], res["detail"]))
    report_extras(channel, results, verdict)
    # a run aborted by the motion guards (or a skipped phase) leaves its speed's evidence incomplete even when the
    # footer is complete
    for name, (speed, result) in sorted(aborted.items()):
        if speed is None:
            continue
        sc.withhold(verdict, [n for _, n, _ in verdict.items if n.startswith(("input_", "wheel_", "prbs_"))
                              and ("_%gkmh" % speed) in n], "phase %s ended %s" % (name, result))
    evidence_state = verdict.tag_of("evidence")
    if evidence_state != probe_common.PASS:
        sc.withhold(verdict, [name for _, name, _ in verdict.items if name.startswith(("input_", "wheel_", "prbs_"))]
                    + ["steer_sign"], "the play is not complete")
        raw_states.append(None)
    verdict.info("raw_engine_path", "PARTIAL %s (cmd -> steer_input -> wheel_angle; coverage %s); CC-08 as a whole "
                                    "is INCOMPLETE until the llc checks exist%s" % (
                                        STATE_TEXT[sc.combine(raw_states)], "FULL" if coverage_ok else "PREVIEW",
                                        sc.observed(raw_states)))
    verdict.info("channel_validated", "yes: %s (CC-07 step 0)" % channel if channel else "no")
    return results


def input_model(play, args):
    """The CC-08 cmd -> steer_input model for CC-07 (a)'s deterministic-dynamics exception: (state, model, detail).
    state True: every input bin with events passes (R2, every parameter stable and resolved), they select the same
    structure and no input PRBS fails; model = that structure with the median parameters (R absolute, units / s).
    False: an input bin or PRBS fails. None: no usable bin, an unresolved parameter, structures that differ."""
    ok, detail = sc.play_evidence(play, "e23")
    if not ok:
        return None, None, detail
    steps = sc.load_steps(play, "e23")
    events = play.csv("e23_events", EVENT_COLUMNS)
    dt = sc.step_dt(play)
    if dt is None:
        return None, None, "e23: no physics step in the run header"
    bins, _, gaps = event_windows(steps, events, "input", None, None, dt)
    fitted = []
    for key in sorted(set(bins) | set(gaps)):
        if len(bins.get(key, [])) + gaps.get(key, (0, 0))[0] >= args.min_events:
            fitted.append((key, analyse_bin(bins.get(key, []), dt, args, key[1], gaps.get(key, (0, 0)))))
    if not fitted:
        return None, None, "e23: no input bin with >= %d events" % args.min_events
    stopped = sorted("%s %s" % (name, result) for name, (speed, result) in aborted_speeds(play).items()
                     if speed in set(key[0] for key, _ in fitted))
    if stopped:
        return None, None, "e23 phases of the model's speeds did not end ok: %s" % ", ".join(stopped)
    texts = ["%g km/h x %g: %s %s" % (key[0], key[1], STATE_TEXT[res["state"]], res["detail"].split(";")[0])
             for key, res in fitted]
    states = [res["state"] for _, res in fitted]
    results = {("input", key): res for key, res in fitted}
    results[("input", "prbs")] = prbs_windows(steps, "input", None, None, dt, prbs_status(play), prbs_sequence(play))
    speeds = sorted(set(key[0] for key, _ in fitted))
    for speed in speeds:
        # the model is claimed at this speed only with a usable, passing held-out PRBS there
        state, text = prbs_check(results, "input", speed, dt, args)
        states.append(state)
        texts.append("PRBS %g km/h %s" % (speed, STATE_TEXT[state]))
    state = sc.combine(states)
    if state is not True:
        return state, None, "; ".join(texts) + sc.observed(states)
    structures = set(res["selected"] for _, res in fitted)
    if len(structures) != 1:
        return None, None, "the input bins select different structures (%s)" % ", ".join(sorted(structures))
    structure = structures.pop()
    if structure == "M4":
        return None, None, "M4 on the input path: the one-step check needs the hidden relaxation state"
    params = [res["params"] for _, res in fitted]
    model = {"structure": structure, "dt": dt,
             "d": int(round(sc.median([p.d for p in params]))),
             "g": sc.median([p.g for p in params]),
             "tau": sc.median([p.tau for p in params]),
             "rate": sc.median([res["rate"] for _, res in fitted]) if structure == "M3" else None}
    # scope: the validated cells only, each bin speed +-SPEED_BIN_SHARE with the largest amplitude fitted at that
    # speed; no gap filling between speeds and no extrapolation to standstill
    cells = []
    for speed in speeds:
        amp = max(key[1] for key, _ in fitted if key[0] == speed)
        cells.append((speed * (1.0 - SPEED_BIN_SHARE), speed * (1.0 + SPEED_BIN_SHARE), amp))
    model["scope"] = {"cells": cells, "text": "; ".join("%g..%g km/h |cmd| <= %g" % cell for cell in cells)}
    return True, model, "%s d %d g %s tau %s s rate %s, from %d e23 input bins with PRBS" % (
        structure, model["d"], sc.fmt(model["g"], 4), sc.fmt(model["tau"], 3), sc.fmt(model["rate"], 4), len(fitted))


def report_extras(channel, results, verdict):
    """steer_sign, rate limit vs speed, centering, L/R ratio and the script-setter limit."""
    signs = []
    for (path, key), res in results.items():
        if path != "kappa" or key == "prbs":
            continue
        for w in res["windows"]:
            p = plateau(w)
            if p is not None and w["amplitude"] != 0:
                signs.append(1 if p / w["amplitude"] > 0 else -1)
    if len(signs) >= 6:
        positive = sum(1 for s in signs if s > 0)
        sign = 1 if positive * 2 >= len(signs) else -1
        verdict.check("steer_sign", sign == 1, "%+d (%d of %d kappa plateaus with the command's sign; 1.8.0.13 and "
                                               "1.9.0.37 e01 measured +1: positive steering turns right)" % (
                                                   sign, positive, len(signs)))
    else:
        verdict.check("steer_sign", None, "%d kappa plateaus (< 6)" % len(signs))

    def bins_of(path):
        return sorted((key, res) for (p, key), res in results.items() if p == path and key != "prbs")

    rates = []
    for path in RAW_PATHS + ("kappa",):
        for key, res in bins_of(path):
            if res["selected"]:
                rates.append("%s %g km/h x %g: %s" % (path, key[0], key[1],
                                                      sc.fmt(res["rate"], 4) + "/s" if res["rate"] else "none"))
    verdict.info("rate_limit_vs_speed", "; ".join(rates) or "-")
    setter = [(key, res["rate"]) for key, res in bins_of("input") if res["rate"]]
    verdict.info("script_setter_rate_limit", "yes, cmd -> input is rate-limited (Maximum Steering Speed acts on "
                                             "the script setter): %s" % ", ".join(
                                                 "%g km/h x %g: %s/s" % (k[0], k[1], sc.fmt(r, 4)) for k, r in setter)
                 if setter else "no rate limit selected on cmd -> input")
    lines = []
    for path in RAW_PATHS:
        for key, res in bins_of(path):
            if not res["windows"]:
                continue
            t10 = [centering(w)[0] for w in res["windows"]]
            residual = [centering(w)[1] for w in res["windows"]]
            lines.append("%s %g km/h x %g: t10 %s s, residual %s" % (
                path, key[0], key[1], sc.fmt(sc.median([v for v in t10 if v is not None]), 3),
                sc.fmt(sc.median([v for v in residual if v is not None]), 3)))
    verdict.info("centering_on_release", "; ".join(lines) or "-")
    ratios = []
    if channel and channel != "sig":
        for key, res in bins_of("wheel"):
            per = []
            for w in res["windows"]:
                rel = w.get("release_index")
                if rel is None or rel < 3:
                    continue
                left = [sc.channel_pair(s, channel)[0] for s in w["rows"][rel - 3:rel]]
                right = [sc.channel_pair(s, channel)[1] for s in w["rows"][rel - 3:rel]]
                left = [v for v in left if v is not None]
                right = [v for v in right if v is not None]
                if not left or not right:
                    continue
                dl, dr = sum(left) / len(left), sum(right) / len(right)
                inner, outer = (dr, dl) if w["amplitude"] > 0 else (dl, dr)
                if abs(outer) > 1e-4:
                    per.append(abs(inner) / abs(outer))
            if per:
                ratios.append("%g km/h x %g: inner/outer %s" % (key[0], key[1], sc.fmt(sc.median(per), 4)))
    verdict.info("lr_angle_ratio", "; ".join(ratios) or "- (no wheel pair)")


def add_step0_args(parser):
    parser.add_argument("--signal-scale-deg", type=float, default=33.75)
    parser.add_argument("--max-step0-error", type=float, default=0.05)
    parser.add_argument("--max-lag-steps", type=float, default=1.0)
    parser.add_argument("--max-unmeasurable", type=float, default=0.2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("play_dir")
    parser.add_argument("--e22", help="e22 play dir: the wheel-angle channel validated by CC-07 step 0")
    parser.add_argument("--min-r2", type=float, default=0.9)
    parser.add_argument("--max-cv", type=float, default=0.2)
    parser.add_argument("--min-events", type=int, default=6)
    add_step0_args(parser)
    args = parser.parse_args(argv)
    play = probe_common.Play(args.play_dir)
    if play.probe and play.probe != "e23":
        raise probe_common.InputError("%s is a %s play, not e23" % (play.dir, play.probe))
    verdict = sc.StrictVerdict("CC-08 steering actuator structure (e23)", play.dir)
    analyse(play, args, verdict)
    verdict.report()
    return 0


if __name__ == "__main__":
    sys.exit(sc.run_cli(main))
