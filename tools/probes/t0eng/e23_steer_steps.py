#!/usr/bin/env python3
"""CC-08 steering actuator structure from the S2 steering steps of probe e23 (DD-14b, DC_ProbeSteerSteps).

    python3 -I -B tools/probes/t0eng/e23_steer_steps.py <e23 play_dir> [--e22 <e22 play_dir>] [--channel auto]
                                                        [--min-r2 0.9] [--max-cv 0.2] [--min-events 6]

<play_dir> is one $profile:DC_Probes/e23/<launch_id>/play_NN directory (run_header.txt, e23_steps.csv,
e23_events.csv, e23_phases.csv, run_footer.txt). Nothing is written.

MIGRATION_1_9.md 6.1 CC-08: steps 0 -> +-0.5 and 0 -> +1.0 at 10/30/60 (+80) km/h, repetitions with random phase,
plus a held-out PRBS; competing structures fitted per speed and amplitude bin, separately for the raw engine path and
the full kappa path; pass: the selected structure (lowest held-out error, BIC as tie-break) reaches held-out
R2 >= 0.9 in every bin with parameters stable across repetitions (CV <= 20%); the rate limit as a function of speed,
the centering on release, the L/R angle ratio, steer_sign and whether a steering-speed limit acts on the script setter
are recorded.

Paths (one fit per path, speed and |amplitude| bin, both signs together):
  input   cmd_steer -> input_post_steer (the engine's steering input after the step)
  wheel   cmd_steer -> delta_eff of the wheel-angle channel (steer_common.delta_eff; the channel validated by CC-07
          step 0 when --e22 is given, else the first candidate with data, reported unvalidated)
  kappa   cmd_steer -> kappa_realised = angvel_l_y / vel_l_z (the llc path with the gain as a parameter)
Structures (discrete, one physics step dt; target = g x u[k - d]):
  M1 delay           y = target
  M2 + first order   y1 += (1 - a) (target - y1), a = exp(-dt / tau)
  M3 + rate limit    the M2 step clamped to +-R dt (R as a share of the bin's step size per second, reported in
                     output units / s)
  M4 + relaxation    M3 then a second first order with tau2 = sigma / v (tyre relaxation length sigma)
Windows: from 3 steps before each onset to the end of the event's centering gap, the output measured from the
run's hold baseline (command 0 at speed) and every model stage starting at the measured value, so the state the
previous event left is the initial condition, not an error.
g is the least-squares gain of each candidate (closed form: the output scales with g; the free response of the
initial state is separated). Fit on the even events of the bin: a grid over d (0..8 steps) and tau; for M3 / M4 a
joint grid of tau with R (M3) or sigma (M4) over the delay and up to 3 steps less; coordinate descent; then
golden-section refinement of tau, R and sigma between their grid neighbours, at the best delay and at +-1 step.
Held-out = the odd events. Selection: the lowest held-out MSE; within 2% of it (or within the noise floor, 1e-4 of
the output variance), the lowest BIC (n ln(SSE / n) + k ln n on the training events, the SSE floored at that noise
floor, so the parameter count decides between candidates that both reach it). Parameter CV:
the selected structure fitted to each event alone (d and tau from the bin start); CV = stdev / |mean| per continuous
parameter (g always, tau when the bin tau is >= one step, R when finite, sigma when > 0); the delay as a range.
A planned bin (params m_sStepGrid) with fewer than --min-events ok events, a missing path signal or a wrong
steer_sign fails, never passes.
Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402
import steer_common as sc  # noqa: E402

PRE_STEPS = 3
D_MAX = 8
TAU_GRID_S = [0.0] + [0.017 * (1.5 / 0.017) ** (i / 11.0) for i in range(12)]
Q_GRID = [None] + [0.3 * (60.0 / 0.3) ** (i / 13.0) for i in range(14)]     # rate limit: steps per second
SIGMA_GRID_M = [0.0] + [0.05 * (5.0 / 0.05) ** (i / 9.0) for i in range(10)]
TIE_SHARE = 0.02
NOISE_FLOOR_SHARE = 1e-4      # a held-out MSE below this share of the output variance (R2 >= 0.9999) is a tie
STRUCTURES = ("M1", "M2", "M3", "M4")
STRUCTURE_NAMES = {"M1": "delay", "M2": "delay+first order", "M3": "delay+first order+rate limit",
                   "M4": "delay+first order+rate limit+relaxation"}
PARAM_COUNT = {"M1": 2, "M2": 3, "M3": 4, "M4": 5}
PATHS = ("input", "wheel", "kappa")
EVENT_COLUMNS = ("event_idx", "phase", "target_kmh", "amplitude", "run_idx", "jitter_steps", "onset_step",
                 "release_step", "end_step", "onset_kmh", "result")


# ---------------------------------------------------------------------------------------------------------------
# models


class Params:
    """One candidate: delay d (steps), gain g, tau (s), q (rate limit in step sizes per second, None = none),
    sigma (relaxation length m, 0 = none)."""

    __slots__ = ("d", "g", "tau", "q", "sigma")

    def __init__(self, d=0, g=1.0, tau=0.0, q=None, sigma=0.0):
        self.d, self.g, self.tau, self.q, self.sigma = d, g, tau, q, sigma

    def copy(self):
        return Params(self.d, self.g, self.tau, self.q, self.sigma)


def simulate(window, params, dt, structure, with_input=True, with_state=True):
    """Output of a structure over one window: u (the command), y0 (the measured output at the first row, the
    initial state of every stage), u_hist (the commands of the D_MAX rows before the window, oldest first), v,
    step_size. with_input / with_state: the forced and the free response alone (their sum is the output of the
    linear structures; the gain fit uses them)."""
    u = window["u"]
    hist = window.get("u_hist", [])
    a = math.exp(-dt / params.tau) if params.tau > 0 else 0.0
    rate = None
    if structure in ("M3", "M4") and params.q is not None:
        rate = params.q * abs(params.g) * window["step_size"] * dt
    b = 0.0
    if structure == "M4" and params.sigma > 0 and window["v"] > 0.1:
        b = math.exp(-dt * window["v"] / params.sigma)
    y0 = window.get("y0", 0.0) if with_state else 0.0
    y1 = y0
    y2 = y0
    out = []
    for k in range(len(u)):
        j = k - params.d
        if not with_input:
            command = 0.0
        elif j >= 0:
            command = u[j]
        else:
            command = hist[j] if -j <= len(hist) else 0.0
        target = params.g * command
        if structure == "M1":
            y = target
        else:
            step = (1.0 - a) * (target - y1)
            if rate is not None:
                step = max(-rate, min(rate, step))
            y1 += step
            y = y1
            if structure == "M4" and b > 0:
                y2 = b * y2 + (1.0 - b) * y1
                y = y2
        out.append(y)
    return out


def pairs(window, sim):
    """(simulated, measured) of the rows with a measured value"""
    return [(ys, ym) for ys, ym in zip(sim, window["y"]) if ym is not None]


def sse(windows, params, dt, structure):
    total = 0.0
    n = 0
    for w in windows:
        for ys, ym in pairs(w, simulate(w, params, dt, structure)):
            total += (ys - ym) ** 2
            n += 1
    return total, n


def score(windows, params, dt, structure):
    """SSE of a candidate with its least-squares gain (set on params). The output is the forced response (linear
    in g; the rate limit is a share of g times the step size) plus the free response of the measured initial state:
    g = sum(A (y - B)) / sum(A A) from the unit-gain forced response A and the free response B, then the SSE of a
    full simulation with that g (exact also where the clamp couples the two)."""
    unit = params.copy()
    unit.g = 1.0
    num = 0.0
    den = 0.0
    parts = []
    for w in windows:
        forced = simulate(w, unit, dt, structure, with_state=False)
        free = simulate(w, unit, dt, structure, with_input=False)
        parts.append((forced, free, w["y"]))
        for fa, fb, ym in zip(forced, free, w["y"]):
            if ym is None:
                continue
            num += fa * (ym - fb)
            den += fa * fa
    params.g = num / den if den > 0 else 0.0
    clamped = structure in ("M3", "M4") and params.q is not None
    if clamped:
        return sse(windows, params, dt, structure)[0]
    total = 0.0
    for forced, free, measured in parts:
        for fa, fb, ym in zip(forced, free, measured):
            if ym is not None:
                total += (params.g * fa + fb - ym) ** 2
    return total


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
    sigma between their grid neighbours (so a richer structure never wins by the grid alone); returns
    (Params, SSE, n). joint: the two-parameter grids of M3 / M4 (the bin fits; the per-event fits start next to the
    bin optimum and skip them)."""
    best = None
    best_sse = None

    def consider(p):
        nonlocal best, best_sse
        value = score(windows, p, dt, structure)
        if best_sse is None or value < best_sse - 1e-15:
            best, best_sse = p.copy(), value

    if start is None:
        taus = [0.0] if structure == "M1" else TAU_GRID_S
        for d in range(D_MAX + 1):
            for tau in taus:
                consider(Params(d=d, tau=tau))
    else:
        consider(start.copy())
    # joint grids of the two parameters that trade off against each other (a lag can mimic a rate limit or a
    # relaxation), then coordinate descent over all of them
    # an added stage takes over part of the dead time, so the joint grids also try up to 3 steps less delay
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
        """Rounds of golden-section refinement of the continuous parameters from params; (params, SSE)"""
        current = params.copy()
        current_sse = score(windows, current, dt, structure)
        names = []
        if structure != "M1":
            names.append(("tau", TAU_GRID_S))
        if structure in ("M3", "M4") and current.q is not None:
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
                    return score(windows, p, dt, structure)

                x, value = golden(evaluate, lo, hi)
                if value < current_sse:
                    setattr(current, name, x)
                    current_sse = score(windows, current, dt, structure)
        return current, current_sse

    # continuous refinement at the best delay and at its two neighbours (a refined lag or rate limit can move the
    # best dead time by a step)
    refined, refined_sse = refine_from(best)
    for d in (refined.d - 1, refined.d + 1):
        if d < 0 or d > D_MAX or structure == "M1":
            continue
        trial = refined.copy()
        trial.d = d
        candidate, candidate_sse = refine_from(trial)
        if candidate_sse < refined_sse:
            refined, refined_sse = candidate, candidate_sse
    consider(refined)
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


def fit_bin(windows, dt):
    """All four structures on the even windows, held-out on the odd ones; the selection and its numbers."""
    train = windows[0::2]
    test = windows[1::2]
    floor_mse = NOISE_FLOOR_SHARE * variance(train)
    result = {"structures": {}, "floor_mse": floor_mse}
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
        result["selected"] = None
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
    result["selected"] = selected
    return result


def per_event_params(windows, dt, structure, start):
    """The selected structure fitted to every window alone (coordinate descent from the bin fit)."""
    out = []
    for w in windows:
        params, _, _ = fit_structure([w], dt, structure, start, joint=False)
        if structure == "M2":
            # M2 has no descent stage in fit_structure: refine tau and d around the bin start
            best, best_sse = params, score([w], params, dt, structure)
            for tau in TAU_GRID_S:
                p = best.copy()
                p.tau = tau
                value = score([w], p, dt, structure)
                if value < best_sse:
                    best, best_sse = p.copy(), value
            for d in range(D_MAX + 1):
                p = best.copy()
                p.d = d
                value = score([w], p, dt, structure)
                if value < best_sse:
                    best, best_sse = p.copy(), value
            params = best
        elif structure == "M1":
            best, best_sse = params, score([w], params, dt, structure)
            for d in range(D_MAX + 1):
                p = best.copy()
                p.d = d
                value = score([w], p, dt, structure)
                if value < best_sse:
                    best, best_sse = p.copy(), value
            params = best
        out.append(params)
    return out


def cv(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    if abs(m) < 1e-12:
        return None
    return sc.stdev(values) / abs(m)


# ---------------------------------------------------------------------------------------------------------------
# data


def path_value(step, path, channel):
    if path == "input":
        return step.f.get("input_post_steer")
    if path == "kappa":
        return step.kappa
    return sc.channel_angle(step, channel)


def run_baselines(steps, path, channel):
    """{(phase, run): the median output over the run's hold segment} (command 0, at speed: the at-rest level that
    every window of the run is measured from, so toe and self-steer offsets cancel)"""
    values = {}
    for step in steps:
        if step.segment != "hold":
            continue
        value = path_value(step, path, channel)
        if value is not None:
            values.setdefault((step.phase, step.i.get("run_idx")), []).append(value)
    return {key: sc.median(v) for key, v in values.items()}


def make_window(steps, first, last, path, channel, baselines):
    """u (the command), y (the output minus the run's baseline), y0, u_hist over steps[first..last]; None without a
    baseline, an initial value or enough values"""
    rows = steps[first:last + 1]
    base = baselines.get((rows[0].phase, rows[0].i.get("run_idx")))
    if base is None:
        return None
    u = []
    y = []
    for step in rows:
        cmd = step.f.get("cmd_steer")
        val = path_value(step, path, channel)
        u.append(cmd if cmd is not None else 0.0)
        y.append(None if val is None else val - base)
    if y[0] is None or sum(1 for v in y if v is not None) < 0.8 * len(y):
        return None
    hist = []
    for k in range(max(0, first - D_MAX), first):
        prev = steps[k]
        same = prev.phase == rows[0].phase and prev.i.get("run_idx") == rows[0].i.get("run_idx")
        hist.append(prev.f.get("cmd_steer") or 0.0 if same else 0.0)
    return {"u": u, "y": y, "y0": y[0], "u_hist": hist, "rows": rows}


def event_windows(steps, events, path, channel):
    """{(speed, |amplitude|): [window]} of the ok events: from PRE_STEPS before the onset to the event's end step,
    in deviation from the run's hold baseline, the model state starting at the measured value (so what the previous
    event left is the initial condition, not an error)."""
    by_step = {}
    for idx, step in enumerate(steps):
        number = step.i.get("physics_step")
        if number is not None:
            by_step[number] = idx
    baselines = run_baselines(steps, path, channel)
    bins = {}
    for event in events:
        if event.get("result") != "ok":
            continue
        onset = probe_common.to_int(event.get("onset_step"))
        end = probe_common.to_int(event.get("end_step"))
        speed = probe_common.to_float(event.get("target_kmh"))
        amplitude = probe_common.to_float(event.get("amplitude"))
        if onset is None or end is None or speed is None or amplitude is None or onset not in by_step:
            continue
        first = by_step[onset] - PRE_STEPS
        last = by_step.get(end)
        if first < 0 or last is None or last <= by_step[onset]:
            continue
        window = make_window(steps, first, last, path, channel, baselines)
        if window is None:
            continue
        rows = window["rows"]
        v_ms = rows[PRE_STEPS].f.get("vel_l_z") or speed / 3.6
        window.update({"v": max(v_ms, 0.1), "step_size": abs(amplitude), "amplitude": amplitude,
                       "onset_index": PRE_STEPS, "release_index": None})
        release = probe_common.to_int(event.get("release_step"))
        if release is not None and release in by_step:
            window["release_index"] = by_step[release] - first
        key = (round(speed, 3), round(abs(amplitude), 3))
        bins.setdefault(key, []).append(window)
    return bins


def prbs_windows(steps, path, channel):
    """{speed: window} of the PRBS segments, like the event windows (the run's hold baseline, the measured initial
    state)."""
    out = {}
    index = {id(s): i for i, s in enumerate(steps)}
    baselines = run_baselines(steps, path, channel)
    for phase, run, segment, rows in sc.segments(steps):
        if segment != "prbs" or len(rows) < 20:
            continue
        begin = index.get(id(rows[0]), 0)
        end = index.get(id(rows[-1]), begin)
        speed = rows[-1].f.get("seg_target")
        window = make_window(steps, begin, end, path, channel, baselines)
        if window is None or speed is None:
            continue
        window["v"] = max(rows[-1].f.get("vel_l_z") or speed / 3.6, 0.1)
        window["step_size"] = 1.0
        out[round(speed, 3)] = window
    return out


def planned_bins(play):
    """(speed, |amplitude|) bins of params m_sStepGrid."""
    text = play.param("m_sStepGrid", "") or ""
    out = []
    for group in text.split(";"):
        if ":" not in group:
            continue
        speed_text, specs = group.split(":", 1)
        speed = probe_common.to_float(speed_text.strip())
        for spec in specs.split(","):
            if "x" not in spec:
                continue
            amp = probe_common.to_float(spec.split("x", 1)[0].strip())
            if speed is not None and amp is not None:
                out.append((round(speed, 3), round(abs(amp), 3)))
    return sorted(set(out))


def choose_channel(steps, events, requested):
    """The wheel-angle channel: requested if given, else the first of rot / bone / ent with data on >= 90% of the
    event rows"""
    if requested and requested != "auto":
        return requested
    rows = steps
    for key, left, right in sc.CHANNELS:
        if right is None:
            continue
        valid = sum(1 for s in rows if s.f.get(left) is not None and s.f.get(right) is not None)
        if rows and valid >= 0.9 * len(rows):
            return key
    return None


def base_value(window, values=None):
    """The output just before the onset (the reference of the plateau and of the centering)"""
    values = values or window["y"]
    for k in range(window["onset_index"] - 1, -1, -1):
        if values[k] is not None:
            return values[k]
    return None


def plateau(window, values=None):
    """Mean output over the last 3 steps before the release, minus the output before the onset (None without a
    release index)."""
    rel = window.get("release_index")
    values = values or window["y"]
    base = base_value(window, values)
    if rel is None or base is None or rel - 3 < window["onset_index"]:
        return None
    tail = [v for v in values[rel - 3:rel] if v is not None]
    return sum(tail) / len(tail) - base if tail else None


def centering(window):
    """Seconds from the release until the output is back within 10% of the plateau of its pre-onset value, and the
    remaining share of the plateau at the window end."""
    p = plateau(window)
    rel = window.get("release_index")
    base = base_value(window)
    if p is None or rel is None or base is None or abs(p) < 1e-9:
        return None, None
    rows = window["rows"]
    dt = sum((r.f.get("timeslice_s") or 0.0) for r in rows) / max(1, len(rows))
    t10 = None
    for k in range(rel, len(window["y"])):
        val = window["y"][k]
        if val is not None and abs(val - base) <= 0.1 * abs(p):
            t10 = (k - rel) * dt
            break
    last = [v for v in window["y"][-3:] if v is not None]
    residual = (sum(last) / len(last) - base) / p if last else None
    return t10, residual


# ---------------------------------------------------------------------------------------------------------------
# analysis


def analyse(play, args, verdict):
    steps = sc.load_steps(play, "e23")
    events = play.csv("e23_events", EVENT_COLUMNS)
    dt = probe_common.to_float(play.param("dt_s"), None) or sc.median([s.f.get("timeslice_s") for s in steps
                                                                         if s.f.get("timeslice_s")]) or 1.0 / 30
    planned = planned_bins(play)
    channel = None
    validated = False
    if args.e22:
        import e21_selfsteer  # noqa: E402  (step 0 lives with CC-07)
        load = probe_common.Play(args.e22)
        selection = e21_selfsteer.step0_selection(load, args, extra_play=play)
        channel = selection.get("selected")
        validated = channel is not None
        verdict.info("channel", "step 0 (e22 %s): %s" % (load.dir, selection.get("detail")))
    if channel is None:
        channel = choose_channel(steps, events, args.channel)
        verdict.info("channel", "%s (%s)" % (channel, "requested" if args.channel not in (None, "auto")
                                             else "first candidate with data; NOT validated by CC-07 step 0, pass "
                                                  "--e22"))
    ok_events = [e for e in events if e.get("result") == "ok"]
    verdict.info("events", "%d rows, %d ok; planned bins %s" % (
        len(events), len(ok_events), ", ".join("%g km/h x %g" % b for b in planned) or "-"))
    if not planned:
        verdict.check("bins", False, "no planned bin (params m_sStepGrid missing or empty)")

    selected_info = {}
    all_ok = True
    for path in PATHS:
        if path == "wheel" and channel is None:
            verdict.check("path_wheel", False, "no wheel-angle channel with data (no data -> FAIL)")
            all_ok = False
            continue
        bins = event_windows(steps, events, path, channel)
        for key in planned:
            windows = bins.get(key, [])
            name = "%s_%gkmh_%g" % (path, key[0], key[1])
            if len(windows) < args.min_events:
                verdict.check(name, False, "%d ok events with a %s signal (< %d): no data" % (
                    len(windows), path, args.min_events))
                all_ok = False
                continue
            fit = fit_bin(windows, dt)
            selected = fit["selected"]
            if selected is None:
                verdict.check(name, False, "no held-out events")
                all_ok = False
                continue
            info = fit["structures"][selected]
            params = info["params"]
            per_event = per_event_params(windows, dt, selected, params)
            cvs = {"g": cv([p.g for p in per_event])}
            if selected != "M1" and params.tau >= dt:
                cvs["tau"] = cv([max(p.tau, dt * 0.5) for p in per_event])
            if selected in ("M3", "M4") and params.q is not None:
                cvs["R"] = cv([p.q * abs(p.g) for p in per_event if p.q is not None])
            if selected == "M4" and params.sigma > 0:
                cvs["sigma"] = cv([max(p.sigma, 0.01) for p in per_event])
            delays = [p.d for p in per_event]
            worst_cv = max([c for c in cvs.values() if c is not None], default=None)
            r2_ok = info["test_r2"] is not None and info["test_r2"] >= args.min_r2
            cv_ok = worst_cv is not None and worst_cv <= args.max_cv
            if cvs.get("g") is None:
                cv_ok = False
            rate = None
            if selected in ("M3", "M4") and params.q is not None:
                rate = params.q * abs(params.g) * key[1]
            others = ", ".join("%s R2 %s BIC %s" % (s, sc.fmt(v["test_r2"], 4), sc.fmt(v["bic"], 6))
                               for s, v in fit["structures"].items())
            detail = ("%s (%s): held-out R2 %s (>= %g), d %d steps (events %d..%d), g %s, tau %s s, rate limit %s "
                      "/s, sigma %s m; CV %s (<= %g); %d events; others: %s" % (
                          selected, STRUCTURE_NAMES[selected], sc.fmt(info["test_r2"], 4), args.min_r2, params.d,
                          min(delays), max(delays), sc.fmt(params.g, 4), sc.fmt(params.tau, 3),
                          sc.fmt(rate, 4) if rate is not None else "none",
                          sc.fmt(params.sigma, 3) if selected == "M4" else "-",
                          ", ".join("%s %s" % (k, sc.fmt(v, 3)) for k, v in cvs.items()), args.max_cv,
                          len(windows), others))
            verdict.check(name, r2_ok and cv_ok, detail)
            all_ok = all_ok and r2_ok and cv_ok
            selected_info[(path, key)] = (selected, params, rate, windows)

    report_extras(play, steps, channel, dt, selected_info, verdict, args)
    verdict.info("mode", "CC-08 %s" % ("PASS: every bin of every path" if all_ok else "FAIL (see the bins)"))
    verdict.info("llc_limiter_rule", "llc_v1's limiter vs the identified engine limit and the 10 Hz ZOH kappa "
                                     "tracking are DD-7 checks (they need llc_v1 in the loop); the engine limit per "
                                     "speed is the rate_limit_vs_speed line")
    verdict.info("channel_validated", "yes (CC-07 step 0)" if validated else "no")


def report_extras(play, steps, channel, dt, selected_info, verdict, args):
    """steer_sign, rate limit vs speed, centering, L/R ratio, the script-setter limit and the PRBS held-out."""
    # steer_sign: sign of the kappa plateau over the command
    signs = []
    for (path, key), (selected, params, rate, windows) in selected_info.items():
        if path != "kappa":
            continue
        for w in windows:
            p = plateau(w)
            if p is not None and w["amplitude"] != 0:
                signs.append(1 if p / w["amplitude"] > 0 else -1)
    if signs:
        positive = sum(1 for s in signs if s > 0)
        sign = 1 if positive * 2 >= len(signs) else -1
        verdict.check("steer_sign", sign == 1, "%+d (%d of %d kappa plateaus with the command's sign; 1.8.0.13 and "
                                               "1.9.0.37 e01 measured +1: positive steering turns right)" % (
                                                   sign, positive, len(signs)))
    else:
        verdict.check("steer_sign", False, "no kappa plateau (no data)")

    rates = []
    for (path, key), (selected, params, rate, windows) in sorted(selected_info.items()):
        rates.append("%s %g km/h x %g: %s" % (path, key[0], key[1], sc.fmt(rate, 4) + "/s" if rate else "none"))
    verdict.info("rate_limit_vs_speed", "; ".join(rates) or "-")
    setter = [(key, rate) for (path, key), (selected, params, rate, windows) in selected_info.items()
              if path == "input" and rate]
    verdict.info("script_setter_rate_limit", "yes, cmd -> input is rate-limited (Maximum Steering Speed acts on "
                                             "the script setter): %s" % ", ".join(
                                                 "%g km/h x %g: %s/s" % (k[0], k[1], sc.fmt(r, 4)) for k, r in setter)
                 if setter else "no rate limit selected on cmd -> input")

    lines = []
    for (path, key), (selected, params, rate, windows) in sorted(selected_info.items()):
        if path == "kappa":
            continue
        t10 = [centering(w)[0] for w in windows]
        res = [centering(w)[1] for w in windows]
        lines.append("%s %g km/h x %g: t10 %s s, residual %s" % (
            path, key[0], key[1], sc.fmt(sc.median([v for v in t10 if v is not None]), 3),
            sc.fmt(sc.median([v for v in res if v is not None]), 3)))
    verdict.info("centering_on_release", "; ".join(lines) or "-")

    ratios = []
    if channel and channel != "sig":
        for (path, key), (selected, params, rate, windows) in sorted(selected_info.items()):
            if path != "wheel":
                continue
            per = []
            for w in windows:
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
            ratios.append("%g km/h x %g: inner/outer %s" % (key[0], key[1], sc.fmt(sc.median(per), 4)))
    verdict.info("lr_angle_ratio", "; ".join(ratios) or "- (no wheel pair)")

    prbs_lines = []
    for path in PATHS:
        if path == "wheel" and channel is None:
            continue
        for speed, window in sorted(prbs_windows(steps, path, channel).items()):
            choices = [(abs(key[1] - window["step_size"]), key, info) for (p, key), info in selected_info.items()
                       if p == path and key[0] == speed]
            if not choices:
                continue
            choices.sort(key=lambda c: c[0])
            _, key, (selected, params, rate, _) = choices[0]
            window = dict(window)
            window["step_size"] = key[1]          # the bin's rate limit in absolute units
            prbs_lines.append("%s %g km/h (model of x %g, %s): R2 %s" % (
                path, speed, key[1], selected, sc.fmt(r2([window], params, dt, selected), 4)))
    verdict.info("prbs_held_out", "; ".join(prbs_lines) or "- (no PRBS segment)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("play_dir")
    parser.add_argument("--e22", help="e22 play dir: the wheel-angle channel validated by CC-07 step 0")
    parser.add_argument("--channel", default="auto", choices=("auto", "rot", "bone", "ent"))
    parser.add_argument("--min-r2", type=float, default=0.9)
    parser.add_argument("--max-cv", type=float, default=0.2)
    parser.add_argument("--min-events", type=int, default=6)
    parser.add_argument("--signal-scale-deg", type=float, default=33.75)
    parser.add_argument("--max-step0-error", type=float, default=0.05)
    parser.add_argument("--max-lag-steps", type=float, default=1.0)
    args = parser.parse_args(argv)
    play = probe_common.Play(args.play_dir)
    verdict = probe_common.Verdict("CC-08 steering actuator structure (e23)", play.dir)
    verdict.check_footer(play)
    if play.probe and play.probe != "e23":
        raise probe_common.InputError("%s is a %s play, not e23" % (play.dir, play.probe))
    analyse(play, args, verdict)
    return verdict.report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
