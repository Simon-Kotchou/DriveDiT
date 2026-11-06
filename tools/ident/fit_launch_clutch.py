#!/usr/bin/env python3
"""Launch-clutch analysis of an E-01 play: what the clutch did, the powertrain fit and its replay (DD-14d).

Run it with the isolated interpreter (stdlib only):

    python3 -I -B tools/ident/fit_launch_clutch.py --e01 <e01 play_dir> [--fit] [--replay] [--acceptance]

<play_dir> is one $profile:DC_Probes/e01/<launch_id>/play_NN directory of the probe kit (run_header.txt,
e01_steps.csv). Nothing is written. Sealed map: a path or run header naming eden / everon is refused (exit 2).

ALWAYS (the check of a Play): the launch clutch of the run header (launch_clutch.mode and its fields), then per launch
phase (move_check, throttle_*, gear_scripted; LAUNCH_PHASES) the openings (cmd_clutch > 0 on one step and 0 on the
next while the throttle is on: the DD-14c chatter's dumps) with their step offset and speed, the first cmd_clutch = 1
after the throttle starts, the lowest rpm and the top speed in gear while the throttle is on, and cmd_clutch's
reversals over the first 3 s; per braking stop (the clutch's last opening of each phase) the rpm on the call before
and at it; the run header counters (launch_clutch.stalls, restarts, guard_drops, max_slip_s, lockups); and, from
DD-14d plays on, EngineGetRPMFeedback (column rpm_feedback) against the engine and the speed-implied driveline rpm.
PASS needs a complete capture (steps, run footer status complete, the launch_clutch block and its counters) with at
least one launch phase, and in every launch phase found (times are physics time, physics_step deltas x the step length,
so merged callbacks count their steps):
  - evidence of its start (a handbrake row, then the release into first gear with the clutch open) and every physics
    step covered from there (each row's physics_step = the last one's + its covered steps);
  - a complete window (MIN_WINDOW_S of throttle, or one that couples) and motion above MOTION_KMH;
  - c = 1 by COUPLE_DEADLINE_S at throttle >= COUPLE_DEADLINE_THROTTLE, else within the window;
  - no opening, and every in-gear rpm at or above stall_guard_rpm + MARGIN_RPM (also at the braking stops);
and no stall: engine_on never falls in the steps, and the counters stalls and guard_drops are 0.

--fit: the shooting fit of tools/ident/launch_clutch_model.py's plant on the play's launches (15-step segments from
the measured state, every 5 steps, first gear): engine inertia, clutch capacity, rolling load and the gross torque at
TORQUE_KNOTS_RPM by Levenberg-Marquardt (numeric Jacobian) from FIT_START (DD-14c's toy values), with the mass, idle
and free-decay drag fixed. The drag comes from the play's free decays (throttle 0, clutch 0). It prints the values,
the rms errors and S105_19's for comparison.

--replay: the launch clutches of the DD-14c card (ramp) and of the 1.9.0.37 card (slip) in closed loop on the plant
(the fit with --fit, else S105_19), each launch with the play's own readback timing: openings and first c = 1, next to
what the play recorded.

--acceptance: the DD-14d acceptance table on the plant for both, flat and at +-5 % grade, timing U and A. It covers
launches at throttle 0.3 / 0.5 / 0.6 / 1.0, a brake of 0.3 / 0.6 / 1.0 to a stop from 15 / 30 / 40 km/h, and the
shift policy's shifts.

Exit codes: 0 PASS (or the requested analysis done), 1 the play's launch clutch failed the check, 2 input error.
"""

import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import launch_clutch_model as model  # noqa: E402

SEALED_RE = re.compile(r"eden|everon", re.IGNORECASE)
MARGIN_RPM = 100.0              # PASS: every in-gear rpm at least stall_guard_rpm + this
LAUNCH_KEYS = ("mode", "engage_rpm", "full_rpm", "couple_time_s", "uncouple_time_s", "stall_guard_rpm",
               "restart_on_stall", "restart_cooldown_s", "slip_kp", "slip_ki", "lock_tolerance_rpm")
DD14C_RAMP = {"mode": "ramp", "engage_rpm": 1100.0, "full_rpm": 1700.0, "couple_time_s": 0.85,
              "uncouple_time_s": 0.45, "stall_guard_rpm": 600.0, "restart_on_stall": True, "restart_cooldown_s": 1.0}
CARD19 = os.path.join(HERE, "cards", "1.9.0.37", "S105.json")


class InputError(Exception):
    pass


def read_header(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                out[key] = value
    return out


def card_launch_clutch():
    import json
    with open(CARD19, encoding="ascii") as f:
        return json.load(f)["gearbox"]["launch_clutch"]


def clutch_of(config):
    return model.LaunchClutch(**{k: v for k, v in config.items() if k in model.LaunchClutch.DEFAULTS})


# ---------------------------------------------------------------------------------------------------------------
# the check of a play

COUNTER_KEYS = ("stalls", "restarts", "guard_drops", "max_slip_s")    # the probe kit's stop summary (DD-14c)
MOTION_KMH = 5.0                # a launch moved: above this within its window
COUPLE_DEADLINE_S = 2.5         # c = 1 within this at throttle >= COUPLE_DEADLINE_THROTTLE (else within the window)
COUPLE_DEADLINE_THROTTLE = 0.5
MIN_WINDOW_S = 2.5              # a launch window of at least this, or ending coupled


def step_dt_of(header, rows):
    """The play's step length [s]: the run header's clock.step_dt_s, else engine.fixed_time_slice_s, else 1/30 s (a
    row's timeslice_s is the time it covers, n steps on a merged callback, so it is no step length)."""
    for key in ("clock.step_dt_s", "engine.fixed_time_slice_s"):
        try:
            value = float(header.get(key, ""))
        except ValueError:
            continue
        if value > 0:
            return value
    return model.STEP_DT


def launch_evidence(rows, first, step_dt, seconds=8.0):
    """A launch of the play from rows[first] (its first excite row with throttle): (last row index, problems). The
    start must be evident: the row before it in the same phase with the handbrake commanded, this one released, in
    first gear, with cmd_clutch 0. Its rows must cover every physics step: each row's physics_step is the previous
    one's plus the steps it covers (timeslice_s / step dt, merged callbacks included). The window runs while excite,
    first gear and within seconds of physics time."""
    problems = []
    row = rows[first]
    if first == 0 or rows[first - 1]["phase"] != row["phase"] or not rows[first - 1]["cmd_handbrake"]:
        problems.append("no evidence of its start (no handbrake row before the first throttle row)")
    if row["cmd_handbrake"] or row["cmd_gear"] != model.FIRST_GEAR or row["cmd_clutch"] != 0:
        problems.append("its first row is not a release into first gear with the clutch open")
    last = first
    while (last + 1 < len(rows) and rows[last + 1]["phase"] == row["phase"] and rows[last + 1]["segment"] == "excite"
           and rows[last + 1]["cmd_gear"] == model.FIRST_GEAR
           and (rows[last + 1]["physics_step"] - row["physics_step"]) * step_dt < seconds):
        last += 1
    for i in range(first + 1, last + 1):
        covered = max(1, int(round(rows[i]["timeslice_s"] / step_dt))) if rows[i].get("timeslice_s", 0) > 0 else 1
        if rows[i]["physics_step"] - rows[i - 1]["physics_step"] != covered:
            problems.append("a gap in the steps at physics_step %d (an incomplete capture)" % rows[i]["physics_step"])
            break
    return last, problems


def check_play(rows, header):
    """The launch clutch of a play (see the module docstring); returns (lines, passed). Refuses (FAIL) an incomplete
    capture: no rows, a run footer that is not status complete, a missing launch_clutch block or counter, or no launch
    phase. Every launch phase found must show its start (launch_evidence) and cover every physics step to its coupling,
    have a complete window (MIN_WINDOW_S, or one that couples), move above MOTION_KMH, couple fully by
    COUPLE_DEADLINE_S at throttle >= COUPLE_DEADLINE_THROTTLE (else within the window), never open the clutch and stay
    at or above stall_guard_rpm + MARGIN_RPM in gear; the recorded engine_on must never fall (no stall), and the
    counters stalls and guard_drops must be 0. Times are physics time: physics_step deltas x the step length."""
    lines, passed = [], True

    def fail(text):
        lines.append("  FAIL: " + text)
        return False

    launch = {k: header.get("launch_clutch." + k) for k in LAUNCH_KEYS if "launch_clutch." + k in header}
    lines.append("launch clutch: %s" % (", ".join("%s %s" % kv for kv in launch.items()) or "no launch_clutch block"))
    if not rows:
        return lines + ["  FAIL: no steps (an incomplete capture)"], False
    if header.get("status") != "complete":
        passed = fail("the run footer is not status=complete (%r): an incomplete capture" % header.get("status"))
    if "mode" not in launch:
        passed = fail("the run header has no launch_clutch block")
    guard = float(launch.get("stall_guard_rpm") or 600.0)
    step_dt = step_dt_of(header, rows)
    starts = {}
    for i, r in enumerate(rows):
        if (r["phase"] in model.LAUNCH_PHASES and r["phase"] not in starts and r["segment"] == "excite"
                and r["cmd_throttle"] > 0):
            starts[r["phase"]] = i
    if not starts:
        passed = fail("no launch phase (%s) in the steps" % ", ".join(model.LAUNCH_PHASES))
    for name, first in starts.items():
        last, problems = launch_evidence(rows, first, step_dt)
        seg = rows[first:last + 1]
        throttle = model.LAUNCH_PHASES[name]
        t0 = rows[first]["physics_step"]
        opens = [((rows[i]["physics_step"] - t0) * step_dt, rows[i]["speed_kmh"]) for i in range(first + 1, last + 1)
                 if rows[i - 1]["cmd_clutch"] > 0 and rows[i]["cmd_clutch"] == 0]
        coupled = next((i for i in range(first, last + 1) if rows[i]["cmd_clutch"] >= 1.0), None)
        coupled_s = None if coupled is None else (rows[coupled]["physics_step"] - t0) * step_dt
        low = min(r["rpm"] for r in seg)
        top = max(r["speed_kmh"] for r in seg)
        span_s = (rows[last]["physics_step"] - t0) * step_dt + step_dt
        cs = [r["cmd_clutch"] for r in seg if (r["physics_step"] - t0) * step_dt < 3.0]
        nz = [b - a for a, b in zip(cs, cs[1:]) if abs(b - a) > 1e-6]
        rev = sum(1 for a, b in zip(nz, nz[1:]) if a * b < 0)
        if opens:
            problems.append("%d openings" % len(opens))
        if low < guard + MARGIN_RPM:
            problems.append("rpm %.0f below stall_guard_rpm + %g" % (low, MARGIN_RPM))
        if top <= MOTION_KMH:
            problems.append("no motion (max %.1f km/h)" % top)
        if coupled_s is None:
            problems.append("never coupled (c = 1)")
        elif throttle >= COUPLE_DEADLINE_THROTTLE and coupled_s > COUPLE_DEADLINE_S:
            problems.append("coupled at %.2f s, after %.1f s" % (coupled_s, COUPLE_DEADLINE_S))
        if span_s < MIN_WINDOW_S and coupled_s is None:
            problems.append("an incomplete window (%.2f s)" % span_s)
        passed = passed and not problems
        lines.append("  %-14s throttle %.2f: %d openings %s, c = 1 at %s, lowest rpm %.0f, max %.1f km/h, c reversals "
                     "in 3 s %d%s" % (name, throttle, len(opens),
                                      "[" + ", ".join("%.2f s at %.1f km/h" % o for o in opens) + "]",
                                      "-" if coupled_s is None else "%.2f s" % coupled_s, low, top, rev,
                                      "" if not problems else "  <-- FAIL: " + "; ".join(problems)))
    stalls = sum(1 for a, b in zip(rows, rows[1:]) if a["engine_on"] and not b["engine_on"])
    if stalls:
        passed = fail("%d stalls (engine_on 1 -> 0 in the steps)" % stalls)
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if (a["cmd_clutch"] > 0 and b["cmd_clutch"] == 0 and b["cmd_brake"] > 0 and b["cmd_throttle"] == 0
                and b["cmd_gear"] >= model.FIRST_GEAR):
            later = [r["rpm"] for r in rows[i:i + 3]]
            ok = min([a["rpm"]] + later) >= guard + MARGIN_RPM
            passed = passed and ok
            lines.append("  stop %-14s step %d brake %.2f gear %d: rpm %.0f before, %.0f at the opening, lowest %.0f "
                         "(%.1f km/h)%s" % (b["phase"], b["physics_step"], b["cmd_brake"], int(b["cmd_gear"]),
                                            a["rpm"], b["rpm"], min(later), b["speed_kmh"],
                                            "" if ok else "  <-- below stall_guard_rpm + %g" % MARGIN_RPM))
    counters = {k: header.get("launch_clutch." + k) for k in COUNTER_KEYS + ("lockups",)}
    lines.append("run header counters: %s" % ", ".join("%s %s" % (k, v if v is not None else "-")
                                                       for k, v in counters.items()))
    missing = [k for k in COUNTER_KEYS if counters[k] is None]
    if missing:
        passed = fail("the run header misses launch_clutch.%s (an incomplete capture)" % ", launch_clutch.".join(missing))
    for key in ("stalls", "guard_drops"):
        if counters[key] not in (None, "0"):
            passed = fail("launch_clutch.%s %s" % (key, counters[key]))
    feedback = rpm_feedback_text(rows)
    if feedback:
        lines.append(feedback)
    return lines, passed


def rpm_feedback_text(rows):
    """DD-14d: EngineGetRPMFeedback (column rpm_feedback) against the engine rpm and the first gear's speed-implied
    driveline rpm while coupled in first gear and moving; None without the column."""
    pairs = [(r["rpm_feedback"], r["rpm"], r["vel_l_z"] / model.CARD_MPS_PER_KRPM[model.FIRST_GEAR] * 1000.0)
             for r in rows if "rpm_feedback" in r and r["cmd_clutch"] >= 1.0 and r["cmd_gear"] == model.FIRST_GEAR
             and r["vel_l_z"] > 1.0 and r["rpm"] > 0]
    if not pairs:
        return None if not any("rpm_feedback" in r for r in rows) else "rpm_feedback: no coupled first-gear steps"
    to_rpm = sorted(f / e for f, e, _ in pairs)
    to_drive = sorted(f / d for f, _, d in pairs if d > 0)
    return ("rpm_feedback (EngineGetRPMFeedback) over %d coupled first-gear steps: median %.3f x the engine rpm, %.3f x "
            "the speed-implied driveline rpm" % (len(pairs), to_rpm[len(to_rpm) // 2], to_drive[len(to_drive) // 2]))


def free_decay_per_s(rows, idle=800.0):
    """The drag rate of (rpm - idle) from the free decays: throttle 0 and clutch 0 on the step and the two before,
    the engine on, rpm above idle + 50; the median per-step fraction as a rate per second."""
    fractions = []
    for i in range(3, len(rows)):
        w = rows[i - 3:i + 1]
        if any(r["cmd_throttle"] > 0 or r["cmd_clutch"] > 0 or not r["engine_on"] for r in w):
            continue
        before = rows[i - 1]["rpm"] - idle
        if before < 50:
            continue
        drop = (rows[i - 1]["rpm"] - rows[i]["rpm"]) / before
        if 0 < drop < 0.5:
            fractions.append(drop)
    if not fractions:
        return None, 0
    fractions.sort()
    median = fractions[len(fractions) // 2]
    return -math.log(1.0 - median) / model.STEP_DT, len(fractions)


# ---------------------------------------------------------------------------------------------------------------
# the fit

# The fit starts here, not at S105_19, so its agreement with S105_19 says something: DD-14c's toy (0.12 kg m2, 130 Nm)
# and a flat 60 Nm torque
FIT_START = {"j_kgm2": 0.12, "clutch_nm": 130.0, "rolling_n": 200.0, "torque_nm": (60.0,) * len(model.TORQUE_KNOTS_RPM)}


def pack(params):
    return [math.log(params["j_kgm2"]), math.log(params["clutch_nm"]), params["rolling_n"]] + list(params["torque_nm"])


def unpack(x, base):
    params = dict(base)
    params["j_kgm2"], params["clutch_nm"], params["rolling_n"] = math.exp(x[0]), math.exp(x[1]), x[2]
    params["torque_nm"] = tuple(x[3:])
    return params


def residual_vector(x, base, rows, rho, windows):
    e_rpm, e_v = model.shooting_residuals(unpack(x, base), rows, rho, windows)
    return [e / 50.0 for e in e_rpm] + [e / 0.05 for e in e_v]


def solve(a, b):
    """Gaussian elimination with partial pivoting (a square, small)."""
    n = len(b)
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-300:
            raise InputError("singular normal equations in the fit")
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def fit(rows, rho, windows, base, iterations=25, log=print):
    """Levenberg-Marquardt over pack(params): returns (params, rms rpm, rms m/s)."""
    x = pack(base)
    r = residual_vector(x, base, rows, rho, windows)
    cost = sum(v * v for v in r)
    lam = 1e-3
    for it in range(iterations):
        jac = []
        for k in range(len(x)):
            h = 1e-4 if k < 2 else 1e-2
            xk = list(x)
            xk[k] += h
            rk = residual_vector(xk, base, rows, rho, windows)
            jac.append([(b - a) / h for a, b in zip(r, rk)])
        jtj = [[sum(a * b for a, b in zip(jac[i], jac[j])) for j in range(len(x))] for i in range(len(x))]
        jtr = [sum(a * b for a, b in zip(jac[i], r)) for i in range(len(x))]
        improved = False
        for _ in range(8):
            a = [[jtj[i][j] + (lam * jtj[i][i] if i == j else 0.0) for j in range(len(x))] for i in range(len(x))]
            delta = solve(a, [-v for v in jtr])
            xn = [xi + di for xi, di in zip(x, delta)]
            rn = residual_vector(xn, base, rows, rho, windows)
            cn = sum(v * v for v in rn)
            if cn < cost:
                x, r, gain, cost = xn, rn, cost - cn, cn
                lam = max(lam / 3.0, 1e-7)
                improved = True
                break
            lam *= 4.0
        log("  iteration %d: cost %.1f, lambda %.1e" % (it + 1, cost, lam))
        if not improved or gain < 1e-4 * cost:
            break
    params = unpack(x, base)
    e_rpm, e_v = model.shooting_residuals(params, rows, rho, windows)
    return params, model.rms(e_rpm), model.rms(e_v)


def describe_params(params):
    return ("J %.4f kg m2, clutch %.1f Nm, rolling %.0f N, torque at throttle 1 %s Nm at %s rpm, drag %.3f / s; clutch "
            "moves %.0f rpm per step at c = 1" % (
                params["j_kgm2"], params["clutch_nm"], params["rolling_n"],
                " / ".join("%.1f" % t for t in params["torque_nm"]),
                " / ".join("%.0f" % k for k in model.TORQUE_KNOTS_RPM), params["drag_per_s"],
                params["clutch_nm"] / params["j_kgm2"] * model.RPM_PER_RADS * model.STEP_DT))


# ---------------------------------------------------------------------------------------------------------------
# replay and acceptance

def replay(rows, rho, windows, params, config):
    out = {}
    for name, (first, _last) in windows.items():
        sim = model.launch(clutch_of(config), model.LAUNCH_PHASES[name], seconds=5.0, params=params,
                           timing=tuple(rho[first:first + 200]), policy=model.ShiftPolicy(first_only=True))
        m = model.metrics(sim)
        opens = [round(b[0] / model.STEP_DT) for a, b in zip(sim, sim[1:]) if a[1] > 0 and b[1] == 0]
        out[name] = (m["openings"], opens, m["lockup_s"])
    return out


def acceptance(config, params=None, timings=("U", "A")):
    """The DD-14d table for one launch clutch on the plant: {row: text}, and whether it passed."""
    rows_out, passed = {}, True
    guard = config.get("stall_guard_rpm", 600.0)
    for throttle in (0.3, 0.5, 0.6, 1.0):
        for grade in (0.0, 0.05, -0.05):
            parts = []
            for timing in timings:
                m = model.metrics(model.launch(clutch_of(config), throttle, seconds=6.0, grade=grade, params=params,
                                               timing=timing))
                ok = (m["openings"] == 0 and m["stalls"] == 0 and m["alternations"] <= 1
                      and m["min_rpm"] >= guard + MARGIN_RPM
                      and (throttle != 0.5 or grade != 0.0 or (m["lockup_s"] or 99.0) <= 2.0))
                passed = passed and ok
                parts.append("%s: %d openings, c = 1 at %s, min %.0f rpm, alternations %d%s" % (
                    timing, m["openings"], "-" if m["lockup_s"] is None else "%.2f s" % m["lockup_s"], m["min_rpm"],
                    m["alternations"], "" if ok else " FAIL"))
            rows_out["launch throttle %.1f grade %+.0f %%" % (throttle, grade * 100)] = "; ".join(parts)
    for brake in (0.3, 0.6, 1.0):
        parts = []
        for from_kmh in (15.0, 30.0, 40.0):
            low, ok_all = [], True
            for timing in timings:
                sim, first = model.brake_to_stop(clutch_of(config), brake, from_kmh, params=params, timing=timing)
                m = model.metrics(sim, first)
                ok = m["stalls"] == 0 and m["end_c"] == 0.0 and m["end_on"] and m["min_rpm"] >= guard + MARGIN_RPM
                ok_all = ok_all and ok
                low.append(m["min_rpm"])
            passed = passed and ok_all
            parts.append("from %.0f km/h min %.0f rpm%s" % (from_kmh, min(low), "" if ok_all else " FAIL"))
        rows_out["brake %.1f to a stop" % brake] = "; ".join(parts)
    parts = []
    for throttle in (0.6, 1.0):
        sim = model.launch(clutch_of(config), throttle, seconds=12.0, params=params)
        gears = sorted(set(r[7] for r in sim))
        first_one = next((i for i, r in enumerate(sim) if r[1] >= 1.0), None)
        after = [r[1] for r in sim[first_one:]] if first_one is not None else [0.0]
        ok = min(after) >= 1.0
        passed = passed and ok
        parts.append("throttle %.1f gears %s: c after the first lock-up %s" % (
            throttle, gears, "stays 1" if ok else "drops to %.2f FAIL" % min(after)))
    rows_out["shifts under the probe policy"] = "; ".join(parts)
    return rows_out, passed


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch-clutch analysis of an E-01 play (DD-14d)")
    ap.add_argument("--e01", required=True, help="E-01 play directory")
    ap.add_argument("--fit", action="store_true", help="fit the powertrain model on the play's launches")
    ap.add_argument("--replay", action="store_true", help="replay the ramp and slip clutches on the model")
    ap.add_argument("--acceptance", action="store_true", help="the DD-14d acceptance table on the model")
    args = ap.parse_args(argv)
    try:
        play = os.path.abspath(args.e01)
        if SEALED_RE.search(play):
            raise InputError("%s names a sealed map" % play)
        header = read_header(os.path.join(play, "run_header.txt"))
        header.update(read_header(os.path.join(play, "run_footer.txt")) or {})
        if any(SEALED_RE.search(v) for k, v in header.items() if k.startswith(("world.", "launch."))):
            raise InputError("the run header names a sealed map")
        steps = os.path.join(play, "e01_steps.csv")
        if not os.path.isfile(steps):
            raise InputError("%s has no e01_steps.csv" % play)
        rows = model.read_steps(steps)
    except (InputError, OSError, KeyError, ValueError) as e:
        print("REFUSED: %s" % e)
        return 2
    print("E-01 launch clutch check: %s (launch %s, pin %s)" % (play, header.get("launch.id", "?"),
                                                               header.get("launch.pin_id", "?")))
    lines, passed = check_play(rows, header)
    for line in lines:
        print(line)
    rho = model.readback_timing(rows)
    windows = model.launch_windows(rows)
    params = dict(model.S105_19)
    if args.fit:
        drag, count = free_decay_per_s(rows, params["idle_rpm"])
        if drag is not None:
            params["drag_per_s"] = drag
        print("fit: %d launches, readback timing %.0f %% half a step; free-decay drag %s from %d steps" % (
            len(windows), 100.0 * sum(1 for r in rho if r == 0.5) / max(len(rho), 1),
            "-" if drag is None else "%.3f / s" % drag, count))
        before = model.shooting_residuals(params, rows, rho, windows)
        print("  S105_19: %s; rms %.1f rpm, %.4f m/s" % (describe_params(params), model.rms(before[0]),
                                                         model.rms(before[1])))
        params, e_rpm, e_v = fit(rows, rho, windows, dict(params, **FIT_START))
        print("  fitted from %s: %s; rms %.1f rpm, %.4f m/s" % (
            "J %.2f, clutch %.0f, torque %.0f" % (FIT_START["j_kgm2"], FIT_START["clutch_nm"], FIT_START["torque_nm"][0]),
            describe_params(params), e_rpm, e_v))
    if args.replay:
        print("replay (closed loop, each launch with the play's readback timing; %s):"
              % ("the fit" if args.fit else "S105_19"))
        recorded = {}
        for name, (first, last) in model.launch_windows(rows, seconds=8.0).items():
            opens = [i - first for i in range(first + 1, last + 1)
                     if rows[i - 1]["cmd_clutch"] > 0 and rows[i]["cmd_clutch"] == 0 and rows[i]["cmd_throttle"] > 0]
            coupled = next((i - first for i in range(first, last + 1) if rows[i]["cmd_clutch"] >= 1.0), None)
            recorded[name] = (len(opens), opens, None if coupled is None else coupled * model.STEP_DT)
        for label, config in (("DD-14c ramp", DD14C_RAMP), ("1.9.0.37 card " + card_launch_clutch()["mode"],
                                                            card_launch_clutch())):
            for name, (n, opens, lockup) in replay(rows, rho, windows, params, config).items():
                rec = recorded.get(name, (None, [], None))
                print("  %-20s %-14s %d openings at steps %s, c = 1 at %s | play (%s): %s openings at %s, c = 1 at %s"
                      % (label, name, n, opens, "-" if lockup is None else "%.2f s" % lockup,
                         header.get("launch_clutch.mode", "?"), rec[0], rec[1],
                         "-" if rec[2] is None else "%.2f s" % rec[2]))
    if args.acceptance:
        for label, config in (("DD-14c ramp", DD14C_RAMP), ("1.9.0.37 card", card_launch_clutch())):
            table, ok = acceptance(config, params)
            print("acceptance, %s (%s): %s" % (label, config["mode"], "PASS" if ok else "FAIL"))
            for row, text in table.items():
                print("  %-34s %s" % (row, text))
    print("RESULT %s" % ("PASS" if passed else "FAIL"))
    return 0 if passed or args.fit or args.replay or args.acceptance else 1


if __name__ == "__main__":
    sys.exit(main())
