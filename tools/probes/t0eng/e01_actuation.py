#!/usr/bin/env python3
"""E-01 / E-02a analysis of the T0-ENG actuation probes (DC_ProbeSequences, DD-4-probe).

Run it with the isolated interpreter (stdlib only):

    python3 -I -B tools/probes/t0eng/e01_actuation.py e01 <e01 play_dir> [--max-lag 30] [--max-long-lag 120]
    python3 -I -B tools/probes/t0eng/e01_actuation.py e02a <e02a play_dir> [--min-cells 3] [--min-r2 0.9]
                                                      [--max-alat 4.0] [--steady-kmh 1.0] [--min-steady-steps 10]

<play_dir> is one $profile:DC_Probes/<probe>/<launch_id>/play_NN directory (run_header.txt, <probe>_steps.csv,
e01_phases.csv or e02a_cells.csv, run_footer.txt). Nothing is written.

E-01 (r2 4.4: "The car moves. input_pre vs input_post vs command recorded. Step delay identified. steer_sign
measured. Auto-gear works or a scripted gearbox is needed. Recorded whether it works without ForceEnableSimulation"):
  moves                    max speed_kmh >= 5 in an accel or excite segment
  inputs_recorded          per channel the max |input_pre - cmd| and |input_post - cmd| on steps whose command was
                           unchanged since the previous step: "echo" (<= 1e-3) or "transformed" (both are results)
  steer_delay_steps        lag 0..--max-lag maximising |corr(d cmd_steer, d yaw rate)| on steer_prbs excite (the
                           yaw rate from heading_deg differences; differences make the peak sit at the dead time
                           even with a first-order lag), cross-checked by the first step after each steering change
                           of steer_reversal whose yaw rate leaves the pre-change mean by 3 sigma (median)
  throttle_delay_steps     first step after the throttle step where acc_lz leaves the settle baseline +-3 sigma
  brake_delay_steps        the same for the brake steps (baseline: end of the accel segment); median over phases
  steer_sign               sign of corr(cmd_steer, heading rate) at the level-correlation peak (contract: right turn
                           positive; r2 expects -1, CarSteering +1 = left); INFO yaw_axis_sign = sign of
                           corr(angvel_ly, heading rate)
  gearbox                  "automatic works" when cur_gear rises in gear_observe (cmd_gear -1) while speed rises,
                           else "scripted gearbox needed"; INFO gear_scripted honoured or overridden, shift loops
  force_enable_simulation  "not needed" (move_check ok), "needed" (only move_check_forced moved), "does not help"
  INFO                     engine_on 0 steps while driving (stall watch, expected 0 on 1.8), persistent-handbrake
                           re-releases, the level-correlation peak (dead time plus lag), the rerun19.* checklist
  RESULT PASS when the car moved and every identification has a value.

E-02a (r2 4.4: "fit usable by llc_v1"): per cell the steady window is the last half of excite, steps within
--steady-kmh of the target speed; kappa = heading change / distance driven (1/m, right positive), a_lat = v^2 kappa.
Per speed a least-squares line kappa = a * s + b over the valid cells (steady window, |a_lat| <= --max-alat).
PASS ("usable") when every speed has >= --min-cells valid cells and R2 >= --min-r2 [design target; r2 gives no number;
E-02Q at T3 is the real qualification]. The fit itself belongs to DD-7-min (tools/ident/fit_steer_kappa.py).

A play without a complete run_footer.txt is analysed as far as possible; its result is at best INCOMPLETE.
Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402  (shared reader and verdict printer, PROBE_DESIGN.md 4.7)

ECHO_TOL = 1e-3
MOVE_KMH = 5.0
MIN_CORR = 0.3
SIGMA_K = 3.0
BASELINE_STEPS = 30
SHIFT_LOOP_WINDOW_S = 2.0
SHIFT_LOOP_MAX_CHANGES = 2
SHIFT_LOOP_SPEED_KMH = 5.0
GEAR_HONOURED_SHARE = 0.9

CHANNELS = (
    ("steer", "cmd_steer", "input_pre_steer", "input_post_steer"),
    ("throttle", "cmd_throttle", "input_pre_throttle", "input_post_throttle"),
    ("brake", "cmd_brake", "input_pre_brake", "input_post_brake"),
    ("clutch", "cmd_clutch", "input_pre_clutch", "input_post_clutch"),
    ("gear", "cmd_gear", "input_pre_gear", "input_post_gear"),
    ("handbrake", "cmd_handbrake", "input_pre_handbrake", "input_post_handbrake"),
)
STEP_COLUMNS = ("physics_step", "timeslice_s", "phase", "segment", "phase_step", "speed_kmh", "heading_deg",
                "angvel_ly", "acc_lz", "pos_x", "pos_z", "cur_gear", "engine_on") + tuple(
    name for channel in CHANNELS for name in channel[1:])
DRIVING = ("accel", "excite")


# ---------------------------------------------------------------------------------------------------------------
# helpers


def prbs_bits(order, seed, count):
    """The DC_ProbePrbs generator (Fibonacci LFSR, output bit 0, feedback into bit order-1)."""
    taps = {5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7)}.get(order)
    state = seed & ((1 << order) - 1)
    if taps is None or state == 0:
        raise ValueError("unsupported PRBS order %r or zero seed" % order)
    shifts = [order - tap for tap in taps]
    out = []
    for _ in range(count):
        out.append(state & 1)
        feedback = 0
        for shift in shifts:
            feedback ^= (state >> shift) & 1
        state = (state >> 1) | (feedback << (order - 1))
    return out


def header_value(play, key, default=None):
    """A run header value written directly (AppendHeaderLines) or through GetProbeParams (params.<key>)."""
    if key in play.header:
        return play.header[key]
    return play.header.get("params." + key, default)


def num(value):
    if value is None or value == "":
        return None
    return float(value)


def wrap_deg(delta):
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def mean(values):
    return sum(values) / len(values) if values else None


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def pearson(pairs):
    if len(pairs) < 3:
        return None
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / math.sqrt(sxx * syy)


def corr_at_lag(xs, ys, lag):
    """corr(x[s], y[s + lag]) over the physics steps s where both exist (dicts step -> value)."""
    pairs = [(xv, ys[s + lag]) for s, xv in xs.items() if xv is not None and ys.get(s + lag) is not None]
    return pearson(pairs)


def best_lag(xs, ys, max_lag):
    """(lag, corr) maximising |corr| over 0..max_lag, or (None, None)."""
    best = (None, None)
    for lag in range(max_lag + 1):
        c = corr_at_lag(xs, ys, lag)
        if c is not None and (best[1] is None or abs(c) > abs(best[1])):
            best = (lag, c)
    return best


def differences(series):
    """step -> value - value of the previous step (only where both steps exist)."""
    return {s: v - series[s - 1] for s, v in series.items()
            if v is not None and series.get(s - 1) is not None}


def onset_delay(series, change_step, baseline_steps, max_lag):
    """First step >= change_step whose value leaves mean +- 3 sigma of the baseline_steps values before the change.
    Returns the delay in steps or None (no baseline or no onset within max_lag)."""
    baseline = [series[s] for s in range(change_step - baseline_steps, change_step) if series.get(s) is not None]
    if len(baseline) < 5:
        return None
    m = mean(baseline)
    sigma = max(std(baseline), 1e-9)
    for s in range(change_step, change_step + max_lag + 1):
        value = series.get(s)
        if value is not None and abs(value - m) > SIGMA_K * sigma:
            return s - change_step
    return None


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.*f" % (digits, value)
    return str(value)


class Steps:
    """The <probe>_steps.csv rows grouped by phase, with the derived heading rate."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: int(r["physics_step"]))
        self.by_phase = {}
        for row in self.rows:
            self.by_phase.setdefault(row["phase"], []).append(row)
        self.heading_rate = {}       # step -> deg/s (right turn positive)
        for phase_rows in self.by_phase.values():
            previous = None
            for row in phase_rows:
                step = int(row["physics_step"])
                heading = num(row["heading_deg"])
                dt = num(row["timeslice_s"])
                if previous is not None and previous[0] == step - 1 and heading is not None and dt:
                    self.heading_rate[step] = wrap_deg(heading - previous[1]) / dt
                if heading is not None:
                    previous = (step, heading)

    def phases(self, prefix):
        return [name for name in self.by_phase if name == prefix or name.startswith(prefix + "_")]

    def series(self, phase, column, segments=None):
        out = {}
        for row in self.by_phase.get(phase, []):
            if segments is None or row["segment"] in segments:
                out[int(row["physics_step"])] = num(row[column])
        return out

    def yaw_series(self, phase, segments=None):
        out = {}
        for row in self.by_phase.get(phase, []):
            step = int(row["physics_step"])
            if (segments is None or row["segment"] in segments) and step in self.heading_rate:
                out[step] = self.heading_rate[step]
        return out

    def first_step(self, phase, segment):
        for row in self.by_phase.get(phase, []):
            if row["segment"] == segment:
                return int(row["physics_step"])
        return None


# ---------------------------------------------------------------------------------------------------------------
# E-01


def analyse_e01(play, max_lag=30, max_long_lag=120):
    verdict = probe_common.Verdict("E-01 actuation", play.dir)
    steps = Steps(play.csv("e01_steps", required=STEP_COLUMNS))
    phases = play.csv("e01_phases", required=("phase_idx", "phase", "result", "note"))
    verdict.check_footer(play)

    # moves
    best = (0.0, None)
    for row in steps.rows:
        if row["segment"] in DRIVING:
            speed = num(row["speed_kmh"]) or 0.0
            if speed > best[0]:
                best = (speed, row["phase"])
    verdict.check("moves", best[0] >= MOVE_KMH, "max speed %.2f km/h in an accel/excite segment (%s), needs >= %.1f"
                  % (best[0], best[1] or "-", MOVE_KMH))

    # inputs_recorded
    details = []
    recorded = True
    for name, cmd_col, pre_col, post_col in CHANNELS:
        worst_pre = worst_post = None
        compared = 0
        previous = None
        for row in steps.rows:
            step = int(row["physics_step"])
            cmd = num(row[cmd_col])
            if previous is not None and previous[0] == step - 1 and previous[1] == cmd and cmd is not None:
                if not (name == "gear" and cmd < 0):
                    pre, post = num(row[pre_col]), num(row[post_col])
                    if pre is not None and post is not None:
                        compared += 1
                        worst_pre = max(worst_pre or 0.0, abs(pre - cmd))
                        worst_post = max(worst_post or 0.0, abs(post - cmd))
            previous = (step, cmd)
        if compared == 0:
            if name != "gear":
                recorded = False
            details.append("%s no comparable steps" % name)
            continue
        details.append("%s pre %s (max %.4f) post %s (max %.4f)" % (
            name, "echo" if worst_pre <= ECHO_TOL else "transformed", worst_pre,
            "echo" if worst_post <= ECHO_TOL else "transformed", worst_post))
    verdict.check("inputs_recorded", recorded, "; ".join(details))

    # steering: dead time from the PRBS, cross-check from the reversals, sign
    prbs_phases = steps.phases("steer_prbs")
    rev_phases = steps.phases("steer_reversal")
    cmd_prbs, yaw_prbs, yaw_prbs_all, angvel_prbs_all = {}, {}, {}, {}
    for phase in prbs_phases:
        cmd_prbs.update(steps.series(phase, "cmd_steer", ("excite",)))
        yaw_prbs.update(steps.yaw_series(phase, ("excite",)))
        yaw_prbs_all.update(steps.yaw_series(phase))
        angvel_prbs_all.update(steps.series(phase, "angvel_ly"))
    if len(cmd_prbs) < 3:
        verdict.check("steer_delay_steps", None, "no steer_prbs excite steps")
    else:
        lag, c = best_lag(differences(cmd_prbs), differences(yaw_prbs_all), max_lag)
        lag_w, c_w = best_lag(differences(cmd_prbs), differences(angvel_prbs_all), max_lag)
        level_lag, level_c = best_lag(cmd_prbs, yaw_prbs_all, max_lag)
        cross = []
        for phase in rev_phases:
            cmd = steps.series(phase, "cmd_steer")
            yaw = steps.yaw_series(phase)
            excite = set(steps.series(phase, "cmd_steer", ("excite",)))
            for s in sorted(excite):
                if cmd.get(s - 1) is not None and abs(cmd[s] - cmd[s - 1]) > 1e-6:
                    delay = onset_delay(yaw, s, BASELINE_STEPS, max_lag)
                    if delay is not None:
                        cross.append(delay)
        identified = lag is not None and c is not None and abs(c) >= MIN_CORR and lag < max_lag
        verdict.check("steer_delay_steps", identified,
                      "%s steps (|corr| %s of d cmd_steer vs d yaw rate over %d PRBS steps; angvel_ly: lag %s "
                      "corr %s); reversal onsets median %s steps over %d changes; level-correlation peak (dead time "
                      "+ lag) %s steps" % (fmt(lag), fmt(abs(c) if c is not None else None), len(cmd_prbs),
                                           fmt(lag_w), fmt(c_w), fmt(median(cross)), len(cross), fmt(level_lag)))

    # steer_sign over every steering excite step, at the level-correlation peak
    cmd_all, yaw_all = dict(cmd_prbs), dict(yaw_prbs_all)
    for phase in rev_phases:
        cmd_all.update(steps.series(phase, "cmd_steer", ("excite",)))
        yaw_all.update(steps.yaw_series(phase))
    if len(cmd_all) < 3:
        verdict.check("steer_sign", None, "no steering excite steps")
    else:
        lag, c = best_lag(cmd_all, yaw_all, max_lag)
        measured = c is not None and abs(c) >= MIN_CORR
        sign = (1 if c > 0 else -1) if measured else None
        verdict.check("steer_sign", measured, "%s (corr %s at lag %s; +1 = positive steer turns right, r2 expects -1); "
                      "used by e02a of the same Play" % (fmt(sign), fmt(c), fmt(lag)))
        angvel = {}
        for phase in prbs_phases + rev_phases:
            angvel.update(steps.series(phase, "angvel_ly", DRIVING))
        c_axis = corr_at_lag(angvel, yaw_all, 0)
        verdict.info("yaw_axis_sign", "%s (corr(angvel_ly, heading rate) %s)" % (
            fmt(None if c_axis is None else (1 if c_axis > 0 else -1)), fmt(c_axis)))
        header_sign = header_value(play, "m_iAssumedSteerSign")
        verdict.info("steer_sign_attribute", "m_iAssumedSteerSign %s" % fmt(header_sign))

    # throttle / brake dead time on acc_lz
    for kind, base_segment in (("throttle", "settle"), ("brake", "accel")):
        delays, ran = [], 0
        for phase in steps.phases(kind):
            excite_start = steps.first_step(phase, "excite")
            if excite_start is None:
                continue
            ran += 1
            acc = steps.series(phase, "acc_lz")
            delay = onset_delay(acc, excite_start, BASELINE_STEPS, max_long_lag)
            if delay is not None:
                delays.append(delay)
        name = "%s_delay_steps" % kind
        if ran == 0:
            verdict.check(name, None, "no %s phase with an excite segment" % kind)
        else:
            verdict.check(name, bool(delays), "median %s steps (%s) over %d of %d %s steps; acc_lz leaves the %s "
                          "baseline +-3 sigma" % (fmt(median(delays)), ",".join(str(d) for d in delays) or "-",
                                                  len(delays), ran, kind, base_segment))

    # gearbox
    observe = steps.by_phase.get("gear_observe", [])
    observe = [r for r in observe if r["segment"] == "excite"]
    if not observe:
        verdict.check("gearbox", None, "no gear_observe excite steps")
    else:
        gears = [int(float(r["cur_gear"])) for r in observe]
        no_setgear = all(int(float(r["cmd_gear"])) < 0 for r in observe)
        upshifts = sum(1 for a, b in zip(gears, gears[1:]) if b > a)
        sequence = [gears[0]] + [b for a, b in zip(gears, gears[1:]) if b != a]
        speed_gain = (num(observe[-1]["speed_kmh"]) or 0.0) - (num(observe[0]["speed_kmh"]) or 0.0)
        automatic = no_setgear and upshifts >= 1 and speed_gain >= MOVE_KMH
        verdict.check("gearbox", True, "%s (cur_gear %s, %d upshifts, speed +%.1f km/h, cmd_gear -1 throughout: %s)" % (
            "automatic works" if automatic else "scripted gearbox needed", "->".join(str(g) for g in sequence[:20]),
            upshifts, speed_gain, "yes" if no_setgear else "no"))
    scripted = [r for r in steps.by_phase.get("gear_scripted", []) if r["segment"] == "excite"]
    if scripted:
        blocks = []
        for row in scripted:
            gear = int(float(row["cmd_gear"]))
            if not blocks or blocks[-1][0] != gear:
                blocks.append((gear, []))
            blocks[-1][1].append(row)
        parts = []
        for gear, rows in blocks:
            tail = rows[len(rows) // 2:]
            share = sum(1 for r in tail
                        if int(float(r["cur_gear"])) == gear or int(float(r["input_post_gear"])) == gear)
            share = share / len(tail) if tail else 0.0
            parts.append("%d:%s(%.0f%%)" % (gear, "honoured" if share >= GEAR_HONOURED_SHARE else "overridden",
                                            100.0 * share))
        verdict.info("gear_scripted", "SetGear readback per scripted gear: " + " ".join(parts))
    verdict.info("shift_loops", "%d windows with more than %d gear changes within %.0f s while speed changed < %.0f "
                 "km/h (1.9 shift-loop watch, MIGRATION_1_9.md F9)" % (count_shift_loops(steps), SHIFT_LOOP_MAX_CHANGES,
                                                                   SHIFT_LOOP_WINDOW_S, SHIFT_LOOP_SPEED_KMH))

    # ForceEnableSimulation: a move check moved when its max speed reached m_fMoveCheckMinKmh (note max_kmh)
    min_kmh = num(header_value(play, "m_fMoveCheckMinKmh", "3")) or 3.0
    checks = {r["phase"]: r for r in phases if r["phase"] in ("move_check", "move_check_forced")}

    def moved(name):
        row = checks.get(name)
        if row is None:
            return None
        if row["result"] == "ok":
            return True
        for token in (row.get("note") or "").split("/"):
            if token.startswith("max_kmh:"):
                return (num(token.split(":", 1)[1]) or 0.0) >= min_kmh
        return False

    plain, forced = moved("move_check"), moved("move_check_forced")
    if plain is None:
        verdict.check("force_enable_simulation", None, "no move_check row in e01_phases.csv")
    else:
        if plain:
            decision = "not needed"
        elif forced:
            decision = "needed"
        elif forced is None:
            decision = None
        else:
            decision = "does not help"
        verdict.check("force_enable_simulation", decision is not None, "%s (move_check %s, move_check_forced %s)" % (
            decision or "undecided", checks["move_check"]["result"],
            checks["move_check_forced"]["result"] if "move_check_forced" in checks else "-"))

    # INFO
    stalled = sum(1 for r in steps.rows if r["segment"] in DRIVING and num(r["engine_on"]) == 0)
    verdict.info("engine_off_steps", "%d driving steps with engine_on 0 (stall watch; expected 0 on 1.8)" % stalled)
    resets = 0
    for row in phases:
        for token in (row.get("note") or "").split("/"):
            if token.startswith("phb_resets:"):
                resets += int(float(token.split(":", 1)[1] or 0))
    verdict.info("persistent_handbrake_resets", "%d steps on which the controller had set it again" % resets)
    verdict.info("phases", " ".join("%s=%s" % (r["phase"], r["result"]) for r in phases))
    for key in sorted(k for k in play.header if k.split("params.", 1)[-1].startswith("rerun19.")):
        verdict.info(key.split("params.", 1)[-1], play.header[key])
    return verdict


def count_shift_loops(steps):
    """Windows of SHIFT_LOOP_WINDOW_S with more than SHIFT_LOOP_MAX_CHANGES cur_gear changes while the speed changed
    by less than SHIFT_LOOP_SPEED_KMH (counted once per window start, windows do not overlap)."""
    loops = 0
    for rows in steps.by_phase.values():
        changes = []
        previous = None
        for row in rows:
            if row["segment"] not in DRIVING:
                previous = None
                continue
            gear = int(float(row["cur_gear"]))
            t = int(row["physics_step"]) * (num(row["timeslice_s"]) or 0.0)
            if previous is not None and gear != previous:
                changes.append((t, num(row["speed_kmh"]) or 0.0))
            previous = gear
        i = 0
        while i < len(changes):
            window = [c for c in changes[i:] if c[0] - changes[i][0] <= SHIFT_LOOP_WINDOW_S]
            speeds = [c[1] for c in window]
            if len(window) > SHIFT_LOOP_MAX_CHANGES and max(speeds) - min(speeds) < SHIFT_LOOP_SPEED_KMH:
                loops += 1
                i += len(window)
            else:
                i += 1
    return loops


# ---------------------------------------------------------------------------------------------------------------
# E-02a


def cell_kappa(rows, target_kmh, steady_kmh, min_steps):
    """(kappa 1/m, mean speed m/s, steps used) of the steady window, or (None, None, n) without one."""
    excite = [r for r in rows if r["segment"] == "excite"]
    tail = excite[len(excite) // 2:]
    turn = dist = speed_sum = 0.0
    used = 0
    previous = None
    for row in tail:
        step = int(row["physics_step"])
        speed = num(row["speed_kmh"])
        point = (num(row["pos_x"]), num(row["pos_z"]), num(row["heading_deg"]))
        steady = speed is not None and abs(speed - target_kmh) <= steady_kmh
        if previous is not None and previous[0] == step - 1 and steady and None not in point:
            turn += math.radians(wrap_deg(point[2] - previous[1][2]))
            dist += math.hypot(point[0] - previous[1][0], point[1] - previous[1][1])
            speed_sum += speed / 3.6
            used += 1
        previous = (step, point)
    if used < min_steps or dist <= 0:
        return None, None, used
    return turn / dist, speed_sum / used, used


def fit_line(points):
    """Least squares y = a x + b: (a, b, r2, rms) or None with fewer than 2 distinct x."""
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0:
        return None
    a = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx
    b = my - a * mx
    residuals = [p[1] - (a * p[0] + b) for p in points]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((p[1] - my) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return a, b, r2, math.sqrt(ss_res / n)


def analyse_e02a(play, min_cells=3, min_r2=0.9, max_alat=4.0, steady_kmh=1.0, min_steady_steps=10):
    verdict = probe_common.Verdict("E-02a steer-kappa", play.dir)
    steps = Steps(play.csv("e02a_steps", required=STEP_COLUMNS))
    cells = play.csv("e02a_cells", required=("cell_idx", "target_kmh", "steer", "result", "note"))
    verdict.check_footer(play)
    verdict.info("steer_sign_used", "%s (%s)" % (header_value(play, "steer_sign_used", "n/a"),
                                                 header_value(play, "steer_sign_source", "n/a")))

    by_speed = {}
    flagged = []
    for cell in cells:
        speed = float(cell["target_kmh"])
        steer = float(cell["steer"])
        phase = "cell_%s" % cell["cell_idx"]
        kappa, v, used = cell_kappa(steps.by_phase.get(phase, []), speed, steady_kmh, min_steady_steps)
        if kappa is None:
            flagged.append("%s (%.0f km/h, s %+.3f, %s): no steady window (%d steps)" % (
                phase, speed, steer, cell["result"], used))
            continue
        a_lat = v * v * kappa
        line = "%s s %+.3f kappa %+.5f 1/m a_lat %+.2f m/s2 (%s, %d steps)" % (
            phase, steer, kappa, a_lat, cell["result"], used)
        if abs(a_lat) > max_alat:
            flagged.append(line + ": |a_lat| > %.1f" % max_alat)
            continue
        by_speed.setdefault(speed, []).append((steer, kappa, line))

    speeds = sorted({float(c["target_kmh"]) for c in cells})
    if not speeds:
        verdict.check("cells", None, "no cell in e02a_cells.csv")
    usable = bool(speeds)
    for speed in speeds:
        valid = by_speed.get(speed, [])
        fit = fit_line([(s, k) for s, k, _ in valid])
        enough = len(valid) >= min_cells
        good = fit is not None and fit[2] is not None and fit[2] >= min_r2
        if not (enough and good):
            usable = False
        if fit is None:
            verdict.check("fit_%gkmh" % speed, False, "%d valid cells, no fit" % len(valid))
        else:
            a, b, r2, rms = fit
            verdict.check("fit_%gkmh" % speed, enough and good,
                          "kappa = %+.5f * s %+.6f (R2 %s, residual RMS %.6f 1/m, %d valid cells, needs >= %d and "
                          "R2 >= %.2f)" % (a, b, fmt(r2, 4), rms, len(valid), min_cells, min_r2))
        for _, _, line in valid:
            verdict.info("cell", line)
    for line in flagged:
        verdict.info("flagged", line)
    verdict.info("usable", "fit usable by llc_v1: %s [design target; E-02Q at T3 qualifies]" % (
        "yes" if usable else "no"))
    return verdict


# ---------------------------------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="E-01 / E-02a actuation analysis (DD-4-probe)")
    sub = parser.add_subparsers(dest="probe", required=True)
    p01 = sub.add_parser("e01", help="Mode A actuation identification")
    p01.add_argument("play_dir")
    p01.add_argument("--max-lag", type=int, default=30, help="steering lag search 0..N steps (default 30)")
    p01.add_argument("--max-long-lag", type=int, default=120, help="throttle/brake onset search N steps (default 120)")
    p02 = sub.add_parser("e02a", help="first-pass steer -> kappa")
    p02.add_argument("play_dir")
    p02.add_argument("--min-cells", type=int, default=3)
    p02.add_argument("--min-r2", type=float, default=0.9)
    p02.add_argument("--max-alat", type=float, default=4.0)
    p02.add_argument("--steady-kmh", type=float, default=1.0)
    p02.add_argument("--min-steady-steps", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        play = probe_common.Play(args.play_dir)
        if args.probe == "e01":
            verdict = analyse_e01(play, args.max_lag, args.max_long_lag)
        else:
            verdict = analyse_e02a(play, args.min_cells, args.min_r2, args.max_alat, args.steady_kmh,
                                   args.min_steady_steps)
    except probe_common.InputError as exc:
        print("input error: %s" % exc)
        return probe_common.EXIT_INPUT_ERROR
    return verdict.report()


if __name__ == "__main__":
    sys.exit(main())
