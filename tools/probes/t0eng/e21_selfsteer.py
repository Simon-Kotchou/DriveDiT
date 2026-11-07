#!/usr/bin/env python3
"""CC-07 Mode A self-steer from the G19-1 steering probes e21 (S1), e22 (step 0 and load) and e23 (DD-14b).

    python3 -I -B tools/probes/t0eng/e21_selfsteer.py <e21 play_dir> --e22 <e22 play_dir> [--e23 <e23 play_dir>]
                  [--baseline <1.8 e21 play_dir>] [--signal-scale-deg 33.75] [--max-step0-error 0.05]
                  [--max-lag-steps 1] [--max-unmeasurable 0.2] [--readback-tol 1e-3] [--max-kappa 1e-4]
                  [--bias-tol 2e-4] [--angle-tol-deg 0.25] [--angle-tol-share 0.05]

Plays are $profile:DC_Probes/<probe>/<launch_id>/play_NN directories (DC_ProbeSelfSteer, DC_ProbeSteerLoad,
DC_ProbeSteerSteps). Nothing is written. The analysis decides Mode A against Mode B, so every item is PASS, FAIL or
INCOMPLETE (steer_common: missing, interrupted or partial evidence is INCOMPLETE, never PASS; an item with a missing
part is INCOMPLETE even when another part failed, the failure kept as "[observed FAIL ...]"; a production mode is
printed only when every item is complete). An item that depends on a supplied play without a complete footer (e23
included) is withheld as INCOMPLETE with its observed tag. The plays must carry the same build, pin, prefab, map and addon
identity (else input error); a --baseline must be a complete 1.8 play of the same prefab and map. The mandated
conditions are encoded here (FLAT_SPEEDS, CROSS_SPEEDS, ALAT_TARGETS, BRAKE_DECEL), not read from the run.
MIGRATION_1_9.md 6.1 CC-07, item by item:

  evidence       the e21 and e22 footers complete, and e23's when it is given
  step0_channel  e22 stair and ref rows flagged steady in the low-slip regime (10..20 km/h, |a_lat| <= 1 m/s2,
                 |kappa| >= 0.004 1/m): per channel (rot, bone, ent, sig) the median |tan(delta_eff) / L - kappa| /
                 |kappa| <= --max-step0-error over >= 30 rows, and EVERY qualifying command change (the e22 stair
                 levels, the e23 onsets and releases with --e23) lags t50(delta_eff) - t50(input_post_steer) <=
                 --max-lag-steps engine steps. A change is unmeasurable when a value is missing, no t50 is found or a
                 row it depends on covers merged steps (the resolution is then coarser than one step); unmeasurable
                 changes are counted, and more than --max-unmeasurable of them (or < 3 measurable) is INCOMPLETE.
                 The first channel that passes is the wheel_angle (the steering0 signal at --signal-scale-deg per
                 unit; that scale is carried to every reader); none passes and one is INCOMPLETE -> INCOMPLETE
  a_readback     every expected called row (all rows outside e21's no-call segment) of e21, e22 and e23 must be
                 there (each phase's physics_step start..end, one row per dispatched step) with steer_call 1 and a
                 finite command and readback, else INCOMPLETE; then |input_post_steer - cmd_steer| <= --readback-tol.
                 A deviation is explained only by the CC-08 cmd -> steer_input model of e23 (input_model: every input
                 bin and the held-out PRBS at every bin speed PASS, one structure), inside its scope (the bin speeds,
                 or from 0 km/h when the model is the same at every bin speed; |cmd| up to the largest bin), when both
                 its one-step prediction from the previous measured input and its free-running prediction from the
                 block's first settled row are within --readback-tol, and the free-running residual's mean over every
                 1 s stays within --bias-tol (a persistent offset). No e23 model, a deviation outside the scope or
                 without a predecessor -> INCOMPLETE; an unexplained deviation or bias -> FAIL
  b_flat         flat_20 and flat_40 (assistance FULL), phase ok, the measure segment >= 200 m at the speed (mean
                 within 10%): the local absolute curvature |dpsi| / ds and |mean r| / v over every 5 m of path (1 m
                 steps; WINDOW_M below) has p95 <= --max-kappa (1e-4 1/m) and max <= the bound + the noise floor. The
                 floors are measured from the data (steer_common.curvature_stats: 3 sqrt(2) sigma_h / 5 m from the
                 per-row heading noise, 3 x rms(yaw - heading statistic)); a floor above the bound is INCOMPLETE (not
                 resolvable). Over the bound with a valid 1.8 baseline: the allowance max(1e-4, 1.5 x the baseline's
                 p95); without one: INCOMPLETE. INFO the net drift, the no-call segment, flat_<v>_none (NONE)
  c_cross        site.cross.found, its zone >= 5% cross-slope; cross_20 and cross_40 (phase ok, >= 100 m at the
                 speed): |heading drift per 100 m| <= max(1.1 x |b|, |b| + 2 sigma) against a valid 1.8 baseline b
                 that drove the same surveyed strip (site.cross.reuse: start within 1 m, heading within 0.5 deg, the
                 same length and zone, the centre tyre-pair heights within 5 cm, assistance FULL); sigma =
                 sqrt(sigma_now^2 + sigma_baseline^2), each from the heading noise and the two halves of the segment
                 (steer_common.drift_estimate); no baseline (P1 has no Mode B reference) or no sigma -> INCOMPLETE.
                 INFO the wheel-angle deviation from the flat zero command
  d_load         e22 load phases with both command signs, phase ok: delta_eff,ref = the median over the steady ref
                 rows (low slip); every mandated a_lat level (the steady rows of the level whose median |v r| is
                 within 0.3 m/s2 of 2.0 / 2.75 / 3.5, >= 1 s) and the brake segment (steady rows, median decel
                 within 0.75 m/s2 of 3.0, >= 0.5 s): p95 |delta_eff - delta_eff,ref| <= max(--angle-tol-deg,
                 --angle-tol-share x |delta_eff,ref|); needs the step-0 channel
  mode_decision  only when every item is decided: all PASS -> Mode A stays the production default; a FAIL in a..d ->
                 Mode B (CC-14) becomes the production candidate; otherwise no decision, with the reasons
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

KAPPA_FLOOR = 0.004
STEP0_MIN_ROWS = 30
STEP0_MIN_EVENTS = 3
LOW_SLIP_KMH = (10.0, 20.0)
LOW_SLIP_ALAT = 1.0
CHANGE_MIN = 0.04
SETTLE_FROM = 6
SETTLE_TO = 15
# mandated conditions (MIGRATION_1_9.md 6.1 CC-07)
FLAT_SPEEDS = (20.0, 40.0)
CROSS_SPEEDS = (20.0, 40.0)
ALAT_TARGETS = (2.0, 2.75, 3.5)
BRAKE_DECEL = 3.0
MIN_FLAT_M = 200.0
MIN_CROSS_M = 100.0
MIN_CROSS_SLOPE = 0.05
SPEED_SHARE = 0.1
ALAT_TOL = 0.3
DECEL_TOL = 0.75
MIN_ALAT_S = 1.0
MIN_BRAKE_S = 0.5
MIN_REF_S = 1.0
# (b): |dpsi| / ds and |mean r| / v over every 5 m of path (1 m steps). 5 m is short against any curvature that could
# cancel (a 25 m wave of 8e-4 1/m keeps 94% of its amplitude) and long against the CSV heading resolution: heading_deg
# has 5 decimals (1.7e-7 rad), 3.5e-8 1/m over 5 m; the floor that counts is the measured per-row heading noise
# (steer_common.curvature_stats; e01 zero command: 1.9e-6 rad on 1.8.0.13, 1.1e-5..2.7e-5 rad on 1.9.0.37, floors
# 1.6e-6 and 0.9e-5..2.3e-5 1/m)
WINDOW_M = 5.0
WINDOW_STEP_M = 1.0
MIN_WINDOWS = 20
# (c): the baseline must be the same surveyed strip: start within 1 m, heading within 0.5 deg (2 m apart at 175 m
# along is less than the 2.6 m of the tyre-track sample spacing x 2), and the centre tyre-pair heights within 5 cm
SITE_MATCH_M = 1.0
SITE_MATCH_DEG = 0.5
PROFILE_MATCH_M = 0.05
DRIFT_MIN_HALF_M = 25.0
# (a) model exception: the free-running prediction's mean over any 1 s window
BIAS_WINDOW_S = 1.0
ASSIST_FULL = 2
ASSIST_NONE = 0
STATE_TEXT = {True: "PASS", False: "FAIL", None: "INCOMPLETE"}


# ---------------------------------------------------------------------------------------------------------------
# step 0


def transitions(steps, segment_names):
    """Indices k where cmd_steer changes by >= CHANGE_MIN from row k-1 inside one segment of segment_names."""
    out = []
    for k in range(1, len(steps)):
        a, b = steps[k - 1], steps[k]
        if b.segment not in segment_names or a.segment != b.segment or a.phase != b.phase:
            continue
        ca, cb = a.f.get("cmd_steer"), b.f.get("cmd_steer")
        if ca is None or cb is None or abs(cb - ca) < CHANGE_MIN:
            continue
        out.append(k)
    return out


def t50_row(values, start, end, pre, final):
    span = final - pre
    if abs(span) < 1e-9:
        return None
    for i in range(start, end):
        if values[i] is not None and (values[i] - pre) / span >= 0.5:
            return i
    return None


def lag_of(steps, inputs, angles, k, settle_to, dt):
    """(lag in engine steps, None) of the command change at row k, or (None, reason) when it is not measurable"""
    end = min(len(inputs), k + settle_to)
    if k < 1 or end - k <= SETTLE_FROM:
        return None, "short"
    pre_i, pre_a = inputs[k - 1], angles[k - 1]
    tail_i = [v for v in inputs[k + SETTLE_FROM:end] if v is not None]
    tail_a = [v for v in angles[k + SETTLE_FROM:end] if v is not None]
    if pre_i is None or pre_a is None or not tail_i or not tail_a:
        return None, "missing"
    ti = t50_row(inputs, k, end, pre_i, sc.median(tail_i))
    ta = t50_row(angles, k, end, pre_a, sc.median(tail_a))
    if ti is None or ta is None:
        return None, "no_t50"
    lo, hi = min(ti, ta), max(ti, ta)
    if any(sc.covered_steps(steps[i], dt) > 1 for i in range(k - 1, hi + 1)):
        return None, "merged"
    lag = sum(sc.covered_steps(steps[i], dt) for i in range(lo + 1, hi + 1))
    return (lag if ta >= ti else -lag), None


def step0_selection(load_play, args, extra_play=None):
    """CC-07 step 0 on an e22 play (and the e23 onsets of extra_play): {'state', 'selected', 'scale_rad', 'per',
    'detail', 'L'}"""
    steps = sc.load_steps(load_play, "e22")
    wheelbase = sc.wheelbase(load_play)
    dt = sc.step_dt(load_play)
    scale = math.radians(args.signal_scale_deg)
    result = {"state": None, "selected": None, "scale_rad": None, "per": {}, "L": wheelbase}
    if wheelbase is None or dt is None:
        result["detail"] = "no wheelbase (steer.wheelbase_m) or physics step (dt_s) in the e22 run header"
        return result
    extra_steps = sc.load_steps(extra_play, "e23") if extra_play is not None else []
    extra_dt = sc.step_dt(extra_play) if extra_play is not None else None
    low = []
    for s in steps:
        if s.segment not in ("stair", "ref") or s.i.get("seg_flag") != 1:
            continue
        v = s.f.get("speed_kmh")
        if v is None or s.kappa is None or s.a_lat is None:
            continue
        if LOW_SLIP_KMH[0] <= v <= LOW_SLIP_KMH[1] and abs(s.a_lat) <= LOW_SLIP_ALAT and abs(s.kappa) >= KAPPA_FLOOR:
            low.append(s)
    changes = [(0, k, SETTLE_TO, dt) for k in transitions(steps, ("stair",))]
    if extra_steps and extra_dt:
        changes += [(1, k, 12, extra_dt) for k in transitions(extra_steps, ("train",))]
    rows_of = (steps, extra_steps)
    inputs = ([s.f.get("input_post_steer") for s in steps], [s.f.get("input_post_steer") for s in extra_steps])
    parts = []
    states = []
    for key, _, _ in sc.CHANNELS:
        angles = ([sc.channel_angle(s, key, scale) for s in steps],
                  [sc.channel_angle(s, key, scale) for s in extra_steps])
        errors = []
        sig_scales = []
        for s in low:
            delta = sc.channel_angle(s, key, scale)
            if delta is None:
                continue
            errors.append(abs(math.tan(delta) / wheelbase - s.kappa) / abs(s.kappa))
            if key == "sig" and s.f.get("sig_steer"):
                sig_scales.append(math.atan(s.kappa * wheelbase) / s.f.get("sig_steer"))
        lags, reasons = [], {}
        for which, k, settle_to, step in changes:
            lag, reason = lag_of(rows_of[which], inputs[which], angles[which], k, settle_to, step)
            if lag is None:
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                lags.append(lag)
        err = sc.median(errors)
        err_state = None if len(errors) < STEP0_MIN_ROWS else err <= args.max_step0_error
        violations = sum(1 for x in lags if x > args.max_lag_steps)
        unmeasurable = sum(reasons.values())
        share = unmeasurable / float(len(changes)) if changes else 1.0
        if violations:
            lag_state = False
        elif len(lags) < STEP0_MIN_EVENTS or share > args.max_unmeasurable:
            lag_state = None
        else:
            lag_state = True
        state = sc.combine([err_state, lag_state])
        states.append(state)
        entry = {"error": err, "rows": len(errors), "lags": lags, "violations": violations, "state": state,
                 "unmeasurable": reasons}
        if key == "sig":
            entry["fit_scale_deg"] = math.degrees(sc.median(sig_scales)) if sig_scales else None
        result["per"][key] = entry
        text = "%s: median error %s over %d rows; lag <= %g steps on %d of %d measurable changes (max %s), %d of " \
               "%d unmeasurable (%s) -> %s" % (
                   key, sc.fmt(err * 100 if err is not None else None, 3, "%"), len(errors), args.max_lag_steps,
                   len(lags) - violations, len(lags), sc.fmt(max(lags) if lags else None, 3), unmeasurable,
                   len(changes), ", ".join("%s %d" % r for r in sorted(reasons.items())) or "-", STATE_TEXT[state])
        if key == "sig":
            text += " (scale %g deg/unit; best fit %s deg)" % (args.signal_scale_deg,
                                                               sc.fmt(entry["fit_scale_deg"], 4))
        parts.append(text)
        if state is True and result["selected"] is None:
            result["selected"] = key
            result["scale_rad"] = scale if key == "sig" else None
    if result["selected"] is not None:
        result["state"] = True
    elif any(s is None for s in states):
        result["state"] = None
    else:
        result["state"] = False
    result["detail"] = "L %s m, %d low-slip rows, %d command changes; %s; selected %s" % (
        sc.fmt(wheelbase, 4), len(low), len(changes), "; ".join(parts), result["selected"] or "none")
    return result


# ---------------------------------------------------------------------------------------------------------------
# helpers


def phase_segment(steps, phase, segment):
    """Rows of phase in segment (the longest contiguous block)"""
    blocks = [rows for p, run, seg, rows in sc.segments(steps) if p == phase and seg == segment]
    if not blocks:
        return []
    return max(blocks, key=len)


def with_lead(steps, rows):
    """rows plus the row before the first of them: the probe ends a distance segment once the car is that far from
    where it stood at the segment start (the last row of the previous segment), so the path length counts from there"""
    if not rows:
        return rows
    index = rows[0].index
    if index <= 0 or index >= len(steps) or steps[index] is not rows[0]:
        return rows
    return [steps[index - 1]] + list(rows)


def angle_bound(reference, args):
    return max(math.radians(args.angle_tol_deg), args.angle_tol_share * abs(reference or 0.0))


def deviations(rows, channel, scale, reference):
    out = []
    for s in rows:
        delta = sc.channel_angle(s, channel, scale)
        if delta is not None:
            out.append(abs(delta - reference))
    return out


def achieved(rows, target_kmh):
    speeds = [s.f.get("speed_kmh") for s in rows if s.f.get("speed_kmh") is not None]
    mean = sc.mean(speeds)
    return mean, mean is not None and abs(mean - target_kmh) <= SPEED_SHARE * target_kmh


def drive_phase(steps, results, name, target_kmh, min_m, segment="measure"):
    """(rows incl. the lead, None) of a zero-command driving phase, or (None, why it is not usable)"""
    if name not in results:
        return None, "%s: not in the phases file" % name
    if results[name] != "ok":
        return None, "%s: phase result %s" % (name, results[name])
    rows = phase_segment(steps, name, segment)
    if not rows:
        return None, "%s: no %s rows" % (name, segment)
    lead = with_lead(steps, rows)
    _, distance, _ = sc.path_curvature(lead)
    if distance < min_m:
        return None, "%s: %s %s m (< %g m)" % (name, segment, sc.fmt(distance, 4), min_m)
    mean, ok = achieved(rows, target_kmh)
    if not ok:
        return None, "%s: mean speed %s km/h, not within %d%% of %g" % (name, sc.fmt(mean, 3), SPEED_SHARE * 100,
                                                                       target_kmh)
    if any(s.i.get("steer_call") != 1 or abs(s.f.get("cmd_steer") or 0.0) > 1e-9 for s in rows):
        return None, "%s: a %s row without the SetSteering(0) call" % (name, segment)
    return lead, None


def flat_statistic(rows, bound):
    """(state, p95, max, text) of the local absolute curvature against bound: None when it is not resolvable (a
    floor above the bound) or too short; True when p95 <= bound and max <= bound + floor; else False"""
    st = sc.curvature_stats(rows, WINDOW_M, WINDOW_STEP_M)
    p95 = max(v for v in (st["k_p95"], st["r_p95"], 0.0) if v is not None)
    top = max(v for v in (st["k_max"], st["r_max"], 0.0) if v is not None)
    floor = max(v for v in (st["k_floor"], st["r_floor"], 0.0) if v is not None)
    text = "%d windows of %g m: |dpsi|/ds p95 %s max %s, |r|/v p95 %s max %s 1/m; noise floors %s / %s 1/m (sigma_h %s " \
           "rad, q %s rad)" % (st["windows"], WINDOW_M, sc.fmt(st["k_p95"], 3), sc.fmt(st["k_max"], 3), sc.fmt(st["r_p95"], 3),
                              sc.fmt(st["r_max"], 3), sc.fmt(st["k_floor"], 3), sc.fmt(st["r_floor"], 3),
                              sc.fmt(st["sigma_h"], 3), sc.fmt(st["q_h"], 2))
    if st["windows"] < MIN_WINDOWS or st["k_floor"] is None or st["r_floor"] is None:
        return None, p95, top, text + ": too short or the noise cannot be measured"
    if floor > bound:
        return None, p95, top, text + ": not resolvable (a floor above %s)" % sc.fmt(bound, 3)
    return p95 <= bound and top <= bound + floor, p95, top, text


# ---------------------------------------------------------------------------------------------------------------
# CC-07 items


def model_step(model, y, u_e, start, n, dt):
    """The CC-08 input model's output after n engine steps from state y (u_e: commands per engine step, start: the
    first of the n)"""
    a = math.exp(-dt / model["tau"]) if model["structure"] != "M1" and model["tau"] > 0 else 0.0
    for e in range(start, start + n):
        j = e - model["d"]
        command = u_e[j] if j >= 0 else 0.0
        target = model["g"] * command
        if model["structure"] == "M1":
            y = target
            continue
        step = (1.0 - a) * (target - y)
        if model["rate"] is not None:
            limit = model["rate"] * dt
            step = max(-limit, min(limit, step))
        y += step
    return y


def called(step):
    """Every steering-probe row calls SetSteering except the e21 no-call segment"""
    return step.segment != "nocall"


def in_scope(step, scope):
    v = step.f.get("speed_kmh")
    cmd = step.f.get("cmd_steer")
    return v is not None and cmd is not None and scope["kmh"][0] <= v <= scope["kmh"][1] and \
        abs(cmd) <= scope["amplitude"] + 1e-9


def model_residuals(model, scope, steps, dt, tol, bias_tol):
    """The CC-08 model against one play's called rows, block by block (a block: consecutive called rows of one phase
    and run; a teleport between phases is not a model step). Per block: the free-running prediction from the first
    settled row (|input - command| <= tol) and the one-step prediction from the previous measured input of the same
    block. A deviating row (|input - command| > tol) is explained only when both predictions are within tol, it lies
    inside the model's scope and it has a finite predecessor; the free-running residual's mean over every
    BIAS_WINDOW_S must stay within bias_tol on every block (a persistent offset the one-step prediction would absorb).
    Returns the maxima and the counts: unexplained, outside (the scope), no_state (no predecessor or no settled
    start before a deviation), bias_windows (windows over bias_tol)"""
    u_e, ends = [], []
    for s in steps:
        u_e.extend([s.f.get("cmd_steer") or 0.0] * sc.covered_steps(s, dt))
        ends.append(len(u_e))
    out = {"one_step": 0.0, "free": 0.0, "bias": 0.0, "accumulated": 0.0, "unexplained": 0, "outside": 0,
           "no_state": 0, "bias_windows": 0, "rows": 0}
    k = 0
    while k < len(steps):
        s = steps[k]
        if not called(s) or s.i.get("steer_call") != 1:
            k += 1
            continue
        end = k
        while end + 1 < len(steps) and called(steps[end + 1]) and steps[end + 1].i.get("steer_call") == 1 and \
                steps[end + 1].phase == s.phase and steps[end + 1].i.get("run_idx") == s.i.get("run_idx"):
            end += 1
        free_y = None
        residuals = []
        for idx in range(k, end + 1):
            row = steps[idx]
            cmd, post = row.f.get("cmd_steer"), row.f.get("input_post_steer")
            out["rows"] += 1
            deviates = abs(post - cmd) > tol
            if deviates and not in_scope(row, scope):
                out["outside"] += 1
            one = None
            if idx > k:
                prev = steps[idx - 1].f.get("input_post_steer")
                if prev is not None and math.isfinite(prev):
                    one = abs(post - model_step(model, prev, u_e, ends[idx - 1], ends[idx] - ends[idx - 1], dt))
                    out["one_step"] = max(out["one_step"], one)
            free = None
            if free_y is not None:
                free_y = model_step(model, free_y, u_e, ends[idx - 1], ends[idx] - ends[idx - 1], dt)
                r = post - free_y
                free = abs(r)
                out["free"] = max(out["free"], free)
                residuals.append((r, row.f.get("timeslice_s") or dt))
            elif not deviates:
                free_y = post
            if deviates:
                if one is None or free is None:
                    out["no_state"] += 1
                elif one > tol or free > tol:
                    out["unexplained"] += 1
        total, acc = 0.0, 0.0
        window = []
        for r, slice_s in residuals:
            acc += r * slice_s
            window.append((r, slice_s))
            total += slice_s
            while window and total - window[0][1] >= BIAS_WINDOW_S:
                total -= window[0][1]
                window.pop(0)
            if total >= BIAS_WINDOW_S * 0.999:
                bias = abs(sum(x * w for x, w in window) / total)
                out["bias"] = max(out["bias"], bias)
                if bias > bias_tol:
                    out["bias_windows"] += 1
        out["accumulated"] = max(out["accumulated"], abs(acc))
        k = end + 1
    return out


def check_readback(sets, verdict, args, e23_play):
    """sets: [(name, play, probe, steps, dt)] of e21, e22 (and e23). Every called row is checked; missing rows or
    values are INCOMPLETE; deviations are explained only by the CC-08 input model (one-step, free-running, bias)"""
    states = []
    parts = []
    for name, play, probe, steps, _ in sets:
        complete, text = sc.phase_rows_complete(steps, play, probe, called)
        states.append(complete)
        parts.append(text)
    total = 0
    devs = []
    per_segment = {}
    late_zero = 0
    for name, play, probe, steps, _ in sets:
        zero_run = 0
        for s in steps:
            cmd, post = s.f.get("cmd_steer"), s.f.get("input_post_steer")
            if not called(s) or s.i.get("steer_call") != 1 or cmd is None or post is None:
                zero_run = 0
                continue
            total += 1
            if abs(post - cmd) > args.readback_tol:
                devs.append((name, s))
                key = "%s/%s" % (name, s.segment)
                per_segment[key] = per_segment.get(key, 0) + 1
                if zero_run >= 60 and abs(cmd) < 1e-9:
                    late_zero += 1
            zero_run = zero_run + 1 if abs(cmd) < 1e-9 else 0
    worst = max([abs(s.f["input_post_steer"] - s.f["cmd_steer"]) for _, s in devs] or [0.0])
    head = "%d called rows in %s: %d off by > %g (max %s; %d of them after >= 60 zero-command rows; %s)" % (
        total, "+".join(n for n, _, _, _, _ in sets), len(devs), args.readback_tol, sc.fmt(worst, 4), late_zero,
        ", ".join("%s %d" % kv for kv in sorted(per_segment.items())) or "-")
    parts.append(head)
    if total < 100:
        states.append(None)
        parts.append("< 100 called rows")
    elif not devs:
        states.append(True)
    elif e23_play is None:
        states.append(None)
        parts.append("the deterministic-dynamics exception needs the CC-08 input model (a complete --e23)")
    else:
        import e23_steer_steps  # noqa: E402  (CC-08 lives there)
        state, model, detail = e23_steer_steps.input_model(e23_play, args)
        parts.append("CC-08 input model %s: %s" % (STATE_TEXT[state], detail))
        if state is not True:
            states.append(state)
        else:
            merged = {"one_step": 0.0, "free": 0.0, "bias": 0.0, "accumulated": 0.0, "unexplained": 0, "outside": 0,
                      "no_state": 0, "bias_windows": 0}
            for name, play, probe, steps, dt in sets:
                res = model_residuals(model, model["scope"], steps, dt, args.readback_tol, args.bias_tol)
                for key in merged:
                    merged[key] = max(merged[key], res[key]) if isinstance(merged[key], float) else \
                        merged[key] + res[key]
            parts.append("model residuals: one-step max %s, free-running max %s, 1 s mean max %s (<= %g, %d windows "
                         "over), accumulated max %s s; %d deviating rows not explained, %d outside the scope (%s), %d "
                         "without a predecessor or settled start" % (
                             sc.fmt(merged["one_step"], 3), sc.fmt(merged["free"], 3), sc.fmt(merged["bias"], 3),
                             args.bias_tol, merged["bias_windows"], sc.fmt(merged["accumulated"], 3),
                             merged["unexplained"], merged["outside"], model["scope"]["text"], merged["no_state"]))
            if merged["unexplained"] or merged["bias_windows"]:
                states.append(False)
            if merged["outside"] or merged["no_state"]:
                states.append(None)
            if not (merged["unexplained"] or merged["bias_windows"] or merged["outside"] or merged["no_state"]):
                states.append(True)
    state = sc.combine(states)
    verdict.check("a_readback", state, "; ".join(parts) + sc.observed(states))
    return state


def info_nocall(steps, verdict):
    nocall = [s for s in steps if s.segment == "nocall" and s.i.get("steer_call") == 0]
    values = [s.f.get("input_post_steer") for s in nocall if s.f.get("input_post_steer") is not None]
    if values:
        verdict.info("a_nocall", "input_post_steer with no SetSteering call (%d rows): max |.| %s, mean %s, last "
                                 "%s" % (len(values), sc.fmt(max(abs(v) for v in values), 4),
                                         sc.fmt(sc.mean(values), 3), sc.fmt(values[-1], 4)))
    else:
        verdict.info("a_nocall", "no no-call rows")


def check_flat(steps, results, verdict, args, baseline):
    states = []
    parts = []
    for speed in FLAT_SPEEDS:
        name = "flat_%d" % speed
        rows, why = drive_phase(steps, results, name, speed, MIN_FLAT_M)
        if rows is None:
            states.append(None)
            parts.append(why)
            continue
        if any(s.i.get("assist_mode") != ASSIST_FULL for s in rows[1:]):
            states.append(None)
            parts.append("%s: assistance not FULL on every measure row" % name)
            continue
        kappa, distance, turn = sc.path_curvature(rows)
        state, p95, top, text = flat_statistic(rows, args.max_kappa)
        allowance = None
        base_text = "no baseline"
        if baseline is not None:
            base_rows, base_why = drive_phase(baseline["steps"], baseline["results"], name, speed, MIN_FLAT_M)
            if base_rows is None:
                base_text = "baseline not usable: " + base_why
            else:
                base_state, base_p95, _, base_stat = flat_statistic(base_rows, args.max_kappa)
                if base_state is None:
                    base_text = "baseline not usable: " + base_stat
                else:
                    allowance = max(args.max_kappa, 1.5 * base_p95)
                    base_text = "1.8 baseline p95 %s 1/m -> allowance %s" % (sc.fmt(base_p95, 3), sc.fmt(allowance, 3))
        if state is False:
            if allowance is None:
                state = None
            else:
                state, _, _, _ = flat_statistic(rows, allowance)
        states.append(state)
        parts.append("%s: %s (bound %g; %s) -> %s; net %s 1/m over %s m (turn %s deg)" % (
            name, text, args.max_kappa, base_text, STATE_TEXT[state], sc.fmt(kappa, 3), sc.fmt(distance, 4),
            sc.fmt(turn, 3)))
        nocall = with_lead(steps, phase_segment(steps, name, "nocall"))
        nk, nd, _ = sc.path_curvature(nocall)
        _, n95, _, _ = flat_statistic(nocall, args.max_kappa)
        verdict.info("b_nocall_%d" % speed, "no-call segment: p95 %s 1/m, net %s 1/m over %s m" % (
            sc.fmt(n95, 3), sc.fmt(nk, 3), sc.fmt(nd, 4)))
        none_name = name + "_none"
        none_rows, none_why = drive_phase(steps, results, none_name, speed, 50.0)
        if none_rows is None:
            verdict.info("b_" + none_name, none_why)
        else:
            mode = sc.median([s.i.get("assist_mode") for s in none_rows[1:] if s.i.get("assist_mode") is not None])
            none_state, none_p95, _, _ = flat_statistic(none_rows, args.max_kappa)
            verdict.info("b_" + none_name, "assistance %s (NONE = %d): p95 %s 1/m (%s against the %g bound)" % (
                sc.fmt(mode, 2), ASSIST_NONE, sc.fmt(none_p95, 3), STATE_TEXT[none_state], args.max_kappa))
    state = sc.combine(states)
    verdict.check("b_flat", state, "; ".join(parts) + sc.observed(states))
    return state


def site_of(play, kind):
    start = [probe_common.to_float(v) for v in (play.header.get("site.%s.start" % kind) or "").split()]
    heading = probe_common.to_float(play.header.get("site.%s.heading_deg" % kind))
    if len(start) != 3 or None in start or heading is None:
        return None
    return start, heading


def same_site(play, base_play, kind):
    """(True, text) when the baseline drove the same surveyed strip: start within SITE_MATCH_M, heading within
    SITE_MATCH_DEG, the same strip length and measure zone, and the centre tyre-pair heights within PROFILE_MATCH_M;
    else (None, why)"""
    here, there = site_of(play, kind), site_of(base_play, kind)
    if not here or not there:
        return None, "no %s site in %s" % (kind, "the baseline" if here else "this play")
    gap = math.hypot(here[0][0] - there[0][0], here[0][2] - there[0][2])
    turn = abs(sc.wrap_deg(here[1] - there[1]))
    text = "baseline site %s m / %s deg away" % (sc.fmt(gap, 3), sc.fmt(turn, 3))
    if gap > SITE_MATCH_M or turn > SITE_MATCH_DEG:
        return None, text + " (needs <= %g m and %g deg: drive the baseline on site.%s.reuse)" % (
            SITE_MATCH_M, SITE_MATCH_DEG, kind)
    for key in ("length_m", "zone_from_m", "zone_to_m"):
        a = play.header.get("site.%s.%s" % (kind, key))
        b = base_play.header.get("site.%s.%s" % (kind, key))
        if a is None or a != b:
            return None, text + "; %s %s vs %s" % (key, a, b)
    p, q = sc.track_profile(play, kind), sc.track_profile(base_play, kind)
    if not p or not q or len(p) != len(q):
        return None, text + "; no comparable track_profile"
    worst = max(abs(x - y) for x, y in zip(p, q))
    text += ", track profile max |dy| %s m over %d samples" % (sc.fmt(worst, 3), len(p))
    if worst > PROFILE_MATCH_M:
        return None, text + " (> %g m: not the same ground)" % PROFILE_MATCH_M
    return True, text


def check_cross(play, steps, results, verdict, args, channel, scale, baseline):
    found = play.header.get("site.cross.found")
    if found != "1":
        return verdict.check("c_cross", None, "no cross-slope site was found (site.cross.found=%s; first terrain "
                                              "reject %s, first obstacle %s): no_site" % (
                                                  found, play.header.get("site.cross.search.first_terrain_reject",
                                                                         "-"),
                                                  play.header.get("site.cross.search.first_obstacle", "-")))
    cross_min = probe_common.to_float(play.header.get("site.cross.cross_min_signed"))
    detail_site = "site %s heading %s, cross %s..%s (mean %s), grade %s, fallback %s" % (
        play.header.get("site.cross.start", "-"), play.header.get("site.cross.heading_deg", "-"),
        play.header.get("site.cross.cross_min_signed", "-"), play.header.get("site.cross.cross_max_signed", "-"),
        play.header.get("site.cross.cross_mean", "-"), play.header.get("site.cross.grade", "-"),
        play.header.get("site.cross.fallback", "-"))
    if cross_min is None or cross_min < MIN_CROSS_SLOPE:
        return verdict.check("c_cross", None, "the site's measure zone is below %g cross-slope: %s" % (
            MIN_CROSS_SLOPE, detail_site))
    base_site = None
    if baseline is not None:
        base_site = same_site(play, baseline["play"], "cross")
    states = []
    parts = []
    for speed in CROSS_SPEEDS:
        name = "cross_%d" % speed
        rows, why = drive_phase(steps, results, name, speed, MIN_CROSS_M)
        if rows is None:
            states.append(None)
            parts.append(why)
            continue
        drift, sigma, how = sc.drift_estimate(rows, DRIFT_MIN_HALF_M)
        state = None
        base_text = "no 1.8 baseline (P1 has no Mode B reference)"
        if drift is None:
            base_text = "the drift uncertainty cannot be estimated: " + how
        elif base_site is not None and base_site[0] is not True:
            base_text = "baseline not usable: " + base_site[1]
        elif base_site is not None:
            base_rows, base_why = drive_phase(baseline["steps"], baseline["results"], name, speed, MIN_CROSS_M)
            if base_rows is not None and any(s.i.get("assist_mode") != ASSIST_FULL for s in base_rows[1:]):
                base_rows, base_why = None, "%s: baseline assistance not FULL" % name
            if base_rows is None:
                base_text = "baseline not usable: " + base_why
            else:
                base_drift, base_sigma, base_how = sc.drift_estimate(base_rows, DRIFT_MIN_HALF_M)
                if base_drift is None:
                    base_text = "baseline drift uncertainty cannot be estimated: " + base_how
                else:
                    sigma_all = math.sqrt(sigma * sigma + base_sigma * base_sigma)
                    bound = max(1.1 * abs(base_drift), abs(base_drift) + 2.0 * sigma_all)
                    state = abs(drift) <= bound
                    base_text = "1.8 baseline %s +- %s deg/100 m (%s; %s) -> tolerance max(1.1 x, + 2 sigma %s) = %s" % (
                        sc.fmt(base_drift, 3), sc.fmt(base_sigma, 3), base_site[1], base_how, sc.fmt(sigma_all, 3),
                        sc.fmt(bound, 3))
        states.append(state)
        text = "%s: heading drift %s +- %s deg/100 m (%s) (%s) -> %s" % (
            name, sc.fmt(drift, 3), sc.fmt(sigma, 3), how, base_text, STATE_TEXT[state])
        if channel is not None:
            flat = phase_segment(steps, "flat_%d" % speed, "measure")
            refs = [r for r in (sc.channel_angle(s, channel, scale) for s in flat) if r is not None]
            reference = sc.median(refs)
            devs = deviations(rows[1:], channel, scale, reference) if reference is not None else []
            if devs:
                text += "; diagnostic p95 |delta_eff - flat zero command| %s deg" % sc.fmt(
                    sc.deg(probe_common.percentile(devs, 95)), 3)
        parts.append(text)
    state = sc.combine(states)
    verdict.check("c_cross", state, "; ".join(parts) + "; " + detail_site + sc.observed(states))
    return state


def steady(rows):
    return [s for s in rows if s.i.get("seg_flag") == 1]


def check_load(load_play, steps, verdict, args, channel, scale):
    if channel is None:
        return verdict.check("d_load", None, "no wheel-angle channel validated by step 0")
    results = sc.phase_results(load_play, "e22")
    commands = {}
    for phase in results:
        if phase.startswith("load_"):
            value = probe_common.to_float(phase[len("load_"):])
            if value is not None and value != 0:
                commands[phase] = value
    states = []
    parts = []
    for sign in (1, -1):
        if not any((c > 0) == (sign > 0) for c in commands.values()):
            states.append(None)
            parts.append("no load phase with a %s command" % ("positive" if sign > 0 else "negative"))
    for phase in sorted(commands):
        if results[phase] != "ok":
            states.append(None)
            parts.append("%s: phase result %s" % (phase, results[phase]))
            continue
        ref_rows = steady(phase_segment(steps, phase, "ref"))
        refs = [r for r in (sc.channel_angle(s, channel, scale) for s in ref_rows) if r is not None]
        ref_speed = sc.median([s.f.get("speed_kmh") for s in ref_rows if s.f.get("speed_kmh") is not None])
        ref_alat = sc.median([abs(s.a_lat) for s in ref_rows if s.a_lat is not None])
        if sc.duration_s(ref_rows) < MIN_REF_S or len(refs) < 0.9 * len(ref_rows) or ref_speed is None \
                or not 5.0 <= ref_speed <= LOW_SLIP_KMH[1] or ref_alat is None or ref_alat > LOW_SLIP_ALAT:
            states.append(None)
            parts.append("%s: no low-slip steady ref of >= %g s with %s (speed %s km/h, a_lat %s)" % (
                phase, MIN_REF_S, channel, sc.fmt(ref_speed, 3), sc.fmt(ref_alat, 3)))
            continue
        reference = sc.median(refs)
        bound = angle_bound(reference, args)
        levels = {}
        for s in steady(phase_segment(steps, phase, "alat")):
            levels.setdefault(s.i.get("event_idx"), []).append(s)
        for target in ALAT_TARGETS:
            best = None
            for rows in levels.values():
                a_lat = sc.median([abs(s.a_lat) for s in rows if s.a_lat is not None])
                if a_lat is not None and (best is None or abs(a_lat - target) < abs(best[0] - target)):
                    best = (a_lat, rows)
            if best is None or abs(best[0] - target) > ALAT_TOL or sc.duration_s(best[1]) < MIN_ALAT_S:
                states.append(None)
                parts.append("%s a_lat %g: no steady level within %g m/s2 of it for >= %g s (closest %s)" % (
                    phase, target, ALAT_TOL, MIN_ALAT_S, sc.fmt(best[0] if best else None, 3)))
                continue
            devs = deviations(best[1], channel, scale, reference)
            if len(devs) < 0.9 * len(best[1]):
                states.append(None)
                parts.append("%s a_lat %g: the channel is missing on the level" % (phase, target))
                continue
            p95 = probe_common.percentile(devs, 95)
            state = p95 <= bound
            states.append(state)
            parts.append("%s a_lat %g (measured %s, %s s): p95 %s deg vs bound %s deg (ref %s deg) -> %s" % (
                phase, target, sc.fmt(best[0], 3), sc.fmt(sc.duration_s(best[1]), 3), sc.fmt(sc.deg(p95), 3),
                sc.fmt(sc.deg(bound), 3), sc.fmt(sc.deg(reference), 4), STATE_TEXT[state]))
        brake = steady(phase_segment(steps, phase, "brake"))
        decel = sc.median([-s.f["acc_l_z"] for s in brake if s.f.get("acc_l_z") is not None])
        devs = deviations(brake, channel, scale, reference)
        if decel is None or abs(decel - BRAKE_DECEL) > DECEL_TOL or sc.duration_s(brake) < MIN_BRAKE_S \
                or len(devs) < 0.9 * len(brake):
            states.append(None)
            parts.append("%s brake: no steady braking at %g +- %g m/s2 for >= %g s (measured %s over %s s)" % (
                phase, BRAKE_DECEL, DECEL_TOL, MIN_BRAKE_S, sc.fmt(decel, 3), sc.fmt(sc.duration_s(brake), 3)))
            continue
        p95 = probe_common.percentile(devs, 95)
        state = p95 <= bound
        states.append(state)
        parts.append("%s brake %g m/s2 (measured %s): p95 %s deg vs bound %s deg -> %s" % (
            phase, BRAKE_DECEL, sc.fmt(decel, 3), sc.fmt(sc.deg(p95), 3), sc.fmt(sc.deg(bound), 3),
            STATE_TEXT[state]))
    state = sc.combine(states)
    verdict.check("d_load", state, "; ".join(parts) + " (bound max(%g deg, %g x |ref|))" % (
        args.angle_tol_deg, args.angle_tol_share) + sc.observed(states))
    return state


def info_lines(play, steps, verdict, channel):
    verdict.info("assistance", "original %s (%s), sets %s, restores %s, at the stop %s" % (
        play.header.get("assist.original", "-"), play.header.get("assist.original_name", "-"),
        play.header.get("assist.sets", "-"), play.header.get("assist.restores", "-"),
        play.header.get("assist.at_stop", "-")))
    verdict.info("site_flat", "found %s, %s, heading %s, grade %s, cross %s, benign hits %s" % (
        play.header.get("site.flat.found", "-"), play.header.get("site.flat.source", "-"),
        play.header.get("site.flat.heading_deg", "-"), play.header.get("site.flat.grade", "-"),
        play.header.get("site.flat.cross_mean", "-"), play.header.get("site.flat.benign_hits", "-")))
    verdict.info("vehicle", "noise_steer_sensitivity %s, wheelbase %s m, steer sign %s" % (
        play.header.get("vehicle.noise_steer_sensitivity", "-"), sc.fmt(sc.wheelbase(play), 4),
        play.header.get("steer.sign_used", "-")))
    if channel and channel != "sig":
        rows = [s for s in steps if s.segment == "measure" and s.phase.startswith("flat_")]
        left = [sc.channel_pair(s, channel)[0] for s in rows]
        right = [sc.channel_pair(s, channel)[1] for s in rows]
        verdict.info("toe_zero_command", "%s: left %s deg, right %s deg (median over the flat measure rows)" % (
            channel, sc.fmt(sc.deg(sc.median([v for v in left if v is not None])), 3),
            sc.fmt(sc.deg(sc.median([v for v in right if v is not None])), 3)))


def open_play(path, probe):
    play = probe_common.Play(path)
    if play.probe and play.probe != probe:
        raise probe_common.InputError("%s is a %s play, not %s" % (play.dir, play.probe, probe))
    return play


def decide(verdict):
    """The mode decision: printed only when every item is complete (PASS or FAIL); a FAIL observed inside an
    incomplete item is listed, never decided on"""
    items = ("evidence", "step0_channel", "a_readback", "b_flat", "c_cross", "d_load")
    tags = {name: verdict.tag_of(name) for name in items}
    open_items = [n for n in items if tags[n] not in (probe_common.PASS, probe_common.FAIL)]
    failed = [n for n in items[2:] if tags[n] == probe_common.FAIL]
    seen = [name for tag, name, detail in verdict.items if name in items and tag == probe_common.INCOMPLETE
            and "[observed FAIL" in detail]
    if all(tags[n] == probe_common.PASS for n in items):
        verdict.info("mode_decision", "CC-07 PASS on this build: Mode A stays the production default")
    elif not open_items and failed:
        verdict.info("mode_decision", "CC-07 FAIL (%s): Mode B (seated driver, CC-14) becomes the production "
                                      "candidate (decision 11)" % ", ".join(failed))
    else:
        verdict.info("mode_decision", "none: INCOMPLETE %s%s%s; no production mode is chosen from missing or partial "
                                      "evidence" % (", ".join(open_items) or "-",
                                                    "; FAIL so far: " + ", ".join(failed) if failed else "",
                                                    "; FAIL observed inside incomplete items: " + ", ".join(seen)
                                                    if seen else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("play_dir", help="e21 play dir")
    parser.add_argument("--e22", help="e22 play dir (step 0, a and d)")
    parser.add_argument("--e23", help="e23 play dir (step-0 changes, a's readback rows and the CC-08 input model)")
    parser.add_argument("--baseline", help="1.8 e21 play dir (the b allowance and the c reference)")
    parser.add_argument("--readback-tol", type=float, default=1e-3)
    parser.add_argument("--max-kappa", type=float, default=1e-4)
    parser.add_argument("--bias-tol", type=float, default=2e-4)
    parser.add_argument("--angle-tol-deg", type=float, default=0.25)
    parser.add_argument("--angle-tol-share", type=float, default=0.05)
    parser.add_argument("--min-r2", type=float, default=0.9)
    parser.add_argument("--max-cv", type=float, default=0.2)
    parser.add_argument("--min-events", type=int, default=6)
    parser.add_argument("--signal-scale-deg", type=float, default=33.75)
    parser.add_argument("--max-step0-error", type=float, default=0.05)
    parser.add_argument("--max-lag-steps", type=float, default=1.0)
    parser.add_argument("--max-unmeasurable", type=float, default=0.2)
    args = parser.parse_args(argv)
    play = open_play(args.play_dir, "e21")
    load = open_play(args.e22, "e22") if args.e22 else None
    extra = open_play(args.e23, "e23") if args.e23 else None
    identity = sc.check_identity([("e21", play), ("e22", load), ("e23", extra)])
    verdict = sc.StrictVerdict("CC-07 Mode A self-steer (e21, e22, e23)", play.dir)
    steps = sc.load_steps(play, "e21")
    results = sc.phase_results(play, "e21")
    e21_ok, e21_text = sc.play_evidence(play, "e21")
    e22_ok, e22_text = sc.play_evidence(load, "e22")
    texts = [e21_text, e22_text]
    evidence = [e21_ok, e22_ok]
    e23_ok = True
    if extra is not None:
        e23_ok, e23_text = sc.play_evidence(extra, "e23")
        texts.append(e23_text)
        evidence.append(e23_ok)
    verdict.check("evidence", sc.combine(evidence), "; ".join(texts))
    baseline = None
    if args.baseline:
        base_play = open_play(args.baseline, "e21")
        ok, text = sc.baseline_identity(base_play, identity)
        verdict.info("baseline", text if ok else "not usable: " + text)
        if ok:
            baseline = {"play": base_play, "steps": sc.load_steps(base_play, "e21"),
                        "results": sc.phase_results(base_play, "e21")}
    else:
        verdict.info("baseline", "none (--baseline): b's 1.8 allowance and c's reference are missing")

    channel = scale = None
    load_steps = sc.load_steps(load, "e22") if load is not None else []
    if load is not None:
        selection = step0_selection(load, args, extra)
        verdict.check("step0_channel", selection["state"], selection["detail"])
        if selection["state"] is True:
            channel, scale = selection["selected"], selection["scale_rad"]
    else:
        verdict.check("step0_channel", None, "no e22 play (--e22)")

    sets = [("e21", play, "e21", steps, sc.step_dt(play))]
    if load is not None:
        sets.append(("e22", load, "e22", load_steps, sc.step_dt(load)))
    if extra is not None:
        sets.append(("e23", extra, "e23", sc.load_steps(extra, "e23"), sc.step_dt(extra)))
    if load is None:
        verdict.check("a_readback", None, "no e22 play (--e22): the non-zero-command rows are missing")
    elif any(dt is None for _, _, _, _, dt in sets):
        verdict.check("a_readback", None, "a play without a physics step in its run header")
    else:
        check_readback(sets, verdict, args, extra)
    info_nocall(steps, verdict)
    check_flat(steps, results, verdict, args, baseline)
    check_cross(play, steps, results, verdict, args, channel, scale, baseline)
    if load is not None:
        check_load(load, load_steps, verdict, args, channel, scale)
    else:
        verdict.check("d_load", None, "no e22 play (--e22)")

    # an item that depends on a play without a complete footer is not decided (its observed tag stays in the detail)
    needs = {"a_readback": [e21_ok, e22_ok, e23_ok], "b_flat": [e21_ok], "c_cross": [e21_ok],
             "d_load": [e22_ok, e23_ok], "step0_channel": [e22_ok, e23_ok]}
    for name, plays_ok in needs.items():
        if not all(ok is True for ok in plays_ok):
            sc.withhold(verdict, (name,), "a play it depends on is not complete")
    info_lines(play, steps, verdict, channel)
    decide(verdict)
    return verdict.report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
