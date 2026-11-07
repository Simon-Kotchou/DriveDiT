#!/usr/bin/env python3
"""CC-07 Mode A self-steer from the G19-1 steering probes e21 (S1) and e22 (step 0 and load) (DD-14b).

    python3 -I -B tools/probes/t0eng/e21_selfsteer.py <e21 play_dir> --e22 <e22 play_dir> [--e23 <e23 play_dir>]
                  [--baseline <1.8 e21 play_dir>] [--signal-scale-deg 33.75] [--max-step0-error 0.05]
                  [--max-lag-steps 1] [--readback-tol 1e-3] [--max-kappa 1e-4] [--min-flat-m 200]
                  [--min-cross 0.05] [--angle-tol-deg 0.25] [--angle-tol-share 0.05]

Plays are $profile:DC_Probes/<probe>/<launch_id>/play_NN directories (DC_ProbeSelfSteer, DC_ProbeSteerLoad,
DC_ProbeSteerSteps). Nothing is written. MIGRATION_1_9.md 6.1 CC-07, item by item:

  step0_channel  e22 stair and ref rows flagged steady (seg_flag 1) in the low-slip regime (10..20 km/h,
                 |a_lat| <= 1 m/s2, |kappa| >= 0.004 1/m): per candidate channel (rot, bone, ent, sig; steer_common)
                 the median of |tan(delta_eff) / L - kappa| / |kappa| <= --max-step0-error (5%) over >= 30 rows, and
                 the lag: median over the command changes (e22 stair levels, plus the e23 step onsets with --e23) of
                 t50(delta_eff) - t50(input_post_steer) <= --max-lag-steps (one physics step). The first channel that
                 passes both is the validated wheel_angle; none -> FAIL. L = steer.wheelbase_m. The signal is scaled
                 by --signal-scale-deg per unit (the S105 MaxSteeringAngle), its best-fit scale is INFO
  a_readback     e21 rows with SetSteering called (steer_call 1) in hold and measure, above 5 km/h:
                 |input_post_steer - cmd_steer| <= --readback-tol on every row (the command is constant there, so a
                 rate limit cannot explain a deviation); INFO the no-call rows (the input with nobody calling)
  b_flat         every speed of params m_sFlatSpeedsKmh, phase flat_<v> (the original assistance mode): the measure
                 segment's mean curvature (unwrapped heading change / path length) |kappa| <= --max-kappa over
                 >= --min-flat-m; INFO the yaw-rate equivalent, the no-call segment, the assistance variant and the
                 1.8 baseline (--baseline, INFO until a 1.8 S1 baseline is adopted)
  c_cross        site.cross.found and its measure zone at >= --min-cross; every speed of params m_sCrossSpeedsKmh,
                 phase cross_<v>, measure segment: p95 of |delta_eff - delta_eff(flat, zero command)| <=
                 max(--angle-tol-deg, --angle-tol-share x |reference|) with the step-0 channel (no channel -> FAIL);
                 INFO the heading drift per 100 m (vs --baseline when given; CC-07 c's relative bound needs a Mode B
                 or 1.8 reference)
  d_load         e22 load_<c> phases: delta_eff,ref = median over the ref rows flagged steady; every a_lat level
                 (alat rows flagged steady, event_idx = the level, median |v r| within 0.3 m/s2 of its target) and the
                 brake segment (flagged rows, median decel within 0.75 m/s2 of its target): p95 |delta_eff -
                 delta_eff,ref| <= max(--angle-tol-deg, --angle-tol-share x |delta_eff,ref|)
  INFO           the Mode A verdict (FAIL -> Mode B, CC-14, becomes the production candidate), assistance (original
                 mode, sets, restores, at the stop), the sites, toe at zero command, steer sign, wheelbase
Missing data never passes: a missing play, phase, segment, column value or site fails the item.
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
DRIVING_KMH = 5.0
CHANGE_MIN = 0.04
SETTLE_FROM = 6
SETTLE_TO = 15
ALAT_TOL = 0.3
DECEL_TOL = 0.75


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


def lag_of(inputs, angles, k, settle_to):
    """t50(delta_eff) - t50(input_post_steer) in steps for the command change at row k (None if not measurable);
    inputs / angles: input_post_steer and the channel's delta_eff per row"""
    end = min(len(inputs), k + settle_to)
    if k < 1 or end - k <= SETTLE_FROM:
        return None
    pre_i, pre_a = inputs[k - 1], angles[k - 1]
    tail_i = [v for v in inputs[k + SETTLE_FROM:end] if v is not None]
    tail_a = [v for v in angles[k + SETTLE_FROM:end] if v is not None]
    if pre_i is None or pre_a is None or not tail_i or not tail_a:
        return None
    ti = sc.t50_index(inputs[:end], k, pre_i, sc.median(tail_i))
    ta = sc.t50_index(angles[:end], k, pre_a, sc.median(tail_a))
    if ti is None or ta is None:
        return None
    return ta - ti


def step0_selection(load_play, args, extra_play=None):
    """CC-07 step 0 on an e22 play (and the e23 step onsets of extra_play): {'selected', 'per', 'detail', 'L'}"""
    steps = sc.load_steps(load_play, "e22")
    wheelbase = sc.wheelbase(load_play)
    scale = math.radians(args.signal_scale_deg)
    result = {"selected": None, "per": {}, "L": wheelbase}
    if wheelbase is None:
        result["detail"] = "no wheelbase in the e22 run header (steer.wheelbase_m): no data"
        return result
    extra_steps = sc.load_steps(extra_play, "e23") if extra_play is not None else []
    rows = [s for s in steps if s.segment in ("stair", "ref") and s.i.get("seg_flag") == 1]
    low = []
    for s in rows:
        v = s.f.get("speed_kmh")
        if v is None or s.kappa is None or s.a_lat is None:
            continue
        if not LOW_SLIP_KMH[0] <= v <= LOW_SLIP_KMH[1] or abs(s.a_lat) > LOW_SLIP_ALAT:
            continue
        if abs(s.kappa) < KAPPA_FLOOR:
            continue
        low.append(s)
    stair_changes = transitions(steps, ("stair",))
    extra_changes = transitions(extra_steps, ("train",)) if extra_steps else []
    inputs = [s.f.get("input_post_steer") for s in steps]
    extra_inputs = [s.f.get("input_post_steer") for s in extra_steps]
    parts = []
    for key, _, _ in sc.CHANNELS:
        errors = []
        sig_scales = []
        for s in low:
            delta = sc.channel_angle(s, key, scale)
            if delta is None:
                continue
            predicted = math.tan(delta) / wheelbase
            errors.append(abs(predicted - s.kappa) / abs(s.kappa))
            if key == "sig" and s.f.get("sig_steer"):
                sig_scales.append(math.atan(s.kappa * wheelbase) / s.f.get("sig_steer"))
        angles = [sc.channel_angle(s, key, scale) for s in steps]
        extra_angles = [sc.channel_angle(s, key, scale) for s in extra_steps]
        lags = [lag_of(inputs, angles, k, SETTLE_TO) for k in stair_changes]
        lags += [lag_of(extra_inputs, extra_angles, k, 12) for k in extra_changes]
        lags = [x for x in lags if x is not None]
        err = sc.median(errors)
        lag = sc.median(lags)
        ok_err = err is not None and len(errors) >= STEP0_MIN_ROWS and err <= args.max_step0_error
        ok_lag = lag is not None and len(lags) >= STEP0_MIN_EVENTS and lag <= args.max_lag_steps
        entry = {"error": err, "rows": len(errors), "lag": lag, "events": len(lags), "ok": ok_err and ok_lag}
        if key == "sig":
            entry["fit_scale_deg"] = math.degrees(sc.median(sig_scales)) if sig_scales else None
        result["per"][key] = entry
        text = "%s: median error %s over %d rows, lag %s steps over %d changes -> %s" % (
            key, sc.fmt(err * 100 if err is not None else None, 3, "%"), len(errors), sc.fmt(lag, 3), len(lags),
            "ok" if entry["ok"] else "no")
        if key == "sig":
            text += " (scale %g deg/unit; best fit %s deg)" % (args.signal_scale_deg,
                                                               sc.fmt(entry["fit_scale_deg"], 4))
        parts.append(text)
        if entry["ok"] and result["selected"] is None:
            result["selected"] = key
    result["detail"] = "L %s m, %d low-slip rows; %s; selected %s" % (
        sc.fmt(wheelbase, 4), len(low), "; ".join(parts), result["selected"] or "none")
    return result


# ---------------------------------------------------------------------------------------------------------------
# helpers


def phase_segment(steps, phase, segment):
    """Rows of the first run of phase in segment (the longest contiguous block)"""
    blocks = [rows for p, run, seg, rows in sc.segments(steps) if p == phase and seg == segment]
    if not blocks:
        return []
    return max(blocks, key=len)


def with_lead(steps, rows):
    """rows plus the row before the first of them: the probe ends a distance segment once the car is that far from
    where it stood at the segment start (the last row of the previous segment), so the path length counts from there"""
    if not rows:
        return rows
    index = next((i for i, s in enumerate(steps) if s is rows[0]), None)
    if index is None or index == 0:
        return rows
    return [steps[index - 1]] + list(rows)


def float_list(text):
    out = []
    for token in (text or "").split(","):
        value = probe_common.to_float(token.strip())
        if value is not None:
            out.append(value)
    return out


def name_kmh(value):
    return "%d" % round(value)


def angle_bound(reference, args):
    return max(math.radians(args.angle_tol_deg), args.angle_tol_share * abs(reference or 0.0))


def deviations(rows, channel, scale, reference):
    out = []
    for s in rows:
        delta = sc.channel_angle(s, channel, scale)
        if delta is not None:
            out.append(abs(delta - reference))
    return out


# ---------------------------------------------------------------------------------------------------------------
# CC-07 items


def check_readback(steps, verdict, args):
    held = [s for s in steps if s.i.get("steer_call") == 1 and s.segment in ("hold", "measure")
            and (s.f.get("speed_kmh") or 0) >= DRIVING_KMH]
    devs = []
    per_phase = {}
    for s in held:
        cmd, post = s.f.get("cmd_steer"), s.f.get("input_post_steer")
        if cmd is None or post is None:
            continue
        dev = post - cmd
        devs.append(dev)
        per_phase.setdefault(s.phase, []).append(dev)
    if not devs:
        verdict.check("a_readback", False, "no held zero-command rows with a readback (no data)")
        return False
    worst = max(abs(d) for d in devs)
    off = sum(1 for d in devs if abs(d) > args.readback_tol)
    phases = "; ".join("%s max %s, %d/%d off" % (p, sc.fmt(max(abs(x) for x in v), 3),
                                                 sum(1 for x in v if abs(x) > args.readback_tol), len(v))
                       for p, v in per_phase.items())
    ok = off == 0
    verdict.check("a_readback", ok, "input_post_steer - cmd_steer on %d held rows: max |.| %s (tol %g), %d rows "
                                    "off, mean |.| %s, signed mean %s; per phase: %s" % (
                                        len(devs), sc.fmt(worst, 4), args.readback_tol, off,
                                        sc.fmt(sc.mean([abs(d) for d in devs]), 3), sc.fmt(sc.mean(devs), 3),
                                        phases))
    nocall = [s for s in steps if s.segment == "nocall" and s.i.get("steer_call") == 0]
    values = [s.f.get("input_post_steer") for s in nocall if s.f.get("input_post_steer") is not None]
    if values:
        verdict.info("a_nocall", "input_post_steer with no SetSteering call (%d rows): max |.| %s, mean %s, last "
                                 "%s" % (len(values), sc.fmt(max(abs(v) for v in values), 4),
                                         sc.fmt(sc.mean(values), 3), sc.fmt(values[-1], 4)))
    else:
        verdict.info("a_nocall", "no no-call rows")
    return ok


def flat_metrics(steps, phase, segment):
    rows = phase_segment(steps, phase, segment)
    kappa, distance, turn = sc.path_curvature(with_lead(steps, rows))
    speeds = [s.f.get("vel_l_z") for s in rows if s.f.get("vel_l_z") is not None]
    rates = [s.f.get("angvel_l_y") for s in rows if s.f.get("angvel_l_y") is not None]
    return {"rows": rows, "kappa": kappa, "distance": distance, "turn": turn,
            "v": sc.mean(speeds), "r": sc.mean(rates)}


def check_flat(play, steps, verdict, args, baseline_steps):
    speeds = float_list(play.param("m_sFlatSpeedsKmh", ""))
    if not speeds:
        verdict.check("b_flat", False, "no flat speeds in the run header (params m_sFlatSpeedsKmh)")
        return False
    ok = True
    parts = []
    for speed in speeds:
        name = "flat_" + name_kmh(speed)
        m = flat_metrics(steps, name, "measure")
        if m["kappa"] is None or m["distance"] < args.min_flat_m:
            ok = False
            parts.append("%s: measure %s m (< %g m or no rows): no data" % (name, sc.fmt(m["distance"], 4),
                                                                            args.min_flat_m))
            continue
        passed = abs(m["kappa"]) <= args.max_kappa
        ok = ok and passed
        bound_r = args.max_kappa * (m["v"] or 0.0)
        parts.append("%s: kappa %s 1/m over %s m (turn %s deg), mean yaw rate %s rad/s (bound %s at %s m/s) -> %s"
                     % (name, sc.fmt(m["kappa"], 3), sc.fmt(m["distance"], 4), sc.fmt(m["turn"], 3),
                        sc.fmt(m["r"], 3), sc.fmt(bound_r, 3), sc.fmt(m["v"], 3), "ok" if passed else "over"))
        nocall = flat_metrics(steps, name, "nocall")
        verdict.info("b_nocall_" + name_kmh(speed), "no-call segment: kappa %s 1/m over %s m (turn %s deg)" % (
            sc.fmt(nocall["kappa"], 3), sc.fmt(nocall["distance"], 4), sc.fmt(nocall["turn"], 3)))
        if baseline_steps is not None:
            base = flat_metrics(baseline_steps, name, "measure")
            verdict.info("b_baseline_" + name_kmh(speed), "1.8 baseline kappa %s 1/m (relative bound max(%g, "
                                                         "1.5 x |baseline|) = %s; INFO only)" % (
                                                             sc.fmt(base["kappa"], 3), args.max_kappa,
                                                             sc.fmt(max(args.max_kappa, 1.5 * abs(base["kappa"]))
                                                                    if base["kappa"] is not None else None, 3)))
    verdict.check("b_flat", ok, "; ".join(parts) + " (bound |kappa| <= %g 1/m over >= %g m)" % (
        args.max_kappa, args.min_flat_m))
    assist_parts = []
    for phase in sorted(set(s.phase for s in steps if "_assist" in s.phase)):
        held = flat_metrics(steps, phase, "measure")
        nocall = flat_metrics(steps, phase, "nocall")
        mode = sc.median([s.i.get("assist_mode") for s in held["rows"] if s.i.get("assist_mode") is not None])
        devs = [abs(s.f["input_post_steer"] - s.f["cmd_steer"]) for s in held["rows"]
                if s.f.get("input_post_steer") is not None and s.f.get("cmd_steer") is not None]
        assist_parts.append("%s (mode %s): held kappa %s over %s m, readback max |dev| %s, %d rows off; no-call "
                            "kappa %s" % (phase, sc.fmt(mode, 2), sc.fmt(held["kappa"], 3),
                                          sc.fmt(held["distance"], 4), sc.fmt(max(devs) if devs else None, 3),
                                          sum(1 for d in devs if d > args.readback_tol), sc.fmt(nocall["kappa"], 3)))
    verdict.info("assist_variant", "; ".join(assist_parts) or "- (no assistance-variant phase)")
    return ok


def check_cross(play, steps, verdict, args, channel, baseline_steps):
    found = play.header.get("site.cross.found")
    if found != "1":
        verdict.check("c_cross", False, "no cross-slope site was found (site.cross.found=%s; first terrain reject %s, "
                                        "first obstacle %s)" % (found, play.header.get(
                                            "site.cross.search.first_terrain_reject", "-"),
                                            play.header.get("site.cross.search.first_obstacle", "-")))
        return False
    cross_min = probe_common.to_float(play.header.get("site.cross.cross_min_signed"))
    detail_site = "site %s heading %s, cross %s..%s (mean %s), grade %s, fallback %s" % (
        play.header.get("site.cross.start", "-"), play.header.get("site.cross.heading_deg", "-"),
        play.header.get("site.cross.cross_min_signed", "-"), play.header.get("site.cross.cross_max_signed", "-"),
        play.header.get("site.cross.cross_mean", "-"), play.header.get("site.cross.grade", "-"),
        play.header.get("site.cross.fallback", "-"))
    if cross_min is None or cross_min < args.min_cross:
        verdict.check("c_cross", False, "the site's measure zone is below %g cross-slope: %s" % (args.min_cross,
                                                                                               detail_site))
        return False
    if channel is None:
        verdict.check("c_cross", False, "no wheel-angle channel validated by step 0, so the applied angle on the "
                                        "slope cannot be judged; %s" % detail_site)
        return False
    scale = math.radians(args.signal_scale_deg)
    speeds = float_list(play.param("m_sCrossSpeedsKmh", ""))
    if not speeds:
        verdict.check("c_cross", False, "no cross speeds (params m_sCrossSpeedsKmh)")
        return False
    ok = True
    parts = []
    for speed in speeds:
        name = "cross_" + name_kmh(speed)
        rows = phase_segment(steps, name, "measure")
        flat = phase_segment(steps, "flat_" + name_kmh(speed), "measure")
        refs = [sc.channel_angle(s, channel, scale) for s in flat]
        refs = [r for r in refs if r is not None]
        reference = sc.median(refs)
        devs = deviations(rows, channel, scale, reference) if reference is not None else []
        if not devs:
            ok = False
            parts.append("%s: no measure rows with %s, or no flat zero-command reference: no data" % (name, channel))
            continue
        p95 = probe_common.percentile(devs, 95)
        bound = angle_bound(reference, args)
        passed = p95 <= bound
        ok = ok and passed
        kappa, distance, turn = sc.path_curvature(with_lead(steps, rows))
        drift = turn / distance * 100.0 if turn is not None and distance > 0 else None
        parts.append("%s: p95 |delta_eff - flat zero-command %s deg| %s deg (bound %s deg) over %d rows, heading "
                     "drift %s deg/100 m over %s m -> %s" % (
                         name, sc.fmt(sc.deg(reference), 3), sc.fmt(sc.deg(p95), 3), sc.fmt(sc.deg(bound), 3),
                         len(devs), sc.fmt(drift, 3), sc.fmt(distance, 4), "ok" if passed else "over"))
        if baseline_steps is not None:
            base = phase_segment(baseline_steps, name, "measure")
            bk, bd, bt = sc.path_curvature(with_lead(baseline_steps, base))
            verdict.info("c_baseline_" + name_kmh(speed), "1.8 baseline heading drift %s deg/100 m (INFO; CC-07 c: "
                                                         "Mode B or 1.8 reference + 10%%)" % sc.fmt(
                                                             bt / bd * 100.0 if bt is not None and bd else None, 3))
    verdict.check("c_cross", ok, "; ".join(parts) + "; " + detail_site)
    return ok


def check_load(load_play, verdict, args, channel):
    if channel is None:
        verdict.check("d_load", False, "no wheel-angle channel validated by step 0 (no data)")
        return False
    steps = sc.load_steps(load_play, "e22")
    scale = math.radians(args.signal_scale_deg)
    phases = sorted(set(s.phase for s in steps if s.phase.startswith("load_")))
    if not phases:
        verdict.check("d_load", False, "no load_<c> phase in e22 (no data)")
        return False
    targets = float_list(load_play.param("m_sALatTargets", ""))
    decel_target = probe_common.to_float(load_play.param("m_fBrakeDecelMs2"), 3.0)
    ok = True
    parts = []
    for phase in phases:
        ref_rows = [s for s in phase_segment(steps, phase, "ref") if s.i.get("seg_flag") == 1]
        refs = [sc.channel_angle(s, channel, scale) for s in ref_rows]
        refs = [r for r in refs if r is not None]
        reference = sc.median(refs)
        if reference is None:
            ok = False
            parts.append("%s: no steady ref rows with %s (no data)" % (phase, channel))
            continue
        bound = angle_bound(reference, args)
        alat_rows = phase_segment(steps, phase, "alat")
        for level, target in enumerate(targets):
            rows = [s for s in alat_rows if s.i.get("seg_flag") == 1 and s.i.get("event_idx") == level]
            a_lat = sc.median([abs(s.a_lat) for s in rows if s.a_lat is not None])
            devs = deviations(rows, channel, scale, reference)
            if not devs or a_lat is None:
                ok = False
                parts.append("%s a_lat %g: no steady rows (no data)" % (phase, target))
                continue
            p95 = probe_common.percentile(devs, 95)
            at_target = abs(a_lat - target) <= ALAT_TOL
            passed = at_target and p95 <= bound
            ok = ok and passed
            parts.append("%s a_lat %g (measured %s): p95 %s deg vs bound %s deg (ref %s deg) -> %s" % (
                phase, target, sc.fmt(a_lat, 3), sc.fmt(sc.deg(p95), 3), sc.fmt(sc.deg(bound), 3),
                sc.fmt(sc.deg(reference), 4), "ok" if passed else ("not at the target" if not at_target else "over")))
        brake = [s for s in phase_segment(steps, phase, "brake") if s.i.get("seg_flag") == 1]
        decel = sc.median([-s.f["acc_l_z"] for s in brake if s.f.get("acc_l_z") is not None])
        devs = deviations(brake, channel, scale, reference)
        if not devs or decel is None:
            ok = False
            parts.append("%s brake: no steady brake rows (no data)" % phase)
            continue
        p95 = probe_common.percentile(devs, 95)
        at_target = abs(decel - decel_target) <= DECEL_TOL
        passed = at_target and p95 <= bound
        ok = ok and passed
        parts.append("%s brake %g m/s2 (measured %s): p95 %s deg vs bound %s deg -> %s" % (
            phase, decel_target, sc.fmt(decel, 3), sc.fmt(sc.deg(p95), 3), sc.fmt(sc.deg(bound), 3),
            "ok" if passed else ("not at the target" if not at_target else "over")))
    verdict.check("d_load", ok, "; ".join(parts) + " (bound max(%g deg, %g x |ref|))" % (args.angle_tol_deg,
                                                                                         args.angle_tol_share))
    return ok


def info_lines(play, steps, verdict, channel, args):
    verdict.info("assistance", "original %s (%s), sets %s, restores %s, at the stop %s" % (
        play.header.get("assist.original", "-"), play.header.get("assist.original_name", "-"),
        play.header.get("assist.sets", "-"), play.header.get("assist.restores", "-"),
        play.header.get("assist.at_stop", "-")))
    verdict.info("site_flat", "found %s, %s, heading %s, grade %s, cross %s, obstacle hits %s" % (
        play.header.get("site.flat.found", "-"), play.header.get("site.flat.source", "-"),
        play.header.get("site.flat.heading_deg", "-"), play.header.get("site.flat.grade", "-"),
        play.header.get("site.flat.cross_mean", "-"), play.header.get("site.flat.obstacle_hits", "-")))
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("play_dir", help="e21 play dir")
    parser.add_argument("--e22", help="e22 play dir (step 0 and d)")
    parser.add_argument("--e23", help="e23 play dir (more step-0 lag events)")
    parser.add_argument("--baseline", help="1.8 e21 play dir (INFO comparisons)")
    parser.add_argument("--signal-scale-deg", type=float, default=33.75)
    parser.add_argument("--max-step0-error", type=float, default=0.05)
    parser.add_argument("--max-lag-steps", type=float, default=1.0)
    parser.add_argument("--readback-tol", type=float, default=1e-3)
    parser.add_argument("--max-kappa", type=float, default=1e-4)
    parser.add_argument("--min-flat-m", type=float, default=200.0)
    parser.add_argument("--min-cross", type=float, default=0.05)
    parser.add_argument("--angle-tol-deg", type=float, default=0.25)
    parser.add_argument("--angle-tol-share", type=float, default=0.05)
    args = parser.parse_args(argv)
    play = probe_common.Play(args.play_dir)
    if play.probe and play.probe != "e21":
        raise probe_common.InputError("%s is a %s play, not e21" % (play.dir, play.probe))
    verdict = probe_common.Verdict("CC-07 Mode A self-steer (e21, e22)", play.dir)
    verdict.check_footer(play)
    steps = sc.load_steps(play, "e21")
    load = None
    channel = None
    if args.e22:
        load = probe_common.Play(args.e22)
        if load.probe and load.probe != "e22":
            raise probe_common.InputError("%s is a %s play, not e22" % (load.dir, load.probe))
        if load.footer is None or load.footer.get("status") != "complete":
            verdict.check("e22_footer", None if load.footer is None else False, "e22 play %s: status %s" % (
                load.dir, (load.footer or {}).get("status", "no footer")))
        extra = probe_common.Play(args.e23) if args.e23 else None
        selection = step0_selection(load, args, extra)
        channel = selection["selected"]
        verdict.check("step0_channel", channel is not None, selection["detail"])
    else:
        verdict.check("step0_channel", False, "no e22 play (--e22): the wheel-angle channel is not validated")
    baseline_steps = sc.load_steps(probe_common.Play(args.baseline), "e21") if args.baseline else None
    results = [channel is not None]
    results.append(check_readback(steps, verdict, args))
    results.append(check_flat(play, steps, verdict, args, baseline_steps))
    results.append(check_cross(play, steps, verdict, args, channel, baseline_steps))
    if load is not None:
        results.append(check_load(load, verdict, args, channel))
    else:
        verdict.check("d_load", False, "no e22 play (--e22): no data")
        results.append(False)
    info_lines(play, steps, verdict, channel, args)
    if all(results):
        verdict.info("mode_a", "CC-07 PASS on this build: Mode A stays the production default")
    else:
        verdict.info("mode_a", "CC-07 FAIL: Mode B (seated driver, CC-14) becomes the production candidate "
                               "(decision 11)")
    return verdict.report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
