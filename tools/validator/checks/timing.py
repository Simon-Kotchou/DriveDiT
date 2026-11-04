"""C08 (cadence: rows per callback, decision and image grids), C28 (physics tick) and C37 (10 Hz image-rate
evidence)."""

import bisect
import math

from .base import FAIL, HARD, NA, NOT_EVALUATED, PASS, SOFT, first, percentile, result, unreadable


# dc-2.0 TBD-2 (freeze decision D-B): 5 Hz is the production default; 10 Hz only as the session setting
# clocks.image_hz_setting trial_conf. image_hz and decision_hz are equal.
IMAGE_HZ_DEFAULT = 5.0
IMAGE_HZ_OVERRIDE = 10.0
SETTING_OF_RATE = {IMAGE_HZ_DEFAULT: "default", IMAGE_HZ_OVERRIDE: "trial_conf"}
# D-B: a 10 Hz session must show its own capture utilisation u95 <= 0.85 (the capture stall p95 over the slot period)
U95_MAX_10HZ = 0.85
STEP_DT_TOL = 0.05          # the measured step dt of section 6 agrees with the engine's within 5%


def check_c08(ctx):
    """Cadence (r2 5.1 C08; dc-2.0.md section 6, freeze decisions D-A and D-B):
      * rates: clocks.image_hz is 5 (clocks.image_hz_setting default) or 10 (setting trial_conf, the non-default
        session setting); decision_hz equals image_hz (the decision interval is the image interval); N =
        clocks.physics_steps_per_decision follows the D-A rule from the measured engine.step_dt_s (N = floor(1000 /
        decision_hz / step_dt_ms + 0.5)), never from engine.fixed_time_slice_s; the thresholds' image_hz,
        telemetry_every_k_physics_steps, decision_hz and C27.physics_steps_per_decision are pre-freeze program values
        and are not used;
      * telemetry: one row per callback: 0 ctl rows without a telemetry row (C08.max_missed_telemetry_rows) and no
        telemetry row after the session's last step;
      * decisions: the grid callbacks of the world-time grid (anchor = world time of the first ctl row) are rebuilt;
        grid points a hitch skipped are reported, not failed, and must equal the writer's count
        session_end.json observed.decision_grid_points_skipped when it is present;
      * image grid: every slot's requested_world_ms on grid point g (anchor + g x 1000 / image_hz - img_lead_ms, the
        same anchor), the grid points of the slots 0, 1, 2, ... without a hole, and every grid point due before the
        session end (less the clocks.img_inflight_timeout_ms drain) has a slot row;
      * issued slots store both timestamps and are issued within C08.img_slot_jitter_frames_p99 render frames of the
        first frame at or past their due time once the anchor exists (p99)."""
    if ctx.load_error:
        return unreadable(ctx, "C08")
    s = ctx.session
    thr = ctx.thr("C08")
    reader = ctx.reader
    issues = []
    clocks = s.clocks
    n, dt = clocks["N"], clocks["step_dt_s"]
    image_hz, dhz, setting = clocks["image_hz"], clocks["decision_hz"], clocks["image_hz_setting"]
    want_setting = SETTING_OF_RATE.get(float(image_hz))
    if want_setting is None:
        issues.append("clocks.image_hz %r is not 5 (the default) or 10 (the trial-conf setting, TBD-2)" % (image_hz,))
    elif setting != want_setting:
        issues.append("clocks.image_hz %r needs clocks.image_hz_setting %r, the session says %r (10 Hz only as an "
                      "explicit non-default session setting, D-B)" % (image_hz, want_setting, setting))
    if abs(float(dhz) - float(image_hz)) > 1e-9:
        issues.append("clocks.decision_hz %r differs from image_hz %r (TBD-2: the decision interval is the image "
                      "interval)" % (dhz, image_hz))
    rule_n = reader.steps_per_decision(dhz, dt)
    if rule_n != n:
        issues.append("clocks.physics_steps_per_decision %d, the D-A rule for step dt %.6f s at %s Hz gives %d"
                      % (n, dt, dhz, rule_n))
    ctl, tel = s.streams["ctl"], s.streams["telemetry"]
    ctl_steps, steps = ctl.col("physics_step"), tel.col("physics_step")
    end_step = s.end_get("end.physics_step") if s.end is not None else None
    if not (isinstance(end_step, int) and end_step >= 0):
        end_step = max(ctl_steps) if ctl_steps else (max(steps) if steps else None)
    tel_set = set(steps)
    missed = sum(1 for p in ctl_steps if p not in tel_set)
    late = [p for p in steps if end_step is not None and p > end_step]
    if late:
        issues.append("%d telemetry row(s) after the session's last step %d" % (len(late), end_step))
    max_missed = thr.get("max_missed_telemetry_rows", 0)
    if missed > max_missed:
        issues.append("%d ctl row(s) without their telemetry row (max %d): telemetry is one row per callback"
                      % (missed, max_missed))
    callbacks, skipped = reader.decision_callbacks(ctl.col("world_time_ms"), dhz)
    written_skipped = s.end_get("observed.decision_grid_points_skipped") if s.end is not None else None
    if isinstance(written_skipped, int) and not isinstance(written_skipped, bool) and written_skipped != skipped:
        issues.append("session_end observed.decision_grid_points_skipped %d, the rebuilt grid skips %d"
                      % (written_skipped, skipped))
    anchor = ctl.col("world_time_ms")[0] if len(ctl) else None
    img = s.streams["img"]
    period = reader.grid_period_ms(image_hz) or 1000.0 / float(image_hz)
    lead = clocks["img_lead_ms"] or 0
    off = holes = tail = 0
    if anchor is None:
        issues.append("no ctl row: the decision and image grid anchor is unknown")
    else:
        expected = 0
        for due in img.col("requested_world_ms"):
            g = int(round((due - anchor + lead) / period))
            if abs(anchor + g * period - lead - due) > 1.0 or g < expected:
                off += 1
                continue
            holes += g - expected
            expected = g + 1
        end_ms = s.end_get("end.world_ms") if s.end is not None else None
        drain = clocks.get("img_inflight_timeout_ms") or 0
        if isinstance(end_ms, int):
            last_due = int(math.floor((end_ms - drain - anchor + lead) / period))
            tail = max(last_due + 1 - expected, 0)
        if off:
            issues.append("%d img row(s) whose requested_world_ms is not the next grid point" % off)
        if holes or tail:
            issues.append("%d grid point(s) without a slot row (%d inside the slot sequence, %d before the session "
                          "end): every grid point reached becomes the next slot (section 6)"
                          % (holes + tail, holes, tail))
    hf = s.streams["hf"]
    hw, frames = hf.col("world_time_ms"), hf.col("render_frame")
    # a slot can be issued once the anchor exists: in the frames after the first callback's frame
    first_cb_frame = ctl.col("render_frame")[0] if len(ctl) else None
    usable = [(w, f) for w, f in zip(hw, frames) if first_cb_frame is not None and f > first_cb_frame]
    ws = [w for w, _ in usable]
    jitters = []
    missing_ts = 0
    for i in range(len(img)):
        if img.col("issue_status")[i] != "issued":
            continue
        due, issue_w, issue_f = img.col("requested_world_ms")[i], img.col("issue_world_ms")[i], \
            img.col("issue_render_frame")[i]
        if issue_w is None or issue_f is None:
            missing_ts += 1
            continue
        j = bisect.bisect_left(ws, due)
        if j >= len(usable):
            continue
        jitters.append(abs(issue_f - usable[j][1]))
    if missing_ts:
        issues.append("%d issued slot(s) without issue_world_ms / issue_render_frame" % missing_ts)
    p99 = percentile(jitters, 99)
    limit = thr.get("img_slot_jitter_frames_p99", 1)
    if p99 is not None and p99 > limit:
        issues.append("image slot jitter p99 %.2f frames > %s" % (p99, limit))
    value = {"missed_telemetry_rows": missed, "decision_callbacks": len(callbacks),
             "decision_grid_points_skipped": skipped, "writer_grid_points_skipped": written_skipped,
             "img_jitter_p99_frames": p99,
             "missing_grid_points": holes + tail, "image_hz": image_hz, "image_hz_setting": setting,
             "decision_hz": dhz, "N": n, "step_dt_s": dt}
    if not steps and not jitters:
        return result(ctx, "C08", HARD, NOT_EVALUATED, value, thr, "no telemetry and no issued image slot")
    return result(ctx, "C08", HARD, FAIL if issues else PASS, value, thr,
                  first(issues) if issues else "N=%d at %s Hz (%s) from step dt %.6f s; %d telemetry rows, one per "
                  "callback; %d decision callbacks (%d grid points skipped); every grid point has a slot; slot jitter "
                  "p99 %s frames over %d slots" % (n, dhz, setting, dt, len(steps), len(callbacks), skipped, p99,
                                                   len(jitters)))


def check_c28(ctx):
    """Physics tick (r2 5.1 C28; dc-2.0.md section 6, D-A): the fixed-step clock advances at 1 / engine.step_dt_s steps
    per world second within C28.tol_frac over the session (counter-based, so a lost row is C06's, not a tick error),
    excluding paused intervals (events pause .. resume); 0 tick gaps, a tick gap being two consecutive ctl rows whose
    world time advanced more than (the steps advanced + 1) step dt + 1 ms, the steps advanced being the later row's
    covered_steps when no row is missing (world time moved without physics steps; probe run 4: 0 in 10215
    callbacks); session.json engine.step_dt_s within 5% of the session's own measure
    (world time span / steps). Rows per full world second are reported."""
    if ctx.load_error:
        return unreadable(ctx, "C28")
    s = ctx.session
    tol = ctx.thr("C28", "tol_frac", 0.01)
    dt_s = s.clocks["step_dt_s"]
    ctl = s.streams["ctl"]
    steps, world, covered = ctl.col("physics_step"), ctl.col("world_time_ms"), ctl.col("covered_steps")
    if len(steps) < 2:
        return result(ctx, "C28", HARD, NOT_EVALUATED, None, {"tol_frac": tol}, "fewer than 2 ctl rows")
    ev = s.streams["events"]
    pauses = []
    start = None
    for kind, w in zip(ev.col("kind"), ev.col("world_time_ms")):
        if kind == "pause":
            start = w
        elif kind == "resume" and start is not None:
            pauses.append((start, w))
            start = None
    dt_ms = dt_s * 1000.0
    gaps = []
    paused_ms = 0.0
    paused_steps = 0
    for i in range(1, len(steps)):
        dstep = steps[i] - steps[i - 1]
        dw = world[i] - world[i - 1]
        if any(a <= world[i - 1] and world[i] <= b + dt_ms for a, b in pauses):
            paused_ms += dw
            paused_steps += dstep
            continue
        if dw > (dstep + 1) * dt_ms + 1.0:
            gaps.append("steps %d..%d: %d ms over %d step(s)" % (steps[i - 1], steps[i], dw, dstep))
    span_ms = (world[-1] - world[0]) - paused_ms
    span_steps = (steps[-1] - steps[0]) - paused_steps
    rate = span_steps / (span_ms / 1000.0) if span_ms > 0 else None
    expected = 1.0 / dt_s
    issues = list(gaps)
    if rate is None or abs(rate / expected - 1.0) > tol:
        issues.insert(0, "physics steps per world second %s, expected %.3f (1 / engine.step_dt_s) +-%.1f%%" % (
            "n/a" if rate is None else "%.3f" % rate, expected, tol * 100))
    if rate is not None and abs(rate * dt_s - 1.0) > STEP_DT_TOL:
        issues.append("engine.step_dt_s %.6f s is not within %.0f%% of the session's %.6f s" % (
            dt_s, 100 * STEP_DT_TOL, 1.0 / rate))
    per_second = {}
    for st, w in zip(steps, world):
        per_second[w // 1000] = per_second.get(w // 1000, 0) + 1
    full = [c for sec, c in per_second.items() if sec != min(per_second) and sec != max(per_second)]
    value = {"steps_per_world_s": rate, "tick_gaps": len(gaps),
             "rows_per_full_second": [min(full), max(full)] if full else None,
             "covered_steps": {str(c): covered.count(c) for c in sorted(set(c for c in covered if c is not None))}}
    return result(ctx, "C28", HARD, FAIL if issues else PASS, value, {"tol_frac": tol},
                  first(issues) if issues else "%.3f steps per world second (expected %.3f), 0 tick gaps"
                  % (rate, expected))


def check_c37(ctx):
    """10 Hz image-rate evidence (SOFT; freeze decision D-B, contract TBD-2): 5 Hz sessions are N/A. A session at the
    non-default 10 Hz setting warns unless it shows its own capture utilisation u95 <= 0.85: u95 is the p95 over the
    issued slots with a done_render_frame of the capture stall (the world time of the captured frame C = done - 1,
    hf row C + 1 minus hf row C; world/real 1.000 in probe runs 3-4, and the engine caps one frame's world time at
    100 ms, so a longer stall reads as u = 1 at 10 Hz) divided by the slot period. done_render_frame is evidence only
    where the decoded fiducial confirms it (captured frame = done - 1): a slot whose decoded frame contradicts it
    makes the evidence untrustworthy (a warning, never a pass), and without a decoded image it cannot be confirmed
    (NOT_EVALUATED). Without done_render_frame or the hf rows the evidence is missing (a warning). Telemetry-only
    sessions are N/A."""
    if ctx.load_error:
        return unreadable(ctx, "C37", SOFT)
    s = ctx.session
    thr = {"u95_max_10hz": U95_MAX_10HZ}
    image_hz = float(s.clocks["image_hz"])
    img = s.streams["img"]
    issued = [i for i in range(len(img)) if img.col("issue_status")[i] == "issued"]
    if image_hz != IMAGE_HZ_OVERRIDE:
        return result(ctx, "C37", SOFT, NA, {"image_hz": image_hz}, thr, "the 5 Hz default needs no evidence")
    if not issued:
        return result(ctx, "C37", SOFT, NA, {"image_hz": image_hz}, thr, "no image request (telemetry only)")
    hf_world = dict(zip(s.streams["hf"].col("render_frame"), s.streams["hf"].col("world_time_ms")))
    period = 1000.0 / image_hz
    decoded = {r["slot_id"]: r["captured_render_frame"] for r in s.images if r["fid_ok"]}
    contradicted = []
    stalls = []
    for i in issued:
        done, slot = img.col("done_render_frame")[i], img.col("slot_id")[i]
        if done is None or slot not in decoded:
            continue
        if decoded[slot] != done - 1:
            contradicted.append(slot)
            continue
        if done - 1 not in hf_world or done not in hf_world:
            continue
        stalls.append((hf_world[done] - hf_world[done - 1]) / period)
    if contradicted:
        return result(ctx, "C37", SOFT, FAIL, {"image_hz": image_hz, "contradicted_slots": len(contradicted),
                                              "issued": len(issued)}, thr,
                      "done_render_frame contradicted by the decoded frame on %d slot(s) (%s): the utilisation "
                      "evidence is refused" % (len(contradicted), first(contradicted, 5)))
    if not decoded:
        return result(ctx, "C37", SOFT, NOT_EVALUATED, {"image_hz": image_hz, "issued": len(issued)}, thr,
                      "no decoded image confirms done_render_frame")
    if len(stalls) < max(1, len(issued) // 2):
        return result(ctx, "C37", SOFT, FAIL, {"image_hz": image_hz, "slots_with_evidence": len(stalls),
                                              "issued": len(issued)}, thr,
                      "10 Hz session without its capture utilisation evidence (done_render_frame on %d of %d issued "
                      "slots): 10 Hz needs u95 <= %.2f (D-B)" % (len(stalls), len(issued), U95_MAX_10HZ))
    u95 = percentile(stalls, 95)
    value = {"image_hz": image_hz, "u95": u95, "slots_with_evidence": len(stalls), "issued": len(issued)}
    return result(ctx, "C37", SOFT, PASS if u95 <= U95_MAX_10HZ else FAIL, value, thr,
                  "capture utilisation u95 %.3f %s %.2f over %d slots at 10 Hz" % (
                      u95, "<=" if u95 <= U95_MAX_10HZ else ">", U95_MAX_10HZ, len(stalls)))
