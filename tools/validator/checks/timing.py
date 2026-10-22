"""C08 (telemetry cadence and image slot timing) and C28 (physics tick)."""

import bisect
import math

from .base import FAIL, HARD, NOT_EVALUATED, PASS, first, percentile, result, unreadable


IMAGE_HZ_FALLBACK = 5.0    # dc-2.0 TBD-2: the 5 Hz fallback; image_hz and decision_hz drop to 5 Hz together


def check_c08(ctx):
    """Cadence (r2 5.1 C08; dc-2.0.md section 6, TBD-1, TBD-2):
      * rates: clocks.image_hz is the thresholds' image_hz or the TBD-2 fallback (5 Hz); decision_hz equals image_hz
        (the decision interval is the image interval); N and k follow the TBD-1 rule for the session's fixed time
        slice at that decision rate (N = floor(1 / decision_hz / slice + 0.5), k the largest divisor of N with
        k x slice <= 50 ms), so N = 12 at the 5 Hz fallback; decision_hz = 1 / (N x slice) within 1%;
      * telemetry: rows only on steps with step mod k == 0, and 0 missed rows over every k-th step from 0 to the
        session's last step (session_end end.physics_step; count-based, so a lost tail counts);
      * image grid: every slot's requested_world_ms on grid point g (anchor + g x 1000 / image_hz - img_lead_ms,
        anchor = world time of physics step 0), the grid points of the slots 0, 1, 2, ... without a hole, and every
        grid point due before the session end (less the clocks.img_inflight_timeout_ms drain) has a slot row;
      * issued slots store both timestamps and are issued within C08.img_slot_jitter_frames_p99 render frames of the
        first frame at or past their due time (p99)."""
    if ctx.load_error:
        return unreadable(ctx, "C08")
    s = ctx.session
    thr = ctx.thr("C08")
    reader = ctx.reader
    issues = []
    clocks = s.clocks
    slice_s = clocks["slice_s"]
    k, n = clocks["k"], clocks["N"]
    image_hz, dhz = clocks["image_hz"], clocks["decision_hz"]
    allowed = sorted(set([float(ctx.thresholds.get("image_hz", 10)), IMAGE_HZ_FALLBACK]))
    if float(image_hz) not in allowed:
        issues.append("clocks.image_hz %r is not one of %s (the thresholds' image_hz or the TBD-2 fallback)"
                      % (image_hz, allowed))
    dhz_ok = isinstance(dhz, (int, float)) and not isinstance(dhz, bool) and dhz > 0
    if not (dhz_ok and abs(float(dhz) - float(image_hz)) < 1e-9):
        issues.append("clocks.decision_hz %r differs from image_hz %r (TBD-2: the decision interval is the image "
                      "interval)" % (dhz, image_hz))
    if isinstance(slice_s, (int, float)) and slice_s > 0:
        if dhz_ok:
            rule_n, rule_k = reader.tbd1_rule(slice_s, dhz)
            if (rule_n, rule_k) != (n, k):
                issues.append("clocks N=%d k=%d, TBD-1 rule for slice %.6f s at %s Hz gives N=%d k=%d"
                              % (n, k, slice_s, dhz, rule_n, rule_k))
            if abs(dhz * n * slice_s - 1.0) > 0.01:
                issues.append("decision_hz %r is not 1 / (N x slice) = %.3f" % (dhz, 1.0 / (n * slice_s)))
    else:
        issues.append("engine.fixed_time_slice_s %r unusable" % (slice_s,))
    steps = s.streams["telemetry"].col("physics_step")
    ctl_steps = s.streams["ctl"].col("physics_step")
    end_step = s.end_get("end.physics_step") if s.end is not None else None
    if not (isinstance(end_step, int) and end_step >= 0):
        end_step = max(ctl_steps) if ctl_steps else (max(steps) if steps else None)
    off_grid = [p for p in steps if p % k]
    late = [p for p in steps if end_step is not None and p > end_step]
    missed = 0
    if end_step is not None:
        missed = len(set(range(0, end_step + 1, k)) - set(steps))
    if off_grid:
        issues.append("%d telemetry row(s) off the k=%d grid (%s)" % (len(off_grid), k, first(off_grid, 3)))
    if late:
        issues.append("%d telemetry row(s) after the session's last step %d" % (len(late), end_step))
    max_missed = thr.get("max_missed_telemetry_rows", 0)
    if missed > max_missed:
        issues.append("%d missed telemetry row(s) over steps 0..%s (max %d)" % (missed, end_step, max_missed))
    ctl = s.streams["ctl"]
    anchor = None
    idx0 = ctl.index().get(0)
    if idx0 is not None:
        anchor = ctl.col("world_time_ms")[idx0]
    img = s.streams["img"]
    period = 1000.0 / float(image_hz)
    lead = clocks["img_lead_ms"] or 0
    off = holes = tail = 0
    if anchor is None:
        issues.append("no ctl row for physics step 0: the image grid anchor is unknown")
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
    hw, lps, frames = hf.col("world_time_ms"), hf.col("last_physics_step"), hf.col("render_frame")
    usable = [(w, f) for w, step, f in zip(hw, lps, frames) if step is not None and step >= 0]
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
    value = {"missed_telemetry_rows": missed, "off_grid_rows": len(off_grid), "img_jitter_p99_frames": p99,
             "missing_grid_points": holes + tail, "image_hz": image_hz, "decision_hz": dhz, "N": n, "k": k}
    if not steps and not jitters:
        return result(ctx, "C08", HARD, NOT_EVALUATED, value, thr, "no telemetry and no issued image slot")
    return result(ctx, "C08", HARD, FAIL if issues else PASS, value, thr,
                  first(issues) if issues else "k=%d N=%d by the TBD-1 rule at %s Hz; %d telemetry rows, 0 missed; "
                  "every grid point has a slot; slot jitter p99 %s frames over %d slots"
                  % (k, n, dhz, len(steps), p99, len(jitters)))


def check_c28(ctx):
    """Physics tick (r2 5.1 C28): ctl rows advance physics_step at 1 / fixed_time_slice per world second within
    C28.tol_frac over the session (counter-based, so a lost row is C06's, not a tick error), excluding paused
    intervals (events pause .. resume); 0 tick gaps, a tick gap being two consecutive ctl rows whose world time
    advanced more than (steps advanced + 1) slices + 1 ms (world time moved without physics steps). Rows per
    full world second are reported."""
    if ctx.load_error:
        return unreadable(ctx, "C28")
    s = ctx.session
    tol = ctx.thr("C28", "tol_frac", 0.01)
    slice_s = s.clocks["slice_s"]
    ctl = s.streams["ctl"]
    steps, world = ctl.col("physics_step"), ctl.col("world_time_ms")
    if len(steps) < 2 or not (isinstance(slice_s, (int, float)) and slice_s > 0):
        return result(ctx, "C28", HARD, NOT_EVALUATED, None, {"tol_frac": tol}, "fewer than 2 ctl rows or no slice")
    ev = s.streams["events"]
    pauses = []
    start = None
    for kind, w in zip(ev.col("kind"), ev.col("world_time_ms")):
        if kind == "pause":
            start = w
        elif kind == "resume" and start is not None:
            pauses.append((start, w))
            start = None
    slice_ms = slice_s * 1000.0
    gaps = []
    paused_ms = 0.0
    paused_steps = 0
    for i in range(1, len(steps)):
        dstep = steps[i] - steps[i - 1]
        dw = world[i] - world[i - 1]
        if any(a <= world[i - 1] and world[i] <= b + slice_ms for a, b in pauses):
            paused_ms += dw
            paused_steps += dstep
            continue
        if dw > (dstep + 1) * slice_ms + 1.0:
            gaps.append("steps %d..%d: %d ms" % (steps[i - 1], steps[i], dw))
    span_ms = (world[-1] - world[0]) - paused_ms
    span_steps = (steps[-1] - steps[0]) - paused_steps
    rate = span_steps / (span_ms / 1000.0) if span_ms > 0 else None
    expected = 1.0 / slice_s
    issues = list(gaps)
    if rate is None or abs(rate / expected - 1.0) > tol:
        issues.insert(0, "physics steps per world second %s, expected %.3f +-%.1f%%" % (
            "n/a" if rate is None else "%.3f" % rate, expected, tol * 100))
    per_second = {}
    for st, w in zip(steps, world):
        per_second[w // 1000] = per_second.get(w // 1000, 0) + 1
    full = [c for sec, c in per_second.items() if sec != min(per_second) and sec != max(per_second)]
    value = {"steps_per_world_s": rate, "tick_gaps": len(gaps),
             "rows_per_full_second": [min(full), max(full)] if full else None}
    return result(ctx, "C28", HARD, FAIL if issues else PASS, value, {"tol_frac": tol},
                  first(issues) if issues else "%.3f steps per world second (expected %.3f), 0 tick gaps"
                  % (rate, expected))
