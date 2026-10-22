"""C08 (telemetry cadence and image slot timing) and C28 (physics tick)."""

import bisect

from .base import FAIL, HARD, NOT_EVALUATED, PASS, first, percentile, result, unreadable


def check_c08(ctx):
    """Cadence (r2 5.1 C08; dc-2.0.md section 6): the session's k and N equal the TBD-1 rule applied to its fixed
    time slice and decision_hz = 1 / (N x slice); telemetry rows only on steps with step mod k == 0 and 0 missed
    rows over the telemetry counter's range (count-based); every image slot's requested_world_ms on the grid
    anchor + g x 1000 / image_hz - img_lead_ms (anchor = world time of physics step 0); issued slots store both
    timestamps and are issued within C08.img_slot_jitter_frames_p99 render frames of the first frame at or past
    their due time (p99)."""
    if ctx.load_error:
        return unreadable(ctx, "C08")
    s = ctx.session
    thr = ctx.thr("C08")
    reader = ctx.reader
    issues = []
    clocks = s.clocks
    slice_s = clocks["slice_s"]
    k, n = clocks["k"], clocks["N"]
    if isinstance(slice_s, (int, float)) and slice_s > 0:
        rule_n, rule_k = reader.tbd1_rule(slice_s)
        if (rule_n, rule_k) != (n, k):
            issues.append("clocks N=%d k=%d, TBD-1 rule for slice %.6f s gives N=%d k=%d" % (n, k, slice_s, rule_n,
                                                                                           rule_k))
        dhz = clocks["decision_hz"]
        if not (isinstance(dhz, (int, float)) and abs(dhz * n * slice_s - 1.0) <= 0.01):
            issues.append("decision_hz %r is not 1 / (N x slice) = %.3f" % (dhz, 1.0 / (n * slice_s)))
    else:
        issues.append("engine.fixed_time_slice_s %r unusable" % (slice_s,))
    steps = s.streams["telemetry"].col("physics_step")
    off_grid = [p for p in steps if p % k]
    missed = 0
    if steps:
        uniq = sorted(set(steps))
        expected = (uniq[-1] - uniq[0]) // k + 1
        missed = max(expected - len([p for p in uniq if p % k == 0]), 0)
    if off_grid:
        issues.append("%d telemetry row(s) off the k=%d grid (%s)" % (len(off_grid), k, first(off_grid, 3)))
    if missed > thr.get("max_missed_telemetry_rows", 0):
        issues.append("%d missed telemetry row(s) (max %d)" % (missed, thr.get("max_missed_telemetry_rows", 0)))
    ctl = s.streams["ctl"]
    anchor = None
    idx0 = ctl.index().get(0)
    if idx0 is not None:
        anchor = ctl.col("world_time_ms")[idx0]
    img = s.streams["img"]
    period = clocks["image_period_ms"]
    lead = clocks["img_lead_ms"] or 0
    off = 0
    if anchor is None:
        issues.append("no ctl row for physics step 0: the image grid anchor is unknown")
    else:
        last_g = -1
        for due in img.col("requested_world_ms"):
            g = round((due - anchor + lead) / period)
            if abs(anchor + g * period - lead - due) > 1.0 or g <= last_g:
                off += 1
            last_g = g
        if off:
            issues.append("%d img row(s) whose requested_world_ms is not the next grid point" % off)
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
             "N": n, "k": k}
    if not steps and not jitters:
        return result(ctx, "C08", HARD, NOT_EVALUATED, value, thr, "no telemetry and no issued image slot")
    return result(ctx, "C08", HARD, FAIL if issues else PASS, value, thr,
                  first(issues) if issues else "k=%d N=%d by the TBD-1 rule; %d telemetry rows, 0 missed; "
                  "slot jitter p99 %s frames over %d slots" % (k, n, len(steps), p99, len(jitters)))


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
