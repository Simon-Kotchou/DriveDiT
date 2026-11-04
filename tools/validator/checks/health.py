"""C20 (engine-log triage) and C21 (health: heartbeat, FPS, memory)."""

import math
import re

import numpy

from .base import FAIL, HARD, NOT_EVALUATED, PASS, first, percentile, result, unreadable
from .images import _log_lines

HEARTBEAT_PERIOD_MS = 5000  # dc-2.0.md 4.2: one heartbeat row every 5 s
# C21 world/real guard: per-frame FPS comes from hf.world_time_ms deltas, which follow real time only while no frame
# exceeds the engine's 100 ms clamp on one frame's world time. Probe runs 2-4 measured world/real 0.999-1.000; run 1
# (capture stalls of about 103 ms, clamped to 100 ms) 0.934. +-2% separates the two with margin
WORLD_REAL_TOL = 0.02
# C21 FPS evidence floor: at least this many non-screenshot frame-time samples per world second (and FPS_MIN_SAMPLES
# in all). Runs 3-4 at 10 Hz, the heaviest case, gave about 19.6 per second (17630 frames, 5842 screenshot rows in
# 600 s); 5 leaves a factor of 4
FPS_SAMPLES_PER_S = 5
FPS_MIN_SAMPLES = 30
OUR_CLASS_RE = re.compile(r"\b(SCR_ML\w*|SCR_DC\w*|DC_\w+)")
OUR_SOURCE = "Scripts/Game/DataCapture/"
VME = "Virtual Machine Exception"


def triage(logs):
    """HARD and SOFT pattern counts of r1 [TL] 5.3 plus r2's write_fail: {name: [lines]}."""
    hard = {"vme_ours": [], "script_error_ours": [], "crash_log": [], "write_fail": []}
    soft = {"vme_other": [], "rpl_component": []}
    for name, lines in sorted(logs.items()):
        if name.lower() == "crash.log" and any(line.strip() for line in lines):
            hard["crash_log"].append("%s: %d line(s)" % (name, len(lines)))
        for i, line in enumerate(lines):
            if VME in line:
                block = lines[i:i + 12]
                cls = [b for b in block if "Class:" in b]
                if cls and OUR_CLASS_RE.search(cls[0].split("Class:", 1)[1]):
                    hard["vme_ours"].append("%s:%d %s" % (name, i + 1, cls[0].strip()))
                else:
                    soft["vme_other"].append("%s:%d" % (name, i + 1))
            if "SCRIPT" in line and "(E)" in line and OUR_SOURCE in line:
                hard["script_error_ours"].append("%s:%d %s" % (name, i + 1, line.strip()[:160]))
            if "write_fail" in line:
                hard["write_fail"].append("%s:%d %s" % (name, i + 1, line.strip()[:160]))
            if "SCR_BaseGameMode is missing RplComponent!" in line:
                soft["rpl_component"].append("%s:%d" % (name, i + 1))
    return hard, soft


def check_c20(ctx):
    """Engine-log triage (r1 [TL] 5.3, r2 5.3): over the session's sliced engine logs (--engine-logs, default
    <session>/_engine_logs/*.log, written by the harvester) 0 HARD patterns: a Virtual Machine Exception whose
    Class: line names SCR_ML*, SCR_DC* or DC_*, a SCRIPT (E) line from Scripts/Game/DataCapture/, a non-empty
    crash.log, a write_fail line. Vanilla VMEs and the RplComponent line are reported. NOT_EVALUATED without logs."""
    ldir, logs = _log_lines(ctx)
    if logs is None:
        return result(ctx, "C20", HARD, NOT_EVALUATED, None, "0 HARD patterns",
                      "no engine logs (--engine-logs or <session>/_engine_logs/)")
    hard, soft = triage(logs)
    n_hard = sum(len(v) for v in hard.values())
    value = {"hard": {k: len(v) for k, v in hard.items()}, "soft": {k: len(v) for k, v in soft.items()},
             "files": sorted(logs)}
    offenders = [line for v in hard.values() for line in v]
    return result(ctx, "C20", HARD, FAIL if n_hard else PASS, value, "0 HARD patterns",
                  first(offenders) if offenders else "%d log file(s), %d lines: 0 HARD patterns" % (
                      len(logs), sum(len(v) for v in logs.values())))


def frame_fps(s):
    """The per-frame FPS population of C21 (user decision 2026-11-04). Returns {"samples": [(render_frame, fps)],
    "excluded": set of render frames, "problems": [...], "zero_delta": n, "paused_advancing": n, "span_ms": n}.

    Sample: 1000 / (hf.world_time_ms of row n - that of row n - 1), the frame time of frame n - 1 as world time records
    it (the EOnFrame time slice; world/real 1.000 in probe runs 3-4). The paused flag never removes a sample: while the
    game time is paused world time does not advance (dc-2.0 hf.world_time_ms), so a paused frame gives no delta and no
    sample anyway; a row flagged paused whose world time advances counts like any other (review: setting paused on
    slow rows hid them). Rows with no world advance give no sample (zero_delta). A decreasing world time and missing
    hf frames (render_frame not +1 per row) are problems: every frame is needed.

    Excluded: a row is a screenshot frame only with decoded evidence, the frame after its slot's decoded capture
    (issued, fid_ok, done_render_frame = captured C + 1): its delta is C's duration, the cost the image checks judge
    (every frame of 50 ms or more in runs 3-4 was such a row: 5842 of 5842, 5836 of 5836, 707 of 707). A completion
    without decoded evidence stays in the population; a decoded slot whose done_render_frame is not C + 1, and two
    completions in one frame (one request in flight at a time), are problems."""
    hf = s.streams["hf"]
    frames, world, paused = hf.col("render_frame"), hf.col("world_time_ms"), hf.col("paused")
    problems, samples = [], []
    zero = advancing_paused = 0
    missing = sum(max(0, frames[i] - frames[i - 1] - 1) for i in range(1, len(frames)))
    if missing or any(frames[i] <= frames[i - 1] for i in range(1, len(frames))):
        problems.append("hf render_frame is not +1 per row (%d frame(s) missing): per-frame FPS needs every frame"
                        % missing)
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            continue
        dt = world[i] - world[i - 1]
        if dt < 0:
            problems.append("hf world_time_ms decreases at render frame %d" % frames[i])
        elif dt == 0:
            zero += 1
        else:
            samples.append((frames[i], 1000.0 / dt))
            advancing_paused += int(bool(paused[i]))
    issued = [r for r in s.images if r["issue_status"] == "issued" and r["done_render_frame"] is not None]
    done = [r["done_render_frame"] for r in issued]
    if len(set(done)) != len(done):
        problems.append("%d completed request(s) share a done_render_frame: one request is in flight at a time"
                        % (len(done) - len(set(done))))
    off = [r["slot_id"] for r in issued if r["fid_ok"] and r["done_render_frame"] != r["captured_render_frame"] + 1]
    if off:
        problems.append("%d excluded row(s) are not the frame after their decoded capture (done_render_frame != C + "
                        "1; slots %s): they are not screenshot frames" % (len(off), first(off, 5)))
    excluded = set(r["done_render_frame"] for r in issued if r["fid_ok"] and
                   r["done_render_frame"] == r["captured_render_frame"] + 1)
    span = world[-1] - world[0] if len(world) >= 2 else 0
    return {"samples": samples, "excluded": excluded, "problems": problems, "zero_delta": zero,
            "paused_advancing": advancing_paused, "span_ms": span}


def heartbeat_schedule(s, period_ms=HEARTBEAT_PERIOD_MS):
    """Problems of heartbeat.csv against the writer's schedule (DC_Session.HandleFrame): one row in the first frame
    whose world time reaches the next boundary (next = world - world mod period + period, from one period), with the
    world_ms and render_frame of that frame, so every row matches its hf frame and none can be dropped or moved; the
    expected list may hold one more row at the last hf frame (a seal on failure in that frame precedes the row).
    Wall time: unix_s non-decreasing, and world time never outruns it (world <= wall + 1 s per pair, unix_s being
    whole seconds)."""
    hf = s.streams["hf"]
    expected, boundary = [], period_ms
    for frame, w in zip(hf.col("render_frame"), hf.col("world_time_ms")):
        if w >= boundary:
            expected.append((frame, w))
            boundary = w - w % period_ms + period_ms
    got = [(r["render_frame"], r["world_ms"]) for r in s.heartbeat]
    last = hf.col("render_frame")[-1] if len(hf) else None
    problems = []
    if got != expected and not (expected and got == expected[:-1] and expected[-1][0] == last):
        wrong = [(i, g, e) for i, (g, e) in enumerate(zip(got, expected)) if g != e]
        problems.append("heartbeat rows do not follow the %d s world-time schedule of the hf frames: %d row(s), %d "
                        "expected%s" % (period_ms // 1000, len(got), len(expected),
                                        "; row %d is (frame, world_ms) %s, the schedule gives %s" % wrong[0]
                                        if wrong else ""))
    unix = [r["unix_s"] for r in s.heartbeat]
    if any(b < a for a, b in zip(unix, unix[1:])):
        problems.append("heartbeat unix_s decreases")
    outran = [i for i in range(1, len(got)) if
              s.heartbeat[i]["world_ms"] - s.heartbeat[i - 1]["world_ms"] > 1000 * (unix[i] - unix[i - 1] + 1) + 1]
    if outran:
        problems.append("world time outran wall time between heartbeat rows %s (more world ms than unix_s allows)"
                        % first(outran, 3))
    return problems


def world_real(s):
    """(ratio, low, high) of world time over wall time from the heartbeat rows (first to last; unix_s is whole
    seconds, so the wall span is known within +-1 s: low and high bound the ratio); None without two rows a second
    or more apart."""
    hb = s.heartbeat
    if len(hb) < 2:
        return None
    du = hb[-1]["unix_s"] - hb[0]["unix_s"]
    dw = hb[-1]["world_ms"] - hb[0]["world_ms"]
    if du < 1 or dw <= 0:
        return None
    high = dw / (1000.0 * (du - 1)) if du > 1 else float("inf")
    return dw / (1000.0 * du), dw / (1000.0 * (du + 1)), high


def check_c21(ctx):
    """Health (r2 5.1 C21; FPS population by the user's decision of 2026-11-04). FAIL on any of: a heartbeat gap above
    2 periods (from the session start to the first row, between rows, from the last row to the end);
    session_end.heartbeats unequal to the rows; heartbeat rows off the writer's schedule (heartbeat_schedule: each row
    at the first hf frame past its 5 s boundary with that frame's world_ms, so a dropped or edited row fails), unix_s
    decreasing, or world time outrunning wall time; an FPS population problem (frame_fps); per-frame FPS p5 <
    C21.fps_p5_min over the non-screenshot frames; from T1 a memory slope (least squares of mem_kb over unix_s) >=
    C21.mem_slope_mb_per_h_max.

    A world-time FPS can only overstate the real FPS (world time never outruns wall time, and the engine's 100 ms clamp
    only loses world time), so an FPS failure is final; a pass needs more evidence, else NOT_EVALUATED, never PASS:
    at least max(FPS_MIN_SAMPLES, FPS_SAMPLES_PER_S per world second) samples, and a world/real ratio (heartbeat,
    first to last row; unix_s whole seconds bound it) that is, from T1, certainly within 1 +- WORLD_REAL_TOL, which
    needs a heartbeat span of about 51 s, and at T0 / S1 (SOFT, short shakedown sessions) not certainly outside it.
    From T1 a memory slope that cannot be computed (fewer than 2 heartbeat rows or no wall-time span) is NOT_EVALUATED.

    Limit (review of fa37e93, item 2): the FPS distribution trusts the writer's hf.world_time_ms between callbacks.
    Callback frames are pinned (ctl and telemetry carry the same world time, C07) and so are heartbeat frames, but the
    world times of frames without a callback can be redistributed (10/50 ms into 30/30 ms) without any trace: hf.fps
    (System.GetFPS()) is a smoothed rate that sums over such frames, and the fixed-step accumulator is too loose to pin
    them while capturing (the fixed-step clock lags world time by up to 250-310 ms, runs 3-4).

    Reported: fps_p5 (non-screenshot), fps_p5_all, excluded_rows, frames, samples, zero_delta_rows,
    paused_rows_advancing, hf_fps_p5 (System.GetFPS(), a smoothed engine rate, information only), world_real and its
    bounds, mem_slope_mb_per_h. SOFT before T1 (CAPTURE_PROGRAM 4.4 T0-MIN), HARD from T1."""
    if ctx.load_error:
        return unreadable(ctx, "C21")
    s = ctx.session
    thr = ctx.thr("C21")
    issues, undetermined = [], []
    hb = s.heartbeat
    end_ms = s.end_get("end.world_ms") if s.end is not None else None
    marks = [0] + [r["world_ms"] for r in hb] + ([end_ms] if isinstance(end_ms, int) else [])
    gaps = [(a, b) for a, b in zip(marks, marks[1:]) if b - a > 2 * HEARTBEAT_PERIOD_MS]
    if gaps:
        issues.append("%d heartbeat gap(s) > %d ms (%s)" % (len(gaps), 2 * HEARTBEAT_PERIOD_MS, first(gaps, 2)))
    if s.end is not None and s.end_get("heartbeats") != len(hb):
        issues.append("session_end heartbeats %r, heartbeat.csv rows %d" % (s.end_get("heartbeats"), len(hb)))
    issues.extend(heartbeat_schedule(s))
    pop = frame_fps(s)
    issues.extend(pop["problems"])
    kept = [fps for frame, fps in pop["samples"] if frame not in pop["excluded"]]
    p5 = percentile(kept, 5)
    p5_all = percentile([fps for _, fps in pop["samples"]], 5)
    if p5 is None or p5 < thr.get("fps_p5_min", 30):
        issues.append("FPS p5 %s over %d non-screenshot frames < %s" % (
            None if p5 is None else round(p5, 1), len(kept), thr.get("fps_p5_min")))
    need = max(FPS_MIN_SAMPLES, int(math.ceil(FPS_SAMPLES_PER_S * pop["span_ms"] / 1000.0)))
    if len(kept) < need:
        undetermined.append("%d non-screenshot frame-time samples, at least %d needed for %.1f world seconds"
                            % (len(kept), need, pop["span_ms"] / 1000.0))
    ratio = world_real(s)
    if ratio is None:
        if ctx.rank >= 2:
            undetermined.append("no world/real ratio (fewer than 2 heartbeat rows a second apart): world-time deltas "
                                "cannot be tied to wall time")
    elif ratio[2] < 1.0 - WORLD_REAL_TOL or ratio[1] > 1.0 + WORLD_REAL_TOL:
        undetermined.append("world/real %.4f (heartbeat, bounds %.4f..%.4f) is outside 1 +- %.2f: world-time deltas "
                            "do not measure frame time (frames beyond the engine's 100 ms clamp lose world time)"
                            % (ratio[0], ratio[1], ratio[2], WORLD_REAL_TOL))
    elif ctx.rank >= 2 and (ratio[1] < 1.0 - WORLD_REAL_TOL or ratio[2] > 1.0 + WORLD_REAL_TOL):
        undetermined.append("world/real %.4f is not bounded within 1 +- %.2f (bounds %.4f..%.4f: the heartbeat span "
                            "is too short for whole-second unix_s; about 51 s are needed)"
                            % (ratio[0], WORLD_REAL_TOL, ratio[1], ratio[2]))
    engine = [f for f in s.streams["hf"].col("fps") if f is not None]
    slope = None
    if len(hb) >= 2:
        t = numpy.array([r["unix_s"] for r in hb], dtype=float)
        m = numpy.array([r["mem_kb"] for r in hb], dtype=float)
        if numpy.ptp(t) > 0:
            slope = float(numpy.polyfit(t, m, 1)[0]) * 3600.0 / 1024.0
    if ctx.rank >= 2:
        if slope is None:
            undetermined.append("no memory slope (fewer than 2 heartbeat rows with a wall-time span)")
        elif slope >= thr.get("mem_slope_mb_per_h_max", 100):
            issues.append("memory slope %.1f MB/h >= %s" % (slope, thr.get("mem_slope_mb_per_h_max")))
    value = {"heartbeats": len(hb), "fps_p5": p5, "fps_p5_all": p5_all,
             "excluded_rows": len(pop["samples"]) - len(kept), "frames": len(pop["samples"]), "samples": len(kept),
             "samples_needed": need, "zero_delta_rows": pop["zero_delta"],
             "paused_rows_advancing": pop["paused_advancing"], "hf_fps_p5": percentile(engine, 5),
             "world_real": None if ratio is None else round(ratio[0], 4),
             "world_real_bounds": None if ratio is None else [round(ratio[1], 4), round(min(ratio[2], 99.0), 4)],
             "mem_slope_mb_per_h": slope}
    severity = ctx.severity("C21")
    if issues:
        return result(ctx, "C21", severity, FAIL, value, thr, first(issues))
    if undetermined:
        return result(ctx, "C21", severity, NOT_EVALUATED, value, thr, "; ".join(undetermined))
    return result(ctx, "C21", severity, PASS, value, thr,
                  "%d heartbeats on schedule, FPS p5 %.1f over %d non-screenshot frames (%.1f over all %d, %d "
                  "screenshot rows excluded), world/real %s, memory slope %s MB/h%s" % (
                      len(hb), p5, len(kept), p5_all, len(pop["samples"]), value["excluded_rows"],
                      value["world_real"], None if slope is None else round(slope, 1),
                      "" if ctx.rank >= 2 else " (judged from T1)"))
