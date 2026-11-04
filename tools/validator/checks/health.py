"""C20 (engine-log triage) and C21 (health: heartbeat, FPS, memory)."""

import re

import numpy

from .base import FAIL, HARD, NOT_EVALUATED, PASS, first, percentile, result, unreadable
from .images import _log_lines

HEARTBEAT_PERIOD_MS = 5000  # dc-2.0.md 4.2: one heartbeat row every 5 s
# C21 world/real guard: per-frame FPS comes from hf.world_time_ms deltas, which follow real time only while no frame
# exceeds the engine's 100 ms clamp on one frame's world time. Probe runs 2-4 measured world/real 0.999-1.000; run 1
# (capture stalls of about 103 ms, clamped to 100 ms) 0.934. +-2% separates the two with margin
WORLD_REAL_TOL = 0.02
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
    """Per-frame FPS of the hf stream (C21, user decision 2026-11-04): 1000 / (world_time_ms of row n - that of row
    n - 1), the frame time of frame n - 1 as world time records it (the EOnFrame time slice; world/real 1.000 in
    probe runs 3-4), over consecutive unpaused rows with an advancing world time. Rows whose render_frame is an issued
    slot's done_render_frame are the screenshot frames: their delta is the captured frame C's duration (C + 1 = done;
    every frame of 50 ms or more in runs 3-4 was such a row: 5842 of 5842, 5836 of 5836, 707 of 707), the cost
    the image checks already judge. Returns (all [(render_frame, fps)], excluded render frames, problems)."""
    hf = s.streams["hf"]
    frames, world, paused = hf.col("render_frame"), hf.col("world_time_ms"), hf.col("paused")
    rows = []
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1 or paused[i] or paused[i - 1]:
            continue
        dt = world[i] - world[i - 1]
        if dt > 0:
            rows.append((frames[i], 1000.0 / dt))
    problems = []
    done = [r["done_render_frame"] for r in s.images if r["issue_status"] == "issued" and
            r["done_render_frame"] is not None]
    if len(set(done)) != len(done):
        problems.append("%d completed request(s) share a done_render_frame: one request is in flight at a time"
                        % (len(done) - len(set(done))))
    # every excluded row must be the frame after a decoded capture (C13a's tie, done = C + 1)
    off = [r["slot_id"] for r in s.images if r["issue_status"] == "issued" and r["done_render_frame"] is not None
           and r["fid_ok"] and r["done_render_frame"] != r["captured_render_frame"] + 1]
    if off:
        problems.append("%d excluded row(s) are not the frame after their decoded capture (done_render_frame != C + "
                        "1; slots %s): they are not screenshot frames" % (len(off), first(off, 5)))
    return rows, set(done), problems


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
    """Health (r2 5.1 C21; FPS population by the user's decision of 2026-11-04): heartbeat rows every 5 s with no gap
    above 2 periods (from the session start to the first row, between rows, from the last row to the end) and
    session_end.heartbeats equal to the rows; per-frame FPS p5 >= C21.fps_p5_min over the non-screenshot frames
    (frame_fps: 1 / frame time from hf.world_time_ms, the rows at an issued slot's done_render_frame excluded;
    structurally at most one excluded row per completed request, each the frame after its decoded capture, else FAIL);
    memory slope of the heartbeat mem_kb (least squares over unix_s) < C21.mem_slope_mb_per_h_max from T1 on
    (reported in T0). World-time deltas measure frame time only while world time follows wall time: when the
    heartbeat's world/real ratio is certainly outside 1 +- WORLD_REAL_TOL (frames beyond the engine's 100 ms clamp
    lose world time, run 1: 0.934) the check is NOT_EVALUATED, never PASS. Reported: fps_p5 (non-screenshot),
    fps_p5_all, excluded_rows, hf.fps p5 (System.GetFPS(), a smoothed engine rate: information only) and the ratio.
    SOFT before T1 (CAPTURE_PROGRAM 4.4 T0-MIN), HARD from T1."""
    if ctx.load_error:
        return unreadable(ctx, "C21")
    s = ctx.session
    thr = ctx.thr("C21")
    issues = []
    hb = s.heartbeat
    end_ms = s.end_get("end.world_ms") if s.end is not None else None
    marks = [0] + [r["world_ms"] for r in hb] + ([end_ms] if isinstance(end_ms, int) else [])
    gaps = [(a, b) for a, b in zip(marks, marks[1:]) if b - a > 2 * HEARTBEAT_PERIOD_MS]
    if gaps:
        issues.append("%d heartbeat gap(s) > %d ms (%s)" % (len(gaps), 2 * HEARTBEAT_PERIOD_MS, first(gaps, 2)))
    if s.end is not None and s.end_get("heartbeats") != len(hb):
        issues.append("session_end heartbeats %r, heartbeat.csv rows %d" % (s.end_get("heartbeats"), len(hb)))
    rows, excluded, structure = frame_fps(s)
    issues.extend(structure)
    kept = [fps for frame, fps in rows if frame not in excluded]
    p5 = percentile(kept, 5)
    p5_all = percentile([fps for _, fps in rows], 5)
    fps_issue = None
    if p5 is None or p5 < thr.get("fps_p5_min", 30):
        fps_issue = "FPS p5 %s over %d non-screenshot frames < %s" % (
            None if p5 is None else round(p5, 1), len(kept), thr.get("fps_p5_min"))
    engine = [f for f in s.streams["hf"].col("fps") if f is not None]
    slope = None
    if len(hb) >= 2:
        t = numpy.array([r["unix_s"] for r in hb], dtype=float)
        m = numpy.array([r["mem_kb"] for r in hb], dtype=float)
        if numpy.ptp(t) > 0:
            slope = float(numpy.polyfit(t, m, 1)[0]) * 3600.0 / 1024.0
    if slope is not None and ctx.rank >= 2 and slope >= thr.get("mem_slope_mb_per_h_max", 100):
        issues.append("memory slope %.1f MB/h >= %s" % (slope, thr.get("mem_slope_mb_per_h_max")))
    ratio = world_real(s)
    value = {"heartbeats": len(hb), "fps_p5": p5, "fps_p5_all": p5_all, "excluded_rows": len(rows) - len(kept),
             "frames": len(rows), "hf_fps_p5": percentile(engine, 5),
             "world_real": None if ratio is None else round(ratio[0], 4), "mem_slope_mb_per_h": slope}
    if ratio is not None and (ratio[2] < 1.0 - WORLD_REAL_TOL or ratio[1] > 1.0 + WORLD_REAL_TOL):
        # the FPS cannot be judged; the heartbeat, structure and memory rules still can
        reason = ("world/real %.4f (heartbeat, bounds %.4f..%.4f) is outside 1 +- %.2f: world-time deltas do not "
                  "measure frame time (frames beyond the engine's 100 ms clamp lose world time); FPS not judged"
                  % (ratio[0], ratio[1], ratio[2], WORLD_REAL_TOL))
        if issues:
            return result(ctx, "C21", ctx.severity("C21"), FAIL, value, thr, first(issues) + "; " + reason)
        return result(ctx, "C21", ctx.severity("C21"), NOT_EVALUATED, value, thr, reason)
    if fps_issue:
        issues.append(fps_issue)
    return result(ctx, "C21", ctx.severity("C21"), FAIL if issues else PASS, value, thr,
                  first(issues) if issues else "%d heartbeats, FPS p5 %.1f over %d non-screenshot frames (%.1f over "
                  "all %d, %d screenshot rows excluded), world/real %s, memory slope %s MB/h%s" % (
                      len(hb), p5, len(kept), p5_all, len(rows), len(rows) - len(kept),
                      value["world_real"], None if slope is None else round(slope, 1),
                      "" if ctx.rank >= 2 else " (judged from T1)"))
