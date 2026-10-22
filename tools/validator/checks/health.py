"""C20 (engine-log triage) and C21 (health: heartbeat, FPS, memory)."""

import re

import numpy

from .base import FAIL, HARD, NOT_EVALUATED, PASS, first, percentile, result, unreadable
from .images import _log_lines

HEARTBEAT_PERIOD_MS = 5000  # dc-2.0.md 4.2: one heartbeat row every 5 s
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


def check_c21(ctx):
    """Health (r2 5.1 C21): heartbeat rows every 5 s with no gap above 2 periods (from the session start to the
    first row, between rows, from the last row to the end) and session_end.heartbeats equal to the rows; FPS p5 of
    hf.fps >= C21.fps_p5_min; memory slope of the heartbeat mem_kb (least squares over unix_s) <
    C21.mem_slope_mb_per_h_max from T1 on (reported in T0). SOFT before T1 (CAPTURE_PROGRAM 4.4 T0-MIN), HARD from
    T1."""
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
    fps = [f for f in s.streams["hf"].col("fps") if f is not None]
    p5 = percentile(fps, 5)
    if p5 is None or p5 < thr.get("fps_p5_min", 30):
        issues.append("FPS p5 %s < %s" % (p5, thr.get("fps_p5_min")))
    slope = None
    if len(hb) >= 2:
        t = numpy.array([r["unix_s"] for r in hb], dtype=float)
        m = numpy.array([r["mem_kb"] for r in hb], dtype=float)
        if numpy.ptp(t) > 0:
            slope = float(numpy.polyfit(t, m, 1)[0]) * 3600.0 / 1024.0
    if slope is not None and ctx.rank >= 2 and slope >= thr.get("mem_slope_mb_per_h_max", 100):
        issues.append("memory slope %.1f MB/h >= %s" % (slope, thr.get("mem_slope_mb_per_h_max")))
    value = {"heartbeats": len(hb), "fps_p5": p5, "mem_slope_mb_per_h": slope}
    return result(ctx, "C21", ctx.severity("C21"), FAIL if issues else PASS, value, thr,
                  first(issues) if issues else "%d heartbeats, FPS p5 %s, memory slope %s MB/h%s" % (
                      len(hb), p5, None if slope is None else round(slope, 1),
                      "" if ctx.rank >= 2 else " (judged from T1)"))
