#!/usr/bin/env python3
"""E-10 tick counting: analysis of one e10 play (DC_ProbeEngine, DD-4-probe).

    python3 -I -B tools/probes/t0eng/e10_ticks.py <e10 play_dir>

r2 4.4 criterion (E-10): GetFixedTimeSlice logged; EOnPostSimulate count per world-second = 1/dt +-1; telemetry
every k steps, 0 missed. Checked here:
  fixed_time_slice_logged   engine.fixed_time_slice_s of the run header is > 0
  steps_per_world_second    post-simulate steps in every full, unpaused world-second = 1/dt +-1. Steps are the
                            e10_steps.csv rows (world_time_ms_post) when present, else the per-frame counts of
                            e10_frames.csv binned by the frame's world time. In the "fixedframe" fallback the
                            EOnPostFixedFrame count is used and EOnPostSimulate is reported as absent.
  post_simulate_every_step  every dispatched step number between the first and the last has its e10_steps.csv row
                            (written in the post call): a single lost post call is invisible to the +-1 above
  telemetry_every_k        e10_telemetry.csv holds exactly the steps that are multiples of k (params.k) within the
                            dispatched step range, each once: 0 missed, 0 duplicated, 0 off-grid
  run_footer                the probe ended normally (status complete) - at best INCOMPLETE otherwise
INFO: k and dt (TBD-1), the step source, the EOnSimulate / EOnFixedFrame totals, the sum of GetFixedSteps() against
the simulate count, the steps-per-frame histogram, whether world time advances per step or per frame (TBD-1), and
post calls without a pre call.

Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error.
"""

import argparse
import collections
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402

LABEL = "E-10 tick counting"
FRAME_COLUMNS = ("render_frame", "world_time_ms", "n_simulate", "n_postsimulate", "n_fixedframe",
                 "n_postfixedframe", "simulate_total", "postsimulate_total", "fixedframe_total",
                 "postfixedframe_total", "fixed_steps", "paused", "step_source")
STEP_COLUMNS = ("physics_step", "world_time_ms_pre", "world_time_ms_post", "render_frame_post")
TELEMETRY_COLUMNS = ("physics_step", "world_time_ms", "k")
MAX_LISTED = 5


def full_second_counts(events, paused_seconds):
    """events: list of (world_ms, count). Returns {second: count} for every full second between the first and the
    last event (both partial seconds dropped), excluding seconds that contain a paused frame."""
    if not events:
        return {}
    first = min(t for t, _ in events)
    last = max(t for t, _ in events)
    lo = int(math.floor(first / 1000.0)) + 1
    hi = int(math.floor(last / 1000.0)) - 1
    counts = {s: 0 for s in range(lo, hi + 1) if s not in paused_seconds}
    for t, n in events:
        s = int(math.floor(t / 1000.0))
        if s in counts:
            counts[s] += n
    return counts


def analyse(play):
    v = pc.Verdict(LABEL, play.dir)
    dt = pc.to_float(play.header.get("engine.fixed_time_slice_s"))
    v.check("fixed_time_slice_logged", dt is not None and dt > 0,
            "engine.fixed_time_slice_s=%s" % play.header.get("engine.fixed_time_slice_s", "(absent)"))

    frames = None
    if play.has("e10_frames.csv"):
        frames = play.csv("e10_frames", FRAME_COLUMNS)
    steps = play.csv("e10_steps", STEP_COLUMNS) if play.has("e10_steps.csv") else None
    telemetry = play.csv("e10_telemetry", TELEMETRY_COLUMNS) if play.has("e10_telemetry.csv") else None

    if frames is None:
        v.check("steps_per_world_second", None, "e10_frames.csv missing")
        v.check("telemetry_every_k", None, "e10_frames.csv missing")
        v.check_footer(play)
        return v

    sources = [r["step_source"] for r in frames if r["step_source"]]
    step_source = sources[-1] if sources else ""
    last = frames[-1] if frames else {}
    totals = {name: pc.to_int(last.get(name), 0) for name in
              ("simulate_total", "postsimulate_total", "fixedframe_total", "postfixedframe_total")}

    paused_seconds = set()
    for r in frames:
        t = pc.to_float(r["world_time_ms"])
        if t is not None and r["paused"] == "1":
            paused_seconds.add(int(math.floor(t / 1000.0)))

    # Steps per full, unpaused world-second
    if step_source == "fixedframe":
        count_column, event_name = "n_postfixedframe", "EOnPostFixedFrame"
    else:
        count_column, event_name = "n_postsimulate", "EOnPostSimulate"
    if steps and step_source in ("simulate", "fixedframe"):
        events = [(t, 1) for t in pc.floats(steps, "world_time_ms_post") if t is not None]
        basis = "e10_steps.csv post times"
    else:
        events = [(t, n or 0) for t, n in zip(pc.floats(frames, "world_time_ms"), pc.ints(frames, count_column))
                  if t is not None]
        basis = "e10_frames.csv %s by frame time" % count_column
    counts = full_second_counts(events, paused_seconds)
    if dt is None or dt <= 0:
        v.check("steps_per_world_second", None, "no fixed time slice to compare with")
    elif not counts:
        v.check("steps_per_world_second", None, "no full unpaused world-second in the data (%s)" % basis)
    else:
        expected = 1.0 / dt
        bad = [(s, c) for s, c in sorted(counts.items()) if abs(c - expected) > 1.0]
        good = len(counts) - len(bad)
        detail = "%.2f +-1 in %d/%d full seconds (dt %.6f s; %s from %s)" % (
            expected, good, len(counts), dt, event_name, basis)
        if bad:
            detail += "; outside: " + ", ".join("s%d=%d" % sc for sc in bad[:MAX_LISTED])
            if len(bad) > MAX_LISTED:
                detail += " ..."
        v.check("steps_per_world_second", not bad, detail)
        values = list(counts.values())
        v.info("steps_per_second_range", "min %d, p50 %.1f, max %d over %d full seconds, %d paused second(s) skipped"
               % (min(values), pc.percentile(values, 50), max(values), len(values), len(paused_seconds)))

    # Every dispatched step got its post call: the +-1 above cannot see a single lost step, this can
    if steps:
        numbers = [s for s in pc.ints(steps, "physics_step") if s is not None]
        if numbers:
            seen = set(numbers)
            lost = [s for s in range(min(numbers), max(numbers) + 1) if s not in seen]
            detail = "%d step rows, steps %d..%d, %d without a post row" % (
                len(numbers), min(numbers), max(numbers), len(lost))
            if lost:
                detail += " (first: %s)" % ", ".join(str(s) for s in lost[:MAX_LISTED])
            v.check("post_simulate_every_step", not lost, detail)
        else:
            v.check("post_simulate_every_step", None, "e10_steps.csv has no rows")

    # Telemetry every k steps
    k = pc.to_int(play.param("k"))
    if telemetry is None:
        v.check("telemetry_every_k", None, "e10_telemetry.csv missing")
    elif not k or k < 1:
        v.check("telemetry_every_k", None, "params.k missing in the run header")
    else:
        tele_steps = [s for s in pc.ints(telemetry, "physics_step") if s is not None]
        if steps:
            dispatched = [s for s in pc.ints(steps, "physics_step") if s is not None]
        else:
            dispatched = list(tele_steps)
        if not dispatched:
            v.check("telemetry_every_k", None, "k=%d, no dispatched steps recorded" % k)
        else:
            first = min(dispatched)
            last_step = max(dispatched)
            expected_steps = list(range(int(math.ceil(first / float(k))) * k, last_step + 1, k))
            expected_steps = [s for s in expected_steps if s > 0]
            present = collections.Counter(tele_steps)
            missed = [s for s in expected_steps if present[s] == 0]
            duplicated = [s for s, n in present.items() if n > 1]
            off_grid = [s for s in present if s % k != 0]
            outside = [s for s in present if s % k == 0 and (s < first or s > last_step)]
            detail = "k=%d, %d missed of %d" % (k, len(missed), len(expected_steps))
            if duplicated:
                detail += ", %d duplicated" % len(duplicated)
            if off_grid:
                detail += ", %d off-grid" % len(off_grid)
            if missed:
                detail += " (first missed: %s)" % ", ".join(str(s) for s in missed[:MAX_LISTED])
            if outside:
                v.info("telemetry_outside_step_range",
                       "%d row(s) outside steps %d..%d" % (len(outside), first, last_step))
            v.check("telemetry_every_k", not missed and not duplicated and not off_grid and expected_steps != [],
                    detail)

    # INFO
    v.info("k_rule", "k=%s, dt_s=%s, period %s s (TBD-1)" % (
        play.param("k", "?"), play.param("dt_s", play.header.get("engine.fixed_time_slice_s", "?")),
        play.param("m_fE10TelemetryPeriodS", "?")))
    v.info("step_source", step_source or "(undecided)")
    v.info("event_totals", "EOnSimulate %d, EOnPostSimulate %d, EOnFixedFrame %d, EOnPostFixedFrame %d, frames %d" % (
        totals["simulate_total"], totals["postsimulate_total"], totals["fixedframe_total"],
        totals["postfixedframe_total"], len(frames)))
    if step_source == "fixedframe" and totals["postsimulate_total"] == 0:
        v.info("eonpostsimulate", "absent on the game mode entity: the fallback EOnPostFixedFrame was counted")
    fixed_sum = sum(n or 0 for n in pc.ints(frames, "fixed_steps"))
    v.info("fixed_steps_vs_simulate", "sum of GetFixedSteps() over frames %d, EOnSimulate %d, EOnFixedFrame %d" % (
        fixed_sum, totals["simulate_total"], totals["fixedframe_total"]))
    hist = collections.Counter(n or 0 for n in pc.ints(frames, count_column))
    v.info("steps_per_frame", ", ".join("%d:%d" % (n, c) for n, c in sorted(hist.items())))
    if steps:
        resolution = world_time_resolution(steps)
        v.info("world_time_resolution", resolution)
        no_pre = sum(1 for r in steps if not r["world_time_ms_pre"])
        v.info("post_without_pre", "%d of %d step rows" % (no_pre, len(steps)))
    v.check_footer(play)
    return v


def world_time_resolution(steps):
    """'per step' when two steps of one render frame have different post world times, else 'per frame'."""
    by_frame = collections.defaultdict(set)
    for r in steps:
        by_frame[r["render_frame_post"]].add(r["world_time_ms_post"])
    multi = [times for times in by_frame.values() if len(times) > 1]
    shared = sum(1 for r in steps if r["render_frame_post"]) - len(by_frame)
    if multi:
        return "per step (%d frame(s) with several steps at different world times)" % len(multi)
    if shared > 0:
        return "per frame (steps of one frame share its world time; TBD-1)"
    return "undetermined (one step per frame)"


def main(argv=None):
    ap = argparse.ArgumentParser(description="E-10 tick counting (r2 4.4)")
    ap.add_argument("play_dir", help="an e10 play directory ($profile/DC_Probes/e10/<launch_id>/play_NN)")
    args = ap.parse_args(argv)
    play = pc.Play(args.play_dir)
    return analyse(play).report()


if __name__ == "__main__":
    sys.exit(pc.run_cli(main))
