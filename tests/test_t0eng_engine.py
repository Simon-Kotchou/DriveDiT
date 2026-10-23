#!/usr/bin/env python3
"""Tests for the engine-basics analysis scripts e10_ticks.py, e19_write.py, e20_settings.py and e04_components.py
(tools/probes/t0eng/, DD-4-probe) on synthetic plays.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_engine.py

Every fixture is generated here, in the file layout DC_ProbeEngine writes, into a make_temp_dir directory (never
/tmp) that is removed afterwards. Nothing is read from a real capture.
"""

import contextlib
import io
import os
import re
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
T0ENG = os.path.join(REPO, "tools", "probes", "t0eng")
sys.path.insert(0, T0ENG)
import probe_common as pc  # noqa: E402
import e04_components  # noqa: E402
import e10_ticks  # noqa: E402
import e19_write  # noqa: E402
import e20_settings  # noqa: E402

LAUNCH = "20261022T010203Z-wb-abcd"
DT = 1.0 / 60.0
COMPLETE = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done", "note": ""}
ENFORCE_ENGINE = os.path.join(REPO, "src", "enfusion", "datacapture", "DC_ProbeEngine.c")
# e10_steps.csv as DC_ProbeEngine.StartE10 writes it: the columns of the plays written before 2026-10-22 evening, then
# the engine's raw step-callback argument and the steps the dispatched step covers (DC_ProbeRunner STEP TIMING)
E10_FIXTURE_STEP_COLUMNS = ("physics_step", "world_time_ms_pre", "timeslice_pre_s", "render_frame_pre",
                            "world_time_ms_post", "timeslice_post_s", "render_frame_post", "world_ts_ms_post")
E10_RAW_STEP_COLUMNS = ("engine_timeslice_pre_s", "engine_timeslice_post_s", "covered_steps")


def enforce_e10_step_row():
    """(columns, value counts) of the e10_steps.csv row in DC_ProbeEngine.c: the literal stepColumns list of StartE10
    and the number of m_Row.Add* calls on the path that takes every 'if' and on the one that takes every 'else' of the
    row block in OnProbePostSimulate (DC_ProbeCsv.WriteRow fails the probe on any other count)."""
    with open(ENFORCE_ENGINE, "r", encoding="ascii") as handle:
        source = handle.read()
    columns = re.search(r"array<string> stepColumns = \{([^}]*)\};", source)
    block = re.search(r"if \(m_E10Steps\)\s*\{(.*?)m_E10Steps\.WriteRow\(m_Row\);", source, re.S)
    if not columns or not block:
        raise AssertionError("stepColumns or the e10_steps row block not found in %s" % ENFORCE_ENGINE)
    body = block.group(1)
    branch = r"(?:\{[^{}]*\}|[^;{}]*;)"
    if_path = re.sub(r"\belse\s*" + branch, "", body)
    else_path = re.sub(r"\belse\b", "", re.sub(r"\bif\s*\([^()]*\)\s*" + branch, "", body))
    adds = re.compile(r"m_Row\.Add\w+\(")
    return re.findall(r'"([^"]+)"', columns.group(1)), (len(adds.findall(if_path)), len(adds.findall(else_path)))


def header(probe, **extra):
    h = {"schema": "dc-probe-run/1", "probe": probe, "launch.mode": "cli", "launch.id": LAUNCH,
         "engine.fixed_time_slice_s": "%.6f" % DT, "engine.exe": "wb"}
    h.update(extra)
    return h


def run(module, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = module.main(argv)
    return code, out.getvalue()


class EngineAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pc.make_temp_dir("t0eng_engine_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def play(self, probe, n=1):
        return os.path.join(self.tmp, "DC_Probes", probe, LAUNCH, "play_%02d" % n)

    # --------------------------------------------------------------------------------------------
    # E-10
    # --------------------------------------------------------------------------------------------
    def e10_play(self, seconds=10, k=3, drop_step=None, drop_telemetry=None, n=1, step_rows=True, dt=DT,
                 source="simulate", fixed_slice=None, raw_columns=False, covered=None, footer_extra=None):
        """raw_columns: the engine_timeslice_* / covered_steps columns of DC_ProbeEngine (covered: {step: n});
        source fixedframe counts EOnFixedFrame / EOnPostFixedFrame instead of EOnSimulate / EOnPostSimulate;
        fixed_slice: engine.fixed_time_slice_s of the header (default dt)."""
        t0 = 5000.5
        n_steps = int(round(seconds / dt))
        steps, frames, telemetry = [], [], []
        totals = [0, 0]
        fallback = source == "fixedframe"
        for i in range(1, n_steps + 1):
            t = t0 + i * dt * 1000.0
            frame = 100 + i
            posted = i != drop_step
            totals[0] += 1
            totals[1] += 1 if posted else 0
            if posted:
                row = [i, "%.3f" % t, "%.6f" % dt, frame, "%.3f" % t, "%.6f" % dt, frame, "%.3f" % (t - t0)]
                if raw_columns:
                    steps_covered = (covered or {}).get(i, 1)
                    row += ["%.6f" % (dt * steps_covered), "%.6f" % (dt * steps_covered), steps_covered]
                steps.append(row)
            counts = [0, 0, 1, 1 if posted else 0] if fallback else [1, 1 if posted else 0, 0, 0]
            running = [0, 0, totals[0], totals[1]] if fallback else [totals[0], totals[1], 0, 0]
            frames.append([frame, "%.3f" % t, 1000 + i * 17, "%.6f" % dt, 1] + counts + running
                          + ["60.000", 0, source])
            if posted and i % k == 0 and i != drop_telemetry:
                telemetry.append([i, "%.3f" % t, k, "-1.000", "%.3f" % (t - t0)])
        step_columns = list(E10_FIXTURE_STEP_COLUMNS)
        if raw_columns:
            step_columns += list(E10_RAW_STEP_COLUMNS)
        csvs = {
            "e10_frames": (["render_frame", "world_time_ms", "tick_ms", "frame_timeslice_s", "fixed_steps",
                            "n_simulate", "n_postsimulate", "n_fixedframe", "n_postfixedframe", "simulate_total",
                            "postsimulate_total", "fixedframe_total", "postfixedframe_total", "fps_avg", "paused",
                            "step_source"], frames),
            "e10_steps": (step_columns, steps),
            "e10_telemetry": (["physics_step", "world_time_ms", "k", "speed_kmh", "world_ts_ms"], telemetry),
        }
        if not step_rows:
            del csvs["e10_steps"]
        slice_s = dt if fixed_slice is None else fixed_slice
        h = header("e10", **{"params.k": str(k), "params.dt_s": "%.6f" % slice_s,
                             "params.m_fE10TelemetryPeriodS": "0.05", "engine.fixed_time_slice_s": "%.6f" % slice_s})
        footer = dict(COMPLETE)
        footer.update(footer_extra or {})
        return pc.write_synthetic_play(self.play("e10", n), h, csvs, footer=footer)

    def test_e10_sixty_steps_per_second_pass(self):
        code, out = run(e10_ticks, [self.e10_play()])
        self.assertEqual(code, 0, out)
        self.assertIn("[PASS] steps_per_world_second: 60.00 +-1 in 9/9 full seconds", out)
        self.assertIn("[PASS] telemetry_every_k: k=3, 0 missed of 200", out)
        self.assertIn("[INFO] world_time_resolution: undetermined", out)
        self.assertIn("[INFO] world_timestamp: WorldTimestamp", out)
        self.assertTrue(out.rstrip().endswith("RESULT E-10: PASS"), out)

    def test_e10_runner_step_timing_is_info_only(self):
        d = self.e10_play()
        footer = dict(COMPLETE)
        footer.update({"clock.step_source": "fixedframe", "clock.step_decided_by": "frame",
                       "clock.step_dt_s": "0.033333", "clock.step_dt_source": "engine",
                       "clock.measured_fixed_step_s": "0.033357", "clock.engine_fixed_steps": "1069",
                       "clock.merged_fixed_steps": "3", "clock.fixed_time_slice_min_s": "0.100000",
                       "clock.fixed_time_slice_max_s": "0.100000", "clock.fixedframe_timeslice_min_s": "0.033333",
                       "clock.fixedframe_timeslice_max_s": "0.100000",
                       "clock.fixedframe_timeslice_one_step_s": "0.033333",
                       "clock.fixedframe_frame_mismatches": "0"})
        with open(os.path.join(d, pc.RUN_FOOTER), "w", encoding="ascii", newline="") as f:
            f.write("".join("%s=%s\n" % kv for kv in footer.items()))
        code, out = run(e10_ticks, [d])
        self.assertEqual(code, 0, out)
        self.assertIn("[INFO] runner_step_timing: step source fixedframe decided by frame; step dt 0.033333 s "
                      "(engine, 30.00 steps/s); measured fixed step 0.033357 s over 1069 engine fixed steps, 3 merged",
                      out)
        self.assertIn("GetFixedTimeSlice 0.100000..0.100000 s", out)
        self.assertIn("under another frame number than the last EOnFrame 0", out)
        self.assertTrue(out.rstrip().endswith("RESULT E-10: PASS"), out)

    def test_e10_without_runner_step_timing_has_no_info(self):
        code, out = run(e10_ticks, [self.e10_play()])
        self.assertEqual(code, 0, out)
        self.assertNotIn("runner_step_timing", out)
        self.assertNotIn("steps_per_second_vs_step_dt", out)
        self.assertIn("[INFO] engine_timeslice: no engine_timeslice_post_s values", out)

    def test_e10_fallback_fail_against_a_constant_fixed_time_slice_is_marked(self):
        # 1.8.0.13 fallback (stall-fix review R-4): 30 dispatched steps per world second, GetFixedTimeSlice 0.1, step
        # dt 1/30. The r2 rule 1/dt +-1 still FAILs; the detail says why, and the INFO line holds the count against
        # 1/step dt.
        d = self.e10_play(dt=1.0 / 30.0, source="fixedframe", fixed_slice=0.1, k=1,
                          footer_extra={"clock.step_source": "fixedframe", "clock.step_dt_s": "0.033333"})
        code, out = run(e10_ticks, [d])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] steps_per_world_second: 10.00 +-1 in 0/9 full seconds", out)
        self.assertIn("expected in the fixedframe fallback: GetFixedTimeSlice 0.100000 s is not the runner's step dt "
                      "0.033333 s", out)
        self.assertIn("[INFO] steps_per_second_vs_step_dt: 30.00 +-1 (1/clock.step_dt_s 0.033333 s) in 9/9 full "
                      "seconds", out)
        self.assertIn("[PASS] post_simulate_every_step:", out)
        self.assertIn("[PASS] telemetry_every_k:", out)

    def test_e10_a_real_rate_failure_is_not_marked_as_expected(self):
        # Fallback with GetFixedTimeSlice = step dt = 1/30 s, but the steps come at 20/s: a real FAIL, no "expected"
        d = self.e10_play(dt=0.05, source="fixedframe", fixed_slice=1.0 / 30.0, k=1,
                          footer_extra={"clock.step_source": "fixedframe", "clock.step_dt_s": "0.033333"})
        code, out = run(e10_ticks, [d])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] steps_per_world_second: 30.00 +-1 in 0/9 full seconds", out)
        self.assertIn("[INFO] steps_per_second_vs_step_dt: 30.00 +-1 (1/clock.step_dt_s 0.033333 s) in 0/9", out)
        self.assertNotIn("expected in the fixedframe fallback", out)

    def test_e10_engine_timeslice_and_covered_steps_are_info(self):
        d = self.e10_play(dt=1.0 / 30.0, source="fixedframe", k=1, raw_columns=True, covered={100: 2, 200: 3})
        code, out = run(e10_ticks, [d])
        self.assertEqual(code, 0, out)
        self.assertIn("[INFO] engine_timeslice: raw time slice 0.033333..0.100000 s (p50 0.033333) over 300 step rows; "
                      "covered_steps > 1 on 2 row(s), 3 merged engine step(s)", out)

    def test_e10_step_columns_come_from_the_enforce_producer(self):
        columns, adds = enforce_e10_step_row()
        self.assertEqual(columns, list(E10_FIXTURE_STEP_COLUMNS + E10_RAW_STEP_COLUMNS))
        self.assertEqual(adds, (len(columns), len(columns)), "values per e10_steps row (if path, else path)")
        missing = [name for name in e10_ticks.STEP_COLUMNS if name not in columns]
        self.assertEqual(missing, [], "e10_ticks.STEP_COLUMNS names the producer does not write")

    def test_e10_without_step_rows_is_incomplete(self):
        code, out = run(e10_ticks, [self.e10_play(step_rows=False)])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] post_simulate_every_step: e10_steps.csv missing", out)
        self.assertIn("[INCOMPLETE] telemetry_every_k: e10_steps.csv missing", out)

    def test_e10_one_missing_step_fails(self):
        code, out = run(e10_ticks, [self.e10_play(drop_step=301)])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] post_simulate_every_step:", out)
        self.assertIn("1 without a post row (first: 301)", out)

    def test_e10_one_missing_telemetry_row_fails(self):
        code, out = run(e10_ticks, [self.e10_play(drop_telemetry=300)])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] telemetry_every_k: k=3, 1 missed of 200 (first missed: 300)", out)

    def test_e10_without_footer_is_incomplete(self):
        d = self.e10_play()
        os.remove(os.path.join(d, pc.RUN_FOOTER))
        code, out = run(e10_ticks, [d])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] run_footer: no run_footer.txt", out)

    def test_e10_missing_header_is_input_error(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = pc.run_cli(e10_ticks.main, [self.tmp])
        self.assertEqual(code, 2)

    # --------------------------------------------------------------------------------------------
    # E-19
    # --------------------------------------------------------------------------------------------
    def e19_play(self, c05_returned="0", c09="ok", c11_length=68):
        d = self.play("e19")
        rows = [
            ["c01_ascii_row", "new_file_write_17_bytes", "p", "WRITE", 1, 17, 17, 17, 17, "ret:17/len:17", 0, "ok"],
            ["c02_length_param", "write_5_of_16_bytes", "p", "WRITE", 1, 5, 5, 5, 5, "ret:5/len:5", 0, "ok"],
            ["c03_sizes_1", "single_write_of_n_bytes", "p", "WRITE", 1, 1, 1, 1, 1, "ret:1", 0, "ok"],
            ["c03_sizes_1024", "single_write_of_n_bytes", "p", "WRITE", 1, 1024, 1024, 1025, 1025, "ret:1024", 0, "ok"],
            ["c03_sizes_65536", "single_write_of_n_bytes", "p", "WRITE", 1, 65536, 65536, 66561, 66561, "ret:65536",
             0, "ok"],
            ["c03_sizes_1048576", "single_write_of_n_bytes", "p", "WRITE", 1, 1048576, 1048576, 1115137, 1115137,
             "ret:1048576", 0, "ok"],
            ["c03_sizes_total", "four_writes_one_file", "p", "WRITE", 1, 1115137, 1115137, "", 1115137,
             "len:1115137", 0, "ok"],
            ["c04_writeline", "writeline_abc", "p", "WRITE", 1, 3, "", 5, 5, "pos:5_crlf", 0, "observed"],
            ["c06_missing_dir", "open_in_missing_dir", "p", "WRITE", 0, 1, "", "", "", "open_null_or_dir_created", 1,
             "ok"],
            ["c07_dir_as_file", "open_directory_as_file", "p", "WRITE", 0, 1, "", "", "", "open_null_or_short", 1,
             "ok"],
            ["c08_second_writer_h1", "first_of_two_write_handles", "p", "WRITE", 1, 1, 1, 1, "", "observed", 0,
             "observed"],
            ["c08_second_writer_h2", "second_of_two_write_handles", "p", "WRITE", 1, 1, 1, 1, 1, "observed", 0,
             "observed"],
        ]
        if c09 == "prepared":
            rows.append(["c09_readonly_target", "target_created_set_it_read_only", "p", "WRITE", 1, 2, 2, "", "",
                         "ret:0_or_open_null", 0, "prepared"])
        else:
            rows.append(["c09_readonly_target", "append_to_read_only_target", "p", "APPEND", 0, 2, "", "", "",
                         "ret:0_or_open_null", 1, "ok"])
        rows.append(["c10_overwrite_shorter", "rewrite_17_byte_file_with_5_bytes", "p", "WRITE", 1, 5, 5, 5, 5,
                     "len:5_truncates_or_17_keeps_tail", 0, "truncates"])
        rows.append(["c11_append_reopen", "append_reopen_close_per_line", "p", "APPEND", 1, 68, c11_length, "",
                     c11_length, "len:68", int(c11_length != 68), "ok" if c11_length == 68 else "unexpected"])
        detected = "1" if c05_returned != "1" else "0"
        verdict = "ok" if c05_returned != "1" else "unexpected"
        rows.append(["c05_read_handle", "write_through_read_handle", "p", "READ", 1, 1, c05_returned, 0, 17, "ret:0",
                     detected, verdict])
        columns = ["case_id", "description", "path", "mode", "open_ok", "bytes_requested", "bytes_returned",
                   "pos_after", "length_after", "expected", "detected", "verdict"]
        return pc.write_synthetic_play(d, header("e19"), {"e19_cases": (columns, rows)}, footer=COMPLETE)

    def test_e19_alone_is_incomplete_until_the_kill_run(self):
        code, out = run(e19_write, [self.e19_play()])
        self.assertEqual(code, 3, out)
        self.assertIn("[PASS] short_or_failed_writes_detected", out)
        self.assertIn("[PASS] readonly_target_detected", out)
        self.assertIn("[PASS] overwrite_semantics_recorded: WRITE over a 17-byte file with 5 bytes: truncates", out)
        self.assertIn("[PASS] append_reopen_recorded", out)
        self.assertIn("[INFO] c04_writeline: position after WriteLine(\"abc\") = 5 (CR LF)", out)
        self.assertIn("[INCOMPLETE] kill_run", out)
        self.assertIn("RESULT E-19: INCOMPLETE", out)

    def test_e19_first_play_with_a_prepared_target_is_incomplete(self):
        code, out = run(e19_write, [self.e19_play(c09="prepared"), "--kill", self.kill_play()])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] readonly_target_detected", out)
        self.assertIn("prepared", out)

    def test_e19_append_reopen_short_fails(self):
        code, out = run(e19_write, [self.e19_play(c11_length=51), "--kill", self.kill_play()])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] append_reopen_recorded", out)

    def test_e19_read_handle_returning_one_fails(self):
        code, out = run(e19_write, [self.e19_play(c05_returned="1")])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] read_handle_detected", out)
        self.assertIn("[FAIL] short_or_failed_writes_detected", out)

    def kill_play(self, chunks=5, torn=True, uncommitted=True, short_chunk=None, append_log=True):
        d = self.play("e19kill")
        files = {}
        commit = ["chunk_id,first_key,last_key,n_rows,n_bytes"]
        key = 0
        for cid in range(1, chunks + 2):
            rows = []
            first = key + 1
            for _ in range(6):
                key += 1
                rows.append("%d,%.3f,p%09d%s" % (key, 1000.0 + cid * 100, key, "x" * 54))
            data = ("key,world_time_ms,payload\n" + "".join(r + "\n" for r in rows)).encode("ascii")
            if cid <= chunks:
                commit.append("%d,%d,%d,%d,%d" % (cid, first, key, 6, len(data)))
                if cid == short_chunk:
                    data = data[:-1]
                files["chunks/chunk_%05d.csv" % cid] = data
            elif uncommitted:
                files["chunks/chunk_%05d.csv" % cid] = data[:40]     # being written when the kill came
        text = "\n".join(commit) + "\n"
        if torn:
            text += "%d,%d,%d" % (chunks + 1, key - 5, key)          # torn commit line, no LF
        files["e19_commit.csv"] = text.encode("ascii")
        if append_log:
            files["e19_commit_append.csv"] = text.encode("ascii")
        return pc.write_synthetic_play(d, header("e19kill"), {}, footer=None, files=files)

    def test_e19_kill_torn_line_and_uncommitted_chunk_pass(self):
        code, out = run(e19_write, ["--kill", self.kill_play()])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] writer_cases", out)
        self.assertIn("[PASS] commit_log_consistent: 5 of 5 committed chunks verified", out)
        self.assertIn("[PASS] commit_log_append_consistent: 5 of 5 committed chunks verified", out)
        self.assertIn("[INFO] torn_commit_line:", out)
        self.assertIn("[INFO] uncommitted_chunks: e19_commit.csv: 1 trailing chunk file(s)", out)
        self.assertIn("[PASS] kill_happened", out)

    def test_e19_kill_committed_chunk_one_byte_short_fails(self):
        code, out = run(e19_write, ["--kill", self.kill_play(short_chunk=3)])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] commit_log_consistent: 4 of 5 committed chunks verified", out)

    def test_e19_kill_without_the_append_log_is_incomplete(self):
        code, out = run(e19_write, [self.e19_play(), "--kill", self.kill_play(append_log=False)])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] commit_log_append_consistent: e19_commit_append.csv missing", out)

    def test_e19_and_kill_together(self):
        code, out = run(e19_write, [self.e19_play(), "--kill", self.kill_play()])
        self.assertEqual(code, 0, out)
        self.assertIn("RESULT E-19: PASS", out)

    def test_e19_kill_without_kill_is_incomplete(self):
        d = self.kill_play(torn=False, uncommitted=False)
        with open(os.path.join(d, pc.RUN_FOOTER), "w", newline="") as f:
            f.write("status=complete\nreason=done\n")
        code, out = run(e19_write, ["--kill", d])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] kill_happened", out)

    # --------------------------------------------------------------------------------------------
    # E-20
    # --------------------------------------------------------------------------------------------
    def e20_play(self, n=1, max_fps="60", drop_display_module=False, exe="wb"):
        rows = [
            ["engine", "VideoUserSettings", 0, "ResolutionScale", "SCALAR", "1.000000"],
            ["engine", "VideoUserSettings", 1, "MaxFps", "INTEGER", max_fps],
            ["engine", "VideoUserSettings", 2, "Vsync", "BOOLEAN", "0"],
            ["engine", "VideoUserSettings", 3, "Fsaa", "INTEGER", "2"],
            ["engine", "VideoUserSettings", 4, "SomeTexture", "TEXTURE", ""],
            ["game", "SCR_VideoSettings", 0, "m_bNearDofEffect", "BOOLEAN", "1"],
        ]
        if drop_display_module:
            rows.append(["engine", "DisplayUserSettings", -1, "", "missing_module", ""])
        else:
            rows.append(["engine", "DisplayUserSettings", 0, "OverallQuality", "INTEGER", "2"])
        columns = ["root", "module", "var_index", "var_name", "var_type", "value"]
        display = ("workspace_w=1920\nworkspace_h=1080\nrendering_w=1920\nrendering_h=1080\ncli.maxFPS=60\n"
                   "module.engine.VideoUserSettings.vars=5\n").encode("ascii")
        h = header("e20", **{"engine.exe": exe, "params.m_sE20EngineModules": "VideoUserSettings,DisplayUserSettings"})
        return pc.write_synthetic_play(self.play("e20", n), h, {"e20_settings": (columns, rows)}, footer=COMPLETE,
                                       files={"e20_display.txt": display})

    def test_e20_dump_pass(self):
        code, out = run(e20_settings, [self.e20_play()])
        self.assertEqual(code, 0, out)
        self.assertIn("[PASS] engine_modules_dumped: VideoUserSettings 5 variables, DisplayUserSettings 1 variables",
                      out)
        self.assertIn("[INFO] pinned VideoUserSettings.MaxFps: 60 (INTEGER)", out)
        self.assertIn("[INFO] pinned SCR_VideoSettings.m_bNearDofEffect: 1 (BOOLEAN)", out)
        self.assertIn("workspace_w=1920", out)
        self.assertNotIn("module.engine", out)

    def test_e20_missing_engine_module_fails(self):
        code, out = run(e20_settings, [self.e20_play(drop_display_module=True)])
        self.assertEqual(code, 1, out)
        self.assertIn("[FAIL] engine_modules_dumped", out)
        self.assertIn("[INFO] missing_modules: engine/DisplayUserSettings", out)

    def test_e20_compare_and_profile_dir(self):
        a = self.e20_play(1)
        b = self.e20_play(2, max_fps="30", exe="game")
        profile = os.path.join(self.tmp, "profile")
        os.makedirs(os.path.join(profile, ".save", "settings"))
        with open(os.path.join(profile, ".save", "settings", "Engine.conf"), "wb") as f:
            f.write(b"VideoUserSettings {\n MaxFps 60\n}\n")
        with open(os.path.join(profile, "other.txt"), "wb") as f:
            f.write(b"nothing here\n")
        code, out = run(e20_settings, [a, "--compare", b, "--profile-dir", profile])
        self.assertEqual(code, 0, out)
        self.assertIn("[INFO] compare: wb vs game", out)
        self.assertIn("1 different", out)
        self.assertIn("[INFO] differs engine/VideoUserSettings.MaxFps: wb=60 game=30", out)
        self.assertIn("1 file(s) under %s mention VideoUserSettings: %s" % (
            profile, os.path.join(".save", "settings", "Engine.conf")), out)

    # --------------------------------------------------------------------------------------------
    # E-04
    # --------------------------------------------------------------------------------------------
    def e04_play(self, gearbox="1"):
        vehicle = ("prefab={DA79E34823120087}Prefabs/Vehicles/Wheeled/S105/S105_base.et\nentity_class=Vehicle\n"
                   "has_controller=1\nhas_automatic_gearbox=%s\ndriving_assistance_mode=2\n"
                   "driving_assistance_mode_name=FULL\ngearbox_gears=6\ngearbox_forward_gears=4\nwheel_count=4\n"
                   "wheel.0.name=FL\nwheel.1.name=FR\nwheel.2.name=RL\nwheel.3.name=RR\nmass_kg=875.00\n"
                   "bounds_min=-0.800 0.000 -1.900\nbounds_max=0.800 1.400 1.900\n" % gearbox).encode("ascii")
        comps = (["idx", "class_name"], [[0, "CarControllerComponent"], [1, "VehicleWheeledSimulation"]])
        return pc.write_synthetic_play(self.play("e04"), header("e04"), {"e04_components": comps}, footer=COMPLETE,
                                       files={"e04_vehicle.txt": vehicle})

    def test_e04_pass(self):
        code, out = run(e04_components, [self.e04_play()])
        self.assertEqual(code, 0, out)
        self.assertIn("[PASS] gearbox_read: has_automatic_gearbox=1", out)
        self.assertIn("[PASS] components_listed: 2 component classes", out)
        self.assertIn("[INFO] wheels: FL, FR, RL, RR", out)
        self.assertIn("RESULT E-04: PASS", out)

    def test_e04_gearbox_not_read_is_incomplete(self):
        code, out = run(e04_components, [self.e04_play(gearbox="")])
        self.assertEqual(code, 3, out)
        self.assertIn("[INCOMPLETE] gearbox_read", out)


if __name__ == "__main__":
    unittest.main()
