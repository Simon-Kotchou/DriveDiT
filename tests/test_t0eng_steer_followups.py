#!/usr/bin/env python3
"""Follow-ups of the second 1.9.0.37 steering Play (DD-14f): the step-0 lag test on levels too short to settle, the
step-0 error breakdown, the input model's raw-readback diagnostic, CC-07 (a)'s excluded zero-command contexts and the
cross-slope site survey.

    PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_steer_followups.py

The plays are synthetic (tests/test_t0eng_steer.py's SteerCar and builders, written into a fresh
probe_common.make_temp_dir() directory and removed afterwards); the one game-data excerpt is the 1-row pulse below,
22 rows of numbers from that Play's e23 (no launch id, no path).
"""

import importlib.util
import math
import os
import random
import re
import shutil
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ts = _load("test_t0eng_steer_helpers", os.path.join(REPO, "tests", "test_t0eng_steer.py"))
probe_common = ts.probe_common
sc = ts.steer_common
e21 = ts.e21
e23 = ts.e23
enforce_subset = ts.enforce_subset
read = ts.read
DT = ts.DT

# The release of a 1-row +1.0 pulse at 60 km/h (the second 1.9.0.37 Play, e23 steps_60): rows k - 8 .. k + 14 around
# the release k (index 8), cmd_steer, input_post_steer and the rot channel's delta_eff [rad]. The wheel follows the
# input one row later (0.915 at k - 1 -> 0.4546 rad at k); the old lag test read lag 4 here
PULSE_CMD = [0.0] * 7 + [1.0] + [0.0] * 15
PULSE_INPUT = [0.0, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, 0.9149, 0.0708, 0.0052, 0.0008, 0.0001,
               -0.0001, -0.0003, -0.0002, -0.0002, -0.0002, -0.0001, -0.0001, -0.0, 0.0001, 0.0001, 0.0]
PULSE_ROT = [1.4e-05, 3.9e-05, 1.4e-05, -5.9e-05, -1.8e-05, -0.000132, -3e-05, -4.4e-05, 0.454561, 0.037712, -6.5e-05,
             0.000386, -0.000101, -0.000225, -7e-05, -0.000318, -3.6e-05, -0.000245, -3.6e-05, -0.000103, -8e-06,
             7.5e-05, 1.8e-05]


def pulse_steps():
    rows = []
    for k, cmd in enumerate(PULSE_CMD):
        raw = {c: "" for c in sc.STEP_COLUMNS}
        raw.update({"physics_step": str(5600 + k), "timeslice_s": "%.6f" % DT, "phase": "steps_60", "segment": "train",
                    "run_idx": "1", "cmd_steer": "%.4f" % cmd})
        rows.append(sc.Step(raw, k))
    return rows


def rewrite(play_dir, probe, fn):
    """Rewrites <probe>_steps.csv: fn(row dict) -> the row (changed in place), or None to drop it"""
    path = os.path.join(play_dir, probe + "_steps.csv")
    with open(path, "r", encoding="ascii") as handle:
        lines = handle.read().splitlines()
    header = lines[0].split(",")
    out = [lines[0]]
    for line in lines[1:]:
        row = dict(zip(header, line.split(",")))
        row = fn(row)
        if row is not None:
            out.append(",".join(str(row[c]) for c in header))
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(out) + "\n")


def angle_noise(seed=7, sigma=3e-5):
    """fn for rewrite: deterministic noise on the wheel-angle candidates (the real channels carry ~1e-4 rad)"""
    rng = random.Random(seed)

    def fn(row):
        for column in ("wa_rot_l", "wa_rot_r", "wa_bone_l", "wa_bone_r"):
            if row[column]:
                row[column] = "%.6f" % (float(row[column]) + rng.gauss(0.0, sigma))
        return row
    return fn


def build_short_stair_e22(play_dir, rows_per_level):
    """An e22 whose stair levels are each held rows_per_level rows (the ref segment as build_e22's)"""
    car = ts.SteerCar()
    phases = []
    name = "stair_10.5"
    first = car.step_no + 1
    v = 10.5 / 3.6
    ts.drive_start(car, name, 1, v)
    for level, steer in enumerate((0.1, -0.1, 0.2, -0.2, 0.3, -0.3) * 6):
        for k in range(rows_per_level):
            car.run(name, "stair", 1, steer, v, seg_flag=0, event_idx=level, seg_target=10.5)
    ts.drive_stop(car, name, 1)
    phases.append(ts.phase_row(car, 0, name, "stair", "flat", 10.5, 0, -1, first))
    name = "load_+0.30"
    first = car.step_no + 1
    ts.drive_start(car, name, 1, 10.0 / 3.6)
    for k in range(180):
        car.run(name, "ref", 1, 0.3, 10.0 / 3.6, seg_flag=int(k >= 90), seg_target=10.0)
    ts.drive_stop(car, name, 1)
    phases.append(ts.phase_row(car, 1, name, "load", "flat", 10, 0.3, -1, first))
    header = ts.common_header("e22")
    probe_common.write_synthetic_play(play_dir, header, {"e22_steps": (ts.STEP_COLUMNS, car.rows),
                                                         "e22_phases": (ts.PHASE_COLUMNS, phases)}, ts.FOOTER)
    return play_dir


def step0_args(**kw):
    values = dict(signal_scale_deg=33.75, max_step0_error=0.05, max_lag_steps=1.0, max_unmeasurable=0.2)
    values.update(kw)
    return types.SimpleNamespace(**values)


class Step0HoldTests(unittest.TestCase):
    """The step-0 lag on a change whose levels are held too short for the channel to settle"""

    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_dd14f_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def path(self, *parts):
        return os.path.join(self.base, *parts)

    def test_the_hold_minimum_follows_the_lag_allowance(self):
        self.assertEqual([e21.min_hold_rows(x) for x in (0.0, 0.5, 1.0, 2.0, 3.0)], [7, 7, 7, 8, 9])
        self.assertEqual(sc.settle_rows(1.0), sc.SETTLE_FROM)
        self.assertEqual(e21.SETTLE_FROM, sc.SETTLE_FROM)

    def test_a_one_row_pulse_is_the_d3ca_artefact_and_is_excluded(self):
        steps = pulse_steps()
        k = 8                                                    # the release (1.0 -> 0)
        # the old reading: pre value at k - 1 (the wheel had not moved yet), tail of the next level: a lag of 4
        lag, _ = e21.lag_of(steps, PULSE_INPUT, PULSE_ROT, k, 12, DT, e21.SETTLE_FROM, None)
        self.assertNotEqual(lag, 1)
        self.assertEqual(lag, 4)
        old, new = e21.level_holds(steps, k, 20)
        self.assertEqual(old, 1)                                 # the 1.0 level lasted one row: excluded
        self.assertLess(old, e21.min_hold_rows(1.0))
        self.assertGreaterEqual(new, e21.min_hold_rows(1.0))
        # the steepest-step diagnostic reads the same pulse as one step late
        self.assertEqual(e21.steepest_lag(steps, PULSE_INPUT, PULSE_ROT, k, DT, 3), 1)
        self.assertEqual(e21.steepest_lag(steps, PULSE_INPUT, PULSE_ROT, k - 1, DT, 3), 1)    # the onset

    def test_short_holds_are_excluded_and_the_settled_changes_decide(self):
        a = ts.build_e21(self.path("e21", "play_01"))
        b = ts.build_e22(self.path("e22", "play_01"), ts.SteerCar(bone_lag=1))
        c = ts.build_e23(self.path("e23", "play_01"), ts.SteerCar(bone_lag=1),
                         grid="10:0.5x6,1.0x4;30:0.5x6,1.0x4;60:0.5x6,1.0x4")
        rewrite(c, "e23", angle_noise())
        rewrite(b, "e22", angle_noise(seed=9))
        code, text = ts.run_main(e21, [a, "--e22", b, "--e23", c])
        detail = ts.detail(text, "step0_channel")
        self.assertEqual(ts.item(text, "step0_channel"), "PASS", text)
        self.assertIn("selected rot", detail)
        m = re.search(r"rot: median error [^;]*; lag <= 1 steps on (\d+) of (\d+) measurable changes \(max 1\)"
                      r"[^;]*short_hold excluded (\d+) \(steepest-step lag 1: (\d+)\)", detail)
        self.assertIsNotNone(m, detail)
        measured, total, short, steepest = (int(g) for g in m.groups())
        self.assertEqual(measured, total)
        self.assertGreater(short, 0)
        self.assertEqual(steepest, short)
        self.assertIn("with both levels held >= 7 rows, settled from row 6", detail)

    def test_an_all_short_hold_play_never_passes(self):
        a = ts.build_e21(self.path("e21", "play_01"))
        b = build_short_stair_e22(self.path("e22", "play_01"), rows_per_level=4)
        code, text = ts.run_main(e21, [a, "--e22", b])
        self.assertEqual(ts.item(text, "step0_channel"), "INCOMPLETE", text)
        detail = ts.detail(text, "step0_channel")
        self.assertIn("(0 with both levels held >= 7 rows", detail)
        self.assertRegex(detail, r"rot: median error [^;]*; lag <= 1 steps on 0 of 0 measurable changes[^;]*"
                                 r"short_hold excluded [1-9]")
        # a longer allowance asks for longer levels still
        selection = e21.step0_selection(probe_common.Play(b), step0_args(max_lag_steps=3.0))
        self.assertIsNone(selection["state"])
        self.assertIn("held >= 9 rows, settled from row 8", selection["detail"])

    def test_a_two_step_lag_on_settled_levels_still_fails(self):
        code, text = ts.run_main(e21, [ts.build_e21(self.path("e21", "play_01")), "--e22",
                                       ts.build_e22(self.path("e22", "play_01"), ts.SteerCar(bone_lag=2))])
        detail = ts.detail(text, "step0_channel")
        self.assertRegex(detail, r"rot: median error [^;]*; lag <= 1 steps on 0 of 11 measurable changes \(max 2\)"
                                 r"[^;]*-> FAIL")
        self.assertIn("selected ent", detail)

    def test_a_smoothed_channel_on_long_levels_still_fails(self):
        # Codex DD-14f, HIGH: the new level was counted only up to the hold minimum, so even a 90-row level was read at
        # one tail row; a wheel angle smoothed as y = 0.75 y + 0.25 x then passed with lag 1
        b = ts.build_e22(self.path("e22", "play_01"))
        state = {"l": 0.0, "r": 0.0}

        def smooth(row):
            for column, key in (("wa_rot_l", "l"), ("wa_rot_r", "r")):
                state[key] = 0.75 * state[key] + 0.25 * float(row[column])
                row[column] = "%.6f" % state[key]
            return row

        rewrite(b, "e22", smooth)
        code, text = ts.run_main(e21, [ts.build_e21(self.path("e21", "play_01")), "--e22", b])
        detail = ts.detail(text, "step0_channel")
        self.assertRegex(detail, r"rot: median error [^;]*; lag <= 1 steps on \d+ of 11 measurable changes "
                                 r"\(max [2-9]\)[^;]*-> FAIL")
        steps = sc.load_steps(probe_common.Play(b), "e22")
        k = e21.transitions(steps, ("stair",))[0]
        self.assertGreater(e21.level_holds(steps, k, 99)[1], e21.min_hold_rows(1.0))

    def test_e23_plateaus_need_a_settled_hold(self):
        def window(rel, level=1.0):
            y = [0.9] + [level] * (rel - 1) + [0.1] + [0.0] * 10
            return {"release_index": rel, "y": y, "y0": 0.0, "rows": [None] * len(y)}
        self.assertIsNone(e23.plateau(window(4)))                # 40 / 30 km/h holds: the transient rows
        self.assertIsNone(e23.plateau(window(8)))                # rows 5..7 start before the settle
        self.assertAlmostEqual(e23.plateau(window(9)), 1.0)
        self.assertAlmostEqual(e23.plateau(window(12)), 1.0)
        self.assertIsNone(e23.plateau(window(12), settle=10))
        self.assertEqual(e23.centering(window(4)), (None, None))


class Step0ErrorBreakdownTests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_dd14f_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def test_the_breakdown_names_an_asymmetric_channel_and_changes_no_verdict(self):
        a = ts.build_e21(os.path.join(self.base, "e21", "play_01"))
        b = ts.build_e22(os.path.join(self.base, "e22", "play_01"))

        def left_larger(row):
            if float(row["cmd_steer"]) < 0:
                for column in ("wa_rot_l", "wa_rot_r"):
                    row[column] = "%.6f" % (float(row[column]) * 1.04)
            return row

        rewrite(b, "e22", left_larger)
        code, text = ts.run_main(e21, [a, "--e22", b])
        line = ts.detail(text, "step0_error")
        self.assertTrue(line.startswith("rot, L 2.42 m (diagnostic only, the bound stays 5%)"), line)
        right = re.search(r"right \(kappa > 0\) n \d+ median \|err\| ([-\d.e]+)% signed ([-\d.e]+)%", line)
        left = re.search(r"left \(kappa < 0\) n \d+ median \|err\| ([-\d.e]+)% signed ([-\d.e]+)%", line)
        self.assertIsNotNone(right, line)
        self.assertLess(abs(float(right.group(2))), 1.0)
        self.assertGreater(float(left.group(2)), 3.0)
        self.assertRegex(line, r"per level \(right / left\): ref 0\.3: [^;]*; stair 0\.05: \|delta\| [\d.]+ / "
                               r"[\d.]+ deg \(left / right 1\.0[34]\d*\)")
        self.assertIn("a wheelbase / gain scale and an understeer gradient are not separable", line)
        for label in ("by |kappa| 1/m:", "by |a_lat| m/s2:", "by speed km/h:", "by |delta_eff| deg:"):
            self.assertIn(label, line)
        # the verdict is the median error against the unchanged bound, whatever the breakdown says
        m = re.search(r"rot: median error ([\d.]+)% over \d+ rows", ts.detail(text, "step0_channel"))
        state = "PASS" if float(m.group(1)) <= 5.0 else "FAIL"
        self.assertRegex(ts.detail(text, "step0_channel"), r"rot: median error [^;]*; [^;]*-> %s" % state)


def zero_angle_step(k, cmd, kappa):
    raw = {c: "" for c in sc.STEP_COLUMNS}
    raw.update({"physics_step": str(100 + k), "phase": "stair_10.5", "segment": "stair", "cmd_steer": "%.4f" % cmd,
                "speed_kmh": "10.5", "vel_l_z": "2.9167", "angvel_l_y": "%.7f" % (kappa * 2.9167),
                "wa_rot_l": "0.000000", "wa_rot_r": "0.000000"})
    return sc.Step(raw, k)


class ErrorBreakdownGuardTests(unittest.TestCase):
    def test_zero_angles_leave_the_ratios_unavailable(self):
        # Codex DD-14f, MEDIUM: a channel reading 0 on both signs divided by zero and stopped the analysis
        rows = [zero_angle_step(k, 0.1 if k % 2 else -0.1, 0.02 if k % 2 else -0.02) for k in range(40)]
        line = e21.error_breakdown({"low_rows": rows, "L": 2.42}, "rot", None, 0.05)
        self.assertIn("stair 0.1: |delta| 0 / 0 deg (left / right -), |kappa| 0.02 / 0.02 1/m, implied L 0 m", line)
        rows = [zero_angle_step(k, 0.1 if k % 2 else -0.1, 0.0) for k in range(40)]
        self.assertTrue(e21.error_breakdown({"low_rows": rows, "L": 2.42}, "rot", None, 0.05))


class RawReadbackTests(unittest.TestCase):
    """The input model's per-event first-order stage when the post-step readback sometimes shows the raw command"""

    @classmethod
    def setUpClass(cls):
        cls.base = probe_common.make_temp_dir("t0eng_dd14f_")
        def car():
            return ts.SteerCar(actuator=ts.Actuator(tau=0.014))

        cls.clean = ts.build_e23(os.path.join(cls.base, "clean", "play_01"), car(), grid="10:0.5x6")
        cls.raw = ts.build_e23(os.path.join(cls.base, "raw", "play_01"), car(), grid="10:0.5x6")
        state = {"previous": None}

        # events 4..11: the readback equals the command on the row of each of their command changes, the onset and
        # the release (the d3ca pattern; the release row still carries the released event's index)
        def raw_rows(row):
            cmd = row["cmd_steer"]
            changed = state["previous"] is not None and cmd != state["previous"]
            state["previous"] = cmd
            if changed and row["segment"] == "train" and int(row["event_idx"]) >= 4:
                row["input_post_steer"] = "%.5f" % float(cmd)
            return row

        rewrite(cls.raw, "e23", raw_rows)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, True)

    def test_a_raw_readback_phase_is_named_and_stays_unresolved(self):
        code, text = ts.run_main(e23, [self.clean])
        self.assertEqual(ts.item(text, "input_10kmh_0.5"), "PASS", text)
        self.assertNotIn("raw readback", ts.detail(text, "input_10kmh_0.5"))
        code, text = ts.run_main(e23, [self.raw])
        detail = ts.detail(text, "input_10kmh_0.5")
        self.assertEqual(ts.item(text, "input_10kmh_0.5"), "INCOMPLETE", text)
        self.assertRegex(detail, r"tau unresolved \(the stage is active in \d+ of 12 events\)")
        self.assertRegex(detail, r"raw readback on a transient row in [1-9]\d* of 12 events")
        # CC-07 (a)'s model exception names it too and stays unavailable
        args = types.SimpleNamespace(min_events=6, max_cv=0.2, min_r2=0.9)
        state, model, text = e23.input_model(probe_common.Play(self.raw), args)
        self.assertIsNone(state)
        self.assertIsNone(model)
        self.assertIn("raw readback on a transient row", text)


class RawReadbackRowTests(unittest.TestCase):
    def window(self, posts, cmd=0.5):
        rows = []
        for k, post in enumerate(posts):
            raw = {c: "" for c in sc.STEP_COLUMNS}
            raw.update({"cmd_steer": "%.4f" % cmd, "input_post_steer": "%.5f" % post, "timeslice_s": "%.6f" % DT,
                        "phase": "steps_10", "event_idx": "7"})
            rows.append(sc.Step(raw, k))
        return {"rows": rows, "u_e": [0.0] * 8, "offset": 8}

    def test_a_row_the_model_puts_on_the_command_is_no_raw_readback(self):
        # Codex DD-14f, MEDIUM: tau 0.01 s and gain 1 smooth 0.5 to 0.48216, 0.49936, 0.49998: the third row equals
        # the command within 5e-5 by the model itself, so it is no raw readback
        smooth = self.window([0.48216, 0.49936, 0.49998, 0.5])
        self.assertEqual(e23.raw_readback([smooth], e23.Params(d=0, g=1.0, tau=0.01), DT), ([], 0))
        # the d3ca pattern: the first row already on the command where the model says 0.46: named, with its event
        raw = self.window([0.5, 0.4853, 0.4974, 0.4975])
        self.assertEqual(e23.raw_readback([raw], e23.Params(d=0, g=0.995, tau=0.0125), DT), (["steps_10#7"], 1))


class ExcludedContextTests(unittest.TestCase):
    """CC-07 (a): zero-command rows of the probe's own teleport, settle, stop braking and launch below 2 km/h are
    reported apart and not scored (user decision 2026-11-13); everywhere else a deviation still counts"""

    @classmethod
    def setUpClass(cls):
        cls.base = probe_common.make_temp_dir("t0eng_dd14f_")

        def act():
            return ts.Actuator(tau=0.014)

        cls.plays = {"e21": ts.build_e21(os.path.join(cls.base, "e21", "play_01"), ts.SteerCar(actuator=act())),
                     "e22": ts.build_e22(os.path.join(cls.base, "e22", "play_01"), ts.SteerCar(actuator=act())),
                     "e23": ts.build_e23(os.path.join(cls.base, "e23", "play_01"), ts.SteerCar(actuator=act()))}
        cls.case = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, True)

    def session(self, **edits):
        """Copies of the three plays with edits {probe: fn(row) -> row or None}; the e21 argv"""
        ExcludedContextTests.case += 1
        root = os.path.join(self.base, "case%d" % self.case)
        out = {}
        for probe, src in self.plays.items():
            dst = os.path.join(root, probe, "play_01")
            shutil.copytree(src, dst)
            if probe in edits:
                rewrite(dst, probe, edits[probe])
            out[probe] = dst
        return [out["e21"], "--e22", out["e22"], "--e23", out["e23"]]

    def test_the_clean_session_passes_and_reports_the_contexts(self):
        code, text = ts.run_main(e21, self.session())
        self.assertEqual(ts.item(text, "a_readback"), "PASS", text)
        self.assertRegex(ts.detail(text, "a_excluded"), r"^zero-command rows of the probe's own teleport, settle after "
                                                        r"it, stop braking and launch below 2 km/h \(user decision "
                                                        r"2026-11-13\): [1-9]\d* rows, \d+ off by > 0\.001")
        # the release of a steering command into the stop is one of them (e22: the load command, then the stop)
        self.assertRegex(ts.detail(text, "a_excluded"), r"e22/stop [1-9]")

    def test_deviations_in_the_excluded_contexts_are_reported_not_scored(self):
        launch = {"seen": set()}

        def deviate(row):
            if row["segment"] in ("stop", "settle") and float(row["cmd_steer"]) == 0.0:
                row["input_post_steer"] = "0.01200"
            if row["segment"] == "accel" and float(row["speed_kmh"]) < 2.0:
                row["input_post_steer"] = "-0.01400"
            return row

        code, text = ts.run_main(e21, self.session(e21=deviate, e22=deviate))
        self.assertEqual(ts.item(text, "a_readback"), "PASS", text)
        excluded = ts.detail(text, "a_excluded")
        for key, top in (("e21/launch", "0.014"), ("e21/settle", "0.012"), ("e21/stop", "0.012"),
                         ("e22/launch", "0.014"), ("e22/settle", "0.012"), ("e22/stop", "0.012")):
            self.assertRegex(excluded, r"%s [1-9]\d* \(max %s\)" % (key, re.escape(top)))
        self.assertIn("not scored: zero-command rows", ts.detail(text, "a_readback"))
        del launch

    def test_a_zero_command_deviation_anywhere_else_still_fails(self):
        def hold(row):
            if row["phase"] == "flat_20" and row["segment"] == "hold":
                row["input_post_steer"] = "0.01200"
            return row

        def late_accel(row):
            if row["segment"] == "accel" and 9.0 <= float(row["speed_kmh"]) <= 11.0:
                row["input_post_steer"] = "0.01200"
            return row

        for edit, where in ((hold, "e21/hold"), (late_accel, "e21/accel")):
            code, text = ts.run_main(e21, self.session(e21=edit))
            self.assertEqual(ts.item(text, "a_readback"), "FAIL", text)
            scored = ts.detail(text, "a_readback").split("not scored:")[0]
            self.assertRegex(scored, r"%s [1-9]" % where)
            self.assertRegex(ts.detail(text, "a_readback"), r" [1-9]\d* deviating rows not explained")

    def test_the_launch_is_the_accel_start_only(self):
        # a row below 2 km/h after the car first reached it (here: an accel row at ~5 km/h read as 1.5) is scored
        state = {"done": False}

        def dip(row):
            if not state["done"] and row["segment"] == "accel" and 4.5 <= float(row["speed_kmh"]) <= 5.5:
                row["speed_kmh"] = "1.500"
                row["input_post_steer"] = "0.01200"
                state["done"] = True
            return row

        code, text = ts.run_main(e21, self.session(e21=dip))
        self.assertNotEqual(ts.item(text, "a_readback"), "PASS", text)
        self.assertIn("e21/accel 1", ts.detail(text, "a_readback").split("not scored:")[0])

    def test_a_non_zero_command_is_never_excluded(self):
        def settle_command(row):
            if row["segment"] == "settle" and row["phase"] == "stair_10.5":
                row["cmd_steer"] = "0.3000"
                row["input_post_steer"] = "0.20000"
            return row

        code, text = ts.run_main(e21, self.session(e22=settle_command))
        self.assertNotEqual(ts.item(text, "a_readback"), "PASS", text)
        self.assertRegex(ts.detail(text, "a_readback").split("not scored:")[0], r"e22/settle [1-9]")

    def test_the_evidence_gate_counts_the_excluded_rows_too(self):
        # Codex DD-14f, MEDIUM: 100 settle rows and 20 hold rows were 20 scored rows (< 100: INCOMPLETE), and the
        # unexplained hold deviations were never judged
        steps = []
        for k in range(120):
            segment = "settle" if k < 100 else "hold"
            raw = {c: "" for c in sc.STEP_COLUMNS}
            raw.update({"physics_step": str(1000 + k), "timeslice_s": "%.6f" % DT, "phase": "flat_20",
                        "segment": segment, "run_idx": "1", "steer_call": "1", "cmd_steer": "0.0000",
                        "input_post_steer": "0.01200" if segment == "hold" else "0.00000",
                        "speed_kmh": "0.000" if segment == "settle" else "20.000"})
            steps.append(sc.Step(raw, k))

        class OnePhase:
            def has(self, name):
                return True

            def csv(self, name, required=()):
                return [{"phase": "flat_20", "start_step": "1000", "end_step": "1119", "result": "ok"}]

        model = {"structure": "M2", "d": 0, "g": 1.0, "tau": 0.013, "rate": None,
                 "scope": {"cells": [(17.0, 23.0, 1.0)], "text": "17..23 km/h"}}
        args = types.SimpleNamespace(readback_tol=1e-3, bias_tol=2e-4)
        verdict = sc.StrictVerdict("test", "-")
        original = e23.input_model
        e23.input_model = lambda play, args: (True, model, "stub")
        try:
            state = e21.check_readback([("e21", OnePhase(), "e21", steps, DT)], verdict, args, object())
        finally:
            e23.input_model = original
        self.assertIs(state, False, verdict.items)
        text = [d for t, n, d in verdict.items if n == "a_readback"][0]
        self.assertIn("20 scored called rows in e21: 20 off by > 0.001", text)
        self.assertNotIn("< 100 called rows", text)
        # fewer than 100 called rows in all: INCOMPLETE, the observed FAIL kept
        verdict = sc.StrictVerdict("test", "-")
        e23.input_model = lambda play, args: (True, model, "stub")
        try:
            state = e21.check_readback([("e21", OnePhase(), "e21", steps[30:], DT)], verdict, args, object())
        finally:
            e23.input_model = original
        self.assertIsNone(state)
        self.assertIn("[observed FAIL in 1 part(s); withheld: incomplete]",
                      [d for t, n, d in verdict.items if n == "a_readback"][0])

    def test_an_excluded_context_never_hides_missing_rows(self):
        def drop_stop(row):
            return None if row["segment"] == "stop" and row["phase"] == "flat_40" else row

        code, text = ts.run_main(e21, self.session(e21=drop_stop))
        self.assertEqual(ts.item(text, "a_readback"), "INCOMPLETE", text)
        self.assertRegex(ts.detail(text, "a_readback"), r"e21: flat_40: \d+ of \d+ rows")


# ------------------------------------------------------------------------------------------------------------------
# the site survey (DC_SteerSites.c, e21_selfsteer.py)

SITE_SOURCE = ts.SITE_SOURCE
SELF_SOURCE = ts.SELF_SOURCE
BASE_SOURCE = ts.BASE_SOURCE


def survey_class():
    return enforce_subset.translate(read(SITE_SOURCE), ["DC_SteerSurvey"])["DC_SteerSurvey"]


def longest_run(flags):
    best = run = 0
    for ok in flags:
        run = run + 1 if ok else 0
        best = max(best, run)
    return best


class SiteSurveyTests(unittest.TestCase):
    def test_the_pre_check_survey_keeps_the_exact_longest_run(self):
        Survey = survey_class()
        survey = Survey()
        rng = random.Random(11)
        expected_hist = {}
        sampled = 0
        for strip in range(400):
            n = rng.choice((21, 41))
            flags = [rng.random() < rng.choice((0.3, 0.7, 0.95)) for _ in range(n)]
            # ContinuePreCheck's control flow: the refusal at the first station outside, the sampling on while the
            # run can still grow
            survey.StartStrip()
            rejected = False
            for station in range(n):
                if rejected and not survey.CanImprove(n - station):
                    break
                survey.AddStation(flags[station])
                sampled += 1
                if not flags[station]:
                    rejected = True
            self.assertEqual(rejected, not all(flags))
            if rejected:
                self.assertEqual(survey.m_iBestRun, longest_run(flags))
                start = survey.m_iBestRunFrom
                if survey.m_iBestRun:
                    self.assertTrue(all(flags[start:start + survey.m_iBestRun]))
                survey.CountRefused()
                expected_hist[longest_run(flags)] = expected_hist.get(longest_run(flags), 0) + 1
        self.assertEqual(survey.m_iPreRejected, sum(expected_hist.values()))
        self.assertEqual(survey.HistogramText(), ";".join("%d:%d" % kv for kv in sorted(expected_hist.items())))
        self.assertEqual(survey.m_iLongestRun, max(expected_hist))
        keys = enforce_subset.EArray([enforce_subset.f32(v) for v in (0.9, 0.7, 0.7)], "float")
        self.assertEqual(survey.RankPosition(keys, enforce_subset.f32(0.7), 10), 3)     # ties after the earlier
        self.assertEqual(survey.RankPosition(keys, enforce_subset.f32(0.8), 10), 1)
        self.assertEqual(survey.RankPosition(keys, enforce_subset.f32(0.6), 3), -1)
        self.assertEqual(survey.TOP, 10)

    def test_the_producer_runs_the_survey_without_changing_the_refusal(self):
        sites = read(SITE_SOURCE)
        body = sites[sites.index("protected bool ContinuePreCheck("):sites.index("protected void CountReject(")]
        code = " ".join("\n".join(line.split("//", 1)[0] for line in body.split("\n")).split())
        self.assertIn("int stationsLeft = (total - m_iPreIndex + 1) / 2; if (m_bPreRejected && "
                      "!m_Survey.CanImprove(stationsLeft)) break; if (!budget.TakeSamples(1)) return false;", code)
        self.assertIn("bool inWindow = DC_SteerTerrainMath.ZoneCrossOk(c, m_Sampling.m_iSide, cross); "
                      "m_Survey.AddStation(inWindow); if (!inWindow) m_bPreRejected = true;", code)
        self.assertIn("if (m_bPreRejected) { m_Sampling.m_iPreRun = m_Survey.m_iBestRun;", code)
        self.assertIn("m_Survey.CountRefused(); } return true;", code)
        self.assertIn("m_Survey.StartStrip();", sites[sites.index("protected void StartSampling("):])
        # the lists: the near misses by zone share, the runs, the obstacle refusals, each bounded
        reject = sites[sites.index("protected void CountReject("):sites.index("string DescribeRejects()")]
        self.assertIn("AddToList(m_aRunList, m_aRunKeys, site, site.m_iPreRun);", reject)
        self.assertIn("AddToList(m_aNearList, m_aNearKeys, site, site.m_fZoneShare - 0.0001 * site.m_iFailedCount);",
                      reject)
        self.assertIn("while (list.Count() > DC_SteerSurvey.TOP)", reject)
        self.assertIn("if (m_aObstacleList.Count() < DC_SteerSurvey.TOP) m_aObstacleList.Insert(site);",
                      " ".join(sites[sites.index("protected void StepObstacles("):].split()))

    def test_every_survey_row_has_every_column(self):
        sites = read(SITE_SOURCE)
        columns = re.findall(r'"([^"]+)"', re.search(r"array<string> names = \{([^}]*)\};", sites).group(1))
        self.assertEqual(len(columns), 26)
        self.assertEqual(len(set(columns)), 26)

        def adds(text):
            count = len(re.findall(r"row\.Add\w+\(", text))
            for loop in re.finditer(r"for \(int e = 0; e < (\d+); e\+\+\)\s*\{\s*row\.AddEmpty\(\);\s*\}", text):
                count += int(loop.group(1)) - 1
            return count

        write = sites[sites.index("bool WriteSurvey("):sites.index("protected bool WriteSurveySite(")]
        histogram = write[write.index("DC_ProbeRow row = new DC_ProbeRow();"):]
        self.assertEqual(adds(histogram), 26)
        site = sites[sites.index("protected bool WriteSurveySite("):sites.index("string DescribeSearch()")]
        head = site[:site.index('if (list == "pre_run")')]
        pre = site[site.index('if (list == "pre_run")'):site.index("return csv.WriteRow(row);")]
        rest = site[site.index("return csv.WriteRow(row);") + 1:]
        self.assertEqual(adds(head) + adds(pre), 26)
        self.assertEqual(adds(head) + adds(rest), 26)
        # the analysis reads columns the producer writes, from the file the e21 probe opens
        for column in e21.SURVEY_COLUMNS:
            self.assertIn(column, columns)
        selfsteer = read(SELF_SOURCE)
        self.assertIn('m_SurveyCsv = ctx.OpenCsv(PROBE_E21 + "_site_survey", columns);', selfsteer)
        self.assertEqual(e21.SURVEY_FILE, "e21_site_survey")
        self.assertIn('if (m_SurveyCsv && !full.WriteSurvey(m_SurveyCsv, "cross_full"))', selfsteer)
        base = read(BASE_SOURCE)
        steps = base[base.index("protected void StepSearches()"):]
        self.assertLess(steps.index("m_Ctx.AppendHeaderLines(lines)"), steps.index("WriteSiteSurveys();"))
        snapshot = base[base.index("protected void AppendSearchSnapshot("):]
        self.assertIn("WriteSiteSurveys();", snapshot[:snapshot.index("protected bool RefuseStart(")])
        for key in ("pre_rejected=", "zone_stations=", "longest_run_stations=", "longest_run_m=", "run_histogram=",
                    "near_listed=", "obstacle_listed=", "runs_listed="):
            self.assertIn('lines.Insert(p + "%s"' % key, sites)
        self.assertIn('AppendSurveyLines(lines, prefix + ".survey.");', sites)

    def test_apply_lists_every_failed_criterion(self):
        math_ = ts.translated_terrain_math()
        crit = ts.CriteriaStub("cross", half_width=8.0, corridor_half=5.0, corridor_points=ts.corridor_points(5.0),
                               m_fGradeMax=0.02, m_fGradeWindowMax=0.035, m_fGradeWindowMaxAll=0.06, m_fCrossMin=0.05,
                               m_fCrossMax=0.12, m_fCrossAbsMaxAll=0.15, m_fCrossTarget=0.065, m_fRmsMax=0.5)

        def apply(height):
            site = ts.SiteStub(210.0, 8.0, 75.0, 175.0)
            lateral = crit.offsets()
            along = [i * 5.0 for i in range(43)]

            def arr(values):
                return enforce_subset.EArray([enforce_subset.f32(v) for v in values], "float")

            math_.Apply(site, crit, arr(along), arr(lateral), arr([height(a, lat) for a in along for lat in lateral]))
            return site

        good = apply(lambda a, lat: 50.0 - 0.065 * lat)
        self.assertTrue(good.m_bTerrainOk)
        self.assertEqual((good.m_sFailed, good.m_iFailedCount), ("", 0))
        # a 6.5% cross-slope near the shore that also climbs 4%: low, graded, windowed
        bad = apply(lambda a, lat: 2.0 + 0.04 * a - 0.065 * lat)
        self.assertFalse(bad.m_bTerrainOk)
        self.assertEqual(bad.m_sTerrainReason, "below_min_surface_y")
        self.assertEqual(bad.m_sFailed, "below_min_surface_y|grade|grade_window")
        self.assertEqual(bad.m_iFailedCount, 3)


def survey_header(kind, refused, stations, longest, histogram, near=0, blocked=0, runs=0, step="5.00"):
    p = "site.%s." % kind
    return {p + "crit.step_m": step, p + "survey.pre_rejected": str(refused), p + "survey.zone_stations": str(stations),
            p + "survey.longest_run_stations": str(longest), p + "survey.longest_run_m": "%.1f" % ((longest - 1) * 5.0),
            p + "survey.run_histogram": histogram, p + "survey.near_listed": str(near),
            p + "survey.obstacle_listed": str(blocked), p + "survey.runs_listed": str(runs)}


class SurveyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_dd14f_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def test_the_analysis_summarises_the_survey(self):
        header = survey_header("cross", 15040, 21, 14, "0:9000;3:4000;10:2000;11:30;14:10", near=2, blocked=1, runs=2)
        header.update(survey_header("cross_full", 15100, 41, 12, "0:9100;12:6000"))
        play = ts.build_e21(os.path.join(self.base, "e21", "play_01"), cross_found=False, header_extra=header)
        columns = []
        sites = read(SITE_SOURCE)
        columns = re.findall(r'"([^"]+)"', re.search(r"array<string> names = \{([^}]*)\};", sites).group(1))

        def row(**values):
            return ",".join(str(values.get(c, "")) for c in columns)

        lines = [",".join(columns),
                 row(search="cross", list="near_miss", rank=1, source="grid", start_x="3000.4", start_z="756.1",
                     heading_deg="45.25", failed="below_min_surface_y", zone_share="1.0000", cross_min_signed="0.0662",
                     cross_max_signed="0.0943", grade="-0.0181", grade_window_max="0.0275",
                     grade_window_max_all="0.1244", cross_abs_max_all="0.3444", wide_cross_abs_max="0.1156",
                     rms_m="0.200", min_y="1.20", zone_stations=21),
                 row(search="cross", list="near_miss", rank=2, source="grid", failed="below_min_surface_y|grade_window",
                     zone_share="0.9990", zone_stations=21),
                 row(search="cross", list="obstacle", rank=1, source="grid", failed="obstacle",
                     obstacle="SCR_IndestructibleEnvironmentalEntity@2521.9_6.9_3183.4", zone_stations=21),
                 row(search="cross", list="pre_run", rank=1, source="grid", start_x="1000.0", start_z="2000.0",
                     heading_deg="10.00", failed="cross_zone_pre", run_stations=14, run_m="65.0", run_from_m="80.0",
                     zone_stations=21),
                 row(search="cross", list="run_histogram", rank=14, run_stations=14, run_m="65.0", zone_stations=21,
                     strips=10)]
        with open(os.path.join(play, "e21_site_survey.csv"), "w", encoding="ascii") as handle:
            handle.write("\n".join(lines) + "\n")
        code, text = ts.run_main(e21, [play, "--e22", ts.build_e22(os.path.join(self.base, "e22", "play_01"))])
        survey = ts.detail(text, "c_cross_survey")
        self.assertIn("cross_full: 15100 pre-check refusals, zone 41 stations, longest run 12 stations (55.0 m); "
                      "6000 with a run of >= 11 stations (50 m), 0 with a run of >= 16 stations (75 m)", survey)
        self.assertIn("cross: 15040 pre-check refusals, zone 21 stations, longest run 14 stations (65.0 m); 40 with a "
                      "run of >= 11 stations (50 m), 0 with a run of >= 16 stations (75 m)", survey)
        self.assertIn("near misses: #1 below_min_surface_y (share 1.0000, side x cross 0.0662..0.0943", survey)
        self.assertIn("#2 below_min_surface_y|grade_window", survey)
        self.assertIn("(listed near misses failing one criterion only: below_min_surface_y 1)", survey)
        self.assertIn("1 terrain-ok strips listed as refused by an obstacle (first: "
                      "SCR_IndestructibleEnvironmentalEntity", survey)
        self.assertIn("longest runs at 1000.0 2000.0 heading 10.00 (65.0 m)", survey)
        self.assertEqual(ts.item(text, "c_cross"), "INCOMPLETE", text)
        # a play from a producer before the survey
        plain = ts.build_e21(os.path.join(self.base, "plain", "play_01"), cross_found=False)
        code, text = ts.run_main(e21, [plain, "--e22", ts.build_e22(os.path.join(self.base, "e22b", "play_01"))])
        self.assertEqual(ts.detail(text, "c_cross_survey"), "no site survey in this play (a producer before DD-14f)")


if __name__ == "__main__":
    unittest.main()
