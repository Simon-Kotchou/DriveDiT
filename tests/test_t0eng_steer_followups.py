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


if __name__ == "__main__":
    unittest.main()
