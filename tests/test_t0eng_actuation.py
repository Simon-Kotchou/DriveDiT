#!/usr/bin/env python3
"""Synthetic-fixture tests for tools/probes/t0eng/e01_actuation.py (E-01 and E-02a, DD-4-probe).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_actuation.py

Every case writes a synthetic play directory (run_header.txt, the probe CSVs, run_footer.txt) with
probe_common.write_synthetic_play into a fresh probe_common.make_temp_dir() directory (RAM or E:, never /tmp) and
removes it afterwards. The fixtures are generated here from a small vehicle model; nothing binary is committed.

  - E-01: a 4-step steering dead time with a first-order yaw lag, steer_sign -1, a 6-step longitudinal dead time and
    an automatic gearbox that shifts up -> every identification has a value, PASS, and the identified values match;
  - E-01 with a car that never moves -> FAIL (moves), and ForceEnableSimulation "does not help";
  - E-02a with kappa = G(v) * s -> R2 ~ 1, PASS; one speed with only 2 valid cells -> FAIL;
  - the PRBS generator mirrors DC_ProbePrbs: maximum length for every supported order;
  - the step-CSV header of the fixtures is built from the Enforce source (DC_ProbeSequences.c: the stepColumns list,
    DC_ProbeActuator.AddStepColumns and DC_ProbeCsv.AddVectorColumns' _x/_y/_z), and every column the reader
    requires is in it, so a reader/producer name mismatch fails here instead of on the first real play.
"""

import contextlib
import importlib.util
import io
import math
import os
import random
import re
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
T0ENG = os.path.join(REPO, "tools", "probes", "t0eng")
sys.path.insert(0, T0ENG)
import probe_common  # noqa: E402


def load_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e01 = load_tool("e01_actuation", os.path.join(T0ENG, "e01_actuation.py"))

DT = 1.0 / 60.0
ENFORCE_SEQUENCES = os.path.join(REPO, "src", "enfusion", "datacapture", "DC_ProbeSequences.c")


def enforce_step_columns(path=ENFORCE_SEQUENCES):
    """The <probe>_steps.csv header exactly as DC_ProbeSequences.OnProbeStart builds it: the literal stepColumns list,
    DC_ProbeActuator.AddStepColumns (columns.Insert and DC_ProbeCsv.AddVectorColumns -> _x, _y, _z), then the
    stepColumns.Insert calls after it."""
    with open(path, "r", encoding="ascii") as handle:
        source = handle.read()
    start = re.search(r"array<string> stepColumns = \{([^}]*)\};", source)
    body = re.search(r"static void AddStepColumns\(notnull array<string> columns\)\s*\{(.*?)\n\t\}", source, re.S)
    if not start or not body:
        raise AssertionError("stepColumns or AddStepColumns not found in %s" % path)
    columns = re.findall(r'"([^"]+)"', start.group(1))
    for match in re.finditer(r'columns\.Insert\("([^"]+)"\)|AddVectorColumns\(columns, "([^"]+)"\)', body.group(1)):
        if match.group(1):
            columns.append(match.group(1))
        else:
            columns.extend(match.group(2) + suffix for suffix in ("_x", "_y", "_z"))
    tail = source[start.end():]
    after = tail[tail.index("AddStepColumns(stepColumns);"):tail.index("OpenCsv(probeId + \"_steps\"")]
    columns.extend(re.findall(r'stepColumns\.Insert\("([^"]+)"\)', after))
    return columns


STEP_COLUMNS = enforce_step_columns()
PHASE_COLUMNS = ["phase_idx", "phase", "start_step", "end_step", "start_world_ms", "end_world_ms", "target_kmh",
                 "amplitude", "force_sim", "result", "note"]
CELL_COLUMNS = ["cell_idx", "target_kmh", "steer", "start_step", "excite_start_step", "end_step", "result", "note"]


class Car:
    """Point-mass car with a dead time and a first-order lag on the yaw rate, a longitudinal dead time and an
    automatic gearbox. Writes one e01/e02a step row per physics step."""

    def __init__(self, steer_delay=4, yaw_tau=6.0, long_delay=6, steer_sign=-1, yaw_gain_deg=30.0, moves=True,
                 seed=7):
        self.rng = random.Random(seed)
        self.steer_delay = steer_delay
        self.alpha = 1.0 - math.exp(-1.0 / yaw_tau)
        self.long_delay = long_delay
        self.steer_sign = steer_sign
        self.yaw_gain_deg = yaw_gain_deg
        self.moves = moves
        self.kappa_fn = None          # E-02a: kappa(v_kmh, s) in 1/m replaces the yaw model
        self.step = 0
        self.speed = 0.0              # m/s
        self.heading = 0.0            # deg
        self.x, self.z = 1495.0, 2626.0
        self.yaw_rate = 0.0           # deg/s
        self.acc = 0.0
        self.cmd_hist = []
        self.long_hist = []
        self.rows = []

    def reset(self):
        self.speed = 0.0
        self.heading = 0.0
        self.yaw_rate = 0.0
        self.acc = 0.0
        self.x, self.z = 1495.0, 2626.0
        self.cmd_hist = []
        self.long_hist = []

    def run(self, phase, segment, phase_step, steer=0.0, throttle=0.0, brake=0.0, gear=-1, handbrake=False,
            hold_kmh=-1.0):
        self.step += 1
        self.cmd_hist.append(steer)
        self.long_hist.append((throttle, brake, handbrake))
        delayed_steer = self.cmd_hist[-1 - self.steer_delay] if len(self.cmd_hist) > self.steer_delay else 0.0
        thr, brk, hb = self.long_hist[-1 - self.long_delay] if len(self.long_hist) > self.long_delay else (0, 1, True)
        if not self.moves:
            self.acc = 0.0
        elif hb or brk > 0:
            self.acc = -8.0 * max(brk, 1.0 if hb else 0.0) if self.speed > 0.05 else 0.0
        else:
            self.acc = 3.0 * thr - 0.02 * self.speed
        self.acc += self.rng.gauss(0, 0.01)
        self.speed = max(0.0, self.speed + self.acc * DT) if self.moves else 0.0
        if self.kappa_fn is not None:
            target = math.degrees(self.kappa_fn(self.speed * 3.6, delayed_steer) * self.speed)
            self.yaw_rate = target
        elif self.speed > 0.5:
            target = self.steer_sign * self.yaw_gain_deg * delayed_steer
            self.yaw_rate += (target - self.yaw_rate) * self.alpha
        else:
            self.yaw_rate = 0.0
        self.heading = self.heading + self.yaw_rate * DT + self.rng.gauss(0, 1e-5)
        rad = math.radians(self.heading)
        self.x += self.speed * math.sin(rad) * DT
        self.z += self.speed * math.cos(rad) * DT
        kmh = self.speed * 3.6
        cur_gear = 1 + min(3, int(kmh / 20.0)) if gear < 0 else gear
        self.rows.append({
            "physics_step": self.step, "world_time_ms": "%.3f" % (self.step * DT * 1000), "timeslice_s": "%.6f" % DT,
            "phase": phase, "segment": segment, "phase_step": phase_step, "force_sim": 0,
            "cmd_steer": "%.4f" % steer, "cmd_throttle": "%.4f" % throttle, "cmd_brake": "%.4f" % brake,
            "cmd_clutch": "1.0000", "cmd_gear": gear, "cmd_handbrake": int(handbrake),
            "speed_hold_kmh": "%.3f" % hold_kmh, "input_pre_steer": "%.4f" % steer,
            "input_pre_throttle": "%.4f" % throttle, "input_pre_brake": "%.4f" % brake,
            "input_pre_clutch": "1.0000", "input_pre_gear": cur_gear, "input_pre_handbrake": int(handbrake),
            "input_post_steer": "%.4f" % steer, "input_post_throttle": "%.4f" % throttle,
            "input_post_brake": "%.4f" % brake, "input_post_clutch": "1.0000", "input_post_gear": cur_gear,
            "input_post_handbrake": int(handbrake), "cur_gear": cur_gear, "fut_gear": cur_gear,
            "rpm": "%.1f" % (900 + 40 * kmh), "engine_on": 1, "speed_kmh": "%.3f" % kmh,
            "vel_l_x": "0.000", "vel_l_y": "0.000", "vel_l_z": "%.3f" % self.speed,
            "angvel_l_x": "0.000000", "angvel_l_y": "%.6f" % math.radians(self.yaw_rate), "angvel_l_z": "0.000000",
            "acc_l_x": "%.4f" % (self.speed * math.radians(self.yaw_rate)), "acc_l_y": "0.0000",
            "acc_l_z": "%.4f" % self.acc,
            "pos_x": "%.3f" % self.x, "pos_y": "40.000", "pos_z": "%.3f" % self.z,
            "fwd_x": "%.6f" % math.sin(rad), "fwd_y": "0.000000", "fwd_z": "%.6f" % math.cos(rad),
            "heading_deg": "%.5f" % self.heading, "ground_contact": 1, "wheels_contact": 4, "in_area": 1})
        return kmh


def build_e01(play_dir, moves=True):
    car = Car(moves=moves)
    phases = []

    def phase(name, target_kmh, amplitude, body):
        car.reset()
        start = car.step + 1
        n = 0
        for _ in range(60):
            car.run(name, "settle", n, brake=1.0, handbrake=True)
            n += 1
        result = body(name, n)
        phases.append({"phase_idx": len(phases), "phase": name, "start_step": start, "end_step": car.step,
                       "start_world_ms": "", "end_world_ms": "", "target_kmh": target_kmh, "amplitude": amplitude,
                       "force_sim": 0, "result": result, "note": "phb_resets:0"})
        return result

    def accel(name, n, target):
        for _ in range(1500):
            throttle = 1.0 if car.speed * 3.6 < target else 0.2
            kmh = car.run(name, "accel", n, throttle=throttle, hold_kmh=target)
            n += 1
            if kmh >= target - 1:
                return n, True
        return n, False

    def stop(name, n):
        for _ in range(600):
            kmh = car.run(name, "stop", n, brake=1.0)
            n += 1
            if kmh < 1:
                break
        return n

    def move_check(name, n):
        top = 0.0
        for _ in range(120):
            top = max(top, car.run(name, "excite", n, throttle=0.5))
            n += 1
        return "ok" if top >= 3 else "no_motion"

    def prbs(name, n):
        n, ok = accel(name, n, 20)
        if not ok:
            return "no_speed"
        for bit in e01.prbs_bits(7, 1, 127):
            for _ in range(12):
                car.run(name, "excite", n, steer=0.3 if bit else -0.3, throttle=0.2, hold_kmh=20)
                n += 1
        stop(name, n)
        return "ok"

    def reversal(name, n):
        n, ok = accel(name, n, 20)
        if not ok:
            return "no_speed"
        for half in range(8):
            steer = 0.0 if half == 7 else (0.3 if half % 2 == 0 else -0.3)
            for _ in range(60):
                car.run(name, "excite", n, steer=steer, throttle=0.2, hold_kmh=20)
                n += 1
        stop(name, n)
        return "ok"

    def throttle_step(value):
        def body(name, n):
            for _ in range(120):
                car.run(name, "excite", n, throttle=value)
                n += 1
            stop(name, n)
            return "ok"
        return body

    def brake_step(value):
        def body(name, n):
            n, ok = accel(name, n, 40)
            if not ok:
                return "no_speed"
            for _ in range(600):
                kmh = car.run(name, "excite", n, brake=value)
                n += 1
                if kmh < 1:
                    break
            return "ok"
        return body

    def gear_observe(name, n):
        for _ in range(900):
            car.run(name, "excite", n, throttle=0.7)
            n += 1
        stop(name, n)
        return "ok"

    def gear_scripted(name, n):
        for gear in (2, 3, 4, 3, 2):
            for _ in range(60):
                car.run(name, "excite", n, throttle=0.4, gear=gear)
                n += 1
        stop(name, n)
        return "ok"

    result = phase("move_check", "", "0.5000", move_check)
    if result == "no_motion":
        result = phase("move_check_forced", "", "0.5000", move_check)
        if result == "no_motion":
            for name in ("steer_prbs", "steer_reversal"):
                phases.append({"phase_idx": len(phases), "phase": name, "result": "skipped",
                               "note": "no_motion_with_force_sim"})
    if result == "ok":
        phase("steer_prbs", "20.000", "0.3000", prbs)
        phase("steer_reversal", "20.000", "0.3000", reversal)
        for value in (0.3, 0.6, 1.0):
            phase("throttle_%.2f" % value, "", "%.4f" % value, throttle_step(value))
        for value in (0.3, 0.6, 1.0):
            phase("brake_%.2f" % value, "40.000", "%.4f" % value, brake_step(value))
        phase("gear_observe", "", "0.7000", gear_observe)
        phase("gear_scripted", "", "0.4000", gear_scripted)

    header = {"schema": "dc-probe-run/1", "probe": "e01", "engine.fixed_time_slice_s": "%.6f" % DT,
              "params.m_iAssumedSteerSign": "-1", "params.dt_s": "%.6f" % DT,
              "rerun19.compare": "e01_actuation.py e01 on both play dirs"}
    footer = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}
    probe_common.write_synthetic_play(play_dir, header, {"e01_steps": (STEP_COLUMNS, car.rows),
                                                         "e01_phases": (PHASE_COLUMNS, phases)}, footer)


def build_e02a(play_dir, starve_speed=None):
    """Cells at 15/30/50 km/h x 5 levels with kappa = G(v) * s * steer_sign (G(v) = 0.15 / (1 + (v/20)^2)).
    starve_speed: the cells of that speed except two never reach their speed (no steady window)."""
    car = Car()
    car.kappa_fn = lambda kmh, s: -0.15 / (1.0 + (kmh / 20.0) ** 2) * s
    grid = {15: (-0.5, -0.25, 0.1, 0.25, 0.5), 30: (-0.25, -0.12, 0.05, 0.12, 0.25),
            50: (-0.12, -0.06, 0.03, 0.06, 0.12)}
    cells = []
    for speed, levels in grid.items():
        for i, steer in enumerate(levels):
            idx = len(cells)
            name = "cell_%d" % idx
            car.reset()
            start = car.step + 1
            n = 0
            for _ in range(30):
                car.run(name, "settle", n, brake=1.0, handbrake=True)
                n += 1
            starved = speed == starve_speed and i >= 2
            target_reach = speed - 5 if starved else speed
            while car.speed * 3.6 < target_reach - 0.05:
                car.run(name, "accel", n, throttle=1.0, hold_kmh=speed)
                n += 1
            car.speed = target_reach / 3.6
            excite_start = car.step + 1
            for _ in range(300):
                car.speed = target_reach / 3.6
                car.run(name, "excite", n, steer=steer, throttle=0.2, hold_kmh=speed)
                n += 1
            car.speed = 0.0
            car.run(name, "stop", n, brake=1.0)
            cells.append({"cell_idx": idx, "target_kmh": "%.3f" % speed, "steer": "%.4f" % steer, "start_step": start,
                          "excite_start_step": excite_start, "end_step": car.step,
                          "result": "no_speed" if starved else "ok", "note": "lateral_start_m:0.00"})
    header = {"schema": "dc-probe-run/1", "probe": "e02a", "params.steer_sign_used": "-1",
              "params.steer_sign_source": "measured_e01"}
    footer = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}
    probe_common.write_synthetic_play(play_dir, header, {"e02a_steps": (STEP_COLUMNS, car.rows),
                                                         "e02a_cells": (CELL_COLUMNS, cells)}, footer)


def run_tool(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = e01.main(argv)
    return code, out.getvalue()


class ActuationTests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_actuation_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def test_e01_identifies_everything(self):
        play = os.path.join(self.base, "e01_ok")
        build_e01(play)
        code, text = run_tool(["e01", play])
        self.assertEqual(code, 0, text)
        self.assertIn("[PASS] moves", text)
        self.assertIn("[PASS] steer_delay_steps: 4 steps", text)
        self.assertIn("reversal onsets median 4", text)
        self.assertIn("[PASS] steer_sign: -1", text)
        self.assertIn("[PASS] throttle_delay_steps: median 6", text)
        self.assertIn("[PASS] brake_delay_steps: median 6", text)
        self.assertIn("automatic works", text)
        self.assertIn("not needed", text)
        self.assertIn("steer pre echo", text)
        self.assertIn("2:honoured", text)
        self.assertIn("[PASS] local_axes: z_forward 1", text)
        self.assertIn("x_right 1", text)
        self.assertIn("x right, z forward: yes", text)
        self.assertIn("[INFO] gear_indices:", text)

    def test_reader_columns_come_from_the_enforce_producer(self):
        columns = enforce_step_columns()
        self.assertEqual(len(columns), len(set(columns)), "duplicate step column")
        self.assertEqual(columns[:6], ["physics_step", "world_time_ms", "timeslice_s", "phase", "segment",
                                       "phase_step"])
        self.assertEqual(columns[-1], "in_area")
        for name in ("angvel_l_y", "acc_l_z", "acc_l_x", "vel_l_z", "fwd_x", "fwd_z"):
            self.assertIn(name, columns)
        missing = [name for name in e01.STEP_COLUMNS if name not in columns]
        self.assertEqual(missing, [], "e01_actuation.STEP_COLUMNS names the producer does not write")

    def test_a_flipped_local_x_axis_is_reported(self):
        play = os.path.join(self.base, "e01_flipped")
        build_e01(play)
        path = os.path.join(play, "e01_steps.csv")
        with open(path, "r", encoding="ascii", newline="") as handle:
            lines = handle.read().split("\n")
        index = lines[0].split(",").index("acc_l_x")
        flipped = [lines[0]]
        for line in lines[1:]:
            if line:
                parts = line.split(",")
                value = float(parts[index])
                parts[index] = "%.4f" % (-value) if value else parts[index]
                line = ",".join(parts)
            flipped.append(line)
        with open(path, "w", encoding="ascii", newline="") as handle:
            handle.write("\n".join(flipped))
        code, text = run_tool(["e01", play])
        self.assertIn("x_right -1", text)
        self.assertIn("x right, z forward: no", text)

    def test_e01_no_motion_fails(self):
        play = os.path.join(self.base, "e01_still")
        build_e01(play, moves=False)
        code, text = run_tool(["e01", play])
        self.assertEqual(code, 1, text)
        self.assertIn("[FAIL] moves", text)
        self.assertIn("does not help", text)

    def test_e01_without_footer_is_incomplete_at_best(self):
        play = os.path.join(self.base, "e01_nofooter")
        build_e01(play)
        os.remove(os.path.join(play, "run_footer.txt"))
        code, text = run_tool(["e01", play])
        self.assertEqual(code, 3, text)

    def test_e02a_linear_map_passes(self):
        play = os.path.join(self.base, "e02a_ok")
        build_e02a(play)
        code, text = run_tool(["e02a", play])
        self.assertEqual(code, 0, text)
        for speed in (15, 30, 50):
            self.assertIn("[PASS] fit_%dkmh" % speed, text)
        self.assertIn("R2 1.0000", text)

    def test_e02a_two_valid_cells_fail(self):
        play = os.path.join(self.base, "e02a_starved")
        build_e02a(play, starve_speed=30)
        code, text = run_tool(["e02a", play])
        self.assertEqual(code, 1, text)
        self.assertIn("[FAIL] fit_30kmh", text)
        self.assertIn("2 valid cells", text)
        self.assertIn("no steady window", text)

    def test_missing_play_is_input_error(self):
        code, _ = run_tool(["e01", os.path.join(self.base, "no_such_play")])
        self.assertEqual(code, 2)

    def test_prbs_matches_dc_probe_prbs(self):
        for order in range(5, 11):
            period = (1 << order) - 1
            bits = e01.prbs_bits(order, 1, 2 * period)
            self.assertEqual(bits[:period], bits[period:], order)
            self.assertEqual(sum(bits[:period]), (period + 1) // 2, order)
            for shift in range(1, period):
                if bits[:period] == bits[shift:shift + period]:
                    self.fail("order %d repeats after %d bits" % (order, shift))
        # first bits of the default order 7 / seed 1 (DC_ProbePrbs: output bit 0, feedback into bit 6)
        self.assertEqual(e01.prbs_bits(7, 1, 8), [1, 0, 0, 0, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            e01.prbs_bits(4, 1, 1)


if __name__ == "__main__":
    unittest.main()
