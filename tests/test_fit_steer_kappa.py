#!/usr/bin/env python3
"""Synthetic-data tests for tools/ident/fit_steer_kappa.py (DD-7-min) and the committed S105 card.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_fit_steer_kappa.py

No E-01 / E-02a play exists yet (the user's probe Plays are pending), so the plays are generated here from a vehicle
with a known steer -> curvature map, dead time, first-order lag and signs, in the probe kit's format: the step-CSV
header comes from the Enforce producer (tests/test_t0eng_actuation.py enforce_step_columns), the play files from
probe_common.write_synthetic_play into a fresh make_temp_dir() directory (RAM or E:, never /tmp), removed afterwards.

  - the fit recovers the map per speed, the offset, steer_sign and yaw_axis_sign, the dead time and lag (the
    delay + first-order structure wins on the held-out PRBS half), the echoing input, the throttle and brake gains,
    the automatic gearbox and the forward gear indices; steer_sign +1 too;
  - an identified card needs one pin and one build_key (--build-key fills a dd-launch/1 header), complete plays and a
    usable E-02a fit; Everon inputs are refused;
  - the committed card tools/ident/cards/1.8.0.13/S105.json is exactly the generator's PLACEHOLDER (status
    placeholder, no identification session, build_key null);
  - every card keeps the format llc_v1 reads: each scanned array key once, numbers without exponent, ASCII + LF, the
    keys DC_VehicleCard.Load requires; a Python mirror of DC_JsonScan.ReadNumbers reads the same arrays as json;
  - the Enforce reader (src/enfusion/datacapture/DC_LLCv1.c) asks for exactly the llc_v1 keys and arrays the
    generator writes;
  - DC_Sha256 (the card and source digests llc_v1 stamps): its round constants and initial hash are FIPS 180-4's,
    and an int32 emulation of its code (wrapping addition, arithmetic right shift, the engine semantics its SelfTest
    checks) equals hashlib.sha256 on the card, the route and other messages.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
T0ENG = os.path.join(REPO, "tools", "probes", "t0eng")
sys.path.insert(0, T0ENG)
import probe_common  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load_module("fit_steer_kappa", os.path.join(REPO, "tools", "ident", "fit_steer_kappa.py"))
actuation_tests = load_module("test_t0eng_actuation", os.path.join(REPO, "tests", "test_t0eng_actuation.py"))

DT = 1.0 / 60.0
STEP_COLUMNS = actuation_tests.enforce_step_columns()
PHASE_COLUMNS = actuation_tests.PHASE_COLUMNS
CELL_COLUMNS = actuation_tests.CELL_COLUMNS
CARD = os.path.join(REPO, "tools", "ident", "cards", "1.8.0.13", "S105.json")
ROUTE = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes", "arland_t0_loop.json")
LLC_SOURCE = os.path.join(REPO, "src", "enfusion", "datacapture", "DC_LLCv1.c")
LLC_KEYS = [name for name, _ in fit.LLC_DEFAULTS]
LAUNCH_ID = "20261023T101500Z-wb-00a1"


def gain(v):
    """True curvature per contract steer [1/m] at speed v [m/s]."""
    return 0.22 / (1.0 + (v / 10.0) ** 2)


class Truth:
    """Vehicle with a known actuator: contract steer c = steer_sign x engine steer, curvature kappa follows
    gain(v) x c(t - delay) + offset with a first-order lag tau; a longitudinal dead time; throttle 3 m/s2 per unit,
    brake 8 m/s2 per unit; an automatic gearbox that shifts up every 20 km/h (input gear 3, 4, 5, 6)."""

    def __init__(self, steer_sign=-1, delay=4, tau=0.1, offset=0.001, long_delay=6):
        self.steer_sign = steer_sign
        self.delay = delay
        self.alpha = 1.0 - math.exp(-DT / tau)
        self.offset = offset
        self.long_delay = long_delay
        self.step = 0
        self.rows = []
        self.reset()

    def reset(self):
        self.speed = 0.0
        self.heading = 0.0
        self.x, self.z = 1495.0, 2626.0
        self.kappa = 0.0
        self.steer_hist = []
        self.long_hist = []

    def run(self, phase, segment, n, steer=0.0, throttle=0.0, brake=0.0, gear=-1, handbrake=False, hold=-1.0,
            speed=None):
        self.step += 1
        self.steer_hist.append(steer)
        self.long_hist.append((throttle, brake, handbrake))
        delayed = self.steer_hist[-1 - self.delay] if len(self.steer_hist) > self.delay else 0.0
        thr, brk, hb = self.long_hist[-1 - self.long_delay] if len(self.long_hist) > self.long_delay else (0, 1, True)
        if hb or brk > 0:
            acc = -8.0 * max(brk, 1.0 if hb else 0.0) if self.speed > 0.05 else 0.0
        else:
            acc = 3.0 * thr
        if speed is not None:
            acc = 0.0
            self.speed = speed
        self.speed = max(0.0, self.speed + acc * DT)
        target = gain(self.speed) * self.steer_sign * delayed + self.offset
        self.kappa += self.alpha * (target - self.kappa)
        yaw = self.speed * self.kappa if self.speed > 0.3 else 0.0
        self.heading += math.degrees(yaw * DT)
        rad = math.radians(self.heading)
        self.x += self.speed * math.sin(rad) * DT
        self.z += self.speed * math.cos(rad) * DT
        kmh = self.speed * 3.6
        cur = 3 + min(3, int(kmh / 20.0)) if gear < 0 else gear
        self.rows.append({
            "physics_step": self.step, "world_time_ms": "%.3f" % (self.step * DT * 1000), "timeslice_s": "%.6f" % DT,
            "phase": phase, "segment": segment, "phase_step": n, "force_sim": 0, "cmd_steer": "%.4f" % steer,
            "cmd_throttle": "%.4f" % throttle, "cmd_brake": "%.4f" % brake, "cmd_clutch": "1.0000", "cmd_gear": gear,
            "cmd_handbrake": int(handbrake), "speed_hold_kmh": "%.3f" % hold,
            "input_pre_steer": "%.4f" % steer, "input_pre_throttle": "%.4f" % throttle,
            "input_pre_brake": "%.4f" % brake, "input_pre_clutch": "1.0000", "input_pre_gear": cur,
            "input_pre_handbrake": int(handbrake), "input_post_steer": "%.4f" % steer,
            "input_post_throttle": "%.4f" % throttle, "input_post_brake": "%.4f" % brake,
            "input_post_clutch": "1.0000", "input_post_gear": cur, "input_post_handbrake": int(handbrake),
            "cur_gear": cur, "fut_gear": cur, "rpm": "%.1f" % (900 + 40 * kmh), "engine_on": 1,
            "speed_kmh": "%.4f" % kmh, "vel_l_x": "0.000", "vel_l_y": "0.000", "vel_l_z": "%.4f" % self.speed,
            "angvel_l_x": "0.000000", "angvel_l_y": "%.7f" % yaw, "angvel_l_z": "0.000000",
            "acc_l_x": "%.4f" % (self.speed * yaw), "acc_l_y": "0.0000", "acc_l_z": "%.4f" % acc,
            "pos_x": "%.4f" % self.x, "pos_y": "40.000", "pos_z": "%.4f" % self.z, "fwd_x": "%.6f" % math.sin(rad),
            "fwd_y": "0.000000", "fwd_z": "%.6f" % math.cos(rad), "heading_deg": "%.6f" % self.heading,
            "ground_contact": 1, "wheels_contact": 4, "in_area": 1})
        return kmh


def header(probe, build_key="bk-test-1", pin="1.8.0.13", map_id="arland"):
    return {"schema": "dc-probe-run/1", "probe": probe, "launch.id": LAUNCH_ID, "launch.pin_id": pin,
            "launch.build_key": build_key, "engine.fixed_time_slice_s": "%.6f" % DT, "world.map_id": map_id,
            "ego.prefab": fit.PREFABS["S105"], "params.dt_s": "%.6f" % DT}


FOOTER = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}


def build_e01(play_dir, truth, **head):
    phases = []

    def phase(name, target, amplitude, body):
        truth.reset()
        start = truth.step + 1
        n = 0
        for _ in range(60):
            truth.run(name, "settle", n, brake=1.0, handbrake=True)
            n += 1
        result = body(name, n)
        phases.append({"phase_idx": len(phases), "phase": name, "start_step": start, "end_step": truth.step,
                       "start_world_ms": "", "end_world_ms": "", "target_kmh": target, "amplitude": amplitude,
                       "force_sim": 0, "result": result, "note": "phb_resets:0"})

    def cruise(name, n, kmh, steer):
        truth.run(name, "excite", n, steer=steer, throttle=0.2, hold=kmh, speed=kmh / 3.6)
        return n + 1

    def accel(name, n, kmh):
        while truth.speed * 3.6 < kmh:
            truth.run(name, "accel", n, throttle=1.0, hold=kmh)
            n += 1
        return n

    def stop(name, n):
        for _ in range(600):
            n += 1
            if truth.run(name, "stop", n, brake=1.0) < 1:
                break
        return n

    def move_check(name, n):
        for _ in range(120):
            truth.run(name, "excite", n, throttle=0.5)
            n += 1
        return "ok"

    def prbs(name, n):
        n = accel(name, n, 20)
        for bit in actuation_tests.e01.prbs_bits(7, 1, 127):
            for _ in range(12):
                n = cruise(name, n, 20.0, 0.3 if bit else -0.3)
        stop(name, n)
        return "ok"

    def reversal(name, n):
        n = accel(name, n, 20)
        for half in range(8):
            for _ in range(60):
                n = cruise(name, n, 20.0, 0.0 if half == 7 else (0.3 if half % 2 == 0 else -0.3))
        stop(name, n)
        return "ok"

    def throttle_step(value):
        def body(name, n):
            for _ in range(120):
                truth.run(name, "excite", n, throttle=value)
                n += 1
            stop(name, n)
            return "ok"
        return body

    def brake_step(value):
        def body(name, n):
            n = accel(name, n, 40)
            for _ in range(600):
                n += 1
                if truth.run(name, "excite", n, brake=value) < 1:
                    break
            return "ok"
        return body

    def gear_observe(name, n):
        for _ in range(900):
            truth.run(name, "excite", n, throttle=0.7)
            n += 1
        stop(name, n)
        return "ok"

    phase("move_check", "", "0.5000", move_check)
    phase("steer_prbs", "20.000", "0.3000", prbs)
    phase("steer_reversal", "20.000", "0.3000", reversal)
    for value in (0.3, 0.6, 1.0):
        phase("throttle_%.2f" % value, "", "%.4f" % value, throttle_step(value))
    for value in (0.3, 0.6, 1.0):
        phase("brake_%.2f" % value, "40.000", "%.4f" % value, brake_step(value))
    phase("gear_observe", "", "0.7000", gear_observe)
    probe_common.write_synthetic_play(play_dir, header("e01", **head), {
        "e01_steps": (STEP_COLUMNS, truth.rows), "e01_phases": (PHASE_COLUMNS, phases)}, FOOTER)


def build_e02a(play_dir, truth, starve_speed=None, **head):
    grid = {15: (-0.5, -0.25, 0.1, 0.25, 0.5), 30: (-0.25, -0.12, 0.05, 0.12, 0.25),
            50: (-0.12, -0.06, 0.03, 0.06, 0.12)}
    cells = []
    for kmh, levels in grid.items():
        for i, steer in enumerate(levels):
            idx = len(cells)
            name = "cell_%d" % idx
            truth.reset()
            start = truth.step + 1
            n = 0
            for _ in range(30):
                truth.run(name, "settle", n, brake=1.0, handbrake=True)
                n += 1
            starved = kmh == starve_speed and i >= 2
            hold = kmh - 5 if starved else kmh
            for _ in range(90):
                truth.run(name, "accel", n, throttle=0.3, hold=kmh, speed=hold / 3.6)
                n += 1
            excite = truth.step + 1
            for _ in range(480):
                truth.run(name, "excite", n, steer=steer, throttle=0.2, hold=kmh, speed=hold / 3.6)
                n += 1
            truth.run(name, "stop", n, brake=1.0, speed=0.0)
            cells.append({"cell_idx": idx, "target_kmh": "%.3f" % kmh, "steer": "%.4f" % steer, "start_step": start,
                          "excite_start_step": excite, "end_step": truth.step,
                          "result": "no_speed" if starved else "ok", "note": ""})
    probe_common.write_synthetic_play(play_dir, header("e02a", **head), {
        "e02a_steps": (STEP_COLUMNS, truth.rows), "e02a_cells": (CELL_COLUMNS, cells)}, FOOTER)


def run_fit(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fit.main(argv)
    return code, out.getvalue()


def scan_numbers(text, key):
    """Python mirror of DC_JsonScan.ReadNumbers (DC_LLCv1.c): the key once, then '[' numbers ',' ... ']'."""
    quoted = '"%s"' % key
    at = text.find(quoted)
    if at < 0 or text.find(quoted, at + 1) >= 0:
        return None
    rest = text[at + len(quoted):].lstrip()
    if not rest.startswith(":"):
        return None
    rest = rest[1:].lstrip()
    if not rest.startswith("["):
        return None
    body = rest[1:rest.index("]")]
    return [float(item) for item in body.replace("\n", " ").split(",") if item.strip()]


def i32(value):
    """An Enforce int: 32-bit two's complement (wrapping)."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


class EnforceSha256:
    """DC_Sha256 of DC_LLCv1.c, line by line, on emulated Enforce ints (constants parsed from the source)."""

    def __init__(self, source):
        self.k = [i32(int(v, 16)) for v in re.findall(r"0x([0-9a-f]{8})", source.split("s_aK = {", 1)[1].split("};", 1)[0])]
        self.h0 = [i32(int(v, 16)) for v in re.findall(r"0x([0-9a-f]{8})", source.split("s_aH0 = {", 1)[1].split("};", 1)[0])]

    @staticmethod
    def shr(value, count):
        return (value >> count) & (0x7FFFFFFF >> (count - 1))

    def rotr(self, value, count):
        return self.shr(value, count) | i32(value << (32 - count))

    def hash_text(self, text):
        data = [ord(ch) for ch in text]
        if any(code < 1 or code > 127 for code in data):
            return ""
        length = len(data)
        data.append(0x80)
        while len(data) % 64 != 56:
            data.append(0)
        for word in (self.shr(length, 29), i32(length << 3)):
            data += [(word >> 24) & 0xFF, (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF]
        state = list(self.h0)
        schedule = [0] * 64
        for block in range(0, len(data), 64):
            for t in range(16):
                at = block + 4 * t
                schedule[t] = i32((data[at] << 24) | (data[at + 1] << 16) | (data[at + 2] << 8) | data[at + 3])
            self.compress(state, schedule)
        return "".join("".join("0123456789abcdef"[(word >> shift) & 15] for shift in range(28, -1, -4))
                       for word in state)

    def compress(self, state, w):
        for t in range(16, 64):
            s0 = self.rotr(w[t - 15], 7) ^ self.rotr(w[t - 15], 18) ^ self.shr(w[t - 15], 3)
            s1 = self.rotr(w[t - 2], 17) ^ self.rotr(w[t - 2], 19) ^ self.shr(w[t - 2], 10)
            w[t] = i32(w[t - 16] + s0 + w[t - 7] + s1)
        a, b, c, d, e, f, g, h = state
        for r in range(64):
            big1 = self.rotr(e, 6) ^ self.rotr(e, 11) ^ self.rotr(e, 25)
            choose = (e & f) ^ (i32(~e) & g)
            temp1 = i32(h + big1 + choose + self.k[r] + w[r])
            big0 = self.rotr(a, 2) ^ self.rotr(a, 13) ^ self.rotr(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            temp2 = i32(big0 + majority)
            h, g, f, e, d, c, b, a = g, f, e, i32(d + temp1), c, b, a, i32(temp1 + temp2)
        for i, v in enumerate((a, b, c, d, e, f, g, h)):
            state[i] = i32(state[i] + v)


def primes(count):
    out = []
    n = 2
    while len(out) < count:
        if all(n % p for p in out):
            out.append(n)
        n += 1
    return out


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = probe_common.make_temp_dir("fit_steer_kappa_")
        cls.e01 = os.path.join(cls.base, "e01", LAUNCH_ID, "play_01")
        cls.e02a = os.path.join(cls.base, "e02a", LAUNCH_ID, "play_01")
        build_e01(cls.e01, Truth())
        build_e02a(cls.e02a, Truth())
        cls.out = os.path.join(cls.base, "cards", "1.8.0.13", "S105.json")
        cls.code, cls.text = run_fit(["--e01", cls.e01, "--e02a", cls.e02a, "--out", cls.out])
        with open(cls.out, "r", encoding="ascii") as f:
            cls.card_text = f.read()
        cls.card = json.loads(cls.card_text)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, True)

    def setUp(self):
        self.tmp = probe_common.make_temp_dir("fit_steer_kappa_case_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_identified_card_recovers_the_truth(self):
        self.assertEqual(self.code, 0, self.text)
        card = self.card
        self.assertEqual((card["status"], card["pin_id"], card["build_key"]), ("identified", "1.8.0.13", "bk-test-1"))
        self.assertEqual(card["identification_sessions"], ["e01/%s/play_01" % LAUNCH_ID, "e02a/%s/play_01" % LAUNCH_ID])
        self.assertEqual((card["signs"]["steer_sign"], card["signs"]["yaw_axis_sign"], card["signs"]["measured"]),
                         (-1, 1, True))
        ss = card["kappa_ss"]
        self.assertEqual(len(ss["speeds_mps"]), 3)
        for v, g, b in zip(ss["speeds_mps"], ss["kappa_per_steer"], ss["kappa_offset"]):
            self.assertAlmostEqual(g, gain(v), delta=0.02 * gain(v))
            self.assertAlmostEqual(b, 0.001, delta=2e-4)
        for f in ss["fits"]:
            self.assertGreater(f["r2"], 0.99)
        raw = card["actuator"]["raw_engine_path"]
        self.assertEqual((raw["input_echo"], raw["steer_rate_limit_per_s"]), (True, None))
        self.assertEqual(raw["selected_structure"], "delay_first_order")
        prbs = raw["bins"][0]["structures"]
        self.assertEqual(prbs["delay_first_order"]["dead_time_steps"], 4)
        self.assertAlmostEqual(prbs["delay_first_order"]["tau_s"], 0.1, delta=0.03)
        self.assertGreater(prbs["delay_first_order"]["r2_heldout"], 0.98)
        self.assertGreater(len(raw["bins"]), 10)            # the PRBS bin + one per E-02a cell (speed x amplitude)
        self.assertIsNone(card["actuator"]["full_llc_path"]["structure"])
        lon = card["longitudinal"]
        self.assertAlmostEqual(lon["throttle"]["per_mps2"], 1.0 / 3.0, delta=0.05)
        self.assertAlmostEqual(lon["brake"]["per_mps2"], 1.0 / 8.0, delta=0.02)
        self.assertEqual(lon["throttle"]["delay_steps"], 6)
        self.assertEqual(card["llc_v1"]["throttle_per_mps2"], lon["throttle"]["per_mps2"])
        self.assertEqual(card["llc_v1"]["steer_rate_per_s"], fit.LLC_DEFAULT_RATE)
        self.assertEqual(card["gearbox"]["mode"], "automatic")
        self.assertEqual(card["gear_indices_forward"], [3, 4, 5, 6])
        self.assertIsNone(card["gear_index_neutral"])
        self.assertIsNone(card["gear_index_reverse"])

    def test_positive_steer_sign(self):
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_02")
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_02")
        build_e01(e01, Truth(steer_sign=1))
        build_e02a(e02a, Truth(steer_sign=1))
        out = os.path.join(self.tmp, "card.json")
        code, text = run_fit(["--e01", e01, "--e02a", e02a, "--out", out])
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            card = json.load(f)
        self.assertEqual(card["signs"]["steer_sign"], 1)
        for v, g in zip(card["kappa_ss"]["speeds_mps"], card["kappa_ss"]["kappa_per_steer"]):
            self.assertAlmostEqual(g, gain(v), delta=0.02 * gain(v))

    def test_build_key_comes_from_the_headers_or_the_argument(self):
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_03")
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_03")
        build_e01(e01, Truth(), build_key="")
        build_e02a(e02a, Truth(), build_key="")
        code, text = run_fit(["--e01", e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("--build-key", text)
        code, text = run_fit(["--e01", e01, "--e02a", e02a, "--build-key", "bk-from-harvest"])
        self.assertEqual(code, 0, text)
        self.assertIn("build_key bk-from-harvest", text)
        code, text = run_fit(["--e01", self.e01, "--e02a", self.e02a, "--build-key", "another"])
        self.assertEqual(code, 2, text)

    def test_plays_of_two_pins_are_refused(self):
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_04")
        build_e02a(e02a, Truth(), pin="1.9.0.10-exp")
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("one pin", text)

    def test_unusable_e02a_is_refused(self):
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_05")
        build_e02a(e02a, Truth(), starve_speed=30)
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("not usable", text)

    def test_incomplete_play_is_refused(self):
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_06")
        build_e02a(e02a, Truth())
        os.remove(os.path.join(e02a, "run_footer.txt"))
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("complete", text)

    def test_everon_is_refused(self):
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_07")
        build_e02a(e02a, Truth(), map_id="everon")
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("sealed", text)
        sealed = os.path.join(self.tmp, "everon_plays", "play_01")
        code, text = run_fit(["--e01", sealed, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("sealed", text)

    def test_check_mode(self):
        code, text = run_fit(["--e01", self.e01, "--e02a", self.e02a, "--check", self.out])
        self.assertEqual(code, 0, text)
        other = os.path.join(self.tmp, "other.json")
        with open(other, "w", encoding="ascii") as f:
            f.write(self.card_text.replace('"identified"', '"placeholder"'))
        code, text = run_fit(["--e01", self.e01, "--e02a", self.e02a, "--check", other])
        self.assertEqual(code, 1, text)

    def test_committed_card_is_the_placeholder(self):
        code, text = run_fit(["--placeholder", "--pin-id", "1.8.0.13", "--check", CARD])
        self.assertEqual(code, 0, text)
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)
        self.assertEqual((card["status"], card["identification_sessions"], card["build_key"]),
                         ("placeholder", [], None))
        self.assertEqual((card["card_id"], card["pin_id"]), ("S105", "1.8.0.13"))
        self.assertIn("PLACEHOLDER", card["note"])
        self.assertIn("production", card["use"])
        self.assertEqual(card["gear_indices_forward"], [])

    def test_cards_keep_the_llc_v1_format(self):
        with open(CARD, "r", encoding="ascii", newline="") as f:
            committed = f.read()
        for text in (committed, self.card_text):
            self.assertNotIn("\r", text)
            self.assertTrue(text.endswith("}\n"))
            text.encode("ascii")
            card = json.loads(text)
            for key in fit.ENGINE_ARRAY_KEYS:
                self.assertEqual(text.count('"%s":' % key), 1, key)
            self.assertEqual(scan_numbers(text, "speeds_mps"), card["kappa_ss"]["speeds_mps"])
            self.assertEqual(scan_numbers(text, "kappa_per_steer"), card["kappa_ss"]["kappa_per_steer"])
            self.assertEqual(scan_numbers(text, "kappa_offset"), card["kappa_ss"]["kappa_offset"])
            self.assertEqual(scan_numbers(text, "upshift_mps"), card["gearbox"]["upshift_mps"])
            self.assertEqual(card["schema"], "drivedit.vehicle_card/1")
            self.assertIn(card["signs"]["steer_sign"], (-1, 1))
            self.assertIn(card["signs"]["yaw_axis_sign"], (-1, 1))
            self.assertIn(card["gearbox"]["mode"], ("automatic", "scripted"))
            self.assertEqual(sorted(card["llc_v1"]), sorted(LLC_KEYS))
            for value in card["llc_v1"].values():
                self.assertIsInstance(value, float)        # written with a decimal point
            speeds = card["kappa_ss"]["speeds_mps"]
            self.assertEqual(speeds, sorted(speeds))
            self.assertTrue(all(g > 0 for g in card["kappa_ss"]["kappa_per_steer"]))
            self.assertNotRegex(json.dumps(card), r'": -?[0-9.]+e')

    def test_enforce_reader_asks_for_the_written_keys(self):
        with open(LLC_SOURCE, "r", encoding="ascii") as f:
            source = f.read()
        body = source.split("protected bool ReadLlcNumbers", 1)[1].split("\n\t}\n", 1)[0]
        self.assertEqual(sorted(re.findall(r'ReadContextNumber\(context, "(\w+)"', body)), sorted(LLC_KEYS))
        arrays = re.findall(r'DC_JsonScan\.ReadNumbers\(text, "(\w+)"', source)
        self.assertEqual(sorted(arrays), sorted(fit.ENGINE_ARRAY_KEYS))
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)
        for key in re.findall(r'context\.ReadValue\("(\w+)"', source.split("class DC_VehicleCard", 1)[1].split(
                "class DC_LLCv1", 1)[0]):
            self.assertTrue(key in card or key in card["signs"] or key in card["gearbox"], key)

    def test_enforce_sha256(self):
        with open(LLC_SOURCE, "r", encoding="ascii") as f:
            enforce = EnforceSha256(f.read())
        p = primes(64)
        self.assertEqual(enforce.k, [i32(int(math.floor((n ** (1.0 / 3.0) % 1) * 2 ** 32))) for n in p])
        self.assertEqual(enforce.h0, [i32(int(math.floor((math.sqrt(n) % 1) * 2 ** 32))) for n in p[:8]])
        messages = ["", "abc", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "a" * 55, "a" * 56,
                    "a" * 63, "a" * 64, "a" * 119, "".join(chr(32 + (i * 7) % 95) for i in range(1000))]
        for path in (CARD, ROUTE):
            with open(path, "r", encoding="ascii", newline="") as f:
                messages.append(f.read())
        for text in messages:
            self.assertEqual(enforce.hash_text(text), hashlib.sha256(text.encode("ascii")).hexdigest(), len(text))
        self.assertEqual(enforce.hash_text("caf\u00e9"), "")

    def test_placeholder_needs_a_pin_and_takes_no_play(self):
        self.assertEqual(run_fit(["--placeholder"])[0], 2)
        self.assertEqual(run_fit(["--placeholder", "--pin-id", "1.8.0.13", "--e01", self.e01])[0], 2)

    def test_outputs_only_on_e_or_ram(self):
        code, text = run_fit(["--placeholder", "--pin-id", "1.8.0.13", "--out", "/tmp/S105.json"])
        self.assertEqual(code, 2, text)
        self.assertFalse(os.path.exists("/tmp/S105.json"))


if __name__ == "__main__":
    unittest.main()
