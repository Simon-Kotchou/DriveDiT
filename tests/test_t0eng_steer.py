#!/usr/bin/env python3
"""Synthetic-fixture tests for the G19-1 steering analyses (DD-14b): tools/probes/t0eng/e21_selfsteer.py (CC-07)
and tools/probes/t0eng/e23_steer_steps.py (CC-08), and the Enforce site arithmetic DC_SteerTerrainMath.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_steer.py

Every case writes synthetic e21 / e22 / e23 plays (run_header.txt, the probe CSVs, run_footer.txt) with
probe_common.write_synthetic_play into a fresh probe_common.make_temp_dir() directory (RAM or E:, never /tmp) and
removes it afterwards. The plays come from a small vehicle model (SteerCar): a steering actuator cmd -> input
(delay, first order, rate limit), the input -> wheel angle (0.5 rad per unit, an Ackermann pair, optional lag of the
bone channels, toe, a load or slope term), and a kinematic bicycle with a speed-dependent curvature lag. The step-CSV
header is built from the Enforce source (DC_ProbeSteerBase.OnProbeStart, DC_ProbeActuator.AddStepColumns,
DC_SteerChannels.AddColumns), so a reader / producer name mismatch fails here.

  - CC-07: a clean S1 / step-0 / load fixture passes every item; the 1.9.0.37-like readback at zero command fails
    (a); a 3e-4 1/m drift fails (b); wrong-sign wheel channels fail step 0 (and with it c and d); a lagged bone
    channel is skipped for an unlagged one; a missing e22 play, a missing cross site or an angle that follows the
    load fail; nothing passes without data;
  - CC-08: the selected structure is the true one on the cmd -> input path (delay, first order, rate limit), every
    bin reaches R2 >= 0.9; a planned bin without events and a wrong-sign kappa fail;
  - delta_eff inverts an Ackermann pair; the site search's contour heading puts the downhill on the requested side;
    DC_SteerTerrainMath (translated from the Enforce source) accepts a flat runway and a 6% cross-slope on the right
    side, refuses the wrong side, a bumpy strip and a short one, and agrees with a Python reference;
  - every header key and CSV name the analyses read is written by the Enforce producer.
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
import steer_common  # noqa: E402


def load_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e21 = load_tool("e21_selfsteer", os.path.join(T0ENG, "e21_selfsteer.py"))
e23 = load_tool("e23_steer_steps", os.path.join(T0ENG, "e23_steer_steps.py"))
_SPEC = importlib.util.spec_from_file_location("enforce_subset", os.path.join(REPO, "tests", "enforce_subset.py"))
enforce_subset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(enforce_subset)

SOURCES = os.path.join(REPO, "src", "enfusion", "datacapture")
BASE_SOURCE = os.path.join(SOURCES, "DC_ProbeSteerBase.c")
SEQ_SOURCE = os.path.join(SOURCES, "DC_ProbeSequences.c")
CHANNEL_SOURCE = os.path.join(SOURCES, "DC_SteerChannels.c")
SITE_SOURCE = os.path.join(SOURCES, "DC_SteerSites.c")
STEPS_SOURCE = os.path.join(SOURCES, "DC_ProbeSteerSteps.c")
SELF_SOURCE = os.path.join(SOURCES, "DC_ProbeSelfSteer.c")
LOAD_SOURCE = os.path.join(SOURCES, "DC_ProbeSteerLoad.c")
DT = 1.0 / 30.0
WHEELBASE = 2.42
TRACK = 1.30
DMAX = 0.5


def read(path):
    with open(path, "r", encoding="ascii") as handle:
        return handle.read()


def enforce_step_columns():
    """The <probe>_steps.csv header exactly as DC_ProbeSteerBase.OnProbeStart builds it"""
    base = read(BASE_SOURCE)
    seq = read(SEQ_SOURCE)
    chan = read(CHANNEL_SOURCE)
    start = re.search(r"array<string> stepColumns = \{([^}]*)\};", base)
    columns = re.findall(r'"([^"]+)"', start.group(1))
    act = re.search(r"static void AddStepColumns\(notnull array<string> columns\)\s*\{(.*?)\n\t\}", seq, re.S)
    for match in re.finditer(r'columns\.Insert\("([^"]+)"\)|AddVectorColumns\(columns, "([^"]+)"\)', act.group(1)):
        if match.group(1):
            columns.append(match.group(1))
        else:
            columns.extend(match.group(2) + s for s in ("_x", "_y", "_z"))
    tail = base[start.end():base.index("DC_SteerChannels.AddColumns(stepColumns);")]
    columns.extend(re.findall(r'stepColumns\.Insert\("([^"]+)"\)', tail))
    body = re.search(r"static void AddColumns\(notnull array<string> columns\)\s*\{(.*?)\n\t\}", chan, re.S)
    columns.extend(re.findall(r'columns\.Insert\("([^"]+)"\)', body.group(1)))
    return columns


STEP_COLUMNS = enforce_step_columns()
PHASE_COLUMNS = re.findall(r'"([^"]+)"', re.search(r'array<string> phaseColumns = \{([^}]*)\};',
                                                   read(BASE_SOURCE)).group(1))
EVENT_COLUMNS = re.findall(r'"([^"]+)"', re.search(r'array<string> columns = \{([^}]*)\};',
                                                   read(STEPS_SOURCE)).group(1))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Actuator:
    """cmd -> steering input: delay (steps), first order (tau s), rate limit (units / s)"""

    def __init__(self, delay=0, tau=0.0, rate=None):
        self.delay, self.tau, self.rate = delay, tau, rate
        self.hist = []
        self.y = 0.0

    def step(self, cmd, call=True):
        if not call:
            return self.y
        self.hist.append(cmd)
        target = self.hist[-1 - self.delay] if len(self.hist) > self.delay else 0.0
        a = math.exp(-DT / self.tau) if self.tau > 0 else 0.0
        step = (1.0 - a) * (target - self.y)
        if self.rate is not None:
            step = clamp(step, -self.rate * DT, self.rate * DT)
        self.y += step
        return self.y


def ackermann(delta):
    """(left, right) of a bicycle angle (positive right: the right wheel is the inner one)"""
    if abs(delta) < 1e-6:
        return delta, delta
    cot = 1.0 / math.tan(delta)
    half = TRACK / (2.0 * WHEELBASE)
    if delta > 0:
        return math.atan(1.0 / (cot + half)), math.atan(1.0 / (cot - half))
    return math.atan(1.0 / (cot + half)), math.atan(1.0 / (cot - half))


class SteerCar:
    """The fixture vehicle. Options: actuator (cmd -> input), wheel_sign (the channels' sign), bone_lag (steps of
    lag of the rot and bone channels; ent stays unlagged), toe (rad, left + / right -), kappa_bias (1/m added to the
    path), zero_noise (readback deviations at zero command, the 1.9.0.37 pattern), load_gain (rad of extra wheel
    angle per m/s2 of a_lat), slope_gain (rad of wheel angle per unit cross-slope), sigma (relaxation length m)."""

    def __init__(self, actuator=None, wheel_sign=1.0, bone_lag=0, toe=0.0, kappa_bias=0.0, zero_noise=False,
                 load_gain=0.0, slope_gain=0.0, sigma=0.6, kappa_sign=1.0, seed=3):
        self.act = actuator or Actuator()
        self.wheel_sign, self.bone_lag, self.toe = wheel_sign, bone_lag, toe
        self.kappa_bias, self.zero_noise, self.load_gain = kappa_bias, zero_noise, load_gain
        self.slope_gain, self.sigma, self.kappa_sign = slope_gain, sigma, kappa_sign
        self.rng = random.Random(seed)
        self.step_no = 100
        self.rows = []
        self.lag_hist = []
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, z, heading_deg):
        self.x, self.z, self.heading = x, z, heading_deg
        self.x0, self.z0, self.h0 = x, z, heading_deg
        self.v, self.kappa, self.a_lat = 0.0, 0.0, 0.0
        self.dist = 0.0
        self.act.hist = []
        self.act.y = 0.0
        self.lag_hist = []

    def run(self, phase, segment, run_idx, cmd, v_target, steer_call=1, seg_target=-1.0, seg_flag=0, event_idx=-1,
            site="flat", assist=2, mode="hold", slope=0.0, acc=None):
        self.step_no += 1
        if mode == "accel":
            self.v = min(v_target, self.v + 2.5 * DT)
        elif mode == "stop":
            self.v = max(0.0, self.v - 4.0 * DT)
        elif mode == "brake":
            self.v = max(0.0, self.v - 3.0 * DT)
        elif mode == "park":
            self.v = 0.0
        else:
            self.v = v_target
        steer_input = self.act.step(cmd, call=bool(steer_call))
        post = steer_input
        if self.zero_noise and abs(cmd) < 1e-9 and self.v > 2 and self.rng.random() < 0.16:
            post += self.rng.choice((-1, 1)) * self.rng.uniform(0.0002, 0.037)
        delta = steer_input * DMAX + self.load_gain * abs(self.a_lat) * (1 if cmd >= 0 else -1) + \
            self.slope_gain * slope
        left, right = ackermann(delta)
        left, right = left + self.toe, right - self.toe
        self.lag_hist.append((left, right))
        lagged = self.lag_hist[-1 - self.bone_lag] if len(self.lag_hist) > self.bone_lag else (0.0, 0.0)
        target_kappa = self.kappa_sign * math.tan(steer_common.delta_eff(left, right)) / WHEELBASE + self.kappa_bias
        tau = max(DT, self.sigma / max(self.v, 0.5))
        self.kappa += (target_kappa - self.kappa) * (1.0 - math.exp(-DT / tau))
        r = self.kappa * self.v
        self.a_lat = self.v * r
        self.heading += math.degrees(r * DT)
        rad = math.radians(self.heading)
        self.x += self.v * math.sin(rad) * DT
        self.z += self.v * math.cos(rad) * DT
        self.dist += self.v * DT
        h0 = math.radians(self.h0)
        along = (self.x - self.x0) * math.sin(h0) + (self.z - self.z0) * math.cos(h0)
        lateral = (self.x - self.x0) * math.cos(h0) - (self.z - self.z0) * math.sin(h0)
        s = self.wheel_sign
        acc_z = acc if acc is not None else (-3.0 if mode == "brake" else 0.0)
        row = {c: "" for c in STEP_COLUMNS}
        row.update({
            "physics_step": self.step_no, "world_time_ms": "%.3f" % (self.step_no * DT * 1000.0),
            "timeslice_s": "%.6f" % DT, "phase": phase, "segment": segment, "phase_step": 0, "run_idx": run_idx,
            "force_sim": 0, "cmd_steer": "%.4f" % cmd, "cmd_throttle": "0.3000", "cmd_brake": "0.0000",
            "cmd_clutch": "1.0000", "cmd_gear": 2, "cmd_handbrake": 0, "speed_hold_kmh": "%.3f" % (v_target * 3.6),
            "input_pre_steer": "%.4f" % cmd, "input_post_steer": "%.4f" % post, "engine_on": 1,
            "speed_kmh": "%.3f" % (self.v * 3.6), "vel_l_x": "0.000", "vel_l_y": "0.000", "vel_l_z": "%.3f" % self.v,
            "angvel_l_x": "0.000000", "angvel_l_y": "%.6f" % r, "angvel_l_z": "0.000000",
            "acc_l_x": "%.4f" % self.a_lat, "acc_l_y": "0.0000", "acc_l_z": "%.4f" % acc_z,
            "pos_x": "%.3f" % self.x, "pos_y": "39.000", "pos_z": "%.3f" % self.z,
            "fwd_x": "%.6f" % math.sin(rad), "fwd_y": "0.000000", "fwd_z": "%.6f" % math.cos(rad),
            "heading_deg": "%.5f" % self.heading, "ground_contact": 1, "wheels_contact": 4,
            "steer_call": int(steer_call), "assist_mode": assist, "site": site, "along_m": "%.3f" % along,
            "lateral_m": "%.3f" % lateral, "in_area": 1, "seg_target": "%.4f" % seg_target, "seg_flag": seg_flag,
            "event_idx": event_idx,
            "wa_rot_l": "%.6f" % (s * lagged[0]), "wa_rot_r": "%.6f" % (s * lagged[1]),
            "wa_bone_l": "%.6f" % (s * lagged[0]), "wa_bone_r": "%.6f" % (s * lagged[1]),
            "wa_ent_l": "%.6f" % (s * left), "wa_ent_r": "%.6f" % (s * right),
            "sig_steer": "%.6f" % steer_input, "wa_rot_l_pf": "%.6f" % (s * left), "wa_rot_r_pf": "%.6f" % (s * right),
            "slip_lat_l": "0.0", "slip_lat_r": "0.0", "vdir_l": "%.6f" % left, "vdir_r": "%.6f" % right})
        self.rows.append(row)


def drive_start(car, phase, run, v_target, site="flat", assist=2):
    car.reset(1000.0, 2000.0, 75.0)
    for _ in range(2):
        car.run(phase, "reset", run, 0.0, 0.0, mode="park", site=site, assist=assist)
    for _ in range(5):
        car.run(phase, "settle", run, 0.0, 0.0, mode="park", site=site, assist=assist)
    while car.v < v_target - 0.2:
        car.run(phase, "accel", run, 0.0, v_target, mode="accel", seg_target=v_target * 3.6, site=site, assist=assist)


def drive_stop(car, phase, run, site="flat", assist=2):
    while car.v > 0.2:
        car.run(phase, "stop", run, 0.0, 0.0, mode="stop", site=site, assist=assist)


def common_header(probe):
    return {"schema": "dc-probe-run/1", "probe": probe, "params.dt_s": "%.6f" % DT, "steer.wheelbase_m": WHEELBASE,
            "steer.wheel_index_l": 0, "steer.wheel_index_r": 1, "steer.wheel_count": 4, "steer.sign_used": 1,
            "assist.original": 2, "assist.original_name": "FULL", "assist.sets": 2, "assist.restores": 2,
            "assist.at_stop": 2, "vehicle.noise_steer_sensitivity": "0.0000", "site.flat.found": 1,
            "site.flat.source": "explicit", "site.flat.heading_deg": "75.110", "site.flat.grade": "0.00001",
            "site.flat.cross_mean": "0.00010", "site.flat.obstacle_hits": 0}


FOOTER = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}


def build_e21(play_dir, car=None, cross_found=True, cross_min=0.06, flat_m=200.0):
    car = car or SteerCar()
    phases = []

    def zero_phase(name, kmh, held, nocall, site="flat", assist=2, slope=0.0):
        v = kmh / 3.6
        drive_start(car, name, 1, v, site, assist)
        for _ in range(60):
            car.run(name, "hold", 1, 0.0, v, site=site, assist=assist, slope=slope, seg_target=kmh)
        for segment, length, call in (("measure", held, 1), ("nocall", nocall, 0)):
            start = car.dist
            while car.dist - start < length:
                car.run(name, segment, 1, 0.0, v, steer_call=call, seg_flag=1, site=site, assist=assist,
                        slope=slope, seg_target=kmh)
        drive_stop(car, name, 1, site, assist)
        phases.append([len(phases), name, site, site, "", "", "", "", kmh, 0, assist, 1, "ok", ""])

    for kmh in (20, 40):
        zero_phase("flat_%d" % kmh, kmh, flat_m, 200.0)
    for kmh in (20, 40):
        zero_phase("flat_%d_assist0" % kmh, kmh, 100.0, 100.0, assist=0)
    if cross_found:
        for kmh in (20, 40):
            zero_phase("cross_%d" % kmh, kmh, 100.0, 100.0, site="cross", slope=0.06)
    header = common_header("e21")
    header.update({"params.m_sFlatSpeedsKmh": "20,40", "params.m_sCrossSpeedsKmh": "20,40",
                   "site.cross.found": int(cross_found), "site.cross.cross_min_signed": "%.5f" % cross_min,
                   "site.cross.cross_max_signed": "0.07000", "site.cross.cross_mean": "0.06500",
                   "site.cross.start": "2000.0 0.0 1500.0", "site.cross.heading_deg": "33.0",
                   "site.cross.grade": "0.004", "site.cross.fallback": 0})
    probe_common.write_synthetic_play(play_dir, header, {"e21_steps": (STEP_COLUMNS, car.rows),
                                                         "e21_phases": (PHASE_COLUMNS, phases)}, FOOTER)
    return play_dir


def build_e22(play_dir, car=None, steer_load=(0.3, -0.3), targets=(2.0, 2.75, 3.5)):
    car = car or SteerCar()
    phases = []
    for kmh, levels in ((12, (0.1, -0.1, 0.2, -0.2, 0.3, -0.3)), (18, (0.05, -0.05, 0.1, -0.1, 0.15, -0.15))):
        name = "stair_%d" % kmh
        v = kmh / 3.6
        drive_start(car, name, 1, v)
        for level, steer in enumerate(levels):
            for k in range(90):
                car.run(name, "stair", 1, steer, v, seg_flag=int(k >= 30), event_idx=level, seg_target=kmh)
        drive_stop(car, name, 1)
        phases.append([len(phases), name, "stair", "flat", "", "", "", "", kmh, 0, -1, 1, "ok", ""])
    for steer in steer_load:
        name = "load_%+.2f" % steer
        v = 10.0 / 3.6
        drive_start(car, name, 1, v)
        for k in range(180):
            car.run(name, "ref", 1, steer, v, seg_flag=int(k >= 90), seg_target=10.0)
        kappa = abs(math.tan(steer * DMAX) / WHEELBASE)
        for level, target in enumerate(targets):
            vt = math.sqrt(target / kappa)
            for k in range(150):
                car.run(name, "alat", 1, steer, vt, seg_flag=int(k >= 40), event_idx=level, seg_target=target)
        while car.v > 5.0 / 3.6:
            car.run(name, "brake", 1, steer, car.v, mode="brake", seg_flag=1, seg_target=3.0)
        drive_stop(car, name, 1)
        phases.append([len(phases), name, "load", "flat", "", "", "", "", 10, steer, -1, 1, "ok", ""])
    header = common_header("e22")
    header.update({"params.m_sALatTargets": ",".join("%g" % t for t in targets), "params.m_fBrakeDecelMs2": "3.000"})
    probe_common.write_synthetic_play(play_dir, header, {"e22_steps": (STEP_COLUMNS, car.rows),
                                                         "e22_phases": (PHASE_COLUMNS, phases)}, FOOTER)
    return play_dir


def build_e23(play_dir, car=None, grid="10:0.5x6,1.0x4;30:0.5x6,1.0x4;60:0.5x6", drop_speed=None, seed=5):
    car = car or SteerCar()
    rng = random.Random(seed)
    events = []
    phases = []
    prbs = e23_prbs_bits(5, 11, 31)
    for group in grid.split(";"):
        speed_text, specs = group.split(":")
        kmh = float(speed_text)
        amps = []
        for spec in specs.split(","):
            amp, count = spec.split("x")
            amps.append([float(amp), int(count)])
        order = []
        while any(c > 0 for _, c in amps):
            for item in amps:
                if item[1] > 0:
                    order += [item[0], -item[0]]
                    item[1] -= 1
        if drop_speed is not None and kmh == drop_speed:
            continue
        name = "steps_%d" % kmh
        v = kmh / 3.6
        drive_start(car, name, 1, v)
        for _ in range(30):
            car.run(name, "hold", 1, 0.0, v, seg_target=kmh)
        open_event = -1
        pending = None
        for idx, amp in enumerate(order):
            gap = 18 + rng.randint(0, 9)
            for k in range(gap):
                car.run(name, "train", 1, 0.0, v, event_idx=open_event, seg_target=kmh)
                if k == 0 and pending is not None:
                    pending["release_step"] = car.step_no
            if pending is not None:
                pending["end_step"] = car.step_no
                events.append(pending)
            open_event = idx
            for k in range(12):
                car.run(name, "train", 1, amp, v, seg_flag=1, event_idx=idx, seg_target=kmh)
                if k == 0:
                    pending = {"event_idx": idx, "phase": name, "target_kmh": "%.3f" % kmh, "amplitude": "%.4f" % amp,
                               "run_idx": 1, "jitter_steps": gap - 18, "onset_step": car.step_no,
                               "release_step": "", "end_step": "", "onset_kmh": "%.3f" % kmh,
                               "onset_along_m": "0.0", "result": "ok"}
        for k in range(18):
            car.run(name, "train", 1, 0.0, v, event_idx=open_event, seg_target=kmh)
            if k == 0:
                pending["release_step"] = car.step_no
        pending["end_step"] = car.step_no
        events.append(pending)
        amp = {10.0: 0.5, 30.0: 0.3, 60.0: 0.2}.get(kmh, 0.0)
        for bit in prbs:
            for _ in range(4):
                car.run(name, "prbs", 1, amp if bit else -amp, v, seg_flag=1, seg_target=kmh)
        drive_stop(car, name, 1)
        phases.append([len(phases), name, "steps", "flat", "", "", "", "", kmh, 0, -1, 1, "ok", ""])
    header = common_header("e23")
    header.update({"params.m_sStepGrid": grid})
    probe_common.write_synthetic_play(play_dir, header, {"e23_steps": (STEP_COLUMNS, car.rows),
                                                         "e23_phases": (PHASE_COLUMNS, phases),
                                                         "e23_events": (EVENT_COLUMNS, events)}, FOOTER)
    return play_dir


def e23_prbs_bits(order, seed, count):
    taps = {5: (5, 3)}[order]
    state = seed & ((1 << order) - 1)
    out = []
    for _ in range(count):
        out.append(state & 1)
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (order - tap)) & 1
        state = (state >> 1) | (feedback << (order - 1))
    return out


def run_main(module, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = probe_common.run_cli(module.main, argv)
    return code, out.getvalue()


def item(text, name):
    """The [TAG] of check name in a report"""
    m = re.search(r"\[(PASS|FAIL|INCOMPLETE|INFO)\] %s:" % re.escape(name), text)
    return m.group(1) if m else None


class SteerFixtureTests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_steer_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def plays(self, e21_car=None, e22_car=None, **kw):
        a = build_e21(os.path.join(self.base, "e21", "play_01"), e21_car, **kw)
        b = build_e22(os.path.join(self.base, "e22", "play_01"), e22_car)
        return a, b

    # --- CC-07 -------------------------------------------------------------------------------------------------
    def test_clean_fixture_passes_cc07(self):
        a, b = self.plays()
        code, text = run_main(e21, [a, "--e22", b])
        for name in ("step0_channel", "a_readback", "b_flat", "c_cross", "d_load"):
            self.assertEqual(item(text, name), "PASS", "%s in\n%s" % (name, text))
        self.assertEqual(code, 0, text)
        self.assertIn("selected rot", text)
        self.assertIn("Mode A stays the production default", text)

    def test_zero_command_readback_like_1_9_fails_a(self):
        a, b = self.plays(e21_car=SteerCar(zero_noise=True))
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "a_readback"), "FAIL", text)
        self.assertEqual(code, 1)
        self.assertIn("Mode B (seated driver, CC-14)", text)

    def test_straight_line_drift_fails_b(self):
        a, b = self.plays(e21_car=SteerCar(kappa_bias=3e-4))
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "b_flat"), "FAIL", text)
        self.assertEqual(code, 1)

    def test_short_flat_segment_is_no_data(self):
        a, b = self.plays(flat_m=120.0)
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "b_flat"), "FAIL", text)
        self.assertIn("no data", text)

    def test_wrong_sign_wheel_channels_fail_step0(self):
        a, b = self.plays(e21_car=SteerCar(wheel_sign=-1.0), e22_car=SteerCar(wheel_sign=-1.0))
        code, text = run_main(e21, ["--signal-scale-deg", "1", a, "--e22", b])
        self.assertEqual(item(text, "step0_channel"), "FAIL", text)
        self.assertEqual(item(text, "c_cross"), "FAIL", text)
        self.assertEqual(item(text, "d_load"), "FAIL", text)
        self.assertEqual(code, 1)

    def test_lagged_bones_select_the_entity_channel(self):
        a, b = self.plays(e22_car=SteerCar(bone_lag=3))
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "step0_channel"), "PASS", text)
        self.assertIn("selected ent", text)
        self.assertRegex(text, r"rot: median error [^;]*lag 3 steps[^;]*-> no")

    def test_without_e22_step0_and_load_fail(self):
        a = build_e21(os.path.join(self.base, "e21", "play_01"))
        code, text = run_main(e21, [a])
        self.assertEqual(item(text, "step0_channel"), "FAIL")
        self.assertEqual(item(text, "d_load"), "FAIL")
        self.assertEqual(item(text, "c_cross"), "FAIL")
        self.assertEqual(code, 1)

    def test_missing_cross_site_fails_c(self):
        a, b = self.plays(cross_found=False)
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "c_cross"), "FAIL", text)
        self.assertIn("no cross-slope site", text)

    def test_shallow_cross_site_fails_c(self):
        a, b = self.plays(cross_min=0.03)
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "c_cross"), "FAIL", text)

    def test_slope_self_steer_fails_c(self):
        a, b = self.plays(e21_car=SteerCar(slope_gain=0.2))
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "c_cross"), "FAIL", text)
        self.assertEqual(item(text, "b_flat"), "PASS", text)

    def test_angle_that_follows_the_load_fails_d(self):
        e22_car = SteerCar(load_gain=math.radians(0.5))
        a, b = self.plays(e22_car=e22_car)
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(item(text, "d_load"), "FAIL", text)

    def test_empty_steps_are_no_data(self):
        a = probe_common.write_synthetic_play(os.path.join(self.base, "e21_empty", "play_01"), common_header("e21"),
                                              {"e21_steps": (STEP_COLUMNS, []), "e21_phases": (PHASE_COLUMNS, [])},
                                              FOOTER)
        b = probe_common.write_synthetic_play(os.path.join(self.base, "e22_empty", "play_01"), common_header("e22"),
                                              {"e22_steps": (STEP_COLUMNS, []), "e22_phases": (PHASE_COLUMNS, [])},
                                              FOOTER)
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(code, 1, text)
        self.assertNotIn("[PASS] step0", text)
        self.assertNotIn("[PASS] a_readback", text)
        self.assertNotIn("[PASS] b_flat", text)

    def test_missing_column_is_input_error(self):
        columns = [c for c in STEP_COLUMNS if c != "wa_rot_l"]
        a = probe_common.write_synthetic_play(os.path.join(self.base, "e21_col", "play_01"), common_header("e21"),
                                              {"e21_steps": (columns, [])}, FOOTER)
        code, text = run_main(e21, [a])
        self.assertEqual(code, probe_common.EXIT_INPUT_ERROR, text)

    # --- CC-08 -------------------------------------------------------------------------------------------------
    def selected_on_input(self, text, kmh, amp):
        m = re.search(r"\[(PASS|FAIL)\] input_%gkmh_%g: (M\d)" % (kmh, amp), text)
        self.assertIsNotNone(m, text)
        return m.group(1), m.group(2)

    def test_steps_select_the_true_structure(self):
        cases = (("M1", Actuator(delay=2)), ("M2", Actuator(tau=0.12)), ("M3", Actuator(rate=2.0)))
        for expected, actuator in cases:
            with self.subTest(expected=expected):
                play = build_e23(os.path.join(self.base, "e23_" + expected, "play_01"), SteerCar(actuator=actuator),
                                 grid="10:0.5x6,1.0x4;30:0.5x6")
                code, text = run_main(e23, [play])
                for kmh, amp in ((10, 0.5), (10, 1.0), (30, 0.5)):
                    tag, structure = self.selected_on_input(text, kmh, amp)
                    self.assertEqual(structure, expected, text)
                    self.assertEqual(tag, "PASS", text)
                self.assertEqual(item(text, "steer_sign"), "PASS", text)

    def test_full_grid_passes_with_validated_channel(self):
        play = build_e23(os.path.join(self.base, "e23", "play_01"), SteerCar(actuator=Actuator(tau=0.05)))
        e22_play = build_e22(os.path.join(self.base, "e22", "play_01"))
        code, text = run_main(e23, [play, "--e22", e22_play])
        self.assertEqual(code, 0, text)
        self.assertIn("channel_validated: yes", text)
        self.assertEqual(item(text, "wheel_60kmh_0.5"), "PASS", text)
        self.assertEqual(item(text, "kappa_30kmh_1"), "PASS", text)
        self.assertIn("prbs_held_out", text)

    def test_missing_speed_bin_fails(self):
        play = build_e23(os.path.join(self.base, "e23", "play_01"), drop_speed=60.0)
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "input_60kmh_0.5"), "FAIL", text)
        self.assertIn("no data", text)
        self.assertEqual(code, 1)

    def test_wrong_kappa_sign_fails_steer_sign(self):
        play = build_e23(os.path.join(self.base, "e23", "play_01"), SteerCar(kappa_sign=-1.0),
                         grid="10:0.5x6;30:0.5x6")
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "steer_sign"), "FAIL", text)
        self.assertEqual(code, 1)

    def test_no_events_is_no_data(self):
        play = build_e23(os.path.join(self.base, "e23", "play_01"), grid="10:0.5x6")
        events = os.path.join(play, "e23_events.csv")
        with open(events, "w", encoding="ascii") as handle:
            handle.write(",".join(EVENT_COLUMNS) + "\n")
        code, text = run_main(e23, [play])
        self.assertEqual(code, 1, text)
        self.assertNotIn("[PASS] input_", text)

    # --- math and contracts ------------------------------------------------------------------------------------
    def test_delta_eff_inverts_ackermann(self):
        for delta in (-0.4, -0.1, -0.01, 0.0, 0.003, 0.05, 0.2, 0.45):
            left, right = ackermann(delta)
            self.assertAlmostEqual(steer_common.delta_eff(left, right), delta, delta=max(2e-4, abs(delta) * 0.003))

    def test_contour_heading_puts_the_downhill_on_the_side(self):
        # DC_SteerSiteSearch.GenerateStrips: heading = atan2(side gz/g, -side gx/g); right(h) = (cos h, -sin h)
        for gx, gz in ((0.06, 0.0), (0.0, 0.06), (-0.04, 0.05), (0.03, -0.07)):
            g = math.hypot(gx, gz)
            for side in (1, -1):
                h = math.atan2(side * gz / g, -side * gx / g)
                right = (math.cos(h), -math.sin(h))
                # moving to the right goes downhill when side is +1: the gradient . right < 0
                self.assertAlmostEqual(side * (gx * right[0] + gz * right[1]), -g, places=9)
                forward = (math.sin(h), math.cos(h))
                self.assertAlmostEqual(gx * forward[0] + gz * forward[1], 0.0, places=9)
        source = read(SITE_SOURCE)
        self.assertIn("Math.Atan2(side * gz / g, -side * gx / g)", source)

    def test_header_keys_and_files_come_from_the_enforce_producer(self):
        sources = "\n".join(read(p) for p in (BASE_SOURCE, SITE_SOURCE, CHANNEL_SOURCE, SELF_SOURCE, LOAD_SOURCE,
                                              STEPS_SOURCE))
        for key in ("steer.wheelbase_m", "steer.wheel_index_l", "steer.wheel_count", "assist.original",
                    "assist.original_name", "assist.sets", "assist.restores", "assist.at_stop", "steer.sign_used",
                    "site.cross.fallback", "vehicle.noise_steer_sensitivity"):
            tail = key.split(".")[-1]
            self.assertIn(tail + "=", sources, key)
        for name in (".found=", ".cross_min_signed=", ".cross_max_signed=", ".cross_mean=", ".heading_deg=",
                     ".grade=", ".obstacle_hits=", ".start=", "first_terrain_reject=", "first_obstacle="):
            self.assertIn(name, sources, name)
        for param in ("m_sFlatSpeedsKmh", "m_sCrossSpeedsKmh", "m_sALatTargets", "m_fBrakeDecelMs2", "m_sStepGrid"):
            self.assertIn('"%s="' % param, sources, param)
        for csv_name in ('probeId + "_steps"', 'probeId + "_phases"', 'PROBE_E23 + "_events"'):
            self.assertIn(csv_name, sources)
        for column in steer_common.STEP_COLUMNS:
            self.assertIn(column, STEP_COLUMNS, column)
        for column in e23.EVENT_COLUMNS:
            self.assertIn(column, EVENT_COLUMNS, column)
        for segment in ("hold", "measure", "nocall", "stair", "ref", "alat", "brake", "train", "prbs"):
            self.assertIn('return "%s";' % segment, sources, segment)

    def test_manifest_and_layer_carry_the_probes(self):
        manifest = read(os.path.join(REPO, "addons", "DataCapture", "deploy_manifest.txt"))
        for name in ("DC_SteerChannels.c", "DC_SteerSites.c", "DC_ProbeSteerBase.c", "DC_ProbeSelfSteer.c",
                     "DC_ProbeSteerLoad.c", "DC_ProbeSteerSteps.c"):
            self.assertIn("script src/enfusion/datacapture/" + name, manifest)
        layer = read(os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Probes_Arland_Layers",
                                  "default.layer"))
        self.assertIn('m_sPlan "e10,e04,e20,e19,e08a,e01,e02a,e05,e06a,e21,e22,e23"', layer)
        for cls in ("DC_ProbeSelfSteer", "DC_ProbeSteerLoad", "DC_ProbeSteerSteps"):
            self.assertRegex(layer, r'  %s "\{DC4A50B30000000[789]\}" \{' % cls)

    def test_no_1_9_only_member_in_the_new_scripts(self):
        # the overlay convention: GetHandbrake, EngineGetRPMRedline, Update*Configuration and
        # VehicleWheelEffectComponent stay out of every script that reaches a 1.8.0.13 target
        for path in (BASE_SOURCE, SITE_SOURCE, CHANNEL_SOURCE, SELF_SOURCE, LOAD_SOURCE, STEPS_SOURCE):
            text = read(path)
            code = "\n".join(line.split("//", 1)[0] for line in text.split("\n"))
            for member in ("GetHandbrake(", "EngineGetRPMRedline(", "Configuration(", "VehicleWheelEffectComponent",
                           "TraceDisplacement("):
                self.assertNotIn(member, code, "%s in %s" % (member, path))


# ------------------------------------------------------------------------------------------------------------------
# DC_SteerTerrainMath, translated from the Enforce source


class CriteriaStub:
    def __init__(self, kind, **values):
        self.m_sKind = kind
        self.m_fStepM = enforce_subset.f32(5.0)
        self.m_fGradeWindowM = enforce_subset.f32(20.0)
        defaults = {"m_fGradeMax": 0.005, "m_fGradeWindowMax": 0.01, "m_fGradeWindowMaxAll": 0.02,
                    "m_fCrossMin": 0.0, "m_fCrossMax": 0.01, "m_fCrossAbsMaxAll": 0.02, "m_fCrossTarget": 0.0,
                    "m_fRmsMax": 0.1, "m_fMinSurfaceY": 3.0}
        defaults.update(values)
        for key, value in defaults.items():
            setattr(self, key, enforce_subset.f32(value))

    def IsCross(self):
        return self.m_sKind == "cross"


class SiteStub:
    def __init__(self, length, half_width, zone_from, zone_to, side=1, min_y=39.0):
        self.m_fLengthM = enforce_subset.f32(length)
        self.m_fHalfWidthM = enforce_subset.f32(half_width)
        self.m_fZoneFromM = enforce_subset.f32(zone_from)
        self.m_fZoneToM = enforce_subset.f32(zone_to)
        self.m_iSide = side
        self.m_fMinY = enforce_subset.f32(min_y)
        for name in ("m_fGrade", "m_fGradeWindowMax", "m_fGradeWindowMaxAll", "m_fCrossMean", "m_fCrossMin",
                     "m_fCrossMax", "m_fCrossAbsMaxAll", "m_fRmsM", "m_fScore"):
            setattr(self, name, enforce_subset.f32(0.0))
        self.m_bTerrainOk = False
        self.m_sTerrainReason = ""


def translated_terrain_math():
    source = read(SITE_SOURCE)
    fields = {"DC_SteerSite": enforce_subset.class_field_types(source, "DC_SteerSite"),
              "DC_SteerSiteCriteria": enforce_subset.class_field_types(source, "DC_SteerSiteCriteria")}
    classes = enforce_subset.translate(source, ["DC_SteerTerrainMath"],
                                       externals={"DC_SteerSite": SiteStub, "DC_SteerSiteCriteria": CriteriaStub},
                                       external_fields=fields)
    return classes["DC_SteerTerrainMath"]()


def strip_samples(length, w, height_fn, step=5.0):
    n = int(length // step) + 1
    along = [i * step for i in range(n)]
    left = [height_fn(a, -w) for a in along]
    centre = [height_fn(a, 0.0) for a in along]
    right = [height_fn(a, w) for a in along]

    def arr(values):
        return enforce_subset.EArray(values, "float")

    return arr(along), arr(left), arr(centre), arr(right)


class TerrainMathTests(unittest.TestCase):
    def setUp(self):
        self.math = translated_terrain_math()

    def apply(self, site, criteria, height_fn):
        along, left, centre, right = strip_samples(float(site.m_fLengthM), float(site.m_fHalfWidthM), height_fn)
        site.m_fMinY = enforce_subset.f32(min(min(left), min(centre), min(right)))
        self.math.Apply(site, criteria, along, left, centre, right)
        return site

    def test_flat_runway_passes(self):
        site = self.apply(SiteStub(1000, 22, 0, 1000), CriteriaStub("flat"), lambda a, lat: 39.0 + 0.00002 * a)
        self.assertTrue(site.m_bTerrainOk, site.m_sTerrainReason)
        self.assertAlmostEqual(float(site.m_fGrade), 0.00002, places=6)
        self.assertLess(float(site.m_fRmsM), 0.01)

    def test_cross_slope_on_the_right_passes_and_the_left_does_not(self):
        crit = CriteriaStub("cross", m_fGradeMax=0.02, m_fGradeWindowMax=0.035, m_fGradeWindowMaxAll=0.06,
                            m_fCrossMin=0.05, m_fCrossMax=0.12, m_fCrossAbsMaxAll=0.15, m_fCrossTarget=0.065,
                            m_fRmsMax=0.5)

        def slope(a, lat):          # right side (lateral +) lower by 6%
            return 50.0 - 0.06 * lat + 0.005 * a

        site = self.apply(SiteStub(310, 8, 75, 275, side=1), crit, slope)
        self.assertTrue(site.m_bTerrainOk, site.m_sTerrainReason)
        self.assertAlmostEqual(float(site.m_fCrossMean), 0.06, places=5)
        self.assertAlmostEqual(float(site.m_fCrossMin), 0.06, places=5)
        self.assertAlmostEqual(float(site.m_fGrade), 0.005, places=5)
        wrong = self.apply(SiteStub(310, 8, 75, 275, side=-1), crit, slope)
        self.assertFalse(wrong.m_bTerrainOk)
        self.assertEqual(wrong.m_sTerrainReason, "cross_zone")

    def test_bumpy_and_short_strips_are_refused(self):
        lax = CriteriaStub("flat", m_fGradeWindowMax=1, m_fGradeWindowMaxAll=1, m_fCrossMax=1, m_fCrossAbsMaxAll=1)
        bumpy = self.apply(SiteStub(1000, 22, 0, 1000), lax, lambda a, lat: 39.0 + 0.5 * math.sin(a / 7.0))
        self.assertFalse(bumpy.m_bTerrainOk)
        self.assertIn(bumpy.m_sTerrainReason, ("rms", "grade"))
        short = self.apply(SiteStub(8, 22, 0, 8), CriteriaStub("flat"), lambda a, lat: 39.0)
        self.assertFalse(short.m_bTerrainOk)
        self.assertEqual(short.m_sTerrainReason, "too_short")
        low = self.apply(SiteStub(1000, 22, 0, 1000), CriteriaStub("flat"), lambda a, lat: 1.0)
        self.assertEqual(low.m_sTerrainReason, "below_min_surface_y")

    def test_matches_a_python_reference(self):
        rng = random.Random(9)
        crit = CriteriaStub("cross", m_fGradeMax=0.05, m_fGradeWindowMax=0.1, m_fGradeWindowMaxAll=0.1,
                            m_fCrossMin=0.0, m_fCrossMax=0.2, m_fCrossAbsMaxAll=0.2, m_fCrossTarget=0.065,
                            m_fRmsMax=5.0)
        coeffs = [rng.uniform(-0.02, 0.02) for _ in range(3)]

        def h(a, lat):
            return 60.0 + coeffs[0] * a - (0.07 + coeffs[1]) * lat + 0.3 * math.sin(a / 40.0 + coeffs[2])

        site = self.apply(SiteStub(300, 6, 60, 260), crit, h)
        along, left, centre, right = strip_samples(300.0, 6.0, h)
        n = len(along)
        ma = sum(along) / n
        my = sum(centre) / n
        grade = sum((a - ma) * (c - my) for a, c in zip(along, centre)) / sum((a - ma) ** 2 for a in along)
        zone = [(l_ - r_) / 12.0 for a, l_, r_ in zip(along, left, right) if 60 <= a <= 260]
        self.assertAlmostEqual(float(site.m_fGrade), grade, places=4)
        self.assertAlmostEqual(float(site.m_fCrossMean), sum(zone) / len(zone), places=4)
        self.assertAlmostEqual(float(site.m_fCrossMin), min(zone), places=4)
        self.assertAlmostEqual(float(site.m_fCrossMax), max(zone), places=4)


if __name__ == "__main__":
    unittest.main()
