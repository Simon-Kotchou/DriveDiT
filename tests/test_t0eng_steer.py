#!/usr/bin/env python3
"""Synthetic-fixture tests for the G19-1 steering analyses (DD-14b): tools/probes/t0eng/e21_selfsteer.py (CC-07)
and tools/probes/t0eng/e23_steer_steps.py (CC-08), and the Enforce site arithmetic DC_SteerTerrainMath.

    PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_steer.py

Every case writes synthetic e21 / e22 / e23 plays (run_header.txt with the build / pin / prefab / map / addon
identity, the probe CSVs, run_footer.txt) with probe_common.write_synthetic_play into a fresh
probe_common.make_temp_dir() directory and removes it afterwards. The plays come from a small
vehicle model (SteerCar), written independently of the analyses: a steering actuator cmd -> input (delay, first
order, rate limit, gain), the input -> a bicycle wheel angle (0.5 rad per unit plus optional load and slope terms),
an exact Ackermann pair from it (plus toe), the path curvature tan(angle) / L with a speed-dependent lag (never the
analysis's delta_eff), optional merged physics steps (a row covering n engine steps). The step-CSV header is built
from the Enforce source, so a reader / producer name mismatch fails here.

Each counterexample of the Codex round-1 review (PR #27 @ 9fdc360) is a test here; the missing-evidence cases must
come out INCOMPLETE (never PASS, never a production mode), the physical failures FAIL.
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
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
PREFAB = "{DA79E34823120087}Prefabs/Vehicles/Wheeled/S105/S105_base.et"


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
    """cmd -> steering input per engine step: delay (steps), first order (tau s), rate limit (units / s), gain"""

    def __init__(self, delay=0, tau=0.0, rate=None, gain=1.0):
        self.delay, self.tau, self.rate, self.gain = delay, tau, rate, gain
        self.hist = []
        self.y = 0.0

    def step(self, cmd, call=True):
        if not call:
            return self.y
        self.hist.append(cmd)
        target = self.gain * (self.hist[-1 - self.delay] if len(self.hist) > self.delay else 0.0)
        a = math.exp(-DT / self.tau) if self.tau > 0 else 0.0
        step = (1.0 - a) * (target - self.y)
        if self.rate is not None:
            step = clamp(step, -self.rate * DT, self.rate * DT)
        self.y += step
        return self.y


def ackermann(delta):
    """(left, right) of a bicycle angle by the exact Ackermann geometry (positive right: the right wheel is inner)"""
    if abs(delta) < 1e-9:
        return delta, delta
    cot = 1.0 / math.tan(delta)
    half = TRACK / (2.0 * WHEELBASE)
    return math.atan(1.0 / (cot + half)), math.atan(1.0 / (cot - half))


class SteerCar:
    """The fixture vehicle. Options: actuator (cmd -> input), wheel_sign (the channels' sign), bone_lag (rows of lag
    of the rot and bone channels), late_changes (every command change k with k % 5 in (1, 3) lags the rot / bone
    channels by late_changes rows until the next change: 4 of 10), toe (rad), kappa_bias (1/m on the path),
    kappa_wave ((amplitude 1/m, wavelength m) on the path), slope_kappa (1/m of path curvature per unit cross-slope,
    wheels straight), zero_noise (readback deviations at zero command after >= 60 zero rows, the 1.9.0.37 e01
    pattern), stair_error (added to the readback of every non-zero command), load_gain (rad of wheel angle per m/s2
    of a_lat), slope_gain (rad of wheel angle per unit cross-slope), sigma (relaxation length m), kappa_sign, merge
    (engine steps per row: one row per dispatched step, physics_step + 1, timeslice n x dt, as DC_ProbeRunner writes
    it), heading_noise (rad rms on the written heading), readback_offset (added to the readback of every zero-command
    measure row after 60 zero rows), yaw_kappa_offset (1/m: a constant disagreement of the written yaw rate),
    kappa_bump ((start m, length m, 1/m) on every measure segment), late_kappa ((from m, 1/m) on every measure
    segment). The default slope_kappa gives the cross runs a steady drift of about 0.5 deg / 100 m (a reference
    that a 10% margin can resolve)."""

    def __init__(self, actuator=None, wheel_sign=1.0, bone_lag=0, late_changes=0, toe=0.0, kappa_bias=0.0,
                 kappa_wave=None, slope_kappa=1.45e-3, zero_noise=False, stair_error=0.0, load_gain=0.0,
                 slope_gain=0.0, sigma=0.6, kappa_sign=1.0, merge=1, heading_noise=0.0, readback_offset=0.0,
                 yaw_kappa_offset=0.0, kappa_bump=None, late_kappa=None, seed=3):
        self.act = actuator or Actuator()
        self.wheel_sign, self.bone_lag, self.late_changes, self.toe = wheel_sign, bone_lag, late_changes, toe
        self.kappa_bias, self.kappa_wave, self.slope_kappa = kappa_bias, kappa_wave, slope_kappa
        self.zero_noise, self.stair_error, self.load_gain = zero_noise, stair_error, load_gain
        self.slope_gain, self.sigma, self.kappa_sign, self.merge = slope_gain, sigma, kappa_sign, merge
        self.heading_noise, self.readback_offset = heading_noise, readback_offset
        self.yaw_kappa_offset, self.kappa_bump, self.late_kappa = yaw_kappa_offset, kappa_bump, late_kappa
        self.segment_key = None
        self.segment_start = 0.0
        self.offset_y = 0.0
        self.rng = random.Random(seed)
        self.step_no = 100
        self.world_ms = 3000.0
        self.rows = []
        self.lag_hist = []
        self.changes = 0
        self.last_cmd = 0.0
        self.zero_rows = 0
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
            site="flat", assist=2, mode="hold", slope=0.0, acc=None, merge=None):
        n = merge or self.merge
        if (phase, segment) != self.segment_key:
            self.segment_key = (phase, segment)
            self.segment_start = self.dist
        if abs(cmd - self.last_cmd) >= 0.04:
            self.changes += 1
        self.last_cmd = cmd
        self.step_no += 1
        for _ in range(n):
            self.world_ms += DT * 1000.0
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
            delta = steer_input * DMAX + self.load_gain * abs(self.a_lat) * (1 if cmd >= 0 else -1) + \
                self.slope_gain * slope
            target_kappa = self.kappa_sign * math.tan(delta) / WHEELBASE + self.kappa_bias + self.slope_kappa * slope
            if self.kappa_wave:
                target_kappa += self.kappa_wave[0] * math.sin(2.0 * math.pi * self.dist / self.kappa_wave[1])
            into = self.dist - self.segment_start
            if self.kappa_bump and segment == "measure" and \
                    self.kappa_bump[0] <= into < self.kappa_bump[0] + self.kappa_bump[1]:
                target_kappa += self.kappa_bump[2]
            if self.late_kappa and segment == "measure" and into >= self.late_kappa[0]:
                target_kappa += self.late_kappa[1]
            tau = max(DT, self.sigma / max(self.v, 0.5))
            self.kappa += (target_kappa - self.kappa) * (1.0 - math.exp(-DT / tau))
            r = self.kappa * self.v
            self.a_lat = self.v * r
            self.heading += math.degrees(r * DT)
            rad = math.radians(self.heading)
            self.x += self.v * math.sin(rad) * DT
            self.z += self.v * math.cos(rad) * DT
            self.dist += self.v * DT
        post = steer_input
        self.zero_rows = self.zero_rows + 1 if abs(cmd) < 1e-9 and steer_call else 0
        if self.zero_noise and self.zero_rows >= 60 and segment == "measure" and self.rng.random() < 0.16:
            post += self.rng.choice((-1, 1)) * self.rng.uniform(0.002, 0.0146)
        if self.stair_error and abs(cmd) > 1e-9:
            post += self.stair_error
        if self.readback_offset:
            # a hidden first order (tau 0.2 s) towards the offset: each row is within 1e-3 of a one-step prediction
            # from the previous readback, the sustained prediction is 0
            target = self.readback_offset if self.zero_rows >= 60 and segment == "measure" else 0.0
            self.offset_y += (1.0 - math.exp(-DT / 0.2)) * (target - self.offset_y)
            post += self.offset_y
        left, right = ackermann(delta)
        left, right = left + self.toe, right - self.toe
        self.lag_hist.append((left, right))
        lag = self.bone_lag
        if self.late_changes and self.changes % 5 in (1, 3):
            lag = self.late_changes
        lagged = self.lag_hist[-1 - lag] if len(self.lag_hist) > lag else (0.0, 0.0)
        h0 = math.radians(self.h0)
        along = (self.x - self.x0) * math.sin(h0) + (self.z - self.z0) * math.cos(h0)
        lateral = (self.x - self.x0) * math.cos(h0) - (self.z - self.z0) * math.sin(h0)
        s = self.wheel_sign
        acc_z = acc if acc is not None else (-3.0 if mode == "brake" else 0.0)
        row = {c: "" for c in STEP_COLUMNS}
        row.update({
            "physics_step": self.step_no, "world_time_ms": "%.3f" % self.world_ms,
            "timeslice_s": "%.6f" % (n * DT), "phase": phase, "segment": segment, "phase_step": 0, "run_idx": run_idx,
            "force_sim": 0, "cmd_steer": "%.4f" % cmd, "cmd_throttle": "0.3000", "cmd_brake": "0.0000",
            "cmd_clutch": "1.0000", "cmd_gear": 2, "cmd_handbrake": 0, "speed_hold_kmh": "%.3f" % (v_target * 3.6),
            "input_pre_steer": "%.4f" % cmd, "input_post_steer": "%.5f" % post, "engine_on": 1,
            "speed_kmh": "%.3f" % (self.v * 3.6), "vel_l_x": "0.000", "vel_l_y": "0.000", "vel_l_z": "%.3f" % self.v,
            "angvel_l_x": "0.000000", "angvel_l_y": "%.7f" % (r + self.yaw_kappa_offset * self.v),
            "angvel_l_z": "0.000000",
            "acc_l_x": "%.4f" % self.a_lat, "acc_l_y": "0.0000", "acc_l_z": "%.4f" % acc_z,
            "pos_x": "%.4f" % self.x, "pos_y": "39.000", "pos_z": "%.4f" % self.z,
            "fwd_x": "%.6f" % math.sin(rad), "fwd_y": "0.000000", "fwd_z": "%.6f" % math.cos(rad),
            "heading_deg": "%.5f" % (self.heading + (math.degrees(self.rng.gauss(0.0, self.heading_noise))
                                                     if self.heading_noise else 0.0)),
            "ground_contact": 1, "wheels_contact": 4,
            "steer_call": int(steer_call), "assist_mode": assist, "site": site, "along_m": "%.3f" % along,
            "lateral_m": "%.3f" % lateral, "in_area": 1, "seg_target": "%.4f" % seg_target, "seg_flag": seg_flag,
            "event_idx": event_idx,
            "wa_rot_l": "%.6f" % (s * lagged[0]), "wa_rot_r": "%.6f" % (s * lagged[1]),
            "wa_bone_l": "%.6f" % (s * lagged[0]), "wa_bone_r": "%.6f" % (s * lagged[1]),
            "wa_ent_l": "%.6f" % (s * left), "wa_ent_r": "%.6f" % (s * right),
            "sig_steer": "%.6f" % steer_input, "wa_rot_l_pf": "%.6f" % (s * left), "wa_rot_r_pf": "%.6f" % (s * right),
            "slip_lat_l": "0.0", "slip_lat_r": "0.0", "vdir_l": "%.6f" % left, "vdir_r": "%.6f" % right})
        self.rows.append(row)


def phase_row(car, index, name, kind, site, kmh, steer, assist, first_step, result="ok", prbs=("", "", "", "")):
    """A <probe>_phases.csv row as DC_ProbeSteerBase writes it (start / end = the phase's first and last physics_step;
    prbs = bits, expected, bit steps, result: e23 only)"""
    return [index, name, kind, site, first_step, car.step_no, "", "", kmh, steer, assist, 1, result, ""] + list(prbs)


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


def common_header(probe, build="1.9.0.37"):
    return {"schema": "dc-probe-run/1", "probe": probe, "params.dt_s": "%.6f" % DT, "steer.wheelbase_m": WHEELBASE,
            "engine.build_version": build, "launch.pin_id": build, "launch.build_key": "", "ego.prefab": PREFAB,
            "world.map_id": "arland", "addon.git_sha": "0123456789abcdef0123456789abcdef01234567",
            "steer.wheel_index_l": 0, "steer.wheel_index_r": 1, "steer.wheel_count": 4, "steer.sign_used": 1,
            "assist.original": 2, "assist.original_name": "FULL", "assist.sets": 2, "assist.restores": 2,
            "assist.at_stop": 2, "vehicle.noise_steer_sensitivity": "0.0000", "site.flat.found": 1,
            "site.flat.source": "explicit", "site.flat.heading_deg": "75.110", "site.flat.grade": "0.00001",
            "site.flat.cross_mean": "0.00010", "site.flat.benign_hits": 0}


FOOTER = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}


CROSS_PROFILE = " ".join("%.3f" % (50.0 + 0.004 * 5 * k + (0.039 if j else 0.117)) for k in range(63) for j in (0, 1))


def build_e21(play_dir, car=None, cross_found=True, cross_min=0.06, flat_m=200.0, flat_kmh=None, build="1.9.0.37",
              cross_start="2000.0 0.0 1500.0", cross_heading="33.0", profile=CROSS_PROFILE, footer=FOOTER,
              header_extra=None):
    car = car or SteerCar()
    phases = []

    def zero_phase(name, kmh, held, nocall, site="flat", assist=2, slope=0.0, drive_kmh=None):
        first = car.step_no + 1
        v = (drive_kmh or kmh) / 3.6
        drive_start(car, name, 1, v, site, assist)
        for _ in range(60):
            car.run(name, "hold", 1, 0.0, v, site=site, assist=assist, slope=slope, seg_target=kmh)
        for segment, length, call in (("measure", held, 1), ("nocall", nocall, 0)):
            start = car.dist
            while car.dist - start < length + 0.5:      # the probe ends the segment past the distance (along)
                car.run(name, segment, 1, 0.0, v, steer_call=call, seg_flag=1, site=site, assist=assist,
                        slope=slope, seg_target=kmh)
        drive_stop(car, name, 1, site, assist)
        phases.append(phase_row(car, len(phases), name, site, site, kmh, 0, assist, first))

    for kmh in (20, 40):
        zero_phase("flat_%d" % kmh, kmh, flat_m, 200.0, drive_kmh=(flat_kmh or {}).get(kmh))
    for kmh in (20, 40):
        zero_phase("flat_%d_none" % kmh, kmh, 100.0, 100.0, assist=0)
    if cross_found:
        for kmh in (20, 40):
            zero_phase("cross_%d" % kmh, kmh, 100.0, 100.0, site="cross", slope=0.06)
    header = common_header("e21", build)
    header.update({"params.m_sFlatSpeedsKmh": "20,40", "params.m_sCrossSpeedsKmh": "20,40",
                   "site.cross.found": int(cross_found), "site.cross.cross_min_signed": "%.5f" % cross_min,
                   "site.cross.cross_max_signed": "0.07000", "site.cross.cross_mean": "0.06500",
                   "site.cross.start": cross_start, "site.cross.heading_deg": cross_heading,
                   "site.cross.length_m": "310.0", "site.cross.zone_from_m": "75.0", "site.cross.zone_to_m": "275.0",
                   "site.cross.track_profile": profile, "site.cross.grade": "0.004", "site.cross.fallback": 0})
    header.update(header_extra or {})
    probe_common.write_synthetic_play(play_dir, header, {"e21_steps": (STEP_COLUMNS, car.rows),
                                                         "e21_phases": (PHASE_COLUMNS, phases)}, footer)
    return play_dir


def build_e22(play_dir, car=None, steer_load=(0.3, -0.3), targets=(2.0, 2.75, 3.5), steady_rows=110,
              header_extra=None):
    car = car or SteerCar()
    phases = []
    for kmh, levels in ((10.5, (0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.05, -0.05, 0.15, -0.15, 0.25, -0.25)),):
        name = "stair_%g" % kmh
        first = car.step_no + 1
        v = kmh / 3.6
        drive_start(car, name, 1, v)
        for level, steer in enumerate(levels):
            for k in range(90):
                car.run(name, "stair", 1, steer, v, seg_flag=int(k >= 30), event_idx=level, seg_target=kmh)
        drive_stop(car, name, 1)
        phases.append(phase_row(car, len(phases), name, "stair", "flat", kmh, 0, -1, first))
    for steer in steer_load:
        name = "load_%+.2f" % steer
        first = car.step_no + 1
        v = 10.0 / 3.6
        drive_start(car, name, 1, v)
        for k in range(180):
            car.run(name, "ref", 1, steer, v, seg_flag=int(k >= 90), seg_target=10.0)
        kappa = abs(math.tan(steer * DMAX) / WHEELBASE)
        for level, target in enumerate(targets):
            vt = math.sqrt(target / kappa)
            for k in range(40 + steady_rows):
                if k < 40 and k % 8 == 7 and abs(car.a_lat) > 0.1:
                    vt *= math.sqrt(target / abs(car.a_lat))       # the probe adapts the speed to the measured a_lat
                car.run(name, "alat", 1, steer, vt, seg_flag=int(k >= 40), event_idx=level, seg_target=target)
        while car.v > 10.0 / 3.6:
            car.run(name, "brake", 1, steer, car.v, mode="brake", seg_flag=1, seg_target=3.0)
        drive_stop(car, name, 1)
        phases.append(phase_row(car, len(phases), name, "load", "flat", 10, steer, -1, first))
    header = common_header("e22")
    header.update({"params.m_sALatTargets": ",".join("%g" % t for t in targets), "params.m_fBrakeDecelMs2": "3.000"})
    header.update(header_extra or {})
    probe_common.write_synthetic_play(play_dir, header, {"e22_steps": (STEP_COLUMNS, car.rows),
                                                         "e22_phases": (PHASE_COLUMNS, phases)}, FOOTER)
    return play_dir


STEPS_TEXT = read(STEPS_SOURCE)
HEADING_BUDGET_DEG = float(re.search(r"HEADING_BUDGET_DEG = ([0-9.]+);", STEPS_TEXT).group(1))
KAPPA_PER_STEER = float(re.search(r"KAPPA_PER_STEER = ([0-9.]+);", STEPS_TEXT).group(1))
ON_STEPS = 12          # m_fStepOnS 0.4 s at 30 Hz
BIT_STEPS = 5          # m_fPrbsBitS 0.15 s at 30 Hz (DC_ProbeSequences.StepsFor rounds 4.5 up)
PRBS_AMPLITUDES = {10.0: 0.5, 20.0: 0.4, 30.0: 0.3, 40.0: 0.25, 60.0: 0.2}


def hold_steps(amplitude, kmh):
    """DC_ProbeSteerSteps.HoldStepsFor"""
    per_step = KAPPA_PER_STEER * abs(amplitude) * max(kmh, 1.0) / 3.6 * DT
    return max(1, min(ON_STEPS, int(math.floor(math.radians(HEADING_BUDGET_DEG) / per_step))))


def max_running_sum(bits):
    total, top = 0, 1
    for bit in bits:
        total += 1 if bit else -1
        top = max(top, abs(total))
    return top


def prbs_bit_steps(amplitude, kmh, bits):
    """DC_ProbeSteerSteps.PrbsBitStepsFor"""
    per_step = KAPPA_PER_STEER * abs(amplitude) * max(kmh, 1.0) / 3.6 * DT * max_running_sum(bits)
    return max(1, min(BIT_STEPS, int(math.floor(math.radians(HEADING_BUDGET_DEG) / per_step))))


def build_e23(play_dir, car=None, grid="10:0.5x6,1.0x4;20:0.5x4;30:0.5x6,1.0x4;40:0.5x4;60:0.5x6", seed=5,
              prbs=True, prbs_gain=None, slow_kmh=None, alt_tau=None, prbs_bits_done=None, phase_results=None):
    """slow_kmh: {speed: the speed the phase decays to while its train runs}; alt_tau: (tau_a, tau_b) alternating per
    event; prbs_gain: the actuator gain during the PRBS (a PRBS the step model cannot predict); prbs_bits_done:
    {speed: bits} of a PRBS cut short; phase_results: {speed: result} (an abort that still ends with a complete
    footer). Holds and PRBS bits follow the probe's heading budget (hold_steps, prbs_bit_steps)"""
    car = car or SteerCar()
    rng = random.Random(seed)
    events = []
    phases = []
    prbs_bits = e23_prbs_bits(5, 11, 31)
    for group in grid.split(";"):
        speed_text, specs = group.split(":")
        kmh = float(speed_text)
        amps = []
        for spec in specs.split(","):
            amp, count = spec.split("x")
            amps.append([float(amp), int(count)])
        order = []
        while any(c > 0 for _, c in amps):
            for item_ in amps:
                if item_[1] > 0:
                    order += [item_[0], -item_[0]]
                    item_[1] -= 1
        name = "steps_%d" % kmh
        first = car.step_no + 1
        v = kmh / 3.6
        drive_start(car, name, 1, v)
        for _ in range(30):
            car.run(name, "hold", 1, 0.0, v, seg_target=kmh)
        open_event = -1
        pending = None
        end_v = (slow_kmh or {}).get(kmh, kmh) / 3.6
        for idx, amp in enumerate(order):
            vt = v + (end_v - v) * idx / max(1, len(order) - 1)
            if alt_tau:
                car.act.tau = alt_tau[idx % 2]
            gap = 18 + rng.randint(0, 9)
            for k in range(gap):
                car.run(name, "train", 1, 0.0, vt, event_idx=open_event, seg_target=kmh)
                if k == 0 and pending is not None:
                    pending["release_step"] = car.step_no
            if pending is not None:
                pending["end_step"] = car.step_no
                events.append(pending)
            open_event = idx
            hold = hold_steps(amp, kmh)
            for k in range(hold):
                car.run(name, "train", 1, amp, vt, seg_flag=1, event_idx=idx, seg_target=kmh)
                if k == 0:
                    pending = {"event_idx": idx, "phase": name, "target_kmh": "%.3f" % kmh, "amplitude": "%.4f" % amp,
                               "run_idx": 1, "jitter_steps": gap - 18, "onset_step": car.step_no,
                               "release_step": "", "end_step": "", "onset_kmh": "%.3f" % (vt * 3.6),
                               "onset_along_m": "0.0", "hold_steps": hold, "result": "ok"}
        for k in range(18):
            car.run(name, "train", 1, 0.0, end_v, event_idx=open_event, seg_target=kmh)
            if k == 0:
                pending["release_step"] = car.step_no
        pending["end_step"] = car.step_no
        events.append(pending)
        prbs_cols = (0, 0, 1, "none")
        if prbs:
            gain = car.act.gain
            if prbs_gain is not None:
                car.act.gain = prbs_gain
            amp = PRBS_AMPLITUDES.get(kmh, 0.2)
            bit_steps = prbs_bit_steps(amp, kmh, prbs_bits)
            done = (prbs_bits_done or {}).get(kmh, len(prbs_bits))
            for bit in prbs_bits[:done]:
                for _ in range(bit_steps):
                    car.run(name, "prbs", 1, amp if bit else -amp, end_v, seg_flag=1, seg_target=kmh)
            car.act.gain = gain
            prbs_cols = (done, len(prbs_bits), bit_steps, "ok" if done == len(prbs_bits) else "incomplete")
        drive_stop(car, name, 1)
        phases.append(phase_row(car, len(phases), name, "steps", "flat", kmh, 0, -1, first,
                                result=(phase_results or {}).get(kmh, "ok"), prbs=prbs_cols))
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
        code = steer_common.run_cli(module.main, argv)
    return code, out.getvalue()


def item(text, name):
    """The [TAG] of check name in a report"""
    m = re.search(r"\[(PASS|FAIL|INCOMPLETE|INFO)\] %s:" % re.escape(name), text)
    return m.group(1) if m else None


def detail(text, name):
    m = re.search(r"\[(?:PASS|FAIL|INCOMPLETE|INFO)\] %s: (.*)" % re.escape(name), text)
    return m.group(1) if m else ""


def drop_footer(play_dir):
    os.remove(os.path.join(play_dir, "run_footer.txt"))


def blank_columns(play_dir, probe, columns):
    path = os.path.join(play_dir, probe + "_steps.csv")
    with open(path, "r", encoding="ascii") as handle:
        lines = handle.read().splitlines()
    header = lines[0].split(",")
    index = [header.index(c) for c in columns]
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        for i in index:
            cells[i] = ""
        out.append(",".join(cells))
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(out) + "\n")


def edit_column(play_dir, probe, where, column, value):
    """Sets column to value on every <probe>_steps.csv row where(row dict) accepts"""
    path = os.path.join(play_dir, probe + "_steps.csv")
    with open(path, "r", encoding="ascii") as handle:
        lines = handle.read().splitlines()
    header = lines[0].split(",")
    index = header.index(column)
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        if where(dict(zip(header, cells))):
            cells[index] = value
        out.append(",".join(cells))
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(out) + "\n")


def blank_segment(play_dir, probe, segment, column):
    edit_column(play_dir, probe, lambda r: r["segment"] == segment, column, "")


def no_mode(test, text):
    """Coordinator decision (round 3): never an automated Mode A / B decision, only the manual-review line"""
    test.assertNotIn("Mode A stays", text)
    test.assertNotIn("Mode B (seated driver", text)
    test.assertIn("mode_decision: none (manual review: G19 admission record); items: evidence ", text)


class Cc07Tests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_steer_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def path(self, *parts):
        return os.path.join(self.base, *parts)

    def session(self, e21_car=None, e22_car=None, baseline_car=None, baseline=True, **kw):
        a = build_e21(self.path("e21", "play_01"), e21_car, **kw)
        b = build_e22(self.path("e22", "play_01"), e22_car)
        argv = [a, "--e22", b]
        if baseline:
            base = build_e21(self.path("base", "play_01"), baseline_car, build="1.8.0.13")
            argv += ["--baseline", base]
        return argv

    def test_clean_session_with_baseline_passes_cc07(self):
        code, text = run_main(e21, self.session())
        for name in ("evidence", "step0_channel", "a_readback", "b_flat", "c_cross", "d_load"):
            self.assertEqual(item(text, name), "PASS", "%s in\n%s" % (name, text))
        self.assertEqual(code, 0, text)
        self.assertIn("selected rot", text)
        no_mode(self, text)
        self.assertIn("d_load PASS (complete)", text)

    def test_without_a_baseline_c_is_incomplete_and_no_mode_is_chosen(self):
        code, text = run_main(e21, self.session(baseline=False))
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)
        self.assertEqual(item(text, "b_flat"), "PASS", text)
        self.assertEqual(code, 0, text)
        no_mode(self, text)

    # (a) readback
    def test_first_order_input_lag_is_explained_by_the_cc08_model(self):
        def act():
            return Actuator(tau=0.014)       # the 1.9.0.37 e01 reversal fit

        argv = self.session(e21_car=SteerCar(actuator=act()), e22_car=SteerCar(actuator=act()))
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertIn("needs the CC-08 input model", text)
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()))
        code, text = run_main(e21, argv + ["--e23", e23_play])
        self.assertEqual(item(text, "a_readback"), "PASS", text)
        self.assertIn(" 0 deviating rows not explained", text)

    def test_unexplained_zero_command_deviations_fail_a(self):
        # Codex: 71 deviations > 1e-3 after >= 60 zero-command rows (max 0.0146) that the short lag cannot explain
        def act():
            return Actuator(tau=0.014)

        argv = self.session(e21_car=SteerCar(actuator=act(), zero_noise=True), e22_car=SteerCar(actuator=act()))
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()))
        code, text = run_main(e21, argv + ["--e23", e23_play])
        self.assertEqual(item(text, "a_readback"), "FAIL", text)
        self.assertRegex(detail(text, "a_readback"), r" [1-9]\d* deviating rows not explained")
        self.assertRegex(detail(text, "a_readback"), r"[1-9]\d* of them after >= 60 zero-command rows")
        self.assertEqual(code, 0, text)
        no_mode(self, text)

    def test_nonzero_command_readback_error_never_passes(self):
        # Codex: staircase rows with a 0.1 readback error left a PASS (only zero-command rows were read)
        argv = self.session(e22_car=SteerCar(stair_error=0.1))
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertIn("e22/stair", text)
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(), grid="10:0.5x6;30:0.5x6")
        code, text = run_main(e21, argv + ["--e23", e23_play])
        # the e22 load rows at 20..27 km/h lie outside this model's cells (10 and 30 km/h only): never PASS
        self.assertIn(item(text, "a_readback"), ("FAIL", "INCOMPLETE"), text)
        self.assertRegex(detail(text, "a_readback"), r" [1-9]\d* deviating rows not explained")

    # (b) flat
    def test_alternating_curvature_fails_b_despite_zero_net_drift(self):
        # Codex: peak 8e-4 1/m alternating left / right, net heading change zero, passed the old net check
        code, text = run_main(e21, self.session(e21_car=SteerCar(kappa_wave=(8e-4, 100.0))))
        self.assertEqual(item(text, "b_flat"), "FAIL", text)
        self.assertRegex(detail(text, "b_flat"), r"net -?\d")

    def test_baseline_allowance_admits_a_matching_drift(self):
        # Codex: current and baseline both 3e-4 1/m failed although the printed bound was 4.5e-4
        argv = self.session(e21_car=SteerCar(kappa_bias=3e-4), baseline_car=SteerCar(kappa_bias=3e-4))
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "b_flat"), "PASS", text)
        self.assertIn("allowance 0.00045", text)
        code, text = run_main(e21, self.session(e21_car=SteerCar(kappa_bias=3e-4), baseline=False))
        self.assertEqual(item(text, "b_flat"), "INCOMPLETE", text)

    def test_drift_beyond_the_baseline_allowance_fails_b(self):
        code, text = run_main(e21, self.session(e21_car=SteerCar(kappa_bias=3e-4)))
        self.assertEqual(item(text, "b_flat"), "FAIL", text)

    def test_a_baseline_from_the_same_build_is_not_used(self):
        a = build_e21(self.path("e21", "play_01"), SteerCar(kappa_bias=3e-4))
        b = build_e22(self.path("e22", "play_01"))
        base = build_e21(self.path("base", "play_01"), SteerCar(kappa_bias=3e-4))
        code, text = run_main(e21, [a, "--e22", b, "--baseline", base])
        self.assertIn("baseline: not usable", text)
        self.assertEqual(item(text, "b_flat"), "INCOMPLETE", text)

    def test_short_or_off_speed_flat_segments_are_incomplete(self):
        code, text = run_main(e21, self.session(flat_m=120.0))
        self.assertEqual(item(text, "b_flat"), "INCOMPLETE", text)
        self.assertIn("< 200 m", text)
        code, text = run_main(e21, self.session(flat_kmh={40: 30}))
        self.assertEqual(item(text, "b_flat"), "INCOMPLETE", text)
        self.assertIn("not within 10% of 40", text)

    # (c) cross
    def test_cross_drift_with_straight_wheels_fails_c(self):
        # Codex: > 2 deg / 100 m of heading drift with straight wheels passed on the wheel-angle criterion
        code, text = run_main(e21, self.session(e21_car=SteerCar(slope_kappa=7e-3)))
        self.assertEqual(item(text, "c_cross"), "FAIL", text)
        self.assertEqual(item(text, "b_flat"), "PASS", text)
        self.assertRegex(detail(text, "c_cross"), r"heading drift 2\.\d")
        code, text = run_main(e21, self.session(e21_car=SteerCar(slope_kappa=7e-3), baseline=False))
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)
        no_mode(self, text)

    def test_a_baseline_on_another_cross_site_is_not_a_reference(self):
        a = build_e21(self.path("e21", "play_01"))
        b = build_e22(self.path("e22", "play_01"))
        base = build_e21(self.path("base", "play_01"), build="1.8.0.13", cross_start="2100.0 0.0 1500.0")
        code, text = run_main(e21, [a, "--e22", b, "--baseline", base])
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)
        self.assertIn("baseline site 100 m", text)

    def test_missing_or_shallow_cross_site_is_incomplete_not_mode_b(self):
        # Codex: no cross site printed "Mode B ... becomes the production candidate"
        code, text = run_main(e21, self.session(cross_found=False))
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)
        self.assertIn("no_site", text)
        no_mode(self, text)
        code, text = run_main(e21, self.session(cross_min=0.03))
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)

    # evidence and identity
    def test_an_interrupted_play_prints_no_mode(self):
        # Codex: without the e21 footer the result was INCOMPLETE but still printed "Mode A stays"
        argv = self.session()
        drop_footer(argv[0])
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "evidence"), "INCOMPLETE", text)
        self.assertIn("withheld: a play it depends on is not complete]", text)
        self.assertEqual(code, 0, text)
        no_mode(self, text)

    def test_contradictory_identities_are_refused(self):
        # Codex: e21 and e22 with contradictory build and prefab identities still produced CC-07 PASS
        a = build_e21(self.path("e21", "play_01"))
        b = build_e22(self.path("e22", "play_01"), header_extra={"ego.prefab": "{0000}Prefabs/Other.et"})
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(code, 1, text)
        self.assertIn("differ in ego.prefab", text)
        c = build_e22(self.path("e22b", "play_01"), header_extra={"engine.build_version": "1.8.0.13"})
        code, text = run_main(e21, [a, "--e22", c])
        self.assertEqual(code, 1, text)

    def test_a_play_without_identity_stamps_is_refused(self):
        a = build_e21(self.path("e21", "play_01"), header_extra={"launch.pin_id": ""})
        b = build_e22(self.path("e22", "play_01"))
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(code, 1, text)
        self.assertIn("lacks the identity field(s) launch.pin_id", text)

    # step 0
    def test_wrong_sign_wheel_channels_fail_step0(self):
        argv = self.session(e21_car=SteerCar(wheel_sign=-1.0), e22_car=SteerCar(wheel_sign=-1.0))
        code, text = run_main(e21, ["--signal-scale-deg", "1"] + argv)
        self.assertEqual(item(text, "step0_channel"), "FAIL", text)
        self.assertEqual(item(text, "d_load"), "INCOMPLETE", text)
        self.assertEqual(code, 0, text)
        no_mode(self, text)

    def test_intermittently_lagged_bones_are_not_validated(self):
        # Codex: 4 of 10 transitions delayed by 4 steps still selected rot (the median lag stayed 0)
        code, text = run_main(e21, self.session(e22_car=SteerCar(late_changes=4)))
        self.assertEqual(item(text, "step0_channel"), "PASS", text)
        self.assertIn("selected ent", text)
        self.assertRegex(text, r"rot: median error [^;]*; lag <= 1 steps on \d+ of \d+ measurable changes \(max 4\)"
                               r"[^;]*-> FAIL")

    def test_merged_steps_leave_the_one_step_lag_unresolved(self):
        code, text = run_main(e21, self.session(e22_car=SteerCar(merge=2)))
        self.assertEqual(item(text, "step0_channel"), "INCOMPLETE", text)
        self.assertIn("merged", detail(text, "step0_channel"))

    # (d) load
    def test_angle_that_follows_the_load_fails_d(self):
        code, text = run_main(e21, self.session(e22_car=SteerCar(load_gain=math.radians(0.5))))
        self.assertEqual(item(text, "d_load"), "FAIL", text)

    def test_missing_load_conditions_are_incomplete(self):
        # Codex: the negative-steer phase removed, one sample per condition, no lateral-load rows and an empty
        # target list all passed
        a = build_e21(self.path("e21", "play_01"))
        base = build_e21(self.path("base", "play_01"), build="1.8.0.13")
        cases = {"no_negative": dict(steer_load=(0.3,)), "one_row": dict(steady_rows=1),
                 "no_targets": dict(targets=())}
        for name, kw in cases.items():
            with self.subTest(name):
                b = build_e22(self.path("e22_" + name, "play_01"), **kw)
                code, text = run_main(e21, [a, "--e22", b, "--baseline", base])
                self.assertEqual(item(text, "d_load"), "INCOMPLETE", text)
                no_mode(self, text)

    def test_empty_plays_never_pass(self):
        a = probe_common.write_synthetic_play(self.path("e21_empty", "play_01"), common_header("e21"),
                                              {"e21_steps": (STEP_COLUMNS, []), "e21_phases": (PHASE_COLUMNS, [])},
                                              FOOTER)
        b = probe_common.write_synthetic_play(self.path("e22_empty", "play_01"), common_header("e22"),
                                              {"e22_steps": (STEP_COLUMNS, []), "e22_phases": (PHASE_COLUMNS, [])},
                                              FOOTER)
        code, text = run_main(e21, [a, "--e22", b])
        self.assertEqual(code, 0, text)
        self.assertNotIn("] step0_channel: L", text.replace("[INCOMPLETE] step0_channel: L", ""))
        for name in ("step0_channel", "a_readback", "b_flat", "c_cross", "d_load"):
            self.assertNotEqual(item(text, name), "PASS", text)
        no_mode(self, text)

    def test_missing_column_is_input_error(self):
        columns = [c for c in STEP_COLUMNS if c != "wa_rot_l"]
        a = probe_common.write_synthetic_play(self.path("e21_col", "play_01"), common_header("e21"),
                                              {"e21_steps": (columns, [])}, FOOTER)
        code, text = run_main(e21, [a])
        self.assertEqual(code, 1, text)


    # --- Codex round 2 (a58ddd6) ------------------------------------------------------------------------------
    def test_a_failed_part_never_hides_a_missing_one(self):
        # Codex: no negative-load phase plus a failed positive load selected Mode B (exit 1)
        a = build_e21(self.path("e21", "play_01"))
        b = build_e22(self.path("e22", "play_01"), SteerCar(load_gain=math.radians(0.5)), steer_load=(0.3,))
        base = build_e21(self.path("base", "play_01"), build="1.8.0.13")
        code, text = run_main(e21, [a, "--e22", b, "--baseline", base])
        self.assertEqual(item(text, "d_load"), "INCOMPLETE", text)
        self.assertIn("[observed FAIL", detail(text, "d_load"))
        self.assertEqual(code, 0, text)
        no_mode(self, text)
        self.assertIn("d_load INCOMPLETE (incomplete, a FAIL observed inside)", text)

    def test_an_interrupted_e23_withholds_its_dependents(self):
        # Codex: removing e23's footer gave evidence PASS, exit 0 and Mode A (e23 silently dropped)
        def act():
            return Actuator(tau=0.014)

        argv = self.session(e21_car=SteerCar(actuator=act()), e22_car=SteerCar(actuator=act()))
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()))
        drop_footer(e23_play)
        code, text = run_main(e21, argv + ["--e23", e23_play])
        self.assertEqual(item(text, "evidence"), "INCOMPLETE", text)
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertEqual(item(text, "step0_channel"), "INCOMPLETE", text)
        self.assertEqual(code, 0, text)
        no_mode(self, text)

    def test_missing_called_rows_or_values_are_incomplete(self):
        # Codex: blanking every e22 braking readback still gave a_readback PASS and Mode A
        argv = self.session()
        e22_steps = os.path.join(argv[2], "e22_steps.csv")
        with open(e22_steps, "r", encoding="ascii") as handle:
            lines = handle.read().splitlines()
        header = lines[0].split(",")
        seg, post = header.index("segment"), header.index("input_post_steer")
        out = [lines[0]]
        for line in lines[1:]:
            cells = line.split(",")
            if cells[seg] == "brake":
                cells[post] = ""
            out.append(",".join(cells))
        with open(e22_steps, "w", encoding="ascii") as handle:
            handle.write("\n".join(out) + "\n")
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertIn("no finite command / readback", detail(text, "a_readback"))
        no_mode(self, text)
        # rows missing from a phase (a gap in physics_step) are just as incomplete
        argv = self.session()
        e21_steps = os.path.join(argv[0], "e21_steps.csv")
        with open(e21_steps, "r", encoding="ascii") as handle:
            lines = handle.read().splitlines()
        del lines[400:410]
        with open(e21_steps, "w", encoding="ascii") as handle:
            handle.write("\n".join(lines) + "\n")
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertRegex(detail(text, "a_readback"), r"\d+ of \d+ rows")

    def test_a_persistent_offset_fails_the_free_running_model(self):
        # Codex: a validated tau 0.2 s model passed 300 zero-command rows carrying a 0.005 offset (one-step 0.000768)
        def act():
            return Actuator(tau=0.2)

        argv = self.session(e21_car=SteerCar(actuator=act(), readback_offset=0.005), e22_car=SteerCar(actuator=act()))
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()))
        code, text = run_main(e21, argv + ["--e23", e23_play])
        # FAIL, or INCOMPLETE when e23's own slow stops leave deviations outside the validated cells; never PASS
        self.assertIn(item(text, "a_readback"), ("FAIL", "INCOMPLETE"), text)
        if item(text, "a_readback") == "INCOMPLETE":
            self.assertIn("[observed FAIL", detail(text, "a_readback"))
        m = re.search(r"one-step max ([0-9.e-]+), free-running max ([0-9.e-]+); on every row: 1 s mean max ([0-9.e-]+)",
                      detail(text, "a_readback"))
        self.assertIsNotNone(m, text)
        self.assertLess(float(m.group(1)), 1e-3)
        self.assertGreater(float(m.group(2)), 4e-3)
        self.assertGreater(float(m.group(3)), 2e-4)

    def test_the_model_exception_needs_a_passing_prbs(self):
        # Codex: a 10:0.5x6 first-order fixture without PRBS qualified as a validated model and gave Mode A
        def act():
            return Actuator(tau=0.014)

        argv = self.session(e21_car=SteerCar(actuator=act()), e22_car=SteerCar(actuator=act()))
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()), grid="10:0.5x6", prbs=False)
        code, text = run_main(e21, argv + ["--e23", e23_play])
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertIn("PRBS 10 km/h INCOMPLETE", detail(text, "a_readback"))
        no_mode(self, text)

    def test_short_wave_curvature_fails_b(self):
        # Codex: 8e-4 1/m at a 25 m wavelength kept the 25 m window maxima at 1.15e-5 and passed
        code, text = run_main(e21, self.session(e21_car=SteerCar(kappa_wave=(8e-4, 25.0))))
        self.assertEqual(item(text, "b_flat"), "FAIL", text)
        m = re.search(r"heading p95 ([0-9.e-]+)", detail(text, "b_flat"))
        self.assertGreater(float(m.group(1)), 5e-4)

    def test_an_unresolvable_noise_floor_is_incomplete(self):
        code, text = run_main(e21, self.session(e21_car=SteerCar(heading_noise=5e-4),
                                                baseline_car=SteerCar(heading_noise=5e-4)))
        self.assertEqual(item(text, "b_flat"), "INCOMPLETE", text)
        self.assertIn("not resolvable", detail(text, "b_flat"))

    def test_a_near_zero_baseline_gets_no_uncalibrated_floor(self):
        # Codex: a zero-drift baseline and 0.09 deg/100 m passed on the 0.1 deg/100 m floor
        code, text = run_main(e21, self.session(e21_car=SteerCar(slope_kappa=2.62e-4),
                                                baseline_car=SteerCar(slope_kappa=0.0)))
        self.assertRegex(detail(text, "c_cross"), r"heading drift 0\.0[89]")
        self.assertEqual(item(text, "c_cross"), "FAIL", text)

    def test_the_baseline_must_drive_the_surveyed_strip(self):
        # Codex: 4.9 m and 1.9 deg apart were accepted as the same site
        a = build_e21(self.path("e21", "play_01"))
        b = build_e22(self.path("e22", "play_01"))
        cases = {"close": (dict(cross_start="2000.6 0.0 1500.5", cross_heading="33.4"), "PASS"),
                 "far": (dict(cross_start="2004.9 0.0 1500.0", cross_heading="34.9"), "INCOMPLETE"),
                 "other_ground": (dict(profile=" ".join("%.3f" % (float(v) + 0.1) for v in CROSS_PROFILE.split())),
                                  "INCOMPLETE")}
        for name, (kw, expected) in cases.items():
            with self.subTest(name):
                base = build_e21(self.path("base_" + name, "play_01"), build="1.8.0.13", **kw)
                code, text = run_main(e21, [a, "--e22", b, "--baseline", base])
                self.assertEqual(item(text, "c_cross"), expected, text)
        self.assertIn("not the same ground", detail(text, "c_cross"))


    # --- Codex round 3 (91bb2cb) ------------------------------------------------------------------------------
    def test_a_second_half_drift_is_unresolved_not_tolerated(self):
        # Codex: drift only in the second half of each measure segment widened its own tolerance (sigma 3.41) and
        # passed against a zero-drift baseline
        code, text = run_main(e21, self.session(e21_car=SteerCar(late_kappa=(50.0, 6e-4)),
                                                baseline_car=SteerCar(slope_kappa=0.0)))
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)
        self.assertIn("unresolved: the two halves differ", detail(text, "c_cross"))
        no_mode(self, text)

    def test_curvature_magnitude_survives_every_wavelength(self):
        # Codex: a 4.9 m wave of 8e-4 1/m averaged to 3.5e-5 and passed; the magnitude is now taken per step
        for wavelength in (2.0, 4.9, 10.0, 25.0, 50.0):
            with self.subTest(wavelength=wavelength):
                code, text = run_main(e21, self.session(e21_car=SteerCar(kappa_wave=(8e-4, wavelength))))
                self.assertNotEqual(item(text, "b_flat"), "PASS", text)
                p95 = float(re.search(r"heading p95 ([0-9.e-]+)", detail(text, "b_flat")).group(1))
                self.assertGreater(p95, 1e-4, text)
                if wavelength >= 4.9:
                    self.assertEqual(item(text, "b_flat"), "FAIL", text)
        code, text = run_main(e21, self.session(e21_car=SteerCar(kappa_wave=(3e-5, 4.9))))
        self.assertEqual(item(text, "b_flat"), "PASS", text)

    def test_an_excursion_with_a_channel_offset_is_not_a_pass(self):
        # Codex: a short excursion plus a constant yaw / heading disagreement passed on the larger channel's floor
        car = SteerCar(kappa_bump=(80.0, 2.0, 6e-4), yaw_kappa_offset=7e-5)
        code, text = run_main(e21, self.session(e21_car=car))
        self.assertNotEqual(item(text, "b_flat"), "PASS", text)
        self.assertRegex(detail(text, "b_flat"), r"systematic heading - yaw offset -7(\.\d+)?e-05")

    def test_a_truncated_or_aborted_prbs_is_not_held_out(self):
        # Codex: 8 of 31 PRBS bits in a phase that ended aborted_heading, complete footer, still validated the model
        def act():
            return Actuator(tau=0.014)

        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()), prbs_bits_done={10.0: 8},
                             phase_results={10.0: "aborted_heading"})
        code, text = run_main(e23, [e23_play])
        self.assertEqual(code, 0, text)
        self.assertEqual(item(text, "prbs_input_10kmh"), "INCOMPLETE", text)
        self.assertEqual(item(text, "input_10kmh_0.5"), "INCOMPLETE", text)
        self.assertIn("ended aborted_heading", detail(text, "input_10kmh_0.5"))
        argv = self.session(e21_car=SteerCar(actuator=act()), e22_car=SteerCar(actuator=act()))
        code, text = run_main(e21, argv + ["--e23", e23_play])
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)
        self.assertIn("did not end ok", detail(text, "a_readback"))

    def test_the_model_scope_has_no_gaps(self):
        # Codex: bins at 10 and 60 km/h gave a 0..69 km/h, |cmd| <= 1 scope
        play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=Actuator(tau=0.014)),
                         grid="10:0.5x6,1.0x4;60:0.5x6")

        class Args:
            min_events, min_r2, max_cv = 6, 0.9, 0.2

        state, model, _ = e23.input_model(probe_common.Play(play), Args())
        self.assertTrue(state)

        def row(kmh, cmd):
            return steer_common.Step({"speed_kmh": "%g" % kmh, "cmd_steer": "%g" % cmd})

        for kmh, cmd, inside in ((10, 1.0, True), (60, 0.5, True), (60, 1.0, False), (30, 0.3, False),
                                 (40, 0.0, False), (0, 0.0, False), (70, 0.2, False)):
            self.assertIs(e21.in_scope(row(kmh, cmd), model["scope"]), inside, (kmh, cmd))

    def test_missing_readbacks_with_a_model_are_incomplete(self):
        # Codex: the missing-readback case with a validated model raised a TypeError
        def act():
            return Actuator(tau=0.014)

        argv = self.session(e21_car=SteerCar(actuator=act()), e22_car=SteerCar(actuator=act()))
        blank_segment(argv[2], "e22", "brake", "input_post_steer")
        e23_play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=act()))
        code, text = run_main(e21, argv + ["--e23", e23_play])
        self.assertEqual(code, 0, text)
        self.assertEqual(item(text, "a_readback"), "INCOMPLETE", text)

    def test_current_cross_assistance_must_be_full(self):
        # Codex: assist_mode 0 on every current cross row still passed against a FULL baseline
        argv = self.session()
        edit_column(argv[0], "e21", lambda r: r["phase"].startswith("cross_"), "assist_mode", "0")
        code, text = run_main(e21, argv)
        self.assertEqual(item(text, "c_cross"), "INCOMPLETE", text)
        self.assertIn("assistance not FULL", detail(text, "c_cross"))

    def test_exit_codes(self):
        code, text = run_main(e21, self.session(baseline=False))
        self.assertEqual(code, 0, text)                 # ran; the items carry the verdicts
        code, text = run_main(e21, [self.path("nowhere", "play_01")])
        self.assertEqual(code, 1, text)
        code, text = run_main(e21, ["--no-such-option"])
        self.assertEqual(code, 1, text)


class Cc08Tests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_steer_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def path(self, *parts):
        return os.path.join(self.base, *parts)

    def selected(self, text, name):
        m = re.search(r"\[(PASS|FAIL|INCOMPLETE)\] %s: PARTIAL \(raw engine path\): (M\d)" % re.escape(name), text)
        self.assertIsNotNone(m, text)
        return m.group(1), m.group(2)

    def test_steps_select_the_true_structure_but_cc08_stays_incomplete(self):
        cases = (("M1", Actuator(delay=2)), ("M2", Actuator(tau=0.12)), ("M3", Actuator(rate=2.0)))
        for expected, actuator in cases:
            with self.subTest(expected=expected):
                play = build_e23(self.path("e23_" + expected, "play_01"), SteerCar(actuator=actuator),
                                 grid="10:0.5x6,1.0x4;30:0.5x6")
                code, text = run_main(e23, [play])
                for name in ("input_10kmh_0.5", "input_10kmh_1", "input_30kmh_0.5"):
                    tag, structure = self.selected(text, name)
                    self.assertEqual(structure, expected, text)
                    if expected == "M3" and name == "input_10kmh_1":
                        # 2 /s never reaches 1.0 in the 0.4 s on-time: the gain is not identified, not a pass
                        self.assertEqual(tag, "INCOMPLETE", text)
                        self.assertIn("g unresolved", detail(text, name))
                    else:
                        self.assertEqual(tag, "PASS", text)
                self.assertEqual(item(text, "steer_sign"), "PASS", text)
                self.assertEqual(item(text, "coverage"), "INCOMPLETE", text)
                self.assertIn("PREVIEW", detail(text, "coverage"))
                self.assertEqual(item(text, "llc_path"), "INCOMPLETE", text)
                self.assertEqual(code, 0, text)

    def test_an_unvalidated_channel_without_prbs_is_not_a_pass(self):
        # Codex: an unvalidated wheel channel and no PRBS returned exit 0 and "CC-08 PASS"
        play = build_e23(self.path("e23", "play_01"), prbs=False, grid="10:0.5x6;30:0.5x6")
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "wheel_10kmh_0.5"), "INCOMPLETE", text)
        self.assertEqual(item(text, "prbs_input_10kmh"), "INCOMPLETE", text)
        self.assertEqual(code, 0, text)
        self.assertNotIn("RESULT CC-08: PASS", text)

    def test_a_poorly_predicted_prbs_fails(self):
        # Codex: arbitrarily poor PRBS predictions were informational only
        play = build_e23(self.path("e23", "play_01"), prbs_gain=-0.6, grid="10:0.5x6;30:0.5x6")
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "prbs_input_10kmh"), "FAIL", text)
        # the other spec bins are missing, so the aggregate stays INCOMPLETE with the FAIL kept as a diagnostic
        self.assertIn("PARTIAL INCOMPLETE", detail(text, "raw_engine_path"))
        self.assertIn("observed FAIL", detail(text, "raw_engine_path"))

    def test_a_reduced_grid_is_preview_coverage(self):
        # Codex: a 10:0.5x6 experiment alone returned CC-08 PASS
        play = build_e23(self.path("e23", "play_01"), grid="10:0.5x6")
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "input_10kmh_0.5"), "PASS", text)
        self.assertEqual(item(text, "input_30kmh_0.5"), "INCOMPLETE", text)
        self.assertEqual(item(text, "coverage"), "INCOMPLETE", text)
        self.assertEqual(code, 0, text)

    def test_a_slowing_run_is_binned_by_its_measured_speed(self):
        # Codex: a slowing 60 km/h run stayed in the 60 km/h bin
        play = build_e23(self.path("e23", "play_01"), grid="10:0.5x6;60:0.5x6", slow_kmh={60.0: 30.0})
        code, text = run_main(e23, [play])
        self.assertRegex(detail(text, "coverage"), r"off_speed [1-9]")
        self.assertEqual(item(text, "input_60kmh_0.5"), "INCOMPLETE", text)

    def test_scored_windows_are_disjoint_and_both_signs_are_held_out(self):
        # Codex: the next window included 3 already-scored rows; even / odd split = + trains, - tests
        play = build_e23(self.path("e23", "play_01"), grid="10:0.5x6")
        steps = steer_common.load_steps(probe_common.Play(play), "e23")
        events = probe_common.Play(play).csv("e23_events", e23.EVENT_COLUMNS)
        bins, _ = e23.event_windows(steps, events, "input", None, None, DT)
        windows = bins[(10.0, 0.5)]
        train, test, n_train, n_test = e23.split_doublets(windows)
        scored = [id(r) for w in windows for r in w["rows"]]
        self.assertEqual(len(scored), len(set(scored)))
        train_rows = set(id(r) for w in train for r in w["rows"])
        self.assertFalse(train_rows & set(id(r) for w in test for r in w["rows"]))
        for side in (train, test):
            self.assertEqual(sorted(set(w["amplitude"] > 0 for w in side)), [False, True])
        self.assertEqual((n_train, n_test), (3, 3))

    def test_an_alternating_lag_fails_the_parameter_cv(self):
        # Codex: an alternating-lag case selected M2 at R2 0.995; tau CV about 70% passed (only the gain was gated)
        play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=Actuator(tau=0.05)), grid="10:0.5x8",
                         alt_tau=(0.04, 0.2))
        code, text = run_main(e23, [play, "--min-r2", "0.5"])
        self.assertEqual(item(text, "input_10kmh_0.5"), "FAIL", text)
        self.assertRegex(detail(text, "input_10kmh_0.5"), r"tau CV (0\.[3-9]|[1-9])")

    def test_rate_limited_gain_comes_from_the_full_simulation(self):
        # Codex: an exact rate-limited trace with a non-zero initial state and gain 0.6 fitted g 0.9454, SSE 0.265
        truth = e23.Params(d=1, g=0.6, tau=0.05, q=3.0)
        windows = []
        for k, amp in enumerate((0.5, -0.5, 0.5, -0.5)):
            u = [amp] * 12 + [0.0] * 18
            w = {"u": u, "n": [1] * 30, "y0": 0.08 * (k + 1), "u_e": [0.0] * 8 + u, "offset": 8, "v": 3.0,
                 "step_size": 0.5}
            w["y"] = e23.simulate(w, truth, DT, "M3")
            windows.append(w)
        params, total, _ = e23.fit_structure(windows, DT, "M3")
        self.assertAlmostEqual(params.g, 0.6, delta=0.01)
        self.assertLess(total, 1e-4)

    def test_merged_steps_fit_the_true_tau(self):
        # Codex: with two physics steps per row a true tau 0.10 s fitted as 0.050 s
        play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=Actuator(tau=0.10), merge=2),
                         grid="10:0.5x6")
        code, text = run_main(e23, [play])
        m = re.search(r"input_10kmh_0.5: PARTIAL \(raw engine path\): M2 [^;]* tau ([0-9.]+) s", text)
        self.assertIsNotNone(m, text)
        self.assertAlmostEqual(float(m.group(1)), 0.10, delta=0.01)

    def test_the_validated_wheel_channel_and_its_signal_scale_are_used(self):
        e22_play = build_e22(self.path("e22", "play_01"))
        play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=Actuator(tau=0.05)), grid="10:0.5x6;30:0.5x6")
        code, text = run_main(e23, [play, "--e22", e22_play])
        self.assertIn("channel_validated: yes: rot", text)
        self.assertEqual(item(text, "wheel_30kmh_0.5"), "PASS", text)
        self.assertEqual(item(text, "prbs_wheel_10kmh"), "PASS", text)
        # Codex: a validated steering0 channel could not be fitted (the reader dropped the signal scale)
        bones = ("wa_rot_l", "wa_rot_r", "wa_bone_l", "wa_bone_r", "wa_ent_l", "wa_ent_r")
        blank_columns(e22_play, "e22", bones)
        blank_columns(play, "e23", bones)
        code, text = run_main(e23, [play, "--e22", e22_play, "--signal-scale-deg", "%g" % math.degrees(DMAX)])
        self.assertIn("channel_validated: yes: sig", text)
        self.assertEqual(item(text, "wheel_30kmh_0.5"), "PASS", text)

    def test_wrong_kappa_sign_fails_steer_sign(self):
        play = build_e23(self.path("e23", "play_01"), SteerCar(kappa_sign=-1.0), grid="10:0.5x6;30:0.5x6")
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "steer_sign"), "FAIL", text)

    def test_no_events_never_pass(self):
        play = build_e23(self.path("e23", "play_01"), grid="10:0.5x6")
        with open(os.path.join(play, "e23_events.csv"), "w", encoding="ascii") as handle:
            handle.write(",".join(EVENT_COLUMNS) + "\n")
        code, text = run_main(e23, [play])
        self.assertEqual(code, 0, text)
        self.assertNotIn("[PASS] input_", text)


    def test_a_missing_footer_prints_no_bin_or_prbs_pass(self):
        # Codex: CC-08 printed input-bin and PRBS PASS with its own footer missing
        play = build_e23(self.path("e23", "play_01"), grid="10:0.5x6;30:0.5x6")
        drop_footer(play)
        code, text = run_main(e23, [play])
        self.assertEqual(item(text, "evidence"), "INCOMPLETE", text)
        self.assertNotIn("[PASS] input_", text)
        self.assertNotIn("[PASS] prbs_", text)
        self.assertIn("[observed PASS; withheld: the play is not complete]", text)

    def test_a_stage_present_in_some_repetitions_is_unresolved(self):
        # Codex: relaxation [1, 1, 1, 0] reported CV 0; three active events plus one inactive passed every gate
        state, text = e23.stage_stability("sigma", [1.0, 1.0, 1.0, None], 0.2)
        self.assertIsNone(state)
        self.assertIn("active in 3 of 4", text)
        per = [e23.Params(d=1, g=1.0, tau=0.05, q=3.0, sigma=s) for s in (1.0, 1.0, 1.0, 0.0)]
        windows = [{"release_index": 5, "amplitude": 0.5, "y0": 0.0, "u_e": [0.5] * 20, "offset": 0, "n": [1] * 12,
                    "v": 3.0, "step_size": 0.5} for _ in per]
        out = e23.stability("M4", per[0], per, windows, DT, 0.5, 0.2)
        self.assertIsNone(out["sigma"][0])
        rates = e23.stability("M3", per[0], [e23.Params(d=1, g=1.0, tau=0.05, q=q) for q in (3.0, 3.0, 3.0, None)],
                              windows, DT, 0.5, 0.2)
        self.assertIsNone(rates["R"][0])


    def test_the_default_grid_stays_under_the_heading_guard(self):
        # Codex round 3: 0.4 s holds reached 43 deg at 30 km/h x 1.0 and 40 deg at 60 km/h x 0.5 (guard 30 deg)
        grid = re.search(r'\[Attribute\("([^"]+)", UIWidgets\.EditBox, "Step grid', STEPS_TEXT).group(1)
        amps = re.search(r'\[Attribute\("([^"]+)", UIWidgets\.EditBox, "Held-out PRBS amplitude', STEPS_TEXT).group(1)
        self.assertEqual({float(k): float(v) for k, v in (g.split(":") for g in amps.split(";"))}, PRBS_AMPLITUDES)
        play = build_e23(self.path("e23", "play_01"), SteerCar(actuator=Actuator(tau=0.014)), grid=grid)
        steps = steer_common.load_steps(probe_common.Play(play), "e23")
        worst = {}
        for s in steps:
            if s.segment in ("train", "prbs"):
                off = abs(steer_common.wrap_deg(s.f["heading_deg"] - 75.0))
                key = (s.phase, s.segment)
                worst[key] = max(worst.get(key, 0.0), off)
        print("max heading off the strip per phase and segment:", {k: round(v, 1) for k, v in sorted(worst.items())})
        self.assertLess(max(worst.values()), 25.0, worst)
        code, text = run_main(e23, [play])
        for name in ("input_10kmh_0.5", "input_10kmh_1", "input_30kmh_0.5", "input_30kmh_1", "input_60kmh_0.5"):
            self.assertEqual(item(text, name), "PASS", text)
        for speed in (10, 30, 60):
            self.assertEqual(item(text, "prbs_input_%dkmh" % speed), "PASS", text)
        self.assertIn("input_20kmh_0.5: outside the CC-08 grid", text)


class ContractTests(unittest.TestCase):
    def test_delta_eff_inverts_ackermann(self):
        for delta in (-0.4, -0.1, -0.01, 0.0, 0.003, 0.05, 0.2, 0.45):
            left, right = ackermann(delta)
            self.assertAlmostEqual(steer_common.delta_eff(left, right), delta, delta=max(2e-4, abs(delta) * 0.003))

    def test_strict_verdict_and_combine(self):
        self.assertIsNone(steer_common.combine([True, None]))
        # Codex round 2: a FAIL must not hide an INCOMPLETE part
        self.assertIsNone(steer_common.combine([True, False, None]))
        self.assertIn("observed FAIL in 1 part", steer_common.observed([True, False, None]))
        self.assertIs(steer_common.combine([True, False]), False)
        self.assertEqual(steer_common.observed([True, False]), "")
        self.assertIsNone(steer_common.combine([]))
        v = steer_common.StrictVerdict("X")
        v.check("a", False, "")
        v.check("b", None, "")
        self.assertEqual(v.result(), probe_common.INCOMPLETE)

    def test_contour_heading_puts_the_downhill_on_the_side(self):
        # DC_SteerSiteSearch.GenerateStrips: heading = atan2(side gz/g, -side gx/g); right(h) = (cos h, -sin h)
        for gx, gz in ((0.06, 0.0), (0.0, 0.06), (-0.04, 0.05), (0.03, -0.07)):
            g = math.hypot(gx, gz)
            for side in (1, -1):
                h = math.atan2(side * gz / g, -side * gx / g)
                right = (math.cos(h), -math.sin(h))
                self.assertAlmostEqual(side * (gx * right[0] + gz * right[1]), -g, places=9)
                forward = (math.sin(h), math.cos(h))
                self.assertAlmostEqual(gx * forward[0] + gz * forward[1], 0.0, places=9)
        self.assertIn("Math.Atan2(side * gz / g, -side * gx / g)", read(SITE_SOURCE))

    def test_header_keys_and_files_come_from_the_enforce_producer(self):
        sources = "\n".join(read(p) for p in (BASE_SOURCE, SITE_SOURCE, CHANNEL_SOURCE, SELF_SOURCE, LOAD_SOURCE,
                                              STEPS_SOURCE))
        for key in ("steer.wheelbase_m", "steer.wheel_index_l", "steer.wheel_count", "assist.original",
                    "assist.original_name", "assist.sets", "assist.restores", "assist.at_stop", "steer.sign_used",
                    "site.cross.fallback", "vehicle.noise_steer_sensitivity"):
            self.assertIn(key.split(".")[-1] + "=", sources, key)
        for name in (".found=", ".cross_min_signed=", ".cross_max_signed=", ".cross_mean=", ".heading_deg=",
                     ".grade=", ".benign_hits=", ".start=", "first_terrain_reject=", "first_obstacle="):
            self.assertIn(name, sources, name)
        runner = read(os.path.join(SOURCES, "DC_ProbeRunner.c")) + read(os.path.join(SOURCES, "DC_LaunchGuard.c"))
        for key in steer_common.IDENTITY_KEYS:
            self.assertIn('"%s="' % key, runner, key)
        self.assertIn('"dt_s="', sources)
        for csv_name in ('probeId + "_steps"', 'probeId + "_phases"', 'PROBE_E23 + "_events"'):
            self.assertIn(csv_name, sources)
        for column in steer_common.STEP_COLUMNS:
            self.assertIn(column, STEP_COLUMNS, column)
        for column in e23.EVENT_COLUMNS:
            self.assertIn(column, EVENT_COLUMNS, column)
        for segment in ("hold", "measure", "nocall", "stair", "ref", "alat", "brake", "train", "prbs"):
            self.assertIn('return "%s";' % segment, sources, segment)
        # the phase names the analysis requires come from the producer, with the explicit assistance modes
        selfsteer = read(SELF_SOURCE)
        self.assertIn('"flat_" + flatKmh.ToString(-1, 0), SITE_FLAT, flatKmh, ASSIST_FULL', selfsteer)
        self.assertIn('"flat_" + assistKmh.ToString(-1, 0) + "_none", SITE_FLAT, assistKmh, ASSIST_NONE', selfsteer)
        self.assertIn('"cross_" + crossKmh.ToString(-1, 0), SITE_CROSS, crossKmh, ASSIST_FULL', selfsteer)
        self.assertIn("static const int ASSIST_FULL = 2;", selfsteer)
        self.assertIn("static const int ASSIST_NONE = 0;", selfsteer)

    def test_manifest_and_layer_carry_the_probes(self):
        manifest = read(os.path.join(REPO, "addons", "DataCapture", "deploy_manifest.txt"))
        for name in ("DC_SteerChannels.c", "DC_SteerSites.c", "DC_ProbeSteerBase.c", "DC_ProbeSelfSteer.c",
                     "DC_ProbeSteerLoad.c", "DC_ProbeSteerSteps.c"):
            self.assertIn("script src/enfusion/datacapture/" + name, manifest)
        layer = read(os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Probes_Arland_Layers",
                                  "default.layer"))
        self.assertIn('m_sPlan "e10,e04,e20,e19,e08a,e01,e02a,e21,e22,e23,e05,e06a"', layer)
        for cls in ("DC_ProbeSelfSteer", "DC_ProbeSteerLoad", "DC_ProbeSteerSteps"):
            self.assertRegex(layer, r'  %s "\{DC4A50B30000000[789]\}" \{' % cls)

    def test_no_1_9_only_member_in_the_new_scripts(self):
        # the overlay convention: GetHandbrake, EngineGetRPMRedline, Update*Configuration and
        # VehicleWheelEffectComponent stay out of every script that reaches a 1.8.0.13 target
        for path in (BASE_SOURCE, SITE_SOURCE, CHANNEL_SOURCE, SELF_SOURCE, LOAD_SOURCE, STEPS_SOURCE):
            code = "\n".join(line.split("//", 1)[0] for line in read(path).split("\n"))
            for member in ("GetHandbrake(", "EngineGetRPMRedline(", "Configuration(", "VehicleWheelEffectComponent",
                           "TraceDisplacement("):
                self.assertNotIn(member, code, "%s in %s" % (member, path))

    def test_site_safety_rules_in_the_source(self):
        sites = read(SITE_SOURCE)
        base = read(BASE_SOURCE)
        code = "\n".join(line.split("//", 1)[0] for line in (sites + base).split("\n"))
        # Codex: a trusted explicit candidate turned a blocking hit into a warning
        self.assertNotIn("Trust", code)
        self.assertIn("IsBenign(", code)
        self.assertRegex(base, r'\[Attribute\("", UIWidgets\.EditBox, "Obstacle sweeps: [^"]*"[^\]]*\]\s*'
                               r'protected string m_sBenignColliders;')
        # vehicle-sized volume sweeps with an initial-overlap check and the stopping envelope
        for needle in ("new TraceOBB()", "TracePosition(", "TraceMove(", "c.m_fEnvelopeLongM = stopping;"):
            self.assertIn(needle, code)
        # one budget per frame shared by every search (Codex round 2: each search got the whole budget)
        steps = base[base.index("protected void StepSearches()"):]
        steps = steps[:steps.index("OnSearchesDone()")]
        self.assertIn("m_Budget.Reset(m_iSiteSamplesPerFrame, m_iSiteTracesPerFrame);", steps)
        self.assertIn("search.StepSearch(world, m_Budget)", steps)
        self.assertLess(steps.index("m_Budget.Reset("), steps.index("foreach (DC_SteerSiteSearch search"))

    def test_every_native_call_of_the_search_is_charged_first(self):
        # Codex round 2: one charged sweep could make TracePosition, TraceMove and 8 retries
        sites = read(SITE_SOURCE)
        body = sites[sites.index("class DC_SteerSiteSearch"):sites.index("class DC_SteerTerrainMath")]
        code = "\n".join(line.split("//", 1)[0] for line in body.split("\n"))
        methods = {}
        for m in re.finditer(r"\n\t(?:protected )?\w+ (\w+)\(([^)]*)\)\n\t\{(.*?)\n\t\}", code, re.S):
            methods[m.group(1)] = m.group(3)
        natives = {name: (len(re.findall(r"\.GetSurfaceY\(", text)), len(re.findall(r"\.Trace(?:Position|Move)\(", text)))
                   for name, text in methods.items()}
        self.assertEqual({n: c for n, c in natives.items() if c != (0, 0)},
                         {"GenerateStrips": (4, 0), "ContinueSampling": (1, 0), "SweepEnds": (2, 0),
                          "TraceOnce": (0, 2)})
        once = methods["TraceOnce"]                                        # one call: an if / else of the two
        self.assertRegex(once, r"if \(m_Plan\.GetCall\(\) == DC_SteerSweepPlan\.CALL_POSITION\)\s*\{\s*blocked = world\."
                               r"TracePosition\(m_Box, null\) < 0;\s*\}\s*else\s*\{\s*fraction = world\.TraceMove")
        self.assertRegex(methods["StepGrid"], r"if \(!budget\.TakeSamples\(4\)\)\s*return;\s*(?:.*\n)*?.*GenerateStrips\(")
        self.assertRegex(methods["ContinueSampling"], r"if \(!budget\.TakeSamples\(1\)\)\s*return false;(?:.*\n)*?.*"
                                                      r"GetSurfaceY")
        self.assertRegex(methods["StepObstacles"], r"if \(!budget\.TakeSamples\(2\)\)\s*return;\s*SweepEnds\(")
        self.assertRegex(methods["StepObstacles"], r"if \(!budget\.TakeTrace\(\)\)\s*return;\s*m_Plan\.Report\("
                                                   r"TraceOnce\(")
        for name in ("StepGrid", "ContinueSampling", "StepObstacles", "StepStages"):
            self.assertNotIn("m_iSiteSamplesPerFrame", methods.get(name, ""))

    def test_undersized_budgets_are_refused(self):
        # Codex round 3: budgets below a unit's atomic cost (4 samples, 1 trace) made no progress
        base = read(BASE_SOURCE)
        refusal = base[base.index("if (m_iSiteSamplesPerFrame < 4 || m_iSiteTracesPerFrame < 1)"):]
        self.assertTrue(refusal.split("\n", 1)[1].strip().startswith("return RefuseStart("))
        start = base.index("protected bool StartSteerProbe") if "protected bool StartSteerProbe" in base else 0
        self.assertLess(base.index("m_iSiteTracesPerFrame < 1", start), base.index("AddSearch(SITE_FLAT", start))

    def test_the_sweep_plan_and_budget_with_a_mock_tracer(self):
        # the translated DC_SteerSweepPlan / DC_SteerBudget, driven the way StepObstacles drives them: at most the
        # frame's traces per frame, a benign retry continues in the next frame, a block ends the plan
        classes = enforce_subset.translate(read(SITE_SOURCE), ["DC_SteerBudget", "DC_SteerSweepPlan"])
        Budget, Plan = classes["DC_SteerBudget"], classes["DC_SteerSweepPlan"]
        K = Plan()                                                         # the static constants

        def run(results, lines, segments, traces_per_frame):
            plan, budget = Plan(), Budget()
            plan.Reset(lines, segments)
            calls, frames, log = [], 0, []
            while not plan.IsDone() and frames < 100:
                frames += 1
                budget.Reset(1000, traces_per_frame)
                made = 0
                while not plan.IsDone():
                    if not budget.TakeTrace():
                        break
                    call = (plan.GetLine(), plan.GetSegment(), plan.GetCall())
                    result = results.get(len(calls), K.RESULT_CLEAR)
                    calls.append(call)
                    plan.Report(result)
                    made += 1
                log.append(made)
                self.assertLessEqual(budget.m_iTracesUsed, traces_per_frame)
            return plan, calls, log

        plan, calls, log = run({1: K.RESULT_BENIGN, 2: K.RESULT_BENIGN}, 2, 2, 3)
        self.assertTrue(plan.IsClear())
        p, m = K.CALL_POSITION, K.CALL_MOVE
        self.assertEqual(calls, [(0, 0, p), (0, 0, m), (0, 0, m), (0, 0, m), (0, 1, m), (1, 0, p), (1, 0, m),
                                 (1, 1, m)])
        self.assertEqual(log, [3, 3, 2])
        self.assertEqual(plan.m_iCalls, 8)
        plan, calls, _ = run({4: K.RESULT_BLOCKED}, 3, 3, 2)
        self.assertTrue(plan.IsDone())
        self.assertFalse(plan.IsClear())
        self.assertEqual(len(calls), 5)
        plan, calls, _ = run({k: K.RESULT_BENIGN for k in range(1, 20)}, 1, 1, 4)
        self.assertTrue(plan.m_bRetryCap)
        self.assertEqual(len(calls), 1 + K.MAX_BENIGN_RETRIES + 1)
        budget = Budget()
        budget.Reset(5, 1)
        self.assertTrue(budget.TakeSamples(4))
        self.assertFalse(budget.TakeSamples(2))
        self.assertTrue(budget.TakeSamples(1))
        self.assertEqual(budget.m_iSamplesUsed, 5)

    def test_probe_rules_in_the_source(self):
        base = read(BASE_SOURCE)
        steps = read(STEPS_SOURCE)
        load = read(LOAD_SOURCE)
        channels = read(CHANNEL_SOURCE)
        # Codex: the axle choice latched before the zero-steer settle; rejected axes stayed unusable
        settle = base[base.index("case DC_ESteerSegment.SETTLE:"):]
        settle = settle[:settle.index("EnterSegment(DC_ESteerSegment.ACCEL)")]
        for needle in ("m_Channels.ArmAxleSelection();", "m_Channels.DisarmAxleSelection();", "GetPostSteer()"):
            self.assertIn(needle, settle)
        self.assertIn("AXLE_MAX_TRIES", channels)
        # Codex: unbounded doublet counts reached BuildEvents before any check
        plan = steps[steps.index("override bool BuildSteerPlan"):]
        self.assertLess(plan.index("MAX_EVENTS"), plan.index("BuildEvents("))
        self.assertIn("!InRange(count, 1, MAX_DOUBLETS)", steps)
        self.assertIn("m_aALatTargets.IsEmpty()", load)
        # Codex: load counters counted callbacks, not engine steps
        self.assertIn("m_iLevelSteps += covered;", load)
        self.assertIn("m_fBrakeIntegral += error * m_fDt * m_iLastCovered;", load)
        # Codex round 2: e22's controller allowed 90 km/h against a 35 km/h envelope; heading assumed, not enforced
        self.assertNotIn("ALAT_MAX_KMH", load)
        self.assertEqual(load.count("Math.Clamp(m_fLevelKmh"), 2)
        self.assertEqual(len(re.findall(r"Math\.Clamp\(m_fLevelKmh[^;]*SPEED_CAP_KMH\);", load)), 2)
        self.assertIn("return SPEED_CAP_KMH;", load)
        self.assertRegex(load, r"override bool AllowsAnyHeading\(\)\s*\{\s*return true;")
        guard = base[base.index('EndRun(phase, "aborted_area");'):]
        guard = guard[:guard.index('EndRun(phase, "aborted_heading");')]
        self.assertIn("speedNow > GetMaxSpeedKmh() + OVERSPEED_KMH", guard)
        self.assertIn('EndRun(phase, "overspeed");', guard)
        self.assertIn("!AllowsAnyHeading() && headingOff > GetEnvelopeAngleDeg()", guard)
        self.assertIn("c.m_fEnvelopeBackM = stopping;", base)
        self.assertIn("c.m_fEnvelopeLatM = stopping * Math.Sin(GetEnvelopeAngleDeg() * Math.DEG2RAD);", base)
        # the steering probes stay within 15 min of Play together
        timeouts = [float(re.search(r"TIMEOUT_E2%d_S = (\d+)" % k, read(path)).group(1))
                    for k, path in ((1, SELF_SOURCE), (2, LOAD_SOURCE), (3, STEPS_SOURCE))]
        self.assertLessEqual(sum(timeouts), 900)


# ------------------------------------------------------------------------------------------------------------------
# DC_SteerTerrainMath, translated from the Enforce source


class CriteriaStub:
    def __init__(self, kind, half_width=22.0, corridor_half=19.0, corridor_points=11, **values):
        self.m_sKind = kind
        self.m_fStepM = enforce_subset.f32(5.0)
        self.m_fGradeWindowM = enforce_subset.f32(20.0)
        self.m_fHalfWidthM = enforce_subset.f32(half_width)
        self.m_fCorridorHalfM = enforce_subset.f32(corridor_half)
        self.m_iCorridorPoints = corridor_points
        self.m_fTrackHalfM = enforce_subset.f32(0.65)
        defaults = {"m_fGradeMax": 0.005, "m_fGradeWindowMax": 0.01, "m_fGradeWindowMaxAll": 0.02,
                    "m_fCrossMin": 0.0, "m_fCrossMax": 0.01, "m_fCrossAbsMaxAll": 0.02, "m_fCrossTarget": 0.0,
                    "m_fRmsMax": 0.1, "m_fMinSurfaceY": 3.0}
        defaults.update(values)
        for key, value in defaults.items():
            setattr(self, key, enforce_subset.f32(value))

    def IsCross(self):
        return self.m_sKind == "cross"

    def offsets(self):
        """DC_SteerSiteCriteria.OffsetAt for every sample index (0 / 1 the edges, 2 the centre, then the tracks)"""
        out = [-float(self.m_fHalfWidthM), float(self.m_fHalfWidthM), 0.0]
        points = max(1, self.m_iCorridorPoints)
        for k in range(points):
            position = 0.0
            if points > 1:
                position = -float(self.m_fCorridorHalfM) + k * (2 * float(self.m_fCorridorHalfM) / (points - 1))
            out += [position - float(self.m_fTrackHalfM), position + float(self.m_fTrackHalfM)]
        return out


class SiteStub:
    def __init__(self, length, half_width, zone_from, zone_to, side=1):
        self.m_fLengthM = enforce_subset.f32(length)
        self.m_fHalfWidthM = enforce_subset.f32(half_width)
        self.m_fZoneFromM = enforce_subset.f32(zone_from)
        self.m_fZoneToM = enforce_subset.f32(zone_to)
        self.m_iSide = side
        for name in ("m_fGrade", "m_fGradeWindowMax", "m_fGradeWindowMaxAll", "m_fCrossMean", "m_fCrossMin",
                     "m_fCrossMax", "m_fCrossAbsMaxAll", "m_fWideCrossAbsMax", "m_fRmsM", "m_fScore", "m_fMinY",
                     "m_fMaxY"):
            setattr(self, name, enforce_subset.f32(0.0))
        self.m_bTerrainOk = False
        self.m_sTerrainReason = ""
        self.m_aTrackProfile = enforce_subset.EArray([], "float")


def translated_terrain_math():
    source = read(SITE_SOURCE)
    fields = {"DC_SteerSite": enforce_subset.class_field_types(source, "DC_SteerSite"),
              "DC_SteerSiteCriteria": enforce_subset.class_field_types(source, "DC_SteerSiteCriteria")}
    classes = enforce_subset.translate(source, ["DC_SteerTerrainMath"],
                                       externals={"DC_SteerSite": SiteStub, "DC_SteerSiteCriteria": CriteriaStub},
                                       external_fields=fields)
    return classes["DC_SteerTerrainMath"]()


def corridor_points(corridor_half, track_half=0.65):
    """DC_ProbeSteerBase.SetSiteGeometry: an odd count, the centre included, at most 2 x track_half apart"""
    return 2 * int(math.ceil(corridor_half / (2.0 * track_half))) + 1


class TerrainMathTests(unittest.TestCase):
    def setUp(self):
        self.math = translated_terrain_math()

    def apply(self, site, criteria, height_fn):
        lateral = criteria.offsets()
        n = int(float(site.m_fLengthM) // 5.0) + 1
        along = [i * 5.0 for i in range(n)]
        heights = [height_fn(a, lat) for a in along for lat in lateral]

        def arr(values):
            return enforce_subset.EArray([enforce_subset.f32(v) for v in values], "float")

        self.math.Apply(site, criteria, arr(along), arr(lateral), arr(heights))
        return site

    def cross_criteria(self, points=None, **kw):
        values = dict(m_fGradeMax=0.02, m_fGradeWindowMax=0.035, m_fGradeWindowMaxAll=0.06, m_fCrossMin=0.05,
                      m_fCrossMax=0.12, m_fCrossAbsMaxAll=0.15, m_fCrossTarget=0.065, m_fRmsMax=0.5)
        values.update(kw)
        return CriteriaStub("cross", half_width=8.0, corridor_half=5.0,
                            corridor_points=points or corridor_points(5.0), **values)

    def test_the_corridor_rule_samples_the_centre_pair_without_gaps(self):
        self.assertIn("int half = Math.Ceil(c.m_fCorridorHalfM / (2 * c.m_fTrackHalfM));", read(BASE_SOURCE))
        self.assertIn("c.m_iCorridorPoints = 2 * half + 1;", read(BASE_SOURCE))
        for corridor in (0.0, 1.0, 5.0, 7.0, 12.0, 19.0):
            crit = CriteriaStub("cross", half_width=corridor + 3, corridor_half=corridor,
                                corridor_points=corridor_points(corridor))
            tracks = sorted(crit.offsets()[3:])
            self.assertIn(-0.65, [round(t, 6) for t in tracks])
            self.assertIn(0.65, [round(t, 6) for t in tracks])
            grid = [-corridor + 0.01 * k for k in range(int(200 * corridor) + 1)]
            self.assertLessEqual(max(min(abs(g - t) for t in tracks) for g in grid), 0.65 + 1e-6)

    def test_a_two_metre_flat_lane_between_shoulders_is_not_a_cross_site(self):
        # Codex round 2: the default positions (-5, -1.667, 1.667, 5 m) missed the centre tyres; a 2 m flat lane
        # between 6% shoulders passed as a 6% cross-slope
        def lane(a, lat):
            return 50.0 - 0.06 * math.copysign(max(0.0, abs(lat) - 1.0), lat) + 0.004 * a

        old = self.apply(SiteStub(310, 8, 75, 275, side=1), self.cross_criteria(points=4), lane)
        self.assertTrue(old.m_bTerrainOk)            # what the round-1 layout accepted
        site = self.apply(SiteStub(310, 8, 75, 275, side=1), self.cross_criteria(), lane)
        self.assertFalse(site.m_bTerrainOk)
        self.assertEqual(site.m_sTerrainReason, "cross_zone")
        self.assertLess(float(site.m_fCrossMin), 0.01)
        self.assertEqual(len(site.m_aTrackProfile), 2 * 63)

    def test_flat_runway_passes(self):
        site = self.apply(SiteStub(1000, 22, 0, 1000), CriteriaStub("flat"), lambda a, lat: 39.0 + 0.00002 * a)
        self.assertTrue(site.m_bTerrainOk, site.m_sTerrainReason)
        self.assertAlmostEqual(float(site.m_fGrade), 0.00002, places=6)
        self.assertLess(float(site.m_fRmsM), 0.01)

    def test_cross_slope_on_the_right_passes_and_the_left_does_not(self):
        def slope(a, lat):          # right side (lateral +) lower by 6%
            return 50.0 - 0.06 * lat + 0.005 * a

        site = self.apply(SiteStub(310, 8, 75, 275, side=1), self.cross_criteria(), slope)
        self.assertTrue(site.m_bTerrainOk, site.m_sTerrainReason)
        self.assertAlmostEqual(float(site.m_fCrossMean), 0.06, places=4)
        self.assertAlmostEqual(float(site.m_fCrossMin), 0.06, places=4)
        self.assertAlmostEqual(float(site.m_fGrade), 0.005, places=5)
        wrong = self.apply(SiteStub(310, 8, 75, 275, side=-1), self.cross_criteria(), slope)
        self.assertFalse(wrong.m_bTerrainOk)
        self.assertEqual(wrong.m_sTerrainReason, "cross_zone")

    def test_a_flat_road_between_sloping_shoulders_is_not_a_cross_site(self):
        # Codex: a 6 m flat road with sloping shoulders passed as a 6% cross-slope (three lateral samples)
        def road(a, lat):
            excess = max(0.0, abs(lat) - 3.0)
            return 50.0 - 0.12 * excess * (1 if lat > 0 else -1) + 0.004 * a

        site = self.apply(SiteStub(310, 8, 75, 275, side=1), self.cross_criteria(), road)
        self.assertFalse(site.m_bTerrainOk)
        self.assertIn(site.m_sTerrainReason, ("cross_zone", "cross_all", "rms"))
        self.assertLess(float(site.m_fCrossMin), 0.05)

    def test_a_sloping_shoulder_inside_the_flat_corridor_is_refused(self):
        def road(a, lat):
            return 39.0 - 0.05 * max(0.0, lat - 10.0)

        site = self.apply(SiteStub(1000, 22, 0, 1000), CriteriaStub("flat"), road)
        self.assertFalse(site.m_bTerrainOk)
        self.assertIn(site.m_sTerrainReason, ("cross_all", "cross_zone"))

    def test_bumpy_short_and_low_strips_are_refused(self):
        lax = CriteriaStub("flat", m_fGradeWindowMax=1, m_fGradeWindowMaxAll=1, m_fCrossMax=1, m_fCrossAbsMaxAll=1)
        bumpy = self.apply(SiteStub(1000, 22, 0, 1000), lax, lambda a, lat: 39.0 + 0.5 * math.sin(a / 7.0))
        self.assertFalse(bumpy.m_bTerrainOk)
        self.assertIn(bumpy.m_sTerrainReason, ("rms", "grade"))
        short = self.apply(SiteStub(8, 22, 0, 8), CriteriaStub("flat"), lambda a, lat: 39.0)
        self.assertFalse(short.m_bTerrainOk)
        self.assertEqual(short.m_sTerrainReason, "too_short")
        low = self.apply(SiteStub(1000, 22, 0, 1000), CriteriaStub("flat"), lambda a, lat: 1.0)
        self.assertEqual(low.m_sTerrainReason, "below_min_surface_y")

    def test_track_cross_matches_a_python_reference(self):
        rng = random.Random(9)
        crit = self.cross_criteria(m_fGradeMax=0.05, m_fGradeWindowMax=0.1, m_fGradeWindowMaxAll=0.1,
                                   m_fCrossMin=0.0, m_fCrossMax=0.2, m_fCrossAbsMaxAll=0.3, m_fRmsMax=5.0)
        coeffs = [rng.uniform(-0.02, 0.02) for _ in range(3)]

        def h(a, lat):
            return 60.0 + coeffs[0] * a - (0.07 + coeffs[1]) * lat - 0.004 * lat * lat + \
                0.3 * math.sin(a / 40.0 + coeffs[2])

        site = self.apply(SiteStub(300, 8, 60, 260), crit, h)
        lateral = crit.offsets()
        along = [i * 5.0 for i in range(61)]
        zone = []
        for a in along:
            if 60 <= a <= 260:
                for k in range(crit.m_iCorridorPoints):
                    left, right = lateral[3 + 2 * k], lateral[4 + 2 * k]
                    zone.append((h(a, left) - h(a, right)) / (right - left))
        centre = [h(a, 0.0) for a in along]
        ma = sum(along) / len(along)
        my = sum(centre) / len(centre)
        grade = sum((a - ma) * (c - my) for a, c in zip(along, centre)) / sum((a - ma) ** 2 for a in along)
        self.assertAlmostEqual(float(site.m_fGrade), grade, places=4)
        self.assertAlmostEqual(float(site.m_fCrossMean), sum(zone) / len(zone), places=4)
        self.assertAlmostEqual(float(site.m_fCrossMin), min(zone), places=4)
        self.assertAlmostEqual(float(site.m_fCrossMax), max(zone), places=4)


if __name__ == "__main__":
    unittest.main()
