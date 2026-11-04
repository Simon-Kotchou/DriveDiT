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
  - the engine input path: a rate limit (1 /s, 18 /s) is input_shaping rate_limit with its limit, a 0.08 s lag is
    lag with its tau and no limit (review F-05); the limiter rule needs a known path (F-04);
  - a scripted gearbox stores E-01's clutch dip and is refused without it (F-02); a neutral index reported while
    rolling stays in gear_indices_forward (TBD-7) but can never be a scripted gear (Codex CX-1);
  - a manual-gearbox E-01 without gear_observe (probe kit 699d8e2 skips it, note manual_gearbox; the car driven by
    the probe's shift policy, neutral with the handbrake) gives a scripted card in both run header styles: the probe
    kit's (vehicle.gearbox=manual, vehicle.gear_index_neutral / _first / _top) and the earlier one
    (vehicle.has_automatic_gearbox=0 and the gear counts, indices by the vanilla rule); the default scripted gears
    run from the first gear upward while E-01 shows propulsion, the speed per 1000 rpm of each gear is measured, and
    a play that disagrees with itself or with E-04 is refused; the phase note alone says manual but gives no index;
  - an identified card needs one pin, one build_key (--build-key fills a dd-launch/1 header), one fixed time slice
    (F-11), complete plays and a usable E-02a fit with gains >= MIN_KAPPA_PER_STEER, and is checked as written
    against llc_v1's load rules (F-12); Everon inputs are refused;
  - the committed card tools/ident/cards/1.8.0.13/S105.json is exactly the generator's PLACEHOLDER (status
    placeholder, no identification session, build_key null, signs.measured false, the TBD-5 signs dc-2.0 decided:
    steer_sign +1) with kappa_ss and the scripted gears 2, 3, 4 and their shift speeds of the provisional run-3
    first-pass fit (RUN3_*; compared with the fit's own card when DD_RUN3_CARD names it), an unidentified actuator,
    and the S105's gear indices of T0-ENG probe run 2 (E-04: 6 gears, 4 forward -> forward 2..5, neutral 1, reverse 0
    by the vanilla rule); llc_v1 still refuses it at production (AcceptCard, mirrored);
  - llc_v1's scripted-gear policy (a Python mirror of DC_LLCv1 Reset / ScriptedGear / the clutch dip, tied to the
    Enforce source) drives the placeholder's 3 scripted gears at the frozen dc-2.0 step dt (1/30 s, callbacks of 1 to
    3 steps, D-A): first gear at the start, one up and one down shift per shift speed with hysteresis, never neutral
    or reverse, engine speeds within the probe policy's range by E-01 run 3's measured speed per 1000 rpm, the clutch
    dip after every change, counted in covered steps;
    the gear-index rules (reverse below neutral, increasing scripted gears above neutral) and
    E-04's indices in an identified card; llc_v1 refuses an automatic card on the manual ego (CheckEgoGearbox);
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

DT = 1.0 / 60.0             # the synthetic probe plays' fixed time slice (their headers carry it; the fit reads it)
STEP_DT = 1.0 / 30.0        # the engine step dt of a dc-2.0 capture session on 1.8 (D-A), llc_v1's step time
STEP_COLUMNS = actuation_tests.enforce_step_columns()
PHASE_COLUMNS = actuation_tests.PHASE_COLUMNS
CELL_COLUMNS = actuation_tests.CELL_COLUMNS
CARD = os.path.join(REPO, "tools", "ident", "cards", "1.8.0.13", "S105.json")
ROUTE = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes", "arland_t0_loop.json")
LLC_SOURCE = os.path.join(REPO, "src", "enfusion", "datacapture", "DC_LLCv1.c")
ACTUATOR_SOURCE = os.path.join(REPO, "src", "enfusion", "datacapture", "DC_StreamRecorder.c")
LAYER = os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Arland_Layers", "default.layer")
LLC_KEYS = [name for name, _ in fit.LLC_DEFAULTS]
LAUNCH_ID = "20261023T101500Z-wb-00a1"


def gain(v):
    """True curvature per contract steer [1/m] at speed v [m/s]."""
    return 0.22 / (1.0 + (v / 10.0) ** 2)


# A manual gearbox as the S105's (vanilla indices: reverse 0, neutral 1, forward 2..5) with a speed per 1000 rpm per
# gear, driven by the probe kit's shift policy where no explicit gear is set (699d8e2: neutral while the handbrake is
# commanded, then a gear by speed; cmd_gear = the policy gear): 2 below 15 km/h, 3 below 30, 4 above
MANUAL_SPEED_PER_KRPM = {2: 1.5, 3: 2.7, 4: 4.1, 5: 6.0}
KIT_GEARBOX_LINES = {"vehicle.has_automatic_gearbox": "0", "vehicle.gearbox_gears": "6",
                     "vehicle.gearbox_forward_gears": "4", "vehicle.gearbox": "manual",
                     "vehicle.gear_index_neutral": "1", "vehicle.gear_index_first": "2", "vehicle.gear_index_top": "5",
                     "vehicle.gear_index_rule": "neutral gears-forward-1, first gears-forward, top gears-1",
                     "gear_policy.active": "1", "gear_policy.clutch_dip_s": "0.000"}
EARLIER_GEARBOX_LINES = {"vehicle.has_automatic_gearbox": "0", "vehicle.gearbox_gears": "6",
                         "vehicle.gearbox_forward_gears": "4"}
# The producer lines of what the fit reads, pinned from src/enfusion/datacapture/DC_ProbeSequences.c of probe kit
# 699d8e2 (feat/t0eng-probe-kit; leading whitespace removed): DC_ProbeActuator.AppendGearboxLines :533-540,
# DC_ProbeSequencePhase.AddNote :1245, AppendVehicleLines :1613 and :1630-1631, the gear_observe phase :1979,
# BeginPhase's skip :2120, SkipPhase :2578 and BoolText :2824 / :2826
KIT_699D8E2_LINES = (
    'string kind = "manual";',
    'if (m_bAutomaticGearbox)',
    'kind = "automatic";',
    'lines.Insert("vehicle.gearbox=" + kind);',
    'lines.Insert("vehicle.gear_index_neutral=" + m_iGearNeutral.ToString());',
    'lines.Insert("vehicle.gear_index_first=" + m_iGearFirst.ToString());',
    'lines.Insert("vehicle.gear_index_top=" + m_iGearTop.ToString());',
    'm_sNote += "/" + token;',
    'lines.Insert("vehicle.has_automatic_gearbox=" + BoolText(controller.HasAutomaticGearbox()));',
    'lines.Insert("vehicle.gearbox_gears=" + gears.ToString());',
    'lines.Insert("vehicle.gearbox_forward_gears=" + forwardGears.ToString());',
    'AddPhase(DC_EProbeSequenceKind.GEAR_OBSERVE, "gear_observe", -1, m_fGearObserveThrottle, 0);',
    'SkipPhase(phase, "manual_gearbox",',
    'phase.m_sResult = "skipped";',
    'return "1";',
    'return "0";',
)


class Truth:
    """Vehicle with a known actuator: contract steer c = steer_sign x engine steer, curvature kappa follows
    gain_scale x gain(v) x c(t - delay) + offset with a first-order lag tau; a longitudinal dead time; throttle 3 m/s2
    per unit, brake 8 m/s2 per unit; an automatic gearbox that shifts up every 20 km/h (input gear 3, 4, 5, 6), or
    without one gear 3 unless SetGear sets another, or (manual) the S105-like manual gearbox under the probe's shift
    policy with the engine speed of MANUAL_SPEED_PER_KRPM; optionally a neutral index reported while braking above
    5 km/h."""

    def __init__(self, steer_sign=-1, delay=4, tau=0.1, offset=0.001, long_delay=6, input_rate=None, input_tau=None,
                 automatic=True, neutral_when_braking=None, gain_scale=1.0, manual=False):
        self.input_rate = input_rate      # an engine-side steering rate limit on the input [1/s], None = none
        self.input_alpha = None if input_tau is None else 1.0 - math.exp(-DT / input_tau)   # an engine input lag
        self.manual = manual
        self.automatic = automatic and not manual
        self.neutral_when_braking = neutral_when_braking
        self.gain_scale = gain_scale
        self.steer_input = 0.0
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
        if self.input_rate is not None:
            limit = self.input_rate * DT
            self.steer_input += max(-limit, min(limit, steer - self.steer_input))
        elif self.input_alpha is not None:
            self.steer_input += self.input_alpha * (steer - self.steer_input)
        else:
            self.steer_input = steer
        self.steer_hist.append(self.steer_input)
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
        target = self.gain_scale * gain(self.speed) * self.steer_sign * delayed + self.offset
        self.kappa += self.alpha * (target - self.kappa)
        yaw = self.speed * self.kappa if self.speed > 0.3 else 0.0
        self.heading += math.degrees(yaw * DT)
        rad = math.radians(self.heading)
        self.x += self.speed * math.sin(rad) * DT
        self.z += self.speed * math.cos(rad) * DT
        kmh = self.speed * 3.6
        rpm = 900 + 40 * kmh
        if gear >= 0:
            cur = gear
        elif self.manual:
            cur = 1 if handbrake else 2 + min(2, int(kmh / 15.0))
        elif self.neutral_when_braking is not None and brake > 0 and kmh >= 5.0:
            cur = self.neutral_when_braking
        elif self.automatic:
            cur = 3 + min(3, int(kmh / 20.0))
        else:
            cur = 3
        if self.manual and cur in MANUAL_SPEED_PER_KRPM:
            rpm = max(800.0, self.speed / MANUAL_SPEED_PER_KRPM[cur] * 1000.0)
        cmd_gear = gear if gear >= 0 or not self.manual else cur
        self.rows.append({
            "physics_step": self.step, "world_time_ms": "%.3f" % (self.step * DT * 1000), "timeslice_s": "%.6f" % DT,
            "phase": phase, "segment": segment, "phase_step": n, "force_sim": 0, "cmd_steer": "%.4f" % steer,
            "cmd_throttle": "%.4f" % throttle, "cmd_brake": "%.4f" % brake, "cmd_clutch": "1.0000",
            "cmd_gear": cmd_gear,
            "cmd_handbrake": int(handbrake), "speed_hold_kmh": "%.3f" % hold,
            "input_pre_steer": "%.4f" % steer, "input_pre_throttle": "%.4f" % throttle,
            "input_pre_brake": "%.4f" % brake, "input_pre_clutch": "1.0000", "input_pre_gear": cur,
            "input_pre_handbrake": int(handbrake), "input_post_steer": "%.4f" % self.steer_input,
            "input_post_throttle": "%.4f" % throttle, "input_post_brake": "%.4f" % brake,
            "input_post_clutch": "1.0000", "input_post_gear": cur, "input_post_handbrake": int(handbrake),
            "cur_gear": cur, "fut_gear": cur, "rpm": "%.1f" % rpm, "engine_on": 1,
            "speed_kmh": "%.4f" % kmh, "vel_l_x": "0.000", "vel_l_y": "0.000", "vel_l_z": "%.4f" % self.speed,
            "angvel_l_x": "0.000000", "angvel_l_y": "%.7f" % yaw, "angvel_l_z": "0.000000",
            "acc_l_x": "%.4f" % (self.speed * yaw), "acc_l_y": "0.0000", "acc_l_z": "%.4f" % acc,
            "pos_x": "%.4f" % self.x, "pos_y": "40.000", "pos_z": "%.4f" % self.z, "fwd_x": "%.6f" % math.sin(rad),
            "fwd_y": "0.000000", "fwd_z": "%.6f" % math.cos(rad), "heading_deg": "%.6f" % self.heading,
            "ground_contact": 1, "wheels_contact": 4, "in_area": 1})
        return kmh


def header(probe, build_key="bk-test-1", pin="1.8.0.13", map_id="arland", prefab=None, slice_s=DT, clutch_dip=None,
           lines=None):
    out = {"schema": "dc-probe-run/1", "probe": probe, "launch.id": LAUNCH_ID, "launch.pin_id": pin,
           "launch.build_key": build_key, "engine.fixed_time_slice_s": "%.6f" % slice_s, "world.map_id": map_id,
           "ego.prefab": prefab or fit.PREFABS["S105"], "params.dt_s": "%.6f" % slice_s}
    if clutch_dip is not None:
        out["params.m_fGearClutchDipS"] = "%.3f" % clutch_dip      # DC_ProbeSequences.GetProbeParams
    out.update(lines or {})                                         # e.g. the gearbox lines of the run header
    return out


FOOTER = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}


def build_e01(play_dir, truth, gear_sequence=(), observe="run", **head):
    """The E-01 phases the fit reads; gear_sequence adds a gear_scripted phase (SetGear to each gear for 3 s at
    throttle 0.4, as DC_ProbeSequences' GEAR_SCRIPTED). observe: "run" (gear_observe drives 15 s without SetGear),
    "skipped" (the probe kit's row on a manual gearbox: result skipped, note manual_gearbox, no steps) or "absent"
    (no gear_observe row at all)."""
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

    def gear_scripted(name, n):
        for gear in gear_sequence:
            for _ in range(180):
                truth.run(name, "excite", n, throttle=0.4, gear=gear)
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
    if observe == "run":
        phase("gear_observe", "", "0.7000", gear_observe)
    elif observe == "skipped":
        phases.append({"phase_idx": len(phases), "phase": "gear_observe", "target_kmh": "", "amplitude": "0.7000",
                       "force_sim": 0, "result": "skipped", "note": "manual_gearbox"})
    if gear_sequence:
        phase("gear_scripted", "", "0.4000", gear_scripted)
    probe_common.write_synthetic_play(play_dir, header("e01", **head), {
        "e01_steps": (STEP_COLUMNS, truth.rows), "e01_phases": (PHASE_COLUMNS, phases)}, FOOTER)


def build_e02a(play_dir, truth, starve_speed=None, starved=(), **head):
    """Cells at 15/30/50 km/h x 5 levels. A starved cell (every cell from the third on at starve_speed, and the
    (km/h, level index) pairs of starved) never reaches its speed: as DC_ProbeSequences writes it, no excite segment,
    an empty excite_start_step and result no_speed."""
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
            no_speed = (kmh == starve_speed and i >= 2) or (kmh, i) in starved
            for _ in range(90):
                truth.run(name, "accel", n, throttle=0.3, hold=kmh, speed=(kmh - 5 if no_speed else kmh) / 3.6)
                n += 1
            excite = None
            if not no_speed:
                excite = truth.step + 1
                for _ in range(480):
                    truth.run(name, "excite", n, steer=steer, throttle=0.2, hold=kmh, speed=kmh / 3.6)
                    n += 1
            truth.run(name, "stop", n, brake=1.0, speed=0.0)
            cells.append({"cell_idx": idx, "target_kmh": "%.3f" % kmh, "steer": "%.4f" % steer, "start_step": start,
                          "excite_start_step": excite, "end_step": truth.step,
                          "result": "no_speed" if no_speed else "ok", "note": ""})
    probe_common.write_synthetic_play(play_dir, header("e02a", **head), {
        "e02a_steps": (STEP_COLUMNS, truth.rows), "e02a_cells": (CELL_COLUMNS, cells)}, FOOTER)


# The provisional run-3 first-pass fit (T0-ENG probe run 3, launch 20261103T061122Z-wb-86bd: E-01, E-02a, E-04 on
# 1.8.0.13; its identified card waits for the DD-14a build_key) whose kappa_ss and scripted gearbox the placeholder
# copies. DD_RUN3_CARD may name that fit's card (S105.provisional.json) to compare the two directly.
RUN3_KAPPA_SS = {"speeds_mps": [4.254893, 8.350237, 14.07005], "kappa_per_steer": [0.230344, 0.228379, 0.229872],
                 "kappa_offset": [-0.0001, 0.0, 0.0]}
RUN3_GEARBOX = {"mode": "scripted", "has_automatic_gearbox": False, "scripted_gears": [2, 3, 4],
                "upshift_mps": [7.0, 12.5], "downshift_mps": [5.0, 7.5], "clutch_dip_s": 0.0}
RUN3_CARD = os.environ.get("DD_RUN3_CARD", "")
# E-01 run 3's measured speed per 1000 engine rpm in the gears it drove (the fit's gearbox.detail.speed_per_krpm), the
# basis of the shift speeds: the probe policy upshifts at 4500 rpm and downshifts at 1800 rpm in the higher gear
RUN3_SPEED_PER_KRPM = {2: 1.54871, 3: 2.772066, 4: 4.169103}


def engine_rpm(speed, gear):
    return speed / RUN3_SPEED_PER_KRPM[gear] * 1000.0


def accept_card(card, use_class, session_pin_id):
    """Mirror of the first rules of DC_LLCv1.AcceptCard (asserted in test_placeholder_is_refused_at_production): a
    placeholder only at a qualification use_class (dev, probe), and only for the session's pin."""
    if card["status"] == "placeholder" and use_class not in ("dev", "probe"):
        return False
    return card["pin_id"] == session_pin_id


class LlcGears:
    """Python mirror of llc_v1's scripted gearbox (DC_LLCv1.c: Init's dip steps from the measured step dt, Reset's gear
    slot from the speed, ScriptedGear's one-slot hysteresis, Step's clutch dip after each change of the commanded gear,
    counted in the steps each fixed-frame callback covers)."""

    def __init__(self, card, dt):
        gearbox = card["gearbox"]
        self.gears, self.up, self.down = gearbox["scripted_gears"], gearbox["upshift_mps"], gearbox["downshift_mps"]
        self.dip_steps = int(math.floor(gearbox["clutch_dip_s"] / dt + 0.5))     # Math.Round
        self.reset(0.0)

    def reset(self, speed):
        self.slot, self.last, self.dip_left = 0, -1, 0
        while self.slot < len(self.up) and speed >= self.up[self.slot]:
            self.slot += 1

    def step(self, speed, covered=1):
        steps = covered if covered >= 1 else 1
        last = len(self.gears) - 1
        if self.slot < last and speed >= self.up[self.slot]:
            self.slot += 1
        elif self.slot > 0 and speed < self.down[self.slot - 1]:
            self.slot -= 1
        gear = int(round(self.gears[self.slot]))
        if gear != self.last:
            self.dip_left = self.dip_steps
        self.last = gear
        clutch = 1.0
        if self.dip_left > 0:
            clutch = 0.0
            self.dip_left -= steps
            if self.dip_left < 0:
                self.dip_left = 0
        return gear, clutch


def enforce_body(source, cls, method):
    """One method of an Enforce class, whitespace collapsed (the statements the Python mirrors follow)."""
    body = source.split("\nclass %s" % cls, 1)[1]
    found = re.search(r"\n\t(?:(?:override|protected|static) )*\w+ %s\(" % method, body)
    body = body[found.end():].split("\n\t}\n", 1)[0]
    return " ".join(body.split())


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
        self.assertEqual((raw["input_echo"], raw["input_shaping"], raw["steer_rate_limit_per_s"], raw["input_lag_tau_s"]),
                         (True, "echo", None, None))
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
        self.assertEqual(card["llc_v1"]["steer_rate_per_s"], 0.0)
        self.assertEqual(card["gearbox"]["mode"], "automatic")
        self.assertEqual(card["gearbox"]["clutch_dip_s"], 0.0)          # no SetGear, no dip
        self.assertEqual(card["gear_indices_forward"], [3, 4, 5, 6])
        self.assertEqual(card["gearbox"]["detail"]["propulsion_gears"], [3, 4, 5, 6])
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

    def test_engine_rate_limit_is_identified(self):
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_09")
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_09")
        build_e01(e01, Truth(input_rate=1.0))
        build_e02a(e02a, Truth(input_rate=1.0))
        out = os.path.join(self.tmp, "limited.json")
        code, text = run_fit(["--e01", e01, "--e02a", e02a, "--out", out])
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            card = json.load(f)
        raw = card["actuator"]["raw_engine_path"]
        self.assertFalse(raw["input_echo"])
        self.assertEqual(raw["input_shaping"], "rate_limit")
        self.assertAlmostEqual(raw["steer_rate_limit_per_s"], 1.0, delta=0.01)
        self.assertIsNone(raw["input_lag_tau_s"])
        self.assertEqual(card["llc_v1"]["steer_rate_per_s"], 0.0)       # the identity: never stacks (CC-08)

    def test_a_fast_engine_limit_is_not_an_echo(self):
        # 18 /s: a 0.6 command step takes two physics steps, so the input lags on the change step only (RX-1)
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_11")
        build_e01(e01, Truth(input_rate=18.0))
        out = os.path.join(self.tmp, "fast_limit.json")
        code, text = run_fit(["--e01", e01, "--e02a", self.e02a, "--out", out])
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            raw = json.load(f)["actuator"]["raw_engine_path"]
        self.assertFalse(raw["input_echo"])
        self.assertEqual(raw["input_shaping"], "rate_limit")       # the lag model misses the arrival steps
        self.assertAlmostEqual(raw["steer_rate_limit_per_s"], 18.0, delta=0.1)

    def test_an_engine_lag_is_not_a_rate_limit(self):
        # F-05: a 0.08 s first-order lag on the input has an initial slope (6.8 /s here), not a rate limit; an llc
        # limiter would stack with it, so the card identifies no limit and llc_v1 may run only the identity
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_12")
        build_e01(e01, Truth(input_tau=0.08))
        out = os.path.join(self.tmp, "lagged_input.json")
        code, text = run_fit(["--e01", e01, "--e02a", self.e02a, "--out", out])
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            card = json.load(f)
        raw = card["actuator"]["raw_engine_path"]
        self.assertEqual((raw["input_echo"], raw["input_shaping"], raw["steer_rate_limit_per_s"]), (False, "lag", None))
        self.assertAlmostEqual(raw["input_lag_tau_s"], 0.08, delta=0.004)
        self.assertAlmostEqual(raw["input_max_slope_per_s"], 0.6 * (1.0 - math.exp(-DT / 0.08)) / DT, delta=0.05)
        self.assertEqual(fit.enforce_load_check(card), [])
        card["llc_v1"]["steer_rate_per_s"] = 2.0                    # "slower" than the slope, but it would stack
        self.assertTrue(any("steer_rate_per_s" in v for v in fit.enforce_load_check(card)))

    def test_limiter_rule_needs_a_known_engine_path(self):
        # F-04: a non-identity llc limiter only with an echoing input or an identified pure rate limit (and then
        # strictly slower); a placeholder (engine path unknown) keeps the identity
        with open(CARD, "r", encoding="ascii") as f:
            placeholder = json.load(f)
        self.assertEqual(fit.enforce_load_check(placeholder), [])
        placeholder["llc_v1"]["steer_rate_per_s"] = 2.0
        self.assertTrue(any("steer_rate_per_s" in v for v in fit.enforce_load_check(placeholder)))
        card = json.loads(self.card_text)                           # input echo
        card["llc_v1"]["steer_rate_per_s"] = 2.0
        self.assertEqual(fit.enforce_load_check(card), [])
        card["actuator"]["raw_engine_path"].update({"input_echo": False, "input_shaping": "rate_limit",
                                                    "steer_rate_limit_per_s": 4.0})
        self.assertEqual(fit.enforce_load_check(card), [])
        card["llc_v1"]["steer_rate_per_s"] = 4.0
        self.assertTrue(any("not slower" in v for v in fit.enforce_load_check(card)))
        card["actuator"]["raw_engine_path"]["input_shaping"] = "unknown"
        self.assertTrue(fit.enforce_load_check(card))                # a limit without a rate_limit shaping

    def test_scripted_gearbox_carries_the_clutch_dip(self):
        # F-02: a scripted card stores the clutch dip E-01's gear_scripted phase ran with (llc_v1 reproduces it) and is
        # refused when the run header does not say it
        args = ["--e02a", self.e02a, "--scripted-gears", "3,4,5", "--upshift-mps", "5,10", "--downshift-mps", "4,8"]
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_13")
        build_e01(e01, Truth(automatic=False), gear_sequence=(3, 4, 5), clutch_dip=0.25)
        out = os.path.join(self.tmp, "scripted.json")
        code, text = run_fit(["--e01", e01, "--out", out] + args)
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            card = json.load(f)
        gearbox = card["gearbox"]
        self.assertEqual((gearbox["mode"], gearbox["scripted_gears"], gearbox["clutch_dip_s"]), ("scripted", [3, 4, 5], 0.25))
        self.assertEqual(card["gear_indices_forward"], [3, 4, 5])
        self.assertEqual(fit.enforce_load_check(card), [])
        undeclared = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_14")
        build_e01(undeclared, Truth(automatic=False), gear_sequence=(3, 4, 5))
        code, text = run_fit(["--e01", undeclared] + args)
        self.assertEqual(code, 2, text)
        self.assertIn("m_fGearClutchDipS", text)
        card["gearbox"]["mode"] = "automatic"                        # a dip without SetGear is no card llc_v1 loads
        self.assertTrue(fit.enforce_load_check(card))

    def test_neutral_reported_while_rolling_is_no_scripted_gear(self):
        # Codex CX-1 reproduced: a neutral index (1) reported while braking above 5 km/h enters gear_indices_forward
        # (the TBD-7 definition, kept), but it shows no propulsion, so it can never be a scripted gear
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_15")
        build_e01(e01, Truth(neutral_when_braking=1))
        steps = fit.e01_actuation.Steps(probe_common.Play(e01).csv("e01_steps"))
        automatic, forward, propulsion, _ = fit.fit_gearbox(steps)
        self.assertEqual((automatic, forward, propulsion), (True, [1, 3, 4, 5, 6], [3, 4, 5, 6]))
        bad = type("Args", (), {"scripted_gears": "1,3", "upshift_mps": "5", "downshift_mps": "4"})()
        with self.assertRaises(fit.Refusal) as caught:
            fit.scripted_gearbox(bad, forward, propulsion)
        self.assertIn("propulsion", str(caught.exception))
        good = type("Args", (), {"scripted_gears": "3,4", "upshift_mps": "5", "downshift_mps": "4"})()
        self.assertEqual(fit.scripted_gearbox(good, forward, propulsion), ([3, 4], [5.0], [4.0]))

    def test_plays_of_two_time_slices_are_refused(self):
        # F-11: the PRBS bin, the E-02a step responses and the delays are fitted with one dt
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_16")
        build_e02a(e02a, Truth(), slice_s=1.0 / 120.0)
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("time slice", text)

    def test_a_gain_too_small_for_llc_v1_is_refused(self):
        # F-12: a slope below MIN_KAPPA_PER_STEER makes the fit unusable, and a card whose gain the 6-decimal
        # serialization would turn into 0 is refused before it is written (DC_VehicleCard.CheckMap refuses gain <= 0)
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_17")
        build_e02a(e02a, Truth(gain_scale=0.003))
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("not usable", text)
        card = json.loads(self.card_text)
        card["kappa_ss"]["kappa_per_steer"][0] = 0.0000399996
        self.assertEqual(fit.clean(card["kappa_ss"]["kappa_per_steer"][0]), 0.0)
        with self.assertRaises(fit.Refusal) as caught:
            fit.card_text(card)
        self.assertIn("kappa_per_steer", str(caught.exception))

    def test_a_cell_that_never_reached_its_speed(self):
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_10")
        build_e02a(e02a, Truth(), starved={(30, 4)})
        out = os.path.join(self.tmp, "one_failed_cell.json")
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a, "--out", out])
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            card = json.load(f)
        fits = {f["target_kmh"]: f["valid_cells"] for f in card["kappa_ss"]["fits"]}
        self.assertEqual(fits, {15.0: 5, 30.0: 4, 50.0: 5})
        failed = [c for c in card["kappa_ss"]["cells"] if c["result"] == "no_speed"]
        self.assertEqual(len(failed), 1)
        self.assertNotIn("valid", failed[0])
        self.assertEqual(len(card["actuator"]["raw_engine_path"]["bins"]), 1 + 14)    # PRBS + the cells with a step

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

    def test_plays_of_two_vehicles_are_refused(self):
        e02a = os.path.join(self.tmp, "e02a", LAUNCH_ID, "play_08")
        build_e02a(e02a, Truth(), prefab="{259EE7B78C51B624}Prefabs/Vehicles/Wheeled/UAZ469/UAZ469.et")
        code, text = run_fit(["--e01", self.e01, "--e02a", e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("one vehicle", text)

    def test_scripted_gearbox_arguments_are_checked(self):
        args = type("Args", (), {"scripted_gears": "3,4,5", "upshift_mps": "5,10", "downshift_mps": "4,8"})()
        self.assertEqual(fit.scripted_gearbox(args, [3, 4, 5, 6], [3, 4, 5, 6]), ([3, 4, 5], [5.0, 10.0], [4.0, 8.0]))
        for gears, up, down in (("3,4,5", "5", "4"), ("3,4", "5", "6"), ("3,4,5", "10,5", "4,3"), ("3,9", "5", "4"),
                                ("3,x", "5", "4")):
            bad = type("Args", (), {"scripted_gears": gears, "upshift_mps": up, "downshift_mps": down})()
            with self.assertRaises(fit.Refusal, msg=(gears, up, down)):
                fit.scripted_gearbox(bad, [3, 4, 5, 6], [3, 4, 5, 6])

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
        # TBD-5 as dc-2.0 decided it (E-01 run 3: a positive SetSteering turns right): steer_sign +1, yaw_axis_sign
        # +1, not measured by an identification session of this card; the same sign as the capture core's own
        # session.json control.steer_sign fallback on DC_Arland (DC_StreamRecorder m_iSteerSign 1)
        self.assertEqual((card["signs"]["steer_sign"], card["signs"]["yaw_axis_sign"], card["signs"]["measured"]),
                         (1, 1, False))
        self.assertEqual(fit.PLACEHOLDER_SIGNS, (1, 1))
        with open(LAYER, "r", encoding="ascii") as f:
            layer = f.read()
        recorder = layer.split('DC_StreamRecorder "{', 1)[1].split("\n  }", 1)[0]
        self.assertIn("\n   m_iSteerSign %d\n" % card["signs"]["steer_sign"], recorder)
        self.assertIn("production", card["use"])
        # the S105's manual gearbox (T0-ENG probe run 2, E-04: has_automatic_gearbox 0, 6 gears, 4 forward) by the
        # vanilla rule: reverse 0, neutral 1, forward 2..5; SetGear drives the scripted gears among them
        self.assertEqual(fit.vanilla_gear_indices(6, 4), {"neutral": 1, "reverse": 0, "forward": [2, 3, 4, 5]})
        self.assertEqual((card["gear_indices_forward"], card["gear_index_neutral"], card["gear_index_reverse"]),
                         ([2, 3, 4, 5], 1, 0))
        # kappa_ss and the scripted gearbox are run 3's first-pass fit, copied (provisional, not measured for this
        # card's build): flat at about 0.23 1/m per contract steer; gears 2, 3, 4 (gear 5 was not driven in E-01)
        self.assertEqual({k: card["kappa_ss"][k] for k in RUN3_KAPPA_SS}, RUN3_KAPPA_SS)
        self.assertEqual({k: card["gearbox"][k] for k in RUN3_GEARBOX}, RUN3_GEARBOX)
        self.assertEqual((card["kappa_ss"]["fits"], card["kappa_ss"]["cells"]), ([], []))   # no fit of this card
        for slot, gear in enumerate(RUN3_GEARBOX["scripted_gears"][:-1]):
            self.assertAlmostEqual(engine_rpm(RUN3_GEARBOX["upshift_mps"][slot], gear), 4500.0, delta=25.0)
            self.assertAlmostEqual(engine_rpm(RUN3_GEARBOX["downshift_mps"][slot], gear + 1), 1800.0, delta=25.0)
        for text in (card["note"], card["gearbox"]["basis"], card["kappa_ss"]["basis"]):
            for word in ("COPIED", "run 3", "provisional", "DD-14a", "not measured for this card's build"):
                self.assertIn(word, text)
        # the actuator stays unidentified: llc_v1 reads the raw engine path only for the limiter rule, which the
        # identity limiter (steer_rate_per_s 0) never needs
        raw = card["actuator"]["raw_engine_path"]
        self.assertEqual((raw["input_echo"], raw["input_shaping"], raw["steer_rate_limit_per_s"],
                          raw["dead_time_steps"], raw["selected_structure"], raw["bins"]),
                         (None, "unknown", None, None, None, []))
        self.assertEqual(card["llc_v1"]["steer_rate_per_s"], 0.0)
        self.assertEqual(card["llc_v1"], dict(fit.LLC_DEFAULTS))           # no longitudinal fit is copied
        self.assertIsNone(card["longitudinal"])
        if RUN3_CARD:
            with open(RUN3_CARD, "r", encoding="ascii") as f:
                run3 = json.load(f)
            for key in ("speeds_mps", "kappa_per_steer", "kappa_offset"):
                self.assertEqual(card["kappa_ss"][key], run3["kappa_ss"][key], key)
            for key in RUN3_GEARBOX:
                self.assertEqual(card["gearbox"][key], run3["gearbox"][key], key)
            self.assertEqual((card["signs"]["steer_sign"], card["signs"]["yaw_axis_sign"]),
                             (run3["signs"]["steer_sign"], run3["signs"]["yaw_axis_sign"]))
            self.assertEqual({str(g): v for g, v in RUN3_SPEED_PER_KRPM.items()},
                             run3["gearbox"]["detail"]["speed_per_krpm"])
            self.assertEqual((card["gear_index_neutral"], card["gear_index_reverse"]),
                             (run3["gear_index_neutral"], run3["gear_index_reverse"]))
        self.assertEqual(fit.enforce_load_check(card), [])

    def test_placeholder_is_refused_at_production(self):
        # DC_LLCv1.AcceptCard's first rule, and DC_VehicleCard's placeholder test, as the Enforce source has them
        with open(LLC_SOURCE, "r", encoding="ascii") as f:
            source = f.read()
        self.assertIn('protected static ref array<string> s_aQualificationUseClasses = {"dev", "probe"};', source)
        accept = enforce_body(source, "DC_LLCv1", "AcceptCard")
        self.assertIn("warning = string.Empty; if (card.IsPlaceholder() && s_aQualificationUseClasses.Find(useClass) "
                      "< 0) { failure =", accept)
        self.assertIn("return m_sStatus == STATUS_PLACEHOLDER;",
                      enforce_body(source, "DC_VehicleCard", "IsPlaceholder"))
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)
        # the copied run-3 values do not make it identified: still a placeholder without sessions or build_key
        self.assertEqual((card["status"], card["identification_sessions"], card["build_key"],
                          card["signs"]["measured"]), ("placeholder", [], None, False))
        self.assertFalse(accept_card(card, "production", "1.8.0.13"))
        for use_class in ("dev", "probe"):
            self.assertTrue(accept_card(card, use_class, "1.8.0.13"))
        self.assertFalse(accept_card(card, "dev", "1.9.0.10"))
        self.assertIn("production", card["use"])

    def test_vanilla_gear_indices(self):
        self.assertEqual(fit.vanilla_gear_indices(7, 4), {"neutral": 2, "reverse": 1, "forward": [3, 4, 5, 6]})
        self.assertEqual(fit.vanilla_gear_indices(5, 4), {"neutral": 0, "reverse": None, "forward": [1, 2, 3, 4]})
        for gears, forward in ((4, 4), (6, 0), (None, 4), (6, None)):
            self.assertIsNone(fit.vanilla_gear_indices(gears, forward), (gears, forward))
        # the same rule as the Mode A actuator of the ego (DC_ActuatorModeA.Init, DD-4-min)
        with open(ACTUATOR_SOURCE, "r", encoding="ascii") as f:
            init = enforce_body(f.read(), "DC_ActuatorModeA", "Init")
        for statement in ("int neutral = m_iGearCount - m_iForwardGearCount - 1;",
                          "bool indicesKnown = m_iForwardGearCount >= 1 && neutral >= 0;",
                          "m_iNeutralGear = neutral; m_iFirstGear = neutral + 1; m_iTopGear = m_iGearCount - 1;",
                          "m_iParkGear = m_iNeutralGear;"):
            self.assertIn(statement, init)

    def test_placeholder_scripted_policy_drives_its_three_gears(self):
        # llc_v1's scripted gearbox on the placeholder card: the mirror follows the Enforce statements ...
        with open(LLC_SOURCE, "r", encoding="ascii") as f:
            source = f.read()
        reset = enforce_body(source, "DC_LLCv1", "Reset")
        self.assertIn("while (m_Card && m_iGearSlot < m_Card.m_aUpshiftMps.Count() && m_fSpeedRef >= "
                      "m_Card.m_aUpshiftMps[m_iGearSlot]) { m_iGearSlot++; }", reset)
        self.assertIn("m_iLastGear = -1;", reset)
        shift = enforce_body(source, "DC_LLCv1", "ScriptedGear")
        self.assertIn("if (m_iGearSlot < last && speed >= m_Card.m_aUpshiftMps[m_iGearSlot]) m_iGearSlot++; else if "
                      "(m_iGearSlot > 0 && speed < m_Card.m_aDownshiftMps[m_iGearSlot - 1]) m_iGearSlot--;", shift)
        step = enforce_body(source, "DC_LLCv1", "Step")
        self.assertIn("gear = ScriptedGear(speed); if (gear != m_iLastGear) m_iDipStepsLeft = m_iClutchDipSteps; "
                      "m_iLastGear = gear;", step)
        self.assertIn("m_Actuator.Apply(m_fSteerOut, throttle, brake, false, clutch, gear);", step)
        # the dip counts the steps a callback covers (dc-2.0 D-A), its length the measured step dt
        self.assertIn("if (m_iDipStepsLeft > 0) { clutch = 0; m_iDipStepsLeft -= steps; if (m_iDipStepsLeft < 0) "
                      "m_iDipStepsLeft = 0; }", step)
        self.assertIn("int steps = coveredSteps; if (steps < 1) steps = 1;", step)
        self.assertIn("int dipSteps = Math.Round(card.m_fClutchDipS / m_fStepDtS);",
                      enforce_body(source, "DC_LLCv1", "Init"))
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)
        neutral, reverse = card["gear_index_neutral"], card["gear_index_reverse"]
        up, down = card["gearbox"]["upshift_mps"], card["gearbox"]["downshift_mps"]
        # ... over an episode from standstill: 0 -> 16 m/s (above the expert's 40 km/h cruise and the 50 km/h E-02a
        # cell) at 1.5 m/s2, 2 s at 16 m/s, back to 0 at 2 m/s2, one callback per step of the frozen step dt
        speeds = [min(16.0, 1.5 * STEP_DT * i) for i in range(int(16.0 / 1.5 / STEP_DT) + 60)]
        speeds += [max(0.0, 16.0 - 2.0 * STEP_DT * i) for i in range(1, int(16.0 / 2.0 / STEP_DT) + 15)]
        llc = LlcGears(card, STEP_DT)
        llc.reset(speeds[0])
        trace = [(v,) + llc.step(v) for v in speeds]
        gears = [g for _, g, _ in trace]
        self.assertEqual(gears[0], 2)                                   # first gear at the start
        self.assertEqual(set(gears), {2, 3, 4})                         # the three scripted gears (run 3's)
        self.assertTrue(all(g > neutral and g != reverse for g in gears))   # never neutral or reverse
        changes = [(trace[i][0], gears[i - 1], gears[i]) for i in range(1, len(trace)) if gears[i] != gears[i - 1]]
        self.assertEqual([(a, b) for _, a, b in changes], [(2, 3), (3, 4), (4, 3), (3, 2)])
        for speed, before, after in changes:
            slot = card["gearbox"]["scripted_gears"].index(min(before, after))
            if after > before:
                self.assertGreaterEqual(speed, up[slot])
            else:
                self.assertLess(speed, down[slot])
        # engine speeds by E-01 run 3's measured speed per 1000 rpm: the probe policy's band, never above about
        # 4500 rpm, and at least about 1800 rpm whenever the car is above the lowest downshift speed
        rpms = [(v, engine_rpm(v, g)) for v, g, _ in trace]
        self.assertLessEqual(max(r for _, r in rpms), 4560.0)
        self.assertGreaterEqual(min(r for v, r in rpms if v >= down[0]), 1790.0)
        # no dip on the placeholder (clutch 1, as vanilla VehicleCinematicTrack)
        self.assertTrue(all(c == 1.0 for _, _, c in trace))
        # hysteresis: speed jitter of +-0.3 m/s around an upshift speed shifts once
        llc.reset(up[0] - 0.5)
        jitter = [2] + [llc.step(up[0] + 0.3 * math.sin(i / 3.0))[0] for i in range(600)]
        self.assertEqual(sum(1 for a, b in zip(jitter, jitter[1:]) if a != b), 1)
        # an episode that begins while rolling starts in the gear for that speed: 12 m/s second (index 3), 13 m/s
        # third (index 4)
        llc.reset(12.0)
        self.assertEqual(llc.step(12.0)[0], 3)
        llc.reset(13.0)
        self.assertEqual(llc.step(13.0)[0], 4)
        # with a clutch dip (E-01's m_fGearClutchDipS), clutch 0 for the dip after the first SetGear and every change
        dipped = dict(card, gearbox=dict(card["gearbox"], clutch_dip_s=0.25))
        llc = LlcGears(dipped, STEP_DT)
        llc.reset(0.0)
        trace = [llc.step(v) for v in speeds]
        steps = int(math.floor(0.25 / STEP_DT + 0.5))
        starts = [0] + [i for i in range(1, len(trace)) if trace[i][0] != trace[i - 1][0]]
        self.assertEqual(len(starts), 5)
        for start in starts:
            self.assertEqual([c for _, c in trace[start:start + steps + 1]], [0.0] * steps + [1.0])
        self.assertEqual(sum(1 for _, c in trace if c == 0.0), 5 * steps)
        # the same episode in callbacks of 1, 2 and 3 steps (a capture session's shapes): the same gears, and each dip
        # lasts its steps rounded up to whole callbacks (at least the card's, at most 2 steps more)
        pattern = [1, 1, 2, 1, 3, 1, 2, 3, 1, 1]
        llc = LlcGears(dipped, STEP_DT)
        llc.reset(0.0)
        merged, index = [], 0
        while index < len(speeds):
            covered = min(pattern[len(merged) % len(pattern)], len(speeds) - index)
            merged.append((covered,) + llc.step(speeds[index], covered))      # the speed at the callback's top
            index += covered
        self.assertEqual(trace[0][0], merged[0][1])
        changes = [i for i in range(1, len(merged)) if merged[i][1] != merged[i - 1][1]]
        self.assertEqual([(merged[i - 1][1], merged[i][1]) for i in changes], [(2, 3), (3, 4), (4, 3), (3, 2)])
        for start in [0] + changes:
            run, dipped_steps = start, 0
            while run < len(merged) and merged[run][2] == 0.0:
                dipped_steps += merged[run][0]
                run += 1
            self.assertGreaterEqual(dipped_steps, steps)
            self.assertLessEqual(dipped_steps, steps + 2)

    def test_gear_index_rules(self):
        # the card's gear indices as DC_VehicleCard.CheckGearbox checks them (gearbox_violations mirrors it)
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)

        def violations(**changes):
            changed = json.loads(json.dumps(card))
            for key, value in changes.items():
                if key in changed["gearbox"]:
                    changed["gearbox"][key] = value
                else:
                    changed[key] = value
            return fit.gearbox_violations(changed)

        self.assertEqual(violations(), [])
        self.assertTrue(violations(gear_index_reverse=1))                     # reverse not below neutral
        self.assertTrue(violations(gear_index_neutral=True))                   # neither null nor an index
        self.assertTrue(violations(gear_index_reverse=-1))
        self.assertTrue(violations(scripted_gears=[3, 2, 4, 5]))               # not increasing: start gear unclear
        self.assertTrue(violations(gear_indices_forward=[1, 2, 3, 4, 5], scripted_gears=[1, 2, 3, 4]))   # neutral
        self.assertEqual(violations(gear_index_neutral=None, gear_index_reverse=None), [])
        self.assertTrue(violations(mode="automatic", clutch_dip_s=0.0, gear_index_reverse=2))
        with open(LLC_SOURCE, "r", encoding="ascii") as f:
            source = f.read()
        check = enforce_body(source, "DC_VehicleCard", "CheckGearbox")
        for statement in ("m_fNeutralGear >= 0 && m_fReverseGear >= 0 && m_fReverseGear >= m_fNeutralGear",
                          "if (gear <= previous)", "if (m_fNeutralGear >= 0 && gear <= m_fNeutralGear)"):
            self.assertIn(statement, check)
        # llc_v1 checks the card against the ego (DC_Actuator, DD-4-min): no automatic card on a manual gearbox, the
        # scripted gears in the ego's forward range and the card's neutral the ego's
        ego = enforce_body(source, "DC_LLCv1", "CheckEgoGearbox")
        for statement in ("if (kind == DC_Actuator.GEARBOX_MANUAL)", "if (first < 0 || top < first)",
                          "if (gear < first || gear > top)",
                          "if (card.m_fNeutralGear >= 0 && card.m_fNeutralGear != neutral)"):
            self.assertIn(statement, ego)
        self.assertIn("if (!CheckEgoGearbox(card, actuator, failure)) return false;",
                      enforce_body(source, "DC_LLCv1", "Init"))

    def test_e04_gives_the_neutral_and_reverse_indices(self):
        # an identified card records E-04's indices by the vanilla rule (here 7 gears, 4 forward: neutral 2, reverse
        # 1, forward 3..6 as the synthetic gearbox); scripted gears must lie above that neutral
        args = ["--e02a", self.e02a, "--scripted-gears", "3,4,5", "--upshift-mps", "5,10", "--downshift-mps", "4,8"]
        e01 = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_21")
        build_e01(e01, Truth(automatic=False), gear_sequence=(3, 4, 5), clutch_dip=0.0)

        def e04(name, gears, forward, automatic=0):
            play = os.path.join(self.tmp, "e04", LAUNCH_ID, name)
            text = "has_automatic_gearbox=%d\ngearbox_gears=%d\ngearbox_forward_gears=%d\n" % (automatic, gears, forward)
            probe_common.write_synthetic_play(play, header("e04"), {}, FOOTER, {"e04_vehicle.txt": text.encode("ascii")})
            return play

        out = os.path.join(self.tmp, "e04.json")
        code, text = run_fit(["--e01", e01, "--e04", e04("play_22", 7, 4), "--out", out] + args)
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            card = json.load(f)
        self.assertEqual((card["gear_index_neutral"], card["gear_index_reverse"]), (2, 1))
        self.assertEqual(card["gearbox"]["has_automatic_gearbox"], False)
        self.assertEqual(card["gearbox"]["detail"]["e04"]["gearbox_gears"], 7)
        self.assertEqual(fit.enforce_load_check(card), [])
        code, text = run_fit(["--e01", e01, "--e04", e04("play_23", 8, 4)] + args)    # neutral 3: gear 3 is no forward
        self.assertEqual(code, 2, text)
        self.assertIn("above the neutral index", text)
        code, text = run_fit(["--e01", self.e01, "--e02a", self.e02a, "--e04", e04("play_24", 7, 4)])
        self.assertEqual(code, 2, text)                                 # E-01 automatic, E-04 manual: disagree
        self.assertIn("disagree", text)

    def manual_e01(self, name, lines, observe="skipped", **head):
        """A manual-gearbox E-01 play driven by the probe kit's shift policy, with gear_scripted 2, 3, 4, 3, 2 (the
        probe's default m_sGearSequence) and no clutch dip."""
        play = os.path.join(self.tmp, "e01", LAUNCH_ID, name)
        build_e01(play, Truth(manual=True), gear_sequence=(2, 3, 4, 3, 2), observe=observe, clutch_dip=0.0,
                  lines=lines, **head)
        return play

    def fit_card(self, argv):
        out = os.path.join(self.tmp, "card_%d.json" % len(os.listdir(self.tmp)))
        code, text = run_fit(argv + ["--out", out])
        self.assertEqual(code, 0, text)
        with open(out, "r", encoding="ascii") as f:
            return json.load(f), text

    def test_manual_gearbox_without_gear_observe_probe_kit_header(self):
        # probe kit 699d8e2 on the S105: gear_observe skipped (note manual_gearbox), the run header's gearbox lines
        # (vehicle.gearbox=manual, vehicle.gear_index_neutral 1 / first 2 / top 5); E-01 drives gears 2..4 only
        e01 = self.manual_e01("play_30", KIT_GEARBOX_LINES)
        speeds = ["--upshift-mps", "4.5,8.5", "--downshift-mps", "3.5,7.0"]
        card, text = self.fit_card(["--e01", e01, "--e02a", self.e02a] + speeds)
        gearbox = card["gearbox"]
        self.assertEqual((gearbox["mode"], gearbox["has_automatic_gearbox"], gearbox["clutch_dip_s"]),
                         ("scripted", False, 0.0))
        self.assertIn("vehicle.gearbox=manual", gearbox["basis"])
        self.assertIn("gear_observe not run", gearbox["basis"])
        # default scripted gears: from the first gear (2) upward while E-01 shows propulsion; 5 was never driven
        self.assertEqual(gearbox["scripted_gears"], [2, 3, 4])
        self.assertEqual((gearbox["upshift_mps"], gearbox["downshift_mps"]), ([4.5, 8.5], [3.5, 7.0]))
        self.assertEqual(card["gear_indices_forward"], [2, 3, 4])
        self.assertEqual((card["gear_index_neutral"], card["gear_index_reverse"]), (1, 0))
        detail = gearbox["detail"]
        self.assertEqual(detail["propulsion_gears"], [2, 3, 4])
        self.assertEqual((detail["e01_header"]["indices"]["first"], detail["e01_header"]["indices"]["top"]), (2, 5))
        self.assertIn("default", detail["scripted_gears_rule"])
        for gear in (2, 3, 4):
            self.assertAlmostEqual(detail["speed_per_krpm"][str(gear)], MANUAL_SPEED_PER_KRPM[gear], delta=0.01)
        self.assertEqual(fit.enforce_load_check(card), [])
        self.assertIn("neutral 1, reverse 0; scripted gears [2, 3, 4]", text)
        # explicit gears: within first..top, above neutral and driven forward with propulsion
        card, _ = self.fit_card(["--e01", e01, "--e02a", self.e02a, "--scripted-gears", "2,3", "--upshift-mps", "5",
                                 "--downshift-mps", "4"])
        self.assertEqual(card["gearbox"]["scripted_gears"], [2, 3])
        self.assertNotIn("scripted_gears_rule", card["gearbox"]["detail"])
        for gears, up, down, reason in (("2,3,4,5", "4.5,8.5,13", "3.5,7,11", "not seen while driving forward"),
                                        ("1,2", "3", "2", "above the neutral index")):
            code, text = run_fit(["--e01", e01, "--e02a", self.e02a, "--scripted-gears", gears, "--upshift-mps", up,
                                  "--downshift-mps", down])
            self.assertEqual(code, 2, text)
            self.assertIn(reason, text)
        # without shift speeds: refused, naming the default gears and the measured speed per 1000 rpm
        code, text = run_fit(["--e01", e01, "--e02a", self.e02a])
        self.assertEqual(code, 2, text)
        self.assertIn("--upshift-mps and --downshift-mps", text)
        self.assertIn("[2, 3, 4]", text)
        self.assertIn("4: 4.100 m/s", text)

    def test_manual_gearbox_earlier_header_style(self):
        # a play before the gearbox fix (probe run 2 style): only vehicle.has_automatic_gearbox=0 and the gear counts,
        # and no gear_observe row: the indices follow by the vanilla rule, the card equals the probe kit style's
        e01 = self.manual_e01("play_31", EARLIER_GEARBOX_LINES, observe="absent")
        card, _ = self.fit_card(["--e01", e01, "--e02a", self.e02a, "--upshift-mps", "4.5,8.5", "--downshift-mps",
                                 "3.5,7.0"])
        gearbox = card["gearbox"]
        self.assertEqual((gearbox["mode"], gearbox["has_automatic_gearbox"], gearbox["scripted_gears"]),
                         ("scripted", False, [2, 3, 4]))
        self.assertIn("vehicle.has_automatic_gearbox=0", gearbox["basis"])
        self.assertNotIn("vehicle.gearbox=", gearbox["basis"])
        self.assertEqual((card["gear_indices_forward"], card["gear_index_neutral"], card["gear_index_reverse"]),
                         ([2, 3, 4], 1, 0))
        self.assertIn("gear counts 6 / 4", gearbox["detail"]["e01_header"]["indices"]["source"])
        self.assertEqual(fit.enforce_load_check(card), [])

    def test_manual_gearbox_from_the_phase_note_alone(self):
        # no gearbox line in the header: the skipped gear_observe says manual, but nothing gives the indices
        e01 = self.manual_e01("play_32", {})
        speeds = ["--upshift-mps", "4.5,8.5", "--downshift-mps", "3.5,7.0"]
        code, text = run_fit(["--e01", e01, "--e02a", self.e02a] + speeds)
        self.assertEqual(code, 2, text)
        self.assertIn("pass --scripted-gears", text)
        card, _ = self.fit_card(["--e01", e01, "--e02a", self.e02a, "--scripted-gears", "2,3,4"] + speeds)
        self.assertEqual((card["gearbox"]["mode"], card["gearbox"]["has_automatic_gearbox"]), ("scripted", False))
        self.assertIn("gear_observe skipped (note manual_gearbox)", card["gearbox"]["basis"])
        self.assertEqual((card["gear_index_neutral"], card["gear_index_reverse"]), (None, None))
        # E-04's counts give them (and the default gears)
        e04 = os.path.join(self.tmp, "e04", LAUNCH_ID, "play_33")
        probe_common.write_synthetic_play(e04, header("e04"), {}, FOOTER, {"e04_vehicle.txt": (
            b"has_automatic_gearbox=0\ngearbox_gears=6\ngearbox_forward_gears=4\n")})
        card, _ = self.fit_card(["--e01", e01, "--e02a", self.e02a, "--e04", e04] + speeds)
        self.assertEqual((card["gearbox"]["scripted_gears"], card["gear_index_neutral"], card["gear_index_reverse"]),
                         ([2, 3, 4], 1, 0))

    def test_manual_gearbox_contradictions_are_refused(self):
        def said(lines, rows=()):
            play = type("Play", (), {"header": dict(lines), "dir": "e01/x/play_01"})()
            return fit.e01_gearbox(play, list(rows))

        skipped = [{"phase": "gear_observe", "result": "skipped", "note": "manual_gearbox"}]
        self.assertEqual(said(KIT_GEARBOX_LINES)["indices"],
                         {"neutral": 1, "first": 2, "top": 5, "reverse": 0,
                          "source": "E-01 run header vehicle.gear_index_* (the probe kit, vanilla rule)"})
        self.assertEqual(said({}, skipped)["manual"], True)
        self.assertEqual(said({})["manual"], None)
        self.assertEqual(said({"vehicle.gearbox": "automatic"})["manual"], False)
        for lines, rows, reason in (
                ({"vehicle.gearbox": "automatic"}, skipped, "disagrees with itself"),
                (dict(KIT_GEARBOX_LINES, **{"vehicle.has_automatic_gearbox": "1"}), (), "disagrees with itself"),
                (dict(KIT_GEARBOX_LINES, **{"vehicle.gear_index_neutral": "2", "vehicle.gear_index_first": "3"}), (),
                 "vanilla rule"),
                (dict(KIT_GEARBOX_LINES, **{"vehicle.gear_index_top": ""}), (), "incomplete"),
                ({"vehicle.gearbox": "sequential"}, (), "neither manual nor automatic"),
                ({"vehicle.gear_index_neutral": "1.5"}, (), "integer"),
                ({"vehicle.gearbox_gears": "4", "vehicle.gearbox_forward_gears": "4"}, (), "no neutral")):
            with self.assertRaises(fit.Refusal, msg=lines) as caught:
                said(lines, rows)
            self.assertIn(reason, str(caught.exception))
        # without gear_observe a play that does not say manual decides nothing
        e01 = self.manual_e01("play_34", {})
        steps = fit.e01_actuation.Steps(probe_common.Play(e01).csv("e01_steps"))
        with self.assertRaises(fit.Refusal) as caught:
            fit.fit_gearbox(steps)
        self.assertIn("does not say its gearbox is manual", str(caught.exception))
        self.assertEqual(fit.fit_gearbox(steps, True)[:3], (False, [2, 3, 4], [2, 3, 4]))
        # gear_observe upshifts without SetGear on a play that says manual
        auto = os.path.join(self.tmp, "e01", LAUNCH_ID, "play_35")
        build_e01(auto, Truth(), lines=KIT_GEARBOX_LINES)
        steps = fit.e01_actuation.Steps(probe_common.Play(auto).csv("e01_steps"))
        with self.assertRaises(fit.Refusal) as caught:
            fit.fit_gearbox(steps, True)
        self.assertIn("disagrees with itself", str(caught.exception))
        # E-04 against the E-01 header: the gearbox kind and the indices (neutral, reverse, first and top) must agree;
        # 4 gears with 2 forward keep neutral 1 and reverse 0 but end at top 3, where the header says 5
        kit = self.manual_e01("play_36", KIT_GEARBOX_LINES)
        speeds = ["--upshift-mps", "4.5,8.5", "--downshift-mps", "3.5,7.0"]
        for name, text, reason in (("play_37", b"has_automatic_gearbox=1\n", "has_automatic_gearbox 1"),
                                   ("play_38", b"has_automatic_gearbox=0\ngearbox_gears=7\ngearbox_forward_gears=4\n",
                                    "neutral 2, reverse 1, first 3 and top 6"),
                                   ("play_39", b"has_automatic_gearbox=0\ngearbox_gears=4\ngearbox_forward_gears=2\n",
                                    "neutral 1, reverse 0, first 2 and top 3, the E-01 run header neutral 1, reverse "
                                    "0, first 2 and top 5")):
            e04 = os.path.join(self.tmp, "e04", LAUNCH_ID, name)
            probe_common.write_synthetic_play(e04, header("e04"), {}, FOOTER, {"e04_vehicle.txt": text})
            code, out = run_fit(["--e01", kit, "--e02a", self.e02a, "--e04", e04] + speeds)
            self.assertEqual(code, 2, out)
            self.assertIn(reason, out)
            self.assertIn("disagree", out)

    def test_header_keys_are_the_probe_kits(self):
        # the run header lines and the phase note the fit reads are the ones the probe kit writes: the lines of
        # DC_ProbeSequences.c at probe kit 699d8e2 pinned in KIT_699D8E2_LINES (this branch does not have that file's
        # AppendGearboxLines yet; once it does, its file must hold every pinned line) and AppendVehicleLines of this
        # tree
        with open(os.path.join(REPO, "src", "enfusion", "datacapture", "DC_ProbeSequences.c"), "r",
                  encoding="ascii") as f:
            source = f.read()
        pinned = "\n".join(KIT_699D8E2_LINES)
        for key in (fit.HEADER_KIND, fit.HEADER_AUTOMATIC) + fit.HEADER_GEARS + fit.HEADER_INDICES:
            self.assertIn('lines.Insert("%s=" +' % key, pinned)
        self.assertIn('SkipPhase(phase, "%s",' % fit.NOTE_MANUAL, pinned)
        self.assertIn('phase.m_sResult = "skipped";', pinned)
        self.assertIn('m_sNote += "/" + token;', pinned)            # note tokens joined by "/", as the fit splits them
        play = type("Play", (), {"header": {}, "dir": "e01/x/play_01"})()
        rows = [{"phase": "gear_observe", "result": "skipped", "note": "other_token/" + fit.NOTE_MANUAL}]
        self.assertEqual(fit.e01_gearbox(play, rows)["manual"], True)
        self.assertIn('"gear_observe"', pinned)
        kind = re.search(r'string kind = "(\w+)";\s+if \(m_bAutomaticGearbox\)\s+kind = "(\w+)";', pinned)
        self.assertEqual(kind.groups(), ("manual", "automatic"))   # the values e01_gearbox accepts
        self.assertIn('return "1";', pinned)                        # BoolText: has_automatic_gearbox is 0 or 1
        self.assertIn('return "0";', pinned)
        for key in (fit.HEADER_AUTOMATIC,) + fit.HEADER_GEARS:
            self.assertIn('lines.Insert("%s=" +' % key, source)
        if "AppendGearboxLines" in source:
            for line in KIT_699D8E2_LINES:
                self.assertIn(line, source)
        self.assertEqual(fit.HEADER_INDICES, ("vehicle.gear_index_neutral", "vehicle.gear_index_first",
                                              "vehicle.gear_index_top"))
        self.assertEqual((fit.HEADER_KIND, fit.NOTE_MANUAL), ("vehicle.gearbox", "manual_gearbox"))

    def test_cards_keep_the_llc_v1_format(self):
        with open(CARD, "r", encoding="ascii", newline="") as f:
            committed = f.read()
        for text in (committed, self.card_text):
            self.assertNotIn("\r", text)
            self.assertTrue(text.endswith("}\n"))
            text.encode("ascii")
            card = json.loads(text)
            for key in fit.ENGINE_SCANNED_KEYS:
                self.assertEqual(text.count('"%s"' % key), 1, key)         # DC_JsonScan.FindValue's rule
            self.assertEqual(scan_numbers(text, "speeds_mps"), card["kappa_ss"]["speeds_mps"])
            self.assertEqual(scan_numbers(text, "kappa_per_steer"), card["kappa_ss"]["kappa_per_steer"])
            self.assertEqual(scan_numbers(text, "kappa_offset"), card["kappa_ss"]["kappa_offset"])
            self.assertEqual(scan_numbers(text, "upshift_mps"), card["gearbox"]["upshift_mps"])
            self.assertEqual(scan_numbers(text, "scripted_gears"), card["gearbox"]["scripted_gears"])
            self.assertEqual(scan_numbers(text, "gear_indices_forward"), card["gear_indices_forward"])
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
            self.assertEqual(fit.enforce_load_check(card), [])              # the rules llc_v1 loads it with

    def test_enforce_reader_asks_for_the_written_keys(self):
        with open(LLC_SOURCE, "r", encoding="ascii") as f:
            source = f.read()
        body = source.split("protected bool ReadLlcNumbers", 1)[1].split("\n\t}\n", 1)[0]
        self.assertEqual(sorted(re.findall(r'ReadContextNumber\(context, "(\w+)"', body)), sorted(LLC_KEYS))
        arrays = re.findall(r'DC_JsonScan\.ReadNumbers\(text, "(\w+)"', source)
        self.assertEqual(sorted(arrays), sorted(fit.ENGINE_ARRAY_KEYS))
        scanned = re.findall(r'DC_JsonScan\.(?:ReadStrings|IsNull|ReadIndexOrNull)\(text, "(\w+)"', source)
        self.assertEqual(sorted(scanned), sorted(set(fit.ENGINE_SCANNED_KEYS) - set(fit.ENGINE_ARRAY_KEYS)))
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)
        reader = source.split("class DC_VehicleCard", 1)[1].split("class DC_LLCv1", 1)[0]
        keys = re.findall(r'context\.ReadValue\("(\w+)"', reader)
        keys += re.findall(r'ReadContextNumber\(context, "(\w+)"', reader.split("protected bool ReadLlcNumbers", 1)[0])
        self.assertIn("clutch_dip_s", keys)
        self.assertIn("input_shaping", keys)
        for key in keys:
            self.assertTrue(key in card or key in card["signs"] or key in card["gearbox"]
                            or key in card["actuator"]["raw_engine_path"], key)
        shapings = re.search(r"s_aInputShapings = \{([^}]*)\}", reader).group(1)
        self.assertEqual(tuple(re.findall(r'"(\w+)"', shapings)), fit.INPUT_SHAPINGS)
        self.assertIn("CLUTCH_DIP_MAX_S = %g;" % fit.CLUTCH_DIP_MAX_S, reader)

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
