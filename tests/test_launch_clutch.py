#!/usr/bin/env python3
"""Offline check of the Mode A launch clutch (DD-14c, src/enfusion/datacapture/DC_LaunchClutch.c).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_launch_clutch.py

Why: the 1.9.0.37 powertrain stalls where 1.8.0.13 slipped. At the first 1.9 Workbench Play (launch
20261106T052424Z-wb-addc, e01 play_01) Mode A released the handbrake into first gear at throttle 0.5 and clutch 1 from
800 rpm. The next step read 528 rpm, the one after 18 rpm with the engine off, and the car never moved. Vanilla script
has no anti-stall for direct setters (G19_0_STATIC_1.9.0.37.md section 1), so DC_LaunchClutch ramps the clutch by rpm.

  - LaunchClutch is a Python mirror of DC_LaunchClutch.Step, ConsumeRestart and the counters. The tests tie it to
    the Enforce source statement by statement.
  - ToyPowertrain is a TOY S105 in first gear: one engine inertia with an idle governor and a torque curve, a friction
    clutch of MaxClutchTorque capacity, and the vehicle's inertia through first gear. It has no wheel slip, no shifts
    and no tyre model. It uses the run header facts (rpm_idle 800, peak power at 4800 rpm, max 8000 rpm, step dt
    1/30 s), plus an engine inertia and a stall rpm estimated from the 1.9 Play's 800 -> 528 -> 18 rpm. It exists to
    catch control-loop bugs in the launch clutch. It does not predict the 1.9 powertrain: every in-engine behaviour
    stays for a Workbench Play.
  - With the defaults in mode "ramp" the toy launches from standstill at throttle 0.3, 0.5 and 1.0 without a stall
    and without a sustained limit cycle in c. It also does so in capture callbacks of 1 to 3 steps.
  - In mode "off" it stalls the way the 1.9 Play did.
  - Braking to a stop does not stall. That includes a full brake from 16 and 40 km/h in first and second gear, in
    callbacks of 1, 2 and 3 steps and mixes (Codex round 1). A stall is restarted, never twice within the cooldown,
    whatever the callback sizes.
  - CheckEngine bounds the band by the engine's idle and max rpm.
  - The defaults are the same in DC_LaunchClutchConfig, the DC_ProbeSequences attributes, the provisional 1.9.0.37 card
    and its generator. The three actuation paths (DC_ProbeActuator, llc_v1, DC_ActuatorModeA.Start) call the helper as
    described, and the deploy manifest carries the new file and the 1.9.0.37 card.

DD-14d (mode slip). The first 1.9.0.37 Play with the ramp (launch 20261106T194516Z-wb-ab6a, e01 play_01) chattered.
tools/ident/launch_clutch_model.py holds the S105 powertrain fitted to that Play (tools/ident/fit_launch_clutch.py) and
the mirror of DC_LaunchClutch in every mode (LaunchClutch here). tests/fixtures/launch_clutch_19/e01_launches.csv is
that Play's launches and a free decay, trimmed to the step columns the fit reads.
  - The fit: from DD-14c's toy values it lands on S105_19, rms about 40 rpm / 0.035 m/s over 15-step shooting segments;
    the free decay gives the drag rate.
  - Replayed in closed loop with each launch's recorded readback timing, the DD-14c ramp chatters as the Play did, and
    mode slip does not.
  - Acceptance on the plant (timing U and A, callbacks of 1, 2, 3 and mixed steps): launches at throttle 0.3 / 0.5 /
    0.6 / 1.0, flat and at +-5 % grade, with no opening, no step-to-step oscillation of c (the longest alternation run at
    most 1), rpm at or above the guard + 100 and c = 1 at throttle 0.5 by LOCKUP_LIMIT_S (2.0 s with 1-step callbacks,
    the probe kit's); the same over every combination of engine inertia and clutch capacity +-30 % and torque +-20 %
    (2592 launches); stops by a brake of 0.3 / 0.6 / 1.0 before the lock-up (3 / 5 / 7 km/h) and coupled (15 / 30 / 40
    km/h, also at the inertia / capacity corners) open without a stall, above the guard + 100; Codex round 1's cases;
    c stays 1 through the probe policy's shifts and gear_scripted's 2, 3, 4, 3, 2.
  - The slip mirror follows DC_LaunchClutch.StepSlip statement by statement; the defaults (DC_LaunchClutchConfig.Slip(),
    the attributes, the card, the generator) agree; the probe kit and llc_v1 pass the driveline rpm as described.
"""

import importlib.util
import itertools
import json
import math
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS = os.path.join(REPO, "src", "enfusion", "datacapture")
HELPER_SOURCE = os.path.join(SCRIPTS, "DC_LaunchClutch.c")
PROBE_SOURCE = os.path.join(SCRIPTS, "DC_ProbeSequences.c")
LLC_SOURCE = os.path.join(SCRIPTS, "DC_LLCv1.c")
RECORDER_SOURCE = os.path.join(SCRIPTS, "DC_StreamRecorder.c")
EXPERT_SOURCE = os.path.join(SCRIPTS, "DC_ExpertHL.c")
MANIFEST = os.path.join(REPO, "addons", "DataCapture", "deploy_manifest.txt")
CARD = os.path.join(REPO, "tools", "ident", "cards", "1.9.0.37", "S105.json")     # the card with the launch clutch
STEP_DT = 1.0 / 30.0        # run header clock.step_dt_s of the 1.9 Play (and of 1.8)
RPM_PER_RADS = 60.0 / (2.0 * math.pi)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load_module("fit_steer_kappa", os.path.join(REPO, "tools", "ident", "fit_steer_kappa.py"))
model = load_module("launch_clutch_model", os.path.join(REPO, "tools", "ident", "launch_clutch_model.py"))
analysis = load_module("fit_launch_clutch", os.path.join(REPO, "tools", "ident", "fit_launch_clutch.py"))
FIXTURE = os.path.join(REPO, "tests", "fixtures", "launch_clutch_19", "e01_launches.csv")


def read(path):
    with open(path, "r", encoding="ascii", newline="") as f:
        return f.read()


def code_only(source):
    """The source without // comments (string literals here never contain //)."""
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def enforce_body(source, cls, method):
    """One method of an Enforce class, comments removed and whitespace collapsed (the statements the mirror follows)."""
    body = code_only(source).split("\nclass %s" % cls, 1)[1]
    found = re.search(r"\n\t(?:(?:override|protected|static) )*[\w<>]+ %s\(" % method, body)
    body = body[found.end():].split("\n\t}\n", 1)[0]
    return " ".join(body.split())


def class_source(source, cls):
    """The text of one Enforce class (to the next top-level class)."""
    match = re.search(r"\nclass %s\b(?!\w)[^\n]*\n" % cls, source)
    rest = source[match.end():]
    end = re.search(r"\n//-+\n//!", rest)
    return rest[:end.start()] if end else rest


def class_method(source, cls, method):
    """enforce_body for classes whose names prefix others (DC_LaunchClutch vs DC_LaunchClutchConfig)."""
    body = code_only(class_source(source, cls))
    found = re.search(r"\n\t(?:(?:override|protected|static) )*[\w<>]+ %s\(" % method, body)
    body = body[found.end():].split("\n\t}\n", 1)[0]
    return " ".join(body.split())


def enforce_constants(source, cls):
    """static const float / bool NAME = value; of one class, as Python numbers or bools."""
    out = {}
    for kind, name, value in re.findall(r"static const (float|bool) (\w+) = ([^;]+);", class_source(source, cls)):
        out[name] = (value.strip() == "true") if kind == "bool" else float(value)
    return out


def attribute_defaults(source):
    """{member: default text} of the [Attribute("...", ...)] declarations of a component."""
    return dict((name, default) for default, name in re.findall(
        r'\[Attribute\("([^"]*)"[^\n]*\]\n\tprotected \w+ (m_\w+);', source))


# The mirror of DC_LaunchClutch (every mode) lives in tools/ident/launch_clutch_model.py, where fit_launch_clutch.py
# replays it; LaunchClutchTiesTest and SlipTiesTest tie it to the Enforce source
LaunchClutch = model.LaunchClutch


def ramp(**config):
    return LaunchClutch(mode="ramp", **config)


def check_ranges(config):
    """Mirror of DC_LaunchClutchConfig.Check (the ranges, every mode; the slip fields too)."""
    c = dict(LaunchClutch.DEFAULTS, **config)
    return (c["mode"] in ("off", "ramp", "slip") and 0 < c["engage_rpm"] < c["full_rpm"] <= 10000.0
            and 0 <= c["stall_guard_rpm"] < c["engage_rpm"] and 0 < c["couple_time_s"] <= 10.0
            and 0 < c["uncouple_time_s"] <= 10.0 and 0 <= c["restart_cooldown_s"] <= 10.0
            and 0 <= c["slip_kp"] <= 0.01 and 0 < c["slip_ki"] <= 0.1 and 0 <= c["lock_tolerance_rpm"] <= 1000.0)


def probe_set_launch_clutch(config, manual, downshift_rpm, rpm_idle, rpm_max, upshift_rpm=4500.0,
                            mps_per_krpm=(1.54871, 2.772066, 4.169103)):
    """Mirror of DC_ProbeActuator.SetLaunchClutch: the ranges, then on a manual gearbox with mode ramp or slip full_rpm
    below the policy's downshift rpm, for slip every m/s per 1000 rpm > 0 and every policy upshift landing at or above
    2 x engage_rpm (CheckSlipUpshifts), then CheckEngine; the one validation of the configuration the actuator drives
    with."""
    if not check_ranges(config):
        return False
    if config["mode"] in ("ramp", "slip") and manual and config["full_rpm"] >= downshift_rpm:
        return False
    if config["mode"] == "slip" and manual:
        if any(v <= 0 for v in mps_per_krpm):
            return False
        ceiling = 2.0 * config["engage_rpm"]
        if any(upshift_rpm * a / b < ceiling for a, b in zip(mps_per_krpm, mps_per_krpm[1:])):
            return False
    return not manual or check_engine(config, rpm_idle, rpm_max)


def check_engine(config, rpm_idle, rpm_max):
    """Mirror of DC_LaunchClutchConfig.CheckEngine: mode ramp or slip needs engage_rpm above the idle and full_rpm below
    FULL_RPM_MAX_FRACTION (0.5) of a known max rpm."""
    if config["mode"] not in ("ramp", "slip"):
        return True
    if config["engage_rpm"] <= rpm_idle:
        return False
    return rpm_max > 0 and config["full_rpm"] < 0.5 * rpm_max


class ToyPowertrain:
    """A TOY S105 in first gear (see the module docstring). It catches control-loop bugs, and it does not predict
    the 1.9 powertrain."""

    J_ENGINE = 0.12             # [kg m2]: 800 -> 528 rpm in one 1/30 s step against a 130 Nm clutch (the 1.9 Play)
    RPM_IDLE = 800.0            # run header vehicle.rpm_idle
    RPM_STALL = 550.0           # the 1.9 engine died between 528 rpm and the next step
    GOVERNOR_GAIN = 6.0         # idle governor: throttle floor gain x (idle - rpm) / idle ...
    GOVERNOR_MAX = 0.35         # ... at most this (an idle air valve, not full throttle)
    CLUTCH_MAX_NM = 130.0       # S105_base.et MaxClutchTorque
    MASS_KG = 1000.0
    WHEEL_RADIUS_M = 0.295
    FIRST_GEAR_RATIO = 19.95    # 3.8 x 5.25 (G19_0_STATIC_1.9.0.37.md 1.1)
    # second gear (index 3) by T0-ENG run 3's measured speed per 1000 rpm: 1.54871 m/s in first (index 2, which with
    # the 0.295 m wheel gives the same 19.95) and 2.772066 m/s in second (tools/ident/fit_steer_kappa.py's run-3 fit)
    SECOND_GEAR_RATIO = FIRST_GEAR_RATIO * 1.54871 / 2.772066
    ROLLING = 0.015
    BRAKE_N = 9000.0            # at brake 1, at the contact patches (about 0.9 g)
    SUBSTEPS = 40               # integration substeps per 1/30 s
    # full-throttle torque [Nm] by rpm: peak torque 72.8 Nm at 3000 (Engine_S_742.10.conf), 65.9 Nm at the run
    # header's peak power rpm 4800 (33.1 kW, MaxPower), 0 at the run header's rpm_max 8000
    TORQUE_CURVE = ((0.0, 0.0), (500.0, 40.0), (800.0, 52.0), (3000.0, 72.8), (4800.0, 65.9), (5800.0, 55.0),
                    (7500.0, 20.0), (8000.0, 0.0))

    def __init__(self, rpm=RPM_IDLE, speed_mps=0.0, locked=False, j_engine=None, ratio=FIRST_GEAR_RATIO):
        self.j_engine = j_engine or self.J_ENGINE
        self.ratio = ratio
        self.we = rpm / RPM_PER_RADS
        self.v = speed_mps
        self.on = True
        self.locked = locked
        self.j_vehicle = self.MASS_KG * self.WHEEL_RADIUS_M ** 2 / self.ratio ** 2

    @property
    def rpm(self):
        return self.we * RPM_PER_RADS

    @property
    def kmh(self):
        return self.v * 3.6

    def curve(self, rpm):
        for (r0, t0), (r1, t1) in zip(self.TORQUE_CURVE, self.TORQUE_CURVE[1:]):
            if rpm <= r1:
                return t0 + (t1 - t0) * max(rpm - r0, 0.0) / (r1 - r0)
        return 0.0

    def engine_torque(self, throttle):
        friction = 4.0 + 0.0015 * self.rpm if self.we > 0 else 0.0
        if not self.on:
            return -friction
        governor = min(max(self.GOVERNOR_GAIN * (self.RPM_IDLE - self.rpm) / self.RPM_IDLE, 0.0), self.GOVERNOR_MAX)
        return max(throttle, governor) * self.curve(self.rpm) - friction

    def load_torque(self, brake):
        force = self.ROLLING * self.MASS_KG * 9.81 + 0.5 * 1.2 * 0.7 * self.v ** 2 + brake * self.BRAKE_N
        return force * self.WHEEL_RADIUS_M / self.ratio

    def force_start(self):
        """ForceStartEngine: the engine runs again at idle or above."""
        self.on = True
        self.we = max(self.we, self.RPM_IDLE / RPM_PER_RADS)

    def step(self, dt, throttle, brake, clutch):
        substeps = max(1, int(round(dt / STEP_DT * self.SUBSTEPS)))
        h = dt / substeps
        capacity = clutch * self.CLUTCH_MAX_NM
        for _ in range(substeps):
            wg = self.v / self.WHEEL_RADIUS_M * self.ratio
            te, load = self.engine_torque(throttle), self.load_torque(brake)
            if capacity > 0 and (self.locked or abs(self.we - wg) < 0.5):
                accel = (te - load) / (self.j_engine + self.j_vehicle)
                if abs(te - self.j_engine * accel) <= capacity:
                    self.locked = True
                    wg = max(wg + accel * h, 0.0)
                    self.we, self.v = wg, wg / self.ratio * self.WHEEL_RADIUS_M
                    self.stall_check()
                    continue
            self.locked = False
            slip = self.we - wg
            tc = capacity * (1.0 if slip > 0 else -1.0 if slip < 0 else 0.0)
            we_new = max(self.we + (te - tc) / self.j_engine * h, 0.0)
            wg_new = max(wg + (tc - load) / self.j_vehicle * h, 0.0)
            if capacity > 0 and slip * (we_new - wg_new) < 0:
                self.locked = True                      # the slip changed sign within the substep: lock
                we_new = wg_new = 0.5 * (we_new + wg_new)
            self.we, self.v = we_new, wg_new / self.ratio * self.WHEEL_RADIUS_M
            self.stall_check()

    def stall_check(self):
        if self.on and self.rpm < self.RPM_STALL:
            self.on = False


def drive(clutch, car, seconds, throttle, brake=0.0, pattern=(1,), brake_after=None):
    """Mode A over the toy as the probe actuator runs it: each callback (covering pattern[i] steps) reads the
    previous state (rpm, engine_on), the launch clutch gives the applied clutch for first gear with the handbrake
    released and a commanded clutch 1, a requested restart is ForceStartEngine before the step; returns the rows
    (t, c, rpm, engine_on, km/h)."""
    rows, t, i = [], 0.0, 0
    while t < seconds - 1e-9:
        covered = pattern[i % len(pattern)]
        i += 1
        dt = covered * STEP_DT
        thr, brk = (0.0, brake) if brake_after is not None and t >= brake_after else (throttle, 0.0)
        applied = clutch.step(dt, True, False, car.rpm, car.on, 1.0, False)
        if clutch.consume_restart():
            car.force_start()
        car.step(dt, thr, brk, applied)
        t += dt
        rows.append((t, applied, car.rpm, car.on, car.kmh))
    return rows


def reversals(values, eps=1e-4):
    """Direction changes of a sequence (steps smaller than eps ignored)."""
    deltas = [b - a for a, b in zip(values, values[1:]) if abs(b - a) > eps]
    return sum(1 for a, b in zip(deltas, deltas[1:]) if a * b < 0)


class LaunchClutchStepTest(unittest.TestCase):
    """The per-step rules of DC_LaunchClutch.Step, on the mirror."""

    def test_off_passes_the_commanded_clutch(self):
        lc = LaunchClutch(mode="off")
        for commanded, expected in ((1.0, 1.0), (0.0, 0.0), (0.4, 0.4), (1.7, 1.0), (-0.2, 0.0)):
            self.assertEqual(lc.step(STEP_DT, True, False, 800.0, True, commanded, False), expected)
        self.assertEqual(lc.step(STEP_DT, False, True, 0.0, False, 1.0, False), 1.0)   # even neutral, handbrake, off
        self.assertFalse(lc.consume_restart())                                         # never restarts ...
        self.assertEqual((lc.stalls, lc.restarts), (1, 0))                             # ... but counts the stall

    def test_open_in_neutral_with_the_handbrake_and_during_a_dip(self):
        lc = ramp()
        for _ in range(60):
            lc.step(STEP_DT, True, False, 2500.0, True, 1.0, False)
        self.assertEqual(lc.coupling, 1.0)
        for args in ((False, False, False), (True, True, False), (True, False, True)):
            lc.coupling = 1.0
            in_gear, handbrake, dip = args
            self.assertEqual(lc.step(STEP_DT, in_gear, handbrake, 2500.0, True, 1.0, dip), 0.0, args)
            self.assertEqual(lc.coupling, 0.0, args)
        # after the dip it ramps again from 0 at the couple rate
        self.assertAlmostEqual(lc.step(STEP_DT, True, False, 2500.0, True, 1.0, False), STEP_DT / 0.85)

    def test_rate_limits_target_and_upper_bound(self):
        lc = ramp()
        self.assertEqual(lc.step(STEP_DT, True, False, 800.0, True, 1.0, False), 0.0)   # idle never bites
        self.assertEqual(lc.step(STEP_DT, True, False, 1100.0, True, 1.0, False), 0.0)  # engage_rpm: target 0
        c = lc.step(STEP_DT, True, False, 1700.0, True, 1.0, False)                    # full_rpm: target 1
        self.assertAlmostEqual(c, STEP_DT / 0.85)                                       # rises at most dt / couple
        ups = [lc.step(STEP_DT, True, False, 3000.0, True, 1.0, False) for _ in range(40)]
        self.assertEqual(1 + ups.index(1.0) + 1, math.ceil(0.85 / STEP_DT))           # 0 -> 1 in ceil(0.85 / dt) steps
        self.assertEqual(ups[-1], 1.0)
        down = lc.step(STEP_DT, True, False, 1650.0, True, 1.0, False)                 # in the band: falls at most
        self.assertAlmostEqual(down, 1.0 - STEP_DT / 0.45)                              # dt / uncouple
        lc.coupling = 1.0
        self.assertEqual(lc.step(STEP_DT, True, False, 1100.0, True, 1.0, False), 0.0)  # at engage_rpm: open at once
        self.assertEqual(lc.guard_drops, 0)                                             # (not a guard drop)
        lc.coupling = 0.2
        self.assertAlmostEqual(lc.step(STEP_DT, True, False, 1400.0, True, 1.0, False), 0.2 + STEP_DT / 0.85)
        lc.coupling = 0.9
        self.assertAlmostEqual(lc.step(STEP_DT, True, False, 1400.0, True, 1.0, False), 0.9 - STEP_DT / 0.45)
        lc.coupling = 0.5
        self.assertAlmostEqual(lc.step(STEP_DT, True, False, 1400.0, True, 1.0, False), 0.5)   # at the target
        self.assertAlmostEqual(lc.step(STEP_DT, True, False, 1400.0, True, 0.3, False), 0.3)   # commanded bounds it
        self.assertAlmostEqual(lc.coupling, 0.5)                                                # c is not lowered
        # a step that covers 3 engine steps moves c 3 times as far (dt = covered steps x step dt)
        lc.coupling = 0.0
        self.assertAlmostEqual(lc.step(3 * STEP_DT, True, False, 3000.0, True, 1.0, False), 3 * STEP_DT / 0.85)

    def test_stall_guard_opens_at_once(self):
        lc = ramp()
        lc.coupling = 0.8
        self.assertEqual(lc.step(STEP_DT, True, False, 599.0, True, 1.0, False), 0.0)
        self.assertEqual(lc.guard_drops, 1)
        self.assertEqual(lc.step(STEP_DT, True, False, 599.0, True, 1.0, False), 0.0)
        self.assertEqual(lc.guard_drops, 1)                     # a drop counts once, when c was > 0

    def test_restart_cooldown_handbrake_and_switch(self):
        lc = ramp()
        lc.step(STEP_DT, True, False, 2000.0, True, 1.0, False)
        restarts = []
        for k in range(75):                                    # 2.5 s with the engine off, handbrake released
            lc.step(STEP_DT, True, False, 0.0, False, 1.0, False)
            if lc.consume_restart():
                restarts.append(k)
            self.assertFalse(lc.consume_restart())             # true once per request
        self.assertEqual(lc.stalls, 1)                         # one on -> off transition
        self.assertEqual(len(restarts), 3)                     # the first at once, then one per 1 s cooldown
        self.assertEqual(restarts[0], 0)
        for a, b in zip(restarts, restarts[1:]):
            self.assertIn(b - a, (30, 31))                     # 1 s of 1/30 s steps (float sums may need one more)
        self.assertEqual(lc.restarts, 3)
        held = ramp()
        for _ in range(60):
            held.step(STEP_DT, True, True, 0.0, False, 1.0, False)   # handbrake commanded: never
        self.assertFalse(held.consume_restart())
        self.assertEqual(held.restarts, 0)
        no_restart = ramp(restart_on_stall=False)
        no_restart.step(STEP_DT, True, False, 0.0, False, 1.0, False)
        self.assertFalse(no_restart.consume_restart())
        no_cooldown = ramp(restart_cooldown_s=0.0)             # cooldown 0: every step with the engine off
        for _ in range(5):
            no_cooldown.step(STEP_DT, True, False, 0.0, False, 1.0, False)
        self.assertEqual(no_cooldown.restarts, 5)
        lc.reset()                                             # an episode start clears the cooldown
        lc.step(STEP_DT, True, False, 0.0, False, 1.0, False)
        self.assertTrue(lc.consume_restart())

    def test_cooldown_holds_in_callbacks_of_varying_size(self):
        # Codex round 1: the cooldown counted the current call's steps before deciding, so a 3-step call after 1-step
        # ones gave restarts 0.967 s apart. It is the time elapsed at the restart point now, never below the cooldown
        for pattern in ((1,), (2,), (3,), (1, 3), (3, 1, 1, 2), (1, 1, 2, 1, 3, 1, 2, 3, 1, 1)):
            lc = ramp()
            t, times, i = 0.0, [], 0
            while t < 6.0:
                covered = pattern[i % len(pattern)]
                i += 1
                lc.step(covered * STEP_DT, True, False, 0.0, False, 1.0, False)
                if lc.consume_restart():
                    times.append(t)
                t += covered * STEP_DT
            gaps = [b - a for a, b in zip(times, times[1:])]
            self.assertGreaterEqual(len(times), 5, pattern)
            self.assertTrue(all(g >= 1.0 - 1e-9 for g in gaps), (pattern, gaps))
            self.assertTrue(all(g <= 1.0 + 3 * STEP_DT + 1e-9 for g in gaps), (pattern, gaps))   # at the first chance

    def test_check_engine(self):
        # DC_LaunchClutchConfig.CheckEngine on the S105's run header values (vehicle.rpm_idle 800, rpm_max 8000)
        defaults = dict(LaunchClutch.DEFAULTS, mode="ramp")
        self.assertTrue(check_engine(defaults, 800.0, 8000.0))
        for change, idle, rpm_max in (({"engage_rpm": 800.0}, 800.0, 8000.0), ({"full_rpm": 4000.0}, 800.0, 8000.0),
                                      ({}, 800.0, 0.0), ({}, 1200.0, 8000.0), ({}, 800.0, 3000.0),
                                      ({"engage_rpm": 9000.0, "full_rpm": 10000.0}, 800.0, 8000.0)):
            self.assertFalse(check_engine(dict(defaults, **change), idle, rpm_max), (change, idle, rpm_max))
        self.assertTrue(check_engine(dict(defaults, full_rpm=3999.0), 800.0, 8000.0))
        self.assertTrue(check_engine(dict(defaults, mode="off", engage_rpm=9000.0, full_rpm=10000.0), 800.0, 0.0))

    def test_engine_off_is_never_coupled(self):
        lc = ramp(restart_on_stall=False)
        lc.coupling = 1.0
        self.assertEqual(lc.step(STEP_DT, True, False, 3000.0, False, 1.0, False), 0.0)   # rpm = the wheels' drag

    def test_slip_span(self):
        lc = ramp()
        for _ in range(15):
            lc.step(STEP_DT, True, False, 1400.0, True, 1.0, False)        # target 0.5: slips
        self.assertAlmostEqual(lc.slip_s, 15 * STEP_DT)
        lc.step(STEP_DT, True, True, 1400.0, True, 1.0, False)             # handbrake ends the span
        self.assertEqual(lc.slip_s, 0.0)
        self.assertAlmostEqual(lc.max_slip_s, 15 * STEP_DT)
        lc.max_slip_s = lc.slip_s                                          # ResetMaxSlip (a probe phase)
        self.assertEqual(lc.max_slip_s, 0.0)
        self.assertAlmostEqual(lc.max_slip_total_s, 15 * STEP_DT)


class ToyPowertrainTest(unittest.TestCase):
    """The defaults on the TOY powertrain (catches control-loop bugs; does not predict 1.9)."""

    def test_off_stalls_like_the_1_9_play(self):
        # launch 20261106T052424Z-wb-addc: first gear, throttle 0.5, clutch 1 from 800 rpm -> 528 rpm, then off
        car, lc = ToyPowertrain(), LaunchClutch(mode="off")
        rows = drive(lc, car, 3.0, 0.5)
        self.assertGreater(rows[0][2], 300.0)
        self.assertLess(rows[0][2], 700.0)                       # one step after the release (the Play: 528 rpm)
        first_off = next(i for i, r in enumerate(rows) if not r[3])
        self.assertLessEqual(first_off, 2)                       # off within two steps, as on 1.9
        self.assertFalse(rows[-1][3])
        self.assertLess(max(r[4] for r in rows), 3.0)            # a lurch at most (the Play: <= 0.67 km/h) ...
        self.assertEqual(rows[-1][4], 0.0)                       # ... then standing with a dead engine
        self.assertEqual((lc.stalls, lc.restarts), (1, 0))

    def test_ramp_launches_without_stalling(self):
        for throttle in (0.3, 0.5, 1.0):
            car, lc = ToyPowertrain(), ramp()
            rows = drive(lc, car, 8.0, throttle)
            self.assertTrue(all(r[3] for r in rows), throttle)                     # never stalls
            self.assertEqual((lc.stalls, lc.restarts, lc.guard_drops), (0, 0, 0), throttle)
            self.assertGreater(min(r[2] for r in rows), 800.0 * 0.95, throttle)    # never sags below idle
            moving = next(r[0] for r in rows if r[4] > 1.0)
            self.assertLess(moving, 1.5, throttle)                                 # moving within 1.5 s
            self.assertGreater(rows[-1][4], 10.0, throttle)
            cs = [r[1] for r in rows]
            self.assertLessEqual(reversals(cs), 3, throttle)                       # no sustained limit cycle in c
            self.assertTrue(all(c == 1.0 for c in cs[-60:]), throttle)             # fully coupled at the end
            self.assertGreater(lc.max_slip_total_s, 0.5, throttle)                 # it did slip: a ramp, no dump

    def test_ramp_in_callbacks_of_one_to_three_steps(self):
        # a capture callback covers 1 to 3 engine steps (dc-2.0 D-A): dt = covered steps x step dt
        pattern = (1, 1, 2, 1, 3, 1, 2, 3, 1, 1)
        for throttle in (0.3, 0.5, 1.0):
            single = drive(ramp(), ToyPowertrain(), 8.0, throttle)
            lc = ramp()
            rows = drive(lc, ToyPowertrain(), 8.0, throttle, pattern=pattern)
            self.assertTrue(all(r[3] for r in rows), throttle)
            self.assertEqual((lc.stalls, lc.guard_drops), (0, 0), throttle)
            self.assertLessEqual(reversals([r[1] for r in rows]), 6, throttle)    # a 3-step callback triples the gain
            self.assertTrue(all(r[1] == 1.0 for r in rows if r[0] > 6.0), throttle)
            coupled = next(r[0] for r in rows if r[1] == 1.0)
            coupled_single = next(r[0] for r in single if r[1] == 1.0)
            self.assertLess(abs(coupled - coupled_single), 0.5, throttle)          # world time, not callbacks

    def test_braking_to_a_stop_in_first_gear(self):
        car, lc = ToyPowertrain(), ramp()
        rows = drive(lc, car, 16.0, 0.5, brake=0.6, brake_after=6.0)
        self.assertTrue(all(r[3] for r in rows))                     # no stall on the way down
        self.assertEqual((lc.stalls, lc.restarts, lc.guard_drops), (0, 0, 0))
        self.assertEqual(rows[-1][4], 0.0)                           # stopped ...
        self.assertEqual(rows[-1][1], 0.0)                           # ... with the clutch open ...
        self.assertAlmostEqual(rows[-1][2], 800.0, delta=60.0)       # ... and the engine at idle
        self.assertGreater(max(r[4] for r in rows), 20.0)
        # the same stop with the clutch held at 1 (mode off) from 20 km/h, coupled: the engine dies
        car = ToyPowertrain(rpm=20.0 / 3.6 / ToyPowertrain.WHEEL_RADIUS_M * ToyPowertrain.FIRST_GEAR_RATIO
                            * RPM_PER_RADS, speed_mps=20.0 / 3.6, locked=True)
        off = LaunchClutch(mode="off")
        rows = drive(off, car, 8.0, 0.0, brake=0.6, brake_after=0.0)
        self.assertFalse(rows[-1][3])
        self.assertEqual(off.stalls, 1)

    def full_brake(self, ratio, kmh, pattern):
        """Brake 1, throttle 0 from kmh coupled in a gear, the clutch as cruising there leaves it."""
        v = kmh / 3.6
        rpm = v / ToyPowertrain.WHEEL_RADIUS_M * ratio * RPM_PER_RADS
        car, lc = ToyPowertrain(rpm=rpm, speed_mps=v, locked=True, ratio=ratio), ramp()
        lc.engine_seen = 1
        lc.coupling = 0.0 if rpm <= 1100.0 else min((rpm - 1100.0) / 600.0, 1.0)
        return drive(lc, car, 6.0, 0.0, brake=1.0, brake_after=0.0, pattern=pattern), lc

    def test_full_brake_to_a_stop_never_stalls(self):
        # Codex round 1: a full brake stalled the toy (606 -> 504 rpm at c 0.333) while the release below the band was
        # rate-limited. Now the clutch opens at engage_rpm: from 16 and 40 km/h, in first and second gear, in
        # callbacks of 1, 2 and 3 steps and mixes, with the defaults: no stall, so no restart
        patterns = ((1,), (2,), (3,), (1, 1, 2, 1, 3, 1, 2, 3, 1, 1), (3, 1, 2))
        for ratio, gear in ((ToyPowertrain.FIRST_GEAR_RATIO, "first"), (ToyPowertrain.SECOND_GEAR_RATIO, "second")):
            for kmh in (16.0, 40.0):
                for pattern in patterns:
                    label = (gear, kmh, pattern)
                    rows, lc = self.full_brake(ratio, kmh, pattern)
                    self.assertTrue(all(r[3] for r in rows), label)
                    self.assertEqual((lc.stalls, lc.restarts), (0, 0), label)
                    self.assertGreater(min(r[2] for r in rows), ToyPowertrain.RPM_STALL, label)
                    self.assertEqual(rows[-1][4], 0.0, label)                     # stopped, clutch open
                    self.assertEqual(rows[-1][1], 0.0, label)
        # Codex's case: 6 s at throttle 0.5 from standstill, then a full brake
        for pattern in patterns:
            lc = ramp()
            rows = drive(lc, ToyPowertrain(), 16.0, 0.5, brake=1.0, brake_after=6.0, pattern=pattern)
            self.assertTrue(all(r[3] for r in rows), pattern)
            self.assertEqual((lc.stalls, lc.restarts), (0, 0), pattern)
            self.assertEqual(rows[-1][4], 0.0, pattern)

    def test_a_stall_is_restarted(self):
        car, lc = ToyPowertrain(), ramp()
        drive(lc, car, 6.0, 0.5)
        car.on = False                                               # a stall from outside the clutch
        rows = drive(lc, car, 4.0, 0.5)
        self.assertEqual((lc.stalls, lc.restarts), (1, 1))
        self.assertTrue(all(r[3] for r in rows))                     # restarted before the next step
        self.assertTrue(all(r[1] == 1.0 for r in rows[-30:]))        # and coupled again
        self.assertGreater(rows[-1][4], 20.0)

    def test_loop_gain_margin(self):
        # The sampled loop at the bite point: c follows (rpm - engage) / (full - engage) one step late, and rpm answers
        # c x MaxClutchTorque through the engine inertia. Gain = dt x (60 / 2 pi) / J x 130 / (full - engage); below
        # 1 there is no overshoot. With the toy's (and the Play's estimated) 0.12 kg m2 it is about 0.58. A much
        # lighter 1.9 engine (below about 0.07 kg m2) would make c ring while it slips: check the Workbench steps for
        # it (cmd_clutch and rpm in the first second of every launch)
        defaults = LaunchClutch.DEFAULTS
        gain = STEP_DT * RPM_PER_RADS / ToyPowertrain.J_ENGINE * ToyPowertrain.CLUTCH_MAX_NM / (
            defaults["full_rpm"] - defaults["engage_rpm"])
        self.assertLess(gain, 0.65)
        for j_engine in (0.08, 0.2, 0.4):
            car, lc = ToyPowertrain(j_engine=j_engine), ramp()
            rows = drive(lc, car, 10.0, 0.5)
            self.assertTrue(all(r[3] for r in rows), j_engine)
            self.assertLessEqual(reversals([r[1] for r in rows]), 3, j_engine)


class LaunchClutchTiesTest(unittest.TestCase):
    """The mirror, the defaults and the three actuation paths against the Enforce sources."""

    @classmethod
    def setUpClass(cls):
        cls.helper = read(HELPER_SOURCE)
        cls.probe = read(PROBE_SOURCE)
        cls.llc = read(LLC_SOURCE)
        cls.recorder = read(RECORDER_SOURCE)
        cls.expert = read(EXPERT_SOURCE)

    def test_mirror_follows_the_enforce_step(self):
        step = class_method(self.helper, "DC_LaunchClutch", "Step")
        for statement in (
                "float commanded = Math.Clamp(commandedClutch, 0, 1); float slice = Math.Max(dt, 0);",
                "if (m_iEngineSeen == 1 && !engineOn) m_iStalls++; m_iEngineSeen = 0; if (engineOn) m_iEngineSeen = 1;",
                # the cooldown is decided on the time before this call and advanced by it afterwards
                "if (!m_Config.IsActive()) return commanded; bool cooled = m_fSinceRestartS < 0 || m_fSinceRestartS >= "
                "m_Config.m_fRestartCooldownS;",
                "if (!engineOn && !handbrake && m_Config.m_bRestartOnStall && cooled) { m_bRestartDue = true; "
                "m_iRestarts++; m_fSinceRestartS = 0; m_fCoupling = 0; }",
                "bool driving = inGear && !handbrake && !dip && engineOn; if (m_Config.IsSlip()) { StepSlip(slice, "
                "driving, rpm, drivelineRpm, throttle > 0 && brake <= 0, gear); } else if (!driving) { m_fCoupling = 0; "
                "} "
                "else if (rpm < m_Config.m_fStallGuardRpm) { if (m_fCoupling > 0) m_iGuardDrops++; m_fCoupling = 0; } "
                "else if (rpm <= m_Config.m_fEngageRpm) { m_fCoupling = 0; } "
                "else { float target = Math.Clamp((rpm - m_Config.m_fEngageRpm) / (m_Config.m_fFullRpm - "
                "m_Config.m_fEngageRpm), 0, 1); if (target > m_fCoupling) m_fCoupling = Math.Min(target, m_fCoupling + "
                "slice / m_Config.m_fCoupleTimeS); else m_fCoupling = Math.Max(target, m_fCoupling - slice / "
                "m_Config.m_fUncoupleTimeS); }",
                "if (driving && m_fCoupling > 0 && m_fCoupling < 1) m_fSlipS += slice; else m_fSlipS = 0; "
                "if (m_fSinceRestartS >= 0) m_fSinceRestartS += slice;",
                "m_fMaxSlipS = Math.Max(m_fMaxSlipS, m_fSlipS); m_fMaxSlipTotalS = Math.Max(m_fMaxSlipTotalS, "
                "m_fSlipS); return Math.Min(commanded, m_fCoupling);"):
            self.assertIn(statement, step)
        self.assertIn("if (!m_bRestartDue) return false; m_bRestartDue = false; return true;",
                      class_method(self.helper, "DC_LaunchClutch", "ConsumeRestart"))
        self.assertIn("m_fCoupling = 0; m_fSinceRestartS = -1; m_bRestartDue = false; m_fSlipS = 0; m_bLocked = false; "
                      "m_iAgreeCalls = 0; m_fSyncLeftS = 0; m_iLastGear = -1; m_fCallSteps = 1; ForgetRpm();",
                      class_method(self.helper, "DC_LaunchClutch", "Reset"))
        self.assertIn("m_fLastRpm = -1; m_fLastMeanRpm = -1; m_iHistory = 0; m_fLastSliceS = 0; m_fPrevSliceS = 0;",
                      class_method(self.helper, "DC_LaunchClutch", "ForgetRpm"))
        self.assertIn("float Step(float dt, bool inGear, bool handbrake, float rpm, bool engineOn, float commandedClutch, "
                      "bool dip, float drivelineRpm = -1, float throttle = 1, float brake = 0, int gear = -1)", self.helper)
        self.assertIn("m_fMaxSlipS = m_fSlipS;", class_method(self.helper, "DC_LaunchClutch", "ResetMaxSlip"))
        # the helper calls no engine API and writes no capture stream or event (dc-2.0 is frozen)
        code = code_only(self.helper)
        for engine in ("ForceStartEngine", "SetClutch", "EngineGetRPM", "EngineIsOn", "LogEvent", "Print("):
            self.assertNotIn(engine, code)
        self.helper.encode("ascii")                     # DC_TextFile.Read (the expert hashes it) needs 7-bit ASCII
        self.assertNotIn("\r", self.helper)

    def test_defaults_agree_everywhere(self):
        consts = enforce_constants(self.helper, "DC_LaunchClutchConfig")
        enforce_defaults = {"engage_rpm": consts["DEFAULT_ENGAGE_RPM"], "full_rpm": consts["DEFAULT_FULL_RPM"],
                            "couple_time_s": consts["DEFAULT_COUPLE_TIME_S"],
                            "uncouple_time_s": consts["DEFAULT_UNCOUPLE_TIME_S"],
                            "stall_guard_rpm": consts["DEFAULT_STALL_GUARD_RPM"],
                            "restart_on_stall": consts["DEFAULT_RESTART_ON_STALL"],
                            "restart_cooldown_s": consts["DEFAULT_RESTART_COOLDOWN_S"],
                            "slip_kp": consts["DEFAULT_SLIP_KP"], "slip_ki": consts["DEFAULT_SLIP_KI"],
                            "lock_tolerance_rpm": consts["DEFAULT_LOCK_TOLERANCE_RPM"]}
        mirror = dict(LaunchClutch.DEFAULTS)
        self.assertEqual(mirror.pop("mode"), "off")                 # a DC_LaunchClutchConfig starts off ...
        self.assertIn("string m_sMode = MODE_OFF;", self.helper)
        self.assertEqual(enforce_defaults, mirror)
        self.assertIn("config.m_sMode = MODE_RAMP;", class_method(self.helper, "DC_LaunchClutchConfig", "Ramp"))
        base_keys = list(fit.LAUNCH_CLUTCH_KEYS)
        ramp_defaults = dict(((k, enforce_defaults[k]) for k in base_keys[1:]), mode="ramp")   # Ramp(): DD-14c's
        self.assertEqual(fit.PLACEHOLDER_LAUNCH_CLUTCH, ramp_defaults)
        # ... Slip() (DD-14d) is the probe kit's default and the 1.9.0.37 card's: full_rpm the held rpm
        self.assertEqual(class_method(self.helper, "DC_LaunchClutchConfig", "Slip"),
                         ") { DC_LaunchClutchConfig config = new DC_LaunchClutchConfig(); config.m_sMode = MODE_SLIP; "
                         "config.m_fFullRpm = DEFAULT_SLIP_FULL_RPM; return config;")
        slip_defaults = dict(enforce_defaults, mode="slip", full_rpm=consts["DEFAULT_SLIP_FULL_RPM"])
        self.assertEqual(LaunchClutch.slip().config, slip_defaults)
        self.assertEqual(LaunchClutch.SLIP_FULL_RPM, consts["DEFAULT_SLIP_FULL_RPM"])
        self.assertEqual(fit.PLACEHOLDER_SLIP_LAUNCH_CLUTCH, slip_defaults)
        with open(CARD, "r", encoding="ascii") as f:
            self.assertEqual(json.load(f)["gearbox"]["launch_clutch"], slip_defaults)
        attrs = attribute_defaults(self.probe)
        names = {"mode": "m_sLaunchClutchMode", "engage_rpm": "m_fLaunchEngageRpm", "full_rpm": "m_fLaunchHoldRpm",
                 "couple_time_s": "m_fLaunchCoupleS", "uncouple_time_s": "m_fLaunchUncoupleS",
                 "stall_guard_rpm": "m_fLaunchStallGuardRpm", "restart_on_stall": "m_bLaunchRestartOnStall",
                 "restart_cooldown_s": "m_fLaunchRestartCooldownS", "slip_kp": "m_fLaunchSlipKp",
                 "slip_ki": "m_fLaunchSlipKi", "lock_tolerance_rpm": "m_fLaunchLockTolRpm"}
        for key, name in names.items():
            value = slip_defaults[key]
            if isinstance(value, bool):
                self.assertEqual(attrs[name], "1" if value else "0", name)
            elif isinstance(value, str):
                self.assertEqual(attrs[name], value, name)
            else:
                self.assertEqual(float(attrs[name]), value, name)
        # DD-14d round 1 (Codex): full_rpm per mode, so mode ramp is exactly DD-14c's ramp (engage 1100, full 1700)
        self.assertEqual(float(attrs["m_fLaunchFullRpm"]), ramp_defaults["full_rpm"])
        build = enforce_body(self.probe, "DC_ProbeSequences", "BuildLaunchClutchConfig")
        self.assertIn("config.m_fFullRpm = m_fLaunchFullRpm; if (config.IsSlip()) config.m_fFullRpm = m_fLaunchHoldRpm;",
                      build)
        # full_rpm below the shift policy's downshift rpm (attribute and actuator default): fully coupled wherever
        # the policy drives; the idle (800 rpm, run header vehicle.rpm_idle) below engage_rpm never bites
        self.assertLess(ramp_defaults["full_rpm"], float(attrs["m_fShiftDownRpm"]))
        self.assertLess(slip_defaults["full_rpm"], float(attrs["m_fShiftDownRpm"]))
        self.assertIn("static const float DEFAULT_DOWNSHIFT_RPM = %g;" % float(attrs["m_fShiftDownRpm"]), self.probe)
        self.assertGreater(ramp_defaults["engage_rpm"], ToyPowertrain.RPM_IDLE)
        self.assertLess(ramp_defaults["stall_guard_rpm"], ToyPowertrain.RPM_IDLE)
        # the ranges of DC_LaunchClutchConfig.Check are the Python validator's
        self.assertEqual((consts["RPM_MAX"], consts["TIME_MAX_S"]), (fit.LAUNCH_RPM_MAX, fit.LAUNCH_TIME_MAX_S))
        check = class_method(self.helper, "DC_LaunchClutchConfig", "Check")
        for rule in ("if (m_fEngageRpm <= 0 || m_fFullRpm <= m_fEngageRpm || m_fFullRpm > RPM_MAX)",
                     "if (m_fStallGuardRpm < 0 || m_fStallGuardRpm >= m_fEngageRpm)",
                     "if (m_fCoupleTimeS <= 0 || m_fCoupleTimeS > TIME_MAX_S || m_fUncoupleTimeS <= 0 || "
                     "m_fUncoupleTimeS > TIME_MAX_S)",
                     "if (m_fRestartCooldownS < 0 || m_fRestartCooldownS > TIME_MAX_S)"):
            self.assertIn(rule, check)
        self.assertIn("if (m_fSlipKp < 0 || m_fSlipKp > SLIP_KP_MAX || m_fSlipKi <= 0 || m_fSlipKi > SLIP_KI_MAX)", check)
        self.assertIn("if (m_fLockToleranceRpm < 0 || m_fLockToleranceRpm > LOCK_TOLERANCE_MAX_RPM)", check)
        self.assertEqual(fit.LAUNCH_CLUTCH_MODES, ("off", "ramp", "slip"))
        self.assertIn("if (m_sMode != MODE_OFF && m_sMode != MODE_RAMP && m_sMode != MODE_SLIP)", check)
        self.assertIn('static const string MODE_OFF = "off";', self.helper)
        self.assertIn('static const string MODE_RAMP = "ramp";', self.helper)
        self.assertIn('static const string MODE_SLIP = "slip";', self.helper)

    def test_check_engine_follows_the_enforce_source(self):
        check = class_method(self.helper, "DC_LaunchClutchConfig", "CheckEngine")
        for statement in ("if (!IsActive()) { failure = string.Empty; return true; }",
                          "if (m_fEngageRpm <= rpmIdle) {",
                          "float limit = FULL_RPM_MAX_FRACTION * rpmMax; if (rpmMax <= 0 || m_fFullRpm >= limit) {"):
            self.assertIn(statement, check)
        self.assertEqual(enforce_constants(self.helper, "DC_LaunchClutchConfig")["FULL_RPM_MAX_FRACTION"], 0.5)
        init = enforce_body(self.llc, "DC_LLCv1", "Init")
        self.assertIn("if (card.m_sGearbox == DC_VehicleCard.GEARBOX_SCRIPTED && !card.m_LaunchClutch.CheckEngine("
                      "m_Simulation.EngineGetRPMIdle(), m_Simulation.EngineGetRPMMax(), failure))", init)
        self.assertLess(init.index("CheckEgoGearbox(card, actuator, failure)"), init.index("CheckEngine("))

    def test_probe_validates_only_the_configuration_it_drives_with(self):
        # Codex round 2: Init validated the default ramp before OnProbeStart applied the attributes, so an engine the
        # default does not fit (idle 600, max 3000 rpm: full_rpm 1700 >= 0.5 x 3000) refused the probe even in mode
        # off. Discovery (InitComponents) is now separate; the selected configuration is validated once, after the
        # shift policy it is checked against
        init_components = enforce_body(self.probe, "DC_ProbeActuator", "InitComponents")
        for absent in ("SetLaunchClutch", "CheckEngine", "Configure", "Ramp()"):
            self.assertNotIn(absent, init_components)
        self.assertIn("DetectGearbox(); return true;", init_components)
        init = enforce_body(self.probe, "DC_ProbeActuator", "Init")
        self.assertEqual(init, "notnull IEntity vehicle) { if (!InitComponents(vehicle)) return false; "
                               "DC_LaunchClutchConfig launchClutch = DC_LaunchClutchConfig.Slip(); return "
                               'SetLaunchClutch(launchClutch, "DC_LaunchClutchConfig defaults");')
        start = enforce_body(self.probe, "DC_ProbeSequences", "OnProbeStart")
        self.assertIn("if (m_Actuator.InitComponents(ego))", start)
        self.assertNotIn("m_Actuator.Init(", start)
        self.assertEqual(start.count("SetLaunchClutch("), 1)
        self.assertLess(start.index("m_Actuator.SetShiftPolicy("), start.index("SetLaunchClutch("))
        # the probes that drive with the defaults still get them validated (Init)
        for name in ("DC_ProbeImages.c", "DC_ProbeMarkers.c"):
            self.assertIn("m_Actuator.Init(m_Ego)", read(os.path.join(SCRIPTS, name)), name)
        # on that engine: mode off and a fitting custom ramp start, the default ramp is refused (and only it)
        defaults = dict(LaunchClutch.DEFAULTS, mode="ramp")
        off = dict(LaunchClutch.DEFAULTS, mode="off")
        custom = dict(defaults, engage_rpm=900.0, full_rpm=1400.0)
        self.assertTrue(probe_set_launch_clutch(off, True, 1800.0, 600.0, 3000.0))
        self.assertTrue(probe_set_launch_clutch(custom, True, 1800.0, 600.0, 3000.0))
        self.assertFalse(probe_set_launch_clutch(defaults, True, 1800.0, 600.0, 3000.0))
        self.assertTrue(probe_set_launch_clutch(defaults, True, 1800.0, 800.0, 8000.0))    # the S105
        self.assertTrue(probe_set_launch_clutch(dict(off, full_rpm=5000.0), True, 1800.0, 600.0, 3000.0))
        self.assertTrue(probe_set_launch_clutch(defaults, False, 1800.0, 600.0, 3000.0))   # automatic: not ramped
        # DD-14d: the slip defaults on the S105 (the policy's upshifts land at about 2514 and 2992 rpm, above 2200);
        # refused with an upshift landing below 2 x engage_rpm, a non-positive m/s per 1000 rpm, or full_rpm at the
        # downshift rpm
        slip = LaunchClutch.slip().config
        self.assertTrue(probe_set_launch_clutch(slip, True, 1800.0, 800.0, 8000.0))
        self.assertFalse(probe_set_launch_clutch(slip, True, 1800.0, 800.0, 8000.0, upshift_rpm=3800.0))
        self.assertFalse(probe_set_launch_clutch(slip, True, 1800.0, 800.0, 8000.0, mps_per_krpm=(1.5, 0.0, 4.1)))
        self.assertFalse(probe_set_launch_clutch(dict(slip, full_rpm=1800.0), True, 1800.0, 800.0, 8000.0))
        self.assertTrue(probe_set_launch_clutch(slip, False, 1800.0, 800.0, 8000.0, upshift_rpm=3800.0))
        # the mirror follows SetLaunchClutch
        set_launch = enforce_body(self.probe, "DC_ProbeActuator", "SetLaunchClutch")
        order = ["if (!config.Check(failure))", "if (config.IsActive() && m_bShiftPolicy && config.m_fFullRpm >= "
                 "m_fDownshiftRpm)", "if (config.IsSlip() && m_bShiftPolicy && !CheckSlipUpshifts(config)) return "
                 "false;", "if (m_bShiftPolicy && !config.CheckEngine(", "m_LaunchClutch.Configure(config);"]
        self.assertEqual([set_launch.index(x) for x in order], sorted(set_launch.index(x) for x in order))

    def test_card_downshifts_lie_above_the_band(self):
        # the 1.9.0.37 card's downshift speeds, by its scripted_mps_per_krpm, in the gear each shifts down from: about
        # 1804 and 1799 rpm, at least full_rpm (1400 in DD-14d's mode slip) + DOWNSHIFT_RPM_MARGIN 50, so llc_v1 never
        # drives a gear above the
        # first with the clutch slipping; Codex round 2's [2, 4] m/s would be about 721 and 959 rpm (refused)
        with open(CARD, "r", encoding="ascii") as f:
            gearbox = json.load(f)["gearbox"]
        per_krpm, down = gearbox["scripted_mps_per_krpm"], gearbox["downshift_mps"]
        margin = enforce_constants(self.helper, "DC_LaunchClutchConfig")["DOWNSHIFT_RPM_MARGIN"]
        rpms = [speed / per_krpm[slot + 1] * 1000.0 for slot, speed in enumerate(down)]
        self.assertTrue(all(r >= gearbox["launch_clutch"]["full_rpm"] + margin for r in rpms), rpms)
        low = [speed / per_krpm[slot + 1] * 1000.0 for slot, speed in enumerate((2.0, 4.0))]
        self.assertEqual([round(r) for r in low], [721, 959])
        self.assertTrue(all(r < gearbox["launch_clutch"]["full_rpm"] + margin for r in low))

    def test_run_header_block_is_the_cards_fields(self):
        # AppendHeaderLines writes launch_clutch.<card field> (fit_steer_kappa.launch_clutch_of reads them back)
        lines = class_method(self.helper, "DC_LaunchClutchConfig", "AppendHeaderLines")
        self.assertEqual(re.findall(r'lines\.Insert\(prefix \+ "\.(\w+)="', lines),
                         list(fit.LAUNCH_CLUTCH_KEYS + fit.LAUNCH_CLUTCH_SLIP_KEYS))
        self.assertEqual(fit.HEADER_LAUNCH_CLUTCH, "launch_clutch")
        gearbox_lines = enforce_body(self.probe, "DC_ProbeActuator", "AppendGearboxLines")
        self.assertIn('m_LaunchClutch.GetConfig().AppendHeaderLines(lines, "launch_clutch");', gearbox_lines)
        summary = enforce_body(self.probe, "DC_ProbeActuator", "AppendDriveSummaryLines")
        self.assertIn('m_LaunchClutch.AppendSummaryLines(lines, "launch_clutch");', summary)
        for key in ("stalls", "restarts", "guard_drops", "max_slip_s"):
            self.assertIn('lines.Insert(prefix + ".%s=" +' % key, self.helper)
            self.assertIn('phase.AddNote("%s:" +' % key, self.probe)
        self.assertIn('lines.Insert(prefix + ".lockups=" + m_iLockups.ToString());', self.helper)
        self.assertIn('lines.Insert("launch_clutch.mps_per_krpm=" + FloatsText(m_aLaunchMpsPerKrpm));', gearbox_lines)
        self.assertIn('lines.Insert("launch_clutch.active=" + DC_LaunchGuard.BoolText(m_bShiftPolicy && '
                      'm_LaunchClutch.IsActive()));', gearbox_lines)
        params = enforce_body(self.probe, "DC_ProbeSequences", "GetProbeParams")
        build = enforce_body(self.probe, "DC_ProbeSequences", "BuildLaunchClutchConfig")
        for name in ("m_sLaunchClutchMode", "m_fLaunchEngageRpm", "m_fLaunchFullRpm", "m_fLaunchCoupleS",
                     "m_fLaunchUncoupleS", "m_fLaunchStallGuardRpm", "m_bLaunchRestartOnStall",
                     "m_fLaunchRestartCooldownS", "m_fLaunchSlipKp", "m_fLaunchSlipKi", "m_fLaunchLockTolRpm",
                     "m_fLaunchHoldRpm"):
            self.assertIn('outKeyValues.Insert("%s=" +' % name, params)
            self.assertIn(name, build)
        self.assertIn('outKeyValues.Insert("m_sLaunchMpsPerKrpm=" + m_sLaunchMpsPerKrpm);', params)

    def test_probe_actuator_runs_it(self):
        apply_step = enforce_body(self.probe, "DC_ProbeActuator", "ApplyStep")
        call = ("if (m_bShiftPolicy) { bool inGear = m_iAppliedGear >= 0 && m_iAppliedGear != m_iGearNeutral; "
                "bool engineOn = m_bEngineOn || !m_bStateRead; bool dip = m_fAppliedClutch <= 0; float drivelineRpm = "
                "-1; int forwardSlot = m_iAppliedGear - m_iGearFirst; if (forwardSlot >= 0 && forwardSlot < "
                "m_aLaunchMpsPerKrpm.Count() && m_aLaunchMpsPerKrpm[forwardSlot] > 0) drivelineRpm = m_vVelLocal[2] / "
                "m_aLaunchMpsPerKrpm[forwardSlot] * 1000; m_fAppliedClutch = m_LaunchClutch.Step(timeSlice, inGear, "
                "m_bCmdHandbrake, m_fRpm, engineOn, m_fAppliedClutch, dip, drivelineRpm, m_fCmdThrottle, m_fCmdBrake, "
                "m_iAppliedGear); if (m_LaunchClutch.ConsumeRestart()) StartModeA(); }")
        self.assertIn(call, apply_step)
        # after the speed hold (m_fCmdThrottle, m_fCmdBrake), the shift policy and its dip, before the setters;
        # cmd_clutch is the value passed to SetClutch
        self.assertLess(apply_step.index("RunSpeedHold(timeSlice);"), apply_step.index(call))
        self.assertLess(apply_step.index("m_fDipLeftS -= timeSlice;"), apply_step.index(call))
        self.assertLess(apply_step.index(call), apply_step.index("m_Simulation.SetSteering(m_fCmdSteer);"))
        self.assertIn("m_Simulation.SetClutch(m_fAppliedClutch);", apply_step)
        self.assertIn("row.AddFloat(m_fAppliedClutch, 4);", enforce_body(self.probe, "DC_ProbeActuator",
                                                                          "AddStepValues"))
        self.assertIn("m_bStateRead = true;", enforce_body(self.probe, "DC_ProbeActuator", "ReadPost"))
        init = enforce_body(self.probe, "DC_ProbeActuator", "Init")
        self.assertIn("DC_LaunchClutchConfig launchClutch = DC_LaunchClutchConfig.Slip(); "
                      'return SetLaunchClutch(launchClutch, "DC_LaunchClutchConfig defaults");', init)
        start = enforce_body(self.probe, "DC_ProbeActuator", "StartModeA")
        self.assertIn("m_Controller.SetPersistentHandBrake(false); if (!m_Controller.IsEngineOn()) "
                      "m_Controller.ForceStartEngine();", start)
        set_launch = enforce_body(self.probe, "DC_ProbeActuator", "SetLaunchClutch")
        self.assertIn("if (!config.Check(failure))", set_launch)
        self.assertIn("if (config.IsActive() && m_bShiftPolicy && config.m_fFullRpm >= m_fDownshiftRpm)", set_launch)
        self.assertIn("if (m_bShiftPolicy && !config.CheckEngine(m_Simulation.EngineGetRPMIdle(), "
                      "m_Simulation.EngineGetRPMMax(), failure))", set_launch)
        self.assertIn("m_LaunchClutch.Configure(config);", set_launch)
        start_probe = enforce_body(self.probe, "DC_ProbeSequences", "OnProbeStart")
        self.assertLess(start_probe.index("m_Actuator.SetShiftPolicy("), start_probe.index("SetLaunchClutch("))
        self.assertIn('if (m_Actuator.SetLaunchClutch(launchClutch, "DC_ProbeSequences attributes")) '
                      "readyActuator = m_Actuator;", start_probe)

    def test_llc_and_mode_a_run_it(self):
        step = enforce_body(self.llc, "DC_LLCv1", "Step")
        call = ("float drivelineRpm = -1; if (m_iGearSlot < m_Card.m_aScriptedMpsPerKrpm.Count() && "
                "m_Card.m_aScriptedMpsPerKrpm[m_iGearSlot] > 0) drivelineRpm = speed / "
                "m_Card.m_aScriptedMpsPerKrpm[m_iGearSlot] * 1000; clutch = m_LaunchClutch.Step(sliceS, true, false, "
                "m_Simulation.EngineGetRPM(), m_Simulation.EngineIsOn(), clutch, clutch <= 0, drivelineRpm, throttle, "
                "brake, gear); if (m_LaunchClutch.ConsumeRestart()) m_Actuator.Start(); clutchRecorded = clutch; }")
        self.assertIn(call, step)
        scripted = step.index("if (m_Card.m_sGearbox == DC_VehicleCard.GEARBOX_SCRIPTED) {")
        dip = step.index("if (m_iDipStepsLeft > 0) { clutch = 0;")
        self.assertLess(scripted, dip)
        self.assertLess(dip, step.index(call))                      # after the dip, inside the scripted block
        self.assertLess(step.index(call), step.index("m_Actuator.Apply(m_fSteerOut, throttle, brake, false, clutch, "
                                                     "gear);"))      # before the setters; llc_clutch = passed
        self.assertIn("llc.m_fClutch = clutchRecorded;", step)
        self.assertIn("m_LaunchClutch.Configure(card.m_LaunchClutch);", enforce_body(self.llc, "DC_LLCv1", "Init"))
        self.assertIn("m_LaunchClutch.Reset();", enforce_body(self.llc, "DC_LLCv1", "Reset"))
        # Mode A: Start is the restart (ForceStartEngine when the engine is off), Apply passes the clutch it is given
        start = enforce_body(self.recorder, "DC_ActuatorModeA", "Start")
        self.assertIn("m_Controller.SetPersistentHandBrake(false); if (m_Controller.IsEngineOn()) return; "
                      "m_Controller.ForceStartEngine(); m_iEngineForceStarts++;", start)
        self.assertIn("LogLevel.WARNING", start)
        self.assertIn("m_Simulation.SetClutch(Math.Clamp(clutch, 0, 1));",
                      enforce_body(self.recorder, "DC_ActuatorModeA", "Apply"))
        # the expert logs the counters and the helper's sha256 (console only: no stamp, column or event)
        self.assertIn("m_LLC.DescribeLaunchClutch()", enforce_body(self.expert, "DC_ExpertDriver", "EndEpisode"))
        begin = enforce_body(self.expert, "DC_ExpertDriver", "BeginDriver")
        self.assertIn("HashSource(DC_LaunchClutch.SOURCE_FILE, failure);", begin)
        self.assertIn("ctx.SetStamps(m_sLlcSha, m_sExpertSha, m_Card.m_sCardId, m_Card.m_sSha256);", begin)
        self.assertIn('static const string SOURCE_FILE = "DC_LaunchClutch.c";', self.helper)

    def test_deploy_manifest_carries_the_helper(self):
        with open(MANIFEST, "r", encoding="ascii") as f:
            entries = [line.split("#", 1)[0].split() for line in f]
        self.assertIn(["script", "src/enfusion/datacapture/DC_LaunchClutch.c"], entries)
        self.assertIn(["config", "tools/ident/cards/1.9.0.37/S105.json"], entries)    # the provisional 1.9 card
        self.assertIn(["config", "tools/ident/cards/1.8.0.13/S105.json"], entries)




def slip(**config):
    return LaunchClutch.slip(**config)


def hold(lc, rpm, steps=1, driveline=-1.0, dt=STEP_DT, throttle=0.5, brake=0.0, gear=2):
    """steps calls of the slip clutch in gear, no handbrake, commanded 1, engine on at rpm, throttle and brake as
    given; returns the last c."""
    c = None
    for _ in range(steps):
        c = lc.step(dt, True, False, rpm, True, 1.0, False, driveline, throttle, brake, gear)
    return c


def coupled(rpm, **config):
    """A slip clutch locked at rpm with a steady history."""
    lc = slip(**config)
    lc.locked, lc.coupling = True, 1.0
    lc.last_rpm = lc.last_mean_rpm = rpm
    lc.history, lc.last_slice_s, lc.prev_slice_s, lc.last_gear = 3, STEP_DT, STEP_DT, 2
    return lc


class SlipStepTest(unittest.TestCase):
    """The per-call rules of DC_LaunchClutch.StepSlip (mode slip, DD-14d), on the mirror."""

    def test_the_pi_while_slipping(self):
        lc = slip()
        self.assertEqual(hold(lc, 800.0), 0.0)                       # idle never bites
        self.assertEqual(hold(lc, 1100.0), 0.0)                      # engage_rpm: open
        lc = slip()
        cs = [hold(lc, 2000.0) for _ in range(5)]                    # above the held 1400: the integral couples
        self.assertTrue(all(b > a for a, b in zip(cs, cs[1:])), cs)
        self.assertAlmostEqual(cs[1] - cs[0], STEP_DT * 0.001 * (2000.0 - 1400.0), delta=1e-5)
        lc = slip()
        for rpm in (1500.0, 1500.0, 1500.0):
            hold(lc, rpm)
        c0 = lc.coupling
        c = hold(lc, 1700.0)                                         # proportional: on the mean's rate [rpm/s]
        rate = 2.0 * (1600.0 - 1500.0) / (2 * STEP_DT)
        self.assertAlmostEqual(c, c0 + STEP_DT * (0.0003 * rate + 0.001 * (1600.0 - 1400.0)), delta=1e-5)
        self.assertFalse(lc.locked)

    def test_rate_limits_bounds_and_the_held_call_length(self):
        lc = slip()
        self.assertAlmostEqual(hold(lc, 8000.0), STEP_DT / 0.85)    # up by at most dt / couple_time_s
        lc = slip(slip_kp=0.01)                                      # a strong P: 1500 -> 1450 asks c -0.25 ...
        lc.coupling = 0.9
        lc.last_rpm = lc.last_mean_rpm = 1500.0
        lc.history, lc.last_slice_s, lc.prev_slice_s = 3, STEP_DT, STEP_DT
        self.assertAlmostEqual(hold(lc, 1450.0, throttle=1.0), 0.9 - STEP_DT / 0.45)   # ... down by dt / uncouple
        lc = slip(slip_ki=0.1)
        self.assertAlmostEqual(hold(lc, 8000.0, dt=3 * STEP_DT), 3 * STEP_DT / 0.85)   # a 3-step call: 3 x as far
        lc = slip()                                                  # ... and its integral gain / 9
        self.assertAlmostEqual(hold(lc, 8000.0, dt=3 * STEP_DT), 3 * STEP_DT * 0.001 / 9.0 * (8000.0 - 1400.0),
                               delta=1e-4)
        self.assertAlmostEqual(lc.call_steps, 3 * STEP_DT / 0.033333)
        # the gains are per GAIN_STEP_S: after a 3-step call they are divided by 3 (P) and 9 (I), recovering by
        # CALL_STEPS_DECAY per call
        c0 = lc.coupling
        c = hold(lc, 1500.0, throttle=1.0)
        self.assertAlmostEqual(lc.call_steps, 3 * STEP_DT / 0.033333 - 0.05)
        self.assertLess(c - c0, STEP_DT / 0.85)
        lc = slip()
        self.assertEqual(hold(lc, 1400.0), 0.0)                      # first call: no rate, no error
        self.assertEqual(lc.coupling, 0.0)

    def test_the_mean_follows_no_alternating_readback(self):
        # the readback alternates between half a step and no step after the setters, so a steady slip reads 1500,
        # 1600, 1500, ...; the mean of two readbacks is steady, and c moves one way only
        lc = slip()
        cs = [hold(lc, 1500.0 + 100.0 * (k % 2)) for k in range(12)]
        deltas = [b - a for a, b in zip(cs, cs[1:])]
        self.assertTrue(all(d > 0 for d in deltas[2:]), deltas)

    def test_slipping_opens_without_drive_or_at_a_predicted_lug(self):
        # Codex round 1, HIGH: a brake before the lock-up stalled (219 rpm); a slipping clutch opens at once without
        # drive (a brake or no throttle), before the brake acts (one step after it is set)
        for throttle, brake in ((0.5, 0.3), (0.0, 0.0), (0.0, 1.0)):
            lc = slip()
            hold(lc, 1600.0, steps=8)
            self.assertGreater(lc.coupling, 0)
            self.assertEqual(hold(lc, 1600.0, throttle=throttle, brake=brake), 0.0, (throttle, brake))
            self.assertFalse(lc.locked)
        # a fast fall while slipping: predicted 1300 - 1.5 x dt x 9000 rpm/s = 850 <= the lug 900: open
        lc = slip()
        hold(lc, 1600.0, steps=8)
        lc.last_rpm, lc.last_mean_rpm = 1600.0, 1600.0
        hold(lc, 1300.0)                                             # mean 1450: rate 2 x -150 / 2 dt = -4500 rpm/s
        self.assertGreater(lc.coupling, 0)
        self.assertEqual(hold(lc, 1000.0 + 150.0), 0.0)              # mean 1225: rate -6750: 1150 - 337 < 900
        self.assertEqual(lc.lug_rpm(), 900.0)

    def test_lockup_needs_drive_and_two_agreeing_calls(self):
        lc = slip()
        hold(lc, 1500.0, steps=3)
        self.assertGreater(lc.coupling, 0)
        hold(lc, 1500.0, driveline=1350.0)                           # 150 rpm apart: no agreement
        self.assertEqual(lc.agree_calls, 0)
        c = hold(lc, 1500.0, driveline=1401.0)                       # within 100 rpm: one call is not enough
        self.assertLess(c, 1.0)
        self.assertEqual(lc.agree_calls, 1)
        self.assertEqual(hold(lc, 1500.0, driveline=1420.0), 1.0)   # the second in a row: coupled at once
        self.assertTrue(lc.locked)
        self.assertEqual(lc.lockups, 1)
        # Codex round 1, HIGH: a brake's wheel slip makes body speed read above the shaft; under a brake there is no
        # lock-up (the slipping clutch opens instead), and an interrupted agreement starts again
        lc = slip()
        hold(lc, 1500.0, steps=3)
        hold(lc, 1500.0, driveline=1450.0)
        self.assertEqual(hold(lc, 1500.0, driveline=1450.0, brake=1.0), 0.0)
        self.assertEqual((lc.locked, lc.agree_calls), (False, 0))
        lc = slip()
        hold(lc, 1300.0, driveline=1300.0, steps=3)                  # c 0 below the held rpm: no agreement
        self.assertEqual((lc.coupling, lc.locked), (0.0, False))
        lc = slip()
        lc.coupling = 0.99
        lc.last_rpm = lc.last_mean_rpm = 3000.0
        self.assertEqual(hold(lc, 3000.0), 1.0)                      # no driveline: when the PI reaches 1
        self.assertTrue(lc.locked)

    def test_coupled_floor_lookahead_and_drop_guard(self):
        lc = coupled(1200.0)
        self.assertEqual(hold(lc, 1200.0, steps=10, throttle=0.0, brake=0.1), 1.0)   # steady in the old band
        self.assertEqual(hold(lc, 1100.0, throttle=0.0, brake=0.1), 0.0)             # no drive: open at engage_rpm
        lc = coupled(1050.0)
        self.assertEqual(hold(lc, 1050.0, steps=10, throttle=0.4), 1.0)  # under drive it lugs below engage_rpm ...
        self.assertEqual(hold(lc, 900.0, throttle=0.4), 0.0)             # ... down to the lug rpm 900
        lc = coupled(1050.0)
        self.assertEqual(hold(lc, 1050.0, throttle=0.4, brake=0.2), 0.0)  # a brake there: engage_rpm again, open
        # a full brake: 1700 -> 1500 (mean 1600) -> 1250 (mean 1375): rate -6750 rpm/s, predicted 912 <= 1100: open
        lc = coupled(1700.0)
        self.assertEqual(hold(lc, 1500.0, throttle=0.0, brake=1.0), 1.0)
        self.assertEqual(hold(lc, 1250.0, throttle=0.0, brake=1.0), 0.0)
        # above the ceiling (2 x engage_rpm) no lookahead
        lc = coupled(4500.0)
        self.assertEqual(hold(lc, 2600.0, throttle=0.5), 1.0)
        self.assertEqual(hold(lc, 2300.0, throttle=0.5), 1.0)
        self.assertEqual(lc.lookahead_ceiling_rpm(), 2200.0)
        # the drop guard: without drive, a fall faster than 12000 rpm/s opens even above the ceiling ...
        lc = coupled(4000.0)
        hold(lc, 3400.0, throttle=0.0, brake=1.0)                    # mean 3700: rate -9000
        self.assertEqual(lc.coupling, 1.0)
        self.assertEqual(hold(lc, 2600.0, throttle=0.0, brake=1.0), 0.0)   # mean 3000: rate -21000
        # ... but not under drive (an engine that synchronises down after the PI reached 1 at a high flare)
        lc = coupled(4000.0)
        hold(lc, 3400.0, throttle=1.0)
        self.assertEqual(hold(lc, 2600.0, throttle=1.0), 1.0)

    def test_upshift_synchronisation(self):
        # Codex round 1, MEDIUM: gear_scripted's upshift into a low-rpm gear fired the braking lookahead on the
        # synchronisation drop. An upshift restarts the history and waits SYNC_S (0.3 s) before predicting
        lc = coupled(2000.0)
        self.assertEqual(hold(lc, 1990.0, throttle=0.4, gear=2), 1.0)
        for rpm in (1700.0, 1400.0, 1150.0, 1110.0):                 # into gear 3: about 300 rpm per step
            self.assertEqual(hold(lc, rpm, throttle=0.4, gear=3), 1.0, rpm)
        self.assertTrue(lc.locked)
        self.assertGreater(lc.sync_left_s, 0.0)
        lc = coupled(2000.0)                                         # the same drop without a gear change opens
        for rpm in (1700.0, 1400.0, 1150.0):
            c = hold(lc, rpm, throttle=0.0, brake=0.0)
        self.assertEqual(c, 0.0)
        lc = coupled(1900.0)                                         # a downshift keeps the history
        lc.last_gear = 3
        hold(lc, 1900.0, throttle=0.0, brake=1.0, gear=3)
        hold(lc, 2600.0, throttle=0.0, brake=1.0, gear=2)
        self.assertEqual((lc.history, lc.sync_left_s), (3, 0.0))

    def test_zero_cases_guard_and_restart(self):
        for args in ((False, False, False, True), (True, True, False, True), (True, False, True, True)):
            lc = coupled(2500.0)
            in_gear, handbrake, dip, on = args
            self.assertEqual(lc.step(STEP_DT, in_gear, handbrake, 2500.0, on, 0.0 if dip else 1.0, dip, 2500.0, 0.5,
                                     0.0, 2), 0.0)
            self.assertFalse(lc.locked, args)
        lc = coupled(2500.0, restart_on_stall=False)
        self.assertEqual(lc.step(STEP_DT, True, False, 2500.0, False, 1.0, False, 2500.0, 0.5, 0.0, 2), 0.0)
        self.assertFalse(lc.locked)
        lc = coupled(2500.0)
        self.assertEqual(hold(lc, 599.0), 0.0)                       # below the guard: a guard drop
        self.assertEqual((lc.guard_drops, lc.locked), (1, False))
        lc = slip()
        hold(lc, 2000.0)
        restarts = []
        for k in range(75):                                          # the ramp's stall restart, in mode slip too
            lc.step(STEP_DT, True, False, 0.0, False, 1.0, False, -1.0, 0.5, 0.0, 2)
            if lc.consume_restart():
                restarts.append(k)
        self.assertEqual((lc.stalls, len(restarts), restarts[0]), (1, 3, 0))
        lc.reset()
        self.assertEqual((lc.locked, lc.last_rpm, lc.last_mean_rpm, lc.coupling, lc.call_steps, lc.last_gear),
                         (False, -1.0, -1.0, 0.0, 1.0, -1))


def fixture_rows():
    rows = model.read_steps(FIXTURE)
    return rows, model.readback_timing(rows), model.launch_windows(rows)


# What the Play recorded with the DD-14c ramp (e01_steps.csv of launch 20261106T194516Z-wb-ab6a): openings while the
# throttle is on and the first cmd_clutch = 1 after the launch [s]
RECORDED = {"move_check": (2, 2.00), "throttle_0.30": (4, 3.83), "throttle_0.60": (1, 1.63), "throttle_1.00": (0, 0.93),
            "gear_scripted": (3, 2.47)}
COMPLETE_HEADER = {"status": "complete", "launch_clutch.mode": "slip", "launch_clutch.stall_guard_rpm": "600.0",
                   "launch_clutch.stalls": "0", "launch_clutch.restarts": "0", "launch_clutch.guard_drops": "0",
                   "launch_clutch.max_slip_s": "0.500", "launch_clutch.lockups": "5"}


class CalibrationTest(unittest.TestCase):
    """The fitted plant against the 1.9.0.37 ramp Play (tests/fixtures/launch_clutch_19/e01_launches.csv)."""

    @classmethod
    def setUpClass(cls):
        cls.rows, cls.rho, cls.windows = fixture_rows()

    def test_the_fixture_is_the_plays_chatter(self):
        lines, passed = analysis.check_play(self.rows, dict(COMPLETE_HEADER, **{"launch_clutch.mode": "ramp"}))
        self.assertFalse(passed)
        self.assertEqual(sorted(self.windows), sorted(RECORDED))
        for name, (first, last) in model.launch_windows(self.rows, seconds=8.0).items():
            opens = sum(1 for i in range(first + 1, last + 1) if self.rows[i - 1]["cmd_clutch"] > 0
                        and self.rows[i]["cmd_clutch"] == 0 and self.rows[i]["cmd_throttle"] > 0)
            coupled_at = next(i - first for i in range(first, last + 1) if self.rows[i]["cmd_clutch"] >= 1.0)
            self.assertEqual(opens, RECORDED[name][0], name)
            self.assertAlmostEqual(coupled_at * STEP_DT, RECORDED[name][1], delta=0.02, msg=name)
        self.assertTrue(any("move_check     throttle 0.50: 2 openings" in line for line in lines), lines)

    def test_check_play_refuses_what_is_not_a_launch(self):
        # Codex round 1, MEDIUM: check_play passed without a launch. An empty capture, a capture without its footer or
        # counters, and a fixture-shaped run that never moves or couples all FAIL
        self.assertFalse(analysis.check_play([], {})[1])
        self.assertFalse(analysis.check_play([], COMPLETE_HEADER)[1])
        frozen = [dict(r, cmd_clutch=0.0, speed_kmh=0.0, vel_l_z=0.0, rpm=1400.0) for r in self.rows]
        lines, passed = analysis.check_play(frozen, COMPLETE_HEADER)
        self.assertFalse(passed)
        self.assertTrue(any("no motion" in line and "never coupled" in line for line in lines), lines)
        no_launch = [r for r in self.rows if r["phase"] not in model.LAUNCH_PHASES]
        lines, passed = analysis.check_play(no_launch, COMPLETE_HEADER)
        self.assertFalse(passed)
        self.assertTrue(any("no launch phase" in line for line in lines), lines)
        # a replay of the card's slip on the plant, written as a play's rows, passes with a complete header ...
        good = []
        for name, throttle in model.LAUNCH_PHASES.items():
            sim = model.launch(LaunchClutch(**analysis.card_launch_clutch()), throttle, seconds=4.0)
            for k, r in enumerate(sim):
                good.append({"physics_step": len(good), "phase": name, "segment": "excite", "cmd_throttle": throttle,
                             "cmd_brake": 0.0, "cmd_clutch": r[1], "input_post_clutch": r[1], "cmd_gear": 2.0,
                             "cur_gear": 2.0, "rpm": r[2], "engine_on": 1.0, "speed_kmh": r[4],
                             "vel_l_z": r[4] / 3.6})
        lines, passed = analysis.check_play(good, COMPLETE_HEADER)
        self.assertTrue(passed, lines)
        # ... and fails without the footer's status, a counter, or with a stall in the steps
        self.assertFalse(analysis.check_play(good, dict(COMPLETE_HEADER, status="incomplete"))[1])
        partial = dict(COMPLETE_HEADER)
        del partial["launch_clutch.guard_drops"]
        self.assertFalse(analysis.check_play(good, partial)[1])
        stalled = [dict(r) for r in good]
        stalled[50]["engine_on"] = 0.0
        self.assertFalse(analysis.check_play(stalled, COMPLETE_HEADER)[1])

    def test_fit_quality(self):
        e_rpm, e_v = model.shooting_residuals(None, self.rows, self.rho, self.windows)
        self.assertLess(model.rms(e_rpm), 45.0)
        self.assertLess(model.rms(e_v), 0.040)
        drag, count = analysis.free_decay_per_s(self.rows)
        self.assertGreater(count, 50)
        self.assertAlmostEqual(drag, model.S105_19["drag_per_s"], delta=0.01)
        # the clutch moves the rpm by about 965 rpm per step at c = 1: the ramp's 600 rpm band is a loop gain of 1.6
        per_step = model.S105_19["clutch_nm"] / model.S105_19["j_kgm2"] * model.RPM_PER_RADS * STEP_DT
        self.assertAlmostEqual(per_step, 965.0, delta=5.0)
        self.assertGreater(per_step / 600.0, 1.5)

    def test_the_fit_lands_on_s105_19_from_the_toy(self):
        params, e_rpm, e_v = analysis.fit(self.rows, self.rho, self.windows,
                                          dict(model.S105_19, **analysis.FIT_START), log=lambda *_: None)
        for key in ("j_kgm2", "clutch_nm", "rolling_n"):
            self.assertAlmostEqual(params[key] / model.S105_19[key], 1.0, delta=0.03, msg=key)
        for got, want in zip(params["torque_nm"], model.S105_19["torque_nm"]):
            self.assertAlmostEqual(got / want, 1.0, delta=0.03)
        self.assertLess(e_rpm, 45.0)

    def test_the_ramp_replay_chatters_like_the_play(self):
        # closed loop on the plant, each launch with its recorded readback timing: the DD-14c ramp opens the clutch as
        # the Play did (throttle 0.3 is the most sensitive: 4 in the Play) and couples fully at the Play's times
        replay = analysis.replay(self.rows, self.rho, self.windows, None, analysis.DD14C_RAMP)
        for name, (count, opens, lockup) in replay.items():
            want_count, want_lockup = RECORDED[name]
            if name == "throttle_0.30":
                self.assertTrue(3 <= count <= 6, (name, count, opens))
            else:
                self.assertEqual(count, want_count, (name, opens))
            self.assertAlmostEqual(lockup, want_lockup, delta=0.1, msg=name)
        self.assertEqual(replay["move_check"][1][:2], [18, 34])        # the Play: steps 17 and 34 after the release

    def test_the_slip_replay_does_not(self):
        replay = analysis.replay(self.rows, self.rho, self.windows, None, analysis.card_launch_clutch())
        for name, (count, opens, lockup) in replay.items():
            self.assertEqual(count, 0, (name, opens))
            self.assertLess(lockup, RECORDED[name][1] + 0.25, name)
        self.assertLessEqual(replay["move_check"][2], 2.0)


ACCEPT_THROTTLES = (0.3, 0.5, 0.6, 1.0)
ACCEPT_GRADES = (0.0, 0.05, -0.05)
GUARD_MARGIN_RPM = 100.0
CALLS = {"1": (1,), "2": (2,), "3": (3,), "mixed": (1, 1, 2, 1, 3, 1, 2, 3, 1, 1)}
# c = 1 at throttle 0.5 on the nominal plant, flat: 2.0 s with 1-step callbacks (the probe kit: the 1.9.0.37 Play had
# 0 merged fixed steps of 3567, clock.merged_fixed_steps); llc_v1's callbacks of 2 and 3 steps slow the PI (its gains
# are per 1/30 s step), so the envelope is stated per callback (DD-14d round 1)
LOCKUP_LIMIT_S = {"1": 2.0, "2": 2.0, "3": 2.2, "mixed": 2.1}


def variant_params(j=1.0, clutch=1.0, torque=1.0, **extra):
    return dict({"j_kgm2": model.S105_19["j_kgm2"] * j, "clutch_nm": model.S105_19["clutch_nm"] * clutch,
                 "torque_nm": tuple(t * torque for t in model.S105_19["torque_nm"])}, **extra)


class SlipAcceptanceTest(unittest.TestCase):
    """DD-14d's acceptance on the fitted plant (tools/ident/launch_clutch_model.py), for the 1.9.0.37 card's slip."""

    @classmethod
    def setUpClass(cls):
        with open(CARD, "r", encoding="ascii") as f:
            cls.config = json.load(f)["gearbox"]["launch_clutch"]
        cls.floor = cls.config["stall_guard_rpm"] + GUARD_MARGIN_RPM

    def clutch(self):
        return LaunchClutch(**self.config)

    def test_launches(self):
        for calls in CALLS:
            for timing in ("U", "A"):
                for throttle in ACCEPT_THROTTLES:
                    for grade in ACCEPT_GRADES:
                        label = (calls, timing, throttle, grade)
                        m = model.metrics(model.launch(self.clutch(), throttle, grade=grade, timing=timing,
                                                       calls=CALLS[calls]))
                        self.assertEqual((m["openings"], m["stalls"]), (0, 0), label)
                        self.assertLessEqual(m["alternations"], 1, label)          # no step-to-step oscillation
                        self.assertGreaterEqual(m["min_rpm"], self.floor, label)
                        if grade <= 0:
                            self.assertIsNotNone(m["lockup_s"], label)
                        if throttle == 0.5 and grade == 0.0:
                            self.assertLessEqual(m["lockup_s"], LOCKUP_LIMIT_S[calls], label)

    def test_combined_robustness_grid(self):
        # Codex round 1, MEDIUM: one factor at a time did not establish the envelope. Every combination of engine
        # inertia and clutch capacity +-30 %, torque +-20 %, callbacks of 1, 2, 3 and mixed steps, both readback
        # timings, throttle 0.3 / 0.5 / 0.6 / 1.0, flat and +-5 %: no opening, no stall, no step-to-step oscillation,
        # rpm at or above the guard + 100 (2592 launches)
        worst = 0.0
        for j, clutch, torque in itertools.product((0.7, 1.0, 1.3), (0.7, 1.0, 1.3), (0.8, 1.0, 1.2)):
            params = variant_params(j, clutch, torque)
            for calls in CALLS:
                for timing in ("U", "A"):
                    for throttle in ACCEPT_THROTTLES:
                        for grade in ACCEPT_GRADES:
                            label = (j, clutch, torque, calls, timing, throttle, grade)
                            m = model.metrics(model.launch(self.clutch(), throttle, grade=grade, timing=timing,
                                                           calls=CALLS[calls], params=params))
                            self.assertEqual((m["openings"], m["stalls"]), (0, 0), label)
                            self.assertLessEqual(m["alternations"], 1, label)
                            self.assertGreaterEqual(m["min_rpm"], self.floor, label)
                            if throttle == 0.5 and grade == 0.0:
                                worst = max(worst, m["lockup_s"])
        self.assertLessEqual(worst, 3.1)                              # the slowest corner (torque -20 %, 3 steps)

    def test_stops_before_the_lockup(self):
        # Codex round 1, HIGH: a brake while still slipping (3 / 5 / 7 km/h; the lock-up comes at about 7.8 km/h) with
        # callbacks of 1, 2, 3 and mixed steps and both timings: the clutch opens at the brake command, no stall
        for calls in CALLS:
            for timing in ("U", "A"):
                for throttle in (0.3, 0.5, 1.0):
                    for brake in (0.3, 0.6, 1.0):
                        for at_kmh in (3.0, 5.0, 7.0):
                            label = (calls, timing, throttle, brake, at_kmh)
                            rows, first = model.brake_to_stop(self.clutch(), brake, at_kmh, throttle=throttle,
                                                              timing=timing, calls=CALLS[calls])
                            m = model.metrics(rows, first)
                            self.assertEqual((m["stalls"], model.openings(rows[:first])), (0, 0), label)
                            self.assertGreaterEqual(m["min_rpm"], self.floor, label)
                            self.assertTrue(m["end_on"] and m["end_c"] == 0.0, label)

    def test_braking_to_a_stop_opens_first(self):
        for j, clutch in ((1.0, 1.0), (0.7, 0.7), (0.7, 1.3), (1.3, 0.7), (1.3, 1.3)):
            params = variant_params(j, clutch)
            for calls in CALLS:
                for timing in ("U", "A"):
                    for brake in (0.3, 0.6, 1.0):
                        for from_kmh in (15.0, 30.0, 40.0):
                            label = (j, clutch, calls, timing, brake, from_kmh)
                            rows, first = model.brake_to_stop(self.clutch(), brake, from_kmh, timing=timing,
                                                              calls=CALLS[calls], params=params)
                            m = model.metrics(rows, first)
                            self.assertEqual(model.openings(rows[:first]), 0, label)
                            self.assertEqual(m["stalls"], 0, label)
                            self.assertTrue(m["end_on"] and m["end_c"] == 0.0 and m["end_kmh"] == 0.0, label)
                            self.assertGreaterEqual(m["min_rpm"], self.floor, label)

    def test_codex_round_1_cases(self):
        # the exact reproductions of the review, now safe (the DD-14c ramp survives the first two as well)
        rows, first = model.brake_to_stop(self.clutch(), 1.0, 5.0, throttle=0.5, timing="A", calls=(3,))
        self.assertGreaterEqual(model.metrics(rows, first)["min_rpm"], self.floor)     # was 219 rpm, a stall
        for brake_slip in (0.25, 1.5, 2.0):
            rows, first = model.brake_to_stop(self.clutch(), 1.0, 7.0, throttle=0.5, timing="U",
                                              params={"brake_slip_mps": brake_slip})
            m = model.metrics(rows, first)
            self.assertEqual(m["stalls"], 0, brake_slip)                                # was 426 rpm at 1.5 m/s
            self.assertGreaterEqual(m["min_rpm"], self.floor, brake_slip)
        m = model.metrics(model.launch(self.clutch(), 0.3, params=variant_params(j=0.7), calls=(3,)))
        self.assertEqual((m["openings"], m["alternations"] <= 1), (0, True))           # was 3 openings, 4 in a row
        m = model.metrics(model.launch(self.clutch(), 1.0, calls=CALLS["mixed"]))
        self.assertLessEqual(m["alternations"], 1)                                      # was 2
        # wheel lock while coupled: brake slip up to 2 m/s from 15 and 40 km/h
        for brake_slip in (0.5, 1.0, 2.0):
            for from_kmh in (15.0, 40.0):
                rows, first = model.brake_to_stop(self.clutch(), 1.0, from_kmh, params={"brake_slip_mps": brake_slip})
                m = model.metrics(rows, first)
                self.assertEqual(m["stalls"], 0, (brake_slip, from_kmh))
                self.assertGreaterEqual(m["min_rpm"], self.floor, (brake_slip, from_kmh))

    def test_shifts_stay_coupled(self):
        # the probe policy (up at 4500, down at 1800 rpm) up to fourth gear: after the first lock-up c stays 1
        for calls in CALLS:
            for throttle in (0.6, 1.0):
                rows = model.launch(self.clutch(), throttle, seconds=14.0, calls=CALLS[calls])
                gears = [r[7] for r in rows]
                self.assertGreaterEqual(max(gears), 4, (calls, throttle))
                first = next(i for i, r in enumerate(rows) if r[1] >= 1.0)
                self.assertEqual(min(r[1] for r in rows[first:]), 1.0, (calls, throttle))
        # and a speed hold that coasts down through a downshift (throttle off at 30 km/h, then on again at 15)
        state = {"coast": False}

        def schedule(t, kmh):
            if kmh >= 30.0:
                state["coast"] = True
            elif kmh <= 15.0:
                state["coast"] = False
            return (0.0, 0.1) if state["coast"] else (0.4, 0.0)

        rows = model.drive(self.clutch(), schedule, 30.0)
        first = next(i for i, r in enumerate(rows) if r[1] >= 1.0)
        self.assertEqual(min(r[1] for r in rows[first:]), 1.0)
        self.assertGreater(sum(1 for a, b in zip(rows, rows[1:]) if b[7] < a[7]), 0)   # it did shift down

    def test_gear_scripted_full_sequence(self):
        # Codex round 1, MEDIUM: E-01's gear_scripted (m_sGearSequence 2,3,4,3,2, m_fGearHoldS 3 s, throttle 0.4, no
        # dip) reopened the clutch at its upshifts. The upshift into gear 4 lands near 1075 rpm, below engage_rpm:
        # under drive the coupled clutch lugs down to 900 rpm, so the whole sequence keeps it coupled
        sequence = (2, 3, 4, 3, 2)

        def gear_fn(t, kmh):
            return sequence[min(int(t // 3.0), len(sequence) - 1)]

        for calls in CALLS:
            for timing in ("U", "A"):
                rows = model.drive(self.clutch(), lambda t, kmh: (0.4, 0.0), 15.0, timing=timing, gear_fn=gear_fn,
                                   calls=CALLS[calls])
                m = model.metrics(rows)
                label = (calls, timing)
                self.assertEqual((m["openings"], m["stalls"]), (0, 0), label)
                self.assertGreaterEqual(m["min_rpm"], self.floor, label)
                self.assertEqual(sorted(set(r[7] for r in rows)), [2, 3, 4], label)
                first = next(i for i, r in enumerate(rows) if r[1] >= 1.0)
                self.assertLess(rows[first][0], 3.0, label)                    # coupled before the first upshift
                self.assertEqual(min(r[1] for r in rows[first:]), 1.0, label)  # and through every shift
        ramp_rows = model.drive(LaunchClutch(**analysis.DD14C_RAMP), lambda t, kmh: (0.4, 0.0), 15.0, gear_fn=gear_fn)
        self.assertEqual(model.openings(ramp_rows), 3)                          # the ramp: the Play's 3 openings

    def test_the_dd14c_ramp_fails_it(self):
        ramp_clutch = LaunchClutch(**analysis.DD14C_RAMP)
        m = model.metrics(model.launch(ramp_clutch, 0.5))
        self.assertEqual(m["openings"], 2)
        self.assertGreaterEqual(m["lockup_s"], 1.95)
        _, passed = analysis.acceptance(analysis.DD14C_RAMP)
        self.assertFalse(passed)
        table, passed = analysis.acceptance(self.config)
        self.assertTrue(passed, table)


class SlipTiesTest(unittest.TestCase):
    """The slip mirror, its constants and the inputs of both callers against the Enforce sources (DD-14d)."""

    @classmethod
    def setUpClass(cls):
        cls.helper = read(HELPER_SOURCE)
        cls.probe = read(PROBE_SOURCE)
        cls.llc = read(LLC_SOURCE)

    def test_mirror_follows_step_slip(self):
        body = class_method(self.helper, "DC_LaunchClutch", "StepSlip")
        for statement in (
                "float engage = m_Config.m_fEngageRpm; if (gear >= 0 && m_iLastGear >= 0 && gear > m_iLastGear) { "
                "ForgetRpm(); m_fSyncLeftS = DC_LaunchClutchConfig.SYNC_S; m_iAgreeCalls = 0; } m_iLastGear = gear;",
                "float last = m_fLastRpm; if (last < 0) last = rpm; float mean = 0.5 * (rpm + last); float meanLast = "
                "m_fLastMeanRpm; if (meanLast < 0) meanLast = mean; float spanS = m_fLastSliceS + m_fPrevSliceS; float "
                "rate = 0; if (m_iHistory >= 2 && spanS > 0) rate = 2 * (mean - meanLast) / spanS;",
                "m_fLastRpm = rpm; m_fLastMeanRpm = mean; if (m_iHistory < 3) m_iHistory++; m_fPrevSliceS = "
                "m_fLastSliceS; m_fLastSliceS = slice; m_fCallSteps = Math.Max(Math.Max(slice / "
                "DC_LaunchClutchConfig.GAIN_STEP_S, 1), m_fCallSteps - DC_LaunchClutchConfig.CALL_STEPS_DECAY); bool "
                "inSync = m_fSyncLeftS > 0; m_fSyncLeftS = Math.Max(m_fSyncLeftS - slice, 0); float predicted = rpm + "
                "DC_LaunchClutchConfig.RELEASE_LOOKAHEAD_CALLS * slice * Math.Min(rate, 0);",
                "if (!driving) { Open(); return; }",
                "if (rpm < m_Config.m_fStallGuardRpm) { if (m_fCoupling > 0) m_iGuardDrops++; Open(); return; }",
                "if (m_bLocked) { float openRpm = engage; if (drive) openRpm = m_Config.GetLugRpm(); if (rpm <= openRpm) "
                "{ Open(); return; } if (!inSync && rpm < m_Config.GetLookaheadCeilingRpm() && predicted <= openRpm) { "
                "Open(); return; } if (!drive && !inSync && rate < -DC_LaunchClutchConfig.MAX_DROP_RPM_PER_S) { Open(); "
                "return; } m_fCoupling = 1; return; }",
                "if (rpm <= engage || !drive || (!inSync && predicted <= m_Config.GetLugRpm())) { Open(); return; }",
                "float kp = m_Config.m_fSlipKp / m_fCallSteps; float ki = m_Config.m_fSlipKi / (m_fCallSteps * "
                "m_fCallSteps); float want = m_fCoupling + slice * (kp * rate + ki * (mean - m_Config.m_fFullRpm)); "
                "float lowest = Math.Max(m_fCoupling - slice / m_Config.m_fUncoupleTimeS, 0); float highest = "
                "Math.Min(m_fCoupling + slice / m_Config.m_fCoupleTimeS, 1); m_fCoupling = Math.Clamp(want, lowest, "
                "highest);",
                "if (m_fCoupling > 0 && drivelineRpm > 0 && Math.AbsFloat(rpm - drivelineRpm) <= "
                "m_Config.m_fLockToleranceRpm) m_iAgreeCalls++; else m_iAgreeCalls = 0; if (m_iAgreeCalls >= "
                "DC_LaunchClutchConfig.LOCK_AGREE_CALLS || m_fCoupling >= 1) { m_bLocked = true; m_fCoupling = 1; "
                "m_iLockups++; }"):
            self.assertIn(statement, body)
        self.assertIn("m_fCoupling = 0; m_bLocked = false; m_iAgreeCalls = 0;",
                      class_method(self.helper, "DC_LaunchClutch", "Open"))
        consts = enforce_constants(self.helper, "DC_LaunchClutchConfig")
        for name in ("RELEASE_LOOKAHEAD_CALLS", "RELEASE_LOOKAHEAD_ENGAGE_FACTOR", "SYNC_S", "LUG_MARGIN_RPM",
                     "MAX_DROP_RPM_PER_S", "GAIN_STEP_S", "CALL_STEPS_DECAY"):
            self.assertEqual(consts[name], getattr(LaunchClutch, name), name)
        self.assertIn("static const int LOCK_AGREE_CALLS = %d;" % LaunchClutch.LOCK_AGREE_CALLS, self.helper)
        self.assertIn("return RELEASE_LOOKAHEAD_ENGAGE_FACTOR * m_fEngageRpm;",
                      class_method(self.helper, "DC_LaunchClutchConfig", "GetLookaheadCeilingRpm"))
        self.assertIn("return Math.Min(m_fStallGuardRpm + LUG_MARGIN_RPM, m_fEngageRpm);",
                      class_method(self.helper, "DC_LaunchClutchConfig", "GetLugRpm"))
        self.assertIn("return m_sMode == MODE_RAMP || m_sMode == MODE_SLIP;",
                      class_method(self.helper, "DC_LaunchClutchConfig", "IsActive"))
        self.helper.encode("ascii")

    def test_the_mps_per_krpm_agree(self):
        with open(CARD, "r", encoding="ascii") as f:
            card = json.load(f)
        per_krpm = card["gearbox"]["scripted_mps_per_krpm"]
        self.assertEqual(per_krpm, [model.CARD_MPS_PER_KRPM[g] for g in card["gearbox"]["scripted_gears"]])
        attr = attribute_defaults(self.probe)["m_sLaunchMpsPerKrpm"]
        self.assertEqual([float(v) for v in attr.split(",")], per_krpm)
        self.assertIn("protected ref array<float> m_aLaunchMpsPerKrpm = {%s};" % ", ".join(repr(v) for v in per_krpm),
                      self.probe)
        # the card's ratios against the 1.9.0.37 Play's: within 1.3 %, far inside the 100 rpm lock-up tolerance
        for gear, value in model.CARD_MPS_PER_KRPM.items():
            self.assertAlmostEqual(value / model.S105_19_MPS_PER_KRPM[gear], 1.0, delta=0.013)

    def test_the_probe_checks_the_upshifts(self):
        check = enforce_body(self.probe, "DC_ProbeActuator", "CheckSlipUpshifts")
        self.assertIn("float ceiling = config.GetLookaheadCeilingRpm(); for (int slot = 0; slot + 1 < "
                      "m_aLaunchMpsPerKrpm.Count(); slot++) { float landing = m_fUpshiftRpm * m_aLaunchMpsPerKrpm[slot] "
                      "/ m_aLaunchMpsPerKrpm[slot + 1]; if (landing < ceiling) {", check)
        start = enforce_body(self.probe, "DC_ProbeSequences", "OnProbeStart")
        self.assertIn("if (!ParseFloatList(m_sLaunchMpsPerKrpm, m_aLaunchMpsPerKrpm, LAUNCH_MPS_PER_KRPM_MIN, "
                      "LAUNCH_MPS_PER_KRPM_MAX))", start)
        self.assertLess(start.index("m_Actuator.SetLaunchMpsPerKrpm(m_aLaunchMpsPerKrpm);"),
                        start.index("SetLaunchClutch("))
        landings = [4500.0 * a / b for a, b in zip((1.54871, 2.772066, 4.169103), (2.772066, 4.169103))]
        self.assertTrue(all(x >= 2200.0 for x in landings), landings)

    def test_the_image_probes_drive_with_slip(self):
        for name in ("DC_ProbeImages.c", "DC_ProbeMarkers.c"):
            self.assertIn("m_Actuator.Init(m_Ego)", read(os.path.join(SCRIPTS, name)), name)
        self.assertIn("DC_LaunchClutchConfig.Slip();", enforce_body(self.probe, "DC_ProbeActuator", "Init"))
        self.assertEqual(attribute_defaults(self.probe)["m_sLaunchClutchMode"], "slip")

    def test_rpm_feedback_is_recorded_only(self):
        # Codex round 1 / DD-14d: EngineGetRPMFeedback (declared on 1.8.0.13 and 1.9.0.37 alike) goes into the step
        # CSV as the last actuator column, rpm_feedback, so the next Play shows whether it is a usable driveline rpm;
        # the launch clutch does not read it
        read_post = enforce_body(self.probe, "DC_ProbeActuator", "ReadPost")
        self.assertIn("m_fRpm = m_Simulation.EngineGetRPM(); m_fRpmFeedback = m_Simulation.EngineGetRPMFeedback();",
                      read_post)
        columns = enforce_body(self.probe, "DC_ProbeActuator", "AddStepColumns")
        self.assertTrue(columns.endswith('columns.Insert("wheels_contact"); columns.Insert("rpm_feedback");'), columns)
        values = enforce_body(self.probe, "DC_ProbeActuator", "AddStepValues")
        self.assertTrue(values.endswith("row.AddInt(m_iWheelsContact); row.AddFloat(m_fRpmFeedback, 1);"), values)
        self.assertEqual(code_only(self.probe).count("m_fRpmFeedback"), 3)       # declared, read, written, nothing else
        self.assertNotIn("RPMFeedback", code_only(self.helper))
        self.assertNotIn("RPMFeedback", code_only(self.llc))
        self.assertEqual(model.OPTIONAL_STEP_COLUMNS, ("rpm_feedback",))


if __name__ == "__main__":
    unittest.main()
