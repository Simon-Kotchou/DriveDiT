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
  - Braking to a stop in first gear does not stall. A stall is restarted once per cooldown.
  - The defaults are the same in DC_LaunchClutchConfig, the DC_ProbeSequences attributes, the placeholder card and its
    generator. The three actuation paths (DC_ProbeActuator, llc_v1, DC_ActuatorModeA.Start) call the helper as
    described, and the deploy manifest carries the new file.
"""

import importlib.util
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
CARD = os.path.join(REPO, "tools", "ident", "cards", "1.8.0.13", "S105.json")
STEP_DT = 1.0 / 30.0        # run header clock.step_dt_s of the 1.9 Play (and of 1.8)
RPM_PER_RADS = 60.0 / (2.0 * math.pi)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load_module("fit_steer_kappa", os.path.join(REPO, "tools", "ident", "fit_steer_kappa.py"))


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


class LaunchClutch:
    """Python mirror of DC_LaunchClutch (Configure, Reset, Step, ConsumeRestart, the counters), statement for statement;
    LaunchClutchTiesTest ties it to the Enforce source."""

    DEFAULTS = {"mode": "off", "engage_rpm": 1100.0, "full_rpm": 1700.0, "couple_time_s": 0.85,
                "uncouple_time_s": 0.45, "stall_guard_rpm": 600.0, "restart_on_stall": True,
                "restart_cooldown_s": 1.0}

    def __init__(self, **config):
        self.config = dict(self.DEFAULTS, **config)
        self.reset()
        self.engine_seen = -1
        self.max_slip_s = self.max_slip_total_s = 0.0
        self.stalls = self.restarts = self.guard_drops = 0

    def reset(self):
        self.coupling = 0.0
        self.since_restart_s = -1.0
        self.restart_due = False
        self.slip_s = 0.0

    def step(self, dt, in_gear, handbrake, rpm, engine_on, commanded_clutch, dip):
        cfg = self.config
        commanded = min(max(commanded_clutch, 0.0), 1.0)
        slice_s = max(dt, 0.0)
        if self.engine_seen == 1 and not engine_on:
            self.stalls += 1
        self.engine_seen = 0
        if engine_on:
            self.engine_seen = 1
        if cfg["mode"] != "ramp":
            return commanded
        if self.since_restart_s >= 0:
            self.since_restart_s += slice_s
        cooled = self.since_restart_s < 0 or self.since_restart_s >= cfg["restart_cooldown_s"]
        if not engine_on and not handbrake and cfg["restart_on_stall"] and cooled:
            self.restart_due = True
            self.restarts += 1
            self.since_restart_s = 0.0
            self.coupling = 0.0
        driving = in_gear and not handbrake and not dip and engine_on
        if not driving:
            self.coupling = 0.0
        else:
            target = min(max((rpm - cfg["engage_rpm"]) / (cfg["full_rpm"] - cfg["engage_rpm"]), 0.0), 1.0)
            if rpm < cfg["stall_guard_rpm"]:
                if self.coupling > 0:
                    self.guard_drops += 1
                self.coupling = 0.0
            elif target > self.coupling:
                self.coupling = min(target, self.coupling + slice_s / cfg["couple_time_s"])
            else:
                self.coupling = max(target, self.coupling - slice_s / cfg["uncouple_time_s"])
        if driving and 0 < self.coupling < 1:
            self.slip_s += slice_s
        else:
            self.slip_s = 0.0
        self.max_slip_s = max(self.max_slip_s, self.slip_s)
        self.max_slip_total_s = max(self.max_slip_total_s, self.slip_s)
        return min(commanded, self.coupling)

    def consume_restart(self):
        if not self.restart_due:
            return False
        self.restart_due = False
        return True


def ramp(**config):
    return LaunchClutch(mode="ramp", **config)


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
    ROLLING = 0.015
    BRAKE_N = 9000.0            # at brake 1, at the contact patches (about 0.9 g)
    SUBSTEPS = 40               # integration substeps per 1/30 s
    # full-throttle torque [Nm] by rpm: peak torque 72.8 Nm at 3000 (Engine_S_742.10.conf), 65.9 Nm at the run
    # header's peak power rpm 4800 (33.1 kW, MaxPower), 0 at the run header's rpm_max 8000
    TORQUE_CURVE = ((0.0, 0.0), (500.0, 40.0), (800.0, 52.0), (3000.0, 72.8), (4800.0, 65.9), (5800.0, 55.0),
                    (7500.0, 20.0), (8000.0, 0.0))

    def __init__(self, rpm=RPM_IDLE, speed_mps=0.0, locked=False, j_engine=None):
        self.j_engine = j_engine or self.J_ENGINE
        self.we = rpm / RPM_PER_RADS
        self.v = speed_mps
        self.on = True
        self.locked = locked
        self.j_vehicle = self.MASS_KG * self.WHEEL_RADIUS_M ** 2 / self.FIRST_GEAR_RATIO ** 2

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
        return force * self.WHEEL_RADIUS_M / self.FIRST_GEAR_RATIO

    def force_start(self):
        """ForceStartEngine: the engine runs again at idle or above."""
        self.on = True
        self.we = max(self.we, self.RPM_IDLE / RPM_PER_RADS)

    def step(self, dt, throttle, brake, clutch):
        substeps = max(1, int(round(dt / STEP_DT * self.SUBSTEPS)))
        h = dt / substeps
        capacity = clutch * self.CLUTCH_MAX_NM
        for _ in range(substeps):
            wg = self.v / self.WHEEL_RADIUS_M * self.FIRST_GEAR_RATIO
            te, load = self.engine_torque(throttle), self.load_torque(brake)
            if capacity > 0 and (self.locked or abs(self.we - wg) < 0.5):
                accel = (te - load) / (self.j_engine + self.j_vehicle)
                if abs(te - self.j_engine * accel) <= capacity:
                    self.locked = True
                    wg = max(wg + accel * h, 0.0)
                    self.we, self.v = wg, wg / self.FIRST_GEAR_RATIO * self.WHEEL_RADIUS_M
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
            self.we, self.v = we_new, wg_new / self.FIRST_GEAR_RATIO * self.WHEEL_RADIUS_M
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
        down = lc.step(STEP_DT, True, False, 1100.0, True, 1.0, False)                 # target 0: falls at most
        self.assertAlmostEqual(down, 1.0 - STEP_DT / 0.45)                              # dt / uncouple
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
                "if (!m_Config.IsRamp()) return commanded; if (m_fSinceRestartS >= 0) m_fSinceRestartS += slice;",
                "bool cooled = m_fSinceRestartS < 0 || m_fSinceRestartS >= m_Config.m_fRestartCooldownS;",
                "if (!engineOn && !handbrake && m_Config.m_bRestartOnStall && cooled) { m_bRestartDue = true; "
                "m_iRestarts++; m_fSinceRestartS = 0; m_fCoupling = 0; }",
                "bool driving = inGear && !handbrake && !dip && engineOn; if (!driving) { m_fCoupling = 0; }",
                "float target = Math.Clamp((rpm - m_Config.m_fEngageRpm) / (m_Config.m_fFullRpm - "
                "m_Config.m_fEngageRpm), 0, 1);",
                "if (rpm < m_Config.m_fStallGuardRpm) { if (m_fCoupling > 0) m_iGuardDrops++; m_fCoupling = 0; }",
                "else if (target > m_fCoupling) { m_fCoupling = Math.Min(target, m_fCoupling + slice / "
                "m_Config.m_fCoupleTimeS); }",
                "else { m_fCoupling = Math.Max(target, m_fCoupling - slice / m_Config.m_fUncoupleTimeS); }",
                "if (driving && m_fCoupling > 0 && m_fCoupling < 1) m_fSlipS += slice; else m_fSlipS = 0;",
                "m_fMaxSlipS = Math.Max(m_fMaxSlipS, m_fSlipS); m_fMaxSlipTotalS = Math.Max(m_fMaxSlipTotalS, "
                "m_fSlipS); return Math.Min(commanded, m_fCoupling);"):
            self.assertIn(statement, step)
        self.assertIn("if (!m_bRestartDue) return false; m_bRestartDue = false; return true;",
                      class_method(self.helper, "DC_LaunchClutch", "ConsumeRestart"))
        self.assertIn("m_fCoupling = 0; m_fSinceRestartS = -1; m_bRestartDue = false; m_fSlipS = 0;",
                      class_method(self.helper, "DC_LaunchClutch", "Reset"))
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
                            "restart_cooldown_s": consts["DEFAULT_RESTART_COOLDOWN_S"]}
        mirror = dict(LaunchClutch.DEFAULTS)
        self.assertEqual(mirror.pop("mode"), "off")                 # a DC_LaunchClutchConfig starts off ...
        self.assertIn("string m_sMode = MODE_OFF;", self.helper)
        self.assertEqual(enforce_defaults, mirror)
        self.assertIn("config.m_sMode = MODE_RAMP;", class_method(self.helper, "DC_LaunchClutchConfig", "Ramp"))
        ramp_defaults = dict(enforce_defaults, mode="ramp")          # ... Ramp() is the probe kit's default
        self.assertEqual(fit.PLACEHOLDER_LAUNCH_CLUTCH, ramp_defaults)
        with open(CARD, "r", encoding="ascii") as f:
            self.assertEqual(json.load(f)["gearbox"]["launch_clutch"], ramp_defaults)
        attrs = attribute_defaults(self.probe)
        names = {"mode": "m_sLaunchClutchMode", "engage_rpm": "m_fLaunchEngageRpm", "full_rpm": "m_fLaunchFullRpm",
                 "couple_time_s": "m_fLaunchCoupleS", "uncouple_time_s": "m_fLaunchUncoupleS",
                 "stall_guard_rpm": "m_fLaunchStallGuardRpm", "restart_on_stall": "m_bLaunchRestartOnStall",
                 "restart_cooldown_s": "m_fLaunchRestartCooldownS"}
        for key, name in names.items():
            value = ramp_defaults[key]
            if isinstance(value, bool):
                self.assertEqual(attrs[name], "1" if value else "0", name)
            elif isinstance(value, str):
                self.assertEqual(attrs[name], value, name)
            else:
                self.assertEqual(float(attrs[name]), value, name)
        # full_rpm below the shift policy's downshift rpm (attribute and actuator default): fully coupled wherever
        # the policy drives; the idle (800 rpm, run header vehicle.rpm_idle) below engage_rpm never bites
        self.assertLess(ramp_defaults["full_rpm"], float(attrs["m_fShiftDownRpm"]))
        self.assertIn("static const float DEFAULT_DOWNSHIFT_RPM = %g;" % float(attrs["m_fShiftDownRpm"]), self.probe)
        self.assertGreater(ramp_defaults["engage_rpm"], ToyPowertrain.RPM_IDLE)
        self.assertLess(ramp_defaults["stall_guard_rpm"], ToyPowertrain.RPM_IDLE)
        # the ranges of DC_LaunchClutchConfig.Check are the Python validator's
        self.assertEqual((consts["RPM_MAX"], consts["TIME_MAX_S"]), (fit.LAUNCH_RPM_MAX, fit.LAUNCH_TIME_MAX_S))
        check = class_method(self.helper, "DC_LaunchClutchConfig", "Check")
        for rule in ("if (m_sMode != MODE_OFF && m_sMode != MODE_RAMP)",
                     "if (m_fEngageRpm <= 0 || m_fFullRpm <= m_fEngageRpm || m_fFullRpm > RPM_MAX)",
                     "if (m_fStallGuardRpm < 0 || m_fStallGuardRpm >= m_fEngageRpm)",
                     "if (m_fCoupleTimeS <= 0 || m_fCoupleTimeS > TIME_MAX_S || m_fUncoupleTimeS <= 0 || "
                     "m_fUncoupleTimeS > TIME_MAX_S)",
                     "if (m_fRestartCooldownS < 0 || m_fRestartCooldownS > TIME_MAX_S)"):
            self.assertIn(rule, check)
        self.assertEqual(fit.LAUNCH_CLUTCH_MODES, ("off", "ramp"))
        self.assertIn('static const string MODE_OFF = "off";', self.helper)
        self.assertIn('static const string MODE_RAMP = "ramp";', self.helper)

    def test_run_header_block_is_the_cards_fields(self):
        # AppendHeaderLines writes launch_clutch.<card field> (fit_steer_kappa.launch_clutch_of reads them back)
        lines = class_method(self.helper, "DC_LaunchClutchConfig", "AppendHeaderLines")
        self.assertEqual(re.findall(r'lines\.Insert\(prefix \+ "\.(\w+)="', lines), list(fit.LAUNCH_CLUTCH_KEYS))
        self.assertEqual(fit.HEADER_LAUNCH_CLUTCH, "launch_clutch")
        gearbox_lines = enforce_body(self.probe, "DC_ProbeActuator", "AppendGearboxLines")
        self.assertIn('m_LaunchClutch.GetConfig().AppendHeaderLines(lines, "launch_clutch");', gearbox_lines)
        summary = enforce_body(self.probe, "DC_ProbeActuator", "AppendDriveSummaryLines")
        self.assertIn('m_LaunchClutch.AppendSummaryLines(lines, "launch_clutch");', summary)
        for key in ("stalls", "restarts", "guard_drops", "max_slip_s"):
            self.assertIn('lines.Insert(prefix + ".%s=" +' % key, self.helper)
            self.assertIn('phase.AddNote("%s:" +' % key, self.probe)
        params = enforce_body(self.probe, "DC_ProbeSequences", "GetProbeParams")
        build = enforce_body(self.probe, "DC_ProbeSequences", "BuildLaunchClutchConfig")
        for name in ("m_sLaunchClutchMode", "m_fLaunchEngageRpm", "m_fLaunchFullRpm", "m_fLaunchCoupleS",
                     "m_fLaunchUncoupleS", "m_fLaunchStallGuardRpm", "m_bLaunchRestartOnStall",
                     "m_fLaunchRestartCooldownS"):
            self.assertIn('outKeyValues.Insert("%s=" +' % name, params)
            self.assertIn(name, build)

    def test_probe_actuator_runs_it(self):
        apply_step = enforce_body(self.probe, "DC_ProbeActuator", "ApplyStep")
        call = ("if (m_bShiftPolicy) { bool inGear = m_iAppliedGear >= 0 && m_iAppliedGear != m_iGearNeutral; "
                "bool engineOn = m_bEngineOn || !m_bStateRead; bool dip = m_fAppliedClutch <= 0; m_fAppliedClutch = "
                "m_LaunchClutch.Step(timeSlice, inGear, m_bCmdHandbrake, m_fRpm, engineOn, m_fAppliedClutch, dip); "
                "if (m_LaunchClutch.ConsumeRestart()) StartModeA(); }")
        self.assertIn(call, apply_step)
        # after the shift policy and its dip, before the setters; cmd_clutch is the value passed to SetClutch
        self.assertLess(apply_step.index("m_fDipLeftS -= timeSlice;"), apply_step.index(call))
        self.assertLess(apply_step.index(call), apply_step.index("m_Simulation.SetSteering(m_fCmdSteer);"))
        self.assertIn("m_Simulation.SetClutch(m_fAppliedClutch);", apply_step)
        self.assertIn("row.AddFloat(m_fAppliedClutch, 4);", enforce_body(self.probe, "DC_ProbeActuator",
                                                                          "AddStepValues"))
        self.assertIn("m_bStateRead = true;", enforce_body(self.probe, "DC_ProbeActuator", "ReadPost"))
        init = enforce_body(self.probe, "DC_ProbeActuator", "Init")
        self.assertIn("DC_LaunchClutchConfig launchClutch = DC_LaunchClutchConfig.Ramp(); "
                      "m_LaunchClutch.Configure(launchClutch);", init)
        start = enforce_body(self.probe, "DC_ProbeActuator", "StartModeA")
        self.assertIn("m_Controller.SetPersistentHandBrake(false); if (!m_Controller.IsEngineOn()) "
                      "m_Controller.ForceStartEngine();", start)
        set_launch = enforce_body(self.probe, "DC_ProbeActuator", "SetLaunchClutch")
        self.assertIn("if (!config.Check(failure))", set_launch)
        self.assertIn("if (config.IsRamp() && m_bShiftPolicy && config.m_fFullRpm >= m_fDownshiftRpm)", set_launch)
        start_probe = enforce_body(self.probe, "DC_ProbeSequences", "OnProbeStart")
        self.assertLess(start_probe.index("m_Actuator.SetShiftPolicy("), start_probe.index("SetLaunchClutch("))
        self.assertIn('if (m_Actuator.SetLaunchClutch(launchClutch, "DC_ProbeSequences attributes")) '
                      "readyActuator = m_Actuator;", start_probe)

    def test_llc_and_mode_a_run_it(self):
        step = enforce_body(self.llc, "DC_LLCv1", "Step")
        call = ("clutch = m_LaunchClutch.Step(sliceS, true, false, m_Simulation.EngineGetRPM(), "
                "m_Simulation.EngineIsOn(), clutch, clutch <= 0); if (m_LaunchClutch.ConsumeRestart()) "
                "m_Actuator.Start(); clutchRecorded = clutch; }")
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


if __name__ == "__main__":
    unittest.main()
