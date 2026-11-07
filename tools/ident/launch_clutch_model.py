#!/usr/bin/env python3
"""The S105 powertrain of Reforger 1.9.0.37 as fitted to E-01, and the Python mirror of DC_LaunchClutch (DD-14d).

Stdlib only. tools/ident/fit_launch_clutch.py fits and replays with it; tests/test_launch_clutch.py ties the mirror
to src/enfusion/datacapture/DC_LaunchClutch.c statement by statement and checks the launch clutch on the plant.

THE PLANT (Plant). One engine inertia J, a Coulomb clutch, the car through the gear, integrated in DT / 4 / MICRO
steps:
  - engine: J dw/dt = throttle x T(rpm) - J k (w - w_idle) for w above idle - T_clutch. T is piecewise linear over
    TORQUE_KNOTS_RPM. Below idle there is no drag and no governor: a Play's rpm stays where it is with the clutch
    open, as at 607 rpm after the full brakes of launch 20261106T194516Z-wb-ab6a. It stalls below stall_rpm (an
    assumption: alive at 528, off at 18 on the first 1.9 Play);
  - clutch: c x clutch_nm while slipping, and locked while the torque it must carry is within that. The rpm falls by a
    constant 38.5 rpm per step for each 0.039 of c, whatever the slip speed, so it is Coulomb, not viscous;
  - car: m dv/dt = T_clutch / (r / G) - rolling_n - aero v^2 - brake x brake_n. r / G comes from the gear's m/s per
    1000 rpm. Under a brake the driveline turns brake x brake_slip_mps slower than the body (a full brake drags the
    engine down faster than the speed implies);
  - timing: a SetClutch value acts from the step it is set until the next; SetThrottle and SetBreak act one step
    later. The rpm readback of step N is the state rho steps after it was set (rho alternates between 0.5 and 0 in
    many phases of that Play, and is 0.5 in others: Plant timing patterns U and A). The velocity readback lags the
    rpm by half a step more.
S105_19 holds the fitted values (fit_launch_clutch.py --fit, launch 20261106T194516Z-wb-ab6a, e01 play_01; 1020 kg
from the S105 config, the 1.9.0.37 m/s per 1000 rpm measured in that Play). Only the ratios J : clutch_nm : torque :
m set the dynamics, so the absolute scale rests on the mass.

THE MIRROR (LaunchClutch): DC_LaunchClutch's Configure / Reset / Step / ConsumeRestart and counters, every mode (off,
ramp, slip).

THE HARNESS (drive): Mode A as DC_ProbeActuator runs it. Each call reads the last readback and the launch clutch gives
the clutch; the probe's shift policy (ShiftPolicy) or a gear function picks the gear. It takes timing patterns,
callbacks of 1..n steps (llc_v1), an extra clutch delay, a grade and a throttle / brake schedule. metrics() reads the
rows.
"""

import csv
import math

RPM_PER_RADS = 60.0 / (2.0 * math.pi)
STEP_DT = 1.0 / 30.0            # run header clock.step_dt_s of the 1.9.0.37 Play
QUARTERS = 4                    # the plant's time grid: readbacks at whole and half steps
MICRO = 2                       # integration steps per quarter step

# The fit on launch 20261106T194516Z-wb-ab6a, e01 play_01 (fit_launch_clutch.py --fit, from DD-14c's toy values: within
# 1 % of these): the five launches in first gear, 15-step shooting segments, rms 40 rpm / 0.035 m/s. Of the timings
# tried (clutch 0 / 0.5, throttle 1 / 1.5, velocity 0 / 0.5 steps) clutch 0, throttle 1, velocity 0.5 fitted best
TORQUE_KNOTS_RPM = (800.0, 1500.0, 2500.0, 3500.0, 4500.0, 5500.0)
S105_19 = {
    "j_kgm2": 0.038755,          # engine inertia
    "clutch_nm": 117.608,        # clutch capacity at c = 1
    "torque_nm": (58.078, 66.547, 72.458, 76.140, 78.696, 76.839),   # gross torque at throttle 1 on TORQUE_KNOTS_RPM
    "drag_per_s": 1.134,         # (rpm - idle) decays at this rate with throttle 0 and the clutch open (free decays)
    "idle_rpm": 800.0,           # run header vehicle.rpm_idle
    "mass_kg": 1020.0,           # S105 config
    "rolling_n": 378.07,         # fitted: rolling and driveline losses
    "aero_n_per_mps2": 0.4,      # assumed (immaterial below 50 km/h)
    "brake_n": 12400.0,          # 4.3 / 7.9 / 12.5 m/s2 at brake 0.3 / 0.6 / 1.0 (the Play's stops, clutch open)
    "brake_slip_mps": 0.25,      # engine below the speed-implied rpm under a brake (5 % at 15 km/h, 16 % at 4 km/h)
    "stall_rpm": 450.0,          # assumed
}
# m/s per 1000 rpm of the forward gears (index 2..5) in that Play, clutch locked, steady (gear 5 never driven: a guess)
S105_19_MPS_PER_KRPM = {2: 1.540, 3: 2.755, 4: 4.13, 5: 5.6}
# the card's (1.8.0.13 E-01 run 3, gearbox.scripted_mps_per_krpm), what the probe kit and llc_v1 compute with
CARD_MPS_PER_KRPM = {2: 1.54871, 3: 2.772066, 4: 4.169103}
FIRST_GEAR = 2
TOP_GEAR = 5

# Readback timing patterns (rho per step, repeating): U = every readback half a step after the setters, A = alternating
# no step / half a step, A2 = the other phase of A
TIMING = {"U": (0.5,), "A": (0.0, 0.5), "A2": (0.5, 0.0)}


# ---------------------------------------------------------------------------------------------------------------
# the mirror of DC_LaunchClutch

class LaunchClutch:
    """Python mirror of DC_LaunchClutch (Configure, Reset, Step, StepSlip, ConsumeRestart, the counters), statement for
    statement; tests/test_launch_clutch.py ties it to the Enforce source."""

    DEFAULTS = {"mode": "off", "engage_rpm": 1100.0, "full_rpm": 1700.0, "couple_time_s": 0.85,
                "uncouple_time_s": 0.45, "stall_guard_rpm": 600.0, "restart_on_stall": True,
                "restart_cooldown_s": 1.0, "slip_kp": 0.0003, "slip_ki": 0.001, "lock_tolerance_rpm": 100.0,
                "brake_drop_rpm_per_s": 29000.0}
    SLIP_FULL_RPM = 1400.0                  # DC_LaunchClutchConfig.DEFAULT_SLIP_FULL_RPM (Slip())
    RELEASE_LOOKAHEAD_CALLS = 1.5           # predicted rpm: this far ahead in units of the call's own duration
    RELEASE_LOOKAHEAD_ENGAGE_FACTOR = 2.0   # coupled: the lookahead acts below this x engage_rpm
    LOCK_AGREE_CALLS = 2                    # lock-up: engine and driveline agree on this many calls in a row
    SYNC_S = 0.3                            # after an upshift: no lookahead or drop guard without a brake
    LUG_MARGIN_RPM = 300.0                  # coupled under drive: open at stall_guard_rpm + this (at most engage_rpm)
    DROP_GUARD_RPM_PER_S = 12000.0          # coupled under a brake: a fall faster than this opens (a full brake in
                                            # first gear is about 8000)
    BRAKE_LAG_S = 0.016667                  # the brake rule's horizon beyond this call: half a step (the readback)
    BRAKE_FULL_FALL_COMMAND = 0.8           # the brake rule's full fall rate from this brake on, brake / this below
    GAIN_STEP_S = 0.033333                  # slip_kp / slip_ki are per call of this length (the fixed time slice)
    CALL_STEPS_DECAY = 0.05                 # the held call length (in GAIN_STEP_S) decays by this per call

    def __init__(self, **config):
        self.config = dict(self.DEFAULTS, **config)
        self.reset()
        self.engine_seen = -1
        self.max_slip_s = self.max_slip_total_s = 0.0
        self.stalls = self.restarts = self.guard_drops = self.lockups = 0

    @classmethod
    def slip(cls, **config):
        """DC_LaunchClutchConfig.Slip() with changes."""
        return cls(**dict({"mode": "slip", "full_rpm": cls.SLIP_FULL_RPM}, **config))

    def reset(self):
        self.coupling = 0.0
        self.since_restart_s = -1.0
        self.restart_due = False
        self.slip_s = 0.0
        self.locked = False
        self.agree_calls = 0
        self.sync_left_s = 0.0
        self.last_gear = -1
        self.call_steps = 1.0
        self.forget_rpm()

    def forget_rpm(self):
        """DC_LaunchClutch.ForgetRpm: no readback history (no change of the mean, no rate)."""
        self.last_rpm = -1.0
        self.last_mean_rpm = -1.0
        self.history = 0
        self.last_slice_s = 0.0
        self.prev_slice_s = 0.0

    def lookahead_ceiling_rpm(self):
        return self.RELEASE_LOOKAHEAD_ENGAGE_FACTOR * self.config["engage_rpm"]

    def lug_rpm(self):
        return min(self.config["stall_guard_rpm"] + self.LUG_MARGIN_RPM, self.config["engage_rpm"])

    def step(self, dt, in_gear, handbrake, rpm, engine_on, commanded_clutch, dip, driveline_rpm=-1.0, throttle=1.0,
             brake=0.0, gear=-1):
        cfg = self.config
        commanded = min(max(commanded_clutch, 0.0), 1.0)
        slice_s = max(dt, 0.0)
        if self.engine_seen == 1 and not engine_on:
            self.stalls += 1
        self.engine_seen = 0
        if engine_on:
            self.engine_seen = 1
        if cfg["mode"] not in ("ramp", "slip"):
            return commanded
        cooled = self.since_restart_s < 0 or self.since_restart_s >= cfg["restart_cooldown_s"]
        if not engine_on and not handbrake and cfg["restart_on_stall"] and cooled:
            self.restart_due = True
            self.restarts += 1
            self.since_restart_s = 0.0
            self.coupling = 0.0
        driving = in_gear and not handbrake and not dip and engine_on
        if cfg["mode"] == "slip":
            self.step_slip(slice_s, driving, rpm, driveline_rpm, throttle, brake, gear)
        elif not driving:
            self.coupling = 0.0
        elif rpm < cfg["stall_guard_rpm"]:
            if self.coupling > 0:
                self.guard_drops += 1
            self.coupling = 0.0
        elif rpm <= cfg["engage_rpm"]:
            self.coupling = 0.0
        else:
            target = min(max((rpm - cfg["engage_rpm"]) / (cfg["full_rpm"] - cfg["engage_rpm"]), 0.0), 1.0)
            if target > self.coupling:
                self.coupling = min(target, self.coupling + slice_s / cfg["couple_time_s"])
            else:
                self.coupling = max(target, self.coupling - slice_s / cfg["uncouple_time_s"])
        if driving and 0 < self.coupling < 1:
            self.slip_s += slice_s
        else:
            self.slip_s = 0.0
        if self.since_restart_s >= 0:
            self.since_restart_s += slice_s
        self.max_slip_s = max(self.max_slip_s, self.slip_s)
        self.max_slip_total_s = max(self.max_slip_total_s, self.slip_s)
        return min(commanded, self.coupling)

    def open_clutch(self):
        """DC_LaunchClutch.Open: c = 0, not locked, no agreement."""
        self.coupling = 0.0
        self.locked = False
        self.agree_calls = 0

    def step_slip(self, slice_s, driving, rpm, driveline_rpm, throttle, brake, gear):
        cfg = self.config
        engage = cfg["engage_rpm"]
        braking = brake > 0
        drive = throttle > 0 and not braking
        # an upshift: its synchronisation drop is no fall for the lookahead and the drop guard unless the brake could
        # lock the wheels (the history stays, so a brake during the synchronisation still predicts)
        if gear >= 0 and self.last_gear >= 0 and gear > self.last_gear:
            self.sync_left_s = self.SYNC_S
            self.agree_calls = 0
        self.last_gear = gear
        last = self.last_rpm
        if last < 0:
            last = rpm
        mean = 0.5 * (rpm + last)
        mean_last = self.last_mean_rpm
        if mean_last < 0:
            mean_last = mean
        span_s = self.last_slice_s + self.prev_slice_s
        rate = 0.0
        if self.history >= 2 and span_s > 0:
            rate = 2.0 * (mean - mean_last) / span_s
        self.last_rpm = rpm
        self.last_mean_rpm = mean
        if self.history < 3:
            self.history += 1
        self.prev_slice_s = self.last_slice_s
        self.last_slice_s = slice_s
        self.call_steps = max(slice_s / self.GAIN_STEP_S, 1.0, self.call_steps - self.CALL_STEPS_DECAY)
        syncing = self.sync_left_s > 0
        guarded = (braking and brake >= self.BRAKE_FULL_FALL_COMMAND) or not syncing
        self.sync_left_s = max(self.sync_left_s - slice_s, 0.0)
        predicted = rpm + self.RELEASE_LOOKAHEAD_CALLS * slice_s * min(rate, 0.0)
        if not driving:
            self.open_clutch()
            return
        if rpm < cfg["stall_guard_rpm"]:
            if self.coupling > 0:
                self.guard_drops += 1
            self.open_clutch()
            return
        # a brake: open now if the drop it could cause over this call and the readback's lag reaches engage_rpm,
        # before the brake acts (one step later). The fall rate: locked wheels drag the engine at clutch capacity /
        # inertia, a lighter brake (below BRAKE_FULL_FALL_COMMAND) proportionally less; inside an upshift's wait also
        # the fall to the new gear's driveline rpm (the full rate when unknown). A drop within the band between
        # engage_rpm and the lug rpm is left to the coupled floor at engage_rpm
        share = min(brake / self.BRAKE_FULL_FALL_COMMAND, 1.0)
        landing = rpm
        if syncing:
            if driveline_rpm > 0:
                landing = min(rpm, driveline_rpm)
            else:
                share = 1.0
        drop = rpm - landing + cfg["brake_drop_rpm_per_s"] * share * (slice_s + self.BRAKE_LAG_S)
        if braking and drop > engage - self.lug_rpm() and rpm - drop <= engage:
            self.open_clutch()
            return
        if self.locked:
            open_rpm = self.lug_rpm()
            if braking:
                open_rpm = engage
            if rpm <= open_rpm:
                self.open_clutch()
                return
            if guarded and rpm < self.lookahead_ceiling_rpm() and predicted <= open_rpm:
                self.open_clutch()
                return
            if braking and guarded and rate < -self.DROP_GUARD_RPM_PER_S:
                self.open_clutch()
                return
            self.coupling = 1.0
            return
        if rpm <= engage or braking or (guarded and predicted <= self.lug_rpm()):
            self.open_clutch()
            return
        kp = cfg["slip_kp"] / self.call_steps
        ki = cfg["slip_ki"] / (self.call_steps * self.call_steps)
        want = self.coupling + slice_s * (kp * rate + ki * (mean - cfg["full_rpm"]))
        lowest = max(self.coupling - slice_s / cfg["uncouple_time_s"], 0.0)
        highest = min(self.coupling + slice_s / cfg["couple_time_s"], 1.0)
        self.coupling = min(max(want, lowest), highest)                     # Math.Clamp(want, lowest, highest)
        if drive and self.coupling > 0 and driveline_rpm > 0 and abs(rpm - driveline_rpm) <= cfg["lock_tolerance_rpm"]:
            self.agree_calls += 1
        else:
            self.agree_calls = 0
        if self.agree_calls >= self.LOCK_AGREE_CALLS or (drive and self.coupling >= 1):
            self.locked = True
            self.coupling = 1.0
            self.lockups += 1

    def consume_restart(self):
        if not self.restart_due:
            return False
        self.restart_due = False
        return True


# ---------------------------------------------------------------------------------------------------------------
# the plant

class Plant:
    """The fitted S105 powertrain (see the module docstring)."""

    def __init__(self, params=None, rpm=None, speed_mps=0.0, gear=FIRST_GEAR, mps_per_krpm=None, locked=False,
                 grade=0.0, micro=MICRO):
        self.p = dict(S105_19, **(params or {}))
        self.mps_per_krpm = dict(mps_per_krpm or S105_19_MPS_PER_KRPM)
        self.w = (self.p["idle_rpm"] if rpm is None else rpm) / RPM_PER_RADS
        self.v = speed_mps
        self.on = True
        self.locked = locked
        self.grade_n = self.p["mass_kg"] * 9.81 * grade
        self.micro = micro
        self.set_gear(gear)
        self.locked = locked

    @property
    def rpm(self):
        return self.w * RPM_PER_RADS

    def set_gear(self, gear):
        """A gear change: the driveline ratio changes now, a coupled clutch then slips until it is synchronised."""
        self.gear = gear
        self.g = self.mps_per_krpm[gear] / 1000.0 * RPM_PER_RADS        # m/s per rad/s of engine
        self.locked = False

    def torque(self, rpm):
        knots, values = TORQUE_KNOTS_RPM, self.p["torque_nm"]
        if rpm <= knots[0]:
            return values[0]
        for r0, t0, r1, t1 in zip(knots, values, knots[1:], values[1:]):
            if rpm <= r1:
                return t0 + (t1 - t0) * (rpm - r0) / (r1 - r0)
        return values[-1]

    def engine_torque(self, throttle):
        p = self.p
        if not self.on:
            return -p["j_kgm2"] * p["drag_per_s"] * self.w
        drag = p["j_kgm2"] * p["drag_per_s"] * max(self.w - p["idle_rpm"] / RPM_PER_RADS, 0.0)
        return throttle * self.torque(self.rpm) - drag

    def resistance(self, brake):
        p = self.p
        if self.v <= 1e-4:
            return 0.0
        return p["rolling_n"] + self.grade_n + p["aero_n_per_mps2"] * self.v * self.v + brake * p["brake_n"]

    def force_start(self):
        """ForceStartEngine: the engine runs again at idle or above."""
        self.on = True
        self.w = max(self.w, self.p["idle_rpm"] / RPM_PER_RADS)

    def advance(self, seconds, throttle, brake, clutch):
        p = self.p
        h = seconds / self.micro
        capacity = clutch * p["clutch_nm"]
        lag = p["brake_slip_mps"] * brake
        j, m, g = p["j_kgm2"], p["mass_kg"], self.g
        for _ in range(self.micro):
            wg = max(self.v - lag, 0.0) / g
            te = self.engine_torque(throttle)
            fres = self.resistance(brake)
            if self.v <= 1e-4 and self.grade_n < 0:
                fres = self.grade_n                                         # downhill: gravity alone at rest
            if capacity > 0 and (self.locked or abs(self.w - wg) < 0.3):
                acc = (te - fres * g) / (j + m * g * g)
                if abs(te - j * acc) <= capacity:
                    self.locked = True
                    self.v = max(self.v + acc * g * h, 0.0)
                    self.w = max(self.v - lag, 0.0) / g
                    self.stall_check()
                    continue
            self.locked = False
            slip = self.w - wg
            tc = capacity * (1.0 if slip > 0 else -1.0 if slip < 0 else 0.0)
            w_new = max(self.w + (te - tc) / j * h, 0.0)
            v_new = max(self.v + (tc / g - fres) / m * h, 0.0)
            wg_new = max(v_new - lag, 0.0) / g
            if capacity > 0 and slip * (w_new - wg_new) < 0:                # synchronised within the step: lock
                self.locked = True
                w_new = (j * w_new + m * g * g * wg_new) / (j + m * g * g)
                v_new = w_new * g + (lag if w_new > 0 else 0.0)
            self.w, self.v = w_new, v_new
            self.stall_check()

    def stall_check(self):
        if self.on and self.rpm < self.p["stall_rpm"]:
            self.on = False


class ShiftPolicy:
    """DC_ProbeActuator.RunShiftPolicy while driving (manual gearbox, gear -1): first gear below first_below_kmh, up one
    at rpm >= up_rpm, down one at rpm <= down_rpm, at most one change per min_s; rpm and speed of the last readback."""

    def __init__(self, up_rpm=4500.0, down_rpm=1800.0, min_s=1.5, first_below_kmh=8.0, first_only=False):
        self.up_rpm, self.down_rpm, self.min_s = up_rpm, down_rpm, min_s
        self.first_below_kmh, self.first_only = first_below_kmh, first_only
        self.gear = FIRST_GEAR
        self.since_s = 0.0

    def step(self, dt, rpm, kmh):
        self.since_s += dt
        gear = self.gear
        if self.first_only or kmh < self.first_below_kmh:
            gear = FIRST_GEAR
        elif self.since_s >= self.min_s:
            if rpm >= self.up_rpm and gear < TOP_GEAR:
                gear += 1
            elif rpm <= self.down_rpm and gear > FIRST_GEAR:
                gear -= 1
        if gear != self.gear:
            self.gear = gear
            self.since_s = 0.0
        return gear


def drive(clutch, schedule, seconds, timing="U", calls=(1,), plant=None, policy=None, gear_fn=None,
          clutch_delay_steps=0.0, mps_per_krpm=None, commanded=1.0):
    """Mode A over the plant as DC_ProbeActuator runs it (the module docstring). clutch: a LaunchClutch;
    schedule(t, kmh) -> (throttle, brake); timing: a TIMING key or a rho tuple; calls: the steps each callback covers
    (repeating); gear_fn(t, kmh) -> gear overrides the policy (gear_scripted); mps_per_krpm: what the controller's
    driveline rpm uses (CARD_MPS_PER_KRPM by default; the plant uses its own). Returns rows (t, c, rpm read, engine on,
    km/h, locked, true rpm, gear, throttle, brake)."""
    plant = plant or Plant()
    policy = policy or ShiftPolicy()
    policy.gear = plant.gear
    rho = TIMING[timing] if isinstance(timing, str) else tuple(timing)
    known = dict(CARD_MPS_PER_KRPM if mps_per_krpm is None else mps_per_krpm)
    total = int(round(seconds / STEP_DT))
    inputs = []                   # per step: (throttle, brake, clutch)
    w_hist = {0: plant.rpm}
    v_hist = {0: plant.v}
    rows = []
    rpm_read, on_read, v_read = plant.rpm, plant.on, plant.v
    q, step, call = 0, 0, 0
    delay_q = int(round(clutch_delay_steps * QUARTERS))
    while step < total:
        covered = calls[call % len(calls)]
        call += 1
        t = step * STEP_DT
        throttle, brake = schedule(t, v_read * 3.6)
        gear = gear_fn(t, v_read * 3.6) if gear_fn else policy.step(covered * STEP_DT, rpm_read, v_read * 3.6)
        if gear_fn:
            policy.gear = gear
        if gear != plant.gear:
            plant.set_gear(gear)
        driveline = v_read / known[gear] * 1000.0 if known.get(gear, 0) > 0 else -1.0
        c = clutch.step(covered * STEP_DT, True, False, rpm_read, on_read, commanded, commanded <= 0, driveline,
                        throttle, brake, gear)
        for _ in range(covered):
            inputs.append((throttle, brake, c))
        last = step + covered - 1
        end_q = (last + 1) * QUARTERS
        while q < end_q:
            k_in = math.floor(q / QUARTERS) - 1                             # throttle and brake: one step later
            thr_in, brk_in = (inputs[k_in][0], inputs[k_in][1]) if k_in >= 0 else (0.0, 0.0)
            k_c = (q - delay_q) // QUARTERS
            c_in = inputs[k_c][2] if k_c >= 0 else 0.0
            plant.advance(STEP_DT / QUARTERS, thr_in, brk_in, c_in)
            q += 1
            w_hist[q] = plant.rpm
            v_hist[q] = plant.v
        r = rho[last % len(rho)]
        q_rpm = int(round((last + r) * QUARTERS))
        q_v = int(round((last + r - 0.5) * QUARTERS))
        rpm_read = w_hist.get(q_rpm, plant.rpm)
        v_read = v_hist.get(max(q_v, 0), plant.v)
        on_read = plant.on
        step += covered
        rows.append((step * STEP_DT, c, rpm_read, plant.on, plant.v * 3.6, plant.locked, plant.rpm, gear, throttle,
                     brake))
        if clutch.consume_restart():
            plant.force_start()
        for old in [k for k in w_hist if k < q - 4 * QUARTERS]:
            del w_hist[old], v_hist[old]
    return rows


def openings(rows):
    """Clutch openings: c > 0 on one call, 0 on the next, whatever the throttle and brake (the chatter's dumps, and a
    stop's opening)."""
    return sum(1 for a, b in zip(rows, rows[1:]) if a[1] > 0 and b[1] == 0)


# The oscillation criterion's amplitude deadband (DD-14d round 2, stated): a reversal of c counts only when both changes
# are at least this. Over 8127 launches with random 1-3 step callback sequences, engine inertia 0.7 / 1 / 1.3 x, throttle
# 0.3 / 0.5 / 1.0 and all three readback timings, the largest change in a run of two alternations was 0.0097 (about 1.1
# Nm of clutch torque): the dither of the call-length gain scaling, not an oscillation
ALTERNATION_DEADBAND = 0.01


def alternations(values, eps=ALTERNATION_DEADBAND):
    """The longest run of step-to-step sign changes of the change of c, each change at least eps
    (ALTERNATION_DEADBAND): 0 or 1 = no oscillation."""
    deltas = [b - a for a, b in zip(values, values[1:])]
    best = run = 0
    for a, b in zip(deltas, deltas[1:]):
        if abs(a) >= eps and abs(b) >= eps and a * b < 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def metrics(rows, start=0):
    """Openings, stalls, c's reversals and alternation runs, first c = 1 (lock-up) time, the lowest true and read rpm,
    and the end state of rows[start:]."""
    r = rows[start:]
    cs = [x[1] for x in r]
    nz = [b - a for a, b in zip(cs, cs[1:]) if abs(b - a) > 1e-6]
    return {"openings": openings(r), "stalls": sum(1 for a, b in zip(r, r[1:]) if a[3] and not b[3]),
            "reversals": sum(1 for a, b in zip(nz, nz[1:]) if a * b < 0), "alternations": alternations(cs),
            "lockup_s": next((x[0] for x in r if x[1] >= 1.0), None),
            "min_rpm": min(x[6] for x in r), "min_read_rpm": min(x[2] for x in r),
            "end_c": cs[-1], "end_kmh": r[-1][4], "end_on": r[-1][3]}


def launch(clutch, throttle, seconds=6.0, grade=0.0, **kw):
    """Throttle from standstill at idle in first gear (an E-01 throttle phase or move check)."""
    plant = kw.pop("plant", None) or Plant(params=kw.pop("params", None), grade=grade)
    return drive(clutch, lambda t, kmh: (throttle, 0.0), seconds, plant=plant, **kw)


def brake_to_stop(clutch, brake, from_kmh, throttle=0.6, **kw):
    """Throttle up to from_kmh (the shift policy shifting), then brake to a stop; returns (rows, index of the first
    braking call)."""
    state = {"braking": False}

    def schedule(t, kmh):
        if kmh >= from_kmh:
            state["braking"] = True
        return (0.0, brake) if state["braking"] else (throttle, 0.0)

    plant = kw.pop("plant", None) or Plant(params=kw.pop("params", None))
    rows = drive(clutch, schedule, 40.0, plant=plant, **kw)
    first = next(i for i, x in enumerate(rows) if x[9] > 0)
    stop = next((i for i, x in enumerate(rows) if i > first and x[4] <= 0.0), len(rows) - 1)
    return rows[:stop + 30], first


# ---------------------------------------------------------------------------------------------------------------
# E-01 steps and the shooting fit (fit_launch_clutch.py)

STEP_COLUMNS = ("physics_step", "phase", "segment", "cmd_throttle", "cmd_brake", "cmd_clutch", "input_post_clutch",
                "cmd_gear", "cur_gear", "rpm", "engine_on", "speed_kmh", "vel_l_z", "cmd_handbrake", "timeslice_s",
                "world_time_ms")
LAUNCH_PHASES = {"move_check": 0.5, "throttle_0.30": 0.3, "throttle_0.60": 0.6, "throttle_1.00": 1.0,
                 "gear_scripted": 0.4}


OPTIONAL_STEP_COLUMNS = ("rpm_feedback",)     # DD-14d: EngineGetRPMFeedback, recorded only (plays from DD-14d on)


def read_steps(path):
    """The STEP_COLUMNS of an e01_steps.csv (or of the trimmed fixture), numbers as floats, and the
    OPTIONAL_STEP_COLUMNS a play has. A missing column raises KeyError."""
    out = []
    with open(path, newline="", encoding="ascii") as f:
        for row in csv.DictReader(f):
            d = {"phase": row["phase"], "segment": row["segment"]}
            for key in STEP_COLUMNS[3:]:
                d[key] = float(row[key])
            for key in OPTIONAL_STEP_COLUMNS:
                if row.get(key) not in (None, ""):
                    d[key] = float(row[key])
            d["physics_step"] = int(row["physics_step"])
            out.append(d)
    return out


def readback_timing(rows, reach=6):
    """rho per row: 0.5 when input_post_clutch shows the command of a step that changed it, 0 when it still shows the
    previous one. A row without a change takes 0.5 when no row within reach shows 0 (timing U), else the alternation
    of its nearest row that shows one (timing A)."""
    n = len(rows)
    seen = [None] * n
    for i in range(1, n):
        a, b = rows[i - 1], rows[i]
        if abs(b["cmd_clutch"] - a["cmd_clutch"]) < 1e-4:
            continue
        seen[i] = 0.5 if abs(b["input_post_clutch"] - b["cmd_clutch"]) < 1e-4 else 0.0
    out = []
    for i in range(n):
        if seen[i] is not None:
            out.append(seen[i])
            continue
        near = [j for j in range(max(i - reach, 0), min(i + reach + 1, n)) if seen[j] is not None]
        if not near or all(seen[j] == 0.5 for j in near):
            out.append(0.5)
            continue
        j = min(near, key=lambda k: (abs(k - i), k))
        out.append(seen[j] if (i - j) % 2 == 0 else 0.5 - seen[j])
    return out


def launch_windows(rows, seconds=4.0):
    """{phase: (first, last)} row indices of each launch in LAUNCH_PHASES: from the first excite step with throttle
    to seconds later, while excite and first gear."""
    out = {}
    for name in LAUNCH_PHASES:
        first = next((i for i, r in enumerate(rows) if r["phase"] == name and r["segment"] == "excite"
                      and r["cmd_throttle"] > 0), None)
        if first is None:
            continue
        last = first
        while (last + 1 < len(rows) and rows[last + 1]["phase"] == name and rows[last + 1]["segment"] == "excite"
               and rows[last + 1]["cmd_gear"] == FIRST_GEAR and last + 1 - first < seconds / STEP_DT):
            last += 1
        out[name] = (first, last)
    return out


def shoot(params, rows, rho, first, last, per_krpm=S105_19_MPS_PER_KRPM[FIRST_GEAR]):
    """Predicted (rpm, speed) readbacks of rows first..last from the measured state at row first - 1 with the recorded
    commands (clutch from its step, throttle and brake one step later, the velocity half a step behind the rpm)."""
    a = rows[first - 1]
    plant = Plant(params=params, rpm=a["rpm"], mps_per_krpm={FIRST_GEAR: per_krpm})
    v0, v1 = rows[first - 1]["vel_l_z"], rows[first]["vel_l_z"]
    plant.v = max(v0 + 0.5 * (v1 - v0), 0.0)
    plant.locked = a["cmd_clutch"] > 0 and abs(plant.w - plant.v / plant.g) * RPM_PER_RADS < 15
    q0 = int(round((first - 1 + rho[first - 1]) * QUARTERS))
    w_hist, v_hist = {q0: plant.rpm}, {q0: plant.v}
    for q in range(q0, (last + 1) * QUARTERS):
        k_c = min(max(q // QUARTERS, 0), len(rows) - 1)
        k_in = min(max(q // QUARTERS - 1, 0), len(rows) - 1)
        plant.advance(STEP_DT / QUARTERS, rows[k_in]["cmd_throttle"], rows[k_in]["cmd_brake"], rows[k_c]["cmd_clutch"])
        w_hist[q + 1] = plant.rpm
        v_hist[q + 1] = plant.v
    rpm, speed = [], []
    for i in range(first, last + 1):
        rpm.append(w_hist.get(int(round((i + rho[i]) * QUARTERS)), plant.rpm))
        speed.append(v_hist.get(int(round((i + rho[i] - 0.5) * QUARTERS)), v_hist[q0]))
    return rpm, speed


def shooting_residuals(params, rows, rho, windows, length=15, stride=5):
    """(rpm errors, speed errors) of the length-step shooting segments every stride steps over the windows."""
    e_rpm, e_v = [], []
    for first, last in windows.values():
        s = first
        while s + 3 <= last:
            e = min(s + length - 1, last)
            rpm, speed = shoot(params, rows, rho, s, e)
            for k, i in enumerate(range(s, e + 1)):
                e_rpm.append(rpm[k] - rows[i]["rpm"])
                e_v.append(speed[k] - rows[i]["vel_l_z"])
            s += stride
    return e_rpm, e_v


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")
