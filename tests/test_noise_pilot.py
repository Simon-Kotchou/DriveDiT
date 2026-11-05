#!/usr/bin/env python3
"""DD-7-ident's perturbation pilot (src/enfusion/datacapture/DC_Noise.c and its wiring) against dc-2.0 and the targets.

    TMPDIR=/dev/shm/<x> python3 -I -B -m pytest --noconftest -p no:cacheprovider -q tests/test_noise_pilot.py

The Enforce code cannot run here, so these tests read it, as tests/test_expert_driver_dc2.py does:
  - DC_NoiseRng: the xorshift32 port on emulated Enforce ints gives the SelfTest constants of the source;
  - NoiseMirror is DC_Noise statement for statement (the parameter table is parsed from the source, the order of
    the five draws of a segment and the formulas of every kind are asserted in the source), and over a simulated hour
    of the pilot (the T0 loop both ways round, DC_ExpertHL's pure pursuit and speed profile, a lagged kinematic car,
    5 Hz decisions over 30 Hz steps) the pilot_v1 defaults meet the research targets (delta 2026-11 section 3.3):
    at least 30% of decision steps off-expert, at least 40% of the segments longitudinal, at least 10 stops (stand
    still, then drive off) per hour, the turns of each signed kind balanced within 2 per episode, the applied
    curvature's left and right turning within 1.2x over the two directions, at least 90% of the segments recovered
    (|lane_lat_err| < 0.5 m and |lane_head_err| < 3 deg held 2 s, reached within 10 s) and no termination;
  - one seed gives one schedule (kinds, signs, amplitudes, durations, gaps), another seed another;
  - the contract: the perturbation acts on the HL command (action_cmd) only, exp_*_now stays the clean expert
    (action_label), noise_active / noise_id come from the held command, a disturbance only in a mode-N episode, the
    perturb_start / perturb_end payload is noise_id, kind, amplitude, duration_s, the core records mode N, and a trial
    conf without "perturb" (T0L) keeps every episode mode E;
  - the trial conf: DC_NoiseConfig refuses an unknown profile, perturb_params without one, an unknown, misspelt or
    repeated key, a value out of range and a lo above its hi (a Python mirror of Read on the same text).
"""

import json
import math
import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SOURCES = os.path.join(REPO, "src", "enfusion", "datacapture")
NOISE_SOURCE = os.path.join(SOURCES, "DC_Noise.c")
EXPERT_SOURCE = os.path.join(SOURCES, "DC_ExpertHL.c")
RECORDER_SOURCE = os.path.join(SOURCES, "DC_StreamRecorder.c")
EGO_SOURCE = os.path.join(SOURCES, "DC_EgoSession.c")
SESSION_SOURCE = os.path.join(SOURCES, "DC_Session.c")
MANIFEST = os.path.join(REPO, "addons", "DataCapture", "deploy_manifest.txt")
ROUTES = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes")
ROUTE_CW = os.path.join(ROUTES, "arland_t0_loop.json")
ROUTE_CCW = os.path.join(ROUTES, "arland_t0_loop_rev.json")
STEP_DT = 1.0 / 30.0
STEPS_PER_DECISION = 6          # N at 5 Hz (dc-2.0 D-B)
DECISION_DT = STEPS_PER_DECISION * STEP_DT
# DC_ExpertDriver attribute defaults (DC_ExpertHL.c, checked in test_expert_defaults)
EXPERT = {"cruise_kmh": 40.0, "alat": 2.75, "accel_max": 1.5, "brake_max": 4.0, "profile_decel": 2.0,
          "speed_gain": 1.0, "preview_s": 1.0, "lookahead_s": 0.9, "lookahead_min": 6.0, "lookahead_max": 15.0,
          "kappa_max": 0.2, "end_tol": 5.0, "stop_kmh": 1.0, "offroad_margin": 0.5, "offroad_s": 3.0,
          "stuck_kmh": 1.0, "stuck_s": 20.0, "timeout_s": 480.0}
TARGET_OFF_EXPERT = 0.30
TARGET_LONGITUDINAL = 0.40
TARGET_STOPS_PER_HOUR = 10
TARGET_LR_RATIO = 1.2
TARGET_RECOVERED = 0.90


def read(path):
    with open(path, "r", encoding="ascii", newline="") as f:
        return f.read()


def code_only(source):
    return "\n".join(line.split("//", 1)[0] for line in source.split("\n"))


def flat(text):
    return " ".join(code_only(text).split())


def class_text(source, cls):
    found = re.search(r"\nclass %s(?:\s*:\s*\w+)?\s*\n\{" % cls, source)
    rest = source[found.start() + 1:]
    following = re.search(r"\n(?://[^\n]*\n|\[[^\n]*\]\n)*class \w+", rest[1:])
    return rest if following is None else rest[:following.start() + 1]


def method_body(source, cls, method):
    body = class_text(source, cls)
    head = re.search(r"\n\t(?:(?:override|protected|static) )*[\w<>]+ %s\(" % method, body)
    return body[head.start():].split("\n\t}\n", 1)[0]


def enforce_array(text, name):
    """The items of `... <name> = { ... };` (strings or numbers)."""
    block = re.search(r"\b%s = \{(.*?)\};" % name, text, re.S).group(1)
    strings = re.findall(r'"([^"]*)"', block)
    if strings:
        return strings
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", code_only(block))]


# ------------------------------------------------------------------------------------------------
# Enforce ints and DC_NoiseRng
# ------------------------------------------------------------------------------------------------
def i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def shr(v, c):
    return (v >> c) & (0x7FFFFFFF >> (c - 1))


class Rng:
    """DC_NoiseRng on emulated Enforce ints."""
    SEED_MIX, SEED_ZERO, WARMUP, UNIT = 0x5DEECE66, 0x6D2B79F5, 8, 16777216.0

    def __init__(self):
        self.state = self.SEED_ZERO

    def seed(self, seed, stream):
        x = i32(i32(seed << 8) ^ stream ^ self.SEED_MIX)
        self.state = self.SEED_ZERO if x == 0 else x
        for _ in range(self.WARMUP):
            self.next_int()

    def next_int(self):
        x = self.state
        x = i32(x ^ i32(x << 13))
        x = i32(x ^ shr(x, 17))
        x = i32(x ^ i32(x << 5))
        self.state = x
        return x

    def uniform01(self):
        return shr(self.next_int(), 8) / self.UNIT

    def uniform_open01(self):
        return (shr(self.next_int(), 8) + 1) / self.UNIT

    def gauss(self):
        u1 = self.uniform_open01()
        u2 = self.uniform01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def enforce_round(x):
    """Math.Round: half away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


# ------------------------------------------------------------------------------------------------
# DC_NoiseConfig and DC_Noise, statement for statement
# ------------------------------------------------------------------------------------------------
class Table:
    """DC_NoiseConfig's parameter table, parsed from the source."""

    def __init__(self, source):
        cfg = class_text(source, "DC_NoiseConfig")
        self.keys = enforce_array(cfg, "s_aKeys")
        self.defaults = enforce_array(cfg, "s_aPilotV1")
        self.lo = enforce_array(cfg, "s_aMin")
        self.hi = enforce_array(cfg, "s_aMax")
        self.pairs = enforce_array(cfg, "s_aPairs")
        noise = class_text(source, "DC_Noise")
        self.kinds = enforce_array(noise, "s_aKindNames")
        self.weight_keys = enforce_array(noise, "s_aWeightKeys")


class ConfigError(ValueError):
    pass


def read_config(table, conf_text):
    """Mirror of DC_NoiseConfig.Read on a trial conf's text: (profile, {key: value}, given) or ConfigError."""
    at = conf_text.find('"perturb')
    while at >= 0:
        nxt = at + len('"perturb')
        if not (conf_text.startswith('"', nxt) or conf_text.startswith('_params"', nxt)):
            raise ConfigError("misspelt perturb key")
        at = conf_text.find('"perturb', nxt)
    conf = json.loads(conf_text)
    profile = "none"
    if "perturb" in conf:
        if not isinstance(conf["perturb"], str):
            raise ConfigError("perturb is not a string")
        if conf["perturb"] not in ("none", "pilot_v1"):
            raise ConfigError("unknown profile")
        profile = conf["perturb"]
    values = dict(zip(table.keys, table.defaults))
    if "perturb_params" not in conf:
        return profile, values, {}
    if profile == "none":
        raise ConfigError("perturb_params without a profile")
    if conf_text.count('"perturb_params"') != 1 or not isinstance(conf["perturb_params"], dict):
        raise ConfigError("perturb_params is repeated or not an object")
    keys = re.findall(r'"(\w+)"\s*:', conf_text[conf_text.index('"perturb_params"') + 16:].split("}", 1)[0])
    if len(keys) != len(set(keys)):
        raise ConfigError("repeated key")
    given = {}
    for key, value in conf["perturb_params"].items():
        if key not in values:
            raise ConfigError("unknown key %s" % key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("not a number")
        values[key] = float(value)
        given[key] = float(value)
    for key, lo, hi in zip(table.keys, table.lo, table.hi):
        if not lo <= values[key] <= hi:
            raise ConfigError("%s out of range" % key)
    for pair in table.pairs:
        if values[pair + "_lo"] > values[pair + "_hi"]:
            raise ConfigError("%s lo above hi" % pair)
    if sum(values[k] for k in table.weight_keys) <= 0:
        raise ConfigError("no weight")
    if values["stop_dwell_s_hi"] > values["stop_max_s"]:
        raise ConfigError("dwell above the cap")
    return profile, values, given


class Cmd:
    __slots__ = ("kappa", "accel", "dir", "noise", "noise_id")

    def __init__(self, kappa=0.0, accel=0.0, direction=1, noise=False, noise_id=-1):
        self.kappa, self.accel, self.dir, self.noise, self.noise_id = kappa, accel, direction, noise, noise_id

    def copy(self):
        return Cmd(self.kappa, self.accel, self.dir, self.noise, self.noise_id)


class NoiseMirror:
    """DC_Noise: BeginEpisode, Decide, ApplySegment, CanStart, StartSegment, EndSegment, DrawSegment, UpdateRecovery,
    EndEpisode."""
    K_PULSE, K_STEP, K_OU, A_PULSE, A_OU, SPEED, STOP, BRAKE = range(8)
    BALANCE_MAX = 2

    def __init__(self, table, values, decision_dt, kappa_cmd_max, accel_max, brake_max, stop_speed):
        self.p, self.dt = values, decision_dt
        self.kappa_cmd_max, self.accel_max, self.brake_max, self.stop_speed = (kappa_cmd_max, accel_max, brake_max,
                                                                              stop_speed)
        self.weights = [values[k] for k in table.weight_keys]
        self.schedule, self.innovation = Rng(), Rng()
        self.next_noise_id = 0
        self.kind_count = [0] * 8
        self.decisions = self.perturbed = 0
        self.recovered_in_window = self.recovered_late = self.not_recovered = 0
        self.stood = 0                  # stop segments that reached a standstill (a stop and start)
        self.drawn = []                 # (kind, sign, amplitude, planned steps, gap steps) per DrawSegment
        self.segments = []              # per started segment: dict(kind, id, sign, amplitude, steps applied)
        self.episode_n = False
        self.active = False

    def steps(self, seconds):
        return max(enforce_round(seconds / self.dt), 0)

    def begin_episode(self, trial_seed, episode_id):
        self.episode = episode_id
        self.schedule.seed(trial_seed, episode_id * 2)
        self.innovation.seed(trial_seed, episode_id * 2 + 1)
        self.episode_n = self.schedule.uniform01() < self.p["episode_n_share"]
        self.balance = [0] * 8
        self.active = False
        self.noise_id = -1
        self.await_recovery = False
        self.recovered_steps = 0
        self.draw_segment()
        self.gap_steps = self.steps(self.p["start_delay_s"])
        return self.episode_n

    def decide(self, clean, inp):
        applied = clean.copy()
        applied.noise, applied.noise_id = False, -1
        if not self.episode_n:
            return applied
        self.decisions += 1
        self.update_recovery(inp)
        if not self.active:
            if self.gap_steps > 0:
                self.gap_steps -= 1
                return applied
            if not self.can_start(clean, inp):
                return applied
            self.start_segment()
        if clean.dir != 1 or not self.apply_segment(clean, inp, applied):
            self.end_segment()
            applied = clean.copy()
            applied.noise, applied.noise_id = False, -1
            return applied
        applied.noise, applied.noise_id = True, self.noise_id
        self.step += 1
        self.perturbed += 1
        self.segments[-1]["applied"] += 1
        return applied

    def kappa_scale(self, v):
        v = max(v, self.p["min_speed_mps"])
        return min(self.kappa_cmd_max, self.p["kappa_alat_mps2"] / (v * v))

    def pulse(self, i, n):
        return math.sin(math.pi * (i + 0.5) / n)

    def next_ou(self, tau):
        rho = math.exp(-self.dt / tau)
        self.ou = rho * self.ou + self.amplitude * math.sqrt(1.0 - rho * rho) * self.innovation.gauss()
        return self.ou

    def apply_segment(self, clean, inp, applied):
        v = inp["speed"]
        k = self.kind
        if k in (self.K_PULSE, self.K_STEP, self.K_OU):
            if self.step >= self.planned:
                return False
            dk = self.amplitude * self.kappa_scale(v)
            if k == self.K_PULSE:
                dk *= self.pulse(self.step, self.planned)
            elif k == self.K_OU:
                dk = self.next_ou(self.p["k_ou_tau_s"]) * self.kappa_scale(v)
            applied.kappa = min(max(clean.kappa + dk, -self.kappa_cmd_max), self.kappa_cmd_max)
            return True
        if k in (self.A_PULSE, self.A_OU):
            if self.step >= self.planned:
                return False
            da = self.amplitude * self.pulse(self.step, self.planned)
            if k == self.A_OU:
                da = self.next_ou(self.p["a_ou_tau_s"])
            applied.accel = min(max(clean.accel + da, -self.p["decel_cmd_max"]), self.p["accel_cmd_max"])
            return True
        if k == self.SPEED:
            if self.step >= self.planned:
                return False
            target = self.amplitude * inp["profile_ahead"]
            applied.accel = min(max(self.p["speed_gain"] * (target - v), -self.brake_max), self.accel_max)
            return True
        if k == self.STOP:
            if self.step >= self.cap:
                return False
            if not self.stopped and v < self.stop_speed:
                self.stopped = True
                self.stood += 1
            if self.stopped:
                if self.dwell_done >= self.planned:
                    return False
                self.dwell_done += 1
            applied.dir = 0
            applied.accel = self.amplitude
            return True
        if self.step >= self.cap or v < self.p["brake_end_mps"]:
            return False
        applied.accel = self.amplitude
        return True

    def can_start(self, clean, inp):
        if clean.dir != 1 or inp["speed"] < self.p["min_speed_mps"]:
            return False
        if inp["remaining"] < self.p["end_margin_m"]:
            return False
        return self.recovered_steps >= self.steps(self.p["recover_hold_s"])

    def start_segment(self):
        self.active = True
        self.noise_id = self.next_noise_id
        self.next_noise_id += 1
        self.step = 0
        self.stopped = False
        self.dwell_done = 0
        self.ou = 0.0
        self.await_recovery = False
        self.kind_count[self.kind] += 1
        self.segments.append({"kind": self.kind, "id": self.noise_id, "sign": self.sign, "episode": self.episode,
                              "amplitude": self.amplitude, "applied": 0})

    def end_segment(self):
        self.active = False
        self.noise_id = -1
        self.await_recovery = True
        self.since_end = 0
        self.draw_segment()

    def lerp(self, key, u):
        lo, hi = self.p[key + "_lo"], self.p[key + "_hi"]
        return lo + (hi - lo) * u

    def draw_segment(self):
        u_kind, u_sign, u_amp, u_dur = (self.schedule.uniform01() for _ in range(4))
        u_gap = self.schedule.uniform_open01()
        total = sum(self.weights)
        self.kind = -1
        cumulative = 0.0
        for k, w in enumerate(self.weights):
            cumulative += w
            if self.kind < 0 and w > 0 and u_kind * total < cumulative:
                self.kind = k
        for last in range(len(self.weights) - 1, -1, -1):
            if self.kind >= 0:
                break
            if self.weights[last] > 0:
                self.kind = last
        sign = -1 if u_sign >= 0.5 else 1
        balance = self.balance[self.kind]
        if balance >= self.BALANCE_MAX:
            sign = -1
        elif balance <= -self.BALANCE_MAX:
            sign = 1
        self.sign = sign
        self.cap = 0
        k, p = self.kind, self.p
        if k == self.K_PULSE:
            self.amplitude, duration = sign * self.lerp("k_pulse_amp", u_amp), self.lerp("k_pulse_s", u_dur)
        elif k == self.K_STEP:
            self.amplitude, duration = sign * self.lerp("k_step_amp", u_amp), self.lerp("k_step_s", u_dur)
        elif k == self.K_OU:
            self.amplitude, duration = p["k_ou_sigma"], self.lerp("ou_s", u_dur)
        elif k == self.A_PULSE:
            self.amplitude, duration = sign * self.lerp("a_pulse_amp", u_amp), self.lerp("a_pulse_s", u_dur)
        elif k == self.A_OU:
            self.amplitude, duration = p["a_ou_sigma"], self.lerp("ou_s", u_dur)
        elif k == self.SPEED:
            self.amplitude, duration = self.lerp("speed_f", u_amp), self.lerp("speed_s", u_dur)
        elif k == self.STOP:
            self.amplitude, duration = -self.lerp("stop_decel", u_amp), self.lerp("stop_dwell_s", u_dur)
            self.cap = self.steps(p["stop_max_s"])
        else:
            self.amplitude, duration = -p["brake_mps2"], p["brake_max_s"]
            self.cap = self.steps(duration)
        if k in (self.K_PULSE, self.K_STEP, self.A_PULSE):
            self.balance[k] = balance + sign
        self.planned = max(self.steps(duration), 1)
        self.gap_steps = self.steps(p["gap_min_s"] - p["gap_mean_s"] * math.log(u_gap))
        self.drawn.append((k, sign, round(self.amplitude, 6), self.planned, self.gap_steps))

    def update_recovery(self, inp):
        inside = (abs(inp["lateral"]) < self.p["recover_lat_m"]
                  and abs(inp["heading_err"]) < math.radians(self.p["recover_head_deg"]))
        self.recovered_steps = self.recovered_steps + 1 if inside else 0
        if not self.await_recovery:
            return
        self.since_end += 1
        hold = self.steps(self.p["recover_hold_s"])
        if self.recovered_steps < hold or self.recovered_steps < 1:
            return
        self.await_recovery = False
        if self.since_end - self.recovered_steps <= self.steps(self.p["recover_window_s"]):
            self.recovered_in_window += 1
        else:
            self.recovered_late += 1

    def end_episode(self):
        if self.active:
            self.active = False
            self.noise_id = -1
            self.await_recovery = True
        if self.await_recovery:
            self.not_recovered += 1
        self.await_recovery = False
        self.episode_n = False


# ------------------------------------------------------------------------------------------------
# The route, DC_ExpertHL and a car: one simulated pilot
# ------------------------------------------------------------------------------------------------
def wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


class RouteMirror:
    """DC_Route: points, arc lengths, curvature over +-5 m, Project in a window, the lane corridor."""

    def __init__(self, path):
        with open(path, "r", encoding="ascii") as f:
            obj = json.load(f)
        self.xz = [(p[0], p[2]) for p in obj["polyline"]]
        self.s = [0.0]
        for a, b in zip(self.xz, self.xz[1:]):
            self.s.append(self.s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.length = self.s[-1]
        self.kappa = []
        for sj in self.s:
            lo, hi = max(sj - 5.0, 0.0), min(sj + 5.0, self.length)
            self.kappa.append(wrap(self.heading_at(hi) - self.heading_at(lo)) / (hi - lo) if hi - lo > 0.001 else 0.0)
        self.corr_s = obj["source"]["lane_corridor_s_m"]
        self.corr_half = obj["source"]["lane_corridor_half_width_m"]

    def seg(self, s):
        if s <= 0:
            return 0
        if s >= self.s[-1]:
            return len(self.s) - 2
        lo, hi = 0, len(self.s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.s[mid] <= s:
                lo = mid
            else:
                hi = mid
        return lo

    def point_at(self, s):
        i = self.seg(s)
        t = min(max((s - self.s[i]) / (self.s[i + 1] - self.s[i]), 0.0), 1.0)
        a, b = self.xz[i], self.xz[i + 1]
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def heading_at(self, s):
        i = self.seg(s)
        a, b = self.xz[i], self.xz[i + 1]
        return math.atan2(b[0] - a[0], b[1] - a[1])

    def project(self, x, z, s_lo, s_hi):
        s_lo, s_hi = max(0.0, s_lo), min(self.length, s_hi)
        best = None
        i = self.seg(s_lo)
        while i < len(self.xz) - 1 and self.s[i] <= s_hi:
            a, b = self.xz[i], self.xz[i + 1]
            seg = self.s[i + 1] - self.s[i]
            t_lo, t_hi = max(0.0, (s_lo - self.s[i]) / seg), min(1.0, (s_hi - self.s[i]) / seg)
            if t_lo <= t_hi:
                dx, dz = b[0] - a[0], b[1] - a[1]
                t = min(max(((x - a[0]) * dx + (z - a[1]) * dz) / (seg * seg), t_lo), t_hi)
                px, pz = a[0] + t * dx, a[1] + t * dz
                d = math.hypot(x - px, z - pz)
                if best is None or d < best[1]:
                    best = (self.s[i] + t * seg, d, ((x - px) * dz - (z - pz) * dx) / seg, math.atan2(dx, dz))
            i += 1
        return best

    def corridor(self, s):
        piece = 0
        for i in range(1, len(self.corr_s)):
            if self.corr_s[i] > s:
                break
            piece = i
        return self.corr_half[piece]


class ExpertMirror:
    """DC_ExpertHL Init (speed profile), Update (ordered progress, lane errors), Evaluate (pure pursuit, speed law)."""

    def __init__(self, route):
        e = EXPERT
        self.route, self.cruise = route, e["cruise_kmh"] / 3.6
        prof = []
        for k in route.kappa:
            v = self.cruise
            if abs(k) > 0.0001:
                v = min(v, math.sqrt(e["alat"] / abs(k)))
            prof.append(v)
        prof[-1] = 0.0
        for j in range(len(prof) - 2, -1, -1):
            ds = route.s[j + 1] - route.s[j]
            prof[j] = min(prof[j], math.sqrt(prof[j + 1] ** 2 + 2.0 * e["profile_decel"] * ds))
        self.profile = prof
        self.reset()

    def reset(self):
        self.progress = self.lateral = self.heading_err = 0.0
        self.dir = 1

    def profile_at(self, s):
        r = self.route
        i = r.seg(s)
        t = min(max((s - r.s[i]) / (r.s[i + 1] - r.s[i]), 0.0), 1.0)
        return self.profile[i] + t * (self.profile[i + 1] - self.profile[i])

    def update(self, x, z, psi):
        found = self.route.project(x, z, self.progress - 30.0, self.progress + 30.0)
        if found is None:
            return
        s, _, lat, tangent = found
        self.progress = max(self.progress, s)
        self.lateral = lat
        self.heading_err = wrap(psi - tangent)

    def evaluate(self, x, z, psi, v):
        e, r = EXPERT, self.route
        if r.length - self.progress <= e["end_tol"] and v < e["stop_kmh"] / 3.6:
            self.dir = 0
        target = self.profile_at(self.progress + v * e["preview_s"])
        accel = min(max(e["speed_gain"] * (target - v), -e["brake_max"]), e["accel_max"])
        look = min(max(e["lookahead_s"] * v, e["lookahead_min"]), e["lookahead_max"])
        px, pz = r.point_at(self.progress + look)
        dx, dz = px - x, pz - z
        fwd = dx * math.sin(psi) + dz * math.cos(psi)
        right = dx * math.cos(psi) - dz * math.sin(psi)
        d2 = fwd * fwd + right * right
        kappa = min(max(2.0 * right / d2, -e["kappa_max"]), e["kappa_max"]) if d2 > 1.0 else 0.0
        if self.dir == 0:
            kappa, accel = 0.0, -e["profile_decel"]
        return Cmd(kappa, accel, self.dir)


class Car:
    """A kinematic car: curvature through a first-order lag, acceleration through a lag and the brake limit."""
    TAU_KAPPA_S, TAU_ACCEL_S, DECEL_LIMIT, ACCEL_LIMIT = 0.12, 0.25, 7.0, 3.0

    def __init__(self, x, z, psi):
        self.x, self.z, self.psi, self.v, self.kappa, self.acc = x, z, psi, 0.0, 0.0, 0.0

    def step(self, cmd, dt):
        self.kappa += (cmd.kappa - self.kappa) * (1.0 - math.exp(-dt / self.TAU_KAPPA_S))
        want = cmd.accel if cmd.dir == 1 else min(cmd.accel, 0.0) if self.v > 0 else 0.0
        if cmd.dir == 0 and self.v <= 0.05:
            want = 0.0
        self.acc += (want - self.acc) * (1.0 - math.exp(-dt / self.TAU_ACCEL_S))
        self.acc = min(max(self.acc, -self.DECEL_LIMIT), self.ACCEL_LIMIT)
        self.v = max(0.0, self.v + self.acc * dt)
        self.psi = wrap(self.psi + self.v * self.kappa * dt)
        self.x += self.v * math.sin(self.psi) * dt
        self.z += self.v * math.cos(self.psi) * dt


def simulate(table, values, routes, seconds_per_route, trial_seed=1, profile_active=True):
    """The pilot on each route for seconds_per_route: episodes (laps from standstill at the start, 3 s apart) of
    5 Hz decisions over 30 Hz steps; DC_Noise on the clean expert in mode-N episodes. Returns the mirror and the
    per-decision records."""
    records = []
    noise = NoiseMirror(table, values, DECISION_DT, EXPERT["kappa_max"], EXPERT["accel_max"], EXPERT["brake_max"],
                        EXPERT["stop_kmh"] / 3.6)
    episode_id = 0
    terminations = []
    for route_index, route in enumerate(routes):
        expert = ExpertMirror(route)
        t_route = 0.0
        while t_route < seconds_per_route:
            x0, z0 = route.xz[0]
            car = Car(x0, z0, route.heading_at(0.0))
            expert.reset()
            mode_n = profile_active and noise.begin_episode(trial_seed, episode_id)
            held = Cmd(0.0, 0.0, 1)
            t_ep, off_s, slow_s, term = 0.0, 0.0, 0.0, None
            while term is None:
                expert.update(car.x, car.z, car.psi)
                clean = expert.evaluate(car.x, car.z, car.psi, car.v)
                inp = {"speed": car.v, "remaining": route.length - expert.progress, "lateral": expert.lateral,
                       "heading_err": expert.heading_err,
                       "profile_ahead": expert.profile_at(expert.progress + car.v * EXPERT["preview_s"])}
                held = noise.decide(clean, inp) if mode_n else clean
                active = noise.segments[-1] if held.noise else None
                records.append({"route": route_index, "episode": episode_id, "mode": "N" if mode_n else "E",
                                "noise": held.noise,
                                "noise_id": held.noise_id, "kind": active["kind"] if active else None,
                                "speed": car.v, "kappa_cmd": held.kappa, "kappa_label": clean.kappa,
                                "accel_cmd": held.accel, "accel_label": clean.accel, "dir_cmd": held.dir,
                                "dir_label": clean.dir, "lateral": expert.lateral})
                for _ in range(STEPS_PER_DECISION):
                    car.step(held, STEP_DT)
                t_ep += DECISION_DT
                if expert.dir == 0 and expert.progress >= 0.98 * route.length:
                    term = "route_complete"
                elif abs(expert.lateral) > 30.0:
                    term = "route_deviation"
                off_s = off_s + DECISION_DT if abs(expert.lateral) > route.corridor(expert.progress) + \
                    EXPERT["offroad_margin"] else 0.0
                slow_s = slow_s + DECISION_DT if car.v < EXPERT["stuck_kmh"] / 3.6 else 0.0
                if off_s > EXPERT["offroad_s"]:
                    term = "offroad"
                elif slow_s >= EXPERT["stuck_s"]:
                    term = "stuck"
                elif t_ep >= EXPERT["timeout_s"]:
                    term = "timeout"
            if mode_n:
                noise.end_episode()
            terminations.append(term)
            t_route += t_ep + 3.0
            episode_id += 1
    return noise, records, terminations


class NoisePilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read(NOISE_SOURCE)
        cls.expert_source = read(EXPERT_SOURCE)
        cls.table = Table(cls.source)
        cls.defaults = dict(zip(cls.table.keys, cls.table.defaults))
        cls.routes = [RouteMirror(ROUTE_CW), RouteMirror(ROUTE_CCW)]
        cls.noise, cls.records, cls.terms = simulate(cls.table, cls.defaults, cls.routes, 1800.0)

    # -------------------------------------------------------------------------------------- the mirror is the code
    def test_rng_self_test_constants(self):
        rng = Rng()
        rng.seed(1, 0)
        got = [rng.next_int() for _ in range(3)]
        body = method_body(self.source, "DC_NoiseRng", "SelfTest")
        expected = [int(v) for v in re.search(r"array<int> expected = \{([^}]*)\}", body).group(1).split(",")]
        self.assertEqual(got, expected)
        seed = flat(method_body(self.source, "DC_NoiseRng", "Seed"))
        self.assertIn("int x = (seed << 8) ^ stream ^ SEED_MIX; if (x == 0) x = SEED_ZERO;", seed)
        self.assertIn("for (int i = 0; i < WARMUP_ROUNDS; i++) { NextInt(); }", seed)
        nxt = flat(method_body(self.source, "DC_NoiseRng", "NextInt"))
        self.assertIn("x = x ^ (x << 13); x = x ^ Shr(x, 17); x = x ^ (x << 5);", nxt)
        for name, value in (("SEED_MIX", 0x5DEECE66), ("SEED_ZERO", 0x6D2B79F5), ("WARMUP_ROUNDS", 8)):
            self.assertRegex(self.source, r"static const int %s = (0x%X|%d);" % (name, value, value))

    def test_table_is_complete(self):
        t = self.table
        self.assertEqual(len(t.keys), len(set(t.keys)))
        self.assertEqual((len(t.defaults), len(t.lo), len(t.hi)), (len(t.keys),) * 3)
        for key, d, lo, hi in zip(t.keys, t.defaults, t.lo, t.hi):
            self.assertTrue(lo <= d <= hi, key)
        for pair in t.pairs:
            self.assertLessEqual(self.defaults[pair + "_lo"], self.defaults[pair + "_hi"], pair)
        self.assertEqual(t.kinds, ["k_pulse", "k_step", "k_ou", "a_pulse", "a_ou", "speed", "stop", "brake"])
        self.assertEqual(t.weight_keys, ["w_" + k for k in t.kinds])
        # every key the code asks for exists in the table
        asked = set(re.findall(r'Get\("(\w+)"\)', code_only(self.source)))
        asked |= {k + suffix for k in re.findall(r'Lerp\("(\w+)"', code_only(self.source)) for suffix in ("_lo", "_hi")}
        self.assertTrue(asked <= set(t.keys), asked - set(t.keys))
        # the stop cap stays inside the expert's stuck rule (slower than 1 km/h for 20 s)
        self.assertLess(t.hi[t.keys.index("stop_max_s")], EXPERT["stuck_s"])

    def test_draw_order_and_formulas(self):
        draw = flat(method_body(self.source, "DC_Noise", "DrawSegment"))
        order = ["float uKind = m_Schedule.Uniform01();", "float uSign = m_Schedule.Uniform01();",
                 "float uAmplitude = m_Schedule.Uniform01();", "float uDuration = m_Schedule.Uniform01();",
                 "float uGap = m_Schedule.UniformOpen01();"]
        positions = [draw.index(s) for s in order]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(draw.count("m_Schedule."), 5)          # five draws per segment, always
        for statement in ("if (m_iKind < 0 && m_aWeights[k] > 0 && uKind * total < cumulative) m_iKind = k;",
                          "if (uSign >= 0.5) sign = -1;", "if (balance >= SIGN_BALANCE_MAX) sign = -1; else if "
                          "(balance <= -SIGN_BALANCE_MAX) sign = 1;",
                          'm_fAmplitude = sign * Lerp("k_pulse_amp", uAmplitude); durationS = Lerp("k_pulse_s", '
                          'uDuration);',
                          'm_fAmplitude = -Lerp("stop_decel", uAmplitude); durationS = Lerp("stop_dwell_s", uDuration); '
                          'm_iCapSteps = Steps(m_Config.Get("stop_max_s"));',
                          'float gapS = m_Config.Get("gap_min_s") - m_Config.Get("gap_mean_s") * Math.Log(uGap);'):
            self.assertIn(statement, draw)
        apply = flat(method_body(self.source, "DC_Noise", "ApplySegment"))
        for statement in ("applied.m_fKappa = Math.Clamp(clean.m_fKappa + kappaDelta, -m_fKappaCmdMax, m_fKappaCmdMax);",
                          "applied.m_fAccel = Math.Clamp(clean.m_fAccel + accelDelta, -decelMax, accelMax);",
                          "float target = m_fAmplitude * input.m_fProfileAheadMps;",
                          'applied.m_fAccel = Math.Clamp(m_Config.Get("speed_gain") * (target - speed), '
                          '-m_fExpertBrakeMax, m_fExpertAccelMax);',
                          "applied.m_iDir = 0; applied.m_fAccel = m_fAmplitude; return true;",
                          'if (m_iStep >= m_iCapSteps || speed < m_Config.Get("brake_end_mps")) return false;'):
            self.assertIn(statement, apply)
        # the perturbation touches the HL targets only (kappa, accel, dir), never llc or a setter
        self.assertNotRegex(code_only(self.source), r"\b(SetSteering|SetThrottle|SetBreak|Apply\(|DC_LLCOutput)\b")
        decide = flat(method_body(self.source, "DC_Noise", "Decide"))
        self.assertIn("applied.CopyFrom(clean); applied.m_bNoise = false; applied.m_iNoiseId = -1;", decide)
        self.assertIn("applied.m_bNoise = true; applied.m_iNoiseId = m_iNoiseId;", decide)
        start = flat(method_body(self.source, "DC_Noise", "CanStart"))
        self.assertIn('if (input.m_fRemainingM < m_Config.Get("end_margin_m")) return false;', start)
        self.assertIn('return m_iRecoveredSteps >= Steps(m_Config.Get("recover_hold_s"));', start)

    def test_expert_defaults(self):
        """The simulation's expert is DC_ExpertDriver with its attribute defaults."""
        for attr, value in (("m_fCruiseKmh", 40), ("m_fAlatMaxMps2", 2.75), ("m_fAccelMaxMps2", 1.5),
                            ("m_fBrakeMaxMps2", 4), ("m_fKappaMax", 0.2), ("m_fStuckS", 20), ("m_fOffroadS", 3),
                            ("m_fOffroadMarginM", 0.5), ("m_fStopKmh", 1)):
            m = re.search(r'\[Attribute\("([\d.]+)"[^\n]*\n\s*protected (?:float|int) %s;' % attr, self.expert_source)
            self.assertEqual(float(m.group(1)), value, attr)

    # -------------------------------------------------------------------------------------- the targets
    def test_off_expert_share(self):
        n = [r for r in self.records if r["mode"] == "N"]
        off = sum(1 for r in n if r["noise"]) / float(len(n))
        self.assertGreaterEqual(off, TARGET_OFF_EXPERT)
        # every perturbed decision differs from the label in at least one channel, and no clean one does
        for r in n:
            differs = (abs(r["kappa_cmd"] - r["kappa_label"]) > 1e-9 or abs(r["accel_cmd"] - r["accel_label"]) > 1e-9
                       or r["dir_cmd"] != r["dir_label"])
            if not r["noise"]:
                self.assertFalse(differs)
        self.assertGreater(sum(1 for r in n if r["noise"] and r["kind"] is not None), 0)

    def test_longitudinal_share_and_stops(self):
        segments = self.noise.segments
        longitudinal = sum(1 for s in segments if s["kind"] >= NoiseMirror.A_PULSE)
        self.assertGreaterEqual(longitudinal / float(len(segments)), TARGET_LONGITUDINAL)
        hours = sum(DECISION_DT for _ in self.records) / 3600.0
        self.assertGreaterEqual(self.noise.stood / hours, TARGET_STOPS_PER_HOUR)
        brakes = [s for s in segments if s["kind"] == NoiseMirror.BRAKE]
        self.assertGreater(len(brakes), 0)
        # every kind occurs
        self.assertTrue(all(c > 0 for c in self.noise.kind_count), self.noise.kind_count)

    def test_turn_balance(self):
        """Signed kinds: right and left within SIGN_BALANCE_MAX per kind and episode; the applied curvature's right
        and left turning (kappa x v) over the two directions within TARGET_LR_RATIO, though the clockwise loop alone
        turns mostly right."""
        sums = {}
        for seg in self.noise.segments:
            if seg["kind"] in (NoiseMirror.K_PULSE, NoiseMirror.K_STEP, NoiseMirror.A_PULSE):
                key = (seg["episode"], seg["kind"])
                sums[key] = sums.get(key, 0) + seg["sign"]
        self.assertTrue(sums)
        self.assertLessEqual(max(abs(v) for v in sums.values()), NoiseMirror.BALANCE_MAX)

        def turning(records):
            right = sum(r["kappa_cmd"] * r["speed"] for r in records if r["kappa_cmd"] > 0)
            left = -sum(r["kappa_cmd"] * r["speed"] for r in records if r["kappa_cmd"] < 0)
            return right, left

        right, left = turning(self.records)
        self.assertLessEqual(max(right, left) / min(right, left), TARGET_LR_RATIO)
        right_cw, left_cw = turning([r for r in self.records if r["route"] == 0])
        self.assertGreater(right_cw / left_cw, 1.5)

    def test_recovery_and_no_termination(self):
        n = self.noise
        judged = n.recovered_in_window + n.recovered_late + n.not_recovered
        self.assertGreater(judged, 100)
        self.assertGreaterEqual(n.recovered_in_window / float(judged), TARGET_RECOVERED)
        self.assertEqual(set(self.terms), {"route_complete"}, self.terms)
        # recovery between segments: a segment starts only after recover_hold_s inside the bounds
        for prev, nxt in zip(self.records, self.records[1:]):
            if nxt["noise"] and not prev["noise"]:
                self.assertLess(abs(prev["lateral"]), self.defaults["recover_lat_m"])

    # -------------------------------------------------------------------------------------- determinism
    def test_schedule_is_a_function_of_the_seed(self):
        def schedule(seed):
            m = NoiseMirror(self.table, self.defaults, DECISION_DT, 0.2, 1.5, 4.0, 1.0 / 3.6)
            out = []
            for episode in range(3):
                m.begin_episode(seed, episode)
                for _ in range(40):
                    m.end_segment()
                out.append(list(m.drawn))
                m.drawn = []
            return out

        self.assertEqual(schedule(7), schedule(7))
        self.assertNotEqual(schedule(7), schedule(8))
        self.assertNotEqual(schedule(7)[0], schedule(7)[1])         # episodes differ within a session
        again = simulate(self.table, self.defaults, self.routes[:1], 300.0, trial_seed=3)[1]
        self.assertEqual(again, simulate(self.table, self.defaults, self.routes[:1], 300.0, trial_seed=3)[1])

    # -------------------------------------------------------------------------------------- contract and wiring
    def test_profile_none_keeps_mode_e(self):
        """T0L: no "perturb" in the conf, no DC_Noise, every episode mode E (the mirror without a profile too)."""
        begin = flat(method_body(self.expert_source, "DC_ExpertDriver", "BeginNoise"))
        self.assertIn('if (!config || !config.IsActive()) { ctx.Log("no perturbation profile: every episode is mode E"); '
                      'return true; }', begin)
        # the decision period is the world-time grid's (200 ms at 5 Hz, the mirror's DECISION_DT)
        self.assertIn("float decisionDtS = ctx.GetDecisionPeriodMs() / 1000.0;", begin)
        self.assertAlmostEqual(DECISION_DT, 0.2)
        self.assertIn("if (!DC_NoiseRng.SelfTest(failure)) return false;", begin)
        episode = flat(method_body(self.expert_source, "DC_ExpertDriver", "BeginEpisode"))
        self.assertIn("m_bEpisodeN = false; if (m_Noise) m_bEpisodeN = m_Noise.BeginEpisode(m_Context.GetTrialSeed(), "
                      "episodeId);", episode)
        self.assertIn('info.m_sMode = "E"; if (m_bEpisodeN) info.m_sMode = DC_StreamRecorder.MODE_NOISE;', episode)
        profile, _, _ = read_config(self.table, '{"trial_id": "T0L", "duration_s": 600.0, "warmup_s": 20.0, '
                                                '"seed": 1, "route": "arland_t0_loop", "images": true}')
        self.assertEqual(profile, "none")
        _, records, _ = simulate(self.table, self.defaults, self.routes[:1], 300.0, profile_active=False)
        self.assertTrue(all(r["mode"] == "E" and not r["noise"] for r in records))
        self.assertTrue(all(r["kappa_cmd"] == r["kappa_label"] and r["accel_cmd"] == r["accel_label"]
                            for r in records))

    def test_label_stays_clean(self):
        """exp_*_now is the clean expert (m_Now); only m_Decision (action_cmd) carries the disturbance."""
        post = flat(method_body(self.expert_source, "DC_ExpertDriver", "PostStep"))
        self.assertIn("m_Expert.Evaluate(position, headingRad, speed, m_Now);", post)
        self.assertIn("m_Decision.CopyFrom(m_Now); if (m_bEpisodeN) PerturbDecision(speed);", post)
        self.assertIn("post.m_fExpKappa = m_Now.m_fKappa; post.m_fExpAccel = m_Now.m_fAccel; post.m_iExpDir = "
                      "m_Now.m_iDir;", post)
        perturb = flat(method_body(self.expert_source, "DC_ExpertDriver", "PerturbDecision"))
        self.assertIn("m_Noise.Decide(m_Now, m_NoiseInput, m_Decision);", perturb)
        decide = flat(method_body(self.expert_source, "DC_ExpertDriver", "Decide"))
        self.assertIn("command.CopyFrom(m_Decision); NoteAppliedNoise(m_Decision.m_iNoiseId, m_sDecisionPayload, "
                      "m_fDecisionPlannedS);", decide)

    def test_events_and_core(self):
        note = flat(method_body(self.expert_source, "DC_ExpertDriver", "NoteAppliedNoise"))
        self.assertIn('m_Context.LogEvent("perturb_end", m_sAppliedPayload + ";duration_s=" + realisedS.ToString(-1, 2))',
                      note)
        self.assertIn('m_Context.LogEvent("perturb_start", payload + ";duration_s=" + plannedS.ToString(-1, 2))', note)
        payload = flat(method_body(self.source, "DC_Noise", "PayloadBase"))
        self.assertIn('return "noise_id=" + m_iNoiseId.ToString() + ";kind=" + KindName(m_iKind) + ";amplitude=" + '
                      'm_fAmplitude.ToString(-1, 6);', payload)
        end = flat(method_body(self.expert_source, "DC_ExpertDriver", "EndEpisode"))
        self.assertIn("NoteAppliedNoise(-1, string.Empty, 0);", end)
        recorder = read(RECORDER_SOURCE)
        command = flat(class_text(recorder, "DC_HLCommand"))
        self.assertIn("m_bNoise = other.m_bNoise; m_iNoiseId = other.m_iNoiseId;", command)
        row = flat(method_body(recorder, "DC_StreamRecorder", "WriteCtlRow"))
        self.assertIn("if (m_bHasHeld) { noiseActive = m_Held.m_bNoise; noiseId = m_Held.m_iNoiseId; } "
                      "m_Row.AddBool(noiseActive); m_Row.AddInt(noiseId);", row)
        decide = flat(method_body(recorder, "DC_StreamRecorder", "OnStepDecide"))
        self.assertIn("bool noiseModeOk = !m_Pending.m_bNoise || (m_EpisodeInfo && m_EpisodeInfo.m_sMode == MODE_NOISE);",
                      decide)
        self.assertIn('static const string MODE_NOISE = "N";', recorder)
        context = flat(method_body(recorder, "DC_DriverContext", "LogEvent"))
        self.assertIn("if (!m_Session || (!m_Session.IsCapturing() && m_Session.GetEpisodeId() < 0)) return false;",
                      context)
        modes = re.search(r"s_aEpisodeModes = \{([^}]*)\}", read(EGO_SOURCE)).group(1)
        self.assertEqual(re.findall(r'"(\w)"', modes), ["E", "N", "D", "H", "A"])

    def test_trial_conf(self):
        session = read(SESSION_SOURCE)
        resolve = flat(method_body(session, "DC_Session", "ResolveTrial"))
        self.assertIn("if (!m_NoiseConfig.Read(context, m_sTrialConfRaw, noiseFailure)) { failure = \"trial conf \" + "
                      "confPath + \": \" + noiseFailure; return false; }", resolve)
        self.assertLess(resolve.index("ReadTrialConfText(confPath);"), resolve.index("m_NoiseConfig.Read("))
        self.assertIn("m_NoiseConfig.WriteConf(json);", session)
        base = '{"trial_id": "T0P", "duration_s": 600, "warmup_s": 20, "seed": 101, "route": "arland_t0_loop"'
        self.assertEqual(read_config(self.table, base + ', "perturb": "pilot_v1"}')[0], "pilot_v1")
        prof, values, given = read_config(self.table, base + ', "perturb": "pilot_v1", "perturb_params": '
                                                             '{"gap_mean_s": 4, "w_stop": 0.2}}')
        self.assertEqual((prof, values["gap_mean_s"], given), ("pilot_v1", 4.0, {"gap_mean_s": 4.0, "w_stop": 0.2}))
        for bad in (', "perturb": "pilot_v2"}', ', "perturb_params": {"gap_mean_s": 4}}',
                    ', "perturb": "none", "perturb_params": {}}',
                    ', "perturb": "pilot_v1", "perturb_params": {"gap_mean": 4}}',
                    ', "perturb": "pilot_v1", "perturb_params": {"gap_mean_s": 400}}',
                    ', "perturb": "pilot_v1", "perturb_params": {"speed_f_lo": 1.2, "speed_f_hi": 1.0}}',
                    ', "perturb": "pilot_v1", "perturb_param": {"gap_mean_s": 4}}',
                    ', "perturb": "pilot_v1", "perturb_params": {"stop_dwell_s_hi": 15, "stop_max_s": 10}}',
                    ', "perturb": "pilot_v1", "perturb_params": {"w_k_pulse": 0, "w_k_step": 0, "w_k_ou": 0, '
                    '"w_a_pulse": 0, "w_a_ou": 0, "w_speed": 0, "w_stop": 0, "w_brake": 0}}'):
            with self.assertRaises(ConfigError, msg=bad):
                read_config(self.table, base + bad)
        read_fn = flat(method_body(self.source, "DC_NoiseConfig", "Read"))
        for statement in ("if (!CheckKeyNames(rawText, failure)) return false;",
                          "if (profile != PROFILE_NONE && profile != PROFILE_PILOT_V1)",
                          "if (!IsActive()) { failure = \"\\\"\" + KEY_PARAMS + \"\\\" without a perturbation profile\";",
                          "if (!ScanParams(rawText, failure)) return false; return CheckValues(failure);"):
            self.assertIn(statement, read_fn)
        scan = flat(method_body(self.source, "DC_NoiseConfig", "ScanParams"))
        self.assertIn("if (keyIndex < 0 || m_aGivenKeys.Find(key) >= 0)", scan)

    def test_deployed(self):
        with open(MANIFEST, "r", encoding="ascii") as f:
            entries = [line.split() for line in f if line.strip() and not line.startswith("#")]
        self.assertIn(["script", "src/enfusion/datacapture/DC_Noise.c"], entries)
        self.assertIn(["config", "addons/DataCapture/Configs/routes/arland_t0_loop_rev.json"], entries)
        # no duplicate class name across the deployed scripts
        names = []
        for kind, path in entries:
            if kind == "script":
                names += re.findall(r"(?m)^(?:modded )?class (\w+)", read(os.path.join(REPO, path)))
        self.assertEqual(len(names), len(set(names)), sorted(n for n in names if names.count(n) > 1))
        noise = code_only(self.source)
        self.assertNotRegex(noise, r"\?[^\n:]*:")                 # no ternary operator in Enforce
        self.assertNotIn("ScriptComponent", noise)


if __name__ == "__main__":
    unittest.main()
