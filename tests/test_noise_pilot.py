#!/usr/bin/env python3
"""DD-7-ident's perturbation pilot (src/enfusion/datacapture/DC_Noise.c and its wiring) against dc-2.0 and the targets.

    TMPDIR=/dev/shm/<x> python3 -I -B -m pytest --noconftest -p no:cacheprovider -q tests/test_noise_pilot.py

The Enforce code cannot run in the engine here, so these tests run it in Python: tests/enforce_subset.py translates
DC_NoiseRng and DC_Noise of the source statement for statement (32-bit ints, C division, Math.Round half away from
zero) and the simulation calls the translated Init, BeginEpisode, Decide, DrawSegment and EndEpisode. A change of
the scheduler or the recovery gate changes what the tests see (removing UpdateRecovery, breaking the sign balance,
the gap count-down, the hold or the kind draw each fails them); a statement outside the subset fails the translation.
  - DC_NoiseRng (translated, and an independent port) gives the SelfTest constants of the source;
  - over a simulated hour of the pilot (the T0 loop both ways round, DC_ExpertHL's pure pursuit and speed profile,
    5 Hz decisions on a 200 ms world-time grid over 30 Hz steps, pilot_v1 defaults) DC_Noise meets the research
    targets (delta 2026-11 section 3.3): at least 30% of decision steps off-expert, at least 40% of the segments
    longitudinal, at least 10 stops (stand still, then drive off) per hour, the turns of each signed kind balanced
    within 2 per episode, the applied curvature's left and right turning within 1.2x over the two directions, at
    least 90% of the segments recovered within 10 s, no termination. The car is kinematic with lags: it bypasses
    llc_v1 and the S105, so this checks the scheduler and the gate, not the vehicle;
  - the recovery gate holds 2 s of world time from the first decision inside the bounds (the 11th decision at
    200 ms, not the 10th) and restarts on a decision outside;
  - one seed gives one schedule (kinds, signs, amplitudes, durations, gaps), another seed another;
  - the contract: the perturbation acts on the HL command (action_cmd) only, exp_*_now stays the clean expert
    (action_label), noise_active / noise_id come from the held command, a disturbance only in a mode-N episode, the
    perturb_start / perturb_end payload is noise_id, kind, amplitude, duration_s, the core records mode N, and a trial
    conf without "perturb" (T0L) keeps every episode mode E;
  - the trial conf (a Python port of DC_NoiseConfig.Read with its statements asserted): an unknown profile,
    perturb_params without one, an unknown or repeated parameter, a value out of range, a lo above its hi and a
    misspelt perturb key with a profile refuse the session; string values are never keys (a T0L conf with "note":
    "perturbation deferred" is read as before), a misspelt key without a profile is only warned about; every
    accepted override and an empty perturb_params reach session.json trial.conf exactly as written.
"""

import importlib.util
import json
import math
import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_SPEC = importlib.util.spec_from_file_location("enforce_subset",
                                               os.path.join(REPO, "tests", "enforce_subset.py"))
enforce_subset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(enforce_subset)

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


def check_key_names(text):
    """DC_NoiseConfig.CheckKeyNames: None, or the first key (a string followed by ':' outside strings) that starts
    with "perturb" and is neither "perturb" nor "perturb_params"."""
    index, length = 0, len(text)
    while index < length:
        if text[index] != '"':
            index += 1
            continue
        close = index + 1
        while close < length and text[close] != '"':
            if text[close] == "\\":
                close += 1
            close += 1
        if close >= length:
            return None
        after = close + 1
        while after < length and text[after] in " \t\r\n":
            after += 1
        key_length = close - index - 1
        if after < length and text[after] == ":" and key_length >= len("perturb"):
            head = text[index + 1:index + 1 + min(key_length, 64)]
            known = (key_length == 7 and head == "perturb") or (key_length == 14 and head == "perturb_params")
            if head.startswith("perturb") and not known:
                return head
        index = close + 1
    return None


def read_config(table, conf_text):
    """Mirror of DC_NoiseConfig.Read on a trial conf's text: (profile, {key: value}, given, warning) or ConfigError;
    given is {key: the number text as written} (what WriteConf records), None without perturb_params."""
    conf = json.loads(conf_text)
    profile = "none"
    if "perturb" in conf:
        if not isinstance(conf["perturb"], str):
            raise ConfigError("perturb is not a string")
        if conf["perturb"] not in ("none", "pilot_v1"):
            raise ConfigError("unknown profile")
        profile = conf["perturb"]
    values = dict(zip(table.keys, table.defaults))
    has_params = "perturb_params" in conf
    misspelt = check_key_names(conf_text)
    warning = None
    if misspelt is not None:
        if "perturb" in conf or has_params:
            raise ConfigError("misspelt perturb key %s" % misspelt)
        warning = misspelt
    if not has_params:
        return profile, values, None, warning
    if profile == "none":
        raise ConfigError("perturb_params without a profile")
    if conf_text.count('"perturb_params"') != 1 or not isinstance(conf["perturb_params"], dict):
        raise ConfigError("perturb_params is repeated or not an object")
    body = conf_text[conf_text.index('"perturb_params"') + 16:].split(":", 1)[1].split("}", 1)[0]
    pairs = re.findall(r'"(\w+)"\s*:\s*([^,\s}]+)', body)
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ConfigError("repeated key")
    given = {}
    for key, text in pairs:
        if key not in values:
            raise ConfigError("unknown key %s" % key)
        if not re.fullmatch(r"[-+.eE0-9]{1,32}", text):
            raise ConfigError("not a number of at most 32 characters")
        values[key] = float(text)
        given[key] = text
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
    return profile, values, given, warning


def written_conf(profile_given, profile, given):
    """DC_NoiseConfig.WriteConf through DC_JsonWriter: the trial.conf members it adds, as JSON text."""
    lines = []
    if profile_given:
        lines.append('"perturb": "%s"' % profile)
    if given is not None:
        lines.append('"perturb_params": {%s}' % ", ".join('"%s": %s' % (k, t) for k, t in given.items()))
    return "{" + ", ".join(lines) + "}"


class Cmd:
    """DC_HLCommand (DC_StreamRecorder.c): the fields DC_Noise reads and writes."""

    def __init__(self, kappa=0.0, accel=0.0, direction=1):
        self.m_fKappa, self.m_fAccel, self.m_iDir = kappa, accel, direction
        self.m_bNoise, self.m_iNoiseId = False, -1

    def CopyFrom(self, other):
        self.m_fKappa, self.m_fAccel, self.m_iDir = other.m_fKappa, other.m_fAccel, other.m_iDir
        self.m_bNoise, self.m_iNoiseId = other.m_bNoise, other.m_iNoiseId


class NoiseInput:
    """DC_NoiseInput."""

    def __init__(self, speed, remaining, lateral, heading_err, profile_ahead, world_ms):
        self.m_fSpeedMps, self.m_fRemainingM, self.m_fLateralM = speed, remaining, lateral
        self.m_fHeadingErrRad, self.m_fProfileAheadMps, self.m_iWorldMs = heading_err, profile_ahead, world_ms


class ConfigStub:
    """DC_NoiseConfig as DC_Noise uses it: Get(key) of the resolved parameters."""

    def __init__(self, values):
        self.values = dict(values)

    def Get(self, key):
        return self.values[key]


def enforce_noise_classes(source):
    """DC_NoiseRng and DC_Noise of the source, translated statement for statement (tests/enforce_subset.py)."""
    return enforce_subset.translate(source, ["DC_NoiseRng", "DC_Noise"],
                                    externals={"DC_NoiseConfig": ConfigStub, "DC_HLCommand": Cmd,
                                               "DC_NoiseInput": NoiseInput})


class Pilot:
    """DC_Noise itself (translated from the Enforce source) bound as DC_ExpertDriver binds it: Init with the
    expert's limits and the 200 ms grid; per segment the kind, sign, amplitude and episode, read from its fields
    when a new noise id is first applied."""
    K_PULSE, K_STEP, K_OU, A_PULSE, A_OU, SPEED, STOP, BRAKE = range(8)

    def __init__(self, source, values):
        self.classes = enforce_noise_classes(source)
        self.noise = self.classes["DC_Noise"]()
        self.noise.Init(ConfigStub(values), DECISION_DT, EXPERT["kappa_max"], EXPERT["accel_max"],
                        EXPERT["brake_max"], EXPERT["stop_kmh"] / 3.6)
        self.segments = []
        self.episode = None
        self.last_id = -1

    def begin_episode(self, trial_seed, episode_id):
        self.episode = episode_id
        self.last_id = -1
        return self.noise.BeginEpisode(trial_seed, episode_id)

    def decide(self, clean, inp):
        applied = Cmd()
        self.noise.Decide(clean, inp, applied)
        if applied.m_iNoiseId >= 0 and applied.m_iNoiseId != self.last_id:
            n = self.noise
            self.segments.append({"kind": n.m_iKind, "id": applied.m_iNoiseId, "sign": int(n.m_fSign),
                                  "episode": self.episode, "amplitude": n.m_fAmplitude})
        self.last_id = applied.m_iNoiseId
        return applied

    def end_episode(self):
        return self.noise.EndEpisode()


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
        self.kappa += (cmd.m_fKappa - self.kappa) * (1.0 - math.exp(-dt / self.TAU_KAPPA_S))
        want = cmd.m_fAccel if cmd.m_iDir == 1 else min(cmd.m_fAccel, 0.0) if self.v > 0 else 0.0
        if cmd.m_iDir == 0 and self.v <= 0.05:
            want = 0.0
        self.acc += (want - self.acc) * (1.0 - math.exp(-dt / self.TAU_ACCEL_S))
        self.acc = min(max(self.acc, -self.DECEL_LIMIT), self.ACCEL_LIMIT)
        self.v = max(0.0, self.v + self.acc * dt)
        self.psi = wrap(self.psi + self.v * self.kappa * dt)
        self.x += self.v * math.sin(self.psi) * dt
        self.z += self.v * math.cos(self.psi) * dt


def simulate(source, values, routes, seconds_per_route, trial_seed=1, profile_active=True):
    """The pilot on each route for seconds_per_route: episodes (laps from standstill at the start, 3 s apart) of
    5 Hz decisions (world time +200 ms each) over 30 Hz steps; DC_Noise (translated from source) on the clean expert
    in mode-N episodes. The car is kinematic with lags: it bypasses llc_v1 and the S105, so the recovery and
    termination results check the scheduler and the gate, not the vehicle. Returns (Pilot, records, terminations)."""
    records = []
    pilot = Pilot(source, values)
    episode_id = 0
    world_ms = 0
    terminations = []
    for route_index, route in enumerate(routes):
        expert = ExpertMirror(route)
        t_route = 0.0
        while t_route < seconds_per_route:
            x0, z0 = route.xz[0]
            car = Car(x0, z0, route.heading_at(0.0))
            expert.reset()
            mode_n = profile_active and pilot.begin_episode(trial_seed, episode_id)
            t_ep, off_s, slow_s, term = 0.0, 0.0, 0.0, None
            while term is None:
                expert.update(car.x, car.z, car.psi)
                clean = expert.evaluate(car.x, car.z, car.psi, car.v)
                inp = NoiseInput(car.v, route.length - expert.progress, expert.lateral, expert.heading_err,
                                 expert.profile_at(expert.progress + car.v * EXPERT["preview_s"]), world_ms)
                held = pilot.decide(clean, inp) if mode_n else clean
                active = pilot.segments[-1] if held.m_bNoise else None
                records.append({"route": route_index, "episode": episode_id, "mode": "N" if mode_n else "E",
                                "noise": held.m_bNoise, "noise_id": held.m_iNoiseId,
                                "kind": active["kind"] if active else None, "speed": car.v,
                                "kappa_cmd": held.m_fKappa, "kappa_label": clean.m_fKappa,
                                "accel_cmd": held.m_fAccel, "accel_label": clean.m_fAccel,
                                "dir_cmd": held.m_iDir, "dir_label": clean.m_iDir, "lateral": expert.lateral,
                                "world_ms": world_ms})
                for _ in range(STEPS_PER_DECISION):
                    car.step(held, STEP_DT)
                t_ep += DECISION_DT
                world_ms += int(round(DECISION_DT * 1000))
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
                pilot.end_episode()
            terminations.append(term)
            t_route += t_ep + 3.0
            world_ms += 3000
            episode_id += 1
    return pilot, records, terminations


class NoisePilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read(NOISE_SOURCE)
        cls.expert_source = read(EXPERT_SOURCE)
        cls.table = Table(cls.source)
        cls.defaults = dict(zip(cls.table.keys, cls.table.defaults))
        cls.routes = [RouteMirror(ROUTE_CW), RouteMirror(ROUTE_CCW)]
        cls.pilot, cls.records, cls.terms = simulate(cls.source, cls.defaults, cls.routes, 1800.0)
        cls.noise = cls.pilot.noise

    # -------------------------------------------------------------------------------------- the code runs here
    def test_rng_self_test_constants(self):
        """DC_NoiseRng translated from the source gives its own SelfTest constants, and so does an independent
        port on emulated Enforce ints."""
        body = method_body(self.source, "DC_NoiseRng", "SelfTest")
        expected = [int(v) for v in re.search(r"array<int> expected = \{([^}]*)\}", body).group(1).split(",")]
        enforce = enforce_noise_classes(self.source)["DC_NoiseRng"]()
        enforce.Seed(1, 0)
        self.assertEqual([enforce.NextInt() for _ in range(3)], expected)
        rng = Rng()
        rng.seed(1, 0)
        self.assertEqual([rng.next_int() for _ in range(3)], expected)
        for seed, stream in ((0, 0), (7, 3), (1000003, 255), (-5, 1)):
            a, b = enforce_noise_classes(self.source)["DC_NoiseRng"](), Rng()
            a.Seed(seed, stream)
            b.seed(seed, stream)
            self.assertEqual([a.NextInt() for _ in range(50)], [b.next_int() for _ in range(50)], (seed, stream))
            a.Seed(seed, stream)
            u = [a.Uniform01() for _ in range(2000)]
            self.assertTrue(all(0.0 <= x < 1.0 for x in u))
            self.assertAlmostEqual(sum(u) / len(u), 0.5, delta=0.03)

    def test_the_translator_refuses_what_it_does_not_know(self):
        """A statement outside the translated subset fails the translation (it is never skipped)."""
        broken = self.source.replace("x = x ^ (x << 13);", "x = x ^ (x << 13) ? 1 : 0;")
        with self.assertRaises(enforce_subset.TranslateError):
            enforce_noise_classes(broken)

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
        asked = set(re.findall(r'Get\("(\w+)"\)', code_only(self.source)))
        asked |= {k + suffix for k in re.findall(r'Lerp\("(\w+)"', code_only(self.source)) for suffix in ("_lo", "_hi")}
        self.assertTrue(asked <= set(t.keys), asked - set(t.keys))
        self.assertLess(t.hi[t.keys.index("stop_max_s")], EXPERT["stuck_s"])      # under the stuck rule (20 s)
        self.assertNotRegex(code_only(self.source), r"\b(SetSteering|SetThrottle|SetBreak|Apply\(|DC_LLCOutput)\b")

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
        self.assertLessEqual(off, 0.6)
        for r in n:          # a clean decision is the expert's; a perturbed one differs from the label somewhere
            differs = (abs(r["kappa_cmd"] - r["kappa_label"]) > 1e-12 or abs(r["accel_cmd"] - r["accel_label"]) > 1e-12
                       or r["dir_cmd"] != r["dir_label"])
            if not r["noise"]:
                self.assertFalse(differs)
        # DC_Noise's own counters agree with the applied commands
        self.assertEqual(self.noise.m_iPerturbedDecisions, sum(1 for r in n if r["noise"]))
        self.assertEqual(self.noise.m_iDecisions, len(n))

    def test_longitudinal_share_and_stops(self):
        segments = self.pilot.segments
        longitudinal = sum(1 for seg in segments if seg["kind"] >= Pilot.A_PULSE)
        self.assertGreaterEqual(longitudinal / float(len(segments)), TARGET_LONGITUDINAL)
        hours = len(self.records) * DECISION_DT / 3600.0
        self.assertGreaterEqual(self.noise.m_iStops / hours, TARGET_STOPS_PER_HOUR)
        self.assertEqual(sorted(set(seg["kind"] for seg in segments)), list(range(8)))
        self.assertEqual(list(self.noise.m_aKindCount), [sum(1 for seg in segments if seg["kind"] == k)
                                                         for k in range(8)])
        # a stop stands, then the expert drives off; a full brake asks for -brake_mps2
        stops = [r for r in self.records if r["noise"] and r["kind"] == Pilot.STOP]
        self.assertTrue(stops and all(r["dir_cmd"] == 0 for r in stops))
        self.assertTrue(any(r["speed"] < 0.1 for r in stops))
        brakes = [r for r in self.records if r["noise"] and r["kind"] == Pilot.BRAKE]
        self.assertTrue(brakes and all(r["accel_cmd"] == -self.defaults["brake_mps2"] for r in brakes))

    def test_turn_balance(self):
        """Signed kinds: right and left within SIGN_BALANCE_MAX per kind and episode; the applied curvature's right
        and left turning (kappa x v) over the two directions within TARGET_LR_RATIO, though the clockwise loop alone
        turns mostly right."""
        sums = {}
        for seg in self.pilot.segments:
            if seg["kind"] in (Pilot.K_PULSE, Pilot.K_STEP, Pilot.A_PULSE):
                key = (seg["episode"], seg["kind"])
                sums[key] = sums.get(key, 0) + seg["sign"]
                self.assertEqual(seg["sign"], 1 if seg["amplitude"] > 0 else -1)
        self.assertTrue(sums)
        self.assertLessEqual(max(abs(v) for v in sums.values()), 2)
        signs = [seg["sign"] for seg in self.pilot.segments if seg["kind"] in (Pilot.K_PULSE, Pilot.K_STEP)]
        self.assertLessEqual(abs(sum(signs)), 2 * len(set(seg["episode"] for seg in self.pilot.segments)))

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
        judged = n.m_iRecoveredInWindow + n.m_iRecoveredLate + n.m_iNotRecovered
        self.assertGreater(judged, 100)
        self.assertGreaterEqual(n.m_iRecoveredInWindow / float(judged), TARGET_RECOVERED)
        self.assertEqual(set(self.terms), {"route_complete"}, self.terms)
        # every segment starts after recover_hold_s inside the bounds (world time from the first decision inside)
        hold_ms = self.defaults["recover_hold_s"] * 1000
        inside_since = None
        for r in self.records:
            if r["mode"] != "N":
                continue
            inside = abs(r["lateral"]) < self.defaults["recover_lat_m"]
            if r["noise"] and r["noise_id"] != getattr(self, "_last", -1):
                self.assertIsNotNone(inside_since, r)
                self.assertGreaterEqual(r["world_ms"] - inside_since, hold_ms)
            self._last = r["noise_id"]
            if not inside:
                inside_since = None
            elif inside_since is None:
                inside_since = r["world_ms"]

    def test_recovery_gate_holds_two_seconds(self):
        """Decision by decision on the translated DC_Noise: with the gap over, a segment starts at the first
        decision 2000 ms after the first decision inside the bounds (the 11th at 200 ms, not the 10th), and a
        decision outside the bounds restarts the hold."""
        def first_start(lateral_by_decision):
            pilot = Pilot(self.source, dict(self.defaults, start_delay_s=0.0))
            pilot.begin_episode(11, 0)
            pilot.noise.m_iGapSteps = 0
            clean = Cmd(0.0, 0.0, 1)
            for k, lateral in enumerate(lateral_by_decision):
                applied = pilot.decide(clean, NoiseInput(10.0, 1000.0, lateral, 0.0, 11.0, 200 * k))
                if applied.m_bNoise:
                    return k
            return None

        self.assertEqual(first_start([0.0] * 30), 10)
        self.assertEqual(first_start([0.9] * 3 + [0.0] * 30), 13)
        self.assertEqual(first_start([0.0] * 9 + [0.9] + [0.0] * 30), 20)
        self.assertIsNone(first_start([0.9] * 30))

    # -------------------------------------------------------------------------------------- determinism
    def test_schedule_is_a_function_of_the_seed(self):
        def schedule(seed):
            pilot = Pilot(self.source, self.defaults)
            out = []
            for episode in range(3):
                pilot.begin_episode(seed, episode)
                drawn = []
                for _ in range(40):
                    n = pilot.noise
                    drawn.append((n.m_iKind, n.m_fSign, round(n.m_fAmplitude, 9), n.m_iPlannedSteps, n.m_iGapSteps,
                                  n.m_iCapSteps))
                    n.DrawSegment()
                out.append(drawn)
            return out

        self.assertEqual(schedule(7), schedule(7))
        self.assertNotEqual(schedule(7), schedule(8))
        self.assertNotEqual(schedule(7)[0], schedule(7)[1])         # episodes differ within a session
        again = simulate(self.source, self.defaults, self.routes[:1], 300.0, trial_seed=3)[1]
        self.assertEqual(again, simulate(self.source, self.defaults, self.routes[:1], 300.0, trial_seed=3)[1])

    # -------------------------------------------------------------------------------------- contract and wiring
    def test_profile_none_keeps_mode_e(self):
        """T0L: no "perturb" in the conf, no DC_Noise, every episode mode E."""
        begin = flat(method_body(self.expert_source, "DC_ExpertDriver", "BeginNoise"))
        self.assertIn('if (!config || !config.IsActive()) { ctx.Log("no perturbation profile: every episode is mode E"); '
                      'return true; }', begin)
        self.assertIn("float decisionDtS = ctx.GetDecisionPeriodMs() / 1000.0;", begin)    # the 200 ms grid
        self.assertAlmostEqual(DECISION_DT, 0.2)
        self.assertIn("if (!DC_NoiseRng.SelfTest(failure)) return false;", begin)
        episode = flat(method_body(self.expert_source, "DC_ExpertDriver", "BeginEpisode"))
        self.assertIn("m_bEpisodeN = false; if (m_Noise) m_bEpisodeN = m_Noise.BeginEpisode(m_Context.GetTrialSeed(), "
                      "episodeId);", episode)
        self.assertIn('info.m_sMode = "E"; if (m_bEpisodeN) info.m_sMode = DC_StreamRecorder.MODE_NOISE;', episode)
        perturb = flat(method_body(self.expert_source, "DC_ExpertDriver", "PerturbDecision"))
        self.assertIn("m_NoiseInput.m_iWorldMs = m_Context.GetWorldMs();", perturb)
        t0l = ('{"trial_id": "T0L", "duration_s": 600.0, "warmup_s": 20.0, "seed": 1, "route": "arland_t0_loop", '
               '"images": true}')
        self.assertEqual(read_config(self.table, t0l)[0], "none")
        _, records, _ = simulate(self.source, self.defaults, self.routes[:1], 300.0, profile_active=False)
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
        # the translated Decide never touches the clean command
        pilot = Pilot(self.source, dict(self.defaults, start_delay_s=0.0))
        pilot.begin_episode(5, 0)
        pilot.noise.m_iGapSteps = 0
        clean = Cmd(0.01, 0.5, 1)
        for k in range(200):
            pilot.decide(clean, NoiseInput(10.0, 1000.0, 0.0, 0.0, 11.0, 200 * k))
            self.assertEqual((clean.m_fKappa, clean.m_fAccel, clean.m_iDir, clean.m_bNoise, clean.m_iNoiseId),
                             (0.01, 0.5, 1, False, -1))

    def test_events_and_core(self):
        note = flat(method_body(self.expert_source, "DC_ExpertDriver", "NoteAppliedNoise"))
        self.assertIn('m_Context.LogEvent("perturb_end", m_sAppliedPayload + ";duration_s=" + realisedS.ToString(-1, 2))',
                      note)
        self.assertIn('m_Context.LogEvent("perturb_start", payload + ";duration_s=" + plannedS.ToString(-1, 2))', note)
        pilot = Pilot(self.source, self.defaults)
        pilot.begin_episode(3, 0)
        payload = pilot.noise.PayloadBase()
        self.assertRegex(payload, r"^noise_id=-?\d+;kind=(k_pulse|k_step|k_ou|a_pulse|a_ou|speed|stop|brake);"
                                  r"amplitude=-?\d+\.\d{6}$")
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
        prof, values, given, _ = read_config(self.table, base + ', "perturb": "pilot_v1", "perturb_params": '
                                                                '{"gap_mean_s": 4, "w_stop": 0.2}}')
        self.assertEqual((prof, values["gap_mean_s"], given), ("pilot_v1", 4.0, {"gap_mean_s": "4", "w_stop": "0.2"}))
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
        for statement in ("string keyFailure; if (!CheckKeyNames(rawText, keyFailure)) { if (m_bProfileGiven || "
                          "hasParams) { failure = keyFailure; return false; }",
                          "if (profile != PROFILE_NONE && profile != PROFILE_PILOT_V1)",
                          "if (!IsActive()) { failure = \"\\\"\" + KEY_PARAMS + \"\\\" without a perturbation profile\";",
                          "if (!ScanParams(rawText, failure)) return false; m_bParamsGiven = true; "
                          "return CheckValues(failure);"):
            self.assertIn(statement, read_fn)
        scan = flat(method_body(self.source, "DC_NoiseConfig", "ScanParams"))
        self.assertIn("if (keyIndex < 0 || m_aGivenKeys.Find(key) >= 0)", scan)

    def test_string_values_are_not_keys(self):
        """CheckKeyNames scans keys only: a T0L conf with "note": "perturbation deferred" is read as on main; a
        misspelt perturb key is refused with a profile and only warned about without one; escaped quotes inside a
        value do not end it."""
        keys = flat(method_body(self.source, "DC_NoiseConfig", "CheckKeyNames"))
        for statement in ("if (rawText.ToAscii(close) == CHAR_BACKSLASH) close++; close++;",
                          "int after = DC_JsonScan.SkipSpace(rawText, close + 1);",
                          "if (after < length && rawText.ToAscii(after) == DC_JsonScan.CHAR_COLON && keyLength >= "
                          "KEY_PROFILE.Length())"):
            self.assertIn(statement, keys)
        t0l = ('{"trial_id": "T0L", "duration_s": 600.0, "warmup_s": 20.0, "seed": 1, "route": "arland_t0_loop", '
               '"images": true, "note": "perturbation deferred"}')
        self.assertEqual(read_config(self.table, t0l)[0::3], ("none", None))
        self.assertIsNone(check_key_names(t0l))
        quoted = t0l.replace('"perturbation deferred"', '"say \\"perturb_x\\": later"')
        self.assertIsNone(check_key_names(quoted))
        typo = t0l.replace('"note"', '"perturbs"')
        self.assertEqual(read_config(self.table, typo)[0::3], ("none", "perturbs"))         # warned, mode E
        with self.assertRaises(ConfigError):
            read_config(self.table, typo.replace("}", ', "perturb": "pilot_v1"}'))

    def test_overrides_round_trip_exactly(self):
        """session.json trial.conf carries every accepted override as the conf wrote it, and an empty
        perturb_params as an empty object, so the recorded conf reads back to the same parameters."""
        write = flat(method_body(self.source, "DC_NoiseConfig", "WriteConf"))
        self.assertIn("if (!m_bParamsGiven) return; json.BeginObject(KEY_PARAMS); for (int i = 0; i < "
                      "m_aGivenKeys.Count(); i++) { json.AddNumberText(m_aGivenKeys[i], m_aGivenTexts[i]); }", write)
        self.assertNotIn("AddFloat", write)
        scan = flat(method_body(self.source, "DC_NoiseConfig", "ScanParams"))
        self.assertIn("m_aGivenTexts.Insert(text.Substring(numberStart, index - numberStart));", scan)
        self.assertIn("void AddNumberText(string key, string numberText) { AddLine(KeyText(key) + numberText); }",
                      flat(class_text(read(SESSION_SOURCE), "DC_JsonWriter")))
        base = '{"trial_id": "T0P", "duration_s": 600, "warmup_s": 20, "seed": 1, "perturb": "pilot_v1"'
        zero_weights = ", ".join('"%s": 0' % k for k in self.table.weight_keys if k != "w_k_pulse")
        for params in ('{"w_k_pulse": 0.0000004, %s}' % zero_weights, '{"gap_mean_s": 3.0000001}', "{}",
                       '{"kappa_alat_mps2": 4e0}'):
            conf = base + ', "perturb_params": %s}' % params
            profile, values, given, _ = read_config(self.table, conf)
            recorded = json.loads(written_conf(True, profile, given))
            replay = base + ', "perturb_params": %s}' % json.dumps(recorded["perturb_params"])
            self.assertEqual(read_config(self.table, replay)[1], values, params)
            self.assertEqual(recorded["perturb_params"], json.loads(params))
        # six fixed decimals (the writer before) would record 0.0000004 as 0.000000: every weight 0, refused
        self.assertEqual("%.6f" % 0.0000004, "0.000000")

    def test_deployed(self):
        with open(MANIFEST, "r", encoding="ascii") as f:
            entries = [line.split() for line in f if line.strip() and not line.startswith("#")]
        self.assertIn(["script", "src/enfusion/datacapture/DC_Noise.c"], entries)
        self.assertIn(["config", "addons/DataCapture/Configs/routes/arland_t0_loop_rev.json"], entries)
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
