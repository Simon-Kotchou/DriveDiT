#!/usr/bin/env python3
"""Synthetic dc-2.0 sessions for DD-5-min: the golden fixture, the negative fixtures and the C17 loop sessions.

    python3 -I -B tests/fixtures/capture_dc2/make_synthetic.py --out DIR [--kind golden|t0_loop|golden_10hz|client_1408]
        [--variant gap|missing_commit|reused_slot|crash|crash_unexplained|destroyed_unexplained]
        [--golden-json FILE] [--routes-dir DIR]

writes DIR/<session_id>/ and, with --golden-json, the expected reader output (golden.json). The committed golden
fixture was written by

    python3 -I -B tests/fixtures/capture_dc2/make_synthetic.py --out tests/fixtures/capture_dc2/golden \
        --golden-json tests/fixtures/capture_dc2/golden.json --routes-dir tests/fixtures/capture_dc2/routes

SYNTHETIC: no T0b session exists yet (DD-4-min has not run). The real golden cut from the first clean T0b session
replaces this one (README.md).

What a session holds, in the shape DD-4-min's writer writes it at the dc-2.0 freeze (docs/capture_contract/dc-2.0.md
section 6, decisions D-A and D-B):
  * every stream of the streams JSON in its column order, values formatted per unit (fixed point: 4 decimals for
    m, m/s, m/s2 and 6 for rad, rad/s, 1/m and unitless numbers), chunks closed at chunk_max_rows, after 10 s of
    world time or at the end, each committed by one commit.csv line; session.json, session_end.json, heartbeat.csv;
  * the harvester's sidecars: harvest.json (build_key of a dd-launch/1 session), _engine_logs/console.log (the
    sliced engine log: session markers and one "Screenshot saved to" line per image) and _launch/launch_<id>.json
    (the session's dd-launch/1 preflight record, copied from the pin's launches directory; C02 joins it);
  * frames/f_<slot:08d>.png (TBD-4): small images whose side bands carry real fiducial v2 patches
    (data/dc2/fiducial.py layout, the DD-4-probe encoder) of the captured render frame, so the reader's img -> hf
    join runs on decoded codes. The core shows an azimuth texture that moves with the heading and stripes that move
    with the distance driven, so C12 and C13b see motion.

The clock model (D-A, probe runs 3 and 4): render frames of a fixed duration pattern (about 39 FPS) whose world time is
the frame clock; the frame in which a screenshot is captured lasts STALL_MS (the capture stall). Engine fixed steps of
1/30 s run after each frame's EOnFrame, at most 3 per frame, catching up with the frame's world time; a frame with at
least one step has one callback, and ctl and telemetry get one row per callback with physics_step (the index of the
last step it covers), covered_steps (1 to 3), render_frame and world_time_ms. The held command is applied at the
callback, so the covered steps share its inputs. The vehicle is a kinematic model integrated over every frame's world
time with the inputs in effect (the logged pose follows the frame clock): curvature = KAPPA_PER_STEER x contract
steering input, yaw rate = speed x curvature, an llc_v1 stand-in (rate limiter and first-order lag over the callback's
covered steps) and a pure-pursuit expert on a drivedit_nav route. hf row f shows the state at the world time of frame
f - 1; the image captured in frame C (issued in its EOnFrame, done_render_frame C + 1) shows the state of hf row C + 1.

Decisions (R-01 on the world-time grid): grid point g is due at anchor + g x 1000 / decision_hz ms, anchor = world
time of the first callback; the decision callback is the first callback at or past it; the expert decides after the
rows of that callback, the command applies from the next callback (hl_seq increments there). Images use the same
anchor at image_hz (5 Hz by default), issued in EOnFrame of the first frame at or past the due time once the anchor is
known, one request in flight; each img row is written when its slot resolves (done_render_frame = the next frame).

golden.json is computed here from the simulation's own values, not by the reader, so the reader test compares two
independent computations. The R-01 samples come from the simulation's decision events: action_cmd_t and hl_seq are
the command the simulated expert decided at C_t and its decision index, action_label_t the expert evaluation after
C_t, the decision interval is held when the simulation applied that same command in every callback of it, and
action_exec is read through the simulation's own step-to-callback map; none of them is read back from the ctl rows.
The masks and the obs rule repeat the contract's rules (section 6).

Kind golden_10hz is the golden session at the non-default 10 Hz setting (image_hz = decision_hz = 10, N = 3,
clocks.image_hz_setting trial_conf, the trial conf naming image_hz 10); the tests generate it, it is not committed.

Kind client_1408 (DD-6-min) is the golden drive captured by a packaged client instead of Workbench: exe cl, instance
i01, source class client_1408x720, 1408x720 images (a 1280x720 core between two 64 px bands), a dd-launch/1 record
of exe client. The harvester tests generate it; it is not committed.

Variants (negative fixtures; each breaks one check):
  gap             two consecutive ctl rows inside an episode-0 decision interval are missing (C06); telemetry, commit
                  lines and counts stay consistent
  missing_commit  the commit line of the last ctl chunk is missing while session_end still counts it (C34)
  reused_slot     from 1 s into episode 1 the ego entity id changes while ego_slot_id stays (C35)
  crash_unexplained      the crash session below without the ego collision and veh_damage rows of episode 1: its
                         termination collision has no terminal ego event (C36)
  destroyed_unexplained  the crash session below without its veh_damage state=destroyed row: ego_destroyed is
                         unexplained (C36)
and one positive variant, crash (contract 9.5.1), in the shape DD-4-min writes it: in episode 1 the ego stops dead
against a rock (its speed drops to 0 in one frame) and is destroyed at the first callback, two or more after the
detector's onset, of a frame that issued an image request: the seal (no drain) leaves that request pending, its file
is still written and its pose row never exists (contract section 6). The
crash watch's rows at that terminal callback L: the collision (onset_step = the onset callback, dv_mps and v_mps from
the reference detector data/dc2/crash_dv.py on the simulated velocities, src dv_contact, other rock), veh_damage
state=destroyed and veh_fire_start; then the seal closes the open episode at L with termination collision and the
session ends with ego_destroyed, status ok. Nothing is written after L.

telemetry.dv_max_mps is the reference detector's delta-v (contract 9.5.1: 100 ms of world time, interpolated, gap
rule) of the row's callback, empty while it was not evaluated.

Writes only under /dev/shm/ or on the data drive of the local config (never the system drive or /tmp). Stdlib, numpy and drivedit_nav; PNGs are written by
a small zlib PNG writer, so no cv2 is needed here.
"""

import argparse
import calendar
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import drivedit_nav  # noqa: E402

STREAMS_JSON = os.path.join(REPO, "docs", "capture_contract", "dc-2.0.streams.json")
CARD_FILE = os.path.join(HERE, "cards", "S105_synthetic.json")
# DD-22: outputs only under /dev/shm/ or on the data drive of the local config (tools/local_config.py: the drive of
# DATA_ROOT; never SYSTEM_DRIVE or /tmp/). Without the config only /dev/shm/ is writable.
_LOCAL_CONFIG_PY = os.path.join(REPO, "tools", "local_config.py")


def write_refusal(path):
    """None when an output may go to path, else the refusal message (tools/local_config.py simple_write_refusal)."""
    import importlib.util
    import sys
    mod = sys.modules.get("dd_local_config")
    if mod is None or getattr(mod, "__file__", None) != _LOCAL_CONFIG_PY:
        if not os.path.isfile(_LOCAL_CONFIG_PY):
            real = os.path.realpath(os.path.abspath(path)) if path else ""
            return None if path and real.startswith("/dev/shm/") else (
                "refusing output %r: only /dev/shm/... here (%s, which knows the data drive, is missing)"
                % (path, _LOCAL_CONFIG_PY))
        spec = importlib.util.spec_from_file_location("dd_local_config", _LOCAL_CONFIG_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dd_local_config"] = mod
        spec.loader.exec_module(mod)
    return mod.simple_write_refusal(path)
VARIANTS = ("gap", "missing_commit", "reused_slot", "crash", "crash_unexplained", "destroyed_unexplained")
CRASH_VARIANTS = ("crash", "crash_unexplained", "destroyed_unexplained")
# session.json events.* (contract 10.1, TBD-10 defaults)
EVENT_PARAMS = {"collision_dv_mps": 4.0, "collision_window_ms": 100, "collision_window_tol_ms": 50,
                "collision_peak_ms": 200, "collision_refractory_ms": 1000, "episode_end_wait_ms": 250,
                "crash_cooldown_ms": 3000, "episode_block_max_ms": 60000, "rollover_up_y": 0.5,
                "rollover_hold_ms": 500, "state_poll_ms": 100,
                "contact_source": "vanilla_process_collision"}
KINDS = ("golden", "t0_loop", "golden_10hz", "client_1408", "t0_loop_noise")
RECORD_EXE = {"wb": "wb", "cl": "client", "cld": "client"}    # dd-launch exe of a session exe.kind

STEP_MS3 = 100                  # one engine fixed step in thirds of a ms (1/30 s, TBD-1 decided)
STEP_DT_S = 1.0 / 30.0
MAX_STEPS_PER_FRAME = 3         # probe runs 3-4: a callback covers at most 3 steps
FRAME_MS = (25, 26, 25, 27, 24)     # frame durations [ms] without a capture (about 39 FPS)
STALL_MS = 80                   # the frame in which a screenshot is captured (run 3: stall p50 80 ms)
FIRST_FRAME_MS = 5              # world time of the first frame after the session open
STEER_SIGN = 1                  # TBD-5 decided: a positive SetSteering turns right
YAW_AXIS_SIGN = 1
KAPPA_PER_STEER = 0.15          # synthetic vehicle: curvature (1/m) per unit contract steering
STEER_RATE = 2.0                # llc stand-in rate limit (1/s)
STEER_TAU_S = 0.08              # llc stand-in first-order lag
GEAR_FORWARD = 2                # S105 (TBD-7 decided): first gear
GEAR_IDLE = 1                   # S105 neutral while no command is held (dir_state 0)
RAD_PER_PX = 0.004              # synthetic camera: azimuth per core pixel column
DECIMALS = {"m": 4, "m/s": 4, "m/s2": 4, "rad": 6, "rad/s": 6, "1/m": 6, "1": 6, "deg": 4, "degC": 2, "rpm": 2,
            "1/s": 3, "h": 6, "s": 3}
FIDUCIAL_PATH = os.path.join(REPO, "data", "dc2", "fiducial.py")
CRASH_DV_PATH = os.path.join(REPO, "data", "dc2", "crash_dv.py")


def _load_fiducial():
    spec = importlib.util.spec_from_file_location("dc2_fiducial", FIDUCIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fiducial = _load_fiducial()


def _load_crash_dv():
    spec = importlib.util.spec_from_file_location("dc2_crash_dv", CRASH_DV_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


crash_dv = _load_crash_dv()


def steps_per_decision(decision_hz):
    """N = floor(1000 / decision_hz / step_dt_ms + 0.5) (contract section 6): 6 at 5 Hz, 3 at 10 Hz."""
    return int(math.floor(1000.0 / decision_hz / (1000.0 * STEP_DT_S) + 0.5))


def _hex(text, n=64):
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:n]


def check_out(path):
    why = write_refusal(path)
    if why:
        raise ValueError(why)
    return os.path.realpath(path)


# ------------------------------------------------------------------------------------------------
# Specs and routes
# ------------------------------------------------------------------------------------------------
SPECS = {
    # About 13 s: 2 s warm-up, two 5 s mode-E episodes on the S-curve with a teleport back to its start between them,
    # 0.5 s tail. Every stream gets a second chunk (10 s rule); the render frame crosses 2^18 (fiducial unwrap).
    "golden": {
        "session_id": "arland_T0b_20261025T091512Z_wb_i00_5e00", "trial_id": "T0b", "seed": 0,
        "start_utc": (2026, 10, 25, 9, 15, 12), "render_frame0": 261900, "decision_hz": 5,
        "warmup_ms": 2000, "route": "golden_s_curve_v1", "v_ref": 8.0,
        "episodes": [{"max_ms": 5000}, {"max_ms": 5000}], "between_ms": 500, "teleport_offset_ms": 250,
        "tail_ms": 500, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": True,
    },
    # The golden session at the non-default 10 Hz setting (D-B): image_hz = decision_hz = 10, N = 3, set by the
    # trial conf (generated in the tests, not committed).
    "golden_10hz": {
        "session_id": "arland_T0b_20261025T093000Z_wb_i00_5e10", "trial_id": "T0b", "seed": 0, "decision_hz": 10,
        "start_utc": (2026, 10, 25, 9, 30, 0), "render_frame0": 261900,
        "warmup_ms": 2000, "route": "golden_s_curve_v1", "v_ref": 8.0,
        "episodes": [{"max_ms": 5000}, {"max_ms": 5000}], "between_ms": 500, "teleport_offset_ms": 250,
        "tail_ms": 500, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": True,
    },
    # The golden drive in the packaged client (DD-6-min harvester tests): the client source class, 1408x720 images
    # with a 1280x720 core (generated in the tests, not committed).
    "client_1408": {
        "session_id": "arland_T0b_20261025T094500Z_cl_i01_c140", "trial_id": "T0b", "seed": 0,
        "start_utc": (2026, 10, 25, 9, 45, 0), "render_frame0": 261900, "decision_hz": 5,
        "warmup_ms": 2000, "route": "golden_s_curve_v1", "v_ref": 8.0,
        "episodes": [{"max_ms": 5000}, {"max_ms": 5000}], "between_ms": 500, "teleport_offset_ms": 250,
        "tail_ms": 500, "image_w": 1408, "image_h": 720, "band_px": 64, "fill_route_sha": True,
        "exe": "cl", "instance": "i01", "source_class": "client_1408x720",
        # the golden azimuth span over the wider core, so C13b's shift search sees one texture period
        "rad_per_px": RAD_PER_PX * 128.0 / 1280.0,
    },
    # 108 s: the 60 s C17 warm-up standing, then one mode-E lap of the synthetic loop (trial T1, so C17 is SOFT).
    "t0_loop": {
        "session_id": "arland_T1_20261026T101500Z_wb_i00_c17a", "trial_id": "T1", "seed": 1,
        "start_utc": (2026, 10, 26, 10, 15, 0), "render_frame0": 40000, "decision_hz": 5,
        "warmup_ms": 60000, "route": "t0_loop_synthetic_v1", "v_ref": 8.0,
        "episodes": [{"max_ms": 50000}], "between_ms": 0, "teleport_offset_ms": None,
        "tail_ms": 500, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": False,
    },
    # DD-7-ident's pilot shape: the t0_loop lap as a mode-N episode (trial T0P) with one disturbance of each family
    # DC_Noise draws, in the decision callbacks DC_ExpertDriver applies them: ctl hl_* the applied command,
    # exp_*_now the clean expert, noise_active / noise_id on every row holding a disturbed command, perturb_start
    # and perturb_end events (payload noise_id, kind, amplitude, duration_s). Generated in the tests, not committed.
    "t0_loop_noise": {
        "session_id": "arland_T0P_20261106T101500Z_wb_i00_a01e", "trial_id": "T0P", "seed": 101,
        "start_utc": (2026, 11, 6, 10, 15, 0), "render_frame0": 40000, "decision_hz": 5,
        "warmup_ms": 60000, "route": "t0_loop_synthetic_v1", "v_ref": 8.0,
        "episodes": [{"max_ms": 60000}], "between_ms": 0, "teleport_offset_ms": None,
        "tail_ms": 500, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": True,
        "episode_mode": "N",
        # (start, duration) in ms after the episode start, the amplitudes as DC_Noise logs them (its pilot_v1
        # semantics): k_pulse / k_step a signed fraction of kappa_max(v) = min(0.2, 4 / max(v, 3)^2) added to
        # kappa_target (the pulse a half-sine sampled at the middle of each decision interval), a_pulse m/s2 added to
        # a_target, speed the factor of the speed target (here x v_ref) for the expert's gain law, stop dir 0 with
        # a_target = amplitude
        "noise": [{"kind": "k_pulse", "start_ms": 6000, "dur_ms": 2000, "amplitude": 0.4},
                  {"kind": "a_pulse", "start_ms": 13000, "dur_ms": 2000, "amplitude": -1.5},
                  {"kind": "k_step", "start_ms": 19000, "dur_ms": 1200, "amplitude": -0.3},
                  {"kind": "stop", "start_ms": 25000, "dur_ms": 5000, "amplitude": -2.5},
                  {"kind": "speed", "start_ms": 36000, "dur_ms": 4000, "amplitude": 0.6}],
    },
}


def route_dict(route_id, xz, closed, description):
    pts = [[round(x, 4), 30.0, round(z, 4)] for x, z in xz]
    if closed:
        pts[-1] = list(pts[0])
    length = 0.0
    for i in range(1, len(pts)):
        length += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][2] - pts[i - 1][2])
    return {"schema": "drivedit.route/1", "route_id": route_id, "map_id": "arland", "split": "train",
            "nav_version": "nav_v1", "lane_side": "right", "closed": closed, "polyline": pts, "maneuvers": [],
            "stops": [{"s_m": round(length, 6), "kind": "route_end"}], "length_m": round(length, 6),
            "source": {"generator": "tests/fixtures/capture_dc2/make_synthetic.py", "synthetic": True},
            "description": description}


def golden_route():
    """S-curve near (1000, 3000), outside the VAL-L box and its buffer: 5 m north, right arc R 25 m over 50 deg,
    left arc R 25 m over 50 deg, 20 m straight."""
    xz = [(1000.0, 3000.0 + s) for s in range(0, 5)]
    x, z, psi = 1000.0, 3004.0, 0.0
    step = 0.5
    for sign in (1.0, -1.0):
        n = int(round(math.radians(50.0) * 25.0 / step))
        for _ in range(n):
            psi += sign * step / 25.0
            x += step * math.sin(psi)
            z += step * math.cos(psi)
            xz.append((x, z))
    for _ in range(40):
        x += step * math.sin(psi)
        z += step * math.cos(psi)
        xz.append((x, z))
    return route_dict("golden_s_curve_v1", xz, False, "SYNTHETIC golden S-curve (DD-5-min fixture)")


def t0_loop_route():
    """Asymmetric clockwise loop (about 330 m) standing in for the T0 loop of DD-7-min: x = cx + 60 sin t + 18 sin 2t,
    z = cz + 45 cos t, t from 0 to 2 pi; it starts at its north point heading east."""
    cx, cz = 1200.0, 3200.0
    xz = []
    n = 1400
    for i in range(n + 1):
        t = 2.0 * math.pi * i / n
        xz.append((cx + 60.0 * math.sin(t) + 18.0 * math.sin(2.0 * t), cz + 45.0 * math.cos(t)))
    return route_dict("t0_loop_synthetic_v1", xz, True, "SYNTHETIC stand-in for the T0 loop (DD-5-min C17 test)")


def mirror_route(obj, route_id=None):
    """The route reflected across the line through its first point along its first segment: same start pose, every
    turn in the other direction."""
    pts = obj["polyline"]
    x0, z0 = pts[0][0], pts[0][2]
    dx, dz = pts[1][0] - x0, pts[1][2] - z0
    norm = math.hypot(dx, dz)
    dx, dz = dx / norm, dz / norm
    xz = []
    for p in pts:
        rx, rz = p[0] - x0, p[2] - z0
        along = rx * dx + rz * dz
        xz.append((x0 + 2.0 * along * dx - rx, z0 + 2.0 * along * dz - rz))
    out = route_dict(route_id or obj["route_id"], xz, obj["closed"], "mirror of " + obj["route_id"])
    return out


ROUTES = {"golden_s_curve_v1": golden_route, "t0_loop_synthetic_v1": t0_loop_route}


# ------------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------------
def fmt(field, value):
    """CSV text of one value (contract section 8)."""
    if value is None:
        return ""
    t = field["type"]
    if t == "int":
        return "%d" % int(value)
    if t == "bool":
        return "1" if value else "0"
    if t == "float":
        text = "%.*f" % (DECIMALS.get(field["unit"], 4), float(value))
        return text[1:] if text.startswith("-") and not text.strip("-0.") else text    # no "-0.000"
    return str(value)


def quantise(field, text):
    """The value a reader gets back from text."""
    if text == "":
        return None
    t = field["type"]
    if t in ("int", "bool"):
        return int(text)
    if t == "float":
        return float(text)
    return text


def wrap(a):
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


# ------------------------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------------------------
class Sim:
    """Truth of one session: render frames, callbacks (one ctl and telemetry row each), events, episodes, decisions
    and image slots."""

    def __init__(self):
        self.frames = []        # per render frame: index, render_frame, world, lps, state (pose at its EOnFrame), cb
        self.callbacks = []     # per callback: the row values (state after its frame, inputs, command, expert)
        self.events = []        # dicts: step, episode, frame, world, kind, payload
        self.episodes = []      # dicts: episode_id, start_cb, end_cb, termination
        self.decisions = []     # per decision callback: cb, grid, cmd (the expert's command, or None)
        self.slots = []         # per image slot: slot_id, due, episode, issue_frame, done_frame
        self.dv = {}            # callback index -> reference detector dv_h (or None)
        self.crash = None
        self.end_reason = "duration"


def expert(route, x, z, psi, v, progress, v_ref):
    remaining = route.length_m - progress
    target = min(v_ref, math.sqrt(2.0 * 1.2 * max(remaining - 0.5, 0.0)))
    accel = max(-3.0, min(2.0, 0.8 * (target - v)))
    look = max(6.0, 1.2 * v)
    tx, tz = route.point_at(progress + look)
    fwd, right = drivedit_nav.to_ego(drivedit_nav.Pose(x, z, psi), tx, tz)
    d2 = fwd * fwd + right * right
    kappa = 2.0 * right / d2 if d2 > 0.01 else 0.0
    return (kappa, accel, 1)


NOISE_KAPPA_MAX = 0.2           # DC_ExpertDriver m_fKappaMax: DC_Noise's kappa_max(v) cap and the clamp of kappa
NOISE_KAPPA_ALAT_MPS2 = 4.0     # pilot_v1 kappa_alat_mps2
NOISE_MIN_SPEED_MPS = 3.0       # pilot_v1 min_speed_mps


def noise_kappa_scale(v):
    """DC_Noise.KappaScale: kappa_max(v) = min(kappa_max, kappa_alat / max(v, min_speed)^2)."""
    v = max(v, NOISE_MIN_SPEED_MPS)
    return min(NOISE_KAPPA_MAX, NOISE_KAPPA_ALAT_MPS2 / (v * v))


def perturb(spec, exp, v, t_ms, applied_counts):
    """(command, noise id, segment) at a decision t_ms after the episode start: the expert's command exp with the
    spec's disturbance of that time, as DC_Noise.ApplySegment computes it (the HL targets only; a pulse is a
    half-sine over the segment's n decisions, sampled at the middle of decision i = applied_counts[id]), else
    (exp, -1, None)."""
    period_ms = 1000 // spec["decision_hz"]
    for noise_id, seg in enumerate(spec.get("noise", ())):
        if seg["start_ms"] <= t_ms < seg["start_ms"] + seg["dur_ms"]:
            kappa, accel, direction = exp
            n = max(1, int(math.floor(seg["dur_ms"] / float(period_ms) + 0.5)))
            i = applied_counts.get(noise_id, 0)
            applied_counts[noise_id] = i + 1
            shape = math.sin(math.pi * (i + 0.5) / n)
            if seg["kind"] == "k_pulse":
                kappa = kappa + seg["amplitude"] * noise_kappa_scale(v) * shape
            elif seg["kind"] == "k_step":
                kappa = kappa + seg["amplitude"] * noise_kappa_scale(v)
            elif seg["kind"] == "a_pulse":
                accel = accel + seg["amplitude"] * shape
            elif seg["kind"] == "speed":
                accel = max(-3.0, min(2.0, seg["amplitude"] * spec["v_ref"] - v))
            elif seg["kind"] == "stop":
                accel, direction = seg["amplitude"], 0
            kappa = max(-NOISE_KAPPA_MAX, min(NOISE_KAPPA_MAX, kappa))
            return (kappa, accel, direction), noise_id, seg
    return exp, -1, None


def _event(sim, step, episode, frame, world, kind, payload=""):
    sim.events.append({"step": step, "episode": episode, "frame": frame, "world": world, "kind": kind,
                       "payload": payload})


def simulate(spec, route):
    """Frame by frame: EOnFrame (teleport, hf state, image resolve and issue), then the frame's fixed steps (one
    callback: command, llc stand-in, rows, warm-up, episodes and the decision), with the vehicle integrated over the
    frame's world time in between. Returns a Sim."""
    sim = Sim()
    decision_hz = spec["decision_hz"]
    period_d = 1000 // decision_hz
    period_i = 1000 // decision_hz             # image_hz = decision_hz (TBD-2)
    p0 = route.xz[0]
    p1 = route.point_at(1.0)
    psi0 = math.atan2(p1[0] - p0[0], p1[1] - p0[1])
    st = {"x": p0[0], "z": p0[1], "psi": psi0, "v": 0.0, "dist": 0.0, "yaw": 0.0, "acc": 0.0}
    steer_c, throttle, brake = 0.0, 0.0, 0.0
    held = pending = None
    hl_count = -1
    episode, ep_index, ep_start_cb, ep_start_world, max_ms, progress = -1, 0, None, None, None, 0.0
    phase, phase_end_ms, warmup_done, stopping = "warmup", spec["warmup_ms"], False, False
    teleport_ms = None
    crash_ms, crash_onset_cb, crash_hit = None, None, False
    llc_gear_due = False
    applied_noise, applied_since, applied_counts = -1, None, {}
    episode_mode = spec.get("episode_mode", "E")
    det = crash_dv.Detector()
    det.watch(0)
    world, prev_world, clock3, step, anchor, grid_d, grid_i = FIRST_FRAME_MS, 0, 0, -1, None, 0, 0
    inflight = None
    f = 0
    pattern = 0
    while True:
        # ---------------- EOnFrame f
        frame = {"index": f, "render_frame": spec["render_frame0"] + f, "world": world, "lps": step, "cb": None,
                 "issued": False}
        issued_slot = None
        if teleport_ms is not None and world >= teleport_ms:
            st.update({"x": p0[0], "z": p0[1], "psi": psi0, "v": 0.0, "yaw": 0.0, "acc": 0.0})
            steer_c, progress = 0.0, 0.0
            det.reposition(world)
            _event(sim, step, episode, f, world, "teleport", "reason=episode_reset")
            teleport_ms = None
        frame["state"] = dict(st)
        if inflight is not None:
            inflight["done_frame"] = f
            inflight = None
        if stopping:
            sim.frames.append(frame)
            break                                   # drained: the seal follows this frame's hf row
        if anchor is not None:
            while anchor + grid_i * period_i <= world:
                slot = {"slot_id": grid_i, "due": anchor + grid_i * period_i, "episode": episode,
                        "issue_frame": None, "done_frame": None}
                if inflight is None and not frame["issued"]:
                    slot["issue_frame"] = f
                    frame["issued"] = True
                    inflight = slot
                    issued_slot = slot
                sim.slots.append(slot)
                grid_i += 1
        sim.frames.append(frame)
        # ---------------- the frame's fixed steps: at most one callback
        covered = min(MAX_STEPS_PER_FRAME, (3 * world - clock3) // STEP_MS3)
        cb = None
        in_ep = False
        if covered >= 1:
            clock3 += STEP_MS3 * covered
            cb_index = len(sim.callbacks)
            if anchor is None:
                anchor = world
            decision = world >= anchor + grid_d * period_d
            if episode < 0:
                held = pending = None
            elif pending is not None:                       # EOnFixedFrame: the previous decision applies now
                held, pending = pending, None
                hl_count += 1
            if decision and phase == "between" and world >= phase_end_ms:
                episode = ep_index
                ep_index += 1
                ep_start_cb, ep_start_world = cb_index, world
                max_ms = spec["episodes"][episode]["max_ms"]
                phase, progress, llc_gear_due = "episode", 0.0, True
                if spec.get("crash") and episode == 1:
                    crash_ms = world + spec["crash_after_ms"]
                _event(sim, step, episode, f, world, "episode_start", "route_id=%s;mode=%s" % (route.route_id,
                                                                                              episode_mode))
            in_ep = phase == "episode"
            cmd = held if in_ep else None
            dt_cb = covered * STEP_DT_S
            llc_gear = None
            if cmd is not None:
                raw_c = max(-1.0, min(1.0, cmd[0] / KAPPA_PER_STEER))
                limited = steer_c + max(-STEER_RATE * dt_cb, min(STEER_RATE * dt_cb, raw_c - steer_c))
                steer_c = steer_c + (limited - steer_c) * (dt_cb / (STEER_TAU_S + dt_cb))
                throttle = min(1.0, cmd[1] / 3.0) if cmd[1] >= 0 else 0.0
                brake = min(1.0, -cmd[1] / 6.0) if cmd[1] < 0 else 0.0
                if llc_gear_due:
                    llc_gear, llc_gear_due = GEAR_FORWARD, False
            else:
                raw_c = None
                steer_c = steer_c if in_ep else 0.0
                throttle = brake = 0.0
            cb = {"index": cb_index, "covered": covered, "frame": f, "world": world,
                  "episode": episode if in_ep else -1, "mode": episode_mode if in_ep else "I",
                  "driver": "expert" if in_ep else "none", "hl_seq": hl_count, "cmd": cmd,
                  "raw_c": raw_c, "lim_c": steer_c if cmd is not None else None, "steer_in_c": steer_c,
                  "throttle": throttle, "brake": brake, "llc_on": cmd is not None, "llc_gear": llc_gear,
                  "gear": GEAR_FORWARD if in_ep else GEAR_IDLE,
                  "handbrake": 1 if (not in_ep and st["v"] < 0.1) else 0}
        # ---------------- the vehicle over the frame's world time (the frame clock), inputs in effect
        dt = (world - prev_world) / 1000.0
        kappa_act = KAPPA_PER_STEER * steer_c
        yaw = st["v"] * kappa_act
        acc = 3.0 * throttle - 6.0 * brake - 0.05 * st["v"]
        v_new = max(0.0, st["v"] + acc * dt)
        if crash_ms is not None and world >= crash_ms and episode == 1:
            v_new = 0.0                 # the crash: stopped dead against the rock (contract 9.5.1 crash rows)
            crash_hit = True
        acc_ach = (v_new - st["v"]) / dt if dt > 0 else 0.0
        st["psi"] = wrap(st["psi"] + yaw * dt)
        st["x"] += v_new * math.sin(st["psi"]) * dt
        st["z"] += v_new * math.cos(st["psi"]) * dt
        st["dist"] += v_new * dt
        st["v"], st["yaw"], st["acc"] = v_new, yaw, acc_ach
        if issued_slot is not None:
            issued_slot["capture_state"] = dict(st)     # frame C is rendered after its physics: hf row C + 1's state
        if cb is not None:
            # ---------------- EOnPostFixedFrame: the rows of the callback, then warm-up, end and the decision
            step += covered
            exp = None
            if in_ep:
                s_proj, _ = route.project(st["x"], st["z"], progress - 30.0, progress + 30.0)
                progress = max(progress, s_proj)
                exp = expert(route, st["x"], st["z"], st["psi"], st["v"], progress, spec["v_ref"])
            cb.update({"p": step, "exp": exp, "x": st["x"], "z": st["z"], "psi": st["psi"], "v": st["v"],
                       "dist": st["dist"], "yaw": st["yaw"], "acc": st["acc"], "progress": progress if in_ep else None})
            frame["cb"] = cb["index"]
            sim.callbacks.append(cb)
            sim.dv[cb["index"]] = det.step(world, st["v"] * math.sin(st["psi"]), st["v"] * math.cos(st["psi"]), step)
            if not warmup_done and world >= spec["warmup_ms"]:
                warmup_done = True
                _event(sim, step, -1, f, world, "warmup_end")
                phase, phase_end_ms = "between", world
            # the crash: the terminal callback L, two callbacks after the detector's onset and in the frame of an
            # image request, so the seal (no drain) catches it in flight: its row is pending, the engine still writes
            # the file, and its pose row (C + 1) never exists (contract section 6, review round 1); the destroyed ego
            # ends the session after its rows
            crash_onsets = [o for o in det.onsets if crash_ms is not None and o["t_ms"] >= crash_ms]
            if crash_hit and crash_onset_cb is None and crash_onsets:
                crash_onset_cb = [c["index"] for c in sim.callbacks if c["p"] == crash_onsets[0]["step"]][0]
            if crash_onset_cb is not None and cb["index"] >= crash_onset_cb + 2 and inflight is not None:
                _end_episode(sim, episode, ep_start_cb, cb["index"], "collision", step, f, world)
                sim.crash = {"onset_cb": crash_onset_cb, "stop_cb": cb["index"], "onset": crash_onsets[0]}
                break
            if decision:
                g = (world - anchor) // period_d
                grid_d = g + 1
                dec = {"cb": cb["index"], "grid": g, "cmd": None}
                if in_ep:
                    complete = route.length_m - progress <= 1.5 and st["v"] < 0.5
                    if complete or world - ep_start_world >= max_ms:
                        if applied_noise >= 0:          # DC_ExpertDriver.EndEpisode closes it before episode_end
                            _perturb_event(sim, spec, "perturb_end", applied_noise, (world - applied_since) / 1000.0,
                                           step, episode, f, world)
                            applied_noise = -1
                        _end_episode(sim, episode, ep_start_cb, cb["index"], "route_complete" if complete
                                     else "timeout", step, f, world)
                        episode, held, pending = -1, None, None
                        phase = "between" if ep_index < len(spec["episodes"]) else "tail"
                        phase_end_ms = world + (spec["between_ms"] if phase == "between" else spec["tail_ms"])
                        if phase == "between" and spec["teleport_offset_ms"] is not None:
                            teleport_ms = world + spec["teleport_offset_ms"]
                    else:
                        cmd_d, noise_id, seg = perturb(spec, exp, st["v"], world - ep_start_world, applied_counts)
                        if noise_id != applied_noise:   # DC_ExpertDriver.NoteAppliedNoise, in the decision callback
                            if applied_noise >= 0:
                                _perturb_event(sim, spec, "perturb_end", applied_noise,
                                               (world - applied_since) / 1000.0, step, episode, f, world)
                            if noise_id >= 0:
                                _perturb_event(sim, spec, "perturb_start", noise_id, seg["dur_ms"] / 1000.0, step,
                                               episode, f, world)
                                applied_since = world
                            applied_noise = noise_id
                        pending = cmd_d + ((noise_id >= 0, noise_id),) if spec.get("noise") else exp
                        dec["cmd"] = pending
                sim.decisions.append(dec)
            if phase == "tail" and world >= phase_end_ms:
                stopping = True                     # the planned end: no new slot; the drain ends the session
        prev_world = world
        if frame["issued"]:                 # a spec may cycle several capture stalls (the C37 tests)
            stalls = spec.get("stall_ms", (STALL_MS,))
            world += stalls[sum(1 for sl in sim.slots if sl.get("issue_frame") is not None and
                                sl["issue_frame"] < f) % len(stalls)]
        else:                               # a spec may set its own ordinary frame times (the C21 tests)
            frame_ms = spec.get("frame_ms", FRAME_MS)
            world += frame_ms[pattern % len(frame_ms)]
        if not frame["issued"]:
            pattern += 1
        f += 1
    return sim


def _perturb_event(sim, spec, kind, noise_id, duration_s, step, episode, frame, world):
    seg = spec["noise"][noise_id]
    _event(sim, step, episode, frame, world, kind, "noise_id=%d;kind=%s;amplitude=%.6f;duration_s=%.2f"
           % (noise_id, seg["kind"], seg["amplitude"], duration_s))


def _end_episode(sim, episode, start_cb, end_cb, term, step, frame, world):
    _event(sim, step, episode, frame, world, "episode_end", "termination_reason=%s" % term)
    sim.episodes.append({"episode_id": episode, "start_cb": start_cb, "end_cb": end_cb, "termination": term})


def inject_crash(sim, variant):
    """The crash rows of the crash variants (module docstring) at the terminal callback L, before the seal's
    episode_end, in DD-4-min's order. Sets sim.end_reason."""
    if sim.crash is None:
        return
    o = sim.crash["onset"]
    onset_cb = sim.callbacks[sim.crash["onset_cb"]]
    stop_cb = sim.callbacks[sim.crash["stop_cb"]]
    psi = onset_cb["psi"]
    fwd = (math.sin(psi), math.cos(psi))
    right = (math.cos(psi), -math.sin(psi))
    dlon = o["peak_vec"][0] * fwd[0] + o["peak_vec"][1] * fwd[1]
    dlat = o["peak_vec"][0] * right[0] + o["peak_vec"][1] * right[1]
    end = sim.events.pop()
    assert end["kind"] == "episode_end" and end["step"] == stop_cb["p"], end
    rows = []
    if variant != "crash_unexplained":
        rows.append(("collision", "onset_step=%d;dv_mps=%.2f;dv_lon_mps=%.2f;dv_lat_mps=%.2f;v_mps=%.2f;other=rock;"
                     "other_prefab=Prefabs/Rocks/Granite/GraniteStone_03.et;impulse_ns=15210.40;src=dv_contact"
                     % (o["step"], o["peak"], dlon, dlat, math.hypot(*o["ref"]))))
        if variant != "destroyed_unexplained":
            rows.append(("veh_damage", "state=destroyed;health=0.00;movement_damage=1.00"))
        rows.append(("veh_fire_start", "fire_state=burning"))
    for kind, payload in rows:
        _event(sim, end["step"], end["episode"], end["frame"], end["world"], kind, payload)
    sim.events.append(end)
    sim.end_reason = "stop" if variant == "crash_unexplained" else "ego_destroyed"


def gap_callbacks(sim):
    """The two ctl rows the gap variant drops: consecutive callbacks inside an episode-0 decision interval that are
    neither a decision callback nor the callback after one, nor the callback of a frame an image was captured in."""
    deciders = set(d["cb"] for d in sim.decisions)
    captured = set(sl["issue_frame"] for sl in sim.slots if sl["issue_frame"] is not None)
    ok = [c["index"] for c in sim.callbacks if c["episode"] == 0 and c["index"] not in deciders
          and c["index"] - 1 not in deciders and c["frame"] not in captured and c["index"] > 30]
    for a in ok:
        if a + 1 in ok:
            return (a, a + 1)
    raise AssertionError("no gap rows found")


# ------------------------------------------------------------------------------------------------
# Images
# ------------------------------------------------------------------------------------------------
TEX = ((0.21, 1.0, 0.3), (0.37, 0.7, 1.9), (0.53, 0.5, 4.1), (0.97, 0.6, 2.2))


def render_image(spec, step, code):
    """H x W x 3 uint8: grey side bands with the fiducial v2 patches of code; a core that moves with heading
    (azimuth texture, upper part) and with the distance driven (stripes everywhere)."""
    w, h, band = spec["image_w"], spec["image_h"], spec["band_px"]
    img = numpy.full((h, w, 3), 70, dtype=numpy.uint8)
    core_w = w - 2 * band
    u = numpy.arange(core_w, dtype=float) - core_w / 2.0 + 0.5
    az = step["psi"] + u * spec.get("rad_per_px", RAD_PER_PX)
    tex = numpy.zeros(core_w)
    for period, amp, phase in TEX:
        tex += amp * numpy.sin(2.0 * math.pi * az / period + phase)
    tex /= sum(a for _, a, _ in TEX)
    horizon = int(h * 0.45)
    y = numpy.arange(h, dtype=float)[:, None]
    dist = step["dist"]
    # integer terms that depend on x only or on y only keep every PNG row alike after the Sub filter (small files)
    sky_x = numpy.floor(125.5 + 70.0 * tex)[None, :]
    sky_y = numpy.floor(0.5 + 14.0 * numpy.sin(2.0 * math.pi * dist / 7.3 + y[:horizon] / 23.0))
    ground_y = y[horizon:] - horizon + 6.0
    road = numpy.floor(95.5 + 40.0 * numpy.sin(2.0 * math.pi * (dist + 300.0 / ground_y) / 5.0))
    core = numpy.vstack([sky_x + sky_y, road + numpy.zeros((h - horizon, core_w))])
    rgb = numpy.stack([core, core - 10.0, core - 25.0], axis=2)
    img[:, band:band + core_w, :] = numpy.clip(rgb, 0, 255).astype(numpy.uint8)
    fiducial.render(img, code)
    return img


def png_bytes(rgb):
    """PNG (8-bit RGB, Sub filter, zlib level 9) of an H x W x 3 uint8 array."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(h, w * 3).astype(numpy.int16)
    filt = numpy.empty((h, w * 3), dtype=numpy.uint8)
    filt[:, :3] = flat[:, :3]
    filt[:, 3:] = (flat[:, 3:] - flat[:, :-3]) % 256
    raw = b"".join(b"\x01" + filt[i].tobytes() for i in range(h))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))




# ------------------------------------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------------------------------------
def sun_dir():
    elev = math.radians(45.0)
    return (0.3 * math.cos(elev), math.sin(elev), 0.95 * math.cos(elev))


def build_rows(spec, sim, route, schema, variant):
    """{stream: [row dict by contract column]} with raw values."""
    ego_slot = 1
    entity = "0x00010000A1"
    entity_b = "0x00010000B7"
    rows = {name: [] for name in schema["streams"]}
    rf0 = spec["render_frame0"]
    reuse_from = None
    if variant == "reused_slot":
        start1 = [ep for ep in sim.episodes if ep["episode_id"] == 1][0]["start_cb"]
        reuse_from = sim.callbacks[start1]["world"] + 1000
    for cb in sim.callbacks:
        cmd, exp = cb["cmd"], cb["exp"]
        lim_e = None if cb["lim_c"] is None else STEER_SIGN * cb["lim_c"]
        raw_e = None if cb["raw_c"] is None else STEER_SIGN * cb["raw_c"]
        in_e = STEER_SIGN * cb["steer_in_c"]
        ctl = {"physics_step": cb["p"], "covered_steps": cb["covered"], "render_frame": rf0 + cb["frame"],
               "world_time_ms": cb["world"], "episode_id": cb["episode"], "mode": cb["mode"], "driver": cb["driver"],
               "hl_seq": cb["hl_seq"],
               "hl_kappa": None if cmd is None else cmd[0], "hl_accel": None if cmd is None else cmd[1],
               "hl_dir": None if cmd is None else cmd[2],
               "exp_kappa_now": None if exp is None else exp[0], "exp_accel_now": None if exp is None else exp[1],
               "exp_dir_now": None if exp is None else exp[2],
               "llc_steer_raw": raw_e, "llc_steer_lim": lim_e,
               "llc_throttle": cb["throttle"] if cb["llc_on"] else None,
               "llc_brake": cb["brake"] if cb["llc_on"] else None,
               "llc_handbrake": 0 if cb["llc_on"] else None, "llc_gear": cb["llc_gear"], "llc_clutch": None,
               "noise_active": int(cmd is not None and len(cmd) > 3 and cmd[3][0]),
               "noise_id": cmd[3][1] if cmd is not None and len(cmd) > 3 else -1, "steer_signal_raw": None,
               "wheel_angle_l_rad": None,
               "wheel_angle_r_rad": None, "react_veh_slot_id": None, "react_visible": None}
        for when in ("pre", "post"):
            ctl.update({"input_%s_steer" % when: in_e, "input_%s_throttle" % when: cb["throttle"],
                        "input_%s_brake" % when: cb["brake"], "input_%s_clutch" % when: 0.0,
                        "input_%s_gear" % when: cb["gear"], "input_%s_handbrake" % when: cb["handbrake"]})
        rows["ctl"].append(ctl)
        reused = reuse_from is not None and cb["world"] >= reuse_from
        rows["telemetry"].append(_telemetry_row(cb, route, ego_slot, entity_b if reused else entity,
                                                sim.dv[cb["index"]], rf0))
    for fr in sim.frames:
        rows["hf"].append(_hf_row(fr))
    for sl in sim.slots:
        issued = sl["issue_frame"] is not None
        fr = sim.frames[sl["issue_frame"]] if issued else None
        rows["img"].append({"slot_id": sl["slot_id"], "episode_id": sl["episode"], "ego_slot_id": ego_slot,
                            "requested_world_ms": sl["due"], "issue_status": "issued" if issued else "skipped_busy",
                            "issue_world_ms": fr["world"] if issued else None,
                            "issue_render_frame": fr["render_frame"] if issued else None,
                            "issue_physics_step": fr["lps"] if issued else None,
                            "path": "frames/f_%08d" % sl["slot_id"] if issued else None,
                            "cam_current": 1, "cam_attached": 1,
                            "done_render_frame": None if sl["done_frame"] is None
                            else rf0 + sl["done_frame"]})
    events = [{"step": -1, "episode": -1, "frame": -1, "world": 0, "kind": "session_start", "payload": ""}]
    for seq, ev in enumerate(events + sim.events):
        rows["events"].append({"event_seq": seq, "episode_id": ev["episode"], "physics_step": ev["step"],
                               "render_frame": rf0 + ev["frame"], "world_time_ms": ev["world"], "kind": ev["kind"],
                               "veh_slot_id": ego_slot, "payload": ev["payload"]})
    for ep in sim.episodes:
        a, b = sim.callbacks[ep["start_cb"]], sim.callbacks[ep["end_cb"]]
        reused = variant == "reused_slot" and ep["episode_id"] == 1
        rows["episodes"].append({
            "episode_id": ep["episode_id"], "start_physics_step": a["p"], "end_physics_step": b["p"],
            "start_world_ms": a["world"], "end_world_ms": b["world"], "ego_slot_id": ego_slot,
            "ego_entity_id": entity_b if reused else entity, "ego_prefab": "synthetic/S105.et",
            "vehicle_card_id": "S105_synthetic", "route_id": route.route_id,
            "route_sha256": route.sha256 if spec["fill_route_sha"] else None,
            "mode": spec.get("episode_mode", "E"), "driver": "expert",
            "style_speed_factor": 1.0, "style_lane_offset_m": 0.0, "style_alat_max_mps2": 2.75,
            "env_weather_state": "Clear", "env_tod_h_start": 12.0, "env_rain": 0.0, "env_fog": 0.0,
            "env_wetness": 0.0, "seed": spec["seed"] * 100 + ep["episode_id"],
            "termination_reason": ep["termination"], "expert_sha": _hex("synthetic expert"),
            "llc_sha": _hex("synthetic llc"), "nav_version": "nav_v1", "noise_steer_sensitivity": 1.0,
            "roughness_sensitivity": 1.0, "veh_dirt_body": None, "veh_dirt_glass": None})
    if variant == "gap":
        dropped = set(sim.callbacks[i]["p"] for i in gap_callbacks(sim))
        rows["ctl"] = [r for r in rows["ctl"] if r["physics_step"] not in dropped]
    for name, stream_rows in rows.items():
        cols = [f["name"] for f in schema["streams"][name]["fields"]]
        for r in stream_rows:
            if sorted(r) != sorted(cols):
                raise AssertionError("%s row keys differ from the streams JSON: %s"
                                     % (name, sorted(set(r) ^ set(cols))))
    return rows


def _telemetry_row(cb, route, ego_slot, entity, dv_max, rf0):
    psi = cb["psi"]
    fwd = (math.sin(psi), 0.0, math.cos(psi))
    right = (math.cos(psi), 0.0, -math.sin(psi))
    lane_lat = lane_head = None
    if cb["progress"] is not None:
        s, _ = route.project(cb["x"], cb["z"], cb["progress"] - 30.0, cb["progress"] + 30.0)
        a = route.point_at(s - 0.5)
        b = route.point_at(s + 0.5)
        tangent = math.atan2(b[0] - a[0], b[1] - a[1])
        px, pz = route.point_at(s)
        _, lane_lat = drivedit_nav.to_ego(drivedit_nav.Pose(px, pz, tangent), cb["x"], cb["z"])
        lane_head = wrap(psi - tangent)
    sun = sun_dir()
    row = {"physics_step": cb["p"], "covered_steps": cb["covered"], "render_frame": rf0 + cb["frame"],
           "world_time_ms": cb["world"], "episode_id": cb["episode"],
           "ego_slot_id": ego_slot, "ego_entity_id": entity, "pos_x": cb["x"], "pos_y": 30.0, "pos_z": cb["z"],
           "right_x": right[0], "right_y": right[1], "right_z": right[2], "up_x": 0.0, "up_y": 1.0, "up_z": 0.0,
           "fwd_x": fwd[0], "fwd_y": fwd[1], "fwd_z": fwd[2], "vel_local_x": 0.0, "vel_local_y": 0.0,
           "vel_local_z": cb["v"], "angvel_local_x": 0.0, "angvel_local_y": YAW_AXIS_SIGN * cb["yaw"],
           "angvel_local_z": 0.0, "acc_local_x": cb["v"] * cb["yaw"], "acc_local_y": 0.0, "acc_local_z": cb["acc"],
           "speed_engine_mps": cb["v"], "dv_max_mps": dv_max, "gear": cb["gear"], "rpm": 800.0 + 120.0 * cb["v"],
           "engine_on": 1,
           "handbrake": cb["handbrake"], "ground_contact": 1, "wheel_count": 4,
           "road_key": "12/1000.00:3000.00/1003.40:3068.10", "road_dist_m": 0.4, "road_width_m": 7.0,
           "lane_lat_err_m": lane_lat, "lane_head_err_rad": lane_head, "route_progress_m": cb["progress"],
           "env_weather_state": "Clear", "env_rain": 0.0, "env_fog": 0.0, "env_fog_height_density": 0.0,
           "env_wetness": 0.0, "env_tod_h": 12.0 + cb["world"] / 3600000.0, "sun_dir_x": sun[0],
           "sun_dir_y": sun[1], "sun_dir_z": sun[2], "moon_phase01": 0.5, "sun_elev_deg": 45.0, "traffic_n": 0,
           "traffic_slots": None}
    for w in range(4):
        row.update({"w%d_contact" % w: 1, "w%d_slip_long" % w: 0.01 * cb["acc"] / 3.0,
                    "w%d_slip_lat" % w: 0.002 * cb["v"] * cb["yaw"], "w%d_material" % w: "Asphalt",
                    "w%d_temp_c" % w: None})
    return row


def _hf_row(fr):
    """hf row of render frame f: the pose at the frame's EOnFrame (the state at the world time of frame f - 1)."""
    src = fr["state"]
    psi = src["psi"]
    fwd = (math.sin(psi), 0.0, math.cos(psi))
    right = (math.cos(psi), 0.0, -math.sin(psi))
    pitch = math.radians(-4.0)
    cam_fwd = (fwd[0] * math.cos(pitch), math.sin(pitch), fwd[2] * math.cos(pitch))
    cam_up = (-fwd[0] * math.sin(pitch), math.cos(pitch), -fwd[2] * math.sin(pitch))
    row = {"render_frame": fr["render_frame"], "world_time_ms": fr["world"],
           "last_physics_step": fr["lps"], "paused": 0,
           "cam_pos_x": src["x"] + 1.8 * fwd[0], "cam_pos_y": 31.5, "cam_pos_z": src["z"] + 1.8 * fwd[2],
           "cam_right_x": right[0], "cam_right_y": right[1], "cam_right_z": right[2],
           "cam_up_x": cam_up[0], "cam_up_y": cam_up[1], "cam_up_z": cam_up[2],
           "cam_fwd_x": cam_fwd[0], "cam_fwd_y": cam_fwd[1], "cam_fwd_z": cam_fwd[2],
           "veh_pos_x": src["x"], "veh_pos_y": 30.0, "veh_pos_z": src["z"],
           "veh_right_x": right[0], "veh_right_y": right[1], "veh_right_z": right[2],
           "veh_up_x": 0.0, "veh_up_y": 1.0, "veh_up_z": 0.0, "veh_fwd_x": fwd[0], "veh_fwd_y": fwd[1],
           "veh_fwd_z": fwd[2], "fps": 60.0}
    return row

# ------------------------------------------------------------------------------------------------
# Writer (chunks, commit lines, session files)
# ------------------------------------------------------------------------------------------------
def write_stream(sdir, spec_stream, rows, chunk_max_s, drop_last_commit=False):
    """Chunks at chunk_max_rows, after chunk_max_s of world time or at the end; one commit line per chunk.
    Returns (quantised rows, counts {rows, chunks, last_key} as the writer believes them)."""
    fields = spec_stream["fields"]
    cols = [f["name"] for f in fields]
    key = spec_stream["key"][0]
    os.makedirs(sdir)
    header = ",".join(cols)
    chunks = []
    cur = []
    t0 = None
    for r in rows:
        w = next((r[c] for c in ("world_time_ms", "requested_world_ms", "end_world_ms") if c in r), None)
        if cur and (len(cur) >= spec_stream["chunk_max_rows"] or (w is not None and t0 is not None
                                                                   and w - t0 >= chunk_max_s * 1000)):
            chunks.append(cur)
            cur = []
        if not cur:
            t0 = w
        cur.append(r)
    if cur:
        chunks.append(cur)
    quantised = []
    commit_lines = ["chunk_id,first_key,last_key,n_rows,n_bytes"]
    for cid, chunk in enumerate(chunks):
        lines = [header]
        for r in chunk:
            texts = [fmt(f, r[f["name"]]) for f in fields]
            lines.append(",".join(texts))
            quantised.append({f["name"]: quantise(f, t) for f, t in zip(fields, texts)})
        data = ("\n".join(lines) + "\n").encode("ascii")
        with open(os.path.join(sdir, "chunk_%05d.csv" % cid), "wb") as f:
            f.write(data)
        commit_lines.append("%d,%d,%d,%d,%d" % (cid, chunk[0][key], chunk[-1][key], len(chunk), len(data)))
    if drop_last_commit and len(commit_lines) > 1:
        commit_lines.pop()
    if chunks:
        with open(os.path.join(sdir, "commit.csv"), "wb") as f:
            f.write(("\n".join(commit_lines) + "\n").encode("ascii"))
    counts = {"rows": len(rows), "chunks": len(chunks), "last_key": rows[-1][key] if rows else None}
    return quantised, counts


def _iso(unix_s):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(unix_s))


def trial_conf(spec):
    """The trial conf as parsed into session.json trial.conf (warmup_s is set from the simulation); the 10 Hz
    setting names image_hz (TBD-2)."""
    conf = {"trial_id": spec["trial_id"], "route": spec["route"], "seed": spec["seed"], "warmup_s": 0.0,
            "synthetic": True}
    if spec["decision_hz"] != 5:
        conf["image_hz"] = spec["decision_hz"]
    if spec.get("noise"):
        conf["perturb"] = "pilot_v1"            # DC_NoiseConfig's profile key (DD-7-ident)
    return conf


def session_json(spec, start_unix, card_sha):
    sid = spec["session_id"]
    exe = spec.get("exe", "wb")
    launch_id = "%s-%s-%s" % (_iso(start_unix - 60).replace("-", "").replace(":", ""), RECORD_EXE[exe], sid[-4:])
    w, h, band = spec["image_w"], spec["image_h"], spec["band_px"]
    render_settings = {"synthetic": True, "VideoUserSettings": {}, "DisplayUserSettings": {}}
    return {
        "contract_version": "dc-2.0", "session_id": sid, "created_utc": _iso(start_unix),
        "start": {"unix_s": start_unix, "local_hms": time.strftime("%H:%M:%S", time.gmtime(start_unix)),
                  "engine_world_ms": 183250, "render_frame": spec["render_frame0"]},
        "map": {"map_id": "arland", "terrain": "Arland", "world_file": "worlds/DataCapture/DC_Arland.ent",
                "root_world": "{A9806AF617972E97}worlds/Arland/Arland.ent",
                "fingerprint": {"size_m": 4096.0, "latitude_deg": 47.411, "ok": True}},
        "trial": {"trial_id": spec["trial_id"], "source": "cli",
                  "conf": trial_conf(spec),
                  "seed": spec["seed"], "warmup_s": 0.0, "duration_s": 0.0},
        "exe": {"kind": exe, "instance": spec.get("instance", "i00"),
                "cli": {"dcLaunchId": launch_id, "dcTrial": spec["trial_id"]}},
        "engine": {"build_version": "1.8.0.13", "build_time": "synthetic", "fixed_time_slice_s": round(STEP_DT_S, 6),
                   "step_dt_s": round(STEP_DT_S, 6), "step_dt_source": "engine"},
        "pin": {"pin_id": "1.8.0.13", "pin_sha256": _hex("synthetic pin 1.8.0.13"), "build_key": None,
                "channel": "stable", "launch_id": launch_id, "launch_schema": "dd-launch/1",
                "game_buildid": 24903726, "tools_buildid": 24870687},
        "use_class": "dev",
        "addon": {"addon_id": "DataCapture", "addon_guid": "SYNTHETIC-ADDON-GUID", "git_sha": _hex("synthetic", 40),
                  "git_branch": "synthetic", "git_dirty": False, "content_sha256": _hex("synthetic content"),
                  "manifest_sha256": _hex("synthetic manifest"), "deploy_utc": _iso(start_unix - 3600),
                  "target_pin": None},
        "control": {"actuation_mode": "A", "llc_sha": _hex("synthetic llc"), "expert_sha": _hex("synthetic expert"),
                    "vehicle_card_id": "S105_synthetic", "vehicle_card_sha": card_sha, "steer_sign": STEER_SIGN,
                    "yaw_axis_sign": YAW_AXIS_SIGN, "steering_geometry": "unknown_on_1.8"},
        "authority": {"standalone": True, "rpl_ok": True},
        "versions": {"features_version": "features_v1", "nav_version": "nav_v1"},
        "clocks": {"physics_steps_per_decision": steps_per_decision(spec["decision_hz"]),
                   "decision_hz": float(spec["decision_hz"]), "image_hz": float(spec["decision_hz"]),
                   "image_hz_setting": image_hz_setting(spec), "img_inflight_timeout_ms": 2000, "img_lead_ms": 0,
                   "chunk_max_s": 10.0},
        "events": dict(EVENT_PARAMS),
        "rig": {"rig_id": "R1-S105-synthetic-%dx%d-n0.03" % (w, h), "rig_name": "R1-S105",
                "ego_prefab": "synthetic/S105.et",
                "mount": {"fwd_m": 1.8, "up_m": 1.5, "right_m": 0.0, "pitch_deg": -4.0}, "vfov_deg": 50.0,
                "near_plane_m": 0.03,
                "writer": "MakeScreenshot", "source_class": spec.get("source_class", "wb_viewport"),
                "capture_wh": [w, h], "core_wh": [w - 2 * band, h], "band_px": band, "derivative_wh": [512, 288],
                "derivative_codec": "jpeg_q95",
                "fiducial": {"version": "v2", "bits": 18, "patches": 6, "bits_per_patch": 3, "patch_px": 32,
                             "margin_px": 16, "gap_px": 8, "mirrored": True, "carrier": "shape", "depth_m": 0.05},
                "dpi_scale": 1.0, "render_settings": render_settings,
                "render_settings_hash": _hex(json.dumps(render_settings, sort_keys=True))},
        "env_initial": {"weather_state": "Clear", "rain": 0.0, "fog": 0.0, "wetness": 0.0, "tod_h": 12.0,
                        "date": "1989-08-20", "day_auto_advance": False},
        "launch_guard": {"launch_mode": "cli", "record_path": "$profile:dc_launch/launch_%s.json" % launch_id,
                         "record_launch_id": launch_id, "cli_seen": True, "profile_root_ok": True,
                         "escape_binding": None},
    }


PREFLIGHT_CHECKS = ("acf", "appid", "buildid", "target_buildid", "auto_update_behavior", "state_flags", "depots",
                    "steam_appid_txt", "binaries", "paks", "launch_path")


def launch_record(meta):
    """The session's dd-launch/1 record as DD-0's preflight writes it (tools/pin/preflight.py make_record): status
    ok, the 22 named checks of both apps, the observed appmanifest buildids. Synthetic values, no host name."""
    pin = meta["pin"]
    exe = RECORD_EXE[meta["exe"]["kind"]]
    instance = meta["exe"]["instance"] if exe == "client" else None
    observed = {"game": {"appid": 1874880, "buildid": pin["game_buildid"], "target_buildid": pin["game_buildid"],
                         "auto_update_behavior": "1"},
                "tools": {"appid": 1874910, "buildid": pin["tools_buildid"], "target_buildid": pin["tools_buildid"],
                          "auto_update_behavior": "1"}}
    return {"schema": pin["launch_schema"], "launch_id": pin["launch_id"], "status": "ok",
            "utc": meta["created_utc"], "exe": exe, "instance": instance,
            "pin": {"path": "synthetic/pin_%s.json" % pin["pin_id"], "pin_id": pin["pin_id"],
                    "sha256": pin["pin_sha256"], "resolved_by": "active-pointer", "pointer": "synthetic/ACTIVE"},
            "observed": observed,
            "checks": [{"name": "%s.%s" % (app, c), "ok": True, "detail": "synthetic"}
                       for app in ("game", "tools") for c in PREFLIGHT_CHECKS],
            "host": {"hostname": "synthetic", "wsl_distro": None, "python": "synthetic"},
            "argv": ["--pin-active", "--exe", exe] + (["--instance", instance] if instance else [])}


def _json_text(obj):
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"



def generate(out_dir, kind="golden", variant=None, golden_json=None, routes_dir=None, route_obj=None):
    """Write one synthetic session under out_dir. Returns {"session_dir", "session_id", "route", "golden"}."""
    if kind not in KINDS:
        raise ValueError("kind must be one of %s" % ", ".join(KINDS))
    if variant is not None and variant not in VARIANTS:
        raise ValueError("variant must be one of %s" % ", ".join(VARIANTS))
    out_dir = check_out(out_dir)
    spec = dict(SPECS[kind])
    if variant in CRASH_VARIANTS:
        spec.update({"crash": True, "crash_after_ms": 4000})
    with open(STREAMS_JSON, "rb") as f:
        schema = json.loads(f.read().decode("ascii"))
    route_obj = route_obj or ROUTES[spec["route"]]()
    route = drivedit_nav.route_from_dict(route_obj)
    session_dir = os.path.join(out_dir, spec["session_id"])
    if os.path.exists(session_dir):
        raise ValueError("refusing to overwrite %s" % session_dir)
    os.makedirs(session_dir)
    sim = simulate(spec, route)
    inject_crash(sim, variant)
    rows = build_rows(spec, sim, route, schema, variant)
    start_unix = calendar.timegm(spec["start_utc"])
    with open(CARD_FILE, "rb") as f:
        card_bytes = f.read()
    card = json.loads(card_bytes.decode("ascii"))
    counts = {}
    quantised = {}
    for name, srows in rows.items():
        sspec = schema["streams"][name]
        if sspec["status"] != "required" and not srows:
            continue
        quantised[name], counts[name] = write_stream(
            os.path.join(session_dir, sspec["dir"]), sspec, srows, 10.0,
            drop_last_commit=(variant == "missing_commit" and name == "ctl"))
    # images: the image captured in frame C shows the state after its physics (hf row C + 1); the engine writes the file
    # of every issued request, also of one a seal without drain left pending
    os.makedirs(os.path.join(session_dir, "frames"))
    pixels = {}
    done = [sl for sl in sim.slots if sl["done_frame"] is not None]
    written = [sl for sl in sim.slots if sl.get("capture_state") is not None]
    pixel_slots = _pixel_slots([sl["slot_id"] for sl in done])
    for sl in written:
        captured = sim.frames[sl["issue_frame"]]
        pose_frame = sl["issue_frame"] + 1         # the pose row; none for a request the seal caught in flight
        state = sim.frames[pose_frame]["state"] if pose_frame < len(sim.frames) else sl["capture_state"]
        img = render_image(spec, state, fiducial.code_for(captured["render_frame"]))
        with open(os.path.join(session_dir, "frames", "f_%08d.png" % sl["slot_id"]), "wb") as f:
            f.write(png_bytes(img))
        if sl["slot_id"] in pixel_slots:
            pixels[sl["slot_id"]] = img
    # heartbeat (every 5 s of world time, in the first frame past each boundary)
    hb_lines = ["unix_s,world_ms,physics_step,render_frame,fps,mem_kb,img_ok,img_fail"]
    n_hb = 0
    next_hb = 5000
    for fr in sim.frames:
        if fr["world"] >= next_hb:
            next_hb = fr["world"] - fr["world"] % 5000 + 5000
            ok = sum(1 for sl in done if sl["done_frame"] <= fr["index"])
            hb_lines.append("%d,%d,%d,%d,%.3f,%d,%d,%d" % (start_unix + fr["world"] // 1000, fr["world"], fr["lps"],
                                                          fr["render_frame"], 60.0,
                                                          1800000 + 2 * (fr["world"] // 1000), ok, 0))
            n_hb += 1
    with open(os.path.join(session_dir, "heartbeat.csv"), "wb") as f:
        f.write(("\n".join(hb_lines) + "\n").encode("ascii"))
    meta = session_json(spec, start_unix, hashlib.sha256(card_bytes).hexdigest())
    warm = [e for e in sim.events if e["kind"] == "warmup_end"][0]["world"]
    last_cb, last_frame = sim.callbacks[-1], sim.frames[-1]
    meta["trial"]["warmup_s"] = warm / 1000.0
    meta["trial"]["conf"]["warmup_s"] = warm / 1000.0
    meta["trial"]["duration_s"] = round((last_cb["world"] - warm) / 1000.0, 3)
    end_world = last_cb["world"] if sim.crash is not None else last_frame["world"]
    end_unix = start_unix + end_world // 1000 + 1
    issued = [sl for sl in sim.slots if sl["issue_frame"] is not None]
    end = {"contract_version": "dc-2.0", "session_id": spec["session_id"], "status": "ok",
           "end_reason": sim.end_reason,
           "end": {"unix_s": end_unix, "utc": _iso(end_unix), "world_ms": end_world, "physics_step": last_cb["p"],
                   "render_frame": last_frame["render_frame"]},
           "pin": {"pin_id": "1.8.0.13", "build_key": None}, "engine": {"build_version": "1.8.0.13"},
           "counts": counts,
           "images": {"requested": len(sim.slots), "issued": len(issued),
                      "skipped_busy": len(sim.slots) - len(issued), "make_failed": 0, "timed_out": 0,
                      "pending_at_end": len(issued) - len(done)},
           "episodes": len(sim.episodes), "heartbeats": n_hb,
           "observed": {"decision_grid_points_skipped": sum(b["grid"] - a["grid"] - 1 for a, b in
                                                            zip(sim.decisions, sim.decisions[1:]))}}
    with open(os.path.join(session_dir, "session.json"), "wb") as f:
        f.write(_json_text(meta).encode("ascii"))
    with open(os.path.join(session_dir, "harvest.json"), "wb") as f:
        f.write(_json_text({"schema": "drivedit.harvest/0-synthetic", "session_id": spec["session_id"],
                            "build_key": _hex("synthetic build_key 1.8.0.13"),
                            "note": "DD-6-min writes harvest.json; this synthetic one carries only build_key"}
                           ).encode("ascii"))
    _write_logs(session_dir, spec, start_unix, written, sim.end_reason, end_world)
    os.makedirs(os.path.join(session_dir, "_launch"))
    with open(os.path.join(session_dir, "_launch", "launch_%s.json" % meta["pin"]["launch_id"]), "wb") as f:
        f.write(_json_text(launch_record(meta)).encode("ascii"))
    with open(os.path.join(session_dir, "session_end.json"), "wb") as f:
        f.write(_json_text(end).encode("ascii"))
    if routes_dir is not None:
        routes_dir = check_out(routes_dir)
        os.makedirs(routes_dir, exist_ok=True)
        with open(os.path.join(routes_dir, route.route_id + ".json"), "wb") as f:
            f.write((json.dumps(route_obj, indent=1) + "\n").encode("ascii"))
    golden = None
    if golden_json is not None:
        golden = golden_truth(spec, sim, quantised, card, pixels)
        with open(check_out(golden_json), "wb") as f:
            f.write((json.dumps(golden, indent=1, sort_keys=True) + "\n").encode("ascii"))
    return {"session_dir": session_dir, "session_id": spec["session_id"], "route": route_obj, "golden": golden,
            "sim": sim}


def _pixel_slots(slot_ids):
    n = len(slot_ids)
    return sorted(set(slot_ids[i] for i in (0, n // 3, (2 * n) // 3, n - 1)))


def _write_logs(session_dir, spec, start_unix, slots, end_reason, end_world):
    sid = spec["session_id"]
    ldir = os.path.join(session_dir, "_engine_logs")
    os.makedirs(ldir)

    def hms(ms):
        t = start_unix + ms // 1000
        return "%s.%03d" % (time.strftime("%H:%M:%S", time.gmtime(t)), ms % 1000)

    lines = ["%s SCRIPT       : [DC] session_start id=%s" % (hms(0), sid)]
    for sl in slots:
        lines.append("%s RESOURCES    : Screenshot saved to $profile:DC_Sessions/%s/frames/f_%08d.png"
                     % (hms(sl["due"] + 40), sid, sl["slot_id"]))
    lines.append("%s SCRIPT       : [DC] session_end reason=%s" % (hms(end_world), end_reason))
    with open(os.path.join(ldir, "console.log"), "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))


# ------------------------------------------------------------------------------------------------
# golden.json (expected reader output, from the simulation)
# ------------------------------------------------------------------------------------------------
def _r6(values):
    return None if values is None else [None if v is None else (round(v, 6) if isinstance(v, float) else v)
                                        for v in values]


def _card_dir_state(card, gear):
    if gear in set(card["gear_indices_forward"]):
        return 1
    if gear == card.get("gear_index_reverse"):
        return -1
    if gear == card.get("gear_index_neutral") or card.get("gear_index_neutral") is None:
        return 0
    return None


def golden_truth(spec, sim, q, card, pixels):
    """Expected reader output. The R-01 samples come from the simulation's decision events (sim.decisions: the
    decision callback, its grid point and the command the expert decided there), not from the ctl rows: action_cmd_t
    and hl_seq are that command and its index, action_label_t the expert evaluation after C_t, the decision interval
    is held when the simulation applied that command in every callback of it, and action_exec reads each step through
    the simulation's step-to-callback map. State, exec and obs values are the values as written (q)."""
    decision_hz = spec["decision_hz"]
    n = steps_per_decision(decision_hz)
    period = 1000.0 / decision_hz
    cbs = sim.callbacks
    ctl = {r["physics_step"]: r for r in q["ctl"]}
    tel = {r["physics_step"]: r for r in q["telemetry"]}
    step_cb = {}
    for c in cbs:
        for s in range(c["p"] - c["covered"] + 1, c["p"] + 1):
            step_cb[s] = c
    cmd_index = {d["cb"]: i for i, d in enumerate([d for d in sim.decisions if d["cmd"] is not None])}
    images, cands = [], []
    for sl in sim.slots:
        if sl["done_frame"] is None:
            continue
        cf = sim.frames[sl["issue_frame"]]
        images.append({"slot_id": sl["slot_id"], "file": "frames/f_%08d.png" % sl["slot_id"], "file_exists": 1,
                       "png_w": spec["image_w"], "png_h": spec["image_h"], "fid_ok": 1,
                       "captured_render_frame": cf["render_frame"]})
        if sl["issue_frame"] + 1 < len(sim.frames):
            pose = sim.frames[sl["issue_frame"] + 1]
            if pose["lps"] >= 0:      # a pose row before the first callback joins neither ctl nor telemetry
                cands.append((cf["render_frame"], pose["lps"], cf["world"], sl["slot_id"],
                              step_cb[pose["lps"]]["episode"]))
    warmup_end = [e for e in sim.events if e["kind"] == "warmup_end"][0]["world"]
    teleports = [e["step"] for e in sim.events if e["kind"] in ("teleport", "respawn")]
    decisions = []
    for k, dec in enumerate(sim.decisions):
        c0 = cbs[dec["cb"]]
        p, world, episode = c0["p"], c0["world"], c0["episode"]
        nxt = cbs[sim.decisions[k + 1]["cb"]] if k + 1 < len(sim.decisions) else None
        p_next = nxt["p"] if nxt is not None else None
        after = cbs[dec["cb"] + 1] if dec["cb"] + 1 < len(cbs) else None
        interval = cbs[dec["cb"] + 1:nxt["index"] + 1] if nxt is not None else []
        complete = nxt is not None
        one_ep = complete and episode >= 0 and all(c["episode"] == episode for c in interval)
        t0 = tel[p]
        gear_state = _card_dir_state(card, t0["gear"])
        state = [t0["vel_local_z"], YAW_AXIS_SIGN * t0["angvel_local_y"], t0["acc_local_z"], gear_state]
        label = _r6(list(c0["exp"])) if c0["exp"] is not None else [None, None, None]
        if dec["cmd"] is not None:
            hl_seq, cmd = cmd_index[dec["cb"]], _r6(list(dec["cmd"]))
        elif after is not None:
            hl_seq = after["hl_seq"]
            cmd = _r6(list(after["cmd"])) if after["cmd"] is not None else [None, None, None]
        else:
            hl_seq, cmd = None, None
        held = dec["cmd"] is not None and complete and all(
            c["cmd"] == dec["cmd"] and c["hl_seq"] == cmd_index[dec["cb"]] for c in interval)
        exec_rows = []
        for s in range(p + 1, p + n + 1):
            if p_next is None or s > p_next or s not in step_cb:
                exec_rows.append(None)
                continue
            r = ctl[step_cb[s]["p"]]
            exec_rows.append([None if r["llc_steer_lim"] is None else STEER_SIGN * r["llc_steer_lim"],
                              r["llc_throttle"], r["llc_brake"], STEER_SIGN * r["input_post_steer"]])
        inside = exec_rows[:min(n, p_next - p)] if p_next is not None else []
        exec_ok = bool(inside) and all(e is not None and None not in e for e in inside)
        eligible = [c for c in cands if c[1] <= p and c[0] <= spec["render_frame0"] + c0["frame"]]
        obs = max(eligible) if eligible else None
        obs_valid = 1 if obs is not None and world - obs[2] <= period else 0
        same_ep = obs_valid == 1 and obs[4] == episode
        first_q = obs[1] if obs_valid else p
        end_q = p_next if p_next is not None else p + n
        teleport = 1 if any(first_q <= t <= end_q for t in teleports) else 0
        mode = after["mode"] if after is not None else c0["mode"]
        wm_valid = int(complete and one_ep and held and exec_ok and obs_valid == 1 and same_ep and not teleport)
        policy_valid = int(episode >= 0 and None not in label and mode not in ("X", "A", "I") and obs_valid == 1
                           and same_ep and not teleport)
        decisions.append({
            "decision_step": p, "grid_point": dec["grid"], "episode_id": episode, "hl_seq": hl_seq,
            "interval_steps": None if p_next is None else p_next - p,
            "state": state, "action_cmd": cmd, "action_label": label, "action_exec": exec_rows,
            "obs_slot_id": None if obs is None else obs[3],
            "obs_captured_render_frame": None if obs is None else obs[0],
            "obs_last_physics_step": None if obs is None else obs[1],
            "obs_age_steps": None if obs is None else p - obs[1],
            "obs_age_ms": None if obs is None else world - obs[2], "obs_valid": obs_valid,
            "masks": {"policy_valid": policy_valid, "wm_valid": wm_valid, "warmup": 1 if world < warmup_end else 0,
                      "env_transition": 0, "teleport": teleport}})
    segments = []
    cur = None
    for d in decisions:
        ok = d["masks"]["wm_valid"] == 1 and d["masks"]["warmup"] == 0 and d["episode_id"] >= 0
        if ok and cur is not None and cur["episode_id"] == d["episode_id"] and d["grid_point"] == cur["grid"] + 1:
            cur["last_decision_step"] = d["decision_step"]
            cur["grid"] = d["grid_point"]
            cur["n_steps"] += 1
            continue
        if cur is not None:
            segments.append(cur)
            cur = None
        if ok:
            cur = {"episode_id": d["episode_id"], "first_decision_step": d["decision_step"],
                   "last_decision_step": d["decision_step"], "grid": d["grid_point"], "n_steps": 1}
    if cur is not None:
        segments.append(cur)
    for seg in segments:
        del seg["grid"]
    band = spec["band_px"]
    stride = 16
    pix = []
    for slot in sorted(pixels):
        core = pixels[slot][:, band:spec["image_w"] - band, :].transpose(2, 0, 1)[:, ::stride, ::stride]
        pix.append({"slot_id": slot, "core": numpy.round(core.astype(float) / 255.0, 6).tolist()})
    return {"schema": "drivedit.dc2-golden/1", "session_id": spec["session_id"], "contract_version": "dc-2.0",
            "clocks": {"physics_steps_per_decision": n, "decision_hz": float(decision_hz),
                       "image_hz": float(decision_hz), "image_hz_setting": image_hz_setting(spec),
                       "img_lead_ms": 0, "step_dt_s": round(STEP_DT_S, 6)},
            "counts": {name: len(rows) for name, rows in sorted(q.items())},
            "img_resolved": images, "segments": segments, "decision_steps": decisions,
            "pixels": {"stride": stride, "slots": pix}}


def image_hz_setting(spec):
    """clocks.image_hz_setting: the 5 Hz default, or the 10 Hz trial-conf setting (TBD-2)."""
    return "default" if spec["decision_hz"] == 5 else "trial_conf"




# ------------------------------------------------------------------------------------------------
# dc-1.0 legacy session (the legacy self-test and the enfusion_preprocessing guard test)
# ------------------------------------------------------------------------------------------------
# The dc-1.0 telemetry.csv columns in writer order, copied verbatim from WriteTelemetryHeader of the dc-1.0 writer
# src/enfusion/datacapture/SCR_MLDataCollector.c, so the legacy fixture never invents a schema (CR-109). DD-12 deleted
# that writer from main; it is kept in tag archive/dc1-rust:
#     git show archive/dc1-rust:src/enfusion/datacapture/SCR_MLDataCollector.c
LEGACY_HEADER = (
    "frame_id", "timestamp_ms",
    "pos_x", "pos_y", "pos_z",
    "fwd_x", "fwd_y", "fwd_z",
    "up_x", "up_y", "up_z",
    "right_x", "right_y", "right_z",
    "speed_kmh", "steering", "throttle", "brake", "clutch",
    "gear", "engine_rpm", "engine_on", "handbrake",
    "acceleration_kmh_s", "distance_total_m",
    "waypoint_type", "speed_limit_kmh",
    "waypoint_idx", "vehicle_id",
)


def legacy_header():
    """The dc-1.0 telemetry.csv columns (LEGACY_HEADER, from the dc-1.0 writer's WriteTelemetryHeader)."""
    return list(LEGACY_HEADER)


def write_legacy_session(out_dir, name="session_9001", n_frames=40, vehicles=5, frames_present=(0, 10, 20)):
    """A small dc-1.0 session as the DD-1 writer leaves it: telemetry.csv (CRLF, one row per vehicle per tick, tick
    intervals of 200 and 400 ms as in sessions 2702 and 4811), summary.txt and frames/frame_<id:06d>.bmp.png for
    frames_present only. Returns the session directory."""
    out_dir = check_out(out_dir)
    sdir = os.path.join(out_dir, name)
    os.makedirs(os.path.join(sdir, "frames"))
    cols = legacy_header()
    lines = [",".join(cols)]
    t = 2902.6
    for f in range(n_frames):
        for v in range(vehicles):
            values = {"frame_id": "%d" % f, "timestamp_ms": "%.1f" % t, "vehicle_id": "%d" % v,
                      "pos_x": "%.4f" % (2749.8628 + 0.5 * f + 10 * v), "pos_y": "%.4f" % 31.2,
                      "pos_z": "%.4f" % (1200.0 + v), "fwd_z": "1.0000", "up_y": "1.0000", "right_x": "1.0000",
                      "speed_kmh": "%.3f" % (20.0 + v), "gear": "2", "engine_rpm": "1500.0", "engine_on": "1"}
            lines.append(",".join(values.get(c, "0.0000") for c in cols))
        t += 200.0 if f % 2 == 0 else 400.0
    with open(os.path.join(sdir, "telemetry.csv"), "wb") as fh:
        fh.write(("\r\n".join(lines) + "\r\n").encode("ascii"))
    with open(os.path.join(sdir, "summary.txt"), "wb") as fh:
        fh.write(("total_frames=%d\r\nvehicles_tracked=%d\r\n" % (n_frames, vehicles)).encode("ascii"))
    img = numpy.full((40, 64, 3), 128, dtype=numpy.uint8)
    for f in frames_present:
        with open(os.path.join(sdir, "frames", "frame_%06d.bmp.png" % f), "wb") as fh:
            fh.write(png_bytes(img))
    return sdir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="parent directory of the new session (under /dev/shm or the data drive)")
    ap.add_argument("--kind", choices=KINDS, default="golden")
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--golden-json", default=None, help="also write the expected reader output here")
    ap.add_argument("--routes-dir", default=None, help="also write the route file here")
    args = ap.parse_args(argv)
    try:
        result = generate(args.out, args.kind, args.variant, args.golden_json, args.routes_dir)
    except ValueError as e:
        print("refused: %s" % e)
        return 2
    print(result["session_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
