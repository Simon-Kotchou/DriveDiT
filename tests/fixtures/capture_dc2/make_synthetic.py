#!/usr/bin/env python3
"""Synthetic dc-2.0 sessions for DD-5-min: the golden fixture, the negative fixtures and the C17 loop sessions.

    python3 -I -B tests/fixtures/capture_dc2/make_synthetic.py --out DIR [--kind golden|t0_loop|golden_5hz]
        [--variant gap|missing_commit|reused_slot] [--golden-json FILE] [--routes-dir DIR]

writes DIR/<session_id>/ and, with --golden-json, the expected reader output (golden.json). The committed golden
fixture was written by

    python3 -I -B tests/fixtures/capture_dc2/make_synthetic.py --out tests/fixtures/capture_dc2/golden \
        --golden-json tests/fixtures/capture_dc2/golden.json --routes-dir tests/fixtures/capture_dc2/routes

SYNTHETIC: no T0b session exists yet (DD-4-min has not run). The real golden cut from the first clean T0b session
replaces this one (README.md).

What a session holds, as DD-4-min's writer will write it (docs/capture_contract/dc-2.0.md):
  * every stream of the streams JSON in its column order, values formatted per unit (fixed point: 4 decimals for
    m, m/s, m/s2 and 6 for rad, rad/s, 1/m and unitless numbers), chunks closed at chunk_max_rows, after 10 s of
    world time or at the end, each committed by one commit.csv line; session.json, session_end.json, heartbeat.csv;
  * the harvester's sidecars: harvest.json (build_key of a dd-launch/1 session), _engine_logs/console.log (the
    sliced engine log: session markers and one "Screenshot saved to" line per image) and _launch/launch_<id>.json
    (the session's dd-launch/1 preflight record, copied from the pin's launches directory; C02 joins it);
  * frames/f_<slot:08d>.png (TBD-4 default name): small images whose side bands carry real fiducial v2 patches
    (data/dc2/fiducial.py layout, the DD-4-probe encoder) of the captured render frame, so the reader's img -> hf
    join runs on decoded codes. The core shows an azimuth texture that moves with the heading and stripes that move
    with the distance driven, so C12 and C13b see motion.

The vehicle (60 Hz, k = 3, N = 6, image_hz 10, the TBD defaults) is a kinematic model: curvature = KAPPA_PER_STEER x
contract steering input, yaw rate = speed x curvature, an llc_v1 stand-in (rate limiter and first-order lag) and a
pure-pursuit expert on a drivedit_nav route. Decisions follow R-01: the expert decides in EOnPostSimulate of P_t, the
command applies from P_t + 1, hl_seq increments at P_t + 1.

golden.json is computed here from the simulation's own values, not by the reader, so the reader test compares two
independent computations. The R-01 samples come from the simulation's decision events: action_cmd_t and hl_seq are
the command the simulated expert decided at P_t and its decision index, action_label_t the expert evaluation after
step P_t, and a window is "held" when the simulation applied that same command in every step of it; none of them
is read back from the ctl rows at P_t + 1. The masks and the obs rule repeat the contract's rules (section 6).

Kind golden_5hz is the golden session at the TBD-2 fallback (image_hz = decision_hz = 5, N = 12, k = 3); the tests
generate it, it is not committed.

Variants (negative fixtures; each breaks one check):
  gap             ctl rows of physics steps 304 and 305 are missing (C06); commit lines and counts stay consistent
  missing_commit  the commit line of the last ctl chunk is missing while session_end still counts it (C34)
  reused_slot     from step 450 (episode 1) the ego entity id changes while ego_slot_id stays (C35)

Writes only under /mnt/e/ or /dev/shm/ (never /mnt/c or /tmp). Stdlib, numpy and drivedit_nav; PNGs are written by
a small zlib PNG writer, so no cv2 is needed here.
"""

import argparse
import bisect
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
WRITE_ROOTS = ("/mnt/e/", "/dev/shm/")
FORBIDDEN_ROOTS = ("/mnt/c/", "/tmp/")
VARIANTS = ("gap", "missing_commit", "reused_slot")
KINDS = ("golden", "t0_loop", "golden_5hz")

SLICE_S = 1.0 / 60.0
STEER_SIGN = -1                 # TBD-5 default: CarSteering +1 is left, the contract is right-positive
YAW_AXIS_SIGN = 1
KAPPA_PER_STEER = 0.15          # synthetic vehicle: curvature (1/m) per unit contract steering
STEER_RATE = 2.0                # llc stand-in rate limit (1/s)
STEER_ALPHA = SLICE_S / (0.08 + SLICE_S)   # first-order lag, tau 0.08 s
GEAR_FORWARD = 3                # S105_synthetic card: gear_indices_forward [3]
GEAR_IDLE = 1                   # unmapped gear while no command is held (dir_state 0)
RAD_PER_PX = 0.004              # synthetic camera: azimuth per core pixel column
DECIMALS = {"m": 4, "m/s": 4, "m/s2": 4, "rad": 6, "rad/s": 6, "1/m": 6, "1": 6, "deg": 4, "degC": 2, "rpm": 2,
            "1/s": 3, "h": 6, "s": 3}
FIDUCIAL_PATH = os.path.join(REPO, "data", "dc2", "fiducial.py")


def _load_fiducial():
    spec = importlib.util.spec_from_file_location("dc2_fiducial", FIDUCIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fiducial = _load_fiducial()


def _hex(text, n=64):
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:n]


def check_out(path):
    real = os.path.realpath(path)
    probe = real if real.endswith("/") else real + "/"
    if any(probe.startswith(r) for r in FORBIDDEN_ROOTS) or not any(probe.startswith(r) for r in WRITE_ROOTS):
        raise ValueError("refusing output %r: only /mnt/e/... or /dev/shm/... (never /mnt/c or /tmp)" % path)
    return real


# ------------------------------------------------------------------------------------------------
# Specs and routes
# ------------------------------------------------------------------------------------------------
SPECS = {
    # 13 s: 2 s warm-up, two 5 s mode-E episodes on the S-curve with a teleport back to its start between them,
    # 0.5 s tail. Every stream gets a second chunk (10 s rule); the render frame crosses 2^18 (fiducial unwrap).
    "golden": {
        "session_id": "arland_T0b_20261025T091512Z_wb_i00_5e00", "trial_id": "T0b", "seed": 0,
        "start_utc": (2026, 10, 25, 9, 15, 12), "render_frame0": 261900, "lead_frames": 2,
        "warmup_steps": 120, "route": "golden_s_curve_v1", "v_ref": 8.0,
        "episodes": [{"max_steps": 300}, {"max_steps": 300}], "between_steps": 30, "teleport_offset": 15,
        "tail_steps": 30, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": True,
    },
    # The golden session at the TBD-2 5 Hz fallback: image_hz = decision_hz = 5, N = 12, k = 3 (generated in the
    # tests, not committed).
    "golden_5hz": {
        "session_id": "arland_T0b_20261025T093000Z_wb_i00_5e05", "trial_id": "T0b", "seed": 0, "decision_hz": 5.0,
        "start_utc": (2026, 10, 25, 9, 30, 0), "render_frame0": 261900, "lead_frames": 2,
        "warmup_steps": 120, "route": "golden_s_curve_v1", "v_ref": 8.0,
        "episodes": [{"max_steps": 300}, {"max_steps": 300}], "between_steps": 30, "teleport_offset": 15,
        "tail_steps": 30, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": True,
    },
    # 108 s: the 60 s C17 warm-up standing, then one mode-E lap of the synthetic loop (trial T1, so C17 is SOFT).
    "t0_loop": {
        "session_id": "arland_T1_20261026T101500Z_wb_i00_c17a", "trial_id": "T1", "seed": 1,
        "start_utc": (2026, 10, 26, 10, 15, 0), "render_frame0": 40000, "lead_frames": 2,
        "warmup_steps": 3600, "route": "t0_loop_synthetic_v1", "v_ref": 8.0,
        "episodes": [{"max_steps": 3000}], "between_steps": 0, "teleport_offset": None,
        "tail_steps": 30, "image_w": 256, "image_h": 240, "band_px": 64, "fill_route_sha": False,
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


def world_ms(step):
    """World time of physics step p since the session start: (p + 1) slices, integer ms (floor)."""
    return (step + 1) * 1000 // 60


# ------------------------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------------------------
class Sim:
    """Per-step truth of one session (lists indexed by physics step)."""

    def __init__(self):
        self.steps = []         # dicts per step (state after the step, inputs, command, expert, phase)
        self.events = []        # (physics_step, kind, payload)
        self.episodes = []      # dicts
        self.decisions = []     # (P_t, command) of every expert decision, in order (R-01: applies from P_t + 1)


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


def n_per_decision(spec):
    """Physics steps per decision at the spec's decision rate (6 at 10 Hz, 12 at the 5 Hz fallback)."""
    return int(round(1.0 / spec.get("decision_hz", 10.0) / SLICE_S))


def simulate(spec, route):
    sim = Sim()
    n_dec = n_per_decision(spec)
    p0 = route.xz[0]
    p1 = route.point_at(1.0)
    psi0 = math.atan2(p1[0] - p0[0], p1[1] - p0[1])
    x, z, psi, v, dist, steer_c = p0[0], p0[1], psi0, 0.0, 0.0, 0.0
    progress = 0.0
    hl_count = -1
    pending = False
    held = None             # (kappa, accel, dir) applied from the step after its decision
    episode = -1
    ep_start = None
    ep_index = 0
    phase = "warmup"
    phase_end = spec["warmup_steps"] - 1
    max_steps = None
    teleport_step = None
    p = 0
    while True:
        if pending:
            hl_count += 1
            pending = False
        if teleport_step is not None and p == teleport_step:
            x, z, psi, v, steer_c, progress = p0[0], p0[1], psi0, 0.0, 0.0, 0.0
            sim.events.append((p - 1, "teleport", "reason=episode_reset"))
        in_ep = phase == "episode"
        cmd = held if in_ep and p > ep_start else None
        if cmd is not None:
            raw_c = max(-1.0, min(1.0, cmd[0] / KAPPA_PER_STEER))
            limited = steer_c + max(-STEER_RATE * SLICE_S, min(STEER_RATE * SLICE_S, raw_c - steer_c))
            steer_c = steer_c + (limited - steer_c) * STEER_ALPHA
            throttle = min(1.0, cmd[1] / 3.0) if cmd[1] >= 0 else 0.0
            brake = min(1.0, -cmd[1] / 6.0) if cmd[1] < 0 else 0.0
        else:
            raw_c = None
            steer_c = 0.0 if not in_ep else steer_c
            throttle = brake = 0.0
        kappa_act = KAPPA_PER_STEER * steer_c
        yaw = v * kappa_act
        acc = 3.0 * throttle - 6.0 * brake - 0.05 * v
        v_new = max(0.0, v + acc * SLICE_S)
        acc_ach = (v_new - v) / SLICE_S
        psi = wrap(psi + yaw * SLICE_S)
        x += v_new * math.sin(psi) * SLICE_S
        z += v_new * math.cos(psi) * SLICE_S
        dist += v_new * SLICE_S
        v_before, v = v, v_new
        exp = None
        if in_ep:
            s_proj, _ = route.project(x, z, progress - 30.0, progress + 30.0)
            progress = max(progress, s_proj)
            exp = expert(route, x, z, psi, v, progress, spec["v_ref"])
        gear = GEAR_FORWARD if in_ep else GEAR_IDLE
        step = {"p": p, "episode": episode if in_ep else -1, "mode": "E" if in_ep else "I",
                "driver": "expert" if in_ep else "none", "hl_seq": hl_count, "cmd": cmd, "exp": exp,
                "raw_c": raw_c, "lim_c": steer_c if cmd is not None else None, "steer_in_c": steer_c,
                "throttle": throttle, "brake": brake, "llc_on": cmd is not None,
                "llc_gear": GEAR_FORWARD if (in_ep and p == ep_start + 1) else None, "gear": gear,
                "handbrake": 1 if (not in_ep and v_before < 0.1) else 0,
                "x": x, "z": z, "psi": psi, "v": v, "dist": dist, "yaw": yaw, "acc": acc_ach,
                "progress": progress if in_ep else None, "world": world_ms(p)}
        sim.steps.append(step)
        # decision in EOnPostSimulate of P_t (R-01): applies from P_t + 1
        if in_ep and p % n_dec == 0:
            complete = route.length_m - progress <= 1.5 and v < 0.5
            if complete or (p - ep_start) >= max_steps:
                term = "route_complete" if complete else "timeout"
                _end_episode(sim, spec, route, episode, ep_start, p, term)
                held = None
                phase = "between" if ep_index < len(spec["episodes"]) else "tail"
                phase_end = p + (spec["between_steps"] if phase == "between" else spec["tail_steps"]) - 1
                if phase == "between" and spec["teleport_offset"] is not None:
                    teleport_step = p + spec["teleport_offset"]
            else:
                held = exp
                pending = True
                sim.decisions.append((p, exp))
        elif phase == "warmup" and p == phase_end:
            sim.events.append((p, "warmup_end", ""))
            phase = "between"
            phase_end = p
        if phase == "between" and p >= phase_end and (p + 1) % n_dec == 0:
            phase = "episode"
            episode = ep_index
            ep_index += 1
            ep_start = p + 1
            max_steps = spec["episodes"][episode]["max_steps"]
            progress = 0.0
            sim.events.append((p + 1, "episode_start", "route_id=%s" % route.route_id))
        if phase == "tail" and p >= phase_end:
            break
        p += 1
    return sim


def _end_episode(sim, spec, route, episode, start, end, term):
    sim.events.append((end, "episode_end", "termination_reason=%s" % term))
    sim.episodes.append({"episode_id": episode, "start": start, "end": end, "termination": term})


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
    az = step["psi"] + u * RAD_PER_PX
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
def frames_of(spec, sim):
    """hf frames: lead frames before the first step (last_physics_step -1), then one frame per physics step."""
    lead = spec["lead_frames"]
    last = sim.steps[-1]["p"]
    return [{"index": f, "render_frame": spec["render_frame0"] + f, "lps": f - lead if f >= lead else -1}
            for f in range(last + lead + 1)]


def image_slots(spec, sim, frames):
    """Slot grid at 10 Hz phase-locked to physics step 0 (img_lead_ms 0); issue in the first frame at or past the
    due time; the screenshot shows the next frame (capture offset 1) and resolves one frame later."""
    anchor = world_ms(0)
    period_ms = int(round(1000.0 / spec.get("decision_hz", 10.0)))
    slots = []
    i = 0
    g = 0
    while True:
        due = anchor + g * period_ms
        while i < len(frames) and (frames[i]["lps"] < 0 or world_ms(frames[i]["lps"]) < due):
            i += 1
        if i + 2 >= len(frames):
            break
        slots.append({"slot_id": g, "due": due, "issue": frames[i], "captured": frames[i + 1]})
        g += 1
    return slots


def sun_dir():
    elev = math.radians(45.0)
    return (0.3 * math.cos(elev), math.sin(elev), 0.95 * math.cos(elev))


def build_rows(spec, sim, route, frames, slots, schema, variant):
    """{stream: [row dict by contract column]} with raw values."""
    ego_slot = 1
    entity = "0x00010000A1"
    entity_b = "0x00010000B7"
    rows = {name: [] for name in schema["streams"]}
    steps = sim.steps
    for st in steps:
        cmd, exp = st["cmd"], st["exp"]
        lim_e = None if st["lim_c"] is None else STEER_SIGN * st["lim_c"]
        raw_e = None if st["raw_c"] is None else STEER_SIGN * st["raw_c"]
        in_e = STEER_SIGN * st["steer_in_c"]
        ctl = {"physics_step": st["p"], "world_time_ms": st["world"], "episode_id": st["episode"],
               "mode": st["mode"], "driver": st["driver"], "hl_seq": st["hl_seq"],
               "hl_kappa": None if cmd is None else cmd[0], "hl_accel": None if cmd is None else cmd[1],
               "hl_dir": None if cmd is None else cmd[2],
               "exp_kappa_now": None if exp is None else exp[0], "exp_accel_now": None if exp is None else exp[1],
               "exp_dir_now": None if exp is None else exp[2],
               "llc_steer_raw": raw_e, "llc_steer_lim": lim_e,
               "llc_throttle": st["throttle"] if st["llc_on"] else None,
               "llc_brake": st["brake"] if st["llc_on"] else None,
               "llc_handbrake": 0 if st["llc_on"] else None, "llc_gear": st["llc_gear"], "llc_clutch": None,
               "noise_active": 0, "noise_id": -1, "steer_signal_raw": None, "wheel_angle_l_rad": None,
               "wheel_angle_r_rad": None, "react_veh_slot_id": None, "react_visible": None}
        for when in ("pre", "post"):
            ctl.update({"input_%s_steer" % when: in_e, "input_%s_throttle" % when: st["throttle"],
                        "input_%s_brake" % when: st["brake"], "input_%s_clutch" % when: 0.0,
                        "input_%s_gear" % when: st["gear"], "input_%s_handbrake" % when: st["handbrake"]})
        rows["ctl"].append(ctl)
        if st["p"] % 3 == 0:
            rows["telemetry"].append(_telemetry_row(st, route, ego_slot,
                                                    entity_b if (variant == "reused_slot" and st["p"] >= 450)
                                                    else entity))
    for fr in frames:
        st = steps[fr["lps"]] if fr["lps"] >= 0 else None
        rows["hf"].append(_hf_row(fr, st, steps[0]))
    for sl in slots:
        issue = sl["issue"]
        st = steps[issue["lps"]]
        rows["img"].append({"slot_id": sl["slot_id"], "episode_id": st["episode"], "ego_slot_id": ego_slot,
                            "requested_world_ms": sl["due"], "issue_status": "issued",
                            "issue_world_ms": st["world"], "issue_render_frame": issue["render_frame"],
                            "issue_physics_step": issue["lps"], "path": "frames/f_%08d" % sl["slot_id"],
                            "cam_current": 1, "cam_attached": 1})
    lead = spec["lead_frames"]
    events = [(-1, "session_start", "")] + sim.events
    for seq, (pstep, kind, payload) in enumerate(events):
        episode = steps[pstep]["episode"] if pstep >= 0 else -1
        rows["events"].append({"event_seq": seq, "episode_id": episode, "physics_step": pstep,
                               "render_frame": spec["render_frame0"] + (pstep + lead if pstep >= 0 else 0),
                               "world_time_ms": world_ms(pstep) if pstep >= 0 else 0, "kind": kind,
                               "veh_slot_id": ego_slot, "payload": payload})
    for ep in sim.episodes:
        reused = variant == "reused_slot" and ep["start"] >= 450
        rows["episodes"].append({
            "episode_id": ep["episode_id"], "start_physics_step": ep["start"], "end_physics_step": ep["end"],
            "start_world_ms": world_ms(ep["start"]), "end_world_ms": world_ms(ep["end"]), "ego_slot_id": ego_slot,
            "ego_entity_id": entity_b if reused else entity, "ego_prefab": "synthetic/S105.et",
            "vehicle_card_id": "S105_synthetic", "route_id": route.route_id,
            "route_sha256": route.sha256 if spec["fill_route_sha"] else None, "mode": "E", "driver": "expert",
            "style_speed_factor": 1.0, "style_lane_offset_m": 0.0, "style_alat_max_mps2": 2.75,
            "env_weather_state": "Clear", "env_tod_h_start": 12.0, "env_rain": 0.0, "env_fog": 0.0,
            "env_wetness": 0.0, "seed": spec["seed"] * 100 + ep["episode_id"],
            "termination_reason": ep["termination"], "expert_sha": _hex("synthetic expert"),
            "llc_sha": _hex("synthetic llc"), "nav_version": "nav_v1", "noise_steer_sensitivity": 1.0,
            "roughness_sensitivity": 1.0, "veh_dirt_body": None, "veh_dirt_glass": None})
    if variant == "gap":
        rows["ctl"] = [r for r in rows["ctl"] if r["physics_step"] not in (304, 305)]
    for name, stream_rows in rows.items():
        cols = [f["name"] for f in schema["streams"][name]["fields"]]
        for r in stream_rows:
            if sorted(r) != sorted(cols):
                raise AssertionError("%s row keys differ from the streams JSON: %s"
                                     % (name, sorted(set(r) ^ set(cols))))
    return rows


def _telemetry_row(st, route, ego_slot, entity):
    psi = st["psi"]
    fwd = (math.sin(psi), 0.0, math.cos(psi))
    right = (math.cos(psi), 0.0, -math.sin(psi))
    lane_lat = lane_head = None
    if st["progress"] is not None:
        s, _ = route.project(st["x"], st["z"], st["progress"] - 30.0, st["progress"] + 30.0)
        a = route.point_at(s - 0.5)
        b = route.point_at(s + 0.5)
        tangent = math.atan2(b[0] - a[0], b[1] - a[1])
        px, pz = route.point_at(s)
        _, lane_lat = drivedit_nav.to_ego(drivedit_nav.Pose(px, pz, tangent), st["x"], st["z"])
        lane_head = wrap(psi - tangent)
    sun = sun_dir()
    row = {"physics_step": st["p"], "world_time_ms": st["world"], "episode_id": st["episode"],
           "ego_slot_id": ego_slot, "ego_entity_id": entity, "pos_x": st["x"], "pos_y": 30.0, "pos_z": st["z"],
           "right_x": right[0], "right_y": right[1], "right_z": right[2], "up_x": 0.0, "up_y": 1.0, "up_z": 0.0,
           "fwd_x": fwd[0], "fwd_y": fwd[1], "fwd_z": fwd[2], "vel_local_x": 0.0, "vel_local_y": 0.0,
           "vel_local_z": st["v"], "angvel_local_x": 0.0, "angvel_local_y": YAW_AXIS_SIGN * st["yaw"],
           "angvel_local_z": 0.0, "acc_local_x": st["v"] * st["yaw"], "acc_local_y": 0.0, "acc_local_z": st["acc"],
           "speed_engine_mps": st["v"], "gear": st["gear"], "rpm": 800.0 + 120.0 * st["v"], "engine_on": 1,
           "handbrake": st["handbrake"], "ground_contact": 1, "wheel_count": 4,
           "road_key": "12/1000.00:3000.00/1003.40:3068.10", "road_dist_m": 0.4, "road_width_m": 7.0,
           "lane_lat_err_m": lane_lat, "lane_head_err_rad": lane_head, "route_progress_m": st["progress"],
           "env_weather_state": "Clear", "env_rain": 0.0, "env_fog": 0.0, "env_fog_height_density": 0.0,
           "env_wetness": 0.0, "env_tod_h": 12.0 + st["world"] / 3600000.0, "sun_dir_x": sun[0],
           "sun_dir_y": sun[1], "sun_dir_z": sun[2], "moon_phase01": 0.5, "sun_elev_deg": 45.0, "traffic_n": 0,
           "traffic_slots": None}
    for w in range(4):
        row.update({"w%d_contact" % w: 1, "w%d_slip_long" % w: 0.01 * st["acc"] / 3.0,
                    "w%d_slip_lat" % w: 0.002 * st["v"] * st["yaw"], "w%d_material" % w: "Asphalt",
                    "w%d_temp_c" % w: None})
    return row


def _hf_row(fr, st, st0):
    # before the first step (last_physics_step -1) the ego stands at its spawn pose, which step 0 (standing) keeps
    src = st if st is not None else {"x": st0["x"], "z": st0["z"], "psi": st0["psi"], "world": 0}
    psi = src["psi"]
    fwd = (math.sin(psi), 0.0, math.cos(psi))
    right = (math.cos(psi), 0.0, -math.sin(psi))
    pitch = math.radians(-4.0)
    cam_fwd = (fwd[0] * math.cos(pitch), math.sin(pitch), fwd[2] * math.cos(pitch))
    cam_up = (-fwd[0] * math.sin(pitch), math.cos(pitch), -fwd[2] * math.sin(pitch))
    row = {"render_frame": fr["render_frame"], "world_time_ms": src["world"],
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


def session_json(spec, start_unix, card_sha):
    sid = spec["session_id"]
    launch_id = "%s-wb-%s" % (_iso(start_unix - 60).replace("-", "").replace(":", ""), sid[-4:])
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
                  "conf": {"trial_id": spec["trial_id"], "route": spec["route"], "seed": spec["seed"],
                           "warmup_s": spec["warmup_steps"] / 60.0, "synthetic": True},
                  "seed": spec["seed"], "warmup_s": spec["warmup_steps"] / 60.0, "duration_s": 0.0},
        "exe": {"kind": "wb", "instance": "i00", "cli": {"dcLaunchId": launch_id, "dcTrial": spec["trial_id"]}},
        "engine": {"build_version": "1.8.0.13", "build_time": "synthetic", "fixed_time_slice_s": SLICE_S},
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
        "clocks": {"telemetry_every_k": 3, "physics_steps_per_decision": n_per_decision(spec),
                   "decision_hz": spec.get("decision_hz", 10.0), "image_hz": spec.get("decision_hz", 10.0),
                   "img_inflight_timeout_ms": 2000, "img_lead_ms": 0, "chunk_max_s": 10.0},
        "rig": {"rig_id": "R1-S105-synthetic-%dx%d" % (w, h), "rig_name": "R1-S105", "ego_prefab": "synthetic/S105.et",
                "mount": {"fwd_m": 1.8, "up_m": 1.5, "right_m": 0.0, "pitch_deg": -4.0}, "vfov_deg": 50.0,
                "writer": "MakeScreenshot", "source_class": "wb_viewport", "capture_wh": [w, h],
                "core_wh": [w - 2 * band, h], "band_px": band, "derivative_wh": [512, 288],
                "derivative_codec": "jpeg_q95",
                "fiducial": {"version": "v2", "bits": 18, "patches": 6, "bits_per_patch": 3, "patch_px": 32,
                             "mirrored": True},
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
    observed = {"game": {"appid": 1874880, "buildid": pin["game_buildid"], "target_buildid": pin["game_buildid"],
                         "auto_update_behavior": "1"},
                "tools": {"appid": 1874910, "buildid": pin["tools_buildid"], "target_buildid": pin["tools_buildid"],
                          "auto_update_behavior": "1"}}
    return {"schema": pin["launch_schema"], "launch_id": pin["launch_id"], "status": "ok",
            "utc": meta["created_utc"], "exe": "wb", "instance": None,
            "pin": {"path": "synthetic/pin_%s.json" % pin["pin_id"], "pin_id": pin["pin_id"],
                    "sha256": pin["pin_sha256"], "resolved_by": "active-pointer", "pointer": "synthetic/ACTIVE"},
            "observed": observed,
            "checks": [{"name": "%s.%s" % (app, c), "ok": True, "detail": "synthetic"}
                       for app in ("game", "tools") for c in PREFLIGHT_CHECKS],
            "host": {"hostname": "synthetic", "wsl_distro": None, "python": "synthetic"},
            "argv": ["--pin-active", "--exe", "wb"]}


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
    with open(STREAMS_JSON, "rb") as f:
        schema = json.loads(f.read().decode("ascii"))
    route_obj = route_obj or ROUTES[spec["route"]]()
    route = drivedit_nav.route_from_dict(route_obj)
    session_dir = os.path.join(out_dir, spec["session_id"])
    if os.path.exists(session_dir):
        raise ValueError("refusing to overwrite %s" % session_dir)
    os.makedirs(session_dir)
    sim = simulate(spec, route)
    frames = frames_of(spec, sim)
    slots = image_slots(spec, sim, frames)
    rows = build_rows(spec, sim, route, frames, slots, schema, variant)
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
    # images
    os.makedirs(os.path.join(session_dir, "frames"))
    pixels = {}
    pixel_slots = _pixel_slots(len(slots))
    for sl in slots:
        st = sim.steps[sl["captured"]["lps"]]
        img = render_image(spec, st, fiducial.code_for(sl["captured"]["render_frame"]))
        with open(os.path.join(session_dir, "frames", "f_%08d.png" % sl["slot_id"]), "wb") as f:
            f.write(png_bytes(img))
        if sl["slot_id"] in pixel_slots:
            pixels[sl["slot_id"]] = img
    # heartbeat (every 5 s of world time)
    hb_lines = ["unix_s,world_ms,physics_step,render_frame,fps,mem_kb,img_ok,img_fail"]
    n_hb = 0
    for st in sim.steps:
        if st["world"] // 5000 > world_ms(st["p"] - 1) // 5000 and st["p"] > 0:
            ok = sum(1 for sl in slots if sl["captured"]["index"] + 1 <= st["p"] + spec["lead_frames"])
            hb_lines.append("%d,%d,%d,%d,%.3f,%d,%d,%d" % (start_unix + st["world"] // 1000, st["world"], st["p"],
                                                          spec["render_frame0"] + st["p"] + spec["lead_frames"],
                                                          60.0, 1800000 + 2 * (st["world"] // 1000), ok, 0))
            n_hb += 1
    with open(os.path.join(session_dir, "heartbeat.csv"), "wb") as f:
        f.write(("\n".join(hb_lines) + "\n").encode("ascii"))
    meta = session_json(spec, start_unix, hashlib.sha256(card_bytes).hexdigest())
    meta["trial"]["duration_s"] = round((sim.steps[-1]["p"] + 1 - spec["warmup_steps"]) / 60.0, 3)
    last = sim.steps[-1]
    end_unix = start_unix + last["world"] // 1000 + 1
    end = {"contract_version": "dc-2.0", "session_id": spec["session_id"], "status": "ok", "end_reason": "duration",
           "end": {"unix_s": end_unix, "utc": _iso(end_unix), "world_ms": last["world"], "physics_step": last["p"],
                   "render_frame": frames[-1]["render_frame"]},
           "pin": {"pin_id": "1.8.0.13", "build_key": None}, "engine": {"build_version": "1.8.0.13"},
           "counts": counts,
           "images": {"requested": len(slots), "issued": len(slots), "skipped_busy": 0, "make_failed": 0,
                      "timed_out": 0, "pending_at_end": 0},
           "episodes": len(sim.episodes), "heartbeats": n_hb}
    with open(os.path.join(session_dir, "session.json"), "wb") as f:
        f.write(_json_text(meta).encode("ascii"))
    with open(os.path.join(session_dir, "harvest.json"), "wb") as f:
        f.write(_json_text({"schema": "drivedit.harvest/0-synthetic", "session_id": spec["session_id"],
                            "build_key": _hex("synthetic build_key 1.8.0.13"),
                            "note": "DD-6-min writes harvest.json; this synthetic one carries only build_key"}
                           ).encode("ascii"))
    _write_logs(session_dir, spec, start_unix, slots)
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
        golden = golden_truth(spec, sim, quantised, slots, card, pixels)
        with open(check_out(golden_json), "wb") as f:
            f.write((json.dumps(golden, indent=1, sort_keys=True) + "\n").encode("ascii"))
    return {"session_dir": session_dir, "session_id": spec["session_id"], "route": route_obj, "golden": golden}


def _pixel_slots(n):
    return sorted(set([0, n // 3, (2 * n) // 3, n - 1]))


def _write_logs(session_dir, spec, start_unix, slots):
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
    lines.append("%s SCRIPT       : [DC] session_end reason=duration" % hms(slots[-1]["due"] + 200))
    with open(os.path.join(ldir, "console.log"), "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))


# ------------------------------------------------------------------------------------------------
# golden.json (expected reader output, from the simulation)
# ------------------------------------------------------------------------------------------------
def _r6(values):
    return None if values is None else [None if v is None else (round(v, 6) if isinstance(v, float) else v)
                                        for v in values]


def golden_truth(spec, sim, q, slots, card, pixels):
    """Expected reader output. The R-01 samples come from the simulation's decision events (sim.decisions: P_t and
    the command decided after step P_t), not from the ctl rows: action_cmd_t and hl_seq are that command and its
    index, action_label_t the expert evaluation after step P_t, and the window is held when the simulation applied
    that command in each of P_t + 1 .. P_t + N. State, exec and obs values are the values as written (q)."""
    n, k = n_per_decision(spec), 3
    period = 1000.0 / spec.get("decision_hz", 10.0)
    steps = sim.steps
    last = steps[-1]["p"]
    ctl = {r["physics_step"]: r for r in q["ctl"]}
    tel = {r["physics_step"]: r for r in q["telemetry"]}
    forward = set(card["gear_indices_forward"])
    decided = {p: (i, cmd) for i, (p, cmd) in enumerate(sim.decisions)}
    decision_steps = [p for p, _ in sim.decisions]
    images = []
    cands = []
    for sl in slots:
        cf = sl["captured"]
        images.append({"slot_id": sl["slot_id"], "file": "frames/f_%08d.png" % sl["slot_id"], "file_exists": 1,
                       "png_w": spec["image_w"], "png_h": spec["image_h"], "fid_ok": 1,
                       "captured_render_frame": cf["render_frame"]})
        if cf["lps"] >= 0:      # a frame before the first step joins neither ctl nor telemetry
            cands.append((cf["render_frame"], cf["lps"], world_ms(cf["lps"]), sl["slot_id"],
                          steps[cf["lps"]]["episode"]))
    warmup_end = next(world_ms(p) for p, kind, _ in sim.events if kind == "warmup_end")
    teleports = [p for p, kind, _ in sim.events if kind in ("teleport", "respawn")]
    decisions = []
    for p in range(0, last + 1, n):
        st, t0 = steps[p], tel[p]
        world, episode = st["world"], st["episode"]
        window = list(range(p + 1, p + n + 1))
        has_next = p + 1 <= last
        complete = p + n <= last
        one_ep = complete and episode >= 0 and all(steps[j]["episode"] == episode for j in window)
        state = [t0["vel_local_z"], YAW_AXIS_SIGN * t0["angvel_local_y"], t0["acc_local_z"],
                 1 if t0["gear"] in forward else 0]
        label = _r6(list(st["exp"])) if st["exp"] is not None else [None, None, None]
        if p in decided:
            hl_seq, cmd = decided[p][0], _r6(list(decided[p][1]))
        else:
            hl_seq = bisect.bisect_right(decision_steps, p) - 1 if has_next else None
            cmd = [None, None, None] if has_next else None
        held = p in decided and complete and all(
            steps[j]["cmd"] == decided[p][1] and steps[j]["hl_seq"] == decided[p][0] for j in window)
        exec_ok = complete and all(steps[j]["llc_on"] for j in window)
        exec_rows = [None if ctl.get(j) is None else
                     [None if ctl[j]["llc_steer_lim"] is None else STEER_SIGN * ctl[j]["llc_steer_lim"],
                      ctl[j]["llc_throttle"], ctl[j]["llc_brake"], STEER_SIGN * ctl[j]["input_post_steer"]]
                     for j in window]
        eligible = [c for c in cands if c[1] <= p]
        obs = max(eligible) if eligible else None
        obs_valid = 1 if obs is not None and world - obs[2] <= period else 0
        same_ep = obs_valid == 1 and obs[4] == episode
        first_q = obs[1] if obs_valid else p
        teleport = 1 if any(first_q <= t <= p + n for t in teleports) else 0
        mode = steps[p + 1]["mode"] if has_next else st["mode"]
        wm_valid = int(complete and one_ep and held and exec_ok and obs_valid == 1 and same_ep and not teleport)
        policy_valid = int(episode >= 0 and None not in label and mode not in ("X", "A", "I") and obs_valid == 1
                           and same_ep and not teleport)
        decisions.append({
            "decision_step": p, "episode_id": episode, "hl_seq": hl_seq,
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
        if ok and cur is not None and cur["episode_id"] == d["episode_id"] \
                and d["decision_step"] == cur["last_decision_step"] + n:
            cur["last_decision_step"] = d["decision_step"]
            cur["n_steps"] += 1
            continue
        if cur is not None:
            segments.append(cur)
            cur = None
        if ok:
            cur = {"episode_id": d["episode_id"], "first_decision_step": d["decision_step"],
                   "last_decision_step": d["decision_step"], "n_steps": 1}
    if cur is not None:
        segments.append(cur)
    band = spec["band_px"]
    stride = 16
    pix = []
    for slot in sorted(pixels):
        core = pixels[slot][:, band:spec["image_w"] - band, :].transpose(2, 0, 1)[:, ::stride, ::stride]
        pix.append({"slot_id": slot, "core": numpy.round(core.astype(float) / 255.0, 6).tolist()})
    return {"schema": "drivedit.dc2-golden/1", "session_id": spec["session_id"], "contract_version": "dc-2.0",
            "clocks": {"telemetry_every_k": k, "physics_steps_per_decision": n,
                       "decision_hz": spec.get("decision_hz", 10.0), "image_hz": spec.get("decision_hz", 10.0),
                       "img_lead_ms": 0},
            "counts": {name: len(rows) for name, rows in sorted(q.items())},
            "img_resolved": images, "segments": segments, "decision_steps": decisions,
            "pixels": {"stride": stride, "slots": pix}}


# ------------------------------------------------------------------------------------------------
# dc-1.0 legacy session (the legacy self-test and the enfusion_preprocessing guard test)
# ------------------------------------------------------------------------------------------------
LEGACY_WRITER = os.path.join(REPO, "src", "enfusion", "datacapture", "SCR_MLDataCollector.c")


def legacy_header():
    """The dc-1.0 telemetry.csv columns, read from the writer's WriteTelemetryHeader (SCR_MLDataCollector.c), so the
    legacy fixture never invents a schema (CR-109)."""
    with open(LEGACY_WRITER, "rb") as f:
        src = f.read().decode("utf-8", errors="replace")
    start = src.index("bool WriteTelemetryHeader()")
    stop = src.index("WriteLine(header)", start)
    text = ""
    for line in src[start:stop].splitlines():
        line = line.strip()
        if line.startswith("string header =") or line.startswith("header +="):
            text += line.split('"')[1]
    return text.split(",")


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
    ap.add_argument("--out", required=True, help="parent directory of the new session (under /mnt/e or /dev/shm)")
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
