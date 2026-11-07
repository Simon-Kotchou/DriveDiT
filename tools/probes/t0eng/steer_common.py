#!/usr/bin/env python3
"""Shared readers and math of the G19-1 steering analyses (DD-14b): e21_selfsteer.py (CC-07) and
e23_steer_steps.py (CC-08). Not a script itself; stdlib only, imported after probe_common like every t0eng tool.

What the steering probes write (src/enfusion/datacapture/DC_ProbeSteerBase.c): <probe>_steps.csv with one row per
physics step (the DC_ProbeActuator columns, then steer_call, assist_mode, site, along_m, lateral_m, in_area,
seg_target, seg_flag, event_idx and the DC_SteerChannels wheel-angle candidates), <probe>_phases.csv, e23_events.csv,
and run header lines steer.*, site.<kind>.*, assist.*.

Conventions (DC_SteerChannels.c): wheel angles in rad in the vehicle frame, positive = the wheel points to the right
(clockwise from above), the sign of heading_deg and of a right-turn yaw rate; positive steering turns right on both
builds (e01 / e02a steer_sign_used=1). kappa_realised = angvel_l_y / vel_l_z [1/m], a_lat = vel_l_z x angvel_l_y.

delta_eff, the bicycle-model equivalent of the two front wheel angles: for an ideal Ackermann pair the turn centre
on the rear axle line gives cot(delta_eff) = (cot(delta_l) + cot(delta_r)) / 2, i.e. tan(delta_eff) = the harmonic
mean of the two tangents. That form is used when both wheels point the same way by at least HARMONIC_MIN_TAN; near
zero (where a toe offset would make the harmonic mean meaningless) the arithmetic mean of the tangents is used. The
two agree to < 0.3% at the step-0 angles (<= 0.15 rad).
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402

HARMONIC_MIN_TAN = 0.02
MIN_SPEED_MS = 1.0
# The wheel-angle candidates in the order step 0 tries them (DC_SteerChannels columns)
CHANNELS = (
    ("rot", "wa_rot_l", "wa_rot_r"),
    ("bone", "wa_bone_l", "wa_bone_r"),
    ("ent", "wa_ent_l", "wa_ent_r"),
    ("sig", "sig_steer", None),
)
# Columns every steering-probe step CSV must have (the analyses refuse a play without them)
STEP_COLUMNS = ("physics_step", "timeslice_s", "phase", "segment", "phase_step", "run_idx", "cmd_steer",
                "input_pre_steer", "input_post_steer", "speed_kmh", "vel_l_z", "angvel_l_y", "acc_l_z",
                "heading_deg", "pos_x", "pos_z", "engine_on", "steer_call", "assist_mode", "site", "along_m",
                "lateral_m", "in_area", "seg_target", "seg_flag", "event_idx", "wa_rot_l", "wa_rot_r", "wa_bone_l",
                "wa_bone_r", "wa_ent_l", "wa_ent_r", "sig_steer", "wa_rot_l_pf", "wa_rot_r_pf", "slip_lat_l",
                "slip_lat_r", "vdir_l", "vdir_r")
FLOAT_COLUMNS = ("timeslice_s", "cmd_steer", "input_pre_steer", "input_post_steer", "speed_kmh", "vel_l_z",
                 "vel_l_x", "angvel_l_y", "acc_l_z", "acc_l_x", "heading_deg", "pos_x", "pos_z", "along_m",
                 "lateral_m", "seg_target", "wa_rot_l", "wa_rot_r", "wa_bone_l", "wa_bone_r", "wa_ent_l", "wa_ent_r",
                 "sig_steer", "wa_rot_l_pf", "wa_rot_r_pf", "slip_lat_l", "slip_lat_r", "vdir_l", "vdir_r",
                 "speed_hold_kmh")
INT_COLUMNS = ("physics_step", "phase_step", "run_idx", "engine_on", "steer_call", "assist_mode", "in_area",
               "seg_flag", "event_idx")


class Step:
    """One step row with typed fields (floats / ints, None for an empty field) and the derived kappa and a_lat."""

    __slots__ = ("raw", "phase", "segment", "site", "f", "i", "kappa", "a_lat")

    def __init__(self, raw):
        self.raw = raw
        self.phase = raw.get("phase", "")
        self.segment = raw.get("segment", "")
        self.site = raw.get("site", "")
        self.f = {name: probe_common.to_float(raw.get(name)) for name in FLOAT_COLUMNS}
        self.i = {name: probe_common.to_int(raw.get(name)) for name in INT_COLUMNS}
        v = self.f.get("vel_l_z")
        r = self.f.get("angvel_l_y")
        self.kappa = r / v if v is not None and r is not None and v >= MIN_SPEED_MS else None
        self.a_lat = v * r if v is not None and r is not None else None

    def num(self, name):
        if name in self.f:
            return self.f[name]
        return self.i.get(name)


def load_steps(play, probe):
    """<probe>_steps.csv of play as Step objects (InputError when the file or a required column is missing)."""
    rows = play.csv(probe + "_steps", STEP_COLUMNS)
    return [Step(r) for r in rows]


def header_float(play, key, default=None):
    """A run header value as float (written directly or as params.<key>)."""
    value = play.header.get(key)
    if value is None:
        value = play.header.get("params." + key)
    return probe_common.to_float(value, default)


def wheelbase(play):
    """steer.wheelbase_m of the run header, else from the steer.wheel.<i>.pos lines; None when neither exists."""
    value = header_float(play, "steer.wheelbase_m")
    if value is not None and value > 0.5:
        return value
    front_l = probe_common.to_int(play.header.get("steer.wheel_index_l"), 0)
    front_r = probe_common.to_int(play.header.get("steer.wheel_index_r"), 1)
    count = probe_common.to_int(play.header.get("steer.wheel_count"), 0)
    front, rear = [], []
    for i in range(count):
        text = play.header.get("steer.wheel.%d.pos" % i, "")
        parts = text.split()
        if len(parts) != 3:
            continue
        z = probe_common.to_float(parts[2])
        if z is None:
            continue
        (front if i in (front_l, front_r) else rear).append(z)
    if len(front) == 2 and rear:
        return sum(front) / 2.0 - sum(rear) / len(rear)
    return None


def delta_eff(left, right):
    """Bicycle-model equivalent angle of two front wheel angles [rad] (module docstring); None without both."""
    if left is None or right is None:
        return None
    tl, tr = math.tan(left), math.tan(right)
    if tl * tr > 0 and min(abs(tl), abs(tr)) >= HARMONIC_MIN_TAN:
        return math.atan(2.0 * tl * tr / (tl + tr))
    return math.atan((tl + tr) / 2.0)


def channel_angle(step, channel, signal_scale_rad=None):
    """delta_eff of a channel on a step [rad]: the L/R pair of a bone or entity source, or the steering signal x
    signal_scale_rad (None without a scale)."""
    for key, left, right in CHANNELS:
        if key != channel:
            continue
        if right is None:
            sig = step.f.get(left)
            if sig is None or signal_scale_rad is None:
                return None
            return sig * signal_scale_rad
        return delta_eff(step.f.get(left), step.f.get(right))
    raise ValueError("unknown channel %r" % channel)


def channel_pair(step, channel):
    """(left, right) wheel angles of a bone or entity channel, (None, None) for the signal"""
    for key, left, right in CHANNELS:
        if key == channel and right is not None:
            return step.f.get(left), step.f.get(right)
    return None, None


def segments(steps):
    """Contiguous runs of rows with the same (phase, run_idx, segment): list of (phase, run, segment, [steps])."""
    out = []
    for step in steps:
        key = (step.phase, step.i.get("run_idx"), step.segment)
        if out and out[-1][0] == key:
            out[-1][1].append(step)
        else:
            out.append((key, [step]))
    return [(k[0], k[1], k[2], rows) for k, rows in out]


def wrap_deg(delta):
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def path_curvature(rows):
    """(mean curvature [1/m], distance [m], heading change [deg]) over consecutive rows from the unwrapped
    heading_deg and the positions; (None, distance, None) when the rows are too short."""
    distance = 0.0
    turn = 0.0
    prev = None
    for step in rows:
        x, z, h = step.f.get("pos_x"), step.f.get("pos_z"), step.f.get("heading_deg")
        if x is None or z is None or h is None:
            prev = None
            continue
        if prev is not None:
            distance += math.hypot(x - prev[0], z - prev[1])
            turn += wrap_deg(h - prev[2])
        prev = (x, z, h)
    if distance <= 1.0:
        return None, distance, None
    return math.radians(turn) / distance, distance, turn


def median(values):
    return probe_common.percentile(values, 50)


def mean(values):
    return sum(values) / len(values) if values else None


def stdev(values):
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def fmt(value, digits=4, unit=""):
    """A number for a detail line ('-' for None)."""
    if value is None:
        return "-"
    return ("%%.%dg" % digits) % value + unit


def deg(rad):
    return None if rad is None else math.degrees(rad)


def t50_index(values, start, pre, final):
    """First index i >= start with (values[i] - pre) / (final - pre) >= 0.5; None when it never does or the step is
    too small to measure."""
    span = final - pre
    if abs(span) < 1e-9:
        return None
    for i in range(start, len(values)):
        if values[i] is None:
            continue
        if (values[i] - pre) / span >= 0.5:
            return i
    return None
