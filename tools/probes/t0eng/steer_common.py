#!/usr/bin/env python3
"""Shared readers, evidence rules and math of the G19-1 steering analyses (DD-14b): e21_selfsteer.py (CC-07) and
e23_steer_steps.py (CC-08). Not a script itself; stdlib only, imported after probe_common like every t0eng tool.

What the steering probes write (src/enfusion/datacapture/DC_ProbeSteerBase.c): <probe>_steps.csv with one row per
dispatched physics step (the DC_ProbeActuator columns, then steer_call, assist_mode, site, along_m, lateral_m,
in_area, seg_target, seg_flag, event_idx and the DC_SteerChannels wheel-angle candidates), <probe>_phases.csv,
e23_events.csv, and run header lines steer.*, site.<kind>.*, assist.*.

Conventions (DC_SteerChannels.c): wheel angles in rad in the vehicle frame, positive = the wheel points to the right
(clockwise from above), the sign of heading_deg and of a right-turn yaw rate; positive steering turns right on both
builds (e01 / e02a steer_sign_used=1). kappa_realised = angvel_l_y / vel_l_z [1/m], a_lat = vel_l_z x angvel_l_y.

delta_eff, the bicycle-model equivalent of the two front wheel angles: for an ideal Ackermann pair the turn centre
on the rear axle line gives cot(delta_eff) = (cot(delta_l) + cot(delta_r)) / 2, i.e. tan(delta_eff) = the harmonic
mean of the two tangents. That form is used when both wheels point the same way by at least HARMONIC_MIN_TAN; near
zero (where a toe offset would make the harmonic mean meaningless) the arithmetic mean of the tangents is used. The
two agree to < 0.3% at the step-0 angles (<= 0.15 rad).

Evidence rules shared by both analyses (they decide Mode A against Mode B, so missing or partial evidence never
passes and never selects a production mode):
  - every item is PASS, FAIL or INCOMPLETE (with the reason), and completeness is never lost in an aggregation:
    combine() is INCOMPLETE as soon as one part is (a FAIL seen in a part is kept as a diagnostic, observed()), so an
    item is FAIL only when all its parts are decided. StrictVerdict's RESULT is INCOMPLETE while any item is;
  - a play counts only with a complete run footer (play_evidence); an interrupted, failed or missing play makes the
    items that need it INCOMPLETE;
  - identity: every input play must carry engine.build_version, launch.pin_id, ego.prefab, world.map_id and
    addon.git_sha, the same in all of them (launch.build_key too when any play has one); otherwise the analysis
    refuses (InputError, exit 2). A 1.8 baseline is checked on its own (baseline_identity): a 1.8 build with a
    complete footer, the same prefab and map;
  - merged physics steps: a row covers round(timeslice_s / dt_s) engine steps (covered_steps); durations and lags
    use the world time and the covered steps, never row counts.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402

HARMONIC_MIN_TAN = 0.02
MIN_SPEED_MS = 1.0
IDENTITY_KEYS = ("engine.build_version", "launch.pin_id", "ego.prefab", "world.map_id", "addon.git_sha")
# The wheel-angle candidates in the order step 0 tries them (DC_SteerChannels columns)
CHANNELS = (
    ("rot", "wa_rot_l", "wa_rot_r"),
    ("bone", "wa_bone_l", "wa_bone_r"),
    ("ent", "wa_ent_l", "wa_ent_r"),
    ("sig", "sig_steer", None),
)
# Columns every steering-probe step CSV must have (the analyses refuse a play without them)
STEP_COLUMNS = ("physics_step", "world_time_ms", "timeslice_s", "phase", "segment", "phase_step", "run_idx",
                "cmd_steer", "input_pre_steer", "input_post_steer", "speed_kmh", "vel_l_z", "angvel_l_y", "acc_l_z",
                "heading_deg", "pos_x", "pos_z", "engine_on", "steer_call", "assist_mode", "site", "along_m",
                "lateral_m", "in_area", "seg_target", "seg_flag", "event_idx", "wa_rot_l", "wa_rot_r", "wa_bone_l",
                "wa_bone_r", "wa_ent_l", "wa_ent_r", "sig_steer", "wa_rot_l_pf", "wa_rot_r_pf", "slip_lat_l",
                "slip_lat_r", "vdir_l", "vdir_r")
FLOAT_COLUMNS = ("world_time_ms", "timeslice_s", "cmd_steer", "input_pre_steer", "input_post_steer", "speed_kmh",
                 "vel_l_z", "vel_l_x", "angvel_l_y", "acc_l_z", "acc_l_x", "heading_deg", "pos_x", "pos_z", "along_m",
                 "lateral_m", "seg_target", "wa_rot_l", "wa_rot_r", "wa_bone_l", "wa_bone_r", "wa_ent_l", "wa_ent_r",
                 "sig_steer", "wa_rot_l_pf", "wa_rot_r_pf", "slip_lat_l", "slip_lat_r", "vdir_l", "vdir_r",
                 "speed_hold_kmh")
INT_COLUMNS = ("physics_step", "phase_step", "run_idx", "engine_on", "steer_call", "assist_mode", "in_area",
               "seg_flag", "event_idx")


# ---------------------------------------------------------------------------------------------------------------
# evidence


class StrictVerdict(probe_common.Verdict):
    """probe_common.Verdict whose RESULT is INCOMPLETE while any item is INCOMPLETE (FAIL needs every item decided)."""

    def result(self):
        tags = [t for t, _, _ in self.items]
        if probe_common.INCOMPLETE in tags:
            return probe_common.INCOMPLETE
        if probe_common.FAIL in tags:
            return probe_common.FAIL
        if probe_common.PASS in tags:
            return probe_common.PASS
        return probe_common.INCOMPLETE

    def tag_of(self, name):
        for tag, item, _ in self.items:
            if item == name:
                return tag
        return None


def combine(states):
    """Tri-state aggregate of True / False / None (PASS / FAIL / INCOMPLETE) that never loses completeness: None
    when there is no state or any state is None (even next to a False: a decision needs every part), else False when
    any is False, else True. A False hidden by a None is reported by observed()"""
    if not states or any(s is None for s in states):
        return None
    if any(s is False for s in states):
        return False
    return True


def observed(states):
    """The diagnostic of an incomplete aggregate: ' [observed FAIL in N part(s); withheld: incomplete]' when a part
    failed although the aggregate is INCOMPLETE, else ''"""
    fails = sum(1 for s in states if s is False)
    if fails and combine(states) is None:
        return " [observed FAIL in %d part(s); withheld: incomplete]" % fails
    return ""


def withhold(verdict, names, reason):
    """Every decided item of names becomes INCOMPLETE (its observed tag kept in the detail): the play it depends on
    is not complete evidence"""
    for i, (tag, name, detail) in enumerate(verdict.items):
        if name in names and tag in (probe_common.PASS, probe_common.FAIL):
            verdict.items[i] = (probe_common.INCOMPLETE, name, "%s [observed %s; withheld: %s]" % (detail, tag, reason))


def play_evidence(play, name):
    """(True, detail) for a play with a complete footer, else (None, the reason): the evidence is missing"""
    if play is None:
        return None, "%s: no play given" % name
    if play.footer is None:
        return None, "%s: no run_footer.txt (crash, kill or Stop before the probe ended)" % name
    status = play.footer.get("status", "")
    if status != "complete":
        return None, "%s: footer status %s reason %s" % (name, status or "-", play.footer.get("reason", "") or "-")
    return True, "%s: complete" % name


def identity(play):
    return {key: (play.header.get(key) or "").strip() for key in IDENTITY_KEYS + ("launch.build_key",)}


def check_identity(plays):
    """Refuses (InputError) a play without the identity fields, or plays whose identities differ; returns the
    common identity"""
    first = None
    for name, play in plays:
        if play is None:
            continue
        ident = identity(play)
        missing = [k for k in IDENTITY_KEYS if not ident[k]]
        if missing:
            raise probe_common.InputError("refused: %s (%s) lacks the identity field(s) %s" % (
                name, play.dir, ", ".join(missing)))
        if first is None:
            first = (name, ident)
            continue
        for key in IDENTITY_KEYS + ("launch.build_key",):
            a, b = first[1][key], ident[key]
            if key == "launch.build_key" and not a and not b:
                continue
            if a != b:
                raise probe_common.InputError("refused: %s and %s differ in %s (%r vs %r)" % (first[0], name, key, a, b))
    return first[1] if first else None


def baseline_identity(baseline, current):
    """(True, detail) when the baseline play is a usable 1.8 baseline of the current plays, else (None, reason)"""
    ident = identity(baseline)
    missing = [k for k in IDENTITY_KEYS if not ident[k]]
    if missing:
        return None, "baseline lacks %s" % ", ".join(missing)
    if not ident["engine.build_version"].startswith("1.8"):
        return None, "baseline build %s is not a 1.8 build" % ident["engine.build_version"]
    if current and ident["engine.build_version"] == current["engine.build_version"]:
        return None, "baseline build equals the current build"
    for key in ("ego.prefab", "world.map_id"):
        if current and ident[key] != current[key]:
            return None, "baseline %s %r differs from %r" % (key, ident[key], current[key])
    ok, detail = play_evidence(baseline, "baseline")
    if not ok:
        return None, detail
    return True, "baseline %s, pin %s" % (ident["engine.build_version"], ident["launch.pin_id"])


def phase_results(play, probe):
    """{phase name: result} of <probe>_phases.csv ({} without the file)"""
    if not play.has(probe + "_phases.csv"):
        return {}
    return {row.get("phase", ""): row.get("result", "") for row in play.csv(probe + "_phases", ("phase", "result"))}


# ---------------------------------------------------------------------------------------------------------------
# rows


class Step:
    """One step row with typed fields (floats / ints, None for an empty field) and the derived kappa and a_lat."""

    __slots__ = ("raw", "phase", "segment", "site", "f", "i", "kappa", "a_lat", "index")

    def __init__(self, raw, index=0):
        self.raw = raw
        self.index = index
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
    return [Step(r, k) for k, r in enumerate(rows)]


def header_float(play, key, default=None):
    """A run header value as float (written directly or as params.<key>)."""
    value = play.header.get(key)
    if value is None:
        value = play.header.get("params." + key)
    return probe_common.to_float(value, default)


def step_dt(play):
    """The runner's physics step [s] (params.dt_s, else clock.step_dt_s); None when neither is a positive number"""
    value = header_float(play, "dt_s")
    if value is None or value <= 0:
        value = probe_common.to_float(play.header.get("clock.step_dt_s"))
    return value if value and value > 0 else None


def covered_steps(step, dt):
    """Engine steps the row covers: round(timeslice_s / dt), at least 1"""
    slice_s = step.f.get("timeslice_s")
    if not slice_s or not dt:
        return 1
    return max(1, int(round(slice_s / dt)))


def duration_s(rows):
    """World time the rows cover: the sum of their timeslice_s"""
    return sum((s.f.get("timeslice_s") or 0.0) for s in rows)


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
        parts = play.header.get("steer.wheel.%d.pos" % i, "").split()
        if len(parts) != 3:
            continue
        z = probe_common.to_float(parts[2])
        if z is not None:
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


def path_points(rows):
    """[(s m, psi rad, r rad/s, v m/s, row)] along the rows: the path length and the unwrapped heading"""
    points = []
    s = 0.0
    psi = 0.0
    prev = None
    for step in rows:
        x, z, h = step.f.get("pos_x"), step.f.get("pos_z"), step.f.get("heading_deg")
        if x is None or z is None or h is None:
            continue
        if prev is not None:
            s += math.hypot(x - prev[0], z - prev[1])
            psi += math.radians(wrap_deg(h - prev[2]))
        prev = (x, z, h)
        points.append((s, psi, step.f.get("angvel_l_y"), step.f.get("vel_l_z"), step))
    return points


def local_windows(rows, window_m, step_m):
    """Sliding windows of window_m path length every step_m: [(|dpsi| / ds 1/m, |mean r| / mean v 1/m, rows in the
    window)]. The absolute value is taken per window, so a curvature that alternates over more than a window length
    is not cancelled"""
    points = path_points(rows)
    out = []
    start = 0
    begin = 0.0
    while points and begin + window_m <= points[-1][0]:
        while start < len(points) and points[start][0] < begin:
            start += 1
        end = start
        while end < len(points) and points[end][0] < points[start][0] + window_m:
            end += 1
        if end >= len(points) or end <= start:
            break
        ds = points[end][0] - points[start][0]
        dpsi = points[end][1] - points[start][1]
        rates = [p[2] for p in points[start + 1:end + 1] if p[2] is not None]
        speeds = [p[3] for p in points[start + 1:end + 1] if p[3] is not None]
        yaw = abs(mean(rates)) / mean(speeds) if rates and speeds and mean(speeds) > MIN_SPEED_MS else None
        if ds > 0:
            out.append((abs(dpsi) / ds, yaw, end - start))
        begin += step_m
    return out


def decimals(texts):
    """The most decimals of the number strings (the CSV resolution)"""
    best = 0
    for text in texts:
        if text and "." in text:
            best = max(best, len(text.split(".", 1)[1].rstrip()))
    return best


def heading_noise(rows):
    """(sigma_h rad, quantisation step rad): the per-row heading noise from the data, the rms of the second
    differences of the unwrapped heading over consecutive rows / sqrt(6) (white noise; a smooth curvature adds
    almost nothing at one step), at least the quantisation q / sqrt(12) of the CSV's heading decimals"""
    points = path_points(rows)
    q = math.radians(10.0 ** -decimals(p[4].raw.get("heading_deg", "") for p in points)) if points else None
    second = [points[k + 1][1] - 2.0 * points[k][1] + points[k - 1][1] for k in range(1, len(points) - 1)]
    if len(second) < 10 or q is None:
        return None, q
    jitter = math.sqrt(sum(d * d for d in second) / len(second) / 6.0)
    return max(jitter, q / math.sqrt(12.0)), q


def yaw_noise(rows):
    """sigma_r rad/s per row: the rms of the first differences of angvel_l_y / sqrt(2), at least the quantisation of
    its decimals / sqrt(12)"""
    values = [s.f.get("angvel_l_y") for s in rows if s.f.get("angvel_l_y") is not None]
    diffs = [values[k] - values[k - 1] for k in range(1, len(values))]
    if len(diffs) < 10:
        return None
    q = 10.0 ** -decimals(s.raw.get("angvel_l_y", "") for s in rows)
    return max(math.sqrt(sum(d * d for d in diffs) / len(diffs) / 2.0), q / math.sqrt(12.0))


def curvature_stats(rows, window_m, step_m):
    """The local absolute-curvature statistic of zero-command rows and its noise floor: {'k_p95', 'k_max', 'r_p95',
    'r_max', 'k_floor', 'r_floor', 'windows', 'sigma_h', 'sigma_r', 'q_h'}. Floors (3 sigma of one window):
    heading 3 sqrt(2) sigma_h / window_m (the two end headings), yaw 3 rms(yaw statistic - heading statistic)"""
    windows = local_windows(rows, window_m, step_m)
    sigma_h, q_h = heading_noise(rows)
    sigma_r = yaw_noise(rows)
    out = {"windows": len(windows), "sigma_h": sigma_h, "sigma_r": sigma_r, "q_h": q_h}
    kappas = [w[0] for w in windows]
    yaws = [w[1] for w in windows if w[1] is not None]
    out["k_p95"] = probe_common.percentile(kappas, 95) if kappas else None
    out["k_max"] = max(kappas) if kappas else None
    out["r_p95"] = probe_common.percentile(yaws, 95) if yaws else None
    out["r_max"] = max(yaws) if yaws else None
    out["k_floor"] = 3.0 * math.sqrt(2.0) * sigma_h / window_m if sigma_h is not None else None
    # the yaw statistic measures the same curvature from the sampled rate: 3 x the rms of its difference to the
    # heading statistic on the same windows (the per-row rate jitter integrates away in the heading, so a white-noise
    # floor from sigma_r would overstate it; e01 1.9.0.37: rms 1e-5..2.5e-5 1/m), at least its quantisation
    diffs = [w[1] - w[0] for w in windows if w[1] is not None]
    speeds = [s.f.get("vel_l_z") for s in rows if s.f.get("vel_l_z") is not None]
    v = mean(speeds)
    if len(diffs) >= 10 and sigma_r is not None and v:
        q_r = 10.0 ** -decimals(s.raw.get("angvel_l_y", "") for s in rows)
        out["r_floor"] = max(3.0 * math.sqrt(sum(d * d for d in diffs) / len(diffs)), q_r / v)
    else:
        out["r_floor"] = None
    return out


def drift_estimate(rows, min_half_m):
    """(drift deg / 100 m, sigma deg / 100 m, detail) of a zero-command segment: the heading change over the path
    length; sigma = sqrt(noise^2 + split^2) with noise = sqrt(2) sigma_h / length (heading_noise) and split = |d1 -
    d2| / 2 of the two halves (the repeat variance). (None, None, why) when a half is shorter than min_half_m or the
    noise cannot be measured"""
    points = path_points(rows)
    if len(points) < 4:
        return None, None, "too few rows"
    length = points[-1][0] - points[0][0]
    if length < 2 * min_half_m:
        return None, None, "%s m (< 2 x %g m halves)" % (fmt(length, 4), min_half_m)
    mid = next(k for k, p in enumerate(points) if p[0] - points[0][0] >= length / 2.0)
    scale = math.degrees(1.0) * 100.0

    def rate(a, b):
        return (points[b][1] - points[a][1]) / (points[b][0] - points[a][0]) * scale

    drift = rate(0, len(points) - 1)
    d1, d2 = rate(0, mid), rate(mid, len(points) - 1)
    sigma_h, _ = heading_noise(rows)
    if sigma_h is None:
        return None, None, "the heading noise cannot be measured"
    noise = math.sqrt(2.0) * sigma_h / length * scale
    split = abs(d1 - d2) / 2.0
    sigma = math.sqrt(noise * noise + split * split)
    return drift, sigma, "halves %s / %s, noise %s, split %s deg/100 m" % (fmt(d1, 3), fmt(d2, 3), fmt(noise, 3),
                                                                            fmt(split, 3))


def phase_rows_complete(steps, play, probe, expect_call):
    """(True, detail) when every phase of <probe>_phases.csv with a start step has all its rows (physics_step
    start_step..end_step, one row per dispatched step) and every row is called or not as expect_call(step) says,
    with a finite cmd_steer and input_post_steer on every called row; else (None, what is missing)"""
    if not play.has(probe + "_phases.csv"):
        return None, "%s: no phases file" % probe
    by_phase = {}
    for s in steps:
        by_phase.setdefault(s.phase, []).append(s)
    problems = []
    called = 0
    for row in play.csv(probe + "_phases", ("phase", "start_step", "end_step")):
        start = probe_common.to_int(row.get("start_step"))
        end = probe_common.to_int(row.get("end_step"))
        name = row.get("phase", "")
        if start is None or end is None:
            if by_phase.get(name):
                problems.append("%s: rows but no start / end step" % name)
            continue
        rows = by_phase.get(name, [])
        numbers = [s.i.get("physics_step") for s in rows]
        expected = end - start + 1
        if len(rows) != expected or numbers != list(range(start, end + 1)):
            problems.append("%s: %d of %d rows (steps %d..%d)" % (name, len(rows), expected, start, end))
        bad = 0
        for s in rows:
            want = expect_call(s)
            if s.i.get("steer_call") != (1 if want else 0):
                bad += 1
            elif want:
                called += 1
                cmd, post = s.f.get("cmd_steer"), s.f.get("input_post_steer")
                if cmd is None or post is None or not math.isfinite(cmd) or not math.isfinite(post):
                    bad += 1
        if bad:
            problems.append("%s: %d rows with a wrong steer_call or no finite command / readback" % (name, bad))
    if problems:
        return None, "%s: %s" % (probe, "; ".join(problems[:6]) + ("; ..." if len(problems) > 6 else ""))
    return True, "%s: %d called rows, every phase complete" % (probe, called)


def track_profile(play, kind):
    """The site's centre tyre-pair heights per station (site.<kind>.track_profile), or None"""
    text = play.header.get("site.%s.track_profile" % kind) or ""
    values = [probe_common.to_float(v) for v in text.split()]
    if not values or any(v is None for v in values):
        return None
    return values


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


def t50_time(rows, values, start, pre, final):
    """World time [ms] of the first row i >= start with (values[i] - pre) / (final - pre) >= 0.5; None when it never
    does, the step is too small or the row has no world time."""
    span = final - pre
    if abs(span) < 1e-9:
        return None
    for i in range(start, len(values)):
        if values[i] is None:
            continue
        if (values[i] - pre) / span >= 0.5:
            return rows[i].f.get("world_time_ms")
    return None
