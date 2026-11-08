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


def decimals(texts):
    """The most decimals of the number strings (the CSV resolution)"""
    best = 0
    for text in texts:
        if text and "." in text:
            best = max(best, len(text.split(".", 1)[1].rstrip()))
    return best


def white_sigma(values):
    """The white-noise sigma of a series from its fourth differences: rms(d4) / sqrt(70) (var(d4) = 70 sigma^2 for
    white noise). A real smooth signal leaks in as (2 sin(pi / N))^4 of its amplitude for N samples per wavelength:
    5% at 13 samples (a 4.9 m wave at 40 km/h), against 23% for second differences; below about 6 samples per
    wavelength the wave is not separable from noise and its floor rises (not resolvable). None for < 15 values"""
    d4 = [values[k + 2] - 4.0 * values[k + 1] + 6.0 * values[k] - 4.0 * values[k - 1] + values[k - 2]
          for k in range(2, len(values) - 2)]
    if len(d4) < 10:
        return None
    return math.sqrt(sum(d * d for d in d4) / len(d4) / 70.0)


def heading_noise(rows):
    """(sigma_h rad, quantisation step rad): the per-row heading noise of the unwrapped heading (white_sigma), at
    least the quantisation q / sqrt(12) of the CSV's heading decimals"""
    points = path_points(rows)
    q = math.radians(10.0 ** -decimals(p[4].raw.get("heading_deg", "") for p in points)) if points else None
    jitter = white_sigma([p[1] for p in points])
    if jitter is None or q is None:
        return None, q
    return max(jitter, q / math.sqrt(12.0)), q


def yaw_noise(rows):
    """sigma_r rad/s per row of angvel_l_y (white_sigma), at least the quantisation of its decimals / sqrt(12)"""
    values = [s.f.get("angvel_l_y") for s in rows if s.f.get("angvel_l_y") is not None]
    jitter = white_sigma(values)
    if jitter is None:
        return None
    q = 10.0 ** -decimals(s.raw.get("angvel_l_y", "") for s in rows)
    return max(jitter, q / math.sqrt(12.0))


def channel_samples(rows):
    """Per-channel samples along the path, each channel on its own: heading steps need only the heading and the
    positions of two consecutive rows, yaw samples only the yaw rate and the speed of a row (placed at that row's
    path position). Returns (heading [(s, ds, |dpsi| / ds, dpsi / ds)], yaw [(s, ds, |r| / v, r / v)], missing
    {'heading': rows without a heading or position, 'yaw': rows without a yaw rate or speed >= MIN_SPEED_MS, or
    without a position to place them})"""
    heading, yaw = [], []
    missing = {"heading": 0, "yaw": 0}
    s_pos = 0.0
    psi = 0.0
    prev = None
    for step in rows:
        x, z, h = step.f.get("pos_x"), step.f.get("pos_z"), step.f.get("heading_deg")
        r, v = step.f.get("angvel_l_y"), step.f.get("vel_l_z")
        if x is None or z is None or h is None:
            missing["heading"] += 1
            missing["yaw"] += 1              # without a position no yaw sample can be placed on the path either
            prev = None
            continue
        ds = 0.0
        if prev is not None:
            ds = math.hypot(x - prev[0], z - prev[1])
            dpsi = math.radians(wrap_deg(h - prev[2]))
            s_pos += ds
            psi += dpsi
            if ds > 0.01:
                heading.append((s_pos, ds, abs(dpsi) / ds, dpsi / ds))
        prev = (x, z, h)
        if r is None or v is None or v < MIN_SPEED_MS:
            missing["yaw"] += 1
        elif ds > 0.0:
            yaw.append((s_pos, ds, abs(r) / v, r / v))
    return heading, yaw, missing


def windowed(samples, total, window_m, step_m):
    """Path-weighted means of the per-sample magnitudes over sliding windows: ([mean], [count])"""
    means, counts = [], []
    begin = 0.0
    a = 0
    while begin + window_m <= total:
        while a < len(samples) and samples[a][0] <= begin:
            a += 1
        b = a
        while b < len(samples) and samples[b][0] <= begin + window_m:
            b += 1
        length = sum(smp[1] for smp in samples[a:b])
        if b > a and length > 0:
            means.append(sum(smp[1] * smp[2] for smp in samples[a:b]) / length)
            counts.append(b - a)
        begin += step_m
    return means, counts


def channel_stats(rows, window_m, step_m):
    """The flat statistic per channel, each channel computed on its own (a missing yaw sample never drops a heading
    step): per-step magnitudes averaged (path-weighted) over sliding windows of window_m every step_m, so a curvature
    of any wavelength keeps its magnitude. {'heading': {...}, 'yaw': {...}, 'offset': the mean signed heading-minus-yaw
    curvature over the rows that have both (a systematic disagreement, reported, never noise), 'windows'}. Each
    channel: p95, max, 'missing' (rows of the run without that channel's samples: the channel is then incomplete)
    and its own noise floor = the noise-only window statistic mu + 3 s from its own measured noise (heading:
    per-step dpsi noise sqrt(2) sigma_h, heading_noise; yaw: sigma_r, yaw_noise): mu = E|noise| / step (sqrt(2 / pi)
    sigma), s = its window fluctuation (x 2 for the correlation of neighbouring heading steps)"""
    heading, yaw, missing = channel_samples(rows)
    sigma_h, q_h = heading_noise(rows)
    sigma_r = yaw_noise(rows)
    total = heading[-1][0] if heading else 0.0
    yaw_by_s = {round(smp[0], 6): smp[3] for smp in yaw}
    pairs = [hs[3] - yaw_by_s[round(hs[0], 6)] for hs in heading if round(hs[0], 6) in yaw_by_s]
    out = {"windows": 0, "offset": mean(pairs) if pairs else None, "sigma_h": sigma_h, "sigma_r": sigma_r,
           "q_h": q_h}
    speeds = [st for st in (s.f.get("vel_l_z") for s in rows) if st is not None and st >= MIN_SPEED_MS]
    v = mean(speeds)
    for name, samples, sigma, scale, corr in (
            ("heading", heading, math.sqrt(2.0) * sigma_h if sigma_h else None,
             mean([smp[1] for smp in heading]) if heading else None, 2.0),
            ("yaw", yaw, sigma_r, v, 1.0)):
        values, counts = windowed(samples, total, window_m, step_m)
        ch = {"p95": probe_common.percentile(values, 95) if values else None, "max": max(values) if values else None,
              "floor": None, "mu": None, "missing": missing[name], "windows": len(values)}
        n = mean(counts) if counts else None
        if sigma is not None and scale and n:
            mu = sigma * math.sqrt(2.0 / math.pi) / scale
            spread = sigma * math.sqrt(corr * (1.0 - 2.0 / math.pi) / n) / scale
            ch["mu"] = mu
            ch["floor"] = mu + 3.0 * spread
        out[name] = ch
    out["windows"] = min(out["heading"]["windows"], out["yaw"]["windows"])
    return out


def end_estimate(points, start, end):
    """The heading at the middle of points[start:end] from a least-squares line in s, and its standard error from
    the residuals with the lag-1 correlation (effective sample size n (1 - rho) / (1 + rho)): (psi, sigma, s)"""
    chunk = points[start:end]
    n = len(chunk)
    if n < 5:
        return None, None, None
    s_mid = sum(p[0] for p in chunk) / n
    p_mid = sum(p[1] for p in chunk) / n
    sxx = sum((p[0] - s_mid) ** 2 for p in chunk)
    slope = sum((p[0] - s_mid) * (p[1] - p_mid) for p in chunk) / sxx if sxx > 0 else 0.0
    residuals = [p[1] - (p_mid + slope * (p[0] - s_mid)) for p in chunk]
    var = sum(r * r for r in residuals) / max(1, n - 2)
    rho = 0.0
    if var > 0:
        rho = sum(residuals[k] * residuals[k - 1] for k in range(1, n)) / ((n - 1) * var)
        rho = max(-0.9, min(0.95, rho))
    n_eff = max(1.0, n * (1.0 - rho) / (1.0 + rho))
    return p_mid, math.sqrt(var / n_eff), s_mid


def drift_estimate(rows, end_m):
    """The heading drift of a zero-command segment from measurement noise only: (drift deg / 100 m, sigma deg /
    100 m, nonstationary, detail). The headings at the start, the middle and the end come from line fits over
    end_m of path (their standard errors from the residuals, corrected for their lag-1 correlation, at least the
    heading quantisation); drift = (end - start) / length. nonstationary when the two halves differ by more than 3
    sigma of their difference: within-run variation is unresolved evidence, never extra tolerance. (None, None,
    None, why) without the rows. Repeated paired runs on the same strip are the proper uncertainty source for (c);
    they are to be implemented once a session has them (paired differences, a small-sample t interval, INCOMPLETE on
    insufficient or invalid pairs)"""
    points = path_points(rows)
    if len(points) < 20:
        return None, None, None, "too few rows"
    total = points[-1][0] - points[0][0]
    if total < 4 * end_m:
        return None, None, None, "%s m (< 4 x %g m)" % (fmt(total, 4), end_m)
    _, q = heading_noise(rows)
    floor = (q or 0.0) / math.sqrt(12.0)

    def at(centre):
        lo = next(k for k, p in enumerate(points) if p[0] >= centre - end_m / 2.0)
        hi = next((k for k, p in enumerate(points) if p[0] > centre + end_m / 2.0), len(points))
        psi, sigma, s_at = end_estimate(points, lo, hi)
        return psi, max(sigma or 0.0, floor), s_at

    s0 = points[0][0]
    a_psi, a_sig, a_s = at(s0 + end_m / 2.0)
    m_psi, m_sig, m_s = at(s0 + total / 2.0)
    e_psi, e_sig, e_s = at(points[-1][0] - end_m / 2.0)
    if None in (a_psi, m_psi, e_psi):
        return None, None, None, "the end fits failed"
    scale = math.degrees(1.0) * 100.0
    length = e_s - a_s
    drift = (e_psi - a_psi) / length * scale
    sigma = math.sqrt(a_sig ** 2 + e_sig ** 2) / length * scale
    d1 = (m_psi - a_psi) / (m_s - a_s) * scale
    d2 = (e_psi - m_psi) / (e_s - m_s) * scale
    sigma_diff = math.sqrt(a_sig ** 2 + 4.0 * m_sig ** 2 + e_sig ** 2) / (length / 2.0) * scale
    nonstationary = abs(d1 - d2) > 3.0 * sigma_diff
    return drift, sigma, nonstationary, "halves %s / %s (3 sigma %s), noise sigma %s deg/100 m" % (
        fmt(d1, 3), fmt(d2, 3), fmt(3.0 * sigma_diff, 3), fmt(sigma, 3))


def margin_test(current, sigma_current, baseline, sigma_baseline, z=1.96):
    """CC-07 (c): |current| <= 1.1 |baseline| tested with a confidence interval of the difference: True when its
    upper bound is <= 0, False when its lower bound is > 0, None in between; (state, difference, half width)"""
    diff = abs(current) - 1.1 * abs(baseline)
    half = z * math.sqrt(sigma_current ** 2 + (1.1 * sigma_baseline) ** 2)
    if diff + half <= 0:
        return True, diff, half
    if diff - half > 0:
        return False, diff, half
    return None, diff, half


def aborted_phases(play, probe):
    """{phase: result} of the phases that did not end ok (an abort ends a run but the probe can still finish with
    a complete footer, so the dependent evidence must look here)"""
    return {name: result for name, result in phase_results(play, probe).items() if result != "ok"}


def run_cli(main, argv=None):
    """The steering analyses' exit codes: 0 the analysis ran (whatever the items say: the Mode A / B decision is a
    manual review), 1 an input or identity error or a usage error"""
    try:
        main(argv)
    except probe_common.InputError as error:
        print("input error: %s" % error)
        return 1
    except SystemExit as stop:
        return 0 if stop.code in (0, None) else 1
    return 0


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
