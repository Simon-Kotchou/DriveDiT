#!/usr/bin/env python3
"""First-pass steer -> curvature fit of the DD-4-probe E-01 / E-02a plays and the vehicle card of llc_v1 (DD-7-min).

Run it with the isolated interpreter (stdlib only; it reuses the probe kit's reader and E-01 / E-02a helpers):

    python3 -I -B tools/ident/fit_steer_kappa.py --e01 <e01 play_dir> --e02a <e02a play_dir> [--e04 <e04 play_dir>]
        [--vehicle S105] [--build-key KEY] [--scripted-gears 2,3,4 --upshift-mps 5,10 --downshift-mps 4,8]
        [--out FILE | --check FILE]
    python3 -I -B tools/ident/fit_steer_kappa.py --placeholder --pin-id 1.8.0.13 [--vehicle S105] [--out | --check]

<play_dir> is one $profile:DC_Probes/<probe>/<launch_id>/play_NN directory of the probe kit (run_header.txt,
<probe>_steps.csv, e01_phases.csv or e02a_cells.csv, run_footer.txt; PROBE_DESIGN.md 5.3). Without --out the card is
printed; --out writes it (default place: tools/ident/cards/<pin_id>/<vehicle>.json, the path llc_v1 and the DD-5-min
reader look up), --check compares it byte for byte with FILE (exit 1 when they differ). Writes only under /mnt/e/ or
/dev/shm/.

WHAT IS FITTED (contract signs: steer, curvature and yaw rate right-positive; c = steer_sign x engine steer)
  signs        steer_sign = sign of corr(cmd_steer, heading rate) on the E-01 steering excite steps at the level-
               correlation peak (the e01_actuation.py rule, |corr| >= 0.3); yaw_axis_sign = sign of corr(angvel_l_y,
               heading rate). An E-02a slope of the other sign refuses the fit.
  kappa_ss     per E-02a speed: kappa = a x c + b by least squares over the valid cells (steady window = last half of
               excite within 1 km/h of the target, |a_lat| <= 4 m/s2; e01_actuation.cell_kappa / fit_line), at the mean
               steady speed of the cells; a >= MIN_KAPPA_PER_STEER (1e-3 1/m per unit steer) required. usable = every
               speed has >= 3 valid cells and R2 >= 0.9 (the probe kit's E-02a criterion; E-02Q at T3 is the real
               qualification).
  vehicle      every play drove the same ego prefab (run header ego.prefab), the card's vehicle when it is known.
  time slice   every play ran at one engine.fixed_time_slice_s (the dt of every structure fit; plays at different
               slices are refused).
  actuator     the CC-08 structure of the raw engine path (MIGRATION_1_9 6.1; first pass, one build):
                 E-01 PRBS (20 km/h, amplitude 0.3): competing structures pure delay (kappa = G c(t - d) + b) and delay
                 + first order (time constant tau), each fitted on the first half of the excite and scored on the
                 held-out second half; the lower held-out error wins (BIC breaks a tie within 1 %);
                 E-02a cells (one bin per speed and amplitude): the step response of the cell (steer 0 -> s at the
                 excite start) with both structures, chosen by BIC;
               the engine input (input_path): input_echo = input_pre_steer and input_post_steer equal cmd_steer
               (+-1e-3) on every step (command changes included); otherwise input_shaping tells a pure rate limit
               (steer_rate_limit_per_s) from a first-order lag (input_lag_tau_s) by which one-step model predicts
               input_post_steer, else "unknown"; input_max_slope_per_s is the largest change per second while the
               input lags (a rate limit, or a lag's initial slope). Only an identified pure rate limit lets llc_v1 run
               a non-identity limiter (strictly slower). The full llc path (kappa target -> kappa realised) needs
               llc_v1 sessions and stays null here (CC-08, DD-7-ident).
  longitudinal throttle_per_mps2 and brake_per_mps2 = 1 / the slope through the origin of the acc_l_z response to the
               E-01 throttle and brake steps (after their onset delay: throttle while below 15 km/h, brake while above
               3 km/h); the onset delays in physics steps.
  gearbox      "automatic" when E-01's gear_observe shifts up by itself (the e01_actuation.py rule), else "scripted",
               which needs --scripted-gears / --upshift-mps / --downshift-mps (forward gears only, TBD-7);
               gear_indices_forward = the input_post_gear values seen while driving forward (>= 5 km/h, the TBD-7
               definition); neutral and reverse stay null (forward-only until a probe measures them); scripted gears
               must be among them and must also show propulsion (throttle > 0 with forward acceleration in that
               gear: a neutral index reported while rolling is not a gear to drive in). clutch_dip_s is the clutch dip
               of E-01's gear_scripted phase (params.m_fGearClutchDipS, required for a scripted card): llc_v1 dips the
               clutch the same way after each scripted gear change; 0 with an automatic gearbox.
  llc_v1       the controller parameters of DC_LLCv1 (LLC_DEFAULTS) with the fitted throttle and brake gains; the llc
               steering rate limiter is the identity (CC-08: it never stacks with an engine limit).

THE CARD (drivedit.vehicle_card/1): card_id, vehicle, prefab, status (identified | placeholder), pin_id, build_key,
identification_sessions (<probe>/<launch_id>/<play>), signs, gear_indices_forward / gear_index_neutral /
gear_index_reverse (read by the DD-5-min reader for dir_state), gearbox, kappa_ss, actuator {raw_engine_path,
full_llc_path}, longitudinal, llc_v1, mode_a, fit. llc_v1 (DC_VehicleCard) reads signs, gearbox.mode and
clutch_dip_s, the kappa_ss and gearbox arrays, actuator.raw_engine_path input_echo, input_shaping and
steer_rate_limit_per_s, and llc_v1; the keys it scans by name (ENGINE_SCANNED_KEYS: the number arrays,
identification_sessions, build_key and input_echo) occur once in the file, and no number uses an exponent
(string.ToFloat).

An identified card needs complete plays (run_footer status complete) of one pin and one build_key (from the run
headers, or --build-key when the headers carry none: dd-launch/1), a usable E-02a fit and every identification above;
otherwise nothing is written (exit 2). Every card is checked after serialization against the rules llc_v1 applies when
it loads and accepts it (enforce_load_check: DC_VehicleCard.Check and the AcceptCard limiter rule), so a card that
says usable is never one the engine refuses. --placeholder writes the PLACEHOLDER card instead: status placeholder, no
identification session, build_key null, the TBD-5 draft signs and guessed kappa_ss; llc_v1 accepts it only for
qualification (use_class dev or probe) and refuses it at any other use_class, production included.

Sealed map: any path or run header naming eden / everon is refused with exit 2 before anything is fitted.
Exit codes: 0 card built (and written or equal), 1 --check found a difference, 2 refused (input, fit or path).
"""

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
T0ENG = os.path.join(REPO, "tools", "probes", "t0eng")
sys.path.insert(0, T0ENG)
import probe_common  # noqa: E402  (the probe kit's play reader, PROBE_DESIGN.md 4.7)
import e01_actuation  # noqa: E402  (its E-01 / E-02a helpers: Steps, cell_kappa, fit_line, best_lag, ...)

GENERATOR = "tools/ident/fit_steer_kappa.py"
GENERATOR_VERSION = 1
CARD_SCHEMA = "drivedit.vehicle_card/1"
CARDS_DIR = os.path.join(REPO, "tools", "ident", "cards")
DEFAULT_VEHICLE = "S105"
PREFABS = {"S105": "{DA79E34823120087}Prefabs/Vehicles/Wheeled/S105/S105_base.et"}
ENGINE_ARRAY_KEYS = ("speeds_mps", "kappa_per_steer", "kappa_offset", "scripted_gears", "upshift_mps", "downshift_mps",
                     "gear_indices_forward")
ENGINE_SCANNED_KEYS = ENGINE_ARRAY_KEYS + ("identification_sessions", "build_key", "input_echo")

# E-02a "usable" criterion and steady window: the probe kit's defaults (e01_actuation.py analyse_e02a)
MIN_CELLS = 3
MIN_R2 = 0.9
MAX_ALAT = 4.0
STEADY_KMH = 1.0
MIN_STEADY_STEPS = 10
MIN_CORR = 0.3
MAX_LAG_STEPS = 30
STEP_FIT_BEFORE = 30            # E-02a step response: steps before the excite start ...
STEP_FIT_AFTER = 150            # ... and after it
TAU_GRID_S = tuple(0.01 * 1.2 ** k for k in range(30))   # 0.01 .. 2.0 s
TIE_REL = 0.01
ECHO_TOL = 1e-3
SHAPING_TOL = 1e-3              # one-step RMS within which a rate limit or a lag model explains input_post_steer
MIN_SHAPING_STEPS = 3
MIN_KAPPA_PER_STEER = 1e-3      # [1/m per unit steer]; the S105 guesses are 0.07 .. 0.2 (a smaller gain is no fit)
DT_TOL = 1e-6                   # [s] fixed time slices printed with 6 decimals
MOVE_KMH = 5.0
THROTTLE_FIT_MAX_KMH = 15.0
BRAKE_FIT_MIN_KMH = 3.0
LONG_MAX_LAG_STEPS = 120
GEAR_MIN_SHARE = 0.01
MIN_PROPULSION_STEPS = 10       # steps with throttle and forward acceleration that make a gear one to drive in
CLUTCH_DIP_MAX_S = 2.0          # the probe kit's m_fGearClutchDipS range and DC_VehicleCard.CLUTCH_DIP_MAX_S
INPUT_SHAPINGS = ("echo", "rate_limit", "lag", "unknown")

# Controller parameters of llc_v1 that are not identified (versioned with DC_LLCv1.c; tuned on the scratch
# simulation of DD7_IMPL.md, to be re-checked on the first T0-MIN loop sessions)
LLC_DEFAULTS = (
    ("steer_max", 1.0),
    ("steer_rate_per_s", 0.0),         # the identity: never stacks with an engine limit (CC-08); DD-7-ident may slow it
    ("steer_lag_tau_s", 0.03),
    ("kappa_trim_gain", 0.0),
    ("kappa_trim_max", 0.1),
    ("trim_min_speed_mps", 3.0),
    ("speed_kp", 0.8),
    ("speed_ki", 0.2),
    ("speed_window_mps", 2.0),
    ("integral_max_mps2", 1.5),
    ("throttle_per_mps2", 0.35),
    ("brake_per_mps2", 0.15),
    ("throttle_max", 0.8),
    ("coast_band_mps2", 0.05),
    ("brake_hold", 0.6),
    ("stop_speed_mps", 0.3),
)

# The PLACEHOLDER card's guesses (no probe has run): E-02a grid speeds 15 / 30 / 50 km/h, curvature per contract steer
# falling with speed (an engine steering reduction is expected on 1.8), the TBD-5 draft signs
PLACEHOLDER_SPEEDS_MPS = (4.17, 8.33, 13.89)
PLACEHOLDER_KAPPA_PER_STEER = (0.2, 0.12, 0.07)
PLACEHOLDER_SIGNS = (-1, 1)

WRITE_ROOTS = ("/mnt/e/", "/dev/shm/")
FORBIDDEN_ROOTS = ("/mnt/c/", "/tmp/")
SEALED_RE = re.compile(r"eden|everon", re.IGNORECASE)
PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class Refusal(Exception):
    """An input, fit or path problem: exit 2, nothing written."""


# ---------------------------------------------------------------------------------------------------------------
# inputs


def sealed_reason(text):
    if text and SEALED_RE.search(text):
        return "refusing %s: it names the sealed map (eden / everon); Everon is the zero-shot test map" % text
    return None


def check_out_path(path):
    real = os.path.realpath(os.path.abspath(path))
    probe = real if real.endswith("/") else real + "/"
    if any(probe.startswith(root) for root in FORBIDDEN_ROOTS) or not any(probe.startswith(r) for r in WRITE_ROOTS):
        raise Refusal("refusing output %s: only /mnt/e/... or /dev/shm/... (never /mnt/c or /tmp)" % path)
    return real


def open_play(path, probe):
    """A complete play of the given probe, not on the sealed map."""
    for text in (path, os.path.realpath(path)):
        reason = sealed_reason(text)
        if reason:
            raise Refusal(reason)
    try:
        play = probe_common.Play(path)
    except probe_common.InputError as e:
        raise Refusal(str(e))
    if play.probe != probe:
        raise Refusal("%s is a %r play, expected %s" % (path, play.probe, probe))
    for key in ("world.map_id", "world.terrain", "world.file"):
        reason = sealed_reason(play.header.get(key, ""))
        if reason:
            raise Refusal("%s: %s" % (path, reason))
    if not play.complete:
        raise Refusal("%s did not end normally (run_footer status %s): an identified card needs complete plays"
                      % (path, (play.footer or {}).get("status", "missing")))
    return play


def session_id(play):
    """<probe>/<launch_id>/<play_NN> of a play (the identification session id in the card)."""
    launch_id = play.header.get("launch.id", "")
    if not launch_id:
        raise Refusal("%s: run header has no launch.id" % play.dir)
    return "%s/%s/%s" % (play.probe, launch_id, os.path.basename(play.dir))


def vehicle_prefab(plays, vehicle):
    """The one ego prefab every play drove (run header ego.prefab), the vehicle's when it is known."""
    prefabs = sorted({p.header.get("ego.prefab", "") for p in plays})
    if len(prefabs) != 1 or not prefabs[0]:
        raise Refusal("the plays drove the ego prefabs %s: a card is fitted from one vehicle" % prefabs)
    expected = PREFABS.get(vehicle)
    if expected and prefabs[0] != expected:
        raise Refusal("the plays drove %s, card %s is %s" % (prefabs[0], vehicle, expected))
    return prefabs[0]


def scripted_gearbox(args, forward, propulsion):
    """(gears, upshift, downshift) of --scripted-gears / --upshift-mps / --downshift-mps, checked as llc_v1 checks a
    card (DC_VehicleCard.CheckGearbox): n >= 1 forward gears, n - 1 up and down shift speeds, down < up, up
    increasing; every gear one the plays drove forward in and one with propulsion (fit_gearbox)."""
    try:
        gears = [int(v) for v in args.scripted_gears.split(",")]
        up = [float(v) for v in args.upshift_mps.split(",") if v.strip()]
        down = [float(v) for v in args.downshift_mps.split(",") if v.strip()]
    except ValueError as e:
        raise Refusal("scripted gearbox arguments: %s" % e)
    ok = gears and len(up) == len(gears) - 1 and len(down) == len(gears) - 1
    ok = ok and all(d < u for d, u in zip(down, up)) and all(b > a for a, b in zip(up, up[1:]))
    if not ok:
        raise Refusal("scripted gearbox: n >= 1 gears, n - 1 upshift and downshift speeds, downshift < upshift, "
                      "upshifts increasing (got %s / %s / %s)" % (gears, up, down))
    unseen = [g for g in gears if g not in forward]
    if unseen:
        raise Refusal("scripted gears %s were not seen while driving forward (gear_indices_forward %s)"
                      % (unseen, forward))
    idle = [g for g in gears if g not in propulsion]
    if idle:
        raise Refusal("scripted gears %s show no propulsion in E-01 (throttle > 0 with forward acceleration; gears "
                      "with propulsion %s): a neutral index reported while rolling is not a forward gear (TBD-7)"
                      % (idle, propulsion))
    return gears, up, down


def clutch_dip(play):
    """The clutch dip [s] of E-01's gear_scripted phase (params.m_fGearClutchDipS of the run header): what llc_v1
    must reproduce after each scripted gear change (DC_ProbeSequences: clutch 0 for this long at the start of every
    gear hold). Required for a scripted card."""
    text = e01_actuation.header_value(play, "m_fGearClutchDipS")
    value = probe_common.to_float(text)
    if value is None:
        raise Refusal("%s: run header has no params.m_fGearClutchDipS: the clutch dip of the scripted gear phase is "
                      "unknown, and llc_v1 must reproduce it" % play.dir)
    if value < 0 or value > CLUTCH_DIP_MAX_S:
        raise Refusal("E-01 clutch dip %s s is outside [0, %s] s" % (value, CLUTCH_DIP_MAX_S))
    return value


def time_slice(plays):
    """engine.fixed_time_slice_s shared by every play: the PRBS bin, the E-02a step responses and the longitudinal
    delays are fitted with one dt, so plays at different fixed time slices are refused."""
    slices = []
    for play in plays:
        dt = probe_common.to_float(play.header.get("engine.fixed_time_slice_s"))
        if not dt or dt <= 0:
            raise Refusal("%s: run header has no engine.fixed_time_slice_s" % play.dir)
        slices.append(dt)
    if max(slices) - min(slices) > DT_TOL:
        raise Refusal("the plays ran at fixed time slices %s s: a card is fitted from plays of one time slice"
                      % sorted(set(slices)))
    return slices[0]


def build_identity(plays, build_key_arg):
    """(pin_id, build_key) shared by every play; --build-key fills a dd-launch/1 header's empty build_key."""
    pins = sorted({p.header.get("launch.pin_id", "") for p in plays})
    keys = sorted({p.header.get("launch.build_key", "") for p in plays})
    if len(pins) != 1 or not pins[0]:
        raise Refusal("the plays name pins %s: a card is fitted from one pin" % pins)
    if len(keys) != 1:
        raise Refusal("the plays name build_keys %s: a card is fitted from one build" % keys)
    build_key = keys[0]
    if build_key and build_key_arg and build_key_arg != build_key:
        raise Refusal("--build-key %s differs from the plays' build_key %s" % (build_key_arg, build_key))
    build_key = build_key or build_key_arg or ""
    if not build_key:
        raise Refusal("the plays carry no build_key (dd-launch/1 records): pass --build-key (the pin's build_key, "
                      "as the harvester's harvest.json records it)")
    if not PIN_RE.match(pins[0]):
        raise Refusal("pin_id %r is not a pin id" % pins[0])
    return pins[0], build_key


# ---------------------------------------------------------------------------------------------------------------
# numerics


def least_squares(rows, targets):
    """Coefficients of targets ~ rows (normal equations, 1 to 3 regressors); None when singular."""
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    atb = [sum(r[i] * t for r, t in zip(rows, targets)) for i in range(k)]
    return solve(ata, atb)


def solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def first_order(values, alpha):
    """y[t] = y[t-1] + alpha (x[t] - y[t-1]), y[0] = x[0]."""
    out = []
    y = values[0]
    for x in values:
        y += alpha * (x - y)
        out.append(y)
    return out


def r_squared(measured, predicted):
    mean = sum(measured) / len(measured)
    ss_tot = sum((m - mean) ** 2 for m in measured)
    ss_res = sum((m - p) ** 2 for m, p in zip(measured, predicted))
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else None, ss_res


def bic(sse, n, k):
    return n * math.log(max(sse, 1e-300) / n) + k * math.log(n)


def fit_structures(u, kappa, train, test, dt):
    """Pure delay vs delay + first order on one steering record.
    u, kappa: lists over consecutive steps (contract steer command, measured curvature); train / test: index ranges.
    Returns {"pure_delay": {...}, "delay_first_order": {...}, "selected": name} (the lower held-out SSE, BIC within
    TIE_REL; with test None both are scored on train and BIC decides)."""
    out = {}
    for name, taus in (("pure_delay", (None,)), ("delay_first_order", TAU_GRID_S)):
        best = None
        for d in range(0, MAX_LAG_STEPS + 1):
            delayed = [u[i - d] if i - d >= 0 else u[0] for i in range(len(u))]
            for tau in taus:
                if tau is None:
                    shaped, ones = delayed, [1.0] * len(u)
                else:
                    alpha = 1.0 - math.exp(-dt / tau)
                    shaped, ones = first_order(delayed, alpha), first_order([1.0] * len(u), alpha)
                rows = [(shaped[i], ones[i]) for i in train]
                coef = least_squares(rows, [kappa[i] for i in train])
                if coef is None:
                    continue
                predicted = [coef[0] * shaped[i] + coef[1] * ones[i] for i in range(len(u))]
                _, sse_train = r_squared([kappa[i] for i in train], [predicted[i] for i in train])
                if best is None or sse_train < best["sse_train"]:
                    best = {"dead_time_steps": d, "tau_s": tau, "gain": coef[0], "offset": coef[1],
                            "sse_train": sse_train, "predicted": predicted}
        if best is None:
            continue
        k = 3 if name == "pure_delay" else 4
        scored = test if test else train
        r2, sse = r_squared([kappa[i] for i in scored], [best["predicted"][i] for i in scored])
        best.update({"r2_heldout" if test else "r2": r2, "sse_scored": sse, "bic": bic(sse, len(scored), k)})
        del best["predicted"]
        out[name] = best
    if not out:
        return None
    names = sorted(out, key=lambda n: out[n]["sse_scored"])
    selected = names[0]
    if len(names) > 1 and test:
        a, b = out[names[0]], out[names[1]]
        if b["sse_scored"] <= a["sse_scored"] * (1.0 + TIE_REL) and b["bic"] < a["bic"]:
            selected = names[1]
    elif len(names) > 1:
        selected = min(names, key=lambda n: out[n]["bic"])
    out["selected"] = selected
    return out


# ---------------------------------------------------------------------------------------------------------------
# E-01


def heading_rate_rad(steps):
    return {s: math.radians(v) for s, v in steps.heading_rate.items()}


def fit_signs(steps):
    """(steer_sign, yaw_axis_sign, detail) from the E-01 steering phases (the e01_actuation.py rules)."""
    phases = steps.phases("steer_prbs") + steps.phases("steer_reversal")
    cmd, yaw, angvel = {}, {}, {}
    for phase in phases:
        cmd.update(steps.series(phase, "cmd_steer", ("excite",)))
        yaw.update(steps.yaw_series(phase))
        angvel.update(steps.series(phase, "angvel_l_y", e01_actuation.DRIVING))
    if len(cmd) < 3:
        raise Refusal("E-01 has no steering excite steps (steer_prbs / steer_reversal)")
    lag, c = e01_actuation.best_lag(cmd, yaw, MAX_LAG_STEPS)
    if c is None or abs(c) < MIN_CORR:
        raise Refusal("steer_sign not measured (|corr| %s < %.1f)" % (e01_actuation.fmt(c), MIN_CORR))
    c_axis = e01_actuation.corr_at_lag(angvel, yaw, 0)
    if c_axis is None or abs(c_axis) < MIN_CORR:
        raise Refusal("yaw_axis_sign not measured (|corr(angvel_l_y, heading rate)| %s < %.1f)"
                      % (e01_actuation.fmt(c_axis), MIN_CORR))
    steer_sign = 1 if c > 0 else -1
    yaw_axis_sign = 1 if c_axis > 0 else -1
    return steer_sign, yaw_axis_sign, {"steer_corr": c, "steer_corr_lag_steps": lag, "yaw_axis_corr": c_axis}


def steering_record(steps, phase, steer_sign, segments):
    """(u, kappa) over the consecutive steps of a phase's segments with speed > 1 m/s: contract steer command and
    curvature = heading rate / speed."""
    cmd = steps.series(phase, "cmd_steer", segments)
    speed = steps.series(phase, "speed_kmh", segments)
    rate = heading_rate_rad(steps)
    u, kappa = [], []
    for s in sorted(cmd):
        v = (speed.get(s) or 0.0) / 3.6
        if s not in rate or v <= 1.0 or cmd[s] is None:
            continue
        u.append(steer_sign * cmd[s])
        kappa.append(rate[s] / v)
    return u, kappa


def fit_prbs(steps, steer_sign, dt):
    """The E-01 PRBS bin: both structures, train on the first half of the excite, score on the second."""
    phases = steps.phases("steer_prbs")
    if not phases:
        raise Refusal("E-01 has no steer_prbs phase")
    u, kappa = steering_record(steps, phases[0], steer_sign, ("excite",))
    if len(u) < 4 * MAX_LAG_STEPS:
        raise Refusal("E-01 steer_prbs has %d usable steps (needs >= %d)" % (len(u), 4 * MAX_LAG_STEPS))
    half = len(u) // 2
    fit = fit_structures(u, kappa, range(MAX_LAG_STEPS, half), range(half, len(u)), dt)
    if fit is None:
        raise Refusal("E-01 steer_prbs: no structure could be fitted")
    speed = steps.series(phases[0], "speed_kmh", ("excite",))
    amplitude = max(abs(steps.series(phases[0], "cmd_steer", ("excite",))[s] or 0.0) for s in speed)
    return {"source": "E-01 steer_prbs", "speed_mps": sum(v for v in speed.values() if v) / len(speed) / 3.6,
            "amplitude": amplitude, "structures": fit}


def input_path(steps, dt):
    """The engine's steering input path (CC-08 raw engine path) as the card's raw_engine_path fields.
    echo        input_pre_steer (read right after the setter) and input_post_steer (after the step) both equal
                cmd_steer within ECHO_TOL on every step, the steps of a command change included (a fast limit shows
                only there: one lagging step).
    Otherwise the engine shapes the input; its kind is read from the steps that begin with input_post_steer away from
    the command (error e = cmd - previous input_post_steer, change d = input_post_steer - previous), with two one-step
    models:
    rate_limit  d = clamp(e, +-c), c the largest change on a step that still lags: it predicts every such step within
                SHAPING_TOL (RMS) and the lag does not; steer_rate_limit_per_s = c / dt;
    lag         d = alpha e, alpha by least squares: it predicts them and the rate limit does not; input_lag_tau_s =
                -dt / ln(1 - alpha);
    unknown     neither, both, or fewer than MIN_SHAPING_STEPS steps (a delay, a mix): no limit is identified.
    input_max_slope_per_s = c / dt (a rate limit, or a lag's initial slope) is kept for every shaped path."""
    worst = 0.0
    lagging = []
    pairs = []
    previous = None
    for row in steps.rows:
        step = int(row["physics_step"])
        cmd = e01_actuation.num(row["cmd_steer"])
        pre = e01_actuation.num(row["input_pre_steer"])
        post = e01_actuation.num(row["input_post_steer"])
        if None not in (cmd, pre, post):
            worst = max(worst, abs(pre - cmd), abs(post - cmd))
        consecutive = previous is not None and previous[0] == step - 1
        if consecutive and None not in (cmd, post, previous[1]) and abs(cmd - previous[1]) > ECHO_TOL:
            pairs.append((cmd - previous[1], post - previous[1]))
            if abs(post - cmd) > ECHO_TOL:
                lagging.append(abs(post - previous[1]))
        previous = (step, post)
    path = {"input_echo": True, "input_shaping": "echo", "steer_rate_limit_per_s": None, "input_lag_tau_s": None,
            "input_max_slope_per_s": None, "shaping_fit": {"steps": len(pairs), "rate_limit_rms": None,
                                                           "lag_rms": None}}
    if worst <= ECHO_TOL:
        return path
    path.update({"input_echo": False, "input_shaping": "unknown"})
    c = max(lagging) if lagging else 0.0
    if c > 0:
        path["input_max_slope_per_s"] = c / dt
    if len(pairs) < MIN_SHAPING_STEPS:
        return path
    n = float(len(pairs))
    rate_rms = None
    if c > ECHO_TOL:
        rate_rms = math.sqrt(sum((d - max(-c, min(c, e))) ** 2 for e, d in pairs) / n)
    alpha = sum(e * d for e, d in pairs) / sum(e * e for e, _ in pairs)
    lag_rms = math.sqrt(sum((d - alpha * e) ** 2 for e, d in pairs) / n)
    path["shaping_fit"].update({"rate_limit_rms": rate_rms, "lag_rms": lag_rms, "lag_alpha": alpha})
    rate_fits = rate_rms is not None and rate_rms <= SHAPING_TOL
    lag_fits = lag_rms <= SHAPING_TOL and 0.0 < alpha < 1.0
    if rate_fits and not lag_fits:
        path.update({"input_shaping": "rate_limit", "steer_rate_limit_per_s": c / dt})
    elif lag_fits and not rate_fits:
        path.update({"input_shaping": "lag", "input_lag_tau_s": -dt / math.log(1.0 - alpha)})
    return path


def fit_longitudinal(steps, phases_csv):
    """throttle / brake gains and onset delays from the E-01 throttle_<v> and brake_<v> phases."""
    out = {}
    for kind, base, keep in (("throttle", "settle", lambda kmh: kmh < THROTTLE_FIT_MAX_KMH),
                             ("brake", "accel", lambda kmh: kmh > BRAKE_FIT_MIN_KMH)):
        pairs, delays = [], []
        for phase in steps.phases(kind):
            start = steps.first_step(phase, "excite")
            level = e01_actuation.num(next((r["amplitude"] for r in phases_csv if r["phase"] == phase), None))
            if start is None or not level:
                continue
            acc = steps.series(phase, "acc_l_z")
            delay = e01_actuation.onset_delay(acc, start, e01_actuation.BASELINE_STEPS, LONG_MAX_LAG_STEPS)
            if delay is None:
                continue
            delays.append(delay)
            speed = steps.series(phase, "speed_kmh", ("excite",))
            values = [acc[s] for s in sorted(speed) if s >= start + delay and acc.get(s) is not None
                      and speed[s] is not None and keep(speed[s])]
            if len(values) >= 10:
                response = sum(values) / len(values)
                pairs.append((level, response if kind == "throttle" else -response))
        if len(pairs) < 2:
            raise Refusal("E-01 %s steps: %d usable phase(s), needs 2" % (kind, len(pairs)))
        slope = sum(level * a for level, a in pairs) / sum(level * level for level, _ in pairs)
        if slope <= 0:
            raise Refusal("E-01 %s steps: the acceleration does not follow the %s (slope %.3f)" % (kind, kind, slope))
        out[kind] = {"mps2_per_unit": slope, "per_mps2": 1.0 / slope, "delay_steps": e01_actuation.median(delays),
                     "points": [{"level": level, "accel_mps2": a} for level, a in pairs]}
    return out


def fit_gearbox(steps):
    """(automatic works, gear_indices_forward, propulsion gears, detail): the e01_actuation.py gear_observe rule, the
    input_post_gear values while driving forward (TBD-7's definition) and, of those, the gears with at least
    MIN_PROPULSION_STEPS steps of throttle (input_post_throttle > 0) and forward acceleration (acc_l_z > 0)."""
    observe = [r for r in steps.by_phase.get("gear_observe", []) if r["segment"] == "excite"]
    if not observe:
        raise Refusal("E-01 has no gear_observe excite steps")
    gears = [int(float(r["cur_gear"])) for r in observe]
    no_setgear = all(int(float(r["cmd_gear"])) < 0 for r in observe)
    upshifts = sum(1 for a, b in zip(gears, gears[1:]) if b > a)
    gain = (e01_actuation.num(observe[-1]["speed_kmh"]) or 0.0) - (e01_actuation.num(observe[0]["speed_kmh"]) or 0.0)
    automatic = no_setgear and upshifts >= 1 and gain >= MOVE_KMH
    counts = {}
    pushed = {}
    moving = 0
    for row in steps.rows:
        speed = e01_actuation.num(row["speed_kmh"])
        gear = e01_actuation.num(row["input_post_gear"])
        if row["segment"] in e01_actuation.DRIVING and speed is not None and speed >= MOVE_KMH \
                and (e01_actuation.num(row["vel_l_z"]) or 0.0) > 0 and gear is not None:
            moving += 1
            counts[int(gear)] = counts.get(int(gear), 0) + 1
            if (e01_actuation.num(row["input_post_throttle"]) or 0.0) > 0 \
                    and (e01_actuation.num(row["acc_l_z"]) or 0.0) > 0:
                pushed[int(gear)] = pushed.get(int(gear), 0) + 1
    forward = sorted(g for g, n in counts.items() if moving and n >= GEAR_MIN_SHARE * moving)
    propulsion = sorted(g for g in forward if pushed.get(g, 0) >= MIN_PROPULSION_STEPS)
    return automatic, forward, propulsion, {
        "upshifts": upshifts, "speed_gain_kmh": gain, "no_setgear": no_setgear, "moving_steps": moving,
        "input_post_gear_counts": {str(k): v for k, v in sorted(counts.items())},
        "propulsion_steps": {str(k): v for k, v in sorted(pushed.items())}, "propulsion_gears": propulsion}


def force_enable_simulation(phases_csv):
    rows = {r["phase"]: r["result"] for r in phases_csv}
    if rows.get("move_check") == "ok":
        return "not needed"
    if rows.get("move_check_forced") == "ok":
        return "needed"
    return "undecided"


# ---------------------------------------------------------------------------------------------------------------
# E-02a


def fit_kappa_ss(play, steer_sign, dt):
    """kappa_ss per speed and the per-cell step-response bins."""
    steps = e01_actuation.Steps(play.csv("e02a_steps", required=e01_actuation.STEP_COLUMNS))
    cells = play.csv("e02a_cells", required=("cell_idx", "target_kmh", "steer", "result", "excite_start_step"))
    by_speed = {}
    cell_rows = []
    bins = []
    for cell in cells:
        target = float(cell["target_kmh"])
        steer = float(cell["steer"])
        phase = "cell_%s" % cell["cell_idx"]
        rows = steps.by_phase.get(phase, [])
        kappa, v, used = e01_actuation.cell_kappa(rows, target, STEADY_KMH, MIN_STEADY_STEPS)
        entry = {"cell": int(cell["cell_idx"]), "target_kmh": target, "steer": steer, "contract_steer": steer_sign * steer,
                 "result": cell["result"], "steady_steps": used}
        if kappa is not None:
            entry.update({"kappa": kappa, "speed_mps": v, "a_lat_mps2": v * v * kappa})
            if abs(v * v * kappa) <= MAX_ALAT:
                by_speed.setdefault(target, []).append((steer_sign * steer, kappa, v))
                entry["valid"] = True
        cell_rows.append(entry)
        excite = probe_common.to_int(cell["excite_start_step"])         # empty when the cell never reached it
        response = step_response(steps, phase, excite, steer_sign, dt) if excite is not None else None
        if response is not None:
            bins.append({"source": "E-02a %s" % phase, "target_kmh": target, "amplitude": abs(steer),
                         "steer": steer, "structures": response})
    speeds, slopes, offsets, fits = [], [], [], []
    usable = True
    for target in sorted({float(c["target_kmh"]) for c in cells}):
        valid = by_speed.get(target, [])
        line = e01_actuation.fit_line([(c, k) for c, k, _ in valid])
        enough = len(valid) >= MIN_CELLS
        good = line is not None and line[2] is not None and line[2] >= MIN_R2 and line[0] >= MIN_KAPPA_PER_STEER
        usable = usable and enough and good
        fit = {"target_kmh": target, "valid_cells": len(valid)}
        if line is not None:
            a, b, r2, rms = line
            fit.update({"speed_mps": sum(v for _, _, v in valid) / len(valid), "slope": a, "offset": b, "r2": r2,
                        "residual_rms": rms})
            speeds.append(fit["speed_mps"])
            slopes.append(a)
            offsets.append(b)
        fits.append(fit)
    if not usable:
        raise Refusal("E-02a fit not usable by llc_v1 (every speed needs >= %d valid cells, R2 >= %.2f and a slope of "
                      "at least %g 1/m per unit steer in the contract sign): %s"
                      % (MIN_CELLS, MIN_R2, MIN_KAPPA_PER_STEER, json.dumps(fits)))
    order = sorted(range(len(speeds)), key=lambda i: speeds[i])
    return {"speeds_mps": [speeds[i] for i in order], "kappa_per_steer": [slopes[i] for i in order],
            "kappa_offset": [offsets[i] for i in order], "fits": fits, "cells": cell_rows, "bins": bins}


def step_response(steps, phase, excite_start, steer_sign, dt):
    """Both structures on a cell's step (steer 0 -> s at the excite start), chosen by BIC; None when too short."""
    cmd = steps.series(phase, "cmd_steer")
    speed = steps.series(phase, "speed_kmh")
    rate = heading_rate_rad(steps)
    u, kappa = [], []
    for s in range(excite_start - STEP_FIT_BEFORE, excite_start + STEP_FIT_AFTER):
        v = (speed.get(s) or 0.0) / 3.6
        if s not in rate or cmd.get(s) is None or v <= 1.0:
            continue
        u.append(steer_sign * cmd[s])
        kappa.append(rate[s] / v)
    if len(u) < STEP_FIT_BEFORE + 2 * MAX_LAG_STEPS:
        return None
    return fit_structures(u, kappa, range(len(u)), None, dt)


# ---------------------------------------------------------------------------------------------------------------
# the card


def llc_parameters(longitudinal):
    """llc_v1's parameters: LLC_DEFAULTS with the fitted throttle and brake gains. The steering rate limiter stays the
    identity, which satisfies the CC-08 rule whatever the engine path does (echo, a rate limit, a lag)."""
    params = dict(LLC_DEFAULTS)
    if longitudinal:
        params["throttle_per_mps2"] = longitudinal["throttle"]["per_mps2"]
        params["brake_per_mps2"] = longitudinal["brake"]["per_mps2"]
    return params


def build_identified(args):
    e01_play = open_play(args.e01, "e01")
    e02a_play = open_play(args.e02a, "e02a")
    plays = [e01_play, e02a_play]
    e04_play = open_play(args.e04, "e04") if args.e04 else None
    if e04_play:
        plays.append(e04_play)
    pin_id, build_key = build_identity(plays, args.build_key)
    prefab = vehicle_prefab(plays, args.vehicle)
    dt = time_slice(plays)
    try:
        steps = e01_actuation.Steps(e01_play.csv("e01_steps", required=e01_actuation.STEP_COLUMNS))
        phases_csv = e01_play.csv("e01_phases", required=("phase", "result", "amplitude"))
        steer_sign, yaw_axis_sign, sign_detail = fit_signs(steps)
        prbs = fit_prbs(steps, steer_sign, dt)
        engine_input = input_path(steps, dt)
        longitudinal = fit_longitudinal(steps, phases_csv)
        automatic, forward, propulsion, gear_detail = fit_gearbox(steps)
        kappa_ss = fit_kappa_ss(e02a_play, steer_sign, dt)
    except probe_common.InputError as e:
        raise Refusal(str(e))
    wrong = [f for f in kappa_ss["fits"] if f.get("slope", 1.0) <= 0]
    if wrong:
        raise Refusal("E-02a slopes are not positive in the contract sign of steer_sign %d: %s" % (steer_sign, wrong))
    gearbox = {"mode": "automatic" if automatic else "scripted", "basis": "E-01 gear_observe (%s)" % (
        "upshifts without SetGear" if automatic else "no upshift without SetGear: scripted gearbox needed"),
        "has_automatic_gearbox": None, "scripted_gears": [], "upshift_mps": [], "downshift_mps": [],
        "clutch_dip_s": 0.0, "detail": gear_detail}
    if e04_play:
        flag = e04_play.header.get("has_automatic_gearbox")
        kv = os.path.join(e04_play.dir, "e04_vehicle.txt")
        if os.path.isfile(kv):
            flag = probe_common.read_kv(kv).get("has_automatic_gearbox", flag)
        gearbox["has_automatic_gearbox"] = None if flag in (None, "") else flag in ("1", "true", "True")
    if not automatic:
        if not (args.scripted_gears and args.upshift_mps and args.downshift_mps):
            raise Refusal("E-01 says a scripted gearbox is needed: pass --scripted-gears, --upshift-mps and "
                          "--downshift-mps (forward gears only)")
    if not forward:
        raise Refusal("no forward gear index seen while driving (input_post_gear >= 5 km/h)")
    if not automatic:
        gearbox["scripted_gears"], gearbox["upshift_mps"], gearbox["downshift_mps"] = scripted_gearbox(
            args, forward, propulsion)
        gearbox["clutch_dip_s"] = clutch_dip(e01_play)
    vehicle = args.vehicle
    sessions = [session_id(p) for p in plays]
    return {
        "schema": CARD_SCHEMA,
        "card_id": vehicle,
        "vehicle": vehicle,
        "prefab": prefab,
        "status": "identified",
        "pin_id": pin_id,
        "build_key": build_key,
        "identification_sessions": sessions,
        "use": "llc_v1 vehicle card; build-scoped: refused by llc_v1 on a session of another build_key",
        "note": "first-pass fit (DD-7-min) of the E-01 / E-02a probe plays; E-02Q and CC-08 (DD-7-ident, DD-14b) are the "
                "full identification",
        "generator": {"tool": GENERATOR, "version": GENERATOR_VERSION},
        "signs": {"steer_sign": steer_sign, "yaw_axis_sign": yaw_axis_sign, "measured": True, "detail": sign_detail},
        "gear_indices_forward": forward,
        "gear_index_neutral": None,
        "gear_index_reverse": None,
        "gearbox": gearbox,
        "kappa_ss": {"speeds_mps": kappa_ss["speeds_mps"], "kappa_per_steer": kappa_ss["kappa_per_steer"],
                     "kappa_offset": kappa_ss["kappa_offset"],
                     "basis": "E-02a steady cells: kappa = kappa_per_steer x contract steer + kappa_offset per speed",
                     "fits": kappa_ss["fits"], "cells": kappa_ss["cells"]},
        "actuator": {
            "raw_engine_path": dict(engine_input, selected_structure=prbs["structures"]["selected"],
                                    dead_time_steps=prbs["structures"][prbs["structures"]["selected"]]["dead_time_steps"],
                                    bins=[prbs] + kappa_ss["bins"]),
            "full_llc_path": {"structure": None, "bins": [],
                              "note": "kappa target -> kappa realised needs llc_v1 sessions (CC-08, DD-7-ident)"},
        },
        "longitudinal": longitudinal,
        "llc_v1": llc_parameters(longitudinal),
        "mode_a": {"force_enable_simulation": force_enable_simulation(phases_csv)},
        "fit": {"dt_s": dt, "min_cells": MIN_CELLS, "min_r2": MIN_R2, "max_alat_mps2": MAX_ALAT,
                "usable": True},
    }


def build_placeholder(args):
    if not args.pin_id or not PIN_RE.match(args.pin_id):
        raise Refusal("--placeholder needs --pin-id (the pin the card is placed under)")
    vehicle = args.vehicle
    return {
        "schema": CARD_SCHEMA,
        "card_id": vehicle,
        "vehicle": vehicle,
        "prefab": PREFABS.get(vehicle, ""),
        "status": "placeholder",
        "pin_id": args.pin_id,
        "build_key": None,
        "identification_sessions": [],
        "use": "PLACEHOLDER: qualification only. llc_v1 accepts it at use_class dev or probe and refuses it at any "
               "other use_class, production included",
        "note": "PLACEHOLDER card (DD-7-min): no E-01 / E-02a play exists yet (the user's probe Plays are pending). Signs "
                "are the TBD-5 draft defaults, kappa_ss is guessed, the actuator is unidentified. Replace it with "
                "fit_steer_kappa.py --e01 --e02a once the plays exist; data driven with it is qualification-only.",
        "generator": {"tool": GENERATOR, "version": GENERATOR_VERSION},
        "signs": {"steer_sign": PLACEHOLDER_SIGNS[0], "yaw_axis_sign": PLACEHOLDER_SIGNS[1], "measured": False,
                  "detail": "dc-2.0 TBD-5 draft: CarSteering +1 = left (steer_sign -1); +angvel_local_y clockwise"},
        "gear_indices_forward": [],
        "gear_index_neutral": None,
        "gear_index_reverse": None,
        "gearbox": {"mode": "automatic", "basis": "dc-2.0 TBD-7 draft (HasAutomaticGearbox, driverless); E-04 / E-01 "
                    "decide", "has_automatic_gearbox": None, "scripted_gears": [], "upshift_mps": [],
                    "downshift_mps": [], "clutch_dip_s": 0.0},
        "kappa_ss": {"speeds_mps": list(PLACEHOLDER_SPEEDS_MPS), "kappa_per_steer": list(PLACEHOLDER_KAPPA_PER_STEER),
                     "kappa_offset": [0.0] * len(PLACEHOLDER_SPEEDS_MPS),
                     "basis": "GUESSED (E-02a grid speeds 15 / 30 / 50 km/h); not a fit", "fits": [], "cells": []},
        "actuator": {
            "raw_engine_path": {"input_echo": None, "input_shaping": "unknown", "steer_rate_limit_per_s": None,
                                "input_lag_tau_s": None, "input_max_slope_per_s": None, "shaping_fit": None,
                                "selected_structure": None, "dead_time_steps": None, "bins": []},
            "full_llc_path": {"structure": None, "bins": [],
                              "note": "kappa target -> kappa realised needs llc_v1 sessions (CC-08, DD-7-ident)"},
        },
        "longitudinal": None,
        "llc_v1": llc_parameters(None),
        "mode_a": {"force_enable_simulation": None},
        "fit": None,
    }


def clean(value):
    """Floats rounded to 6 decimals without exponent notation (DC_JsonScan reads numbers with string.ToFloat)."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Refusal("a card value is not finite")
        out = round(value, 6)
        if out != 0.0 and abs(out) < 1e-4:
            out = round(value, 4)
        return 0.0 if out == 0.0 else out
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def exponent_numbers(value):
    """The floats of a card whose JSON form uses an exponent."""
    if isinstance(value, float):
        return [value] if "e" in json.dumps(value).lower() else []
    if isinstance(value, dict):
        return [x for v in value.values() for x in exponent_numbers(v)]
    if isinstance(value, list):
        return [x for v in value for x in exponent_numbers(v)]
    return []


def enforce_load_check(card):
    """The violations of the rules llc_v1 applies to a card as written (DC_VehicleCard.Check with CheckMap,
    CheckGearbox and CheckActuator, and the limiter rule of DC_LLCv1.AcceptCard); empty when the engine loads it."""
    out = []
    if card.get("schema") != CARD_SCHEMA or not all(isinstance(card.get(k), str) and card.get(k)
                                                    for k in ("card_id", "vehicle", "status", "pin_id")):
        out.append("schema %s and the strings card_id, vehicle, status, pin_id are required" % CARD_SCHEMA)
    status, sessions = card.get("status"), card.get("identification_sessions")
    if status not in ("placeholder", "identified") or not isinstance(sessions, list):
        out.append("status %r is neither placeholder nor identified, or identification_sessions is no list" % status)
    elif (status == "placeholder" and sessions) or (status == "identified" and not (sessions and card.get("build_key"))):
        out.append("a placeholder names no identification session; an identified card needs them and a build_key")
    signs = card.get("signs") or {}
    if signs.get("steer_sign") not in (-1, 1) or signs.get("yaw_axis_sign") not in (-1, 1) \
            or not isinstance(signs.get("measured"), bool):
        out.append("signs: steer_sign and yaw_axis_sign -1 or +1, measured a bool")
    ss = card.get("kappa_ss") or {}
    speeds, gains = ss.get("speeds_mps") or [], ss.get("kappa_per_steer") or []
    if not speeds or len(gains) != len(speeds) or len(ss.get("kappa_offset") or []) != len(speeds):
        out.append("kappa_ss needs speeds_mps, kappa_per_steer and kappa_offset of one length >= 1")
    if any(v < 0 for v in speeds) or any(b <= a for a, b in zip(speeds, speeds[1:])):
        out.append("kappa_ss.speeds_mps must be >= 0 and strictly increasing")
    if any(g <= 0 for g in gains):
        out.append("kappa_ss.kappa_per_steer must be > 0 as written: %s" % gains)
    out += gearbox_violations(card)
    raw = (card.get("actuator") or {}).get("raw_engine_path") or {}
    shaping, limit, echo = raw.get("input_shaping"), raw.get("steer_rate_limit_per_s"), raw.get("input_echo")
    if shaping not in INPUT_SHAPINGS:
        out.append("input_shaping %r is none of %s" % (shaping, ", ".join(INPUT_SHAPINGS)))
    if (shaping == "rate_limit") != (limit is not None and limit > 0):
        out.append("steer_rate_limit_per_s (%r) is a positive number exactly when input_shaping is rate_limit" % limit)
    if (echo is not None and echo != (shaping == "echo")) or (echo is None and shaping != "unknown"):
        out.append("input_echo %r and input_shaping %r disagree" % (echo, shaping))
    llc = card.get("llc_v1") or {}
    if any(not isinstance(llc.get(k), float) for k, _ in LLC_DEFAULTS):
        return out + ["llc_v1 misses a parameter or writes one without a decimal point"]
    ok = 0 < llc["steer_max"] <= 1 and 0 < llc["throttle_max"] <= 1 and 0 <= llc["brake_hold"] <= 1
    ok = ok and llc["speed_window_mps"] > 0 and llc["throttle_per_mps2"] > 0 and llc["brake_per_mps2"] > 0
    ok = ok and all(llc[k] >= 0 for k in ("steer_rate_per_s", "steer_lag_tau_s", "kappa_trim_gain", "kappa_trim_max",
                                          "trim_min_speed_mps", "speed_kp", "speed_ki", "integral_max_mps2",
                                          "coast_band_mps2", "stop_speed_mps"))
    if not ok:
        out.append("an llc_v1 parameter is outside the range DC_VehicleCard.Check allows")
    rate = llc["steer_rate_per_s"]
    rate_limit = shaping == "rate_limit" and limit is not None and limit > 0
    if rate > 0 and not ((echo is True and shaping == "echo") or rate_limit):
        out.append("llc steer_rate_per_s %s needs an echoing engine input or an identified rate limit" % rate)
    elif rate > 0 and rate_limit and rate >= limit:
        out.append("llc steer_rate_per_s %s is not slower than the engine limit %s /s" % (rate, limit))
    return out


def gearbox_violations(card):
    """DC_VehicleCard.CheckGearbox on a card as written."""
    gearbox = card.get("gearbox") or {}
    dip, mode = gearbox.get("clutch_dip_s"), gearbox.get("mode")
    if not isinstance(dip, (int, float)) or isinstance(dip, bool) or not 0 <= dip <= CLUTCH_DIP_MAX_S:
        return ["gearbox.clutch_dip_s %r is not a number in [0, %s] s" % (dip, CLUTCH_DIP_MAX_S)]
    if mode == "automatic":
        return ["gearbox.clutch_dip_s applies to a scripted gearbox only"] if dip > 0 else []
    if mode != "scripted":
        return ["gearbox.mode %r is neither automatic nor scripted" % mode]
    gears, up, down = gearbox.get("scripted_gears") or [], gearbox.get("upshift_mps") or [], \
        gearbox.get("downshift_mps") or []
    if not gears or len(up) != len(gears) - 1 or len(down) != len(gears) - 1 \
            or any(d >= u for d, u in zip(down, up)) or any(b <= a for a, b in zip(up, up[1:])):
        return ["scripted gearbox: n >= 1 gears, n - 1 increasing upshifts, downshift < upshift"]
    forward = card.get("gear_indices_forward") or []
    if any(g < 0 or g != round(g) or g not in forward for g in gears):
        return ["scripted gears %s are not all in gear_indices_forward %s" % (gears, forward)]
    return []


def card_text(card):
    cleaned = clean(card)
    problems = enforce_load_check(cleaned)
    if problems:
        raise Refusal("llc_v1 would refuse this card as written: %s" % "; ".join(problems))
    text = json.dumps(cleaned, indent=2, ensure_ascii=True) + "\n"
    for key in ENGINE_SCANNED_KEYS:
        if text.count('"%s"' % key) != 1:
            raise Refusal("the card names \"%s\" %d times; llc_v1 scans it by key and needs it once"
                          % (key, text.count('"%s"' % key)))
    if exponent_numbers(cleaned):
        raise Refusal("card numbers %s use an exponent; llc_v1 reads numbers with string.ToFloat"
                      % exponent_numbers(cleaned)[:3])
    return text


def default_out(card):
    return os.path.join(CARDS_DIR, card["pin_id"], card["card_id"] + ".json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="First-pass steer -> curvature fit and the llc_v1 vehicle card (DD-7-min)")
    ap.add_argument("--e01", help="E-01 play directory")
    ap.add_argument("--e02a", help="E-02a play directory")
    ap.add_argument("--e04", help="E-04 play directory (optional: has_automatic_gearbox)")
    ap.add_argument("--placeholder", action="store_true", help="write the PLACEHOLDER card (no plays)")
    ap.add_argument("--pin-id", help="pin of the placeholder card")
    ap.add_argument("--vehicle", default=DEFAULT_VEHICLE, help="card id and vehicle (default %(default)s)")
    ap.add_argument("--build-key", help="build_key of the plays when their run headers carry none (dd-launch/1)")
    ap.add_argument("--scripted-gears", help="forward gear indices of a scripted gearbox, comma-separated")
    ap.add_argument("--upshift-mps", help="upshift speeds [m/s] between them, comma-separated")
    ap.add_argument("--downshift-mps", help="downshift speeds [m/s], comma-separated")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--out", nargs="?", const="", help="write the card (default tools/ident/cards/<pin>/<id>.json)")
    group.add_argument("--check", help="compare the card with this file")
    args = ap.parse_args(argv)
    try:
        if not re.match(r"^[A-Za-z0-9_]{1,32}$", args.vehicle):
            raise Refusal("--vehicle %r is not a card id" % args.vehicle)
        if args.placeholder:
            if args.e01 or args.e02a or args.e04:
                raise Refusal("--placeholder takes no play")
            card = build_placeholder(args)
        else:
            if not (args.e01 and args.e02a):
                raise Refusal("an identified card needs --e01 and --e02a (or --placeholder)")
            card = build_identified(args)
        text = card_text(card)
        if args.out is not None:
            out = check_out_path(args.out or default_out(card))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="ascii", newline="") as f:
                f.write(text)
    except Refusal as e:
        print("REFUSED: %s" % e)
        return 2
    print("card %s (%s) pin %s build_key %s sessions %s" % (card["card_id"], card["status"], card["pin_id"],
                                                           card["build_key"], card["identification_sessions"]))
    print("  signs steer %+d yaw_axis %+d (measured %s); gearbox %s (clutch dip %s s); forward gears %s" % (
        card["signs"]["steer_sign"], card["signs"]["yaw_axis_sign"], card["signs"]["measured"], card["gearbox"]["mode"],
        card["gearbox"]["clutch_dip_s"], card["gear_indices_forward"]))
    for v, g, b in zip(card["kappa_ss"]["speeds_mps"], card["kappa_ss"]["kappa_per_steer"], card["kappa_ss"]["kappa_offset"]):
        print("  kappa_ss %.2f m/s: %.5f 1/m per steer, offset %+.6f 1/m" % (v, g, b))
    for fit in (card["kappa_ss"].get("fits") or []):
        print("  fit %g km/h: R2 %s, residual RMS %s 1/m, %d valid cells" % (
            fit["target_kmh"], e01_actuation.fmt(fit.get("r2"), 4), e01_actuation.fmt(fit.get("residual_rms"), 6),
            fit["valid_cells"]))
    raw = card["actuator"]["raw_engine_path"]
    print("  raw engine path: input echo %s, shaping %s (rate limit %s /s, lag tau %s s, max slope %s /s), structure "
          "%s, dead time %s steps" % (raw["input_echo"], raw["input_shaping"], raw["steer_rate_limit_per_s"],
                                      raw["input_lag_tau_s"], raw["input_max_slope_per_s"], raw["selected_structure"],
                                      raw["dead_time_steps"]))
    if args.check:
        with open(args.check, "r", encoding="ascii", newline="") as f:
            if f.read() != text:
                print("DIFFERS: %s is not the card these inputs give" % args.check)
                return 1
        print("check: %s is identical" % args.check)
    elif args.out is not None:
        print("wrote %s" % (args.out or default_out(card)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
