"""C15: dynamics consistency of the ego telemetry and the steering inputs."""

import math

import numpy

from .base import FAIL, HARD, NOT_EVALUATED, PASS, first, result, unreadable

SPEED_TOL_MPS = 0.3         # r1 [TL] 5.1 C15: median |speed - |dpos|/dt| (r2 keeps "speed vs dpos/dt")
ACC_TOL_MPS2 = 0.5          # r1 [TL] 5.1 C15: median |acc_fwd - dv/dt|
EVENT_GUARD_S = 1.0         # r1 [TL] 5.1 C15: no event within +-1 s
DISCONTINUITY_EVENTS = ("episode_start", "episode_end", "perturb_start", "perturb_end", "teleport", "respawn",
                        "stuck", "collision", "camera_reset", "pause", "resume", "env_change")
MIN_SAMPLES = 20


def fit_steer_gain(kappa, steer):
    """Least-squares gain g of kappa = g x steer (through the origin); None without steering."""
    den = float(numpy.dot(steer, steer))
    return float(numpy.dot(kappa, steer)) / den if den > 1e-9 else None


def steer_series(s, steps, delay):
    """Contract-sign steering input (ctl_export.steer_input) at physics step p - delay for each p, NaN when the
    ctl row is missing."""
    index = s.streams["ctl"].index()
    steer = s.ctl_export["steer_input"]
    out = numpy.full(len(steps), numpy.nan)
    for i, p in enumerate(steps):
        j = index.get(p - delay)
        if j is not None and steer[j] is not None:
            out[i] = steer[j]
    return out


def eligible_rows(s, min_speed, max_alat_frac):
    """Telemetry row indices with speed > min_speed, |a_lat| < max_alat_frac x the episode's style_alat_max_mps2
    (nominal 2.75) and no discontinuity event within +-1 s."""
    tel = s.streams["telemetry"]
    texp = s.telemetry_export
    ev = s.streams["events"]
    guard_ms = EVENT_GUARD_S * 1000.0
    event_ms = [w for kind, w in zip(ev.col("kind"), ev.col("world_time_ms")) if kind in DISCONTINUITY_EVENTS]
    alat_max = {e: a for e, a in zip(s.streams["episodes"].col("episode_id"),
                                     s.streams["episodes"].col("style_alat_max_mps2"))}
    rows = []
    for i, (w, e) in enumerate(zip(tel.col("world_time_ms"), tel.col("episode_id"))):
        v = texp["speed_fwd_mps"][i]
        if not (v > min_speed):
            continue
        if abs(texp["acc_lat_mps2"][i]) >= max_alat_frac * alat_max.get(e, 2.75):
            continue
        if any(abs(w - t) <= guard_ms for t in event_ms):
            continue
        rows.append(i)
    return rows


def check_c15(ctx):
    """Dynamics (r2 5.1 C15) on telemetry rows with speed > C15.min_speed_mps, |a_lat| < C15.max_alat_frac x
    a_lat,max and no event within +-1 s:
      * yaw model: yaw_rate vs v x kappa_ss(steer_eff(t - delta)), steer_eff = the contract-sign input_post_steer
        with the lag delta (0..N physics steps) that fits best; kappa_ss = g x steer_eff with g least-squares fitted
        on the session (no vehicle card with a steady-state map exists before DD-7-min). g must be > 0 (right steer
        gives right yaw: catches a wrong steer_sign); residual RMS <= C15.resid_rms_rps_max or <= C15.resid_rel_max
        x the yaw-rate RMS;
      * speed vs horizontal |dpos| / dt and acc_fwd vs dv/dt between consecutive telemetry rows: medians within
        0.3 m/s and 0.5 m/s2 (r1 values; r2 5.2 lists none)."""
    if ctx.load_error:
        return unreadable(ctx, "C15")
    s = ctx.session
    thr = ctx.thr("C15")
    tel = s.streams["telemetry"]
    texp = s.telemetry_export
    rows = eligible_rows(s, thr.get("min_speed_mps", 3), thr.get("max_alat_frac", 0.8))
    if len(rows) < MIN_SAMPLES:
        return result(ctx, "C15", HARD, NOT_EVALUATED, {"samples": len(rows)}, thr,
                      "%d telemetry rows above %s m/s without events (< %d)" % (len(rows), thr.get("min_speed_mps"),
                                                                               MIN_SAMPLES))
    steps = [tel.col("physics_step")[i] for i in rows]
    v = numpy.array([texp["speed_fwd_mps"][i] for i in rows])
    yaw = numpy.array([texp["yaw_rate_rps"][i] for i in rows])
    best = None
    for delay in range(0, s.clocks["N"] + 1):
        steer = steer_series(s, steps, delay)
        ok = ~numpy.isnan(steer)
        if ok.sum() < MIN_SAMPLES:
            continue
        g = fit_steer_gain(yaw[ok] / v[ok], steer[ok])
        if g is None:
            continue
        resid = yaw[ok] - v[ok] * g * steer[ok]
        rms = float(numpy.sqrt(numpy.mean(resid ** 2)))
        if best is None or rms < best[0]:
            best = (rms, delay, g, int(ok.sum()))
    issues = []
    signal = float(numpy.sqrt(numpy.mean(yaw ** 2)))
    value = {"samples": len(rows), "yaw_signal_rms": signal}
    if best is None:
        issues.append("no steering input to fit (no llc rows at the telemetry steps)")
    else:
        rms, delay, g, n = best
        value.update({"yaw_resid_rms": rms, "lag_steps": delay, "kappa_per_steer": g})
        if g <= 0:
            issues.append("fitted curvature per contract steer %.4f <= 0: right steering does not yaw right "
                          "(steer_sign?)" % g)
        if not (rms <= thr.get("resid_rms_rps_max", 0.03) or rms <= thr.get("resid_rel_max", 0.15) * signal):
            issues.append("yaw residual RMS %.4f rad/s > %s and > %s x %.4f" % (
                rms, thr.get("resid_rms_rps_max"), thr.get("resid_rel_max"), signal))
    dt = s.clocks["k"] * s.clocks["slice_s"]
    index = {tel.col("physics_step")[i]: i for i in range(len(tel))}
    speed_err, acc_err = [], []
    x, z = tel.array("pos_x"), tel.array("pos_z")
    row_set = set(rows)
    for i in rows:
        p = tel.col("physics_step")[i]
        j = index.get(p - s.clocks["k"])
        if j is None or j not in row_set:
            continue
        dist = math.hypot(x[i] - x[j], z[i] - z[j])
        v_mean = 0.5 * (texp["speed_fwd_mps"][i] + texp["speed_fwd_mps"][j])
        speed_err.append(abs(abs(v_mean) - dist / dt))
        dv = (texp["speed_fwd_mps"][i] - texp["speed_fwd_mps"][j]) / dt
        acc_err.append(abs(0.5 * (texp["acc_long_mps2"][i] + texp["acc_long_mps2"][j]) - dv))
    if speed_err:
        value["speed_err_median"] = float(numpy.median(speed_err))
        value["acc_err_median"] = float(numpy.median(acc_err))
        if value["speed_err_median"] > SPEED_TOL_MPS:
            issues.append("median |speed - dpos/dt| %.3f m/s > %s" % (value["speed_err_median"], SPEED_TOL_MPS))
        if value["acc_err_median"] > ACC_TOL_MPS2:
            issues.append("median |acc_fwd - dv/dt| %.3f m/s2 > %s" % (value["acc_err_median"], ACC_TOL_MPS2))
    detail = first(issues) if issues else ("yaw residual %.4f rad/s (signal %.3f) at lag %d steps, gain %.4f 1/m per "
                                           "steer (session fit); speed err %.3f m/s, acc err %.3f m/s2 over %d rows"
                                           % (best[0], signal, best[1], best[2], value.get("speed_err_median", 0),
                                              value.get("acc_err_median", 0), len(rows)))
    return result(ctx, "C15", HARD, FAIL if issues else PASS, value, thr, detail,
                  not_evaluated=["vehicle-card kappa_ss map (DD-7-min): gain fitted on the session",
                                 "slip flag (not defined in dc-2.0)"])
