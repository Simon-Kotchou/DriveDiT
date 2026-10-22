"""C17: behavioural profile of mode-E expert episodes against their route (r2 5.1 C17)."""

import math

import numpy

from .base import FAIL, HARD, INFO, NA, NOT_EVALUATED, PASS, SOFT, first, percentile, result, unreadable, \
    wrap_angle
from .dynamics import fit_steer_gain, steer_series

MOVING_MPS = 1.0            # r1 [TL] 5.1 C17: moving = speed > 1 m/s
CURV_HALF_WINDOW_M = 5.0    # route curvature: heading change over +-5 m of arc length
MIN_TURN_RAD = 0.2          # below this much turning (session and route) the L/R comparison has no signal


def route_curvature(route, s):
    """Signed curvature (1/m, right-positive) of a drivedit_nav Route at arc length s: the tangent heading change
    over [s - 5 m, s + 5 m] divided by that length (heading = atan2(dx, dz), clockwise positive)."""
    def tangent(t):
        a = route.point_at(max(t - 0.5, 0.0))
        b = route.point_at(min(t + 0.5, route.length_m))
        return math.atan2(b[0] - a[0], b[1] - a[1])
    lo = max(s - CURV_HALF_WINDOW_M, 0.0)
    hi = min(s + CURV_HALF_WINDOW_M, route.length_m)
    if hi - lo < 1e-6:
        return 0.0
    return wrap_angle(tangent(hi) - tangent(lo)) / (hi - lo)


def left_share(turns):
    """Share of the total absolute turning that is to the left (negative, right-positive convention); None when
    there is no turning."""
    total = float(numpy.abs(turns).sum())
    if total <= 0:
        return None
    return float(numpy.abs(turns[turns < 0]).sum()) / total


def severity_for(ctx):
    """SOFT at T1 (calibration), HARD at T2-T3 for mode E, INFO elsewhere (r2 5.1 C17)."""
    if ctx.tier == "T1":
        return SOFT
    if ctx.tier in ("T2", "T3"):
        return HARD
    return INFO


def check_c17(ctx):
    """Behavioural profile of the mode-E expert episodes after the C17.warmup_s warm-up (world time since the
    session start), r2 5.1 C17 with the C17 thresholds:
      * moving fraction (speed > 1 m/s) >= moving_frac_min; speed p50 / p95 reported;
      * stuck events per 30 min <= stuck_per_30min_max; road adherence P(road_dist_m <= road_width_m / 2 + 1 m)
        >= road_adherence_min;
      * against the route of each episode (drivedit_nav route schema; --route / --routes-dir; a filled
        episodes.route_sha256 must match): progress by drivedit_nav.nav_sequence over the telemetry poses, the
        route's signed curvature kappa_r at each row's progress (+-5 m), and per row the session turning
        yaw_rate x dt and the route-expected turning kappa_r x |v| x dt.
          L/R: |left share of the session's turning - left share of the route-expected turning| <=
          lr_ratio_dev_from_route_max (shares of total absolute turning, bounded where a ratio would not be);
          steering mean: |mean contract steer_input - mean(kappa_r / g)| <= steer_mean_dev_from_route_max, g the
          curvature per contract steer fitted on the session's rows above 3 m/s (no vehicle-card map yet).
    Mode-E qualification only: SOFT at T1, HARD at T2-T3, INFO elsewhere; N/A without mode-E episodes."""
    if ctx.load_error:
        return unreadable(ctx, "C17", INFO)
    import drivedit_nav
    s = ctx.session
    thr = ctx.thr("C17")
    severity = severity_for(ctx)
    ep = s.streams["episodes"]
    episodes = {e: i for i, e in enumerate(ep.col("episode_id"))
                if ep.col("mode")[i] == "E" and ep.col("driver")[i] == "expert"}
    if not episodes:
        return result(ctx, "C17", severity, NA, None, thr, "no mode-E expert episode")
    tel = s.streams["telemetry"]
    texp = s.telemetry_export
    warm_ms = float(thr.get("warmup_s", 60)) * 1000.0
    dt = s.clocks["k"] * s.clocks["slice_s"]
    rows = []
    turn_s, turn_r, steer_rows = [], [], []
    notes = []
    compared = 0
    for e, ei in sorted(episodes.items()):
        idx = [i for i, ee in enumerate(tel.col("episode_id")) if ee == e]
        after = [i for i in idx if tel.col("world_time_ms")[i] >= warm_ms]
        rows += after
        if not after:
            continue
        rid = ep.col("route_id")[ei]
        if not rid:
            notes.append("episode %d has no route_id" % e)
            continue
        route, _, err = ctx.route(rid)
        if route is None:
            notes.append(err)
            continue
        want = ep.col("route_sha256")[ei]
        if want and want != route.sha256:
            notes.append("episode %d: route file %s does not match route_sha256" % (e, rid))
            continue
        poses = [(tel.col("pos_x")[i], tel.col("pos_z")[i], float(texp["heading_rad"][i])) for i in idx]
        progress = {i: nav["progress_m"] for i, nav in zip(idx, drivedit_nav.nav_sequence(route, poses))}
        steer = steer_series(s, [tel.col("physics_step")[i] for i in after], 0)
        for n, i in enumerate(after):
            v = float(texp["speed_fwd_mps"][i])
            k_r = route_curvature(route, progress[i])
            turn_s.append(float(texp["yaw_rate_rps"][i]) * dt)
            turn_r.append(k_r * abs(v) * dt)
            steer_rows.append((float(steer[n]), v, float(texp["yaw_rate_rps"][i]), k_r))
        compared += 1
    if not rows:
        return result(ctx, "C17", severity, NA, None, thr, "no mode-E telemetry after the %.0f s warm-up"
                      % (warm_ms / 1000.0))
    v_all = numpy.array([texp["speed_fwd_mps"][i] for i in rows])
    value = {"rows": len(rows), "moving_frac": float(numpy.mean(v_all > MOVING_MPS)),
             "speed_p50": percentile(list(v_all), 50), "speed_p95": percentile(list(v_all), 95)}
    issues = []
    if value["moving_frac"] < thr.get("moving_frac_min", 0.7):
        issues.append("moving fraction %.3f < %s" % (value["moving_frac"], thr.get("moving_frac_min")))
    ev = s.streams["events"]
    stuck = sum(1 for kind, e, w in zip(ev.col("kind"), ev.col("episode_id"), ev.col("world_time_ms"))
                if kind == "stuck" and e in episodes and w >= warm_ms)
    duration_s = len(rows) * dt
    value["stuck_per_30min"] = stuck * 1800.0 / duration_s if duration_s > 0 else None
    if value["stuck_per_30min"] is not None and value["stuck_per_30min"] > thr.get("stuck_per_30min_max", 2):
        issues.append("%d stuck event(s) = %.2f per 30 min > %s" % (stuck, value["stuck_per_30min"],
                                                                    thr.get("stuck_per_30min_max")))
    road = [(tel.col("road_dist_m")[i], tel.col("road_width_m")[i]) for i in rows]
    road = [(d, w) for d, w in road if d is not None and w is not None]
    if road:
        value["road_adherence"] = sum(1 for d, w in road if d <= w / 2.0 + 1.0) / float(len(road))
        if value["road_adherence"] < thr.get("road_adherence_min", 0.95):
            issues.append("road adherence %.3f < %s" % (value["road_adherence"], thr.get("road_adherence_min")))
    skipped = []
    if compared == 0:
        skipped.append("route comparison (%s)" % first(notes, 2) if notes else "route comparison")
    else:
        ts, tr = numpy.array(turn_s), numpy.array(turn_r)
        if numpy.abs(ts).sum() < MIN_TURN_RAD and numpy.abs(tr).sum() < MIN_TURN_RAD:
            skipped.append("L/R and steering mean (route and session turn less than %.1f rad)" % MIN_TURN_RAD)
        else:
            ls, lr = left_share(ts), left_share(tr)
            value.update({"left_share_session": ls, "left_share_route": lr})
            if ls is None or lr is None:
                issues.append("no turning in the %s" % ("session" if ls is None else "route"))
            else:
                value["lr_dev"] = abs(ls - lr)
                if value["lr_dev"] > thr.get("lr_ratio_dev_from_route_max", 0.25):
                    issues.append("L/R deviation %.3f > %s (left share: session %.3f, route %.3f)" % (
                        value["lr_dev"], thr.get("lr_ratio_dev_from_route_max"), ls, lr))
            sr = numpy.array(steer_rows)
            ok = ~numpy.isnan(sr[:, 0])
            fast = ok & (sr[:, 1] > 3.0)
            g = fit_steer_gain(sr[fast, 2] / sr[fast, 1], sr[fast, 0]) if fast.sum() >= 10 else None
            if g is None or g <= 0:
                skipped.append("steering mean (no positive curvature-per-steer gain to fit)")
            else:
                mean_s = float(numpy.mean(sr[ok, 0]))
                mean_r = float(numpy.mean(sr[ok, 3] / g))
                value.update({"steer_mean_session": mean_s, "steer_mean_route": mean_r,
                              "steer_mean_dev": abs(mean_s - mean_r), "kappa_per_steer": g})
                if value["steer_mean_dev"] > thr.get("steer_mean_dev_from_route_max", 0.03):
                    issues.append("steering mean deviation %.4f > %s (session %.4f, route-expected %.4f)" % (
                        value["steer_mean_dev"], thr.get("steer_mean_dev_from_route_max"), mean_s, mean_r))
        if notes:
            skipped.append("route comparison of some episodes (%s)" % first(notes, 2))
    if issues:
        status = FAIL
    elif compared == 0:
        status = NOT_EVALUATED
    else:
        status = PASS
    detail = first(issues) if issues else "%d rows of %d mode-E episode(s): moving %.3f%s" % (
        len(rows), len(episodes), value["moving_frac"],
        "; L/R dev %.3f, steering mean dev %.4f" % (value.get("lr_dev", float("nan")),
                                                    value.get("steer_mean_dev", float("nan")))
        if compared else "; route comparison not run")
    return result(ctx, "C17", severity, status, value, thr, detail, not_evaluated=skipped)
