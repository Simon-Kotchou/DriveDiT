"""C31: usable yield of training windows."""

from .base import FAIL, INFO, NA, NOT_EVALUATED, PASS, first, result, unreadable


def window_yield(decisions, segments, length):
    """(valid starts, candidate starts) of windows of `length` consecutive decision steps: candidates are the
    starts that fit inside an episode's non-warm-up decision steps, valid ones lie inside one segment (no gap,
    fiducial failure, mask, nav_valid 0 or episode boundary)."""
    per_episode = {}
    for d in decisions:
        if d["episode_id"] is not None and d["episode_id"] >= 0 and d["masks"]["warmup"] == 0:
            per_episode[d["episode_id"]] = per_episode.get(d["episode_id"], 0) + 1
    candidates = sum(max(n - length + 1, 0) for n in per_episode.values())
    valid = sum(max(seg["n_steps"] - length + 1, 0) for seg in segments)
    return valid, candidates


def nav_masked(ctx, s):
    """(decision steps with nav_valid 0, episodes whose route could not be read): drivedit_nav.nav_sequence over each
    route episode's poses at its decision steps (features_v1: nav_valid 0 clears policy_valid and wm_valid). Episodes
    without a route_id (X/A modes) have no nav."""
    import drivedit_nav
    tel = s.streams["telemetry"]
    tel_index = tel.index()
    texp = s.telemetry_export
    ep = s.streams["episodes"]
    routes = {e: rid for e, rid in zip(ep.col("episode_id"), ep.col("route_id")) if rid}
    by_episode = {}
    for d in s.decisions:
        if d["episode_id"] in routes:
            by_episode.setdefault(d["episode_id"], []).append(d["decision_step"])
    masked, skipped = set(), []
    for e, steps in sorted(by_episode.items()):
        route, _, err = ctx.route(routes[e])
        if route is None:
            skipped.append("episode %d: %s" % (e, err))
            continue
        poses, keep = [], []
        for p in sorted(steps):
            i = tel_index.get(p)
            if i is None:
                masked.add(p)       # no pose: no nav at this step
                continue
            poses.append((tel.col("pos_x")[i], tel.col("pos_z")[i], float(texp["heading_rad"][i])))
            keep.append(p)
        for p, nav in zip(keep, drivedit_nav.nav_sequence(route, poses)):
            if nav["nav_valid"] != 1:
                masked.add(p)
    return masked, skipped


def check_c31(ctx):
    """Usable yield (r2 5.1 C31): the share of valid window starts of C31.L_req_frames decision steps (one image
    per decision step) after gaps, fiducial failures, masks, warm-up and nav_valid 0 (nav_v1 of the episode's route,
    --route / --routes-dir), >= C31.min_valid_start_frac from T1 on (HARD; reported as INFO in T0); the shares for
    C31.report_L are reported. N/A when no episode is that long. A route episode whose route file is missing leaves
    the nav part NOT_EVALUATED (nav_fill itself is DD-6's)."""
    if ctx.load_error:
        return unreadable(ctx, "C31")
    s = ctx.session
    thr = ctx.thr("C31")
    severity = ctx.severity("C31")
    length = thr.get("L_req_frames", 128)
    masked, skipped = nav_masked(ctx, s)
    decisions = s.decisions
    segments = s.segments
    if masked:
        decisions = [dict(d, masks=dict(d["masks"], wm_valid=0)) if d["decision_step"] in masked else d
                     for d in s.decisions]
        segments = ctx.reader.segments_of(decisions, s.clocks["N"])
    valid, cand = window_yield(decisions, segments, length)
    value = {"L": length, "valid_starts": valid, "candidate_starts": cand, "nav_invalid_steps": len(masked)}
    for other in thr.get("report_L", []):
        v, c = window_yield(decisions, segments, other)
        value["frac_L%d" % other] = (v / float(c)) if c else None
    not_evaluated = ["nav_valid of %s" % first(skipped, 2)] if skipped else None
    if not cand:
        return result(ctx, "C31", severity, NA, value, thr, "no episode holds %d decision steps after the warm-up"
                      % length, not_evaluated=not_evaluated)
    frac = valid / float(cand)
    value["frac"] = frac
    ok = frac >= thr.get("min_valid_start_frac", 0.9)
    if not ok:
        status = FAIL
    elif skipped and severity != INFO:
        status = NOT_EVALUATED
    else:
        status = PASS
    nav_note = "; nav not checked for %d episode(s)" % len(skipped) if skipped else ""
    return result(ctx, "C31", severity, status, value, thr,
                  "%d of %d window starts valid (%.3f) at L=%d%s" % (valid, cand, frac, length, nav_note),
                  not_evaluated=not_evaluated)
