"""C31: usable yield of training windows."""

from .base import FAIL, HARD, INFO, NA, PASS, result, unreadable


def window_yield(decisions, segments, length):
    """(valid starts, candidate starts) of windows of `length` consecutive decision steps: candidates are the
    starts that fit inside an episode's non-warm-up decision steps, valid ones lie inside one reader segment (no
    gap, fiducial failure, mask or episode boundary)."""
    per_episode = {}
    for d in decisions:
        if d["episode_id"] is not None and d["episode_id"] >= 0 and d["masks"]["warmup"] == 0:
            per_episode[d["episode_id"]] = per_episode.get(d["episode_id"], 0) + 1
    candidates = sum(max(n - length + 1, 0) for n in per_episode.values())
    valid = sum(max(seg["n_steps"] - length + 1, 0) for seg in segments)
    return valid, candidates


def check_c31(ctx):
    """Usable yield (r2 5.1 C31): the share of valid window starts of C31.L_req_frames decision steps (one image
    per decision step) after gaps, fiducial failures, masks and warm-up, >= C31.min_valid_start_frac from T1 on
    (reported in T0); the shares for C31.report_L are reported. N/A when no episode is that long."""
    if ctx.load_error:
        return unreadable(ctx, "C31", INFO)
    s = ctx.session
    thr = ctx.thr("C31")
    severity = HARD if ctx.rank >= 2 else INFO
    length = thr.get("L_req_frames", 128)
    valid, cand = window_yield(s.decisions, s.segments, length)
    value = {"L": length, "valid_starts": valid, "candidate_starts": cand}
    for other in thr.get("report_L", []):
        v, c = window_yield(s.decisions, s.segments, other)
        value["frac_L%d" % other] = (v / float(c)) if c else None
    if not cand:
        return result(ctx, "C31", severity, NA, value, thr, "no episode holds %d decision steps after the warm-up"
                      % length)
    frac = valid / float(cand)
    value["frac"] = frac
    ok = frac >= thr.get("min_valid_start_frac", 0.9)
    return result(ctx, "C31", severity, PASS if ok else FAIL, value, thr,
                  "%d of %d window starts valid (%.3f) at L=%d" % (valid, cand, frac, length))
