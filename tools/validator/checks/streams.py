"""C05 (header-based parse), C06 (keys), C07 (monotonicity), C34 (writer integrity), C35 (entity identity)."""

from .base import FAIL, HARD, PASS, first, result, unreadable

# Streams with their own counter and its step (C06, C07). Telemetry steps by k (TBD-1).
COUNTERS = (("ctl", "physics_step"), ("telemetry", "physics_step"), ("hf", "render_frame"), ("img", "slot_id"),
            ("events", "event_seq"), ("episodes", "episode_id"))
STARTS_AT_ZERO = ("ctl", "telemetry", "img", "events", "episodes")
WORLD_TIME_STREAMS = ("ctl", "telemetry", "hf", "events")
HL_COMMAND = ("hl_kappa", "hl_accel", "hl_dir")


def check_c05(ctx):
    """Every committed chunk and heartbeat.csv parse by header name: no missing required or unknown column, the
    field count of every row, types, finite numbers without exponent, enums, non-empty required fields, ASCII and
    LF only (dc-2.0.md section 8). 0 errors."""
    if ctx.load_error:
        return unreadable(ctx, "C05")
    problems = ["%s: %s" % (p.where, p.message) for p in ctx.session.problems_for("C05")]
    rows = sum(len(st) for st in ctx.session.streams.values())
    return result(ctx, "C05", HARD, FAIL if problems else PASS, len(problems), 0,
                  first(problems) if problems else "%d committed rows parsed by header name" % rows)


def check_c06(ctx):
    """Keys unique and contiguous per stream on its own counter (ctl and hf +1, telemetry +k, img slot_id, events
    event_seq, episodes episode_id +1); ctl, telemetry, img, events and episodes start at 0; every hf row with a
    physics step joins ctl exactly and telemetry as-of (section 6). 0 duplicates, 0 gaps, 0 join failures. hf rows
    past the last committed ctl step are counted but judged by C34 (the stream ends before the session does)."""
    if ctx.load_error:
        return unreadable(ctx, "C06")
    s = ctx.session
    k = s.clocks["k"]
    issues = []
    dups_total = gaps_total = 0
    for name, key in COUNTERS:
        keys = s.streams[name].col(key)
        if not keys:
            continue
        step = k if name == "telemetry" else 1
        seen = set()
        dups = 0
        for v in keys:
            if v in seen:
                dups += 1
            seen.add(v)
        ordered = sorted(seen)
        gaps = 0
        where = []
        for a, b in zip(ordered, ordered[1:]):
            if b - a != step:
                missing = (b - a) // step - 1 if (b - a) % step == 0 else (b - a)
                gaps += max(missing, 1)
                where.append("%d..%d" % (a, b))
        if name in STARTS_AT_ZERO and ordered[0] != 0:
            gaps += ordered[0] // step if ordered[0] > 0 else 1
            where.insert(0, "starts at %d" % ordered[0])
        if dups or gaps:
            issues.append("%s.%s: %d duplicate(s), %d missing key(s) (%s)" % (name, key, dups, gaps, first(where, 2)))
        dups_total += dups
        gaps_total += gaps
    ctl_steps = s.streams["ctl"].col("physics_step")
    last_ctl = max(ctl_steps) if ctl_steps else -1
    joins, past = [], 0
    for i, lps in enumerate(s.streams["hf"].col("last_physics_step")):
        if lps is None or lps < 0:
            continue
        if lps > last_ctl:
            past += 1
        elif s.hf_ctl[i] is None or s.hf_tel[i] is None:
            joins.append("%d (%s)" % (s.streams["hf"].col("render_frame")[i],
                                      "no ctl row" if s.hf_ctl[i] is None else "no telemetry row within k steps"))
    if joins:
        issues.append("%d hf row(s) whose last_physics_step does not join ctl / telemetry (frames %s)"
                      % (len(joins), first(joins, 3)))
    return result(ctx, "C06", HARD, FAIL if issues else PASS,
                  {"duplicates": dups_total, "gaps": gaps_total, "hf_join_failures": len(joins),
                   "hf_rows_past_ctl": past},
                  "0 duplicates, 0 gaps, 0 hf join failures",
                  first(issues) if issues else "all stream counters unique and contiguous; every hf row joins")


def check_c07(ctx):
    """Monotonic counters and clocks (dc-2.0.md section 6): every key strictly increasing in row order;
    world_time_ms non-decreasing in ctl, telemetry, hf and events; two ctl rows never share a world time (world
    time advances per fixed step, TBD-1 default); two hf rows share one only with no physics step between them or
    while paused; hf.last_physics_step non-decreasing; img requested_world_ms and issue_render_frame increasing;
    episodes ordered and not overlapping. The writer's R-01 decision boundary (streams JSON timing.decision): ctl
    hl_seq is -1 at step 0 and changes only on steps with physics_step mod N == 1, by +1; within a window (P_t,
    P_t + N] the held command hl_kappa, hl_accel and hl_dir stays the same (it may change at an episode boundary).
    0 violations."""
    if ctx.load_error:
        return unreadable(ctx, "C07")
    s = ctx.session
    issues = []
    for name, key in COUNTERS:
        keys = s.streams[name].col(key)
        bad = sum(1 for a, b in zip(keys, keys[1:]) if not b > a)
        if bad:
            issues.append("%s.%s not strictly increasing at %d row(s)" % (name, key, bad))
    for name in WORLD_TIME_STREAMS:
        w = s.streams[name].col("world_time_ms")
        bad = sum(1 for a, b in zip(w, w[1:]) if b < a)
        if bad:
            issues.append("%s.world_time_ms decreases at %d row(s)" % (name, bad))
    w = s.streams["ctl"].col("world_time_ms")
    steps = s.streams["ctl"].col("physics_step")
    same = sum(1 for i in range(1, len(w)) if w[i] == w[i - 1] and steps[i] > steps[i - 1])
    if same:
        issues.append("ctl: %d physics step(s) with no world-time advance" % same)
    issues += _decision_boundary(s)
    hf = s.streams["hf"]
    hw, lps, paused = hf.col("world_time_ms"), hf.col("last_physics_step"), hf.col("paused")
    same = sum(1 for i in range(1, len(hw)) if hw[i] == hw[i - 1] and lps[i] != lps[i - 1]
               and not (paused[i] or paused[i - 1]))
    if same:
        issues.append("hf: %d frame(s) with a physics step but the same world time" % same)
    back = sum(1 for a, b in zip(lps, lps[1:]) if b < a)
    if back:
        issues.append("hf.last_physics_step decreases at %d row(s)" % back)
    img = s.streams["img"]
    req = img.col("requested_world_ms")
    bad = sum(1 for a, b in zip(req, req[1:]) if not b > a)
    if bad:
        issues.append("img.requested_world_ms not increasing at %d row(s)" % bad)
    frames = [f for f in img.col("issue_render_frame") if f is not None]
    bad = sum(1 for a, b in zip(frames, frames[1:]) if not b > a)
    if bad:
        issues.append("img.issue_render_frame not increasing at %d row(s)" % bad)
    ep = s.streams["episodes"]
    starts, ends = ep.col("start_physics_step"), ep.col("end_physics_step")
    bad = sum(1 for a, b in zip(starts, ends) if b < a)
    bad += sum(1 for i in range(1, len(starts)) if starts[i] <= ends[i - 1])
    if bad:
        issues.append("episodes: %d episode(s) reversed or overlapping" % bad)
    return result(ctx, "C07", HARD, FAIL if issues else PASS, len(issues), "0 violations",
                  first(issues) if issues else "counters and world time monotonic")


def _decision_boundary(s):
    """R-01 writer issues of the ctl stream (see check_c07)."""
    ctl = s.streams["ctl"]
    n = s.clocks["N"]
    steps, seq, ep = ctl.col("physics_step"), ctl.col("hl_seq"), ctl.col("episode_id")
    index = ctl.index()
    seq_bad, held_bad = [], []
    if 0 in index and seq[index[0]] != -1:
        seq_bad.append(0)
    for i, p in enumerate(steps):
        j = index.get(p - 1)
        if j is None or j == i:
            continue
        if p % n == 1 % n:
            if seq[i] not in (seq[j], seq[j] + 1):
                seq_bad.append(p)
            continue
        if seq[i] != seq[j]:
            seq_bad.append(p)
        elif ep[i] == ep[j] and any(ctl.col(c)[i] != ctl.col(c)[j] for c in HL_COMMAND):
            held_bad.append(p)
    issues = []
    if seq_bad:
        issues.append("ctl.hl_seq changes off the decision boundary (only at physics_step mod %d == 1, by +1; -1 at "
                      "step 0) at %d step(s): %s" % (n, len(seq_bad), first(seq_bad, 4)))
    if held_bad:
        issues.append("ctl hl_kappa/hl_accel/hl_dir change inside a decision window at %d step(s): %s"
                      % (len(held_bad), first(held_bad, 4)))
    return issues


def check_c34(ctx):
    """Writer integrity (r2 5.1 C34; dc-2.0.md section 8): every commit line names the next chunk, whose file exists
    with exactly n_bytes and n_rows (never 0) and the committed first and last key; a sealed session has no chunk
    file without a commit line (a written but never acknowledged row); session_end.json counts {rows, chunks,
    last_key} equal the committed data of every stream; the committed ctl, telemetry and hf rows reach the session
    end (end.physics_step, its last k-th step, end.render_frame: reader problems), so a lost or truncated tail
    fails. Exact."""
    if ctx.load_error:
        return unreadable(ctx, "C34")
    s = ctx.session
    issues = ["%s: %s" % (p.where, p.message) for p in s.problems_for("C34")]
    for name, st in s.streams.items():
        if st.uncommitted and s.sealed:
            issues.append("%s: %d chunk file(s) without a commit line (%s)" % (name, len(st.uncommitted),
                                                                              first(st.uncommitted, 2)))
    counts = s.end_get("counts") if s.end is not None else None
    if isinstance(counts, dict):
        for name, st in s.streams.items():
            c = counts.get(name)
            if c is None:
                if len(st):
                    issues.append("%s: %d committed rows but no session_end counts" % (name, len(st)))
                continue
            if not isinstance(c, dict):
                issues.append("%s: session_end counts is not an object" % name)
                continue
            have = {"rows": len(st), "chunks": len(st.commits),
                    "last_key": st.col(st.key)[-1] if len(st) else None}
            for key in ("rows", "chunks", "last_key"):
                if c.get(key) != have[key]:
                    issues.append("%s: session_end counts %s=%r, committed %r" % (name, key, c.get(key), have[key]))
        for name in counts:
            if name not in s.streams:
                issues.append("session_end counts names unknown stream %s" % name)
    elif s.sealed:
        issues.append("session_end.json has no counts object")
    return result(ctx, "C34", HARD, FAIL if issues else PASS, len(issues), "exact",
                  first(issues) if issues else "%d commit lines match their chunks and session_end counts"
                  % sum(len(st.commits) for st in s.streams.values()))


def check_c35(ctx):
    """Entity identity (r2 5.1 C35): one entity id per vehicle slot id (ego slots in telemetry and episodes, traffic
    slots), no slot shared by the ego and a traffic vehicle, and one ego slot per episode (telemetry, img and the
    episodes row agree). Exact."""
    if ctx.load_error:
        return unreadable(ctx, "C35")
    s = ctx.session
    entities = {}
    tel = s.streams["telemetry"]
    for slot, ent in zip(tel.col("ego_slot_id"), tel.col("ego_entity_id")):
        entities.setdefault(slot, set()).add(ent)
    ep = s.streams["episodes"]
    for slot, ent in zip(ep.col("ego_slot_id"), ep.col("ego_entity_id")):
        entities.setdefault(slot, set()).add(ent)
    ego_slots = set(entities)
    traffic = s.streams.get("traffic")
    if traffic is not None and len(traffic):
        for slot, ent in zip(traffic.col("veh_slot_id"), traffic.col("entity_id")):
            entities.setdefault(slot, set()).add(ent)
    issues = ["slot %s carries %d entity ids (%s)" % (slot, len(ents), ", ".join(sorted(ents)))
              for slot, ents in sorted(entities.items()) if len(ents) > 1]
    if traffic is not None and len(traffic):
        shared = ego_slots & set(traffic.col("veh_slot_id"))
        if shared:
            issues.append("slot(s) %s used by the ego and by traffic" % sorted(shared))
    per_episode = {}
    for e, slot in zip(tel.col("episode_id"), tel.col("ego_slot_id")):
        if e is not None and e >= 0:
            per_episode.setdefault(e, set()).add(slot)
    img = s.streams["img"]
    for e, slot in zip(img.col("episode_id"), img.col("ego_slot_id")):
        if e is not None and e >= 0:
            per_episode.setdefault(e, set()).add(slot)
    for e, slot in zip(ep.col("episode_id"), ep.col("ego_slot_id")):
        per_episode.setdefault(e, set()).add(slot)
    for e, slots in sorted(per_episode.items()):
        if len(slots) > 1:
            issues.append("episode %d has ego slots %s" % (e, sorted(slots)))
    return result(ctx, "C35", HARD, FAIL if issues else PASS, len(issues), "exact",
                  first(issues) if issues else "%d slot(s), one entity each; one ego slot per episode"
                  % len(entities))
