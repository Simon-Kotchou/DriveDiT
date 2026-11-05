"""C05 (header-based parse), C06 (keys), C07 (monotonicity), C34 (writer integrity), C35 (entity identity)."""

import bisect

from .base import FAIL, HARD, PASS, first, load_reader, result, unreadable

# Streams with their own counter (C06, C07). ctl and telemetry are frame-bound (contract section 6, freeze decision
# D-A): physics_step advances by the row's covered_steps; the others advance by 1.
COUNTERS = (("ctl", "physics_step"), ("telemetry", "physics_step"), ("hf", "render_frame"), ("img", "slot_id"),
            ("events", "event_seq"), ("episodes", "episode_id"))
STEP_STREAMS = ("ctl", "telemetry")
STARTS_AT_ZERO = ("img", "events", "episodes")
WORLD_TIME_STREAMS = ("ctl", "telemetry", "hf", "events")
HL_COMMAND = ("hl_kappa", "hl_accel", "hl_dir")
NOISE_MODE = "N"                    # ctl.noise_active 1 only in a mode-N episode (dc-2.0 9.1, r2 3.1)
PERTURB_KINDS = ("perturb_start", "perturb_end")


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
    """Keys unique and contiguous per stream on its own counter (dc-2.0.md section 6, D-A): ctl and telemetry are
    frame-bound, so each row's physics_step is the previous row's plus its covered_steps (>= 1) and the first row's is
    covered_steps - 1 (counted from the session open); ctl and telemetry hold the same keys (one row of each per
    callback); hf render_frame, img slot_id, events event_seq and episodes episode_id advance by 1 (img, events and
    episodes from 0); every hf row with a physics step joins ctl and telemetry exactly. 0 duplicates, 0 gaps, 0 join
    failures. hf rows past the last committed ctl step, and ctl or telemetry keys past the other stream's last
    committed key, are counted but judged by C34 (the stream ends before the session does)."""
    if ctx.load_error:
        return unreadable(ctx, "C06")
    s = ctx.session
    issues = []
    dups_total = gaps_total = 0
    for name, key in COUNTERS:
        st = s.streams[name]
        keys = st.col(key)
        if not keys:
            continue
        seen = set()
        dups = 0
        for v in keys:
            if v in seen:
                dups += 1
            seen.add(v)
        gaps = 0
        where = []
        if name in STEP_STREAMS:
            covered = st.col("covered_steps")
            bad_cov = sum(1 for c in covered if c is None or c < 1)
            if bad_cov:
                gaps += bad_cov
                where.append("%d row(s) with covered_steps < 1" % bad_cov)
            if covered[0] is not None and keys[0] != covered[0] - 1:
                gaps += 1
                where.insert(0, "first row physics_step %d with covered_steps %s (the session open is step -1)"
                             % (keys[0], covered[0]))
            for i in range(1, len(keys)):
                if covered[i] is not None and keys[i] > keys[i - 1] and keys[i] - keys[i - 1] != covered[i]:
                    gaps += 1
                    where.append("%d..%d (covered %d)" % (keys[i - 1], keys[i], covered[i]))
        else:
            ordered = sorted(seen)
            for a, b in zip(ordered, ordered[1:]):
                if b - a != 1:
                    gaps += b - a - 1
                    where.append("%d..%d" % (a, b))
            if name in STARTS_AT_ZERO and ordered[0] != 0:
                gaps += ordered[0] if ordered[0] > 0 else 1
                where.insert(0, "starts at %d" % ordered[0])
        if dups or gaps:
            issues.append("%s.%s: %d duplicate(s), %d missing key(s) (%s)" % (name, key, dups, gaps, first(where, 2)))
        dups_total += dups
        gaps_total += gaps
    ctl_steps = s.streams["ctl"].col("physics_step")
    tel_steps = s.streams["telemetry"].col("physics_step")
    # a stream whose committed rows end early is C34's (a truncated tail), not a key mismatch
    last_tel = max(tel_steps) if tel_steps else -1
    last_both = min(max(ctl_steps) if ctl_steps else -1, last_tel)
    only_ctl = sorted(p for p in set(ctl_steps) - set(tel_steps) if p <= last_both)
    only_tel = sorted(p for p in set(tel_steps) - set(ctl_steps) if p <= last_both)
    if only_ctl or only_tel:
        issues.append("ctl and telemetry keys differ: %d ctl row(s) without telemetry (%s), %d telemetry row(s) "
                      "without ctl (%s)" % (len(only_ctl), first(only_ctl, 3), len(only_tel), first(only_tel, 3)))
    # one callback writes both rows of a key: same render_frame, same world time (the frame clock)
    ctl, tel = s.streams["ctl"], s.streams["telemetry"]
    ctl_clock = dict(zip(ctl_steps, zip(ctl.col("render_frame"), ctl.col("world_time_ms"))))
    split = [k for k, f, w in zip(tel_steps, tel.col("render_frame"), tel.col("world_time_ms"))
             if k in ctl_clock and ctl_clock[k] != (f, w)]
    if split:
        issues.append("%d key(s) whose ctl and telemetry rows differ in render_frame or world_time_ms (%s): one "
                      "callback, one frame clock" % (len(split), first(split, 3)))
    last_ctl = max(ctl_steps) if ctl_steps else -1
    joins, past = [], 0
    for i, lps in enumerate(s.streams["hf"].col("last_physics_step")):
        if lps is None or lps < 0:
            continue
        if lps > last_ctl:
            past += 1
        elif s.hf_ctl[i] is None or s.hf_tel[i] is None:
            joins.append("%d (%s)" % (s.streams["hf"].col("render_frame")[i],
                                      "no ctl row" if s.hf_ctl[i] is None else "no telemetry row"))
    if joins:
        issues.append("%d hf row(s) whose last_physics_step does not join ctl / telemetry (frames %s)"
                      % (len(joins), first(joins, 3)))
    return result(ctx, "C06", HARD, FAIL if issues else PASS,
                  {"duplicates": dups_total, "gaps": gaps_total, "ctl_telemetry_key_mismatch": len(only_ctl) +
                   len(only_tel), "ctl_telemetry_clock_mismatch": len(split), "hf_join_failures": len(joins),
                   "hf_rows_past_ctl": past},
                  "0 duplicates, 0 gaps, 0 key or clock mismatches, 0 hf join failures",
                  first(issues) if issues else "all stream counters unique and contiguous (ctl and telemetry by "
                  "covered_steps); every hf row joins")


def check_c07(ctx):
    """Monotonic counters and clocks (dc-2.0.md section 6, D-A): every key strictly increasing in row order;
    world_time_ms non-decreasing in ctl, telemetry, hf and events; ctl render_frame non-decreasing, and ctl rows of
    one render frame share one world time, the world time of the hf row of that frame (the frame clock); two hf rows
    share a world time only with no physics step between them or while paused; hf.last_physics_step non-decreasing
    and exactly the newest callback of an earlier render frame (a frame's callback runs after its EOnFrame, which the
    image pairing of section 6 relies on; a stale value pairs an image with an older state); img requested_world_ms
    and issue_render_frame increasing, done_render_frame after issue_render_frame; episodes ordered, not overlapping,
    and begun at a decision callback. The writer's R-01 decision boundary on the world-time grid (streams JSON
    timing.decision): ctl hl_seq is -1 (the no-command sentinel: hl_* empty) until the first command and advances by
    1 exactly on the row right after every decision callback of an episode that starts or goes on (a kept command is
    numbered again) and on no other row, so a number before the first episode fails; a command (hl_kappa, hl_accel,
    hl_dir) changes only together with that increment or at an episode boundary. The disturbance of the held command
    (dc-2.0 9.1 noise_active / noise_id, DD-7-ident) follows the same boundary: noise_active 1 exactly when noise_id
    >= 0 (-1 when none), only on a row that holds a command of a mode-N episode (ctl.mode N, episodes.mode N); the
    pair changes only together with an hl_seq increment or at an episode boundary; every disturbance id has one
    perturb_start and one perturb_end (payload noise_id) in its episode, start before end, and its rows are exactly
    the rows after the start's decision callback through the end's (R-01: the command decided at C_t applies from
    P_t + 1). 0 violations."""
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
    ctl = s.streams["ctl"]
    w, frames = ctl.col("world_time_ms"), ctl.col("render_frame")
    back = sum(1 for a, b in zip(frames, frames[1:]) if b < a)
    if back:
        issues.append("ctl.render_frame decreases at %d row(s)" % back)
    split = sum(1 for i in range(1, len(w)) if frames[i] == frames[i - 1] and w[i] != w[i - 1])
    if split:
        issues.append("ctl: %d row(s) of one render frame with different world times (the frame clock)" % split)
    hf_world = dict(zip(s.streams["hf"].col("render_frame"), s.streams["hf"].col("world_time_ms")))
    off_clock = [k for k, f, ww in zip(ctl.col("physics_step"), frames, w) if f in hf_world and hf_world[f] != ww]
    if off_clock:
        issues.append("%d ctl row(s) whose world_time_ms differs from the hf row of their render frame (keys %s): "
                      "the frame clock" % (len(off_clock), first(off_clock, 3)))
    issues += _decision_boundary(s)
    issues += _noise_boundary(s)
    hf = s.streams["hf"]
    hw, lps, paused, hframe = hf.col("world_time_ms"), hf.col("last_physics_step"), hf.col("paused"), \
        hf.col("render_frame")
    same = sum(1 for i in range(1, len(hw)) if hw[i] == hw[i - 1] and lps[i] != lps[i - 1]
               and not (paused[i] or paused[i - 1]))
    if same:
        issues.append("hf: %d frame(s) with a physics step but the same world time" % same)
    back = sum(1 for a, b in zip(lps, lps[1:]) if b < a)
    if back:
        issues.append("hf.last_physics_step decreases at %d row(s)" % back)
    cb_frame = dict(zip(ctl.col("physics_step"), frames))
    late = [hframe[i] for i, step in enumerate(lps) if step in cb_frame and not cb_frame[step] < hframe[i]]
    if late:
        issues.append("%d hf row(s) whose last_physics_step is a callback of the same or a later render frame "
                      "(frames %s): the callback must follow its frame's EOnFrame" % (len(late), first(late, 3)))
    # exactly the newest callback of an earlier frame; an hf row naming a missing ctl key is C06's, a row past the last
    # committed ctl key C34's
    ctl_keys = ctl.col("physics_step")
    last_ctl = max(ctl_keys) if ctl_keys else -1
    stale = []
    for frame, step in zip(hframe, lps):
        if step is None or step > last_ctl or (step >= 0 and step not in cb_frame):
            continue
        j = bisect.bisect_left(frames, frame) - 1
        if step != (ctl_keys[j] if j >= 0 else -1):
            stale.append(frame)
    if stale:
        issues.append("%d hf row(s) whose last_physics_step is not the newest callback of an earlier render frame "
                      "(frames %s): a stale join pairs images with an older state" % (len(stale), first(stale, 3)))
    img = s.streams["img"]
    req = img.col("requested_world_ms")
    bad = sum(1 for a, b in zip(req, req[1:]) if not b > a)
    if bad:
        issues.append("img.requested_world_ms not increasing at %d row(s)" % bad)
    issue_frames = [f for f in img.col("issue_render_frame") if f is not None]
    bad = sum(1 for a, b in zip(issue_frames, issue_frames[1:]) if not b > a)
    if bad:
        issues.append("img.issue_render_frame not increasing at %d row(s)" % bad)
    early = [sl for sl, a, d in zip(img.col("slot_id"), img.col("issue_render_frame"), img.col("done_render_frame"))
             if a is not None and d is not None and not d > a]
    if early:
        issues.append("%d img row(s) whose done_render_frame is not after issue_render_frame (slots %s)"
                      % (len(early), first(early, 3)))
    ep = s.streams["episodes"]
    starts, ends = ep.col("start_physics_step"), ep.col("end_physics_step")
    bad = sum(1 for a, b in zip(starts, ends) if b < a)
    bad += sum(1 for i in range(1, len(starts)) if starts[i] <= ends[i - 1])
    if bad:
        issues.append("episodes: %d episode(s) reversed or overlapping" % bad)
    callbacks, _ = ctx.reader.decision_callbacks(w, s.clocks["decision_hz"])
    deciders = set(ctl.col("physics_step")[i] for i, _ in callbacks)
    off = [a for a in starts if a is not None and a in cb_frame and a not in deciders]
    if off:
        issues.append("episodes: %d episode(s) not begun at a decision callback (start_physics_step %s)"
                      % (len(off), first(off, 3)))
    return result(ctx, "C07", HARD, FAIL if issues else PASS, len(issues), "0 violations",
                  first(issues) if issues else "counters and world time monotonic; R-01 on the world-time grid")


def _decision_boundary(s):
    """R-01 writer issues of the ctl stream (see check_c07): decision callbacks from the world-time grid."""
    reader = load_reader()
    ctl = s.streams["ctl"]
    steps, seq, ep = ctl.col("physics_step"), ctl.col("hl_seq"), ctl.col("episode_id")
    callbacks, _ = reader.decision_callbacks(ctl.col("world_time_ms"), s.clocks["decision_hz"])
    after_decision = set(i + 1 for i, _ in callbacks)
    seq_bad, held_bad, unnumbered = [], [], []
    if seq and seq[0] != -1:
        seq_bad.append(steps[0])
    for i in range(1, len(steps)):
        changed = any(ctl.col(c)[i] != ctl.col(c)[i - 1] for c in HL_COMMAND)
        # every decision callback of an episode that goes on numbers its command (a kept command is numbered again)
        due = i in after_decision and ep[i - 1] is not None and ep[i - 1] >= 0 and ep[i] == ep[i - 1]
        if seq[i] == seq[i - 1] + 1:
            if not due:             # only a decision callback inside a starting or continuing episode numbers one
                seq_bad.append(steps[i])
        elif seq[i] != seq[i - 1]:
            seq_bad.append(steps[i])
        elif due:
            unnumbered.append(steps[i])
        elif changed and ep[i] == ep[i - 1]:
            held_bad.append(steps[i])
    # -1 is the no-command sentinel: no command before the first one, and no row with a command keeps it
    sentinel = [steps[i] for i in range(len(steps)) if seq[i] == -1 and
                any(ctl.col(c)[i] is not None for c in HL_COMMAND)]
    issues = []
    if sentinel:
        issues.append("ctl: %d row(s) with a command but hl_seq -1 (the no-command sentinel): %s"
                      % (len(sentinel), first(sentinel, 4)))
    if seq_bad:
        issues.append("ctl.hl_seq changes off the decision boundary (only on the row after a decision callback of "
                      "the world-time grid inside a starting or continuing episode, by +1; -1 until the first "
                      "command) at %d row(s): %s"
                      % (len(seq_bad), first(seq_bad, 4)))
    if unnumbered:
        issues.append("ctl: no hl_seq increment on %d row(s) after a decision callback inside an episode (every "
                      "decision is numbered, a kept command too): %s" % (len(unnumbered), first(unnumbered, 4)))
    if held_bad:
        issues.append("ctl hl_kappa/hl_accel/hl_dir change inside a decision interval at %d row(s): %s"
                      % (len(held_bad), first(held_bad, 4)))
    return issues


def _payload_noise_id(payload):
    """noise_id of a perturb event payload ('key=value' pairs joined by ';'), None when missing or not an int."""
    for item in (payload or "").split(";"):
        key, _, value = item.partition("=")
        if key == "noise_id":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _noise_boundary(s):
    """The disturbance rules of check_c07 on ctl.noise_active / noise_id and the perturb events."""
    ctl, ev, eps = s.streams["ctl"], s.streams["events"], s.streams["episodes"]
    steps, seq, ep = ctl.col("physics_step"), ctl.col("hl_seq"), ctl.col("episode_id")
    active, nid, mode = ctl.col("noise_active"), ctl.col("noise_id"), ctl.col("mode")
    episode_mode = dict(zip(eps.col("episode_id"), eps.col("mode")))
    disagree, outside, unheld, held_bad = [], [], [], []
    for i in range(len(steps)):
        on = bool(active[i])
        if on != (nid[i] is not None and nid[i] >= 0) or (nid[i] is not None and nid[i] < -1):
            disagree.append(steps[i])
        if on and (mode[i] != NOISE_MODE or ep[i] is None or ep[i] < 0 or episode_mode.get(ep[i]) != NOISE_MODE):
            outside.append(steps[i])
        if on and (seq[i] is None or seq[i] < 0 or any(ctl.col(c)[i] is None for c in HL_COMMAND)):
            unheld.append(steps[i])
        if i and (active[i], nid[i]) != (active[i - 1], nid[i - 1]) and seq[i] == seq[i - 1] and ep[i] == ep[i - 1]:
            held_bad.append(steps[i])
    issues = []
    if disagree:
        issues.append("ctl: noise_active and noise_id disagree on %d row(s) (1 needs an id >= 0, 0 needs -1): %s"
                      % (len(disagree), first(disagree, 4)))
    if outside:
        issues.append("ctl: noise_active 1 outside a mode-N episode on %d row(s): %s" % (len(outside),
                                                                                           first(outside, 4)))
    if unheld:
        issues.append("ctl: noise_active 1 on %d row(s) without a held command: %s" % (len(unheld), first(unheld, 4)))
    if held_bad:
        issues.append("ctl noise_active/noise_id change inside a decision interval at %d row(s): %s"
                      % (len(held_bad), first(held_bad, 4)))
    # the perturb events: one start and one end per id, in its episode; the id's rows are (start step, end step]
    marks = {}
    for kind, step, episode, payload in zip(ev.col("kind"), ev.col("physics_step"), ev.col("episode_id"),
                                            ev.col("payload")):
        if kind not in PERTURB_KINDS:
            continue
        noise_id = _payload_noise_id(payload)
        if noise_id is None or noise_id < 0:
            issues.append("events: %s without a noise_id >= 0 in its payload (%r)" % (kind, payload))
            continue
        marks.setdefault(noise_id, {}).setdefault(kind, []).append((step, episode))
    rows = {}
    for i in range(len(steps)):
        if active[i] and nid[i] is not None and nid[i] >= 0:
            rows.setdefault(nid[i], []).append((steps[i], ep[i]))
    window_bad = []
    for noise_id in sorted(set(marks) | set(rows)):
        m = marks.get(noise_id, {})
        starts, ends = m.get("perturb_start", []), m.get("perturb_end", [])
        if len(starts) != 1 or len(ends) != 1:
            window_bad.append("noise_id %d has %d perturb_start and %d perturb_end" % (noise_id, len(starts),
                                                                                       len(ends)))
            continue
        (start, start_ep), (end, end_ep) = starts[0], ends[0]
        if not start < end or start_ep != end_ep or start_ep is None or start_ep < 0:
            window_bad.append("noise_id %d: perturb_start at step %s (episode %s) is not before its perturb_end at "
                              "step %s in the same episode (%s)" % (noise_id, start, start_ep, end, end_ep))
            continue
        want = [k for k, e in zip(steps, ep) if start < k <= end and e == start_ep]
        have = [k for k, _ in rows.get(noise_id, [])]
        if have != want:
            extra = sorted(set(have) - set(want))
            missing = sorted(set(want) - set(have))
            window_bad.append("noise_id %d: its rows are not exactly the rows after perturb_start (step %d) through "
                              "perturb_end (step %d): %d outside %s, %d missing %s" % (
                                  noise_id, start, end, len(extra), first(extra, 3), len(missing), first(missing, 3)))
    if window_bad:
        issues.append("perturbation windows: %s" % first(window_bad, 3))
    return issues


def check_c34(ctx):
    """Writer integrity (r2 5.1 C34; dc-2.0.md section 8): every commit line names the next chunk, whose file exists
    with exactly n_bytes and n_rows (never 0) and the committed first and last key; a sealed session has no chunk
    file without a commit line (a written but never acknowledged row); session_end.json counts {rows, chunks,
    last_key} equal the committed data of every stream; the committed ctl, telemetry and hf rows reach the session
    end (end.physics_step for ctl and telemetry, end.render_frame: reader problems), so a lost or truncated tail
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


def traffic_slot_pairs(text):
    """([(veh_slot_id, entity_id)], [malformed pair texts]) of a telemetry.traffic_slots value: 'veh_slot_id:entity_id'
    pairs joined by ';', the slot a non-negative integer and the entity id non-empty (contract 9.2)."""
    pairs, bad = [], []
    for pair in (text or "").split(";") if text else []:
        head, sep, entity = pair.partition(":")
        if not sep or not head.isdigit() or not entity or ":" in entity:
            bad.append(pair)
            continue
        pairs.append((int(head), entity))
    return pairs, bad


def check_c35(ctx):
    """Entity identity (r2 5.1 C35): one entity id per vehicle slot id (ego slots in telemetry and episodes, traffic
    slots), no slot shared by the ego and a traffic vehicle, and one ego slot per episode (telemetry, img and the
    episodes row agree). telemetry.traffic_slots pairs parse as veh_slot_id:entity_id and take part in the identity
    and overlap checks. Every events.veh_slot_id >= 0 and every crash payload other_slot (contract 9.5.1) is a known
    ego or traffic slot (the traffic stream or telemetry.traffic_slots). Exact."""
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
    traffic_slots = set()
    traffic = s.streams.get("traffic")
    if traffic is not None and len(traffic):
        for slot, ent in zip(traffic.col("veh_slot_id"), traffic.col("entity_id")):
            entities.setdefault(slot, set()).add(ent)
            traffic_slots.add(slot)
    malformed = []
    for step, text in zip(tel.col("physics_step"), tel.col("traffic_slots")):
        pairs, bad = traffic_slot_pairs(text)
        malformed += ["step %s: %r" % (step, b) for b in bad]
        for slot, ent in pairs:
            entities.setdefault(slot, set()).add(ent)
            traffic_slots.add(slot)
    issues = ["slot %s carries %d entity ids (%s)" % (slot, len(ents), ", ".join(sorted(ents)))
              for slot, ents in sorted(entities.items()) if len(ents) > 1]
    if malformed:
        issues.append("%d malformed telemetry.traffic_slots pair(s) (veh_slot_id:entity_id): %s"
                      % (len(malformed), first(malformed, 2)))
    shared = ego_slots & traffic_slots
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
    known = set(entities)
    ev = s.streams["events"]
    unknown = sorted(set(slot for slot in ev.col("veh_slot_id") if slot is not None and slot >= 0) - known)
    if unknown:
        issues.append("events name unknown vehicle slot(s) %s" % unknown)
    others = sorted(set(c["values"]["other_slot"] for c in s.crash_events
                        if isinstance(c["values"].get("other_slot"), int)) - known)
    if others:
        issues.append("collision payloads name unknown other_slot(s) %s" % others)
    return result(ctx, "C35", HARD, FAIL if issues else PASS, len(issues), "exact",
                  first(issues) if issues else "%d slot(s), one entity each; one ego slot per episode"
                  % len(entities))
