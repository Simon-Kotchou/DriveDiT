"""C36: crash-event consistency (contract dc-2.0 section 9.5.1; HARD)."""

from .base import FAIL, HARD, PASS, first, result, unreadable

DV_TOL_MPS = 0.005          # dv_mps has 2 decimals (9.5.1): a peak at the threshold may round down by half a cent
PAIRS = (("veh_fire_start", "veh_fire_end"), ("dismount", "remount"))
CRASH_TERMINATIONS = ("collision", "rollover")
NORMAL_ENDS = ("duration", "stop", "dc_stop")      # the planned ends: an ego destroyed before them is a contradiction


def check_c36(ctx):
    """Crash events against the streams JSON event_payloads and the episode and session ends (contract 9.5.1):
      * every row of a payload kind (collision, veh_damage, veh_fire_start, veh_fire_end, driver_state, dismount,
        remount, rollover, stuck, teleport) names a vehicle (veh_slot_id >= 0) and its payload parses: no repeated,
        unknown or missing required key, every value of its type and enum, numbers finite without exponent;
      * collision: onset_step <= the row's physics_step; with src dv or dv_contact, dv_mps >=
        session.json events.collision_dv_mps (within the 2-decimal rounding);
      * per slot, veh_fire_start / veh_fire_end and dismount / remount alternate, start first (the last may stay
        open at the session end);
      * termination_reason collision or rollover: the episode has a terminal ego event (reader terminal_ego: a
        collision with src dv or dv_contact, a veh_damage not undamaged or with movement_damage >= 1, a
        veh_fire_start, a rollover) and its first one by step gives that reason;
      * end_reason ego_destroyed: an ego veh_damage state=destroyed row exists and no ctl or telemetry row is later
        than it; an ego destroyed row in a session that ended duration, stop or dc_stop is a contradiction.
    Exact."""
    if ctx.load_error:
        return unreadable(ctx, "C36")
    s = ctx.session
    issues = []
    events = s.crash_events
    threshold = s.meta_get("events.collision_dv_mps")
    for c in events:
        where = "event %s %s" % (c["event_seq"], c["kind"])
        if c["veh_slot_id"] is None or c["veh_slot_id"] < 0:
            issues.append("%s: names no vehicle (veh_slot_id %s)" % (where, c["veh_slot_id"]))
        issues += ["%s: %s" % (where, p) for p in c["problems"]]
        v = c["values"]
        if c["kind"] != "collision":
            continue
        if isinstance(v.get("onset_step"), int) and v["onset_step"] > c["physics_step"]:
            issues.append("%s: onset_step %d after the row's physics_step %d" % (where, v["onset_step"],
                                                                              c["physics_step"]))
        if v.get("src") in ctx.reader.TERMINAL_COLLISION_SRC and "dv_mps" in v and \
                isinstance(threshold, (int, float)) and v["dv_mps"] < threshold - DV_TOL_MPS:
            issues.append("%s: dv_mps %.2f below events.collision_dv_mps %s with src %s" % (
                where, v["dv_mps"], threshold, v["src"]))
    for start, end in PAIRS:
        open_by_slot = {}
        for c in events:
            if c["kind"] not in (start, end):
                continue
            slot = c["veh_slot_id"]
            if c["kind"] == start and open_by_slot.get(slot):
                issues.append("event %s: a second %s on slot %s before its %s" % (c["event_seq"], start, slot, end))
            if c["kind"] == end and not open_by_slot.get(slot):
                issues.append("event %s: %s on slot %s without a %s" % (c["event_seq"], end, slot, start))
            open_by_slot[slot] = c["kind"] == start
    ep = s.streams["episodes"]
    crash_ends = 0
    for e, reason in zip(ep.col("episode_id"), ep.col("termination_reason")):
        if reason not in CRASH_TERMINATIONS:
            continue
        crash_ends += 1
        terminal = s.terminal_ego.get(e)
        if terminal is None:
            issues.append("episode %d: termination_reason %s without a terminal ego crash event" % (e, reason))
        elif terminal["terminal"] != reason:
            issues.append("episode %d: termination_reason %s, but its first terminal ego event (%s at step %d) "
                          "gives %s" % (e, reason, terminal["kind"], terminal["step"], terminal["terminal"]))
    destroyed = [c for c in events
                 if c["ego"] and c["kind"] == "veh_damage" and c["values"].get("state") == "destroyed"]
    end_reason = s.end_get("end_reason") if s.sealed else None
    if end_reason == "ego_destroyed":
        if not destroyed:
            issues.append("end_reason ego_destroyed without an ego veh_damage state=destroyed row")
        else:
            step = destroyed[0]["physics_step"]
            for name in ("ctl", "telemetry"):
                later = [p for p in s.streams[name].col("physics_step") if p > step]
                if later:
                    issues.append("%d %s row(s) after the ego's veh_damage destroyed row (step %d), first %d" % (
                        len(later), name, step, later[0]))
    elif destroyed and end_reason in NORMAL_ENDS:
        issues.append("an ego veh_damage state=destroyed row (event %s) in a session that ended %s, not "
                      "ego_destroyed" % (destroyed[0]["event_seq"], end_reason))
    value = {"crash_events": len(events), "crash_terminations": crash_ends,
             "terminal_ego_episodes": len(s.terminal_ego), "ego_destroyed": end_reason == "ego_destroyed"}
    return result(ctx, "C36", HARD, FAIL if issues else PASS, value, "exact",
                  first(issues) if issues else "%d crash event(s), %d crash termination(s) consistent" % (
                      len(events), crash_ends))
