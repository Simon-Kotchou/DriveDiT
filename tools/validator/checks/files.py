"""C01 (session and episode files) and C02 (provenance)."""

import hashlib
import json
import os
import re

from .base import FAIL, HARD, PASS, first, result, unreadable

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def check_c01(ctx):
    """session.json and session_end.json present and schema-valid against the streams JSON (every required key,
    nulls only where nullable, types, enums), the session id rule and map table, every required stream directory,
    heartbeat.csv, frames/, and an episodes row for every episode id the ctl and telemetry rows use."""
    if ctx.load_error:
        return result(ctx, "C01", HARD, FAIL, 1, "all keys", ctx.load_error)
    s = ctx.session
    problems = ["%s: %s" % (p.where, p.message) for p in s.problems_for("C01")]
    if s.end is None and not any(p.where == "session_end.json" for p in s.problems_for("C01")):
        problems.append("session_end.json: missing (session not sealed)")
    elif s.end is not None:
        if s.end_get("session_id") != s.session_id:
            problems.append("session_end.json: session_id %r differs from session.json" % s.end_get("session_id"))
    for name in ("heartbeat.csv",):
        if not os.path.isfile(os.path.join(s.path, name)):
            problems.append("%s: missing" % name)
    if not os.path.isdir(os.path.join(s.path, s.schema.frames_dir)):
        problems.append("%s/: missing" % s.schema.frames_dir)
    sid = s.session_id if isinstance(s.session_id, str) else ""
    parts = sid.split("_")
    if len(parts) == 6:
        if parts[1] != s.trial_id:
            problems.append("session_id trial %s differs from trial.trial_id %r" % (parts[1], s.trial_id))
        if parts[3] != s.meta_get("exe.kind") or parts[4] != s.meta_get("exe.instance"):
            problems.append("session_id exe %s_%s differs from exe.kind/exe.instance %r/%r" % (
                parts[3], parts[4], s.meta_get("exe.kind"), s.meta_get("exe.instance")))
    used = set(e for e in s.streams["ctl"].col("episode_id") if e is not None and e >= 0)
    used |= set(e for e in s.streams["telemetry"].col("episode_id") if e is not None and e >= 0)
    have = set(s.streams["episodes"].col("episode_id"))
    missing = sorted(used - have)
    if missing:
        problems.append("episodes: no row for episode(s) %s used by ctl/telemetry" % missing)
    status = FAIL if problems else PASS
    return result(ctx, "C01", HARD, status, len(problems), "all keys",
                  first(problems) if problems else "session.json, session_end.json and %d stream dirs valid"
                  % len(s.schema.required_streams()))


def _launch_record_checks(ctx, s, record_path, out):
    try:
        with open(record_path, "rb") as f:
            rec = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        out.append(("launch_record", False, "cannot read %s: %s" % (record_path, e)))
        return
    out.append(("launch_record.status", rec.get("status") == "ok", "status %r" % rec.get("status")))
    out.append(("launch_record.launch_id", rec.get("launch_id") == s.meta_get("pin.launch_id"),
                "record %r, session %r" % (rec.get("launch_id"), s.meta_get("pin.launch_id"))))
    pin = rec.get("pin") or {}
    out.append(("launch_record.pin", pin.get("pin_id") == s.meta_get("pin.pin_id")
                and pin.get("sha256") == s.meta_get("pin.pin_sha256"),
                "record pin %r sha %s" % (pin.get("pin_id"), str(pin.get("sha256"))[:12])))
    bad = [c.get("name") for c in rec.get("checks") or [] if not c.get("ok")]
    out.append(("launch_record.preflight", not bad and bool(rec.get("checks")),
                "preflight checks failed: %s" % bad if bad else "%d preflight checks ok (buildids, TargetBuildID, "
                "AutoUpdateBehavior, pak sizes and mtimes)" % len(rec.get("checks") or [])))


def check_c02(ctx):
    """Provenance (r2 5.1 C02): git SHA, dirty=false from T1, pin and buildids against thresholds C02, start/end
    stamps equal, build_key from session.json or harvest.json, use_class rules, the L8 fingerprint, the launch record
    (--launch-record), expert/llc SHAs in the allow-list (--allow-list) and the split derived from the route tags."""
    if ctx.load_error:
        return unreadable(ctx, "C02")
    s = ctx.session
    thr = ctx.thr("C02")
    checks = []
    skipped = []
    sha = s.meta_get("addon.git_sha")
    checks.append(("git_sha", isinstance(sha, str) and bool(HEX40.match(sha)), "addon.git_sha %r" % (sha,)))
    dirty = s.meta_get("addon.git_dirty")
    if ctx.rank >= 2:
        checks.append(("git_dirty", dirty in (False, 0), "addon.git_dirty %r (must be false from T1)" % (dirty,)))
    pin_id = s.meta_get("pin.pin_id")
    want = thr.get("pin_id")
    checks.append(("pin_id", isinstance(pin_id, str) and (pin_id == want or pin_id.startswith(str(want) + "-")),
                   "pin %r, thresholds %r (pins of one build_key: <pin_id>-<suffix>)" % (pin_id, want)))
    for key in ("game_buildid", "tools_buildid"):
        checks.append((key, s.meta_get("pin." + key) == thr.get(key),
                       "pin.%s %r, thresholds %r" % (key, s.meta_get("pin." + key), thr.get(key))))
    if s.end is not None:
        for key in ("pin.pin_id", "pin.build_key", "engine.build_version"):
            checks.append(("end." + key, s.end_get(key) == s.meta_get(key),
                           "start %r, end %r" % (s.meta_get(key), s.end_get(key))))
    build_key = s.build_key()
    checks.append(("build_key", bool(build_key),
                   "build_key %s" % ("from " + ("session.json" if s.meta_get("pin.build_key") else "harvest.json")
                                     if build_key else "missing from session.json and harvest.json")))
    use_class = s.meta_get("use_class")
    mode = s.meta_get("launch_guard.launch_mode")
    if use_class == "production":
        checks.append(("use_class", False, "production needs a qualified G19 record for build_key %s; G19 records "
                       "are not supported yet (DD-14b)" % build_key))
    if mode == "wb_escape":
        checks.append(("wb_escape", use_class in ("dev", "probe", "appearance"),
                       "wb_escape session with use_class %r (allowed: dev, probe, appearance)" % use_class))
    checks.append(("fingerprint", s.meta_get("map.fingerprint.ok") in (True, 1), "L8 map.fingerprint.ok %r"
                   % (s.meta_get("map.fingerprint.ok"),)))
    record = getattr(ctx.options, "launch_record", None)
    if record:
        _launch_record_checks(ctx, s, record, checks)
    else:
        skipped.append("launch record (pass --launch-record; the launch guard refused to write without an ok "
                       "record)")
    allow = getattr(ctx.options, "allow_list", None)
    if allow:
        with open(allow, "rb") as f:
            lists = json.loads(f.read().decode("utf-8"))
        for key in ("expert_sha", "llc_sha"):
            value = s.meta_get("control." + key)
            if value is not None:
                checks.append((key, value in (lists.get(key) or []), "control.%s %s" % (key, str(value)[:12])))
    else:
        skipped.append("expert_sha/llc_sha allow-list (none exists before DD-7-min; pass --allow-list)")
    splits = set()
    for i in range(len(s.streams["episodes"])):
        rid = s.streams["episodes"].col("route_id")[i]
        if not rid:
            continue
        route, _, err = ctx.route(rid)
        if route is None:
            skipped.append("split of route %s (%s)" % (rid, err))
            continue
        checks.append(("route_map." + rid, route.map_id == s.map_id, "route map %s, session map %s"
                       % (route.map_id, s.map_id)))
        splits.add(route.split)
    split = None
    if len(splits) == 1:
        split = splits.pop()
        if split == "train" and int(hashlib.sha256(s.session_id.encode("ascii")).hexdigest()[:8], 16) % 10 == 0:
            split = "val_c"
    elif len(splits) > 1:
        checks.append(("split", False, "episodes use routes of several splits: %s" % sorted(splits)))
    bad = ["%s: %s" % (name, detail) for name, ok, detail in checks if not ok]
    detail = first(bad) if bad else "%d provenance checks ok; split %s; build_key %s" % (
        len(checks), split, str(build_key)[:16])
    return result(ctx, "C02", HARD, FAIL if bad else PASS, {"failed": len(bad), "split": split}, "exact", detail,
                  not_evaluated=skipped)
