#!/usr/bin/env python3
"""Direct-launch freeze check of an app pair (ticket DD-14a; MIGRATION_1_9 5.5, CC-01).

"Only update this game when I launch it" is not a proven freeze for direct-exe launches: the preflight catches buildid
drift before a launch, not during or after it. So, once per app pair, with Steam running online:

  1. before:  the agent records both appmanifests (buildid, TargetBuildID, StateFlags, AutoUpdateBehavior, depots,
              LastUpdated, file size and mtime) and the pinned exes, after the full preflight check of the pin
              passes. Written to <pin dir>/freeze/freeze_<pin id>_<UTC>.before.json.
  2. the user starts the app through its launcher (wb_R1.bat; the client launcher too once it can run) and closes it.
  3. after:   the agent re-reads everything and lists the launch records written in between for this pin.
              Written to ...after.json with the verdict.

PASS: the build identity in both appmanifests is unchanged (buildid, TargetBuildID, StateFlags, AutoUpdateBehavior,
depot manifests, LastUpdated), every pinned exe and pak is unchanged (the full preflight check passes again), and at
least one REAL launch of this pin happened between the two steps (Codex round 1): a launch record of this pin file
(a JSON object with the dd-launch fields, status ok, exe wb) AND that launch's own console.log, under the launch
path's Workbench profile, naming its -dcLaunchId, started after the before-record and showing "Creating game
instance(...), version X" (equal to the pin's engine_version_expected once it has one). A preflight run alone (the
launcher refused after it, or no exe started) never passes. A rewrite of an appmanifest that changes none of the
identity fields (Steam updating LastPlayed at exit, measured 2026-11-07) is reported, not failed. Steam's online state
cannot be read from WSL: --steam online|offline records what the user says, and only online qualifies for 5.5; an
offline run ends PASS_UNQUALIFIED, which CC-01 does not accept.

    python3 -I -B freeze_check.py before --pin-active --steam online [--dry-run]
    python3 -I -B freeze_check.py after --before <DATA_ROOT>/pin/freeze/freeze_<id>_<UTC>.before.json [--dry-run]

Exit codes: 0 written (before), PASS (after) or dry run; 1 refused (the pin does not pass before), FAIL or
PASS_UNQUALIFIED (after); 2 usage, config or path error. Writes only on the data drive (pinlib.check_write_path),
and only the freeze/ records. <DATA_ROOT> is the local config's (tools/local_config.py).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys


def _load(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    modname = "dd_pin_" + name
    mod = sys.modules.get(modname)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pinlib = _load("pinlib")
preflight = _load("preflight")

SCHEMA = "dd-freeze/1"
FREEZE_DIR = "freeze"
IDENTITY_FIELDS = ("appid", "buildid", "target_buildid", "state_flags", "auto_update_behavior", "last_updated",
                   "depots")


def snapshot(src: dict) -> dict:
    """Appmanifests (identity fields, file stat and sha256; never LastOwner) and the pinned exes of both apps."""
    acf_paths, roots = preflight.pinned_paths(src["pin"])
    out = {}
    for app in pinlib.APPS:
        acf = pinlib.load_acf(acf_paths[app])
        raw = pinlib.read_regular_file(acf_paths[app], "appmanifest", 1 << 20)
        last_played = None
        for line in raw.decode("utf-8", errors="replace").splitlines():
            parts = line.strip().split('"')
            if len(parts) >= 4 and parts[1].lower() == "lastplayed":
                last_played = parts[3]
        st = os.stat(acf_paths[app])
        out[app] = {
            "acf_path": acf_paths[app],
            "acf": {k: acf[k] for k in IDENTITY_FIELDS},
            "acf_file": {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": hashlib.sha256(raw).hexdigest(),
                         "last_played": last_played},
            "exes": {rel: pinlib.stat_entry(os.path.join(roots[app], *rel.split("/")))
                     for rel in sorted(src["pin"][app]["binaries"])},
        }
    return out


def launches_between(src: dict, start_utc: str, end_utc: str) -> list:
    """Launch records of this pin (same pin file sha256) with start_utc <= utc <= end_utc, oldest first. A record that
    is not a JSON object with the dd-launch fields is skipped (never counted)."""
    ldir = os.path.join(src["pin_dir"], "launches")
    found = []
    if not os.path.isdir(ldir):
        return found
    for name in sorted(os.listdir(ldir)):
        if not (name.startswith("launch_") and name.endswith(".json")):
            continue
        try:
            rec = json.loads(pinlib.read_regular_file(os.path.join(ldir, name), "launch record", 1 << 22))
        except (pinlib.PinError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("schema") not in ("dd-launch/1", "dd-launch/2"):
            continue
        pin = rec.get("pin") if isinstance(rec.get("pin"), dict) else {}
        utc = rec.get("utc")
        lid = rec.get("launch_id")
        if (isinstance(utc, str) and pinlib.iso_utc_to_unix(utc) is not None and start_utc <= utc <= end_utc
                and isinstance(lid, str) and name == "launch_%s.json" % lid and pin.get("pin_id") == src["pin_id"]
                and pin.get("sha256") == src["sha256"] and rec.get("status") == "ok"):
            found.append({"launch_id": lid, "exe": rec.get("exe"), "utc": utc})
    return found


def launch_evidence(src: dict, ident: dict, launch: dict, before_unix: int):
    """(evidence or None, why not): the console.log of a wb launch, under the launch path's Workbench profile, that
    names its -dcLaunchId, started after the before-record and reports the build."""
    if launch.get("exe") != "wb":
        return None, "launch %s is a %s launch; only Workbench starts leave a log here" % (launch["launch_id"],
                                                                                         launch.get("exe"))
    wb = pinlib.WB_LAUNCH.get(ident["launch_path_id"])
    if wb is None:
        return None, "launch path %s has no Workbench launcher" % ident["launch_path_id"]
    logs = pinlib.launch_logs(pinlib.win_to_wsl(wb["profile"]), launch["launch_id"], before_unix)
    if not logs:
        return None, ("launch %s has no console.log under %s\\logs started after the before-record (the preflight "
                      "ran, but no Workbench start was observed)" % (launch["launch_id"], wb["profile"]))
    log = logs[-1]
    if log["version"] is None:
        return None, "the console.log of launch %s shows no 'Creating game instance' line" % launch["launch_id"]
    want = ident.get("engine_version_expected")
    if want and log["version"] != want:
        return None, "launch %s reports version %s, the pin expects %s" % (launch["launch_id"], log["version"], want)
    return {"launch_id": launch["launch_id"], "exe": launch["exe"], "console_log": log["path"],
            "log_start_unix": log["start_unix"], "build_version": log["version"]}, None


def compare(before: dict, after: dict) -> tuple:
    """(identity diffs, notes): identity diffs fail the check; notes (appmanifest rewrites) do not."""
    diffs, notes = [], []
    for app in pinlib.APPS:
        b, a = before[app], after[app]
        for key in IDENTITY_FIELDS:
            if b["acf"][key] != a["acf"][key]:
                diffs.append("%s.%s %r -> %r" % (app, key, b["acf"][key], a["acf"][key]))
        for rel in sorted(set(b["exes"]) | set(a["exes"])):
            if b["exes"].get(rel) != a["exes"].get(rel):
                diffs.append("%s exe %s changed" % (app, rel))
        if b["acf_file"]["sha256"] != a["acf_file"]["sha256"] or b["acf_file"]["mtime_ns"] != a["acf_file"]["mtime_ns"]:
            notes.append("%s appmanifest rewritten without a build change (LastPlayed %s -> %s)"
                         % (app, b["acf_file"]["last_played"], a["acf_file"]["last_played"]))
    return diffs, notes


def _out_dir(src, write_roots, create: bool) -> str:
    d = pinlib.check_write_path(os.path.join(src["pin_dir"], FREEZE_DIR), write_roots)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def cmd_before(args, write_roots) -> int:
    try:
        src = preflight.resolve_pin(args.pin, args.pin_active, args.pin_dir)
        ident = preflight.pin_identity(src)
        out_dir = _out_dir(src, write_roots, create=False)
    except pinlib.PinError as exc:
        print("freeze_check: %s" % exc, file=sys.stderr)
        return 2
    checks, _ = preflight.full_checks(src, ident)
    failed = [c for c in checks if not c["ok"]]
    if failed:
        for c in failed:
            print("FAIL %s: %s" % (c["name"], c["detail"]), file=sys.stderr)
        print("freeze_check: pin %s does not match the installs; nothing to freeze-check" % src["pin_id"],
              file=sys.stderr)
        return 1
    now = pinlib.utc_now()
    rec = {"schema": SCHEMA, "step": "before", "utc": pinlib.utc_iso(now), "pin_id": src["pin_id"],
           "pin_sha256": src["sha256"], "pin_path": src["path"], "build_key": ident["build_key"],
           "app_pair": {app: src["pin"][app]["appid"] for app in pinlib.APPS}, "steam_state_user": args.steam,
           "preflight_checks_passed": len(checks), "snapshot": snapshot(src)}
    path = os.path.join(out_dir, "freeze_%s_%s.before.json" % (src["pin_id"], pinlib.utc_compact(now)))
    print(json.dumps(rec, indent=2))
    if args.dry_run:
        print("DRY RUN: would write %s; nothing written" % path, file=sys.stderr)
        return 0
    try:
        _out_dir(src, write_roots, create=True)
        written = pinlib.write_json_once(path, rec, write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("freeze_check: %s" % exc, file=sys.stderr)
        return 2
    print("wrote %s\nNow start the app through its launcher (wb_R1.bat), let it load, close it, then run:\n"
          "  python3 -I -B %s after --before %s" % (written, os.path.abspath(__file__), written), file=sys.stderr)
    return 0


def cmd_after(args, write_roots) -> int:
    try:
        before = json.loads(pinlib.read_regular_file(os.path.abspath(args.before), "before record", 1 << 22))
        if not isinstance(before, dict) or before.get("schema") != SCHEMA or before.get("step") != "before":
            raise pinlib.PinError("%s is not a %s before record" % (args.before, SCHEMA))
        pin_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.before)))
        src = preflight.load_pin_by_id(pin_dir, before["pin_id"])
        if src["sha256"] != before["pin_sha256"]:
            raise pinlib.PinError("pin %s changed since the before step" % before["pin_id"])
        ident = preflight.pin_identity(src)
        out_dir = _out_dir(src, write_roots, create=False)
    except (pinlib.PinError, ValueError, KeyError) as exc:
        print("freeze_check: %s" % exc, file=sys.stderr)
        return 2
    now = pinlib.utc_now()
    after = snapshot(src)
    diffs, notes = compare(before["snapshot"], after)
    checks, _ = preflight.full_checks(src, ident)
    failed = [c["name"] for c in checks if not c["ok"]]
    launches = launches_between(src, before["utc"], pinlib.utc_iso(now))
    before_unix = pinlib.iso_utc_to_unix(before["utc"])
    evidence, missing = [], []
    for launch in launches:
        ev, why = launch_evidence(src, ident, launch, before_unix)
        (evidence.append(ev) if ev else missing.append(why))
    problems = list(diffs)
    if failed:
        problems.append("preflight checks fail now: %s" % ", ".join(failed))
    if not launches:
        problems.append("no launch record of pin %s between %s and now: start the app through its launcher between "
                        "the two steps" % (src["pin_id"], before["utc"]))
    elif not evidence:
        problems.append("no launch between the two steps left evidence of a real start: %s" % "; ".join(missing))
    qualified = before.get("steam_state_user") == "online"
    verdict = "FAIL" if problems else ("PASS" if qualified else "PASS_UNQUALIFIED")
    rec = {"schema": SCHEMA, "step": "after", "utc": pinlib.utc_iso(now), "pin_id": src["pin_id"],
           "pin_sha256": src["sha256"], "build_key": ident["build_key"], "app_pair": before["app_pair"],
           "before": {"path": os.path.abspath(args.before), "utc": before["utc"],
                      "sha256": pinlib.sha256_file(os.path.abspath(args.before))},
           "steam_state_user": before.get("steam_state_user"), "qualified_5_5": qualified, "launches": launches,
           "launch_evidence": evidence, "launches_without_evidence": missing,
           "apps_launched": sorted({"tools" if r["exe"] == "wb" else "game" for r in evidence}),
           "identity_diffs": diffs, "notes": notes, "problems": problems, "verdict": verdict, "snapshot": after}
    path = os.path.join(out_dir, os.path.basename(args.before).replace(".before.json", ".after.json"))
    print(json.dumps(rec, indent=2))
    print("FREEZE %s (pin %s, app pair %s): %s%s" % (
        verdict, src["pin_id"], before["app_pair"], "; ".join(problems) or "nothing changed across %d real launch(es)"
        % len(evidence), "" if qualified or problems else "; Steam was offline, so this does not qualify for 5.5"),
        file=sys.stderr)
    if args.dry_run:
        print("DRY RUN: would write %s; nothing written" % path, file=sys.stderr)
        return 0 if verdict == "PASS" else 1
    try:
        written = pinlib.write_json_once(path, rec, write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("freeze_check: %s" % exc, file=sys.stderr)
        return 2
    print("wrote %s" % written, file=sys.stderr)
    return 0 if verdict == "PASS" else 1


def find_pass(pin_dir: str, pin_sha256: str, build_key) -> list:
    """after records with verdict PASS (Steam online, at least one launch with log evidence) for this pin file (or,
    when given, this build_key): the CC-01 evidence."""
    d = os.path.join(pin_dir, FREEZE_DIR)
    out = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".after.json"):
            continue
        try:
            rec = json.loads(pinlib.read_regular_file(os.path.join(d, name), "freeze record", 1 << 22))
        except (pinlib.PinError, ValueError):
            continue
        if rec.get("schema") == SCHEMA and rec.get("verdict") == "PASS" and rec.get("qualified_5_5") is True \
                and rec.get("launch_evidence") and (
                rec.get("pin_sha256") == pin_sha256 or (build_key and rec.get("build_key") == build_key)):
            out.append(os.path.join(d, name))
    return out


def main(argv=None, write_roots=None) -> int:
    """write_roots: None for pinlib.DEFAULT_WRITE_ROOTS (the data drive of the local config)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("before")
    which = b.add_mutually_exclusive_group(required=True)
    which.add_argument("--pin", default=None)
    which.add_argument("--pin-active", action="store_true")
    b.add_argument("--pin-dir", default=None)
    b.add_argument("--steam", choices=("online", "offline"), required=True,
                   help="Steam's state during the test, as the user reports it (5.5 asks for online)")
    b.add_argument("--dry-run", action="store_true")
    a = sub.add_parser("after")
    a.add_argument("--before", required=True)
    a.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.step == "before":
        if args.pin_dir and not args.pin_active:
            print("freeze_check: --pin-dir is only used with --pin-active", file=sys.stderr)
            return 2
        return cmd_before(args, write_roots)
    return cmd_after(args, write_roots)


if __name__ == "__main__":
    sys.exit(main())
