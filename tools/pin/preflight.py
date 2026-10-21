#!/usr/bin/env python3
"""Build-pin preflight for the DriveDiT launchers (ticket DD-0 / program TKT-0).

Every launcher runs this through `wsl.exe -e python3 -I -B .../preflight.py --pin ... --exe wb|client`
before it starts an exe. It refuses the launch unless, for BOTH the game and the Tools app:
  - buildid equals the pin, and the installed depot manifests equal the pin;
  - TargetBuildID == buildid (no update pending) and StateFlags is the settled value from the pin;
  - AutoUpdateBehavior is an allowed value ("1" = only update when launched; "0" and "2" auto-update);
  - every pinned *.pak exists with the pinned size and mtime, and no extra *.pak exists;
  - the launched exes and steam_appid.txt files match the pin.

Launch mode (default): on success prints ONLY the launch id on stdout, writes
<launch-dir>/launch_<id>.json (default launch dir: <pin dir>/launches) and exits 0. On refusal prints
nothing on stdout, explains on stderr, writes refused_<id>.json (best effort) and exits 1.
Check mode (--check): prints a PASS/FAIL report on stdout, writes nothing, exits 0 or 1.
--allow-fail, --mtime-resolution-s and the --game-acf/--tools-acf/--game-root/--tools-root overrides are
accepted only with --check, so a launch is always checked, in full, against the pinned paths.

Exit codes: 0 ok; 1 refused; 2 usage/config error (pin unreadable, bad path); 3 launch record not written.
Writes are allowed only under /mnt/e (C: is always refused).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys


def _load_pinlib():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinlib.py")
    name = "dd_pin_pinlib"
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pinlib = _load_pinlib()

CHECK_SUFFIXES = ("acf", "appid", "buildid", "target_buildid", "auto_update_behavior", "state_flags",
                  "depots", "steam_appid_txt", "binaries", "paks")
CHECK_NAMES = tuple("%s.%s" % (app, s) for app in pinlib.APPS for s in CHECK_SUFFIXES)
MAX_LISTED = 8


def _fmt_list(items):
    items = list(items)
    more = len(items) - MAX_LISTED
    return "; ".join(items[:MAX_LISTED]) + ("; ... %d more" % more if more > 0 else "")


def _same_mtime(have_ns: int, want_ns, resolution_s: int) -> bool:
    if not isinstance(want_ns, int):
        return False
    if resolution_s <= 0:
        return have_ns == want_ns
    step = resolution_s * 10**9
    return have_ns // step == want_ns // step


def _cmp_entry(pinned: dict, path: str, mtime_resolution_s: int = 0):
    """None if the file at path matches the pinned size and mtime, else a short reason.

    mtime_resolution_s=0 compares mtime_ns exactly (live installs). A copy on E: needs 1: WSL drvfs
    reads sub-second mtimes but truncates them to whole seconds when setting them (measured on E:).
    """
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    diffs = []
    if st.st_size != pinned.get("size"):
        diffs.append("size %d != pin %s" % (st.st_size, pinned.get("size")))
    if not _same_mtime(st.st_mtime_ns, pinned.get("mtime_ns"), mtime_resolution_s):
        diffs.append("mtime %s != pin %s" % (pinlib.iso_utc_from_ns(st.st_mtime_ns), pinned.get("mtime_utc")))
    return ", ".join(diffs) or None


def check_app(app: str, pin_app: dict, steam_policy: dict, acf_path: str, root: str, mtime_resolution_s: int = 0):
    """Return (checks, observed) for one app. checks: list of {name, ok, detail}."""
    checks = []

    def add(suffix, ok, detail):
        checks.append({"name": "%s.%s" % (app, suffix), "ok": bool(ok), "detail": detail})

    observed = None
    try:
        acf = pinlib.load_acf(acf_path)
        observed = acf
        add("acf", True, "read %s" % acf_path)
    except pinlib.PinError as exc:
        add("acf", False, "cannot read appmanifest: %s" % exc)
        for suffix in ("appid", "buildid", "target_buildid", "auto_update_behavior", "state_flags", "depots"):
            add(suffix, False, "appmanifest unreadable")
        acf = None

    if acf is not None:
        add("appid", acf["appid"] == pin_app["appid"], "appid %s (pin %s)" % (acf["appid"], pin_app["appid"]))
        add("buildid", acf["buildid"] == pin_app["buildid"],
            "buildid %s (pin %s)%s" % (acf["buildid"], pin_app["buildid"],
                                       "" if acf["buildid"] == pin_app["buildid"]
                                       else ": Steam installed a different build"))
        tgt_ok = acf["target_buildid"] is not None and acf["target_buildid"] == acf["buildid"]
        add("target_buildid", tgt_ok,
            "TargetBuildID %s, buildid %s%s" % (acf["target_buildid"], acf["buildid"],
                                                 "" if tgt_ok else ": an update is pending or downloading"))
        # A pin can only narrow the code policy (AUTO_UPDATE_ALLOWED: "1"), never widen it. A hand-edited pin
        # that lists "0" (always auto-update) or "2" (high priority, also auto-updates) does not let them through.
        allowed = [str(v) for v in steam_policy.get("auto_update_behavior_allowed", pinlib.AUTO_UPDATE_ALLOWED)
                   if str(v) in pinlib.AUTO_UPDATE_ALLOWED]
        aub = acf["auto_update_behavior"]
        if aub in allowed:
            add("auto_update_behavior", True, "AutoUpdateBehavior %r (%s)" % (aub, pinlib.AUTO_UPDATE_MEANING.get(aub)))
        else:
            add("auto_update_behavior", False,
                "AutoUpdateBehavior is %r (%s); allowed %s. Fix: Steam > Library > %s > Properties > Updates > "
                "\"Only update this game when I launch it\""
                % (aub, pinlib.AUTO_UPDATE_MEANING.get(aub, "unknown value"), allowed, pin_app.get("name") or app))
        want_flags = steam_policy.get("require_state_flags", pin_app.get("state_flags"))
        add("state_flags", acf["state_flags"] == want_flags,
            "StateFlags %r (pin %r)" % (acf["state_flags"], want_flags))
        dep_diff = []
        for depot in sorted(set(acf["depots"]) | set(pin_app["depots"])):
            have = acf["depots"].get(depot, {}).get("manifest")
            want = pin_app["depots"].get(depot, {}).get("manifest")
            if have != want:
                dep_diff.append("depot %s manifest %s != pin %s" % (depot, have, want))
        add("depots", not dep_diff,
            _fmt_list(dep_diff) if dep_diff else "manifests %s" % ",".join(
                "%s:%s" % (d, v["manifest"]) for d, v in sorted(pin_app["depots"].items())))

    # Files: compared against the pin even when the acf is unreadable.
    appid_diff = []
    for rel, want in sorted(pin_app.get("steam_appid_txt", {}).items()):
        full = os.path.join(root, *rel.split("/"))
        try:
            have = pinlib.read_small_text(full)
        except OSError:
            have = None
        if have != want:
            appid_diff.append("%s is %r (pin %r)" % (rel, have, want))
    add("steam_appid_txt", not appid_diff, _fmt_list(appid_diff) if appid_diff else "match")

    bin_diff = []
    for rel, pinned in sorted(pin_app.get("binaries", {}).items()):
        why = _cmp_entry(pinned, os.path.join(root, *rel.split("/")), mtime_resolution_s)
        if why:
            bin_diff.append("%s: %s" % (rel, why))
    add("binaries", not bin_diff, _fmt_list(bin_diff) if bin_diff else "%d exes match" % len(pin_app.get("binaries", {})))

    pak_diff = []
    pinned_paks = pin_app.get("paks", {})
    if not pinned_paks:
        pak_diff.append("pin lists no paks")
    if os.path.isdir(root):
        for rel, pinned in sorted(pinned_paks.items()):
            why = _cmp_entry(pinned, os.path.join(root, *rel.split("/")), mtime_resolution_s)
            if why:
                pak_diff.append("%s: %s" % (rel, why))
        try:
            present = pinlib.scan_paks(root)
        except OSError as exc:
            present = {}
            pak_diff.append("cannot scan %s: %s" % (root, exc))
        for rel in sorted(set(present) - set(pinned_paks)):
            pak_diff.append("%s: not in pin (new pak)" % rel)
    else:
        pak_diff.append("install dir missing: %s" % root)
    add("paks", not pak_diff,
        _fmt_list(pak_diff) if pak_diff else "%d paks match (size + mtime)" % len(pinned_paks))
    return checks, observed


def run_checks(pin: dict, acf_paths: dict, roots: dict, mtime_resolution_s: int = 0):
    checks, observed = [], {}
    for app in pinlib.APPS:
        c, o = check_app(app, pin[app], pin.get("steam", {}), acf_paths[app], roots[app], mtime_resolution_s)
        checks.extend(c)
        observed[app] = o
    return checks, observed


def load_pin(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            pin = json.load(fh)
    except (OSError, ValueError) as exc:
        raise pinlib.PinError("cannot read pin %s: %s" % (path, exc))
    if not isinstance(pin, dict) or pin.get("schema") != pinlib.PIN_SCHEMA:
        raise pinlib.PinError("%s is not a %s pin file" % (path, pinlib.PIN_SCHEMA))
    for app in pinlib.APPS:
        a = pin.get(app)
        if not isinstance(a, dict) or not all(k in a for k in ("appid", "buildid", "depots", "paks", "acf", "install")):
            raise pinlib.PinError("pin %s: section %r is incomplete" % (path, app))
    return pin


def new_launch_id(exe: str) -> str:
    return "%s-%s-%s" % (pinlib.utc_compact(pinlib.utc_now()), exe, os.urandom(2).hex())


def make_record(status, launch_id, args, cli_argv, pin_path, pin, checks, observed) -> dict:
    return {
        "schema": pinlib.LAUNCH_SCHEMA,
        "launch_id": launch_id,
        "status": status,
        "utc": pinlib.utc_iso(pinlib.utc_now()),
        "exe": args.exe,
        "instance": args.instance,
        "pin": {"path": pin_path, "pin_id": pin.get("pin_id"), "sha256": pinlib.sha256_file(pin_path)},
        "observed": observed,
        "checks": checks,
        "host": {"hostname": socket.gethostname(), "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
                 "python": sys.version.split()[0]},
        "argv": cli_argv,
    }


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", required=True, help="pin JSON written by make_pin.py")
    ap.add_argument("--exe", choices=("wb", "client"), default=None,
                    help="which exe the launcher is about to start (required unless --check)")
    ap.add_argument("--instance", default=None, help="client instance id, e.g. i01 (recorded only)")
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    ap.add_argument("--allow-fail", action="append", default=[], metavar="CHECK",
                    help="with --check only: do not count this check as a failure (e.g. for the cold copy)")
    ap.add_argument("--mtime-resolution-s", type=int, default=0, metavar="N",
                    help="with --check only: compare mtimes at N-second resolution (1 for a copy on E:)")
    ap.add_argument("--launch-dir", default=None, help="where launch records go (default: <pin dir>/launches)")
    ap.add_argument("--game-acf", default=None, help="with --check only: override the appmanifest path in the pin")
    ap.add_argument("--tools-acf", default=None, help="with --check only")
    ap.add_argument("--game-root", default=None,
                    help="with --check only: override the install dir in the pin (the cold copy check uses it)")
    ap.add_argument("--tools-root", default=None, help="with --check only")
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(cli_argv)

    if not args.check and args.exe is None:
        print("preflight: --exe wb|client is required unless --check", file=sys.stderr)
        return 2
    if args.allow_fail and not args.check:
        print("preflight: --allow-fail is only accepted with --check (a launch can never skip a check)",
              file=sys.stderr)
        return 2
    if args.mtime_resolution_s and not args.check:
        print("preflight: --mtime-resolution-s is only accepted with --check (launches compare mtimes exactly)",
              file=sys.stderr)
        return 2
    overrides = [flag for flag, val in (("--game-acf", args.game_acf), ("--tools-acf", args.tools_acf),
                                        ("--game-root", args.game_root), ("--tools-root", args.tools_root)) if val]
    if overrides and not args.check:
        # A launch record must describe the files the launcher really starts: the pinned acf and install paths.
        print("preflight: %s only accepted with --check (a launch is always checked against the pinned paths)"
              % ", ".join(overrides), file=sys.stderr)
        return 2
    unknown = [n for n in args.allow_fail if n not in CHECK_NAMES]
    if unknown:
        print("preflight: unknown check name(s) %s; known: %s" % (unknown, ", ".join(CHECK_NAMES)), file=sys.stderr)
        return 2

    pin_path = os.path.abspath(pinlib.win_to_wsl(args.pin))
    launch_dir = None
    try:
        pin = load_pin(pin_path)
        if not args.check:
            launch_dir = pinlib.check_write_path(args.launch_dir or os.path.join(os.path.dirname(pin_path), "launches"),
                                                 write_roots)
    except pinlib.PinError as exc:
        print("preflight: %s" % exc, file=sys.stderr)
        return 2

    acf_paths = {"game": args.game_acf or pin["game"]["acf"]["wsl"],
                 "tools": args.tools_acf or pin["tools"]["acf"]["wsl"]}
    roots = {"game": args.game_root or pin["game"]["install"]["wsl"],
             "tools": args.tools_root or pin["tools"]["install"]["wsl"]}
    checks, observed = run_checks(pin, acf_paths, roots, max(0, args.mtime_resolution_s))
    failed = [c for c in checks if not c["ok"] and c["name"] not in args.allow_fail]

    if args.check:
        for c in checks:
            tag = "PASS" if c["ok"] else ("WAIVED" if c["name"] in args.allow_fail else "FAIL")
            print("%-6s %-27s %s" % (tag, c["name"], c["detail"]))
        if failed:
            print("RESULT: REFUSED (pin %s): %d check(s) failed: %s"
                  % (pin.get("pin_id"), len(failed), ", ".join(c["name"] for c in failed)))
            return 1
        print("RESULT: OK (pin %s)" % pin.get("pin_id"))
        return 0

    launch_id = new_launch_id(args.exe)
    if failed:
        print("PREFLIGHT REFUSED LAUNCH of %s (pin %s, %s):" % (args.exe, pin.get("pin_id"), pin_path),
              file=sys.stderr)
        for c in failed:
            print("  FAIL %s: %s" % (c["name"], c["detail"]), file=sys.stderr)
        try:
            os.makedirs(launch_dir, exist_ok=True)
            rec = pinlib.write_json_new(os.path.join(launch_dir, "refused_%s.json" % launch_id),
                                        make_record("refused", launch_id, args, cli_argv, pin_path, pin, checks, observed),
                                        write_roots)
            print("  record: %s" % rec, file=sys.stderr)
        except (pinlib.PinError, OSError) as exc:
            print("  (refusal record not written: %s)" % exc, file=sys.stderr)
        return 1

    try:
        os.makedirs(launch_dir, exist_ok=True)
        rec = pinlib.write_json_new(os.path.join(launch_dir, "launch_%s.json" % launch_id),
                                    make_record("ok", launch_id, args, cli_argv, pin_path, pin, checks, observed),
                                    write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("PREFLIGHT REFUSED LAUNCH of %s: launch record not written: %s" % (args.exe, exc), file=sys.stderr)
        return 3
    print("PREFLIGHT OK: %s matches pin %s; record %s" % (args.exe, pin.get("pin_id"), rec), file=sys.stderr)
    print(launch_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
