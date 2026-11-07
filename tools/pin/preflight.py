#!/usr/bin/env python3
"""Build-pin preflight for the DriveDiT launchers (ticket DD-0 / program TKT-0).

Every launcher runs this through `wsl.exe -e /usr/bin/python3 -I -B .../preflight.py --pin-active --exe wb|client`
before it starts an exe.

Which pin: --pin-active (what the launchers use) reads <pin dir>/ACTIVE (default pin dir /mnt/e/DriveDiT/pin;
--pin-dir overrides). ACTIVE must be a regular file, not a symlink, holding exactly one valid pin id, such as
1.8.0.13 or 1.9.0.9-exp. The pin is then <pin dir>/pin_<id>.json, which must be a regular file, not a symlink,
whose pin_id equals that id. Anything else is a config error (exit 2). Only pin_activate.py writes ACTIVE, and it
refuses a pin that does not match the live installs. --pin PATH names a pin file explicitly; the two options are
mutually exclusive. 1.8.0.13 is the interim pin; the plan is to move to 1.9 by activating a new pin.

It refuses the launch unless, for BOTH the game and the Tools app:
  - buildid equals the pin, and the installed depot manifests equal the pin;
  - TargetBuildID == buildid (no update pending) and StateFlags is the settled value from the pin;
  - AutoUpdateBehavior is an allowed value ("1" = only update when launched; "0" and "2" auto-update);
  - every pinned *.pak exists with the pinned size and mtime, and no extra *.pak exists;
  - the launched exes and steam_appid.txt files match the pin;
  - the pin was made from the appmanifest and install dir the launchers start (pinlib.LAUNCH_PATHS; check
    <app>.launch_path), so a pin made from another tree can never vouch for the exe a launcher starts.

DD-14a (dd-pin/2): a pin has a channel (stable, experimental, cold) and a launch-path id; a launcher names its own
(--launch-path, default stable) and <app>.launch_path refuses a pin made for another. When the pin's exe hashes are
known (dd-pin/2, or the build_key sidecar pin_<id>.build_key.json of a dd-pin/1 pin) <app>.exe_sha256 hashes every
pinned exe. --pin-active refuses an experimental pin (exit 2). The launch record becomes dd-launch/2 when there is
something to add (a known build_key, a dd-pin/2 pin, a non-stable channel or --use-class): top-level build_key,
channel (the release channel), pin_channel, launch_path_id, use_class, physics_hz_expected and, from the engine
sidecar pin_<id>.engine.json, engine_version_expected. A dd-pin/1 pin without sidecars keeps the dd-launch/1 record.

Launch mode (default): on success prints ONLY the launch id on stdout, writes
<launch-dir>/launch_<id>.json (default launch dir: <pin dir>/launches) and exits 0. The record names the pin id
and how it was resolved (active pointer or explicit path). On refusal prints
nothing on stdout, explains on stderr, writes refused_<id>.json (best effort) and exits 1.
Check mode (--check): prints a PASS/FAIL report on stdout, writes nothing, exits 0 or 1.
--allow-fail, --mtime-resolution-s and the --game-acf/--tools-acf/--game-root/--tools-root overrides are
accepted only with --check, so a launch is always checked, in full, against the pinned paths.

Exit codes: 0 ok; 1 refused; 2 usage/config error (pin unreadable, bad path); 3 launch record not written.
Writes are allowed only under /mnt/e (C: is always refused).
"""
from __future__ import annotations

import argparse
import hashlib
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
                  "depots", "steam_appid_txt", "binaries", "paks", "launch_path")
# The 22 checks of DD-0; every launch record names them (validator C02 requires all 22).
CHECK_NAMES = tuple("%s.%s" % (app, s) for app in pinlib.APPS for s in CHECK_SUFFIXES)
# DD-14a: exe_sha256 runs when the pin's exe hashes are known (dd-pin/2, or a dd-pin/1 pin with its build_key
# sidecar); pak_sha256 only with --verify-hashes (check mode, a pin that records pak hashes: a cold pin).
EXTRA_CHECK_SUFFIXES = ("exe_sha256", "pak_sha256")
ALL_CHECK_NAMES = CHECK_NAMES + tuple("%s.%s" % (app, s) for app in pinlib.APPS for s in EXTRA_CHECK_SUFFIXES)
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


def _pinned_path(pin_app: dict, key: str):
    v = pin_app.get(key)
    return v.get("wsl") if isinstance(v, dict) else None


def check_app(app: str, pin_app: dict, steam_policy: dict, acf_path: str, root: str, mtime_resolution_s: int = 0,
              launch_path=None, exe_sha256=None, launch_path_mismatch=None, verify_hashes: bool = False):
    """Return (checks, observed) for one app. checks: list of {name, ok, detail}.

    launch_path is {"acf", "install"}, the paths the launchers start for this app (pinlib.LAUNCH_PATHS[app]). The
    launch_path check compares them with the paths stored IN THE PIN, not with acf_path/root, which --check may
    point at another tree (the cold copy). launch_path_mismatch, when set, says the pin was made for another
    launch-path id than the launcher's, and fails that check too.

    exe_sha256 ({exe rel path: sha256}, DD-14a) adds the exe_sha256 check: every pinned exe is hashed and compared.
    verify_hashes adds pak_sha256: every pinned pak that records a sha256 (a cold pin) is hashed and compared.
    """
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

    if exe_sha256 is not None:
        sha_diff = []
        want_rels = set(pin_app.get("binaries", {}))
        if set(exe_sha256) != want_rels:
            sha_diff.append("hashed exes %s != pinned exes %s" % (sorted(exe_sha256), sorted(want_rels)))
        for rel, want in sorted(exe_sha256.items()):
            full = os.path.join(root, *rel.split("/"))
            try:
                have = pinlib.sha256_file(full)
            except OSError:
                have = None
            if have != want:
                sha_diff.append("%s: sha256 %s != pin %s" % (rel, (have or "missing")[:16], str(want)[:16]))
        add("exe_sha256", not sha_diff,
            _fmt_list(sha_diff) if sha_diff else "%d exe sha256 match" % len(exe_sha256))

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

    if verify_hashes:
        import time
        t0 = time.monotonic()
        hashed = {rel: p["sha256"] for rel, p in pinned_paks.items() if isinstance(p, dict) and "sha256" in p}
        hash_diff = [] if hashed else ["the pin records no pak sha256 (only a cold pin does)"]
        nbytes = 0
        for rel, want in sorted(hashed.items()):
            full = os.path.join(root, *rel.split("/"))
            try:
                have = pinlib.sha256_file(full)
                nbytes += os.path.getsize(full)
            except OSError:
                have = None
            if have != want:
                hash_diff.append("%s: sha256 %s != pin %s" % (rel, (have or "missing")[:16], str(want)[:16]))
        add("pak_sha256", not hash_diff,
            _fmt_list(hash_diff) if hash_diff else "%d pak sha256 match (%.2f GB hashed in %.1f s)"
            % (len(hashed), nbytes / 1e9, time.monotonic() - t0))

    lp = launch_path or {}
    have = (_pinned_path(pin_app, "acf"), _pinned_path(pin_app, "install"))
    want = (lp.get("acf"), lp.get("install"))
    if launch_path_mismatch:
        add("launch_path", False, launch_path_mismatch)
    elif have == want and None not in want:
        add("launch_path", True, "pinned install is the one the launchers start: %s" % (pinlib.wsl_to_win(want[1])
                                                                                        or want[1]))
    else:
        add("launch_path", False,
            "pin was made from install %s (appmanifest %s), but the launchers start %s (appmanifest %s); only a pin "
            "made from the launch install can be activated or launched" % (have[1], have[0], want[1], want[0]))
    return checks, observed


def run_checks(pin: dict, acf_paths: dict, roots: dict, mtime_resolution_s: int = 0, launch_paths=None,
               exe_sha256=None, launch_path_id=None, verify_hashes: bool = False):
    """All checks for both apps.

    launch_path_id is the launcher's launch-path id (default: the pin's own); launch_paths defaults to
    pinlib.launch_paths_for(it), read at call time (stable is pinlib.LAUNCH_PATHS). A pin made for another id fails
    <app>.launch_path. exe_sha256 is {app: {rel: sha256}} (pin_identity) or None.
    """
    pin_lp = pinlib.pin_launch_path_id(pin)
    want_lp = pin_lp if launch_path_id is None else launch_path_id
    mismatch = None
    if want_lp != pin_lp:
        mismatch = ("pin %s was made for launch path %r, but this launcher starts launch path %r; use that launch "
                    "path's launcher" % (pin.get("pin_id"), pin_lp, want_lp))
    if launch_paths is None:
        launch_paths = pinlib.launch_paths_for(want_lp)
    checks, observed = [], {}
    for app in pinlib.APPS:
        c, o = check_app(app, pin[app], pin.get("steam", {}), acf_paths[app], roots[app], mtime_resolution_s,
                         launch_paths.get(app), None if exe_sha256 is None else exe_sha256.get(app, {}),
                         mismatch, verify_hashes)
        checks.extend(c)
        observed[app] = o
    return checks, observed


def pinned_paths(pin: dict):
    """(acf_paths, roots): the live appmanifests and install dirs the pin was made from."""
    return ({app: pin[app]["acf"]["wsl"] for app in pinlib.APPS},
            {app: pin[app]["install"]["wsl"] for app in pinlib.APPS})


def print_report(checks, allow_fail=()) -> list:
    """Print the PASS/WAIVED/FAIL lines of --check; return the failed checks that were not waived."""
    for c in checks:
        tag = "PASS" if c["ok"] else ("WAIVED" if c["name"] in allow_fail else "FAIL")
        print("%-6s %-27s %s" % (tag, c["name"], c["detail"]))
    return [c for c in checks if not c["ok"] and c["name"] not in allow_fail]


def load_pin_bytes(raw: bytes, path: str) -> dict:
    try:
        pin = json.loads(raw.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        raise pinlib.PinError("cannot read pin %s: %s" % (path, exc))
    if not isinstance(pin, dict) or pin.get("schema") not in pinlib.PIN_SCHEMAS:
        raise pinlib.PinError("%s is not a %s pin file" % (path, " or ".join(pinlib.PIN_SCHEMAS)))
    for app in pinlib.APPS:
        a = pin.get(app)
        if not isinstance(a, dict) or not all(k in a for k in ("appid", "buildid", "depots", "paks", "acf", "install")):
            raise pinlib.PinError("pin %s: section %r is incomplete" % (path, app))
    if pin["schema"] == pinlib.PIN_SCHEMA_V2:
        for key in ("channel", "launch_path_id", "build_key", "build_key_inputs"):
            if key not in pin:
                raise pinlib.PinError("dd-pin/2 pin %s has no %s" % (path, key))
        pinlib.pin_channel(pin)
        for app in pinlib.APPS:
            if not isinstance(pin[app].get("exe_sha256"), dict):
                raise pinlib.PinError("dd-pin/2 pin %s: %s has no exe_sha256" % (path, app))
    return pin


# --------------------------------------------------------------------------- build identity (DD-14a)


def _check_inputs_match_pin(pin: dict, inputs: dict, exe_sha256: dict, channel: str, what: str) -> str:
    """Recompute build_key/1 from inputs; PinError unless they agree with the pin's appids, buildids and exe hashes."""
    key = pinlib.build_key_from_inputs(inputs)
    for app in pinlib.APPS:
        a = inputs[app]
        if (a["appid"], a["buildid"]) != (pin[app]["appid"], pin[app]["buildid"]):
            raise pinlib.PinError("%s: build_key inputs name %s %s/%s, the pin %s/%s"
                                  % (what, app, a["appid"], a["buildid"], pin[app]["appid"], pin[app]["buildid"]))
        if a.get("exe") != pinlib.BUILD_KEY_EXES[channel][app]:
            raise pinlib.PinError("%s: build_key exe for %s is %r, not %r"
                                  % (what, app, a.get("exe"), pinlib.BUILD_KEY_EXES[channel][app]))
        if (exe_sha256.get(app) or {}).get(a["exe"]) != a["exe_sha256"]:
            raise pinlib.PinError("%s: exe_sha256 of %s/%s disagrees with the build_key inputs" % (what, app, a["exe"]))
        if set(exe_sha256.get(app) or {}) != set(pin[app].get("binaries", {})):
            raise pinlib.PinError("%s: exe_sha256 of %s names %s, the pin's exes are %s"
                                  % (what, app, sorted(exe_sha256.get(app) or {}), sorted(pin[app].get("binaries", {}))))
    return key


def load_sidecar(src: dict, kind: str, dir_fd=None):
    """The validated sidecar pin_<id>.<kind>.json next to a loaded pin, or None when there is none.

    A sidecar is bound to the exact pin bytes (pin_sha256) and pin id. A sidecar that exists but does not validate is
    a config error (PinError): it is never ignored, because the harvester reads it too.
    """
    path = pinlib.sidecar_path(src["pin_dir"], src["pin_id"], kind)
    name = os.path.basename(path)
    try:
        pinlib._lstat(name if dir_fd is not None else path, dir_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise pinlib.PinError("cannot stat %s: %s" % (path, exc))
    raw = pinlib.read_regular_file(name if dir_fd is not None else path, "%s sidecar" % kind, pinlib.PIN_MAX_BYTES,
                                   dir_fd=dir_fd, shown=path)
    try:
        side = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise pinlib.PinError("cannot read %s: %s" % (path, exc))
    want_schema = pinlib.BUILD_KEY_SIDECAR_SCHEMA if kind == "build_key" else pinlib.ENGINE_SIDECAR_SCHEMA
    if not isinstance(side, dict) or side.get("schema") != want_schema:
        raise pinlib.PinError("%s is not a %s file" % (path, want_schema))
    if side.get("pin_id") != src["pin_id"] or side.get("pin_sha256") != src["sha256"]:
        raise pinlib.PinError("%s belongs to pin %r sha256 %s, not to %s (sha256 %s)"
                              % (path, side.get("pin_id"), str(side.get("pin_sha256"))[:16], src["pin_id"],
                                 src["sha256"][:16]))
    side["_path"] = path
    side["_sha256"] = hashlib.sha256(raw).hexdigest()
    return side


def pin_identity(src: dict, dir_fd=None) -> dict:
    """Build identity of a loaded pin: channel, release channel, launch-path id, build_key and its source, the exe
    hashes, physics_hz_expected and engine_version_expected.

    dd-pin/2 carries build_key and exe_sha256 itself; a dd-pin/1 pin gets them from its build_key sidecar
    (pin_sidecar.py build-key), or has none (build_key None: the launch record stays dd-launch/1). PinError when a source
    disagrees with the pin or with another source.
    """
    pin = src["pin"]
    channel = pinlib.pin_channel(pin)
    ident = {"schema": pin["schema"], "channel": channel, "release_channel": pinlib.pin_release_channel(pin),
             "launch_path_id": pinlib.pin_launch_path_id(pin), "build_key": None, "build_key_source": None,
             "exe_sha256": None, "physics_hz_expected": None, "engine_version_expected": None, "engine_source": None}
    side = load_sidecar(src, "build_key", dir_fd)
    if pin["schema"] == pinlib.PIN_SCHEMA_V2:
        exe = {app: pin[app]["exe_sha256"] for app in pinlib.APPS}
        key = _check_inputs_match_pin(pin, pin["build_key_inputs"], exe, channel, "pin %s" % src["pin_id"])
        if key != pin["build_key"]:
            raise pinlib.PinError("pin %s: build_key %s does not match its inputs (%s)"
                                  % (src["pin_id"], str(pin["build_key"])[:16], key[:16]))
        ident.update(build_key=key, build_key_source="pin", exe_sha256=exe,
                     physics_hz_expected=pin.get("physics_hz_expected"))
        if side is not None and side.get("build_key") != key:
            raise pinlib.PinError("%s disagrees with the build_key in pin %s" % (side["_path"], src["pin_id"]))
    elif side is not None:
        if side.get("channel", "stable") != channel:
            raise pinlib.PinError("%s is for channel %r, the pin is %r" % (side["_path"], side.get("channel"), channel))
        exe = side.get("exe_sha256")
        if not isinstance(exe, dict) or not isinstance(side.get("build_key_inputs"), dict):
            raise pinlib.PinError("%s has no exe_sha256 or build_key_inputs" % side["_path"])
        key = _check_inputs_match_pin(pin, side["build_key_inputs"], exe, channel, side["_path"])
        if key != side.get("build_key"):
            raise pinlib.PinError("%s: build_key %s does not match its inputs (%s)"
                                  % (side["_path"], str(side.get("build_key"))[:16], key[:16]))
        ev = side.get("evidence") if isinstance(side.get("evidence"), dict) else {}
        if ev.get("source") == "sums" and ev.get("sums") and os.path.exists(ev["sums"]):
            # The SHA256SUMS the exe hashes came from must not have changed since (Codex round 1, item 19).
            if pinlib.sha256_file(ev["sums"]) != ev.get("sums_sha256"):
                raise pinlib.PinError("%s: the SHA256SUMS it was made from (%s) changed since" % (side["_path"],
                                                                                                  ev["sums"]))
        ident.update(build_key=key, build_key_source="sidecar", exe_sha256=exe,
                     physics_hz_expected=side.get("physics_hz_expected"))
    engine = load_sidecar(src, "engine", dir_fd)
    if engine is not None:
        ver = engine.get("engine_version_expected")
        if not isinstance(ver, str) or not ver or len(ver) > 64 or not all(32 < ord(c) < 127 for c in ver):
            raise pinlib.PinError("%s: engine_version_expected %r is not a version string" % (engine["_path"], ver))
        if engine.get("build_key") and ident["build_key"] and engine["build_key"] != ident["build_key"]:
            raise pinlib.PinError("%s names build_key %s, the pin %s" % (engine["_path"], engine["build_key"][:16],
                                                                         ident["build_key"][:16]))
        ident.update(engine_version_expected=ver, engine_source=engine["_path"])
    return ident


def cold_anchor_problem(pin: dict):
    """For a cold pin: why the reference SHA256SUMS it was verified against has changed since, or None (Codex round 1,
    item 19: the sha256 of that file is recorded in the pin as cold_of.sums_sha256)."""
    cold_of = pin.get("cold_of") if isinstance(pin.get("cold_of"), dict) else {}
    path, want = cold_of.get("sums"), cold_of.get("sums_sha256")
    if not path or not want:
        return "the cold pin records no SHA256SUMS anchor (cold_of.sums, cold_of.sums_sha256)"
    try:
        have = pinlib.sha256_file(path)
    except OSError as exc:
        return "cannot read the reference SHA256SUMS %s: %s" % (path, exc)
    if have != want:
        return "the reference SHA256SUMS %s changed since the cold pin was made (sha256 %s, pin %s)" % (
            path, have[:16], want[:16])
    return None


def full_checks(src: dict, ident: dict, acf_paths=None, roots=None, mtime_resolution_s: int = 0,
                launch_path_id=None, verify_hashes: bool = False):
    """run_checks on the pinned paths (or overrides) with the pin's exe hashes, for launch, check and activation.

    A cold pin always hashes every pak against its recorded sha256 (Codex round 1, MIGRATION_1_9 7.1): a pak changed
    with its size and mtime preserved is refused at launch too. No cache: a cache keyed by size, mtime and path could
    not see that change. About 6.5 min for the 28 GB of 1.8.0.13 at the 73 MB/s measured on E: (2026-11-07). It also
    checks the reference SHA256SUMS anchor."""
    pin_acf, pin_roots = pinned_paths(src["pin"])
    acf_paths = {app: (acf_paths or {}).get(app) or pin_acf[app] for app in pinlib.APPS}
    roots = {app: (roots or {}).get(app) or pin_roots[app] for app in pinlib.APPS}
    cold = ident["channel"] == "cold"
    checks, observed = run_checks(src["pin"], acf_paths, roots, mtime_resolution_s, exe_sha256=ident["exe_sha256"],
                                  launch_path_id=launch_path_id, verify_hashes=verify_hashes or cold)
    if cold:
        why = cold_anchor_problem(src["pin"])
        if why:
            for c in checks:
                if c["name"].endswith(".pak_sha256"):
                    c["ok"] = False
                    c["detail"] = why + "; " + c["detail"]
    return checks, observed


def load_pin_by_id(pin_dir: str, pin_id: str, dir_fd=None) -> dict:
    """Load <pin dir>/pin_<id>.json: a valid id, a regular file (not a symlink), and a pin for that id.

    With dir_fd (the opened pin dir, as pin_activate uses it), the file is read relative to that fd.
    Returns the pin source used by resolve_pin; resolved_by and pointer are left for the caller.
    """
    pin_dir = os.path.abspath(pinlib.win_to_wsl(pin_dir))
    path = pinlib.pin_file_for(pin_dir, pin_id)
    raw = pinlib.read_regular_file(os.path.basename(path) if dir_fd is not None else path,
                                   "pin file for %s" % pin_id, pinlib.PIN_MAX_BYTES, dir_fd=dir_fd, shown=path)
    pin = load_pin_bytes(raw, path)
    if pin.get("pin_id") != pin_id:
        raise pinlib.PinError("%s holds pin_id %r, not %r (a pin file must be named pin_<its pin_id>.json)"
                              % (path, pin.get("pin_id"), pin_id))
    return {"path": path, "pin": pin, "pin_id": pin_id, "sha256": hashlib.sha256(raw).hexdigest(),
            "pin_dir": pin_dir, "resolved_by": None, "pointer": None}


def resolve_pin(pin_path=None, pin_active: bool = False, pin_dir=None) -> dict:
    """Find the pin to check: exactly one of an explicit pin_path or the active pointer.

    Returns {path, pin, pin_id, sha256, pin_dir, resolved_by, pointer}. resolved_by is "active-pointer" (pointer
    is the ACTIVE file) or "explicit-path" (pointer is None). The sha256 is of the exact bytes that were parsed.
    """
    if bool(pin_path) == bool(pin_active):
        raise pinlib.PinError("give exactly one of --pin PATH and --pin-active")
    if pin_active:
        pin_dir = os.path.abspath(pinlib.win_to_wsl(pin_dir or pinlib.DEFAULT_PIN_DIR))
        pin_id = pinlib.read_active(pin_dir)
        src = load_pin_by_id(pin_dir, pin_id)
        src.update(resolved_by="active-pointer", pointer=os.path.join(pin_dir, pinlib.ACTIVE_NAME))
        return src
    if pin_dir:
        raise pinlib.PinError("--pin-dir is only used with --pin-active")
    path = os.path.abspath(pinlib.win_to_wsl(pin_path))
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise pinlib.PinError("cannot read pin %s: %s" % (path, exc))
    pin = load_pin_bytes(raw, path)
    return {"path": path, "pin": pin, "pin_id": pin.get("pin_id"), "sha256": hashlib.sha256(raw).hexdigest(),
            "pin_dir": os.path.dirname(path), "resolved_by": "explicit-path", "pointer": None}


def describe_source(src: dict) -> str:
    if src["resolved_by"] == "active-pointer":
        return "pin %s via active pointer %s -> %s" % (src["pin_id"], src["pointer"], src["path"])
    return "pin %s via explicit --pin %s" % (src["pin_id"], src["path"])


def new_launch_id(exe: str) -> str:
    return "%s-%s-%s" % (pinlib.utc_compact(pinlib.utc_now()), exe, os.urandom(2).hex())


def record_is_v2(ident: dict, use_class_given: bool) -> bool:
    """dd-launch/2 whenever there is anything to add to a dd-launch/1 record: a known build_key or
    engine_version_expected, a dd-pin/2 pin, a non-stable channel or an explicit --use-class. A dd-pin/1 pin without
    sidecars keeps DD-0's dd-launch/1 record."""
    return bool(ident["build_key"]) or bool(ident["engine_version_expected"]) \
        or ident["schema"] == pinlib.PIN_SCHEMA_V2 or ident["channel"] != "stable" or use_class_given


def make_record(status, launch_id, args, cli_argv, src, checks, observed, ident=None, use_class=None,
                v2=False) -> dict:
    rec = {
        "schema": pinlib.LAUNCH_SCHEMA_V2 if v2 else pinlib.LAUNCH_SCHEMA,
        "launch_id": launch_id,
        "status": status,
        "utc": pinlib.utc_iso(pinlib.utc_now()),
        "exe": args.exe,
        "instance": args.instance,
        # resolved_by: "active-pointer" (--pin-active; pointer = the ACTIVE file read) or "explicit-path" (--pin).
        "pin": {"path": src["path"], "pin_id": src["pin"].get("pin_id"), "sha256": src["sha256"],
                "resolved_by": src["resolved_by"], "pointer": src["pointer"]},
        "observed": observed,
        "checks": checks,
        "host": {"hostname": socket.gethostname(), "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
                 "python": sys.version.split()[0]},
        "argv": cli_argv,
    }
    if v2 and ident is not None:
        # Top-level keys, read by DC_LaunchGuard before it enters any child object. channel is the release channel
        # (dc-2.0 session.json pin.channel: stable or experimental); pin_channel is the pin's own (cold included).
        # build_key is left out, never null, while it is unknown (JsonLoadContext and a JSON null do not mix).
        if ident["build_key"]:
            rec["build_key"] = ident["build_key"]
            rec["build_key_source"] = ident["build_key_source"]
        rec["channel"] = ident["release_channel"]
        rec["pin_channel"] = ident["channel"]
        rec["launch_path_id"] = ident["launch_path_id"]
        rec["use_class"] = use_class
        if ident["physics_hz_expected"] is not None:
            rec["physics_hz_expected"] = ident["physics_hz_expected"]
        if ident["engine_version_expected"]:
            rec["engine_version_expected"] = ident["engine_version_expected"]
    return rec


def resolve_use_class(requested, ident: dict) -> str:
    """The use_class a launch records. An experimental pin is always compat-preview; production is refused until a
    G19 admission record can qualify it (DD-14b). PinError on a refusal."""
    if requested is not None and requested not in pinlib.USE_CLASSES:
        raise pinlib.PinError("unknown --use-class %r (known: %s)" % (requested, ", ".join(pinlib.USE_CLASSES)))
    if ident["channel"] == "experimental":
        if requested not in (None, pinlib.EXPERIMENTAL_USE_CLASS):
            raise pinlib.PinError("an experimental pin launches only with use_class %s, not %s"
                                  % (pinlib.EXPERIMENTAL_USE_CLASS, requested))
        return pinlib.EXPERIMENTAL_USE_CLASS
    if requested == "production":
        raise pinlib.PinError("use_class production needs a qualified G19 admission record for this build_key "
                              "(MIGRATION_1_9 6.2, DD-14b); the preflight does not accept it yet")
    return requested or pinlib.DEFAULT_USE_CLASS


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--pin", default=None, help="explicit pin JSON written by make_pin.py")
    which.add_argument("--pin-active", action="store_true",
                       help="check the pin named by <pin dir>/ACTIVE (set it with pin_activate.py); the launchers "
                            "use this")
    ap.add_argument("--pin-dir", default=None,
                    help="with --pin-active only: the directory holding ACTIVE and pin_<id>.json (default %s)"
                         % pinlib.DEFAULT_PIN_DIR)
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
    ap.add_argument("--launch-path", default=pinlib.STABLE_LAUNCH_PATH, metavar="ID",
                    help="launch-path id of the launcher (default %(default)s; wb_C18.bat passes cold-1.8.0.13). "
                         "The pin must have been made for it (<app>.launch_path)")
    ap.add_argument("--use-class", default=None, metavar="CLASS",
                    help="use_class recorded in the launch record (%s; default %s; an experimental pin is always "
                         "%s; production is refused until DD-14b's G19 admission record)"
                         % (", ".join(pinlib.USE_CLASSES), pinlib.DEFAULT_USE_CLASS, pinlib.EXPERIMENTAL_USE_CLASS))
    ap.add_argument("--verify-hashes", action="store_true",
                    help="with --check only: also hash every pak the pin records a sha256 for (a cold pin always "
                         "does, at every launch too; reads the whole install)")
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(cli_argv)
    if args.verify_hashes and not args.check:
        print("preflight: --verify-hashes is only accepted with --check", file=sys.stderr)
        return 2

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
    unknown = [n for n in args.allow_fail if n not in ALL_CHECK_NAMES]
    if unknown:
        print("preflight: unknown check name(s) %s; known: %s" % (unknown, ", ".join(ALL_CHECK_NAMES)),
              file=sys.stderr)
        return 2

    if args.pin_dir and not args.pin_active:
        print("preflight: --pin-dir is only accepted with --pin-active", file=sys.stderr)
        return 2

    launch_dir = None
    try:
        src = resolve_pin(args.pin, args.pin_active, args.pin_dir)
        ident = pin_identity(src)
        if args.pin_active and ident["channel"] == "experimental":
            raise pinlib.PinError("ACTIVE names %s, a channel-experimental pin; no production launcher may start "
                                  "Experimental (an experimental pin is used with an explicit --pin only)"
                                  % src["pin_id"])
        pinlib.launch_paths_for(args.launch_path)
        use_class = resolve_use_class(args.use_class, ident)
        if not args.check:
            launch_dir = pinlib.check_write_path(args.launch_dir or os.path.join(src["pin_dir"], "launches"),
                                                 write_roots)
    except pinlib.PinError as exc:
        print("preflight: %s" % exc, file=sys.stderr)
        return 2
    pin = src["pin"]
    v2 = record_is_v2(ident, args.use_class is not None)

    checks, observed = full_checks(src, ident, {"game": args.game_acf, "tools": args.tools_acf},
                                   {"game": args.game_root, "tools": args.tools_root},
                                   max(0, args.mtime_resolution_s), args.launch_path, args.verify_hashes)
    failed = [c for c in checks if not c["ok"] and c["name"] not in args.allow_fail]

    if args.check:
        print("%-6s %s (sha256 %s)" % ("PIN", describe_source(src), src["sha256"]))
        print("%-6s %s channel %s (release %s), launch path %s, build_key %s%s" % (
            "BUILD", ident["schema"], ident["channel"], ident["release_channel"], ident["launch_path_id"],
            ident["build_key"] or "unknown (dd-pin/1 without its build_key sidecar; run pin_sidecar.py build-key)",
            " (from %s)" % ident["build_key_source"] if ident["build_key"] else ""))
        if ident["engine_version_expected"]:
            print("%-6s engine_version_expected %s (%s)" % ("ENGINE", ident["engine_version_expected"],
                                                            ident["engine_source"]))
        print_report(checks, args.allow_fail)
        if failed:
            print("RESULT: REFUSED (pin %s): %d check(s) failed: %s"
                  % (pin.get("pin_id"), len(failed), ", ".join(c["name"] for c in failed)))
            return 1
        print("RESULT: OK (pin %s)" % pin.get("pin_id"))
        return 0

    launch_id = new_launch_id(args.exe)
    if failed:
        print("PREFLIGHT REFUSED LAUNCH of %s (%s):" % (args.exe, describe_source(src)), file=sys.stderr)
        for c in failed:
            print("  FAIL %s: %s" % (c["name"], c["detail"]), file=sys.stderr)
        try:
            os.makedirs(launch_dir, exist_ok=True)
            rec = pinlib.write_json_new(os.path.join(launch_dir, "refused_%s.json" % launch_id),
                                        make_record("refused", launch_id, args, cli_argv, src, checks, observed,
                                                    ident, use_class, v2),
                                        write_roots)
            print("  record: %s" % rec, file=sys.stderr)
        except (pinlib.PinError, OSError) as exc:
            print("  (refusal record not written: %s)" % exc, file=sys.stderr)
        return 1

    try:
        os.makedirs(launch_dir, exist_ok=True)
        rec = pinlib.write_json_new(os.path.join(launch_dir, "launch_%s.json" % launch_id),
                                    make_record("ok", launch_id, args, cli_argv, src, checks, observed, ident, use_class,
                                                v2),
                                    write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("PREFLIGHT REFUSED LAUNCH of %s: launch record not written: %s" % (args.exe, exc), file=sys.stderr)
        return 3
    print("PREFLIGHT OK: %s matches %s; record %s" % (args.exe, describe_source(src), rec), file=sys.stderr)
    print(launch_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
