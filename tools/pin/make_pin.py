#!/usr/bin/env python3
"""Write the DriveDiT build pin for Arma Reforger + Arma Reforger Tools (ticket DD-0 / program TKT-0).

Reads, READ-ONLY, both Steam appmanifests and both install directories, and records:
  - game: appid, buildid, TargetBuildID, StateFlags, AutoUpdateBehavior, depots + manifest ids
  - tools: the same for the Tools app
  - every *.pak in each install (relative path, size, mtime), the launched exes, steam_appid.txt
The JSON goes to --out, which must be under /mnt/e (C: is always refused). The output directory must
already exist; nothing else is written.

    python3 -I -B make_pin.py --pin-id 1.8.0.13 --out /mnt/e/DriveDiT/pin/pin_1.8.0.13.json \
        --expect-game-buildid 24903726 --expect-tools-buildid 24870687

Writing a pin does not make the launchers use it: switch with `pin_activate.py <pin id>` (see README).

Rollback (README "Rollback"): restored installs never get the sub-second mtimes of the original pin back, so they
are re-pinned under a new id, never by overwriting the original pin:

    python3 -I -B make_pin.py --pin-id 1.8.0.13-r1 --restore-of 1.8.0.13 \
        --out /mnt/e/DriveDiT/pin/pin_1.8.0.13-r1.json --expect-game-buildid 24903726 --expect-tools-buildid 24870687

--restore-of REF reads <out dir>/pin_<REF>.json and refuses (exit 1) unless the new snapshot equals it in
everything that identifies the build: appid, buildid, depot manifests, appmanifest and install paths,
steam_appid.txt, and the set and size of every pak and exe. Only file mtimes may differ. The new pin records
restore_of {pin_id, sha256, path} so a validator can treat both ids as one build.

Exit codes: 0 written; 1 install state not pinnable (buildid mismatch, update pending, no paks, not a restore of
--restore-of); 2 usage, path or read error (nothing written).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def _load(name):
    # By explicit path: `python3 -I` does not put this directory on sys.path.
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


class NotPinnable(Exception):
    pass


def _paths(p: str) -> dict:
    return {"wsl": p, "windows": pinlib.wsl_to_win(p)}


def collect_app(app: str, acf_path: str, expect_buildid) -> dict:
    """Snapshot one app. Raises NotPinnable if the install is not in a settled, expected state."""
    acf = pinlib.load_acf(acf_path)
    want_appid = pinlib.APPIDS[app]
    if acf["appid"] != want_appid:
        raise NotPinnable("%s: %s has appid %s, expected %s" % (app, acf_path, acf["appid"], want_appid))
    if acf["buildid"] is None:
        raise NotPinnable("%s: %s has no numeric buildid" % (app, acf_path))
    if expect_buildid is not None and acf["buildid"] != expect_buildid:
        raise NotPinnable("%s: buildid is %s, but --expect-%s-buildid says %s"
                          % (app, acf["buildid"], app, expect_buildid))
    if acf["target_buildid"] != acf["buildid"]:
        raise NotPinnable("%s: TargetBuildID %s != buildid %s (an update is pending; do not pin a half-updated install)"
                          % (app, acf["target_buildid"], acf["buildid"]))
    if acf["state_flags"] != pinlib.STATE_FLAGS_SETTLED:
        raise NotPinnable("%s: StateFlags is %r, expected %r (fully installed, nothing queued)"
                          % (app, acf["state_flags"], pinlib.STATE_FLAGS_SETTLED))
    if not acf["depots"]:
        raise NotPinnable("%s: %s lists no InstalledDepots" % (app, acf_path))

    root = pinlib.install_dir_for_acf(acf_path, acf["installdir"])
    if not os.path.isdir(root):
        raise pinlib.PinError("%s: install dir not found: %s" % (app, root))
    paks = pinlib.scan_paks(root)
    if not paks:
        raise NotPinnable("%s: no *.pak files under %s" % (app, root))

    binaries = {}
    for rel in pinlib.BINARIES[app]:
        full = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(full):
            raise NotPinnable("%s: missing executable %s" % (app, full))
        binaries[rel] = pinlib.stat_entry(full)

    appid_txt = {}
    for rel in pinlib.STEAM_APPID_FILES[app]:
        full = os.path.join(root, *rel.split("/"))
        appid_txt[rel] = pinlib.read_small_text(full) if os.path.isfile(full) else None

    return {
        "appid": acf["appid"],
        "name": acf["name"],
        "buildid": acf["buildid"],
        "target_buildid": acf["target_buildid"],
        "state_flags": acf["state_flags"],
        "auto_update_behavior_at_pin": acf["auto_update_behavior"],
        "last_updated_utc": (pinlib.iso_utc_from_ns(acf["last_updated"] * 10**9)
                             if acf["last_updated"] else None),
        "depots": acf["depots"],
        "acf": _paths(os.path.abspath(acf_path)),
        "install": _paths(os.path.abspath(root)),
        "steam_appid_txt": appid_txt,
        "binaries": binaries,
        "paks": paks,
    }


def restore_diffs(ref: dict, apps: dict) -> list:
    """What differs between a reference pin and a new snapshot, ignoring file mtimes (the rollback re-pin rule)."""
    diffs = []
    for app in pinlib.APPS:
        r, n = ref.get(app) or {}, apps[app]
        for key in ("appid", "buildid"):
            if r.get(key) != n.get(key):
                diffs.append("%s.%s %s != %s" % (app, key, n.get(key), r.get(key)))
        rd = {d: (v or {}).get("manifest") for d, v in (r.get("depots") or {}).items()}
        nd = {d: v.get("manifest") for d, v in n["depots"].items()}
        if rd != nd:
            diffs.append("%s.depots %s != %s" % (app, nd, rd))
        for key in ("acf", "install"):
            rv = (r.get(key) or {}).get("wsl") if isinstance(r.get(key), dict) else None
            if rv != n[key]["wsl"]:
                diffs.append("%s.%s %s != %s" % (app, key, n[key]["wsl"], rv))
        if (r.get("steam_appid_txt") or {}) != n["steam_appid_txt"]:
            diffs.append("%s.steam_appid_txt %s != %s" % (app, n["steam_appid_txt"], r.get("steam_appid_txt")))
        for kind in ("binaries", "paks"):
            rs = {k: (v or {}).get("size") for k, v in (r.get(kind) or {}).items()}
            ns = {k: v["size"] for k, v in n[kind].items()}
            for k in sorted(set(rs) - set(ns)):
                diffs.append("%s.%s %s missing" % (app, kind, k))
            for k in sorted(set(ns) - set(rs)):
                diffs.append("%s.%s %s not in the reference pin" % (app, kind, k))
            for k in sorted(set(rs) & set(ns)):
                if rs[k] != ns[k]:
                    diffs.append("%s.%s %s size %s != %s" % (app, kind, k, ns[k], rs[k]))
    return diffs


def mtimes_changed(ref: dict, apps: dict) -> int:
    n = 0
    for app in pinlib.APPS:
        for kind in ("binaries", "paks"):
            for k, v in apps[app][kind].items():
                if ((ref[app].get(kind) or {}).get(k) or {}).get("mtime_ns") != v["mtime_ns"]:
                    n += 1
    return n


def build_pin(pin_id: str, out_path: str, game: dict, tools: dict) -> dict:
    base = os.path.join(os.path.dirname(out_path), pin_id)
    return {
        "schema": pinlib.PIN_SCHEMA,
        "pin_id": pin_id,
        "ticket": "DD-0",
        "created_utc": pinlib.utc_iso(pinlib.utc_now()),
        "created_by": "tools/pin/make_pin.py",
        "game": game,
        "tools": tools,
        "steam": {
            "auto_update_behavior_allowed": list(pinlib.AUTO_UPDATE_ALLOWED),
            "auto_update_behavior_meaning": pinlib.AUTO_UPDATE_MEANING,
            "auto_update_behavior_mapping_source": "community-documented; Valve docs UNVERIFIED",
            "require_target_buildid_eq_buildid": True,
            "require_state_flags": pinlib.STATE_FLAGS_SETTLED,
            "compare": "pak/exe size and mtime_ns must equal the pin exactly; no extra or missing paks",
        },
        "cold_copy": {
            "game": pinlib.wsl_to_win(os.path.join(base, "game")),
            "tools": pinlib.wsl_to_win(os.path.join(base, "tools")),
        },
        "sha256sums": pinlib.wsl_to_win(os.path.join(base, "SHA256SUMS")),
    }


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="pin JSON to write; must be under /mnt/e; parent must exist")
    ap.add_argument("--pin-id", required=True,
                    help="id of the pinned version, e.g. 1.8.0.13, 1.9.0.12 or 1.9.0.9-exp (pinlib.PIN_ID_RE); "
                         "name the output pin_<id>.json so pin_activate.py and preflight --pin-active find it")
    ap.add_argument("--game-acf", default=pinlib.DEFAULT_ACF["game"])
    ap.add_argument("--tools-acf", default=pinlib.DEFAULT_ACF["tools"])
    ap.add_argument("--expect-game-buildid", type=int, default=None)
    ap.add_argument("--expect-tools-buildid", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="replace an existing pin file (a pin should never change)")
    ap.add_argument("--restore-of", default=None, metavar="PIN_ID",
                    help="rollback re-pin: the installs are a restore of the build pinned as <out dir>/pin_PIN_ID.json; "
                         "refuse unless they equal it in everything but file mtimes (see README 'Rollback')")
    args = ap.parse_args(argv)

    # Path guard first: nothing is read or written before the output path is known to be on E:.
    try:
        pinlib.validate_pin_id(args.pin_id)
        out = pinlib.check_write_path(args.out, write_roots)
    except pinlib.PinError as exc:
        print("make_pin: %s" % exc, file=sys.stderr)
        return 2
    if os.path.basename(out) != "pin_%s.json" % args.pin_id:
        print("make_pin: WARNING: %s is not named pin_%s.json, so pin_activate.py and preflight --pin-active "
              "will not find it" % (out, args.pin_id), file=sys.stderr)
    if os.path.exists(out) and not args.force:
        print("make_pin: %s already exists; a pin is immutable (use --force only to re-pin on purpose)" % out,
              file=sys.stderr)
        return 2
    if not os.path.isdir(os.path.dirname(out)):
        print("make_pin: output directory does not exist (create it first): %s" % os.path.dirname(out),
              file=sys.stderr)
        return 2
    ref = None
    if args.restore_of is not None:
        try:
            ref_id = pinlib.validate_pin_id(args.restore_of)
            if ref_id == args.pin_id:
                raise pinlib.PinError("--restore-of %s: a restored install gets a new pin id (for example %s-r1); "
                                      "pin_%s.json is never overwritten" % (ref_id, ref_id, ref_id))
            ref = _load("preflight").load_pin_by_id(os.path.dirname(out), ref_id)
        except pinlib.PinError as exc:
            print("make_pin: %s" % exc, file=sys.stderr)
            return 2

    try:
        game = collect_app("game", args.game_acf, args.expect_game_buildid)
        tools = collect_app("tools", args.tools_acf, args.expect_tools_buildid)
    except NotPinnable as exc:
        print("make_pin: not pinnable: %s" % exc, file=sys.stderr)
        return 1
    except pinlib.PinError as exc:
        print("make_pin: %s" % exc, file=sys.stderr)
        return 2

    pin = build_pin(args.pin_id, out, game, tools)
    if ref is not None:
        diffs = restore_diffs(ref["pin"], {"game": game, "tools": tools})
        if diffs:
            print("make_pin: not pinnable: the installs are not a restore of pin %s (only file mtimes may differ): %s"
                  % (ref["pin_id"], "; ".join(diffs[:12]) + ("; ... %d more" % (len(diffs) - 12) if len(diffs) > 12
                                                              else "")), file=sys.stderr)
            return 1
        pin["restore_of"] = {
            "pin_id": ref["pin_id"], "sha256": ref["sha256"], "path": ref["path"],
            "rule": "equal to the reference pin in appid, buildid, depot manifests, appmanifest and install paths, "
                    "steam_appid.txt and the set and size of every pak and exe; only file mtimes differ",
            "mtimes_changed": mtimes_changed(ref["pin"], {"game": game, "tools": tools}),
        }
    try:
        written = pinlib.write_json_new(out, pin, write_roots, overwrite=args.force)
    except (pinlib.PinError, OSError) as exc:
        print("make_pin: %s" % exc, file=sys.stderr)
        return 2

    for app in pinlib.APPS:
        a = pin[app]
        print("%-5s appid %d buildid %d depots %s paks %d (%.2f GB) AutoUpdateBehavior %r"
              % (app, a["appid"], a["buildid"],
                 ",".join("%s:%s" % (d, v["manifest"]) for d, v in a["depots"].items()),
                 len(a["paks"]), sum(p["size"] for p in a["paks"].values()) / 1e9,
                 a["auto_update_behavior_at_pin"]))
        if a["auto_update_behavior_at_pin"] not in pinlib.AUTO_UPDATE_ALLOWED:
            print("WARNING: %s AutoUpdateBehavior is %r (%s); preflight will refuse launches until it is set to "
                  "\"Only update this game when I launch it\"."
                  % (app, a["auto_update_behavior_at_pin"],
                     pinlib.AUTO_UPDATE_MEANING.get(a["auto_update_behavior_at_pin"], "unknown")), file=sys.stderr)
    if ref is not None:
        print("restore of pin %s: same build; %d file mtime(s) differ from it"
              % (ref["pin_id"], pin["restore_of"]["mtimes_changed"]))
    print("wrote %s (sha256 %s)" % (written, pinlib.sha256_file(written)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
