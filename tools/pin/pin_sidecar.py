#!/usr/bin/env python3
"""Write the write-once sidecars of a pin (ticket DD-14a; MIGRATION_1_9 2.2, 5.5, 8.2 component 1).

A pin file is immutable. What DD-14a adds to an existing pin goes into a sidecar next to it, bound to the exact pin
bytes (pin_sha256), written once (O_EXCL) and never replaced:

  build-key  <pin dir>/pin_<id>.build_key.json, for a dd-pin/1 pin: its build_key (build_key/1: game and Tools appid
             and buildid, sha256 of the client and the Workbench exe) and the sha256 of every pinned exe. The
             harvester reads this file (tools/harvester/harvest.py resolve_build_key), the preflight copies the
             build_key into every dd-launch/2 record and hashes the exes at every launch (<app>.exe_sha256).
               --from-live  hash the live exes, after the full preflight check of the pin passes against the live
                            installs (so the exes hashed are the pinned ones), and stat them again afterwards;
               --from-sums  take the exe hashes from the pin's cold copy SHA256SUMS (<pin dir>/<id>/SHA256SUMS), and
                            hash the copy's exes to confirm them (for a pin whose installs have moved on, 1.8.0.13).
  engine     <pin dir>/pin_<id>.engine.json: engine_version_expected, the GetBuildVersion() string of a probe run
             header on that pin (SURFACE 5 row 2; the first E-11). The header must name the pin id and the pin's
             sha256. The preflight copies it into the launch record; DC_Session compares it with GetBuildVersion().

    python3 -I -B pin_sidecar.py build-key 1.9.0.37 --from-live --dry-run
    python3 -I -B pin_sidecar.py build-key 1.9.0.37 --from-live --apply
    python3 -I -B pin_sidecar.py build-key 1.8.0.13 --from-sums --dry-run
    python3 -I -B pin_sidecar.py engine 1.9.0.37 --run-header <DC_Probes/e01/<launch id>/play_01/run_header.txt> --dry-run

Exactly one of --dry-run (print the sidecar, write nothing) and --apply (write it) is required. Writes only under
/mnt/e (pinlib.check_write_path). Reads the installs, the copy and the run header read-only.

Exit codes: 0 written (or dry run); 1 refused (the pin does not match its installs or copy, the evidence does not
name the pin); 2 usage, config or path error, or the sidecar already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time


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

RUN_HEADER_MAX_BYTES = 1 << 20


class Refused(Exception):
    pass


def _stat_all(pin_app: dict, root: str) -> dict:
    return {rel: pinlib.stat_entry(os.path.join(root, *rel.split("/"))) for rel in sorted(pin_app["binaries"])}


def build_key_from_live(src: dict) -> tuple:
    """(exe_sha256, detail) from the live installs, after the full preflight check passes (no waivers)."""
    checks, _ = preflight.full_checks(src, preflight.pin_identity(src))
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise Refused("the live installs do not match pin %s (%s); hash only what the pin checked. Use --from-sums "
                      "for a pin whose installs have moved on" % (src["pin_id"], "; ".join(
                          "%s: %s" % (c["name"], c["detail"]) for c in failed[:4])))
    _, roots = preflight.pinned_paths(src["pin"])
    exe, total, t0 = {}, 0, time.monotonic()
    for app in pinlib.APPS:
        before = _stat_all(src["pin"][app], roots[app])
        exe[app], n = pinlib.hash_files(roots[app], sorted(src["pin"][app]["binaries"]))
        total += n
        if _stat_all(src["pin"][app], roots[app]) != before:
            raise Refused("%s exes changed while they were hashed" % app)
    return exe, {"source": "live", "installs": {app: roots[app] for app in pinlib.APPS}, "bytes_hashed": total,
                 "seconds": round(time.monotonic() - t0, 2)}


def build_key_from_sums(src: dict, sums_path=None, verify_copy: bool = True) -> tuple:
    """(exe_sha256, detail) from the cold copy's SHA256SUMS; the copy's exes are hashed to confirm them."""
    copy = os.path.join(src["pin_dir"], src["pin_id"])
    sums_path = os.path.abspath(sums_path or os.path.join(copy, "SHA256SUMS"))
    raw = pinlib.read_regular_file(sums_path, "SHA256SUMS", 64 * 1024 * 1024)
    exe = pinlib.sums_exe_sha256(pinlib.parse_sha256sums(raw))
    total, t0 = 0, time.monotonic()
    for app in pinlib.APPS:
        for rel, digest in sorted(exe[app].items()):
            if rel not in src["pin"][app]["binaries"]:
                raise Refused("SHA256SUMS names %s/%s, which pin %s does not list" % (app, rel, src["pin_id"]))
            if not verify_copy:
                continue
            full = os.path.join(copy, app, *rel.split("/"))
            if not os.path.isfile(full):
                raise Refused("copy exe missing: %s" % full)
            if os.path.getsize(full) != src["pin"][app]["binaries"][rel]["size"]:
                raise Refused("copy exe %s has another size than the pin" % full)
            if pinlib.sha256_file(full) != digest:
                raise Refused("copy exe %s does not match SHA256SUMS" % full)
            total += os.path.getsize(full)
        if set(exe[app]) != set(src["pin"][app]["binaries"]):
            raise Refused("SHA256SUMS lacks some exes of pin %s %s" % (src["pin_id"], app))
    return exe, {"source": "sums", "sums": sums_path, "sums_sha256": hashlib.sha256(raw).hexdigest(),
                 "copy_verified": verify_copy, "bytes_hashed": total, "seconds": round(time.monotonic() - t0, 2)}


def make_build_key_sidecar(src: dict, exe: dict, detail: dict, physics_hz: int) -> dict:
    pin = src["pin"]
    inputs = pinlib.build_key_inputs("stable", {app: pin[app] for app in pinlib.APPS}, exe)
    return {
        "schema": pinlib.BUILD_KEY_SIDECAR_SCHEMA,
        "pin_id": src["pin_id"],
        "pin_sha256": src["sha256"],
        "pin_schema": pin["schema"],
        "channel": "stable",
        "release_channel": "stable",
        "build_key": pinlib.build_key_from_inputs(inputs),
        "build_key_inputs": inputs,
        "exe_sha256": exe,
        "physics_hz_expected": physics_hz,
        "evidence": detail,
        "ticket": "DD-14a",
        "created_utc": pinlib.utc_iso(pinlib.utc_now()),
        "created_by": "tools/pin/pin_sidecar.py build-key",
    }


def parse_run_header(raw: bytes) -> dict:
    out = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key and key not in out:
            out[key.strip()] = value.strip()
    return out


def make_engine_sidecar(src: dict, header_path: str) -> dict:
    raw = pinlib.read_regular_file(header_path, "run header", RUN_HEADER_MAX_BYTES)
    h = parse_run_header(raw)
    if h.get("launch.pin_id") != src["pin_id"] or h.get("launch.pin_sha256") != src["sha256"]:
        raise Refused("run header %s names pin %r sha256 %s, not pin %s sha256 %s"
                      % (header_path, h.get("launch.pin_id"), str(h.get("launch.pin_sha256"))[:16], src["pin_id"],
                         src["sha256"][:16]))
    ver = h.get("engine.build_version", "")
    if not ver or len(ver) > 64 or not all(32 < ord(c) < 127 for c in ver):
        raise Refused("run header %s has no usable engine.build_version (%r)" % (header_path, ver))
    ident = preflight.pin_identity(src)
    if h.get("launch.build_key") and ident["build_key"] and h["launch.build_key"] != ident["build_key"]:
        raise Refused("run header %s names build_key %s, the pin %s" % (header_path, h["launch.build_key"][:16],
                                                                         ident["build_key"][:16]))
    slice_s = h.get("engine.fixed_time_slice_s")
    return {
        "schema": pinlib.ENGINE_SIDECAR_SCHEMA,
        "pin_id": src["pin_id"],
        "pin_sha256": src["sha256"],
        "build_key": ident["build_key"],
        "engine_version_expected": ver,
        "engine_build_time": h.get("engine.build_time"),
        "fixed_time_slice_s_measured": float(slice_s) if slice_s else None,
        "evidence": {"kind": "probe run header", "path": os.path.abspath(header_path),
                     "sha256": hashlib.sha256(raw).hexdigest(), "launch_id": h.get("launch.id"),
                     "launch_mode": h.get("launch.mode"), "engine_exe": h.get("engine.exe")},
        "ticket": "DD-14a",
        "created_utc": pinlib.utc_iso(pinlib.utc_now()),
        "created_by": "tools/pin/pin_sidecar.py engine",
    }


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    for name in ("build-key", "engine"):
        p = sub.add_parser(name)
        p.add_argument("pin_id")
        p.add_argument("--pin-dir", default=pinlib.DEFAULT_PIN_DIR)
        mode = p.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true", help="print the sidecar; write nothing")
        mode.add_argument("--apply", action="store_true", help="write the sidecar (once; never replaced)")
        if name == "build-key":
            src_g = p.add_mutually_exclusive_group(required=True)
            src_g.add_argument("--from-live", action="store_true")
            src_g.add_argument("--from-sums", nargs="?", const="", default=None, metavar="SHA256SUMS")
            p.add_argument("--no-verify-copy", action="store_true",
                           help="--from-sums: do not hash the copy's exes against SHA256SUMS")
            p.add_argument("--physics-hz-expected", type=int, default=pinlib.PHYSICS_HZ_EXPECTED)
        else:
            p.add_argument("--run-header", required=True, help="a probe run_header.txt of a launch on this pin")
    args = ap.parse_args(argv)

    pin_dir = os.path.abspath(pinlib.win_to_wsl(args.pin_dir))
    try:
        pinlib.validate_pin_id(args.pin_id)
        out = pinlib.sidecar_path(pin_dir, args.pin_id, "build_key" if args.kind == "build-key" else "engine")
        if args.apply:
            pinlib.check_write_path(out, write_roots)
        if os.path.lexists(out):
            raise pinlib.PinError("%s already exists; a sidecar is written once and never replaced" % out)
        src = preflight.load_pin_by_id(pin_dir, args.pin_id)
        if args.kind == "build-key":
            if src["pin"]["schema"] != pinlib.PIN_SCHEMA:
                raise pinlib.PinError("pin %s is %s and carries its own build_key; no sidecar is needed"
                                      % (args.pin_id, src["pin"]["schema"]))
            if args.from_live:
                exe, detail = build_key_from_live(src)
            else:
                exe, detail = build_key_from_sums(src, args.from_sums or None, not args.no_verify_copy)
            side = make_build_key_sidecar(src, exe, detail, args.physics_hz_expected)
        else:
            side = make_engine_sidecar(src, args.run_header)
    except Refused as exc:
        print("pin_sidecar: refused: %s" % exc, file=sys.stderr)
        return 1
    except pinlib.PinError as exc:
        print("pin_sidecar: %s" % exc, file=sys.stderr)
        return 2

    print(json.dumps(side, indent=2))
    ev = side["evidence"]
    if args.kind == "build-key":
        print("build_key %s for pin %s (%s; %d bytes hashed in %.2f s)"
              % (side["build_key"], args.pin_id, ev["source"], ev["bytes_hashed"], ev["seconds"]), file=sys.stderr)
    else:
        print("engine_version_expected %r for pin %s (from %s)"
              % (side["engine_version_expected"], args.pin_id, ev["path"]), file=sys.stderr)
    if args.dry_run:
        print("DRY RUN: would write %s; nothing written" % out, file=sys.stderr)
        return 0
    try:
        written = pinlib.write_json_new(out, side, write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("pin_sidecar: %s" % exc, file=sys.stderr)
        return 2
    print("wrote %s (sha256 %s)" % (written, pinlib.sha256_file(written)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
