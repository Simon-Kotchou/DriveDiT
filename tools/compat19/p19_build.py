#!/usr/bin/env python3
"""CC-01 build identity (P19-0) of a pin: the G19-0 static check (ticket DD-14a; MIGRATION_1_9 6.1 row CC-01).

Pass criterion (6.1 CC-01), all four, evaluated read-only:
  1. both appmanifests have StateFlags 4 and AutoUpdateBehavior "1";
  2. the pin is written with its build_key: a dd-pin/2 pin, or a dd-pin/1 pin with its build_key sidecar
     (tools/pin/pin_sidecar.py build-key);
  3. preflight --pin-active --check passes (every check, exe_sha256 included once the exe hashes are known);
  4. the direct-launch freeze check of 5.5 was done for this app pair: a PASS after-record of
     tools/pin/freeze_check.py for this pin file or its build_key, under <pin dir>/freeze/.

    python3 -I -B tools/compat19/p19_build.py [--pin-dir /mnt/e/DriveDiT/pin] [--write]

--write also stores the result as /mnt/e/DriveDiT/compat19/<pin id>/cc01_<UTC>.json (the directory must exist or is
created under /mnt/e/DriveDiT/compat19). Exit codes: 0 PASS, 1 FAIL, 2 config error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_TOOLS = os.path.join(os.path.dirname(HERE), "pin")
COMPAT_ROOT = "/mnt/e/DriveDiT/compat19"


def _load(name):
    path = os.path.join(PIN_TOOLS, name + ".py")
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
freeze_check = _load("freeze_check")


def evaluate(pin_dir=None) -> dict:
    """{pin_id, build_key, items: [{item, ok, detail}], verdict} for the active pin. PinError on a config error."""
    src = preflight.resolve_pin(None, True, pin_dir)
    ident = preflight.pin_identity(src)
    items = []

    def add(item, ok, detail):
        items.append({"item": item, "ok": bool(ok), "detail": detail})

    acf_paths, _ = preflight.pinned_paths(src["pin"])
    flags = []
    for app in pinlib.APPS:
        try:
            acf = pinlib.load_acf(acf_paths[app])
            flags.append((app, acf["state_flags"], acf["auto_update_behavior"]))
        except pinlib.PinError as exc:
            flags.append((app, None, str(exc)))
    add("appmanifests", all(f == pinlib.STATE_FLAGS_SETTLED and a in pinlib.AUTO_UPDATE_ALLOWED for _, f, a in flags),
        "; ".join("%s StateFlags %s AutoUpdateBehavior %s" % f for f in flags))
    add("build_key", bool(ident["build_key"]),
        "%s (from %s)" % (ident["build_key"], ident["build_key_source"]) if ident["build_key"] else
        "the pin has no build_key: %s with no build_key sidecar (tools/pin/pin_sidecar.py build-key)" % ident["schema"])
    checks, _ = preflight.full_checks(src, ident)
    failed = [c["name"] for c in checks if not c["ok"]]
    add("preflight", not failed, "%d/%d checks pass%s" % (len(checks) - len(failed), len(checks),
                                                          "; failed: " + ", ".join(failed) if failed else ""))
    passes = freeze_check.find_pass(src["pin_dir"], src["sha256"], ident["build_key"])
    add("freeze_check", bool(passes), "PASS record %s" % passes[-1] if passes else
        "no PASS after-record in %s (run tools/pin/freeze_check.py before, a user launch, after)"
        % os.path.join(src["pin_dir"], freeze_check.FREEZE_DIR))
    return {"check": "CC-01", "pin_id": src["pin_id"], "pin_sha256": src["sha256"], "build_key": ident["build_key"],
            "utc": pinlib.utc_iso(pinlib.utc_now()), "items": items,
            "verdict": "PASS" if all(i["ok"] for i in items) else "FAIL"}


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS, compat_root=COMPAT_ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin-dir", default=None)
    ap.add_argument("--write", action="store_true", help="also write the result under %s/<pin id>/" % COMPAT_ROOT)
    args = ap.parse_args(argv)
    try:
        res = evaluate(args.pin_dir)
    except pinlib.PinError as exc:
        print("p19_build: %s" % exc, file=sys.stderr)
        return 2
    for i in res["items"]:
        print("%-5s %-13s %s" % ("PASS" if i["ok"] else "FAIL", i["item"], i["detail"]))
    print("CC-01 %s (pin %s, build_key %s)" % (res["verdict"], res["pin_id"], res["build_key"]))
    if args.write:
        try:
            d = pinlib.check_write_path(os.path.join(compat_root, res["pin_id"]), write_roots)
            os.makedirs(d, exist_ok=True)
            path = pinlib.write_json_once(os.path.join(d, "cc01_%s.json" % res["utc"].replace("-", "").replace(":", "")),
                                         res, write_roots)
            print("wrote %s" % path)
        except (pinlib.PinError, OSError) as exc:
            print("p19_build: %s" % exc, file=sys.stderr)
            return 2
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
