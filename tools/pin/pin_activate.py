#!/usr/bin/env python3
"""Switch the active build pin, the one every launcher checks (ticket DD-0 / program TKT-0).

    python3 -I -B pin_activate.py PIN_ID [--pin-dir /mnt/e/DriveDiT/pin]

<pin dir>/ACTIVE holds one line, the active pin id, and `preflight.py --pin-active` checks <pin dir>/pin_<id>.json.
1.8.0.13 is the interim pin. The plan is to move to 1.9 and its new vehicle physics once 1.9 is stable and
compatibility is checked; this command is that switch, and the only supported way to change ACTIVE.

It refuses (and writes nothing) unless:
  - PIN_ID is a valid pin id and <pin dir>/pin_<id>.json is a regular file (not a symlink) holding that pin_id;
  - the full preflight check of that pin passes against the live installs it names, with no waivers. A pin that
    does not match what Steam has installed cannot be activated, so a switch to 1.9 needs Steam to have
    installed 1.9 and a pin made from it; a switch back to 1.8.0.13 needs the 1.8 installs restored;
  - the current ACTIVE, if any, is a regular file (a symlink or anything else is left alone for a human).
Then it writes ACTIVE atomically (temp file and rename in the same directory) and appends
"<UTC ISO time> <old> -> <new> <user>@<host>" to <pin dir>/ACTIVE.log. <old> is "none" when there was no ACTIVE
and "invalid" when ACTIVE held no valid id. If PIN_ID is already active, it re-checks and writes nothing.
Writes are allowed only under /mnt/e (C: is always refused), as for make_pin.py and preflight.py.

Exit codes: 0 activated, or already active; 1 refused (a check failed); 2 usage, config or path error;
3 a write failed (the message says whether ACTIVE changed).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import socket
import stat
import sys


def _load(name):
    # By explicit path, like preflight's _load_pinlib: `python3 -I` does not put this directory on sys.path.
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


def _token(s) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s or "")) or "unknown"


def who() -> str:
    """user@host for the log line, reduced to safe characters (no spaces, no line breaks)."""
    try:
        import pwd
        user = pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        user = os.environ.get("USER") or "uid%d" % os.getuid()
    return "%s@%s" % (_token(user), _token(socket.gethostname()))


def current_active(pin_dir: str) -> str:
    """The id ACTIVE names now: "none" if there is no ACTIVE, "invalid" if it holds no valid id.

    Raises PinError if ACTIVE exists but is not a regular file: a symlink there was not made by this tool, so a
    human should look at it before anything replaces it.
    """
    path = os.path.join(pin_dir, pinlib.ACTIVE_NAME)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "none"
    except OSError as exc:
        raise pinlib.PinError("cannot stat %s: %s" % (path, exc))
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise pinlib.PinError("%s exists but is not a regular file (a symlink?); inspect it and remove it by hand, "
                              "then activate again" % path)
    try:
        return pinlib.read_active(pin_dir)
    except pinlib.PinError:
        return "invalid"


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pin_id", help="the pin to activate, e.g. 1.9.0.12; <pin dir>/pin_<id>.json must exist")
    ap.add_argument("--pin-dir", default=pinlib.DEFAULT_PIN_DIR,
                    help="directory holding the pin files, ACTIVE and ACTIVE.log (default %(default)s)")
    args = ap.parse_args(argv)

    pin_dir = os.path.abspath(pinlib.win_to_wsl(args.pin_dir))
    active = os.path.join(pin_dir, pinlib.ACTIVE_NAME)
    log = os.path.join(pin_dir, pinlib.ACTIVE_LOG_NAME)
    try:
        new = pinlib.validate_pin_id(args.pin_id)
        # Write guard before anything is checked: ACTIVE, its temp file and the log all live in pin_dir.
        pinlib.check_write_path(pin_dir, write_roots)
        src = preflight.load_pin_by_id(pin_dir, new)
        old = current_active(pin_dir)
        # The log must be appendable before ACTIVE changes, so a switch is never left unlogged by a bad log file.
        if os.path.lexists(log):
            st = os.lstat(log)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise pinlib.PinError("%s is not a regular file (a symlink?); inspect it by hand" % log)
    except pinlib.PinError as exc:
        print("pin_activate: %s" % exc, file=sys.stderr)
        return 2

    print("activate %s: %s (sha256 %s); ACTIVE now names %s" % (new, src["path"], src["sha256"], old))
    acf_paths, roots = preflight.pinned_paths(src["pin"])
    checks, _ = preflight.run_checks(src["pin"], acf_paths, roots)
    failed = preflight.print_report(checks)
    if failed:
        print("RESULT: REFUSED (pin %s): %d check(s) failed: %s"
              % (new, len(failed), ", ".join(c["name"] for c in failed)))
        print("pin_activate: %s does not match the live installs; NOT activated, %s unchanged (still %s)"
              % (new, active, old), file=sys.stderr)
        return 1
    print("RESULT: OK (pin %s)" % new)

    if old == new:
        print("pin_activate: %s is already the active pin; nothing written" % new)
        return 0

    line = "%s %s -> %s %s" % (pinlib.utc_iso(pinlib.utc_now()), old, new, who())
    try:
        pinlib.write_text_replace(active, new + "\n", write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("pin_activate: ACTIVE not changed (still %s): %s" % (old, exc), file=sys.stderr)
        return 3
    try:
        pinlib.append_line(log, line, write_roots)
    except (pinlib.PinError, OSError) as exc:
        print("pin_activate: ACTIVE now names %s, but the log line was NOT written to %s: %s\n  missing line: %s"
              % (new, log, exc, line), file=sys.stderr)
        return 3
    try:  # read back through the same validation the launchers use
        if pinlib.read_active(pin_dir) != new:
            raise pinlib.PinError("ACTIVE does not read back as %s" % new)
    except pinlib.PinError as exc:
        print("pin_activate: write did not take: %s" % exc, file=sys.stderr)
        return 3
    print("ACTIVE: %s -> %s (%s)" % (old, new, active))
    print("logged to %s: %s" % (log, line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
