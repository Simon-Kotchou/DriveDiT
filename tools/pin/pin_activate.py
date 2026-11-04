#!/usr/bin/env python3
"""Switch the active build pin, the one every launcher checks (ticket DD-0 / program TKT-0).

    python3 -I -B pin_activate.py PIN_ID [--pin-dir /mnt/e/DriveDiT/pin]

<pin dir>/ACTIVE holds one line, the active pin id, and `preflight.py --pin-active` checks <pin dir>/pin_<id>.json.
1.8.0.13 is the interim pin. The plan is to move to 1.9 and its new vehicle physics once 1.9 is stable and
compatibility is checked; this command is that switch, and the only supported way to change ACTIVE.

It refuses (and changes nothing) unless:
  - PIN_ID is a valid pin id and <pin dir>/pin_<id>.json is a regular file (not a symlink) holding that pin_id;
  - no other activation holds <pin dir>/ACTIVE.lock (created with O_EXCL; held from reading the old ACTIVE to the
    read-back, so switches are serialized; a lock left by a killed run is removed by hand);
  - the full preflight check of that pin passes against the live installs it names, with no waivers. That includes
    <app>.launch_path: the pin must have been made from the installs the launchers start. A pin that does not match
    what Steam has installed cannot be activated, so a switch to 1.9 needs Steam to have installed 1.9 and a pin
    made from it. A switch back to 1.8 after a restore needs a re-pin of the restored installs (make_pin.py
    --restore-of 1.8.0.13 --pin-id 1.8.0.13-r1), because no restore brings back the sub-second mtimes of
    pin_1.8.0.13.json; see README "Rollback";
  - the current ACTIVE and ACTIVE.log, if any, are regular files (a symlink or anything else is left for a human),
    and ACTIVE.log can be opened for appending.
Then it writes ACTIVE atomically (temp file and rename in the same directory) and appends
"<UTC ISO time> <old> -> <new> <user>@<host>" to <pin dir>/ACTIVE.log, through a log fd opened before the switch.
<old> is "none" when there was no ACTIVE and "invalid" when ACTIVE held no valid id. If PIN_ID is already active,
it re-checks and writes nothing, unless the last log line does not record a switch to PIN_ID (a previous run
switched but could not write its line): then it appends "<UTC ISO time> unlogged -> <new> <user>@<host>".
Writes are allowed only on the E: drive (C: is always refused), as for make_pin.py and preflight.py, and all of
them go through one opened fd of the pin dir.

Exit codes: 0 activated, already active, or the missing log line repaired; 1 refused (a check failed); 2 usage,
config, path or lock error; 3 a write failed (the message says whether ACTIVE changed).
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


def current_active(dir_fd: int, pin_dir: str) -> str:
    """The id ACTIVE names now: "none" if there is no ACTIVE, "invalid" if it holds no valid id.

    Raises PinError if ACTIVE exists but is not a regular file: a symlink there was not made by this tool, so a
    human should look at it before anything replaces it.
    """
    path = os.path.join(pin_dir, pinlib.ACTIVE_NAME)
    try:
        st = os.stat(pinlib.ACTIVE_NAME, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return "none"
    except OSError as exc:
        raise pinlib.PinError("cannot stat %s: %s" % (path, exc))
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise pinlib.PinError("%s exists but is not a regular file (a symlink?); inspect it and remove it by hand, "
                              "then activate again" % path)
    try:
        return pinlib.read_active(pin_dir, dir_fd=dir_fd)
    except pinlib.PinError:
        return "invalid"


def logged_switch_to(line, pin_id: str) -> bool:
    """True if a log line has the form "<UTC> <old> -> <pin_id> <user>@<host>"."""
    parts = (line or "").split(" ")
    return len(parts) == 5 and parts[2] == "->" and parts[3] == pin_id


def log_line(old: str, new: str) -> str:
    return "%s %s -> %s %s" % (pinlib.utc_iso(pinlib.utc_now()), old, new, who())


def main(argv=None, write_roots=pinlib.DEFAULT_WRITE_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pin_id", help="the pin to activate, e.g. 1.9.0.12; <pin dir>/pin_<id>.json must exist")
    ap.add_argument("--pin-dir", default=pinlib.DEFAULT_PIN_DIR,
                    help="directory holding the pin files, ACTIVE, ACTIVE.log and ACTIVE.lock (default %(default)s)")
    args = ap.parse_args(argv)

    pin_dir = os.path.abspath(pinlib.win_to_wsl(args.pin_dir))
    lock = os.path.join(pin_dir, pinlib.ACTIVE_LOCK_NAME)
    try:
        new = pinlib.validate_pin_id(args.pin_id)
        # Write guard before anything is read: ACTIVE, its temp file, the lock and the log all live in pin_dir, and
        # every one of them is written through this one directory fd.
        dir_fd = pinlib.open_guarded_dir(pin_dir, write_roots)
    except pinlib.PinError as exc:
        print("pin_activate: %s" % exc, file=sys.stderr)
        return 2
    try:
        try:
            src = preflight.load_pin_by_id(pin_dir, new, dir_fd=dir_fd)
            pinlib.create_lock_at(dir_fd, pinlib.ACTIVE_LOCK_NAME, "pid %d %s %s" % (
                os.getpid(), who(), pinlib.utc_iso(pinlib.utc_now())), shown=lock)
        except pinlib.PinError as exc:
            print("pin_activate: %s" % exc, file=sys.stderr)
            return 2
        try:
            return _activate_locked(dir_fd, pin_dir, new, src)
        finally:
            try:
                pinlib.remove_lock_at(dir_fd, pinlib.ACTIVE_LOCK_NAME)
            except OSError as exc:
                print("pin_activate: WARNING: could not remove %s (%s); delete it by hand" % (lock, exc),
                      file=sys.stderr)
    finally:
        os.close(dir_fd)


def _activate_locked(dir_fd: int, pin_dir: str, new: str, src: dict) -> int:
    active = os.path.join(pin_dir, pinlib.ACTIVE_NAME)
    log = os.path.join(pin_dir, pinlib.ACTIVE_LOG_NAME)
    try:
        old = current_active(dir_fd, pin_dir)
        try:
            st = os.stat(pinlib.ACTIVE_LOG_NAME, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            st = None
        if st is not None and (stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode)):
            raise pinlib.PinError("%s is not a regular file (a symlink?); inspect it by hand" % log)
    except (pinlib.PinError, OSError) as exc:
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
        try:
            last = pinlib.read_last_line(pinlib.ACTIVE_LOG_NAME, "activation log", dir_fd=dir_fd, shown=log)
        except pinlib.PinError as exc:
            print("pin_activate: %s is already the active pin, but %s cannot be read: %s" % (new, log, exc),
                  file=sys.stderr)
            return 3
        if logged_switch_to(last, new):
            print("pin_activate: %s is already the active pin; nothing written" % new)
            return 0
        # ACTIVE names new, but the log never recorded the switch (a run whose log write failed, or a hand edit).
        line = log_line("unlogged", new)
        try:
            fd = pinlib.open_append_at(dir_fd, pinlib.ACTIVE_LOG_NAME, shown=log)
            try:
                pinlib.write_line(fd, line)
            finally:
                os.close(fd)
        except (pinlib.PinError, OSError) as exc:
            print("pin_activate: %s is the active pin, but its switch is not in %s, and the repair line could not "
                  "be written: %s\n  missing line: %s" % (new, log, exc, line), file=sys.stderr)
            return 3
        print("pin_activate: %s was already the active pin, but %s did not record a switch to it; appended: %s"
              % (new, log, line))
        return 0

    line = log_line(old, new)
    try:  # open the log before the switch: a log that cannot be appended to refuses with ACTIVE unchanged
        log_fd = pinlib.open_append_at(dir_fd, pinlib.ACTIVE_LOG_NAME, shown=log)
    except (pinlib.PinError, OSError) as exc:
        print("pin_activate: ACTIVE not changed (still %s): %s" % (old, exc), file=sys.stderr)
        return 3
    try:
        try:
            pinlib.replace_text_at(dir_fd, pinlib.ACTIVE_NAME, new + "\n")
        except (pinlib.PinError, OSError) as exc:
            print("pin_activate: ACTIVE not changed (still %s): %s" % (old, exc), file=sys.stderr)
            return 3
        try:
            pinlib.write_line(log_fd, line)
        except (pinlib.PinError, OSError) as exc:
            print("pin_activate: ACTIVE now names %s, but the log line was NOT written to %s: %s\n  missing line: %s\n"
                  "  Running pin_activate.py %s again appends an 'unlogged' line for this switch."
                  % (new, log, exc, line, new), file=sys.stderr)
            return 3
    finally:
        os.close(log_fd)
    try:  # read back through the same validation the launchers use
        if pinlib.read_active(pin_dir, dir_fd=dir_fd) != new:
            raise pinlib.PinError("ACTIVE does not read back as %s" % new)
    except pinlib.PinError as exc:
        print("pin_activate: write did not take: %s" % exc, file=sys.stderr)
        return 3
    print("ACTIVE: %s -> %s (%s)" % (old, new, active))
    print("logged to %s: %s" % (log, line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
