"""Shared helpers for the DriveDiT build pin (ticket DD-0, capture program TKT-0).

Stdlib only. Must run on the WSL system interpreter (Ubuntu 22.04: Python 3.10),
because the Windows launchers call `wsl.exe -e python3 -I -B ...` without a login
shell. make_pin.py and preflight.py load this file by explicit path (their
_load_pinlib): `python3 -I` does not put the script directory on sys.path, and
loading by path keeps a stray json.py or os.py next to the scripts from shadowing
the stdlib.

Write safety: every path this package writes to goes through check_write_path().
Writes are allowed only under /mnt/e/ (the E: drive). /mnt/c/ is always refused,
whatever the allowed roots are, because C: is nearly full and holds the live
Workbench addon.
"""
from __future__ import annotations

import datetime as _dt
import errno
import hashlib
import json
import os
import re
import stat

PIN_SCHEMA = "dd-pin/1"
LAUNCH_SCHEMA = "dd-launch/1"

APPS = ("game", "tools")
APPIDS = {"game": 1874880, "tools": 1874910}
DEFAULT_ACF = {
    "game": "/mnt/d/SteamLibrary/steamapps/appmanifest_1874880.acf",
    "tools": "/mnt/c/Program Files (x86)/Steam/steamapps/appmanifest_1874910.acf",
}
# Executables recorded in the pin and checked by preflight (paths relative to the install dir).
BINARIES = {
    "game": ("ArmaReforgerSteam.exe", "ArmaReforgerSteamDiag.exe"),
    "tools": ("Workbench/ArmaReforgerWorkbenchSteamDiag.exe",),
}
# steam_appid.txt files that let the exes start without a Steam relaunch (CAPTURE_PROGRAM r2 4.2 item 2).
STEAM_APPID_FILES = {
    "game": ("steam_appid.txt",),
    "tools": ("steam_appid.txt", "Workbench/steam_appid.txt"),
}

# Steam "Automatic updates" setting stored as AutoUpdateBehavior in appmanifest_<appid>.acf.
# Mapping is community-documented (CAPTURE_PROGRAM r2 4.2 item 1); Valve docs are UNVERIFIED.
AUTO_UPDATE_MEANING = {
    "0": "Always keep this game updated",
    "1": "Only update this game when I launch it",
    "2": "High priority: always auto-update this game before others",
}
# Only "1" stops background updates. "0" is what the plan forbids; "2" also auto-updates, so it is refused too.
AUTO_UPDATE_ALLOWED = ("1",)
# StateFlags 4 = fully installed, no update queued or running (community-documented, UNVERIFIED).
STATE_FLAGS_SETTLED = "4"

DEFAULT_WRITE_ROOTS = ("/mnt/e/",)
FORBIDDEN_WRITE_PREFIXES = ("/mnt/c/",)

# Active pin pointer. <pin dir>/ACTIVE holds one line, the id of the pin every launcher checks
# (preflight --pin-active), and that pin is <pin dir>/pin_<id>.json. Only pin_activate.py writes ACTIVE: it
# refuses a pin that does not match the live installs and appends every switch to <pin dir>/ACTIVE.log.
# 1.8.0.13 is the interim pin; the plan is to move to 1.9 once it is stable and compatibility is checked.
DEFAULT_PIN_DIR = "/mnt/e/DriveDiT/pin"
ACTIVE_NAME = "ACTIVE"
ACTIVE_LOG_NAME = "ACTIVE.log"
ACTIVE_MAX_BYTES = 64
PIN_MAX_BYTES = 4 * 1024 * 1024
# A pin id names a file (pin_<id>.json) and a directory (<pin dir>/<id>, the cold copy), so it is strict:
# 2 to 4 dot-separated numbers and an optional lowercase suffix, e.g. 1.8.0.13, 1.9.0.12, 1.9.0.9-exp.
# No path separator, no "..", no whitespace. Always used with fullmatch, so a trailing newline is refused too.
PIN_ID_RE = re.compile(r"[0-9]{1,4}(?:\.[0-9]{1,9}){1,3}(?:-[a-z0-9]{1,16})?")


class PinError(Exception):
    """A configuration or safety error (bad path, unreadable file, malformed data)."""


# --------------------------------------------------------------------------- paths


_WIN_ABS = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL_MNT = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")


def win_to_wsl(path: str) -> str:
    """'E:\\DriveDiT\\pin' -> '/mnt/e/DriveDiT/pin'. Other strings are returned unchanged."""
    m = _WIN_ABS.match(path)
    if not m:
        return path
    rest = m.group(2).replace("\\", "/").strip("/")
    return "/mnt/%s/%s" % (m.group(1).lower(), rest) if rest else "/mnt/%s" % m.group(1).lower()


def wsl_to_win(path: str):
    """'/mnt/d/SteamLibrary/x' -> 'D:\\SteamLibrary\\x'; None for paths outside /mnt/<drive>."""
    m = _WSL_MNT.match(path)
    if not m:
        return None
    rest = (m.group(2) or "").strip("/")
    return "%s:\\%s" % (m.group(1).upper(), rest.replace("/", "\\"))


def check_write_path(path: str, allowed_roots=DEFAULT_WRITE_ROOTS) -> str:
    """Return the resolved absolute path if writing there is allowed, else raise PinError.

    The path is resolved with realpath first, so a symlink under /mnt/e that points to C: is refused.
    /mnt/c is refused even when a caller passes an allowed root that would contain it.
    """
    if not path:
        raise PinError("empty output path")
    resolved = os.path.realpath(os.path.abspath(win_to_wsl(path)))
    low = resolved.lower() + "/"
    for bad in FORBIDDEN_WRITE_PREFIXES:
        if low.startswith(bad):
            raise PinError("refusing to write on C: (%s resolves to %s)" % (path, resolved))
    for root in allowed_roots:
        r = os.path.realpath(os.path.abspath(root))
        if resolved == r or resolved.startswith(r.rstrip("/") + "/"):
            return resolved
    raise PinError("refusing to write outside %s: %s (resolves to %s)"
                   % (", ".join(allowed_roots), path, resolved))


# --------------------------------------------------------------------------- Valve KeyValues (.acf)


def _tokenize_vdf(text: str):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n\ufeff":
            i += 1
        elif c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif c in "{}":
            tokens.append(c)
            i += 1
        elif c == '"':
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise PinError("unterminated string in KeyValues text")
                c = text[i]
                if c == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    buf.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(nxt, "\\" + nxt))
                    i += 2
                elif c == '"':
                    i += 1
                    break
                else:
                    buf.append(c)
                    i += 1
            tokens.append(("s", "".join(buf)))
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n{}"':
                j += 1
            word = text[i:j]
            i = j
            if word.startswith("[") and word.endswith("]"):
                continue  # platform conditional such as [$WIN32]; not used in appmanifests
            tokens.append(("s", word))
    return tokens


def parse_vdf(text: str) -> dict:
    """Parse Valve KeyValues text (the .acf format) into nested dicts. Later duplicate keys win."""
    tokens = _tokenize_vdf(text)
    pos = 0

    def parse_obj(top: bool) -> dict:
        nonlocal pos
        out = {}
        while pos < len(tokens):
            tok = tokens[pos]
            if tok == "}":
                if top:
                    raise PinError("unbalanced '}' in KeyValues text")
                pos += 1
                return out
            if tok == "{":
                raise PinError("'{' where a key was expected in KeyValues text")
            key = tok[1]
            pos += 1
            if pos >= len(tokens):
                raise PinError("key %r has no value" % key)
            val = tokens[pos]
            if val == "{":
                pos += 1
                out[key] = parse_obj(False)
            elif val == "}":
                raise PinError("key %r has no value" % key)
            else:
                out[key] = val[1]
                pos += 1
        if not top:
            raise PinError("missing '}' in KeyValues text")
        return out

    return parse_obj(True)


def kv_get(d: dict, key: str, default=None):
    """Case-insensitive dict lookup (Steam treats KeyValues keys case-insensitively)."""
    if key in d:
        return d[key]
    low = key.lower()
    for k, v in d.items():
        if k.lower() == low:
            return v
    return default


def _to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def load_acf(path: str) -> dict:
    """Read an appmanifest .acf (read-only) and return a normalised summary.

    The summary deliberately leaves out LastOwner (a personal Steam id) and download counters.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        raise PinError("cannot read %s: %s" % (path, exc))
    root = parse_vdf(text)
    app = kv_get(root, "AppState")
    if not isinstance(app, dict):
        raise PinError("%s has no AppState section" % path)
    depots = {}
    inst = kv_get(app, "InstalledDepots", {})
    if isinstance(inst, dict):
        for depot_id, info in inst.items():
            if isinstance(info, dict):
                depots[str(depot_id)] = {
                    "manifest": str(kv_get(info, "manifest", "")),
                    "size": _to_int(kv_get(info, "size")),
                }
    aub = kv_get(app, "AutoUpdateBehavior")
    return {
        "appid": _to_int(kv_get(app, "appid")),
        "name": kv_get(app, "name"),
        "installdir": kv_get(app, "installdir"),
        "buildid": _to_int(kv_get(app, "buildid")),
        "target_buildid": _to_int(kv_get(app, "TargetBuildID")),
        "auto_update_behavior": None if aub is None else str(aub).strip(),
        "state_flags": None if kv_get(app, "StateFlags") is None else str(kv_get(app, "StateFlags")).strip(),
        "update_result": kv_get(app, "UpdateResult"),
        "last_updated": _to_int(kv_get(app, "LastUpdated")),
        "depots": dict(sorted(depots.items())),
    }


def install_dir_for_acf(acf_path: str, installdir: str) -> str:
    """<library>/steamapps/appmanifest_N.acf -> <library>/steamapps/common/<installdir>."""
    if not installdir:
        raise PinError("%s has no installdir" % acf_path)
    return os.path.join(os.path.dirname(os.path.abspath(acf_path)), "common", installdir)


# --------------------------------------------------------------------------- files


def iso_utc_from_ns(ns: int) -> str:
    dt = _dt.datetime.fromtimestamp(ns / 1e9, tz=_dt.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def iso_local_from_ns(ns: int) -> str:
    return _dt.datetime.fromtimestamp(ns / 1e9).astimezone().isoformat(timespec="seconds")


def stat_entry(path: str) -> dict:
    st = os.stat(path)
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "mtime_utc": iso_utc_from_ns(st.st_mtime_ns),
        "mtime_local": iso_local_from_ns(st.st_mtime_ns),
    }


def scan_paks(root: str) -> dict:
    """Every *.pak (any case) under root, keyed by '/'-separated relative path, sorted."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(".pak"):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                found[rel] = stat_entry(full)
    return found


def read_small_text(path: str, limit: int = 4096) -> str:
    with open(path, "rb") as fh:
        return fh.read(limit).decode("utf-8", errors="replace").strip()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def utc_iso(dt: _dt.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_compact(dt: _dt.datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def write_json_new(path: str, obj, allowed_roots=DEFAULT_WRITE_ROOTS, overwrite: bool = False) -> str:
    """Write JSON to a guarded path. Never creates parent directories.

    overwrite=False uses exclusive create, so an existing file is never replaced.
    overwrite=True writes a temp file in the same directory and renames it over the target.
    """
    target = check_write_path(path, allowed_roots)
    parent = os.path.dirname(target)
    if not os.path.isdir(parent):
        raise PinError("output directory does not exist (create it first): %s" % parent)
    data = json.dumps(obj, indent=2) + "\n"
    if not overwrite:
        try:
            with open(target, "x", encoding="utf-8", newline="\n") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except FileExistsError:
            raise PinError("refusing to overwrite existing file: %s" % target)
        return target
    tmp = "%s.tmp-%d" % (target, os.getpid())
    check_write_path(tmp, allowed_roots)
    try:
        with open(tmp, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


# --------------------------------------------------------------------------- active pin pointer


def validate_pin_id(pin_id) -> str:
    """Return pin_id if it is a valid pin id (PIN_ID_RE, whole string), else raise PinError."""
    if not isinstance(pin_id, str) or PIN_ID_RE.fullmatch(pin_id) is None:
        raise PinError("invalid pin id %r: expected 2 to 4 dot-separated numbers and an optional lowercase "
                       "suffix, such as 1.8.0.13, 1.9.0.12 or 1.9.0.9-exp" % (pin_id,))
    return pin_id


def pin_file_for(pin_dir: str, pin_id: str) -> str:
    """<pin dir>/pin_<id>.json for a valid pin id."""
    return os.path.join(pin_dir, "pin_%s.json" % validate_pin_id(pin_id))


def read_regular_file(path: str, what: str, limit: int) -> bytes:
    """The whole content of a regular file. A symlink at the last path component is refused.

    lstat gives the clear message. O_NOFOLLOW keeps the refusal even if the file is swapped for a link
    between the lstat and the open, and O_NONBLOCK keeps a swapped-in FIFO from blocking the open.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise PinError("%s not found: %s" % (what, path))
    except OSError as exc:
        raise PinError("cannot read %s %s: %s" % (what, path, exc))
    if stat.S_ISLNK(st.st_mode):
        raise PinError("%s is a symlink, refused (it must be a regular file): %s" % (what, path))
    if not stat.S_ISREG(st.st_mode):
        raise PinError("%s is not a regular file: %s" % (what, path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PinError("%s is a symlink, refused (it must be a regular file): %s" % (what, path))
        raise PinError("cannot read %s %s: %s" % (what, path, exc))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PinError("%s is not a regular file: %s" % (what, path))
        chunks, total = [], 0
        while True:
            block = os.read(fd, 1 << 16)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise PinError("%s is larger than %d bytes: %s" % (what, limit, path))
            chunks.append(block)
    finally:
        os.close(fd)
    return b"".join(chunks)


def read_active(pin_dir: str) -> str:
    """The pin id named by <pin dir>/ACTIVE.

    Raises PinError unless ACTIVE is a regular file (not a symlink) holding exactly one valid pin id,
    optionally followed by one LF or CRLF. Empty, multi-line, padded or non-ASCII content is refused.
    """
    path = os.path.join(pin_dir, ACTIVE_NAME)
    raw = read_regular_file(path, "active pin pointer", ACTIVE_MAX_BYTES)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise PinError("%s is not plain ASCII; it must hold exactly one pin id line" % path)
    if text.endswith("\r\n"):
        body = text[:-2]
    elif text.endswith("\n"):
        body = text[:-1]
    else:
        body = text
    if not body.strip():
        raise PinError("%s is empty; set it with pin_activate.py <pin id>" % path)
    if "\n" in body or "\r" in body:
        raise PinError("%s holds more than one line; it must hold exactly one pin id" % path)
    try:
        return validate_pin_id(body)
    except PinError as exc:
        raise PinError("%s: %s" % (path, exc))


def guarded_entry(path: str, allowed_roots=DEFAULT_WRITE_ROOTS) -> str:
    """Resolve the PARENT directory of path through check_write_path and return <resolved parent>/<name>.

    For files that are replaced by rename or opened with O_NOFOLLOW: the directory entry itself is written,
    never whatever a symlink at that name points to. The parent must already exist.
    """
    full = os.path.abspath(win_to_wsl(path))
    name = os.path.basename(full)
    if not name or name in (".", ".."):
        raise PinError("not a file path: %s" % path)
    parent = check_write_path(os.path.dirname(full), allowed_roots)
    if not os.path.isdir(parent):
        raise PinError("output directory does not exist (create it first): %s" % parent)
    return os.path.join(parent, name)


def write_text_replace(path: str, text: str, allowed_roots=DEFAULT_WRITE_ROOTS) -> str:
    """Atomically create or replace a small text file on a guarded path.

    A temp file in the same directory is written and fsynced, then renamed over the target, so a reader sees
    the old or the new content, never a mix. A symlink at the target name is replaced, never followed.
    """
    target = guarded_entry(path, allowed_roots)
    tmp = "%s.tmp-%d" % (target, os.getpid())
    try:
        with open(tmp, "x", encoding="ascii", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)
    try:  # make the rename durable; drvfs may not support fsync on a directory
        dfd = os.open(os.path.dirname(target), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return target


def append_line(path: str, line: str, allowed_roots=DEFAULT_WRITE_ROOTS) -> str:
    """Append one ASCII line (a newline is added) to a guarded regular file, creating it if needed.

    The file is opened with O_APPEND and O_NOFOLLOW: a symlink at the name is refused, not followed.
    """
    if "\n" in line or "\r" in line:
        raise PinError("a log line must not contain a line break")
    data = (line + "\n").encode("ascii")
    target = guarded_entry(path, allowed_roots)
    flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(target, flags, 0o644)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PinError("%s is a symlink, refused" % target)
        raise PinError("cannot open %s for append: %s" % (target, exc))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PinError("%s is not a regular file" % target)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return target
