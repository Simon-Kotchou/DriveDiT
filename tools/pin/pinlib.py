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
Workbench addon. /mnt/e is only a name, so the guard also reads /proc/self/mountinfo
and requires the path to lie on the mount of the E: drive itself (WRITE_ROOT_DRIVES):
an unmounted /mnt/e is an empty directory on the WSL disk, which lives on C:.
pin_activate.py writes through one opened directory fd (open_guarded_dir and the
*_at helpers), so a directory swapped after the check cannot redirect its writes.
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

# The installs the launchers start: wb_R1.bat WB_DIR (Tools install + \Workbench), client_R1_i01.bat GAME_DIR,
# and DD-3's launch-workbench.sh WB_DIR. A pin is checked against the appmanifests and install dirs stored in it,
# so preflight's <app>.launch_path check requires those to be exactly these paths. pin_activate.py (no waivers) and
# every launch therefore refuse a pin made from any other tree: the exe a launcher starts is always the one the pin
# checked. Change these only together with the three launchers.
LAUNCH_PATHS = {
    "game": {"acf": DEFAULT_ACF["game"],
             "install": "/mnt/d/SteamLibrary/steamapps/common/Arma Reforger"},
    "tools": {"acf": DEFAULT_ACF["tools"],
              "install": "/mnt/c/Program Files (x86)/Steam/steamapps/common/Arma Reforger Tools"},
}

DEFAULT_WRITE_ROOTS = ("/mnt/e/",)
FORBIDDEN_WRITE_PREFIXES = ("/mnt/c/",)
# A write under one of these mount points must land on the mount of that Windows drive (checked in
# /proc/self/mountinfo), not on whatever directory happens to carry the name.
WRITE_ROOT_DRIVES = {"/mnt/e": "E:"}
MOUNTINFO = "/proc/self/mountinfo"

# Active pin pointer. <pin dir>/ACTIVE holds one line, the id of the pin every launcher checks
# (preflight --pin-active), and that pin is <pin dir>/pin_<id>.json. Only pin_activate.py writes ACTIVE: it
# refuses a pin that does not match the live installs and appends every switch to <pin dir>/ACTIVE.log.
# 1.8.0.13 is the interim pin; the plan is to move to 1.9 once it is stable and compatibility is checked.
DEFAULT_PIN_DIR = "/mnt/e/DriveDiT/pin"
ACTIVE_NAME = "ACTIVE"
ACTIVE_LOG_NAME = "ACTIVE.log"
# Held by pin_activate.py (O_CREAT|O_EXCL) from reading the old ACTIVE to the read-back, so switches are serialized.
ACTIVE_LOCK_NAME = "ACTIVE.lock"
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


def _read_mountinfo() -> str:
    try:
        with open(MOUNTINFO, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        raise PinError("cannot read %s to check which drive a write lands on: %s" % (MOUNTINFO, exc))


def _unescape_mount(s: str) -> str:
    # mountinfo writes space, tab, newline and backslash as \040, \011, \012 and \134.
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def parse_mountinfo(text: str) -> list:
    """[{mount_point, dev, fstype, source, options}] in file order (a later mount at the same point is on top)."""
    mounts = []
    for line in text.splitlines():
        fields = line.split(" ")
        if "-" not in fields[6:]:
            continue
        sep = fields.index("-", 6)
        if len(fields) < sep + 3:
            continue
        major, _, minor = fields[2].partition(":")
        try:
            dev = os.makedev(int(major), int(minor))
        except ValueError:
            continue
        mounts.append({"mount_point": _unescape_mount(fields[4]), "dev": dev, "fstype": fields[sep + 1],
                       "source": _unescape_mount(fields[sep + 2]),
                       "options": _unescape_mount(fields[sep + 3]) if len(fields) > sep + 3 else ""})
    return mounts


def mount_of(resolved: str, mounts: list) -> dict:
    """The mount a resolved absolute path lies on: the longest mount point that contains it (the last one wins)."""
    best = None
    for m in mounts:
        mp = m["mount_point"]
        if mp == "/" or resolved == mp or resolved.startswith(mp.rstrip("/") + "/"):
            if best is None or len(mp) >= len(best["mount_point"]):
                best = m
    return best


def _is_drive(mount: dict, drive: str) -> bool:
    """True if a mountinfo entry is the drvfs mount of the Windows drive "E:" (source E:\\ or option path=E:\\)."""
    want = drive.upper().rstrip(":") + ":"
    if mount["source"].replace("/", "\\").rstrip("\\").upper() == want:
        return True
    opts = re.split(r"[;,]", mount["options"])
    return any(o.replace("/", "\\").rstrip("\\").upper() == "PATH=" + want for o in opts)


def drive_mount_for(resolved: str):
    """For a resolved path under a WRITE_ROOT_DRIVES root, the mountinfo entry of that drive's mount; else None.

    Raises PinError unless the path lies on a mount at exactly that root whose source is that drive. An unmounted
    /mnt/e (an empty directory on the WSL disk), a redirected one, or another filesystem mounted below it is refused.
    """
    for root, drive in WRITE_ROOT_DRIVES.items():
        if resolved == root or resolved.startswith(root.rstrip("/") + "/"):
            m = mount_of(resolved, parse_mountinfo(_read_mountinfo()))
            if m is None or m["mount_point"] != root or not _is_drive(m, drive):
                where = ("nothing (no mount found)" if m is None else
                         "%s (%s from %r)" % (m["mount_point"], m["fstype"], m["source"]))
                raise PinError("refusing to write to %s: it must lie on the %s drive mounted at %s, but it lies on %s"
                               % (resolved, drive, root, where))
            return m
    return None


def check_write_path(path: str, allowed_roots=DEFAULT_WRITE_ROOTS) -> str:
    """Return the resolved absolute path if writing there is allowed, else raise PinError.

    The path is resolved with realpath first, so a symlink under /mnt/e that points to C: is refused.
    /mnt/c is refused even when a caller passes an allowed root that would contain it. A path under /mnt/e must
    also lie on the E: drive's own mount (drive_mount_for), not merely carry the name.
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
            drive_mount_for(resolved)
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


_RD_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)


def _lstat(path: str, dir_fd=None):
    """lstat, relative to dir_fd when given. A separate name so tests can make it lie (the O_NOFOLLOW and fstat
    re-checks below must hold on their own, for a file swapped between this call and the open)."""
    return os.stat(path, dir_fd=dir_fd, follow_symlinks=False)


def _open_regular(path: str, what: str, dir_fd=None, shown=None) -> int:
    """Open a regular file read-only, never through a symlink at the last component; return the fd.

    lstat gives the clear message. O_NOFOLLOW keeps the refusal even if the file is swapped for a link between the
    lstat and the open, O_NONBLOCK keeps a swapped-in FIFO from blocking the open, and fstat re-checks the type.
    """
    shown = shown or path
    try:
        st = _lstat(path, dir_fd)
    except FileNotFoundError:
        raise PinError("%s not found: %s" % (what, shown))
    except OSError as exc:
        raise PinError("cannot read %s %s: %s" % (what, shown, exc))
    if stat.S_ISLNK(st.st_mode):
        raise PinError("%s is a symlink, refused (it must be a regular file): %s" % (what, shown))
    if not stat.S_ISREG(st.st_mode):
        raise PinError("%s is not a regular file: %s" % (what, shown))
    try:
        fd = os.open(path, _RD_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PinError("%s is a symlink, refused (it must be a regular file): %s" % (what, shown))
        raise PinError("cannot read %s %s: %s" % (what, shown, exc))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PinError("%s is not a regular file: %s" % (what, shown))
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_regular_file(path: str, what: str, limit: int, dir_fd=None, shown=None) -> bytes:
    """The whole content of a regular file (see _open_regular). With dir_fd, path is a name in that directory and
    shown is the path used in messages."""
    shown = shown or path
    fd = _open_regular(path, what, dir_fd, shown)
    try:
        chunks, total = [], 0
        while True:
            block = os.read(fd, 1 << 16)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise PinError("%s is larger than %d bytes: %s" % (what, limit, shown))
            chunks.append(block)
    finally:
        os.close(fd)
    return b"".join(chunks)


def read_last_line(path: str, what: str, dir_fd=None, shown=None, tail: int = 4096):
    """The last non-empty line of a regular text file (read from its last `tail` bytes), or None if the file is
    missing or holds no line. Same no-follow rules as read_regular_file."""
    shown = shown or path
    try:
        fd = _open_regular(path, what, dir_fd, shown)
    except PinError:
        try:
            _lstat(path, dir_fd)
        except FileNotFoundError:
            return None
        raise
    try:
        size = os.fstat(fd).st_size
        os.lseek(fd, max(0, size - tail), os.SEEK_SET)
        data = b""
        while len(data) < tail:
            block = os.read(fd, tail - len(data))
            if not block:
                break
            data += block
    finally:
        os.close(fd)
    lines = [l for l in data.decode("ascii", errors="replace").splitlines() if l.strip()]
    return lines[-1] if lines else None


def read_active(pin_dir: str, dir_fd=None) -> str:
    """The pin id named by <pin dir>/ACTIVE (read relative to dir_fd when given, the opened pin dir).

    Raises PinError unless ACTIVE is a regular file (not a symlink) holding exactly one valid pin id,
    optionally followed by one LF or CRLF. Empty, multi-line, padded or non-ASCII content is refused.
    """
    path = os.path.join(pin_dir, ACTIVE_NAME)
    raw = read_regular_file(ACTIVE_NAME if dir_fd is not None else path, "active pin pointer", ACTIVE_MAX_BYTES,
                            dir_fd=dir_fd, shown=path)
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
    if "\n" in body:
        raise PinError("%s holds more than one line; it must hold exactly one pin id" % path)
    if "\r" in body:
        raise PinError("%s holds a carriage return (CR) that does not end a CRLF line ending; it must hold exactly "
                       "one pin id, optionally followed by LF or CRLF" % path)
    try:
        return validate_pin_id(body)
    except PinError as exc:
        raise PinError("%s: %s" % (path, exc))


def _check_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise PinError("not a plain file name: %r" % (name,))
    return name


def open_guarded_dir(path: str, allowed_roots=DEFAULT_WRITE_ROOTS) -> int:
    """Open an existing directory that may be written to and return its fd, for the *_at helpers below.

    The path goes through check_write_path first. The guard then runs again on the directory the fd really refers to
    (its /proc/self/fd link), and that directory must be on the drive's own mount (same st_dev). Every later create,
    replace and unlink uses dir_fd=, so a directory swapped for a symlink after this call cannot redirect a write.
    """
    resolved = check_write_path(path, allowed_roots)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(resolved, flags)
    except FileNotFoundError:
        raise PinError("directory does not exist (create it first): %s" % resolved)
    except OSError as exc:
        raise PinError("cannot open directory %s: %s" % (resolved, exc))
    try:
        try:
            actual = os.readlink("/proc/self/fd/%d" % fd)
        except OSError as exc:
            raise PinError("cannot tell which directory %s opened as: %s" % (resolved, exc))
        if not actual.startswith("/") or actual.endswith(" (deleted)"):
            raise PinError("directory %s is gone or not a path (%s)" % (resolved, actual))
        check_write_path(actual, allowed_roots)
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            raise PinError("not a directory: %s" % actual)
        mount = drive_mount_for(actual)
        if mount is not None and st.st_dev != mount["dev"]:
            raise PinError("directory %s is not on the %s mount (device %d:%d)"
                           % (actual, mount["mount_point"], os.major(mount["dev"]), os.minor(mount["dev"])))
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def replace_text_at(dir_fd: int, name: str, text: str) -> None:
    """Atomically create or replace a small ASCII text file <dir>/<name>, relative to an opened directory fd.

    A temp file in the same directory is created with O_EXCL|O_NOFOLLOW, written and fsynced, then renamed over the
    target, so a reader sees the old or the new content, never a mix. A symlink at the target name is replaced,
    never followed. The directory is fsynced afterwards, best effort (drvfs may not support it).
    """
    _check_name(name)
    data = text.encode("ascii")
    tmp = "%s.tmp-%d" % (name, os.getpid())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(tmp, flags, 0o644, dir_fd=dir_fd)
    try:
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:  # the target is unchanged (rename is atomic); remove our temp file, best effort
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    try:
        os.fsync(dir_fd)
    except OSError:
        pass


def open_append_at(dir_fd: int, name: str, shown=None) -> int:
    """Open <dir>/<name> for appending (created if missing) and return the fd; for write_line.

    O_APPEND|O_NOFOLLOW relative to the directory fd: a symlink at the name is refused, not followed, and the
    file must be a regular file. Opening it before a change is made proves the log can be written to.
    """
    shown = shown or name
    flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(_check_name(name), flags, 0o644, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PinError("%s is a symlink, refused" % shown)
        raise PinError("cannot open %s for append: %s" % (shown, exc))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PinError("%s is not a regular file" % shown)
    except BaseException:
        os.close(fd)
        raise
    return fd


def write_line(fd: int, line: str) -> None:
    """Append one ASCII line (a newline is added) to an fd from open_append_at, then fsync it."""
    if "\n" in line or "\r" in line:
        raise PinError("a log line must not contain a line break")
    _write_all(fd, (line + "\n").encode("ascii"))
    os.fsync(fd)


def create_lock_at(dir_fd: int, name: str, text: str, shown=None) -> None:
    """Take a lock: create <dir>/<name> with O_CREAT|O_EXCL|O_NOFOLLOW and write text (who holds it) into it.

    Raises PinError if the name exists, whatever it is: the holder (or a killed run) owns it, never this caller.
    """
    shown = shown or name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(_check_name(name), flags, 0o644, dir_fd=dir_fd)
    except FileExistsError:
        holder = ""
        try:
            holder = read_regular_file(name, "lock file", 4096, dir_fd=dir_fd, shown=shown).decode(
                "ascii", errors="replace").strip()
        except PinError:
            pass
        raise PinError("%s exists (%s): another pin_activate.py is running, or one was killed. If none is running, "
                       "check ACTIVE and ACTIVE.log, then delete %s by hand"
                       % (shown, holder or "holder unknown", shown))
    except OSError as exc:
        raise PinError("cannot create lock %s: %s" % (shown, exc))
    try:
        _write_all(fd, (text + "\n").encode("ascii", errors="replace"))
    except OSError as exc:
        os.close(fd)
        try:
            os.unlink(name, dir_fd=dir_fd)
        except OSError:
            pass
        raise PinError("cannot write lock %s: %s" % (shown, exc))
    os.close(fd)


def remove_lock_at(dir_fd: int, name: str) -> None:
    os.unlink(_check_name(name), dir_fd=dir_fd)
