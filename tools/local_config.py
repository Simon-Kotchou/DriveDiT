#!/usr/bin/env python3
"""The local machine paths of DriveDiT (DD-22). Stdlib only; runs on Python 3.10, the WSL system interpreter the
Windows launchers call, and later.

The repository holds placeholders and logic only. Every path that differs from machine to machine (the data root,
the game and Tools installs, the WSL distro the Windows launchers call, ...) lives in ONE untracked file,
drivedit.local.env, that each machine supplies. tools/launch/dd_local_config.cmd reads the same file, in the same
order, for the Windows launchers. Copy drivedit.local.env.example (repo root) to start one, or let
tools/migrate_local_config.py write it from the values a machine used before DD-22.

Where the file is looked for, in this order (the first that exists is used; no path is ever guessed):
  1. $DRIVEDIT_LOCAL_CONFIG: the path of the file (a WSL path, or a Windows path X:\\..., mapped to /mnt/x/...).
  2. <tools dir>/../config/drivedit.local.env, <tools dir> being the folder that holds this file. For the installed
     tools (<DATA_ROOT>/tools) that is <DATA_ROOT>/config/drivedit.local.env.
  3. <tools dir>/../drivedit.local.env, only in a git checkout (<tools dir>/../.git exists): the gitignored repo-root
     file of a development worktree.
When none exists, or a key a caller needs is missing, the caller refuses with a message that names the file and the
key (LocalConfigError). Importing this module reads nothing; a file is read on first use and re-read when it changes.

Format (the Windows launchers parse it with `for /f`, so it is deliberately plain):
  * one KEY=VALUE per line, no spaces around "=", no quotes, no variable expansion; a line starting with # is a
    comment, blank lines are ignored; a key may appear once; unknown keys are refused (a typo never goes unseen);
  * a path is the WSL form, absolute and normalised (no trailing "/", no "//", ".", ".."). The Windows form is
    derived by the pure mapping /mnt/<x>/<rest> <-> <X>:\\<rest> (wsl_to_win), the same mapping the batch helper
    applies; `check` compares it with wslpath on the machine;
  * values use letters, digits, space and _ . , + ( ) ~ / - only, so no value can carry a cmd.exe metacharacter;
    DATA_ROOT and WSL_DISTRO, which reach the wsl.exe command line of the launchers unquoted, also no space or
    parenthesis.

Keys (KEYS below): DATA_ROOT, REALIGN_ROOT, CODE_ROOT, GAME_INSTALL, TOOLS_INSTALL, EXPERIMENTAL_GAME_INSTALL,
EXPERIMENTAL_TOOLS_INSTALL, SYSTEM_DRIVE, WSL_DISTRO.

    python3 -I -B tools/local_config.py where              # which file is used (or where it was looked for)
    python3 -I -B tools/local_config.py show               # every key, WSL and Windows form
    python3 -I -B tools/local_config.py get KEY [--win | --drive | --drive-win]   # one value, for shell scripts
    python3 -I -B tools/local_config.py render FILE        # FILE with every <KEY> / <KEY_WIN> filled in
    python3 -I -B tools/local_config.py check [--cmd]      # validate the file on this machine (writes nothing)

`check` exits 0 when every check passes, 1 when one fails, 2 when the file cannot be found or parsed. --cmd also
runs tools/launch/dd_local_config.cmd /print through cmd.exe (it only reads) and compares what it derives, and runs
echo-only copies of the four launchers (their helper call and set lines, then the wsl.exe command and the start line
printed instead of run; in a temporary directory under /dev/shm) and compares them with the Python reading.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

CONFIG_NAME = "drivedit.local.env"
ENV_VAR = "DRIVEDIT_LOCAL_CONFIG"
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_MAX_BYTES = 64 * 1024

# kind: "drvfs" a path on a Windows drive (/mnt/<x>/...), "path" any absolute WSL path, "steam" a drvfs path of the
# form <library>/steamapps/common/<install dir>, "drive" a drive mount /mnt/<x>, "name" a WSL distro name.
KEYS = {
    "DATA_ROOT": ("drvfs", "the capture data root (pin/, addons/, captures/, profiles, logs, reports, the installed "
                           "tools/); letters, digits and _ . / - only"),
    "REALIGN_ROOT": ("drvfs", "the planning and evidence tree (program/, evidence/); the MCP server's work root is "
                              "<REALIGN_ROOT>/mcp"),
    "CODE_ROOT": ("path", "local source checkouts outside this repo (third_party/dinov3)"),
    "GAME_INSTALL": ("steam", "the stable Arma Reforger install the launchers start"),
    "TOOLS_INSTALL": ("steam", "the stable Arma Reforger Tools install the launchers start"),
    "EXPERIMENTAL_GAME_INSTALL": ("steam", "the Arma Reforger Experimental install (pins for --check only)"),
    "EXPERIMENTAL_TOOLS_INSTALL": ("steam", "the Arma Reforger Experimental Tools install (pins for --check only)"),
    "SYSTEM_DRIVE": ("drive", "the mount of the Windows system drive; nothing is ever written there"),
    "WSL_DISTRO": ("name", "the WSL distro the Windows launchers run the preflight in (wsl.exe -d)"),
}
PATH_KINDS = ("drvfs", "path", "steam", "drive")

_KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
_VALUE_RE = re.compile(r"[A-Za-z0-9 _.,+()~/-]+")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ROOT_RE = re.compile(r"[A-Za-z0-9_./-]+")  # DATA_ROOT: no space, no parenthesis (unquoted on the wsl.exe line)
_DRIVE_RE = re.compile(r"/mnt/[a-z]")
_WSL_MNT = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")
_WIN_ABS = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_PLACEHOLDER_RE = re.compile(r"<[^>]*>")


class LocalConfigError(Exception):
    """The local config is missing, unreadable or malformed, or lacks a key a caller needs."""


# --------------------------------------------------------------------------- the pure path mapping


def win_to_wsl(path: str) -> str:
    """'X:\\a\\b' or 'x:/a/b' -> '/mnt/x/a/b'. Other strings are returned unchanged."""
    m = _WIN_ABS.match(path)
    if not m:
        return path
    rest = m.group(2).replace("\\", "/").strip("/")
    return "/mnt/%s/%s" % (m.group(1).lower(), rest) if rest else "/mnt/%s" % m.group(1).lower()


def wsl_to_win(path: str):
    """'/mnt/x/a/b' -> 'X:\\a\\b' ('/mnt/x' -> 'X:\\'); None for a path outside /mnt/<letter>."""
    m = _WSL_MNT.match(path)
    if not m:
        return None
    rest = (m.group(2) or "").strip("/")
    return "%s:\\%s" % (m.group(1).upper(), rest.replace("/", "\\"))


def drive_of(path: str):
    """'/mnt/x/a/b' -> '/mnt/x'; None for a path outside /mnt/<letter>."""
    m = _WSL_MNT.match(path)
    return "/mnt/%s" % m.group(1) if m else None


# --------------------------------------------------------------------------- parsing


def _value_problem(key: str, value: str):
    """Why value is not acceptable for key, or None."""
    kind = KEYS[key][0]
    if _PLACEHOLDER_RE.search(value):
        return "still holds a placeholder (%s); put this machine's value in" % _PLACEHOLDER_RE.search(value).group(0)
    if value != value.strip():
        return "has leading or trailing white space"
    if not _VALUE_RE.fullmatch(value):
        return "holds a character outside letters, digits, space and _ . , + ( ) ~ / -"
    if kind == "name":
        return None if _NAME_RE.fullmatch(value) else "is not a WSL distro name ([A-Za-z0-9._-], no spaces)"
    if not value.startswith("/"):
        return "is not an absolute WSL path (write /mnt/<drive>/..., not a Windows path)"
    if os.path.normpath(value) != value or value.startswith("//"):
        return "is not a normalised path (no trailing '/', '//', '.' or '..'); write %s" % os.path.normpath(value)
    if kind == "drive":
        return None if _DRIVE_RE.fullmatch(value) else "is not a drive mount /mnt/<lower-case letter>"
    if kind in ("drvfs", "steam"):
        if not re.match(r"^/mnt/[a-z]/.", value):
            return "is not a path on a Windows drive (/mnt/<lower-case letter>/...)"
    if kind == "steam":
        parts = value.split("/")
        if len(parts) < 6 or parts[-2] != "common" or parts[-3].lower() != "steamapps":
            return "is not a Steam install dir <library>/steamapps/common/<install dir>"
    if key == "DATA_ROOT" and not _ROOT_RE.fullmatch(value):
        return ("holds a character outside letters, digits and _ . / - (the Windows launchers pass "
                "<DATA_ROOT>/tools/pin/preflight.py unquoted to wsl.exe)")
    return None


def parse(text: str, path: str = "<text>") -> "Config":
    """Parse drivedit.local.env text. LocalConfigError lists every problem (line numbers included)."""
    values, problems = {}, []
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip() or line.startswith("#"):
            continue
        if "=" not in line:
            problems.append("line %d: expected KEY=VALUE" % n)
            continue
        key, value = line.split("=", 1)
        if not _KEY_RE.fullmatch(key):
            problems.append("line %d: %r is not a key (upper-case letters, digits, _; no spaces around '=')" % (n, key))
            continue
        if key not in KEYS:
            problems.append("line %d: unknown key %s (known: %s)" % (n, key, ", ".join(KEYS)))
            continue
        if key in values:
            problems.append("line %d: %s is set twice" % (n, key))
            continue
        why = _value_problem(key, value)
        if why:
            problems.append("line %d: %s %s" % (n, key, why))
            continue
        values[key] = value
    if problems:
        raise LocalConfigError("local config %s is malformed:\n  %s" % (path, "\n  ".join(problems)))
    return Config(path, values)


class Config:
    """A parsed drivedit.local.env. get() and the derived forms raise LocalConfigError naming a missing key."""

    def __init__(self, path: str, values: dict):
        self.path = path
        self.values = dict(values)

    def has(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str) -> str:
        """The value of key (WSL form for a path)."""
        if key not in KEYS:
            raise LocalConfigError("%s is not a local config key (known: %s)" % (key, ", ".join(KEYS)))
        if key not in self.values:
            raise LocalConfigError("local config %s has no %s (%s); add a line %s=... (drivedit.local.env.example "
                                   "lists every key)" % (self.path, key, KEYS[key][1], key))
        return self.values[key]

    def win(self, key: str) -> str:
        """The Windows form of a path key (X:\\...)."""
        value = self.get(key)
        win = wsl_to_win(value)
        if win is None:
            raise LocalConfigError("local config %s: %s=%s is not on a Windows drive, so it has no Windows form"
                                   % (self.path, key, value))
        return win

    def drive(self, key: str) -> str:
        """The drive mount (/mnt/<x>) of a path key."""
        value = self.get(key)
        d = drive_of(value)
        if d is None:
            raise LocalConfigError("local config %s: %s=%s is not on a Windows drive" % (self.path, key, value))
        return d

    def drive_win(self, key: str) -> str:
        """The Windows drive (X:) of a path key."""
        return self.drive(key)[5:].upper() + ":"


# --------------------------------------------------------------------------- finding and loading


def candidates(environ=None) -> list:
    """[(where, path)] in lookup order; the env var only when it is set."""
    environ = os.environ if environ is None else environ
    out = []
    env = environ.get(ENV_VAR, "")
    if env.strip():
        out.append(("$" + ENV_VAR, win_to_wsl(env.strip())))
    parent = os.path.dirname(TOOLS_DIR)
    out.append(("<tools dir>/../config", os.path.join(parent, "config", CONFIG_NAME)))
    if os.path.exists(os.path.join(parent, ".git")):
        out.append(("repo root", os.path.join(parent, CONFIG_NAME)))
    return out


def find(environ=None) -> str:
    """The path of the config in use. LocalConfigError (naming every place looked at) when there is none. A set
    $DRIVEDIT_LOCAL_CONFIG that names no file is an error, never a reason to look further."""
    cands = candidates(environ)
    if cands[0][0].startswith("$"):
        path = cands[0][1]
        if not os.path.isfile(path):
            raise LocalConfigError("$%s names %s, which is not a file" % (ENV_VAR, path))
        return path
    for _, path in cands:
        if os.path.isfile(path):
            return path
    raise LocalConfigError(
        "no local config (%s). DriveDiT keeps this machine's paths in an untracked file; looked for: %s. Set %s to "
        "the file, or create it from drivedit.local.env.example (tools/migrate_local_config.py writes it from a "
        "machine's pre-DD-22 values)" % (CONFIG_NAME, "; ".join("%s (%s)" % (p, w) for w, p in cands), ENV_VAR))


_CACHE = {}
_OVERRIDE = []


class override:
    """Context manager: load() returns cfg (a Config) inside the block, whatever the lookup would find. For tools
    that check a config before it is written (tools/migrate_local_config.py) and for tests; never nested across
    threads."""

    def __init__(self, cfg: "Config"):
        self.cfg = cfg

    def __enter__(self):
        _OVERRIDE.append(self.cfg)
        return self.cfg

    def __exit__(self, *exc):
        _OVERRIDE.pop()
        return False


def load(path: str = None) -> Config:
    """The parsed config at path (default: find()). Cached; re-read when the file's size or mtime changes. Inside an
    override block, load() without a path returns the override."""
    if path is None and _OVERRIDE:
        return _OVERRIDE[-1]
    path = find() if path is None else win_to_wsl(path)
    try:
        st = os.stat(path)
        key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
        hit = _CACHE.get(path)
        if hit and hit[0] == key:
            return hit[1]
        if st.st_size > CONFIG_MAX_BYTES:
            raise LocalConfigError("local config %s is larger than %d bytes" % (path, CONFIG_MAX_BYTES))
        with open(path, "rb") as fh:
            data = fh.read(CONFIG_MAX_BYTES + 1)
    except OSError as exc:
        raise LocalConfigError("cannot read local config %s: %s" % (path, exc))
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise LocalConfigError("local config %s is not plain ASCII" % path)
    cfg = parse(text, path)
    _CACHE[path] = (key, cfg)
    return cfg


def get(key: str) -> str:
    return load().get(key)


def win(key: str) -> str:
    return load().win(key)


# --------------------------------------------------------------------------- simple write guards


SCRATCH_ROOTS = ("/dev/shm/",)   # always writable: RAM, never a drive
NEVER_ROOTS = ("/tmp/",)         # never: the WSL disk image


def simple_write_roots():
    """(allowed, forbidden, why) prefixes of the plain output guards (probes, routes, ident, test fixtures): allowed
    /dev/shm/ and the data drive of DATA_ROOT (/mnt/x/), forbidden /tmp/ and SYSTEM_DRIVE. Without a usable config
    only /dev/shm/ is allowed and why says what is missing; nothing is guessed."""
    try:
        cfg = load()
        return ((cfg.drive("DATA_ROOT") + "/",) + SCRATCH_ROOTS, (cfg.get("SYSTEM_DRIVE") + "/",) + NEVER_ROOTS,
                None)
    except LocalConfigError as exc:
        return SCRATCH_ROOTS, NEVER_ROOTS, str(exc)


def simple_write_refusal(path: str):
    """None when a plain output guard may write to path (resolved), else the refusal message."""
    allowed, forbidden, why = simple_write_roots()
    real = os.path.realpath(os.path.abspath(path)) if path else ""
    probe = real if real.endswith("/") else real + "/"
    if path and not any(probe.startswith(r) for r in forbidden) and any(probe.startswith(r) for r in allowed):
        return None
    return "refusing output %r: only under %s (the data drive of the local config, or RAM), never %s%s" % (
        path, " or ".join(r + "..." for r in allowed), " or ".join(forbidden),
        "; no data drive is known: %s" % why if why else "")


# --------------------------------------------------------------------------- launcher emulation


def bat_values(cfg: Config) -> dict:
    """What tools/launch/dd_local_config.cmd sets for this config (its /print output), computed in Python."""
    return {"DD_LOCAL_CONFIG": wsl_to_win(os.path.abspath(cfg.path)) or cfg.path,
            "DD_WSL_DISTRO": cfg.get("WSL_DISTRO"), "DD_DATA_ROOT": cfg.get("DATA_ROOT"),
            "DD_DATA_ROOT_WIN": cfg.win("DATA_ROOT"), "DD_DATA_DRIVE_WIN": cfg.drive_win("DATA_ROOT"),
            "DD_GAME_INSTALL_WIN": cfg.win("GAME_INSTALL"), "DD_TOOLS_INSTALL_WIN": cfg.win("TOOLS_INSTALL")}


_BAT_SET_RE = re.compile(r'^set "([A-Za-z_][A-Za-z0-9_]*)=([^"]*)"$')
_BAT_VAR_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")


def bat_env(text: str, seed=None) -> dict:
    """The variables a launcher's plain `set "NAME=VALUE"` lines give, in file order, each %NAME% expanded from the
    lines before it and from seed (bat_values: what dd_local_config.cmd sets). A %NAME% nothing sets stays as it is.
    A static reading for the tests and the migration check; it never runs cmd.exe."""
    env = dict(seed or {})
    for line in text.replace("\r\n", "\n").split("\n"):
        m = _BAT_SET_RE.match(line)
        if m:
            env[m.group(1)] = _BAT_VAR_RE.sub(lambda mm: env.get(mm.group(1), mm.group(0)), m.group(2))
    return env


def bat_start_line(text: str, seed=None) -> str:
    """A launcher's one `start ""` line with every variable expanded (bat_env); the launch id becomes <LID>."""
    env = bat_env(text, seed)
    starts = [line for line in text.replace("\r\n", "\n").split("\n") if line.startswith('start "" ')]
    if len(starts) != 1:
        raise LocalConfigError("expected exactly one start line, found %d" % len(starts))
    return _BAT_VAR_RE.sub(lambda m: "<LID>" if m.group(1) == "LID" else env.get(m.group(1), m.group(0)), starts[0])


def render(template: str, cfg: Config) -> str:
    """template with every <KEY> (WSL form) and <KEY_WIN> (Windows form) of a config key replaced; other <...> stay."""
    def sub(m):
        name = m.group(1)
        if name in KEYS:
            return cfg.get(name)
        if name.endswith("_WIN") and name[:-4] in KEYS:
            return cfg.win(name[:-4])
        return m.group(0)
    return re.sub(r"<([A-Z][A-Z0-9_]*)>", sub, template)


# --------------------------------------------------------------------------- check


def _wslpath(flag: str, path: str):
    exe = shutil.which("wslpath")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, flag, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.decode("utf-8", "replace").strip() if out.returncode == 0 else None


def _cmd_values(cfg: Config):
    """(dict, error) from running dd_local_config.cmd /print through cmd.exe for cfg.path (reads only)."""
    helper = os.path.join(TOOLS_DIR, "launch", "dd_local_config.cmd")
    cmd = shutil.which("cmd.exe")
    helper_win = wsl_to_win(helper) or _wslpath("-w", helper)
    cfg_win = wsl_to_win(os.path.abspath(cfg.path)) or _wslpath("-w", os.path.abspath(cfg.path))
    if not cmd or not helper_win or not cfg_win:
        return None, "cmd.exe, or a Windows path for %s and %s, is not available here" % (helper, cfg.path), None
    env = dict(os.environ, DRIVEDIT_LOCAL_CONFIG=cfg_win)
    env["WSLENV"] = ":".join(p for p in (os.environ.get("WSLENV", ""), ENV_VAR) if p)
    try:
        out = subprocess.run([cmd, "/d", "/c", helper_win, "/print"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, cwd=os.path.dirname(helper), timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "cmd.exe failed: %s" % exc, None
    text = out.stdout.decode("utf-8", "replace")
    vals = dict(line.split("=", 1) for line in text.replace("\r", "").splitlines() if line.startswith("DD_") and "=" in line)
    if out.returncode != 0:
        return None, "dd_local_config.cmd exited %d: %s" % (out.returncode, text.strip()), None
    return vals, None, cfg_win


LAUNCHERS = ("wb_R1.bat", "client_R1_i01.bat", "wb_C18.bat", "client_C18_i01.bat")


def _echo_copy(text: str):
    """An echo-only copy of a launcher: its lines up to the first `set "DD_ERR="` (the helper call and the set lines),
    then the wsl.exe preflight command and the start line, each printed instead of run. None if the shape differs."""
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        end = lines.index('set "DD_ERR="')
        start = next(line for line in lines if line.startswith('start "" '))
        loop = next(line for line in lines if line.startswith("for /f") and "wsl.exe -d" in line)
    except (ValueError, StopIteration):
        return None
    m = re.search(r"\('(wsl\.exe .*?) \^\|", loop)
    if not m:
        return None
    body = lines[:end + 1] + ["echo WSL: " + m.group(1), "echo " + start.replace("%LID%", "^<LID^>"), "endlocal",
                              "exit /b 0", ":fail", "echo FAIL %DD_ERR%", "exit /b 1", ""]
    return "\r\n".join(body), start, m.group(1)


def _cmd_launchers(cfg: Config, cfg_win: str) -> list:
    """[(name, ok, detail)]: each launcher's helper call, set lines, wsl.exe command and start line evaluated by cmd.exe
    (echo-only copies in a temporary directory under /dev/shm, removed afterwards; nothing is started) against the
    Python reading of the same launcher (bat_env, bat_start_line)."""
    import tempfile
    out = []
    if not os.path.isdir("/dev/shm"):
        return [("launchers", False, "no /dev/shm for the echo copies")]
    tmp = tempfile.mkdtemp(prefix="dd_launch_echo_", dir="/dev/shm")
    try:
        shutil.copyfile(os.path.join(TOOLS_DIR, "launch", "dd_local_config.cmd"), os.path.join(tmp, "dd_local_config.cmd"))
        seed = bat_values(cfg)
        for name in LAUNCHERS:
            with open(os.path.join(TOOLS_DIR, "launch", name), "rb") as fh:
                text = fh.read().decode("ascii")
            made = _echo_copy(text)
            if made is None:
                out.append((name, False, "not a launcher of the expected shape"))
                continue
            body, _, wsl = made
            copy = os.path.join(tmp, name.replace(".bat", "_echo.cmd"))
            with open(copy, "wb") as fh:
                fh.write(body.encode("ascii"))
            env = dict(os.environ, DRIVEDIT_LOCAL_CONFIG=cfg_win)
            env["WSLENV"] = ":".join(p for p in (os.environ.get("WSLENV", ""), ENV_VAR) if p)
            r = subprocess.run([shutil.which("cmd.exe"), "/d", "/c", _wslpath("-w", copy) or copy],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=TOOLS_DIR, timeout=60)
            got = r.stdout.decode("utf-8", "replace").replace("\r", "").splitlines()
            env_py = bat_env(text, seed)
            want_wsl = _BAT_VAR_RE.sub(lambda m: env_py.get(m.group(1), m.group(0)), wsl)
            want = ["WSL: " + want_wsl, bat_start_line(text, seed)]
            ok = r.returncode == 0 and got == want
            out.append((name, ok, "cmd.exe gives the same wsl.exe command and start line" if ok else
                        "cmd.exe: %r; python: %r" % (got, want)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def check(cfg: Config, with_cmd: bool = False) -> list:
    """[(status, name, detail)] with status PASS, FAIL or SKIP. Reads only."""
    res = []

    def add(ok, name, detail):
        res.append(("PASS" if ok is True else "SKIP" if ok is None else "FAIL", name, detail))

    for key in KEYS:
        add(cfg.has(key), "key." + key, cfg.values.get(key, "missing (%s)" % KEYS[key][1]))
    distro_now = os.environ.get("WSL_DISTRO_NAME")
    if cfg.has("WSL_DISTRO"):
        add(None if not distro_now else distro_now == cfg.get("WSL_DISTRO"), "wsl_distro.current",
            "this shell runs in %s" % distro_now if distro_now else "not running in WSL; cannot compare")
    for key, (kind, _) in KEYS.items():
        if kind not in PATH_KINDS or not cfg.has(key):
            continue
        value = cfg.get(key)
        if kind == "drive":
            add(os.path.ismount(value), "exists." + key, "%s is %sa mount point" % (
                value, "" if os.path.ismount(value) else "NOT "))
        else:
            add(os.path.isdir(value), "exists." + key, "%s %s" % (value, "is a directory" if os.path.isdir(value)
                                                                    else "is NOT a directory"))
        if kind == "steam":
            lib = os.path.dirname(os.path.dirname(value))
            acfs = [n for n in (os.listdir(lib) if os.path.isdir(lib) else [])
                    if n.startswith("appmanifest_") and n.endswith(".acf")]
            add(bool(acfs), "appmanifests." + key, "%s holds %d appmanifest_*.acf" % (lib, len(acfs)))
        if kind != "path" or drive_of(value):
            want = cfg.win(key) if kind != "drive" else cfg.drive_win(key) + "\\"
            got = _wslpath("-w", value)
            back = _wslpath("-u", want)
            if back is not None and kind == "drive":
                back = back.rstrip("/")
            if got is None or back is None:
                add(None, "windows_form." + key, "wslpath is not available here; the pure mapping gives %s" % want)
            else:
                add(got == want and back == value, "windows_form." + key,
                    "mapping %s, wslpath -w %s, wslpath -u of the mapping %s" % (want, got, back))
    if cfg.has("DATA_ROOT") and cfg.has("SYSTEM_DRIVE"):
        add(cfg.drive("DATA_ROOT") != cfg.get("SYSTEM_DRIVE"), "data_root.off_system_drive",
            "DATA_ROOT is on %s, the system drive is %s" % (cfg.drive("DATA_ROOT"), cfg.get("SYSTEM_DRIVE")))
    if with_cmd:
        try:
            want = bat_values(cfg)
        except LocalConfigError as exc:
            add(False, "batch_helper", str(exc))
        else:
            got, err, cfg_win = _cmd_values(cfg)
            if cfg_win:
                want["DD_LOCAL_CONFIG"] = cfg_win
            if err:
                add(False, "batch_helper", err)
            else:
                # The file path may differ in letter case only (cmd.exe echoes what it was given).
                diff = ["%s: cmd %r, python %r" % (k, got.get(k), v) for k, v in want.items()
                        if (got.get(k) or "").lower() != v.lower() or (k != "DD_LOCAL_CONFIG" and got.get(k) != v)]
                add(not diff, "batch_helper", "; ".join(diff) or "dd_local_config.cmd derives the same %d values"
                    % len(want))
                for name, ok, detail in _cmd_launchers(cfg, cfg_win):
                    add(ok, "launcher." + name, detail)
    return res


# --------------------------------------------------------------------------- command line


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DriveDiT local machine paths (DD-22): find, show and check %s."
                                 % CONFIG_NAME)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("where", help="print the config file in use")
    sub.add_parser("show", help="print every key with its WSL and Windows form")
    g = sub.add_parser("get", help="print one value (WSL form unless a flag says otherwise)")
    g.add_argument("key")
    form = g.add_mutually_exclusive_group()
    form.add_argument("--win", action="store_true", help="the Windows form X:\\...")
    form.add_argument("--drive", action="store_true", help="the drive mount /mnt/<x> of the path")
    form.add_argument("--drive-win", action="store_true", help="the Windows drive X: of the path")
    r = sub.add_parser("render", help="print a file with every <KEY> and <KEY_WIN> of a config key filled in")
    r.add_argument("file")
    c = sub.add_parser("check", help="validate the config on this machine (reads only)")
    c.add_argument("--cmd", dest="with_cmd", action="store_true",
                   help="also run tools/launch/dd_local_config.cmd /print through cmd.exe and compare")
    for p in sub.choices.values():
        p.add_argument("--config", default=None, help="this file instead of the lookup order")
    args = ap.parse_args(argv)
    try:
        if args.command == "where":
            print(load(args.config).path if args.config else find())
            return 0
        cfg = load(args.config)
        if args.command == "render":
            with open(args.file, encoding="utf-8") as fh:
                sys.stdout.write(render(fh.read(), cfg))
            return 0
        if args.command == "get":
            fn = cfg.win if args.win else cfg.drive if args.drive else cfg.drive_win if args.drive_win else cfg.get
            print(fn(args.key))
            return 0
        if args.command == "show":
            print("# %s" % cfg.path)
            for key, (kind, _) in KEYS.items():
                if not cfg.has(key):
                    print("%-27s <missing>" % key)
                elif kind in ("drvfs", "steam") or (kind == "path" and drive_of(cfg.get(key))):
                    print("%-27s %s  |  %s" % (key, cfg.get(key), cfg.win(key)))
                else:
                    print("%-27s %s" % (key, cfg.get(key)))
            return 0
        print("local config %s" % cfg.path)
        results = check(cfg, with_cmd=args.with_cmd)
    except LocalConfigError as exc:
        print("local_config: %s" % exc, file=sys.stderr)
        return 2
    for status, name, detail in results:
        print("%-4s %-36s %s" % (status, name, detail))
    count = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "FAIL", "SKIP")}
    print("%d check(s): %d PASS, %d FAIL, %d SKIP" % (len(results), count["PASS"], count["FAIL"], count["SKIP"]))
    return 1 if count["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
