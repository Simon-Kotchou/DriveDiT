#!/usr/bin/env python3
"""harvest: the minimal session harvester of capture contract dc-2.0 (DD-6-min; CAPTURE_PROGRAM r2 TKT-6-min,
runbook step R4).

    python3 -I -B tools/harvester/harvest.py [--session DIR ...] [--dry-run] [--routes-dir DIR]
        [--allow-mixed-builds --reason TEXT] [--log-before-s S] [--log-after-s S] [--screenshot-suffix S]

Validate, decode fiducials, build the derivative, move: Arland sessions of the Workbench and packaged-client profiles
only. Kolguyev and Everon routing, crash salvage, frozen manifests and the loader guards are DD-6.

Where it looks (every root lies under /mnt/e/DriveDiT, and no path below it may pass through a symlink; nothing
outside it is read for sessions or written):
  profiles  wb_profile/, wb_profile_x19/ (the 1.9 Experimental Workbench, MIGRATION_1_9 3.4) and
            client_profiles/iNN/. $profile is <profile>/profile, so sessions are <profile>/profile/DC_Sessions/<id>/
            (dc-2.0 4.2). The dc-1.0 $profile:DrivingData/ and DD-4-probe's $profile:DC_Probes/ are never read.
  launch    <profile>/profile/dc_launch/launch_<launch_id>.json (the copy the launch guard read, dc-2.0 5.1) and
            pin/launches/launch_<launch_id>.json (DD-0's preflight record). When both exist they must be equal.
  logs      <profile>/logs/logs_*/ and, for client iNN, logs/iNN/ (the -logsDir of client_R1_iNN.bat).
  pins      pin/pin_<pin_id>.json, pin/pin_<pin_id>.build_key.json, pin/<pin_id>/SHA256SUMS.

Per session, in this order:
  1. Admission, read-only. A directory that is not named arland_*, or whose session.json map is not Arland (map_id,
     terrain, an Eden-rooted root_world, session_id), is refused with one line and nothing else: no report, no log
     value, no file, and not its directory name (Everon is the sealed test map; Kolguyev routing is DD-6). An
     unsealed session (no session_end.json) is skipped. pin_id, build_key and use_class are read here, before
     validation. A -exp pin (or pin.channel experimental) routes the session to compat19/<pin_id>/, never to
     captures/ (MIGRATION_1_9 3.4); a session of wb_profile_x19 with a stable pin is held. rig.derivative_wh and
     rig.derivative_codec must be the contract's [512, 288] and jpeg_q95. The engine-log slice (below) is cut here
     too, so a session whose logs cannot be attributed is held before anything is written.
  2. Sidecars (the DD-5-min conventions): _launch/launch_<launch_id>.json, _engine_logs/*.log (each engine log
     sliced to the session's time window) and harvest.json (the build stamp; build_key of a dd-launch/1 session,
     dc-2.0 10.1). dc_validate reads all three from the session directory.
  3. dc_validate, run as `tools/validator/dc_validate.py --session DIR --no-report --json --image-suffix S`. Exit 0
     or 2 routes the session to captures/arland/<YYYY-MM-DD>/<trial_id>/<session_id>/ and exit 1 to
     captures/_rejected/<session_id>/. Exit 3 (a refusal or a validator error) holds it in place and removes the
     sidecars again, except that a compat19 session refused for want of a thresholds file (none exists for an
     experimental pin) goes to compat19.
  4. _harvest/: dc_validate.json (the report), img_decoded.csv (img_resolved of dc-2.0 9.8, one row per img row),
     deriv/f_<slot:08d>.jpg, result.json and SHA256SUMS. A PASS (or PASS_WITH_WARNINGS) session gets one derivative
     per fiducial-decoded slot: the centred 16:9 of the core (the 1280x720 core of a 1408x720 client image, the
     centre 16:9 of a Workbench viewport without its bands), resized to 512x288 with OpenCV and written as JPEG
     quality 95. SHA256SUMS lists every file of the session (sha256sum format). All of it is built in
     _harvest.partial/ and renamed to _harvest/ when complete; SHA256SUMS is then read back and verified.
  5. Move: one rename of the session directory, on the same volume only. An existing destination is never touched.
  6. Catalog: one row in catalog/sessions.jsonl (append-only), with pin_id, build_key and use_class columns and the
     run_id of the run that catalogued it.

Raw data is never modified or deleted. Inside a session the harvester writes only its own names (harvest.json,
_launch/, _engine_logs/, _harvest/, _harvest.partial/ and *.dd6tmp), never through a symlink, and the move is a
rename. Deleting raw PNGs after derivative + validation + checksums (the retention rule) is not done here.

Re-runs and kills. A run that finds nothing to do writes nothing. A session held at admission or by a validator
refusal is left as it was found; one held later keeps only harvester-owned names, which the next run replaces. A
complete _harvest/ of this harvester version whose SHA256SUMS verifies, whose harvest.json carries the build stamp
admission found (pin_id, build_key, use_class, channel, screenshot suffix) and whose result.json names this session
and the destination its verdict routes to is moved as it is; any other harvester output is rebuilt. A kill at any
point leaves the whole session either at its source or at its destination. A session that was moved but not
catalogued is catalogued by the next run, after the same map check, a check of its harvest.json and result.json and
a full SHA256SUMS verification at its destination. A catalog row counts only with its line feed: under the lock, an
unterminated last line that is a complete row gets its line feed back, and a torn one (a killed append) is cut off;
complete rows are never changed.

One build per run (MIGRATION_1_9 3.3). The sessions a run will catalogue outside compat19 must share one build_key.
Otherwise the run is refused, unless --allow-mixed-builds --reason TEXT is given (the reason goes into the catalog
rows). manifests/harvest_runs/harvest_<run_id>.json lists the captures/arland sessions of one run_id, with their
build_key set and the reason; it is written from the catalog, so a run killed before its manifest gets it written by
the next run.

build_key (dc-2.0 11) comes from session.json pin.build_key, the launch record (dd-launch/2), the pin (dd-pin/2) or
DD-14a's sidecar pin_<pin_id>.build_key.json. When none of these exists (a dd-launch/1 session on a dd-pin/1 pin),
build_key/1 is computed from the pin and its copy's SHA256SUMS (build_key_v1). All sources found must agree. A
session with no source is held.

Engine logs (C20 is HARD from T0, so a session whose logs cannot be attributed is held, never passed). The run is
the newest logs_* run whose console.log header says it started before the session, and it must name the session.
A run that loaded an Eden-rooted world or names an everon_ or kolguyev_ session is not used at all: another map's
Play ran in that process, so the session is held and no line is copied (dc-2.0 3). Each *.log is sliced to
[start.unix_s - --log-before-s, end.unix_s + --log-after-s]; the window starts at the session's own DD-4-min START
marker when the log has one, and it is cut at the nearest line at or before the start, or at or after the end,
that names another session id, so a neighbouring Play's lines stay out. The slice must name the session. A
headerless crash.log (written by a dying process) is attributed by its modification time: kept whole when that lies
in the window, else left out and listed in harvest.json; any other headerless log holds the session.

TBD-by-probe. TBD-4: --screenshot-suffix (default the reader's IMAGE_SUFFIX, .png) is the suffix the engine appends
to the img request path, one of the names the probe scheduler polls (DC_ProbeImages.c m_aSuffixes, the bare name
included; the reader's IMAGE_SUFFIXES). It goes to dc_validate (--image-suffix), the derivative pass and
img_decoded.csv's file column. A file in frames/ named f_<slot:08d> plus another of those suffixes holds the
session, and the message names the suffix: the reader would not see it (any other stray file is C11's). TBD-6: in
both launch-guard modes (cli, wb_escape) the launch record is found by pin.launch_id. The other TBDs are read from
the session by dc_validate.

Exit codes: 0 every session found was harvested, or there was nothing to do; 1 a session went to captures/_rejected;
2 a session was held, refused or (with --session) skipped and stays in place (2 wins over 1); 3 the run was refused
(arguments, roots, symlinked catalog files, mixed builds, another harvester running) and nothing was written; 4 the
run stopped on an unexpected error (every session is still whole at its source or its destination; the next run
carries on).

--test-root DIR (tests only) replaces /mnt/e/DriveDiT by DIR: an existing directory under /dev/shm/ or /mnt/e/, never
on C: and never overlapping /mnt/e/DriveDiT. Only then does DD6_HARVEST_KILL_AT=<hook> kill the run at that hook
(HOOKS), for the kill-mid-run test.
"""

import argparse
import calendar
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import traceback

import numpy

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
READER_PATH = os.path.join(REPO, "data", "dc2", "reader.py")
DC_VALIDATE = os.path.join(REPO, "tools", "validator", "dc_validate.py")
DEFAULT_ROUTES_DIR = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes")

HARVESTER_VERSION = "harvest/2 (DD-6-min)"
HARVEST_SCHEMA = "drivedit.harvest/1"
RESULT_SCHEMA = "drivedit.harvest-result/1"
CATALOG_SCHEMA = "drivedit.catalog-session/1"
RUN_SCHEMA = "drivedit.harvest-run/1"

DATA_ROOT = "/mnt/e/DriveDiT"
TEST_ROOT_PREFIXES = ("/dev/shm/", "/mnt/e/")
# Paths whose device never receives a write: the C: drive, and / (the WSL disk image, which lives on C:).
FORBIDDEN_DEVICE_PATHS = (("/mnt/c", "the C: drive"), ("/", "the WSL disk image, which lives on the C: drive"))

WB_PROFILES = ("wb_profile", "wb_profile_x19")
X19_PROFILE = "wb_profile_x19"      # 1.9 Experimental Workbench: never harvested into captures/ (MIGRATION_1_9 3.4)
CLIENT_PROFILES = "client_profiles"
CLIENT_INSTANCE_RE = re.compile(r"^i[0-9]{2}$")
PROFILE_DIR = "profile"             # $profile of a -profile root
SESSIONS_DIR = "DC_Sessions"        # $profile:DC_Sessions/<session_id>/ (dc-2.0 4.2)
LAUNCH_COPY_DIR = "dc_launch"       # $profile:dc_launch/launch_<id>.json (dc-2.0 5.1)
CLIENT_LOGS_DIR = "logs"            # <root>/logs/iNN (client_R1_iNN.bat -logsDir)

CAPTURES = "captures"
REJECTED = "_rejected"
COMPAT19 = "compat19"
CATALOG_DIR = "catalog"
CATALOG_NAME = "sessions.jsonl"
LOCK_NAME = "harvest.lock"
RUN_MANIFESTS = "manifests/harvest_runs"

# Harvester-owned names inside a session directory: the only entries it ever writes, replaces or removes there.
HARVEST_JSON = "harvest.json"
LAUNCH_SIDECAR = "_launch"
LOGS_SIDECAR = "_engine_logs"
HARVEST_DIR = "_harvest"
PARTIAL_DIR = "_harvest.partial"
TMP_SUFFIX = ".dd6tmp"
OWN_NAMES = (HARVEST_JSON, LAUNCH_SIDECAR, LOGS_SIDECAR, HARVEST_DIR, PARTIAL_DIR)
SUMS_NAME = "SHA256SUMS"

# The derivative of dc-2.0 10.1 (rig.derivative_wh, rig.derivative_codec) and CAPTURE_PROGRAM r2 3.9: 512x288 JPEG
# q95. A session that states anything else is held; a new value is a contract change.
DERIVATIVE_WH = [512, 288]
DERIVATIVE_CODEC = "jpeg_q95"
DERIVATIVE_QUALITY = 95

PIN_ID_RE = re.compile(r"^[0-9]+(\.[0-9]+){1,3}(-[0-9A-Za-z]+)*$")      # as dc_validate (and the pin file name)
LAUNCH_ID_RE = re.compile(r"^[0-9A-Za-z-]{20,64}$")                     # dc-2.0 5.1
TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,16}$")
BUILD_KEY_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._:-]{7,127}$")
RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{4}$")
SUM_LINE_RE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
FRAME_NAME_RE = re.compile(r"^f_[0-9]{8}(?=\.|$)")
MAX_JSON_BYTES = 4 << 20

# The executables the launchers start (client_R1_iNN.bat GAME_EXE, wb_R1.bat WB_EXE), relative to the install dir
# (DD-0 pinlib BINARIES); build_key/1 hashes them with the app ids and buildids (MIGRATION_1_9 2.2).
BUILD_KEY_EXES = (("game", "ArmaReforgerSteam.exe"), ("tools", "Workbench/ArmaReforgerWorkbenchSteamDiag.exe"))

LOG_HEADER_RE = re.compile(rb"^Log .* started at ([0-9]{4})-([0-9]{2})-([0-9]{2}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) "
                           rb"\(([0-9]{4})-([0-9]{2})-([0-9]{2}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) UTC\)")
LOG_TIME_RE = re.compile(rb"^([0-9]{2}):([0-9]{2}):([0-9]{2})\.([0-9]{3})")
LOG_HEADER_SCAN = 8
LOG_RUN_SLACK_S = 2.0               # a log run may start up to this long after the session's start second
LOG_CUT_S = 0.0005                  # half a log millisecond: a cut line itself stays outside the window
DEFAULT_LOG_BEFORE_S = 2.0          # before session.json start.unix_s (a whole second): the session's first lines
DEFAULT_LOG_AFTER_S = 15.0          # after session_end.json end.unix_s: the shutdown of the capture components
# A log run that names an Eden-rooted world (dc-2.0 3.1: every one counts as Everon) or a session id of another map
# had another map's Play in its process: none of its lines is copied into an Arland session.
LOG_EDEN_WORLD_RE = re.compile(rb"worlds[/\\]eden[/\\]", re.IGNORECASE)
LOG_OTHER_MAP_IDS = (b"everon_", b"kolguyev_")
# DD-4-min DC_Session.OpenSession prints this right after it takes start.unix_s ("[DC_Session] START session <id>
# dir=..."): when present, the session's window starts at it, so a previous Play's last lines stay out.
LOG_START_MARKER = b"[DC_Session] START session %s "
CRASH_LOG = "crash.log"
VALIDATE_TIMEOUT_S = 3 * 3600
NO_THRESHOLDS_REFUSAL = "refused: no thresholds file for pin "     # dc_validate.thresholds_path, exit 3

EXIT_OK, EXIT_REJECTED, EXIT_HELD, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2, 3, 4
VERDICT_ROUTE = {"PASS": "captures", "PASS_WITH_WARNINGS": "captures", "FAIL": "rejected"}
NOT_ARLAND = ("not an Arland session; DD-6-min harvests Arland only (Everon is sealed, Kolguyev routing is DD-6); "
              "nothing was read or written")
KILL_ENV = "DD6_HARVEST_KILL_AT"
HOOKS = ("after_sidecars", "after_validate", "in_derivatives", "before_harvest_rename", "after_harvest_rename",
         "after_move", "after_catalog")
CATALOG_COLUMNS = ("schema", "session_id", "map_id", "trial_id", "date", "exe", "instance", "pin_id", "build_key",
                   "build_key_source", "use_class", "channel", "verdict", "exit_code", "route", "dest", "images",
                   "decoded", "derivatives", "sha256sums_sha256", "harvester_version", "harvested_utc", "run_id",
                   "mixed_builds_reason")


class Refused(Exception):
    """The whole run is refused (exit 3) before anything is written."""


class Held(Exception):
    """One session stays where it is; the message says why."""


def _load_reader():
    spec = importlib.util.spec_from_file_location("dc2_reader", READER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = _load_reader()


# ------------------------------------------------------------------------------------------------
# Paths and small file helpers
# ------------------------------------------------------------------------------------------------
def inside(path, root):
    """True when path is root or lies below it (plain strings, no symlink resolution)."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def real_below(path, root):
    """True when path lies strictly below root (a realpath) and no component below root is a symlink."""
    norm = os.path.normpath(path)
    return norm != root and inside(norm, root) and os.path.realpath(norm) == norm


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_text(unix_s):
    return datetime.datetime.fromtimestamp(unix_s, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hidden(rel):
    """rel with its last component (the directory name of a refused, non-Arland session: its map, trial and start
    time) replaced by a placeholder holding a hash of it, so a refusal line names no session value."""
    head, name = os.path.split(rel)
    digest = sha256_bytes(name.encode("utf-8", "surrogateescape"))[:12]
    return "%s/<not Arland, name sha256 %s>" % (head, digest)


def json_text(obj):
    return (json.dumps(obj, indent=1) + "\n").encode("ascii")


def read_json(path):
    """The JSON object of a regular file (at most MAX_JSON_BYTES), or None when absent, a symlink, unreadable or not
    an object."""
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) > MAX_JSON_BYTES:
            return None
        with open(path, "rb") as f:
            obj = json.loads(f.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write")
        view = view[n:]


def write_file(path, data):
    """Write data to path (never through a symlink) and flush it to the disk."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_file(path, data):
    """Atomic replace: path.dd6tmp, then rename over path (a symlink at path is replaced, never followed)."""
    tmp = path + TMP_SUFFIX
    if os.path.lexists(tmp):
        os.unlink(tmp)
    write_file(tmp, data)
    os.replace(tmp, path)


def fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def get(obj, dotted):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def forbidden_device_reason(path):
    """Why path (or its nearest existing ancestor) is on the device of C: or of the WSL disk image, or None."""
    probe = path
    while not os.path.lexists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    try:
        dev = os.stat(probe).st_dev
    except OSError:
        return None
    for other, label in FORBIDDEN_DEVICE_PATHS:
        try:
            if os.path.isdir(other) and os.stat(other).st_dev == dev:
                return "%s is on the same device as %s (%s)" % (path, other, label)
        except OSError:
            pass
    return None


def data_root(test_root):
    """/mnt/e/DriveDiT (not reached through a symlink), or the --test-root (an existing directory under /dev/shm/ or
    /mnt/e/, never on C: and never overlapping /mnt/e/DriveDiT). Refused otherwise."""
    if test_root is None:
        if not os.path.isdir(DATA_ROOT) or os.path.realpath(DATA_ROOT) != DATA_ROOT:
            raise Refused("%s does not exist or is reached through a symlink" % DATA_ROOT)
        reason = forbidden_device_reason(DATA_ROOT)
        if reason:
            raise Refused(reason)
        return DATA_ROOT
    if re.match(r"^[A-Za-z]:", test_root) or test_root.startswith("\\\\"):
        raise Refused("--test-root %r uses Windows syntax; pass a /dev/shm/ or /mnt/e/ path" % test_root)
    real = os.path.realpath(os.path.abspath(test_root))
    if inside(real.lower(), "/mnt/c"):
        raise Refused("--test-root %s resolves to the C: drive" % test_root)
    if not any(real.startswith(prefix) and len(real) > len(prefix) for prefix in TEST_ROOT_PREFIXES):
        raise Refused("--test-root %s resolves to %s, which is not inside %s" % (
            test_root, real, " or ".join(TEST_ROOT_PREFIXES)))
    if inside(real.lower(), DATA_ROOT.lower()) or inside(DATA_ROOT.lower(), real.lower()):
        raise Refused("--test-root %s overlaps the production root %s" % (test_root, DATA_ROOT))
    reason = forbidden_device_reason(real)
    if reason:
        raise Refused("--test-root: %s" % reason)
    if not os.path.isdir(real):
        raise Refused("--test-root %s is not an existing directory" % test_root)
    return real


def ensure_dir(root, rel):
    """Create root/rel (a relative path below the data root) one component at a time, refusing any component that
    is a symlink or not a directory. Returns the absolute path."""
    cur = root
    for part in rel.split("/"):
        if part in ("", ".", ".."):
            raise Held("bad path component in %r" % rel)
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            raise Held("%s is a symlink; the harvester never writes through one" % cur)
        if not os.path.lexists(cur):
            os.mkdir(cur)
        elif not os.path.isdir(cur):
            raise Held("%s exists and is not a directory" % cur)
    return cur


def remove_own(session_dir, name):
    """Remove a harvester-owned entry of a session directory (OWN_NAMES or a *.dd6tmp name); never anything else.
    A symlink is unlinked, never followed."""
    if name not in OWN_NAMES and not name.endswith(TMP_SUFFIX):
        raise AssertionError("refusing to remove %s: not a harvester-owned name" % name)
    path = os.path.join(session_dir, name)
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def read_below(path, root, what):
    """Bytes of a regular file below root, or None when it does not exist; Held when it is reached through a symlink
    or is not a regular file."""
    if not os.path.lexists(path):
        return None
    if not real_below(path, root) or not os.path.isfile(path):
        raise Held("%s is reached through a symlink or is not a regular file" % what)
    with open(path, "rb") as f:
        return f.read()


# ------------------------------------------------------------------------------------------------
# Profiles and admission
# ------------------------------------------------------------------------------------------------
class Profile:
    """One -profile root: name (wb_profile, wb_profile_x19, client_profiles/iNN), its $profile directories and the
    engine-log directories of its processes."""

    def __init__(self, root, name, client_instance=None):
        self.name = name
        self.base = os.path.join(root, name)
        self.client_instance = client_instance
        self.experimental = name == X19_PROFILE
        self.sessions = os.path.join(self.base, PROFILE_DIR, SESSIONS_DIR)
        self.launch_dir = os.path.join(self.base, PROFILE_DIR, LAUNCH_COPY_DIR)
        self.log_bases = [os.path.join(self.base, "logs")]
        if client_instance:
            self.log_bases.insert(0, os.path.join(root, CLIENT_LOGS_DIR, client_instance))

    def accepts_exe(self, kind, instance):
        if self.client_instance:
            return kind in ("cl", "cld") and instance == self.client_instance
        return kind == "wb" and instance == "i00"


def list_profiles(root):
    """The profiles whose DC_Sessions folder exists below root without any symlink on the way."""
    profiles = [Profile(root, name) for name in WB_PROFILES]
    cdir = os.path.join(root, CLIENT_PROFILES)
    if os.path.isdir(cdir) and real_below(cdir, root):
        for inst in sorted(os.listdir(cdir)):
            if CLIENT_INSTANCE_RE.match(inst):
                profiles.append(Profile(root, "%s/%s" % (CLIENT_PROFILES, inst), inst))
    return [p for p in profiles if os.path.isdir(p.sessions) and real_below(p.sessions, root)]


def is_arland(meta):
    """True only for an Arland session.json: map_id arland, terrain Arland, a root_world that is not Eden-rooted
    and an arland_ session id (dc-2.0 3.1: every Eden-rooted world counts as Everon)."""
    root_world = get(meta, "map.root_world")
    eden = isinstance(root_world, str) and bool(reader.EDEN_ROOT_RE.match(root_world))
    sid = meta.get("session_id")
    return (get(meta, "map.map_id") == "arland" and get(meta, "map.terrain") == "Arland" and not eden
            and isinstance(sid, str) and sid.startswith("arland_"))


class Candidate:
    """A sealed Arland session admitted for harvest: its identity, build stamp, launch record, engine-log slice and
    route class."""

    def __init__(self, path, profile, meta, end):
        self.path = path
        self.profile = profile
        self.meta = meta
        self.end = end
        self.session_id = meta["session_id"]
        self.trial_id = get(meta, "trial.trial_id")
        self.pin_id = get(meta, "pin.pin_id")
        self.channel = get(meta, "pin.channel")
        self.use_class = meta.get("use_class")
        self.launch_id = get(meta, "pin.launch_id")
        self.experimental = "exp" in self.pin_id.split("-")[1:] or self.channel == "experimental"
        self.build_key = None
        self.build_key_source = None
        self.launch_bytes = None
        self.launch_source = None
        self.screenshot_suffix = reader.IMAGE_SUFFIX
        self.logs = None

    def date(self):
        stamp = self.session_id.split("_")[2]
        return "%s-%s-%s" % (stamp[0:4], stamp[4:6], stamp[6:8])

    def compat_dest(self):
        return "%s/%s/%s" % (COMPAT19, self.pin_id, self.session_id)

    def dest_for(self, verdict):
        if self.experimental:
            return "compat19", self.compat_dest()
        route = VERDICT_ROUTE[verdict]
        if route == "rejected":
            return route, "%s/%s/%s" % (CAPTURES, REJECTED, self.session_id)
        return route, "%s/arland/%s/%s/%s" % (CAPTURES, self.date(), self.trial_id, self.session_id)

    def route_of(self, verdict):
        """(route, dest) a verdict sends this session to, or None (a refusal of a stable-pin session is held)."""
        if not self.experimental and verdict not in VERDICT_ROUTE:
            return None
        return self.dest_for(verdict)

    def stamp(self):
        """What a reusable _harvest/ must still match: the build stamp admission found and the TBD-4 suffix."""
        return {"session_id": self.session_id, "pin_id": self.pin_id, "build_key": self.build_key,
                "use_class": self.use_class, "channel": self.channel, "screenshot_suffix": self.screenshot_suffix}


def admit(path, profile, schema):
    """('refused' | 'skipped' | 'held', reason) or ('ok', Candidate) for one directory of a DC_Sessions folder.
    Nothing is written. The map is judged first: a non-Arland directory name is refused without reading any file,
    and a non-Arland session.json without reading anything else (the refusal names no session value)."""
    name = os.path.basename(path)
    if not name.startswith("arland_"):
        return "refused", NOT_ARLAND
    if os.path.islink(path) or not os.path.isdir(path):
        return "held", "not a directory (a symlink or a file)"
    meta = read_json(os.path.join(path, "session.json"))
    if meta is None:
        return "held", "session.json is missing or unreadable, so its map cannot be checked"
    if not is_arland(meta):
        return "refused", ("not an Arland session (session.json map); DD-6-min harvests Arland only (Everon is "
                           "sealed, Kolguyev routing is DD-6); nothing else was read or written")
    if not os.path.isfile(os.path.join(path, "session_end.json")):
        return "skipped", "not sealed (no session_end.json); a killed session waits for DD-6 salvage"
    end = read_json(os.path.join(path, "session_end.json"))
    sid = meta.get("session_id")
    if sid != name or not schema.id_re.match(name):
        return "held", "directory name and session.json session_id differ or break the session-id rule"
    if meta.get("contract_version") != reader.CONTRACT_VERSION:
        return "held", "contract_version %r is not %s" % (meta.get("contract_version"), reader.CONTRACT_VERSION)
    trial = get(meta, "trial.trial_id")
    if not (isinstance(trial, str) and TRIAL_ID_RE.match(trial) and name.split("_")[1] == trial):
        return "held", "trial.trial_id %r is not the session id's trial" % (trial,)
    if not profile.accepts_exe(get(meta, "exe.kind"), get(meta, "exe.instance")):
        return "held", "exe %r/%r does not belong to profile %s" % (
            get(meta, "exe.kind"), get(meta, "exe.instance"), profile.name)
    pin_id = get(meta, "pin.pin_id")
    if not (isinstance(pin_id, str) and PIN_ID_RE.match(pin_id)):
        return "held", "pin.pin_id %r is not a pin id" % (pin_id,)
    lid = get(meta, "pin.launch_id")
    if not (isinstance(lid, str) and LAUNCH_ID_RE.match(lid)):
        return "held", "pin.launch_id %r is not a launch id" % (lid,)
    record_lid = get(meta, "launch_guard.record_launch_id")
    if record_lid is not None and record_lid != lid:
        return "held", "launch_guard.record_launch_id %r differs from pin.launch_id %r" % (record_lid, lid)
    wh, codec = get(meta, "rig.derivative_wh"), get(meta, "rig.derivative_codec")
    if wh != DERIVATIVE_WH or codec != DERIVATIVE_CODEC:
        return "held", "rig.derivative_wh %r / rig.derivative_codec %r are not the contract's %r / %s" % (
            wh, codec, DERIVATIVE_WH, DERIVATIVE_CODEC)
    c = Candidate(path, profile, meta, end)
    if profile.experimental and not c.experimental:
        return "held", ("a session of %s (the 1.9 Experimental Workbench) with the stable pin %s; Experimental "
                        "output is never harvested into captures/ (MIGRATION_1_9 3.4)" % (profile.name, pin_id))
    return "ok", c


def find_launch_record(c, root):
    """(bytes, record, source) of the session's launch record: the profile copy the guard read and DD-0's record
    under pin/launches; when both exist they must be byte-identical. Held when neither exists, when one is reached
    through a symlink, or when the record does not name the session's launch id."""
    name = "launch_%s.json" % c.launch_id
    found = []
    for source, path in (("profile dc_launch", os.path.join(c.profile.launch_dir, name)),
                         ("pin launches", os.path.join(root, "pin", "launches", name))):
        data = read_below(path, root, "%s %s" % (source, name))
        if data is not None:
            found.append((data, source))
    if not found:
        raise Held("no launch record %s in %s/profile/%s or pin/launches" % (name, c.profile.name, LAUNCH_COPY_DIR))
    if len(set(data for data, _ in found)) > 1:
        raise Held("the launch record copies of %s differ (profile dc_launch vs pin launches)" % name)
    try:
        rec = json.loads(found[0][0].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Held("launch record %s does not parse" % name)
    if not isinstance(rec, dict) or rec.get("launch_id") != c.launch_id:
        raise Held("launch record %s does not name launch id %s" % (name, c.launch_id))
    return found[0][0], rec, " + ".join(source for _, source in found)


# ------------------------------------------------------------------------------------------------
# build_key (read before validation, dc-2.0 11; MIGRATION_1_9 2.2)
# ------------------------------------------------------------------------------------------------
def parse_sums(data):
    """{relative path: sha256} of sha256sum-format text (bytes)."""
    sums = {}
    for line in data.decode("utf-8").splitlines():
        m = SUM_LINE_RE.match(line)
        if m:
            sums[m.group(2)] = m.group(1)
    return sums


def read_sums(path):
    with open(path, "rb") as f:
        return parse_sums(f.read())


def build_key_v1(pin, sums):
    """build_key/1 of a pin: sha256 of the ASCII text

        build_key/1
        game.appid=<int>
        game.buildid=<int>
        game.exe_sha256=<sha256 of game/ArmaReforgerSteam.exe>
        tools.appid=<int>
        tools.buildid=<int>
        tools.exe_sha256=<sha256 of tools/Workbench/ArmaReforgerWorkbenchSteamDiag.exe>

    (LF after each line): the game and Tools app ids and buildids and the sha256 of the client and Workbench exes
    (MIGRATION_1_9 2.2), the exe hashes from the pin copy's SHA256SUMS (paths relative to the copy, as DD-0
    cold_copy.sh writes them). Pins of one build give one key. ValueError when an input is missing."""
    lines = ["build_key/1"]
    for app, exe in BUILD_KEY_EXES:
        block = pin.get(app) if isinstance(pin.get(app), dict) else {}
        appid, buildid = block.get("appid"), block.get("buildid")
        digest = sums.get("%s/%s" % (app, exe))
        if not isinstance(appid, int) or not isinstance(buildid, int) or not digest:
            raise ValueError("pin %s: appid, buildid or the sha256 of %s/%s missing" % (app, app, exe))
        lines += ["%s.appid=%d" % (app, appid), "%s.buildid=%d" % (app, buildid), "%s.exe_sha256=%s" % (app, digest)]
    return sha256_bytes(("\n".join(lines) + "\n").encode("ascii"))


def resolve_build_key(c, rec, root):
    """(build_key, source) for a candidate, from every source that exists (they must agree): session.json, the
    launch record, the pin, the pin's build-key sidecar; computed build_key/1 only when none of these exists. The pin
    file, when present, must be the session's pin (pin.pin_sha256). Held otherwise."""
    pin_dir = os.path.join(root, "pin")
    found = []
    if get(c.meta, "pin.build_key"):
        found.append(("session.json", get(c.meta, "pin.build_key")))
    rec_key = rec.get("build_key") or get(rec, "pin.build_key")
    if rec_key:
        found.append(("launch_record", rec_key))
    pin = None
    raw = read_below(os.path.join(pin_dir, "pin_%s.json" % c.pin_id), root, "pin/pin_%s.json" % c.pin_id)
    if raw is not None:
        if sha256_bytes(raw) != get(c.meta, "pin.pin_sha256"):
            raise Held("pin/pin_%s.json is not the pin this session was launched with (pin.pin_sha256)" % c.pin_id)
        try:
            pin = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Held("pin/pin_%s.json does not parse" % c.pin_id)
        if not isinstance(pin, dict):
            raise Held("pin/pin_%s.json is not a JSON object" % c.pin_id)
        if pin.get("build_key"):
            found.append(("pin", pin["build_key"]))
    side_name = "pin/pin_%s.build_key.json" % c.pin_id
    side_raw = read_below(os.path.join(root, side_name), root, side_name)
    if side_raw is not None:
        try:
            sidecar = json.loads(side_raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Held("%s does not parse" % side_name)
        if not isinstance(sidecar, dict) or sidecar.get("pin_id") != c.pin_id:
            raise Held("%s does not name pin %s" % (side_name, c.pin_id))
        found.append(("pin_sidecar", sidecar.get("build_key")))
    if not found and pin is not None:
        sums_name = "pin/%s/%s" % (c.pin_id, SUMS_NAME)
        sums_raw = read_below(os.path.join(root, sums_name), root, sums_name)
        if sums_raw is not None:
            for app in ("game", "tools"):
                if get(pin, "%s.buildid" % app) != get(c.meta, "pin.%s_buildid" % app):
                    raise Held("pin %s %s buildid differs from the session's pin.%s_buildid" % (c.pin_id, app, app))
            try:
                found.append(("computed build_key/1", build_key_v1(pin, parse_sums(sums_raw))))
            except (UnicodeDecodeError, ValueError) as e:
                raise Held("build_key/1 cannot be computed: %s" % e)
    if not found:
        raise Held("no build_key for pin %s: session.json and the launch record carry none (dd-launch/1), and "
                   "pin/pin_%s.json has neither a build_key, a pin_%s.build_key.json sidecar (DD-14a) nor a "
                   "copy's %s/%s" % (c.pin_id, c.pin_id, c.pin_id, c.pin_id, SUMS_NAME))
    keys = set(key for _, key in found)
    if len(keys) > 1:
        raise Held("build_key sources disagree: %s" % ", ".join("%s %s" % (src, str(key)[:16]) for src, key in found))
    key = keys.pop()
    if not (isinstance(key, str) and BUILD_KEY_RE.match(key)):
        raise Held("build_key %r is not a build key" % (key,))
    return key, " + ".join(src for src, _ in found)


def check_frame_names(c):
    """TBD-4: a screenshot written under another name the probe scheduler polls (f_<slot:08d> plus one of the
    reader's IMAGE_SUFFIXES other than this run's --screenshot-suffix) holds the session: the reader would not see
    it, so C10/C11 would reject good data. Any other stray file is left to C11. A symlinked frames/ is held before
    it is listed."""
    fdir = os.path.join(c.path, "frames")
    if os.path.islink(fdir):
        raise Held("frames/ is a symlink; the harvester never reads through one")
    if not os.path.isdir(fdir):
        return
    other = {}
    for name in os.listdir(fdir):
        m = FRAME_NAME_RE.match(name)
        suffix = name[m.end():] if m else None
        if suffix is not None and suffix != c.screenshot_suffix and suffix in reader.IMAGE_SUFFIXES:
            other[suffix] = other.get(suffix, 0) + 1
    if other:
        raise Held("TBD-4: frames/ holds %s; this run reads frames/f_<slot:08d>%s only (--screenshot-suffix %r; "
                   "pass the suffix TBD-4 settles on), so the session stays in place" % (
                       ", ".join("%d file(s) f_<slot:08d>%s" % (n, suffix or " without extension")
                                 for suffix, n in sorted(other.items())), c.screenshot_suffix, c.screenshot_suffix))


# ------------------------------------------------------------------------------------------------
# Engine logs sliced to the session (C20, C11)
# ------------------------------------------------------------------------------------------------
def log_header(data):
    """(utc start epoch, local-minus-UTC offset in s, local midnight epoch of the start day as if UTC) from an
    engine log's 'Log ... started at <local> (<utc> UTC)' line, or None."""
    for line in data.split(b"\n", LOG_HEADER_SCAN)[:LOG_HEADER_SCAN]:
        m = LOG_HEADER_RE.match(line.strip(b"\r"))
        if m:
            v = [int(x) for x in m.groups()]
            local = calendar.timegm((v[0], v[1], v[2], v[3], v[4], v[5]))
            utc = calendar.timegm((v[6], v[7], v[8], v[9], v[10], v[11]))
            return utc, local - utc, calendar.timegm((v[0], v[1], v[2], 0, 0, 0))
    return None


def timed_lines(data):
    """[(UTC time, line)] of one engine log, or None without a header. A line takes the time of its HH:MM:SS.mmm
    prefix (local time, rolled over at midnight); a line without one (the header, a VME block, crash.log text) takes
    the time of the line before it."""
    head = log_header(data)
    if head is None:
        return None
    t, offset, midnight = head
    day = 0
    prev_sod = t + offset - midnight
    out = []
    for line in data.split(b"\n"):
        m = LOG_TIME_RE.match(line)
        if m:
            h, mi, s, ms = (int(x) for x in m.groups())
            sod = h * 3600 + mi * 60 + s + ms / 1000.0
            if sod < prev_sod - 43200:
                day += 1
            prev_sod = sod
            t = midnight + day * 86400 + sod - offset
        out.append((t, line))
    return out


def slice_timed(timed, lo, hi):
    """The lines of timed_lines() output whose UTC time lies in [lo, hi], leading and trailing empty lines dropped."""
    out = [line for t, line in timed if lo <= t <= hi]
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def slice_log(data, lo, hi):
    """The lines of one engine log whose UTC time lies in [lo, hi] (leading and trailing empty lines dropped), or
    None without a header."""
    timed = timed_lines(data)
    if timed is None:
        return None
    return slice_timed(timed, lo, hi)


def find_log_run(c, root, start):
    """(path, note) of the newest engine-log run of the session's profile (a logs_* directory, or a log base holding
    console.log itself) whose console.log header says it started before the session. note lists the runs whose
    header does not parse. Held when there is none."""
    runs = []
    unparsed = []
    for base in c.profile.log_bases:
        if not os.path.isdir(base) or not real_below(base, root):
            continue
        dirs = [os.path.join(base, n) for n in sorted(os.listdir(base)) if n.startswith("logs_")]
        for run in [base] + dirs:
            console = os.path.join(run, "console.log")
            if os.path.isdir(run) and real_below(console, root) and os.path.isfile(console):
                with open(console, "rb") as f:
                    head = log_header(f.read(4096))
                if head is None:
                    unparsed.append(os.path.relpath(run, root))
                elif head[0] <= start + LOG_RUN_SLACK_S:
                    runs.append((head[0], run))
    note = ""
    if unparsed:
        note = " (the console.log header of %s does not parse)" % ", ".join(sorted(unparsed))
    if not runs:
        raise Held("no engine-log run that started before the session in %s%s, so C20 (HARD) cannot be evaluated" % (
            " or ".join(os.path.relpath(base, root) for base in c.profile.log_bases), note))
    return max(runs)[1], note


def engine_log_slice(c, root, before_s, after_s, id_re):
    """{run, window_utc, start_marker, files: {log name: [lines]}, kept_whole: [names], left_out: {name: why}} of the
    engine-log run that holds the session. Held whenever the logs cannot be attributed to the session, so C20 (HARD
    from T0) never passes on missing or foreign lines:

      * the run (find_log_run) must name the session (its Screenshot-saved lines and DD-4-min's markers do);
      * a run that names an Eden-rooted world or an everon_ / kolguyev_ session id had another map's Play in its
        process (dc-2.0 3: Everon is sealed), so no line of it is copied;
      * each *.log is sliced to [start.unix_s - before_s, end.unix_s + after_s]. The window starts at the session's
        own START marker (LOG_START_MARKER) when the log has one, and it is cut at the nearest line at or before the
        start, or at or after the end, that names another session id (a neighbouring Play of the same process). The
        slice must name the session;
      * a headerless crash.log is attributed by its modification time: kept whole when that lies in the window, else
        left out and listed. Any other non-empty headerless log holds the session: its lines have no time."""
    start = get(c.meta, "start.unix_s")
    end = get(c.end, "end.unix_s") if c.end else None
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise Held("start.unix_s / end.unix_s give no time window, so the engine logs cannot be sliced (C20)")
    run, note = find_log_run(c, root, start)
    run_rel = os.path.relpath(run, root)
    logs = {}
    for name in sorted(os.listdir(run)):
        path = os.path.join(run, name)
        if name.endswith(".log") and os.path.isfile(path) and real_below(path, root):
            with open(path, "rb") as f:
                logs[name] = (f.read(), os.fstat(f.fileno()).st_mtime)
    own = c.session_id.encode("ascii")
    if not any(own in data for data, _ in logs.values()):
        raise Held("engine-log run %s does not name the session, so it is not the session's run%s" % (run_rel, note))
    for data, _ in logs.values():
        if LOG_EDEN_WORLD_RE.search(data) or any(m.group(0).startswith(LOG_OTHER_MAP_IDS)
                                                 for m in id_re.finditer(data)):
            raise Held("engine-log run %s names an Eden-rooted world or a session of another map: that Play ran in the "
                       "same process, so no line of the run is copied (Everon is sealed)" % run_rel)
    timed = dict((name, timed_lines(data)) for name, (data, _) in logs.items())
    lo, hi = start - before_s, end + after_s
    marker = LOG_START_MARKER % own
    marked = False
    for lines in timed.values():
        for t, line in lines or []:
            if marker in line and lo <= t <= end:
                lo = max(lo, t)
                marked = True
    for lines in timed.values():
        for t, line in lines or []:
            if lo <= t <= hi and (t <= start or t >= end) and any(m.group(0) != own for m in id_re.finditer(line)):
                if t <= start:
                    lo = max(lo, t + LOG_CUT_S)
                else:
                    hi = min(hi, t - LOG_CUT_S)
    files = {}
    whole = []
    left_out = {}
    for name, (data, mtime) in logs.items():
        if timed[name] is not None:
            lines = slice_timed(timed[name], lo, hi)
            if lines:
                files[name] = lines
        elif not data.strip():
            continue
        elif name.lower() != CRASH_LOG:
            raise Held("%s/%s has no parseable header, so its lines cannot be attributed to the session's window" % (
                run_rel, name))
        elif lo <= mtime <= hi:
            files[name] = data.rstrip(b"\n").split(b"\n")
            whole.append(name)
        else:
            left_out[name] = "no header; written %s, outside the session's window" % utc_text(mtime)
    if not any(own in line for lines in files.values() for line in lines):
        raise Held("no line of engine-log run %s in the session's window names the session" % run_rel)
    return {"run": os.path.basename(run), "window_utc": [lo, hi], "start_marker": marked, "files": files,
            "kept_whole": whole, "left_out": left_out}


# ------------------------------------------------------------------------------------------------
# The derivative (dc-2.0 9.8 deriv_path; 10.1 rig.derivative_wh, rig.derivative_codec)
# ------------------------------------------------------------------------------------------------
def derivative_plan(meta):
    """{box: [x0, y0, w, h], wh: [w, h], codec, quality, interpolation} of a session's rig: the centred crop of the
    core (capture minus rig.band_px on each side) with the aspect of the 512x288 derivative, resized to 512x288. For
    the client class that is the whole 1280x720 core; for a Workbench viewport, the centre 16:9 of its core. Held
    when the rig fields do not describe a banded capture."""
    cap, core, band = get(meta, "rig.capture_wh"), get(meta, "rig.core_wh"), get(meta, "rig.band_px")
    ints = [v for pair in (cap, core) if isinstance(pair, list) and len(pair) == 2 for v in pair]
    if len(ints) != 4 or not all(isinstance(v, int) and v > 0 for v in ints) or not isinstance(band, int) \
            or band < 0:
        raise Held("rig.capture_wh / core_wh / band_px unusable for the derivative")
    if cap[0] != core[0] + 2 * band or cap[1] != core[1]:
        raise Held("rig.capture_wh %r is not rig.core_wh %r plus two %r px bands" % (cap, core, band))
    (cw, ch), (dw, dh) = core, DERIVATIVE_WH
    if cw * dh > ch * dw:
        w, h = (ch * dw + dh // 2) // dh, ch
    else:
        w, h = cw, (cw * dh + dw // 2) // dw
    box = [band + (cw - w) // 2, (ch - h) // 2, w, h]
    area = w >= dw and h >= dh
    return {"box": box, "wh": list(DERIVATIVE_WH), "codec": DERIVATIVE_CODEC, "quality": DERIVATIVE_QUALITY,
            "interpolation": "area" if area else "linear"}


def make_derivative(rgb, plan):
    """JPEG bytes of the derivative of an H x W x 3 uint8 RGB capture."""
    import cv2  # only here: admission and the dry run need no OpenCV

    x0, y0, w, h = plan["box"]
    crop = numpy.ascontiguousarray(rgb[y0:y0 + h, x0:x0 + w, :3])
    interp = cv2.INTER_AREA if plan["interpolation"] == "area" else cv2.INTER_LINEAR
    small = cv2.resize(crop, tuple(plan["wh"]), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, plan["quality"]])
    if not ok:
        raise Held("OpenCV could not encode a derivative")
    return buf.tobytes()


def opencv_version():
    import cv2

    return "opencv %s" % cv2.__version__


# ------------------------------------------------------------------------------------------------
# Checksums
# ------------------------------------------------------------------------------------------------
def tree_files(session_dir, rename=None):
    """Sorted relative paths of every regular file below session_dir (byte order, as LC_ALL=C sort). rename maps a
    top-level directory name to the name it is listed under (the partial _harvest). Held on a symlink or a special
    file."""
    out = []
    for dirpath, dirnames, filenames in os.walk(session_dir):
        rel_dir = os.path.relpath(dirpath, session_dir)
        for name in dirnames + filenames:
            if os.path.islink(os.path.join(dirpath, name)):
                raise Held("%s is a symlink inside the session" % os.path.join(rel_dir, name))
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                raise Held("%s is not a regular file" % os.path.join(rel_dir, name))
            rel = os.path.normpath(os.path.join(rel_dir, name)).replace(os.sep, "/")
            top = rel.split("/", 1)[0]
            if rename and top in rename:
                rel = rename[top] + rel[len(top):]
            out.append((rel, full))
    out.sort(key=lambda item: item[0].encode("utf-8"))
    return out


def sums_text(files, exclude):
    lines = ["%s  %s" % (sha256_file(full), rel) for rel, full in files if rel != exclude]
    return ("\n".join(lines) + "\n").encode("utf-8")


def verify_sums(session_dir):
    """(ok, detail): _harvest/SHA256SUMS lists exactly the files of the session (itself excepted) and every hash
    matches."""
    sums_path = os.path.join(session_dir, HARVEST_DIR, SUMS_NAME)
    if os.path.islink(sums_path) or not os.path.isfile(sums_path):
        return False, "no %s/%s" % (HARVEST_DIR, SUMS_NAME)
    listed = read_sums(sums_path)
    exclude = "%s/%s" % (HARVEST_DIR, SUMS_NAME)
    present = dict((rel, full) for rel, full in tree_files(session_dir) if rel != exclude)
    if set(present) != set(listed):
        extra = sorted(set(present) - set(listed))
        missing = sorted(set(listed) - set(present))
        return False, "files differ from %s (unlisted %s, missing %s)" % (SUMS_NAME, extra[:2], missing[:2])
    bad = [rel for rel, full in sorted(present.items()) if sha256_file(full) != listed[rel]]
    if bad:
        return False, "%d file(s) do not match %s (%s)" % (len(bad), SUMS_NAME, bad[:2])
    return True, "%d files verified" % len(present)


def parse_row(line):
    """The catalog row of one line's bytes (a JSON object with a session_id), or None. A torn append never parses:
    a JSON object is complete only with its closing brace."""
    try:
        row = json.loads(line.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None
    if isinstance(row, dict) and isinstance(row.get("session_id"), str):
        return row
    return None


# ------------------------------------------------------------------------------------------------
# The harvester
# ------------------------------------------------------------------------------------------------
class Harvester:
    def __init__(self, root, args, test_mode):
        self.root = root
        self.args = args
        self.schema = reader.load_schema()
        self.id_re = re.compile(self.schema.raw["session"]["id_regex"].lstrip("^").rstrip("$").encode("ascii"))
        self.kill_at = os.environ.get(KILL_ENV) if test_mode else None
        self.entries = []
        self.lock_fd = None
        self.run_id = "%s_%s" % (utc_now().replace("-", "").replace(":", ""), os.urandom(2).hex())
        self.reason = args.reason if args.allow_mixed_builds else None

    # -- output -----------------------------------------------------------------------------------
    def report(self, status, where, detail, **extra):
        self.entries.append(dict({"status": status, "path": where, "detail": detail}, **extra))
        print("%s %s: %s" % (status, where, detail))

    def hook(self, name):
        if self.kill_at == name:
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)

    # -- discovery --------------------------------------------------------------------------------
    def session_dirs(self, profiles):
        """[(path, Profile)] of the --session arguments (each must be a directory directly inside a profile's
        DC_Sessions folder), or of every directory in every profile's DC_Sessions folder."""
        if self.args.session:
            out = []
            for raw in sorted(set(self.args.session)):
                real = os.path.realpath(os.path.abspath(raw))
                parent = os.path.dirname(real)
                match = [p for p in profiles if p.sessions == parent]
                if not match:
                    raise Refused("--session %s is not a session directory of a profile under %s "
                                  "(<profile>/%s/%s/<session_id>)" % (raw, self.root, PROFILE_DIR, SESSIONS_DIR))
                path = os.path.join(match[0].sessions, os.path.basename(real))
                if path not in [p for p, _ in out]:
                    out.append((path, match[0]))
            return out
        out = []
        for p in profiles:
            for name in sorted(os.listdir(p.sessions)):
                path = os.path.join(p.sessions, name)
                if os.path.isdir(path) or os.path.islink(path):
                    out.append((path, p))
        return out

    def uncatalogued(self, known):
        """Harvested sessions at a destination (they hold _harvest/) that the catalog does not list: moved by a run
        that was killed before its catalog row. Each one passes the map check first (a non-Arland one is refused
        and nothing else of it is read; its name is not printed), then its result.json and harvest.json must name
        it, its destination and route and a valid build_key (a symlinked _harvest/ is not read), and SHA256SUMS
        must verify. Returns [{status, dest, detail, build_key, route}]."""
        out = []
        for parts, depth, route in (((CAPTURES, "arland"), 2, "captures"), ((CAPTURES, REJECTED), 0, "rejected"),
                                    ((COMPAT19,), 1, "compat19")):
            for dest in self._walk_dirs(os.path.join(self.root, *parts), depth):
                name = os.path.basename(dest)
                hdir = os.path.join(dest, HARVEST_DIR)
                if name in known or not (os.path.isdir(hdir) or os.path.islink(hdir)):
                    continue
                rel = os.path.relpath(dest, self.root).replace(os.sep, "/")
                entry = {"dest": rel, "build_key": None, "route": None}
                meta = read_json(os.path.join(dest, "session.json")) if name.startswith("arland_") else None
                if meta is None or not is_arland(meta) or meta.get("session_id") != name:
                    out.append(dict(entry, dest=hidden(rel), status="refused", detail=NOT_ARLAND + " (found at a "
                                    "destination; not catalogued)"))
                    continue
                if os.path.islink(hdir):
                    out.append(dict(entry, status="held", detail="%s/ is a symlink; not read, not catalogued" %
                                    HARVEST_DIR))
                    continue
                result = read_json(os.path.join(hdir, "result.json")) or {}
                stamp = read_json(os.path.join(dest, HARVEST_JSON)) or {}
                key = stamp.get("build_key")
                if result.get("session_id") != name or result.get("dest") != rel or stamp.get("session_id") != name \
                        or result.get("route") != route or not (isinstance(key, str) and BUILD_KEY_RE.match(key)):
                    out.append(dict(entry, status="held", detail="result.json / harvest.json do not name this "
                                    "session, its destination and route and a build_key; not catalogued"))
                    continue
                try:
                    ok, detail = verify_sums(dest)
                except Held as e:
                    ok, detail = False, str(e)
                if not ok:
                    out.append(dict(entry, status="held", detail="moved but %s; not catalogued" % detail))
                    continue
                out.append(dict(entry, status="ok", detail=detail, build_key=stamp.get("build_key"),
                                route=result.get("route")))
        return out

    def _walk_dirs(self, base, depth):
        """Directories exactly depth + 1 levels below base, none reached through a symlink."""
        level = [base] if os.path.isdir(base) and real_below(base, self.root) else []
        for _ in range(depth + 1):
            nxt = []
            for d in level:
                for name in sorted(os.listdir(d)):
                    path = os.path.join(d, name)
                    if os.path.isdir(path) and not os.path.islink(path):
                        nxt.append(path)
            level = nxt
        return level

    # -- catalog and run manifests ----------------------------------------------------------------
    def catalog_path(self):
        """catalog/sessions.jsonl; Refused when the catalog directory or file is a symlink or not regular."""
        cdir = os.path.join(self.root, CATALOG_DIR)
        for path in (cdir, os.path.join(cdir, CATALOG_NAME), os.path.join(cdir, LOCK_NAME)):
            if os.path.islink(path):
                raise Refused("%s is a symlink; the catalog is never reached through one" % path)
        path = os.path.join(cdir, CATALOG_NAME)
        if os.path.lexists(path) and not os.path.isfile(path):
            raise Refused("%s is not a regular file" % path)
        return path

    def catalog_rows(self):
        """The complete rows of the catalog. A row counts only with its line feed: an unterminated last line is
        not counted here (catalog_tail reconciles it under the lock)."""
        path = self.catalog_path()
        if not os.path.isfile(path):
            return []
        rows = []
        with open(path, "rb") as f:
            for line in f.read().split(b"\n")[:-1]:
                row = parse_row(line)
                if row is not None:
                    rows.append(row)
        return rows

    def catalog_row(self, dest):
        """The catalog row of a harvested session, built from its destination's harvest.json and _harvest/ (so a
        later run builds the same row), stamped with this run's run_id and mixed-build reason."""
        stamp = read_json(os.path.join(dest, HARVEST_JSON)) or {}
        result = read_json(os.path.join(dest, HARVEST_DIR, "result.json")) or {}
        meta = read_json(os.path.join(dest, "session.json")) or {}
        row = {"schema": CATALOG_SCHEMA, "session_id": result.get("session_id"), "map_id": get(meta, "map.map_id"),
               "trial_id": get(meta, "trial.trial_id"), "date": result.get("date"), "exe": get(meta, "exe.kind"),
               "instance": get(meta, "exe.instance"), "pin_id": stamp.get("pin_id"),
               "build_key": stamp.get("build_key"), "build_key_source": stamp.get("build_key_source"),
               "use_class": stamp.get("use_class"), "channel": stamp.get("channel"),
               "verdict": result.get("verdict"), "exit_code": result.get("exit_code"), "route": result.get("route"),
               "dest": result.get("dest"), "images": result.get("images"), "decoded": result.get("decoded"),
               "derivatives": result.get("derivatives"),
               "sha256sums_sha256": sha256_file(os.path.join(dest, HARVEST_DIR, SUMS_NAME)),
               "harvester_version": result.get("harvester_version"), "harvested_utc": result.get("created_utc"),
               "run_id": self.run_id, "mixed_builds_reason": self.reason}
        return dict((k, row[k]) for k in CATALOG_COLUMNS)

    def catalog_tail(self, fd):
        """Reconcile an unterminated last line of the catalog (fd open read-write, under the lock): a complete row
        whose line feed was lost gets it back (the row and its run_id are kept), a torn append is cut off; complete
        rows are never changed. Returns "terminated", "cut" or None, and leaves fd at the end of the file."""
        size = os.fstat(fd).st_size
        keep = size
        while keep > 0:
            os.lseek(fd, keep - 1, os.SEEK_SET)
            if os.read(fd, 1) == b"\n":
                break
            keep -= 1
        action = None
        if keep != size:
            os.lseek(fd, keep, os.SEEK_SET)
            if parse_row(os.read(fd, size - keep)) is not None:
                os.lseek(fd, size, os.SEEK_SET)
                write_all(fd, b"\n")
                action = "terminated"
            else:
                os.ftruncate(fd, keep)
                action = "cut"
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_END)
        return action

    def reconcile_catalog(self):
        """catalog_tail on catalog/sessions.jsonl when it exists (under the lock); reports what it did."""
        path = self.catalog_path()
        if not os.path.isfile(path):
            return
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            action = self.catalog_tail(fd)
        finally:
            os.close(fd)
        if action == "terminated":
            self.report("repaired", "%s/%s" % (CATALOG_DIR, CATALOG_NAME), "the last row had lost its line feed; "
                        "the line feed was added, the row is unchanged")
        elif action == "cut":
            self.report("repaired", "%s/%s" % (CATALOG_DIR, CATALOG_NAME), "a torn last line (a killed append) was "
                        "cut off")

    def catalog_append(self, row):
        """Append one row to catalog/sessions.jsonl (under the lock; never through a symlink), after catalog_tail.
        Every byte is written and flushed to the disk."""
        ensure_dir(self.root, CATALOG_DIR)
        line = (json.dumps(row, separators=(",", ":")) + "\n").encode("ascii")
        fd = os.open(self.catalog_path(), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        try:
            self.catalog_tail(fd)
            write_all(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def manifest_rel(self, run_id):
        return "%s/harvest_%s.json" % (RUN_MANIFESTS, run_id)

    def missing_manifests(self):
        """{run_id: [catalog rows]} of the runs whose captures/arland rows have no run manifest yet."""
        runs = {}
        for row in self.catalog_rows():
            rid = row.get("run_id")
            if row.get("route") == "captures" and isinstance(rid, str) and RUN_ID_RE.match(rid):
                runs.setdefault(rid, []).append(row)
        return dict((rid, rows) for rid, rows in runs.items()
                    if not os.path.lexists(os.path.join(self.root, self.manifest_rel(rid))))

    def write_manifest(self, run_id, rows):
        """manifests/harvest_runs/harvest_<run_id>.json from the catalog rows of one run: the captures/arland
        sessions, their build_key set and the mixed-build reason the run was given."""
        builds = sorted(set(r.get("build_key") for r in rows))
        reasons = sorted(set(r.get("mixed_builds_reason") for r in rows if r.get("mixed_builds_reason")))
        doc = {"schema": RUN_SCHEMA, "run_id": run_id, "created_utc": utc_now(),
               "harvester_version": HARVESTER_VERSION, "build_keys": builds, "mixed_builds": len(builds) > 1,
               "reason": reasons[0] if reasons else None,
               "sessions": [dict((k, r.get(k)) for k in ("session_id", "dest", "build_key", "verdict",
                                                         "sha256sums_sha256")) for r in rows]}
        ensure_dir(self.root, RUN_MANIFESTS)
        replace_file(os.path.join(self.root, self.manifest_rel(run_id)), json_text(doc))
        return self.manifest_rel(run_id)

    # -- the run ----------------------------------------------------------------------------------
    def run(self):
        profiles = list_profiles(self.root)
        known = set(r["session_id"] for r in self.catalog_rows())
        candidates = []
        for path, profile in self.session_dirs(profiles):
            rel = os.path.relpath(path, self.root)
            if not os.path.lexists(path) and os.path.basename(path) in known:
                self.report("done", rel, "already harvested (catalog/sessions.jsonl lists it)")
                continue
            status, value = admit(path, profile, self.schema)
            if status == "refused":
                self.report(status, hidden(rel), value)
                continue
            if status != "ok":
                self.report(status, rel, value)
                continue
            value.screenshot_suffix = self.args.screenshot_suffix
            try:
                value.launch_bytes, rec, value.launch_source = find_launch_record(value, self.root)
                value.build_key, value.build_key_source = resolve_build_key(value, rec, self.root)
                check_frame_names(value)
                value.logs = engine_log_slice(value, self.root, self.args.log_before_s, self.args.log_after_s,
                                              self.id_re)
            except Held as e:
                self.report("held", rel, str(e))
                continue
            candidates.append(value)
        pending = []
        for entry in self.uncatalogued(known):
            if entry["status"] == "ok":
                pending.append(entry)
            else:
                self.report(entry["status"], entry["dest"], entry["detail"])
        manifests = self.missing_manifests()
        builds = sorted(set([c.build_key for c in candidates if not c.experimental]
                            + [e["build_key"] for e in pending if e["route"] != "compat19"]))
        if len(builds) > 1 and not self.args.allow_mixed_builds:
            raise Refused("the sessions this run would catalogue outside compat19 carry %d build_keys (%s); a run is "
                          "single-build_key (MIGRATION_1_9 3.3): harvest one build at a time with --session, or "
                          "pass --allow-mixed-builds --reason TEXT" % (len(builds), ", ".join(
                              str(k)[:16] for k in builds)))
        if self.args.dry_run:
            for c in candidates:
                self.report("would-harvest", os.path.relpath(c.path, self.root), "pin %s, build_key %s (%s), "
                            "use_class %s -> %s" % (c.pin_id, c.build_key[:16], c.build_key_source, c.use_class,
                                                    c.compat_dest() if c.experimental else "captures/ by verdict"))
            for e in pending:
                self.report("would-catalogue", e["dest"], "moved by an earlier run, not in the catalog")
            for rid in sorted(manifests):
                self.report("would-write", self.manifest_rel(rid), "run manifest missing")
            return self.finish([])
        if not candidates and not pending and not manifests:
            if not self.entries:
                self.report("nothing", self.root, "no sealed session to harvest and nothing to catalogue")
            return self.finish([])
        self.lock()
        try:
            self.reconcile_catalog()
            for c in candidates:
                rel = os.path.relpath(c.path, self.root)
                try:
                    self.harvest(c)
                except Held as e:
                    self.report("held", rel, str(e))
                except OSError as e:
                    self.report("held", rel, "I/O error: %s (the session is whole where it is; the next run carries "
                                "on)" % e)
            known = set(r["session_id"] for r in self.catalog_rows())
            for e in pending:
                if os.path.basename(e["dest"]) not in known:
                    self.catalog_append(self.catalog_row(os.path.join(self.root, e["dest"])))
                    self.report("catalogued", e["dest"], "moved by an earlier run; catalog row added")
            written = [self.write_manifest(rid, rows) for rid, rows in sorted(self.missing_manifests().items())]
        finally:
            self.unlock()
        return self.finish(written)

    def lock(self):
        """Hold catalog/harvest.lock (flock) for the rest of the run; Refused when another harvester holds it."""
        self.catalog_path()
        path = os.path.join(ensure_dir(self.root, CATALOG_DIR), LOCK_NAME)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise Refused("%s is not a regular file" % path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise Refused("another harvester holds %s" % path)
        self.lock_fd = fd

    def unlock(self):
        if self.lock_fd is not None:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            self.lock_fd = None

    def finish(self, manifests):
        for rel in manifests:
            print("run manifest: %s" % rel)
        statuses = [e["status"] for e in self.entries]
        if any(s in ("held", "refused") for s in statuses) or (self.args.session and "skipped" in statuses):
            return EXIT_HELD
        if any(e.get("route") == "rejected" for e in self.entries):
            return EXIT_REJECTED
        return EXIT_OK

    # -- one session ------------------------------------------------------------------------------
    def harvest(self, c):
        src = c.path
        if not os.path.isdir(src):
            raise Held("gone before it was harvested (another harvester moved it?)")
        if os.path.islink(src) or not real_below(src, self.root):
            raise Held("reached through a symlink since admission; nothing was written")
        result = self.reusable(c)
        if result is None:
            self.clean(src)
            self.write_sidecars(c)
            self.hook("after_sidecars")
            try:
                report, code, refusal = self.validate(c)
            except Held:
                self.drop_sidecars(src)
                raise
            self.hook("after_validate")
            if report is None and not (c.experimental and refusal.startswith(NO_THRESHOLDS_REFUSAL)):
                self.drop_sidecars(src)
                raise Held("dc_validate refused the session or failed (exit %d): %s; the sidecars were removed "
                           "again" % (code, refusal))
            result = self.build(c, report, code, refusal)
            self.hook("before_harvest_rename")
            os.rename(os.path.join(src, PARTIAL_DIR), os.path.join(src, HARVEST_DIR))
            fsync_dir(src)
            self.hook("after_harvest_rename")
            ok, detail = verify_sums(src)
            if not ok:
                raise Held("checksums do not verify after the build: %s" % detail)
        dest = self.move(src, result["dest"])
        self.hook("after_move")
        self.catalog_append(self.catalog_row(dest))
        self.hook("after_catalog")
        detail = "%s -> %s (%s derivative(s) of %s decoded slot(s); sha256 verified)" % (
            result["verdict"], result["dest"], result["derivatives"], result["decoded"])
        self.report("harvested", result["session_id"], detail, route=result["route"], dest=result["dest"],
                    verdict=result["verdict"])

    def reusable(self, c):
        """The result.json of a complete _harvest/ left in the source by an earlier run that was killed before the
        move, when it can be moved as it is: built by this harvester version, its SHA256SUMS verifies, harvest.json
        carries the stamp admission found (c.stamp) and result.json names this session and the route and
        destination its verdict sends it to. None otherwise (it is rebuilt). A symlinked _harvest/ is not read."""
        hdir = os.path.join(c.path, HARVEST_DIR)
        if os.path.islink(hdir) or not os.path.isdir(hdir):
            return None
        stamp = read_json(os.path.join(c.path, HARVEST_JSON)) or {}
        result = read_json(os.path.join(hdir, "result.json")) or {}
        if any(stamp.get(k) != v for k, v in c.stamp().items()) or stamp.get("harvester_version") != HARVESTER_VERSION:
            return None
        if result.get("harvester_version") != HARVESTER_VERSION or result.get("session_id") != c.session_id:
            return None
        if c.route_of(result.get("verdict")) != (result.get("route"), result.get("dest")):
            return None
        return result if verify_sums(c.path)[0] else None

    def clean(self, src):
        """Remove the outputs of an earlier, unfinished run: _harvest/, _harvest.partial/ and *.dd6tmp (the
        sidecars are replaced next). Never raw data."""
        for name in sorted(os.listdir(src)):
            if name in (HARVEST_DIR, PARTIAL_DIR) or name.endswith(TMP_SUFFIX):
                remove_own(src, name)

    def drop_sidecars(self, src):
        """Remove the sidecars of a session that stays in place after a validator refusal, so a held session is
        left as it was found (re-runs do not rewrite it)."""
        for name in (LAUNCH_SIDECAR, LOGS_SIDECAR, HARVEST_JSON):
            remove_own(src, name)
        fsync_dir(src)

    def replace_dir(self, src, name, files):
        """Replace the harvester-owned directory src/name by one holding files ({name: bytes}); without files the
        directory is removed."""
        tmp = name + TMP_SUFFIX
        remove_own(src, tmp)
        if files:
            os.mkdir(os.path.join(src, tmp))
            for fname, data in sorted(files.items()):
                write_file(os.path.join(src, tmp, fname), data)
        remove_own(src, name)
        if files:
            os.rename(os.path.join(src, tmp), os.path.join(src, name))

    def write_sidecars(self, c):
        """_launch/launch_<id>.json, _engine_logs/*.log and harvest.json, written before validation (dc_validate
        reads them from the session directory)."""
        src = c.path
        launch_name = "launch_%s.json" % c.launch_id
        self.replace_dir(src, LAUNCH_SIDECAR, {launch_name: c.launch_bytes})
        files = c.logs["files"]
        self.replace_dir(src, LOGS_SIDECAR, dict((name, b"\n".join(lines) + b"\n") for name, lines in files.items()))
        logs = dict((k, c.logs[k]) for k in ("run", "window_utc", "start_marker", "kept_whole", "left_out"))
        logs["lines"] = dict((n, len(v)) for n, v in sorted(files.items()))
        stamp = {"schema": HARVEST_SCHEMA, "session_id": c.session_id, "harvester_version": HARVESTER_VERSION,
                 "created_utc": utc_now(), "pin_id": c.pin_id, "pin_sha256": get(c.meta, "pin.pin_sha256"),
                 "channel": c.channel, "use_class": c.use_class, "build_key": c.build_key,
                 "build_key_source": c.build_key_source, "screenshot_suffix": c.screenshot_suffix,
                 "launch_record": {"path": "%s/%s" % (LAUNCH_SIDECAR, launch_name),
                                   "sha256": sha256_bytes(c.launch_bytes), "source": c.launch_source},
                 "engine_logs": logs}
        replace_file(os.path.join(src, HARVEST_JSON), json_text(stamp))
        fsync_dir(src)

    def validate(self, c):
        """(report or None, exit code, refusal text) of dc_validate on the session (a subprocess, no report file)."""
        cmd = [sys.executable, "-I", "-B", DC_VALIDATE, "--session", c.path, "--no-report", "--json",
               "--image-suffix=%s" % c.screenshot_suffix]
        if self.args.routes_dir:
            cmd += ["--routes-dir", self.args.routes_dir]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                  timeout=VALIDATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise Held("dc_validate did not finish within %d s" % VALIDATE_TIMEOUT_S)
        out = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode in (0, 1, 2):
            try:
                report = json.loads(out)
            except ValueError:
                raise Held("dc_validate exit %d without a JSON report" % proc.returncode)
            if report.get("exit_code") != proc.returncode:
                raise Held("dc_validate exit %d but report exit_code %r" % (proc.returncode, report.get("exit_code")))
            return report, proc.returncode, None
        lines = [line for line in out.splitlines() if line.strip()]
        lines += [line for line in proc.stderr.decode("utf-8", errors="replace").splitlines() if line.strip()]
        return None, proc.returncode, (lines[0] if lines and lines[0].startswith("refused") else
                                       (lines[-1] if lines else "no output"))

    def build(self, c, report, code, refusal):
        """_harvest.partial/: dc_validate.json, img_decoded.csv, deriv/, result.json, SHA256SUMS. Returns the
        result dict."""
        src = c.path
        partial = os.path.join(src, PARTIAL_DIR)
        os.mkdir(partial)
        if report is not None:
            verdict = report["verdict"]
            write_file(os.path.join(partial, "dc_validate.json"),
                       (json.dumps(report, indent=1) + "\n").encode("ascii"))
        else:
            verdict = "REFUSED"
            write_file(os.path.join(partial, "dc_validate.txt"), ("exit %d: %s\n" % (code, refusal)).encode(
                "ascii", errors="replace"))
        route, dest = c.route_of(verdict)
        plan = derivative_plan(c.meta) if verdict in ("PASS", "PASS_WITH_WARNINGS") else None
        deriv = {}
        if plan is not None:
            os.mkdir(os.path.join(partial, "deriv"))

        def visit(session, row, rgb):
            if plan is None or not row["fid_ok"]:
                return
            if [rgb.shape[1], rgb.shape[0]] != get(c.meta, "rig.capture_wh"):
                raise Held("slot %d is %dx%d, not rig.capture_wh" % (row["slot_id"], rgb.shape[1], rgb.shape[0]))
            name = "f_%08d.jpg" % row["slot_id"]
            write_file(os.path.join(partial, "deriv", name), make_derivative(rgb, plan))
            deriv[row["slot_id"]] = "%s/deriv/%s" % (HARVEST_DIR, name)
            if len(deriv) == 1:
                self.hook("in_derivatives")

        images = None
        try:
            images = reader.open_session(src, strict=False, decode_images=True, image_visitor=visit,
                                         image_suffix=c.screenshot_suffix).images
        except Held:
            raise
        except Exception as e:  # a failed session the reader cannot read still moves to _rejected with its report
            if plan is not None:
                raise Held("the reader cannot read a session dc_validate passed: %s" % e)
        if images is not None:
            write_file(os.path.join(partial, "img_decoded.csv"), self.img_decoded_csv(src, images, deriv))
        images = images or []
        decoded = sum(1 for r in images if r["fid_ok"])
        if plan is not None and len(deriv) != decoded:
            raise Held("%d derivative(s) for %d decoded slot(s)" % (len(deriv), decoded))
        result = {"schema": RESULT_SCHEMA, "session_id": c.session_id, "date": c.date(),
                  "harvester_version": HARVESTER_VERSION, "created_utc": utc_now(), "verdict": verdict,
                  "exit_code": code, "route": route, "dest": dest, "images": len(images), "decoded": decoded,
                  "derivatives": len(deriv),
                  "derivative": dict(plan, library=opencv_version()) if plan is not None else None,
                  "validator": None if report is None else dict(
                      (k, report.get(k)) for k in ("validator_version", "contract_version", "contract_status",
                                                   "thresholds_file", "thresholds_sha256", "trial", "tier"))}
        write_file(os.path.join(partial, "result.json"), json_text(result))
        files = tree_files(src, rename={PARTIAL_DIR: HARVEST_DIR})
        write_file(os.path.join(partial, SUMS_NAME), sums_text(files, "%s/%s" % (HARVEST_DIR, SUMS_NAME)))
        fsync_dir(partial)
        return result

    def img_decoded_csv(self, src, images, deriv):
        """img_decoded.csv: the img_resolved columns of the streams JSON (dc-2.0 9.8), one row per img row; read
        back with the reader's CSV parser before it is kept."""
        fields = self.schema.raw["derived"]["img_resolved"]["fields"]
        lines = [",".join(f["name"] for f in fields)]
        for r in images:
            on_disk = bool(r["file"]) and os.path.isfile(os.path.join(src, r["file"]))
            values = {"slot_id": r["slot_id"], "file": r["file"] if on_disk else None,
                      "file_exists": r["file_exists"], "png_w": r["png_w"], "png_h": r["png_h"],
                      "fid_ok": r["fid_ok"], "captured_render_frame": r["captured_render_frame"],
                      "fid_margin": r["fid_margin"], "deriv_path": deriv.get(r["slot_id"])}
            cells = []
            for f in fields:
                v = values[f["name"]]
                if v is None:
                    cells.append("")
                elif f["type"] == "float":
                    cells.append("%.6f" % v)
                else:
                    cells.append(str(int(v)) if f["type"] in ("int", "bool") else str(v))
            lines.append(",".join(cells))
        data = ("\n".join(lines) + "\n").encode("ascii")
        problems = []
        _, rows = reader.parse_csv_bytes(data, fields, "img_decoded.csv", problems)
        if problems or rows is None or len(rows) != len(images):
            raise Held("img_decoded.csv does not parse back: %s" % (problems[:1],))
        return data

    def move(self, src, dest_rel):
        """Rename the session directory to root/dest_rel on the same volume; never over an existing entry."""
        parent = ensure_dir(self.root, os.path.dirname(dest_rel))
        dest = os.path.join(parent, os.path.basename(dest_rel))
        if os.path.lexists(dest):
            raise Held("destination %s already exists; nothing was moved" % dest_rel)
        if os.stat(src).st_dev != os.stat(parent).st_dev:
            raise Held("%s is on another volume; a move would not be one rename, so nothing was moved" % dest_rel)
        os.rename(src, dest)
        fsync_dir(parent)
        return dest


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print("usage error: %s" % message, file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def parse_args(argv):
    ap = _Parser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", action="append", default=[],
                    help="harvest only this session directory (repeatable); default: every profile's DC_Sessions")
    ap.add_argument("--dry-run", action="store_true", help="admit and plan only; write nothing")
    ap.add_argument("--routes-dir", default=None,
                    help="route files for dc_validate (default %s when it exists)" % os.path.relpath(
                        DEFAULT_ROUTES_DIR, REPO))
    ap.add_argument("--allow-mixed-builds", action="store_true",
                    help="allow sessions of several build_keys in one run (needs --reason)")
    ap.add_argument("--reason", default=None, help="why builds are mixed (recorded in the catalog rows)")
    ap.add_argument("--log-before-s", type=float, default=DEFAULT_LOG_BEFORE_S,
                    help="engine-log slice: seconds before the session start (default %(default)s)")
    ap.add_argument("--log-after-s", type=float, default=DEFAULT_LOG_AFTER_S,
                    help="engine-log slice: seconds after the session end (default %(default)s)")
    ap.add_argument("--screenshot-suffix", choices=reader.IMAGE_SUFFIXES, default=reader.IMAGE_SUFFIX,
                    help="TBD-4: the suffix the engine appends to an img path ('' for the bare path; default "
                         "%(default)s); passed to dc_validate --image-suffix")
    ap.add_argument("--test-root", default=None, help="TESTS ONLY: a /dev/shm/ or /mnt/e/ directory in place of %s"
                    % DATA_ROOT)
    args = ap.parse_args(argv)
    if args.allow_mixed_builds and not (args.reason and args.reason.strip()):
        ap.error("--allow-mixed-builds needs --reason TEXT")
    if args.reason and not args.allow_mixed_builds:
        ap.error("--reason goes with --allow-mixed-builds")
    if args.routes_dir is None and os.path.isdir(DEFAULT_ROUTES_DIR):
        args.routes_dir = DEFAULT_ROUTES_DIR
    if args.log_before_s < 0 or args.log_after_s < 0:
        ap.error("log margins must not be negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        root = data_root(args.test_root)
        return Harvester(root, args, args.test_root is not None).run()
    except Refused as e:
        print("refused: %s" % e)
        return EXIT_REFUSED
    except Exception:  # an I/O error or a bug: exit 4, never a silent success
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
