"""Reader of capture contract dc-2.0 sessions (DD-5-min): committed rows by header name, the offline joins, the
decision samples, masks and segments.

    spec = importlib.util.spec_from_file_location("dc2_reader", "<repo>/data/dc2/reader.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    session = reader.open_session("/mnt/e/DriveDiT/captures/arland/2026-10-25/T0b/<session_id>")
    for step in session.decisions:
        ...

Loaded by file path like data/dc2/fiducial.py: `import data.dc2.reader` would run data/__init__.py, which imports
torch. Needs the stdlib and numpy; cv2 only when images are decoded (fiducial.load_rgb).

What it reads (docs/capture_contract/dc-2.0.md; dc-2.0.streams.json is normative for names, order, types and keys,
so no column list is written down here):
  * session.json, session_end.json (the seal), heartbeat.csv and, when present, the harvester's harvest.json
    (build_key of dd-launch/1 sessions);
  * every stream directory: commit.csv first, then only the chunks it lists (section 8: a row is acknowledged only
    when its chunk's commit line was written). Chunk files without a commit line are uncommitted tails: they are
    discarded and listed in Stream.uncommitted;
  * each committed chunk by header name (C05): a missing required or an unknown column refuses the chunk; each row
    is parsed whole into temporaries (type, finite number without exponent, enum, required non-empty) and appended
    only when every field is valid, so a bad row never leaves the columns at different lengths (CR-093);
  * frames/: each img row's file (path + the image suffix, TBD-4: IMAGE_SUFFIX unless open_session is given
    another of IMAGE_SUFFIXES), its size and its fiducial v2 code
    (data/dc2/fiducial.py); captured_render_frame is the code unwrapped against issue_render_frame (img_resolved).

Rows are frame-bound (contract section 6, freeze decision D-A): ctl and telemetry hold one row per fixed-frame
callback with the same keys; physics_step is the engine's fixed-step clock and advances by the row's covered_steps,
so a row covers the steps physics_step - covered_steps + 1 .. physics_step (covering_row). world_time_ms is the frame
clock, and the logged pose follows it.

Joins and timing (section 6 and the streams JSON "timing"): img -> hf exact on captured_render_frame C (in Workbench
also img.done_render_frame - 1, kept as a cross-check); the image of frame C shows the state of hf row C + 1, its pose
row; hf -> ctl and hf -> telemetry exact on last_physics_step (both have a row per callback). Rates and periods come
from session.json clocks.* (decision_hz, image_hz, physics_steps_per_decision) and engine.step_dt_s and from the
counters, never from an assumed frame rate (CR-094); rows are keyed by their own counters, never by position
(CR-085). Only the ego is read: dc-2.0 has one ego per session and the traffic stream is kept apart.

Decision samples (the R-01 decision boundary on the world-time grid): grid point g is due at anchor + g x 1000 /
decision_hz ms (anchor = world_time_ms of the first ctl row); decision callback C_t is the first ctl row at or past
its due time (decision_callbacks), P_t its physics_step, and the decision interval is (P_t, P_t+1] up to the next
decision callback. state_t = the telemetry row of C_t; action_label_t = ctl.exp_*_now of C_t; action_cmd_t = ctl.hl_*
of the row after C_t (decided after the rows of C_t, applied from step P_t + 1); action_exec_t = the executed values
of the steps P_t + 1 .. P_t + N (N = clocks.physics_steps_per_decision), each from the ctl row that covers it, empty
past P_t+1; obs_t = the newest decoded image whose pose row (hf row C + 1) joins ctl and telemetry and has
last_physics_step <= P_t and whose captured frame is at or before C_t's render frame (never a future image, so
obs_age_ms >= 0), valid when it is at most one image period old (from the world time of hf row C); an image of the
session's last render frame (a seal without drain) has no pose row and is never obs_t (pose_after_seal). A decision
step is wm-valid only with every step of its interval covered by ctl rows, a telemetry row at P_t, one episode over
the interval, the command held unchanged over it (same hl_seq and hl_*), every action_exec entry inside it, a valid
image of the same episode and no teleport or respawn between the image's last physics step and P_t+1. Segments are
the maximal runs of wm-valid, non-warm-up decision steps of one episode on consecutive grid points, so they split at
gaps, skipped grid points, episode boundaries and masks.

Crash events (section 9.5.1): the rows of the kinds that event_payloads defines are parsed into Session.crash_events
(payload values typed by their key spec, ranges and minimums included, with the problems C36 reports). A row belongs
to its episode_id (episode ownership); its step is the onset (onset_step of a collision or rollover, else
physics_step) and must lie inside that episode, a row with episode_id -1 outside every episode, and never after the
row's own physics_step (chronology). Terminal ego events (a collision with src dv or dv_contact, a veh_damage with a
state other than undamaged or movement_damage >= 1, a veh_fire_start, a rollover) count only when both hold:
Session.terminal_ego[e] is the first by onset, Session.terminal_ego_confirmed[e] the first by confirmation (a rollover
is confirmed at its row, the end of the hold; confirmed_inside tells whether that was inside the episode). The core is
the only source of the terminations collision and rollover, so an episode's termination collision or rollover is the
reason of its first terminal ego event by confirmation, confirmed inside it (C36).
A decision step whose interval (P_t, P_t+1] contains or follows the earliest onset is not policy-valid (wm_valid
unchanged: crash dynamics are world-model data). The teleport mask uses only teleport and respawn rows of the ego slot
or of no vehicle (veh_slot_id -1).

Files: every file the reader opens (stream directories, chunks, session files, images) must resolve inside the
session directory (symlinks followed), and img.path must be exactly frames/f_<slot_id:08d> (TBD-4); anything else
is a C01 (outside) or C11 (path) problem and is not read.

The seal (session_end.json, sections 5.3, 8 and 10.2): status ok with a non-error end_reason (C01; ego_destroyed is
a normal end), contract_version
dc-2.0 and episodes equal to the committed episodes rows (C01); the committed ctl, telemetry and hf rows reach the
session end (end.physics_step for ctl and telemetry, end.render_frame: C34), so a lost or truncated tail is seen.

Refusals (ReaderError subclasses): dc-1.0 sessions (LegacySessionError; never mixed with dc-2.0 data, section 14);
Everon (map_id everon, terrain Eden, an Eden-rooted root_world, an everon_ session id, or a path under heldout/ or a
SEALED directory) unless sealed_eval=True (SealedMapError, section 3.2), checked before any stream is opened; any
path on the C: drive. With strict=True (the default, for dataset builders) every C01, C05, C07 (duplicated or
out-of-order key), C34 or C02 (vehicle card hash) problem and an unsealed or failed seal raise as well; salvage=True
reads an unsealed or failed session's committed data (crash salvage) and still raises on the other problems.
strict=False (the validator) records every problem in Session.problems and keeps going.
"""

import bisect
import collections
import hashlib
import importlib.util
import json
import math
import os
import re

import numpy

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STREAMS_JSON = os.path.join(REPO, "docs", "capture_contract", "dc-2.0.streams.json")
CARDS_DIR = os.path.join(REPO, "tools", "ident", "cards")

READER_VERSION = "dc2-reader/1"
CONTRACT_VERSION = "dc-2.0"
IMAGE_SUFFIX = ".png"           # TBD-4 default: the engine appends .png to the extension-less request path
# TBD-4 candidates: the names the probe scheduler polls for an extension-less request path (DC_ProbeImages.c
# m_aSuffixes; "" is the bare path). TBD-4 is decided (.png, dc-2.0 freeze); a caller may still read another one to
# diagnose a build that names its files differently.
IMAGE_SUFFIXES = (".png", ".bmp.png", ".bmp", ".jpg", "")
ENV_TRANSITION_MS = 15000       # features_v1 masks.env_transition: 1 for 15 s after an environment change
LEGACY_MARKERS = ("telemetry.csv", "summary.txt", "session_info.txt")
LEGACY_DIR_RE = re.compile(r"^session_[0-9]+$")
SEALED_PATH_PARTS = ("heldout", "sealed")
FORBIDDEN_ROOTS = ("/mnt/c/",)
CHUNK_RE = re.compile(r"^chunk_([0-9]{5})\.csv$")
INT_RE = re.compile(r"-?[0-9]+")
FLOAT_RE = re.compile(r"-?[0-9]+(\.[0-9]+)?")     # '.' decimal point, no exponent; nan and inf never match
EDEN_ROOT_RE = re.compile(r"^(\{[0-9A-Fa-f]+\})?worlds/eden/", re.IGNORECASE)   # every Eden-rooted world is Everon
# session_end.json end_reason values that end a session in error (section 5.3); duration, stop, dc_stop and
# ego_destroyed (the ego was destroyed after the first row: a crash is valid data, status ok) are the normal endings.
ERROR_END_REASONS = ("script_error:write_fail", "script_error:other", "zero_ego", "ego_lost", "camera_lost")
TERMINAL_COLLISION_SRC = ("dv", "dv_contact")     # src contact is a hit below the detector: never terminal (9.5.1)
STRICT_FATAL = ("C01", "C02", "C05", "C07", "C34")
HL_COLUMNS = ("hl_seq", "hl_kappa", "hl_accel", "hl_dir")

# Stream columns the derived views read (docs dc-2.0.md 9.8 ctl_export / telemetry_export and the decision
# boundary). Schema.check_columns verifies at load time that the streams JSON still defines every one of them.
DERIVED_SOURCES = {
    "ctl": ("physics_step", "covered_steps", "render_frame", "world_time_ms", "episode_id", "mode", "driver",
            "hl_seq", "hl_kappa", "hl_accel", "hl_dir", "exp_kappa_now", "exp_accel_now", "exp_dir_now",
            "llc_steer_lim", "llc_throttle", "llc_brake", "input_post_steer", "react_visible"),
    "telemetry": ("physics_step", "covered_steps", "render_frame", "world_time_ms", "episode_id", "ego_slot_id",
                  "fwd_x", "fwd_z", "vel_local_x", "vel_local_z", "angvel_local_y", "acc_local_x", "acc_local_z",
                  "gear"),
    "hf": ("render_frame", "world_time_ms", "last_physics_step", "paused"),
    "img": ("slot_id", "issue_status", "issue_render_frame", "path", "done_render_frame"),
    "events": ("event_seq", "episode_id", "physics_step", "world_time_ms", "kind", "veh_slot_id", "payload"),
    "episodes": ("episode_id", "ego_slot_id"),
}

Problem = collections.namedtuple("Problem", "check where message")


class ReaderError(Exception):
    """The session cannot be read (or, with strict=True, holds a C01/C05/C34 problem)."""


class LegacySessionError(ReaderError):
    """A dc-1.0 session ($profile:DrivingData/session_<n>): the dc-2.0 reader never reads it (section 14)."""


class SealedMapError(ReaderError, PermissionError):
    """Everon data outside the sealed evaluation path (section 3.2: loaders refuse it unless sealed_eval=True)."""


class SchemaError(ReaderError):
    """A header or file problem that refuses data (names the column or key)."""


def _load_fiducial():
    spec = importlib.util.spec_from_file_location("dc2_fiducial", os.path.join(HERE, "fiducial.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fiducial = _load_fiducial()


# ------------------------------------------------------------------------------------------------
# Schema (the streams JSON)
# ------------------------------------------------------------------------------------------------
class Schema:
    """dc-2.0.streams.json with lookups. frozen is True once the contract status line says FROZEN."""

    def __init__(self, path=None):
        self.path = os.path.realpath(path or STREAMS_JSON)
        with open(self.path, "rb") as f:
            raw_bytes = f.read()
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self.raw = json.loads(raw_bytes.decode("ascii"))
        if self.raw.get("contract_version") != CONTRACT_VERSION:
            raise SchemaError("%s is not the %s stream schema" % (self.path, CONTRACT_VERSION))
        self.status = self.raw.get("status", "")
        self.frozen = self.status.upper().startswith("FROZEN")
        self.streams = self.raw["streams"]
        self.files = self.raw["files"]
        self.maps = {m["map_id"]: m for m in self.raw["maps"]}
        self.id_re = re.compile(self.raw["session"]["id_regex"])
        self.chunk_name = self.raw["session"]["chunk_name"]
        self.commit_name = self.raw["session"]["commit_name"]
        self.frames_dir = self.raw["session"]["frames_dir"]
        self.tbd = {t["id"]: t for t in self.raw["tbd_by_probe"]}
        self.payloads = self.raw.get("event_payloads", {}).get("kinds", {})
        self.check_columns()

    def fields(self, stream):
        return self.streams[stream]["fields"]

    def columns(self, stream):
        return [f["name"] for f in self.fields(stream)]

    def key(self, stream):
        return self.streams[stream]["key"][0]

    def required_streams(self):
        return [name for name, spec in self.streams.items() if spec["status"] == "required"]

    def file_fields(self, name):
        return self.files[name]["fields"]

    def check_columns(self):
        for stream, names in DERIVED_SOURCES.items():
            have = set(self.columns(stream))
            missing = [n for n in names if n not in have]
            if missing:
                raise SchemaError("the stream schema no longer defines %s, which the reader's derived views read"
                                  % ", ".join("%s.%s" % (stream, m) for m in missing))


_SCHEMA_CACHE = {}


def load_schema(path=None):
    """The (cached) Schema of path, default docs/capture_contract/dc-2.0.streams.json."""
    key = os.path.realpath(path or STREAMS_JSON)
    if key not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[key] = Schema(key)
    return _SCHEMA_CACHE[key]


def steps_per_decision(decision_hz, step_dt_s):
    """physics_steps_per_decision N of TBD-1 (freeze decision D-A): N = floor(1000 / decision_hz / step_dt_ms + 0.5)
    from the measured engine step dt (session.json engine.step_dt_s), never from GetFixedTimeSlice(): 6 at 5 Hz and
    3 at 10 Hz with the 1/30 s steps of 1.8."""
    step_dt_s = float(step_dt_s)
    if not (step_dt_s > 0.0 and math.isfinite(step_dt_s)):
        raise ValueError("step dt %r is not a positive number" % step_dt_s)
    decision_hz = float(decision_hz)
    if not (decision_hz > 0.0 and math.isfinite(decision_hz)):
        raise ValueError("decision rate %r is not a positive number" % decision_hz)
    return max(int(math.floor(1000.0 / decision_hz / (step_dt_s * 1000.0) + 0.5)), 1)


def grid_period_ms(hz):
    """Integer period of a world-time grid at hz (the contract's rates 5 and 10 Hz divide 1000 ms); None otherwise."""
    if not isinstance(hz, (int, float)) or isinstance(hz, bool) or not hz > 0:
        return None
    period = 1000.0 / float(hz)
    return int(round(period)) if abs(period - round(period)) < 1e-9 else None


def decision_callbacks(world_times, decision_hz):
    """[(row index, grid point)] of the decision callbacks of one ctl stream in row order (contract section 6): grid
    point g is due at anchor + g x 1000 / decision_hz ms, anchor = the first row's world_time_ms; its decision callback
    is the first row with world_time_ms >= its due time; a row that reaches several grid points takes one decision,
    for the newest of them (the others are skipped: grid point gaps). Returns (decisions, skipped grid points)."""
    period = grid_period_ms(decision_hz)
    if period is None or not world_times:
        return [], 0
    anchor = world_times[0]
    out, skipped, g = [], 0, 0
    for i, w in enumerate(world_times):
        if w is None or w < anchor + g * period:
            continue
        reached = (w - anchor) // period
        skipped += reached - g
        out.append((i, reached))
        g = reached + 1
    return out, skipped


def covering_row(steps, covered, q):
    """Index of the row (steps sorted, covered_steps per row) that covers physics step q, or None: a row covers
    physics_step - covered_steps + 1 .. physics_step."""
    i = bisect.bisect_left(steps, q)
    if i < len(steps) and covered[i] is not None and steps[i] - covered[i] + 1 <= q:
        return i
    return None


# ------------------------------------------------------------------------------------------------
# Field parsing (C05)
# ------------------------------------------------------------------------------------------------
def _conv_int(text):
    if not INT_RE.fullmatch(text):
        raise ValueError("%r is not an integer" % text)
    return int(text)


def _conv_float(text):
    if not FLOAT_RE.fullmatch(text):
        raise ValueError("%r is not a finite decimal number without exponent" % text)
    value = float(text)
    if not math.isfinite(value):        # a digit string too long for a double overflows to inf (section 8)
        raise ValueError("%r overflows to a non-finite number" % (text[:24] + "..." if len(text) > 24 else text))
    return value


def _conv_bool(text):
    if text == "0":
        return 0
    if text == "1":
        return 1
    raise ValueError("%r is not 0 or 1" % text)


def _conv_text(text):
    if '"' in text:
        raise ValueError("text holds a double quote")
    return text


_CONVERTERS = {"int": _conv_int, "float": _conv_float, "bool": _conv_bool, "str": _conv_text, "enum": _conv_text}


def field_parser(field):
    """A function text -> value for one CSV field spec of the streams JSON. Empty text is None (only where the
    field is not required). Raises ValueError naming the reason."""
    conv = _CONVERTERS.get(field["type"])
    if conv is None:
        raise SchemaError("field %s has CSV type %s" % (field["name"], field["type"]))
    required = bool(field["required"])
    allowed = frozenset(field["enum"]) if "enum" in field else None

    def parse(text):
        if text == "":
            if required:
                raise ValueError("empty value in a required column")
            return None
        value = conv(text)
        if allowed is not None and value not in allowed:
            raise ValueError("%r is not one of the allowed values" % (value,))
        return value

    return parse


def parse_csv_bytes(data, fields, where, problems, check="C05", refuse_unknown=True):
    """Header-based parse of one CSV file (bytes). Returns (columns_found, rows) where rows is a list of dicts by
    contract column name (absent optional columns are None); problems gets a Problem per refused line.
    A header with an unknown, duplicated or missing required column refuses the whole file (rows = None)."""
    by_name = {f["name"]: f for f in fields}
    parsers = {f["name"]: field_parser(f) for f in fields}
    lines = data.split(b"\n")
    tail = lines.pop()
    if tail != b"":
        problems.append(Problem(check, where, "last line is not terminated by LF (%d bytes dropped)" % len(tail)))
    if not lines:
        problems.append(Problem(check, where, "no header line"))
        return [], None
    try:
        header_text = lines[0].decode("ascii")
    except UnicodeDecodeError:
        problems.append(Problem(check, where, "header is not ASCII"))
        return [], None
    if "\r" in header_text:
        problems.append(Problem(check, where, "header line ends with CR (LF line ends only)"))
        return [], None
    columns = header_text.split(",")
    unknown = [c for c in columns if c not in by_name]
    dups = sorted(set(c for c in columns if columns.count(c) > 1))
    missing = [f["name"] for f in fields if f["required"] and f["name"] not in columns]
    if (unknown and refuse_unknown) or dups or missing:
        parts = []
        if missing:
            parts.append("missing required column(s) %s" % ", ".join(missing))
        if unknown:
            parts.append("unknown column(s) %s" % ", ".join(unknown))
        if dups:
            parts.append("duplicated column(s) %s" % ", ".join(dups))
        problems.append(Problem(check, where, "header refused: " + "; ".join(parts)))
        return columns, None
    col_parsers = [parsers.get(c) for c in columns]
    absent = [f["name"] for f in fields if f["name"] not in columns]
    rows = []
    for number, raw in enumerate(lines[1:], start=2):
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            problems.append(Problem(check, "%s:%d" % (where, number), "line is not ASCII"))
            continue
        if "\r" in text:
            problems.append(Problem(check, "%s:%d" % (where, number), "line ends with CR (LF line ends only)"))
            continue
        values = text.split(",")
        if len(values) != len(columns):
            problems.append(Problem(check, "%s:%d" % (where, number),
                                    "%d fields, header has %d" % (len(values), len(columns))))
            continue
        row = {}
        bad = None
        for name, parse, value in zip(columns, col_parsers, values):
            if parse is None:
                continue
            try:
                row[name] = parse(value)
            except ValueError as e:
                bad = "column %s: %s" % (name, e)
                break
        if bad is not None:
            problems.append(Problem(check, "%s:%d" % (where, number), bad))
            continue
        for name in absent:
            row[name] = None
        rows.append(row)
    return columns, rows


# ------------------------------------------------------------------------------------------------
# JSON session files (C01)
# ------------------------------------------------------------------------------------------------
def dotted_get(obj, dotted):
    """(found, value) of a dotted key path in nested dicts."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _json_type_ok(value, field):
    ftype = field["type"]
    if ftype == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif ftype == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    elif ftype == "bool":
        ok = isinstance(value, bool) or (isinstance(value, int) and value in (0, 1))
    elif ftype in ("str", "enum"):
        ok = isinstance(value, str)
    elif ftype == "object":
        ok = isinstance(value, dict)
    elif ftype == "array":
        ok = isinstance(value, list)
    else:
        ok = False
    if ok and "enum" in field:
        ok = value in field["enum"]
    return ok


def validate_json_file(obj, fields, where):
    """Problems (check C01) of one session JSON object against its field list: every required key present, null
    only where nullable, types and enums."""
    problems = []
    if not isinstance(obj, dict):
        return [Problem("C01", where, "not a JSON object")]
    for field in fields:
        found, value = dotted_get(obj, field["name"])
        if not found:
            if field["required"]:
                problems.append(Problem("C01", where, "missing key %s" % field["name"]))
            continue
        if value is None:
            if not field.get("nullable"):
                problems.append(Problem("C01", where, "key %s is null" % field["name"]))
            continue
        if not _json_type_ok(value, field):
            problems.append(Problem("C01", where, "key %s: %r is not a valid %s%s" % (
                field["name"], value, field["type"], " of %s" % field["enum"] if "enum" in field else "")))
    return problems


def _read_json(path, where, problems):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        problems.append(Problem("C01", where, "cannot read JSON: %s" % e))
        return None


# ------------------------------------------------------------------------------------------------
# Streams (committed chunks only)
# ------------------------------------------------------------------------------------------------
class Stream:
    """The committed rows of one stream, by contract column name (lists; None = null), plus its chunk bookkeeping.

    commits: the commit.csv lines; chunks: per committed chunk {chunk_id, file, exists, size, n_rows, first_key,
    last_key, refused}; uncommitted: chunk files with no commit line (discarded)."""

    def __init__(self, name, spec, present):
        self.name = name
        self.spec = spec
        self.present = present
        self.columns = [f["name"] for f in spec["fields"]]
        self.key = spec["key"][0]
        self.data = {c: [] for c in self.columns}
        self.commit_file = False
        self.commits = []
        self.chunks = []
        self.uncommitted = []
        self._arrays = {}
        self._index = None

    def __len__(self):
        return len(self.data[self.key])

    def append_rows(self, rows):
        for row in rows:
            for c in self.columns:
                self.data[c].append(row[c])
        self._arrays = {}
        self._index = None

    def col(self, name):
        return self.data[name]

    def array(self, name):
        """Column as a float64 numpy array, NaN for null (cached)."""
        if name not in self._arrays:
            values = self.data[name]
            self._arrays[name] = numpy.array([numpy.nan if v is None else float(v) for v in values], dtype=float)
        return self._arrays[name]

    def index(self):
        """key -> row index of the first row with that key."""
        if self._index is None:
            self._index = {}
            for i, k in enumerate(self.data[self.key]):
                self._index.setdefault(k, i)
        return self._index

    def row(self, i):
        return {c: self.data[c][i] for c in self.columns}


def inside(root, path):
    """True when path resolves (symlinks followed) to an entry strictly below the directory root (a realpath)."""
    return os.path.realpath(path).startswith(root.rstrip("/") + "/")


def read_stream(session_dir, name, schema, problems):
    """Committed rows of one stream directory (section 8). Commit/chunk mismatches are C34 problems, header and row
    problems C05, a directory or file that resolves outside the session directory C01; none raises here.
    session_dir is a realpath."""
    spec = schema.streams[name]
    sdir = os.path.join(session_dir, spec["dir"])
    stream = Stream(name, spec, os.path.isdir(sdir))
    if not stream.present:
        return stream
    if not inside(session_dir, sdir):
        problems.append(Problem("C01", spec["dir"] + "/", "resolves outside the session directory; not read"))
        return stream
    on_disk = {}
    for entry in sorted(os.listdir(sdir)):
        m = CHUNK_RE.match(entry)
        if m:
            on_disk[int(m.group(1))] = entry
    commit_path = os.path.join(sdir, schema.commit_name)
    where_commit = "%s/%s" % (spec["dir"], schema.commit_name)
    if os.path.lexists(commit_path) and not inside(session_dir, commit_path):
        problems.append(Problem("C01", where_commit, "resolves outside the session directory; not read"))
        stream.uncommitted = [on_disk[i] for i in sorted(on_disk)]
        return stream
    if not os.path.isfile(commit_path):
        if on_disk:
            problems.append(Problem("C34", where_commit, "missing: none of the %d chunk files is acknowledged"
                                    % len(on_disk)))
        stream.uncommitted = [on_disk[i] for i in sorted(on_disk)]
        return stream
    stream.commit_file = True
    with open(commit_path, "rb") as f:
        commit_bytes = f.read()
    _, commit_rows = parse_csv_bytes(commit_bytes, schema.file_fields("commit.csv"), where_commit, problems,
                                     check="C34")
    commit_rows = commit_rows or []
    stream.commits = commit_rows
    seen = set()
    for expected_id, commit in enumerate(commit_rows):
        chunk_id = commit["chunk_id"]
        if chunk_id != expected_id:
            problems.append(Problem("C34", where_commit, "commit line %d names chunk %d (chunk ids are 0-based, +1 "
                                    "per commit)" % (expected_id, chunk_id)))
        if chunk_id in seen:
            problems.append(Problem("C34", where_commit, "chunk %d committed twice" % chunk_id))
            continue
        seen.add(chunk_id)
        fname = schema.chunk_name % chunk_id
        where = "%s/%s" % (spec["dir"], fname)
        info = {"chunk_id": chunk_id, "file": fname, "exists": False, "size": None, "n_rows": None,
                "first_key": None, "last_key": None, "refused": False, "commit": commit}
        stream.chunks.append(info)
        if commit["n_rows"] == 0:
            problems.append(Problem("C34", where_commit, "chunk %d committed with 0 rows (a chunk is written only "
                                    "when it holds at least one row, section 8)" % chunk_id))
        path = os.path.join(sdir, fname)
        if os.path.lexists(path) and not inside(session_dir, path):
            problems.append(Problem("C01", where, "resolves outside the session directory; not read"))
            continue
        if not os.path.isfile(path):
            problems.append(Problem("C34", where, "committed chunk file is missing"))
            continue
        with open(path, "rb") as f:
            data = f.read()
        info["exists"] = True
        info["size"] = len(data)
        info["n_rows"] = max(data.count(b"\n") - 1, 0)
        if info["size"] != commit["n_bytes"]:
            problems.append(Problem("C34", where, "%d bytes on disk, commit line says %d"
                                    % (info["size"], commit["n_bytes"])))
        if info["n_rows"] != commit["n_rows"]:
            problems.append(Problem("C34", where, "%d data lines on disk, commit line says %d"
                                    % (info["n_rows"], commit["n_rows"])))
        _, rows = parse_csv_bytes(data, spec["fields"], where, problems)
        if rows is None:
            info["refused"] = True
            continue
        if rows:
            info["first_key"] = rows[0][stream.key]
            info["last_key"] = rows[-1][stream.key]
            if info["first_key"] != commit["first_key"] or info["last_key"] != commit["last_key"]:
                problems.append(Problem("C34", where, "keys %s..%s on disk, commit line says %s..%s" % (
                    info["first_key"], info["last_key"], commit["first_key"], commit["last_key"])))
        stream.append_rows(rows)
    stream.uncommitted = [on_disk[i] for i in sorted(on_disk) if i not in seen]
    return stream


# ------------------------------------------------------------------------------------------------
# Session
# ------------------------------------------------------------------------------------------------
def _path_parts(path):
    return os.path.normpath(os.path.abspath(path)).replace("\\", "/").lower().split("/")


def check_paths(path, sealed_eval):
    """realpath of a session path; ReaderError on the C: drive, SealedMapError under heldout/ or a SEALED directory
    unless sealed_eval (the --legacy self-test calls it too)."""
    real = os.path.realpath(path)
    probe = real if real.endswith("/") else real + "/"
    if any(probe.startswith(root) for root in FORBIDDEN_ROOTS):
        raise ReaderError("refusing %s: no repo tool reads session data from the C: drive (E: copies only)" % path)
    if not sealed_eval and any(p in SEALED_PATH_PARTS for p in _path_parts(real)):
        raise SealedMapError("%s lies under a heldout/ or SEALED directory; it is read only on the sealed evaluation "
                             "path (sealed_eval=True)" % path)
    return real


def is_legacy_session(path):
    """True for a dc-1.0 session directory (session_<n>, telemetry.csv / summary.txt, no session.json)."""
    if os.path.isfile(os.path.join(path, "session.json")):
        return False
    if LEGACY_DIR_RE.match(os.path.basename(os.path.normpath(path))):
        return True
    return any(os.path.exists(os.path.join(path, m)) for m in LEGACY_MARKERS)


class Session:
    """One dc-2.0 session as read by open_session. The main attributes:

    meta, end, harvest       session.json, session_end.json (None when unsealed), harvest.json (or None)
    session_id, map_id, terrain, trial_id, sealed
    clocks                   dict: N, step_dt_s, step_dt_source, decision_hz, image_hz, image_hz_setting, slice_s
                             (raw GetFixedTimeSlice, provenance only), img_lead_ms, img_inflight_timeout_ms,
                             chunk_max_s, image_period_ms, decision_period_ms
    grid_anchor_ms           world_time_ms of the first ctl row (the decision and image grids' anchor)
    streams                  {name: Stream} of committed rows; heartbeat: list of row dicts
    images                   img_resolved rows (one per img row, in img order)
    ctl_export, telemetry_export   per-row derived views (dicts of numpy arrays)
    decisions                decision samples (list of dicts; see _build_decisions)
    crash_events, terminal_ego, terminal_ego_confirmed   parsed crash-event rows and each episode's first terminal
                             ego event by onset and by confirmation (9.5.1)
    segments                 list of {episode_id, first_decision_step, last_decision_step, n_steps}
    problems                 list of Problem (check, where, message)
    salvageable              the problems that salvage=True excuses (an unsealed or failed seal and its tail)
    """

    def __init__(self, path, schema):
        self.path = path
        self.schema = schema
        self.problems = []
        self.salvageable = []
        self.notes = []
        self.meta = None
        self.end = None
        self.harvest = None
        self.heartbeat = []
        self.streams = {}
        self.images = []
        self.image_suffix = IMAGE_SUFFIX
        self.vehicle_card = None
        self.crash_events = []
        self.terminal_ego = {}
        self.terminal_ego_confirmed = {}

    # -- small helpers --------------------------------------------------------------------------
    def meta_get(self, dotted, default=None):
        found, value = dotted_get(self.meta or {}, dotted)
        return value if found else default

    def end_get(self, dotted, default=None):
        found, value = dotted_get(self.end or {}, dotted)
        return value if found else default

    def problems_for(self, check):
        return [p for p in self.problems if p.check == check]

    def build_key(self):
        """build_key from session.json, else from the harvester's harvest.json (section 11); None when neither."""
        key = self.meta_get("pin.build_key")
        if key:
            return key
        if isinstance(self.harvest, dict) and self.harvest.get("build_key"):
            return self.harvest["build_key"]
        return None

    def image_path(self, slot_id):
        for row in self.images:
            if row["slot_id"] == slot_id:
                return os.path.join(self.path, row["file"]) if row["file"] else None
        return None

    def load_image(self, slot_id, region="core"):
        """float32 [3, H, W] in [0, 1] of one slot's image: region "core" drops the rig.band_px fiducial bands on
        both sides, "full" keeps them. ValueError when the slot has no readable image."""
        path = self.image_path(slot_id)
        if not path or not os.path.isfile(path):
            raise ValueError("slot %r has no image file" % (slot_id,))
        rgb = fiducial.load_rgb(path)
        if region == "core":
            band = int(self.meta_get("rig.band_px", 0) or 0)
            rgb = rgb[:, band:rgb.shape[1] - band, :]
        elif region != "full":
            raise ValueError("region must be core or full")
        return numpy.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(numpy.float32) / 255.0

    def golden_view(self, pixel_slots=(), stride=16):
        """The comparable summary that tests/fixtures/capture_dc2/golden.json holds (see make_synthetic.py)."""
        counts = {name: len(s) for name, s in sorted(self.streams.items()) if s.present}
        clocks = {k: self.clocks[k] for k in ("physics_steps_per_decision", "decision_hz", "image_hz",
                                               "image_hz_setting", "img_lead_ms", "step_dt_s")}
        images = [{k: r[k] for k in ("slot_id", "file", "file_exists", "png_w", "png_h", "fid_ok",
                                     "captured_render_frame")} for r in self.images]
        decisions = []
        for d in self.decisions:
            decisions.append({k: d[k] for k in GOLDEN_DECISION_KEYS})
        pixels = []
        for slot in pixel_slots:
            core = self.load_image(slot, "core")
            pixels.append({"slot_id": slot, "core": core[:, ::stride, ::stride].tolist()})
        return {"schema": GOLDEN_SCHEMA, "session_id": self.session_id, "contract_version": CONTRACT_VERSION,
                "clocks": clocks, "counts": counts, "img_resolved": images, "segments": list(self.segments),
                "decision_steps": decisions, "pixels": {"stride": stride, "slots": pixels}}


GOLDEN_SCHEMA = "drivedit.dc2-golden/1"
GOLDEN_DECISION_KEYS = ("decision_step", "grid_point", "episode_id", "hl_seq", "interval_steps", "state",
                        "action_cmd", "action_label", "action_exec", "obs_slot_id", "obs_captured_render_frame",
                        "obs_last_physics_step", "obs_age_steps", "obs_age_ms", "obs_valid", "masks")


def open_session(path, sealed_eval=False, strict=True, schema=None, decode_images=True, vehicle_card=None,
                 salvage=False, image_visitor=None, image_suffix=IMAGE_SUFFIX):
    """Read one dc-2.0 session directory.

    sealed_eval     True only on the sealed evaluation path (Everon, heldout/, SEALED).
    strict          True: raise ReaderError on any C01/C02/C05/C07/C34 problem and on an unsealed or failed seal
                    (builders); False: record problems in Session.problems (validator).
    decode_images   decode every image's fiducial (needs cv2); False leaves img_resolved without codes.
    vehicle_card    dict or path of the vehicle card (gear_indices_forward ...); default
                    tools/ident/cards/<pin_id>/<control.vehicle_card_id>.json when it exists. Without a card
                    dir_state is None (the gear mapping is read back, never assumed; TBD-7). A card whose sha256
                    differs from control.vehicle_card_sha (or a dict card when the session names a sha) is a C02
                    problem and is not used.
    salvage         with strict=True, read the committed data of an unsealed session or of a seal the writer marked
                    failed (crash salvage) anyway; every other problem still raises.
    image_visitor   optional callable(session, img_resolved_row, rgb_uint8_hwc), called once per readable image in
                    slot order after the stream joins exist (the validator's image checks use it, so each image is
                    read once).
    image_suffix    the suffix the engine appends to an img request path (TBD-4): one of IMAGE_SUFFIXES, default
                    IMAGE_SUFFIX. ValueError for any other.
    """
    if image_suffix not in IMAGE_SUFFIXES:
        raise ValueError("image_suffix %r is not one of the TBD-4 names %s" % (image_suffix, IMAGE_SUFFIXES))
    schema = schema or load_schema()
    real = check_paths(path, sealed_eval)
    if not os.path.isdir(real):
        raise ReaderError("not a directory: %s" % path)
    if is_legacy_session(real):
        raise LegacySessionError("%s is a dc-1.0 session (DrivingData/session_<n>: telemetry.csv, no session.json); "
                                 "the dc-2.0 reader never reads dc-1.0 data" % path)
    base = os.path.basename(os.path.normpath(real))
    if base.startswith("everon_") and not sealed_eval:
        raise SealedMapError("%s is an Everon session; it is read only on the sealed evaluation path "
                             "(sealed_eval=True)" % path)
    s = Session(real, schema)
    s.image_suffix = image_suffix
    meta = _read_session_json(s, "session.json")
    if meta is None:
        raise ReaderError("%s: session.json is missing or unreadable (%s)" % (
            path, "; ".join(p.message for p in s.problems)))
    s.meta = meta
    version = meta.get("contract_version") if isinstance(meta, dict) else None
    if isinstance(version, str) and version.startswith("dc-1"):
        raise LegacySessionError("%s: contract_version %s; the dc-2.0 reader never reads dc-1.0 data" % (path, version))
    if version != CONTRACT_VERSION:
        raise ReaderError("%s: contract_version %r is not %s" % (path, version, CONTRACT_VERSION))
    s.problems += validate_json_file(meta, schema.file_fields("session.json"), "session.json")
    s.session_id = meta.get("session_id")
    s.map_id = s.meta_get("map.map_id")
    s.terrain = s.meta_get("map.terrain")
    s.trial_id = s.meta_get("trial.trial_id")
    root_world = s.meta_get("map.root_world")
    eden_rooted = isinstance(root_world, str) and bool(EDEN_ROOT_RE.match(root_world))
    everon_id = isinstance(s.session_id, str) and s.session_id.startswith("everon_")
    if (s.map_id == "everon" or s.terrain == "Eden" or eden_rooted or everon_id) and not sealed_eval:
        raise SealedMapError("%s: map_id %r / terrain %r / root_world %r / session_id %r is the sealed test map "
                             "(every Eden-rooted world counts as Everon); it is read only on the sealed evaluation "
                             "path (sealed_eval=True)" % (path, s.map_id, s.terrain, root_world, s.session_id))
    if s.map_id not in schema.maps:
        raise ReaderError("%s: session.json map.map_id %r is not a canonical map id (%s)" % (
            path, s.map_id, ", ".join(schema.maps)))
    _check_identity(s, base)
    s.end = _read_session_json(s, "session_end.json")
    if s.end is not None:
        s.problems += validate_json_file(s.end, schema.file_fields("session_end.json"), "session_end.json")
    s.sealed = s.end is not None
    s.harvest = _read_session_json(s, "harvest.json")
    _read_clocks(s)
    hb_path = os.path.join(real, "heartbeat.csv")
    if os.path.lexists(hb_path) and not inside(real, hb_path):
        s.problems.append(Problem("C01", "heartbeat.csv", "resolves outside the session directory; not read"))
    elif os.path.isfile(hb_path):
        with open(hb_path, "rb") as f:
            _, rows = parse_csv_bytes(f.read(), schema.file_fields("heartbeat.csv"), "heartbeat.csv", s.problems)
        s.heartbeat = rows or []
    for name in schema.streams:
        s.streams[name] = read_stream(real, name, schema, s.problems)
        if not s.streams[name].present and schema.streams[name]["status"] == "required":
            s.problems.append(Problem("C01", schema.streams[name]["dir"] + "/", "required stream directory missing"))
    _check_key_order(s)
    _check_seal(s)
    s.vehicle_card = _load_card(s, vehicle_card)
    _image_files(s)
    if strict:
        fatal = [p for p in s.problems if p.check in STRICT_FATAL and not (salvage and p in s.salvageable)]
        if fatal:
            raise ReaderError("%s: %d problem(s), first: %s %s: %s" % (
                path, len(fatal), fatal[0].check, fatal[0].where, fatal[0].message))
        if not s.sealed and not salvage:
            raise ReaderError("%s is not sealed (no session_end.json); pass salvage=True to salvage it" % path)
    _build_exports(s)
    _build_hf_joins(s)
    _resolve_images(s, decode_images, image_visitor)
    _read_crash_events(s)
    _build_decisions(s)
    return s


def _read_session_json(s, name):
    """One JSON file of the session directory, or None when absent; a file that resolves outside the directory or
    does not parse is a C01 problem."""
    path = os.path.join(s.path, name)
    if not os.path.lexists(path):
        return None
    if not inside(s.path, path):
        s.problems.append(Problem("C01", name, "resolves outside the session directory; not read"))
        return None
    if not os.path.isfile(path):
        s.problems.append(Problem("C01", name, "not a regular file"))
        return None
    return _read_json(path, name, s.problems)


def _check_key_order(s):
    """C07 problem per stream whose key (composite for traffic) is not strictly increasing in row order: a
    duplicated or out-of-order row. Stream.index() keeps the first row of a key, so strict readers must refuse."""
    for name, st in s.streams.items():
        keys = list(zip(*[st.data[k] for k in st.spec["key"]]))
        bad = sum(1 for a, b in zip(keys, keys[1:]) if not b > a)
        if bad:
            s.problems.append(Problem("C07", st.spec["dir"] + "/", "key %s not strictly increasing at %d row(s) "
                                      "(duplicated or out-of-order rows)" % ("+".join(st.spec["key"]), bad)))


def _check_seal(s):
    """The seal against the committed data (sections 5.3, 8, 10.2): C01 for a failed status or an error end_reason
    (salvageable), the contract version and the episodes count; C34 when the committed ctl, telemetry or hf rows do
    not reach end.physics_step / end.render_frame (a lost or truncated tail; salvageable, since a failed session
    stops acknowledging rows). ctl and telemetry hold the same keys, so both must reach end.physics_step."""
    if s.end is None or not isinstance(s.end, dict):
        return
    status, reason = s.end_get("status"), s.end_get("end_reason")
    if status != "ok" or reason in ERROR_END_REASONS:
        p = Problem("C01", "session_end.json", "the writer sealed the session with status %r, end_reason %r: not a "
                    "complete session (sections 5.3 and 8; salvage only)" % (status, reason))
        s.problems.append(p)
        s.salvageable.append(p)
    if s.end_get("contract_version") != CONTRACT_VERSION:
        s.problems.append(Problem("C01", "session_end.json", "contract_version %r is not %s"
                                  % (s.end_get("contract_version"), CONTRACT_VERSION)))
    episodes = s.end_get("episodes")
    n_episodes = len(s.streams["episodes"]) if "episodes" in s.streams else 0
    if episodes != n_episodes:
        s.problems.append(Problem("C01", "session_end.json", "episodes %r, %d committed episodes row(s)"
                                  % (episodes, n_episodes)))
    end_step, end_frame = s.end_get("end.physics_step"), s.end_get("end.render_frame")
    for stream, key, want, what in (("ctl", "physics_step", end_step, "end.physics_step"),
                                    ("telemetry", "physics_step", end_step, "end.physics_step"),
                                    ("hf", "render_frame", end_frame, "end.render_frame")):
        keys = s.streams[stream].col(key) if stream in s.streams else []
        last = max(keys) if keys else None
        if last != want:
            p = Problem("C34", "%s/" % stream, "committed rows end at %s %r, the session ends at %r (%s): a lost or "
                        "truncated tail" % (key, last, want, what))
            s.problems.append(p)
            s.salvageable.append(p)


def _check_identity(s, base):
    schema = s.schema
    sid = s.session_id
    if not isinstance(sid, str) or not schema.id_re.match(sid):
        s.problems.append(Problem("C01", "session.json", "session_id %r does not match the session id rule" % (sid,)))
        return
    if sid != base:
        s.problems.append(Problem("C01", "session.json", "session_id %s differs from the directory name %s"
                                  % (sid, base)))
    if not sid.startswith(s.map_id + "_"):
        s.problems.append(Problem("C01", "session.json", "session_id %s does not start with map_id %s"
                                  % (sid, s.map_id)))
    card = schema.maps.get(s.map_id)
    if card is not None:
        if s.terrain != card["terrain"]:
            s.problems.append(Problem("C01", "session.json", "map %s has terrain %s, session.json says %r"
                                      % (s.map_id, card["terrain"], s.terrain)))
        if s.meta_get("map.root_world") != card["root_world"]:
            s.problems.append(Problem("C01", "session.json", "map %s has root world %s, session.json says %r"
                                      % (s.map_id, card["root_world"], s.meta_get("map.root_world"))))


def _read_clocks(s):
    get = s.meta_get
    clocks = {
        "physics_steps_per_decision": get("clocks.physics_steps_per_decision"),
        "decision_hz": get("clocks.decision_hz"),
        "image_hz": get("clocks.image_hz"),
        "image_hz_setting": get("clocks.image_hz_setting"),
        "img_lead_ms": get("clocks.img_lead_ms"),
        "img_inflight_timeout_ms": get("clocks.img_inflight_timeout_ms"),
        "chunk_max_s": get("clocks.chunk_max_s"),
        "step_dt_s": get("engine.step_dt_s"),
        "step_dt_source": get("engine.step_dt_source"),
        "slice_s": get("engine.fixed_time_slice_s"),
    }
    n, dhz, ihz, dt = (clocks["physics_steps_per_decision"], clocks["decision_hz"], clocks["image_hz"],
                       clocks["step_dt_s"])
    ok = (isinstance(n, int) and not isinstance(n, bool) and n >= 1 and grid_period_ms(dhz) is not None
          and grid_period_ms(ihz) is not None and isinstance(dt, (int, float)) and not isinstance(dt, bool)
          and dt > 0)
    if not ok:
        raise ReaderError("%s: session.json clocks are unusable (physics_steps_per_decision %r, decision_hz %r, "
                          "image_hz %r, engine.step_dt_s %r); the grid rates must divide 1000 ms"
                          % (s.path, n, dhz, ihz, dt))
    clocks["N"] = n
    clocks["image_period_ms"] = 1000.0 / float(ihz)
    clocks["decision_period_ms"] = grid_period_ms(dhz)
    s.clocks = clocks
    s.steer_sign = get("control.steer_sign", 1)
    s.yaw_axis_sign = get("control.yaw_axis_sign", 1)


def _load_card(s, card):
    """The vehicle card (TBD-7), or None. Its sha256 must equal session.json control.vehicle_card_sha when the
    session names one: a mismatch (or a dict card, whose bytes cannot be hashed) is a C02 problem and the card is not
    used, so dir_state is never read through another card."""
    sha = s.meta_get("control.vehicle_card_sha")
    if isinstance(card, dict):
        if sha:
            s.problems.append(Problem("C02", "control.vehicle_card_sha", "a vehicle card passed as a dict cannot be "
                                      "checked against control.vehicle_card_sha %s; not used" % str(sha)[:12]))
            return None
        s.notes.append("vehicle card passed as a dict; the session names no vehicle_card_sha")
        return card
    path = card
    if path is None:
        card_id = s.meta_get("control.vehicle_card_id")
        pin_id = s.meta_get("pin.pin_id")
        if isinstance(card_id, str) and isinstance(pin_id, str) and "/" not in card_id and "/" not in pin_id \
                and ".." not in card_id and ".." not in pin_id:
            candidate = os.path.join(CARDS_DIR, pin_id, card_id + ".json")
            path = candidate if os.path.isfile(candidate) else None
    if path is None:
        s.notes.append("no vehicle card: dir_state is null (TBD-7)")
        return None
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if sha and digest != sha:
        s.problems.append(Problem("C02", "control.vehicle_card_sha", "vehicle card %s has sha256 %s, session.json "
                                  "names %s; not used" % (path, digest[:12], str(sha)[:12])))
        return None
    if not sha:
        s.notes.append("vehicle card %s used; the session names no vehicle_card_sha" % path)
    return json.loads(raw.decode("utf-8"))


def dir_state_of(card, gear):
    """dir_state of an engine gear index through the vehicle card (TBD-7, section 7): +1 in a forward gear, -1 in the
    reverse gear, 0 in the neutral gear. While the card's gear_index_neutral is null (T0-MIN, forward-only) every
    other gear is 0 (standstill in an unmapped gear); once neutral is measured, a gear the card does not name is
    None (unknown), never guessed."""
    if gear is None:
        return None
    forward = set(card.get("gear_indices_forward") or [])
    neutral, reverse = card.get("gear_index_neutral"), card.get("gear_index_reverse")
    if gear in forward:
        return 1
    if reverse is not None and gear == reverse:
        return -1
    if neutral is None or gear == neutral:
        return 0
    return None


# ------------------------------------------------------------------------------------------------
# Images (img_resolved)
# ------------------------------------------------------------------------------------------------
def fid_margin(image, result):
    """Smallest nearest-colour margin over the 12 patches of a decoded image: (CHANNEL_MARGIN - the worst channel
    median's distance to 0 or 255) / 255. None when the image did not decode."""
    if result.status != fiducial.STATUS_OK:
        return None
    height, width = image.shape[:2]
    geometry = fiducial.layout(width, height)
    offset = (fiducial.PATCH_PX - fiducial.CENTRE_PX) // 2
    worst = 0.0
    for band in ("left", "right"):
        for x0, y0 in geometry[band]:
            block = image[y0 + offset:y0 + offset + fiducial.CENTRE_PX, x0 + offset:x0 + offset + fiducial.CENTRE_PX,
                          0:3].reshape(-1, 3)
            medians = numpy.median(block, axis=0)
            worst = max(worst, float(numpy.max(numpy.minimum(medians, 255.0 - medians))))
    return (fiducial.CHANNEL_MARGIN - worst) / 255.0


def _image_files(s):
    """(file, error) per img row, before anything is read: path must be exactly frames/f_<slot_id:08d> (TBD-4; else
    a C11 problem), the file is path + s.image_suffix and it must resolve inside the session directory (else C01).
    Stored in s.image_files."""
    img = s.streams["img"]
    s.image_files = []
    for slot, path in zip(img.data["slot_id"], img.data["path"]):
        if path is None:
            s.image_files.append((None, None))
            continue
        want = "%s/f_%08d" % (s.schema.frames_dir, slot)
        if path != want:
            s.problems.append(Problem("C11", "img slot %d" % slot, "path %r is not %s (TBD-4); not read"
                                      % (path, want)))
            s.image_files.append((None, "path %r is not %s" % (path, want)))
            continue
        rel = path + s.image_suffix
        full = os.path.join(s.path, rel)
        if os.path.lexists(full) and not inside(s.path, full):
            s.problems.append(Problem("C01", rel, "resolves outside the session directory; not read"))
            s.image_files.append((None, "file resolves outside the session directory"))
            continue
        s.image_files.append((rel, None))


def _resolve_images(s, decode, visitor):
    img = s.streams["img"]
    hf_index = s.streams["hf"].index()
    end_frame = s.end_get("end.render_frame") if s.end is not None else None
    s.images = []
    for i in range(len(img)):
        slot = img.data["slot_id"][i]
        row = {"slot_id": slot, "issue_status": img.data["issue_status"][i],
               "issue_render_frame": img.data["issue_render_frame"][i],
               "done_render_frame": img.data["done_render_frame"][i], "file": None, "file_exists": 0,
               "png_w": None, "png_h": None, "fid_ok": 0, "fid_status": None, "code": None,
               "captured_render_frame": None, "fid_margin": None, "hf_index": None, "pose_hf_index": None,
               "pose_after_seal": 0, "error": None}
        s.images.append(row)
        rel, error = s.image_files[i]
        if rel is None:
            row["error"] = error
            continue
        row["file"] = rel
        full = os.path.join(s.path, rel)
        if not os.path.isfile(full):
            row["error"] = "file missing"
            continue
        if not decode:
            row["file_exists"] = 1
            continue
        try:
            rgb = fiducial.load_rgb(full)
        except ValueError as e:
            row["error"] = str(e)
            continue
        row["file_exists"] = 1
        row["png_h"], row["png_w"] = int(rgb.shape[0]), int(rgb.shape[1])
        result = fiducial.decode(rgb)
        row["fid_status"] = result.status
        if result.status == fiducial.STATUS_OK and row["issue_render_frame"] is not None:
            row["fid_ok"] = 1
            row["code"] = result.code
            row["captured_render_frame"] = fiducial.unwrap(result.code, row["issue_render_frame"])
            row["fid_margin"] = fid_margin(rgb, result)
            row["hf_index"] = hf_index.get(row["captured_render_frame"])
            row["pose_hf_index"] = hf_index.get(row["captured_render_frame"] + 1)     # the image's pose row
            # captured in the last render frame: a seal without drain caught the request in flight, no pose row exists
            row["pose_after_seal"] = int(isinstance(end_frame, int) and row["captured_render_frame"] >= end_frame)
        if visitor is not None:
            visitor(s, row, rgb)


# ------------------------------------------------------------------------------------------------
# Derived views, joins, decisions
# ------------------------------------------------------------------------------------------------
def _signed(values, sign):
    return [None if v is None else sign * v for v in values]


def _build_exports(s):
    ctl = s.streams["ctl"]
    s.ctl_export = {
        "physics_step": list(ctl.data["physics_step"]),
        "steer_cmd": _signed(ctl.data["llc_steer_lim"], s.steer_sign),
        "steer_input": _signed(ctl.data["input_post_steer"], s.steer_sign),
    }
    tel = s.streams["telemetry"]
    fwd_x, fwd_z = tel.array("fwd_x"), tel.array("fwd_z")
    heading = numpy.arctan2(fwd_x, fwd_z)
    heading[heading == -numpy.pi] = numpy.pi
    speed = tel.array("vel_local_z")
    yaw = s.yaw_axis_sign * tel.array("angvel_local_y")
    with numpy.errstate(invalid="ignore", divide="ignore"):
        kappa = numpy.where(numpy.abs(speed) > 3.0, yaw / speed, numpy.nan)
    if s.vehicle_card is None:
        dir_state = [None] * len(tel)
    else:
        dir_state = [dir_state_of(s.vehicle_card, g) for g in tel.data["gear"]]
    s.telemetry_export = {
        "physics_step": list(tel.data["physics_step"]),
        "heading_rad": heading,
        "speed_fwd_mps": speed,
        "yaw_rate_rps": yaw,
        "acc_long_mps2": tel.array("acc_local_z"),
        "acc_lat_mps2": tel.array("acc_local_x"),
        "kappa_realised": kappa,
        "dir_state": dir_state,
    }


def _build_hf_joins(s):
    """hf -> ctl and hf -> telemetry, both exact on last_physics_step (a row per callback): row indices or None."""
    hf = s.streams["hf"]
    ctl_index = s.streams["ctl"].index()
    tel_index = s.streams["telemetry"].index()
    s.hf_ctl = []
    s.hf_tel = []
    for lps in hf.data["last_physics_step"]:
        s.hf_ctl.append(ctl_index.get(lps))
        s.hf_tel.append(tel_index.get(lps))
    ctl = s.streams["ctl"]
    s.grid_anchor_ms = ctl.data["world_time_ms"][0] if len(ctl) else None


def _event_times(s, kinds):
    ev = s.streams["events"]
    return [(ev.data["physics_step"][i], ev.data["world_time_ms"][i])
            for i in range(len(ev)) if ev.data["kind"][i] in kinds]


# ------------------------------------------------------------------------------------------------
# Crash events (section 9.5.1)
# ------------------------------------------------------------------------------------------------
def parse_payload(text):
    """[(key, value text)] of an events.payload: 'key=value' pairs joined by ';' (section 9.5); None or '' is [].
    Raises ValueError for a pair without '=', an empty key or value (a key without a value is left out, 9.5.1)."""
    if not text:
        return []
    pairs = []
    for part in text.split(";"):
        if "=" not in part:
            raise ValueError("pair %r has no '='" % part)
        key, value = part.split("=", 1)
        if not key or value == "":
            raise ValueError("pair %r has an empty key or value" % part)
        pairs.append((key, value))
    return pairs


def payload_value(spec, text):
    """Typed value of one payload key (event_payloads key spec: type, enum, range, min); ValueError when it does not
    parse or lies outside its range or below its minimum."""
    t = spec["type"]
    if t in ("int", "float"):
        value = _conv_int(text) if t == "int" else _conv_float(text)
        if "range" in spec and not spec["range"][0] <= value <= spec["range"][1]:
            raise ValueError("%s is outside [%s, %s]" % (text, spec["range"][0], spec["range"][1]))
        if "min" in spec and value < spec["min"]:
            raise ValueError("%s is below the minimum %s" % (text, spec["min"]))
        return value
    if t == "bool":
        return _conv_bool(text)
    if "=" in text or ";" in text or "," in text or '"' in text:
        raise ValueError("%r holds a separator" % text)
    if t == "enum" and text not in spec["enum"]:
        raise ValueError("%r is not one of %s" % (text, ", ".join(spec["enum"])))
    return text


def check_payload(kind, text, payloads):
    """(values, problems) of one events row of a kind in event_payloads: values {key: typed value} of the keys that
    parse; problems name a payload that does not split, a repeated or unknown key, a missing required key and a
    value that does not parse as its type or enum (C36)."""
    keys = {k["key"]: k for k in payloads[kind]}
    try:
        pairs = parse_payload(text)
    except ValueError as e:
        return {}, ["payload %r: %s" % (text, e)]
    values, problems, seen = {}, [], set()
    for key, value in pairs:
        if key in seen:
            problems.append("key %s repeated" % key)
            continue
        seen.add(key)
        if key not in keys:
            problems.append("key %s is not a %s key" % (key, kind))
            continue
        try:
            values[key] = payload_value(keys[key], value)
        except ValueError as e:
            problems.append("key %s: %s" % (key, e))
    for key, spec in keys.items():
        if spec["required"] and key not in seen:
            problems.append("required key %s missing" % key)
    return values, problems


def onset_and_confirmation(kind, values, physics_step):
    """(onset, confirmation) steps of a crash row: a collision is latched at its onset, a rollover when its hold ends
    (the row's physics_step) after an onset at the first tilted step, a polled kind at its poll (section 9.5.1)."""
    onset = physics_step
    if kind in ("collision", "rollover") and isinstance(values.get("onset_step"), int):
        onset = values["onset_step"]
    confirmation = onset if kind == "collision" else physics_step
    return onset, confirmation


def terminal_reason(kind, values):
    """The episodes.termination_reason an ego crash event makes ('collision', 'rollover'), or None (9.5.1)."""
    if kind == "collision":
        return "collision" if values.get("src") in TERMINAL_COLLISION_SRC else None
    if kind == "veh_damage":
        immobile = values.get("movement_damage") is not None and values["movement_damage"] >= 1.0
        state = values.get("state")
        return "collision" if (state is not None and state != "undamaged") or immobile else None
    if kind == "veh_fire_start":
        return "collision"
    if kind == "rollover":
        return "rollover"
    return None


def ego_slots(s):
    """The ego slot ids of the session (telemetry and episodes; one per session in dc-2.0)."""
    slots = set(v for v in s.streams["telemetry"].col("ego_slot_id") if v is not None)
    slots |= set(v for v in s.streams["episodes"].col("ego_slot_id") if v is not None)
    return slots


def episode_bounds(s):
    """{episode_id: (start_physics_step, end_physics_step)} of the committed episodes rows."""
    ep = s.streams["episodes"]
    return {e: (a, b) for e, a, b in zip(ep.col("episode_id"), ep.col("start_physics_step"),
                                         ep.col("end_physics_step")) if a is not None and b is not None}


def _read_crash_events(s):
    """Session.crash_events: one dict per events row of a kind in event_payloads, in row order: event_seq, episode_id,
    physics_step, world_time_ms, kind, veh_slot_id, values, problems, step (the onset), confirm_step, in_episode (the
    onset lies inside its episode_id's bounds, or inside none for episode_id -1), chrono_ok (the onset is not after
    the row's physics_step), confirmed_inside (confirm_step <= the episode's end_physics_step), ego (the row names an
    ego slot) and terminal (terminal_reason of an ego row). terminal_ego / terminal_ego_confirmed: {episode_id: the
    episode's first valid (in_episode, chrono_ok) terminal ego event by onset / by confirmation}."""
    ev = s.streams["events"]
    payloads = s.schema.payloads
    egos = ego_slots(s)
    bounds = episode_bounds(s)
    s.crash_events = []
    for i in range(len(ev)):
        kind = ev.data["kind"][i]
        if kind not in payloads:
            continue
        values, problems = check_payload(kind, ev.data["payload"][i], payloads)
        step, confirm = onset_and_confirmation(kind, values, ev.data["physics_step"][i])
        e = ev.data["episode_id"][i]
        if e is not None and e >= 0:
            inside = e in bounds and bounds[e][0] <= step <= bounds[e][1]
        else:
            inside = not any(a <= step <= b for a, b in bounds.values())
        ego = ev.data["veh_slot_id"][i] in egos
        s.crash_events.append({"event_seq": ev.data["event_seq"][i], "episode_id": e,
                               "physics_step": ev.data["physics_step"][i],
                               "world_time_ms": ev.data["world_time_ms"][i], "kind": kind,
                               "veh_slot_id": ev.data["veh_slot_id"][i], "values": values, "problems": problems,
                               "step": step, "confirm_step": confirm, "in_episode": inside,
                               "chrono_ok": step <= ev.data["physics_step"][i],
                               "confirmed_inside": e is not None and e in bounds and confirm <= bounds[e][1],
                               "ego": ego,
                               "terminal": terminal_reason(kind, values) if ego else None})
    s.terminal_ego, s.terminal_ego_confirmed = {}, {}
    for c in s.crash_events:
        e = c["episode_id"]
        if c["terminal"] is None or e is None or e < 0 or not c["in_episode"] or not c["chrono_ok"]:
            continue
        if e not in s.terminal_ego or c["step"] < s.terminal_ego[e]["step"]:
            s.terminal_ego[e] = c
        if e not in s.terminal_ego_confirmed or c["confirm_step"] < s.terminal_ego_confirmed[e]["confirm_step"]:
            s.terminal_ego_confirmed[e] = c


def _build_decisions(s):
    ctl, tel, hf = s.streams["ctl"], s.streams["telemetry"], s.streams["hf"]
    n = s.clocks["N"]
    tel_index = tel.index()
    texp = s.telemetry_export
    cexp = s.ctl_export
    period = s.clocks["image_period_ms"]
    steps = ctl.data["physics_step"]
    covered = ctl.data["covered_steps"]
    # obs candidates: decoded images whose captured frame C is in hf and whose pose row C + 1 joins ctl and telemetry
    # (section 6: a frame whose joins fail is not joined and its masks are 0), by captured frame:
    # (C, pose row last_physics_step, world time of hf row C, slot, episode of the pose row's ctl row)
    cands = []
    for row in s.images:
        h, ph = row["hf_index"], row["pose_hf_index"]
        if row["fid_ok"] and h is not None and ph is not None and s.hf_ctl[ph] is not None \
                and s.hf_tel[ph] is not None:
            cands.append((row["captured_render_frame"], hf.data["last_physics_step"][ph], hf.data["world_time_ms"][h],
                          row["slot_id"], ctl.data["episode_id"][s.hf_ctl[ph]]))
    cands.sort()
    cand_lps = [c[1] for c in cands]
    cand_frames = [c[0] for c in cands]
    lps_sorted = all(cand_lps[i] <= cand_lps[i + 1] for i in range(len(cand_lps) - 1))
    warm = _event_times(s, ("warmup_end",))
    warmup_end_ms = warm[0][1] if warm else float(s.meta_get("trial.warmup_s", 0.0) or 0.0) * 1000.0
    env_ms = [w for _, w in _event_times(s, ("env_change",))]
    # a reset of a traffic vehicle never masks ego samples: the ego's rows and the vehicle-independent ones (-1)
    tele_slots = ego_slots(s) | {-1}
    ev = s.streams["events"]
    tele_steps = [ev.data["physics_step"][i] for i in range(len(ev))
                  if ev.data["kind"][i] in ("teleport", "respawn") and ev.data["veh_slot_id"][i] in tele_slots]
    crash_step = {e: c["step"] for e, c in s.terminal_ego.items()}
    s.decisions = []
    callbacks, s.decision_grid_skipped = decision_callbacks(ctl.data["world_time_ms"], s.clocks["decision_hz"])
    for t, (c0, grid) in enumerate(callbacks):
        p = steps[c0]
        nxt = callbacks[t + 1][0] if t + 1 < len(callbacks) else None
        p_next = steps[nxt] if nxt is not None else None
        c1 = c0 + 1 if c0 + 1 < len(ctl) else None
        t0 = tel_index.get(p)
        world = ctl.data["world_time_ms"][c0]
        episode = ctl.data["episode_id"][c0]
        # the decision interval (P_t, P_t+1]: the rows after C_t up to the next decision callback, which must cover
        # every step without a gap
        interval = list(range(c0 + 1, nxt + 1)) if nxt is not None else []
        contiguous = nxt is not None and all(
            steps[j] - covered[j] == steps[j - 1] for j in interval)
        complete = t0 is not None and contiguous
        one_episode = complete and episode is not None and episode >= 0 and all(
            ctl.data["episode_id"][j] == episode for j in interval)
        state = None
        if t0 is not None:
            state = [_num(texp["speed_fwd_mps"][t0]), _num(texp["yaw_rate_rps"][t0]),
                     _num(texp["acc_long_mps2"][t0]), texp["dir_state"][t0]]
        label = [ctl.data["exp_kappa_now"][c0], ctl.data["exp_accel_now"][c0], ctl.data["exp_dir_now"][c0]]
        cmd = None if c1 is None else [ctl.data["hl_kappa"][c1], ctl.data["hl_accel"][c1], ctl.data["hl_dir"][c1]]
        exec_rows = []
        for q in range(p + 1, p + n + 1):
            j = covering_row(steps, covered, q)
            if j is None or p_next is None or q > p_next:
                exec_rows.append(None)
            else:
                exec_rows.append([cexp["steer_cmd"][j], ctl.data["llc_throttle"][j], ctl.data["llc_brake"][j],
                                  cexp["steer_input"][j]])
        hl_seq = None if c1 is None else ctl.data["hl_seq"][c1]
        mode = ctl.data["mode"][c1] if c1 is not None else ctl.data["mode"][c0]
        # obs_t: newest decoded image whose pose row has last_physics_step <= P_t and whose captured frame is at or
        # before C_t's render frame (a frame without a callback keeps last_physics_step while world time advances:
        # the step bound alone would admit a future image, review round 1)
        frame_t = ctl.data["render_frame"][c0]
        obs = None
        if cands:
            if lps_sorted:     # sorted by captured frame, so the eligible candidates are a prefix
                j = min(bisect.bisect_right(cand_lps, p), bisect.bisect_right(cand_frames, frame_t)) - 1
                obs = cands[j] if j >= 0 else None
            else:
                eligible = [c for c in cands if c[1] <= p and c[0] <= frame_t]
                obs = max(eligible) if eligible else None
        obs_fields = {"obs_slot_id": None, "obs_captured_render_frame": None, "obs_last_physics_step": None,
                      "obs_age_steps": None, "obs_age_ms": None}
        obs_valid = 0
        if obs is not None:
            obs_fields = {"obs_slot_id": obs[3], "obs_captured_render_frame": obs[0], "obs_last_physics_step": obs[1],
                          "obs_age_steps": p - obs[1], "obs_age_ms": world - obs[2]}
            if obs_fields["obs_age_ms"] <= period:
                obs_valid = 1
        same_episode = obs_valid == 1 and obs[4] == episode
        warmup = 1 if world < warmup_end_ms else 0
        env_transition = 1 if any(0 <= world - w < ENV_TRANSITION_MS for w in env_ms) else 0
        # a teleport or respawn between the image's last physics step and the interval's end puts the obs and the
        # state (or the interval) on different sides of the jump
        end_q = p_next if p_next is not None else p + n
        first_q = obs[1] if obs_valid == 1 else p
        teleport = 1 if any(first_q <= q <= end_q for q in tele_steps) else 0
        react_hidden = any(ctl.data["react_visible"][j] == 0 for j in [c0] + interval)
        has_cmd = cmd is not None and all(v is not None for v in cmd)
        held = has_cmd and complete and all(
            [ctl.data[c][j] for c in HL_COLUMNS] == [hl_seq] + cmd for j in interval)
        inside = exec_rows[:min(n, p_next - p)] if p_next is not None else []
        exec_ok = bool(inside) and all(r is not None and all(v is not None for v in r) for r in inside)
        has_label = all(v is not None for v in label)
        wm_valid = int(complete and one_episode and held and exec_ok and obs_valid == 1 and same_episode
                       and not teleport)
        # a crash: no decision whose interval (p, p_next] contains or follows the episode's first terminal ego event
        # is a policy label (section 9.5.1); wm_valid keeps the crash dynamics
        crashed = episode in crash_step and crash_step[episode] <= end_q
        policy_valid = int(t0 is not None and episode is not None and episode >= 0 and has_label
                           and mode not in ("X", "A", "I", None) and obs_valid == 1 and same_episode
                           and not react_hidden and not teleport and not crashed)
        d = {"decision_step": p, "grid_point": grid, "render_frame": frame_t, "episode_id": episode, "hl_seq": hl_seq,
             "interval_steps": None if p_next is None else p_next - p, "mode": mode, "world_time_ms": world,
             "state": state, "action_cmd": cmd, "action_label": label, "action_exec": exec_rows,
             "obs_valid": obs_valid,
             "masks": {"policy_valid": policy_valid, "wm_valid": wm_valid, "warmup": warmup,
                       "env_transition": env_transition, "teleport": teleport}}
        d.update(obs_fields)
        s.decisions.append(d)
    s.segments = segments_of(s.decisions)


def _num(v):
    v = float(v)
    return None if math.isnan(v) else v


def segments_of(decisions, n=None):
    """Maximal runs of wm-valid, non-warm-up decision steps of one episode on consecutive grid points (n is unused:
    decision intervals are frame-bound, so consecutive means grid_point + 1)."""
    segments = []
    cur = None
    for d in decisions:
        ok = d["masks"]["wm_valid"] == 1 and d["masks"]["warmup"] == 0 and d["episode_id"] is not None \
            and d["episode_id"] >= 0
        if ok and cur is not None and d["episode_id"] == cur["episode_id"] \
                and d["grid_point"] == cur["last_grid_point"] + 1:
            cur["last_decision_step"] = d["decision_step"]
            cur["last_grid_point"] = d["grid_point"]
            cur["n_steps"] += 1
            continue
        if cur is not None:
            segments.append(cur)
            cur = None
        if ok:
            cur = {"episode_id": d["episode_id"], "first_decision_step": d["decision_step"],
                   "last_decision_step": d["decision_step"], "last_grid_point": d["grid_point"], "n_steps": 1}
    if cur is not None:
        segments.append(cur)
    for seg in segments:
        del seg["last_grid_point"]
    return segments
