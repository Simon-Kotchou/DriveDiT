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
  * frames/: each img row's file (path + IMAGE_SUFFIX, TBD-4), its size and its fiducial v2 code
    (data/dc2/fiducial.py); captured_render_frame is the code unwrapped against issue_render_frame (img_resolved).

Joins and timing (section 6 and the streams JSON "timing"): img -> hf exact on captured_render_frame; hf -> ctl
exact on last_physics_step; hf -> telemetry as-of (the newest telemetry step <= last_physics_step, younger than k
steps); telemetry and ctl by physics_step. Rates and periods come from session.json clocks.* and the counters,
never from an assumed frame rate (CR-094); rows are keyed by their own counters, never by position (CR-085). Only
the ego is read: dc-2.0 has one ego per session and the traffic stream is kept apart.

Decision samples (the R-01 decision boundary): decision step t is the physics step P_t with P_t mod N == 0
(N = clocks.physics_steps_per_decision). state_t = telemetry at P_t; action_label_t = ctl.exp_*_now at P_t;
action_cmd_t = ctl.hl_* at P_t + 1 (decided after step P_t, applied from P_t + 1); action_exec_t = the ctl rows
P_t + 1 .. P_t + N; obs_t = the newest decoded image whose captured frame joins hf, ctl and telemetry and has
last_physics_step <= P_t, valid when it is at most one image period old. A decision step is wm-valid only with every
ctl row of its window, every action_exec channel, a telemetry row at P_t, one episode over the window, the command
held unchanged over the window (same hl_seq and hl_*), a valid image of the same episode and no teleport or respawn
between the image's last physics step and P_t + N. Segments are the maximal runs of wm-valid, non-warm-up decision
steps of one episode, so they split at gaps, episode boundaries and masks.

Files: every file the reader opens (stream directories, chunks, session files, images) must resolve inside the
session directory (symlinks followed), and img.path must be exactly frames/f_<slot_id:08d> (TBD-4); anything else
is a C01 (outside) or C11 (path) problem and is not read.

The seal (session_end.json, sections 5.3, 8 and 10.2): status ok with a non-error end_reason (C01), contract_version
dc-2.0 and episodes equal to the committed episodes rows (C01); the committed ctl, telemetry and hf rows reach the
session end (end.physics_step, end.render_frame: C34), so a lost or truncated tail is seen.

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
ENV_TRANSITION_MS = 15000       # features_v1 masks.env_transition: 1 for 15 s after an environment change
LEGACY_MARKERS = ("telemetry.csv", "summary.txt", "session_info.txt")
LEGACY_DIR_RE = re.compile(r"^session_[0-9]+$")
SEALED_PATH_PARTS = ("heldout", "sealed")
FORBIDDEN_ROOTS = ("/mnt/c/",)
CHUNK_RE = re.compile(r"^chunk_([0-9]{5})\.csv$")
INT_RE = re.compile(r"-?[0-9]+")
FLOAT_RE = re.compile(r"-?[0-9]+(\.[0-9]+)?")     # '.' decimal point, no exponent; nan and inf never match
EDEN_ROOT_RE = re.compile(r"^(\{[0-9A-Fa-f]+\})?worlds/eden/", re.IGNORECASE)   # every Eden-rooted world is Everon
# session_end.json end_reason values that end a session in error (section 5.3); duration, stop and dc_stop are
# the normal endings.
ERROR_END_REASONS = ("script_error:write_fail", "script_error:other", "zero_ego", "ego_lost", "camera_lost")
STRICT_FATAL = ("C01", "C02", "C05", "C07", "C34")
HL_COLUMNS = ("hl_seq", "hl_kappa", "hl_accel", "hl_dir")

# Stream columns the derived views read (docs dc-2.0.md 9.8 ctl_export / telemetry_export and the decision
# boundary). Schema.check_columns verifies at load time that the streams JSON still defines every one of them.
DERIVED_SOURCES = {
    "ctl": ("physics_step", "world_time_ms", "episode_id", "mode", "driver", "hl_seq", "hl_kappa", "hl_accel",
            "hl_dir", "exp_kappa_now", "exp_accel_now", "exp_dir_now", "llc_steer_lim", "llc_throttle", "llc_brake",
            "input_post_steer", "react_visible"),
    "telemetry": ("physics_step", "world_time_ms", "episode_id", "fwd_x", "fwd_z", "vel_local_x", "vel_local_z",
                  "angvel_local_y", "acc_local_x", "acc_local_z", "gear"),
    "hf": ("render_frame", "world_time_ms", "last_physics_step", "paused"),
    "img": ("slot_id", "issue_status", "issue_render_frame", "path"),
    "events": ("event_seq", "physics_step", "world_time_ms", "kind"),
    "episodes": ("episode_id",),
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


def tbd1_rule(slice_s, decision_hz=10.0):
    """(physics_steps_per_decision N, telemetry_every_k k) of TBD-1 for a fixed time slice in seconds: N =
    floor(1 / decision_hz / slice + 0.5) first (100 ms at the 10 Hz default, 200 ms at the 5 Hz fallback of TBD-2,
    where N is 12 at a 60 Hz slice), then k = the largest divisor of N with k * slice <= 50 ms (+1 us), at least 1."""
    slice_s = float(slice_s)
    if not (slice_s > 0.0 and math.isfinite(slice_s)):
        raise ValueError("fixed time slice %r is not a positive number" % slice_s)
    decision_hz = float(decision_hz)
    if not (decision_hz > 0.0 and math.isfinite(decision_hz)):
        raise ValueError("decision rate %r is not a positive number" % decision_hz)
    n = int(math.floor(1.0 / decision_hz / slice_s + 0.5))
    n = max(n, 1)
    k = 1
    for d in range(1, n + 1):
        if n % d == 0 and d * slice_s <= 0.05 + 1e-6:
            k = d
    return n, k


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
    clocks                   dict: k, N, decision_hz, image_hz, slice_s, img_lead_ms, img_inflight_timeout_ms,
                             chunk_max_s, image_period_ms
    streams                  {name: Stream} of committed rows; heartbeat: list of row dicts
    images                   img_resolved rows (one per img row, in img order)
    ctl_export, telemetry_export   per-row derived views (dicts of numpy arrays)
    decisions                decision samples (list of dicts; see _build_decisions)
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
        self.vehicle_card = None

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
        clocks = {k: self.clocks[k] for k in ("telemetry_every_k", "physics_steps_per_decision", "decision_hz",
                                               "image_hz", "img_lead_ms")}
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
GOLDEN_DECISION_KEYS = ("decision_step", "episode_id", "hl_seq", "state", "action_cmd", "action_label",
                        "action_exec", "obs_slot_id", "obs_captured_render_frame", "obs_last_physics_step",
                        "obs_age_steps", "obs_age_ms", "obs_valid", "masks")


def open_session(path, sealed_eval=False, strict=True, schema=None, decode_images=True, vehicle_card=None,
                 salvage=False, image_visitor=None):
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
    """
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
    stops acknowledging rows)."""
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
    k = s.clocks["k"]
    want_tel = end_step - end_step % k if isinstance(end_step, int) and end_step >= 0 else None
    for stream, key, want, what in (("ctl", "physics_step", end_step, "end.physics_step"),
                                    ("telemetry", "physics_step", want_tel, "the last k-th step <= end.physics_step"),
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
        "telemetry_every_k": get("clocks.telemetry_every_k"),
        "physics_steps_per_decision": get("clocks.physics_steps_per_decision"),
        "decision_hz": get("clocks.decision_hz"),
        "image_hz": get("clocks.image_hz"),
        "img_lead_ms": get("clocks.img_lead_ms"),
        "img_inflight_timeout_ms": get("clocks.img_inflight_timeout_ms"),
        "chunk_max_s": get("clocks.chunk_max_s"),
        "slice_s": get("engine.fixed_time_slice_s"),
    }
    k, n = clocks["telemetry_every_k"], clocks["physics_steps_per_decision"]
    ok = (isinstance(k, int) and isinstance(n, int) and k >= 1 and n >= 1 and n % k == 0
          and isinstance(clocks["image_hz"], (int, float)) and clocks["image_hz"] > 0)
    if not ok:
        raise ReaderError("%s: session.json clocks are unusable (k %r, physics_steps_per_decision %r, image_hz %r); "
                          "k must divide N" % (s.path, k, n, clocks["image_hz"]))
    clocks["k"] = k
    clocks["N"] = n
    clocks["image_period_ms"] = 1000.0 / float(clocks["image_hz"])
    s.clocks = clocks
    s.steer_sign = get("control.steer_sign", -1)
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
    a C11 problem) and the file must resolve inside the session directory (else C01). Stored in s.image_files."""
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
        rel = path + IMAGE_SUFFIX
        full = os.path.join(s.path, rel)
        if os.path.lexists(full) and not inside(s.path, full):
            s.problems.append(Problem("C01", rel, "resolves outside the session directory; not read"))
            s.image_files.append((None, "file resolves outside the session directory"))
            continue
        s.image_files.append((rel, None))


def _resolve_images(s, decode, visitor):
    img = s.streams["img"]
    hf_index = s.streams["hf"].index()
    s.images = []
    for i in range(len(img)):
        slot = img.data["slot_id"][i]
        row = {"slot_id": slot, "issue_status": img.data["issue_status"][i],
               "issue_render_frame": img.data["issue_render_frame"][i], "file": None, "file_exists": 0,
               "png_w": None, "png_h": None, "fid_ok": 0, "fid_status": None, "code": None,
               "captured_render_frame": None, "fid_margin": None, "hf_index": None, "error": None}
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
    """hf -> ctl (exact on last_physics_step) and hf -> telemetry (as-of, age below k steps): row indices or None."""
    hf = s.streams["hf"]
    ctl_index = s.streams["ctl"].index()
    tel_steps = s.streams["telemetry"].data["physics_step"]
    order = sorted(range(len(tel_steps)), key=lambda i: tel_steps[i])
    sorted_steps = [tel_steps[i] for i in order]
    k = s.clocks["k"]
    s.hf_ctl = []
    s.hf_tel = []
    for lps in hf.data["last_physics_step"]:
        s.hf_ctl.append(ctl_index.get(lps))
        j = bisect.bisect_right(sorted_steps, lps) - 1
        if j >= 0 and lps - sorted_steps[j] < k:
            s.hf_tel.append(order[j])
        else:
            s.hf_tel.append(None)


def _event_times(s, kinds):
    ev = s.streams["events"]
    return [(ev.data["physics_step"][i], ev.data["world_time_ms"][i])
            for i in range(len(ev)) if ev.data["kind"][i] in kinds]


def _build_decisions(s):
    ctl, tel, hf = s.streams["ctl"], s.streams["telemetry"], s.streams["hf"]
    n = s.clocks["N"]
    ctl_index, tel_index = ctl.index(), tel.index()
    texp = s.telemetry_export
    cexp = s.ctl_export
    period = s.clocks["image_period_ms"]
    # obs candidates: decoded images joined to hf, and through hf to ctl and telemetry (section 6: a frame whose
    # joins fail is not joined and its masks are 0), by captured frame; the last field is the frame's episode
    cands = []
    for row in s.images:
        h = row["hf_index"]
        if row["fid_ok"] and h is not None and s.hf_ctl[h] is not None and s.hf_tel[h] is not None:
            cands.append((row["captured_render_frame"], hf.data["last_physics_step"][h],
                          hf.data["world_time_ms"][h], row["slot_id"], ctl.data["episode_id"][s.hf_ctl[h]]))
    cands.sort()
    cand_lps = [c[1] for c in cands]
    lps_sorted = all(cand_lps[i] <= cand_lps[i + 1] for i in range(len(cand_lps) - 1))
    warm = _event_times(s, ("warmup_end",))
    warmup_end_ms = warm[0][1] if warm else float(s.meta_get("trial.warmup_s", 0.0) or 0.0) * 1000.0
    env_ms = [w for _, w in _event_times(s, ("env_change",))]
    tele_steps = [p for p, _ in _event_times(s, ("teleport", "respawn"))]
    s.decisions = []
    steps = ctl.data["physics_step"]
    if not steps:
        s.segments = []
        return
    first = int(math.ceil(max(min(steps), 0) / float(n))) * n
    last = max(steps)
    for p in range(first, last + 1, n):
        c0 = ctl_index.get(p)
        t0 = tel_index.get(p)
        window = [ctl_index.get(p + j) for j in range(1, n + 1)]
        c1 = window[0]
        world = ctl.data["world_time_ms"][c0] if c0 is not None else (
            tel.data["world_time_ms"][t0] if t0 is not None else None)
        episode = ctl.data["episode_id"][c0] if c0 is not None else (
            tel.data["episode_id"][t0] if t0 is not None else None)
        complete = c0 is not None and t0 is not None and all(w is not None for w in window)
        one_episode = complete and episode is not None and episode >= 0 and all(
            ctl.data["episode_id"][w] == episode for w in window)
        state = None
        if t0 is not None:
            state = [_num(texp["speed_fwd_mps"][t0]), _num(texp["yaw_rate_rps"][t0]),
                     _num(texp["acc_long_mps2"][t0]), texp["dir_state"][t0]]
        label = None if c0 is None else [ctl.data["exp_kappa_now"][c0], ctl.data["exp_accel_now"][c0],
                                         ctl.data["exp_dir_now"][c0]]
        cmd = None if c1 is None else [ctl.data["hl_kappa"][c1], ctl.data["hl_accel"][c1], ctl.data["hl_dir"][c1]]
        exec_rows = []
        for w in window:
            if w is None:
                exec_rows.append(None)
            else:
                exec_rows.append([cexp["steer_cmd"][w], ctl.data["llc_throttle"][w], ctl.data["llc_brake"][w],
                                  cexp["steer_input"][w]])
        hl_seq = None if c1 is None else ctl.data["hl_seq"][c1]
        mode = ctl.data["mode"][c1] if c1 is not None else (ctl.data["mode"][c0] if c0 is not None else None)
        # obs_t: newest decoded image whose captured frame has last_physics_step <= P_t
        obs = None
        if cands:
            if lps_sorted:     # sorted by captured frame, so the last eligible candidate is the newest
                j = bisect.bisect_right(cand_lps, p) - 1
                obs = cands[j] if j >= 0 else None
            else:
                eligible = [c for c in cands if c[1] <= p]
                obs = max(eligible) if eligible else None
        obs_fields = {"obs_slot_id": None, "obs_captured_render_frame": None, "obs_last_physics_step": None,
                      "obs_age_steps": None, "obs_age_ms": None}
        obs_valid = 0
        if obs is not None:
            obs_fields = {"obs_slot_id": obs[3], "obs_captured_render_frame": obs[0], "obs_last_physics_step": obs[1],
                          "obs_age_steps": p - obs[1], "obs_age_ms": None if world is None else world - obs[2]}
            if world is not None and obs_fields["obs_age_ms"] <= period:
                obs_valid = 1
        same_episode = obs_valid == 1 and obs[4] == episode
        warmup = 1 if world is not None and world < warmup_end_ms else 0
        env_transition = 1 if world is not None and any(0 <= world - w < ENV_TRANSITION_MS for w in env_ms) else 0
        # a teleport or respawn between the image's last physics step and the window's end puts the obs and the
        # state (or the window) on different sides of the jump
        first_q = obs[1] if obs_valid == 1 else p
        teleport = 1 if any(first_q <= q <= p + n for q in tele_steps) else 0
        react_hidden = any(w is not None and ctl.data["react_visible"][w] == 0 for w in [c0] + window)
        has_cmd = cmd is not None and all(v is not None for v in cmd)
        held = has_cmd and complete and all(
            [ctl.data[c][w] for c in HL_COLUMNS] == [hl_seq] + cmd for w in window)
        exec_ok = all(r is not None and all(v is not None for v in r) for r in exec_rows)
        has_label = label is not None and all(v is not None for v in label)
        wm_valid = int(complete and one_episode and held and exec_ok and obs_valid == 1 and same_episode
                       and not teleport)
        policy_valid = int(c0 is not None and t0 is not None and episode is not None and episode >= 0 and has_label
                           and mode not in ("X", "A", "I", None) and obs_valid == 1 and same_episode
                           and not react_hidden and not teleport)
        d = {"decision_step": p, "episode_id": episode, "hl_seq": hl_seq, "mode": mode, "world_time_ms": world,
             "state": state, "action_cmd": cmd, "action_label": label, "action_exec": exec_rows,
             "obs_valid": obs_valid,
             "masks": {"policy_valid": policy_valid, "wm_valid": wm_valid, "warmup": warmup,
                       "env_transition": env_transition, "teleport": teleport}}
        d.update(obs_fields)
        s.decisions.append(d)
    s.segments = segments_of(s.decisions, n)


def _num(v):
    v = float(v)
    return None if math.isnan(v) else v


def segments_of(decisions, n):
    """Maximal runs of consecutive decision steps (step + N) of one episode that are wm-valid and not warm-up."""
    segments = []
    cur = None
    for d in decisions:
        ok = d["masks"]["wm_valid"] == 1 and d["masks"]["warmup"] == 0 and d["episode_id"] is not None \
            and d["episode_id"] >= 0
        if ok and cur is not None and d["episode_id"] == cur["episode_id"] \
                and d["decision_step"] == cur["last_decision_step"] + n:
            cur["last_decision_step"] = d["decision_step"]
            cur["n_steps"] += 1
            continue
        if cur is not None:
            segments.append(cur)
            cur = None
        if ok:
            cur = {"episode_id": d["episode_id"], "first_decision_step": d["decision_step"],
                   "last_decision_step": d["decision_step"], "n_steps": 1}
    if cur is not None:
        segments.append(cur)
    return segments
