#!/usr/bin/env python3
"""retention: the raw-screenshot retention rule of DD-6-min (keep the small resized copies, delete the full-size
raw screenshots once they have been checked).

    python3 -I -B tools/harvester/retention.py [--probes] [--apply] [--every-n N] [--min-age-days D]
        [--under DIR] [--list]

A dry run by default: per session (or probe play) it prints the raw files it would delete and keep and the bytes it
would free, then the totals, and writes nothing. --apply deletes. --list prints every file with its action.

Sessions (the default mode). Only captures/arland/<YYYY-MM-DD>/<trial_id>/<session_id>/ below /mnt/e/DriveDiT is
looked at: never captures/_rejected/, compat19/, another map or a profile's DC_Sessions/. A session's raw
screenshot frames/f_<slot:08d><suffix> (suffix: harvest.json screenshot_suffix, one of .png, .bmp.png, .bmp) may be
deleted only when all of these hold; otherwise nothing of the session is deleted:
  * session.json is an Arland dc-2.0 session (harvest.is_arland) whose id is the directory name and whose date and
    trial are the directory's (a dc-1.0 session is refused);
  * _harvest/result.json and _harvest/dc_validate.json say PASS or PASS_WITH_WARNINGS, and result.json routes the
    session to captures and this destination;
  * catalog/sessions.jsonl, read under the harvester's lock, has a complete row for the session, and every row
    naming it has this destination, route captures, a passing verdict and the sha256 of the session's SHA256SUMS;
  * _harvest/SHA256SUMS verifies (harvest.verify_sums hashes every file; a symlink anywhere in the session holds it);
  * the slot's derivative _harvest/deriv/f_<slot:08d>.jpg is listed in SHA256SUMS (so it exists and verified).
The audit sample is kept (harvest.retention_session_rule): the first and the last raw slot, every slot that is a
multiple of --every-n (default 100), and every slot whose image time (img issue_world_ms, else requested_world_ms)
lies within 2 s of an events row of kind collision, stuck, respawn, teleport or script_error. A raw file whose slot
has no img row, or no derivative, is kept.

Probe plays (--probes). <profile>/profile/DC_Probes/<probe>/<launch_id>/play_NN/frames/f_<n:08d>.png (.bmp.png,
.bmp) of wb_profile, wb_profile_x19 and client_profiles/iNN: diagnostic screenshots, not training data. A play's
frames may be deleted only when its run_footer.txt exists (schema dc-probe-run/1: the play ended and closed its
files) and says status=complete, its run_header.txt names world.map_id=arland, and no file of the play (its own
files and frames/; not RETENTION.json) was modified in the last --min-age-days (default 1). A play with frames whose
footer says stopped, failed or anything else is held with every frame kept: it is evidence of a problem. So is a
play whose run_footer.txt repeats any key, or whose run_header.txt repeats schema or world.map_id or repeats any
key with another value (the probe appends header lines; an identical repeat is unambiguous). Kept: the first and the
last frame and every frame whose number is a multiple of --every-n.

The record. Before the first delete, RETENTION.json is written in _harvest/ of a session or in the play directory of
a probe play: RETENTION.json.dd6tmp is created, written and fsynced, renamed over RETENTION.json and the directory is
fsynced, all through the pinned directory descriptor; any failure holds the unit with nothing deleted. It holds the
rule version, N, the event window and kinds, the kept files with their reasons, the deleted files with their sha256
(from SHA256SUMS for a session; hashed just before the record is written for a probe play) and size, the bytes
freed, the time and status "deleting"; once every delete is done it is rewritten with status "done". SHA256SUMS is
never changed: harvest.verify_sums counts a raw file the record lists as deleted as accounted for, so the harvester
still sees a retained session as complete and never rebuilds or re-validates it.

A record is never a licence to delete. Every run recomputes the rule on the unit's inventory (a session's: the raw
files SHA256SUMS lists, its img and events streams and derivatives; a play's: its raw frames now on disk plus the
files the record names) with the record's N, and holds the unit unless the record's kept files are exactly the
audit sample and its deleted files exactly the rest, each with its SHA256SUMS (or recorded) sha256, and every kept
file is still there. A record with deletes still to do must also have this run's --every-n and rule version.

Deletes. For each unit the directory chain is opened from the root one component at a time with O_DIRECTORY |
O_NOFOLLOW; the unit directory, frames/ and the record directory must be the very directories planned (device and
inode) and the unit's /proc/self/fd path must be its path below the root, else nothing is written or deleted. Each
file is opened with O_NOFOLLOW in the pinned frames/ descriptor, must be the regular file planned (device, inode,
size, mtime), is hashed through that descriptor and must match its sha256, and is checked again by lstat just
before its unlinkat. The first file that fails a check stops the unit, which is held with the rest kept. Only
regular files named f_<8 digits>.png / .bmp.png / .bmp (the whole name, no control character) are ever deleted.

Re-runs and kills. A valid record is completed: its remaining deletes are re-checked as above, then it is marked
done. A kill at any point leaves every file the record lists either present (and checked again on the next run) or
deleted. A run with nothing to do writes nothing. --apply holds catalog/harvest.lock (the harvester's), so it never
runs beside the harvester or another retention run: sessions are planned and applied one at a time under it, with
the catalog read under it (without a catalog file at the start no session is retained in that run); probe plays are
planned read-only first, then planned again and applied under the lock.

Scope. Every path lies below /mnt/e/DriveDiT (harvest.data_root: not reached through a symlink, not on C:). --under
DIR narrows the scan; it must lie below the root without a symlink, inside captures/arland (sessions) or a
DC_Probes folder (--probes). A $profile:DrivingData (dc-1.0) path or session is refused, and a symlinked directory
on the way holds what lies below it.

Exit codes: 0 done or nothing to do (skips are normal: unsealed, too new, not PASS, not catalogued); 2 a session or
play was held or refused (checksum mismatch, symlink, bad record, a play whose footer is not status=complete, not
Arland, dc-1.0) and nothing more of it deleted; 3 the run was refused (arguments, roots, --under outside the root,
the lock); 4 an unexpected error.

--test-root DIR (tests only) replaces /mnt/e/DriveDiT by DIR, as in harvest.py; only then does
DD6_RETENTION_KILL_AT=<hook> (HOOKS) kill the run at that hook, for the kill-mid-run test.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import sys
import time
import traceback

HERE = os.path.dirname(os.path.realpath(__file__))


def _load_harvest():
    spec = importlib.util.spec_from_file_location("dd6_harvest_retention", os.path.join(HERE, "harvest.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harvest = _load_harvest()
reader = harvest.reader
Refused = harvest.Refused
raw_shot_slot = harvest.raw_shot_slot
audit_sample = harvest.retention_sample

RULE_VERSION = harvest.RETENTION_RULE_VERSION
RECORD_NAME = harvest.RETENTION_NAME
RECORD_SCHEMA = harvest.RETENTION_SCHEMA
RECORD_TMP = RECORD_NAME + harvest.TMP_SUFFIX
DEFAULT_EVERY_N = 100
DEFAULT_MIN_AGE_DAYS = 1.0
EVENT_WINDOW_MS = harvest.RETENTION_EVENT_WINDOW_MS
EVENT_KINDS = harvest.RETENTION_EVENT_KINDS
RAW_SUFFIXES = harvest.RAW_SHOT_SUFFIXES
PROBE_RE = re.compile(r"[a-z][a-z0-9]{0,31}")                 # fullmatch only, as every name pattern here
PLAY_RE = re.compile(r"play_[0-9]{2,4}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROBE_SCHEMA = "dc-probe-run/1"
COMPLETE_STATUS = "complete"        # run_footer.txt status of a play that ran to its end; any other keeps every frame
CRITICAL_HEADER_KEYS = ("schema", "world.map_id")
PROBES_DIR = "DC_Probes"
LEGACY_DIR = "DrivingData"
FRAMES = "frames"
PASSING = ("PASS", "PASS_WITH_WARNINGS")
MAX_REPORT_BYTES = 256 << 20
KILL_ENV = "DD6_RETENTION_KILL_AT"
HOOKS = ("after_record", "in_deletes", "before_unlink", "before_done")
EXIT_OK, EXIT_HELD, EXIT_REFUSED, EXIT_ERROR = 0, 2, 3, 4
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class Skip(Exception):
    """One session or play is left as it is. status: skipped (normal), held (an anomaly), refused (out of scope),
    done (retained earlier, nothing left to do)."""

    def __init__(self, status, why):
        Exception.__init__(self, why)
        self.status = status


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gib(n):
    return "%.2f GiB" % (n / float(1 << 30))


def ident(st):
    """What must not change between planning, hashing and unlinking a file: device, inode, size, mtime."""
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def dir_ident(path):
    """(device, inode) of a directory that is not a symlink; Skip held otherwise."""
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise Skip("held", "%s is a symlink or not a directory" % os.path.basename(path))
    return (st.st_dev, st.st_ino)


def frame_name(f_rel):
    """The file name of a frames/<raw screenshot> record path, or None."""
    if not isinstance(f_rel, str) or not f_rel.startswith(FRAMES + "/"):
        return None
    name = f_rel[len(FRAMES) + 1:]
    return name if raw_shot_slot(name) is not None else None


def load_json(path, limit):
    """The JSON object of a regular file (not a symlink) of at most limit bytes, or None."""
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) > limit:
            return None
        with open(path, "rb") as f:
            obj = json.loads(f.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def read_kv(path):
    """({key: value}, {key: [values]} of the keys that occur more than once) of a key=value text file
    (run_header.txt, run_footer.txt), or (None, {}) when it is not a regular file."""
    if os.path.islink(path) or not os.path.isfile(path):
        return None, {}
    seen = {}
    with open(path, "rb") as f:
        for line in f.read(1 << 20).decode("utf-8", errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                seen.setdefault(key.strip(), []).append(value.strip())
    return dict((k, v[0]) for k, v in seen.items()), dict((k, v) for k, v in seen.items() if len(v) > 1)


def record_text(doc):
    """RETENTION.json: one key per line, one kept or deleted entry per line (compact, diffable)."""
    lines = ["{"]
    items = list(doc.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if key in ("kept", "deleted") and value:
            lines.append(" %s: [" % json.dumps(key))
            lines += ["  %s%s" % (json.dumps(e, separators=(",", ":")), "," if j < len(value) - 1 else "")
                      for j, e in enumerate(value)]
            lines.append(" ]%s" % comma)
        else:
            lines.append(" %s: %s%s" % (json.dumps(key), json.dumps(value), comma))
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("ascii")


class Plan:
    """What retention does to one session or play: kept [{file, slot, why}], delete [{file, slot, sha256, bytes}]
    (paths relative to the unit directory), the record (an existing valid RETENTION.json, else None), the planned
    identities of the unit, frames/ and record directories (ids) and of every file still to delete (stats)."""

    def __init__(self, kind, path, rel):
        self.kind = kind                    # "session" | "probe_play"
        self.path = path
        self.rel = rel
        self.ident = {}
        self.ids = {}
        self.stats = {}
        self.kept = []
        self.delete = []
        self.record = None
        self.present = set()                # delete entries whose file is still there
        self.notes = []

    def todo(self):
        """The delete entries whose file is still there."""
        return [e for e in self.delete if e["file"] in self.present]

    def has_work(self):
        return bool(self.todo()) or (self.record is not None and self.record.get("status") != "done")

    def bytes_total(self):
        return sum(e["bytes"] for e in self.delete)

    def bytes_todo(self):
        return sum(e["bytes"] for e in self.todo())

    def document(self, every_n, status):
        doc = {"schema": RECORD_SCHEMA, "rule_version": RULE_VERSION, "kind": self.kind}
        doc.update(self.ident)
        doc.update({"every_n": every_n, "event_window_ms": EVENT_WINDOW_MS if self.kind == "session" else None,
                    "event_kinds": list(EVENT_KINDS) if self.kind == "session" else [],
                    "created_utc": utc_now(), "status": status, "completed_utc": None,
                    "n_kept": len(self.kept), "n_deleted": len(self.delete), "bytes_freed": self.bytes_total(),
                    "kept": self.kept, "deleted": self.delete})
        return doc


class Retention:
    def __init__(self, root, args, test_mode):
        self.root = root
        self.args = args
        self.kill_at = os.environ.get(KILL_ENV) if test_mode else None
        self.h = harvest.Harvester(root, harvest.parse_args([]), False)
        self.schema = self.h.schema
        self.statuses = []
        self.totals = {"units": 0, "delete": 0, "kept": 0, "bytes": 0}
        self.under = None

    # -- output -----------------------------------------------------------------------------------
    def report(self, status, rel, detail):
        self.statuses.append(status)
        print("%s %s: %s" % (status, rel, detail))

    def hook(self, name):
        if self.kill_at == name:
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)

    def rel(self, path):
        return os.path.relpath(path, self.root).replace(os.sep, "/")

    # -- scope ------------------------------------------------------------------------------------
    def check_under(self, under):
        """The --under directory (absolute, normalised), which must lie below the root without a symlink, inside
        captures/arland (sessions) or a DC_Probes folder (--probes), never in DrivingData or captures/_rejected."""
        if re.match(r"^[A-Za-z]:", under) or under.startswith("\\\\"):
            raise Refused("--under %r uses Windows syntax; pass a /mnt/e/DriveDiT/... path" % under)
        path = os.path.normpath(os.path.abspath(under))
        if not harvest.inside(path, self.root):
            raise Refused("--under %s is outside %s; retention never touches anything else" % (under, self.root))
        if os.path.realpath(path) != path:
            raise Refused("--under %s is reached through a symlink" % under)
        if not os.path.isdir(path):
            raise Refused("--under %s is not an existing directory" % under)
        parts = self.rel(path).split("/")
        if LEGACY_DIR in parts:
            raise Refused("--under %s: dc-1.0 $profile:%s sessions are out of scope" % (under, LEGACY_DIR))
        if self.args.probes:
            if PROBES_DIR not in parts:
                raise Refused("--under %s is not inside a <profile>/profile/%s folder (--probes)" % (under,
                                                                                                     PROBES_DIR))
        elif parts[:2] != [harvest.CAPTURES, "arland"]:
            raise Refused("--under %s is not inside captures/arland (never captures/_rejected, compat19 or a "
                          "profile; probe plays need --probes)" % under)
        return path

    def wanted(self, path):
        return self.under is None or harvest.inside(path, self.under) or harvest.inside(self.under, path)

    def subdirs(self, base, held):
        """Directories directly in base (sorted); a symlink among them goes to held, a file is ignored."""
        out = []
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if os.path.islink(path):
                held.append(path)
            elif os.path.isdir(path):
                out.append(path)
        return out

    def session_dirs(self):
        """(session directories, symlinked directories) below captures/arland/<date>/<trial>/."""
        base = os.path.join(self.root, harvest.CAPTURES, "arland")
        held = []
        if not os.path.lexists(base):
            return [], []
        if os.path.islink(base) or not os.path.isdir(base) or not harvest.real_below(base, self.root):
            return [], [base]
        out = []
        for date in self.subdirs(base, held):
            for trial in self.subdirs(date, held):
                out += self.subdirs(trial, held)
        return out, held

    def play_dirs(self):
        """(play directories, symlinked directories) below every profile's DC_Probes/<probe>/<launch_id>/."""
        bases = [os.path.join(self.root, name) for name in harvest.WB_PROFILES]
        cdir = os.path.join(self.root, harvest.CLIENT_PROFILES)
        if os.path.isdir(cdir) and harvest.real_below(cdir, self.root):
            bases += [os.path.join(cdir, n) for n in sorted(os.listdir(cdir))
                      if harvest.CLIENT_INSTANCE_RE.fullmatch(n)]
        out, held = [], []
        for base in bases:
            probes = os.path.join(base, harvest.PROFILE_DIR, PROBES_DIR)
            if os.path.islink(probes) or (os.path.isdir(probes) and not harvest.real_below(probes, self.root)):
                held.append(probes)
                continue
            if not os.path.isdir(probes):
                continue
            for probe in self.subdirs(probes, held):
                if not PROBE_RE.fullmatch(os.path.basename(probe)):
                    continue
                for launch in self.subdirs(probe, held):
                    if not harvest.LAUNCH_ID_RE.fullmatch(os.path.basename(launch)):
                        continue
                    out += [p for p in self.subdirs(launch, held) if PLAY_RE.fullmatch(os.path.basename(p))]
        return out, held

    # -- sessions ---------------------------------------------------------------------------------
    def plan_session(self, path, catalog):
        name = os.path.basename(path)
        rel = self.rel(path)
        if reader.LEGACY_DIR_RE.match(name):
            raise Skip("refused", "a dc-1.0 session; out of scope (retention reads dc-2.0 sessions only)")
        if not name.startswith("arland_"):
            raise Skip("refused", harvest.NOT_ARLAND)
        if os.path.islink(path) or not os.path.isdir(path) or not harvest.real_below(path, self.root):
            raise Skip("held", "reached through a symlink or not a directory")
        unit_id = dir_ident(path)
        meta = harvest.read_json(os.path.join(path, "session.json"))
        if meta is None:
            if reader.is_legacy_session(path):
                raise Skip("refused", "a dc-1.0 session; out of scope (retention reads dc-2.0 sessions only)")
            raise Skip("held", "session.json is missing or unreadable")
        if not harvest.is_arland(meta):
            raise Skip("refused", harvest.NOT_ARLAND)
        if meta.get("contract_version") != reader.CONTRACT_VERSION:
            raise Skip("refused", "contract_version %r is not %s; only dc-2.0 sessions are retained" % (
                meta.get("contract_version"), reader.CONTRACT_VERSION))
        trial = harvest.get(meta, "trial.trial_id")
        id_time = name.split("_")[2] if name.count("_") >= 2 else ""
        want = "%s/arland/%s-%s-%s/%s/%s" % (harvest.CAPTURES, id_time[0:4], id_time[4:6], id_time[6:8], trial, name)
        if meta.get("session_id") != name or rel != want:
            raise Skip("held", "not at its destination %s (session.json session_id, date and trial)" % want)
        hdir = os.path.join(path, harvest.HARVEST_DIR)
        if os.path.islink(hdir):
            raise Skip("held", "_harvest/ is a symlink; not read")
        if not os.path.isdir(hdir):
            raise Skip("skipped", "not harvested (no _harvest/)")
        stamp_doc = harvest.read_json(os.path.join(path, harvest.HARVEST_JSON)) or {}
        suffix = stamp_doc.get("screenshot_suffix")
        if stamp_doc.get("session_id") != name:
            raise Skip("skipped", "harvest.json is missing or names another session")
        if suffix not in RAW_SUFFIXES:
            raise Skip("skipped", "screenshot suffix %r is not a raw screenshot suffix (%s); nothing is deleted" % (
                suffix, ", ".join(RAW_SUFFIXES)))
        result = load_json(os.path.join(hdir, "result.json"), MAX_REPORT_BYTES) or {}
        report = load_json(os.path.join(hdir, "dc_validate.json"), MAX_REPORT_BYTES) or {}
        verdict = result.get("verdict")
        if result.get("session_id") != name:
            raise Skip("skipped", "_harvest/result.json is missing or names another session")
        if verdict not in PASSING or report.get("verdict") != verdict:
            raise Skip("skipped", "verdict %s (dc_validate.json %s); only %s sessions are retained" % (
                verdict, report.get("verdict"), " / ".join(PASSING)))
        if result.get("route") != "captures" or result.get("dest") != rel:
            raise Skip("held", "result.json routes it to %r, not here" % (result.get("dest"),))
        sums_path = os.path.join(hdir, harvest.SUMS_NAME)
        if os.path.islink(sums_path) or not os.path.isfile(sums_path):
            raise Skip("held", "no _harvest/%s" % harvest.SUMS_NAME)
        with open(sums_path, "rb") as f:
            sums_bytes = f.read()
        sums_sha = harvest.sha256_bytes(sums_bytes)
        rows = [r for r in catalog if r.get("session_id") == name]
        if not rows:
            raise Skip("skipped", "not in catalog/sessions.jsonl (the harvester catalogues it first)")
        for r in rows:
            if r.get("dest") != rel or r.get("route") != "captures" or r.get("verdict") not in PASSING \
                    or r.get("sha256sums_sha256") != sums_sha:
                raise Skip("held", "a catalog row of the session disagrees with it (dest, route, verdict or the "
                           "sha256 of SHA256SUMS)")
        frames = os.path.join(path, FRAMES)
        if os.path.islink(frames):
            raise Skip("held", "frames/ is a symlink; never followed")
        plan = Plan("session", path, rel)
        plan.ids = {"unit": unit_id, "frames": dir_ident(frames), "record_dir": dir_ident(hdir)}
        try:
            ok, detail = harvest.verify_sums(path)            # every hash, and any record against the rule
        except harvest.Held as e:
            ok, detail = False, str(e)
        if not ok:
            raise Skip("held", "SHA256SUMS does not verify: %s; nothing is deleted" % detail)
        listed = harvest.parse_sums(sums_bytes)
        plan.ident = {"session_id": name, "path": rel, "verdict": verdict, "screenshot_suffix": suffix,
                      "sha256sums_sha256": sums_sha}
        record_path = os.path.join(hdir, RECORD_NAME)
        if os.path.lexists(record_path):
            doc = load_json(record_path, harvest.RETENTION_MAX_BYTES)
            if doc is None or doc.get("path") != rel or doc.get("screenshot_suffix") != suffix \
                    or not all(type(e.get("bytes")) is int and e["bytes"] >= 0 for e in doc["deleted"]):
                raise Skip("held", "%s does not name this session's path and screenshot suffix with a size for "
                           "every delete; nothing is deleted" % RECORD_NAME)
            return self.from_record(plan, doc, doc["deleted"], doc["kept"])
        try:
            raw, keep = harvest.retention_session_rule(path, listed, suffix, self.args.every_n, self.schema)
        except harvest.Held as e:
            raise Skip("held", str(e))
        if not raw:
            raise Skip("skipped", "SHA256SUMS lists no raw screenshot frames/f_<slot:08d>%s" % suffix)
        for slot in sorted(raw):
            f_rel = raw[slot]
            if slot in keep:
                plan.kept.append({"file": f_rel, "slot": slot, "why": keep[slot]})
            else:
                st = os.lstat(os.path.join(path, f_rel))
                plan.delete.append({"file": f_rel, "slot": slot, "sha256": listed[f_rel], "bytes": st.st_size})
                plan.present.add(f_rel)
                plan.stats[f_rel] = ident(st)
        return plan

    # -- probe plays ------------------------------------------------------------------------------
    def plan_play(self, path, now):
        rel = self.rel(path)
        if not harvest.real_below(path, self.root):
            raise Skip("held", "reached through a symlink")
        unit_id = dir_ident(path)
        footer_path = os.path.join(path, "run_footer.txt")
        if os.path.islink(footer_path):
            raise Skip("held", "run_footer.txt is a symlink")
        footer, footer_repeats = read_kv(footer_path)
        if footer is None:
            raise Skip("skipped", "incomplete play: no run_footer.txt; nothing is deleted")
        if footer_repeats:
            raise Skip("held", "run_footer.txt repeats %s, so its status is ambiguous; every frame is kept" % (
                ", ".join(sorted(footer_repeats))))
        if footer.get("schema") != PROBE_SCHEMA:
            raise Skip("held", "run_footer.txt is not a %s footer" % PROBE_SCHEMA)
        header, header_repeats = read_kv(os.path.join(path, "run_header.txt"))
        ambiguous = sorted(k for k, v in header_repeats.items() if k in CRITICAL_HEADER_KEYS or len(set(v)) > 1)
        if ambiguous:
            raise Skip("held", "run_header.txt repeats %s (another value or a key the rule reads), so the play is "
                       "ambiguous; every frame is kept" % ", ".join(ambiguous))
        if header is None or header.get("world.map_id") != "arland":
            raise Skip("refused", "run_header.txt does not name world.map_id=arland; only Arland probe plays are "
                       "retained")
        frames = os.path.join(path, FRAMES)
        if os.path.islink(frames):
            raise Skip("held", "frames/ is a symlink; never followed")
        if not os.path.isdir(frames):
            raise Skip("skipped", "no frames/ (nothing to delete)")
        status = footer.get("status")
        if status != COMPLETE_STATUS:
            raise Skip("held", "run_footer.txt status=%s (reason=%s), not %s: a play that did not complete is "
                       "evidence of a problem, so every frame is kept" % (
                           status if status else "<none>", footer.get("reason") or "<none>", COMPLETE_STATUS))
        plan = Plan("probe_play", path, rel)
        plan.ids = {"unit": unit_id, "frames": dir_ident(frames), "record_dir": unit_id}
        plan.ident = {"path": rel, "probe": rel.split("/")[-3], "launch_id": rel.split("/")[-2],
                      "play": rel.split("/")[-1], "footer_status": status}
        newest = 0.0
        for name in os.listdir(path):
            st = os.lstat(os.path.join(path, name))
            if stat.S_ISREG(st.st_mode) and name not in (RECORD_NAME, RECORD_TMP):
                newest = max(newest, st.st_mtime)
        present = {}                        # f_rel -> (slot, lstat) of the raw frames on disk
        others = 0
        for name in sorted(os.listdir(frames)):
            st = os.lstat(os.path.join(frames, name))
            slot = raw_shot_slot(name)
            if stat.S_ISREG(st.st_mode):
                newest = max(newest, st.st_mtime)
            if slot is not None and stat.S_ISREG(st.st_mode):
                present["%s/%s" % (FRAMES, name)] = (slot, st)
            else:
                others += 1
        if others:
            plan.notes.append("%d other entr%s in frames/ left alone" % (others, "y" if others == 1 else "ies"))
        age = now - newest
        if age < self.args.min_age_days * 86400.0:
            raise Skip("skipped", "newer than --min-age-days %g (last written %.1f h ago); nothing is deleted" % (
                self.args.min_age_days, age / 3600.0))
        for f_rel, (_, st) in present.items():
            plan.stats[f_rel] = ident(st)
        record_path = os.path.join(path, RECORD_NAME)
        if os.path.lexists(record_path):
            doc = load_json(record_path, harvest.RETENTION_MAX_BYTES)
            deleted, kept = self.check_play_record(plan, doc, present)
            return self.from_record(plan, doc, deleted, kept)
        if not present:
            raise Skip("skipped", "frames/ holds no raw screenshot (nothing to delete)")
        keep = audit_sample(sorted(set(slot for slot, _ in present.values())), self.args.every_n)
        for f_rel in sorted(present):
            slot, st = present[f_rel]
            if slot in keep:
                plan.kept.append({"file": f_rel, "slot": slot, "why": keep[slot]})
            else:
                plan.delete.append({"file": f_rel, "slot": slot, "sha256": None, "bytes": st.st_size})
                plan.present.add(f_rel)
        return plan

    def check_play_record(self, plan, doc, present):
        """(deleted, kept) entries of a play's RETENTION.json, checked against the rule, never trusted: this rule
        version, kind and path, and recomputed on the inventory (the raw frames on disk plus every file the record
        names) with the record's N, its kept files must be exactly the audit sample, still on disk, and its deleted
        files exactly the rest, each with a sha256. Held otherwise."""
        bad = "%s exists but is not a valid retention record of this play (%%s); nothing is deleted" % RECORD_NAME
        every_n = doc.get("every_n") if doc else None
        if doc is None or doc.get("schema") != RECORD_SCHEMA or doc.get("kind") != "probe_play" \
                or doc.get("path") != plan.rel or doc.get("rule_version") != RULE_VERSION \
                or type(every_n) is not int or every_n < 1 or not isinstance(doc.get("deleted"), list) \
                or not isinstance(doc.get("kept"), list):
            raise Skip("held", bad % "schema, kind, path, rule_version, every_n")
        named = {}
        for key in ("deleted", "kept"):
            for e in doc[key]:
                f_rel = e.get("file") if isinstance(e, dict) else None
                if frame_name(f_rel) is None or f_rel in named:
                    raise Skip("held", bad % ("%r is not a raw frames/ screenshot or is named twice" % (f_rel,)))
                if key == "deleted" and not (isinstance(e.get("sha256"), str) and SHA256_RE.fullmatch(e["sha256"])
                                             and type(e.get("bytes")) is int and e["bytes"] >= 0):
                    raise Skip("held", bad % ("%s has no sha256 or size" % f_rel))
                named[f_rel] = key
        inventory = set(named) | set(present)
        slot_of = dict((f, raw_shot_slot(frame_name(f))) for f in inventory)
        keep = audit_sample(sorted(set(slot_of.values())), every_n)
        want_kept = set(f for f in inventory if slot_of[f] in keep)
        kept_files = set(f for f, key in named.items() if key == "kept")
        if kept_files != want_kept or set(f for f, key in named.items() if key == "deleted") != inventory - want_kept:
            raise Skip("held", bad % ("its kept and deleted files are not the audit sample of every_n %d and the "
                                      "rest" % every_n))
        missing = sorted(f for f in kept_files if f not in present)
        if missing:
            raise Skip("held", "kept frame %s of %s is missing (%d in all)" % (missing[0], RECORD_NAME, len(missing)))
        return doc["deleted"], doc["kept"]

    # -- an existing record -----------------------------------------------------------------------
    def from_record(self, plan, doc, deleted, kept):
        """The plan of a record that matched the rule (a session's in harvest.verify_sums, a play's in
        check_play_record): only its remaining deletes are done, and only with the record's own --every-n and this
        rule version, so a record never widens what is deleted."""
        plan.record = doc
        plan.kept = kept
        plan.delete = deleted
        for e in deleted:
            full = os.path.join(plan.path, e["file"])
            if os.path.lexists(full):
                plan.present.add(e["file"])
                plan.stats.setdefault(e["file"], ident(os.lstat(full)))
        if plan.todo() and (doc.get("every_n") != self.args.every_n or doc.get("rule_version") != RULE_VERSION):
            raise Skip("held", "%s was recorded with --every-n %s (%s) and still has %d delete(s) to do; this run "
                       "has --every-n %d: a record never widens what is deleted, so pass --every-n %s to complete "
                       "it" % (RECORD_NAME, doc.get("every_n"), doc.get("rule_version"), len(plan.todo()),
                               self.args.every_n, doc.get("every_n")))
        if doc.get("every_n") != self.args.every_n:
            plan.notes.append("recorded with --every-n %s" % doc.get("every_n"))
        return plan

    # -- the run ----------------------------------------------------------------------------------
    def plan_all(self, catalog):
        """Yield the Plan of every unit in scope that retention may act on, one at a time (each is applied before
        the next is planned); every other unit is reported."""
        now = time.time()
        dirs, held = self.play_dirs() if self.args.probes else self.session_dirs()
        for path in held:
            if self.wanted(path):
                self.report("held", self.rel(path), "a symlink; never followed, nothing below it is touched")
        for path in dirs:
            if self.wanted(path):
                plan = self.plan_unit(path, catalog, now)
                if plan is not None:
                    yield plan

    def plan_unit(self, path, catalog, now):
        rel = self.rel(path)
        try:
            plan = self.plan_play(path, now) if self.args.probes else self.plan_session(path, catalog)
        except Skip as e:
            shown = harvest.hidden(rel) if str(e) == harvest.NOT_ARLAND else rel
            self.report(e.status, shown, str(e))
            return None
        except OSError as e:
            self.report("held", rel, "I/O error while planning: %s; nothing is deleted" % e)
            return None
        if not plan.has_work():
            self.report("done", rel, "retained earlier (%s status %s: %d deleted, %d kept, %s freed)" % (
                RECORD_NAME, plan.record.get("status") if plan.record else None, len(plan.delete), len(plan.kept),
                gib(plan.bytes_total())))
            return None
        return plan

    def describe(self, plan, status):
        reasons = {}
        for e in plan.kept:
            for why in e.get("why") or []:
                key = why.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        todo = plan.todo()
        detail = "delete %d raw screenshot(s), keep %d (%s), free %s (%d B)" % (
            len(todo), len(plan.kept), ", ".join("%s %d" % kv for kv in sorted(reasons.items())) or "none",
            gib(plan.bytes_todo()), plan.bytes_todo())
        if plan.record is not None:
            detail += "; completing %s (status %s, %d of %d deletes done)" % (
                RECORD_NAME, plan.record.get("status"), len(plan.delete) - len(todo), len(plan.delete))
        if plan.kind == "probe_play":
            detail += "; footer status %s" % plan.ident.get("footer_status")
        if plan.notes:
            detail += "; " + "; ".join(plan.notes)
        self.report(status, plan.rel, detail)
        if self.args.list:
            for e in plan.kept:
                print("  keep   %s (%s)" % (e["file"], ",".join(e.get("why") or [])))
            for e in todo:
                print("  delete %s %d" % (e["file"], e["bytes"]))
        self.totals["units"] += 1
        self.totals["delete"] += len(todo)
        self.totals["kept"] += len(plan.kept)
        self.totals["bytes"] += plan.bytes_todo()

    def catalog_present(self):
        return os.path.isfile(self.h.catalog_path())

    def run(self):
        if self.args.under:
            self.under = self.check_under(self.args.under)
        if self.args.probes:
            return self.run_probes()
        return self.run_sessions()

    def run_sessions(self):
        """A dry run reads the catalog and plans. --apply holds the lock for the whole run and plans and applies one
        session at a time with the catalog read under the lock; without a catalog file at the start it reads none
        (a catalog a harvester writes meanwhile is the next run's), so no session qualifies and nothing is
        written."""
        if not self.args.apply:
            for plan in self.plan_all(self.h.catalog_rows()):
                self.describe(plan, "would-delete")
            return self.finish()
        if not self.catalog_present():
            for plan in self.plan_all([]):
                raise AssertionError("%s planned without a catalog" % plan.rel)
            return self.finish()
        with self.locked():
            for plan in self.plan_all(self.h.catalog_rows()):
                self.apply_one(plan)
        return self.finish()

    def run_probes(self):
        """A dry run plans. --apply plans read-only first (lstat only) and, when there is something to do, takes the
        lock and plans and applies each play again under it."""
        plans = list(self.plan_all(None))
        if not self.args.apply:
            for plan in plans:
                self.describe(plan, "would-delete")
            return self.finish()
        if plans:
            with self.locked():
                for old in plans:
                    plan = self.plan_unit(old.path, None, time.time())
                    if plan is not None:
                        self.apply_one(plan)
        return self.finish()

    @contextlib.contextmanager
    def locked(self):
        """catalog/harvest.lock (the harvester's flock) for the block; Refused when another run holds it."""
        try:
            self.h.lock()
        except harvest.Held as e:
            raise Refused(str(e))
        try:
            yield
        finally:
            self.h.unlock()

    def apply_one(self, plan):
        try:
            self.apply(plan)
        except Skip as e:
            self.report(e.status, plan.rel, str(e))
        except OSError as e:
            self.report("held", plan.rel, "I/O error: %s (%s, written before the first delete, lists every delete; "
                        "the next run checks and completes it)" % (e, RECORD_NAME))

    def finish(self):
        t = self.totals
        counts = dict((s, self.statuses.count(s)) for s in ("done", "skipped", "held", "refused"))
        print("total: %d %s, %d raw screenshot(s) %s, %d kept, %s %s (%d B); %d done earlier, %d skipped, %d held, "
              "%d refused" % (
                  t["units"], "probe play(s)" if self.args.probes else "session(s)", t["delete"],
                  "deleted" if self.args.apply else "to delete", t["kept"],
                  "freed" if self.args.apply else "to free", gib(t["bytes"]), t["bytes"],
                  counts["done"], counts["skipped"], counts["held"], counts["refused"]))
        if not self.args.apply:
            print("dry run: nothing was deleted or written (pass --apply to delete)")
        return EXIT_HELD if counts["held"] or counts["refused"] else EXIT_OK

    # -- pinned directories -----------------------------------------------------------------------
    def open_unit(self, plan):
        """A descriptor of the unit directory, opened from the root one component at a time with O_DIRECTORY |
        O_NOFOLLOW (a symlink anywhere on the way fails), which must be the directory planned (device, inode) and
        whose /proc/self/fd path must be its path below the root. Skip held otherwise."""
        try:
            fd = os.open(self.root, DIR_FLAGS)
        except OSError as e:
            raise Skip("held", "the root cannot be opened (%s); nothing was written or deleted" % e)
        try:
            done = []
            for part in plan.rel.split("/"):
                if part in ("", ".", ".."):
                    raise Skip("held", "bad path component in %s; nothing was written or deleted" % plan.rel)
                done.append(part)
                try:
                    nfd = os.open(part, DIR_FLAGS, dir_fd=fd)
                except OSError as e:
                    raise Skip("held", "%s cannot be opened as a directory without following a symlink (%s): it "
                               "changed after planning; nothing was written or deleted" % ("/".join(done), e))
                os.close(fd)
                fd = nfd
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != plan.ids["unit"]:
                raise Skip("held", "the directory was replaced after it was planned; nothing was written or deleted")
            try:
                where = os.readlink("/proc/self/fd/%d" % fd)
            except OSError as e:
                raise Skip("held", "the opened directory's path cannot be checked (%s); nothing was written or "
                           "deleted" % e)
            if where != os.path.join(self.root, plan.rel):
                raise Skip("held", "the opened directory resolves to %s, not to its path below the root; nothing was "
                           "written or deleted" % where)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def open_child(self, parent_fd, name, want):
        """A descriptor of directory name in parent_fd (O_DIRECTORY | O_NOFOLLOW) that must be the one planned."""
        try:
            fd = os.open(name, DIR_FLAGS, dir_fd=parent_fd)
        except OSError as e:
            raise Skip("held", "%s/ cannot be opened without following a symlink (%s); nothing was written or "
                       "deleted" % (name, e))
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino) != want:
            os.close(fd)
            raise Skip("held", "%s/ was replaced after it was planned; nothing was written or deleted" % name)
        return fd

    def write_record(self, rfd, doc):
        """RETENTION.json in the pinned record directory, durably: RETENTION.json.dd6tmp created (O_EXCL,
        O_NOFOLLOW), written and fsynced, renamed over RETENTION.json, then the directory fsynced. Every failure
        raises (OSError); a temporary file left behind is removed when possible."""
        data = record_text(doc)
        try:
            os.unlink(RECORD_TMP, dir_fd=rfd)
        except FileNotFoundError:
            pass
        fd = os.open(RECORD_TMP, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=rfd)
        try:
            try:
                harvest.write_all(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(RECORD_TMP, RECORD_NAME, src_dir_fd=rfd, dst_dir_fd=rfd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(RECORD_TMP, dir_fd=rfd)
            raise
        os.fsync(rfd)

    # -- apply one plan ---------------------------------------------------------------------------
    def apply(self, plan):
        ufd = self.open_unit(plan)
        fds = [ufd]
        try:
            ffd = self.open_child(ufd, FRAMES, plan.ids["frames"])
            fds.append(ffd)
            if plan.kind == "session":
                rfd = self.open_child(ufd, harvest.HARVEST_DIR, plan.ids["record_dir"])
                fds.append(rfd)
            else:
                rfd = ufd
            doc = plan.record
            if doc is None:
                if plan.kind == "probe_play":
                    for e in plan.delete:
                        e["sha256"] = self.hash_planned(ffd, e, plan)[0]
                doc = plan.document(self.args.every_n, "deleting")
                try:
                    self.write_record(rfd, doc)
                except OSError as e:
                    raise Skip("held", "%s could not be written durably (%s); nothing was deleted" % (RECORD_NAME, e))
                self.hook("after_record")
            freed = deleted = 0
            problem = None
            for e in plan.todo():
                try:
                    size = self.delete_checked(ffd, e, plan)
                except Skip as s:
                    problem = str(s)
                    break
                if size is None:
                    continue
                freed += size
                deleted += 1
                if deleted == 1:
                    self.hook("in_deletes")
            try:
                os.fsync(ffd)
            except OSError as e:
                problem = problem or "frames/ could not be synced after the deletes (%s)" % e
            self.hook("before_done")
            if problem:
                raise Skip("held", "%s; %d file(s) deleted before it, the rest kept; %s keeps status deleting and "
                           "the next run checks it again" % (problem, deleted, RECORD_NAME))
            if doc.get("status") != "done":
                try:
                    self.write_record(rfd, dict(doc, status="done", completed_utc=utc_now()))
                except OSError as e:
                    raise Skip("held", "%d file(s) deleted, but %s could not be marked done durably (%s); it still "
                               "lists every delete and the next run completes it" % (deleted, RECORD_NAME, e))
        finally:
            for fd in reversed(fds):
                os.close(fd)
        self.report("deleted", plan.rel, "%d raw screenshot(s) deleted, %d kept, %s freed (%d B); %s status done%s" % (
            deleted, len(plan.kept), gib(freed), freed, RECORD_NAME, "".join("; " + n for n in plan.notes)))
        self.totals["units"] += 1
        self.totals["delete"] += deleted
        self.totals["kept"] += len(plan.kept)
        self.totals["bytes"] += freed

    def hash_planned(self, ffd, e, plan):
        """(sha256, fstat) of a planned frame, read through a descriptor opened with O_NOFOLLOW in the pinned frames/
        descriptor; Skip held unless it is the regular file planned (device, inode, size, mtime) before and after
        the read. FileNotFoundError when it is gone."""
        name = frame_name(e["file"])
        if name is None:
            raise AssertionError("refusing %r: not a raw screenshot name" % (e["file"],))
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=ffd)
        except FileNotFoundError:
            raise
        except OSError as err:
            raise Skip("held", "%s cannot be opened without following a symlink (%s)" % (e["file"], err))
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or ident(st) != plan.stats.get(e["file"]):
                raise Skip("held", "%s changed after it was planned (not the same regular file, size or mtime)" %
                           e["file"])
            h = hashlib.sha256()
            for block in iter(lambda: os.read(fd, 1 << 20), b""):
                h.update(block)
            if ident(os.fstat(fd)) != ident(st):
                raise Skip("held", "%s changed while it was hashed" % e["file"])
        finally:
            os.close(fd)
        return h.hexdigest(), st

    def delete_checked(self, ffd, e, plan):
        """Unlink one planned frame in the pinned frames/ descriptor after checking it once more: the regular file
        planned, hashed through its descriptor to its sha256 (SHA256SUMS for a session, the record for a play), and
        still the same file by lstat just before the unlinkat. Returns its size, None when it is already gone; Skip
        held on any mismatch (nothing is unlinked)."""
        try:
            digest, st = self.hash_planned(ffd, e, plan)
        except FileNotFoundError:
            return None
        if digest != e["sha256"]:
            raise Skip("held", "%s does not match its sha256 (%s)" % (
                e["file"], "SHA256SUMS" if plan.kind == "session" else "recorded in %s" % RECORD_NAME))
        name = frame_name(e["file"])
        self.hook("before_unlink")
        now = os.stat(name, dir_fd=ffd, follow_symlinks=False)
        if not stat.S_ISREG(now.st_mode) or ident(now) != ident(st):
            raise Skip("held", "%s changed after it was hashed" % e["file"])
        os.unlink(name, dir_fd=ffd)
        return st.st_size


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
    ap.add_argument("--probes", action="store_true",
                    help="retain probe plays (<profile>/profile/DC_Probes) instead of captures/arland sessions")
    ap.add_argument("--apply", action="store_true", help="delete (default: a dry run that writes nothing)")
    ap.add_argument("--every-n", type=int, default=DEFAULT_EVERY_N,
                    help="keep every slot / frame that is a multiple of N (default %(default)s)")
    ap.add_argument("--min-age-days", type=float, default=DEFAULT_MIN_AGE_DAYS,
                    help="--probes: skip plays with a file younger than this (default %(default)s)")
    ap.add_argument("--under", default=None, help="only sessions / plays below this directory (inside the root)")
    ap.add_argument("--list", action="store_true", help="print every kept and deleted file")
    ap.add_argument("--test-root", default=None, help="TESTS ONLY: a /dev/shm/ or /mnt/e/ directory in place of %s"
                    % harvest.DATA_ROOT)
    args = ap.parse_args(argv)
    if args.every_n < 1:
        ap.error("--every-n must be at least 1")
    if args.min_age_days < 0:
        ap.error("--min-age-days must not be negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        root = harvest.data_root(args.test_root)
        return Retention(root, args, args.test_root is not None).run()
    except Refused as e:
        print("refused: %s" % e)
        return EXIT_REFUSED
    except Exception:  # an I/O error or a bug: exit 4, never a silent success
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
