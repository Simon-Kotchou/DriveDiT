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
  * catalog/sessions.jsonl has a complete row for the session, and every row naming it has this destination, route
    captures, a passing verdict and the sha256 of the session's SHA256SUMS;
  * _harvest/SHA256SUMS verifies (harvest.verify_sums hashes every file; a symlink anywhere in the session holds it);
  * the slot's derivative _harvest/deriv/f_<slot:08d>.jpg is listed in SHA256SUMS (so it exists and verified).
The audit sample is kept: the first and the last raw slot, every slot that is a multiple of --every-n (default
100), and every slot whose image time (img issue_world_ms, else requested_world_ms) lies within 2 s of an events row
of kind collision, stuck, respawn, teleport or script_error (when the session has an events stream). A raw file
whose slot has no img row, or no derivative, is kept.

Probe plays (--probes). <profile>/profile/DC_Probes/<probe>/<launch_id>/play_NN/frames/f_<n:08d>.png (.bmp.png,
.bmp) of wb_profile, wb_profile_x19 and client_profiles/iNN: diagnostic screenshots, not training data. A play's
frames may be deleted only when its run_footer.txt exists (schema dc-probe-run/1: the play ended and closed its
files), its run_header.txt names world.map_id=arland, and no file of the play (its own files and frames/; not
RETENTION.json) was modified in the last --min-age-days (default 1). Kept: the first and the last frame and every
frame whose number is a multiple of --every-n.

The record. Before the first delete, RETENTION.json is written atomically (RETENTION.json.dd6tmp, fsync, rename,
fsync of the directory): in _harvest/ of a session, in the play directory of a probe play. It holds the rule
version, N, the event window and kinds, the kept files with their reasons, the deleted files with their sha256
(from SHA256SUMS for a session; hashed just before the record is written for a probe play) and size, the bytes
freed, the time and status "deleting"; once every delete is done it is rewritten with status "done". SHA256SUMS is
never changed: harvest.verify_sums counts a raw file the record lists as deleted as accounted for, so the harvester
still sees a retained session as complete and never rebuilds or re-validates it.

Re-runs and kills. An existing record is the plan: a re-run deletes what it lists and is still there (a session's
files are verified against SHA256SUMS first, a probe's against the recorded sha256) and then marks it done; a new
--every-n does not change a recorded plan. A kill at any point leaves every file the record lists either present
(and verified on the next run) or deleted, so the next run completes the record. A run with nothing to do writes
nothing. --apply on sessions plans once, under the lock (planning hashes every file), and only when the catalog
exists; on probe plays it plans read-only first and takes the lock only when there is something to delete.

Safety. Every path lies below /mnt/e/DriveDiT (harvest.data_root: not reached through a symlink, not on C:). --under
DIR narrows the scan; it must lie below the root without a symlink, inside captures/arland (sessions) or a
DC_Probes folder (--probes). A $profile:DrivingData (dc-1.0) path or session is refused. Symlinks are never
followed: a symlinked directory on the way holds what lies below it, and a delete is an unlinkat in the frames/
directory opened with O_NOFOLLOW, of a regular file named f_<8 digits>.png / .bmp.png / .bmp, nothing else. --apply
holds catalog/harvest.lock (the harvester's), so it never runs beside the harvester or another retention run.

Exit codes: 0 done or nothing to do (skips are normal: unsealed, too new, not PASS, not catalogued); 2 a session or
play was held or refused (checksum mismatch, symlink, bad record, not Arland, dc-1.0) and nothing of it deleted; 3
the run was refused (arguments, roots, --under outside the root, the lock); 4 an unexpected error.

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

RULE_VERSION = "retention/1 (DD-6-min)"
RECORD_NAME = harvest.RETENTION_NAME
RECORD_SCHEMA = harvest.RETENTION_SCHEMA
DEFAULT_EVERY_N = 100
DEFAULT_MIN_AGE_DAYS = 1.0
EVENT_WINDOW_MS = 2000
EVENT_KINDS = ("collision", "stuck", "respawn", "teleport", "script_error")
RAW_SUFFIXES = (".png", ".bmp.png", ".bmp")
RAW_NAME_RE = re.compile(r"^f_([0-9]{8})(\.png|\.bmp\.png|\.bmp)$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
PROBE_RE = re.compile(r"^[a-z][a-z0-9]{0,31}$")
PLAY_RE = re.compile(r"^play_[0-9]{2,4}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROBE_SCHEMA = "dc-probe-run/1"
PROBES_DIR = "DC_Probes"
LEGACY_DIR = "DrivingData"
FRAMES = "frames"
PASSING = ("PASS", "PASS_WITH_WARNINGS")
MAX_REPORT_BYTES = 256 << 20
KILL_ENV = "DD6_RETENTION_KILL_AT"
HOOKS = ("after_record", "in_deletes", "before_done")
EXIT_OK, EXIT_HELD, EXIT_REFUSED, EXIT_ERROR = 0, 2, 3, 4


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
    """{key: value} of a key=value text file (run_header.txt, run_footer.txt), or None when it is not a regular
    file."""
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    out = {}
    with open(path, "rb") as f:
        for line in f.read(1 << 20).decode("utf-8", errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() not in out:
                out[key.strip()] = value.strip()
    return out


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


def audit_sample(slots, every_n, times=None, events=()):
    """{slot: [reasons]} of the slots kept as the audit sample: first, last, every_<N> (slot % N == 0) and, with
    times ({slot: world ms}), event:<kind> for a slot within EVENT_WINDOW_MS of an event and no_img_time for a slot
    without a time."""
    keep = {}
    if not slots:
        return keep
    first, last = min(slots), max(slots)
    for s in slots:
        why = []
        if s == first:
            why.append("first")
        if s == last:
            why.append("last")
        if s % every_n == 0:
            why.append("every_%d" % every_n)
        if times is not None:
            t = times.get(s)
            if t is None:
                why.append("no_img_time")
            else:
                for kind in sorted(set(k for k, et in events if abs(t - et) <= EVENT_WINDOW_MS)):
                    why.append("event:%s" % kind)
        if why:
            keep[s] = why
    return keep


class Plan:
    """What retention does to one session or play: kept [{file, slot, why}], delete [{file, slot, sha256, bytes}]
    (paths relative to the unit directory), the record (an existing valid RETENTION.json, else None) and what is
    still to do."""

    def __init__(self, kind, path, rel, record_dir):
        self.kind = kind                    # "session" | "probe_play"
        self.path = path
        self.rel = rel
        self.record_dir = record_dir
        self.frames = os.path.join(path, FRAMES)
        self.ident = {}
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
            bases += [os.path.join(cdir, n) for n in sorted(os.listdir(cdir)) if harvest.CLIENT_INSTANCE_RE.match(n)]
        out, held = [], []
        for base in bases:
            probes = os.path.join(base, harvest.PROFILE_DIR, PROBES_DIR)
            if os.path.islink(probes) or (os.path.isdir(probes) and not harvest.real_below(probes, self.root)):
                held.append(probes)
                continue
            if not os.path.isdir(probes):
                continue
            for probe in self.subdirs(probes, held):
                if not PROBE_RE.match(os.path.basename(probe)):
                    continue
                for launch in self.subdirs(probe, held):
                    if not harvest.LAUNCH_ID_RE.match(os.path.basename(launch)):
                        continue
                    out += [p for p in self.subdirs(launch, held) if PLAY_RE.match(os.path.basename(p))]
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
        if os.path.islink(os.path.join(path, FRAMES)):
            raise Skip("held", "frames/ is a symlink; never followed")
        try:
            ok, detail = harvest.verify_sums(path)
        except harvest.Held as e:
            ok, detail = False, str(e)
        if not ok:
            raise Skip("held", "SHA256SUMS does not verify: %s; nothing is deleted" % detail)
        listed = harvest.parse_sums(sums_bytes)
        plan = Plan("session", path, rel, hdir)
        plan.ident = {"session_id": name, "path": rel, "verdict": verdict, "screenshot_suffix": suffix,
                      "sha256sums_sha256": sums_sha}
        record_path = os.path.join(hdir, RECORD_NAME)
        if os.path.lexists(record_path):
            return self.from_record(plan, record_path, listed)     # verify_sums validated it against SHA256SUMS
        raw = {}
        for f_rel in listed:
            head, _, fname = f_rel.partition("/")
            m = RAW_NAME_RE.match(fname)
            if head == FRAMES and m and m.group(2) == suffix:
                raw[int(m.group(1))] = f_rel
        if not raw:
            raise Skip("skipped", "SHA256SUMS lists no raw screenshot frames/f_<slot:08d>%s" % suffix)
        times, events = self.session_times(path)
        keep = audit_sample(sorted(raw), self.args.every_n, times, events)
        for slot in sorted(raw):
            if "%s/deriv/f_%08d.jpg" % (harvest.HARVEST_DIR, slot) not in listed:
                keep.setdefault(slot, []).append("no_derivative")
        for slot in sorted(raw):
            f_rel = raw[slot]
            if slot in keep:
                plan.kept.append({"file": f_rel, "slot": slot, "why": keep[slot]})
            else:
                st = os.lstat(os.path.join(path, f_rel))
                plan.delete.append({"file": f_rel, "slot": slot, "sha256": listed[f_rel], "bytes": st.st_size})
                plan.present.add(f_rel)
        return plan

    def session_times(self, path):
        """({slot: image world ms}, [(kind, event world ms)] of the window kinds) from the img and events streams
        (the reader's committed rows). Held when either stream does not read cleanly."""
        real = os.path.realpath(path)
        problems = []
        img = reader.read_stream(real, "img", self.schema, problems)
        events = reader.read_stream(real, "events", self.schema, problems)
        if problems:
            p = problems[0]
            raise Skip("held", "the img / events streams do not read cleanly (%s %s: %s), so the event windows are "
                       "unknown" % (p.check, p.where, p.message))
        if not img.present:
            raise Skip("held", "no img stream, so the slot times (event windows) are unknown")
        times = {}
        for slot, issued, requested in zip(img.data["slot_id"], img.data["issue_world_ms"],
                                           img.data["requested_world_ms"]):
            times.setdefault(slot, issued if issued is not None else requested)
        ev = [(events.data["kind"][i], events.data["world_time_ms"][i]) for i in range(len(events))
              if events.data["kind"][i] in EVENT_KINDS]
        return times, ev

    # -- probe plays ------------------------------------------------------------------------------
    def plan_play(self, path, now):
        rel = self.rel(path)
        if not harvest.real_below(path, self.root):
            raise Skip("held", "reached through a symlink")
        footer_path = os.path.join(path, "run_footer.txt")
        if os.path.islink(footer_path):
            raise Skip("held", "run_footer.txt is a symlink")
        footer = read_kv(footer_path)
        if footer is None:
            raise Skip("skipped", "incomplete play: no run_footer.txt; nothing is deleted")
        if footer.get("schema") != PROBE_SCHEMA:
            raise Skip("held", "run_footer.txt is not a %s footer" % PROBE_SCHEMA)
        header = read_kv(os.path.join(path, "run_header.txt"))
        if header is None or header.get("world.map_id") != "arland":
            raise Skip("refused", "run_header.txt does not name world.map_id=arland; only Arland probe plays are "
                       "retained")
        frames = os.path.join(path, FRAMES)
        if os.path.islink(frames):
            raise Skip("held", "frames/ is a symlink; never followed")
        if not os.path.isdir(frames):
            raise Skip("skipped", "no frames/ (nothing to delete)")
        plan = Plan("probe_play", path, rel, path)
        plan.ident = {"path": rel, "probe": rel.split("/")[-3], "launch_id": rel.split("/")[-2],
                      "play": rel.split("/")[-1], "footer_status": footer.get("status")}
        newest = 0.0
        for name in os.listdir(path):
            st = os.lstat(os.path.join(path, name))
            if stat.S_ISREG(st.st_mode) and name not in (RECORD_NAME, RECORD_NAME + harvest.TMP_SUFFIX):
                newest = max(newest, st.st_mtime)
        raw = {}
        others = 0
        for name in sorted(os.listdir(frames)):
            st = os.lstat(os.path.join(frames, name))
            m = RAW_NAME_RE.match(name)
            if stat.S_ISREG(st.st_mode):
                newest = max(newest, st.st_mtime)
            if m and stat.S_ISREG(st.st_mode):
                raw.setdefault(int(m.group(1)), []).append((name, st.st_size))
            else:
                others += 1
        if others:
            plan.notes.append("%d other entr%s in frames/ left alone" % (others, "y" if others == 1 else "ies"))
        age = now - newest
        if age < self.args.min_age_days * 86400.0:
            raise Skip("skipped", "newer than --min-age-days %g (last written %.1f h ago); nothing is deleted" % (
                self.args.min_age_days, age / 3600.0))
        record_path = os.path.join(path, RECORD_NAME)
        if os.path.lexists(record_path):
            return self.from_record(plan, record_path, None)
        keep = audit_sample(sorted(raw), self.args.every_n)
        for n in sorted(raw):
            for name, size in raw[n]:
                f_rel = "%s/%s" % (FRAMES, name)
                if n in keep:
                    plan.kept.append({"file": f_rel, "slot": n, "why": keep[n]})
                else:
                    plan.delete.append({"file": f_rel, "slot": n, "sha256": None, "bytes": size})
                    plan.present.add(f_rel)
        if not raw:
            raise Skip("skipped", "frames/ holds no raw screenshot (nothing to delete)")
        return plan

    # -- an existing record -----------------------------------------------------------------------
    def from_record(self, plan, record_path, listed):
        """The plan an existing RETENTION.json records (it is the plan: a re-run only completes it). For a session,
        harvest.verify_sums has already checked it against SHA256SUMS (listed); for a play it must name the play
        and list only raw frames/ names with a sha256. Held when it is not a valid record."""
        doc = load_json(record_path, harvest.RETENTION_MAX_BYTES)
        ok = doc is not None and doc.get("schema") == RECORD_SCHEMA and doc.get("kind") == plan.kind \
            and doc.get("path") == plan.rel and isinstance(doc.get("deleted"), list) \
            and isinstance(doc.get("kept"), list)
        entries = doc.get("deleted") if ok else []
        for e in entries:
            f_rel = e.get("file") if isinstance(e, dict) else None
            digest = e.get("sha256") if isinstance(e, dict) else None
            if not (isinstance(f_rel, str) and f_rel.startswith(FRAMES + "/")
                    and RAW_NAME_RE.match(f_rel[len(FRAMES) + 1:]) and isinstance(digest, str)
                    and SHA256_RE.match(digest) and isinstance(e.get("bytes"), int)
                    and (listed is None or listed.get(f_rel) == digest)):
                ok = False
                break
        if not ok:
            raise Skip("held", "%s exists but is not a valid retention record of this %s; nothing is deleted" % (
                RECORD_NAME, "session" if plan.kind == "session" else "play"))
        plan.record = doc
        plan.kept = doc["kept"]
        plan.delete = entries
        for e in entries:
            if os.path.lexists(os.path.join(plan.path, e["file"])):
                plan.present.add(e["file"])
        if doc.get("every_n") != self.args.every_n:
            plan.notes.append("recorded with --every-n %s; the recorded plan stands" % doc.get("every_n"))
        return plan

    # -- the run ----------------------------------------------------------------------------------
    def plan_all(self):
        """[Plan] of every unit in scope that retention may act on; every other unit is reported."""
        if self.args.under:
            self.under = self.check_under(self.args.under)
        now = time.time()
        if self.args.probes:
            dirs, held = self.play_dirs()
            catalog = None
        else:
            dirs, held = self.session_dirs()
            catalog = self.h.catalog_rows()
        for path in held:
            if self.wanted(path):
                self.report("held", self.rel(path), "a symlink; never followed, nothing below it is touched")
        plans = []
        for path in dirs:
            if not self.wanted(path):
                continue
            plan = self.plan_unit(path, catalog, now)
            if plan is not None:
                plans.append(plan)
        return plans

    def plan_unit(self, path, catalog, now):
        rel = self.rel(path)
        try:
            plan = self.plan_play(path, now) if self.args.probes else self.plan_session(path, catalog)
        except Skip as e:
            shown = harvest.hidden(rel) if str(e) == harvest.NOT_ARLAND else rel
            self.report(e.status, shown, str(e))
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
            for why in e["why"]:
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
                print("  keep   %s (%s)" % (e["file"], ",".join(e["why"])))
            for e in todo:
                print("  delete %s %d" % (e["file"], e["bytes"]))
        self.totals["units"] += 1
        self.totals["delete"] += len(todo)
        self.totals["kept"] += len(plan.kept)
        self.totals["bytes"] += plan.bytes_todo()

    def run(self):
        """A dry run plans and prints. --apply on sessions plans once, under the lock, when there is a catalog
        (planning hashes every file; without a catalog no session qualifies and nothing is written). --apply on
        probe plays plans read-only first (lstat only), takes the lock only when there is something to do and plans
        again under it."""
        if self.args.apply and not self.args.probes and os.path.isfile(self.h.catalog_path()):
            with self.locked():
                for plan in self.plan_all():
                    self.apply_one(plan)
            return self.finish()
        plans = self.plan_all()
        if not self.args.apply:
            for plan in plans:
                self.describe(plan, "would-delete")
            return self.finish()
        if plans:
            with self.locked():
                now = time.time()
                for old in plans:
                    plan = self.plan_unit(old.path, None, now)
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
                        "the next run completes it)" % (e, RECORD_NAME))

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

    # -- apply one plan ---------------------------------------------------------------------------
    def write_record(self, plan, doc):
        if os.path.islink(plan.record_dir) or not os.path.isdir(plan.record_dir):
            raise Skip("held", "%s is a symlink or gone; nothing was written" % self.rel(plan.record_dir))
        harvest.replace_file(os.path.join(plan.record_dir, RECORD_NAME), record_text(doc))
        harvest.fsync_dir(plan.record_dir)

    def apply(self, plan):
        try:
            dfd = os.open(plan.frames, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as e:
            raise Skip("held", "frames/ cannot be opened without following a symlink (%s); nothing was deleted" % e)
        try:
            doc = plan.record
            if doc is None:
                if plan.kind == "probe_play":
                    for e in plan.delete:
                        e["sha256"] = self.hash_at(dfd, e["file"])
                doc = plan.document(self.args.every_n, "deleting")
                self.write_record(plan, doc)
                self.hook("after_record")
            freed = deleted = 0
            problems = []
            for e in plan.todo():
                name = e["file"][len(FRAMES) + 1:]
                if not RAW_NAME_RE.match(name):
                    raise AssertionError("refusing to delete %s: not a raw screenshot name" % e["file"])
                try:
                    st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    problems.append("%s is not a regular file" % e["file"])
                    continue
                if plan.record is not None and plan.kind == "probe_play" and self.hash_at(dfd, e["file"]) \
                        != e["sha256"]:
                    problems.append("%s does not match its recorded sha256" % e["file"])
                    continue
                os.unlink(name, dir_fd=dfd)
                freed += st.st_size
                deleted += 1
                if deleted == 1:
                    self.hook("in_deletes")
            os.fsync(dfd)
        finally:
            os.close(dfd)
        self.hook("before_done")
        if problems:
            raise Skip("held", "%d file(s) were not deleted (%s); %s keeps status deleting" % (
                len(problems), problems[0], RECORD_NAME))
        if doc.get("status") != "done":
            doc = dict(doc, status="done", completed_utc=utc_now())
            self.write_record(plan, doc)
        self.report("deleted", plan.rel, "%d raw screenshot(s) deleted, %d kept, %s freed (%d B); %s status done%s" % (
            deleted, len(plan.kept), gib(freed), freed, RECORD_NAME, "".join("; " + n for n in plan.notes)))
        self.totals["units"] += 1
        self.totals["delete"] += deleted
        self.totals["kept"] += len(plan.kept)
        self.totals["bytes"] += freed

    def hash_at(self, dfd, f_rel):
        """sha256 of a regular file of the frames/ directory (opened with O_NOFOLLOW)."""
        name = f_rel[len(FRAMES) + 1:]
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise Skip("held", "%s is not a regular file; nothing was deleted" % f_rel)
            h = hashlib.sha256()
            for block in iter(lambda: os.read(fd, 1 << 20), b""):
                h.update(block)
            return h.hexdigest()
        finally:
            os.close(fd)


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
