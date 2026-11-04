#!/usr/bin/env python3
"""Tests for tools/harvester/retention.py (DD-6-min, the raw-screenshot retention rule) and the retention-aware
SHA256SUMS verification of tools/harvester/harvest.py.

    TMPDIR=/dev/shm HOME=/dev/shm python3 -I -B -m pytest --noconftest -p no:cacheprovider -q tests/test_retention.py

Every test builds a synthetic data root in a temporary directory under $TMPDIR (on /dev/shm/ or /mnt/e/ outside
/mnt/e/DriveDiT, else /dev/shm/) and passes it with --test-root, so nothing below /mnt/e/DriveDiT is read or
written. Harvested sessions are built by hand (make_captured: the layout harvest.py leaves at captures/arland/..., a
tiny PNG per slot, img and events streams written with the DD-5-min generator's writer, _harvest/ with result.json,
dc_validate.json, deriv/, SHA256SUMS and a catalog row); one test harvests the generator's golden session with the
real harvester first. Probe plays are built by make_play (run_header.txt, run_footer.txt, frames/, old mtimes).

Covered: a PASS session keeps exactly the audit sample (first, last, every Nth, event windows, slots without an img
row or a derivative) and records every delete with its SHA256SUMS hash; FAIL, _rejected, unharvested,
uncatalogued, checksum-mismatched, non-Arland, dc-1.0 and symlinked sessions lose nothing; a re-run is a no-op; a
kill at each hook leaves a record the next run completes; the harvester neither rebuilds nor holds a retained
session; probe plays: complete vs incomplete, too new, not Arland, symlinks, kill and re-verification; dry runs
write nothing; paths outside the root, DrivingData and _rejected are refused.
"""

import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import unittest
import zlib
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RETENTION = os.path.join(REPO, "tools", "harvester", "retention.py")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ret = load("dd6_retention", "tools/harvester/retention.py")
harvest = ret.harvest
synth = load("dc2_make_synthetic_r", "tests/fixtures/capture_dc2/make_synthetic.py")
th = load("dd6_test_harvest_r", "tests/test_harvest.py")        # helpers only (temp_dir, snapshot, make_session)
STREAMS = ret.reader.load_schema().raw["streams"]

DATE, TRIAL = "2026-10-25", "T0b"
PERIOD_MS = 200                     # 5 Hz image slots: +-2 s is +-10 slots
T0_MS = 10000
LAUNCH = "20261023T000000Z-wb-0a0a"
BUILD_KEY = hashlib.sha256(b"synthetic retention build").hexdigest()


def sid_of(tag):
    return "arland_%s_20261025T091512Z_wb_i00_%s" % (TRIAL, tag)


def tiny_png(n):
    """A valid 1x1 RGB PNG whose pixel encodes n (so every slot's file and hash differ)."""
    def chunk(tag, payload):
        crc = zlib.crc32(tag + payload) & 0xffffffff
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)
    pixel = b"\x00" + bytes([n & 255, (n >> 8) & 255, 7])
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixel)) + chunk(b"IEND", b""))


def slot_ms(slot):
    """issue_world_ms of a slot in make_captured."""
    return T0_MS + slot * PERIOD_MS + 5


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def jtext(obj):
    return (json.dumps(obj, indent=1) + "\n").encode("ascii")


def read_json(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def resum(dest):
    files = harvest.tree_files(dest)
    write(os.path.join(dest, "_harvest", "SHA256SUMS"), harvest.sums_text(files, "_harvest/SHA256SUMS"))


def catalogue(root, dest):
    """The session's catalog row, appended under catalog/harvest.lock as harvest.py does (so the lock file exists)."""
    h = harvest.Harvester(root, harvest.parse_args([]), True)
    h.lock()
    try:
        h.catalog_append(h.catalog_row(dest))
    finally:
        h.unlock()


def make_captured(root, tag="5e01", slots=range(250), events=(), verdict="PASS", suffix=".png", harvested=True,
                  catalogued=True, no_deriv=(), no_img=(), where=None):
    """A session as harvest.py leaves it at captures/arland/<date>/<trial>/<id> (where: another parent, relative
    to root): session.json, a tiny PNG per slot, img rows (5 Hz) and events rows ([(kind, world ms)]), harvest.json
    and, when harvested, _harvest/ (dc_validate.json, result.json, deriv/f_<slot>.jpg except no_deriv, SHA256SUMS)
    and, when catalogued, its catalog row. Returns (session id, session directory)."""
    sid = sid_of(tag)
    parent = os.path.join(root, where) if where else os.path.join(root, "captures", "arland", DATE, TRIAL)
    dest = os.path.join(parent, sid)
    rel = os.path.relpath(dest, root)
    meta = {"session_id": sid, "contract_version": "dc-2.0",
            "map": {"map_id": "arland", "terrain": "Arland",
                    "root_world": "{A9806AF617972E97}worlds/Arland/Arland.ent"},
            "trial": {"trial_id": TRIAL}, "exe": {"kind": "wb", "instance": "i00"}}
    write(os.path.join(dest, "session.json"), jtext(meta))
    write(os.path.join(dest, "session_end.json"), jtext({"status": "ok", "session_id": sid}))
    for s in slots:
        write(os.path.join(dest, "frames", "f_%08d%s" % (s, suffix)), tiny_png(s))
    img = [{"slot_id": s, "episode_id": 0, "ego_slot_id": 0, "requested_world_ms": T0_MS + s * PERIOD_MS,
            "issue_status": "issued", "issue_world_ms": slot_ms(s), "issue_render_frame": 100 + 4 * s,
            "issue_physics_step": 50 + 6 * s, "path": "frames/f_%08d" % s, "cam_current": 1, "cam_attached": 1}
           for s in slots if s not in no_img]
    synth.write_stream(os.path.join(dest, "img"), STREAMS["img"], img, 10.0)
    rows = [{"event_seq": i, "episode_id": 0, "physics_step": 10 * i, "render_frame": 20 * i, "world_time_ms": ms,
             "kind": kind, "veh_slot_id": 0, "payload": None}
            for i, (kind, ms) in enumerate([("session_start", 0)] + sorted(events, key=lambda e: e[1]))]
    synth.write_stream(os.path.join(dest, "events"), STREAMS["events"], rows, 10.0)
    write(os.path.join(dest, "harvest.json"), jtext({
        "schema": harvest.HARVEST_SCHEMA, "session_id": sid, "harvester_version": harvest.HARVESTER_VERSION,
        "pin_id": "1.8.0.13", "build_key": BUILD_KEY, "build_key_source": "session.json", "use_class": "dev",
        "channel": "stable", "screenshot_suffix": suffix}))
    if harvested:
        hdir = os.path.join(dest, "_harvest")
        code = {"PASS": 0, "PASS_WITH_WARNINGS": 2, "FAIL": 1}[verdict]
        write(os.path.join(hdir, "dc_validate.json"), jtext({"verdict": verdict, "exit_code": code}))
        write(os.path.join(hdir, "result.json"), jtext({
            "schema": harvest.RESULT_SCHEMA, "session_id": sid, "date": DATE,
            "harvester_version": harvest.HARVESTER_VERSION, "created_utc": "2026-10-25T10:00:00Z",
            "verdict": verdict, "exit_code": code, "route": "captures", "dest": rel, "images": len(slots),
            "decoded": len(slots), "derivatives": len(slots) - len(no_deriv)}))
        for s in slots:
            if s not in no_deriv:
                write(os.path.join(hdir, "deriv", "f_%08d.jpg" % s), b"\xff\xd8\xff synthetic derivative %d" % s)
        resum(dest)
        if catalogued:
            catalogue(root, dest)
    return sid, dest


def make_play(root, probe="e05", launch=LAUNCH, play="play_01", frames=range(250), footer="complete",
              map_id="arland", age_days=3.0, profile="wb_profile", reason="done"):
    """A probe play <profile>/profile/DC_Probes/<probe>/<launch>/<play>/ with run_header.txt, run_footer.txt (unless
    footer is None; footer is its status, "" for a footer without a status line), a CSV and frames/f_<n:08d>.png,
    every file last written age_days ago. Returns the play directory."""
    pdir = os.path.join(root, profile, "profile", "DC_Probes", probe, launch, play)
    write(os.path.join(pdir, "run_header.txt"), ("schema=dc-probe-run/1\nprobe=%s\nlaunch.id=%s\n"
                                                 "world.map_id=%s\n" % (probe, launch, map_id)).encode("ascii"))
    if footer is not None:
        status = "status=%s\n" % footer if footer else ""
        write(os.path.join(pdir, "run_footer.txt"), ("schema=dc-probe-run/1\n%sreason=%s\n" % (
            status, reason)).encode("ascii"))
    write(os.path.join(pdir, "%s_frames.csv" % probe), b"slot,frame\n0,1\n")
    for n in frames:
        write(os.path.join(pdir, "frames", "f_%08d.png" % n), tiny_png(n))
    age(pdir, age_days)
    return pdir


def age(path, days):
    """Set the mtime of every file below path to days ago."""
    t = time.time() - days * 86400.0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            os.utime(os.path.join(dirpath, name), (t, t), follow_symlinks=False)


def run(root, *argv, quiet="0"):
    """(exit code, stdout) of retention.main in this process. The fixtures are fresh, so the --apply preflight
    window is 0 s unless quiet says otherwise (None: the default 10 minutes)."""
    out = io.StringIO()
    with mock.patch.dict(os.environ):
        if quiet is None:
            os.environ.pop(ret.QUIET_ENV, None)
        else:
            os.environ[ret.QUIET_ENV] = quiet
        with contextlib.redirect_stdout(out):
            code = ret.main(["--test-root", root] + list(argv))
    return code, out.getvalue()


def run_subprocess(root, kill_at, *argv):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop(ret.KILL_ENV, None)
    env[ret.QUIET_ENV] = "0"
    if kill_at:
        env[ret.KILL_ENV] = kill_at
    return subprocess.run([sys.executable, "-I", "-B", RETENTION, "--test-root", root] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def frame_slots(dest):
    """The slots of the raw screenshots in frames/ (the staging directory and other names left out)."""
    return sorted(int(n[2:10]) for n in os.listdir(os.path.join(dest, "frames")) if n.startswith("f_"))


def staged(unit):
    """The names in frames/.retention_staging/ (None when there is no staging directory)."""
    sdir = os.path.join(unit, "frames", ret.STAGING)
    return sorted(os.listdir(sdir)) if os.path.isdir(sdir) else None


def line_of(out, status, rel):
    lines = [line for line in out.splitlines() if line.startswith("%s %s: " % (status, rel))]
    return lines[0] if lines else None


class SessionRetentionTest(unittest.TestCase):
    def setUp(self):
        self.root = th.temp_dir("dd6ret_s_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def rel(self, path):
        return os.path.relpath(path, self.root)

    def test_pass_session_keeps_only_the_audit_sample(self):
        sid, dest = make_captured(self.root)
        with open(os.path.join(dest, "_harvest", "SHA256SUMS"), "rb") as f:
            sums = f.read()
        listed = harvest.parse_sums(sums)
        before = th.snapshot(self.root)
        code, out = run(self.root)                                     # the default is a dry run
        self.assertEqual(code, 0, out)
        self.assertEqual(th.snapshot(self.root), before)
        line = line_of(out, "would-delete", self.rel(dest))
        self.assertIsNotNone(line, out)
        self.assertIn("delete 246 raw screenshot(s), keep 4 (every_100 3, first 1, last 1)", line)
        self.assertIn("dry run: nothing was deleted or written", out)

        code, out = run(self.root, "--apply")
        self.assertEqual(code, 0, out)
        self.assertIsNotNone(line_of(out, "deleted", self.rel(dest)), out)
        self.assertEqual(frame_slots(dest), [0, 100, 200, 249])
        self.assertEqual(len(os.listdir(os.path.join(dest, "_harvest", "deriv"))), 250)   # every derivative stays
        with open(os.path.join(dest, "_harvest", "SHA256SUMS"), "rb") as f:
            self.assertEqual(f.read(), sums)                           # SHA256SUMS is never changed
        doc = read_json(os.path.join(dest, "_harvest", "RETENTION.json"))
        self.assertEqual((doc["schema"], doc["rule_version"], doc["kind"], doc["session_id"], doc["status"]),
                         ("drivedit.retention/1", ret.RULE_VERSION, "session", sid, "done"))
        self.assertEqual((doc["every_n"], doc["event_window_ms"], doc["event_kinds"]),
                         (100, 2000, ["collision", "stuck", "respawn", "teleport", "script_error"]))
        self.assertEqual(doc["sha256sums_sha256"], hashlib.sha256(sums).hexdigest())
        self.assertEqual([(k["slot"], k["why"]) for k in doc["kept"]],
                         [(0, ["first", "every_100"]), (100, ["every_100"]), (200, ["every_100"]), (249, ["last"])])
        self.assertEqual(len(doc["deleted"]), 246)
        for e in doc["deleted"]:
            self.assertEqual(e["sha256"], listed[e["file"]], e)       # each hash is the SHA256SUMS line
            self.assertEqual(e["sha256"], hashlib.sha256(tiny_png(e["slot"])).hexdigest())
            self.assertEqual(e["bytes"], len(tiny_png(e["slot"])))
        self.assertEqual(doc["bytes_freed"], sum(len(tiny_png(e["slot"])) for e in doc["deleted"]))
        self.assertRegex(doc["created_utc"], r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z$")
        self.assertRegex(doc["completed_utc"], r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z$")
        ok, detail = harvest.verify_sums(dest)
        self.assertTrue(ok, detail)
        self.assertIn("246 raw screenshot(s) deleted by retention", detail)

        after = th.snapshot(self.root)
        for argv in (["--apply"], ["--apply", "--every-n", "10"], []):    # idempotent; the recorded plan stands
            code, out = run(self.root, *argv)
            self.assertEqual(code, 0, out)
            self.assertIsNotNone(line_of(out, "done", self.rel(dest)), out)
            self.assertEqual(th.snapshot(self.root), after)

    def test_every_n_and_pass_with_warnings(self):
        _, dest = make_captured(self.root, verdict="PASS_WITH_WARNINGS")
        code, out = run(self.root, "--apply", "--every-n", "50")
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(dest), [0, 50, 100, 150, 200, 249])
        self.assertEqual(read_json(os.path.join(dest, "_harvest", "RETENTION.json"))["verdict"], "PASS_WITH_WARNINGS")

    def test_event_windows_and_unplaceable_slots_are_kept(self):
        events = [("collision", slot_ms(120)), ("stuck", slot_ms(40) + 1500), ("respawn", slot_ms(160)),
                  ("teleport", slot_ms(230)), ("script_error", slot_ms(5)),
                  ("pause", slot_ms(70)), ("episode_start", slot_ms(180)), ("nav_command", slot_ms(150))]
        _, dest = make_captured(self.root, events=events, no_img=(77,), no_deriv=(144,))
        code, out = run(self.root, "--apply")
        self.assertEqual(code, 0, out)
        want = {0, 100, 200, 249, 77, 144}
        for kind, ms in events:
            if kind in ret.EVENT_KINDS:
                want |= set(s for s in range(250) if abs(slot_ms(s) - ms) <= 2000)
        self.assertEqual(frame_slots(dest), sorted(want))
        self.assertTrue(set(range(110, 131)) <= want and 109 not in want and 131 not in want)    # +-2 s at 5 Hz
        self.assertTrue(set(range(38, 58)) <= want)                   # the stuck window, off the slot grid
        self.assertNotIn(70, want)                                      # pause, episode_start, nav_command: no window
        kept = dict((k["slot"], k["why"]) for k in read_json(os.path.join(dest, "_harvest", "RETENTION.json"))["kept"])
        self.assertEqual(kept[120], ["event:collision"])
        self.assertEqual(kept[5], ["event:script_error"])
        self.assertEqual(kept[77], ["no_img_time"])
        self.assertEqual(kept[144], ["no_derivative"])
        self.assertTrue(harvest.verify_sums(dest)[0])

    def test_nothing_is_deleted_unless_every_gate_holds(self):
        def fail(root):
            make_captured(root, verdict="FAIL")

        def rejected(root):
            make_captured(root, where="captures/_rejected")

        def unharvested(root):
            make_captured(root, harvested=False)

        def uncatalogued(root):
            make_captured(root, catalogued=False)

        def checksum_mismatch(root):
            _, dest = make_captured(root)
            with open(os.path.join(dest, "frames", "f_00000007.png"), "ab") as f:
                f.write(b"x")

        def missing_file(root):
            _, dest = make_captured(root)
            os.remove(os.path.join(dest, "frames", "f_00000007.png"))

        def catalog_disagrees(root):
            _, dest = make_captured(root)
            write(os.path.join(dest, "notes.txt"), b"added after the catalog row\n")
            resum(dest)                                                 # verifies, but not the catalogued sums

        def symlink_inside(root):
            _, dest = make_captured(root)
            os.symlink(os.path.join(dest, "frames", "f_00000001.png"), os.path.join(dest, "frames", "link.png"))

        def symlinked_frames(root):
            _, dest = make_captured(root)
            outside = os.path.join(root, "outside_frames")
            os.rename(os.path.join(dest, "frames"), outside)
            os.symlink(outside, os.path.join(dest, "frames"))

        def not_arland(root):
            _, dest = make_captured(root)
            meta = read_json(os.path.join(dest, "session.json"))
            meta["map"].update({"map_id": "everon", "terrain": "Eden",
                                "root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"})
            write(os.path.join(dest, "session.json"), jtext(meta))

        def legacy(root):
            synth.write_legacy_session(os.path.join(root, "captures", "arland", DATE, TRIAL))

        def bare_suffix(root):
            make_captured(root, suffix="")

        def bad_record(root):
            _, dest = make_captured(root)
            write(os.path.join(dest, "_harvest", "RETENTION.json"), b'{"schema": "something else"}\n')

        cases = {"fail": (fail, "skipped", 0, "verdict FAIL"),
                 "rejected": (rejected, None, 0, None),
                 "unharvested": (unharvested, "skipped", 0, "not harvested"),
                 "uncatalogued": (uncatalogued, "skipped", 0, "not in catalog/sessions.jsonl"),
                 "checksum_mismatch": (checksum_mismatch, "held", 2, "SHA256SUMS does not verify"),
                 "missing_file": (missing_file, "held", 2, "missing ['frames/f_00000007.png']"),
                 "catalog_disagrees": (catalog_disagrees, "held", 2, "a catalog row of the session disagrees"),
                 "symlink_inside": (symlink_inside, "held", 2, "is a symlink inside the session"),
                 "symlinked_frames": (symlinked_frames, "held", 2, "frames/ is a symlink"),
                 "not_arland": (not_arland, "refused", 2, "not an Arland session"),
                 "legacy": (legacy, "refused", 2, "a dc-1.0 session; out of scope"),
                 "bare_suffix": (bare_suffix, "skipped", 0, "is not a raw screenshot suffix"),
                 "bad_record": (bad_record, "held", 2, "RETENTION.json does not name this session")}
        for name, (build, status, want_code, text) in sorted(cases.items()):
            with self.subTest(name):
                root = os.path.join(self.root, name)
                os.mkdir(root)
                build(root)
                before = th.snapshot(root)
                code, out = run(root, "--apply")
                self.assertEqual(code, want_code, out)
                self.assertEqual(th.snapshot(root), before, out)        # nothing deleted, nothing written
                self.assertNotIn("would-delete", out)
                self.assertNotIn("deleted ", out)
                if status is None:
                    self.assertEqual(out.splitlines()[0], "total: 0 session(s), 0 raw screenshot(s) deleted, 0 "
                                     "kept, freed 0.00 GiB (0 B); 0 done earlier, 0 skipped, 0 held, 0 refused")
                else:
                    self.assertTrue(out.startswith(status + " "), out)
                    self.assertIn(text, out)
        self.assertNotIn(sid_of("5e01"), run(os.path.join(self.root, "not_arland"))[1])     # the name is hidden

    def test_a_tampered_record_accounts_for_nothing(self):
        _, dest = make_captured(self.root)
        self.assertEqual(run(self.root, "--apply")[0], 0)
        path = os.path.join(dest, "_harvest", "RETENTION.json")
        doc = read_json(path)
        good = json.dumps(doc).encode("ascii")
        for name, edit in (("wrong sha256", lambda d: d["deleted"][0].update(sha256="0" * 64)),
                           ("not a raw screenshot", lambda d: d["deleted"].append(
                               {"file": "_harvest/deriv/f_00000000.jpg", "sha256": "0" * 64, "bytes": 1})),
                           ("other session", lambda d: d.update(session_id=sid_of("beef")))):
            with self.subTest(name):
                bad = json.loads(good.decode("ascii"))
                edit(bad)
                write(path, json.dumps(bad).encode("ascii"))
                ok, detail = harvest.verify_sums(dest)
                self.assertFalse(ok)
                self.assertIn("_harvest/RETENTION.json", detail)
                code, out = run(self.root, "--apply")
                self.assertEqual(code, 2, out)
                self.assertTrue(out.startswith("held "), out)
        write(path, good)
        self.assertTrue(harvest.verify_sums(dest)[0])

    def test_kill_at_each_hook_then_the_next_run_completes(self):
        for hook in [h for h in ret.HOOKS if not h.startswith("restore_")]:     # restores: Round3FindingsTest
            with self.subTest(hook):
                root = os.path.join(self.root, hook)
                os.mkdir(root)
                _, dest = make_captured(root)
                proc = run_subprocess(root, hook, "--apply")
                self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stdout + proc.stderr)
                doc = read_json(os.path.join(dest, "_harvest", "RETENTION.json"))    # written before any delete
                self.assertEqual(doc["status"], "deleting")
                left = frame_slots(dest)
                self.assertEqual((len(left), staged(dest)), {
                    "record_renamed": (250, None), "after_record": (250, None), "before_stage": (250, []),
                    "after_stage": (249, ["f_00000001.png"]), "before_unlink": (249, ["f_00000001.png"]),
                    "in_deletes": (249, []), "before_done": (4, None)}[hook])
                ok, detail = harvest.verify_sums(dest)                 # consistent at every point
                self.assertTrue(ok, detail)
                code, out = run(root)                                   # the dry run shows what is left
                if staged(dest) is None:
                    self.assertIn("completing RETENTION.json (status deleting", out)
                else:
                    self.assertIn("holds files of an interrupted retention run", out)
                code, out = run(root, "--apply")
                self.assertEqual(code, 0, out)
                self.assertEqual(staged(dest), None)
                if hook in ("before_stage", "after_stage", "before_unlink", "in_deletes"):
                    self.assertIn("recovered ", out)
                self.assertEqual(frame_slots(dest), [0, 100, 200, 249])
                done = read_json(os.path.join(dest, "_harvest", "RETENTION.json"))
                self.assertEqual((done["status"], done["deleted"], done["created_utc"]),
                                 ("done", doc["deleted"], doc["created_utc"]))
                before = th.snapshot(root)
                self.assertEqual(run(root, "--apply")[0], 0)
                self.assertEqual(th.snapshot(root), before)

    def test_under_narrows_the_scan(self):
        _, one = make_captured(self.root, tag="5e01")
        _, two = make_captured(self.root, tag="5e02")
        code, out = run(self.root, "--apply", "--under", one)
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(one), [0, 100, 200, 249])
        self.assertEqual(len(frame_slots(two)), 250)
        self.assertNotIn(self.rel(two), out)


class HarvesterAfterRetentionTest(unittest.TestCase):
    """The generator's golden Workbench session, harvested by harvest.py, then retained: the harvester's re-run does
    nothing, and an uncatalogued retained destination is catalogued (verified with the record), not rebuilt."""

    def setUp(self):
        self.root = th.temp_dir("dd6ret_h_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_harvester_rerun_after_retention(self):
        th.make_session(self.root, pin_sha=th.write_pin(self.root))
        code, out = th.run(self.root)
        self.assertEqual(code, 0, out)
        dest = th.dest_of(self.root, th.WB_ID)
        code, out = run(self.root, "--apply")
        self.assertEqual(code, 0, out)
        doc = read_json(os.path.join(dest, "_harvest", "RETENTION.json"))
        self.assertTrue(doc["deleted"] and doc["status"] == "done")
        self.assertTrue(any(w.startswith("event:teleport") for k in doc["kept"] for w in k["why"]))  # episode reset
        self.assertEqual(len(frame_slots(dest)), len(doc["kept"]))
        before = th.snapshot(self.root)
        code, out = th.run(self.root)                                  # catalogued: never rebuilt or re-validated
        self.assertEqual(code, 0, out)
        self.assertIn("nothing", out)
        self.assertEqual(th.snapshot(self.root), before)
        ok, detail = th.harvest.verify_sums(dest)
        self.assertTrue(ok, detail)
        self.assertIn("%d raw screenshot(s) deleted by retention" % len(doc["deleted"]), detail)

        os.remove(os.path.join(self.root, "catalog", "sessions.jsonl"))   # moved but not catalogued
        code, out = th.run(self.root)
        self.assertEqual(code, 0, out)
        self.assertIn("catalogued captures/arland/%s/%s/%s: moved by an earlier run" % (DATE, TRIAL, th.WB_ID), out)
        self.assertEqual(len(frame_slots(dest)), len(doc["kept"]))       # nothing rebuilt, nothing restored
        self.assertEqual(read_json(os.path.join(dest, "_harvest", "RETENTION.json")), doc)
        self.assertEqual(run(self.root, "--apply")[0], 0)                # still recognised as retained
        self.assertEqual(len(frame_slots(dest)), len(doc["kept"]))

        os.remove(os.path.join(self.root, "catalog", "sessions.jsonl"))
        tampered = dict(doc, deleted=doc["deleted"] + [{"file": "frames/f_%08d.png" % doc["kept"][0]["slot"],
                                                        "slot": 0, "sha256": "0" * 64, "bytes": 1}])
        write(os.path.join(dest, "_harvest", "RETENTION.json"), json.dumps(tampered).encode("ascii"))
        code, out = th.run(self.root)
        self.assertEqual(code, 2, out)
        self.assertIn("moved but _harvest/RETENTION.json lists", out)


class ProbeRetentionTest(unittest.TestCase):
    def setUp(self):
        self.root = th.temp_dir("dd6ret_p_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def rel(self, path):
        return os.path.relpath(path, self.root)

    def test_complete_play(self):
        pdir = make_play(self.root, frames=[n for n in range(250) if n != 100])
        before = th.snapshot(self.root)
        code, out = run(self.root, "--probes")
        self.assertEqual(code, 0, out)
        self.assertEqual(th.snapshot(self.root), before)
        self.assertIn("would-delete %s: delete 246 raw screenshot(s), keep 3 (every_100 2, first 1, last 1)" % (
            self.rel(pdir)), out)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(pdir), [0, 200, 249])              # 100 was never there
        self.assertEqual(sorted(os.listdir(pdir)), ["RETENTION.json", "e05_frames.csv", "frames", "run_footer.txt",
                                                    "run_header.txt"])
        doc = read_json(os.path.join(pdir, "RETENTION.json"))
        self.assertEqual((doc["kind"], doc["path"], doc["probe"], doc["launch_id"], doc["play"], doc["status"],
                          doc["every_n"], doc["footer_status"]),
                         ("probe_play", self.rel(pdir), "e05", LAUNCH, "play_01", "done", 100, "complete"))
        self.assertEqual(len(doc["deleted"]), 246)
        for e in doc["deleted"]:
            self.assertEqual(e["sha256"], hashlib.sha256(tiny_png(e["slot"])).hexdigest())
        self.assertEqual(doc["bytes_freed"], sum(e["bytes"] for e in doc["deleted"]))
        after = th.snapshot(self.root)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 0, out)
        self.assertIn("done %s: retained earlier" % self.rel(pdir), out)
        self.assertEqual(th.snapshot(self.root), after)

    def test_incomplete_new_or_foreign_plays_lose_nothing(self):
        incomplete = make_play(self.root, launch="20261024T000000Z-wb-0001", footer=None)
        new = make_play(self.root, launch="20261024T000000Z-wb-0002", age_days=0.2)
        everon = make_play(self.root, launch="20261024T000000Z-wb-0003", map_id="everon")
        before = th.snapshot(self.root)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 2, out)                                  # the non-Arland play is refused
        self.assertEqual(th.snapshot(self.root), before)
        self.assertIn("skipped %s: incomplete play: no run_footer.txt" % self.rel(incomplete), out)
        self.assertIn("skipped %s: newer than --min-age-days 1" % self.rel(new), out)
        self.assertIn("refused %s: run_header.txt does not name world.map_id=arland" % self.rel(everon), out)
        code, out = run(self.root, "--probes", "--min-age-days", "0.1")  # old enough for a smaller threshold
        self.assertIn("would-delete %s" % self.rel(new), out)
        self.assertEqual(th.snapshot(self.root), before)

    def test_only_a_complete_footer_lets_frames_go(self):
        """A play whose footer says stopped, failed or has no status is held with every frame kept (evidence of a
        problem), even when it is old; a stopped play without frames has nothing to keep and is only skipped."""
        complete = make_play(self.root, launch="20261024T000000Z-wb-0010")
        plays = {"stopped": make_play(self.root, launch="20261024T000000Z-wb-0011", footer="stopped",
                                      reason="play_end"),
                 "failed": make_play(self.root, launch="20261024T000000Z-wb-0012", footer="failed",
                                     reason="script_error"),
                 "<none>": make_play(self.root, launch="20261024T000000Z-wb-0013", footer="")}
        empty = make_play(self.root, launch="20261024T000000Z-wb-0014", footer="failed", frames=())
        before = dict((k, th.snapshot(p)) for k, p in plays.items())
        code, out = run(self.root, "--probes")                          # the dry run says why
        self.assertEqual(code, 2, out)
        for status, pdir in plays.items():
            line = line_of(out, "held", self.rel(pdir))
            self.assertIsNotNone(line, out)
            self.assertIn("run_footer.txt status=%s (reason=" % status, line)
            self.assertIn("not complete: a play that did not complete is evidence of a problem, so every frame is "
                          "kept", line)
        self.assertIn("held %s: run_footer.txt status=stopped (reason=play_end), not complete" % self.rel(
            plays["stopped"]), out)
        self.assertIn("would-delete %s" % self.rel(complete), out)
        self.assertIn("skipped %s: no frames/" % self.rel(empty), out)
        self.assertIn("total: 1 probe play(s), 246 raw screenshot(s) to delete", out)
        self.assertIn("3 held", out)
        code, out = run(self.root, "--probes", "--apply", "--min-age-days", "0")
        self.assertEqual(code, 2, out)
        self.assertEqual(frame_slots(complete), [0, 100, 200, 249])
        for status, pdir in plays.items():
            self.assertEqual(th.snapshot(pdir), before[status], status)     # every frame kept, nothing written
            self.assertEqual(len(frame_slots(pdir)), 250)
            self.assertFalse(os.path.exists(os.path.join(pdir, "RETENTION.json")))

    def test_symlinks_and_other_entries_are_never_followed_or_deleted(self):
        pdir = make_play(self.root, frames=range(20))
        target = os.path.join(self.root, "keep_me.png")
        write(target, tiny_png(5))
        frames = os.path.join(pdir, "frames")
        os.remove(os.path.join(frames, "f_00000005.png"))
        os.symlink(target, os.path.join(frames, "f_00000005.png"))     # a symlink named like a screenshot
        write(os.path.join(frames, "notes.txt"), b"not a screenshot\n")
        write(os.path.join(frames, "f_00000006.jpg"), b"not a raw screenshot\n")
        os.mkdir(os.path.join(frames, "f_00000007.png.d"))
        age(pdir, 3)
        code, out = run(self.root, "--probes", "--apply", "--every-n", "1000")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(os.listdir(frames)), ["f_00000000.png", "f_00000005.png", "f_00000006.jpg",
                                                      "f_00000007.png.d", "f_00000019.png", "notes.txt"])
        self.assertTrue(os.path.islink(os.path.join(frames, "f_00000005.png")))
        self.assertTrue(os.path.isfile(target))
        self.assertIn("4 other entries in frames/ left alone", out)

        other = make_play(self.root, launch="20261024T000000Z-wb-0004", frames=range(20))
        outside = os.path.join(self.root, "outside_frames")
        os.rename(os.path.join(other, "frames"), outside)
        os.symlink(outside, os.path.join(other, "frames"))
        before = th.snapshot(self.root)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("held %s: frames/ is a symlink" % self.rel(other), out)
        self.assertEqual(th.snapshot(self.root), before)

    def test_kill_then_complete_and_recorded_hashes_are_checked(self):
        pdir = make_play(self.root)
        proc = run_subprocess(self.root, "in_deletes", "--probes", "--apply")
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stdout + proc.stderr)
        self.assertEqual(read_json(os.path.join(pdir, "RETENTION.json"))["status"], "deleting")
        self.assertEqual(len(frame_slots(pdir)), 249)
        changed = os.path.join(pdir, "frames", "f_00000002.png")      # a file the record lists, changed since
        with open(changed, "ab") as f:
            f.write(b"x")
        age(pdir, 3)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("frames/f_00000002.png does not match its sha256 (recorded in RETENTION.json)", out)
        self.assertEqual(frame_slots(pdir), [0] + list(range(2, 250)))  # the play stops at the first mismatch
        self.assertEqual(read_json(os.path.join(pdir, "RETENTION.json"))["status"], "deleting")
        write(changed, tiny_png(2))
        age(pdir, 3)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(pdir), [0, 100, 200, 249])
        self.assertEqual(read_json(os.path.join(pdir, "RETENTION.json"))["status"], "done")

    def test_client_profile_plays_and_sessions_mode_ignores_probes(self):
        pdir = make_play(self.root, profile="client_profiles/i01")
        before = th.snapshot(self.root)
        code, out = run(self.root, "--apply")                           # sessions mode: no probe is touched
        self.assertEqual((code, th.snapshot(self.root)), (0, before), out)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(pdir), [0, 100, 200, 249])


class ReviewFindingsTest(unittest.TestCase):
    """Regression tests for the 2026-11-04 review of 6c8a660..af46db8 (retention/review/CODEX.md, findings 1-7):
    a record is never a licence to delete, files are re-checked right before their unlink, deletes go through
    pinned directory descriptors, the record is durable before any delete, repeated metadata keys hold a play, a
    catalog that appears mid-run is not used without the lock, and screenshot names match whole."""

    def setUp(self):
        self.root = th.temp_dir("dd6ret_v_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def sub(self, name):
        root = os.path.join(self.root, name)
        os.mkdir(root)
        return root

    def killed_after_record(self, root, *argv):
        proc = run_subprocess(root, "after_record", "--apply", *argv)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stdout + proc.stderr)

    # -- finding 1 ------------------------------------------------------------------------------------
    def test_f1_a_record_never_licenses_deleting_a_protected_frame(self):
        """A protected frame appended to the record's deleted list with its correct SHA256SUMS hash holds the
        session; nothing is deleted."""
        for name, slot in sorted({"first": 0, "every_n": 100, "event": 120, "no_derivative": 144,
                                  "last": 249}.items()):
            with self.subTest(name):
                root = self.sub(name)
                _, dest = make_captured(root, events=[("collision", slot_ms(120))], no_deriv=(144,))
                self.killed_after_record(root)
                path = os.path.join(dest, "_harvest", "RETENTION.json")
                doc = read_json(path)
                with open(os.path.join(dest, "_harvest", "SHA256SUMS"), "rb") as f:
                    listed = harvest.parse_sums(f.read())
                f_rel = "frames/f_%08d.png" % slot
                doc["deleted"].append({"file": f_rel, "slot": slot, "sha256": listed[f_rel],
                                       "bytes": len(tiny_png(slot))})
                write(path, json.dumps(doc).encode("ascii"))
                self.assertFalse(harvest.verify_sums(dest)[0])
                code, out = run(root, "--apply")
                self.assertEqual(code, 2, out)
                self.assertTrue(out.startswith("held "), out)
                self.assertIn("SHA256SUMS does not verify: _harvest/RETENTION.json lists '%s' as deleted, which the "
                              "retention rule keeps" % f_rel, out)
                self.assertEqual(len(frame_slots(dest)), 250)

    def test_f1_a_missing_protected_frame_is_not_hidden_by_the_record(self):
        _, dest = make_captured(self.root)
        self.assertEqual(run(self.root, "--apply")[0], 0)
        path = os.path.join(dest, "_harvest", "RETENTION.json")
        doc = read_json(path)
        with open(os.path.join(dest, "_harvest", "SHA256SUMS"), "rb") as f:
            listed = harvest.parse_sums(f.read())
        os.remove(os.path.join(dest, "frames", "f_00000100.png"))
        doc["deleted"].append({"file": "frames/f_00000100.png", "slot": 100, "sha256": listed["frames/f_00000100.png"],
                               "bytes": len(tiny_png(100))})
        write(path, json.dumps(doc).encode("ascii"))
        ok, detail = harvest.verify_sums(dest)
        self.assertFalse(ok)
        self.assertIn("lists 'frames/f_00000100.png' as deleted, which the retention rule keeps", detail)
        code, out = run(self.root)
        self.assertEqual(code, 2, out)
        self.assertTrue(out.startswith("held "), out)

    def test_f1_a_record_of_another_every_n_needs_that_every_n(self):
        """A record rewritten consistently for --every-n 1000 does not widen a default run's deletes."""
        _, dest = make_captured(self.root)
        self.killed_after_record(self.root)
        path = os.path.join(dest, "_harvest", "RETENTION.json")
        doc = read_json(path)
        with open(os.path.join(dest, "_harvest", "SHA256SUMS"), "rb") as f:
            listed = harvest.parse_sums(f.read())
        moved = [k for k in doc["kept"] if k["slot"] in (100, 200)]
        doc["kept"] = [k for k in doc["kept"] if k["slot"] not in (100, 200)]
        doc["deleted"] += [{"file": k["file"], "slot": k["slot"], "sha256": listed[k["file"]],
                            "bytes": len(tiny_png(k["slot"]))} for k in moved]
        doc["every_n"] = 1000
        write(path, json.dumps(doc).encode("ascii"))
        code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("--every-n 1000", out)
        self.assertEqual(len(frame_slots(dest)), 250)
        code, out = run(self.root, "--apply", "--every-n", "1000")         # the operator's explicit choice
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(dest), [0, 249])

    def test_f1_a_probe_record_cannot_add_or_hide_a_kept_frame(self):
        pdir = make_play(self.root)
        self.killed_after_record(self.root, "--probes")
        path = os.path.join(pdir, "RETENTION.json")
        doc = read_json(path)
        added = json.loads(json.dumps(doc))
        added["deleted"].append({"file": "frames/f_00000000.png", "slot": 0,
                                 "sha256": hashlib.sha256(tiny_png(0)).hexdigest(), "bytes": len(tiny_png(0))})
        write(path, json.dumps(added).encode("ascii"))
        age(pdir, 3)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual(code, 2, out)
        self.assertTrue(out.startswith("held "), out)
        self.assertIn("'frames/f_00000000.png' is not a raw frames/ screenshot or is named twice", out)
        self.assertEqual(len(frame_slots(pdir)), 250)
        write(path, json.dumps(doc).encode("ascii"))                    # the genuine record completes
        age(pdir, 3)
        self.assertEqual(run(self.root, "--probes", "--apply")[0], 0)
        self.assertEqual(frame_slots(pdir), [0, 100, 200, 249])
        with open(path, "rb") as f:
            genuine = f.read()
        os.remove(os.path.join(pdir, "frames", "f_00000100.png"))       # a kept frame lost after retention
        age(pdir, 3)
        code, out = run(self.root, "--probes")                          # the genuine record: reported missing
        self.assertEqual(code, 2, out)
        self.assertTrue(out.startswith("held "), out)
        self.assertIn("kept frame frames/f_00000100.png of RETENTION.json is missing", out)
        done = json.loads(genuine.decode("ascii"))                      # relabelling it as deleted does not help
        done["deleted"].append({"file": "frames/f_00000100.png", "slot": 100,
                                "sha256": hashlib.sha256(tiny_png(100)).hexdigest(), "bytes": len(tiny_png(100))})
        done["kept"] = [k for k in done["kept"] if k["slot"] != 100]
        write(path, json.dumps(done).encode("ascii"))
        age(pdir, 3)
        code, out = run(self.root, "--probes")
        self.assertEqual(code, 2, out)
        self.assertTrue(out.startswith("held "), out)
        self.assertIn("are not the audit sample of every_n 100 and the rest", out)

    # -- finding 2 ------------------------------------------------------------------------------------
    def test_f2_a_file_changed_after_planning_is_not_deleted(self):
        for mode in ("session", "probe"):
            with self.subTest(mode):
                root = self.sub(mode)
                if mode == "session":
                    _, unit = make_captured(root)
                    argv = ["--apply"]
                else:
                    unit = make_play(root)
                    argv = ["--probes", "--apply"]
                target = os.path.join(unit, "frames", "f_00000001.png")
                original = ret.Retention.apply

                def apply(r, plan):
                    write(target, b"changed after planning, before the delete")
                    return original(r, plan)
                with mock.patch.object(ret.Retention, "apply", apply):
                    code, out = run(root, *argv)
                self.assertEqual(code, 2, out)
                self.assertIn("frames/f_00000001.png changed after it was planned", out)
                self.assertTrue(os.path.isfile(target), out)
                with open(target, "rb") as f:
                    self.assertEqual(f.read(), b"changed after planning, before the delete")
                self.assertEqual(len(frame_slots(unit)), 250)

    def test_f2_a_file_swapped_between_its_last_check_and_its_move_is_not_deleted(self):
        """The file is replaced (same bytes, another inode) between the last check in frames/ and the rename into
        staging: the staged file is not the one planned, so it is put back and the session is held."""
        _, dest = make_captured(self.root)
        target = os.path.join(dest, "frames", "f_00000001.png")
        swapped = []

        def hook(r, name):
            if name == "before_stage" and not swapped:
                os.rename(target, os.path.join(dest, "f_00000001.png.old"))
                write(target, tiny_png(1))
                swapped.append(name)
        with mock.patch.object(ret.Retention, "hook", hook):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertTrue(os.path.isfile(target), out)
        self.assertTrue(os.path.isfile(os.path.join(dest, "f_00000001.png.old")))
        self.assertIn("frames/f_00000001.png is not the file planned once staged", out)
        self.assertIn("it was put back in frames/", out)
        self.assertEqual(staged(dest), None)

    # -- finding 3 ------------------------------------------------------------------------------------
    def test_f3_an_ancestor_replaced_after_planning_is_not_followed(self):
        for mode, how in (("session", "symlink"), ("session", "copy"), ("probe", "symlink"), ("probe", "copy")):
            with self.subTest(mode=mode, how=how):
                root = self.sub("%s_%s" % (mode, how))
                if mode == "session":
                    _, unit = make_captured(root)
                    ancestor = os.path.dirname(unit)                    # the trial directory
                    argv = ["--apply"]
                else:
                    unit = make_play(root)
                    ancestor = os.path.dirname(unit)                    # the launch directory
                    argv = ["--probes", "--apply"]
                elsewhere = os.path.join(root, "elsewhere")
                moved = os.path.join(root, "moved_away")
                original = ret.Retention.apply

                def apply(r, plan):
                    shutil.copytree(ancestor, elsewhere, symlinks=True)
                    os.rename(ancestor, moved)
                    if how == "symlink":
                        os.symlink(elsewhere, ancestor)
                    else:
                        os.rename(elsewhere, ancestor)
                    return original(r, plan)
                with mock.patch.object(ret.Retention, "apply", apply):
                    code, out = run(root, *argv)
                self.assertEqual(code, 2, out)
                self.assertTrue(out.startswith("held "), out)
                self.assertIn("cannot be opened as a directory without following a symlink" if how == "symlink"
                              else "the directory was replaced after it was planned", out)
                copy = os.path.join(elsewhere if how == "symlink" else ancestor, os.path.basename(unit))
                self.assertEqual(len(frame_slots(copy)), 250)
                self.assertEqual(len(frame_slots(os.path.join(moved, os.path.basename(unit)))), 250)
                for d in (copy, os.path.join(moved, os.path.basename(unit))):
                    for sub in ("", "_harvest"):
                        self.assertFalse(os.path.exists(os.path.join(d, sub, "RETENTION.json")), d)

    # -- finding 4 ------------------------------------------------------------------------------------
    def test_f4_a_record_that_cannot_be_made_durable_stops_every_delete(self):
        real_fsync = os.fsync
        for mode in ("session", "probe"):
            for what in ("directory", "file"):
                with self.subTest(mode=mode, what=what):
                    root = self.sub("%s_%s" % (mode, what))
                    if mode == "session":
                        _, unit = make_captured(root)
                        argv = ["--apply"]
                    else:
                        unit = make_play(root)
                        argv = ["--probes", "--apply"]

                    def fsync(fd):
                        is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
                        if is_dir == (what == "directory"):
                            raise OSError(errno.EIO, "injected fsync failure")
                        return real_fsync(fd)
                    with mock.patch("os.fsync", fsync):
                        code, out = run(root, *argv)
                    self.assertEqual(code, 2, out)
                    self.assertIn("could not be written durably", out)
                    self.assertEqual(len(frame_slots(unit)), 250)

    # -- finding 5 ------------------------------------------------------------------------------------
    def test_f5_repeated_metadata_keys_hold_the_play(self):
        cases = {"footer status complete then failed": ("run_footer.txt", "status=failed\n", "held"),
                 "footer repeats a key with the same value": ("run_footer.txt", "status=complete\n", "held"),
                 "header map arland then everon": ("run_header.txt", "world.map_id=everon\n", "held"),
                 "header repeats world.map_id": ("run_header.txt", "world.map_id=arland\n", "held"),
                 "header repeats another key unchanged": ("run_header.txt", "drive.force_sim=0\n", "would-delete")}
        plays = {}
        for n, (name, (fname, line, _)) in enumerate(sorted(cases.items())):
            pdir = make_play(self.root, launch="20261024T000000Z-wb-%04d" % (20 + n))
            if name.endswith("unchanged"):
                with open(os.path.join(pdir, fname), "ab") as f:
                    f.write(b"drive.force_sim=0\n")
            with open(os.path.join(pdir, fname), "ab") as f:
                f.write(line.encode("ascii"))
            age(pdir, 3)
            plays[name] = pdir
        before = th.snapshot(self.root)
        code, out = run(self.root, "--probes")
        self.assertEqual(code, 2, out)
        for name, (fname, _, status) in sorted(cases.items()):
            with self.subTest(name):
                line = line_of(out, status, os.path.relpath(plays[name], self.root))
                self.assertIsNotNone(line, out)
                if status == "held":
                    self.assertIn("%s repeats" % fname, line)
        code, out = run(self.root, "--probes", "--apply")
        for name, (fname, _, status) in sorted(cases.items()):
            self.assertEqual(len(frame_slots(plays[name])), 250 if status == "held" else 4, name)
        self.assertNotEqual(th.snapshot(self.root), before)

    # -- finding 6 ------------------------------------------------------------------------------------
    def test_f6_a_catalog_that_appears_after_the_check_is_not_used_without_the_lock(self):
        _, dest = make_captured(self.root, catalogued=False)
        original = ret.Retention.plan_all
        made = []

        def plan_all(r, *a, **k):
            if not made:
                catalogue(self.root, dest)                              # a harvester run catalogues it now
                made.append(1)
            return original(r, *a, **k)
        with mock.patch.object(ret.Retention, "plan_all", plan_all):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 0, out)
        self.assertIn("not in catalog/sessions.jsonl", out)
        self.assertEqual(len(frame_slots(dest)), 250)
        code, out = run(self.root, "--apply")                           # the next run reads it under the lock
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(dest), [0, 100, 200, 249])

    # -- finding 7 ------------------------------------------------------------------------------------
    def test_f7_screenshot_names_match_whole(self):
        self.assertEqual(harvest.raw_shot_slot("f_00000005.png"), 5)
        self.assertEqual(harvest.raw_shot_slot("f_00000005.bmp.png", ".bmp.png"), 5)
        for bad in ("f_00000005.png\n", "f_00000005.png\r", "f_0000005\x01.png", "f_00000005.PNG", "f_00000005.png.x",
                    "f_00000005.jpg", "xf_00000005.png", "f_00000005.bmp", ):
            want = 5 if bad == "f_00000005.bmp" else None
            self.assertEqual(harvest.raw_shot_slot(bad), want, repr(bad))
        self.assertIsNone(harvest.raw_shot_slot("f_00000005.bmp", ".png"))
        pdir = make_play(self.root, frames=range(20))
        for name in ("f_00000005.png\n", "f_00000006.png\r"):
            write(os.path.join(pdir, "frames", name), tiny_png(5))
        age(pdir, 3)
        code, out = run(self.root, "--probes", "--apply", "--every-n", "1000")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(os.listdir(os.path.join(pdir, "frames"))),
                         sorted(["f_00000000.png", "f_00000019.png", "f_00000005.png\n", "f_00000006.png\r"]))
        self.assertIn("2 other entries in frames/ left alone", out)


class ReReviewFindingsTest(unittest.TestCase):
    """Regression tests for the second review of 6c8a660..2c21d59 (retention/review/CODEX2.md), as triaged: the
    record validated is the record used (1), the record is made durable on every run before any delete (2), the
    unit's location is checked right before its first mutation (3), deletes go through a private staging directory
    with restore and crash recovery (4), eligibility inputs are re-checked before the first mutation (5), and
    oversized metadata holds the play (6)."""

    def setUp(self):
        self.root = th.temp_dir("dd6ret_w_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def sub(self, name):
        root = os.path.join(self.root, name)
        os.mkdir(root)
        return root

    def kill(self, root, hook, *argv):
        proc = run_subprocess(root, hook, "--apply", *argv)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stdout + proc.stderr)

    @staticmethod
    def failing_dir_fsync():
        real = os.fsync

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "injected directory fsync failure")
            return real(fd)
        return mock.patch("os.fsync", fsync)

    # -- 1: the record validated is the record used ---------------------------------------------------
    def test_r1_a_record_replaced_during_verification_holds_the_session(self):
        _, dest = make_captured(self.root)
        self.kill(self.root, "after_record")
        path = os.path.join(dest, "_harvest", "RETENTION.json")
        tampered = read_json(path)
        with open(os.path.join(dest, "_harvest", "SHA256SUMS"), "rb") as f:
            listed = harvest.parse_sums(f.read())
        tampered["deleted"].append({"file": "frames/f_00000000.png", "slot": 0,
                                    "sha256": listed["frames/f_00000000.png"], "bytes": len(tiny_png(0))})
        original = harvest.sha256_file
        swapped = []

        def sha256_file(full):
            if not swapped:                                         # while verification hashes the session
                write(path, json.dumps(tampered).encode("ascii"))
                swapped.append(full)
            return original(full)
        with mock.patch.object(harvest, "sha256_file", sha256_file):
            code, out = run(self.root, "--apply")
        self.assertTrue(swapped)
        self.assertEqual(code, 2, out)
        self.assertIn("RETENTION.json changed while the session was verified", out)
        self.assertEqual(len(frame_slots(dest)), 250)

    # -- 2: the record is durable before any delete, on every run -------------------------------------
    def test_r2_a_run_after_a_failed_directory_fsync_deletes_nothing_until_the_record_is_durable(self):
        for mode in ("session", "probe"):
            with self.subTest(mode):
                root = self.sub(mode)
                if mode == "session":
                    _, unit = make_captured(root)
                    argv = ["--apply"]
                else:
                    unit = make_play(root)
                    argv = ["--probes", "--apply"]
                for attempt in (1, 2):                              # the second run resumes the renamed record
                    with self.failing_dir_fsync():
                        code, out = run(root, *argv)
                    self.assertEqual(code, 2, (attempt, out))
                    self.assertIn("could not be written durably", out)
                    self.assertEqual(len(frame_slots(unit)), 250)
                    record = os.path.join(unit, "_harvest" if mode == "session" else "", "RETENTION.json")
                    self.assertTrue(os.path.isfile(record))
                    if mode == "probe":
                        age(unit, 3)
                code, out = run(root, *argv)
                self.assertEqual(code, 0, out)
                self.assertEqual(frame_slots(unit), [0, 100, 200, 249])

    def test_r2_a_kill_between_the_rename_and_the_directory_fsync(self):
        _, dest = make_captured(self.root)
        self.kill(self.root, "record_renamed")
        self.assertEqual(read_json(os.path.join(dest, "_harvest", "RETENTION.json"))["status"], "deleting")
        with self.failing_dir_fsync():                              # not durable yet: nothing may go
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertEqual(len(frame_slots(dest)), 250)
        events = []
        real_fsync, real_unlink = os.fsync, os.unlink

        def fsync(fd):
            events.append(("fsync", stat.S_ISDIR(os.fstat(fd).st_mode)))
            return real_fsync(fd)

        def unlink(name, *a, **k):
            events.append(("unlink", name))
            return real_unlink(name, *a, **k)
        with mock.patch("os.fsync", fsync), mock.patch("os.unlink", unlink):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(dest), [0, 100, 200, 249])
        first_delete = events.index(next(e for e in events if e[0] == "unlink" and str(e[1]).startswith("f_")))
        self.assertIn(("fsync", True), events[:first_delete])        # the record's directory, before any delete

    # -- 3: the location is checked right before the first mutation -----------------------------------
    def test_r3_a_unit_moved_after_its_descriptors_were_opened_is_held(self):
        """Where a directory with open descriptors can be renamed (tmpfs, ext4), the move is caught right before the
        first mutation. On drvfs (E:) the rename itself is refused (EACCES) while retention holds the descriptors,
        so nothing moves and the run completes in place."""
        _, dest = make_captured(self.root)
        moved = dest + "_moved"
        original = ret.Retention.recheck_inputs
        outcome = []

        def recheck_inputs(r, plan, *fds):
            result = original(r, plan, *fds)
            try:
                os.rename(dest, moved)                              # after the open, before the first mutation
                outcome.append("moved")
            except PermissionError:
                outcome.append("refused")
            return result
        with mock.patch.object(ret.Retention, "recheck_inputs", recheck_inputs):
            code, out = run(self.root, "--apply")
        if outcome == ["moved"]:
            self.assertEqual(code, 2, out)
            self.assertIn("was moved or replaced after it was planned", out)
            self.assertEqual(len(frame_slots(moved)), 250)
            self.assertFalse(os.path.exists(os.path.join(moved, "_harvest", "RETENTION.json")))
        else:
            self.assertEqual(outcome, ["refused"])
            self.assertEqual(code, 0, out)
            self.assertFalse(os.path.exists(moved))
            self.assertEqual(frame_slots(dest), [0, 100, 200, 249])

    # -- 4: staging, restore and crash recovery ---------------------------------------------------------
    def test_r4_a_staged_file_that_fails_its_check_is_put_back(self):
        _, dest = make_captured(self.root)
        staged_file = os.path.join(dest, "frames", ret.STAGING, "f_00000001.png")
        changed = []

        def hook(r, name):
            if name == "after_stage" and not changed:
                with open(staged_file, "ab") as f:                  # the staged file is not the planned one
                    f.write(b"x")
                changed.append(name)
        with mock.patch.object(ret.Retention, "hook", hook):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("it was put back in frames/", out)
        with open(os.path.join(dest, "frames", "f_00000001.png"), "rb") as f:
            self.assertEqual(f.read(), tiny_png(1) + b"x")
        self.assertEqual((len(frame_slots(dest)), staged(dest)), (250, None))

    def test_r4_a_kill_with_a_file_in_staging_is_restored_then_completed(self):
        for mode in ("session", "probe"):
            with self.subTest(mode):
                root = self.sub(mode)
                if mode == "session":
                    _, unit = make_captured(root)
                    argv = []
                else:
                    unit = make_play(root)
                    argv = ["--probes"]
                self.kill(root, "after_stage", *argv)
                self.assertEqual(staged(unit), ["f_00000001.png"])
                if mode == "session":
                    self.assertTrue(harvest.verify_sums(unit)[0])     # a staged file still counts
                before = th.snapshot(root)
                code, out = run(root, *argv)                        # a dry run holds it and writes nothing
                self.assertEqual(code, 2, out)
                self.assertIn("holds files of an interrupted retention run", out)
                self.assertEqual(th.snapshot(root), before)
                code, out = run(root, "--apply", *argv)
                self.assertEqual(code, 0, out)
                self.assertIn("recovered %s: restored 1 raw screenshot(s)" % os.path.relpath(unit, root), out)
                self.assertEqual((frame_slots(unit), staged(unit)), ([0, 100, 200, 249], None))

    def test_r4_an_unverifiable_staged_file_is_restored_not_deleted(self):
        _, dest = make_captured(self.root)
        self.kill(self.root, "after_stage")
        with open(os.path.join(dest, "frames", ret.STAGING, "f_00000001.png"), "ab") as f:
            f.write(b"changed while staged")
        code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("recovered ", out)
        self.assertIn("SHA256SUMS does not verify", out)                # restored as it is, then held
        with open(os.path.join(dest, "frames", "f_00000001.png"), "rb") as f:
            self.assertEqual(f.read(), tiny_png(1) + b"changed while staged")
        self.assertEqual((len(frame_slots(dest)), staged(dest)), (250, None))

    def test_r4_a_staged_file_whose_name_is_back_in_frames_is_left_alone(self):
        _, dest = make_captured(self.root)
        self.kill(self.root, "after_stage")
        write(os.path.join(dest, "frames", "f_00000001.png"), b"a new file under the same name")
        before = th.snapshot(self.root)
        code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("is both in frames/ and in frames/%s/ as different files; nothing touched" % ret.STAGING,
                      out)
        self.assertEqual(th.snapshot(self.root), before)

    # -- 5: eligibility inputs re-checked before the first mutation -------------------------------------
    def test_r5_eligibility_that_changes_after_planning_holds_the_unit(self):
        def footer_failed(root, unit):
            write(os.path.join(unit, "run_footer.txt"), b"schema=dc-probe-run/1\nstatus=failed\nreason=crash\n")

        def derivative_removed(root, unit):
            os.remove(os.path.join(unit, "_harvest", "deriv", "f_00000001.jpg"))

        def derivative_changed(root, unit):
            with open(os.path.join(unit, "_harvest", "deriv", "f_00000003.jpg"), "ab") as f:
                f.write(b"x")

        def events_changed(root, unit):
            with open(os.path.join(unit, "events", "chunk_00000.csv"), "ab") as f:
                f.write(b"9,0,0,0,0,collision,0,\n")

        def kept_removed(root, unit):
            os.remove(os.path.join(unit, "frames", "f_00000100.png"))

        def catalog_row_added(root, unit):                          # a second row (the run holds the lock)
            h = harvest.Harvester(root, harvest.parse_args([]), True)
            h.catalog_append(h.catalog_row(unit))

        cases = {"footer flips to failed": ("probe", footer_failed, "run_footer.txt changed after the unit"),
                 "derivative removed": ("session", derivative_removed, "_harvest/deriv/f_00000001.jpg is gone"),
                 "derivative changed": ("session", derivative_changed, "_harvest/deriv/f_00000003.jpg is gone or "
                                        "changed"),
                 "event data changed": ("session", events_changed, "events/chunk_00000.csv is gone or changed"),
                 "kept frame removed": ("session", kept_removed, "kept file frames/f_00000100.png is gone"),
                 "catalog row added": ("session", catalog_row_added, "catalog rows changed after it was planned")}
        for n, (name, (mode, change, text)) in enumerate(sorted(cases.items())):
            with self.subTest(name):
                root = self.sub("case%d" % n)
                if mode == "session":
                    _, unit = make_captured(root)
                    argv = ["--apply"]
                else:
                    unit = make_play(root)
                    argv = ["--probes", "--apply"]
                original = ret.Retention.apply

                def apply(r, plan):
                    change(root, unit)
                    return original(r, plan)
                with mock.patch.object(ret.Retention, "apply", apply):
                    code, out = run(root, *argv)
                self.assertEqual(code, 2, out)
                self.assertIn(text, out)
                self.assertGreaterEqual(len(frame_slots(unit)), 249, out)
                self.assertNotIn("deleted ", out)
                for sub in ("", "_harvest"):
                    self.assertFalse(os.path.exists(os.path.join(unit, sub, "RETENTION.json")), out)

    # -- 6: oversized metadata --------------------------------------------------------------------------
    def test_r6_metadata_beyond_the_read_limit_holds_the_play(self):
        limit = ret.KV_LIMIT
        head = b"schema=dc-probe-run/1\nstatus=complete\nreason=done\n"
        cases = {"footer duplicate beyond the limit": ("run_footer.txt", head + b"pad=" + b"x" * limit
                                                       + b"\nstatus=failed\n", "held"),
                 "header map beyond the limit": ("run_header.txt", None, "held"),
                 "footer exactly at the limit": ("run_footer.txt", None, "would-delete"),
                 "footer one byte over the limit": ("run_footer.txt", None, "held")}
        plays = {}
        for n, (name, (fname, data, _)) in enumerate(sorted(cases.items())):
            pdir = make_play(self.root, launch="20261024T000000Z-wb-%04d" % (40 + n))
            path = os.path.join(pdir, fname)
            with open(path, "rb") as f:
                current = f.read()
            if name == "header map beyond the limit":
                data = current + b"pad=" + b"x" * limit + b"\nworld.map_id=everon\n"
            elif name.startswith("footer exactly") or name.startswith("footer one byte"):
                fill = limit - len(head) - len(b"pad=\n") + (1 if "over" in name else 0)
                data = head + b"pad=" + b"x" * fill + b"\n"
                self.assertEqual(len(data), limit + (1 if "over" in name else 0))
            write(path, data)
            age(pdir, 3)
            plays[name] = (pdir, fname)
        code, out = run(self.root, "--probes")
        self.assertEqual(code, 2, out)
        for name, (fname, _, status) in sorted(cases.items()):
            with self.subTest(name):
                line = line_of(out, status, os.path.relpath(plays[name][0], self.root))
                self.assertIsNotNone(line, out)
                if status == "held":
                    self.assertIn("%s is larger than %d bytes" % (fname, limit), line)


class Round3FindingsTest(unittest.TestCase):
    """Regression tests for the third review (retention/review/CODEX3.md), as triaged: a restore keeps its staging
    copy until the frames/ name is durable (fix 1), an interrupted restore (one inode under both names) is completed
    (fix 5), the pre-mutation recheck covers session admission, every kept session file and the play's age (fix 3),
    and --apply refuses to start while a unit in scope was written in the last 10 minutes (the preflight of the
    operational constraint)."""

    def setUp(self):
        self.root = th.temp_dir("dd6ret_x_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def sub(self, name):
        root = os.path.join(self.root, name)
        os.mkdir(root)
        return root

    def kill(self, root, hook, *argv):
        proc = run_subprocess(root, hook, "--apply", *argv)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stdout + proc.stderr)

    def staged_session(self, root):
        """A session killed with frames/f_00000001.png in staging."""
        _, dest = make_captured(root)
        self.kill(root, "after_stage")
        self.assertEqual(staged(dest), ["f_00000001.png"])
        return dest

    @staticmethod
    def frames_fsync_fails(dest):
        """os.fsync raising for the session's frames/ directory only."""
        st = os.lstat(os.path.join(dest, "frames"))
        real = os.fsync

        def fsync(fd):
            got = os.fstat(fd)
            if (got.st_dev, got.st_ino) == (st.st_dev, st.st_ino):
                raise OSError(errno.EIO, "injected frames/ fsync failure")
            return real(fd)
        return mock.patch("os.fsync", fsync)

    def same_inode(self, dest, name="f_00000001.png"):
        a = os.lstat(os.path.join(dest, "frames", name))
        b = os.lstat(os.path.join(dest, "frames", ret.STAGING, name))
        return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)

    # -- fix 1: restore ordering ----------------------------------------------------------------------
    def test_fix1_a_restore_fsyncs_frames_before_it_removes_the_staging_copy(self):
        dest = self.staged_session(self.root)
        frames_id = os.lstat(os.path.join(dest, "frames"))
        events = []
        real = {"link": os.link, "fsync": os.fsync, "unlink": os.unlink}

        def link(src, dst, *a, **k):
            events.append(("link", src))
            return real["link"](src, dst, *a, **k)

        def fsync(fd):
            st = os.fstat(fd)
            events.append(("fsync", "frames" if (st.st_dev, st.st_ino) == (frames_id.st_dev, frames_id.st_ino)
                           else "other"))
            return real["fsync"](fd)

        def unlink(name, *a, **k):
            events.append(("unlink", name))
            return real["unlink"](name, *a, **k)
        with mock.patch("os.link", link), mock.patch("os.fsync", fsync), mock.patch("os.unlink", unlink):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 0, out)
        i_link = events.index(("link", "f_00000001.png"))
        i_alias = events.index(("unlink", "f_00000001.png"), i_link)
        self.assertIn(("fsync", "frames"), events[i_link:i_alias])          # frames/ durable before the alias goes
        self.assertEqual((frame_slots(dest), staged(dest)), ([0, 100, 200, 249], None))

    def test_fix1_a_failed_frames_fsync_keeps_the_staging_copy(self):
        dest = self.staged_session(self.root)
        with self.frames_fsync_fails(dest):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertEqual(staged(dest), ["f_00000001.png"])                    # the recovery copy is kept
        self.assertTrue(self.same_inode(dest))
        code, out = run(self.root, "--apply")                                # and the next run completes it
        self.assertEqual(code, 0, out)
        self.assertEqual((frame_slots(dest), staged(dest)), ([0, 100, 200, 249], None))

    def test_fix1_a_rejected_file_whose_restore_fails_keeps_its_staging_copy(self):
        _, dest = make_captured(self.root)
        staged_file = os.path.join(dest, "frames", ret.STAGING, "f_00000001.png")
        changed = []

        def hook(r, name):
            if name == "after_stage" and not changed:
                with open(staged_file, "ab") as f:
                    f.write(b"x")
                changed.append(name)
        with self.frames_fsync_fails(dest), mock.patch.object(ret.Retention, "hook", hook):
            code, out = run(self.root, "--apply")
        self.assertEqual(code, 2, out)
        self.assertIn("putting it back failed", out)
        self.assertEqual(staged(dest), ["f_00000001.png"])
        with open(staged_file, "rb") as f:
            self.assertEqual(f.read(), tiny_png(1) + b"x")                   # the rejected bytes are preserved
        code, out = run(self.root, "--apply")                                # restored, then held: it changed
        self.assertEqual(code, 2, out)
        self.assertIn("recovered ", out)
        self.assertEqual(staged(dest), None)
        with open(os.path.join(dest, "frames", "f_00000001.png"), "rb") as f:
            self.assertEqual(f.read(), tiny_png(1) + b"x")

    # -- fix 5: an interrupted restore ------------------------------------------------------------------
    def test_fix5_a_kill_at_each_restore_boundary_is_completed_by_the_next_run(self):
        for hook in ("restore_linked", "restore_synced"):
            with self.subTest(hook):
                root = self.sub(hook)
                dest = self.staged_session(root)
                self.kill(root, hook)                                       # killed inside the recovery's restore
                self.assertEqual(staged(dest), ["f_00000001.png"])
                self.assertTrue(self.same_inode(dest))                      # one inode under both names
                self.assertTrue(harvest.verify_sums(dest)[0])                # counted once
                code, out = run(root, "--apply")
                self.assertEqual(code, 0, out)
                self.assertIn("recovered ", out)
                self.assertEqual((frame_slots(dest), staged(dest)), ([0, 100, 200, 249], None))

    # -- fix 3: the pre-mutation recheck ---------------------------------------------------------------
    def test_fix3_admission_kept_files_and_age_are_rechecked_before_the_first_mutation(self):
        def session_json_not_arland(root, unit):
            meta = read_json(os.path.join(unit, "session.json"))
            meta["map"].update({"map_id": "everon", "terrain": "Eden",
                                "root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"})
            write(os.path.join(unit, "session.json"), jtext(meta))

        def kept_frame_truncated(root, unit):
            write(os.path.join(unit, "frames", "f_00000100.png"), tiny_png(100)[:20])

        def metadata_changed(root, unit):
            with open(os.path.join(unit, "session_end.json"), "ab") as f:
                f.write(b" ")

        def kept_derivative_changed(root, unit):
            with open(os.path.join(unit, "_harvest", "deriv", "f_00000000.jpg"), "ab") as f:
                f.write(b"x")

        def probe_csv_written(root, unit):
            with open(os.path.join(unit, "e05_frames.csv"), "ab") as f:
                f.write(b"1,2\n")

        def probe_kept_frame_written(root, unit):
            os.utime(os.path.join(unit, "frames", "f_00000200.png"))

        cases = {"session.json no longer Arland": ("session", session_json_not_arland, "session.json changed after "
                                                   "the session was planned (admission"),
                 "kept frame truncated": ("session", kept_frame_truncated, "frames/f_00000100.png is gone or changed"),
                 "metadata changed": ("session", metadata_changed, "session_end.json is gone or changed"),
                 "kept derivative changed": ("session", kept_derivative_changed,
                                             "_harvest/deriv/f_00000000.jpg is gone or changed"),
                 "probe csv written": ("probe", probe_csv_written, "a file of the play was written after it was "
                                       "planned"),
                 "probe kept frame written": ("probe", probe_kept_frame_written, "a file of the play was written "
                                              "after it was planned")}
        for n, (name, (mode, change, text)) in enumerate(sorted(cases.items())):
            with self.subTest(name):
                root = self.sub("case%d" % n)
                if mode == "session":
                    _, unit = make_captured(root)
                    argv = ["--apply"]
                else:
                    unit = make_play(root)
                    argv = ["--probes", "--apply"]
                original = ret.Retention.apply

                def apply(r, plan):
                    change(root, unit)
                    return original(r, plan)
                with mock.patch.object(ret.Retention, "apply", apply):
                    code, out = run(root, *argv)
                self.assertEqual(code, 2, out)
                self.assertIn(text, out)
                self.assertEqual(len(frame_slots(unit)), 250, out)
                for sub in ("", "_harvest"):
                    self.assertFalse(os.path.exists(os.path.join(unit, sub, "RETENTION.json")), out)

    # -- the --apply preflight ------------------------------------------------------------------------
    def test_preflight_refuses_apply_while_a_unit_in_scope_was_written_recently(self):
        _, dest = make_captured(self.root)                                  # files written just now
        before = th.snapshot(self.root)
        code, out = run(self.root, "--apply", quiet=None)
        self.assertEqual(code, 3, out)
        self.assertIn("refused: --apply preflight: a file of %s was modified" % os.path.relpath(dest, self.root),
                      out)
        self.assertEqual(th.snapshot(self.root), before)
        code, out = run(self.root, quiet=None)                              # a dry run is not refused
        self.assertEqual(code, 0, out)
        self.assertIn("would-delete", out)
        age(dest, 1)
        code, out = run(self.root, "--apply", quiet=None)
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(dest), [0, 100, 200, 249])

    def test_preflight_covers_every_play_in_scope(self):
        old = make_play(self.root, launch="20261024T000000Z-wb-0060")
        busy = make_play(self.root, launch="20261024T000000Z-wb-0061", footer=None, age_days=0)    # being written
        before = th.snapshot(self.root)
        code, out = run(self.root, "--probes", "--apply", quiet=None)
        self.assertEqual(code, 3, out)
        self.assertIn("--apply preflight: a file of %s was modified" % os.path.relpath(busy, self.root), out)
        self.assertEqual(th.snapshot(self.root), before)
        code, out = run(self.root, "--probes", "--apply", "--under", old, quiet=None)   # only the quiet play
        self.assertEqual(code, 0, out)
        self.assertEqual(frame_slots(old), [0, 100, 200, 249])


class ScopeTest(unittest.TestCase):
    def setUp(self):
        self.root = th.temp_dir("dd6ret_r_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_roots_and_under_outside_the_root_are_refused(self):
        self.assertEqual(harvest.DATA_ROOT, "/mnt/e/DriveDiT")
        for bad in ("/tmp/x", "/mnt/c/Users/x", "C:\\DriveDiT", "/mnt/e/DriveDiT", "/mnt/e/DriveDiT/captures", "/",
                    "/dev/shm"):
            code, out = run(bad, "--apply")
            self.assertEqual(code, 3, (bad, out))
            self.assertTrue(out.startswith("refused: "), out)
        _, dest = make_captured(self.root)
        pdir = make_play(self.root)
        legacy = os.path.join(self.root, "wb_profile", "profile", "DrivingData")
        synth.write_legacy_session(legacy)
        os.symlink(os.path.join(self.root, "captures"), os.path.join(self.root, "captures_link"))
        os.makedirs(os.path.join(self.root, "captures", "_rejected"))
        cases = [((), "/dev/shm", "is outside"), ((), os.path.dirname(self.root), "is outside"),
                 ((), legacy, "dc-1.0 $profile:DrivingData sessions are out of scope"),
                 (("--probes",), legacy, "dc-1.0 $profile:DrivingData sessions are out of scope"),
                 ((), os.path.join(legacy, "session_9001"), "out of scope"),
                 ((), os.path.join(self.root, "captures", "_rejected"), "is not inside captures/arland"),
                 ((), os.path.join(self.root, "captures"), "is not inside captures/arland"),
                 ((), pdir, "is not inside captures/arland"),
                 (("--probes",), dest, "is not inside a <profile>/profile/DC_Probes folder"),
                 ((), os.path.join(self.root, "captures_link", "arland"), "is reached through a symlink"),
                 ((), os.path.join(self.root, "captures", "arland", "no_such_dir"), "is not an existing directory")]
        before = th.snapshot(self.root)
        for mode, under, text in cases:
            with self.subTest(under=under, mode=mode):
                code, out = run(self.root, "--apply", "--under", under, *mode)
                self.assertEqual(code, 3, out)
                self.assertTrue(out.startswith("refused: --under"), out)
                self.assertIn(text, out)
        self.assertEqual(th.snapshot(self.root), before)
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            run(self.root, "--every-n", "0")
        self.assertEqual(cm.exception.code, 3)

    def test_dry_run_writes_nothing_and_prints_totals(self):
        make_captured(self.root, tag="5e01")
        make_captured(self.root, tag="5e02", catalogued=False)
        make_play(self.root)
        before = th.snapshot(self.root)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        self.assertIn("total: 1 session(s), 246 raw screenshot(s) to delete, 4 kept, to free", out)
        self.assertIn("1 skipped", out)
        code, out = run(self.root, "--probes", "--list")
        self.assertEqual(code, 0, out)
        self.assertIn("  keep   frames/f_00000100.png (every_100)", out)
        self.assertIn("  delete frames/f_00000001.png %d" % len(tiny_png(1)), out)
        self.assertIn("total: 1 probe play(s), 246 raw screenshot(s) to delete, 4 kept, to free", out)
        self.assertEqual(th.snapshot(self.root), before)

    def test_apply_with_nothing_to_do_writes_nothing(self):
        make_play(self.root, footer=None)
        before = th.snapshot(self.root)
        code, out = run(self.root, "--probes", "--apply")
        self.assertEqual((code, th.snapshot(self.root)), (0, before), out)
        self.assertFalse(os.path.exists(os.path.join(self.root, "catalog")))

    def test_a_second_run_is_refused_while_the_lock_is_held(self):
        import fcntl
        make_play(self.root)
        make_captured(self.root)
        fd = os.open(os.path.join(self.root, "catalog", "harvest.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            before = th.snapshot(self.root)
            for mode in ((), ("--probes",)):
                code, out = run(self.root, "--apply", *mode)
                self.assertEqual((code, th.snapshot(self.root)), (3, before), out)
                self.assertIn("another harvester holds", out)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
