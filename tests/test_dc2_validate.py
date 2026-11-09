#!/usr/bin/env python3
"""Tests for tools/validator/dc_validate.py (DD-5-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_validate.py

dc_validate runs in-process through its main(argv) (stdout captured). Acceptance of the ticket shown here:
  * dc_validate exits 0 on the golden fixture (frame-bound rows at 30 Hz, 5 Hz images) and on the 10 Hz trial-conf
    session with its capture utilisation; the frame-bound rules of the freeze (covered_steps, the callback after
    its frame's EOnFrame, tick gaps on the measured step dt, the rate settings of D-B) fail on mutations;
  * each negative fixture (tests/fixtures/capture_dc2/negative/*/mutation.json, generated here) fails exactly the
    check it names;
  * C17 passes a synthetic session along a loop route and fails the same session against the mirrored route, and
    fails it with an ego collision;
  * the crash session of make_synthetic (contract 9.5.1) passes every check, and C36 (payloads, the collision
    threshold and onset, the start/end pairs, crash terminations, ego_destroyed), C35 (unknown vehicle slots) and
    C15 (only the ego's events mask its rows) fail or hold on mutations of it;
  * dc-1.0 sessions FAIL for their known reasons (synthetic legacy session; the real sessions 4811, 2702 and 2929
    when their data-drive copy exists (DD_TEST_LOCAL_CONFIG opt-in), else that test skips: TICKETS_UNIFIED decision 13 is open);
  * reports are written only on the data drive (a refusal for /dev/shm and the system drive), never to a path built from
    session.json; thresholds equal CAPTURE_PROGRAM 5.2 verbatim, chosen by pin.
FailLoudTest turns each probe of the 2026-10-22 review (t0b/review/REVIEW_DD5_MIN.md) into a test: one defect on a
copy of the golden session (commit lines kept true), and the exit code and failing check it must produce.
Sessions are generated under $TMPDIR (on /dev/shm/, else /dev/shm/). The report-writing tests point the validator's
report root at a temporary directory; the one test against a real data drive runs only when DC_TEST_E_SCRATCH names
an existing directory on the data drive of the local config DD_TEST_LOCAL_CONFIG names (else it skips). Every other
test reads the synthetic local config (DD-22; data drive /mnt/x, system drive /mnt/z), or none: a real default report
root is never used here.
"""

import contextlib
import glob
import importlib.util
import io
import json
import math
import os
import re
import shutil
import tempfile
import types
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# DD-22: test scratch only in validated RAM (tests/fixtures/local_config/scratch.py), never the system temp dir.
_scratch_spec = importlib.util.spec_from_file_location(
    "dd22_scratch", os.path.join(REPO, "tests", "fixtures", "local_config", "scratch.py"))
scratch = importlib.util.module_from_spec(_scratch_spec)
_scratch_spec.loader.exec_module(scratch)
FIXTURES = os.path.join(REPO, "tests", "fixtures", "capture_dc2")
GOLDEN_ID = "arland_T0b_20261025T091512Z_wb_i00_5e00"
GOLDEN_DIR = os.path.join(FIXTURES, "golden", GOLDEN_ID)
ROUTES = os.path.join(FIXTURES, "routes")
CARD = os.path.join(FIXTURES, "cards", "S105_synthetic.json")
# DD-22: documents of a real machine are compared only when DD_TEST_LOCAL_CONFIG names its local config (opt-in).
_evidence_spec = importlib.util.spec_from_file_location(
    "dd_test_evidence", os.path.join(REPO, "tests", "fixtures", "local_config", "evidence.py"))
evidence = importlib.util.module_from_spec(_evidence_spec)
_evidence_spec.loader.exec_module(evidence)
PROGRAM_MD = evidence.evidence_path("REALIGN_ROOT", "program", "CAPTURE_PROGRAM.md")
LEGACY_ROOT = evidence.evidence_path("DATA_ROOT", "captures", "_legacy_DrivingData_C")
LOCAL_CONFIG_FIXTURE = evidence.FIXTURE
_bv_spec = importlib.util.spec_from_file_location(
    "dd22_boundary_vectors", os.path.join(REPO, "tests", "fixtures", "local_config", "boundary_vectors.py"))
bv = importlib.util.module_from_spec(_bv_spec)
_bv_spec.loader.exec_module(bv)
CHECK_IDS = ("C01", "C02", "C05", "C06", "C07", "C08", "C10", "C11", "C12", "C13a", "C13b", "C15", "C17", "C20",
             "C21", "C28", "C31", "C34", "C35", "C36", "C37")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# DD-22: the synthetic local config (also set by tests/conftest.py under pytest), so the repo's fixtures on any
# drive are readable and no real drive is the system drive here.
os.environ["DRIVEDIT_LOCAL_CONFIG"] = os.path.join(REPO, "tests", "fixtures", "local_config", "placeholder.local.env")

dc_validate = load("dc_validate", "tools/validator/dc_validate.py")
LC = dc_validate._lc()
FIXTURE_CFG = LC.load(LOCAL_CONFIG_FIXTURE)
NO_SUCH_E_DIR = FIXTURE_CFG.drive("DATA_ROOT") + "/__dd5_no_such_dir__"   # on the placeholder data drive, never mounted
synth = load("dc2_make_synthetic_v", "tests/fixtures/capture_dc2/make_synthetic.py")
reader = load("dc2_reader_v", "data/dc2/reader.py")


def temp_dir(prefix):
    """A new directory in validated RAM (tests/fixtures/local_config/scratch.py); skipped where the write boundary refuses RAM."""
    return scratch.mkdtemp(prefix, real_boundary=True)


def run(*argv):
    """(exit code, stdout) of dc_validate.main(argv)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = dc_validate.main(list(argv))
    return code, out.getvalue()


def run_json(*argv):
    code, text = run(*(list(argv) + ["--json", "--no-report"]))
    return code, json.loads(text)


def failing(report):
    return sorted(c["id"] for c in report["checks"] if c["status"] == "FAIL")


def by_id(report):
    return {c["id"]: c for c in report["checks"]}


def edit_json(path, fn):
    with open(path, "rb") as f:
        obj = json.loads(f.read().decode("utf-8"))
    fn(obj)
    with open(path, "wb") as f:
        f.write((json.dumps(obj, indent=2) + "\n").encode("ascii"))


_RAM = []


def copy_golden(tmp, name=GOLDEN_ID):
    """A copy of the golden session under tmp (from one RAM copy; copies from drvfs are slow)."""
    if not _RAM:
        base = temp_dir("dc2_golden_src_")
        _RAM.append(base)
        shutil.copytree(GOLDEN_DIR, os.path.join(base, GOLDEN_ID))
    dst = os.path.join(tmp, name)
    shutil.copytree(os.path.join(_RAM[0], GOLDEN_ID), dst)
    return dst


def tearDownModule():
    for tmp in _RAM:
        shutil.rmtree(tmp, ignore_errors=True)


def read_lines(path):
    with open(path, "rb") as f:
        return f.read().decode("ascii").split("\n")[:-1]


def rewrite_chunk(session, stream, chunk_id, fn):
    """fn(columns, rows as lists of text) -> rows; the commit line's n_rows, n_bytes and keys stay true."""
    path = os.path.join(session, stream, "chunk_%05d.csv" % chunk_id)
    lines = read_lines(path)
    cols = lines[0].split(",")
    rows = fn(cols, [line.split(",") for line in lines[1:]])
    data = ("\n".join([lines[0]] + [",".join(r) for r in rows]) + "\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(data)
    commit = os.path.join(session, stream, "commit.csv")
    clines = read_lines(commit)
    parts = clines[chunk_id + 1].split(",")
    parts[3], parts[4] = str(len(rows)), str(len(data))
    if rows:
        parts[1], parts[2] = rows[0][0], rows[-1][0]
    clines[chunk_id + 1] = ",".join(parts)
    with open(commit, "wb") as f:
        f.write(("\n".join(clines) + "\n").encode("ascii"))


def set_column(session, stream, column, fn):
    """Every chunk of a stream: fn(key, text) -> text for one column (key = the row's first column)."""
    for name in sorted(os.listdir(os.path.join(session, stream))):
        if name.startswith("chunk_"):
            def edit(cols, rows):
                j = cols.index(column)
                for r in rows:
                    r[j] = fn(int(r[0]), r[j])
                return rows
            rewrite_chunk(session, stream, int(name[6:11]), edit)


class GoldenTest(unittest.TestCase):
    def test_golden_exits_0(self):
        code, text = run("--session", GOLDEN_DIR, "--no-report", "--routes-dir", ROUTES, "--vehicle-card", CARD)
        self.assertEqual(code, 0, text)
        self.assertIn("VERDICT: PASS (exit 0)", text)

    def test_golden_report_content(self):
        code, rep = run_json("--session", GOLDEN_DIR, "--routes-dir", ROUTES)
        self.assertEqual(code, 0)
        self.assertEqual(tuple(c["id"] for c in rep["checks"]), CHECK_IDS)
        for c in rep["checks"]:
            self.assertIn(c["status"], ("PASS", "N/A"), c)
            self.assertEqual(set(c) - {"not_evaluated"},
                             {"id", "severity", "status", "value", "threshold", "detail", "provisional"})
        checks = by_id(rep)
        self.assertEqual((checks["C17"]["severity"], checks["C31"]["severity"]), ("INFO", "INFO"))   # T0 trial
        # CAPTURE_PROGRAM 4.4 T0-MIN: C12, C13b and C21 SOFT; the rest HARD (C20, C35 stricter by choice); C37 is SOFT
        # by the freeze decision D-B
        self.assertEqual([c for c in CHECK_IDS if checks[c]["severity"] == "SOFT"], ["C12", "C13b", "C21", "C37"])
        self.assertEqual(rep["thresholds_file"], "thresholds_dc2.1.8.0.13.json")
        # provisional = a TBD the streams JSON does not mark decided: after the freeze only TBD-10 (C36)
        self.assertEqual([c for c in CHECK_IDS if checks[c]["provisional"]], ["C36"])
        self.assertFalse(checks["C08"]["provisional"])     # TBD-1 and TBD-2 decided (D-A, D-B)
        self.assertEqual(checks["C13a"]["value"]["join_failures"], 0)
        self.assertEqual(checks["C13a"]["value"]["done_minus_one_agree"], 1.0)
        self.assertEqual(checks["C13b"]["value"]["lag_frames"], 1)        # the image of frame C shows hf row C + 1
        self.assertEqual(checks["C37"]["status"], "N/A")                     # the 5 Hz default
        self.assertEqual(checks["C08"]["value"]["missed_telemetry_rows"], 0)
        self.assertEqual(checks["C08"]["value"]["N"], 6)
        self.assertEqual(checks["C02"]["value"]["split"], "train")
        self.assertEqual((rep["verdict"], rep["tier"], rep["contract_version"]), ("PASS", "T0", "dc-2.0"))
        self.assertEqual(rep["contract_status"], reader.load_schema().status)

    def test_negative_fixtures_fail_exactly_their_check(self):
        specs = sorted(glob.glob(os.path.join(FIXTURES, "negative", "*", "mutation.json")))
        self.assertEqual([os.path.basename(os.path.dirname(p)) for p in specs],
                         ["crash_unexplained", "destroyed_unexplained", "gap", "missing_commit", "reused_slot"])
        tmp = temp_dir("dc2_negative_")
        try:
            for path in specs:
                with open(path, "rb") as f:
                    spec = json.loads(f.read().decode("ascii"))
                out = synth.generate(os.path.join(tmp, spec["variant"]), variant=spec["variant"])
                code, rep = run_json("--session", out["session_dir"], "--routes-dir", ROUTES)
                self.assertEqual(code, 1, spec["variant"])
                self.assertEqual(failing(rep), spec["fails"], (spec["variant"], [
                    (c["id"], c["detail"]) for c in rep["checks"] if c["status"] != "PASS"]))
                self.assertTrue(all(c["status"] in ("PASS", "N/A") for c in rep["checks"]
                                    if c["id"] not in spec["fails"]))
        finally:
            shutil.rmtree(tmp)

    def test_trial_rules(self):
        tmp = temp_dir("dc2_trial_")
        try:
            session = os.path.join(tmp, GOLDEN_ID)
            shutil.copytree(GOLDEN_DIR, session)
            edit_json(os.path.join(session, "session.json"), lambda o: o["addon"].update({"git_dirty": True}))
            code, rep = run_json("--session", session)
            self.assertEqual((code, by_id(rep)["C02"]["status"]), (0, "PASS"))        # T0: a dirty build is fine
            code, rep = run_json("--session", session, "--trial", "T1")
            self.assertEqual((code, failing(rep)), (1, ["C02"]))                       # dirty=false from T1
            self.assertIn("git_dirty", by_id(rep)["C02"]["detail"])
            self.assertEqual(by_id(rep)["C17"]["severity"], "SOFT")
            self.assertEqual(by_id(rep)["C31"]["severity"], "HARD")
        finally:
            shutil.rmtree(tmp)

    def test_missing_engine_logs_is_a_warning(self):
        tmp = temp_dir("dc2_logs_")
        try:
            session = os.path.join(tmp, GOLDEN_ID)
            shutil.copytree(GOLDEN_DIR, session)
            shutil.rmtree(os.path.join(session, "_engine_logs"))
            code, rep = run_json("--session", session)
            self.assertEqual(code, 2)
            self.assertEqual(rep["verdict"], "PASS_WITH_WARNINGS")
            self.assertEqual(by_id(rep)["C20"]["status"], "NOT_EVALUATED")
            with open(os.path.join(session, "x.log"), "w", encoding="ascii") as f:
                f.write("12:00:00.000 SCRIPT    (E): @\"Scripts/Game/DataCapture/DC_Session.c,12\": write_fail\n")
            code, rep = run_json("--session", session, "--engine-logs", session)
            self.assertEqual((code, failing(rep)), (1, ["C11", "C20"]))   # and no Screenshot-saved line
            self.assertEqual(by_id(rep)["C20"]["value"]["hard"]["write_fail"], 1)
        finally:
            shutil.rmtree(tmp)

    def test_unreadable_and_legacy_sessions_fail_c01(self):
        tmp = temp_dir("dc2_unreadable_")
        try:
            session = os.path.join(tmp, GOLDEN_ID)
            shutil.copytree(GOLDEN_DIR, session)
            os.remove(os.path.join(session, "session.json"))
            code, rep = run_json("--session", session)
            self.assertEqual((code, failing(rep)), (1, ["C01"]))
            legacy = synth.write_legacy_session(tmp)
            code, rep = run_json("--session", legacy)
            self.assertEqual((code, failing(rep)), (1, ["C01"]))
            self.assertIn("dc-1.0", by_id(rep)["C01"]["detail"])
        finally:
            shutil.rmtree(tmp)


def add_event(session, after_seq, kind, payload, veh_slot_id, physics_step=None):
    """Insert one events row after event_seq after_seq (same episode, step, frame and world time unless
    physics_step is given), renumber the rows after it, and keep commit.csv and session_end.json counts true."""
    sdir = os.path.join(session, "events")
    names = sorted(n for n in os.listdir(sdir) if n.startswith("chunk_"))
    seq_shift = {"done": False}
    for name in names:
        def edit(cols, rows):
            out = []
            for r in rows:
                if seq_shift["done"]:
                    r[0] = str(int(r[0]) + 1)
                out.append(r)
                if int(r[0]) == after_seq and not seq_shift["done"]:
                    new = list(r)
                    new[0] = str(after_seq + 1)
                    new[cols.index("kind")] = kind
                    new[cols.index("payload")] = payload
                    new[cols.index("veh_slot_id")] = str(veh_slot_id)
                    if physics_step is not None:
                        new[cols.index("physics_step")] = str(physics_step)
                    out.append(new)
                    seq_shift["done"] = True
            return out
        rewrite_chunk(session, "events", int(name[6:11]), edit)
    edit_json(os.path.join(session, "session_end.json"),
              lambda o: o["counts"]["events"].update({"rows": o["counts"]["events"]["rows"] + 1,
                                                      "last_key": o["counts"]["events"]["last_key"] + 1}))


class FailLoudTest(unittest.TestCase):
    """The review's mutation probes (REVIEW_DD5_MIN.md F-1 to F-23): each must now fail, warn or be refused."""

    def setUp(self):
        self.tmp = temp_dir("dc2_failloud_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def golden(self, sub="s"):
        """A golden copy in its own sub-directory (the directory name must stay the session id, C01)."""
        return copy_golden(os.path.join(self.tmp, sub))

    def check(self, session, *extra):
        return run_json("--session", session, "--routes-dir", ROUTES, "--vehicle-card", CARD, *extra)

    def test_failed_seal_fails_c01(self):                                     # F-1
        session = self.golden()
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o.update({"status": "failed", "end_reason": "script_error:write_fail"}))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C01"]))
        self.assertIn("status 'failed'", by_id(rep)["C01"]["detail"])

    def test_seal_fields_fail(self):                                          # F-1
        session = self.golden()
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o.update({"contract_version": "dc-1.0", "episodes": 9}))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C01"]))
        session = self.golden("end")
        edit_json(os.path.join(session, "session_end.json"), lambda o: o["end"].update({"physics_step": 99999}))
        code, rep = self.check(session)
        self.assertIn("C34", failing(rep))

    def _drop_last_chunk(self, session, stream, last_key):
        os.remove(os.path.join(session, stream, "chunk_00001.csv"))
        commit = os.path.join(session, stream, "commit.csv")
        kept = read_lines(commit)[:2]
        with open(commit, "wb") as f:
            f.write(("\n".join(kept) + "\n").encode("ascii"))
        rows = len(read_lines(os.path.join(session, stream, "chunk_00000.csv"))) - 1
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o["counts"][stream].update({"rows": rows, "chunks": 1, "last_key": last_key}))

    def test_truncated_stream_tails_fail(self):                               # F-2
        session = self.golden()
        self._drop_last_chunk(session, "telemetry", 262)                       # counts kept honest
        for trial in ("T0b", "T1"):
            code, rep = self.check(session, "--trial", trial)
            self.assertEqual((code, failing(rep)), (1, ["C06", "C08", "C34"]), trial)   # joins, cadence, tail
            self.assertIn("without their telemetry row", by_id(rep)["C08"]["detail"])
        session = self.golden("ctl")
        self._drop_last_chunk(session, "ctl", 300)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C34"]))
        self.assertIn("ctl/: committed rows end at physics_step 300", by_id(rep)["C34"]["detail"])

    def test_image_grid_holes_fail_c08(self):                                 # F-3
        session = self.golden()
        frames = os.path.join(session, "frames")

        def thin(cols, rows):          # keep every other slot, renumbered: grid points 1, 3, 5 ... have no slot
            out = []
            for r in rows:
                old = int(r[0])
                if old % 2:
                    os.remove(os.path.join(frames, "f_%08d.png" % old))
                    continue
                os.rename(os.path.join(frames, "f_%08d.png" % old), os.path.join(frames, "n_%08d.png" % (old // 2)))
                r[0] = str(old // 2)
                r[cols.index("path")] = "frames/f_%08d" % (old // 2)
                out.append(r)
            return out

        rewrite_chunk(session, "img", 0, thin)
        rewrite_chunk(session, "img", 1, thin)
        for name in os.listdir(frames):
            os.rename(os.path.join(frames, name), os.path.join(frames, "f_" + name[2:]))
        code, rep = self.check(session)
        self.assertEqual(code, 1)
        self.assertIn("C08", failing(rep))
        self.assertIn("grid point(s) without a slot row", by_id(rep)["C08"]["detail"])

    def test_image_rate_rules(self):                                          # F-3, D-B
        cases = ((10.0, None, "needs clocks.image_hz_setting 'trial_conf'"),
                 (10.0, "trial_conf", "differs from image_hz"),
                 (5.0, "trial_conf", "needs clocks.image_hz_setting 'default'"),     # review round 1, X6
                 (1.0, None, "is not 5 (the default) or 10"))
        for image_hz, setting, why in cases:
            session = self.golden("hz%d%s" % (image_hz, setting or ""))

            def edit(o):
                o["clocks"]["image_hz"] = image_hz
                if setting:
                    o["clocks"]["image_hz_setting"] = setting

            edit_json(os.path.join(session, "session.json"), edit)
            code, rep = self.check(session)
            self.assertEqual((code, failing(rep)), (1, ["C08"]), (image_hz, setting))
            self.assertIn(why, by_id(rep)["C08"]["detail"])
        session = self.golden("n7")              # N must follow the measured step dt (D-A), never the raw slice
        edit_json(os.path.join(session, "session.json"),
                  lambda o: o["clocks"].update({"physics_steps_per_decision": 2}))
        code, rep = self.check(session)
        self.assertIn("C08", failing(rep))
        self.assertIn("the D-A rule for step dt", by_id(rep)["C08"]["detail"])

    def test_10hz_setting_passes_with_its_evidence(self):                    # D-B
        out = synth.generate(os.path.join(self.tmp, "g10"), kind="golden_10hz", routes_dir=os.path.join(self.tmp, "r"))
        code, rep = run_json("--session", out["session_dir"], "--routes-dir", os.path.join(self.tmp, "r"),
                             "--vehicle-card", CARD)
        self.assertEqual(code, 0, [(c["id"], c["detail"]) for c in rep["checks"] if c["status"] != "PASS"])
        c08, c37 = by_id(rep)["C08"], by_id(rep)["C37"]
        self.assertEqual((c08["value"]["N"], c08["value"]["decision_hz"], c08["value"]["image_hz_setting"]),
                         (3, 10.0, "trial_conf"))
        self.assertEqual((c37["severity"], c37["status"]), ("SOFT", "PASS"))
        self.assertAlmostEqual(c37["value"]["u95"], 0.8)                  # an 80 ms capture stall per 100 ms slot
        # without done_render_frame the rows are completions the writer dropped (C13a, HARD); C37 still measures the
        # stalls from the decoded frames (review round 2, N2)
        session = os.path.join(self.tmp, "g10_noev", out["session_id"])
        shutil.copytree(out["session_dir"], session)
        set_column(session, "img", "done_render_frame", lambda key, v: "")
        code, rep = run_json("--session", session, "--routes-dir", os.path.join(self.tmp, "r"), "--vehicle-card",
                             CARD)
        # C21 too: without done_render_frame no screenshot row is excluded (C21 decision 2026-11-04)
        self.assertEqual((code, failing(rep)), (1, ["C13a", "C21"]))
        self.assertIn("a completion the writer dropped", by_id(rep)["C13a"]["detail"])
        c37 = by_id(rep)["C37"]
        self.assertEqual((c37["status"], c37["value"]["without_done_render_frame"]), ("PASS", c37["value"]["issued"]))
        self.assertAlmostEqual(c37["value"]["u95"], 0.8)
        # review round 1, X5: a redirected completion (done_render_frame one frame late) is contradicted by the
        # decoded frames: the evidence is refused, never a pass
        session = os.path.join(self.tmp, "g10_late", out["session_id"])
        shutil.copytree(out["session_dir"], session)
        set_column(session, "img", "done_render_frame", lambda key, v: str(int(v) + 1) if v else v)
        code, rep = run_json("--session", session, "--routes-dir", os.path.join(self.tmp, "r"), "--vehicle-card",
                             CARD)
        c37 = by_id(rep)["C37"]
        self.assertEqual((c37["status"], c37["value"]["contradicted_slots"]), ("FAIL", c37["value"]["issued"]))
        self.assertIn("contradicted by the decoded frame", c37["detail"])

    def test_frame_bound_rows_are_checked(self):                              # D-A
        session = self.golden("cov")
        set_column(session, "ctl", "covered_steps", lambda key, v: "2" if key == 154 else v)    # 153 -> 154 is 1
        code, rep = self.check(session)
        self.assertIn("C06", failing(rep))
        self.assertIn("covered", by_id(rep)["C06"]["detail"])
        session = self.golden("frame")              # a callback in the frame of the hf row that names it
        set_column(session, "ctl", "render_frame", lambda key, v: str(int(v) + 1) if key == 154 else v)
        set_column(session, "telemetry", "render_frame", lambda key, v: str(int(v) + 1) if key == 154 else v)
        code, rep = self.check(session)
        self.assertIn("C07", failing(rep))
        self.assertIn("the same or a later render frame", by_id(rep)["C07"]["detail"])
        session = self.golden("tick")               # world time that ran without physics steps: a tick gap
        set_column(session, "ctl", "world_time_ms", lambda key, v: str(int(v) + 150) if key >= 154 else v)
        code, rep = self.check(session)
        self.assertIn("C28", failing(rep))
        self.assertIn("steps 153..154", by_id(rep)["C28"]["detail"])

    def test_frame_clock_cross_stream_rules(self):                            # review round 1, X3 / C-M3
        """ctl and telemetry share render_frame and world_time_ms per key (C06); ctl world_time_ms is the hf world time
        of its frame and hf.last_physics_step the newest earlier callback (C07). Codex's and Claude's mutations."""
        hf = reader.open_session(GOLDEN_DIR, vehicle_card=CARD).streams["hf"]
        lps = dict(zip(hf.col("render_frame"), hf.col("last_physics_step")))
        stale_frame = [f for f in sorted(lps) if lps.get(f - 1) is not None and 0 <= lps[f - 1] < lps[f]][5]
        cases = (("tel_frame", [("telemetry", "render_frame", 154, lambda v: str(int(v) + 10000))], "C06",
                  "differ in render_frame or world_time_ms"),
                 ("tel_world", [("telemetry", "world_time_ms", 154, lambda v: str(int(v) + 1000))], "C06",
                  "differ in render_frame or world_time_ms"),
                 ("p1", [("telemetry", "world_time_ms", 154, lambda v: str(int(v) + 7))], "C06",
                  "differ in render_frame or world_time_ms"),
                 ("p2", [("telemetry", "render_frame", 154, lambda v: str(int(v) - 1))], "C06",
                  "differ in render_frame or world_time_ms"),
                 ("p3", [("ctl", "world_time_ms", 154, lambda v: str(int(v) + 5)),
                         ("telemetry", "world_time_ms", 154, lambda v: str(int(v) + 5))], "C07",
                  "differs from the hf row of their render frame"),
                 ("p4", [("hf", "last_physics_step", 262100, lambda v: "207")], "C07",
                  "not the newest callback of an earlier render frame"),
                 ("stale", [("hf", "last_physics_step", stale_frame, lambda v: str(lps[stale_frame - 1]))], "C07",
                  "not the newest callback of an earlier render frame"))
        for name, edits, check, why in cases:
            session = self.golden(name)
            for stream, column, key, fn in edits:
                set_column(session, stream, column, lambda k, v, key=key, fn=fn: fn(v) if k == key else v)
            code, rep = self.check(session)
            self.assertIn(check, failing(rep), name)
            self.assertIn(why, by_id(rep)[check]["detail"], name)
            self.assertEqual(by_id(rep)[check]["severity"], "HARD", name)

    def test_capture_past_the_seal_fails_c13a(self):                          # review round 2, N1
        """Codex: golden slot 67 (done at 262287, nothing pending) re-encoded to frame 262296, past the seal at
        262291: a join failure, never the seal exemption."""
        session = self.golden()
        render_slot(session, 67, 262296)
        code, rep = self.check(session)
        # C21 too: the excluded row (done) is no longer the frame after the decoded capture
        self.assertEqual((code, failing(rep)), (1, ["C13a", "C21"]))
        c13a = by_id(rep)["C13a"]
        self.assertEqual((c13a["value"]["join_failures"], c13a["value"]["pending_after_seal"]), (1, 0))

    def test_declared_pending_requests_fail_c13a(self):                       # review round 3, N6
        """Codex: the last 3 golden completions cleared and images.pending_at_end 3, no image edited. One request is
        in flight at a time: more than 1 pending fails C13a."""
        session = self.golden()
        last3 = set(range(65, 68))
        set_column(session, "img", "done_render_frame", lambda key, v: "" if key in last3 else v)
        edit_json(os.path.join(session, "session_end.json"), lambda o: o["images"].update({"pending_at_end": 3}))
        code, rep = self.check(session)
        self.assertEqual(code, 1)
        self.assertIn("C13a", failing(rep))
        c13a = by_id(rep)["C13a"]
        self.assertIn("at most 1 can be pending", c13a["detail"])
        self.assertEqual((c13a["value"]["pending_after_seal"], c13a["value"]["dropped_completions"]), (0, 3))

    def test_pending_at_end_type_fails_c13a(self):                            # review round 4
        """Codex: completed golden sessions declaring "1", 1.0, 3.0, true or null passed C13a. Only the JSON integer
        0 or 1 is a pending count (the reader's predicate)."""
        for name, value in (("str", "1"), ("float1", 1.0), ("float3", 3.0), ("bool", True), ("null", None)):
            session = self.golden(name)
            edit_json(os.path.join(session, "session_end.json"),
                      lambda o, value=value: o["images"].update({"pending_at_end": value}))
            code, rep = self.check(session)
            self.assertEqual((code, failing(rep)), (1, ["C13a"]), name)
            self.assertIn("is not the JSON integer 0 or 1", by_id(rep)["C13a"]["detail"], name)
            self.assertEqual(by_id(rep)["C13a"]["value"]["pending_after_seal"], 0, name)

    def test_pending_integrity_rules(self):                                   # review round 3, N6
        """The exemption is the unique newest issued slot: skipped_busy or make_failed rows may follow it (a seal
        queue), an older pending row is a dropped completion, pending_at_end 2 fails."""
        images = dc_validate.checks.images

        def session(rows, pending_end):
            end = {"images.pending_at_end": pending_end}
            return types.SimpleNamespace(images=[{"slot_id": i, "issue_status": st, "pending_at_seal": int(p)}
                                                 for i, (st, p) in enumerate(rows)],
                                         end=end, end_get=lambda k, d=None: end.get(k, d))

        queue = [("issued", False)] * 5 + [("issued", True), ("skipped_busy", False), ("skipped_busy", False)]
        self.assertEqual(images._pending_integrity(session(queue, 1)), {"issues": [], "dropped": []})
        self.assertEqual(images._pending_integrity(session(queue[:5], 0)), {"issues": [], "dropped": []})
        older = [("issued", False), ("issued", True), ("issued", False), ("issued", True)]
        result = images._pending_integrity(session(older, 1))
        self.assertEqual(result["dropped"], [1])
        result = images._pending_integrity(session(older, 2))
        self.assertEqual(result["dropped"], [1, 3])
        self.assertIn("at most 1", result["issues"][0])
        result = images._pending_integrity(session(queue[:5], 1))       # declared 1, nothing pending
        self.assertIn("is not pending", result["issues"][0])

    def test_hl_seq_before_the_first_command_fails_c07(self):                 # review round 2, N5
        session = self.golden()                  # Codex: +1 on every hl_seq after step 0 (0 during the warm-up)
        set_column(session, "ctl", "hl_seq", lambda key, v: str(int(v) + 1) if key > 0 else v)
        code, rep = self.check(session)
        self.assertIn("C07", failing(rep))
        self.assertIn("decision boundary", by_id(rep)["C07"]["detail"])
        session = self.golden("cmd")             # a command under the -1 sentinel
        set_column(session, "ctl", "hl_kappa", lambda key, v: "0.010000" if key == 5 else v)     # a warm-up row
        code, rep = self.check(session)
        self.assertIn("C07", failing(rep))
        self.assertIn("hl_seq -1 (the no-command sentinel)", by_id(rep)["C07"]["detail"])

    def test_unnumbered_commands_fail_c07(self):                              # review round 1, X7
        session = self.golden()
        set_column(session, "ctl", "hl_seq", lambda key, v: "-1")
        code, rep = self.check(session)
        self.assertIn("C07", failing(rep))
        self.assertIn("no hl_seq increment", by_id(rep)["C07"]["detail"])
        session = self.golden("kept")            # from one decision on, hl_seq lags by one: that decision unnumbered
        seq = dict(zip(*[reader.open_session(GOLDEN_DIR, vehicle_card=CARD).streams["ctl"].col(c)
                         for c in ("physics_step", "hl_seq")]))
        bump = [k for k in sorted(seq) if seq[k] >= 5 and seq[k] > seq.get(k - 1, seq[k])][0]
        set_column(session, "ctl", "hl_seq", lambda key, v: str(int(v) - 1) if key >= bump and int(v) >= 0 else v)
        code, rep = self.check(session)
        self.assertIn("C07", failing(rep))

    def test_written_skip_count_is_cross_checked(self):                       # review round 1, C-L4
        session = self.golden()
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o["observed"].update({"decision_grid_points_skipped": 3}))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C08"]))
        self.assertIn("observed.decision_grid_points_skipped 3, the rebuilt grid skips 0", by_id(rep)["C08"]["detail"])

    def test_lag_zero_fails_c13b(self):                                       # review round 1, C-M2
        session = self.golden()             # hf row n gets the pose of row n + 1: the image of C now matches row C
        rows = {}
        for name in sorted(os.listdir(os.path.join(session, "hf"))):
            if name.startswith("chunk_"):
                lines = read_lines(os.path.join(session, "hf", name))
                cols = lines[0].split(",")
                rows.update((int(line.split(",")[0]), line.split(",")) for line in lines[1:])
        for column in [c for c in cols if c.startswith("cam_") or c.startswith("veh_")]:
            j = cols.index(column)
            set_column(session, "hf", column, lambda key, v, j=j: rows[key + 1][j] if key + 1 in rows else v)
        code, rep = self.check(session)
        c13b = by_id(rep)["C13b"]
        self.assertEqual((c13b["severity"], c13b["status"], c13b["value"]["lag_frames"]), ("SOFT", "FAIL", 0))

    def test_future_obs_never_passes(self):                                   # review round 1, X1
        """The reader never picks a future image (test_dc2_reader); should a decision sample hold one anyway, C13a
        fails."""
        vreader = dc_validate.base.load_reader()
        orig = vreader._build_decisions

        def future(s):
            orig(s)
            d = [d for d in s.decisions if d["decision_step"] == 93][0]
            d.update({"obs_captured_render_frame": 261990, "obs_age_ms": -27, "obs_valid": 1})

        with mock.patch.object(vreader, "_build_decisions", future):
            code, rep = self.check(self.golden())
        self.assertEqual((code, failing(rep)), (1, ["C13a"]))
        self.assertIn("future image", by_id(rep)["C13a"]["detail"])

    def test_writer_r01_off_by_one_fails_c07(self):                           # F-5
        session = self.golden()
        orig = {}
        for name in ("chunk_00000.csv", "chunk_00001.csv"):
            lines = read_lines(os.path.join(session, "ctl", name))
            cols = lines[0].split(",")
            for line in lines[1:]:
                vals = line.split(",")
                orig[int(vals[0])] = {c: vals[cols.index(c)] for c in ("hl_seq", "hl_kappa", "hl_accel", "hl_dir")}
        for column in ("hl_seq", "hl_kappa", "hl_accel", "hl_dir"):          # the command one step early
            set_column(session, "ctl", column, lambda key, v: orig[key + 1][column] if key + 1 in orig else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C07"]))
        self.assertIn("decision boundary", by_id(rep)["C07"]["detail"])

    def test_command_changed_inside_a_window_fails_c07(self):                 # F-5
        session = self.golden()
        set_column(session, "ctl", "hl_kappa", lambda key, v: "0.012345" if key == 154 else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C07"]))
        self.assertIn("inside a decision interval", by_id(rep)["C07"]["detail"])

    def test_soft_checks_warn_at_t0_and_fail_from_t1(self):                   # F-6
        """C21 on a session whose ordinary frames are slow (36-45 ms): SOFT at T0, HARD from T1. hf.fps
        (System.GetFPS()) is information only since the C21 decision of 2026-11-04."""
        routes = os.path.join(self.tmp, "r")
        with mock.patch.dict(synth.SPECS["golden"], SLOW_FRAMES):
            out = synth.generate(os.path.join(self.tmp, "slow"), kind="golden", routes_dir=routes)
        code, rep = run_json("--session", out["session_dir"], "--routes-dir", routes, "--vehicle-card", CARD)
        self.assertEqual((code, failing(rep), by_id(rep)["C21"]["severity"]), (2, ["C21"], "SOFT"))
        code, rep = run_json("--session", out["session_dir"], "--routes-dir", routes, "--vehicle-card", CARD,
                             "--trial", "T1")
        self.assertEqual((code, failing(rep), by_id(rep)["C21"]["severity"]), (1, ["C21"], "HARD"))
        session = self.golden("gfps")            # the engine's smoothed hf.fps alone no longer fails C21
        set_column(session, "hf", "fps", lambda key, v: "25.000")
        code, rep = self.check(session)
        self.assertEqual((code, by_id(rep)["C21"]["status"], by_id(rep)["C21"]["value"]["hf_fps_p5"]),
                         (0, "PASS", 25.0))

    def test_img_path_outside_the_session_fails_c11(self):                    # F-7
        session = self.golden()
        set_column(session, "img", "path", lambda key, v: "../elsewhere/frames/f_00000007" if key == 7 else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C10", "C11"]))        # 1 of 68 slots unread: over 1% at T0
        self.assertIn("not frames/f_<slot_id:08d>", by_id(rep)["C11"]["detail"])

    def test_vehicle_card_hash_mismatch_fails_c02(self):                      # F-10
        bad = os.path.join(self.tmp, "card.json")
        with open(CARD, "rb") as f:
            card = json.loads(f.read().decode("ascii"))
        card.update({"gear_indices_forward": [], "gear_index_reverse": 3})
        with open(bad, "w", encoding="ascii") as f:
            json.dump(card, f)
        code, rep = run_json("--session", GOLDEN_DIR, "--routes-dir", ROUTES, "--vehicle-card", bad)
        self.assertEqual((code, failing(rep)), (1, ["C02"]))
        self.assertIn("vehicle_card", by_id(rep)["C02"]["detail"])

    def test_use_class_and_channel_rules(self):                               # F-11 (G19 refusals included)
        cases = ((("experimental", "dev"), 1), (("experimental", "compat-preview"), 0), (("stable", "production"), 1))
        for (channel, use_class), want in cases:
            session = self.golden("%s_%s" % (channel, use_class))

            def stamp(o):
                o["pin"]["channel"] = channel
                o["use_class"] = use_class

            edit_json(os.path.join(session, "session.json"), stamp)
            code, rep = self.check(session)
            self.assertEqual(code, want, (channel, use_class, by_id(rep)["C02"]["detail"]))
            if want:
                self.assertEqual(failing(rep), ["C02"])

    def test_launch_record_missing_is_a_warning(self):                        # F-11
        session = self.golden()
        shutil.rmtree(os.path.join(session, "_launch"))
        code, rep = self.check(session)
        self.assertEqual((code, by_id(rep)["C02"]["status"]), (2, "NOT_EVALUATED"))
        self.assertIn("no launch record", " ".join(by_id(rep)["C02"]["not_evaluated"]))

    def test_launch_record_contents(self):                                    # F-11
        def record(session):
            return glob.glob(os.path.join(session, "_launch", "launch_*.json"))[0]

        edits = {"check_missing": lambda o: o["checks"].pop(5),
                 "check_failed": lambda o: o["checks"][3].update({"ok": False}),
                 "exe": lambda o: o.update({"exe": "client"}),
                 "buildid": lambda o: o["observed"]["game"].update({"buildid": 1})}
        for name, fn in sorted(edits.items()):
            session = self.golden(name)
            edit_json(record(session), fn)
            code, rep = self.check(session)
            self.assertEqual((code, failing(rep)), (1, ["C02"]), name)
            self.assertIn("launch_record", by_id(rep)["C02"]["detail"], name)

    def test_thresholds_follow_the_session_pin(self):                         # F-11
        session = self.golden()
        for path in (os.path.join(session, "session.json"), os.path.join(session, "session_end.json")):
            edit_json(path, lambda o: o["pin"].update({"pin_id": "1.8.0.13-cold"}))
        edit_json(glob.glob(os.path.join(session, "_launch", "launch_*.json"))[0],
                  lambda o: o["pin"].update({"pin_id": "1.8.0.13-cold"}))
        code, rep = self.check(session)
        self.assertEqual((code, rep["thresholds_file"]), (0, "thresholds_dc2.1.8.0.13.json"))
        session = self.golden("p19")
        edit_json(os.path.join(session, "session.json"), lambda o: o["pin"].update({"pin_id": "1.9.0.10"}))
        code, text = run("--session", session, "--no-report")
        self.assertEqual(code, 3, text)
        self.assertIn("no thresholds file for pin 1.9.0.10", text)

    def test_overflow_and_duplicates_and_empty_chunks(self):                  # F-12, F-9, F-19
        session = self.golden("inf")
        set_column(session, "telemetry", "pos_x", lambda key, v: "9" * 400 if key == 30 else v)
        code, rep = self.check(session)
        self.assertEqual(code, 1)
        self.assertIn("C05", failing(rep))
        session = self.golden("dup")
        rewrite_chunk(session, "ctl", 0, lambda cols, rows: rows[:150] + [list(rows[149])] + rows[150:])
        code, rep = self.check(session)
        self.assertIn("C06", failing(rep))
        self.assertIn("C07", failing(rep))
        session = self.golden("empty")
        header = read_lines(os.path.join(session, "episodes", "chunk_00001.csv"))[0] + "\n"
        with open(os.path.join(session, "episodes", "chunk_00002.csv"), "wb") as f:
            f.write(header.encode("ascii"))
        with open(os.path.join(session, "episodes", "commit.csv"), "ab") as f:
            f.write(("2,1,1,0,%d\n" % len(header)).encode("ascii"))
        code, rep = self.check(session)
        self.assertIn("C34", failing(rep))

    def test_negative_capture_offset_fails_c13a(self):                        # F-21
        session = self.golden()
        for column in ("issue_render_frame", "done_render_frame"):           # the request issued after the capture
            set_column(session, "img", column, lambda key, v: str(int(v) + 2) if key == 50 else v)
        code, rep = self.check(session)
        # C21 too: the excluded row (done) is no longer the frame after the decoded capture
        self.assertEqual((code, failing(rep)), (1, ["C13a", "C21"]))
        self.assertIn("captured before their request", by_id(rep)["C13a"]["detail"])

    def test_lower_trial_is_refused(self):                                    # F-23
        session = self.golden()
        edit_json(os.path.join(session, "session.json"), lambda o: o["trial"].update({"trial_id": "T1"}))
        code, text = run("--session", session, "--no-report", "--trial", "T0b")
        self.assertEqual(code, 3, text)
        self.assertIn("lower tier", text)

    def test_everon_side_doors_are_closed(self):                              # F-16
        legacy = synth.write_legacy_session(os.path.join(self.tmp, "heldout"))
        code, text = run("--session", legacy, "--legacy", "--no-report")
        self.assertEqual(code, 3, text)
        with open(os.path.join(ROUTES, "golden_s_curve_v1.json"), "rb") as f:
            route = json.loads(f.read().decode("ascii"))
        route.update({"map_id": "everon", "split": "everon_test"})
        routes = os.path.join(self.tmp, "everon_routes")
        os.makedirs(routes)
        with open(os.path.join(routes, "golden_s_curve_v1.json"), "w", encoding="ascii") as f:
            json.dump(route, f)
        code, text = run("--session", GOLDEN_DIR, "--no-report", "--route",
                         os.path.join(routes, "golden_s_curve_v1.json"))
        self.assertEqual(code, 3, text)
        code, text = run("--session", GOLDEN_DIR, "--no-report", "--routes-dir", routes)
        self.assertEqual(code, 3, text)                                     # never "split skipped"
        self.assertIn("sealed", text)
        session = self.golden()
        edit_json(os.path.join(session, "session.json"),
                  lambda o: o["map"].update({"root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"}))
        code, text = run("--session", session, "--no-report")
        self.assertEqual(code, 3, text)
        self.assertIn("Eden-rooted", text)


def render_slot(session, slot, captured):
    """A slot's file re-encoded to show render frame `captured` (fiducial v2), drawn with the golden's last state."""
    spec = dict(synth.SPECS["golden"])
    sim = synth.simulate(spec, synth.drivedit_nav.route_from_dict(synth.ROUTES[spec["route"]]()))
    state = sim.frames[min(captured + 1 - spec["render_frame0"], len(sim.frames) - 1)]["state"]
    with open(os.path.join(session, "frames", "f_%08d.png" % slot), "wb") as f:
        f.write(synth.png_bytes(synth.render_image(spec, state, synth.fiducial.code_for(captured))))


# C21 fixtures (decision 2026-11-04). RUN_FRAMES follows probe runs 3-4 at 10 Hz on the runway: ordinary frames p5
# 7.4 / p50 12.6 / p95 15.9 ms, capture stalls p5 67 / p50 80 / p95 91 / max 100 ms (run 3 e05_deep; run 4 per-frame
# FPS p5 11.7 over all frames, 66.0 without the done rows, 33% of the rows excluded). SLOW_FRAMES: ordinary frames of
# 36-45 ms (22-28 FPS).
RUN_FRAMES = {"frame_ms": (12, 13, 8, 15, 12, 14, 11, 16, 13, 10, 12, 7, 13, 14, 12, 16, 11, 13, 9, 12),
              "stall_ms": (80, 67, 91, 78, 85, 72, 100, 81, 76, 88, 80, 70)}
SLOW_FRAMES = {"frame_ms": (40, 45, 38, 42, 36)}


class C21Test(unittest.TestCase):
    """C21 per-frame FPS over the non-screenshot frames (the user's decision of 2026-11-04): 1 / frame time from
    hf.world_time_ms, the rows after a decoded capture (done = C + 1) excluded, and the review of fa37e93
    (t0/freeze/review/CODEX_C21.md): no mutation turns a FAIL into a PASS; missing or contradictory evidence gives
    NOT_EVALUATED (a pass needs the evidence) or FAIL. A world-time FPS can only overstate the real one, so an FPS
    failure is final."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temp_dir("dc2_c21_")
        cls.routes = os.path.join(cls.tmp, "r")

        def make(name, kind, extra):
            with mock.patch.dict(synth.SPECS[kind], extra):
                return synth.generate(os.path.join(cls.tmp, name), kind=kind, routes_dir=cls.routes)

        cls.run_like = make("run", "golden_10hz", RUN_FRAMES)
        cls.run_long = make("runlong", "t0_loop", RUN_FRAMES)          # 108 s: a heartbeat span that bounds world/real
        cls.slow45 = make("slow45", "golden", {"frame_ms": (45,) + (25,) * 9})        # Codex: 22.22 FPS
        cls.slow50 = make("slow50", "golden", {"frame_ms": (10, 50) + (20,) * 8})     # Codex: 20 FPS

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def check(self, session, *extra):
        return run_json("--session", session, "--routes-dir", self.routes, "--vehicle-card", CARD, *extra)

    def c21(self, session, *extra):
        code, rep = self.check(session, *extra)
        return by_id(rep)["C21"]

    def copy(self, out, name):
        dst = os.path.join(self.tmp, "m_" + name, out["session_id"])
        shutil.copytree(out["session_dir"], dst)
        return dst

    def golden(self, name):
        return copy_golden(os.path.join(self.tmp, name))

    @staticmethod
    def hf_rows(session):
        rows = {}
        for name in sorted(os.listdir(os.path.join(session, "hf"))):
            if name.startswith("chunk_"):
                lines = read_lines(os.path.join(session, "hf", name))
                cols = lines[0].split(",")
                for line in lines[1:]:
                    vals = line.split(",")
                    rows[int(vals[0])] = dict(zip(cols, vals))
        return rows

    @staticmethod
    def column(session, stream, column):
        out = {}
        for name in sorted(os.listdir(os.path.join(session, stream))):
            if name.startswith("chunk_"):
                lines = read_lines(os.path.join(session, stream, name))
                cols = lines[0].split(",")
                out.update((int(x.split(",")[0]), x.split(",")[cols.index(column)]) for x in lines[1:])
        return out

    def slow_rows(self, session, ms):
        """hf render frames whose world delta is ms and that are not after a capture (done frames)."""
        rows = self.hf_rows(session)
        done = set(int(v) for v in self.column(session, "img", "done_render_frame").values() if v)
        return sorted(f for f in rows if f - 1 in rows and f not in done and
                      int(rows[f]["world_time_ms"]) - int(rows[f - 1]["world_time_ms"]) == ms)

    @staticmethod
    def edit_heartbeat(session, fn):
        path = os.path.join(session, "heartbeat.csv")
        lines = read_lines(path)
        cols = lines[0].split(",")
        rows = [dict(zip(cols, line.split(","))) for line in lines[1:]]
        rows = fn(rows)
        with open(path, "wb") as f:
            f.write(("\n".join([lines[0]] + [",".join(r[c] for c in cols) for r in rows]) + "\n").encode("ascii"))

    # --- valid sessions -------------------------------------------------------------------------------------------
    def test_golden(self):
        """13 s: PASS at T0 (p5 37.0, all 12.5, 68 excluded); from T1 a 5 s heartbeat span cannot bound world/real
        within 2% (unix_s whole seconds), so the FPS pass is NOT_EVALUATED there (the long variant below passes)."""
        c21 = self.c21(GOLDEN_DIR)
        self.assertEqual((c21["status"], c21["value"]["excluded_rows"], c21["value"]["samples"]), ("PASS", 68, 323))
        self.assertAlmostEqual(c21["value"]["fps_p5"], 1000.0 / 27, places=3)
        self.assertAlmostEqual(c21["value"]["fps_p5_all"], 12.5)
        c21 = self.c21(GOLDEN_DIR, "--trial", "T1")
        self.assertEqual((c21["status"], c21["severity"]), ("NOT_EVALUATED", "HARD"))
        self.assertIn("not bounded within 1 +- 0.02", c21["detail"])

    def test_run_shaped_sessions_pass(self):
        """The run 3-4 shape (frames 7-16 ms, every capture's done row 67-100 ms): 10 Hz at T0 (38% excluded) and the
        108 s variant at T1, where the heartbeat span bounds world/real."""
        c21 = self.c21(self.run_like["session_dir"])
        self.assertEqual(c21["status"], "PASS")
        self.assertGreaterEqual(c21["value"]["fps_p5"], 60.0)
        self.assertLess(c21["value"]["fps_p5_all"], 12.0)
        self.assertGreater(c21["value"]["excluded_rows"] / float(c21["value"]["frames"]), 0.3)
        c21 = self.c21(self.run_long["session_dir"], "--trial", "T1")
        self.assertEqual((c21["status"], c21["severity"]), ("PASS", "HARD"))
        self.assertGreaterEqual(c21["value"]["fps_p5"], 60.0)
        lo, hi = c21["value"]["world_real_bounds"]
        self.assertTrue(0.98 <= lo and hi <= 1.02, (lo, hi))

    def test_slow_ordinary_frames_fail(self):
        c21 = self.c21(self.slow45["session_dir"])
        self.assertEqual((c21["status"], c21["severity"]), ("FAIL", "SOFT"))
        self.assertAlmostEqual(c21["value"]["fps_p5"], 1000.0 / 45, places=3)
        self.assertIn("non-screenshot frames < 30", c21["detail"])
        c21 = self.c21(self.slow45["session_dir"], "--trial", "T1")
        self.assertEqual((c21["status"], c21["severity"]), ("FAIL", "HARD"))

    # --- review item 1: pause flags ---------------------------------------------------------------------------------
    def test_pause_flags_do_not_hide_slow_frames(self):
        """Codex: paused = 1 on the 31 slow ordinary rows (no pause events, clocks unchanged) gave 40 FPS and a full
        PASS at T0 and T1; all rows but two paused gave a T1 PASS from one sample. World time does not advance
        while paused, so a row whose world time advances is a sample whatever its flag."""
        slow = set(self.slow_rows(self.slow45["session_dir"], 45))
        self.assertEqual(len(slow), 31)
        session = self.copy(self.slow45, "paused")
        set_column(session, "hf", "paused", lambda key, v: "1" if key in slow else v)
        for extra in ((), ("--trial", "T1")):
            c21 = self.c21(session, *extra)
            self.assertEqual(c21["status"], "FAIL", extra)
            self.assertAlmostEqual(c21["value"]["fps_p5"], 1000.0 / 45, places=3)
            self.assertEqual(c21["value"]["paused_rows_advancing"], 31)
        session = self.copy(self.slow45, "allpaused")
        first_two = sorted(self.hf_rows(session))[:2]
        set_column(session, "hf", "paused", lambda key, v: v if key in first_two else "1")
        c21 = self.c21(session, "--trial", "T1")
        self.assertEqual(c21["status"], "FAIL")
        self.assertAlmostEqual(c21["value"]["fps_p5"], 1000.0 / 45, places=3)
        self.assertGreater(c21["value"]["samples"], 200)

    def test_too_few_samples_is_not_evaluated(self):
        """The sample floor (max(30, 5 per world second)): a pass needs the evidence, a failure does not."""
        health = dc_validate.checks.health
        with mock.patch.object(health, "FPS_SAMPLES_PER_S", 1000):
            c21 = self.c21(GOLDEN_DIR)
            self.assertEqual(c21["status"], "NOT_EVALUATED")
            self.assertIn("frame-time samples, at least", c21["detail"])
            self.assertEqual(self.c21(self.slow45["session_dir"])["status"], "FAIL")

    # --- review item 2: redistributed world times between callbacks (the documented limit) --------------------------
    def test_known_limit_world_times_between_callbacks(self):
        """Codex: frames (10, 50, 20 x 8) ms fail at 20 FPS; moving 25 world times of frames without a callback,
        heartbeat or image request so that 10/50 ms pairs read 30/30 ms passes at 33.3 FPS. No sound cross-check
        exists (hf.fps is a smoothed rate that sums over such frames; the fixed-step accumulator lags world time by
        up to 250-310 ms while capturing): C21 trusts the writer's hf.world_time_ms between callbacks, as its
        docstring states. This test pins the limit, not a wish."""
        out = self.slow50
        self.assertAlmostEqual(self.c21(out["session_dir"])["value"]["fps_p5"], 20.0)
        session = self.copy(out, "reshape")
        rows = self.hf_rows(session)
        callbacks = set(int(v) for v in self.column(session, "ctl", "render_frame").values())
        issued = set(int(v) for v in self.column(session, "img", "issue_render_frame").values() if v)
        beats = set()
        self.edit_heartbeat(session, lambda hb: [beats.add(int(r["render_frame"])) or r for r in hb])
        moved = []
        for f in sorted(rows):
            if len(moved) == 25:
                break
            if f - 1 not in rows or f + 1 not in rows or f in callbacks | issued | beats:
                continue
            w0, w1, w2 = (int(rows[g]["world_time_ms"]) for g in (f - 1, f, f + 1))
            if (w1 - w0, w2 - w1) == (10, 50):
                moved.append(f)
        self.assertEqual(len(moved), 25)
        set_column(session, "hf", "world_time_ms", lambda key, v: str(int(v) + 20) if key in moved else v)
        c21 = self.c21(session)
        self.assertEqual(c21["status"], "PASS")
        self.assertAlmostEqual(c21["value"]["fps_p5"], 1000.0 / 30, places=3)
        self.assertIn("trusts the writer's hf.world_time_ms between callbacks",
                      " ".join(dc_validate.checks.health.check_c21.__doc__.split()))

    # --- review items 3 and 4: heartbeat integrity -------------------------------------------------------------------
    def test_heartbeat_wall_time_cannot_erase_the_memory_check(self):
        """Codex: +1000 KiB on the golden's second heartbeat fails T1 at 710 MB/h; equalising the two unix_s then
        nulled ratio and slope and passed T1. World time cannot outrun wall time: FAIL."""
        session = self.golden("mem")
        self.edit_heartbeat(session, lambda hb: hb[:1] + [dict(hb[1], mem_kb=str(int(hb[1]["mem_kb"]) + 1000))])
        c21 = self.c21(session, "--trial", "T1")
        self.assertEqual(c21["status"], "FAIL")
        self.assertIn("memory slope", c21["detail"])
        self.assertAlmostEqual(c21["value"]["mem_slope_mb_per_h"], 710.16, places=1)
        self.edit_heartbeat(session, lambda hb: hb[:1] + [dict(hb[1], unix_s=hb[0]["unix_s"])])
        c21 = self.c21(session, "--trial", "T1")
        self.assertEqual(c21["status"], "FAIL")
        self.assertIn("world time outran wall time", c21["detail"])

    def test_a_removed_heartbeat_fails(self):
        """Codex: the second heartbeat removed and the seal count set to 1 passed T1 (the 2-period gap rule allows one
        missing row). The writer's schedule puts one row in the first frame past every 5 s boundary: FAIL."""
        session = self.golden("hb1")
        self.edit_heartbeat(session, lambda hb: hb[:1])
        edit_json(os.path.join(session, "session_end.json"), lambda o: o.update({"heartbeats": 1}))
        for extra in ((), ("--trial", "T1")):
            c21 = self.c21(session, *extra)
            self.assertEqual(c21["status"], "FAIL", extra)
            self.assertIn("do not follow the 5 s world-time schedule", c21["detail"])

    def test_the_final_heartbeat_cannot_be_dropped(self):
        """Codex (CODEX_C21b, CODEX_C21c): t0_loop with tail_ms 4714 ends at 115017 ms in frame 43292 with the 23rd
        heartbeat in its last frame (duration, status ok). +100000 KiB on that row fails T1 at 771.3 MB/h; removing it
        with the seal count set to 22 passed, and with the exemption by end reason it still passed after a destroy
        event in frame 43293 sealed with ego_destroyed. No status-ok path of the writer writes a frame's hf row and
        then seals before that frame's heartbeat (a destroyed ego is seen in OnFrameStart, before the hf row, or in a
        callback, after the heartbeat), so there is no exemption: FAIL for every end reason."""
        with mock.patch.dict(synth.SPECS["t0_loop"], {"tail_ms": 4714}):
            out = synth.generate(os.path.join(self.tmp, "tail4714"), kind="t0_loop", routes_dir=self.routes)
        with open(os.path.join(out["session_dir"], "session_end.json"), "rb") as f:
            end = json.loads(f.read().decode("ascii"))
        self.assertEqual((end["end"]["world_ms"], end["end"]["render_frame"], end["end_reason"], end["heartbeats"]),
                         (115017, 43292, "duration", 23))
        self.assertEqual(self.c21(out["session_dir"], "--trial", "T1")["status"], "PASS")
        session = self.copy(out, "memlast")
        self.edit_heartbeat(session, lambda hb: hb[:-1] + [dict(hb[-1], mem_kb=str(int(hb[-1]["mem_kb"]) + 100000))])
        c21 = self.c21(session, "--trial", "T1")
        self.assertEqual((c21["status"], c21["severity"]), ("FAIL", "HARD"))
        self.assertAlmostEqual(c21["value"]["mem_slope_mb_per_h"], 771.30, places=1)
        self.edit_heartbeat(session, lambda hb: hb[:-1])
        edit_json(os.path.join(session, "session_end.json"), lambda o: o.update({"heartbeats": 22}))
        c21 = self.c21(session, "--trial", "T1")
        self.assertEqual((c21["status"], c21["severity"]), ("FAIL", "HARD"))
        self.assertIn("the writer writes every scheduled row, the last frame's included", c21["detail"])
        for reason, status in (("stop", "ok"), ("dc_stop", "ok"), ("ego_disabled", "ok"),
                               ("script_error:other", "failed"), ("camera_lost", "failed")):
            edit_json(os.path.join(session, "session_end.json"),
                      lambda o, reason=reason, status=status: o.update({"end_reason": reason, "status": status}))
            self.assertEqual(self.c21(session, "--trial", "T1", "--checks", "C21")["status"], "FAIL", reason)
        # Codex's ego_destroyed case: the ego's veh_damage state=destroyed in the next frame 43293 (115042 ms), the
        # seal ego_destroyed, status ok, end.render_frame the last committed hf frame 43292
        session = self.copy(out, "destroyed")
        sdir = os.path.join(session, "events")
        names = sorted(n for n in os.listdir(sdir) if n.startswith("chunk_"))

        def destroy(cols, rows):
            new = list(rows[-1])
            new[cols.index("event_seq")] = str(int(rows[-1][cols.index("event_seq")]) + 1)
            for col, value in (("kind", "veh_damage"), ("payload", "state=destroyed"), ("episode_id", "-1"),
                               ("render_frame", "43293"), ("world_time_ms", "115042")):
                new[cols.index(col)] = value
            return rows + [new]

        rewrite_chunk(session, "events", int(names[-1][6:11]), destroy)
        self.edit_heartbeat(session, lambda hb: hb[:-1])

        def seal(o):
            o.update({"end_reason": "ego_destroyed", "status": "ok", "heartbeats": 22})
            o["end"]["world_ms"] = 115042
            o["counts"]["events"].update({"rows": o["counts"]["events"]["rows"] + 1,
                                          "last_key": o["counts"]["events"]["last_key"] + 1})

        edit_json(os.path.join(session, "session_end.json"), seal)
        code, rep = self.check(session, "--trial", "T1")
        c21 = by_id(rep)["C21"]
        self.assertEqual((code, c21["status"], c21["severity"]), (1, "FAIL", "HARD"))
        self.assertIn("do not follow the 5 s world-time schedule", c21["detail"])

    def test_crash_fixtures_write_the_last_frames_heartbeat(self):
        """The generator follows the writer's order: a crash seals in a callback (after its frame's heartbeat), so
        the last hf frame has its scheduled row; every crash variant passes C21 without any exemption (the two
        negative ones fail C36 only)."""
        for variant in ("crash", "crash_unexplained", "destroyed_unexplained"):
            out = synth.generate(os.path.join(self.tmp, "cr_" + variant), variant=variant, routes_dir=self.routes)
            s = reader.open_session(out["session_dir"], vehicle_card=CARD)
            hb = [(r["render_frame"], r["world_ms"]) for r in s.heartbeat]
            hf = list(zip(s.streams["hf"].col("render_frame"), s.streams["hf"].col("world_time_ms")))
            expected, boundary = [], 5000
            for frame, w in hf:
                if w >= boundary:
                    expected.append((frame, w))
                    boundary = w - w % 5000 + 5000
            self.assertEqual(hb, expected, variant)
            self.assertEqual(self.c21(out["session_dir"])["status"], "PASS", variant)

    def test_a_contradictory_heartbeat_world_time_fails(self):
        """Codex: on the 22.22 FPS fixture the second heartbeat's world_ms 10015 -> 7000 (frame and hf unchanged)
        turned the FPS failure into a T1 NOT_EVALUATED. A heartbeat must carry its frame's world time: FAIL."""
        session = self.copy(self.slow45, "hbworld")
        self.edit_heartbeat(session, lambda hb: hb[:1] + [dict(hb[1], world_ms="7000")] + hb[2:])
        c21 = self.c21(session, "--trial", "T1")
        self.assertEqual((c21["status"], c21["severity"]), ("FAIL", "HARD"))
        self.assertIn("do not follow the 5 s world-time schedule", c21["detail"])

    # --- review item 5: exclusions need decoded evidence --------------------------------------------------------------
    def test_exclusions_need_decoded_captures(self):
        """Codex: 31 bogus issued slots, no image files, whose done frames are the slow ordinary rows: C21 excluded
        them (40 FPS, PASS alone). A row is excluded only after a decoded capture (done = C + 1)."""
        session = self.copy(self.slow45, "bogus")
        slow = self.slow_rows(session, 45)
        names = sorted(n for n in os.listdir(os.path.join(session, "img")) if n.startswith("chunk_"))

        def add(cols, rows):
            last = rows[-1]
            for k, frame in enumerate(slow):
                r = list(last)
                r[cols.index("slot_id")] = str(int(last[cols.index("slot_id")]) + 1 + k)
                r[cols.index("issue_status")] = "issued"
                r[cols.index("issue_render_frame")] = str(frame - 1)
                r[cols.index("done_render_frame")] = str(frame)
                r[cols.index("path")] = "frames/f_%08d" % int(r[cols.index("slot_id")])
                rows.append(r)
            return rows

        rewrite_chunk(session, "img", int(names[-1][6:11]), add)
        code, rep = run_json("--session", session, "--routes-dir", self.routes, "--vehicle-card", CARD,
                             "--checks", "C21")
        c21 = by_id(rep)["C21"]
        self.assertEqual((code, c21["status"]), (2, "FAIL"))
        self.assertAlmostEqual(c21["value"]["fps_p5"], 1000.0 / 45, places=3)

    def test_excluded_row_not_after_its_capture_fails(self):
        """A decoded slot whose done is not C + 1, and two completions in one frame, fail."""
        session = self.golden("off")
        set_column(session, "img", "done_render_frame", lambda key, v: str(int(v) + 2) if key == 10 else v)
        code, rep = self.check(session)
        self.assertIn("C21", failing(rep))
        self.assertIn("not the frame after their decoded capture", by_id(rep)["C21"]["detail"])
        session = self.golden("dup")
        done = self.column(GOLDEN_DIR, "img", "done_render_frame")
        set_column(session, "img", "done_render_frame", lambda key, v: done[9] if key == 10 else v)
        code, rep = self.check(session)
        self.assertIn("C21", failing(rep))
        self.assertIn("share a done_render_frame", by_id(rep)["C21"]["detail"])

    # --- review item 6: whole-second unix_s ---------------------------------------------------------------------------
    def test_world_real_needs_a_bounding_heartbeat_span(self):
        """Codex: +1 s on the golden's last heartbeat gives ratio 0.837 with bounds up to 1.0046: at T0 not certainly
        outside 1 +- 2% (PASS, a short shakedown session), from T1 not bounded (NOT_EVALUATED); +2 s is certainly
        outside: NOT_EVALUATED at every tier. A slow session still fails."""
        for plus, t0, t1 in ((1, "PASS", "NOT_EVALUATED"), (2, "NOT_EVALUATED", "NOT_EVALUATED")):
            session = self.golden("wr%d" % plus)
            self.edit_heartbeat(session, lambda hb: hb[:-1] + [dict(hb[-1], unix_s=str(int(hb[-1]["unix_s"]) + plus))])
            c21 = self.c21(session)
            self.assertEqual(c21["status"], t0, plus)
            self.assertEqual(self.c21(session, "--trial", "T1")["status"], t1, plus)
            self.assertAlmostEqual(c21["value"]["world_real"], 1.0046 * 5 / (5 + plus), places=3)
        session = self.copy(self.slow45, "wrslow")
        self.edit_heartbeat(session, lambda hb: hb[:-1] + [dict(hb[-1], unix_s=str(int(hb[-1]["unix_s"]) + 2))])
        self.assertEqual(self.c21(session)["status"], "FAIL")


class SlowCaptureTest(unittest.TestCase):
    """Review round 2, N2: a 10 Hz session whose captures mix 80 and 90 ms stalls (u95 0.9 > 0.85). Dropping the
    done_render_frame of the slow captures must not turn C37 into a pass."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temp_dir("dc2_slow_")
        cls.routes = os.path.join(cls.tmp, "r")
        with mock.patch.dict(synth.SPECS["golden_10hz"], {"stall_ms": (80, 80, 90)}):
            cls.out = synth.generate(os.path.join(cls.tmp, "src"), kind="golden_10hz", routes_dir=cls.routes)
        s = reader.open_session(cls.out["session_dir"], vehicle_card=CARD)
        hf = dict(zip(s.streams["hf"].col("render_frame"), s.streams["hf"].col("world_time_ms")))
        cls.slow = sorted(r["slot_id"] for r in s.images if r["fid_ok"] and
                          hf[r["captured_render_frame"] + 1] - hf[r["captured_render_frame"]] == 90)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def check(self, session):
        return run_json("--session", session, "--routes-dir", self.routes, "--vehicle-card", CARD)

    def copy(self, name):
        dst = os.path.join(self.tmp, name, self.out["session_id"])
        shutil.copytree(self.out["session_dir"], dst)
        return dst

    def test_slow_captures_fail_c37(self):
        self.assertGreater(len(self.slow), 30)
        code, rep = self.check(self.out["session_dir"])
        self.assertEqual((code, failing(rep)), (2, ["C37"]))
        self.assertAlmostEqual(by_id(rep)["C37"]["value"]["u95"], 0.9)

    def test_dropping_the_slow_completions_does_not_pass(self):
        session = self.copy("drop")
        slow = set(self.slow)
        set_column(session, "img", "done_render_frame", lambda key, v: "" if key in slow else v)
        code, rep = self.check(session)
        # C21 too: without done_render_frame the slow rows are not excluded (C21 decision 2026-11-04)
        self.assertEqual((code, failing(rep)), (1, ["C13a", "C21", "C37"]))
        self.assertIn("a completion the writer dropped", by_id(rep)["C13a"]["detail"])
        self.assertEqual(by_id(rep)["C13a"]["value"]["dropped_completions"], len(slow))
        self.assertAlmostEqual(by_id(rep)["C37"]["value"]["u95"], 0.9)

    def test_twenty_declared_pending_captures_do_not_pass(self):              # review round 3, N6
        """Codex: a 10 Hz session whose last 20 captures stall 90 ms; their completions cleared, images.pending_at_end
        20 and their fiducials re-encoded to the terminal frame (cores unchanged). Before: full PASS with 20
        exemptions and u95 0.8 over the rest. Now C13a fails (at most 1 pending, no exemption) and C37 counts the 20
        as unmeasured at the worst case."""
        spec = dict(synth.SPECS["golden_10hz"])
        spec["stall_ms"] = (80,) * 112 + (90,) * 20
        with mock.patch.dict(synth.SPECS["golden_10hz"], {"stall_ms": spec["stall_ms"]}):
            out = synth.generate(os.path.join(self.tmp, "twenty_src"), kind="golden_10hz", routes_dir=self.routes)
        code, rep = self.check(out["session_dir"])
        self.assertEqual((code, failing(rep)), (2, ["C37"]))
        self.assertAlmostEqual(by_id(rep)["C37"]["value"]["u95"], 0.9)
        session = os.path.join(self.tmp, "twenty", out["session_id"])
        shutil.copytree(out["session_dir"], session)
        sim = synth.simulate(spec, synth.drivedit_nav.route_from_dict(synth.ROUTES[spec["route"]]()))
        issued = [sl for sl in sim.slots if sl["issue_frame"] is not None]
        self.assertEqual(len(issued), 132)
        with open(os.path.join(session, "session_end.json"), "rb") as f:
            end_frame = json.loads(f.read().decode("ascii"))["end"]["render_frame"]
        last20 = set(sl["slot_id"] for sl in issued[-20:])
        for sl in issued[-20:]:
            state = sim.frames[sl["issue_frame"] + 1]["state"]          # the original core, a new code
            with open(os.path.join(session, "frames", "f_%08d.png" % sl["slot_id"]), "wb") as f:
                f.write(synth.png_bytes(synth.render_image(spec, state, synth.fiducial.code_for(end_frame))))
        set_column(session, "img", "done_render_frame", lambda key, v: "" if key in last20 else v)
        edit_json(os.path.join(session, "session_end.json"), lambda o: o["images"].update({"pending_at_end": 20}))
        code, rep = self.check(session)
        self.assertEqual(code, 1)
        self.assertIn("C13a", failing(rep))
        self.assertIn("C37", failing(rep))
        c13a, c37 = by_id(rep)["C13a"], by_id(rep)["C37"]
        self.assertEqual(c13a["value"]["pending_after_seal"], 0)
        self.assertIn("at most 1 can be pending", c13a["detail"])
        self.assertEqual(c37["value"]["unmeasured"], 20)

    def test_unmeasured_slow_captures_cannot_pass(self):
        session = self.copy("gone")              # the slow captures' files removed: assumed at the worst case
        for slot in self.slow:
            os.remove(os.path.join(session, "frames", "f_%08d.png" % slot))
        code, rep = self.check(session)
        c37 = by_id(rep)["C37"]
        self.assertEqual((c37["status"], c37["value"]["unmeasured"]), ("FAIL", len(self.slow)))
        self.assertLessEqual(c37["value"]["u95"], 0.85)
        self.assertIn("could not be measured", c37["detail"])


def edit_rows(session, stream, fn):
    """fn(columns, row as a list of text) edits each row of every chunk of a stream in place; commit lines stay true."""
    for name in sorted(os.listdir(os.path.join(session, stream))):
        if name.startswith("chunk_"):
            def edit(cols, rows):
                for r in rows:
                    fn(cols, r)
                return rows
            rewrite_chunk(session, stream, int(name[6:11]), edit)


class CrashTest(unittest.TestCase):
    """C36, C35 and C15 on the crash variant of make_synthetic (contract 9.5.1)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temp_dir("dc2_crash_")
        cls.out = synth.generate(os.path.join(cls.tmp, "src"), variant="crash")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def copy(self, name):
        dst = os.path.join(self.tmp, name, self.out["session_id"])
        shutil.copytree(self.out["session_dir"], dst)
        return dst

    def check(self, session, *extra):
        return run_json("--session", session, "--routes-dir", ROUTES, "--vehicle-card", CARD, *extra)

    def payloads(self, session, kind, fn):
        """fn(payload text) -> text for every events row of a kind."""
        def edit(cols, r):
            if r[cols.index("kind")] == kind:
                r[cols.index("payload")] = fn(r[cols.index("payload")])
        edit_rows(session, "events", edit)

    def test_crash_session_passes(self):
        code, rep = self.check(self.out["session_dir"])
        self.assertEqual(code, 0, [(c["id"], c["detail"]) for c in rep["checks"] if c["status"] != "PASS"])
        c36 = by_id(rep)["C36"]
        self.assertEqual((c36["severity"], c36["status"]), ("HARD", "PASS"))
        self.assertEqual(c36["value"], {"crash_events": 4, "crash_terminations": 1, "terminal_ego_episodes": 1,
                                        "ego_destroyed": True})
        self.assertTrue(c36["provisional"])                    # TBD-10 while dc-2.0 is a DRAFT
        self.assertEqual(by_id(rep)["C01"]["status"], "PASS")  # ego_destroyed is a normal end
        # C-H1: the image the seal caught in flight is pending, not a join failure
        c13a = by_id(rep)["C13a"]
        self.assertEqual((c13a["status"], c13a["value"]["join_failures"], c13a["value"]["pending_after_seal"]),
                         ("PASS", 0, 1))

    def test_golden_has_no_crash(self):
        code, rep = run_json("--session", GOLDEN_DIR, "--routes-dir", ROUTES, "--checks", "C36")
        self.assertEqual((code, by_id(rep)["C36"]["value"]["crash_events"]), (0, 1))   # its teleport row

    def test_collision_below_the_threshold_fails(self):
        session = self.copy("dv")
        self.payloads(session, "collision", lambda v: re.sub(r";dv_mps=[0-9.]+", ";dv_mps=3.10", v))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("below events.collision_dv_mps", by_id(rep)["C36"]["detail"])
        session = self.copy("contact")                         # a contact-only row may lie below the threshold
        self.payloads(session, "collision", lambda v: re.sub(r";dv_mps=[0-9.]+", ";dv_mps=3.10", v).replace(
            "src=dv_contact", "src=contact"))
        code, rep = self.check(session, "--checks", "C36")
        self.assertEqual((code, by_id(rep)["C36"]["status"]), (0, "PASS"))   # veh_damage still explains the end

    def test_onset_after_the_row_fails(self):
        session = self.copy("onset")
        self.payloads(session, "collision", lambda v: "onset_step=99999;" + v.split(";", 1)[1])
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("after the row's physics_step", by_id(rep)["C36"]["detail"])

    def test_bad_payloads_fail(self):
        for name, fn in (("enum", lambda v: v.replace("other=rock", "other=moon")),
                         ("unknown", lambda v: v + ";colour=red"),
                         ("missing", lambda v: re.sub(r";v_mps=[0-9.]+", "", v)),
                         ("exponent", lambda v: re.sub(r";dv_mps=[0-9.]+", ";dv_mps=7e0", v))):
            session = self.copy("payload_" + name)
            self.payloads(session, "collision", fn)
            code, rep = self.check(session, "--checks", "C36")
            self.assertEqual((code, by_id(rep)["C36"]["status"]), (1, "FAIL"), name)

    def test_unpaired_fire_end_fails(self):
        session = self.copy("pair")
        add_event(session, 0, "veh_fire_end", "", 1, physics_step=0)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("veh_fire_end on slot 1 without a veh_fire_start", by_id(rep)["C36"]["detail"])

    def test_termination_must_match_the_first_terminal_event(self):
        session = self.copy("term")
        set_column(session, "episodes", "termination_reason", lambda key, v: "rollover" if key == 1 else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("termination_reason rollover, but its first terminal ego event", by_id(rep)["C36"]["detail"])

    def test_rows_after_the_destroyed_ego_fail(self):
        session = self.copy("late")

        def earlier(cols, r):                                  # the destroyed row 0.5 s before the last ctl row
            if r[cols.index("payload")].startswith("state=destroyed"):
                r[cols.index("physics_step")] = str(int(r[cols.index("physics_step")]) - 30)
        edit_rows(session, "events", earlier)
        code, rep = self.check(session, "--checks", "C36")
        self.assertEqual(by_id(rep)["C36"]["status"], "FAIL")
        self.assertIn("ctl row(s) after the ego's veh_damage destroyed row", by_id(rep)["C36"]["detail"])

    def test_destroyed_ego_with_a_planned_end_fails(self):
        session = self.copy("planned")
        edit_json(os.path.join(session, "session_end.json"), lambda o: o.update({"end_reason": "duration"}))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("ended duration, not ego_destroyed", by_id(rep)["C36"]["detail"])

    def test_unknown_vehicle_slots_fail_c35(self):
        session = self.copy("slot")
        self.payloads(session, "collision", lambda v: v.replace("other=rock", "other=vehicle;other_slot=7"))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C35"]))
        self.assertIn("unknown other_slot(s) [7]", by_id(rep)["C35"]["detail"])
        session = self.copy("veh")
        add_event(session, 0, "rollover", "onset_step=0;up_y=0.20", 9, physics_step=0)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C35"]))
        self.assertIn("unknown vehicle slot(s) [9]", by_id(rep)["C35"]["detail"])

    def test_an_onset_outside_its_episode_fails(self):
        session = self.copy("outside")
        self.payloads(session, "collision", lambda v: re.sub(r"^onset_step=[0-9]+;", "onset_step=-100;", v))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("its step -100 lies outside episode 1", by_id(rep)["C36"]["detail"])
        session = self.copy("between")                      # a row of no episode at a step inside episode 0
        s = dc_validate.base.load_reader().open_session(session, decode_images=False)
        start = s.streams["episodes"].col("start_physics_step")[0]
        add_event(session, 0, "veh_damage", "state=intermediary;health=0.80;movement_damage=0.10", 1,
                  physics_step=start + 30)
        code, rep = self.check(session, "--checks", "C36")
        self.assertEqual(by_id(rep)["C36"]["status"], "FAIL")
        self.assertIn("episode_id -1, but its step", by_id(rep)["C36"]["detail"])

    def test_payload_ranges_fail(self):
        session = self.copy("range")
        self.payloads(session, "veh_damage", lambda v: v.replace("health=0.00", "health=2.00"))
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C36"]))
        self.assertIn("outside [0, 1]", by_id(rep)["C36"]["detail"])

    def test_c15_masks_only_the_egos_events(self):
        dynamics = dc_validate.checks.dynamics
        s = dc_validate.base.load_reader().open_session(GOLDEN_DIR, strict=False, decode_images=False)
        base = len(dynamics.eligible_rows(s, 3, 0.8))
        ev, tel = s.streams["events"], s.streams["telemetry"]
        row = {c: None for c in ev.columns}
        row.update({"event_seq": len(ev), "episode_id": 0, "physics_step": 0, "render_frame": 0, "kind": "collision",
                    "world_time_ms": tel.col("world_time_ms")[len(tel) // 2], "veh_slot_id": 9, "payload": ""})
        ev.append_rows([row])
        self.assertEqual(len(dynamics.eligible_rows(s, 3, 0.8)), base)          # a traffic slot's crash
        ev.append_rows([dict(row, event_seq=len(ev), veh_slot_id=tel.col("ego_slot_id")[0])])
        self.assertLess(len(dynamics.eligible_rows(s, 3, 0.8)), base)           # the ego's crash

class CrashRulesOnGoldenTest(unittest.TestCase):
    """C36 and C35 rules that need an intact episode: the golden session with injected rows."""

    def setUp(self):
        self.tmp = temp_dir("dc2_crash_golden_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def golden_with(self, name, rows, termination=None, end_reason=None):
        """A golden copy with rows (anchor kind, kind, payload, physics_step) inserted after the anchor event (of
        episode 0, or the teleport between the episodes, which has none), with episode 0's termination and the
        session's end reason changed."""
        session = copy_golden(os.path.join(self.tmp, name))
        if termination:
            set_column(session, "episodes", "termination_reason", lambda key, v: termination if key == 0 else v)
        if end_reason:
            edit_json(os.path.join(session, "session_end.json"), lambda o: o.update({"end_reason": end_reason}))
        for anchor, kind, payload, step in rows:
            s = dc_validate.base.load_reader().open_session(session, decode_images=False)
            ev = s.streams["events"]
            i = [j for j, (k, ep) in enumerate(zip(ev.col("kind"), ev.col("episode_id")))
                 if k == anchor and (ep == 0 or anchor in ("warmup_end", "teleport"))][-1]
            add_event(session, ev.col("event_seq")[i], kind, payload, 1, physics_step=step)
        return session

    def episode0(self):
        s = dc_validate.base.load_reader().open_session(GOLDEN_DIR, decode_images=False)
        ep = s.streams["episodes"]
        return ep.col("start_physics_step")[0], ep.col("end_physics_step")[0]

    def c36(self, session):
        code, rep = run_json("--session", session, "--routes-dir", ROUTES, "--checks", "C35,C36")
        return by_id(rep)["C36"]

    def test_a_confirmed_rollover_is_the_only_rollover_termination(self):
        start, end = self.episode0()
        # the core's rollover: tilted at end - 40, confirmed (row) at end - 10, inside the episode
        ok = self.golden_with("confirmed", [("episode_start", "rollover", "onset_step=%d;up_y=0.31" % (end - 40),
                                             end - 10)], termination="rollover")
        self.assertEqual(self.c36(ok)["status"], "PASS", self.c36(ok))
        # an expert that ended at the first tilted step: the hold confirmed only after the episode ended
        late = self.golden_with("late", [("episode_end", "rollover", "onset_step=%d;up_y=0.31" % (end - 3),
                                          end + 30)], termination="rollover")
        self.assertIn("after the episode ended", self.c36(late)["detail"])
        # a tilt that righted itself before the hold: no rollover row, so a rollover termination is unexplained
        none = self.golden_with("unconfirmed", [], termination="rollover")
        self.assertIn("termination_reason rollover without a terminal ego crash event", self.c36(none)["detail"])
        mixed = self.golden_with("mixed", [("episode_start", "rollover", "onset_step=%d;up_y=0.31" % (end - 40),
                                            end - 10)], termination="collision")
        self.assertIn("termination_reason collision, but its first terminal ego event", self.c36(mixed)["detail"])

    def test_a_contact_reported_after_its_episode_ended(self):
        # episode 0 ends at step P; a vanilla contact reported before the next step carries P and episode 0
        start, end = self.episode0()
        payload = "onset_step=%d;dv_mps=1.10;v_mps=2.00;other=static;impulse_ns=2400.00;src=contact" % end
        owned = self.golden_with("owned", [("episode_end", "collision", payload, end)])
        self.assertEqual(self.c36(owned)["status"], "PASS", self.c36(owned))
        # the round-1 writer paired step P with the cleared episode (-1): rejected
        session = copy_golden(os.path.join(self.tmp, "cleared"))
        add_event(session, 0, "collision", payload, 1, physics_step=end)
        self.assertIn("episode_id -1, but its step %d lies inside an episode" % end, self.c36(session)["detail"])

    def test_a_rollover_confirmed_before_its_onset_fails(self):
        start, end = self.episode0()
        bad = self.golden_with("chrono", [("episode_start", "rollover", "onset_step=%d;up_y=0.31" % (start + 40),
                                           start + 20)], termination="rollover")
        detail = self.c36(bad)["detail"]
        self.assertIn("onset_step %d after the row's physics_step %d" % (start + 40, start + 20), detail)

    def test_ego_disabled_needs_a_disabled_ego(self):
        start, end = self.episode0()
        tail = end + 5                           # between episode 0 and 1: a step of no episode (episode_id -1)
        immobile = self.golden_with("immobile", [("teleport", "veh_damage",
                                                   "state=intermediary;health=0.40;movement_damage=1.00", tail)],
                                    end_reason="ego_disabled")
        self.assertEqual(self.c36(immobile)["status"], "PASS", self.c36(immobile))
        code, rep = run_json("--session", immobile, "--routes-dir", ROUTES, "--checks", "C01")
        self.assertEqual(by_id(rep)["C01"]["status"], "PASS")            # ego_disabled is a normal end
        plain = self.golden_with("plain", [], end_reason="ego_disabled")
        self.assertIn("end_reason ego_disabled, but the ego", self.c36(plain)["detail"])

    def test_traffic_slot_pairs_are_checked(self):
        for name, fn, word in (
                ("identity", lambda key, v: "9:0x00010000C1" if key < 300 else "9:0x00010000C2", "slot 9 carries 2"),
                ("overlap", lambda key, v: "1:0x00010000C1", "used by the ego and by traffic"),
                ("malformed", lambda key, v: "9", "malformed telemetry.traffic_slots")):
            session = copy_golden(os.path.join(self.tmp, name))
            set_column(session, "telemetry", "traffic_slots", fn)
            set_column(session, "telemetry", "traffic_n", lambda key, v: "1")
            code, rep = run_json("--session", session, "--routes-dir", ROUTES, "--checks", "C35")
            c35 = by_id(rep)["C35"]
            self.assertEqual(c35["status"], "FAIL", name)
            self.assertIn(word, c35["detail"], name)
        session = copy_golden(os.path.join(self.tmp, "fine"))
        set_column(session, "telemetry", "traffic_slots", lambda key, v: "9:0x00010000C1;12:0x00010000C2")
        code, rep = run_json("--session", session, "--routes-dir", ROUTES, "--checks", "C35")
        self.assertEqual(by_id(rep)["C35"]["status"], "PASS")


class ReportTest(unittest.TestCase):
    def test_default_report_root(self):
        with mock.patch.dict(os.environ, {"DRIVEDIT_LOCAL_CONFIG": LOCAL_CONFIG_FIXTURE}):
            self.assertEqual(dc_validate.DEFAULT_REPORT_DIR, FIXTURE_CFG.get("DATA_ROOT") + "/reports")
            self.assertEqual(dc_validate.REPORT_ROOT, FIXTURE_CFG.drive("DATA_ROOT") + "/")

    def test_unmounted_data_drive_is_refused_before_any_mkdir(self):
        """AP-R2 row 4: the synthetic config's data drive is mounted nowhere, so the default report dir lies on the
        Linux root filesystem; the boundary refuses it before anything is created."""
        with mock.patch.dict(os.environ, {"DRIVEDIT_LOCAL_CONFIG": LOCAL_CONFIG_FIXTURE}):
            code, text = run("--session", GOLDEN_DIR)
            self.assertEqual(code, 3, text)
            self.assertIn("is not mounted", text)
            self.assertFalse(os.path.exists(dc_validate.DEFAULT_REPORT_DIR))

    def test_report_dir_on_a_system_volume_is_refused(self):
        tmp = temp_dir("dc2_sysvol_")
        try:
            with bv.patched_system_volume(LC, tmp) as system, \
                    mock.patch.object(dc_validate, "REPORT_ROOT", tmp + "/"):
                code, text = run("--session", GOLDEN_DIR, "--report-dir", os.path.join(system, "reports"))
                self.assertEqual(code, 3, text)
                self.assertIn("Windows system drive", text)
                self.assertEqual(os.listdir(system), ["Windows"])
        finally:
            shutil.rmtree(tmp)

    def test_report_written_under_e(self):
        """Against a real data-drive report root: only in a scratch directory the caller names (DC_TEST_E_SCRATCH, an
        existing directory on the data drive of the DD_TEST_LOCAL_CONFIG config), never a hard-coded path."""
        scratch, why = evidence.data_drive_scratch()   # every check before anything is created
        if scratch is None:
            self.skipTest(why)
        machine = evidence.machine_config_path()
        tmp = tempfile.mkdtemp(prefix="dc2_report_", dir=scratch)
        try:
            with mock.patch.dict(os.environ, {"DRIVEDIT_LOCAL_CONFIG": machine}):
                code, text = run("--session", GOLDEN_DIR, "--report-dir", tmp)
            self.assertEqual(code, 0, text)
            path = os.path.join(tmp, GOLDEN_ID, "dc_validate.json")
            self.assertIn("report: %s" % path, text)
            with open(path, "rb") as f:
                rep = json.loads(f.read().decode("ascii"))
            self.assertEqual((rep["verdict"], rep["exit_code"], rep["session_id"]), ("PASS", 0, GOLDEN_ID))
            self.assertEqual(rep["report_schema"], "drivedit.dc-validate-report/1")
        finally:
            shutil.rmtree(tmp)

    def test_report_path_never_comes_from_session_json(self):
        """An absolute or ../ session_id, or a symlink below the report root, never moves the report out of it."""
        tmp = temp_dir("dc2_reportpath_")
        try:
            root = os.path.join(tmp, "root")
            outside = os.path.join(tmp, "outside")
            os.makedirs(root)
            os.makedirs(outside)
            session = copy_golden(tmp)
            with mock.patch.object(dc_validate, "REPORT_ROOT", root + "/"):
                for sid in (os.path.join(outside, "escaped"), "../outside/escaped"):
                    edit_json(os.path.join(session, "session.json"), lambda o: o.update({"session_id": sid}))
                    code, text = run("--session", session, "--report-dir", root)
                    self.assertEqual(code, 1, text)                    # C01: the id is invalid
                    self.assertTrue(os.path.isfile(os.path.join(root, GOLDEN_ID, "dc_validate.json")))
                    self.assertEqual(os.listdir(outside), [])
                shutil.rmtree(os.path.join(root, GOLDEN_ID))
                os.symlink(outside, os.path.join(root, GOLDEN_ID))     # a pre-existing link below the root
                code, text = run("--session", session, "--report-dir", root)
                self.assertEqual(code, 3, text)
                self.assertIn("resolves outside", text)
                self.assertEqual(os.listdir(outside), [])
        finally:
            shutil.rmtree(tmp)

    def test_report_dirs_outside_e_are_refused(self):
        tmp = temp_dir("dc2_refuse_")
        try:
            target = os.path.join(tmp, "reports")
            for report_dir in (target, FIXTURE_CFG.get("SYSTEM_DRIVE") + "/__dd5_no_such_dir__/reports", "/tmp/dd5_reports"):
                code, text = run("--session", GOLDEN_DIR, "--report-dir", report_dir)
                self.assertEqual(code, 3, text)
                self.assertIn("refused", text)
            self.assertFalse(os.path.exists(target))
        finally:
            shutil.rmtree(tmp)

    def test_other_refusals_exit_3(self):
        code, text = run("--session", GOLDEN_DIR, "--no-report", "--profile", "eval_lockstep")
        self.assertEqual(code, 3, text)
        code, text = run("--session", GOLDEN_DIR, "--no-report", "--checks", "C99")
        self.assertEqual(code, 3, text)
        code, text = run("--session", "/mnt/z/Synthetic Steam (x86)/__dd5_no_such_dir__/" + GOLDEN_ID, "--no-report")
        self.assertEqual(code, 3, text)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                dc_validate.main(["--no-report"])
        self.assertEqual(cm.exception.code, 3)

    def test_everon_needs_sealed_eval(self):
        tmp = temp_dir("dc2_everon_")
        try:
            everon_id = GOLDEN_ID.replace("arland_", "everon_")
            session = os.path.join(tmp, everon_id)
            shutil.copytree(GOLDEN_DIR, session)

            def to_everon(obj):
                obj["session_id"] = everon_id
                if "map" in obj:
                    obj["map"].update({"map_id": "everon", "terrain": "Eden",
                                       "root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"})

            edit_json(os.path.join(session, "session.json"), to_everon)
            edit_json(os.path.join(session, "session_end.json"), to_everon)
            log = os.path.join(session, "_engine_logs", "console.log")
            with open(log, "rb") as f:
                text = f.read().decode("ascii").replace(GOLDEN_ID, everon_id)
            with open(log, "wb") as f:
                f.write(text.encode("ascii"))
            code, text = run("--session", session, "--no-report")
            self.assertEqual(code, 3, text)
            self.assertIn("sealed", text)
            code, text = run("--session", session, "--sealed-eval", "--report-dir", NO_SUCH_E_DIR + "/reports")
            self.assertEqual(code, 3, text)                 # sealed reports only under a SEALED directory
            self.assertFalse(os.path.exists(NO_SUCH_E_DIR))
            code, rep = run_json("--session", session, "--sealed-eval")
            self.assertEqual(rep["stats"]["map_id"], "everon")
            self.assertEqual(failing(rep), [])
        finally:
            shutil.rmtree(tmp)


class ThresholdsTest(unittest.TestCase):
    def test_thresholds_are_capture_program_5_2_verbatim(self):
        if not PROGRAM_MD or not os.path.isfile(PROGRAM_MD):
            self.skipTest(evidence.skip_reason("CAPTURE_PROGRAM.md") if not PROGRAM_MD else
                          "%s not on this machine" % PROGRAM_MD)
        with open(PROGRAM_MD, "rb") as f:
            lines = f.read().decode("utf-8").split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("### 5.2 Thresholds"))
        a = next(j for j in range(start, len(lines)) if lines[j].startswith("```"))
        b = next(j for j in range(a + 1, len(lines)) if lines[j].startswith("```"))
        block = json.loads("\n".join(lines[a + 1:b]))
        path = dc_validate.thresholds_path("1.8.0.13")
        thresholds, _ = dc_validate.load_thresholds(path)
        self.assertEqual(thresholds, block)
        with open(path, "rb") as f:
            raw = json.loads(f.read().decode("ascii"))
        self.assertIn("section 5.2", raw["_source"])

    def test_thresholds_are_keyed_by_pin(self):
        self.assertEqual(os.path.basename(dc_validate.thresholds_path("1.8.0.13")), "thresholds_dc2.1.8.0.13.json")
        self.assertEqual(dc_validate.thresholds_path("1.8.0.13-cold"), dc_validate.thresholds_path("1.8.0.13"))
        for pin in ("1.9.0.10", "1.9.0.10-exp", "../1.8.0.13", None):
            with self.assertRaises(dc_validate.Refused):
                dc_validate.thresholds_path(pin)


class C17LoopTest(unittest.TestCase):
    """C17 on a synthetic mode-E session (trial T1) along a loop route in the drivedit_nav schema."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temp_dir("dc2_c17_")
        out = synth.generate(os.path.join(cls.tmp, "loop"), kind="t0_loop", routes_dir=os.path.join(cls.tmp, "r"))
        cls.session = out["session_dir"]
        cls.route = out["route"]
        cls.mirror_dir = os.path.join(cls.tmp, "mirror")
        os.makedirs(cls.mirror_dir)
        with open(os.path.join(cls.mirror_dir, cls.route["route_id"] + ".json"), "w", encoding="ascii") as f:
            json.dump(synth.mirror_route(cls.route), f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_routes_follow_the_route_schema(self):
        import drivedit_nav
        self.assertEqual(drivedit_nav.validate_route(self.route), [])
        mirrored = synth.mirror_route(self.route)
        self.assertEqual(drivedit_nav.validate_route(mirrored), [])
        self.assertEqual(mirrored["polyline"][0], self.route["polyline"][0])

    def test_matching_route_passes(self):
        code, rep = run_json("--session", self.session, "--routes-dir", os.path.join(self.tmp, "r"),
                             "--vehicle-card", CARD)
        c17 = by_id(rep)["C17"]
        self.assertEqual((c17["severity"], c17["status"]), ("SOFT", "PASS"), c17)
        self.assertLessEqual(c17["value"]["lr_dev"], 0.25)
        self.assertLessEqual(c17["value"]["steer_mean_dev"], 0.03)
        self.assertEqual(code, 0, [(c["id"], c["detail"]) for c in rep["checks"] if c["status"] != "PASS"])
        self.assertEqual(by_id(rep)["C31"]["status"], "PASS")         # HARD from T1
        self.assertEqual(by_id(rep)["C12"]["not_evaluated"] if "not_evaluated" in by_id(rep)["C12"] else [], [])

    def test_mirrored_route_fails(self):
        code, rep = run_json("--session", self.session, "--routes-dir", self.mirror_dir)
        c17 = by_id(rep)["C17"]
        self.assertEqual((c17["severity"], c17["status"]), ("SOFT", "FAIL"), c17)
        self.assertGreater(c17["value"]["lr_dev"], 0.25)
        self.assertGreater(c17["value"]["steer_mean_dev"], 0.03)
        # the session is also off the mirrored route for nav_v1 (nav_valid 0), so C31 (HARD at T1) fails as well
        self.assertEqual((code, rep["verdict"], failing(rep)), (1, "FAIL", ["C17", "C31"]))
        code, rep = run_json("--session", self.session, "--routes-dir", self.mirror_dir, "--checks", "C17")
        self.assertEqual((code, rep["verdict"]), (2, "PASS_WITH_WARNINGS"))     # C17 alone: SOFT at T1

    def test_without_a_route_c17_is_not_evaluated(self):
        code, rep = run_json("--session", self.session, "--checks", "C17")
        self.assertEqual((code, by_id(rep)["C17"]["status"]), (2, "NOT_EVALUATED"))

    def copy(self, name):
        dst = os.path.join(self.tmp, name, os.path.basename(self.session))
        shutil.copytree(self.session, dst)
        return dst

    def test_an_ego_crash_fails_c17_a_touch_does_not(self):
        for name, payload, fails in (
                ("touch", "onset_step=%d;dv_mps=1.20;v_mps=2.00;other=static;src=contact", []),
                ("crash", "onset_step=%d;dv_mps=6.20;v_mps=7.00;other=static;src=dv", ["C17"])):
            session = self.copy(name)
            s = dc_validate.base.load_reader().open_session(session, decode_images=False)
            ev, ego = s.streams["events"], s.streams["telemetry"].col("ego_slot_id")[0]
            start = ev.col("kind").index("episode_start")
            step = ev.col("physics_step")[start] + 60
            add_event(session, ev.col("event_seq")[start], "collision", payload % step, ego, physics_step=step)
            code, rep = run_json("--session", session, "--routes-dir", os.path.join(self.tmp, "r"))
            c17 = by_id(rep)["C17"]
            self.assertEqual(failing(rep), fails, (name, c17))       # the rows are consistent (C36)
            self.assertEqual(c17["value"]["collision_per_30min_max"], 0)
            if fails:
                self.assertIn("1 terminal ego collision(s)", c17["detail"])

    def test_route_hash_mismatch_fails(self):                                 # F-15
        session = self.copy("hash")
        set_column(session, "episodes", "route_sha256", lambda key, v: "0" * 64)
        code, rep = run_json("--session", session, "--routes-dir", os.path.join(self.tmp, "r"), "--checks", "C17")
        c17 = by_id(rep)["C17"]
        self.assertEqual((code, c17["status"]), (2, "FAIL"))                  # SOFT at T1
        self.assertIn("has sha256", c17["detail"])

    def test_missing_road_fields_leave_c17_not_evaluated(self):               # F-15
        session = self.copy("road")
        set_column(session, "telemetry", "road_dist_m", lambda key, v: "")
        code, rep = run_json("--session", session, "--routes-dir", os.path.join(self.tmp, "r"), "--checks", "C17")
        c17 = by_id(rep)["C17"]
        self.assertEqual(c17["status"], "NOT_EVALUATED")
        self.assertIn("road adherence", " ".join(c17["not_evaluated"]))

    def test_hard_tier_never_judges_with_a_session_fitted_gain(self):         # F-14 guard
        code, rep = run_json("--session", self.session, "--routes-dir", self.mirror_dir, "--trial", "T2",
                             "--checks", "C17")
        c17 = by_id(rep)["C17"]
        self.assertEqual((c17["severity"], c17["status"]), ("HARD", "FAIL"))    # the L/R share still fails
        self.assertNotIn("steer_mean_dev", c17["value"])
        self.assertIn("kappa_mean_dev", c17["value"])                         # reported in 1/m, not judged
        code, rep = run_json("--session", self.session, "--routes-dir", os.path.join(self.tmp, "r"), "--trial", "T2",
                             "--checks", "C17")
        self.assertEqual((code, by_id(rep)["C17"]["status"]), (2, "NOT_EVALUATED"))

    def test_c31_uses_nav_valid(self):                                        # F-18
        moved = json.loads(json.dumps(self.route))
        for pt in moved["polyline"]:
            pt[0] += 100.0                                                     # 100 m east: nav_valid 0
        routes = os.path.join(self.tmp, "moved")
        os.makedirs(routes, exist_ok=True)
        with open(os.path.join(routes, moved["route_id"] + ".json"), "w", encoding="ascii") as f:
            json.dump(moved, f)
        code, rep = run_json("--session", self.session, "--routes-dir", routes, "--checks", "C31")
        c31 = by_id(rep)["C31"]
        self.assertEqual((code, c31["severity"], c31["status"]), (1, "HARD", "FAIL"))
        self.assertGreater(c31["value"]["nav_invalid_steps"], 0)
        code, rep = run_json("--session", self.session, "--checks", "C31")    # no route file: nav not checked
        self.assertEqual((code, by_id(rep)["C31"]["status"]), (2, "NOT_EVALUATED"))


class LegacyTest(unittest.TestCase):
    def test_synthetic_legacy_session_fails_for_its_known_reasons(self):
        tmp = temp_dir("dc2_legacy_")
        try:
            legacy = synth.write_legacy_session(tmp)
            code, rep = run_json("--session", legacy, "--legacy")
            self.assertEqual(code, 1)
            self.assertEqual(failing(rep), ["C01", "C05", "C08", "C10", "C13a"])
            self.assertIn("bimodal", by_id(rep)["C08"]["detail"])
            self.assertIn("37 of 40", by_id(rep)["C10"]["detail"])
        finally:
            shutil.rmtree(tmp)

    def test_legacy_sessions_4811_2702_2929(self):
        if not LEGACY_ROOT or not os.path.isdir(LEGACY_ROOT):
            self.skipTest("%s does not exist: the data-drive copy of the legacy sessions is TICKETS_UNIFIED "
                          "decision 13, still open" % (LEGACY_ROOT or evidence.skip_reason("the legacy sessions")))
        reasons = {"session_4811": ["C01", "C05", "C08", "C10"], "session_2702": ["C01", "C05", "C08", "C10"],
                   "session_2929": ["C01", "C05"]}
        for name, expected in reasons.items():
            path = os.path.join(LEGACY_ROOT, name)
            if not os.path.isdir(path):
                self.fail("%s missing from the legacy copy" % path)
            code, rep = run_json("--session", path, "--legacy")
            self.assertEqual(code, 1, name)
            for check in expected:
                self.assertIn(check, failing(rep), (name, rep["checks"]))


class NoisePilotTest(unittest.TestCase):
    """DD-7-ident's pilot shape (make_synthetic --kind t0_loop_noise): one mode-N lap with a disturbance of each
    family. Every check passes; C17 judges mode-E expert episodes only, so the disturbances are never expert
    failures (N/A here, and an E lap next to them is judged alone); the reader takes action_cmd from the applied
    hl_* and action_label from the clean exp_*_now, and the noise columns agree with the perturb events."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temp_dir("dc2_noise_")
        cls.routes = os.path.join(cls.tmp, "r")
        out = synth.generate(os.path.join(cls.tmp, "noise"), kind="t0_loop_noise", routes_dir=cls.routes)
        cls.session = out["session_dir"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_every_check_passes_and_c17_skips_mode_n(self):
        code, rep = run_json("--session", self.session, "--routes-dir", self.routes, "--vehicle-card", CARD)
        self.assertEqual(code, 0, [(c["id"], c["detail"]) for c in rep["checks"] if c["status"] != "PASS"])
        c17 = by_id(rep)["C17"]
        self.assertEqual((c17["status"], c17["detail"]), ("N/A", "no mode-E expert episode"))

    def test_c17_judges_the_e_lap_only(self):
        """The same lap tagged mode E: C17 compares it with its route (the disturbances are then its own driving)."""
        dst = os.path.join(self.tmp, "as_e", os.path.basename(self.session))
        shutil.copytree(self.session, dst)
        set_column(dst, "episodes", "mode", lambda key, v: "E")
        set_column(dst, "ctl", "mode", lambda key, v: "E" if v == "N" else v)
        code, rep = run_json("--session", dst, "--routes-dir", self.routes, "--checks", "C17")
        c17 = by_id(rep)["C17"]
        self.assertNotEqual(c17["status"], "N/A", c17)
        self.assertIn("mode-E episode", c17["detail"])

    def test_reader_labels_and_noise_windows(self):
        s = dc_validate.base.load_reader().open_session(self.session, decode_images=False)
        ctl, ev = s.streams["ctl"], s.streams["events"]
        self.assertEqual(set(s.streams["episodes"].col("mode")), {"N"})
        starts = {}
        ends = {}
        for kind, step, payload in zip(ev.col("kind"), ev.col("physics_step"), ev.col("payload")):
            if kind in ("perturb_start", "perturb_end"):
                fields = dict(item.split("=", 1) for item in payload.split(";"))
                self.assertEqual(sorted(fields), ["amplitude", "duration_s", "kind", "noise_id"])
                (starts if kind == "perturb_start" else ends)[int(fields["noise_id"])] = step
        self.assertEqual(sorted(starts), list(range(5)))
        self.assertEqual(sorted(ends), list(range(5)))
        for step, active, noise_id, mode in zip(ctl.col("physics_step"), ctl.col("noise_active"),
                                                ctl.col("noise_id"), ctl.col("mode")):
            self.assertEqual(bool(active), noise_id >= 0)
            if active:
                self.assertEqual(mode, "N")
                self.assertTrue(starts[noise_id] < step <= ends[noise_id], (step, noise_id))
        perturbed = [d for d in s.decisions if d["action_cmd"] is not None and d["mode"] == "N"
                     and d["action_cmd"] != d["action_label"]]
        self.assertGreater(len(perturbed), 40)
        self.assertTrue(any(d["action_cmd"][2] == 0 and d["action_label"][2] == 1 for d in perturbed))   # the stop
        self.assertTrue(all(d["masks"]["policy_valid"] == 1 for d in perturbed if d["obs_valid"] == 1))

    def test_fixture_amplitudes_mean_what_dc_noise_logs(self):
        """The k_pulse disturbance (amplitude 0.4 in its payload) is DC_Noise's: 0.4 x kappa_max(v) x
        sin(pi (i + 0.5) / n) on kappa_target at decision i of n = 10, kappa_max(v) = min(0.2, 4 / max(v, 3)^2)."""
        s = dc_validate.base.load_reader().open_session(self.session, decode_images=False)
        ctl = s.streams["ctl"]
        noise_of_seq = dict(zip(ctl.col("hl_seq"), ctl.col("noise_id")))
        pulse = [d for d in s.decisions if d["hl_seq"] is not None and noise_of_seq.get(d["hl_seq"]) == 0]
        self.assertEqual(len(pulse), 10)
        for i, d in enumerate(pulse):
            v = max(d["state"][0], 3.0)
            scale = min(0.2, 4.0 / (v * v))
            got = (d["action_cmd"][0] - d["action_label"][0]) / (0.4 * scale)
            self.assertAlmostEqual(got, math.sin(math.pi * (i + 0.5) / 10), delta=2e-3)

    def test_c07_checks_the_disturbance(self):
        """C07 owns the held command and its R-01 boundary, the disturbance included: each mutation of the noise
        metadata fails it (Codex: every noise_id -1 with the rows still active passed every check before)."""
        def mutated(name, edit):
            dst = os.path.join(self.tmp, name, os.path.basename(self.session))
            shutil.copytree(self.session, dst)
            edit(dst)
            return run_json("--session", dst, "--routes-dir", self.routes, "--checks", "C07")

        def first_active(dst):
            s = dc_validate.base.load_reader().open_session(dst, decode_images=False)
            ctl = s.streams["ctl"]
            return [k for k, a in zip(ctl.col("physics_step"), ctl.col("noise_active")) if a]

        code, rep = run_json("--session", self.session, "--routes-dir", self.routes, "--checks", "C07")
        self.assertEqual((code, by_id(rep)["C07"]["status"]), (0, "PASS"))
        active = first_active(self.session)
        cases = {
            "ids_dropped": (lambda d: set_column(d, "ctl", "noise_id", lambda k, v: "-1"), "disagree"),
            "mode_e": (lambda d: (set_column(d, "ctl", "mode", lambda k, v: "E" if v == "N" else v),
                                  set_column(d, "episodes", "mode", lambda k, v: "E")), "outside a mode-N episode"),
            "row_off": (lambda d: (set_column(d, "ctl", "noise_active", lambda k, v: "0" if k == active[3] else v),
                                   set_column(d, "ctl", "noise_id", lambda k, v: "-1" if k == active[3] else v)),
                        "inside a decision interval"),
            "id_swapped": (lambda d: set_column(d, "ctl", "noise_id",
                                                lambda k, v: "4" if v == "0" else ("0" if v == "4" else v)),
                           "perturbation windows"),
        }
        for name, (edit, detail) in cases.items():
            code, rep = mutated(name, edit)
            c07 = by_id(rep)["C07"]
            self.assertEqual((code, c07["status"]), (1, "FAIL"), (name, c07["detail"]))
            self.assertIn(detail, c07["detail"], name)
        s = dc_validate.base.load_reader().open_session(self.session, decode_images=False)
        ev = s.streams["events"]
        end = [q for q, kind in zip(ev.col("event_seq"), ev.col("kind")) if kind == "perturb_end"][0]

        def drop_end(dst):
            set_column(dst, "events", "kind", lambda k, v: "nav_command" if k == end else v)
        code, rep = mutated("end_dropped", drop_end)
        self.assertEqual(by_id(rep)["C07"]["status"], "FAIL")
        self.assertIn("0 perturb_end", by_id(rep)["C07"]["detail"])

    def test_c07_refuses_an_orphan_window(self):
        """Codex round 2: disturbance 0's active rows cleared and its perturb events moved to episode 999 (which does
        not exist) passed every check, the expected and observed rows both empty. Now the window's episode must
        exist and be mode N, and the window must hold rows."""
        s = dc_validate.base.load_reader().open_session(self.session, decode_images=False)
        ev = s.streams["events"]
        marks = [q for q, kind, payload in zip(ev.col("event_seq"), ev.col("kind"), ev.col("payload"))
                 if kind in ("perturb_start", "perturb_end") and payload.startswith("noise_id=0;")]
        self.assertEqual(len(marks), 2)
        dst = os.path.join(self.tmp, "orphan", os.path.basename(self.session))
        shutil.copytree(self.session, dst)
        set_column(dst, "ctl", "noise_id", lambda k, v: "-1" if v == "0" else v)
        ids = dict(zip(s.streams["ctl"].col("physics_step"), s.streams["ctl"].col("noise_id")))
        set_column(dst, "ctl", "noise_active", lambda k, v: "0" if ids.get(k) == 0 else v)
        set_column(dst, "events", "episode_id", lambda k, v: "999" if k in marks else v)
        code, rep = run_json("--session", dst, "--routes-dir", self.routes, "--vehicle-card", CARD)
        c07 = by_id(rep)["C07"]
        self.assertEqual((code, c07["status"]), (1, "FAIL"), c07["detail"])
        self.assertIn("episode 999 noise_id 0", c07["detail"])
        self.assertIn("does not have", c07["detail"])

    def test_c07_ids_are_per_episode(self):
        """Codex round 2: two valid N episodes that both use noise_id 0 failed C07 (the ids were grouped across the
        session). dc-2.0 9.1 declares no session-wide uniqueness, so windows are per (episode, noise_id); a window
        outside its episode's steps, in a mode-E episode or without rows still fails."""
        stream_checks = dc_validate.checks.streams

        class Stream:
            def __init__(self, **cols):
                self.cols = cols

            def col(self, name):
                return self.cols[name]

        def session(events, modes=("N", "N"), rows=None):
            # two episodes of 6 callbacks each (steps 10..15 and 20..25); decisions at 10, 13, 20, 23
            steps = list(range(10, 16)) + list(range(20, 26))
            ep = [0] * 6 + [1] * 6
            seq = [0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5]
            rows = rows or {11: 0, 12: 0, 13: 0, 21: 0, 22: 0, 23: 0}
            ctl = Stream(physics_step=steps, episode_id=ep, hl_seq=seq, mode=[modes[e] for e in ep],
                         noise_active=[1 if k in rows else 0 for k in steps],
                         noise_id=[rows.get(k, -1) for k in steps],
                         hl_kappa=[0.0] * 12, hl_accel=[0.0] * 12, hl_dir=[1] * 12)
            evs = Stream(kind=[e[0] for e in events], physics_step=[e[1] for e in events],
                         episode_id=[e[2] for e in events], payload=["noise_id=%d;kind=k_pulse;amplitude=0.1;"
                                                                     "duration_s=0.6" % e[3] for e in events])
            eps = Stream(episode_id=[0, 1], mode=list(modes), start_physics_step=[10, 20], end_physics_step=[15, 25])
            return types.SimpleNamespace(streams={"ctl": ctl, "events": evs, "episodes": eps})

        both = [("perturb_start", 10, 0, 0), ("perturb_end", 13, 0, 0), ("perturb_start", 20, 1, 0),
                ("perturb_end", 23, 1, 0)]
        self.assertEqual(stream_checks._noise_boundary(session(both)), [])
        cases = {
            "episode 1 noise_id 0: its episode is mode E, not N": session(both, modes=("N", "E")),
            "not ordered inside the episode's steps": session(both[:3] + [("perturb_end", 27, 1, 0)]),
            "episode 999 noise_id 0: its perturb events name an episode that the episodes stream does not have":
                session([(k, st, 999 if e == 1 else e, n) for k, st, e, n in both]),
            "episode 0 noise_id 0 has 2 perturb_start and 1 perturb_end": session(both + [("perturb_start", 14, 0, 0)]),
        }
        for want, sess in cases.items():
            issues = stream_checks._noise_boundary(sess)
            self.assertTrue(any(want in i for i in issues), (want, issues))


if __name__ == "__main__":
    unittest.main()
