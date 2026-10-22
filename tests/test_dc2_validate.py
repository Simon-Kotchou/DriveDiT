#!/usr/bin/env python3
"""Tests for tools/validator/dc_validate.py (DD-5-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_validate.py

dc_validate runs in-process through its main(argv) (stdout captured). Acceptance of the ticket shown here:
  * dc_validate exits 0 on the golden fixture (and on the 5 Hz fallback session);
  * each negative fixture (tests/fixtures/capture_dc2/negative/*/mutation.json, generated here) fails exactly the
    check it names;
  * C17 passes a synthetic session along a loop route and fails the same session against the mirrored route;
  * dc-1.0 sessions FAIL for their known reasons (synthetic legacy session; the real sessions 4811, 2702 and 2929
    when their E: copy exists, else that test skips: TICKETS_UNIFIED decision 13 is open);
  * reports are written only under /mnt/e (a refusal for /dev/shm and /mnt/c), never to a path built from
    session.json; thresholds equal CAPTURE_PROGRAM 5.2 verbatim, chosen by pin.
FailLoudTest turns each probe of the 2026-10-22 review (t0b/review/REVIEW_DD5_MIN.md) into a test: one defect on a
copy of the golden session (commit lines kept true), and the exit code and failing check it must produce.
Sessions are generated under $TMPDIR (on /dev/shm/ or /mnt/e/, else /dev/shm/). The report-writing tests point the
validator's report root at a temporary directory; the one test against the real /mnt/e root runs only when
DC_TEST_E_SCRATCH names an existing directory under /mnt/e (else it skips). The default report root
/mnt/e/DriveDiT/reports is never used here.
"""

import contextlib
import glob
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures", "capture_dc2")
GOLDEN_ID = "arland_T0b_20261025T091512Z_wb_i00_5e00"
GOLDEN_DIR = os.path.join(FIXTURES, "golden", GOLDEN_ID)
ROUTES = os.path.join(FIXTURES, "routes")
CARD = os.path.join(FIXTURES, "cards", "S105_synthetic.json")
PROGRAM_MD = "/mnt/e/drivedit_realign/program/CAPTURE_PROGRAM.md"
LEGACY_ROOT = "/mnt/e/DriveDiT/captures/_legacy_DrivingData_C"
NO_SUCH_E_DIR = "/mnt/e/__dd5_no_such_dir__"
CHECK_IDS = ("C01", "C02", "C05", "C06", "C07", "C08", "C10", "C11", "C12", "C13a", "C13b", "C15", "C17", "C20",
             "C21", "C28", "C31", "C34", "C35")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dc_validate = load("dc_validate", "tools/validator/dc_validate.py")
synth = load("dc2_make_synthetic_v", "tests/fixtures/capture_dc2/make_synthetic.py")


def temp_dir(prefix):
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not ((base.startswith("/dev/shm/") or base.startswith("/mnt/e/")) and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)) or base.startswith("/mnt/c"):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/ or /mnt/e/")
    return tempfile.mkdtemp(prefix=prefix, dir=base)


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
    """A copy of the golden session under tmp (from one RAM copy; copies from /mnt/e are slow)."""
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
        # CAPTURE_PROGRAM 4.4 T0-MIN: C12, C13b and C21 SOFT; the rest HARD (C20, C35 stricter by choice)
        self.assertEqual([c for c in CHECK_IDS if checks[c]["severity"] == "SOFT"], ["C12", "C13b", "C21"])
        self.assertEqual(rep["thresholds_file"], "thresholds_dc2.1.8.0.13.json")
        self.assertTrue(checks["C08"]["provisional"])      # TBD-1/TBD-2 while dc-2.0 is a DRAFT
        self.assertTrue(checks["C34"]["provisional"])      # TBD-3
        self.assertFalse(checks["C05"]["provisional"])
        self.assertEqual(checks["C13a"]["value"]["join_failures"], 0)
        self.assertEqual(checks["C13b"]["value"]["lag_frames"], 0)
        self.assertEqual(checks["C02"]["value"]["split"], "train")
        self.assertEqual((rep["verdict"], rep["tier"], rep["contract_version"]), ("PASS", "T0", "dc-2.0"))
        self.assertTrue(rep["contract_status"].startswith("DRAFT"))

    def test_negative_fixtures_fail_exactly_their_check(self):
        specs = sorted(glob.glob(os.path.join(FIXTURES, "negative", "*", "mutation.json")))
        self.assertEqual([os.path.basename(os.path.dirname(p)) for p in specs],
                         ["gap", "missing_commit", "reused_slot"])
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
        self._drop_last_chunk(session, "telemetry", 597)                       # counts kept honest
        for trial in ("T0b", "T1"):
            code, rep = self.check(session, "--trial", trial)
            self.assertEqual((code, failing(rep)), (1, ["C06", "C08", "C34"]), trial)   # joins, cadence, tail
            self.assertIn("missed telemetry", by_id(rep)["C08"]["detail"])
        session = self.golden("ctl")
        self._drop_last_chunk(session, "ctl", 599)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C34"]))
        self.assertIn("ctl/: committed rows end at physics_step 599", by_id(rep)["C34"]["detail"])

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

    def test_image_rate_rules(self):                                          # F-3
        for image_hz, why in ((5.0, "differs from image_hz"), (1.0, "is not one of [5.0, 10.0]")):
            session = self.golden("hz%d" % image_hz)
            edit_json(os.path.join(session, "session.json"), lambda o: o["clocks"].update({"image_hz": image_hz}))
            code, rep = self.check(session)
            self.assertEqual((code, failing(rep)), (1, ["C08"]), image_hz)
            self.assertIn(why, by_id(rep)["C08"]["detail"])

    def test_5hz_fallback_passes(self):                                       # F-3
        out = synth.generate(os.path.join(self.tmp, "g5"), kind="golden_5hz", routes_dir=os.path.join(self.tmp, "r"))
        code, rep = run_json("--session", out["session_dir"], "--routes-dir", os.path.join(self.tmp, "r"),
                             "--vehicle-card", CARD)
        self.assertEqual(code, 0, [(c["id"], c["detail"]) for c in rep["checks"] if c["status"] != "PASS"])
        self.assertEqual((by_id(rep)["C08"]["value"]["N"], by_id(rep)["C08"]["value"]["decision_hz"]), (12, 5.0))

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
        set_column(session, "ctl", "hl_kappa", lambda key, v: "0.012345" if key == 303 else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C07"]))
        self.assertIn("inside a decision window", by_id(rep)["C07"]["detail"])

    def test_soft_checks_warn_at_t0_and_fail_from_t1(self):                   # F-6
        session = self.golden()
        set_column(session, "hf", "fps", lambda key, v: "25.000")
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep), by_id(rep)["C21"]["severity"]), (2, ["C21"], "SOFT"))
        code, rep = self.check(session, "--trial", "T1")
        self.assertEqual((code, failing(rep), by_id(rep)["C21"]["severity"]), (1, ["C21"], "HARD"))

    def test_img_path_outside_the_session_fails_c11(self):                    # F-7
        session = self.golden()
        set_column(session, "img", "path", lambda key, v: "../elsewhere/frames/f_00000007" if key == 7 else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C11"]))
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
        rewrite_chunk(session, "ctl", 0, lambda cols, rows: rows[:302] + [list(rows[301])] + rows[302:])
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
        set_column(session, "img", "issue_render_frame", lambda key, v: str(int(v) + 2) if key == 50 else v)
        code, rep = self.check(session)
        self.assertEqual((code, failing(rep)), (1, ["C13a"]))
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


class ReportTest(unittest.TestCase):
    def test_default_report_root(self):
        self.assertEqual(dc_validate.DEFAULT_REPORT_DIR, "/mnt/e/DriveDiT/reports")

    def test_report_written_under_e(self):
        """Against the real /mnt/e report root: only in a scratch directory the caller names (DC_TEST_E_SCRATCH, an
        existing directory under /mnt/e/), never a hard-coded path."""
        scratch = os.environ.get("DC_TEST_E_SCRATCH")
        if not scratch or not os.path.isdir(scratch) or not os.path.realpath(scratch).startswith("/mnt/e/"):
            self.skipTest("set DC_TEST_E_SCRATCH to an existing directory under /mnt/e/ to run this test")
        tmp = tempfile.mkdtemp(prefix="dc2_report_", dir=os.path.realpath(scratch))
        try:
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
            for report_dir in (target, "/mnt/c/__dd5_no_such_dir__/reports"):
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
        code, text = run("--session", "/mnt/c/__dd5_no_such_dir__/" + GOLDEN_ID, "--no-report")
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
        if not os.path.isfile(PROGRAM_MD):
            self.skipTest("%s not on this machine" % PROGRAM_MD)
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
        if not os.path.isdir(LEGACY_ROOT):
            self.skipTest("%s does not exist: the E: copy of the legacy sessions is TICKETS_UNIFIED decision 13, "
                          "still open" % LEGACY_ROOT)
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


if __name__ == "__main__":
    unittest.main()
