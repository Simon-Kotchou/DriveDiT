#!/usr/bin/env python3
"""Tests for tools/validator/dc_validate.py (DD-5-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_validate.py

dc_validate runs in-process through its main(argv) (stdout captured). Acceptance of the ticket shown here:
  * dc_validate exits 0 on the golden fixture;
  * each negative fixture (tests/fixtures/capture_dc2/negative/*/mutation.json, generated here) fails exactly the
    check it names;
  * C17 passes a synthetic session along a loop route and fails the same session against the mirrored route;
  * dc-1.0 sessions FAIL for their known reasons (synthetic legacy session; the real sessions 4811, 2702 and 2929
    when their E: copy exists, else that test skips: TICKETS_UNIFIED decision 13 is open);
  * reports are written only under /mnt/e (a refusal for /dev/shm and /mnt/c); thresholds equal CAPTURE_PROGRAM
    5.2 verbatim.
Sessions are generated under $TMPDIR (on /dev/shm/ or /mnt/e/, else /dev/shm/); the report test writes into a
temporary directory on /mnt/e ($TMPDIR when it is there, else /mnt/e/drivedit_realign/t0b/scratch) and removes it.
The default report root /mnt/e/DriveDiT/reports is never used here.
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

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures", "capture_dc2")
GOLDEN_ID = "arland_T0b_20261025T091512Z_wb_i00_5e00"
GOLDEN_DIR = os.path.join(FIXTURES, "golden", GOLDEN_ID)
ROUTES = os.path.join(FIXTURES, "routes")
CARD = os.path.join(FIXTURES, "cards", "S105_synthetic.json")
PROGRAM_MD = "/mnt/e/drivedit_realign/program/CAPTURE_PROGRAM.md"
LEGACY_ROOT = "/mnt/e/DriveDiT/captures/_legacy_DrivingData_C"
E_SCRATCH = "/mnt/e/drivedit_realign/t0b/scratch"
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


class ReportTest(unittest.TestCase):
    def test_default_report_root(self):
        self.assertEqual(dc_validate.DEFAULT_REPORT_DIR, "/mnt/e/DriveDiT/reports")

    def test_report_written_under_e(self):
        base = os.path.realpath(os.environ.get("TMPDIR") or "/")
        made_scratch = False
        if not base.startswith("/mnt/e/"):
            if not os.path.isdir(os.path.dirname(E_SCRATCH)):
                self.skipTest("no /mnt/e temporary base ($TMPDIR on /mnt/e or %s)" % E_SCRATCH)
            made_scratch = not os.path.isdir(E_SCRATCH)
            os.makedirs(E_SCRATCH, exist_ok=True)
            base = E_SCRATCH
        tmp = tempfile.mkdtemp(prefix="dc2_report_", dir=base)
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
            if made_scratch and not os.listdir(E_SCRATCH):
                os.rmdir(E_SCRATCH)

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
            code, text = run("--session", session, "--sealed-eval", "--report-dir", "/mnt/e/DriveDiT/reports")
            self.assertEqual(code, 3, text)                 # sealed reports only under a SEALED directory
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
        thresholds, _ = dc_validate.load_thresholds(dc_validate.DEFAULT_THRESHOLDS)
        self.assertEqual(thresholds, block)
        with open(dc_validate.DEFAULT_THRESHOLDS, "rb") as f:
            raw = json.loads(f.read().decode("ascii"))
        self.assertIn("section 5.2", raw["_source"])


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
        self.assertEqual((code, rep["verdict"], failing(rep)), (2, "PASS_WITH_WARNINGS", ["C17"]))

    def test_without_a_route_c17_is_not_evaluated(self):
        code, rep = run_json("--session", self.session, "--checks", "C17")
        self.assertEqual((code, by_id(rep)["C17"]["status"]), (2, "NOT_EVALUATED"))


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
