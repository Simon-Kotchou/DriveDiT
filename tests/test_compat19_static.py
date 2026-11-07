"""Tests for the static compatibility checks of DD-14a (tools/compat19): CC-01 build identity (p19_build.py) and the
CC-02 planned-surface check (p19_api.py). Stdlib unittest; fixtures only (a fake Steam layout from
tools/pin/tests, the fixture script dumps of tools/enfusion-mcp/tests).

    python3 -I -B -m unittest discover -s tests -p test_compat19_static.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN_TESTS = os.path.join(REPO, "tools", "pin", "tests")
FIX = os.path.join(REPO, "tools", "enfusion-mcp", "tests", "fixtures")


def _load(name, path):
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("dd_pin_test_pin_helpers", os.path.join(PIN_TESTS, "test_pin.py"))
dd14a = _load("dd_pin_test_pin_dd14a_helpers", os.path.join(PIN_TESTS, "test_pin_dd14a.py"))
run_main, freeze_check = base.run_main, dd14a.freeze_check
p19_build = _load("dd_compat19_p19_build", os.path.join(REPO, "tools", "compat19", "p19_build.py"))
p19_api = _load("dd_compat19_p19_api", os.path.join(REPO, "tools", "compat19", "p19_api.py"))


class CC01Tests(dd14a.V2Case):
    def setUp(self):
        super().setUp()
        self.put_active(b"1.8.0.13\n")

    def before(self):
        rc, out, err = run_main(freeze_check.main, ["before", "--pin-active", "--pin-dir", self.pin_dir, "--steam",
                                                    "online"], self.roots)
        self.assertEqual(rc, 0, err)
        return os.path.join(self.pin_dir, "freeze", [n for n in os.listdir(os.path.join(self.pin_dir, "freeze"))
                                                     if n.endswith(".before.json")][-1])

    def after(self, before):
        return run_main(freeze_check.main, ["after", "--before", before], self.roots)

    def user_launch(self):
        rc, out, err = self.preflight_active("--exe", "wb")
        self.assertEqual(rc, 0, err)

    def cc01(self):
        return run_main(p19_build.main, ["--pin-dir", self.pin_dir], self.roots)

    def test_cc01_fails_then_passes(self):
        rc, out, _ = self.cc01()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL  build_key", out)
        self.assertIn("FAIL  freeze_check", out)
        self.assertIn("PASS  appmanifests", out)
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        b = self.before()
        self.user_launch()
        self.assertEqual(self.after(b)[0], 0)
        rc, out, _ = self.cc01()
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS  preflight     24/24 checks pass", out)
        self.assertIn("CC-01 PASS", out)

    def test_cc01_appmanifest_flags(self):
        self.fs.set_auto_update("0")
        rc, out, _ = self.cc01()
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  appmanifests", out)


class CC02Tests(unittest.TestCase):
    def run_api(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = p19_api.main(["--dump-dirs", os.path.join(FIX, "dump_18", "scripts"),
                               os.path.join(FIX, "dump_19", "scripts")] + list(extra))
        return rc, out.getvalue(), err.getvalue()

    def test_planned_surface_passes(self):
        rc, out, err = self.run_api("--planned", os.path.join(FIX, "planned_fixture.txt"))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("4 members: 2 new-in-new, 2 unchanged", out)
        self.assertIn("CC-02 PASS", out)

    def test_removed_member_fails_unless_triaged(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=base._tmp_base()) as d:
            planned = os.path.join(d, "planned.txt")
            with open(planned, "w") as fh:
                fh.write("ZxHarness CountSuites Start\n")
            rc, out, _ = self.run_api("--planned", planned)
            self.assertEqual(rc, 1, out)
            self.assertIn("FAIL  ZxHarness.CountSuites REMOVED", out)
            self.assertIn("FAIL  ZxHarness.Start SIG_CHANGED", out)
            triage = os.path.join(d, "triage.txt")
            with open(triage, "w") as fh:
                fh.write("ZxHarness.CountSuites REMOVED obsolete, unused\n"
                         "ZxHarness.Start SIG_CHANGED notnull suites; nothing calls it\n")
            rc, out, _ = self.run_api("--planned", planned, "--triage", triage)
            self.assertEqual(rc, 0, out)


REAL_DUMP = os.environ.get("DD14A_SCRIPT_DUMP", "/mnt/e/drivedit_realign/program/1_9/work/dl/sd_stable")


class RealDumpSpotChecks(unittest.TestCase):
    """Opt-in (the official dump is not in the repo; skipped when absent). An independent regex over a handful of
    known declarations on both tags, compared with what the index parser makes of the same file (Codex round 1:
    the diff tests share the parser, so a symmetric parser omission would stay invisible)."""

    CASES = (
        # (file under scripts/, class, member, regex for the independent reading, kind)
        ("Game/generated/Vehicle/VehicleWheeledSimulation.c", "VehicleWheeledSimulation", "SetSteering",
         r"proto external void SetSteering\(([^)]*)\);", "params"),
        ("Game/generated/Vehicle/VehicleWheeledSimulation.c", "VehicleWheeledSimulation", "GetHandbrake",
         r"proto external float GetHandbrake\(([^)]*)\);", "params"),
        ("Game/generated/GameMode/RespawnSystemComponent.c", "RespawnSystemComponent", "DoSpawn",
         r"GenericEntity DoSpawn\(([^)]*)\);", "params"),
        ("Core/generated/Visual/Color.c", "Color", "Black", r"static const ref Color Black = (.*?);", "init"),
        ("GameLib/entities/ScriptCamera.c", "ScriptCamera", None, r"^class ScriptCamera\s*:\s*(\w+)", "parent"),
    )

    def setUp(self):
        if not os.path.isdir(os.path.join(REAL_DUMP, ".git")):
            self.skipTest("the official script dump is not present (%s)" % REAL_DUMP)
        import subprocess
        self.subprocess = subprocess

    def show(self, tag, rel):
        r = self.subprocess.run(["git", "--no-optional-locks", "-C", REAL_DUMP, "show", "%s:scripts/%s" % (tag, rel)],
                                capture_output=True)
        return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None

    def test_parser_agrees_with_an_independent_reading(self):
        import re
        api_index = p19_api.api_index
        checked = 0
        for tag in ("v1.8.0.13", "v1.9.0.37"):
            for rel, cls, member, rx, kind in self.CASES:
                text = self.show(tag, rel)
                self.assertIsNotNone(text, (tag, rel))
                m = re.search(rx, text, re.M)
                parsed = {c["name"]: c for c in api_index.parse_file(rel, text)}[cls]
                if kind == "parent":
                    self.assertEqual(parsed["parents"], [m.group(1)], (tag, cls))
                elif kind == "init":
                    props = {p["name"]: p for p in parsed["properties"] + parsed["protectedProperties"]}
                    self.assertEqual(props[member]["defaultValue"], m.group(1).strip(), (tag, cls, member))
                else:
                    meths = {x["name"]: x for x in parsed["methods"] + parsed["staticMethods"]
                             + parsed["protectedMethods"]}
                    if m is None:
                        self.assertNotIn(member, meths, (tag, cls, member))
                        continue
                    want = ", ".join(" ".join(p.split()) for p in m.group(1).split(",")) if m.group(1).strip() else ""
                    have = meths[member]["signature"]
                    self.assertTrue(have.endswith("%s (%s)" % (member, want)), (tag, have, want))
                checked += 1
        self.assertGreaterEqual(checked, 9)

    def test_known_facts(self):
        api_index = p19_api.api_index
        vws = {}
        for tag in ("v1.8.0.13", "v1.9.0.37"):
            rel = "Game/generated/Vehicle/VehicleWheeledSimulation.c"
            c = {x["name"]: x for x in api_index.parse_file(rel, self.show(tag, rel))}["VehicleWheeledSimulation"]
            vws[tag] = {m["name"] for m in c["methods"]}
        self.assertEqual(vws["v1.9.0.37"] - vws["v1.8.0.13"], {
            "GetHandbrake", "EngineGetRPMRedline", "UpdateEngineConfiguration", "UpdateClutchConfiguration",
            "UpdateIADConfiguration", "UpdateDiffConfiguration", "UpdateWheelConfiguration",
            "UpdateSuspensionConfiguration", "UpdateTyreConfiguration"})


class TsvGuardTests(unittest.TestCase):
    """Codex round 1 (HIGH, R1): --tsv goes through the tools/pin write guard."""

    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="dd14a-tsv-", dir=base._tmp_base())
        self.outside = tempfile.mkdtemp(prefix="dd14a-tsv-outside-", dir=base._tmp_base())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(shutil.rmtree, self.outside, True)
        self.roots = (self.tmp,)

    def run_api(self, tsv, cwd=None):
        out, err = io.StringIO(), io.StringIO()
        old = os.getcwd()
        if cwd:
            os.chdir(cwd)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = p19_api.main(["--dump-dirs", os.path.join(FIX, "dump_18", "scripts"),
                                   os.path.join(FIX, "dump_19", "scripts"), "--planned",
                                   os.path.join(FIX, "planned_fixture.txt"), "--tsv", tsv], write_roots=self.roots)
        finally:
            os.chdir(old)
        return rc, out.getvalue() + err.getvalue()

    def test_inside_the_roots_is_written(self):
        rc, out = self.run_api(os.path.join(self.tmp, "cc02.tsv"))
        self.assertEqual(rc, 0, out)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "cc02.tsv")))

    def test_c_drive_stand_in_refused(self):
        pinlib = base.pinlib
        fake_c = os.path.join(self.tmp, "fake_c_drive")
        os.makedirs(fake_c)
        saved = pinlib.FORBIDDEN_WRITE_PREFIXES
        pinlib.FORBIDDEN_WRITE_PREFIXES = saved + (os.path.realpath(fake_c).lower() + "/",)
        self.addCleanup(setattr, pinlib, "FORBIDDEN_WRITE_PREFIXES", saved)
        rc, out = self.run_api(os.path.join(fake_c, "cc02.tsv"))
        self.assertEqual(rc, 2, out)
        self.assertIn("refusing to write on C:", out)
        self.assertEqual(os.listdir(fake_c), [])

    def test_real_c_path_refused_before_anything_runs(self):
        rc, out = self.run_api("/mnt/c/__dd14a_never_written__/cc02.tsv")
        self.assertEqual(rc, 2, out)
        self.assertIn("refusing to write on C:", out)

    def test_symlinked_parent_refused(self):
        link = os.path.join(self.tmp, "link")
        os.symlink(self.outside, link)
        rc, out = self.run_api(os.path.join(link, "cc02.tsv"))
        self.assertEqual(rc, 2, out)
        self.assertEqual(os.listdir(self.outside), [])

    def test_relative_path_off_the_roots_refused(self):
        # a relative --tsv resolves against the cwd; on the WSL disk (C:-backed) that is outside E:
        rc, out = self.run_api("cc02.tsv", cwd=self.outside)
        self.assertEqual(rc, 2, out)
        self.assertIn("outside", out)
        self.assertEqual(os.listdir(self.outside), [])


if __name__ == "__main__":
    unittest.main()
