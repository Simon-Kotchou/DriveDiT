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
                fh.write("TestHarness GetNSuites Begin\n")
            rc, out, _ = self.run_api("--planned", planned)
            self.assertEqual(rc, 1, out)
            self.assertIn("FAIL  TestHarness.GetNSuites REMOVED", out)
            self.assertIn("FAIL  TestHarness.Begin SIG_CHANGED", out)
            triage = os.path.join(d, "triage.txt")
            with open(triage, "w") as fh:
                fh.write("TestHarness.GetNSuites REMOVED obsolete, unused (AD 2.6)\n"
                         "TestHarness.Begin SIG_CHANGED notnull suites; our tests call ActiveSuite only\n")
            rc, out, _ = self.run_api("--planned", planned, "--triage", triage)
            self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
