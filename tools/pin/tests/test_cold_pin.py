"""Unit tests for the cold-run pin of DD-14a (MIGRATION_1_9 7.1): make_pin.py --channel cold, the cold launch
paths, wb_C18.bat and client_C18_i01.bat, and the golden start lines of the stable launchers. Stdlib unittest only.

    python3 -I -B -m unittest discover -s tools/pin/tests -v

Temp trees as in test_pin.py ($DD_PIN_TEST_TMP, else /dev/shm), deleted afterwards. The DD-0 tests in test_pin.py are
unchanged; this file reuses their fake Steam layout and test_pin_dd14a.py's helpers.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
TOOLS_DIR = os.path.dirname(os.path.dirname(HERE))
LAUNCH_DIR = os.path.join(TOOLS_DIR, "launch")


def _load_path(name, path):
    if name in __import__("sys").modules and getattr(__import__("sys").modules[name], "__file__", None) == path:
        return __import__("sys").modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    __import__("sys").modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The DD-0 test helpers (FakeSteam, run_main, TmpCase, PinnedCase), loaded under a private name so its test classes
# are not collected twice.
base = _load_path("dd_pin_test_pin_helpers", os.path.join(HERE, "test_pin.py"))
pinlib, make_pin, preflight, pin_activate = base.pinlib, base.make_pin, base.preflight, base.pin_activate
pin_sidecar = base._load("pin_sidecar")
run_main = base.run_main
dd14a = _load_path("dd_pin_test_pin_dd14a_helpers", os.path.join(HERE, "test_pin_dd14a.py"))
V2Case, spec_build_key, sha, read_json = dd14a.V2Case, dd14a.spec_build_key, dd14a.sha, dd14a.read_json
REAL_COLD_RUN_ROOTS = dict(pinlib.COLD_RUN_ROOTS)
REAL_COLD_LAUNCH_PATHS = {k: dict(v) for k, v in pinlib.COLD_LAUNCH_PATHS.items()}


# --------------------------------------------------------------------------- cold-run pin


class ColdPinTests(V2Case):
    LP = "cold-1.8.0.13"

    def setUp(self):
        super().setUp()
        self.ref_copy = dd14a.SidecarTests.make_copy(self, "1.8.0.13")
        self.run_root = os.path.join(self.pin_dir, "1.8.0.13-run")
        os.makedirs(os.path.join(self.run_root, "meta"))
        for app in ("game", "tools"):
            shutil.copytree(os.path.join(self.ref_copy, app), os.path.join(self.run_root, app))
        shutil.copyfile(self.fs.game_acf, os.path.join(self.run_root, "meta", "appmanifest_1874880.acf"))
        shutil.copyfile(self.fs.tools_acf, os.path.join(self.run_root, "meta", "appmanifest_1874910.acf"))
        for name, value in (("COLD_RUN_ROOTS", {self.LP: self.run_root}),
                            ("COLD_LAUNCH_PATHS", {self.LP: pinlib._cold_paths(self.run_root)})):
            saved = getattr(pinlib, name)
            setattr(pinlib, name, value)
            self.addCleanup(setattr, pinlib, name, saved)

    def make_cold(self, *extra):
        return run_main(make_pin.main, ["--channel", "cold", "--pin-id", "1.8.0.13-cold", "--cold-of", "1.8.0.13",
                                        "--from-copy", self.run_root, "--out", self.pin_file("1.8.0.13-cold")]
                        + list(extra), self.roots)

    def test_cold_pin_gets_the_reference_build_key(self):
        rc, out, err = self.make_cold()
        self.assertEqual(rc, 0, err)
        pin = read_json(self.pin_file("1.8.0.13-cold"))
        self.assertEqual((pin["schema"], pin["channel"], pin["release_channel"], pin["launch_path_id"]),
                         ("dd-pin/2", "cold", "stable", self.LP))
        want = spec_build_key(1874880, 24903726, sha(b"MZ game"), 1874910, 24870687, sha(b"MZ wb"))
        self.assertEqual((pin["build_key"], pin["cold_of"]["build_key"]), (want, want))
        self.assertIn("computed from", pin["cold_of"]["build_key_source"])  # dd-pin/1 reference without a sidecar
        self.assertEqual(pin["client_args_required"], ["-noBackend"])
        self.assertEqual(pin["game"]["install"]["wsl"], os.path.join(self.run_root, "game"))
        self.assertTrue(all("sha256" in p for p in pin["game"]["paks"].values()))

    def test_cold_launch_and_activation(self):
        self.assertEqual(self.make_cold()[0], 0)
        rc, _, err, rec = self.launch(self.pin_file("1.8.0.13-cold"), "--launch-path", self.LP, exe="client")
        self.assertEqual(rc, 0, err)
        self.assertEqual((rec["channel"], rec["pin_channel"], rec["launch_path_id"]), ("stable", "cold", self.LP))
        self.assertEqual(len(rec["checks"]), 24)
        # a stable launcher refuses it
        rc, out, err, _ = self.launch(self.pin_file("1.8.0.13-cold"))
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("launch path 'cold-1.8.0.13'", err)
        # it can be activated (the preferred rollback), against its own copy
        rc, out, err = self.activate("1.8.0.13-cold")
        self.assertEqual(rc, 0, out + err)
        rc, out, err = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13-cold"), "--launch-path", self.LP,
                                                 "--check", "--verify-hashes"], self.roots)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("PASS   game.pak_sha256", out)

    def test_copy_not_matching_sums_refused(self):
        self.fs.write(os.path.join(self.run_root, "game"), "addons/data/data.pak", b"data-X", base.PIN_MTIME_NS)
        rc, _, err = self.make_cold()
        self.assertEqual(rc, 1, err)
        self.assertIn("does not match SHA256SUMS", err)
        self.assertFalse(os.path.exists(self.pin_file("1.8.0.13-cold")))

    def test_build_key_must_equal_the_reference(self):
        # a reference sidecar that names another key (as if the copy were another build)
        ref = preflight.load_pin_by_id(self.pin_dir, "1.8.0.13")
        exe = {"game": {"ArmaReforgerSteam.exe": "1" * 64, "ArmaReforgerSteamDiag.exe": "2" * 64},
               "tools": {"Workbench/ArmaReforgerWorkbenchSteamDiag.exe": "3" * 64}}
        side = pin_sidecar.make_build_key_sidecar(ref, exe, {"source": "test"}, 60)
        pinlib.write_json_new(os.path.join(self.pin_dir, "pin_1.8.0.13.build_key.json"), side, self.roots)
        rc, _, err = self.make_cold()
        self.assertEqual(rc, 1, err)
        self.assertIn("differs from the reference pin", err)

    def test_unknown_copy_and_the_reference_copy_refused(self):
        rc, _, err = run_main(make_pin.main, ["--channel", "cold", "--pin-id", "1.8.0.13-cold", "--cold-of", "1.8.0.13",
                                              "--from-copy", self.ref_copy, "--out", self.pin_file("1.8.0.13-cold")],
                              self.roots)
        self.assertEqual(rc, 2, err)
        self.assertIn("not a cold run root", err)
        pinlib.COLD_RUN_ROOTS = {self.LP: self.ref_copy}
        rc, _, err = run_main(make_pin.main, ["--channel", "cold", "--pin-id", "1.8.0.13-cold", "--cold-of", "1.8.0.13",
                                              "--from-copy", self.ref_copy, "--out", self.pin_file("1.8.0.13-cold")],
                              self.roots)
        self.assertEqual(rc, 2, err)
        self.assertIn("pristine restore source", err)

    def test_changed_copy_after_pinning_refused_at_launch(self):
        self.assertEqual(self.make_cold()[0], 0)
        self.fs.write(os.path.join(self.run_root, "tools"), "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ wc",
                      base.TOOLS_MTIME_NS)
        rc, out, err, _ = self.launch(self.pin_file("1.8.0.13-cold"), "--launch-path", self.LP)
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("FAIL tools.exe_sha256", err)

    def test_usage_errors(self):
        for argv in (["--channel", "cold", "--pin-id", "1.8.0.13-cold", "--out", self.pin_file("x")],
                     ["--pin-id", "1.8.0.13-cold", "--cold-of", "1.8.0.13", "--out", self.pin_file("1.8.0.13-cold")],
                     ["--channel", "cold", "--pin-id", "1.8.0.13", "--cold-of", "1.8.0.13", "--from-copy",
                      self.run_root, "--out", self.pin_file("1.8.0.13")]):
            rc, _, err = run_main(make_pin.main, argv, self.roots)
            self.assertEqual(rc, 2, (argv, err))


# --------------------------------------------------------------------------- launchers and launch paths


def bat(name):
    with open(os.path.join(LAUNCH_DIR, name), "rb") as fh:
        return fh.read().decode("ascii")


def bat_vars(name):
    env = {}
    for line in bat(name).splitlines():
        m = re.match(r'^set "([A-Za-z_][A-Za-z0-9_]*)=([^"]*)"$', line)
        if m:
            env[m.group(1)] = re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%",
                                     lambda mm: env.get(mm.group(1), mm.group(0)), m.group(2))
    return env


def resolved_start(name):
    env = bat_vars(name)
    starts = [line for line in bat(name).splitlines() if line.startswith('start "" ')]
    assert len(starts) == 1, name
    return re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%",
                  lambda m: "<LID>" if m.group(1) == "LID" else env.get(m.group(1), m.group(0)), starts[0])


class LauncherTests(unittest.TestCase):
    def test_golden_stable_start_lines(self):
        with open(os.path.join(FIXTURES, "golden_start_lines.txt"), encoding="ascii") as fh:
            golden = dict(line.rstrip("\n").split("\t", 1) for line in fh if line.strip() and not line.startswith("#"))
        self.assertEqual(sorted(golden), ["client_R1_i01.bat", "wb_R1.bat"])
        for name, line in golden.items():
            self.assertEqual(resolved_start(name), line, name)

    def test_cold_launchers_match_the_cold_launch_paths(self):
        lp = "cold-1.8.0.13"
        paths = REAL_COLD_LAUNCH_PATHS[lp]
        wb, cl = bat_vars("wb_C18.bat"), bat_vars("client_C18_i01.bat")
        self.assertEqual(pinlib.win_to_wsl(wb["WB_DIR"]), paths["tools"]["install"] + "/Workbench")
        self.assertEqual(pinlib.win_to_wsl(wb["GAME_ADDONS"]), paths["game"]["install"] + "/addons")
        self.assertEqual(pinlib.win_to_wsl(cl["GAME_DIR"]), paths["game"]["install"])
        self.assertEqual((wb["WB_PROFILE"], wb["WB_GPROJ"]), (pinlib.WB_LAUNCH[lp]["profile"],
                                                             pinlib.WB_LAUNCH[lp]["gproj"]))
        self.assertEqual(pinlib.win_to_wsl(cl["DC_ADDONS_DIR"]), pinlib.DEPLOY_ROOTS[lp])
        self.assertEqual(pinlib.win_to_wsl(wb["WB_GPROJ"]), pinlib.DEPLOY_ROOTS[lp] + "/DataCapture/addon.gproj")
        for env in (wb, cl):
            self.assertEqual(env["DD_LAUNCH_PATH"], lp)
            self.assertEqual(env["DD_PIN"], "/mnt/e/DriveDiT/pin/pin_1.8.0.13-cold.json")
        self.assertEqual(REAL_COLD_RUN_ROOTS[lp], "/mnt/e/DriveDiT/pin/1.8.0.13-run")
        self.assertEqual(paths["game"]["acf"], REAL_COLD_RUN_ROOTS[lp] + "/meta/appmanifest_1874880.acf")

    def test_stable_wb_launch_table_matches_wb_r1(self):
        env = bat_vars("wb_R1.bat")
        self.assertEqual((env["WB_PROFILE"], env["WB_GPROJ"]), (pinlib.WB_LAUNCH["stable"]["profile"],
                                                               pinlib.WB_LAUNCH["stable"]["gproj"]))
        self.assertEqual(pinlib.win_to_wsl(bat_vars("client_R1_i01.bat")["DC_ADDONS_DIR"]), pinlib.DEPLOY_ROOTS["stable"])

    def test_cold_launchers_pass_the_explicit_pin_and_forward_the_id(self):
        for name, exe in (("wb_C18.bat", "wb"), ("client_C18_i01.bat", "client")):
            text = bat(name)
            self.assertNotIn("--pin-active", text, name)
            self.assertIn("preflight.py --pin %%DD_PIN%% --launch-path %%DD_LAUNCH_PATH%% --exe %s" % exe, text)
            self.assertIn(" -dcLaunchId <LID>", resolved_start(name))
        self.assertIn('fc /b "%DD_LAUNCHES%\\launch_%LID%.json" "%WB_DC_LAUNCH%\\launch_%LID%.json"',
                      bat("wb_C18.bat"))

    def test_no_backend_only_on_the_cold_client(self):
        self.assertIn(" -noBackend ", resolved_start("client_C18_i01.bat"))
        for name in ("client_R1_i01.bat", "wb_R1.bat", "wb_C18.bat"):
            self.assertNotIn("-noBackend", resolved_start(name), name)

    def test_temp_and_tmp_on_e_before_wsl_and_start(self):
        # R1: a normal Windows environment points TEMP/TMP at the C: profile.
        for name, want in (("wb_C18.bat", "E:\\DriveDiT\\tmp\\wb_c18"), ("client_C18_i01.bat", "E:\\DriveDiT\\tmp\\c18_i01")):
            text, env = bat(name), bat_vars(name)
            self.assertEqual((env["TEMP"], env["TMP"]), (want, want), name)
            self.assertLess(text.index('set "TEMP='), text.index("wsl.exe -d"), name)
            self.assertLess(text.index('set "TMP='), text.index('start "" '), name)
            self.assertIn('if not exist "%DD_TEMP%\\" mkdir "%DD_TEMP%"', text)
            self.assertIn('if /I not "%DD_TEMP:~0,3%"=="E:\\" (', text)
            self.assertLess(text.index('if not exist "%DD_TEMP%\\" ('), text.index('set "TEMP='))

    def test_client_copies_the_record_into_its_profile(self):
        text = bat("client_C18_i01.bat")
        self.assertIn('set "DC_DC_LAUNCH=%DC_PROFILE%\\profile\\dc_launch"', text)
        fc = 'fc /b "%DD_LAUNCHES%\\launch_%LID%.json" "%DC_DC_LAUNCH%\\launch_%LID%.json"'
        self.assertIn(fc, text)
        self.assertLess(text.index(fc), text.index('start "" '))
        self.assertEqual(bat_vars("client_C18_i01.bat")["DC_DC_LAUNCH"], "E:\\DriveDiT\\client_profiles\\c18_i01\\profile\\dc_launch")

    def test_crlf(self):
        for name in ("wb_C18.bat", "client_C18_i01.bat"):
            with open(os.path.join(LAUNCH_DIR, name), "rb") as fh:
                data = fh.read()
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), name)

    def test_launch_path_ids(self):
        self.assertEqual(pinlib.launch_path_ids(), ("stable", "experimental", "cold-1.8.0.13"))
        with self.assertRaises(pinlib.PinError):
            pinlib.launch_paths_for("nonsense")
        self.assertIs(pinlib.launch_paths_for("stable"), pinlib.LAUNCH_PATHS)


if __name__ == "__main__":
    unittest.main()
