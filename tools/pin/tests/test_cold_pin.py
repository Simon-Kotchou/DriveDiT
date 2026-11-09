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
lc, CFG = base.lc, base.CFG   # the synthetic local config every test here reads (DD-22)
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
            base.override(self, pinlib, name, value)

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
        self.assertEqual(len(rec["checks"]), 26)   # the 22, exe_sha256 and the mandatory pak_sha256 of both apps
        self.assertIn("pak sha256 match", [c for c in rec["checks"] if c["name"] == "game.pak_sha256"][0]["detail"])
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

    def test_pak_changed_with_size_and_mtime_kept_refused_at_launch(self):
        # Codex round 1: the size and mtime are preserved, so only the hash can tell
        self.assertEqual(self.make_cold()[0], 0)
        self.fs.write(os.path.join(self.run_root, "game"), "addons/data/data.pak", b"data-X", base.PIN_MTIME_NS)
        rc, out, err, _ = self.launch(self.pin_file("1.8.0.13-cold"), "--launch-path", self.LP)
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("FAIL game.pak_sha256", err)
        self.assertNotIn("FAIL game.paks", err)

    def test_changed_reference_sums_refused(self):
        self.assertEqual(self.make_cold()[0], 0)
        with open(os.path.join(self.ref_copy, "SHA256SUMS"), "a") as fh:
            fh.write("%s  game/extra.txt\n" % ("0" * 64))
        rc, out, err, _ = self.launch(self.pin_file("1.8.0.13-cold"), "--launch-path", self.LP)
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("changed since the cold pin was made", err)

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


LAUNCHERS = ("wb_R1.bat", "client_R1_i01.bat", "wb_C18.bat", "client_C18_i01.bat")
HELPER = "dd_local_config.cmd"


def bat_vars(name, cfg=None):
    """The launcher's set variables, read statically, seeded with what dd_local_config.cmd sets for cfg (DD-22)."""
    return lc.bat_env(bat(name), lc.bat_values(cfg or CFG))


def resolved_start(name, cfg=None):
    return lc.bat_start_line(bat(name), lc.bat_values(cfg or CFG))


def golden_lines():
    with open(os.path.join(FIXTURES, "golden_start_lines.txt"), encoding="ascii") as fh:
        return dict(line.rstrip("\n").split("\t", 1) for line in fh if line.strip() and not line.startswith("#"))


class LauncherTests(unittest.TestCase):
    def test_golden_stable_start_lines(self):
        # The DD-14a golden lines, kept byte for byte as templates over the local config (DD-22).
        golden = golden_lines()
        self.assertEqual(sorted(golden), ["client_R1_i01.bat", "wb_R1.bat"])
        for name, template in golden.items():
            self.assertEqual(resolved_start(name), lc.render(template, CFG), name)
        self.assertEqual(lc.render(golden["wb_R1.bat"], CFG).split(" -profile ")[0],
                         'start "" /D "Z:\\Synthetic Steam (x86)\\steamapps\\common\\Arma Reforger Tools\\Workbench" '
                         '"Z:\\Synthetic Steam (x86)\\steamapps\\common\\Arma Reforger Tools\\Workbench\\'
                         'ArmaReforgerWorkbenchSteamDiag.exe"')

    def test_golden_lines_follow_any_config(self):
        # Another config (other drives, other names) moves every path of the start lines and nothing else.
        with open(base.LOCAL_CONFIG_FIXTURE, encoding="ascii") as fh:
            text = fh.read()
        other = lc.parse(text.replace("/mnt/x/DataRoot", "/mnt/y/Other-Root").replace("/mnt/y/Game Library", "/mnt/x/G"),
                         "other.env")
        for name, template in golden_lines().items():
            self.assertEqual(resolved_start(name, other), lc.render(template, other), name)
            self.assertNotIn("X:\\DataRoot", resolved_start(name, other))

    def test_launchers_name_no_machine_path(self):
        # Every machine path and the distro come from the local config (DD-22); a launcher names none itself.
        for name in LAUNCHERS:
            code = [l for l in bat(name).splitlines() if not l.lower().startswith("rem")]
            for line in code:
                self.assertIsNone(re.search(r"(?<![A-Za-z0-9_%/])[A-Za-z]:\\|/mnt/[a-z]/", line), (name, line))
            text = "\n".join(code)
            self.assertIn("wsl.exe -d %DD_WSL_DISTRO% -e /usr/bin/python3 -I -B %DD_DATA_ROOT%/tools/pin/preflight.py",
                          text, name)
            self.assertEqual(re.findall(r"wsl\.exe -d (\S+)", text), ["%DD_WSL_DISTRO%"] * 2, name)
            # the helper is called right after setlocal, before any DD_ value is used
            call = text.index('call "%~dp0dd_local_config.cmd"')
            self.assertLess(text.index("setlocal"), call, name)
            self.assertLess(call, text.index("%DD_DATA_ROOT"), name)
            self.assertIn('call "%~dp0dd_local_config.cmd"\nif errorlevel 1 goto :fail\n', text, name)

    def test_echo_copies_start_nothing(self):
        # tools/local_config.py check --cmd runs these through cmd.exe: they must only set and print, never run a
        # check, wsl.exe or an exe.
        for name in LAUNCHERS:
            body, start, wsl = lc._echo_copy(bat(name))
            lines = body.split("\r\n")
            self.assertIn('call "%~dp0dd_local_config.cmd"', lines, name)
            self.assertEqual([l for l in lines if l.startswith("echo ")], ["echo WSL: " + wsl,
                                                                           "echo " + start.replace("%LID%", "^<LID^>"),
                                                                           "echo FAIL %DD_ERR%"])
            for line in lines:
                low = line.lower()
                self.assertFalse(line.startswith(("start ", "for /f", "tasklist", "copy ", "mkdir ", "wsl.exe")) or
                                 (" mkdir " in low and not low.startswith("rem")), (name, line))

    def test_helper_sets_what_the_python_twin_computes(self):
        # tools/local_config.py bat_values is the Python twin of the helper; both list the same DD_ values, and the
        # helper's character classes are those of the Python loader (tools/local_config.py check --cmd runs the real
        # helper through cmd.exe and compares the values).
        text = bat(HELPER)
        for var in lc.bat_values(CFG):
            if var != "DD_LOCAL_CONFIG":
                self.assertRegex(text, r'(?m)^set "%s=' % var)
        classes = dict(re.findall(r'findstr /R /X /C:"DDLC_([A-Z_]+)=\[([^\]]+)\]\*"', text))
        self.assertEqual(sorted(classes), ["DATA_ROOT", "GAME_INSTALL", "TOOLS_INSTALL", "WSL_DISTRO"])
        printable = [chr(c) for c in range(32, 127)]
        for key, cls in classes.items():
            allowed = {c for c in printable if c in cls}
            probe = "/mnt/x/a%sb/steamapps/common/c" if key.endswith("INSTALL") else "/mnt/x/a%sb" if key == "DATA_ROOT" \
                else "a%sb"
            py = {c for c in printable if lc._value_problem(key, probe % c) is None}
            self.assertEqual(allowed, py, key)

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
            self.assertEqual(env["DD_PIN"], CFG.get("DATA_ROOT") + "/pin/pin_1.8.0.13-cold.json")
        self.assertEqual(REAL_COLD_RUN_ROOTS[lp], CFG.get("DATA_ROOT") + "/pin/1.8.0.13-run")
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
        # R1: a normal Windows environment points TEMP/TMP at the user's profile on the system drive.
        root = CFG.win("DATA_ROOT")
        for name, want in (("wb_C18.bat", root + "\\tmp\\wb_c18"), ("client_C18_i01.bat", root + "\\tmp\\c18_i01")):
            text, env = bat(name), bat_vars(name)
            self.assertEqual((env["TEMP"], env["TMP"]), (want, want), name)
            self.assertLess(text.index('set "TEMP='), text.index("wsl.exe -d"), name)
            self.assertLess(text.index('set "TMP='), text.index('start "" '), name)
            self.assertIn('if not exist "%DD_TEMP%\\" mkdir "%DD_TEMP%"', text)
            self.assertIn('if /I not "%DD_TEMP:~0,3%"=="%DD_DATA_DRIVE_WIN%\\" (', text)
            self.assertLess(text.index('if not exist "%DD_TEMP%\\" ('), text.index('set "TEMP='))

    def test_client_copies_the_record_into_its_profile(self):
        text = bat("client_C18_i01.bat")
        self.assertIn('set "DC_DC_LAUNCH=%DC_PROFILE%\\profile\\dc_launch"', text)
        fc = 'fc /b "%DD_LAUNCHES%\\launch_%LID%.json" "%DC_DC_LAUNCH%\\launch_%LID%.json"'
        self.assertIn(fc, text)
        self.assertLess(text.index(fc), text.index('start "" '))
        self.assertEqual(bat_vars("client_C18_i01.bat")["DC_DC_LAUNCH"],
                         CFG.win("DATA_ROOT") + "\\client_profiles\\c18_i01\\profile\\dc_launch")

    def test_crlf(self):
        for name in LAUNCHERS + (HELPER,):
            with open(os.path.join(LAUNCH_DIR, name), "rb") as fh:
                data = fh.read()
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), name)

    def test_launch_path_ids(self):
        self.assertEqual(pinlib.launch_path_ids(), ("stable", "experimental", "cold-1.8.0.13"))
        with self.assertRaises(pinlib.PinError):
            pinlib.launch_paths_for("nonsense")
        self.assertEqual(pinlib.launch_paths_for("stable"), pinlib.LAUNCH_PATHS)


if __name__ == "__main__":
    unittest.main()
