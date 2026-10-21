"""Unit tests for tools/pin (ticket DD-0). Stdlib unittest only.

    python3 -I -B -m unittest discover -s tools/pin/tests -v

Temporary Steam-library trees go under $DD_PIN_TEST_TMP, else /dev/shm, else the system temp dir,
and are deleted afterwards. Nothing is ever written under /mnt/c: the C:-refusal tests use paths
whose parent does not exist, and the code refuses them before touching the filesystem.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_DIR = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures")
NEVER_C = "/mnt/c/__dd0_pin_test_never_written__"


def _load(name):
    path = os.path.join(PIN_DIR, name + ".py")
    modname = "dd_pin_" + name
    if modname in sys.modules and getattr(sys.modules[modname], "__file__", None) == path:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pinlib = _load("pinlib")
make_pin = _load("make_pin")
preflight = _load("preflight")

PIN_MTIME_NS = 1788671755 * 10**9          # 2026-09-06 (UTC), like the real paks
TOOLS_MTIME_NS = 1788671758_506855800      # sub-second, like the real Tools paks


def _tmp_base():
    for cand in (os.environ.get("DD_PIN_TEST_TMP"), "/dev/shm"):
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return None


def run_main(fn, argv, roots):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(argv, write_roots=roots)
    return rc, out.getvalue(), err.getvalue()


class FakeSteam:
    """Two fake Steam libraries laid out like D: (game) and C: (Tools)."""

    def __init__(self, base):
        self.base = base
        self.game_acf = os.path.join(base, "dlib", "steamapps", "appmanifest_1874880.acf")
        self.tools_acf = os.path.join(base, "clib", "steamapps", "appmanifest_1874910.acf")
        self.game_root = os.path.join(base, "dlib", "steamapps", "common", "Arma Reforger")
        self.tools_root = os.path.join(base, "clib", "steamapps", "common", "Arma Reforger Tools")
        self.out_dir = os.path.join(base, "out")
        self.pin = os.path.join(self.out_dir, "pin_1.8.0.13.json")
        os.makedirs(self.out_dir)
        for src, dst in ((os.path.join(FIXTURES, "appmanifest_1874880.acf"), self.game_acf),
                         (os.path.join(FIXTURES, "appmanifest_1874910.acf"), self.tools_acf)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        self.write(self.game_root, "ArmaReforgerSteam.exe", b"MZ game", PIN_MTIME_NS)
        self.write(self.game_root, "ArmaReforgerSteamDiag.exe", b"MZ game diag", PIN_MTIME_NS)
        self.write(self.game_root, "steam_appid.txt", b"1874880", PIN_MTIME_NS)
        self.write(self.game_root, "addons/core/data.pak", b"core", PIN_MTIME_NS)
        self.write(self.game_root, "addons/data/data.pak", b"data-0", PIN_MTIME_NS)
        self.write(self.game_root, "addons/data/data001.pak", b"data-1", PIN_MTIME_NS)
        self.write(self.tools_root, "steam_appid.txt", b"1874910\r\n", TOOLS_MTIME_NS)
        self.write(self.tools_root, "Workbench/steam_appid.txt", b"1874910\r\n", TOOLS_MTIME_NS)
        self.write(self.tools_root, "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ wb", TOOLS_MTIME_NS)
        self.write(self.tools_root, "Workbench/addons/core/data.pak", b"wbcore", TOOLS_MTIME_NS)

    @staticmethod
    def write(root, rel, data, mtime_ns):
        full = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(data)
        os.utime(full, ns=(mtime_ns, mtime_ns))
        return full

    def set_acf(self, app, key, value):
        path = self.game_acf if app == "game" else self.tools_acf
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        pat = re.compile(r'(\n\t"%s"\t\t")[^"]*(")' % re.escape(key))
        if value is None:
            text, n = re.subn(r'\n\t"%s"\t\t"[^"]*"' % re.escape(key), "", text)
        else:
            text, n = pat.subn(lambda m: m.group(1) + value + m.group(2), text)
        assert n == 1, (key, n)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    def set_auto_update(self, value):
        self.set_acf("game", "AutoUpdateBehavior", value)
        self.set_acf("tools", "AutoUpdateBehavior", value)

    def make_pin_argv(self, *extra):
        return ["--pin-id", "1.8.0.13", "--out", self.pin, "--game-acf", self.game_acf,
                "--tools-acf", self.tools_acf] + list(extra)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dd0pin-", dir=_tmp_base())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.roots = (self.tmp,)
        self.fs = FakeSteam(self.tmp)


# --------------------------------------------------------------------------- parser


class AcfParserTests(unittest.TestCase):
    def test_game_fixture(self):
        a = pinlib.load_acf(os.path.join(FIXTURES, "appmanifest_1874880.acf"))
        self.assertEqual(a["appid"], 1874880)
        self.assertEqual(a["buildid"], 24903726)
        self.assertEqual(a["target_buildid"], 24903726)
        self.assertEqual(a["auto_update_behavior"], "0")
        self.assertEqual(a["state_flags"], "4")
        self.assertEqual(a["installdir"], "Arma Reforger")
        self.assertEqual(a["depots"], {
            "1874881": {"manifest": "7768706146966449237", "size": 26946681496},
            "1874882": {"manifest": "1650233779375073015", "size": 233489401},
        })
        self.assertNotIn("LastOwner", json.dumps(a))

    def test_tools_fixture(self):
        a = pinlib.load_acf(os.path.join(FIXTURES, "appmanifest_1874910.acf"))
        self.assertEqual((a["appid"], a["buildid"], a["target_buildid"]), (1874910, 24870687, 24870687))
        self.assertEqual(a["depots"], {"1874911": {"manifest": "6937477522608233045", "size": 693405810}})

    def test_escapes_comments_case_and_conditionals(self):
        text = ('// header comment\n"AppState"\n{\n\t"appid"\t"1"\n'
                '\t"LauncherPath"\t\t"C:\\\\Program Files\\\\Steam\\\\steam.exe"\n'
                '\t"name"\t"say \\"hi\\""  // trailing comment\n'
                '\t"autoupdatebehavior"\t"1" [$WIN32]\n'
                '\t"Nested" { "k" "v" }\n}\n')
        root = pinlib.parse_vdf(text)
        app = pinlib.kv_get(root, "appstate")
        self.assertEqual(app["LauncherPath"], "C:\\Program Files\\Steam\\steam.exe")
        self.assertEqual(app["name"], 'say "hi"')
        self.assertEqual(pinlib.kv_get(app, "AutoUpdateBehavior"), "1")
        self.assertEqual(app["Nested"], {"k": "v"})

    def test_malformed(self):
        for bad in ('"A" { "k" "v"', '"A" { "k" }', '"A" "v" }', '"A" { "k" "v" } }', '"unterminated'):
            with self.assertRaises(pinlib.PinError, msg=bad):
                pinlib.parse_vdf(bad)

    def test_unreadable_acf(self):
        with self.assertRaises(pinlib.PinError):
            pinlib.load_acf("/nonexistent/appmanifest_0.acf")


# --------------------------------------------------------------------------- paths


class PathGuardTests(unittest.TestCase):
    def test_conversions(self):
        self.assertEqual(pinlib.win_to_wsl("E:\\DriveDiT\\pin\\x.json"), "/mnt/e/DriveDiT/pin/x.json")
        self.assertEqual(pinlib.win_to_wsl("e:/DriveDiT"), "/mnt/e/DriveDiT")
        self.assertEqual(pinlib.win_to_wsl("/mnt/e/x"), "/mnt/e/x")
        self.assertEqual(pinlib.wsl_to_win("/mnt/d/SteamLibrary/steamapps/common/Arma Reforger"),
                         "D:\\SteamLibrary\\steamapps\\common\\Arma Reforger")
        self.assertEqual(pinlib.wsl_to_win("/mnt/e"), "E:\\")
        self.assertIsNone(pinlib.wsl_to_win("/dev/shm/x"))

    def test_e_drive_allowed(self):
        self.assertEqual(pinlib.check_write_path("/mnt/e/DriveDiT/pin/pin.json"), "/mnt/e/DriveDiT/pin/pin.json")
        self.assertEqual(pinlib.check_write_path("E:\\DriveDiT\\pin\\pin.json"), "/mnt/e/DriveDiT/pin/pin.json")

    def test_c_drive_refused(self):
        for p in (NEVER_C + "/pin.json", "/mnt/C/x.json", "C:\\DriveDiT\\pin.json", "/mnt/c",
                  "/mnt/e/../c/x.json"):
            with self.assertRaises(pinlib.PinError, msg=p):
                pinlib.check_write_path(p)
            with self.assertRaises(pinlib.PinError, msg=p):
                pinlib.check_write_path(p, allowed_roots=("/mnt/", "/"))
        self.assertFalse(os.path.exists(NEVER_C))

    def test_outside_e_refused(self):
        for p in ("/dev/shm/x.json", "/tmp/x.json", "/home/x.json", "/mnt/d/x.json", "relative.json"):
            if p == "relative.json" and os.getcwd().startswith("/mnt/e/"):
                continue
            with self.assertRaises(pinlib.PinError, msg=p):
                pinlib.check_write_path(p)

    def test_symlink_to_c_refused(self):
        tmp = tempfile.mkdtemp(prefix="dd0pin-", dir=_tmp_base())
        self.addCleanup(shutil.rmtree, tmp, True)
        link = os.path.join(tmp, "evil")
        os.symlink(NEVER_C, link)  # dangling; never followed for writing
        with self.assertRaises(pinlib.PinError):
            pinlib.check_write_path(os.path.join(link, "pin.json"), allowed_roots=(tmp,))
        self.assertFalse(os.path.exists(NEVER_C))


# --------------------------------------------------------------------------- make_pin


class MakePinTests(TmpCase):
    def test_writes_pin(self):
        rc, out, err = run_main(make_pin.main, self.fs.make_pin_argv(
            "--expect-game-buildid", "24903726", "--expect-tools-buildid", "24870687"), self.roots)
        self.assertEqual(rc, 0, err)
        self.assertIn("wrote", out)
        self.assertIn("AutoUpdateBehavior is '0'", err)  # fixture is today's state: warn, still pin
        with open(self.fs.pin) as fh:
            pin = json.load(fh)
        self.assertEqual(pin["schema"], "dd-pin/1")
        self.assertEqual(pin["pin_id"], "1.8.0.13")
        g, t = pin["game"], pin["tools"]
        self.assertEqual((g["appid"], g["buildid"], t["appid"], t["buildid"]), (1874880, 24903726, 1874910, 24870687))
        self.assertEqual(g["depots"]["1874881"]["manifest"], "7768706146966449237")
        self.assertEqual(t["depots"]["1874911"]["manifest"], "6937477522608233045")
        self.assertEqual(sorted(g["paks"]), ["addons/core/data.pak", "addons/data/data.pak", "addons/data/data001.pak"])
        self.assertEqual(sorted(t["paks"]), ["Workbench/addons/core/data.pak"])
        self.assertEqual(g["paks"]["addons/data/data.pak"]["size"], 6)
        self.assertEqual(g["paks"]["addons/data/data.pak"]["mtime_ns"], PIN_MTIME_NS)
        # Compare with what the filesystem stored: drvfs (E:) truncates mtimes it sets to whole seconds.
        stored = os.stat(os.path.join(self.fs.tools_root, "Workbench", "addons", "core", "data.pak")).st_mtime_ns
        self.assertEqual(t["paks"]["Workbench/addons/core/data.pak"]["mtime_ns"], stored)
        self.assertEqual(sorted(g["binaries"]), ["ArmaReforgerSteam.exe", "ArmaReforgerSteamDiag.exe"])
        self.assertEqual(t["steam_appid_txt"], {"steam_appid.txt": "1874910", "Workbench/steam_appid.txt": "1874910"})
        self.assertEqual(g["auto_update_behavior_at_pin"], "0")
        self.assertEqual(pin["steam"]["auto_update_behavior_allowed"], ["1"])
        self.assertNotIn("LastOwner", json.dumps(pin))

    def test_refuses_c_drive_output(self):
        argv = ["--pin-id", "1.8.0.13", "--out", NEVER_C + "/pin.json",
                "--game-acf", self.fs.game_acf, "--tools-acf", self.fs.tools_acf]
        rc, _, err = run_main(make_pin.main, argv, pinlib.DEFAULT_WRITE_ROOTS)
        self.assertEqual(rc, 2)
        self.assertIn("C:", err)
        self.assertFalse(os.path.exists(NEVER_C))

    def test_refuses_output_off_e_by_default(self):
        if os.path.realpath(self.tmp).startswith("/mnt/e/"):
            self.skipTest("temp dir is on E:, so the default guard allows it")
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv(), pinlib.DEFAULT_WRITE_ROOTS)
        self.assertEqual(rc, 2)
        self.assertIn("outside", err)
        self.assertFalse(os.path.exists(self.fs.pin))

    def test_missing_output_dir(self):
        argv = self.fs.make_pin_argv()
        argv[argv.index("--out") + 1] = os.path.join(self.tmp, "nodir", "pin.json")
        rc, _, err = run_main(make_pin.main, argv, self.roots)
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "nodir")))

    def test_expected_buildid_mismatch(self):
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv("--expect-game-buildid", "1"), self.roots)
        self.assertEqual(rc, 1)
        self.assertIn("buildid", err)
        self.assertFalse(os.path.exists(self.fs.pin))

    def test_pending_update_not_pinnable(self):
        self.fs.set_acf("tools", "TargetBuildID", "99999999")
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)
        self.assertEqual(rc, 1)
        self.assertIn("pending", err)

    def test_unsettled_state_not_pinnable(self):
        self.fs.set_acf("game", "StateFlags", "6")
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)
        self.assertEqual(rc, 1)

    def test_wrong_appid(self):
        self.fs.set_acf("game", "appid", "1874900")
        rc, _, _ = run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)
        self.assertEqual(rc, 1)

    def test_no_overwrite_without_force(self):
        self.assertEqual(run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)[0], 0)
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)
        self.assertEqual(rc, 2)
        self.assertIn("immutable", err)
        self.assertEqual(run_main(make_pin.main, self.fs.make_pin_argv("--force"), self.roots)[0], 0)
        self.assertEqual([n for n in os.listdir(self.fs.out_dir) if ".tmp-" in n], [])


# --------------------------------------------------------------------------- preflight


class PreflightTests(TmpCase):
    def setUp(self):
        super().setUp()
        self.fs.set_auto_update("1")  # the state the user is asked to set (P0 step 1)
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)
        self.assertEqual(rc, 0, err)
        self.launches = os.path.join(self.fs.out_dir, "launches")

    def launch(self, *extra, exe="client"):
        return run_main(preflight.main, ["--pin", self.fs.pin, "--exe", exe] + list(extra), self.roots)

    def records(self, prefix):
        if not os.path.isdir(self.launches):
            return []
        return sorted(n for n in os.listdir(self.launches) if n.startswith(prefix))

    def assertRefused(self, *names):
        rc, out, err = self.launch()
        self.assertEqual(rc, 1, err)
        self.assertEqual(out, "", "nothing on stdout when refused (the .bat keys on it)")
        self.assertIn("PREFLIGHT REFUSED LAUNCH", err)
        for n in names:
            self.assertIn("FAIL %s:" % n, err)
        self.assertEqual(self.records("launch_"), [])
        refused = self.records("refused_")
        self.assertEqual(len(refused), 1)
        with open(os.path.join(self.launches, refused[0])) as fh:
            rec = json.load(fh)
        self.assertEqual(rec["status"], "refused")
        failed = sorted(c["name"] for c in rec["checks"] if not c["ok"])
        self.assertEqual(failed, sorted(names))
        return err

    def test_ok_launch_writes_record_and_prints_only_the_id(self):
        rc, out, err = self.launch("--instance", "i01")
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        self.assertEqual(len(lines), 1)
        lid = lines[0]
        self.assertRegex(lid, r"^\d{8}T\d{6}Z-client-[0-9a-f]{4}$")
        self.assertNotIn(" ", lid)
        self.assertEqual(self.records("launch_"), ["launch_%s.json" % lid])
        with open(os.path.join(self.launches, "launch_%s.json" % lid)) as fh:
            rec = json.load(fh)
        self.assertEqual(rec["schema"], "dd-launch/1")
        self.assertEqual((rec["status"], rec["exe"], rec["instance"], rec["launch_id"]), ("ok", "client", "i01", lid))
        self.assertEqual(rec["pin"]["sha256"], pinlib.sha256_file(self.fs.pin))
        self.assertTrue(all(c["ok"] for c in rec["checks"]))
        self.assertEqual(sorted(c["name"] for c in rec["checks"]), sorted(preflight.CHECK_NAMES))
        self.assertEqual(rec["observed"]["game"]["buildid"], 24903726)

    def test_launch_ids_unique(self):
        ids = {self.launch(exe="wb")[1].strip() for _ in range(5)}
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(self.records("launch_")), 5)

    def test_check_mode_writes_nothing(self):
        rc, out, _ = run_main(preflight.main, ["--pin", self.fs.pin, "--check"], self.roots)
        self.assertEqual(rc, 0)
        self.assertIn("RESULT: OK", out)
        self.assertFalse(os.path.exists(self.launches))

    def test_refuses_auto_update_always(self):
        self.fs.set_acf("game", "AutoUpdateBehavior", "0")
        err = self.assertRefused("game.auto_update_behavior")
        self.assertIn("AutoUpdateBehavior is '0'", err)
        self.assertIn("Only update this game when I launch it", err)

    def test_auto_update_zero_refused_even_if_pin_allows_it(self):
        with open(self.fs.pin) as fh:
            pin = json.load(fh)
        pin["steam"]["auto_update_behavior_allowed"] = ["0", "1"]
        with open(self.fs.pin, "w") as fh:
            json.dump(pin, fh)
        self.fs.set_acf("tools", "AutoUpdateBehavior", "0")
        self.assertRefused("tools.auto_update_behavior")

    def test_refuses_auto_update_high_priority(self):
        self.fs.set_acf("tools", "AutoUpdateBehavior", "2")
        self.assertRefused("tools.auto_update_behavior")

    def test_refuses_missing_auto_update_key(self):
        self.fs.set_acf("tools", "AutoUpdateBehavior", None)
        self.assertRefused("tools.auto_update_behavior")

    def test_refuses_new_build(self):
        self.fs.set_acf("game", "buildid", "25000000")
        self.fs.set_acf("game", "TargetBuildID", "25000000")
        self.assertRefused("game.buildid")

    def test_refuses_pending_update(self):
        self.fs.set_acf("tools", "TargetBuildID", "25000000")
        self.assertRefused("tools.target_buildid")

    def test_refuses_state_flags(self):
        self.fs.set_acf("game", "StateFlags", "6")
        self.assertRefused("game.state_flags")

    def test_refuses_manifest_change(self):
        path = self.fs.game_acf
        with open(path) as fh:
            text = fh.read()
        with open(path, "w", newline="\n") as fh:
            fh.write(text.replace("7768706146966449237", "1111111111111111111"))
        self.assertRefused("game.depots")

    def test_refuses_pak_size(self):
        self.fs.write(self.fs.game_root, "addons/data/data.pak", b"data-0-longer", PIN_MTIME_NS)
        err = self.assertRefused("game.paks")
        self.assertIn("addons/data/data.pak: size 13 != pin 6", err)

    def test_refuses_pak_mtime(self):
        self.fs.write(self.fs.tools_root, "Workbench/addons/core/data.pak", b"wbcore", TOOLS_MTIME_NS + 2 * 10**9)
        err = self.assertRefused("tools.paks")
        self.assertIn("mtime", err)

    def test_refuses_new_pak(self):
        self.fs.write(self.fs.game_root, "addons/data/data011.pak", b"new", PIN_MTIME_NS)
        err = self.assertRefused("game.paks")
        self.assertIn("not in pin", err)

    def test_refuses_missing_pak(self):
        os.unlink(os.path.join(self.fs.game_root, "addons", "core", "data.pak"))
        err = self.assertRefused("game.paks")
        self.assertIn("missing", err)

    def test_refuses_exe_change(self):
        self.fs.write(self.fs.tools_root, "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ wb2", TOOLS_MTIME_NS)
        self.assertRefused("tools.binaries")

    def test_refuses_steam_appid_change(self):
        os.unlink(os.path.join(self.fs.game_root, "steam_appid.txt"))
        self.assertRefused("game.steam_appid_txt")

    def test_unreadable_acf_refuses(self):
        os.unlink(self.fs.game_acf)
        rc, out, err = self.launch()
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("FAIL game.acf:", err)

    def test_allow_fail_only_in_check_mode(self):
        self.fs.set_auto_update("0")
        rc, out, err = self.launch("--allow-fail", "game.auto_update_behavior")
        self.assertEqual((rc, out), (2, ""))
        rc, out, _ = run_main(preflight.main, ["--pin", self.fs.pin, "--check",
                                               "--allow-fail", "game.auto_update_behavior",
                                               "--allow-fail", "tools.auto_update_behavior"], self.roots)
        self.assertEqual(rc, 0, out)
        self.assertIn("WAIVED", out)
        rc, out, _ = run_main(preflight.main, ["--pin", self.fs.pin, "--check"], self.roots)
        self.assertEqual(rc, 1)
        self.assertIn("RESULT: REFUSED", out)

    def test_unknown_allow_fail_name(self):
        rc, _, _ = run_main(preflight.main, ["--pin", self.fs.pin, "--check", "--allow-fail", "game.nope"], self.roots)
        self.assertEqual(rc, 2)

    def test_exe_required_for_launch(self):
        rc, out, _ = run_main(preflight.main, ["--pin", self.fs.pin], self.roots)
        self.assertEqual((rc, out), (2, ""))

    def test_bad_pin(self):
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w") as fh:
            fh.write("{not json")
        rc, out, _ = run_main(preflight.main, ["--pin", bad, "--exe", "wb"], self.roots)
        self.assertEqual((rc, out), (2, ""))
        with open(bad, "w") as fh:
            json.dump({"schema": "other"}, fh)
        rc, out, _ = run_main(preflight.main, ["--pin", bad, "--exe", "wb"], self.roots)
        self.assertEqual((rc, out), (2, ""))

    def test_launch_dir_on_c_refused(self):
        rc, out, err = run_main(preflight.main, ["--pin", self.fs.pin, "--exe", "wb", "--launch-dir", NEVER_C],
                                pinlib.DEFAULT_WRITE_ROOTS)
        self.assertEqual((rc, out), (2, ""))
        self.assertFalse(os.path.exists(NEVER_C))

    def test_root_override_for_cold_copy(self):
        # cold_copy.sh points preflight at the copy with --game-root/--tools-root and 1 s mtime resolution.
        copy = os.path.join(self.tmp, "copy")
        shutil.copytree(self.fs.game_root, os.path.join(copy, "game"), copy_function=shutil.copy2)
        shutil.copytree(self.fs.tools_root, os.path.join(copy, "tools"), copy_function=shutil.copy2)
        # Simulate drvfs: the copy's sub-second mtime is truncated to whole seconds.
        self.fs.write(os.path.join(copy, "tools"), "Workbench/addons/core/data.pak", b"wbcore",
                      TOOLS_MTIME_NS // 10**9 * 10**9)
        argv = ["--pin", self.fs.pin, "--check", "--game-root", os.path.join(copy, "game"),
                "--tools-root", os.path.join(copy, "tools")]
        rc, out, _ = run_main(preflight.main, argv + ["--mtime-resolution-s", "1"], self.roots)
        self.assertEqual(rc, 0, out)
        with open(self.fs.pin) as fh:
            pinned_ns = json.load(fh)["tools"]["paks"]["Workbench/addons/core/data.pak"]["mtime_ns"]
        if pinned_ns % 10**9:  # original kept its sub-second part (tmpfs): an exact compare must see the cut
            rc, out, _ = run_main(preflight.main, argv, self.roots)
            self.assertEqual(rc, 1, out)
        os.unlink(os.path.join(copy, "tools", "Workbench", "addons", "core", "data.pak"))
        rc, out, _ = run_main(preflight.main, argv + ["--mtime-resolution-s", "1"], self.roots)
        self.assertEqual(rc, 1)

    def test_mtime_resolution_only_in_check_mode(self):
        rc, out, _ = self.launch("--mtime-resolution-s", "1")
        self.assertEqual((rc, out), (2, ""))

    def test_cli_isolated_mode(self):
        # As the launchers run it: python3 -I -B <abs path>. -I drops the script dir from sys.path,
        # so this proves the explicit pinlib loader works.
        script = os.path.join(PIN_DIR, "preflight.py")
        r = subprocess.run([sys.executable, "-I", "-B", script, "--pin", self.fs.pin, "--check"],
                           capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("RESULT: OK", r.stdout)
        r = subprocess.run([sys.executable, "-I", "-B", script, "--pin", self.fs.pin, "--exe", "client",
                            "--launch-dir", NEVER_C], capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertFalse(os.path.exists(NEVER_C))


if __name__ == "__main__":
    unittest.main()
