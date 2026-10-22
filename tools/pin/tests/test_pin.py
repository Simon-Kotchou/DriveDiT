"""Unit tests for tools/pin (ticket DD-0). Stdlib unittest only.

    python3 -I -B -m unittest discover -s tools/pin/tests -v

Temporary Steam-library trees go under $DD_PIN_TEST_TMP, else /dev/shm, else the system temp dir,
and are deleted afterwards. No /mnt/c path is ever handed to code that writes: real /mnt/c strings
appear only in the pure path-guard tests (check_write_path never writes). Tests that run make_pin or
preflight use a stand-in forbidden prefix inside the temp dir (TmpCase.fake_c_drive), or /proc.
TmpCase points pinlib.LAUNCH_PATHS (the installs the launchers start) at the fake Steam layout; tests that
run a script in a subprocess see the real launch paths, so the fake pins fail <app>.launch_path there.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_DIR = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures")
NEVER_C = "/mnt/c/__dd0_pin_test_never_written__"   # pure path-guard tests only; never passed to make_pin/preflight
NEVER_WRITABLE = "/proc/__dd0_pin_test_never_written__"  # outside /mnt/e, and the kernel refuses writes there


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
pin_activate = _load("pin_activate")
COLD_COPY = os.path.join(PIN_DIR, "cold_copy.sh")
LAUNCH_DIR = os.path.join(os.path.dirname(PIN_DIR), "launch")
REAL_LAUNCH_PATHS = pinlib.LAUNCH_PATHS  # captured before any test points it at a fake layout

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

    def launch_paths(self):
        """pinlib.LAUNCH_PATHS for this layout: the fake installs play the role of the D: and C: installs."""
        return {"game": {"acf": self.game_acf, "install": self.game_root},
                "tools": {"acf": self.tools_acf, "install": self.tools_root}}


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dd0pin-", dir=_tmp_base())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.roots = (self.tmp,)
        self.fs = FakeSteam(self.tmp)
        saved = pinlib.LAUNCH_PATHS
        pinlib.LAUNCH_PATHS = self.fs.launch_paths()
        self.addCleanup(setattr, pinlib, "LAUNCH_PATHS", saved)

    def fake_c_drive(self):
        """A stand-in for C: inside the temp dir. The guard refuses it exactly like /mnt/c, even though the
        temp dir is an allowed root, so the refusal paths of make_pin/preflight are tested without ever
        handing them a real /mnt/c path."""
        self.assertIs(make_pin.pinlib, pinlib)
        self.assertIs(preflight.pinlib, pinlib)
        self.assertIs(pin_activate.pinlib, pinlib)
        fake = os.path.join(self.tmp, "fake_c_drive")
        saved = pinlib.FORBIDDEN_WRITE_PREFIXES
        pinlib.FORBIDDEN_WRITE_PREFIXES = saved + (os.path.realpath(fake).lower() + "/",)
        self.addCleanup(setattr, pinlib, "FORBIDDEN_WRITE_PREFIXES", saved)
        return fake


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
        fake_c = self.fake_c_drive()
        os.makedirs(fake_c)  # the parent exists, so only the guard can stop the write
        argv = ["--pin-id", "1.8.0.13", "--out", os.path.join(fake_c, "pin.json"),
                "--game-acf", self.fs.game_acf, "--tools-acf", self.fs.tools_acf]
        rc, _, err = run_main(make_pin.main, argv, self.roots)
        self.assertEqual(rc, 2)
        self.assertIn("C:", err)
        self.assertEqual(os.listdir(fake_c), [])

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

    def test_invalid_pin_id_refused(self):
        # The id names pin_<id>.json and the cold-copy dir <pin dir>/<id>, so make_pin refuses a bad one up front.
        for bad in ("../1.8.0.13", "1.8.0.13/x", "1.8.0.13\n", "latest", ""):
            argv = self.fs.make_pin_argv()
            argv[argv.index("--pin-id") + 1] = bad
            rc, _, err = run_main(make_pin.main, argv, self.roots)
            self.assertEqual(rc, 2, bad)
            self.assertIn("invalid pin id", err)
        self.assertEqual(os.listdir(self.fs.out_dir), [])

    def test_warns_when_not_named_after_the_id(self):
        argv = self.fs.make_pin_argv()
        argv[argv.index("--out") + 1] = os.path.join(self.fs.out_dir, "other.json")
        rc, _, err = run_main(make_pin.main, argv, self.roots)
        self.assertEqual(rc, 0, err)
        self.assertIn("WARNING", err)
        self.assertIn("pin_1.8.0.13.json", err)


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

    def test_pin_cannot_widen_auto_update_policy(self):
        # Review fix: a hand-edited pin may narrow the allowed values, never add "2" (or "0").
        with open(self.fs.pin) as fh:
            pin = json.load(fh)
        pin["steam"]["auto_update_behavior_allowed"] = ["1", "2"]
        with open(self.fs.pin, "w") as fh:
            json.dump(pin, fh)
        self.fs.set_acf("game", "AutoUpdateBehavior", "2")
        self.assertRefused("game.auto_update_behavior")

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
        fake_c = self.fake_c_drive()
        rc, out, err = run_main(preflight.main, ["--pin", self.fs.pin, "--exe", "wb", "--launch-dir", fake_c],
                                self.roots)
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("C:", err)
        self.assertFalse(os.path.exists(fake_c))

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

    def test_path_overrides_only_in_check_mode(self):
        # Review fix: a launch record must describe the pinned install the launcher starts, not another tree.
        for flag, val in (("--game-root", self.fs.game_root), ("--tools-root", self.fs.tools_root),
                          ("--game-acf", self.fs.game_acf), ("--tools-acf", self.fs.tools_acf)):
            rc, out, err = self.launch(flag, val)
            self.assertEqual((rc, out), (2, ""), flag)
            self.assertIn("only accepted with --check", err)
        self.assertEqual(self.records("launch_"), [])
        self.assertEqual(self.records("refused_"), [])

    def test_cli_isolated_mode(self):
        # As the launchers run it: python3 -I -B <abs path>. -I drops the script dir from sys.path,
        # so this proves the explicit pinlib loader works.
        script = os.path.join(PIN_DIR, "preflight.py")
        # A fresh process has the real launch paths, so this fake pin fails exactly the two launch_path checks.
        r = subprocess.run([sys.executable, "-I", "-B", script, "--pin", self.fs.pin, "--check"],
                           capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 1, r.stderr + r.stdout)
        self.assertIn("RESULT: REFUSED (pin 1.8.0.13): 2 check(s) failed: game.launch_path, tools.launch_path",
                      r.stdout)
        r = subprocess.run([sys.executable, "-I", "-B", script, "--pin", self.fs.pin, "--check",
                            "--allow-fail", "game.launch_path", "--allow-fail", "tools.launch_path"],
                           capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("RESULT: OK", r.stdout)
        # Default write roots in a real process: a launch dir outside /mnt/e is refused (exit 2, no stdout).
        r = subprocess.run([sys.executable, "-I", "-B", script, "--pin", self.fs.pin, "--exe", "client",
                            "--launch-dir", NEVER_WRITABLE], capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertIn("outside", r.stderr)
        self.assertFalse(os.path.exists(NEVER_WRITABLE))


# --------------------------------------------------------------------------- active pin pointer


def fake_steam_update(fs, game_buildid="25500000", tools_buildid="25400000"):
    """Make the fake installs look like Steam installed another build: new buildids and a changed pak."""
    for app, bid in (("game", game_buildid), ("tools", tools_buildid)):
        fs.set_acf(app, "buildid", bid)
        fs.set_acf(app, "TargetBuildID", bid)
    fs.write(fs.game_root, "addons/data/data.pak", b"data-0 of the new build", PIN_MTIME_NS + 86400 * 10**9)


def restore_old_build(fs):
    """Undo fake_steam_update the way a restore from the cold copy does: the 1.8 appmanifest values and files come
    back, but every mtime only to whole seconds (drvfs sets whole seconds; a DepotDownloader restore would give
    download times)."""
    for app, bid in (("game", "24903726"), ("tools", "24870687")):
        fs.set_acf(app, "buildid", bid)
        fs.set_acf(app, "TargetBuildID", bid)
    fs.write(fs.game_root, "addons/data/data.pak", b"data-0", PIN_MTIME_NS)
    whole = TOOLS_MTIME_NS // 10**9 * 10**9
    fs.write(fs.tools_root, "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ wb", whole)
    fs.write(fs.tools_root, "Workbench/addons/core/data.pak", b"wbcore", whole)


class PinIdTests(unittest.TestCase):
    def test_valid_ids(self):
        for good in ("1.8.0.13", "1.9.0.12", "1.9.0.9-exp", "1.9", "10.0.1.2", "1.9.0.9-rc2"):
            self.assertEqual(pinlib.validate_pin_id(good), good)

    def test_invalid_ids(self):
        for bad in ("", "1", "1.8.0.13\n", "1.8.0.13\r\n", " 1.8.0.13", "1.8.0.13 ", "../1.8.0.13", "1.8/0.13",
                    "1.8\\0.13", "1..8", "..", "1.8.0.13.1", "1.8.0.13-", "1.8.0.13-EXP", "1.8.0.13-exp/x",
                    "pin_1.8.0.13", "1.8.0.13.json", "\u0661.8.0.13", "1.8.0.13-exp-2", ".1.8", "1.8.", None, 1.8):
            with self.assertRaises(pinlib.PinError, msg=repr(bad)):
                pinlib.validate_pin_id(bad)

    def test_pin_file_for(self):
        self.assertEqual(pinlib.pin_file_for("/x/pin", "1.9.0.9-exp"), "/x/pin/pin_1.9.0.9-exp.json")
        with self.assertRaises(pinlib.PinError):
            pinlib.pin_file_for("/x/pin", "../1.8.0.13")


class PinnedCase(TmpCase):
    """A fake Steam layout set to AutoUpdateBehavior "1" with pin_1.8.0.13.json in out_dir (the pin dir)."""

    def setUp(self):
        super().setUp()
        self.fs.set_auto_update("1")
        rc, _, err = run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)
        self.assertEqual(rc, 0, err)
        self.pin_dir = self.fs.out_dir
        self.active = os.path.join(self.pin_dir, "ACTIVE")
        self.log = os.path.join(self.pin_dir, "ACTIVE.log")
        self.launches = os.path.join(self.pin_dir, "launches")

    def put_active(self, data: bytes):
        with open(self.active, "wb") as fh:
            fh.write(data)

    def pin_file(self, pin_id):
        return os.path.join(self.pin_dir, "pin_%s.json" % pin_id)

    def activate(self, pin_id, roots=None):
        return run_main(pin_activate.main, [pin_id, "--pin-dir", self.pin_dir], roots or self.roots)

    def log_lines(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log, encoding="ascii") as fh:
            return fh.read().splitlines()

    def preflight_active(self, *extra):
        return run_main(preflight.main, ["--pin-active", "--pin-dir", self.pin_dir] + list(extra), self.roots)


class ActivePointerTests(PinnedCase):
    def test_reads_one_valid_line(self):
        for data in (b"1.8.0.13", b"1.8.0.13\n", b"1.8.0.13\r\n"):
            self.put_active(data)
            self.assertEqual(pinlib.read_active(self.pin_dir), "1.8.0.13", data)

    def test_refuses_bad_content(self):
        cases = {
            b"": "empty", b"\n": "empty", b"   \n": "empty",
            b"1.8.0.13\n1.9.0.12\n": "more than one line", b"1.8.0.13\n\n": "more than one line",
            b"1.8.0.13\r": "carriage return", b"\n1.8.0.13": "more than one line",
            b"1.8.0.13\r\r\n": "carriage return",
            b"../1.8.0.13\n": "invalid pin id", b"1.8.0.13 \n": "invalid pin id", b"pin_1.8.0.13.json\n": "invalid pin id",
            b"\xef\xbb\xbf1.8.0.13\n": "not plain ASCII", b"1" * 100: "larger than",
        }
        for data, why in cases.items():
            self.put_active(data)
            with self.assertRaises(pinlib.PinError, msg=data) as cm:
                pinlib.read_active(self.pin_dir)
            self.assertIn(why, str(cm.exception), data)

    def test_refuses_missing(self):
        with self.assertRaisesRegex(pinlib.PinError, "not found"):
            pinlib.read_active(self.pin_dir)

    def test_refuses_symlink(self):
        real = os.path.join(self.tmp, "elsewhere_ACTIVE")
        with open(real, "wb") as fh:
            fh.write(b"1.8.0.13\n")
        os.symlink(real, self.active)
        with self.assertRaisesRegex(pinlib.PinError, "symlink"):
            pinlib.read_active(self.pin_dir)
        os.unlink(self.active)
        os.symlink(os.path.join(self.tmp, "dangling"), self.active)
        with self.assertRaisesRegex(pinlib.PinError, "symlink"):
            pinlib.read_active(self.pin_dir)

    def test_refuses_directory(self):
        os.mkdir(self.active)
        with self.assertRaisesRegex(pinlib.PinError, "not a regular file"):
            pinlib.read_active(self.pin_dir)


class PreflightActiveTests(PinnedCase):
    def record(self, lid):
        with open(os.path.join(self.launches, "launch_%s.json" % lid)) as fh:
            return json.load(fh)

    def test_launch_record_names_pin_and_resolution(self):
        self.put_active(b"1.8.0.13\n")
        rc, out, err = self.preflight_active("--exe", "wb")
        self.assertEqual(rc, 0, err)
        lid = out.strip()
        self.assertRegex(lid, r"^\d{8}T\d{6}Z-wb-[0-9a-f]{4}$")
        rec = self.record(lid)
        self.assertEqual(rec["pin"], {
            "path": self.pin_file("1.8.0.13"), "pin_id": "1.8.0.13", "sha256": pinlib.sha256_file(self.pin_file("1.8.0.13")),
            "resolved_by": "active-pointer", "pointer": self.active})
        self.assertIn("--pin-active", rec["argv"])
        self.assertIn("via active pointer", err)

    def test_explicit_pin_still_works_and_is_recorded_as_such(self):
        rc, out, err = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13"), "--exe", "client"], self.roots)
        self.assertEqual(rc, 0, err)
        rec = self.record(out.strip())
        self.assertEqual((rec["pin"]["resolved_by"], rec["pin"]["pointer"], rec["pin"]["pin_id"]),
                         ("explicit-path", None, "1.8.0.13"))

    def test_refused_record_names_resolution(self):
        self.put_active(b"1.8.0.13\n")
        self.fs.set_acf("game", "AutoUpdateBehavior", "0")
        rc, out, err = self.preflight_active("--exe", "wb")
        self.assertEqual((rc, out), (1, ""))
        refused = [n for n in os.listdir(self.launches) if n.startswith("refused_")]
        self.assertEqual(len(refused), 1)
        with open(os.path.join(self.launches, refused[0])) as fh:
            rec = json.load(fh)
        self.assertEqual((rec["pin"]["resolved_by"], rec["pin"]["pointer"]), ("active-pointer", self.active))

    def test_check_mode_reports_resolution(self):
        self.put_active(b"1.8.0.13\n")
        rc, out, err = self.preflight_active("--check")
        self.assertEqual(rc, 0, out + err)
        first = out.splitlines()[0]
        self.assertTrue(first.startswith("PIN"), first)
        self.assertIn("pin 1.8.0.13 via active pointer %s -> %s" % (self.active, self.pin_file("1.8.0.13")), first)
        self.assertIn("RESULT: OK (pin 1.8.0.13)", out)
        self.assertFalse(os.path.exists(self.launches))

    def test_after_a_steam_update_the_active_pin_refuses(self):
        self.put_active(b"1.8.0.13\n")
        fake_steam_update(self.fs)
        rc, out, err = self.preflight_active("--exe", "client")
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("FAIL game.buildid", err)
        self.assertIn("FAIL tools.buildid", err)

    def test_pointer_problems_are_config_errors(self):
        os.symlink(self.pin_file("1.8.0.13"), self.pin_file("1.9.0.13"))
        shutil.copyfile(self.pin_file("1.8.0.13"), self.pin_file("1.9.0.14"))  # a regular file with pin_id 1.8.0.13
        real = os.path.join(self.tmp, "elsewhere_ACTIVE")
        with open(real, "wb") as fh:
            fh.write(b"1.8.0.13\n")
        cases = [
            ("missing", None, "not found"),
            ("empty", b"", "empty"),
            ("multi-line", b"1.8.0.13\n1.8.0.13\n", "more than one line"),
            ("invalid id", b"../pin_1.8.0.13\n", "invalid pin id"),
            ("symlinked ACTIVE", "symlink", "symlink"),
            ("missing pin file", b"1.9.0.12\n", "not found"),
            ("symlinked pin file", b"1.9.0.13\n", "symlink"),
            ("pin_id mismatch", b"1.9.0.14\n", "holds pin_id '1.8.0.13', not '1.9.0.14'"),
        ]
        for name, data, why in cases:
            with self.subTest(name):
                if os.path.lexists(self.active):
                    os.unlink(self.active)
                if data == "symlink":
                    os.symlink(real, self.active)
                elif data is not None:
                    self.put_active(data)
                for mode in (["--exe", "wb"], ["--check"]):
                    rc, out, err = self.preflight_active(*mode)
                    self.assertEqual((rc, out), (2, ""), (mode, err))
                    self.assertIn("preflight:", err)
                    self.assertIn(why, err)
                self.assertFalse(os.path.exists(self.launches))

    def test_pin_and_pin_active_are_mutually_exclusive(self):
        for argv in (["--pin", self.pin_file("1.8.0.13"), "--pin-active", "--check"],
                     ["--pin-active", "--pin", self.pin_file("1.8.0.13"), "--exe", "wb"],
                     ["--check"]):
            out, err = io.StringIO(), io.StringIO()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                preflight.main(argv, write_roots=self.roots)
            self.assertEqual(cm.exception.code, 2, argv)
            self.assertEqual(out.getvalue(), "", argv)
            self.assertIn("--pin", err.getvalue())
        self.assertFalse(os.path.exists(self.launches))
        with self.assertRaises(pinlib.PinError):
            preflight.resolve_pin(self.pin_file("1.8.0.13"), True, None)
        with self.assertRaises(pinlib.PinError):
            preflight.resolve_pin(None, False, None)

    def test_pin_dir_only_with_pin_active(self):
        rc, out, err = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13"), "--pin-dir", self.pin_dir,
                                                 "--check"], self.roots)
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("--pin-dir is only accepted with --pin-active", err)

    def test_cli_isolated_mode(self):
        self.put_active(b"1.8.0.13\n")
        script = os.path.join(PIN_DIR, "preflight.py")
        r = subprocess.run([sys.executable, "-I", "-B", script, "--pin-active", "--pin-dir", self.pin_dir, "--check",
                            "--allow-fail", "game.launch_path", "--allow-fail", "tools.launch_path"],
                           capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("via active pointer", r.stdout)
        self.assertIn("RESULT: OK (pin 1.8.0.13)", r.stdout)


class PinActivateTests(PinnedCase):
    LOG_RE = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ %s -> %s [A-Za-z0-9._-]+@[A-Za-z0-9._-]+$"

    def test_first_activation_writes_pointer_and_log(self):
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: OK (pin 1.8.0.13)", out)
        self.assertIn("ACTIVE: none -> 1.8.0.13", out)
        with open(self.active, "rb") as fh:
            self.assertEqual(fh.read(), b"1.8.0.13\n")
        lines = self.log_lines()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], self.LOG_RE % ("none", r"1\.8\.0\.13"))
        self.assertEqual([n for n in os.listdir(self.pin_dir) if ".tmp-" in n], [])
        rc, out, _ = self.preflight_active("--check")
        self.assertEqual(rc, 0, out)

    def test_already_active_writes_nothing(self):
        self.assertEqual(self.activate("1.8.0.13")[0], 0)
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 0, err)
        self.assertIn("already the active pin; nothing written", out)
        self.assertEqual(len(self.log_lines()), 1)

    def test_refuses_a_pin_that_does_not_match_the_installs(self):
        self.fs.set_acf("tools", "AutoUpdateBehavior", "0")
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 1)
        self.assertIn("FAIL   tools.auto_update_behavior", out)
        self.assertIn("RESULT: REFUSED", out)
        self.assertIn("NOT activated", err)
        self.assertFalse(os.path.lexists(self.active))
        self.assertFalse(os.path.lexists(self.log))

    def test_switch_to_a_new_build_and_back_is_checked(self):
        self.assertEqual(self.activate("1.8.0.13")[0], 0)
        fake_steam_update(self.fs)  # Steam installs the new build
        self.assertEqual(self.preflight_active("--check")[0], 1)
        rc, out, err = self.activate("1.8.0.13")  # re-activating the old pin does not paper over it
        self.assertEqual(rc, 1, out + err)
        argv = ["--pin-id", "1.9.0.12", "--out", self.pin_file("1.9.0.12"), "--game-acf", self.fs.game_acf,
                "--tools-acf", self.fs.tools_acf, "--expect-game-buildid", "25500000",
                "--expect-tools-buildid", "25400000"]
        rc, _, err = run_main(make_pin.main, argv, self.roots)
        self.assertEqual(rc, 0, err)
        rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 0, out + err)
        with open(self.active, "rb") as fh:
            self.assertEqual(fh.read(), b"1.9.0.12\n")
        lines = self.log_lines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[1], self.LOG_RE % (r"1\.8\.0\.13", r"1\.9\.0\.12"))
        rc, out, _ = self.preflight_active("--check")
        self.assertEqual(rc, 0, out)
        self.assertIn("RESULT: OK (pin 1.9.0.12)", out)
        # Back to 1.8.0.13 is refused while the installs are the new build: ACTIVE and the log stay as they are.
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 1, out + err)
        with open(self.active, "rb") as fh:
            self.assertEqual(fh.read(), b"1.9.0.12\n")
        self.assertEqual(len(self.log_lines()), 2)

    def test_bad_ids_and_pin_files_are_refused(self):
        os.symlink(self.pin_file("1.8.0.13"), self.pin_file("1.9.0.13"))
        shutil.copyfile(self.pin_file("1.8.0.13"), self.pin_file("1.9.0.14"))
        for pin_id, why in (("../1.8.0.13", "invalid pin id"), ("1.8.0.13/..", "invalid pin id"), ("", "invalid pin id"),
                            ("1.9.0.12", "not found"), ("1.9.0.13", "symlink"),
                            ("1.9.0.14", "holds pin_id '1.8.0.13', not '1.9.0.14'")):
            rc, out, err = self.activate(pin_id)
            self.assertEqual(rc, 2, pin_id)
            self.assertIn(why, err, pin_id)
        self.assertFalse(os.path.lexists(self.active))
        self.assertFalse(os.path.lexists(self.log))

    def test_symlinked_active_is_left_alone(self):
        real = os.path.join(self.tmp, "elsewhere_ACTIVE")
        with open(real, "wb") as fh:
            fh.write(b"1.8.0.13\n")
        os.symlink(real, self.active)
        rc, _, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 2)
        self.assertIn("not a regular file", err)
        self.assertTrue(os.path.islink(self.active))
        self.assertFalse(os.path.lexists(self.log))

    def test_symlinked_log_refused_before_the_switch(self):
        os.symlink(os.path.join(self.tmp, "elsewhere_log"), self.log)
        rc, _, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 2)
        self.assertIn("ACTIVE.log is not a regular file", err)
        self.assertFalse(os.path.lexists(self.active))
        self.assertFalse(os.path.lexists(os.path.join(self.tmp, "elsewhere_log")))

    def test_invalid_pointer_is_replaced_and_logged(self):
        self.put_active(b"garbage\n")
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 0, out + err)
        self.assertRegex(self.log_lines()[0], self.LOG_RE % ("invalid", r"1\.8\.0\.13"))

    def test_write_guard_refuses_c_drive(self):
        fake_c = self.fake_c_drive()
        os.makedirs(fake_c)
        shutil.copyfile(self.pin_file("1.8.0.13"), os.path.join(fake_c, "pin_1.8.0.13.json"))
        rc, _, err = run_main(pin_activate.main, ["1.8.0.13", "--pin-dir", fake_c], self.roots)
        self.assertEqual(rc, 2)
        self.assertIn("C:", err)
        self.assertEqual(os.listdir(fake_c), ["pin_1.8.0.13.json"])

    def test_write_guard_refuses_off_e_by_default(self):
        if os.path.realpath(self.tmp).startswith("/mnt/e/"):
            self.skipTest("temp dir is on E:, so the default guard allows it")
        rc, _, err = self.activate("1.8.0.13", roots=pinlib.DEFAULT_WRITE_ROOTS)
        self.assertEqual(rc, 2)
        self.assertIn("outside", err)
        self.assertFalse(os.path.lexists(self.active))

    def test_cli_isolated_mode(self):
        script = os.path.join(PIN_DIR, "pin_activate.py")
        r = subprocess.run([sys.executable, "-I", "-B", script, "../1.8.0.13", "--pin-dir", self.pin_dir],
                           capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual((r.returncode, r.stdout), (2, ""), r.stderr)
        self.assertIn("invalid pin id", r.stderr)
        r = subprocess.run([sys.executable, "-I", "-B", script, "1.8.0.13", "--pin-dir", self.pin_dir],
                           capture_output=True, text=True, cwd=self.tmp)
        if os.path.realpath(self.tmp).startswith("/mnt/e/"):
            # Allowed to write here, but a fresh process has the real launch paths: refused by launch_path.
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn("FAIL   game.launch_path", r.stdout)
            self.assertFalse(os.path.lexists(self.active))
        else:
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("outside", r.stderr)
            self.assertFalse(os.path.lexists(self.active))



# --------------------------------------------------------------------------- review fixes (active-pin review AP-1..AP-9)


class LaunchPathTests(PinnedCase):
    """AP-2: a pin is usable only if it was made from the installs the launchers start (pinlib.LAUNCH_PATHS)."""

    def elsewhere(self, apps=("game", "tools")):
        """Make the launchers 'start' installs other than the ones the fake pin was made from."""
        lp = {app: dict(v) for app, v in pinlib.LAUNCH_PATHS.items()}
        for app in apps:
            lp[app] = dict(REAL_LAUNCH_PATHS[app])
        pinlib.LAUNCH_PATHS = lp

    def test_launch_refused_for_a_pin_from_another_tree(self):
        self.put_active(b"1.8.0.13\n")
        self.elsewhere()
        rc, out, err = self.preflight_active("--exe", "client")
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("FAIL game.launch_path:", err)
        self.assertIn("the launchers start %s" % REAL_LAUNCH_PATHS["game"]["install"], err)
        refused = [n for n in os.listdir(self.launches) if n.startswith("refused_")]
        self.assertEqual(len(refused), 1)
        with open(os.path.join(self.launches, refused[0])) as fh:
            rec = json.load(fh)
        self.assertEqual(sorted(c["name"] for c in rec["checks"] if not c["ok"]),
                         ["game.launch_path", "tools.launch_path"])
        self.assertEqual([n for n in os.listdir(self.launches) if n.startswith("launch_")], [])

    def test_one_app_elsewhere(self):
        self.put_active(b"1.8.0.13\n")
        self.elsewhere(apps=("tools",))
        rc, out, _ = self.preflight_active("--check")
        self.assertEqual(rc, 1, out)
        self.assertIn("1 check(s) failed: tools.launch_path", out)
        self.assertIn("PASS   game.launch_path", out)

    def test_check_mode_can_waive_it_launch_mode_cannot(self):
        self.put_active(b"1.8.0.13\n")
        self.elsewhere()
        waive = ["--allow-fail", "game.launch_path", "--allow-fail", "tools.launch_path"]
        rc, out, _ = self.preflight_active("--check", *waive)
        self.assertEqual(rc, 0, out)
        rc, out, err = self.preflight_active("--exe", "wb", *waive)
        self.assertEqual((rc, out), (2, ""), err)

    def test_activation_refused_for_a_pin_from_another_tree(self):
        self.elsewhere()
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 1, out + err)
        self.assertIn("FAIL   game.launch_path", out)
        self.assertFalse(os.path.lexists(self.active))
        self.assertFalse(os.path.lexists(self.log))

    def test_launch_paths_match_the_launchers(self):
        # The .bat files start fixed exes; LAUNCH_PATHS must name exactly those installs.
        def bat_value(name, var):
            with open(os.path.join(LAUNCH_DIR, name), "rb") as fh:
                text = fh.read().decode("ascii")
            m = re.search(r'^set "%s=([^"%%]+)"\r?$' % var, text, re.M)
            self.assertIsNotNone(m, (name, var))
            return pinlib.win_to_wsl(m.group(1))
        self.assertEqual(bat_value("wb_R1.bat", "WB_DIR"), REAL_LAUNCH_PATHS["tools"]["install"] + "/Workbench")
        self.assertEqual(bat_value("client_R1_i01.bat", "GAME_DIR"), REAL_LAUNCH_PATHS["game"]["install"])
        # A direct exe launch only finds ArmaReforger.gproj through -addonsDir (measured 2026-10-21).
        self.assertEqual(bat_value("wb_R1.bat", "GAME_ADDONS"), REAL_LAUNCH_PATHS["game"]["install"] + "/addons")
        for app in pinlib.APPS:
            self.assertEqual(REAL_LAUNCH_PATHS[app]["acf"], pinlib.DEFAULT_ACF[app])
            self.assertEqual(pinlib.install_dir_for_acf(REAL_LAUNCH_PATHS[app]["acf"],
                                                        os.path.basename(REAL_LAUNCH_PATHS[app]["install"])),
                             REAL_LAUNCH_PATHS[app]["install"])

    def test_launchers_forward_the_launch_id(self):
        # DC_LaunchGuard (DD-4-probe) writes nothing without -dcLaunchId and $profile:dc_launch/launch_<id>.json.
        def bat_text(name):
            with open(os.path.join(LAUNCH_DIR, name), "rb") as fh:
                return fh.read().decode("ascii")
        for name in ("wb_R1.bat", "client_R1_i01.bat"):
            starts = [l for l in bat_text(name).splitlines() if l.startswith('start "" ')]
            self.assertEqual(len(starts), 1, name)
            self.assertIn(" -dcLaunchId %LID%", starts[0], name)
        wb = bat_text("wb_R1.bat")
        self.assertIn('set "WB_DC_LAUNCH=%WB_PROFILE%\\profile\\dc_launch"', wb)
        self.assertIn('fc /b "%DD_LAUNCHES%\\launch_%LID%.json" "%WB_DC_LAUNCH%\\launch_%LID%.json"', wb)
        self.assertLess(wb.index("fc /b"), wb.index('start "" '))


class RollbackTests(PinnedCase):
    """AP-1: after a restore, the original pin stays refused (sub-second mtimes are gone); the restored installs are
    re-pinned under a new id with make_pin --restore-of, which checks they are the same build."""

    def to_new_build(self):
        self.assertEqual(self.activate("1.8.0.13")[0], 0)
        fake_steam_update(self.fs)
        argv = ["--pin-id", "1.9.0.12", "--out", self.pin_file("1.9.0.12"), "--game-acf", self.fs.game_acf,
                "--tools-acf", self.fs.tools_acf]
        self.assertEqual(run_main(make_pin.main, argv, self.roots)[0], 0)
        self.assertEqual(self.activate("1.9.0.12")[0], 0)

    def repin_argv(self, pin_id="1.8.0.13-r1", ref="1.8.0.13", *extra):
        return ["--pin-id", pin_id, "--restore-of", ref, "--out", self.pin_file(pin_id), "--game-acf",
                self.fs.game_acf, "--tools-acf", self.fs.tools_acf] + list(extra)

    def test_rollback_by_repin(self):
        sha_ref = pinlib.sha256_file(self.pin_file("1.8.0.13"))
        with open(self.pin_file("1.8.0.13")) as fh:
            sub_second = json.load(fh)["tools"]["paks"]["Workbench/addons/core/data.pak"]["mtime_ns"] % 10**9
        self.to_new_build()
        restore_old_build(self.fs)
        rc, out, err = self.activate("1.8.0.13")
        if sub_second:  # tmpfs keeps the sub-second mtimes, like NTFS: the restore cannot match the original pin
            self.assertEqual(rc, 1, out + err)
            self.assertIn("FAIL   tools.binaries", out)
            self.assertIn("FAIL   tools.paks", out)
            rc, out, err = run_main(make_pin.main, self.repin_argv("1.8.0.13-r1", "1.8.0.13",
                                                                   "--expect-game-buildid", "24903726",
                                                                   "--expect-tools-buildid", "24870687"), self.roots)
            self.assertEqual(rc, 0, err)
            self.assertIn("restore of pin 1.8.0.13: same build; 2 file mtime(s) differ", out)
            with open(self.pin_file("1.8.0.13-r1")) as fh:
                r1 = json.load(fh)
            self.assertEqual((r1["restore_of"]["pin_id"], r1["restore_of"]["sha256"]), ("1.8.0.13", sha_ref))
            rc, out, err = self.activate("1.8.0.13-r1")
            self.assertEqual(rc, 0, out + err)
            self.assertRegex(self.log_lines()[-1], PinActivateTests.LOG_RE % (r"1\.9\.0\.12", r"1\.8\.0\.13-r1"))
            rc, out, _ = self.preflight_active("--check")
            self.assertEqual(rc, 0, out)
            self.assertIn("RESULT: OK (pin 1.8.0.13-r1)", out)
        else:  # drvfs truncated the pin's own mtimes, so the original pin matches again
            self.assertEqual(rc, 0, out + err)
        self.assertEqual(pinlib.sha256_file(self.pin_file("1.8.0.13")), sha_ref)

    def test_restore_of_refuses_another_build(self):
        self.to_new_build()  # installs are the new build
        rc, _, err = run_main(make_pin.main, self.repin_argv(), self.roots)
        self.assertEqual(rc, 1)
        self.assertIn("not a restore of pin 1.8.0.13", err)
        self.assertIn("game.buildid 25500000 != 24903726", err)
        self.assertIn("addons/data/data.pak size", err)
        self.assertFalse(os.path.exists(self.pin_file("1.8.0.13-r1")))

    def test_restore_of_refuses_changed_files(self):
        self.to_new_build()
        restore_old_build(self.fs)
        self.fs.write(self.fs.game_root, "addons/data/data011.pak", b"extra", PIN_MTIME_NS)
        os.unlink(os.path.join(self.fs.game_root, "ArmaReforgerSteamDiag.exe"))
        self.fs.write(self.fs.game_root, "ArmaReforgerSteamDiag.exe", b"MZ game diag, other", PIN_MTIME_NS)
        rc, _, err = run_main(make_pin.main, self.repin_argv(), self.roots)
        self.assertEqual(rc, 1, err)
        self.assertIn("game.paks addons/data/data011.pak not in the reference pin", err)
        self.assertIn("game.binaries ArmaReforgerSteamDiag.exe size", err)
        self.assertFalse(os.path.exists(self.pin_file("1.8.0.13-r1")))

    def test_restore_of_needs_a_new_id_and_an_existing_reference(self):
        argv = ["--pin-id", "1.8.0.13", "--restore-of", "1.8.0.13", "--out", os.path.join(self.pin_dir, "x.json"),
                "--game-acf", self.fs.game_acf, "--tools-acf", self.fs.tools_acf]
        rc, _, err = run_main(make_pin.main, argv, self.roots)
        self.assertEqual(rc, 2)
        self.assertIn("gets a new pin id", err)
        rc, _, err = run_main(make_pin.main, self.repin_argv("1.8.0.13-r1", "1.7.0.1"), self.roots)
        self.assertEqual(rc, 2)
        self.assertIn("not found", err)
        rc, _, err = run_main(make_pin.main, self.repin_argv("1.8.0.13-r1", "../1.8.0.13"), self.roots)
        self.assertEqual(rc, 2)
        self.assertIn("invalid pin id", err)
        self.assertEqual(sorted(os.listdir(self.pin_dir)), ["pin_1.8.0.13.json"])


class ActivationLogAndLockTests(PinnedCase):
    """AP-3 (a switch is never left unlogged), AP-4 (activations are serialized), AP-7 (fsync and read-back)."""

    LOG_RE = PinActivateTests.LOG_RE

    def new_build_pin(self):
        self.assertEqual(self.activate("1.8.0.13")[0], 0)
        fake_steam_update(self.fs)
        argv = ["--pin-id", "1.9.0.12", "--out", self.pin_file("1.9.0.12"), "--game-acf", self.fs.game_acf,
                "--tools-acf", self.fs.tools_acf]
        self.assertEqual(run_main(make_pin.main, argv, self.roots)[0], 0)

    def read_active_bytes(self):
        with open(self.active, "rb") as fh:
            return fh.read()

    def test_unwritable_log_refuses_before_the_switch(self):
        self.new_build_pin()
        os.chmod(self.log, 0o444)
        self.addCleanup(os.chmod, self.log, 0o644)
        if os.access(self.log, os.W_OK):
            self.skipTest("the log stays writable here (root, or a filesystem that ignores the mode)")
        rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 3, out + err)
        self.assertIn("ACTIVE not changed (still 1.8.0.13)", err)
        self.assertEqual(self.read_active_bytes(), b"1.8.0.13\n")
        os.chmod(self.log, 0o644)
        rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 0, out + err)
        self.assertRegex(self.log_lines()[-1], self.LOG_RE % (r"1\.8\.0\.13", r"1\.9\.0\.12"))

    def test_log_write_failure_is_repaired_on_retry(self):
        self.new_build_pin()
        with mock.patch.object(pinlib, "write_line", side_effect=OSError(5, "Input/output error")):
            rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 3, out + err)
        self.assertIn("ACTIVE now names 1.9.0.12, but the log line was NOT written", err)
        self.assertEqual(self.read_active_bytes(), b"1.9.0.12\n")
        self.assertEqual(len(self.log_lines()), 1)
        rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("did not record a switch to it; appended", out)
        lines = self.log_lines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[1], self.LOG_RE % ("unlogged", r"1\.9\.0\.12"))
        rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("already the active pin; nothing written", out)
        self.assertEqual(len(self.log_lines()), 2)

    def test_hand_written_pointer_gets_an_unlogged_line(self):
        self.put_active(b"1.8.0.13\n")
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 0, out + err)
        self.assertRegex(self.log_lines()[0], self.LOG_RE % ("unlogged", r"1\.8\.0\.13"))

    def test_held_lock_refuses_and_is_left_alone(self):
        lock = os.path.join(self.pin_dir, "ACTIVE.lock")
        with open(lock, "w") as fh:
            fh.write("pid 1 someone@elsewhere 2026-10-21T00:00:00Z\n")
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 2, out + err)
        self.assertIn("ACTIVE.lock exists (pid 1 someone@elsewhere", err)
        self.assertFalse(os.path.lexists(self.active))
        self.assertFalse(os.path.lexists(self.log))
        with open(lock) as fh:
            self.assertEqual(fh.read(), "pid 1 someone@elsewhere 2026-10-21T00:00:00Z\n")
        os.unlink(lock)
        os.symlink(os.path.join(self.tmp, "elsewhere_lock"), lock)  # a dangling symlink counts as held, too
        self.assertEqual(self.activate("1.8.0.13")[0], 2)
        self.assertFalse(os.path.lexists(os.path.join(self.tmp, "elsewhere_lock")))

    def test_lock_is_released_after_success_and_refusal(self):
        lock = os.path.join(self.pin_dir, "ACTIVE.lock")
        self.assertEqual(self.activate("1.8.0.13")[0], 0)
        self.assertFalse(os.path.lexists(lock))
        self.fs.set_acf("game", "AutoUpdateBehavior", "0")
        self.assertEqual(self.activate("1.8.0.13")[0], 1)
        self.assertFalse(os.path.lexists(lock))

    def test_concurrent_activation_is_serialized(self):
        self.new_build_pin()
        real = preflight.run_checks
        nested = []

        def racing(*a, **k):  # a second activation starts while the first one is checking
            if not nested:
                nested.append(run_main(pin_activate.main, ["1.9.0.12", "--pin-dir", self.pin_dir], self.roots))
            return real(*a, **k)

        with mock.patch.object(preflight, "run_checks", racing):
            rc, out, err = self.activate("1.9.0.12")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(nested[0][0], 2, nested[0])
        self.assertIn("ACTIVE.lock exists", nested[0][2])
        lines = self.log_lines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[1], self.LOG_RE % (r"1\.8\.0\.13", r"1\.9\.0\.12"))

    def test_pointer_and_log_are_fsynced(self):
        synced = []
        real = os.fsync

        def spy(fd):
            synced.append(stat.S_ISREG(os.fstat(fd).st_mode))
            return real(fd)

        with mock.patch.object(os, "fsync", spy):
            self.assertEqual(self.activate("1.8.0.13")[0], 0)
        self.assertGreaterEqual(synced.count(True), 2, synced)  # the ACTIVE temp file and the log

    def test_read_back_mismatch_is_a_write_error(self):
        real = pinlib.replace_text_at

        def wrong(dir_fd, name, text):
            return real(dir_fd, name, "1.9.9.9\n")

        with mock.patch.object(pinlib, "replace_text_at", wrong):
            rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 3, out + err)
        self.assertIn("write did not take", err)


class AnchoredWriteTests(PinnedCase):
    """AP-6: pin_activate writes through one opened fd of the pin dir, so swapping the dir afterwards cannot redirect
    a write. AP-5 for the fd: the guard runs again on the directory the fd really is."""

    def swap_pin_dir(self, target):
        moved = self.pin_dir + "_moved"
        os.rename(self.pin_dir, moved)
        os.symlink(target, self.pin_dir)
        return moved

    def test_activation_writes_stay_in_the_opened_dir(self):
        fake_c = self.fake_c_drive()
        os.makedirs(fake_c)
        real = preflight.run_checks
        moved = []

        def swapping(*a, **k):  # the pin dir is swapped for a symlink to "C:" while the checks run
            if not moved:
                moved.append(self.swap_pin_dir(fake_c))
            return real(*a, **k)

        with mock.patch.object(preflight, "run_checks", swapping):
            rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(os.listdir(fake_c), [])
        names = sorted(os.listdir(moved[0]))
        if not os.path.realpath(self.tmp).startswith("/mnt/"):
            # Linux semantics (tmpfs): the fd follows the renamed directory, so every write lands there.
            self.assertEqual(rc, 0, out + err)
            self.assertEqual(names, ["ACTIVE", "ACTIVE.log", "pin_1.8.0.13.json"])
            with open(os.path.join(moved[0], "ACTIVE"), "rb") as fh:
                self.assertEqual(fh.read(), b"1.8.0.13\n")
        elif rc != 0:
            # drvfs (measured on E:): every *at() call through the fd of a directory renamed after the open fails
            # with ENXIO, so the switch fails closed. Nothing is written anywhere; the lock may stay behind.
            self.assertEqual(rc, 3, out + err)
            self.assertIn("ACTIVE not changed (still none)", err)
            self.assertNotIn("ACTIVE", names)
            self.assertNotIn("ACTIVE.log", names)

    def test_dir_swapped_before_the_open_is_refused(self):
        fake_c = self.fake_c_drive()
        os.makedirs(fake_c)
        real = pinlib.check_write_path
        calls = []

        def check_then_swap(path, roots=pinlib.DEFAULT_WRITE_ROOTS):
            out = real(path, roots)
            if not calls:
                calls.append(self.swap_pin_dir(fake_c))
            return out

        with mock.patch.object(pinlib, "check_write_path", check_then_swap):
            with self.assertRaises(pinlib.PinError):
                pinlib.open_guarded_dir(self.pin_dir, self.roots)
        self.assertEqual(os.listdir(fake_c), [])

    def test_parent_swapped_before_the_open_is_refused(self):
        # O_NOFOLLOW covers only the last component; the re-check of the opened fd's real path covers the rest.
        fake_c = self.fake_c_drive()
        os.makedirs(os.path.join(fake_c, "pin"))
        parent = os.path.join(self.tmp, "a")
        os.makedirs(os.path.join(parent, "pin"))
        real = pinlib.check_write_path
        calls = []

        def check_then_swap(path, roots=pinlib.DEFAULT_WRITE_ROOTS):
            out = real(path, roots)
            if not calls:
                os.rename(parent, parent + "_moved")
                os.symlink(fake_c, parent)
                calls.append(path)
            return out

        with mock.patch.object(pinlib, "check_write_path", check_then_swap):
            with self.assertRaisesRegex(pinlib.PinError, "C:"):
                pinlib.open_guarded_dir(os.path.join(parent, "pin"), self.roots)
        self.assertEqual(os.listdir(os.path.join(fake_c, "pin")), [])

    def test_helpers_refuse_symlinks_and_bad_names(self):
        fd = pinlib.open_guarded_dir(self.pin_dir, self.roots)
        try:
            os.symlink(os.path.join(self.tmp, "elsewhere"), os.path.join(self.pin_dir, "LOG"))
            with self.assertRaisesRegex(pinlib.PinError, "symlink"):
                pinlib.open_append_at(fd, "LOG")
            for bad in ("", ".", "..", "a/b"):
                with self.assertRaises(pinlib.PinError):
                    pinlib.open_append_at(fd, bad)
            pinlib.replace_text_at(fd, "LOG", "x\n")  # a symlink at the target name is replaced, never followed
            self.assertFalse(os.path.islink(os.path.join(self.pin_dir, "LOG")))
            self.assertFalse(os.path.lexists(os.path.join(self.tmp, "elsewhere")))
        finally:
            os.close(fd)


SAMPLE_MOUNTINFO = (
    "78 63 8:32 / / rw,relatime - ext4 /dev/sdc rw,discard\n"
    "119 112 0:66 / /dev/shm rw,nosuid,nodev,noatime shared:24 - tmpfs none rw\n"
    "159 78 0:84 / /mnt/c rw,noatime - 9p C:\\134 rw,dirsync,aname=drvfs;path=C:\;uid=1000;gid=1000\n"
    "161 78 0:86 / /mnt/e rw,noatime - 9p E:\\134 rw,dirsync,aname=drvfs;path=E:\;uid=1000;gid=1000\n")


class MountGuardTests(unittest.TestCase):
    """AP-5: /mnt/e is only a name; writes under it must land on the mount of the E: drive."""

    def guard(self, text, path="/mnt/e/DriveDiT/pin/x.json"):
        with mock.patch.object(pinlib, "_read_mountinfo", return_value=text):
            return pinlib.check_write_path(path)

    def test_parse(self):
        mounts = pinlib.parse_mountinfo(SAMPLE_MOUNTINFO + "garbage line\n")
        e = pinlib.mount_of("/mnt/e/DriveDiT/pin", mounts)
        self.assertEqual((e["mount_point"], e["fstype"], e["source"]), ("/mnt/e", "9p", "E:\\"))
        self.assertEqual(e["dev"], os.makedev(0, 86))
        self.assertEqual(pinlib.mount_of("/mnt/eagle/x", mounts)["mount_point"], "/")
        self.assertEqual(pinlib.mount_of("/dev/shm/x", mounts)["mount_point"], "/dev/shm")
        sp = pinlib.parse_mountinfo("1 0 0:9 / /mnt/my\\040drive rw - 9p X:\\134 rw\n")
        self.assertEqual((sp[0]["mount_point"], sp[0]["source"]), ("/mnt/my drive", "X:\\"))

    def test_e_drive_mount_accepted(self):
        self.assertEqual(self.guard(SAMPLE_MOUNTINFO), "/mnt/e/DriveDiT/pin/x.json")
        self.assertEqual(self.guard(SAMPLE_MOUNTINFO, "/mnt/e"), "/mnt/e")
        # Only the drvfs path= option names the drive (older WSL shows the source as "drvfs").
        self.assertEqual(self.guard(SAMPLE_MOUNTINFO.replace("9p E:\\134", "9p drvfs")),
                         "/mnt/e/DriveDiT/pin/x.json")

    def test_unmounted_e_refused(self):
        text = "".join(l + "\n" for l in SAMPLE_MOUNTINFO.splitlines() if " /mnt/e " not in l)
        with self.assertRaisesRegex(pinlib.PinError, r"must lie on the E: drive mounted at /mnt/e, but it lies on / "):
            self.guard(text)

    def test_other_drive_at_e_refused(self):
        text = SAMPLE_MOUNTINFO.replace("9p E:\\134", "9p D:\\134").replace("path=E:\\", "path=D:\\")
        with self.assertRaisesRegex(pinlib.PinError, "lies on /mnt/e"):
            self.guard(text)

    def test_mount_below_e_refused(self):
        text = SAMPLE_MOUNTINFO + "200 161 0:99 / /mnt/e/DriveDiT rw - tmpfs none rw\n"
        with self.assertRaisesRegex(pinlib.PinError, "lies on /mnt/e/DriveDiT"):
            self.guard(text)
        self.assertEqual(self.guard(text, "/mnt/e/other/x.json"), "/mnt/e/other/x.json")
        # the last mount at a point is the one on top
        text = SAMPLE_MOUNTINFO + "201 78 0:98 / /mnt/e rw - tmpfs none rw\n"
        with self.assertRaises(pinlib.PinError):
            self.guard(text)

    def test_unreadable_mountinfo_refused(self):
        with mock.patch.object(pinlib, "MOUNTINFO", "/nonexistent/__dd0_mountinfo__"):
            with self.assertRaisesRegex(pinlib.PinError, "cannot read"):
                pinlib.check_write_path("/mnt/e/DriveDiT/pin/x.json")

    def test_paths_off_drive_roots_are_not_affected(self):
        tmp = tempfile.mkdtemp(prefix="dd0pin-", dir=_tmp_base())
        self.addCleanup(shutil.rmtree, tmp, True)
        if os.path.realpath(tmp).startswith("/mnt/e/"):
            self.skipTest("temp dir is on E:")
        target = os.path.join(tmp, "x.json")
        with mock.patch.object(pinlib, "_read_mountinfo", side_effect=AssertionError("not consulted")):
            self.assertEqual(pinlib.check_write_path(target, allowed_roots=(tmp,)), os.path.realpath(target))

    @unittest.skipUnless(os.path.ismount("/mnt/e"), "/mnt/e is not mounted")
    def test_live_e_mount(self):
        m = pinlib.drive_mount_for("/mnt/e/DriveDiT/pin")
        self.assertEqual(m["mount_point"], "/mnt/e")
        self.assertEqual(m["dev"], os.stat("/mnt/e").st_dev)


class ReadRaceTests(TmpCase):
    """AP-7 (M2): O_NOFOLLOW and the fstat re-check hold on their own, for a file swapped after the lstat."""

    def lying_lstat(self):
        regular = os.path.join(self.tmp, "regular")
        with open(regular, "wb") as fh:
            fh.write(b"1.8.0.13\n")
        st = os.stat(regular)
        return mock.patch.object(pinlib, "_lstat", lambda path, dir_fd=None: st)

    def test_symlink_swapped_in_is_refused(self):
        target = os.path.join(self.tmp, "target")
        with open(target, "wb") as fh:
            fh.write(b"1.8.0.13\n")
        link = os.path.join(self.tmp, "link")
        os.symlink(target, link)
        with self.lying_lstat():
            with self.assertRaisesRegex(pinlib.PinError, "symlink"):
                pinlib.read_regular_file(link, "test file", 64)

    def test_fifo_swapped_in_is_refused_without_blocking(self):
        fifo = os.path.join(self.tmp, "fifo")
        try:
            os.mkfifo(fifo)
        except OSError as exc:
            self.skipTest("no FIFO here: %s" % exc)

        def timeout(signum, frame):
            raise TimeoutError("read_regular_file blocked on a FIFO")

        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(5)
        try:
            with self.lying_lstat():
                with self.assertRaisesRegex(pinlib.PinError, "not a regular file"):
                    pinlib.read_regular_file(fifo, "test file", 64)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


@unittest.skipUnless(shutil.which("bash"), "bash not found")
class ColdCopyResolutionTests(PinnedCase):
    """cold_copy.sh resolves the active pin by default. --dest is a path the guard always refuses (/proc), so these
    runs stop at the DEST guard and write nothing; reaching that guard proves the pin resolved."""

    def cold_copy(self, *args):
        r = subprocess.run(["bash", COLD_COPY] + list(args) + ["--dest", NEVER_WRITABLE],
                           capture_output=True, text=True, cwd=self.tmp)
        self.assertFalse(os.path.exists(NEVER_WRITABLE))
        return r.returncode, r.stdout, r.stderr

    def test_defaults_to_the_active_pin(self):
        self.put_active(b"1.8.0.13\n")
        rc, _, err = self.cold_copy("--pin-dir", self.pin_dir)
        self.assertEqual(rc, 1)
        self.assertIn("refusing to write outside /mnt/e", err)

    def test_pointer_problems_stop_it(self):
        rc, _, err = self.cold_copy("--pin-dir", self.pin_dir)
        self.assertEqual(rc, 1)
        self.assertIn("not found", err)
        self.assertIn("cannot resolve the pin", err)
        os.symlink(os.path.join(self.tmp, "elsewhere"), self.active)
        rc, _, err = self.cold_copy("--pin-dir", self.pin_dir)
        self.assertEqual(rc, 1)
        self.assertIn("symlink", err)

    def test_explicit_pin_and_exclusivity(self):
        rc, _, err = self.cold_copy("--pin", self.pin_file("1.8.0.13"))
        self.assertEqual(rc, 1)
        self.assertIn("refusing to write outside /mnt/e", err)
        rc, _, err = self.cold_copy("--pin", self.pin_file("1.8.0.13"), "--pin-dir", self.pin_dir)
        self.assertEqual(rc, 1)
        self.assertIn("mutually exclusive", err)

    def test_explicit_pin_with_an_invalid_pin_id_is_refused(self):
        # AP-7 (M1): the pin id names the default DEST (<pin dir>/<id>), so an explicit pin holding ".." must stop
        # cold_copy before any DEST is derived or checked.
        with open(self.pin_file("1.8.0.13")) as fh:
            pin = json.load(fh)
        for bad in ("..", "1.8.0.13/../../x", None):
            pin["pin_id"] = bad
            path = os.path.join(self.tmp, "hand_made_pin.json")
            with open(path, "w") as fh:
                json.dump(pin, fh)
            rc, _, err = self.cold_copy("--pin", path)
            self.assertEqual(rc, 1, bad)
            self.assertIn("invalid pin id %r" % (bad,), err)
            self.assertIn("cannot resolve the pin", err)
            self.assertNotIn("refusing to write", err)


if __name__ == "__main__":
    unittest.main()
