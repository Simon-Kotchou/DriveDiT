"""Unit tests for the DD-14a additions to tools/pin: pin channels, build_key, dd-pin/2, the sidecars, launch-time
use_class and the freeze check. Stdlib unittest only.

    python3 -I -B -m unittest discover -s tools/pin/tests -v

Temp trees as in test_pin.py ($DD_PIN_TEST_TMP, else /dev/shm), deleted afterwards. The DD-0 tests in test_pin.py are
unchanged; this file reuses their fake Steam layout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
TOOLS_DIR = os.path.dirname(os.path.dirname(HERE))


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
freeze_check = base._load("freeze_check")
run_main = base.run_main


def spec_build_key(game_appid, game_buildid, game_exe_sha, tools_appid, tools_buildid, tools_exe_sha):
    """build_key/1 as MIGRATION_1_9 2.2 and tools/harvester/harvest.py build_key_v1 define it."""
    text = ("build_key/1\ngame.appid=%d\ngame.buildid=%d\ngame.exe_sha256=%s\ntools.appid=%d\ntools.buildid=%d\n"
            "tools.exe_sha256=%s\n" % (game_appid, game_buildid, game_exe_sha, tools_appid, tools_buildid,
                                       tools_exe_sha))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class V2Case(base.PinnedCase):
    """PinnedCase (pin_1.8.0.13.json, dd-pin/1) plus helpers for dd-pin/2 pins and records."""

    def make_v2(self, pin_id="1.8.0.14", *extra):
        out = self.pin_file(pin_id)
        rc, outp, err = run_main(make_pin.main, ["--channel", "stable", "--pin-id", pin_id, "--out", out,
                                                 "--game-acf", self.fs.game_acf, "--tools-acf", self.fs.tools_acf]
                                 + list(extra), self.roots)
        self.assertEqual(rc, 0, err)
        return read_json(out)

    def launch(self, pin_path, *extra, exe="wb"):
        rc, out, err = run_main(preflight.main, ["--pin", pin_path, "--exe", exe] + list(extra), self.roots)
        rec = None
        if rc == 0:
            rec = read_json(os.path.join(self.launches, "launch_%s.json" % out.strip()))
        return rc, out, err, rec

    def sidecar(self, pin_id="1.8.0.13", *extra):
        return run_main(pin_sidecar.main, ["build-key", pin_id, "--pin-dir", self.pin_dir] + list(extra), self.roots)


# --------------------------------------------------------------------------- build_key


class BuildKeyTests(V2Case):
    def test_formula_is_the_spec(self):
        exe = {"game": {"ArmaReforgerSteam.exe": "a" * 64, "ArmaReforgerSteamDiag.exe": "b" * 64},
               "tools": {"Workbench/ArmaReforgerWorkbenchSteamDiag.exe": "c" * 64}}
        inputs = pinlib.build_key_inputs("stable", {"game": {"appid": 1874880, "buildid": 1},
                                                    "tools": {"appid": 1874910, "buildid": 2}}, exe)
        self.assertEqual(pinlib.build_key_from_inputs(inputs),
                         spec_build_key(1874880, 1, "a" * 64, 1874910, 2, "c" * 64))
        # the Diag exe is recorded but is not part of the key
        exe["game"]["ArmaReforgerSteamDiag.exe"] = "d" * 64
        inputs2 = pinlib.build_key_inputs("stable", {"game": {"appid": 1874880, "buildid": 1},
                                                     "tools": {"appid": 1874910, "buildid": 2}}, exe)
        self.assertEqual(pinlib.build_key_from_inputs(inputs2), pinlib.build_key_from_inputs(inputs))

    def test_missing_input_refused(self):
        with self.assertRaises(pinlib.PinError):
            pinlib.build_key_inputs("stable", {"game": {"appid": 1, "buildid": 1}, "tools": {"appid": 2}},
                                    {"game": {"ArmaReforgerSteam.exe": "a" * 64}, "tools": {}})

    def test_make_pin_v2_stable(self):
        pin = self.make_v2()
        self.assertEqual((pin["schema"], pin["channel"], pin["release_channel"], pin["launch_path_id"]),
                         ("dd-pin/2", "stable", "stable", "stable"))
        self.assertEqual(pin["physics_hz_expected"], 30)   # the measured fixed slice (1/30 s), not the planned 60
        self.assertEqual(pin["game"]["exe_sha256"], {"ArmaReforgerSteam.exe": sha(b"MZ game"),
                                                     "ArmaReforgerSteamDiag.exe": sha(b"MZ game diag")})
        self.assertEqual(pin["tools"]["exe_sha256"], {"Workbench/ArmaReforgerWorkbenchSteamDiag.exe": sha(b"MZ wb")})
        self.assertEqual(pin["build_key"], spec_build_key(1874880, 24903726, sha(b"MZ game"), 1874910, 24870687,
                                                          sha(b"MZ wb")))
        self.assertNotIn("LastOwner", json.dumps(pin))

    def test_one_changed_exe_byte_changes_the_key(self):
        k1 = self.make_v2("1.8.0.14")["build_key"]
        self.fs.write(self.fs.tools_root, "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ wc", base.TOOLS_MTIME_NS)
        k2 = self.make_v2("1.8.0.15")["build_key"]
        self.assertNotEqual(k1, k2)

    def test_dry_run_writes_nothing(self):
        out = self.pin_file("1.8.0.14")
        rc, outp, err = run_main(make_pin.main, ["--channel", "stable", "--pin-id", "1.8.0.14", "--out", out,
                                                 "--game-acf", self.fs.game_acf, "--tools-acf", self.fs.tools_acf,
                                                 "--dry-run"], self.roots)
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(outp)["schema"], "dd-pin/2")
        self.assertIn("DRY RUN", err)
        self.assertFalse(os.path.exists(out))

    def test_without_channel_dd_pin_1_is_unchanged(self):
        pin = read_json(self.pin_file("1.8.0.13"))
        self.assertEqual(pin["schema"], "dd-pin/1")
        for key in ("channel", "build_key", "exe_sha256"):
            self.assertNotIn(key, pin)

    def test_tampered_build_key_is_a_config_error(self):
        pin = self.make_v2()
        pin["build_key"] = "0" * 64
        with open(self.pin_file("1.8.0.14"), "w") as fh:
            json.dump(pin, fh)
        rc, out, err, _ = self.launch(self.pin_file("1.8.0.14"))
        self.assertEqual((rc, out), (2, ""), err)
        self.assertIn("does not match its inputs", err)


# --------------------------------------------------------------------------- launch record, use_class


class LaunchRecordV2Tests(V2Case):
    def test_v2_record_carries_build_key_channel_use_class(self):
        pin = self.make_v2()
        rc, out, err, rec = self.launch(self.pin_file("1.8.0.14"), "--use-class", "probe")
        self.assertEqual(rc, 0, err)
        self.assertEqual(rec["schema"], "dd-launch/2")
        self.assertEqual((rec["build_key"], rec["build_key_source"]), (pin["build_key"], "pin"))
        self.assertEqual((rec["channel"], rec["pin_channel"], rec["launch_path_id"], rec["use_class"]),
                         ("stable", "stable", "stable", "probe"))
        self.assertEqual(rec["physics_hz_expected"], 30)
        names = sorted(c["name"] for c in rec["checks"])
        self.assertEqual(names, sorted(preflight.CHECK_NAMES + ("game.exe_sha256", "tools.exe_sha256")))
        self.assertTrue(all(c["ok"] for c in rec["checks"]))
        self.assertNotIn("engine_version_expected", rec)

    def test_default_use_class_is_dev(self):
        self.make_v2()
        rc, _, err, rec = self.launch(self.pin_file("1.8.0.14"))
        self.assertEqual(rc, 0, err)
        self.assertEqual(rec["use_class"], "dev")

    def test_production_and_unknown_use_class_refused(self):
        self.make_v2()
        for uc in ("production", "nonsense"):
            rc, out, err, _ = self.launch(self.pin_file("1.8.0.14"), "--use-class", uc)
            self.assertEqual((rc, out), (2, ""), (uc, err))
        self.assertFalse(os.path.exists(self.launches))

    def test_dd_pin_1_with_use_class_gets_v2_without_build_key(self):
        rc, _, err, rec = self.launch(self.pin_file("1.8.0.13"), "--use-class", "probe")
        self.assertEqual(rc, 0, err)
        self.assertEqual(rec["schema"], "dd-launch/2")
        self.assertNotIn("build_key", rec)  # unknown: left out, never null
        self.assertEqual((rec["channel"], rec["use_class"]), ("stable", "probe"))
        self.assertEqual(sorted(c["name"] for c in rec["checks"]), sorted(preflight.CHECK_NAMES))

    def test_changed_exe_content_refused_by_exe_sha256(self):
        self.make_v2()
        # same size and mtime: only the hash can tell
        self.fs.write(self.fs.game_root, "ArmaReforgerSteam.exe", b"MZ gamf", base.PIN_MTIME_NS)
        rc, out, err, _ = self.launch(self.pin_file("1.8.0.14"))
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("FAIL game.exe_sha256:", err)
        self.assertNotIn("FAIL game.binaries", err)

    def test_allow_fail_knows_the_new_checks(self):
        self.make_v2()
        self.fs.write(self.fs.game_root, "ArmaReforgerSteam.exe", b"MZ gamf", base.PIN_MTIME_NS)
        rc, out, _ = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.14"), "--check", "--allow-fail",
                                               "game.exe_sha256"], self.roots)
        self.assertEqual(rc, 0, out)
        self.assertIn("WAIVED game.exe_sha256", out)
        self.assertIn("BUILD  dd-pin/2 channel stable", out)

    def test_check_mode_names_the_build_key(self):
        pin = self.make_v2()
        rc, out, _ = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.14"), "--check"], self.roots)
        self.assertEqual(rc, 0, out)
        self.assertIn("build_key %s (from pin)" % pin["build_key"], out)


# --------------------------------------------------------------------------- experimental channel


class FakeExperimental:
    """The E: library layout of the Experimental apps 1890860/1890880."""

    def __init__(self, tmp):
        lib = os.path.join(tmp, "elib", "steamapps")
        self.game_acf = os.path.join(lib, "appmanifest_1890860.acf")
        self.tools_acf = os.path.join(lib, "appmanifest_1890880.acf")
        self.game_root = os.path.join(lib, "common", "Arma Reforger Experimental")
        self.tools_root = os.path.join(lib, "common", "Arma Reforger Experimental Tools")
        os.makedirs(lib)
        for name in ("appmanifest_1890860.acf", "appmanifest_1890880.acf"):
            shutil.copyfile(os.path.join(FIXTURES, name), os.path.join(lib, name))
        w = base.FakeSteam.write
        w(self.game_root, "ArmaReforgerSteam_Experimental.exe", b"MZ exp", base.PIN_MTIME_NS)
        w(self.game_root, "ArmaReforgerSteamDiag.exe", b"MZ exp diag", base.PIN_MTIME_NS)
        w(self.game_root, "steam_appid.txt", b"1890860", base.PIN_MTIME_NS)
        w(self.game_root, "addons/data/data.pak", b"exp-data", base.PIN_MTIME_NS)
        w(self.tools_root, "steam_appid.txt", b"1890880 ", base.PIN_MTIME_NS)
        w(self.tools_root, "Workbench/steam_appid.txt", b"1890880 ", base.PIN_MTIME_NS)
        w(self.tools_root, "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ exp wb", base.PIN_MTIME_NS)
        w(self.tools_root, "Workbench/addons/core/data.pak", b"exp-wb", base.PIN_MTIME_NS)

    def launch_paths(self):
        return {"game": {"acf": self.game_acf, "install": self.game_root},
                "tools": {"acf": self.tools_acf, "install": self.tools_root}}


class ExperimentalTests(V2Case):
    def setUp(self):
        super().setUp()
        self.exp = FakeExperimental(self.tmp)
        saved = pinlib.EXPERIMENTAL_LAUNCH_PATHS
        pinlib.EXPERIMENTAL_LAUNCH_PATHS = self.exp.launch_paths()
        self.addCleanup(setattr, pinlib, "EXPERIMENTAL_LAUNCH_PATHS", saved)
        rc, _, err = run_main(make_pin.main, ["--channel", "experimental", "--pin-id", "1.9.0.10-exp",
                                              "--out", self.pin_file("1.9.0.10-exp"), "--game-acf", self.exp.game_acf,
                                              "--tools-acf", self.exp.tools_acf], self.roots)
        self.assertEqual(rc, 0, err)
        self.exp_pin = read_json(self.pin_file("1.9.0.10-exp"))

    def test_experimental_pin_records_its_channel(self):
        p = self.exp_pin
        self.assertEqual((p["channel"], p["release_channel"], p["launch_path_id"]),
                         ("experimental", "experimental", "experimental"))
        self.assertEqual((p["game"]["appid"], p["tools"]["appid"]), (1890860, 1890880))
        self.assertEqual(p["build_key_inputs"]["game"]["exe"], "ArmaReforgerSteam_Experimental.exe")
        self.assertEqual(p["build_key"], spec_build_key(1890860, 25426149, sha(b"MZ exp"), 1890880, 25426160,
                                                        sha(b"MZ exp wb")))

    def test_stable_ids_refused_for_the_experimental_channel(self):
        rc, _, err = run_main(make_pin.main, ["--channel", "experimental", "--pin-id", "1.9.0.11-exp",
                                              "--out", self.pin_file("1.9.0.11-exp"), "--game-acf", self.fs.game_acf,
                                              "--tools-acf", self.fs.tools_acf], self.roots)
        self.assertEqual(rc, 1, err)
        self.assertIn("expected 1890860", err)

    def test_activation_refused_nothing_written(self):
        rc, out, err = self.activate("1.9.0.10-exp")
        self.assertEqual(rc, 1, out + err)
        self.assertIn("channel-experimental", err)
        for name in ("ACTIVE", "ACTIVE.log", "ACTIVE.lock"):
            self.assertFalse(os.path.lexists(os.path.join(self.pin_dir, name)), name)

    def test_pin_active_refuses_an_active_naming_it(self):
        self.put_active(b"1.9.0.10-exp\n")
        for argv in (["--exe", "wb"], ["--check"]):
            rc, out, err = self.preflight_active(*argv)
            self.assertEqual((rc, out), (2, ""), err)
            self.assertIn("channel-experimental", err)
        self.assertFalse(os.path.exists(self.launches))

    def test_explicit_pin_always_compat_preview(self):
        rc, _, err, rec = self.launch(self.pin_file("1.9.0.10-exp"), "--launch-path", "experimental")
        self.assertEqual(rc, 0, err)
        self.assertEqual((rec["channel"], rec["pin_channel"], rec["use_class"]),
                         ("experimental", "experimental", "compat-preview"))
        rc, out, err, _ = self.launch(self.pin_file("1.9.0.10-exp"), "--launch-path", "experimental",
                                      "--use-class", "dev")
        self.assertEqual((rc, out), (2, ""), err)

    def test_stable_launcher_refuses_the_experimental_pin(self):
        rc, out, err, _ = self.launch(self.pin_file("1.9.0.10-exp"))
        self.assertEqual((rc, out), (1, ""), err)
        self.assertIn("FAIL game.launch_path: pin 1.9.0.10-exp was made for launch path 'experimental'", err)


# --------------------------------------------------------------------------- build_key sidecar (dd-pin/1)


class SidecarTests(V2Case):
    def side_path(self, pin_id="1.8.0.13"):
        return os.path.join(self.pin_dir, "pin_%s.build_key.json" % pin_id)

    def test_dry_run_then_apply_from_live(self):
        rc, out, err = self.sidecar("1.8.0.13", "--from-live", "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertIn("DRY RUN", err)
        self.assertFalse(os.path.exists(self.side_path()))
        dry = json.loads(out)
        rc, out, err = self.sidecar("1.8.0.13", "--from-live", "--apply")
        self.assertEqual(rc, 0, err)
        side = read_json(self.side_path())
        self.assertEqual(side["build_key"], dry["build_key"])
        self.assertEqual(side["build_key"], spec_build_key(1874880, 24903726, sha(b"MZ game"), 1874910, 24870687,
                                                           sha(b"MZ wb")))
        self.assertEqual((side["pin_id"], side["pin_sha256"]), ("1.8.0.13", pinlib.sha256_file(self.pin_file("1.8.0.13"))))
        self.assertEqual(side["evidence"]["bytes_hashed"], len(b"MZ game") + len(b"MZ game diag") + len(b"MZ wb"))
        # the pin file itself is untouched
        self.assertEqual(read_json(self.pin_file("1.8.0.13"))["schema"], "dd-pin/1")

    def test_write_once(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        rc, _, err = self.sidecar("1.8.0.13", "--from-live", "--apply")
        self.assertEqual(rc, 2, err)
        self.assertIn("written once", err)

    def test_refused_when_the_live_installs_do_not_match(self):
        base.fake_steam_update(self.fs)
        rc, _, err = self.sidecar("1.8.0.13", "--from-live", "--apply")
        self.assertEqual(rc, 1, err)
        self.assertFalse(os.path.exists(self.side_path()))

    def test_launch_with_sidecar_is_v2_and_hashes_exes(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        side = read_json(self.side_path())
        self.put_active(b"1.8.0.13\n")
        rc, out, err = self.preflight_active("--exe", "wb")
        self.assertEqual(rc, 0, err)
        rec = read_json(os.path.join(self.launches, "launch_%s.json" % out.strip()))
        self.assertEqual((rec["schema"], rec["build_key"], rec["build_key_source"]),
                         ("dd-launch/2", side["build_key"], "sidecar"))
        self.assertIn("tools.exe_sha256", [c["name"] for c in rec["checks"]])
        # activation re-checks with the hashes too
        rc, out, err = self.activate("1.8.0.13")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("build_key %s (from sidecar)" % side["build_key"], out)

    def test_sidecar_is_bound_to_the_pin_bytes(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        with open(self.pin_file("1.8.0.13"), "a") as fh:
            fh.write("\n")
        rc, out, err = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13"), "--check"], self.roots)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("belongs to pin", err)

    def test_tampered_sidecar_is_a_config_error(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        side = read_json(self.side_path())
        side["build_key"] = "f" * 64
        os.chmod(self.side_path(), 0o644)
        with open(self.side_path(), "w") as fh:
            json.dump(side, fh)
        rc, out, err = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13"), "--check"], self.roots)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("does not match its inputs", err)

    def test_dd_pin_2_needs_no_sidecar(self):
        self.make_v2()
        rc, _, err = self.sidecar("1.8.0.14", "--from-live", "--dry-run")
        self.assertEqual(rc, 2, err)
        self.assertIn("carries its own build_key", err)

    def make_copy(self, pin_id="1.8.0.13", corrupt=False):
        """<pin dir>/<id>/{game,tools,SHA256SUMS} as cold_copy.sh leaves it."""
        copy = os.path.join(self.pin_dir, pin_id)
        lines = []
        for app, root in (("game", self.fs.game_root), ("tools", self.fs.tools_root)):
            shutil.copytree(root, os.path.join(copy, app))
            for dp, _, fns in os.walk(os.path.join(copy, app)):
                for fn in sorted(fns):
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, copy).replace(os.sep, "/")
                    lines.append("%s  %s" % (pinlib.sha256_file(full), rel))
        with open(os.path.join(copy, "SHA256SUMS"), "w") as fh:
            fh.write("\n".join(sorted(lines, key=lambda l: l[66:])) + "\n")
        if corrupt:
            base.FakeSteam.write(os.path.join(copy, "tools"), "Workbench/ArmaReforgerWorkbenchSteamDiag.exe", b"MZ wc",
                                 base.TOOLS_MTIME_NS)
        return copy

    def test_from_sums_after_the_installs_moved_on(self):
        self.make_copy()
        base.fake_steam_update(self.fs)  # live installs are now another build
        rc, out, err = self.sidecar("1.8.0.13", "--from-sums", "--apply")
        self.assertEqual(rc, 0, err)
        side = read_json(self.side_path())
        self.assertEqual(side["evidence"]["source"], "sums")
        self.assertEqual(side["build_key"], spec_build_key(1874880, 24903726, sha(b"MZ game"), 1874910, 24870687,
                                                           sha(b"MZ wb")))

    def test_from_sums_sidecar_is_anchored_to_its_sha256sums(self):
        copy = self.make_copy()
        self.assertEqual(self.sidecar("1.8.0.13", "--from-sums", "--apply")[0], 0)
        self.assertEqual(run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13"), "--check"], self.roots)[0], 0)
        with open(os.path.join(copy, "SHA256SUMS"), "a") as fh:
            fh.write("%s  game/extra.txt\n" % ("0" * 64))
        rc, out, err = run_main(preflight.main, ["--pin", self.pin_file("1.8.0.13"), "--check"], self.roots)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("changed since", err)

    def test_from_sums_refuses_a_copy_that_does_not_match(self):
        self.make_copy(corrupt=True)
        rc, _, err = self.sidecar("1.8.0.13", "--from-sums", "--apply")
        self.assertEqual(rc, 1, err)
        self.assertIn("does not match SHA256SUMS", err)

    def test_restore_r1_gets_the_reference_build_key(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        ref_key = read_json(self.side_path())["build_key"]
        self.assertEqual(self.activate("1.8.0.13")[0], 0)
        base.fake_steam_update(self.fs)
        base.restore_old_build(self.fs)
        pin = self.make_v2("1.8.0.13-r1", "--restore-of", "1.8.0.13")
        self.assertEqual((pin["build_key"], pin["restore_of"]["build_key"]), (ref_key, ref_key))

    def test_restore_with_another_exe_is_refused(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        self.fs.write(self.fs.game_root, "ArmaReforgerSteam.exe", b"MZ gamf", base.PIN_MTIME_NS)
        rc, _, err = run_main(make_pin.main, ["--channel", "stable", "--pin-id", "1.8.0.13-r1", "--restore-of",
                                              "1.8.0.13", "--out", self.pin_file("1.8.0.13-r1"), "--game-acf",
                                              self.fs.game_acf, "--tools-acf", self.fs.tools_acf], self.roots)
        self.assertEqual(rc, 1, err)
        self.assertIn("build_key", err)


# --------------------------------------------------------------------------- engine sidecar


class AtomicPublishTests(V2Case):
    """Codex round 1: a write-once sidecar is published atomically; a partial write never blocks a retry."""

    def side_path(self):
        return os.path.join(self.pin_dir, "pin_1.8.0.13.build_key.json")

    def test_short_write_leaves_nothing_and_a_retry_works(self):
        real = pinlib._write_all

        def short(fd, data):
            real(fd, data[:9])
            raise OSError(28, "No space left on device (simulated)")
        pinlib._write_all = short
        try:
            rc, _, err = self.sidecar("1.8.0.13", "--from-live", "--apply")
        finally:
            pinlib._write_all = real
        self.assertEqual(rc, 2, err)
        self.assertFalse(os.path.lexists(self.side_path()))
        self.assertEqual([n for n in os.listdir(self.pin_dir) if ".tmp-" in n], [])
        rc, _, err = self.sidecar("1.8.0.13", "--from-live", "--apply")
        self.assertEqual(rc, 0, err)
        self.assertEqual(read_json(self.side_path())["pin_id"], "1.8.0.13")

    def test_leftover_temp_file_does_not_block(self):
        with open(os.path.join(self.pin_dir, ".pin_1.8.0.13.build_key.json.tmp-1-deadbeef"), "w") as fh:
            fh.write("{partial")
        rc, _, err = self.sidecar("1.8.0.13", "--from-live", "--apply")
        self.assertEqual(rc, 0, err)

    def test_never_replaces(self):
        target = os.path.join(self.tmp, "x.json")
        pinlib.write_json_once(target, {"a": 1}, self.roots)
        with self.assertRaises(pinlib.PinError):
            pinlib.write_json_once(target, {"a": 2}, self.roots)
        self.assertEqual(read_json(target), {"a": 1})
        # without link(), renameat2(RENAME_NOREPLACE) keeps the no-replace rule
        real_link = os.link
        os.link = lambda a, b: (_ for _ in ()).throw(OSError(1, "Operation not permitted"))
        try:
            with self.assertRaises(pinlib.PinError):
                pinlib.write_json_once(target, {"a": 3}, self.roots)
            pinlib.write_json_once(os.path.join(self.tmp, "y.json"), {"b": 1}, self.roots)
        finally:
            os.link = real_link
        self.assertEqual((read_json(target), read_json(os.path.join(self.tmp, "y.json"))), ({"a": 1}, {"b": 1}))

    def test_fails_closed_without_an_atomic_no_replace(self):
        # Codex round 2 N6: no check-then-rename fallback; nothing is written and no temp file is left
        real_link, real_rn = os.link, pinlib._rename_noreplace
        os.link = lambda a, b: (_ for _ in ()).throw(OSError(1, "Operation not permitted"))
        pinlib._rename_noreplace = lambda a, b: False
        try:
            with self.assertRaises(pinlib.PinError) as cm:
                pinlib.write_json_once(os.path.join(self.tmp, "z.json"), {"c": 1}, self.roots)
        finally:
            os.link, pinlib._rename_noreplace = real_link, real_rn
        self.assertIn("cannot be published atomically", str(cm.exception))
        self.assertFalse(os.path.lexists(os.path.join(self.tmp, "z.json")))
        self.assertEqual([n for n in os.listdir(self.tmp) if ".tmp-" in n], [])


class EngineSidecarTests(V2Case):
    def header(self, pin_id="1.8.0.13", pin_sha=None, version="1.8.0.13", build_key=""):
        path = os.path.join(self.tmp, "run_header.txt")
        with open(path, "w") as fh:
            fh.write("launch.mode=cli\nlaunch.id=20261106T052424Z-wb-addc\nlaunch.pin_id=%s\nlaunch.pin_sha256=%s\n"
                     "launch.build_key=%s\nengine.build_version=%s\nengine.build_time=2026-08-21 7:30:18\n"
                     "engine.fixed_time_slice_s=0.033333\nengine.exe=wb\n"
                     % (pin_id, pin_sha or pinlib.sha256_file(self.pin_file("1.8.0.13")), build_key, version))
        return path

    def engine(self, path, *mode):
        return run_main(pin_sidecar.main, ["engine", "1.8.0.13", "--pin-dir", self.pin_dir, "--run-header", path]
                        + list(mode or ["--apply"]), self.roots)

    def test_written_and_copied_into_the_record(self):
        rc, out, err = self.engine(self.header())
        self.assertEqual(rc, 0, err)
        side = read_json(os.path.join(self.pin_dir, "pin_1.8.0.13.engine.json"))
        self.assertEqual((side["engine_version_expected"], side["fixed_time_slice_s_measured"]), ("1.8.0.13", 0.033333))
        rc, _, err, rec = self.launch(self.pin_file("1.8.0.13"))
        self.assertEqual(rc, 0, err)
        self.assertEqual((rec["schema"], rec["engine_version_expected"]), ("dd-launch/2", "1.8.0.13"))
        self.assertNotIn("build_key", rec)  # no build_key sidecar in this case

    def test_header_of_another_pin_refused(self):
        for path in (self.header(pin_id="1.9.0.37"), self.header(pin_sha="0" * 64)):
            rc, _, err = self.engine(path)
            self.assertEqual(rc, 1, err)
        self.assertFalse(os.path.exists(os.path.join(self.pin_dir, "pin_1.8.0.13.engine.json")))

    def test_header_with_another_build_key_refused(self):
        self.assertEqual(self.sidecar("1.8.0.13", "--from-live", "--apply")[0], 0)
        rc, _, err = self.engine(self.header(build_key="e" * 64))
        self.assertEqual(rc, 1, err)


# --------------------------------------------------------------------------- freeze check


def write_launch_log(profile, launch_id, start_unix, version="1.8.0.13", name=None):
    """A fixture console.log of a Workbench start (the evidence a real launch leaves), not a preflight run."""
    import datetime
    t = datetime.datetime.fromtimestamp(start_unix, datetime.timezone.utc)
    d = os.path.join(profile, "logs", name or "logs_" + t.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(d, exist_ok=True)
    lines = ["---", "Log %s/console.log started at %s (%s UTC)" % (d, t.strftime("%Y-%m-%d %H:%M:%S"),
                                                                  t.strftime("%Y-%m-%d %H:%M:%S")),
             "00:00:01.000   ENGINE       : CLI Params: -profile %s -gproj x.gproj -dcLaunchId %s" % (profile, launch_id)]
    if version:
        lines.append("00:00:08.000   INIT         : Creating game instance(ArmaReforgerScripted), version %s built "
                     "2026-08-21 14:30:18 UTC." % version)
    with open(os.path.join(d, "console.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return d


class FreezeCase(V2Case):
    """Helpers of the freeze-check tests (no tests here; tests/test_compat19_static.py reuses them)."""

    def setUp(self):
        super().setUp()
        self.put_active(b"1.8.0.13\n")
        self.profile = os.path.join(self.tmp, "wb_profile")
        saved = pinlib.WB_LAUNCH
        pinlib.WB_LAUNCH = {"stable": {"profile": self.profile, "gproj": "x.gproj", "launcher": "wb_R1.bat"}}
        self.addCleanup(setattr, pinlib, "WB_LAUNCH", saved)

    def before(self, steam="online"):
        rc, out, err = run_main(freeze_check.main, ["before", "--pin-active", "--pin-dir", self.pin_dir, "--steam",
                                                    steam], self.roots)
        self.assertEqual(rc, 0, err)
        return os.path.join(self.pin_dir, "freeze", [n for n in os.listdir(os.path.join(self.pin_dir, "freeze"))
                                                     if n.endswith(".before.json")][-1])

    def after(self, before):
        return run_main(freeze_check.main, ["after", "--before", before], self.roots)

    def preflight_only(self):
        rc, out, err = self.preflight_active("--exe", "wb")
        self.assertEqual(rc, 0, err)
        return out.strip()

    def user_launch(self, version="1.8.0.13"):
        """What the launcher and a real Workbench start leave: the preflight record, then the start's console.log."""
        lid = self.preflight_only()
        rec = read_json(os.path.join(self.launches, "launch_%s.json" % lid))
        write_launch_log(self.profile, lid, pinlib.iso_utc_to_unix(rec["utc"]) + 2, version)
        return lid


class FreezeTests(FreezeCase):
    def test_pass_with_a_launch_in_between(self):
        b = self.before()
        self.user_launch()
        self.fs.set_acf("tools", "LastPlayed", "1792553999")  # Steam rewrites LastPlayed on a launch
        rc, out, err = self.after(b)
        self.assertEqual(rc, 0, err)
        rec = json.loads(out)
        self.assertEqual((rec["verdict"], rec["apps_launched"], rec["qualified_5_5"]), ("PASS", ["tools"], True))
        self.assertEqual(len(rec["launch_evidence"]), 1)
        self.assertEqual(len(rec["notes"]), 1)
        self.assertIn("LastPlayed", rec["notes"][0])
        self.assertTrue(os.path.exists(b.replace(".before.json", ".after.json")))

    def test_preflight_only_launch_fails(self):
        # Codex round 1: a preflight record without a real start (the launcher stopped after it) never passes
        b = self.before()
        self.preflight_only()
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)
        rec = json.loads(out)
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertIn("no Workbench start was observed", rec["problems"][0])

    def test_log_from_before_the_before_record_is_no_evidence(self):
        b = self.before()
        lid = self.preflight_only()
        write_launch_log(self.profile, lid, pinlib.iso_utc_to_unix(read_json(b)["utc"]) - 30)
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)

    def test_offline_is_unqualified(self):
        b = self.before(steam="offline")
        self.user_launch()
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)
        rec = json.loads(out)
        self.assertEqual((rec["verdict"], rec["qualified_5_5"]), ("PASS_UNQUALIFIED", False))
        self.assertEqual(freeze_check.find_pass(self.pin_dir, pinlib.sha256_file(self.pin_file("1.8.0.13")), None), [])

    def test_wrong_build_in_the_log(self):
        rc, _, err = run_main(pin_sidecar.main, ["engine", "1.8.0.13", "--pin-dir", self.pin_dir, "--run-header",
                                                 EngineSidecarTests.header(self), "--apply"], self.roots)
        self.assertEqual(rc, 0, err)
        b = self.before()
        self.user_launch(version="1.9.0.37")
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)
        self.assertIn("reports version 1.9.0.37, the pin expects 1.8.0.13", json.loads(out)["problems"][0])

    def test_malformed_launch_records_are_skipped(self):
        b = self.before()
        os.makedirs(self.launches, exist_ok=True)
        with open(os.path.join(self.launches, "launch_20261106T000000Z-wb-0000.json"), "w") as fh:
            fh.write("[]")
        with open(os.path.join(self.launches, "launch_20261106T000001Z-wb-0001.json"), "w") as fh:
            json.dump({"launch_id": "20261106T000001Z-wb-0001", "status": "ok"}, fh)
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)
        self.assertEqual(json.loads(out)["launches"], [])

    def test_fail_without_a_launch(self):
        b = self.before()
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)
        self.assertIn("no launch record", json.loads(out)["problems"][0])

    def test_fail_when_steam_changed_the_build(self):
        b = self.before()
        self.user_launch()
        self.fs.set_acf("game", "TargetBuildID", "25500000")
        rc, out, err = self.after(b)
        self.assertEqual(rc, 1, err)
        self.assertIn("game.target_buildid 24903726 -> 25500000", json.loads(out)["identity_diffs"])

    def test_before_refused_when_the_pin_does_not_match(self):
        base.fake_steam_update(self.fs)
        rc, _, err = run_main(freeze_check.main, ["before", "--pin-active", "--pin-dir", self.pin_dir, "--steam",
                                                  "online"], self.roots)
        self.assertEqual(rc, 1, err)
        self.assertFalse(os.path.exists(os.path.join(self.pin_dir, "freeze")))

    def test_dry_runs_write_nothing(self):
        rc, _, err = run_main(freeze_check.main, ["before", "--pin-active", "--pin-dir", self.pin_dir, "--steam",
                                                  "online", "--dry-run"], self.roots)
        self.assertEqual(rc, 0, err)
        self.assertFalse(os.path.exists(os.path.join(self.pin_dir, "freeze")))


# --------------------------------------------------------------------------- launchers and launch paths


if __name__ == "__main__":
    unittest.main()
