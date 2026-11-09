"""Unit tests for tools/local_config.py, the local machine paths of DD-22. Stdlib only.

Every test reads synthetic configs: tests/fixtures/local_config/placeholder.local.env (placeholder drives X:, Y:, Z:)
or text built here. Lookup tests copy the loader into a temporary tree under /dev/shm (or $TMPDIR when it lies there),
so neither this machine's config nor the repo's own config/ folder is ever read.
"""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER = os.path.join(REPO, "tools", "local_config.py")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "local_config", "placeholder.local.env")
EXAMPLE = os.path.join(REPO, "drivedit.local.env.example")


def load_module(path=LOADER, name="dd_test_local_config"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = load_module()


def read(path):
    with open(path, encoding="ascii") as fh:
        return fh.read()


def temp_dir():
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not (base.startswith("/dev/shm/") and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)):
        raise unittest.SkipTest("no writable /dev/shm")
    return tempfile.mkdtemp(prefix="dd22_local_config_", dir=base)


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = lc.parse(read(FIXTURE), FIXTURE)

    def test_fixture_values_and_forms(self):
        self.assertEqual(sorted(self.cfg.values), sorted(lc.KEYS))
        self.assertEqual(self.cfg.get("DATA_ROOT"), "/mnt/x/DataRoot")
        self.assertEqual(self.cfg.win("DATA_ROOT"), "X:\\DataRoot")
        self.assertEqual(self.cfg.win("TOOLS_INSTALL"), "Z:\\Synthetic Steam (x86)\\steamapps\\common\\Arma Reforger Tools")
        self.assertEqual((self.cfg.drive("GAME_INSTALL"), self.cfg.drive_win("GAME_INSTALL")), ("/mnt/y", "Y:"))
        self.assertEqual(self.cfg.get("SYSTEM_DRIVE"), "/mnt/z")
        self.assertEqual(self.cfg.get("WSL_DISTRO"), "Placeholder-Distro")

    def test_fixture_uses_placeholder_drives_only(self):
        for key, value in self.cfg.values.items():
            d = lc.drive_of(value)
            self.assertIn(d, (None, "/mnt/x", "/mnt/y", "/mnt/z"), key)

    def test_missing_key_names_the_key_and_the_file(self):
        cfg = lc.parse("DATA_ROOT=/mnt/x/DataRoot\n", "/somewhere/drivedit.local.env")
        with self.assertRaises(lc.LocalConfigError) as caught:
            cfg.get("GAME_INSTALL")
        msg = str(caught.exception)
        self.assertIn("GAME_INSTALL", msg)
        self.assertIn("/somewhere/drivedit.local.env", msg)
        with self.assertRaisesRegex(lc.LocalConfigError, "not a local config key"):
            cfg.get("NOPE")

    def test_malformed_lines_are_all_reported(self):
        text = ("# comment\n\nDATA_ROOT=/mnt/x/Data Root\nGAME_INSTALL=Y:\\Lib\\steamapps\\common\\A\n"
                "TOOLS_INSTALL=/mnt/z/T/steamapps/common/B/\nWSL_DISTRO=a b\nBOGUS=1\n DATA_ROOT=/mnt/x/a\n"
                "REALIGN_ROOT=/mnt/x/a\nREALIGN_ROOT=/mnt/x/b\nSYSTEM_DRIVE=/mnt/Z\nCODE_ROOT=/mnt/x/<code>\n"
                "EXPERIMENTAL_GAME_INSTALL=/mnt/x/no/steam/layout\nno equals sign\nCODE_ROOT=relative/path\n")
        with self.assertRaises(lc.LocalConfigError) as caught:
            lc.parse(text, "bad.env")
        msg = str(caught.exception)
        for fragment in ("line 3: DATA_ROOT holds a character", "line 4: GAME_INSTALL", "line 5: TOOLS_INSTALL is not a "
                         "normalised path", "line 6: WSL_DISTRO", "line 7: unknown key BOGUS", "line 8:",
                         "line 10: REALIGN_ROOT is set twice", "line 11: SYSTEM_DRIVE", "line 12: CODE_ROOT still "
                         "holds a placeholder", "line 13: EXPERIMENTAL_GAME_INSTALL is not a Steam install dir",
                         "line 14: expected KEY=VALUE", "line 15: CODE_ROOT"):
            self.assertIn(fragment, msg)

    def test_value_characters(self):
        # cmd.exe metacharacters never reach a value; DATA_ROOT and WSL_DISTRO, unquoted on the wsl.exe line, are
        # stricter still
        for bad in ("&", "|", "<", ">", "^", "%", "!", '"', "'", ";", "=", "\\"):
            self.assertIsNotNone(lc._value_problem("TOOLS_INSTALL", "/mnt/z/a%sb/steamapps/common/T" % bad), bad)
        for ok in (" ", "(", ")", "+", ",", "~", ".", "_", "-"):
            self.assertIsNone(lc._value_problem("TOOLS_INSTALL", "/mnt/z/a%sb/steamapps/common/T" % ok), ok)
            if ok not in ".-_":
                self.assertIsNotNone(lc._value_problem("DATA_ROOT", "/mnt/x/a%sb" % ok), ok)
        self.assertIsNone(lc._value_problem("CODE_ROOT", "/dev/shm/code"))  # any absolute path
        self.assertIsNotNone(lc._value_problem("REALIGN_ROOT", "/dev/shm/plan"))  # a Windows drive is required

    def test_crlf_and_ascii(self):
        self.assertEqual(lc.parse(read(FIXTURE).replace("\n", "\r\n")).values, self.cfg.values)

    def test_mapping(self):
        self.assertEqual(lc.wsl_to_win("/mnt/x/a/b"), "X:\\a\\b")
        self.assertEqual(lc.wsl_to_win("/mnt/x"), "X:\\")
        self.assertIsNone(lc.wsl_to_win("/dev/shm/a"))
        self.assertEqual(lc.win_to_wsl("X:\\a\\b"), "/mnt/x/a/b")
        self.assertEqual(lc.win_to_wsl("y:/a"), "/mnt/y/a")
        self.assertEqual(lc.win_to_wsl("/mnt/x/a"), "/mnt/x/a")
        self.assertIsNone(lc.drive_of("/dev/shm"))

    def test_example_lists_every_key_and_only_placeholders(self):
        text = read(EXAMPLE)
        keys = [line.split("=", 1)[0] for line in text.splitlines() if line and not line.startswith("#")]
        self.assertEqual(keys, list(lc.KEYS))
        with self.assertRaises(lc.LocalConfigError) as caught:
            lc.parse(text, EXAMPLE)
        self.assertEqual(str(caught.exception).count("still holds a placeholder"), len(lc.KEYS))

    def test_render(self):
        self.assertEqual(lc.render("<DATA_ROOT_WIN>\\pin <GAME_INSTALL> <LID> <OTHER>", self.cfg),
                         "X:\\DataRoot\\pin /mnt/y/Game Library/steamapps/common/Arma Reforger <LID> <OTHER>")

    def test_bat_emulation(self):
        text = ('@echo off\r\nset "A=%DD_DATA_ROOT_WIN%\\x"\r\nset "B=%A%\\y %UNSET%"\r\n'
                'start "" /D "%A%" "%B%" -id %LID%\r\n')
        env = lc.bat_env(text, lc.bat_values(self.cfg))
        self.assertEqual(env["B"], "X:\\DataRoot\\x\\y %UNSET%")
        self.assertEqual(lc.bat_start_line(text, lc.bat_values(self.cfg)),
                         'start "" /D "X:\\DataRoot\\x" "X:\\DataRoot\\x\\y %UNSET%" -id <LID>')


class LookupTests(unittest.TestCase):
    """The lookup order, on a copy of the loader in a temporary tree (its <tools dir> is <tmp>/tools)."""

    def setUp(self):
        self.tmp = temp_dir()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.makedirs(os.path.join(self.tmp, "tools"))
        shutil.copyfile(LOADER, os.path.join(self.tmp, "tools", "local_config.py"))
        self.mod = load_module(os.path.join(self.tmp, "tools", "local_config.py"), "dd_test_local_config_copy")
        self.installed = os.path.join(self.tmp, "config", "drivedit.local.env")
        self.repo_file = os.path.join(self.tmp, "drivedit.local.env")

    def put(self, path, text=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as fh:
            fh.write(read(FIXTURE) if text is None else text)
        return path

    def test_nothing_found_names_every_place(self):
        with self.assertRaises(self.mod.LocalConfigError) as caught:
            self.mod.find({})
        msg = str(caught.exception)
        self.assertIn(self.installed, msg)
        self.assertNotIn(self.repo_file, msg)  # no .git: not a checkout, so the repo-root file is not looked for
        self.assertIn("DRIVEDIT_LOCAL_CONFIG", msg)

    def test_installed_location(self):
        self.put(self.installed)
        self.assertEqual(self.mod.find({}), self.installed)
        self.assertEqual(self.mod.load(self.mod.find({})).get("DATA_ROOT"), "/mnt/x/DataRoot")

    def test_repo_root_file_only_in_a_checkout(self):
        self.put(self.repo_file)
        with self.assertRaises(self.mod.LocalConfigError):
            self.mod.find({})
        os.makedirs(os.path.join(self.tmp, ".git"))
        self.assertEqual(self.mod.find({}), self.repo_file)
        self.put(self.installed)
        self.assertEqual(self.mod.find({}), self.installed)  # the installed location comes first

    def test_env_var_wins_and_never_falls_back(self):
        self.put(self.installed)
        other = self.put(os.path.join(self.tmp, "elsewhere", "my.env"))
        self.assertEqual(self.mod.find({"DRIVEDIT_LOCAL_CONFIG": other}), other)
        with self.assertRaisesRegex(self.mod.LocalConfigError, "not a file"):
            self.mod.find({"DRIVEDIT_LOCAL_CONFIG": other + ".missing"})
        self.assertEqual(self.mod.find({"DRIVEDIT_LOCAL_CONFIG": "  "}), self.installed)  # blank counts as unset

    def test_env_var_in_windows_form(self):
        self.assertEqual(self.mod.candidates({"DRIVEDIT_LOCAL_CONFIG": "X:\\DataRoot\\config\\drivedit.local.env"})[0],
                         ("$DRIVEDIT_LOCAL_CONFIG", "/mnt/x/DataRoot/config/drivedit.local.env"))

    def test_reread_when_changed(self):
        path = self.put(self.installed)
        self.assertEqual(self.mod.load(path).get("WSL_DISTRO"), "Placeholder-Distro")
        self.put(path, read(FIXTURE).replace("Placeholder-Distro", "Other-Distro-Name"))
        self.assertEqual(self.mod.load(path).get("WSL_DISTRO"), "Other-Distro-Name")

    def test_import_reads_nothing(self):
        # A fresh import with no config anywhere: nothing raises until a value is asked for.
        mod = load_module(os.path.join(self.tmp, "tools", "local_config.py"), "dd_test_local_config_fresh")
        self.assertEqual(mod.KEYS, lc.KEYS)

    def test_cli(self):
        path = self.put(self.installed)
        script = os.path.join(self.tmp, "tools", "local_config.py")
        env = dict(os.environ)
        env.pop("DRIVEDIT_LOCAL_CONFIG", None)

        def run(*args):
            r = subprocess.run([sys.executable, "-I", "-B", script] + list(args), capture_output=True, text=True,
                               env=env, cwd=self.tmp)
            return r.returncode, r.stdout, r.stderr

        self.assertEqual(run("where"), (0, path + "\n", ""))
        self.assertEqual(run("get", "--win", "GAME_INSTALL")[1], "Y:\\Game Library\\steamapps\\common\\Arma Reforger\n")
        self.assertEqual(run("get", "--drive-win", "DATA_ROOT")[1], "X:\n")
        rc, out, err = run("get", "NOPE")
        self.assertEqual(rc, 2, err)
        self.assertIn("not a local config key", err)
        rc, out, _ = run("show")
        self.assertEqual(rc, 0)
        self.assertIn("X:\\DataRoot", out)
        rc, out, _ = run("check")  # the placeholder paths do not exist here
        self.assertEqual(rc, 1, out)
        self.assertRegex(out, r"(?m)^PASS key\.DATA_ROOT ")
        self.assertRegex(out, r"(?m)^FAIL exists\.DATA_ROOT ")
        os.remove(path)
        rc, out, err = run("check")
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("no local config", err)


class CheckTests(unittest.TestCase):
    def test_check_reports_every_key(self):
        cfg = lc.parse(read(FIXTURE), FIXTURE)
        results = lc.check(cfg)
        names = [r[1] for r in results]
        for key in lc.KEYS:
            self.assertIn("key." + key, names)
        self.assertIn(("PASS", "data_root.off_system_drive"), [(r[0], r[1]) for r in results])
        self.assertTrue(all(r[0] in ("PASS", "FAIL", "SKIP") for r in results))

    def test_check_flags_a_missing_key_and_the_system_drive(self):
        cfg = lc.parse("DATA_ROOT=/mnt/z/DataRoot\nSYSTEM_DRIVE=/mnt/z\n", "partial.env")
        status = {r[1]: r[0] for r in lc.check(cfg)}
        self.assertEqual(status["key.GAME_INSTALL"], "FAIL")
        self.assertEqual(status["data_root.off_system_drive"], "FAIL")


class GitignoreTests(unittest.TestCase):
    def test_local_configs_are_ignored_and_the_example_is_not(self):
        if not shutil.which("git") or not os.path.exists(os.path.join(REPO, ".git")):
            self.skipTest("not a git checkout")

        def ignored(rel):
            return subprocess.run(["git", "-C", REPO, "check-ignore", "-q", "--no-index", rel]).returncode == 0

        for rel in ("drivedit.local.env", "config/drivedit.local.env", "tools/drivedit.local.env"):
            self.assertTrue(ignored(rel), rel)
        for rel in ("drivedit.local.env.example", "tests/fixtures/local_config/placeholder.local.env",
                    "tools/launch/dd_local_config.cmd"):
            self.assertFalse(ignored(rel), rel)


if __name__ == "__main__":
    unittest.main()
