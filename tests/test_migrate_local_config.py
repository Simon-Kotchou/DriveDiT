"""Tests for tools/migrate_local_config.py (DD-22). Stdlib only.

Three kinds of test:
* The preservation proof (HistoryTest): the last pre-DD-22 revision, read with git show at run time (no copy of it is
  committed), against the current code: every old launcher line, path table, write root and runtime getter must come
  out the same. It runs only where that revision is in the history and its data drive is this machine's verified data
  drive (the machine whose paths it holds); everywhere else, CI included, it skips, and a failure names only a count.
* The mechanics (MigrateTest), on a synthetic old revision in RAM: the placeholder machine of
  tests/fixtures/local_config/placeholder.local.env written out as pre-DD-22 literals. Its launchers are today's with
  the helper's values written in, so these tests prove derivation and refusals, not preservation. The boundary rows
  run in the fake world of tests/fixtures/local_config/boundary_vectors.py, the runtime getters in a child interpreter.
* The runtime getters are the code's own (GetterTest): a copy of the repo in RAM whose dataset default is changed
  makes the builder row differ, which a restated formula would not.
Temporary trees go under /dev/shm and are removed.
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
FIXTURE = os.path.join(REPO, "tests", "fixtures", "local_config", "placeholder.local.env")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mig = load("dd_test_migrate_local_config", "tools/migrate_local_config.py")
lc = mig.lc
bv = load("dd22_boundary_vectors", "tests/fixtures/local_config/boundary_vectors.py")
CFG = lc.load(FIXTURE)
with open(FIXTURE, encoding="ascii") as _fh:
    FIXTURE_TEXT = _fh.read()
SYS = CFG.get("SYSTEM_DRIVE")
REV = "f145aff"   # the last pre-DD-22 main commit

DATA, DATA_WIN = "/mnt/x/DataRoot", "X:\\DataRoot"
GAME = "/mnt/y/Game Library/steamapps/common/Arma Reforger"
TOOLS = "/mnt/z/Synthetic Steam (x86)/steamapps/common/Arma Reforger Tools"
EXP = "/mnt/x/Experimental Library/steamapps"

# The literal tables of a pre-DD-22 tools/pin/pinlib.py, written out by hand for the placeholder machine.
LEGACY_PINLIB = '''
DEFAULT_ACF = {"game": "/mnt/y/Game Library/steamapps/appmanifest_1874880.acf",
               "tools": "/mnt/z/Synthetic Steam (x86)/steamapps/appmanifest_1874910.acf"}
DEFAULT_ACF_BY_CHANNEL = {"stable": DEFAULT_ACF,
                          "experimental": {"game": "%(exp)s/appmanifest_1890860.acf",
                                           "tools": "%(exp)s/appmanifest_1890880.acf"}}
LAUNCH_PATHS = {"game": {"acf": DEFAULT_ACF["game"], "install": "%(game)s"},
                "tools": {"acf": DEFAULT_ACF["tools"], "install": "%(tools)s"}}
COLD_RUN_ROOTS = {"cold-1.8.0.13": "/mnt/x/DataRoot/pin/1.8.0.13-run"}
COLD_LAUNCH_PATHS = {"cold-1.8.0.13": {
    "game": {"acf": "/mnt/x/DataRoot/pin/1.8.0.13-run/meta/appmanifest_1874880.acf",
             "install": "/mnt/x/DataRoot/pin/1.8.0.13-run/game"},
    "tools": {"acf": "/mnt/x/DataRoot/pin/1.8.0.13-run/meta/appmanifest_1874910.acf",
              "install": "/mnt/x/DataRoot/pin/1.8.0.13-run/tools"}}}
EXPERIMENTAL_LAUNCH_PATHS = {
    "game": {"acf": "%(exp)s/appmanifest_1890860.acf", "install": "%(exp)s/common/Arma Reforger Experimental"},
    "tools": {"acf": "%(exp)s/appmanifest_1890880.acf", "install": "%(exp)s/common/Arma Reforger Experimental Tools"}}
WB_LAUNCH = {
    "stable": {"profile": "X:\\\\DataRoot\\\\wb_profile", "gproj": "X:\\\\DataRoot\\\\addons\\\\DataCapture\\\\addon.gproj",
               "launcher": "wb_R1.bat"},
    "cold-1.8.0.13": {"profile": "X:\\\\DataRoot\\\\wb_profile_c18",
                      "gproj": "X:\\\\DataRoot\\\\addons_c18\\\\DataCapture\\\\addon.gproj", "launcher": "wb_C18.bat"}}
DEPLOY_ROOTS = {"stable": "/mnt/x/DataRoot/addons", "cold-1.8.0.13": "/mnt/x/DataRoot/addons_c18"}
DEFAULT_WRITE_ROOTS = ("/mnt/x/",)
FORBIDDEN_WRITE_PREFIXES = ("/mnt/z/",)
WRITE_ROOT_DRIVES = {"/mnt/x": "X:"}
DEFAULT_PIN_DIR = "/mnt/x/DataRoot/pin"
''' % {"exp": EXP, "game": GAME, "tools": TOOLS}

LEGACY_DEPLOY = '''
DEFAULT_TARGET = "/mnt/x/DataRoot/addons/DataCapture"
ALLOWED_TARGET_ROOTS = ("/mnt/x/DataRoot/addons",)
ALLOW_ROOT_PREFIXES = ("/mnt/x/", "/dev/shm/")
FORBIDDEN_DEVICE_PATHS = (("/mnt/z", "the system drive"), ("/", "the WSL disk image"))
DEFAULT_DEPLOY_ROOTS = {"stable": "/mnt/x/DataRoot/addons", "cold-1.8.0.13": "/mnt/x/DataRoot/addons_c18"}
'''

LEGACY_TEXT = {
    "tools/enfusion-mcp/enfusion-mcp-wsl.sh": 'ROOT="${EMCP_ROOT:-/mnt/x/PlanRoot/mcp}"   # work root\n',
    "tools/enfusion-mcp/mcp_gate.py": '    root = os.path.realpath(env.get("EMCP_ROOT") or "/mnt/x/PlanRoot/mcp")\n',
    "tools/enfusion-mcp/endpoint_handshake.py": ('NETSTAT = "%s/Windows/System32/NETSTAT.EXE"\n'
                                                 'CSCRIPT = "%s/Windows/System32/cscript.exe"\n'
                                                 'MCP_WIN_TEMP = "/mnt/x/DataRoot/tmp/mcp"   # helper temp\n'
                                                 'IPCONFIG = "%s/Windows/System32/ipconfig.exe"\n' % (SYS, SYS, SYS)),
    "tools/harvester/harvest.py": ('DATA_ROOT = "/mnt/x/DataRoot"\nTEST_ROOT_PREFIXES = ("/dev/shm/", "/mnt/x/")\n'
                                   'FORBIDDEN_DEVICE_PATHS = (("/mnt/z", "the system drive"), ("/", "the WSL disk"))\n'),
    "tools/validator/dc_validate.py": 'DEFAULT_REPORT_DIR = "/mnt/x/DataRoot/reports"\nREPORT_ROOT = "/mnt/x/"\n',
    "src/drivedit/data/build.py": ('FORBIDDEN_ROOTS = ("/mnt/z",)  # the system drive\n'
                                   'DATASET_ROOTS = ("/mnt/x/DataRoot/datasets",)\n'
                                   'CAPTURE_ROOTS = ("/mnt/x/DataRoot/captures",)\n'),
    "data/dc2/reader.py": 'FORBIDDEN_ROOTS = ("/mnt/z/",)\n',
    "tools/probes/t0eng/probe_common.py": 'WRITE_ROOTS = ("/mnt/x/", "/dev/shm/")\nFORBIDDEN_ROOTS = ("/mnt/z/", "/tmp/")\n',
    "src/drivedit/codec/dinov3.py": 'DEFAULT_DINOV3_DIR: Final = Path("/mnt/x/CodeRoot/third_party/dinov3")\n',
    "src/enfusion/datacapture/DC_Session.c": ('\t[Attribute("X:/DataRoot", UIWidgets.EditBox, "Launch guard: -profile and '
                                              '$profile must lie inside this folder", category: "Launch")]\n'),
    "addons/DataCapture/worlds/DataCapture/DC_Arland_Layers/default.layer": '   m_sRequiredProfileRoot "X:/DataRoot"\n',
    "addons/DataCapture/worlds/DataCapture/DC_Probes_Arland_Layers/default.layer":
        '   m_sRequiredProfileRoot "X:/DataRoot"\n',
}


def legacy_launcher(name):
    """A pre-DD-22 launcher: today's, with the placeholder machine's literals instead of dd_local_config.cmd."""
    with open(os.path.join(REPO, "tools", "launch", name), "rb") as fh:
        text = fh.read().decode("ascii")
    text = re.sub(r'if not exist "%~dp0dd_local_config\.cmd" \(\r\n.*?\r\n\)\r\ncall "%~dp0dd_local_config\.cmd"\r\n'
                  r'if errorlevel 1 goto :fail\r\n', "", text, flags=re.S)
    for var, value in lc.bat_values(CFG).items():
        text = text.replace("%" + var + "%", value)
    return text


def write_legacy(root, changes=None):
    files = dict(LEGACY_TEXT)
    files["tools/pin/pinlib.py"] = LEGACY_PINLIB
    files["tools/deploy_addon.py"] = LEGACY_DEPLOY
    for name in mig.LAUNCHERS:
        files["tools/launch/" + name] = legacy_launcher(name)
    with open(os.path.join(REPO, *mig.LEGACY["golden"].split("/")), encoding="ascii") as fh:
        files[mig.LEGACY["golden"]] = lc.render(fh.read(), CFG)
    files.update(changes or {})
    for rel, text in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)


def temp_dir():
    if not (os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)):
        raise unittest.SkipTest("no writable /dev/shm")
    return tempfile.mkdtemp(prefix="dd22_migrate_", dir="/dev/shm")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = mig.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def history_available():
    r = subprocess.run(["git", "--no-optional-locks", "-C", REPO, "cat-file", "-e", REV + "^{commit}"],
                       capture_output=True)
    return r.returncode == 0


class HistoryTest(unittest.TestCase):
    def test_the_last_pre_dd22_revision_is_reproduced(self):
        if not history_available():
            self.skipTest("%s is not in this clone's history (a shallow CI checkout)" % REV)
        legacy = mig.Legacy(repo=REPO, rev=REV)
        values = mig.derive(legacy)
        text = mig.config_text(values, legacy.where())
        cfg = lc.parse(text, "<derived from %s>" % REV)
        with lc.override(cfg):
            vol, _ = lc.data_volume(lc.read_mountinfo())
        if vol is None:
            self.skipTest("the data drive of %s is not a verified data drive here: not the machine whose paths it "
                          "holds" % REV)
        rows = mig.compare(legacy, cfg, text, python=sys.executable)
        diffs = [r for r in rows if not r[3]]
        # only a count: the values are this machine's (run the tool with --dry-run to see them)
        self.assertEqual(len(diffs), 0, "%d of %d comparisons differ" % (len(diffs), len(rows)))
        labels = [r[0] for r in rows]
        self.assertGreaterEqual(len(rows), 120)
        for name in mig.LAUNCHERS:   # the old side of every launcher row is the revision's own file (git show)
            self.assertIn("%s start line" % name, labels)
            self.assertIn("%s wsl.exe lines" % name, labels)
        self.assertEqual(legacy.text("tools/launch/wb_R1.bat"), subprocess.run(
            ["git", "--no-optional-locks", "-C", REPO, "show", REV + ":tools/launch/wb_R1.bat"], capture_output=True,
            check=True).stdout.decode("ascii").replace("\r\n", "\n"))
        for want in ("build.DATASET_ROOTS", "dinov3.DEFAULT_DINOV3_DIR", "mcp_gate EMCP_ROOT default",
                     "pinlib.WRITE_ROOT_DRIVES -> the boundary's data drive"):
            self.assertIn(want, labels)
        self.assertTrue(any(l.startswith("boundary refuses ") for l in labels))


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = temp_dir()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.legacy = os.path.join(self.tmp, "legacy")
        write_legacy(self.legacy)

    def in_world(self):
        """The fake world: the placeholder data drive X: verified, Z: a system volume (boundary_vectors)."""
        w = bv.world(prefix="dd22_migrate_world_")
        world = w.__enter__()
        self.addCleanup(w.__exit__, None, None, None)
        use = world.use(lc, "real_like")
        use.__enter__()
        self.addCleanup(use.__exit__, None, None, None)
        return world

    def test_derives_the_config(self):
        values = mig.derive(mig.Legacy(directory=self.legacy))
        self.assertEqual(values, CFG.values)
        text = mig.config_text(values, "directory x")
        self.assertEqual(lc.parse(text).values, CFG.values)
        self.assertFalse([line for line in text.splitlines() if "=" in line and "<" in line and not line.startswith("#")])

    def test_every_old_value_is_reproduced(self):
        self.in_world()
        rows = mig.compare(mig.Legacy(directory=self.legacy), CFG, FIXTURE_TEXT, python=sys.executable)
        self.assertEqual([(r[0], r[1], r[2]) for r in rows if not r[3]], [])
        labels = [r[0] for r in rows]
        for want in ("wb_R1.bat start line", "client_C18_i01.bat set DD_PIN", "wb_C18.bat wsl.exe lines",
                     "golden_start_lines.txt", "pinlib.LAUNCH_PATHS", "pinlib.WB_LAUNCH", "deploy_addon.DEFAULT_TARGET",
                     "DC_Session.c m_sRequiredProfileRoot default", "endpoint_handshake.MCP_WIN_TEMP",
                     "harvest.DATA_ROOT", "dinov3.DEFAULT_DINOV3_DIR", "boundary refuses /mnt/z/ "
                     "(pinlib.FORBIDDEN_WRITE_PREFIXES)", "boundary writes under /mnt/x/ (pinlib.DEFAULT_WRITE_ROOTS)"):
            self.assertIn(want, labels)
        self.assertGreater(len(rows), 100)

    def test_the_boundary_rows_see_a_refused_old_root(self):
        # outside the fake world the placeholder data drive is mounted nowhere: the old write root is refused now
        rows = mig.boundary_rows(mig.Legacy(directory=self.legacy), CFG)
        row = next(r for r in rows if r[0] == "boundary writes under /mnt/x/ (pinlib.DEFAULT_WRITE_ROOTS)")
        self.assertIn("is not mounted", row[2])
        self.assertNotEqual(row[1], row[2])

    def test_dry_run(self):
        self.in_world()
        rc, out, err = run("--from-dir", self.legacy, "--dry-run", "--python", sys.executable)
        self.assertEqual(rc, 0, err)
        self.assertIn("DATA_ROOT=/mnt/x/DataRoot\n", out)
        self.assertIn("0 different", out)
        self.assertIn("DRY RUN: nothing written", out)

    def test_a_difference_refuses(self):
        self.in_world()
        name = "tools/launch/wb_R1.bat"
        changed = legacy_launcher("wb_R1.bat").replace('\\addons"\r\n', '\\addons_elsewhere"\r\n', 1)
        self.assertNotEqual(changed, legacy_launcher("wb_R1.bat"))
        write_legacy(self.legacy, {name: changed})
        rc, out, err = run("--from-dir", self.legacy, "--write", "--python", sys.executable)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("DIFF wb_R1.bat set GAME_ADDONS", out)
        self.assertIn("2 different", out)   # the set line and the start line that uses it
        self.assertIn("nothing written", err)

    def test_a_post_dd22_revision_is_refused(self):
        write_legacy(self.legacy, {"tools/pin/pinlib.py": "LAUNCH = None\n"})
        rc, _, err = run("--from-dir", self.legacy, "--dry-run")
        self.assertEqual(rc, 2)
        self.assertIn("not a pre-DD-22 revision", err)

    def test_launchers_must_agree_on_the_distro(self):
        other = legacy_launcher("client_R1_i01.bat").replace("wsl.exe -d Placeholder-Distro",
                                                             "wsl.exe -d Placeholder-Other")
        write_legacy(self.legacy, {"tools/launch/client_R1_i01.bat": other})
        rc, _, err = run("--from-dir", self.legacy, "--dry-run")
        self.assertEqual(rc, 2)
        self.assertIn("different WSL distros", err)

    def test_write_goes_through_the_guard(self):
        # Every row is the same in the fake world, but the write itself goes to the config's real DATA_ROOT, whose
        # placeholder drive is mounted nowhere: the write boundary refuses it, nothing is written anywhere.
        self.in_world()
        rc, out, err = run("--from-dir", self.legacy, "--write", "--python", sys.executable)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("0 different", out)
        self.assertIn("refusing to write", err)
        self.assertFalse(os.path.exists(CFG.drive("DATA_ROOT")))


class GetterTest(unittest.TestCase):
    """Codex round 2 row 9: the runtime rows read the code's own getters, so a changed default shows."""

    def test_a_changed_builder_default_differs(self):
        tmp = temp_dir()
        self.addCleanup(shutil.rmtree, tmp, True)
        legacy = os.path.join(tmp, "legacy")
        write_legacy(legacy)
        copy = os.path.join(tmp, "repo")
        for rel in ("src", "tools", "data", "docs"):
            shutil.copytree(os.path.join(REPO, rel), os.path.join(copy, rel),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        rows = dict((r[0], r) for r in mig.runtime_rows(mig.Legacy(directory=legacy), CFG, FIXTURE_TEXT, copy,
                                                        sys.executable))
        self.assertEqual(rows["build.DATASET_ROOTS"][1], rows["build.DATASET_ROOTS"][2])
        build = os.path.join(copy, "src", "drivedit", "data", "build.py")
        with open(build, encoding="utf-8") as fh:
            text = fh.read()
        self.assertEqual(text.count('DATASET_ROOT_REL = "datasets"'), 1)
        with open(build, "w", encoding="utf-8") as fh:
            fh.write(text.replace('DATASET_ROOT_REL = "datasets"', 'DATASET_ROOT_REL = "datasets_moved"'))
        rows = dict((r[0], r) for r in mig.runtime_rows(mig.Legacy(directory=legacy), CFG, FIXTURE_TEXT, copy,
                                                        sys.executable))
        self.assertEqual(rows["build.DATASET_ROOTS"][2], [CFG.get("DATA_ROOT") + "/datasets_moved"])
        self.assertNotEqual(rows["build.DATASET_ROOTS"][1], rows["build.DATASET_ROOTS"][2])


if __name__ == "__main__":
    unittest.main()
