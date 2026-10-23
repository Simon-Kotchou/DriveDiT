#!/usr/bin/env python3
"""Self-test for tools/deploy_addon.py (stdlib unittest).

    python3 -I -B tools/test_deploy_addon.py -v

Every case deploys into a fresh temporary target and removes it afterwards. The temporary base directory is
$DEPLOY_ADDON_TEST_TMP if set (it must be under /mnt/e/ or /dev/shm/, the only places deploy_addon.py accepts as
--allow-root), else $TMPDIR when it is on E: (tools/run_tests.sh sets that), else /dev/shm (RAM, so nothing lands on
the WSL disk image on C:). Without any of them the tests are skipped; they never fall back to /tmp. The repo itself
is only read. C: and WSL-disk targets are checked in-process on the pure target_reason(), so no such path is ever
passed to the tool; other refused targets run with --dry-run under a parent that does not exist.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import deploy_addon  # noqa: E402  (pure checks are tested in-process; the tool has no import side effects)
REPO = os.path.dirname(HERE)
TOOL = os.path.join(HERE, "deploy_addon.py")
MANIFEST = os.path.join(REPO, "addons", "DataCapture", "deploy_manifest.txt")
SCRIPT_DST = "Scripts/Game/DataCapture"
WORLD_LAYER = "worlds/DataCapture/DC_Arland_Layers/default.layer"


def tmp_base():
    base = os.environ.get("DEPLOY_ADDON_TEST_TMP")
    if base:
        return base
    tmpdir = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if tmpdir.startswith("/mnt/e/") and os.path.isdir(tmpdir):  # tools/run_tests.sh points TMPDIR at E:
        return tmpdir
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    raise unittest.SkipTest("no $DEPLOY_ADDON_TEST_TMP and no writable /dev/shm; not using the system temp dir")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def manifest_entries(path=MANIFEST):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                kind, src = line.split()
                out.append((kind, src))
    return out


def expected_dst(kind, src):
    if kind == "addon":
        return src[len("addons/DataCapture/"):]
    if kind == "config":
        for src_root, dst_root in deploy_addon.CONFIG_ROOTS:
            if src.startswith(src_root):
                return dst_root + src[len(src_root):]
    return SCRIPT_DST + "/" + src.rsplit("/", 1)[1]


def listing(root):
    out = set()
    for dirpath, _, files in os.walk(root):
        for name in files:
            out.add(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/"))
    return out


class DeployAddonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dd_deploy_test_", dir=tmp_base())
        self.target = os.path.join(self.tmp, "addons", "DataCapture")
        os.makedirs(os.path.dirname(self.target))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self, *args, target=None, allow_root=True):
        cmd = [sys.executable, "-I", "-B", TOOL, "--target", target or self.target] + list(args)
        if allow_root:
            cmd += ["--allow-root", self.tmp]
        p = subprocess.run(cmd, cwd=self.tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout.decode() + p.stderr.decode()

    def write_manifest(self, lines, name="manifest.txt"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def deployed_set(self):
        return {expected_dst(k, s) for k, s in manifest_entries()} | {SCRIPT_DST + "/DC_BuildInfo.c",
                                                                      ".deploy_state.json"}

    # --- happy path -------------------------------------------------------------------------------------------

    def test_dry_run_writes_nothing(self):
        rc, out = self.run_tool("--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertIn("DRY RUN", out)
        self.assertFalse(os.path.exists(self.target), "dry run created the target")

    def test_deploys_exactly_the_manifest_set(self):
        rc, out = self.run_tool()
        self.assertEqual(rc, 0, out)
        self.assertEqual(listing(self.target), self.deployed_set())
        for kind, src in manifest_entries():
            self.assertEqual(sha(os.path.join(REPO, src)), sha(os.path.join(self.target, expected_dst(kind, src))),
                             "%s was not copied byte for byte" % src)
        for banned in ("SCR_AIDrivingSimulator.c", "SCR_MultiCameraRig.c", "SCR_CaptureTests.c",
                       "SCR_CaptureBuffer.c", "SCR_DepthModule.c"):
            self.assertNotIn(banned, " ".join(listing(self.target)))

    def test_build_info_stamp(self):
        rc, out = self.run_tool()
        self.assertEqual(rc, 0, out)
        with open(os.path.join(self.target, SCRIPT_DST, "DC_BuildInfo.c"), encoding="ascii") as f:
            text = f.read()
        self.assertRegex(text, r"(?m)^class DC_BuildInfo$")
        consts = dict(re.findall(r'(?m)^\tstatic const (?:string|bool|int) ([A-Z_0-9]+) = ("[^"\n]*"|true|false|\d+);$',
                                 text))
        self.assertEqual(set(consts), {"ADDON_ID", "ADDON_GUID", "GIT_SHA", "GIT_BRANCH", "GIT_DIRTY", "DEPLOY_UTC",
                                       "MANIFEST_SHA256", "CONTENT_SHA256", "CONTENT_FILE_COUNT", "DEPLOY_PARTIAL"})
        self.assertEqual(consts["ADDON_ID"], '"DataCapture"')
        self.assertEqual(consts["ADDON_GUID"], '"88BEF4187E0D2DFC"')
        self.assertRegex(consts["GIT_SHA"], r'^"([0-9a-f]{40})?"$')
        self.assertIn(consts["GIT_DIRTY"], ("true", "false"))
        self.assertRegex(consts["DEPLOY_UTC"], r'^"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"$')
        self.assertEqual(consts["MANIFEST_SHA256"], '"%s"' % sha(MANIFEST))
        self.assertEqual(consts["CONTENT_FILE_COUNT"], str(len(manifest_entries())))
        self.assertEqual(consts["DEPLOY_PARTIAL"], "false")
        self.assertNotIn("\r", text)

    def test_redeploy_is_idempotent(self):
        self.assertEqual(self.run_tool()[0], 0)
        before = {p: sha(os.path.join(self.target, p)) for p in listing(self.target)}
        rc, out = self.run_tool()
        self.assertEqual(rc, 0, out)
        self.assertNotRegex(out, r"(?m)^\s+(create|remove)\s")
        for p, h in before.items():
            if not p.endswith(("DC_BuildInfo.c", ".deploy_state.json")):
                self.assertEqual(h, sha(os.path.join(self.target, p)), p)

    # --- refusals ---------------------------------------------------------------------------------------------

    def test_refuses_files_outside_the_manifest(self):
        for request in ("src/enfusion/datacapture/core/SCR_CaptureBuffer.c",
                        "src/enfusion/datacapture/modules/SCR_DepthModule.c",
                        "DriveDiT_DataCapture/DriveDiT_DataCapture.gproj",
                        "src/SCR_CaptureTests.c",
                        "src/enfusion/datacapture/SCR_MultiCameraRig.c",
                        "src/enfusion/datacapture/SCR_AIDrivingSimulator.c",
                        "README.md"):
            rc, out = self.run_tool(request)
            self.assertEqual(rc, 2, "%s: %s" % (request, out))
            self.assertIn("not in the deploy manifest", out)
            self.assertFalse(os.path.exists(self.target), "%s: refusal still wrote the target" % request)

    def test_partial_deploy_of_a_manifest_member(self):
        rc, out = self.run_tool("src/enfusion/datacapture/DC_LaunchGuard.c")
        self.assertEqual(rc, 0, out)
        self.assertEqual(listing(self.target), {SCRIPT_DST + "/DC_LaunchGuard.c", SCRIPT_DST + "/DC_BuildInfo.c",
                                                ".deploy_state.json"})
        with open(os.path.join(self.target, SCRIPT_DST, "DC_BuildInfo.c")) as f:
            self.assertIn("DEPLOY_PARTIAL = true;", f.read())

    def test_refuses_manifest_with_forbidden_entries(self):
        bad = [
            "script src/enfusion/datacapture/core/SCR_CaptureBuffer.c",
            "script src/enfusion/datacapture/modules/SCR_SceneModule.c",
            "addon  DriveDiT_DataCapture/DriveDiT_DataCapture.gproj",
            "script src/SCR_CaptureTests.c",
            "script src/enfusion/datacapture/SCR_SceneEnumerator.c",
            "script src/enfusion/datacapture/SCR_AIDrivingSimulator.c",
            "addon  addons/DataCapture/deploy_manifest.txt",
            "addon  src/enfusion/datacapture/SCR_MLDataCollector.c",
            "script src/enfusion/navigation/extractors/SCR_LocalRoadExtractor.c",
            "script src/enfusion/datacapture/../datacapture/SCR_MLDataCollector.c",
            "copy   src/enfusion/datacapture/SCR_MLDataCollector.c",
        ]
        for line in bad:
            manifest = self.write_manifest(["addon addons/DataCapture/addon.gproj", line])
            rc, out = self.run_tool("--manifest", manifest)
            self.assertEqual(rc, 2, "%s: %s" % (line, out))
            self.assertIn("REFUSED", out)
            self.assertFalse(os.path.exists(self.target), "%s: refusal still wrote the target" % line)

    def test_config_entries(self):
        # DD-7-min: routes under the addon's Configs/ and vehicle cards from tools/ident/cards/ deploy byte for byte
        configs = [(k, s) for k, s in manifest_entries() if k == "config"]
        self.assertIn(("config", "addons/DataCapture/Configs/routes/arland_t0_loop.json"), configs)
        self.assertEqual(deploy_addon.manifest_entry("config", "tools/ident/cards/1.8.0.13/S105.json"),
                         ("config", "Configs/ident/cards/1.8.0.13/S105.json"))
        self.assertEqual(deploy_addon.manifest_entry("config", "addons/DataCapture/Configs/routes/arland_t0_loop.json"),
                         ("config", "Configs/routes/arland_t0_loop.json"))
        for kind, src in (("config", "tools/ident/fit_steer_kappa.py"), ("config", "tools/deploy_state.json"),
                          ("config", "addons/DataCapture/worlds/DataCapture/DC_Arland.ent"),
                          ("config", "addons/DataCapture/Configs/routes/bad name.json"),
                          ("addon", "addons/DataCapture/Configs/routes/arland_t0_loop.json")):
            with self.assertRaises(deploy_addon.Refusal, msg=src):
                deploy_addon.manifest_entry(kind, src)
        rc, out = self.run_tool("addons/DataCapture/Configs/routes/arland_t0_loop.json")
        self.assertEqual(rc, 0, out)
        self.assertEqual(listing(self.target), {"Configs/routes/arland_t0_loop.json", SCRIPT_DST + "/DC_BuildInfo.c",
                                                ".deploy_state.json"})
        self.assertEqual(sha(os.path.join(REPO, "addons/DataCapture/Configs/routes/arland_t0_loop.json")),
                         sha(os.path.join(self.target, "Configs/routes/arland_t0_loop.json")))

    def test_config_entries_never_name_the_sealed_map(self):
        # Everon (terrain eden) is the sealed zero-shot test map: no route or card naming it deploys, in any case
        for src in ("addons/DataCapture/Configs/routes/everon_test_loop.json",
                    "addons/DataCapture/Configs/routes/Eden_loop.json",
                    "tools/ident/cards/1.8.0.13-EVERON/S105.json"):
            with self.assertRaises(deploy_addon.Refusal, msg=src) as caught:
                deploy_addon.manifest_entry("config", src)
            self.assertIn("sealed", str(caught.exception))

    def test_refuses_c_drive_and_wsl_disk_targets(self):
        # Checked in-process on the pure target_reason(): no C: or WSL-disk path is ever passed to the tool's argv.
        probes = ["/mnt/c/__dd_deploy_selftest_missing__/DataCapture",
                  "C:\\__dd_deploy_selftest_missing__\\DataCapture",
                  "c:/__dd_deploy_selftest_missing__/DataCapture"]
        if os.path.isdir("/mnt/c"):
            link = os.path.join(self.tmp, "c_alias")
            os.symlink("/mnt/c", link)
            probes.append(os.path.join(link, "__dd_deploy_selftest_missing__", "DataCapture"))
        for target in probes:
            reason = deploy_addon.target_reason(target, [self.tmp])
            self.assertIsNotNone(reason, target)
            self.assertIn("C: drive", reason)
        # The WSL disk image (/tmp, ~) lives on C:; /mnt/d is the read-only game install.
        for target in ("/tmp/__dd_deploy_selftest_missing__/DataCapture",
                       os.path.join(os.path.expanduser("~"), "__dd_deploy_selftest_missing__", "DataCapture"),
                       "/mnt/d/__dd_deploy_selftest_missing__/DataCapture"):
            self.assertIsNotNone(deploy_addon.target_reason(target, [self.tmp]), target)

    def test_refuses_targets_outside_the_allowed_roots(self):
        # A sibling of the allowed root is not inside it, E: outside DriveDiT/addons is not allowed, and an allowed
        # root itself is not a target.
        probes = [os.path.join(os.path.dirname(self.tmp), "__dd_deploy_selftest_missing__", "DataCapture"),
                  "/mnt/e/__dd_deploy_selftest_missing__/DataCapture",
                  "/mnt/e/DriveDiT/addons",
                  self.tmp]
        for target in probes:
            rc, out = self.run_tool("--dry-run", target=target)
            self.assertEqual(rc, 2, "%s: %s" % (target, out))
            self.assertIn("not inside an allowed deploy root", out)
            self.assertNotIn("DRY RUN", out, target)
        for target in probes[:2]:
            self.assertFalse(os.path.exists(os.path.dirname(target)), target)

    def test_default_root_is_accepted_without_allow_root(self):
        self.assertIsNone(deploy_addon.target_reason("/mnt/e/DriveDiT/addons/DataCapture"))
        self.assertIsNone(deploy_addon.target_reason("/mnt/e/DriveDiT/addons/__dd_deploy_selftest__/DataCapture"))
        self.assertIsNotNone(deploy_addon.target_reason(self.target))  # only with --allow-root

    def test_refuses_bad_allow_roots(self):
        for root in ("/tmp", os.path.expanduser("~"), "/mnt/e", "/mnt/d", "/mnt/e/__dd_deploy_selftest_missing__"):
            reason = deploy_addon.allow_root_reason(root)
            self.assertIsNotNone(reason, root)
            self.assertIn("--allow-root", reason)
        self.assertIsNone(deploy_addon.allow_root_reason(self.tmp))
        rc, out = self.run_tool("--dry-run", "--allow-root", "/mnt/e", target="/mnt/e/__dd_deploy_selftest__/x",
                                allow_root=False)
        self.assertEqual(rc, 2, out)
        self.assertIn("--allow-root /mnt/e resolves to /mnt/e", out)

    def test_refuses_target_inside_repo(self):
        rc, out = self.run_tool("--dry-run", target=os.path.join(REPO, "build_addon_test"))
        self.assertEqual(rc, 2, out)
        self.assertIn("overlaps the repo", out)

    def test_refuses_case_variants_of_manifest_paths(self):
        # E: is case-insensitive: a case variant opens the real file, so it must neither bypass the denylist nor
        # deploy under a second spelling.
        for line in ("script src/enfusion/datacapture/scr_binaryserializer.c",
                     "script src/enfusion/datacapture/SCR_DRIVINGSIMDEBUGUI.c",
                     "script src/enfusion/datacapture/scr_mldatacollector.c",
                     "script SRC/enfusion/datacapture/SCR_MLDataCollector.c",
                     "script src/enfusion/DataCapture/CORE/SCR_CaptureBuffer.c",
                     "addon  addons/DataCapture/ADDON.gproj"):
            manifest = self.write_manifest(["addon addons/DataCapture/addon.gproj", line])
            rc, out = self.run_tool("--manifest", manifest)
            self.assertEqual(rc, 2, "%s: %s" % (line, out))
            self.assertIn("REFUSED", out)
            self.assertFalse(os.path.exists(self.target), "%s: refusal still wrote the target" % line)

    def test_refuses_symlinked_directories_inside_the_target(self):
        self.assertEqual(self.run_tool()[0], 0)
        outside = os.path.join(self.tmp, "outside")
        os.makedirs(outside)
        # A Scripts folder that points elsewhere (e.g. a junction into an old C: addon) must not receive the scripts.
        shutil.rmtree(os.path.join(self.target, "Scripts"))
        os.symlink(outside, os.path.join(self.target, "Scripts"))
        rc, out = self.run_tool()
        self.assertEqual(rc, 2, out)
        self.assertIn("symlink or junction", out)
        self.assertEqual(os.listdir(outside), [], "a deploy wrote through a symlinked directory")
        os.remove(os.path.join(self.target, "Scripts"))

        # A redirected backup folder must not receive --force backups.
        self.assertEqual(self.run_tool()[0], 0)
        layer = os.path.join(self.target, *WORLD_LAYER.split("/"))
        with open(layer, "ab") as f:
            f.write(b"// saved by Workbench\n")
        edited = sha(layer)
        os.symlink(outside, os.path.join(self.target, ".deploy_backup"))
        rc, out = self.run_tool("--force")
        self.assertEqual(rc, 2, out)
        self.assertIn("symlink or junction", out)
        self.assertEqual(os.listdir(outside), [])
        self.assertEqual(sha(layer), edited, "a refused deploy still replaced the layer")
        os.remove(os.path.join(self.target, ".deploy_backup"))

        # A symlinked state file is refused before it is read or replaced.
        state = os.path.join(self.target, ".deploy_state.json")
        os.replace(state, os.path.join(outside, "state.json"))
        os.symlink(os.path.join(outside, "state.json"), state)
        rc, out = self.run_tool("--force")
        self.assertEqual(rc, 2, out)
        self.assertIn("symlink or junction", out)
        self.assertEqual(os.listdir(outside), ["state.json"])

    # --- state, Workbench files and drift ------------------------------------------------------------------------

    def test_never_touches_workbench_generated_files(self):
        os.makedirs(os.path.join(self.target, SCRIPT_DST))
        generated = {"resourceDatabase.rdb": b"RDB", "UserMaps.desc": b"UserMapDescClass {\n}",
                     "worlds/DataCapture/DC_Arland.ent.meta": b"META ent",
                     SCRIPT_DST + "/SCR_MLDataCollector.c.meta": b"META c",
                     SCRIPT_DST + "/SCR_AnchorFrameSelector.c.meta": b"META anchor"}
        for rel, data in generated.items():
            os.makedirs(os.path.dirname(os.path.join(self.target, rel)), exist_ok=True)
            with open(os.path.join(self.target, rel), "wb") as f:
                f.write(data)
        self.assertEqual(self.run_tool()[0], 0)
        smaller = ["addon addons/DataCapture/addon.gproj", "script src/enfusion/datacapture/SCR_MLDataCollector.c"]
        rc, out = self.run_tool("--manifest", self.write_manifest(smaller))
        self.assertEqual(rc, 0, out)
        for rel, data in generated.items():
            with open(os.path.join(self.target, rel), "rb") as f:
                self.assertEqual(f.read(), data, "%s was modified" % rel)

    def test_removes_only_files_it_deployed(self):
        self.assertEqual(self.run_tool()[0], 0)
        foreign = os.path.join(self.target, SCRIPT_DST, "Unrelated_Handmade.c")
        with open(foreign, "w") as f:
            f.write("class Unrelated_Handmade {}\n")
        keep = ["addon addons/DataCapture/addon.gproj", "addon addons/DataCapture/worlds/DataCapture/DC_Arland.ent",
                "addon addons/DataCapture/worlds/DataCapture/DC_Arland_Layers/default.layer",
                "script src/enfusion/datacapture/SCR_MLDataCollector.c",
                "script src/enfusion/datacapture/SCR_MLScreenshotCapture.c"]
        rc, out = self.run_tool("--manifest", self.write_manifest(keep))
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(os.path.join(self.target, SCRIPT_DST, "SCR_AnchorFrameSelector.c")))
        self.assertTrue(os.path.exists(foreign), "a file the tool never deployed was deleted")

    def test_refuses_to_overwrite_a_changed_file_without_force(self):
        self.assertEqual(self.run_tool()[0], 0)
        layer = os.path.join(self.target, *WORLD_LAYER.split("/"))
        with open(layer, "ab") as f:
            f.write(b"// saved by Workbench\n")
        edited = sha(layer)
        rc, out = self.run_tool()
        self.assertEqual(rc, 2, out)
        self.assertIn("changed since the last deploy", out)
        self.assertEqual(sha(layer), edited, "refused deploy still overwrote the layer")
        rc, out = self.run_tool("--force")
        self.assertEqual(rc, 0, out)
        self.assertEqual(sha(layer), sha(os.path.join(REPO, "addons", "DataCapture", *WORLD_LAYER.split("/"))))
        backups = [p for p in listing(self.target) if p.startswith(".deploy_backup/")]
        self.assertEqual(len(backups), 1, backups)
        self.assertTrue(backups[0].endswith(WORLD_LAYER + ".bak"))
        self.assertEqual(sha(os.path.join(self.target, backups[0])), edited)

    def test_refuses_to_overwrite_an_untracked_file_without_force(self):
        os.makedirs(self.target)
        with open(os.path.join(self.target, "addon.gproj"), "w") as f:
            f.write('GameProject {\n ID "Other"\n}\n')
        rc, out = self.run_tool()
        self.assertEqual(rc, 2, out)
        self.assertIn("was not deployed by this tool", out)
        self.assertEqual(os.listdir(self.target), ["addon.gproj"])

    def test_refuses_to_delete_a_changed_file_without_force(self):
        self.assertEqual(self.run_tool()[0], 0)
        anchor = os.path.join(self.target, SCRIPT_DST, "DC_RoadDump.c")
        with open(anchor, "a") as f:
            f.write("// local edit\n")
        keep = ["addon addons/DataCapture/addon.gproj", "script src/enfusion/datacapture/DC_LaunchGuard.c"]
        rc, out = self.run_tool("--manifest", self.write_manifest(keep))
        self.assertEqual(rc, 2, out)
        self.assertTrue(os.path.exists(anchor))

    def test_emcp_handlers(self):
        handlers = os.path.join(self.tmp, "handlers")
        os.makedirs(handlers)
        for name in ("EMCP_WB_Ping.c", "EMCP_WB_Play.c", "Other.c", "EMCP_WB_Notes.txt"):
            with open(os.path.join(handlers, name), "w") as f:
                f.write("// %s\n" % name)
        rc, out = self.run_tool("--emcp-handlers", handlers)
        self.assertEqual(rc, 0, out)
        emcp = {p for p in listing(self.target) if p.startswith("Scripts/WorkbenchGame/")}
        self.assertEqual(emcp, {"Scripts/WorkbenchGame/EnfusionMCP/EMCP_WB_Ping.c",
                                "Scripts/WorkbenchGame/EnfusionMCP/EMCP_WB_Play.c"})
        rc, out = self.run_tool()  # without the flag the handlers stay
        self.assertEqual(rc, 0, out)
        self.assertEqual({p for p in listing(self.target) if p.startswith("Scripts/WorkbenchGame/")}, emcp)
        os.remove(os.path.join(handlers, "EMCP_WB_Play.c"))
        rc, out = self.run_tool("--emcp-handlers", handlers)  # with the flag the handler set is synced
        self.assertEqual(rc, 0, out)
        self.assertEqual({p for p in listing(self.target) if p.startswith("Scripts/WorkbenchGame/")},
                         {"Scripts/WorkbenchGame/EnfusionMCP/EMCP_WB_Ping.c"})

    def test_refuses_empty_emcp_dir(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        rc, out = self.run_tool("--emcp-handlers", empty)
        self.assertEqual(rc, 2, out)
        self.assertFalse(os.path.exists(self.target))


if __name__ == "__main__":
    unittest.main()
