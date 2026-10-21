#!/usr/bin/env python3
"""Self-test for tools/deploy_addon.py (stdlib unittest).

    python3 -I -B tools/test_deploy_addon.py -v

Every case deploys into a fresh temporary target and removes it afterwards. The temporary base directory is
$DEPLOY_ADDON_TEST_TMP if set, else /dev/shm (RAM, so nothing lands on the WSL disk), else the system default.
The repo itself is only read. C:-drive cases always run with --dry-run under a parent that does not exist, so even
a broken guard could not write there.
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
REPO = os.path.dirname(HERE)
TOOL = os.path.join(HERE, "deploy_addon.py")
MANIFEST = os.path.join(REPO, "addons", "DataCapture", "deploy_manifest.txt")
SCRIPT_DST = "Scripts/Game/DataCapture"
WORLD_LAYER = "worlds/DataCapture/DC_Arland_Layers/default.layer"


def tmp_base():
    base = os.environ.get("DEPLOY_ADDON_TEST_TMP")
    if base:
        return base
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


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

    def run_tool(self, *args, target=None):
        cmd = [sys.executable, "-I", "-B", TOOL, "--target", target or self.target] + list(args)
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
        rc, out = self.run_tool("src/enfusion/datacapture/SCR_MLDataCollector.c")
        self.assertEqual(rc, 0, out)
        self.assertEqual(listing(self.target), {SCRIPT_DST + "/SCR_MLDataCollector.c", SCRIPT_DST + "/DC_BuildInfo.c",
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

    def test_refuses_c_drive_targets(self):
        # --dry-run plus a parent that does not exist: nothing could be written even if the guard failed.
        probes = ["/mnt/c/__dd_deploy_selftest_missing__/DataCapture",
                  "C:\\__dd_deploy_selftest_missing__\\DataCapture",
                  "c:/__dd_deploy_selftest_missing__/DataCapture"]
        if os.path.isdir("/mnt/c"):
            link = os.path.join(self.tmp, "c_alias")
            os.symlink("/mnt/c", link)
            probes.append(os.path.join(link, "__dd_deploy_selftest_missing__", "DataCapture"))
        for target in probes:
            rc, out = self.run_tool("--dry-run", target=target)
            self.assertEqual(rc, 2, "%s: %s" % (target, out))
            self.assertIn("C: drive", out)
        self.assertFalse(os.path.exists("/mnt/c/__dd_deploy_selftest_missing__"))

    def test_refuses_target_inside_repo(self):
        rc, out = self.run_tool("--dry-run", target=os.path.join(REPO, "build_addon_test"))
        self.assertEqual(rc, 2, out)
        self.assertIn("overlaps the repo", out)

    # --- state, Workbench files and drift ------------------------------------------------------------------------

    def test_never_touches_workbench_generated_files(self):
        os.makedirs(os.path.join(self.target, SCRIPT_DST))
        generated = {"resourceDatabase.rdb": b"RDB", "UserMaps.desc": b"UserMapDescClass {\n}",
                     "worlds/DataCapture/DC_Arland.ent.meta": b"META ent", SCRIPT_DST + "/SCR_MLDataCollector.c.meta": b"META c",
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
        anchor = os.path.join(self.target, SCRIPT_DST, "SCR_AnchorFrameSelector.c")
        with open(anchor, "a") as f:
            f.write("// local edit\n")
        keep = ["addon addons/DataCapture/addon.gproj", "script src/enfusion/datacapture/SCR_MLDataCollector.c"]
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
