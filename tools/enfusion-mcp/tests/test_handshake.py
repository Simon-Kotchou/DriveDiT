"""Tests for the per-pin MCP index, the index diff, the pin gate and the endpoint handshake (ticket DD-14a, CC-05).
Stdlib unittest only; no Workbench, no MCP server, no Windows process: the Windows probes are stubs, the MCP server
is a stub child that echoes. The fixture dumps are synthetic (tests/fixtures/dump_18, dump_19).

    python3 -I -B -m unittest discover -s tools/enfusion-mcp/tests -v

Temp trees go under $DD_PIN_TEST_TMP, else /dev/shm, and are deleted afterwards. Every test reads the synthetic local
config of tools/pin/tests (DD-22; data drive /mnt/x, system drive /mnt/z), never this machine's. The BK source check
runs only when $DD_TEST_BK_SRC names a pinned BK checkout's src/tools.
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
EMCP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EMCP))
FIX = os.path.join(HERE, "fixtures")
PIN_TESTS = os.path.join(REPO, "tools", "pin", "tests")
BK_SRC = os.environ.get("DD_TEST_BK_SRC", "")   # opt-in: <MCP root>/build/enfusion-mcp-BK/src/tools


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
pinlib, make_pin, preflight = base.pinlib, base.make_pin, base.preflight
pin_sidecar = base._load("pin_sidecar")
api_index = _load("dd_emcp_api_index", os.path.join(EMCP, "api_index.py"))
api_index_diff = _load("dd_emcp_api_index_diff", os.path.join(EMCP, "api_index_diff.py"))
eh = _load("dd_emcp_endpoint_handshake", os.path.join(EMCP, "endpoint_handshake.py"))
bk = _load("dd_emcp_bk_tools", os.path.join(EMCP, "bk_tools.py"))
gate_mod = _load("dd_emcp_mcp_gate", os.path.join(EMCP, "mcp_gate.py"))
WB_REL = "Workbench/ArmaReforgerWorkbenchSteamDiag.exe"
REC_UTC = "2026-11-06T05:24:24Z"
REC_UNIX = pinlib.iso_utc_to_unix(REC_UTC)
DUMP = {}


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a, **kw)
    return rc, out.getvalue(), err.getvalue()


def setUpModule():
    """One synthetic dump repo for the module: tag v1.8.0.13 is dump_18, v1.9.0.37 is dump_19."""
    tmp = tempfile.mkdtemp(prefix="dd14a-dump-", dir=base._tmp_base())
    repo = os.path.join(tmp, "sd")
    os.makedirs(repo)
    email = "noreply" + "@" + "localhost"

    def git(*args):
        return subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=" + email,
                               "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"] + list(args), check=True,
                              capture_output=True).stdout.decode().strip()
    git("init", "-q")
    commits = {}
    for dump, tag in (("dump_18", "v1.8.0.13"), ("dump_19", "v1.9.0.37")):
        shutil.rmtree(os.path.join(repo, "scripts"), ignore_errors=True)
        shutil.copytree(os.path.join(FIX, dump, "scripts"), os.path.join(repo, "scripts"))
        git("add", "-A")
        git("commit", "-q", "-m", tag)
        git("tag", tag)
        commits[tag] = git("rev-parse", "HEAD")
    DUMP.update(tmp=tmp, repo=repo, commits=commits)


def tearDownModule():
    shutil.rmtree(DUMP.get("tmp", ""), ignore_errors=True)


def write_sources(path, mapping):
    with open(path, "w") as fh:
        json.dump({"schema": "dd-mcp-index-sources/1", "pins": mapping}, fh)
    return path


class PinCase(base.TmpCase):
    """A fake Steam layout; pins 1.8.0.13 and 1.9.0.37 (both dd-pin/1, from the same fake installs) with build_key and
    engine sidecars; ACTIVE 1.8.0.13; the module's synthetic dump repo and a source map; a fake MCP root."""

    def setUp(self):
        super().setUp()
        self.fs.set_auto_update("1")
        self.pin_dir = self.fs.out_dir
        self.assertEqual(base.run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)[0], 0)
        rc, _, err = base.run_main(make_pin.main, ["--pin-id", "1.9.0.37", "--out", self.pin_file("1.9.0.37"),
                                                   "--game-acf", self.fs.game_acf, "--tools-acf", self.fs.tools_acf],
                                   self.roots)
        self.assertEqual(rc, 0, err)
        for pid, ver in (("1.8.0.13", "1.8.0.13"), ("1.9.0.37", "1.9.0.37")):
            rc, _, err = base.run_main(pin_sidecar.main, ["build-key", pid, "--pin-dir", self.pin_dir, "--from-live",
                                                          "--apply"], self.roots)
            self.assertEqual(rc, 0, err)
            rc, _, err = base.run_main(pin_sidecar.main, ["engine", pid, "--pin-dir", self.pin_dir, "--run-header",
                                                          self.header(pid, ver), "--apply"], self.roots)
            self.assertEqual(rc, 0, err)
        self.src = preflight.load_pin_by_id(self.pin_dir, "1.8.0.13")
        self.ident = preflight.pin_identity(self.src)
        with open(os.path.join(self.pin_dir, "ACTIVE"), "w") as fh:
            fh.write("1.8.0.13\n")
        self.sources = write_sources(os.path.join(self.tmp, "sources.json"), {
            "1.8.0.13": {"tag": "v1.8.0.13", "commit": DUMP["commits"]["v1.8.0.13"]},
            "1.9.0.37": {"tag": "v1.9.0.37", "commit": DUMP["commits"]["v1.9.0.37"]}})
        self.root = os.path.join(self.tmp, "mcp")
        os.makedirs(os.path.join(self.root, "pins"))
        self.project = os.path.join(self.tmp, "addons")
        os.makedirs(os.path.join(self.project, "DataCapture"))
        self.cold_project = os.path.join(self.tmp, "addons_c18")
        os.makedirs(os.path.join(self.cold_project, "DataCapture"))
        self.profile = os.path.join(self.tmp, "wb_profile")
        self.gproj = os.path.join(self.project, "DataCapture", "addon.gproj")
        for name, value in (("DEPLOY_ROOTS", {"stable": self.project, "cold-1.8.0.13": self.cold_project}),
                            ("WB_LAUNCH", {"stable": {"profile": self.profile, "gproj": self.gproj,
                                                      "launcher": "wb_R1.bat"}})):
            base.override(self, pinlib, name, value)

    def pin_file(self, pin_id):
        return os.path.join(self.fs.out_dir, "pin_%s.json" % pin_id)

    def header(self, pin_id, version):
        path = os.path.join(self.tmp, "run_header_%s.txt" % pin_id)
        with open(path, "w") as fh:
            fh.write("launch.mode=cli\nlaunch.id=20261106T052424Z-wb-addc\nlaunch.pin_id=%s\nlaunch.pin_sha256=%s\n"
                     "engine.build_version=%s\nengine.fixed_time_slice_s=0.033333\n"
                     % (pin_id, pinlib.sha256_file(self.pin_file(pin_id)), version))
        return path

    def build(self, pin_id="1.8.0.13", sources=None, extra=(), root=None, expect=0):
        out = os.path.join(root or self.root, "pins", pin_id, "data")
        rc, o, err = quiet(api_index.main, ["build", "--dump-git", DUMP["repo"], "--pin-id", pin_id, "--pin-dir",
                                            self.pin_dir, "--sources", sources or self.sources, "--out", out]
                           + list(extra or ["--apply"]), write_roots=self.roots)
        self.assertEqual(rc, expect, err)
        return out if expect == 0 else err

    def env(self, **extra):
        e = {"EMCP_ROOT": self.root, "EMCP_PIN_DIR": self.pin_dir, "EMCP_PIN": "1.8.0.13",
             "EMCP_INDEX_SOURCES": self.sources, "ENFUSION_WORKBENCH_HOST": "127.0.0.1"}
        e.update(extra)
        return e


# --------------------------------------------------------------------------- the index


class IndexTests(PinCase):
    def test_build_and_stamp(self):
        out = self.build()
        with open(os.path.join(out, "api", "INDEX_PIN.json")) as fh:
            stamp = json.load(fh)
        self.assertEqual((stamp["pin_id"], stamp["pin_sha256"], stamp["build_key"]),
                         ("1.8.0.13", self.src["sha256"], self.ident["build_key"]))
        self.assertEqual((stamp["source"]["tag"], stamp["source"]["commit"]),
                         ("v1.8.0.13", DUMP["commits"]["v1.8.0.13"]))
        _, classes = api_index_diff.load_index(out)
        car = classes["ZxCarSim"]
        self.assertEqual((car["parents"], car["source"]), (["ZxSimBase"], "arma"))
        self.assertEqual([m["name"] for m in car["methods"]], ["GetSteer", "SetSteer", "SetBrakes", "GetGearIndex"])
        self.assertEqual(car["methods"][0]["description"], "Reads the steer input of the synthetic car")
        self.assertEqual(car["methods"][2]["params"], [{"name": "level", "type": "float", "defaultValue": ""},
                                                       {"name": "park", "type": "bool", "defaultValue": ""}])
        cam = classes["ZxCamera"]
        self.assertEqual(cam["source"], "enfusion")
        self.assertEqual(sorted(m["name"] for m in cam["methods"]), ["OnTick", "ZxCamera"])
        self.assertEqual([p["name"] for p in cam["properties"]], ["Fov"])
        self.assertEqual([(p["name"], p["defaultValue"]) for p in cam["protectedProperties"]], [("m_fRate", "1")])
        self.assertEqual([m["name"] for m in classes["ZxHarness"]["staticMethods"]],
                         ["CountSuites", "Start", "CurrentSuite"])
        enum = classes["EZxAssist"]["enums"][0]
        self.assertEqual([(v["name"], v["value"]) for v in enum["values"]], [("OFF", ""), ("SOME", ""), ("ALL", "2")])
        self.assertEqual(enum["values"][0]["description"], "No help at all")
        # Codex round 1: string/vector defaults are kept; an initializer with parentheses is a member variable
        tint = classes["ZxTint"]
        self.assertEqual(tint["methods"][0]["signature"],
                         'proto external ZxEntity Spawn (string what, vector at, vector turn = "0 0 0")')
        self.assertEqual([(p["name"], p["defaultValue"]) for p in tint["properties"]],
                         [("Dark", "new ZxTint(0.0, 0.0, 0.0, 1.0)")])

    def test_multiline_declaration_and_new_class(self):
        out = self.build("1.9.0.37")
        _, classes = api_index_diff.load_index(out)
        m = [x for x in classes["ZxCarSim"]["methods"] if x["name"] == "UpdateTyres"][0]
        self.assertEqual(m["signature"], "proto external bool UpdateTyres (notnull map<string, float> values, "
                                         "int wheel = -1)")
        self.assertIn("ZxWheelFx", classes)

    def test_built_once_and_dry_run(self):
        out = os.path.join(self.root, "pins", "1.8.0.13", "data")
        self.build(extra=["--dry-run"])
        self.assertFalse(os.path.exists(out))
        self.build()
        self.assertIn("already exists", self.build(expect=2))
        self.assertEqual([n for n in os.listdir(os.path.dirname(out)) if n != "data"], [])  # no temp dir left

    def test_out_must_be_the_pins_dir(self):
        rc, _, err = quiet(api_index.main, ["build", "--dump-git", DUMP["repo"], "--pin-id", "1.8.0.13", "--pin-dir",
                                            self.pin_dir, "--sources", self.sources, "--out",
                                            os.path.join(self.root, "pins", "1.9.0.37", "data"), "--apply"],
                           write_roots=self.roots)
        self.assertEqual(rc, 2, err)

    def test_provenance_refusals(self):
        # Codex round 1: the 1.8 dump stamped as pin 1.9.0.37
        self.assertIn("--tag v1.8.0.13, but pin 1.9.0.37 is built from v1.9.0.37",
                      self.build("1.9.0.37", extra=["--tag", "v1.8.0.13", "--dry-run"], expect=2))
        wrong_map = write_sources(os.path.join(self.tmp, "wrong.json"), {
            "1.9.0.37": {"tag": "v1.8.0.13", "commit": DUMP["commits"]["v1.8.0.13"]}})
        self.assertIn("the pin's engine reports 1.9.0.37",
                      self.build("1.9.0.37", sources=wrong_map, extra=["--dry-run"], expect=2))
        moved = write_sources(os.path.join(self.tmp, "moved.json"), {
            "1.8.0.13": {"tag": "v1.8.0.13", "commit": DUMP["commits"]["v1.9.0.37"]}})
        self.assertIn("a moved tag or another repo", self.build(sources=moved, extra=["--dry-run"], expect=2))
        unmapped = write_sources(os.path.join(self.tmp, "unmapped.json"), {})
        self.assertIn("has no entry in the index source map", self.build(sources=unmapped, extra=["--dry-run"],
                                                                         expect=2))
        os.unlink(os.path.join(self.pin_dir, "pin_1.8.0.13.engine.json"))
        self.assertIn("has no engine_version_expected", self.build(extra=["--dry-run"], expect=2))
        self.assertFalse(os.path.exists(os.path.join(self.root, "pins", "1.9.0.37")))

    def test_reads_a_git_tag_without_checkout(self):
        commit, files = api_index.read_tree_git(DUMP["repo"], "v1.8.0.13")
        self.assertEqual(commit, DUMP["commits"]["v1.8.0.13"])
        self.assertEqual(len(files), 6)
        self.assertNotIn("Game/generated/Rigs/ZxWheelFx.c", dict(files))
        _, files19 = api_index.read_tree_git(DUMP["repo"], "v1.9.0.37")
        self.assertIn("Game/generated/Rigs/ZxWheelFx.c", dict(files19))


class IndexDiffTests(PinCase):
    def setUp(self):
        super().setUp()
        self.old = self.build("1.8.0.13")
        self.new = self.build("1.9.0.37")

    def diff(self, expect, *extra):
        return quiet(api_index_diff.main, ["--old", self.old, "--new", self.new, "--expect", expect] + list(extra),
                     write_roots=self.roots)

    def test_fixture_expectations_hold(self):
        rc, out, err = self.diff(os.path.join(FIX, "expect_fixture.txt"))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: OK (7 of 7 assertions hold)", out)
        self.assertIn("planned_fixture.txt: 2 new-in-new, 2 unchanged", out)

    def test_literal_default_changes_are_in_the_delta(self):
        _, old = api_index_diff.load_index(self.old)
        _, new = api_index_diff.load_index(self.new)
        rows = {(r[0], r[1], r[2]) for r in api_index_diff.delta_rows(old, new)}
        self.assertIn(("sig_changed", "ZxTint", "Spawn"), rows)
        self.assertIn(("sig_changed", "ZxTint", "Dark"), rows)

    def test_whitespace_inside_literals_is_a_change(self):
        # Codex round 2 N7: "a b" -> "a  b" must show in the diff (whitespace is collapsed outside literals only)
        old = {"C": {"name": "C", "parents": [], "methods": [
            {"name": "F", "signature": 'void F (string s = "a b")'}], "properties": [
            {"name": "P", "type": "string", "defaultValue": '"a b"'}]}}
        new = {"C": {"name": "C", "parents": [], "methods": [
            {"name": "F", "signature": 'void  F (string s = "a  b")'}], "properties": [
            {"name": "P", "type": "string", "defaultValue": '"a  b"'}]}}
        rows = {(r[0], r[2]) for r in api_index_diff.delta_rows(old, new)}
        self.assertEqual(rows, {("sig_changed", "F"), ("sig_changed", "P")})
        same = {"C": {"name": "C", "parents": [], "methods": [{"name": "F", "signature": 'void   F (string s = "a b")'}]}}
        base_ = {"C": {"name": "C", "parents": [], "methods": [{"name": "F", "signature": 'void F (string s = "a b")'}]}}
        self.assertEqual(api_index_diff.delta_rows(base_, same), [])

    def test_wrong_expectations_fail(self):
        bad = os.path.join(self.tmp, "bad.txt")
        with open(os.path.join(self.tmp, "planned_bad.txt"), "w") as fh:
            fh.write("ZxCarSim SetSteer\nZxHarness CountSuites\n")
        with open(bad, "w") as fh:
            fh.write("class-added ZxCamera\nunchanged ZxHarness Start\nmember-removed ZxCarSim GetSteer\nplanned %s\n"
                     "nonsense x\nplanned %s\n" % (os.path.join(FIX, "planned_fixture.txt"),
                                                   os.path.join(self.tmp, "planned_bad.txt")))
        rc, out, _ = self.diff(bad)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL line 2 unchanged ZxHarness Start: Start SIG_CHANGED", out)
        self.assertIn("ZxHarness.CountSuites REMOVED", out)
        self.assertIn("RESULT: MISMATCH (1 of 6 assertions hold)", out)

    def test_tsv_goes_through_the_write_guard(self):
        expect = os.path.join(FIX, "expect_fixture.txt")
        out_ok = os.path.join(self.tmp, "delta.tsv")
        rc, o, err = self.diff(expect, "--tsv", out_ok)
        self.assertEqual(rc, 0, o + err)
        self.assertTrue(os.path.isfile(out_ok))
        outside = tempfile.mkdtemp(prefix="dd14a-outside-", dir=base._tmp_base())
        self.addCleanup(shutil.rmtree, outside, True)
        link = os.path.join(self.tmp, "link")
        os.symlink(outside, link)
        for tsv in ("/mnt/z/Synthetic Steam (x86)/__dd14a_never_written__/d.tsv", os.path.join(link, "d.tsv")):
            self.assertEqual(self.diff(expect, "--tsv", tsv)[0], 2, tsv)
        cwd = os.getcwd()
        os.chdir(outside)
        try:
            self.assertEqual(self.diff(expect, "--tsv", "d.tsv")[0], 2)
        finally:
            os.chdir(cwd)
        self.assertEqual(os.listdir(outside), [])

    def test_tampered_index_refused(self):
        with open(os.path.join(self.new, "api", "arma-classes.json"), "a") as fh:
            fh.write(" ")
        rc, _, err = self.diff(os.path.join(FIX, "expect_fixture.txt"))
        self.assertEqual(rc, 2, err)
        self.assertIn("does not match the sha256", err)


# --------------------------------------------------------------------------- the pin gate (wrapper config)


ROUTE = ("Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
         "eth0\t00000000\t010200C0\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"     # default via 192.0.2.1
         "eth0\t000200C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n")
RESOLV = ("# This file was automatically generated by WSL. To stop automatic generation of this file, add the "
          "following entry to /etc/wsl.conf:\n# [network]\n# generateResolvConf = false\nnameserver 192.0.2.1\n")
LAN_ROUTE = ROUTE.replace("010200C0", "016433C6")                        # default via 198.51.100.1
LAN_RESOLV = "nameserver 198.51.100.1\n"
TUN0 = "tun0\t00000000\t00000000\t0001\t0\t0\t0\t00000000\t0\t0\t0\n"   # an up, device-only default route
# Synthetic ipconfig.exe output (RFC 5737 documentation addresses, no real host's adapters): English, the WSL switch last, with a LAN adapter first.
IPCONFIG_EN = ("\r\nWindows IP Configuration\r\n\r\n\r\nEthernet adapter Ethernet:\r\n\r\n"
               "   Connection-specific DNS Suffix  . : lan\r\n   IPv4 Address. . . . . . . . . . . : 198.51.100.20\r\n"
               "   Subnet Mask . . . . . . . . . . . : 255.255.255.0\r\n   Default Gateway . . . . . . . . . : "
               "198.51.100.1\r\n\r\nEthernet adapter vEthernet (Default Switch):\r\n\r\n"
               "   IPv4 Address. . . . . . . . . . . : 203.0.113.1\r\n\r\nEthernet adapter vEthernet (WSL):\r\n\r\n"
               "   Connection-specific DNS Suffix  . : \r\n   IPv4 Address. . . . . . . . . . . : 192.0.2.1\r\n"
               "   Subnet Mask . . . . . . . . . . . : 255.255.255.0\r\n   Default Gateway . . . . . . . . . : \r\n")
# The same, with the WSL switch first and another label style ('IPv4-Adresse ... (Bevorzugt)').
IPCONFIG_ORDER = ("Windows-IP-Konfiguration\r\n\r\nEthernet-Adapter vEthernet (WSL (Hyper-V firewall)):\r\n\r\n"
                  "   IPv4-Adresse  . . . . . . . . . . : 192.0.2.1(Bevorzugt)\r\n\r\nEthernet-Adapter Ethernet:"
                  "\r\n\r\n   IPv4-Adresse  . . . . . . . . . . : 198.51.100.20\r\n")
IPCONFIG_NO_WSL = IPCONFIG_EN.replace("vEthernet (WSL)", "vEthernet (Other)")


class GateConfigTests(PinCase):
    def test_accepts_the_pins_own_index(self):
        out = self.build()
        cfg = gate_mod.resolve_config(self.env())
        self.assertEqual((cfg["data_dir"], cfg["project"], cfg["resolved_by"]), (out, self.project, "EMCP_PIN"))
        env = gate_mod.server_env({"X": "1"}, cfg)
        self.assertEqual((env["ENFUSION_MCP_DATA_DIR"], env["ENFUSION_PROJECT_PATH"], env["ENFUSION_GAME_PATH"]),
                         (out, self.project, self.fs.game_root))
        self.assertEqual((env["ENFUSION_WORKBENCH_HOST"], env["ENFUSION_WORKBENCH_PORT"]), ("127.0.0.1", "5775"))

    def test_active_pointer(self):
        self.build()
        env = self.env()
        del env["EMCP_PIN"]
        self.assertEqual(gate_mod.resolve_config(env)["resolved_by"], "ACTIVE")

    def assertRefused(self, env, text):
        with self.assertRaises(gate_mod.GateRefusal) as cm:
            gate_mod.resolve_config(env)
        self.assertIn(text, str(cm.exception))

    def test_no_index_no_fallback(self):
        self.assertRefused(self.env(), "has no API index")

    def test_index_of_another_pin_refused(self):
        other = self.build()
        shutil.copytree(other, os.path.join(self.root, "pins", "1.9.0.37", "data"))
        self.assertRefused(self.env(EMCP_PIN="1.9.0.37"), "but ACTIVE is 1.8.0.13")
        with open(os.path.join(self.pin_dir, "ACTIVE"), "w") as fh:
            fh.write("1.9.0.37\n")
        self.assertRefused(self.env(EMCP_PIN="1.9.0.37"), "is for pin 1.8.0.13")

    def test_stamp_source_must_match_the_map(self):
        out = self.build()
        stamp_path = os.path.join(out, "api", "INDEX_PIN.json")
        with open(stamp_path) as fh:
            stamp = json.load(fh)
        stamp["source"]["commit"] = DUMP["commits"]["v1.9.0.37"]
        with open(stamp_path, "w") as fh:
            json.dump(stamp, fh)
        self.assertRefused(self.env(), "but pin 1.8.0.13 needs v1.8.0.13")

    def test_data_dir_and_project_overrides_must_be_the_pins(self):
        self.build()
        self.assertRefused(self.env(ENFUSION_MCP_DATA_DIR=os.path.join(self.root, "data")), "is not pin 1.8.0.13's index")
        self.assertRefused(self.env(ENFUSION_PROJECT_PATH=self.tmp), "is not pin 1.8.0.13's project root")

    def test_project_deployed_for_another_pin_refused(self):
        self.build()
        with open(os.path.join(self.project, "DataCapture", ".deploy_state.json"), "w") as fh:
            json.dump({"target_pin": "1.9.0.37"}, fh)
        self.assertRefused(self.env(), "was deployed for pin 1.9.0.37")

    def test_tampered_index_refused(self):
        out = self.build()
        with open(os.path.join(out, "api", "groups.json"), "a") as fh:
            fh.write(" ")
        self.assertRefused(self.env(), "does not match its INDEX_PIN.json hash")

    def test_endpoint_host_binding(self):
        self.build()
        nat = dict(EMCP_TEST_ROUTE=ROUTE, EMCP_TEST_RESOLV=RESOLV, EMCP_TEST_IPCONFIG=IPCONFIG_EN)
        self.assertRefused(self.env(ENFUSION_WORKBENCH_HOST="203.0.113.3", **nat),
                           "neither loopback nor the local Windows host")
        cfg = gate_mod.resolve_config(self.env(ENFUSION_WORKBENCH_HOST="192.0.2.1", **nat))
        self.assertEqual((cfg["host"], cfg["gateway"]), ("192.0.2.1", "192.0.2.1"))
        # Codex round 2: a LAN gateway is never taken for the Windows host
        self.assertRefused(self.env(ENFUSION_WORKBENCH_HOST="198.51.100.1", EMCP_TEST_ROUTE=LAN_ROUTE,
                                    EMCP_TEST_RESOLV=LAN_RESOLV, EMCP_TEST_IPCONFIG=IPCONFIG_EN), "not generated by WSL")
        # no proof that the gateway is the Windows host: loopback only
        self.assertRefused(self.env(ENFUSION_WORKBENCH_HOST="192.0.2.1", EMCP_TEST_ROUTE=ROUTE,
                                    EMCP_TEST_RESOLV=RESOLV, EMCP_TEST_IPCONFIG=IPCONFIG_NO_WSL), "no Windows")
        self.assertEqual(gate_mod.resolve_config(self.env(ENFUSION_WORKBENCH_HOST="127.0.0.1"))["host"], "127.0.0.1")

    def test_nat_gateway_rules(self):
        self.assertEqual(eh.nat_gateway(ROUTE, RESOLV, IPCONFIG_EN), ("192.0.2.1", None))
        self.assertEqual(eh.nat_gateway(ROUTE, RESOLV, IPCONFIG_ORDER), ("192.0.2.1", None))   # order, labels
        stale_lan_resolv = RESOLV.replace("nameserver 192.0.2.1", "nameserver 198.51.100.1")
        cases = ((LAN_ROUTE, LAN_RESOLV, IPCONFIG_EN, "not generated by WSL"),
                 (LAN_ROUTE, RESOLV, IPCONFIG_EN, "names ['192.0.2.1'], not only the gateway 198.51.100.1"),
                 # Codex round 3: a stale WSL header naming a LAN gateway; ipconfig shows no WSL adapter with it
                 (LAN_ROUTE, stale_lan_resolv, IPCONFIG_EN, "no Windows 'vEthernet (WSL...)' adapter has the address "
                                                            "198.51.100.1"),
                 # Codex round 3: an up, device-only tun0 default route counts
                 (ROUTE + TUN0, RESOLV, IPCONFIG_EN, "2 up IPv4 default routes"),
                 (ROUTE.replace("eth0\t00000000", "wlan0\t00000000"), RESOLV, IPCONFIG_EN, "via wlan0, not eth0"),
                 (ROUTE + "eth1\t00000000\t016433C6\t0003\t0\t0\t100\t00000000\t0\t0\t0\n", RESOLV,
                  IPCONFIG_EN, "2 up IPv4 default routes"),
                 (ROUTE.splitlines()[0] + "\n" + TUN0, RESOLV, IPCONFIG_EN, "has no gateway"),
                 (ROUTE.splitlines()[0] + "\n", RESOLV, IPCONFIG_EN, "0 up IPv4 default routes"),
                 (ROUTE, RESOLV, IPCONFIG_NO_WSL, "no such adapter"))
        for route, resolv, ipc, text in cases:
            gw, why = eh.nat_gateway(route, resolv, ipc)
            self.assertIsNone(gw, (text, why))
            self.assertIn(text, why)
        self.assertEqual([n for n, _ in eh.parse_ipconfig(IPCONFIG_EN)],
                         ["Ethernet adapter Ethernet", "Ethernet adapter vEthernet (Default Switch)",
                          "Ethernet adapter vEthernet (WSL)"])

    def test_main_refuses_before_starting_anything(self):
        rc, _, err = quiet(gate_mod.main, ["--", "/bin/false"], env=self.env())
        self.assertEqual(rc, 2)
        self.assertIn("the MCP server was not started", err)


# --------------------------------------------------------------------------- the endpoint handshake


class StubProbes:
    def __init__(self, netstat, line):
        self.netstat, self.line = netstat, line

    def netstat_text(self):
        if isinstance(self.netstat, Exception):
            raise self.netstat
        return self.netstat

    def process_line(self, pid):
        return self.line


def wmi_time(unix):
    import datetime
    t = datetime.datetime.fromtimestamp(unix, datetime.timezone(datetime.timedelta(minutes=-420)))
    return t.strftime("%Y%m%d%H%M%S") + ".123456-420"


class HandshakeTests(PinCase):
    LID = "20261106T052424Z-wb-addc"

    def setUp(self):
        super().setUp()
        self.exe = os.path.join(self.fs.tools_root, *WB_REL.split("/"))
        self.created = REC_UNIX + 3
        self.write_record()
        self.write_log()

    def write_record(self, rec=None, **over):
        if rec is None:
            rec = {"schema": "dd-launch/2", "launch_id": self.LID, "status": "ok", "exe": "wb", "utc": REC_UTC,
                   "pin": {"pin_id": "1.8.0.13", "sha256": self.src["sha256"]}, "build_key": self.ident["build_key"],
                   "launch_path_id": "stable"}
            rec.update(over)
        os.makedirs(os.path.join(self.pin_dir, "launches"), exist_ok=True)
        with open(os.path.join(self.pin_dir, "launches", "launch_%s.json" % self.LID), "w") as fh:
            json.dump(rec, fh)

    def write_log(self, version="1.8.0.13", lid=None, gproj=None, name="logs_2026-11-05_22-24-26", start=None,
                  cwd=None):
        import datetime
        d = os.path.join(self.profile, "logs", name)
        os.makedirs(d, exist_ok=True)
        start = REC_UNIX + 4 if start is None else start
        utc = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cwd = cwd or (os.path.join(self.fs.tools_root, "Workbench") + "/")
        lines = ["---", "Log %s/console.log started at 2026-11-05 22:24:26 (%s UTC)" % (d, utc),
                 "22:24:26.605   ENGINE       : FileSystem: Current working directory: '%s'" % cwd,
                 "22:24:27.651   ENGINE       : CLI Params: -profile %s -gproj %s -forceUpdate -maxFPS 60 "
                 "-dcLaunchId %s" % (self.profile, gproj or self.gproj, lid or self.LID)]
        if version:
            lines.append("22:24:30.057   INIT         : Creating game instance(ArmaReforgerScripted), version %s "
                         "built 2026-08-21 14:30:18 UTC." % version)
        with open(os.path.join(d, "console.log"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def probes(self, pids=(20960,), exe=None, cmd=None, addr="0.0.0.0", created=None):
        netstat = "\n".join("  TCP    %s:5775           0.0.0.0:0              LISTENING       %d" % (addr, p)
                            for p in pids)
        netstat += "\n  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4\n"
        cmd = cmd if cmd is not None else ('"%s" -profile "%s" -gproj "%s" -addonsDir "%s" -forceUpdate -maxFPS 60 '
                                           '-dcLaunchId %s' % (exe or self.exe, self.profile, self.gproj,
                                                              os.path.join(self.fs.game_root, "addons"), self.LID))
        line = "%d\t%s\t%s\t%s\r\n" % (pids[0] if pids else 0, exe or self.exe,
                                       wmi_time(self.created if created is None else created), cmd)
        return StubProbes(netstat, line)

    def shake(self, probes, ident=None, host="127.0.0.1", gateway=None):
        return eh.handshake(self.src, ident or self.ident, 5775, probes, {}, host, gateway)

    def test_the_pins_workbench_passes(self):
        res = self.shake(self.probes())
        self.assertTrue(res["ok"], res["reasons"])
        self.assertEqual((res["facts"]["pid"], res["facts"]["launch_id"], res["facts"]["build_version"]),
                         (20960, self.LID, "1.8.0.13"))
        self.assertEqual(res["facts"]["engine_version_expected"], "1.8.0.13")

    def refused(self, probes, text, ident=None, **kw):
        res = self.shake(probes, ident, **kw)
        self.assertFalse(res["ok"])
        self.assertTrue(any(text in r for r in res["reasons"]), res["reasons"])
        return res

    def test_listeners(self):
        self.refused(self.probes(pids=()), "no process listens on 127.0.0.1:5775")
        self.refused(self.probes(pids=(1, 2)), "2 processes listen")
        self.refused(StubProbes(OSError("no interop"), ""), "cannot list listening ports")
        # a loopback-only listener does not serve the NAT gateway host BK would use
        self.refused(self.probes(addr="127.0.0.1"), "no process listens on 192.0.2.1:5775", host="192.0.2.1",
                     gateway="192.0.2.1")
        self.assertTrue(self.shake(self.probes(), host="192.0.2.1", gateway="192.0.2.1")["ok"])

    def test_endpoint_host_must_be_local(self):
        self.refused(self.probes(), "neither loopback nor the local Windows host", host="203.0.113.3",
                     gateway="192.0.2.1")

    def test_another_exe_path(self):
        other = os.path.join(self.tmp, "other", WB_REL)
        os.makedirs(os.path.dirname(other))
        shutil.copyfile(self.exe, other)
        self.refused(self.probes(exe=other), "is not the pin's Workbench")

    def test_another_exe_hash(self):
        base.FakeSteam.write(self.fs.tools_root, WB_REL, b"MZ wc", base.TOOLS_MTIME_NS)
        self.refused(self.probes(), "exe sha256")

    def test_no_exe_hash_without_sidecar(self):
        self.refused(self.probes(), "no Workbench exe hash", dict(self.ident, exe_sha256=None))

    def test_another_profile_or_project(self):
        cmd = '"%s" -profile "Y:\\Game Library\\OneDrive\\ArmaReforgerWorkbench" -gproj "%s" -dcLaunchId %s' % (
            self.exe, self.gproj, self.LID)
        self.refused(self.probes(cmd=cmd), "-profile is")
        cmd = '"%s" -profile "%s" -gproj "Y:\\Game Library\\DataTest\\addon.gproj" -dcLaunchId %s' % (self.exe, self.profile, self.LID)
        self.refused(self.probes(cmd=cmd), "-gproj is")

    def test_started_by_hand(self):
        cmd = '"%s" -profile "%s" -gproj "%s"' % (self.exe, self.profile, self.gproj)
        self.refused(self.probes(cmd=cmd), "no valid -dcLaunchId")

    def test_launch_records(self):
        self.write_record(pin={"pin_id": "1.9.0.37", "sha256": "0" * 64})
        self.refused(self.probes(), "was made for pin 1.9.0.37")
        self.write_record(build_key="e" * 64)
        self.refused(self.probes(), "has build_key eeee")
        # Codex round 1: a valid JSON value that is not an object
        self.write_record(rec=[])
        self.refused(self.probes(), "is a JSON list, not an object")
        self.write_record(rec={"launch_id": self.LID, "status": "ok", "exe": "wb",
                               "pin": {"pin_id": "1.8.0.13", "sha256": self.src["sha256"]}})
        res = self.refused(self.probes(), "schema None is not one of")
        self.assertTrue(any("has no utc time" in r for r in res["reasons"]), res["reasons"])
        os.unlink(os.path.join(self.pin_dir, "launches", "launch_%s.json" % self.LID))
        self.refused(self.probes(), "has no readable record")

    def test_build_version_against_the_engine_sidecar(self):
        self.write_log(version="1.9.0.37")
        self.refused(self.probes(), "reports version 1.9.0.37, the pin expects 1.8.0.13")
        self.write_log()
        self.assertTrue(self.shake(self.probes(), dict(self.ident, engine_version_expected=None))["ok"])

    def test_console_log_binding(self):
        shutil.rmtree(os.path.join(self.profile, "logs"))
        self.refused(self.probes(), "0 console.log(s)")
        self.write_log(version=None)
        self.refused(self.probes(), "still starting")
        self.write_log(lid="20261101T000000Z-wb-0000")
        self.refused(self.probes(), "0 console.log(s)")
        self.write_log(gproj=os.path.join(self.tmp, "other.gproj"))
        self.refused(self.probes(), "names another -gproj")
        self.write_log(cwd="Z:\\Synthetic Steam (x86)\\Elsewhere\\Workbench\\")
        self.refused(self.probes(), "has working directory")
        self.write_log()
        self.write_log(name="logs_2026-11-05_22-25-00", start=REC_UNIX + 30)
        self.refused(self.probes(), "2 console.log(s)")

    def test_process_and_log_times(self):
        # Codex round 1: a restart that reuses an old launch id, and a log of another start
        self.refused(self.probes(created=REC_UNIX + 3600), "a restart that reuses an old launch id")
        self.write_log(start=REC_UNIX + 200)
        self.refused(self.probes(), "not this process's log")
        # Codex round 2: the old process's log at record+4 s does not vouch for a replacement created at +5 s
        self.write_log(start=REC_UNIX + 4)
        self.refused(self.probes(created=REC_UNIX + 5), "not this process's log")
        self.assertTrue(self.shake(self.probes(created=REC_UNIX + 4))["ok"])

    def test_bound_log_must_be_the_newest_of_the_profile(self):
        # a later start in the same profile (by hand: no -dcLaunchId) makes the launch's log stale
        self.write_log(lid="ZZZ-not-a-launch-of-ours", name="logs_2026-11-05_22-30-00", start=REC_UNIX + 400)
        self.refused(self.probes(), "is not the newest log")

    def test_windows_probe_environment(self):
        p = eh.WindowsProbes(win_temp="/mnt/x/DataRoot/tmp/mcp")
        env = p.win_env()
        self.assertEqual(env["WSLENV"], eh.win_env("/mnt/x/DataRoot/tmp/mcp")["WSLENV"])
        self.assertEqual((env["TEMP"], env["TMP"]), ("/mnt/x/DataRoot/tmp/mcp", "/mnt/x/DataRoot/tmp/mcp"))
        self.assertEqual(env["WSLENV"].split(":")[-2:], ["TEMP/p", "TMP/p"])
        self.assertEqual(eh.parse_env_line("TEMP=X:\\DataRoot\\tmp\\mcp\tTMP=X:\\DataRoot\\tmp\\mcp\r\n"),
                         {"TEMP": "X:\\DataRoot\\tmp\\mcp", "TMP": "X:\\DataRoot\\tmp\\mcp"})
        with self.assertRaises(OSError):   # a temp dir off the data drive is refused before cscript runs
            eh.WindowsProbes(win_temp=self.tmp).check_env()
        # DD-22: the Windows tools and the temp dir come from the local config
        self.assertEqual((p.netstat, p.cscript), (base.SYSTEM_DRIVE + "/Windows/System32/NETSTAT.EXE",
                                                  base.SYSTEM_DRIVE + "/Windows/System32/cscript.exe"))
        self.assertEqual(eh.MCP_WIN_TEMP, "/mnt/x/DataRoot/tmp/mcp")
        self.assertEqual(eh.IPCONFIG, base.SYSTEM_DRIVE + "/Windows/System32/ipconfig.exe")

    def test_cli_main_reads_only(self):
        before = sorted(os.listdir(self.pin_dir))
        rc, out, _ = quiet(eh.main, ["--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir, "--host", "127.0.0.1"],
                           probes=self.probes(), gateway="192.0.2.1")
        self.assertEqual(rc, 0, out)
        self.assertIn("HANDSHAKE OK (pin 1.8.0.13)", out)
        rc, out, _ = quiet(eh.main, ["--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir, "--host", "127.0.0.1"],
                           probes=self.probes(pids=()), gateway="192.0.2.1")
        self.assertEqual(rc, 1)
        self.assertEqual(sorted(os.listdir(self.pin_dir)), before)

    def test_parsing(self):
        argv = eh.split_cmdline('"Z:\\Synthetic Steam (x86)\\x.exe" -profile "X:\\DataRoot\\a b" -forceUpdate -maxFPS 60 -dcLaunchId X')
        self.assertEqual(argv[0], "Z:\\Synthetic Steam (x86)\\x.exe")
        self.assertEqual(eh.cli_args(argv[1:]), {"profile": "X:\\DataRoot\\a b", "forceupdate": "", "maxfps": "60",
                                                 "dclaunchid": "X"})
        self.assertEqual(eh.listening_pids("  TCP    0.0.0.0:57750   0.0.0.0:0   LISTENING   9\n"
                                           "  TCP    [::]:5775       [::]:0      LISTENING   7\n", 5775), {7})
        self.assertEqual(eh.wmi_time_to_unix("20261105222427.123456-420"), REC_UNIX + 3)
        self.assertIsNone(eh.wmi_time_to_unix("garbage"))
        self.assertIsNone(eh.parse_process_line("1\ta\n2\tb\n", 1))


# --------------------------------------------------------------------------- the gate policy and the stdio proxy


STUB_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"echo": msg.get("method"),
                                     "name": (msg.get("params") or {}).get("name")}}) + "\n")
        sys.stdout.flush()
'''

# Every denied BK tool and action at the pinned commit (bk_tools.py), with a typical argument set: the writes, and
# (Codex round 3) every tool that takes a filesystem, resource or mod path.
WRITE_CALLS = [
    ("project", {"action": "write", "path": "Scripts/Game/X.c", "content": "x"}),
    ("script_create", {"className": "X"}), ("config_create", {"configType": "faction", "name": "x"}),
    ("layout_create", {"name": "x"}), ("scenario_create_conflict", {"scenarioName": "x"}),
    ("server_config", {"name": "x"}), ("building_setup", {"manifestPath": "m.json"}),
    ("game_duplicate", {"sourcePath": "Prefabs/a.et", "destPath": "Prefabs/b.et"}),
    ("prefab", {"action": "create", "name": "x"}), ("prefab", {"action": "inspect", "path": "x"}),
    ("mod", {"action": "create", "name": "x"}), ("mod", {"action": "build"}), ("mod", {"action": "validate"}),
    ("animation_graph", {"action": "author", "vehicleName": "x"}), ("animation_graph", {"action": "setup"}),
    ("animation_graph", {"action": "inspect"}), ("gamemode_scaffold", {"action": "create"}),
    ("gamemode_scaffold", {"action": "list"}), ("layout_recipe", {"action": "create", "name": "x"}),
    ("menu_create", {"action": "create", "className": "X"}), ("mod_build", {}), ("wb_launch", {}),
    ("wb_cleanup", {"all": True}), ("wb_play", {}), ("wb_stop", {}), ("wb_save", {}),
    ("wb_undo_redo", {"action": "undo"}), ("wb_undo_redo", {"action": "redo"}),
    ("wb_open_resource", {"path": "worlds/DataCapture/DC_Arland.ent"}), ("wb_entity_create", {"name": "x"}),
    ("wb_entity_delete", {"name": "x"}), ("wb_entity_duplicate", {"entityName": "x", "destPath": "Prefabs/x.et"}),
    ("wb_execute_action", {"menuPath": "File/Save"}), ("wb_reload", {"target": "scripts"}),
    ("scenario_create", {"type": "base"}),
] + [("wb_entity_modify", {"action": a, "name": "x"}) for a in (
    "move", "rotate", "rename", "reparent", "setProperty", "clearProperty", "addArrayItem", "removeArrayItem",
    "setObjectClass", "makeVisible")] + [("wb_layers", {"action": a, "name": "x"}) for a in (
    "create", "delete", "rename", "setActive", "setVisibility", "lock", "unlock", "toggleLock")] + [
    ("wb_prefabs", {"action": a}) for a in ("createTemplate", "save")] + [
    ("wb_resources", {"action": a, "path": "Prefabs/x.et"}) for a in ("register", "rebuild", "open")] + [
    ("wb_script_editor", {"action": a, "path": "Scripts/x.c"}) for a in ("setLine", "insertLine", "removeLine",
                                                                         "openFile")] + [
    ("wb_localization", {"action": a}) for a in ("insert", "delete", "modify")] + [
    ("wb_clipboard", {"action": a}) for a in ("copy", "cut", "paste", "pasteAtCursor", "duplicate", "hasCopied")] + [
    ("wb_component", {"action": a, "entityName": "x"}) for a in ("add", "remove")] + [
    ("wb_projects", {"action": "open", "name": "X:\\DataRoot\\addons\\DataCapture\\worlds\\x.ent"}),
    ("wb_entity_select", {"action": "select", "name": "x"}), ("wb_entity_select", {"action": "deselect"}),
    ("wb_entity_select", {"action": "clear"})]
PATH_CALLS = [("project", {"action": "browse"}), ("project", {"action": "read", "path": "Scripts/Game/X.c"}),
              ("workshop_info", {}), ("workshop_info", {"modName": "/x/stable/Elsewhere"}),
              ("wb_read_props", {"resource": "Prefabs/x.et", "path": "a/b"}), ("asset_search", {"query": "x"}),
              ("game_browse", {"path": "Prefabs/Vehicles"}), ("game_read", {"path": "Prefabs/x.et"}),
              ("wb_validate", {"action": "material", "path": "x.emat"}),
              ("wb_validate", {"action": "texture", "path": "x.edds"}),
              ("wb_resources", {"action": "getInfo", "path": "x"}), ("wb_prefabs", {"action": "getGuid"}),
              ("wb_prefabs", {"action": "locate", "searchPath": "x"}),
              ("wb_prefabs", {"action": "getAncestor", "entityName": "x"}),
              ("wb_projects", {"action": "list"}), ("wb_projects", {"action": "locate", "name": "x"}),
              ("wb_diagnose", {})]
READ_CALLS = [("api_search", {"query": "ZxWheelFx"}), ("wiki_search", {"query": "x"}), ("wiki_read", {"title": "x"}),
              ("component_search", {"query": "x"}), ("wb_knowledge", {"query": "index"})]
WB_READ_CALLS = [("wb_connect", {}), ("wb_state", {}), ("wb_entity_list", {"limit": 10}),
                 ("wb_entity_inspect", {"name": "x"}), ("wb_terrain", {"action": "getHeight", "x": 1, "z": 2}),
                 ("wb_terrain", {"action": "getBounds"}), ("wb_layers", {"action": "list", "subScene": 0}),
                 ("wb_layers", {"action": "getInfo", "layerPath": "default"}),
                 ("wb_layers", {"action": "isVisible", "layerPath": "default"}),
                 ("wb_localization", {"action": "getTable"}), ("wb_localization", {"action": "listLanguages"}),
                 ("wb_script_editor", {"action": "getCurrentFile"}), ("wb_script_editor", {"action": "getLine",
                                                                                       "line": 3}),
                 ("wb_script_editor", {"action": "getLinesCount"}), ("wb_entity_select", {"action": "getSelected"}),
                 ("wb_entity_modify", {"action": "getProperty", "name": "x", "propertyPath": "coords"}),
                 ("wb_entity_modify", {"action": "listProperties", "name": "x"}),
                 ("wb_entity_modify", {"action": "listArrayItems", "name": "x", "propertyPath": "c"}),
                 ("wb_entity_modify", {"action": "getWorldTransform", "name": "x"}),
                 ("wb_component", {"action": "list", "entityName": "x"})]


class GatePolicyTests(unittest.TestCase):
    """Coordinator decisions (Codex rounds 2-3): deny by default; only path-free reads, and wb_* getters after the
    handshake."""

    def setUp(self):
        self.calls = []

    def gate(self, ok=True):
        def hs():
            self.calls.append(1)
            return {"ok": ok, "reasons": [] if ok else ["stub endpoint mismatch"], "facts": {}}
        return gate_mod.Gate(hs, log=lambda m: None)

    def test_every_write_and_path_tool_is_denied(self):
        g = self.gate(ok=True)
        for name, args in WRITE_CALLS + PATH_CALLS:
            why = g.decide(name, dict(args))
            self.assertIsNotNone(why, (name, args))
            self.assertIn("is refused", why, (name, args))
        self.assertEqual(self.calls, [])          # denied before any handshake
        self.assertEqual(set(bk.DENIED) - {n for n, _ in WRITE_CALLS + PATH_CALLS}, set())

    def test_material_and_texture_validation_are_denied(self):
        # Codex round 3 R3-1: MaterialValidator can change shared Workbench state
        for action in ("material", "texture", None):
            why = self.gate().decide("wb_validate", {"action": action, "path": "x"})
            self.assertIn("can change shared Workbench state", why)
        self.assertNotIn("wb_validate", bk.ALLOW)

    def test_codex_read_counterexamples_are_denied(self):
        # Codex round 3 R3-2: a symlinked Scripts/ read, a modName outside the root, and LoadContainer's resource
        g = self.gate()
        for name, args in (("project", {"action": "read", "path": "Scripts/proof.c"}),
                           ("project", {"action": "browse", "path": "Scripts"}),
                           ("workshop_info", {"modName": "/x/stable/Elsewhere"}),
                           ("wb_read_props", {"resource": "{0123456789ABCDEF}Prefabs/x.et", "path": "a"})):
            self.assertIn("takes a filesystem, resource or mod path", g.decide(name, args), name)

    def test_diagnose_is_denied_and_connect_stays(self):
        # Codex round 4: wb_diagnose lists each addon's EnfusionMCP handler dir without resolving symlinks
        g = self.gate(ok=True)
        for args in ({}, None):
            why = g.decide("wb_diagnose", args)
            self.assertIn("without resolving symlinks", why)
            self.assertIn("indirect filesystem read", why)
        self.assertEqual(self.calls, [])          # denied before any handshake
        self.assertNotIn("wb_diagnose", bk.ALLOW)
        self.assertIsNone(g.decide("wb_connect", {}))       # a ping; stays, after the handshake
        self.assertEqual(len(self.calls), 1)

    def test_project_switch_is_denied(self):
        why = self.gate().decide("wb_projects", {"action": "open", "name": "x"})
        self.assertIn("is refused", why)

    def test_unknown_and_missing_actions_and_extra_arguments_are_denied(self):
        g = self.gate()
        for name, args in (("wb_terrain", {"action": "UNKNOWN"}), ("wb_terrain", {}), ("wb_layers", {"action": None}),
                           ("wb_layers", {"action": "create", "name": "x"}), ("brand_new_tool", {}),
                           ("wb_script_editor", {"action": "getLine", "line": 1, "path": "Scripts/x.c"}),
                           ("wb_layers", {"action": "list", "name": "x"}), ("api_search", {"query": "x", "path": "/"}),
                           ("wb_state", {"action": "x"}), ("wiki_read", "not-an-object")):
            self.assertIn("is refused", g.decide(name, args), (name, args))
        self.assertEqual(self.calls, [])

    def test_reads_pass_and_wb_reads_need_the_handshake(self):
        for name, args in READ_CALLS:
            self.assertIsNone(self.gate(ok=False).decide(name, dict(args)), name)
        self.assertEqual(self.calls, [])
        for name, args in WB_READ_CALLS:
            self.calls = []
            self.assertIn("handshake refused", self.gate(ok=False).decide(name, dict(args)), name)
            self.assertEqual(len(self.calls), 1, name)
            self.assertIsNone(self.gate(ok=True).decide(name, dict(args)), name)
        allowed = {(n, (a or {}).get("action")) for n, a in READ_CALLS + WB_READ_CALLS}
        for name, (_, actions) in bk.ALLOW.items():      # every allowed entry is exercised
            for action in actions:
                self.assertIn((name, action), allowed)

    def test_sealed_map_tokens_and_guid(self):
        g = self.gate()
        for args in ({"query": "worlds/Eden/Eden.ent"}, {"query": "EdenTerrain"}, {"query": "Everon_route.json"},
                     {"query": "{853E92315D1D9EFE}worlds/X/Y.ent"}, {"query": "dc_everon"},
                     {"query": "x", "source": ["dc", "EVERON"]}):
            self.assertIn("sealed", g.decide("api_search", args), args)
        self.assertIn("sealed", g.decide("wiki_read", {"title": "Everon"}))
        self.assertIn("is refused", g.decide("api_search", {"EVERON": 1}))      # an unknown key: denied first
        for args in ({"query": "foreverOnline"}, {"query": "credentials"}, {"query": "Sweden"}, {"query": "edenic"}):
            self.assertIsNone(g.decide("api_search", args), args)
        self.assertEqual(bk.identifier_tokens("worlds/EdenTerrain_v2"), ["worlds", "eden", "terrain", "v", "2"])

    def test_table_matches_the_bk_source(self):
        # Opt-in: the pinned BK source ($DD_TEST_BK_SRC; absent in CI). Every BK tool is classified (allowed or denied).
        if not BK_SRC or not os.path.isdir(BK_SRC):
            self.skipTest("set DD_TEST_BK_SRC to a pinned BK checkout's src/tools to run this test")
        names = set()
        for fn in os.listdir(BK_SRC):
            if fn.endswith(".ts"):
                with open(os.path.join(BK_SRC, fn), encoding="utf-8") as fh:
                    names |= set(re.findall(r'registerTool\(\s*"([^"]+)"', fh.read()))
        self.assertEqual(names - bk.known_tools(), set())
        self.assertEqual(bk.known_tools() - names, {"mod_build"})   # an old npm name, kept denied


class ProxyTests(unittest.TestCase):
    def run_proxy(self, msgs, handshake_ok=True, root=None):
        calls = []

        def hs():
            calls.append(1)
            return {"ok": handshake_ok, "reasons": [] if handshake_ok else ["stub endpoint mismatch"],
                    "facts": {"pid": 1, "launch_id": "x"}}
        gate = gate_mod.Gate(hs, log=lambda m: None)
        stdin = io.BytesIO(b"".join((json.dumps(m) + "\n").encode() for m in msgs))
        stdout = io.BytesIO()
        rc = gate_mod.proxy(gate, [sys.executable, "-I", "-c", STUB_SERVER], dict(os.environ), stdin=stdin,
                            stdout=stdout)
        self.assertEqual(rc, 0)
        replies = {r["id"]: r for r in (json.loads(x) for x in stdout.getvalue().decode().splitlines())}
        return replies, calls

    def call(self, i, name, args=None):
        return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}

    def test_forwards_and_gates(self):
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                self.call(2, "api_search", {"query": "ZxWheelFx"}),
                self.call(3, "wb_state"),
                self.call(4, "wb_entity_list"),
                self.call(5, "script_create", {"className": "X"})]
        replies, calls = self.run_proxy(msgs)
        self.assertEqual(replies[1]["result"]["echo"], "initialize")
        for i, name in ((2, "api_search"), (3, "wb_state"), (4, "wb_entity_list")):
            self.assertEqual(replies[i]["result"]["name"], name)
        self.assertTrue(replies[5]["result"]["isError"])   # a write: denied, never forwarded
        self.assertNotIn("echo", replies[5]["result"])
        self.assertEqual(len(calls), 2)  # wb_state and wb_entity_list; none for api_search or the denied write

    def test_refused_handshake_is_never_forwarded(self):
        replies, _ = self.run_proxy([self.call(3, "wb_state"), self.call(4, "wb_layers", {"action": "list"}),
                                     self.call(5, "api_search")], handshake_ok=False)
        for i in (3, 4):
            self.assertTrue(replies[i]["result"]["isError"])
            self.assertIn("stub endpoint mismatch", replies[i]["result"]["content"][0]["text"])
            self.assertNotIn("echo", replies[i]["result"])
        self.assertEqual(replies[5]["result"]["name"], "api_search")

    def test_batch_with_a_refused_call(self):
        gate = gate_mod.Gate(lambda: {"ok": True, "reasons": [], "facts": {}}, log=lambda m: None)
        fwd, replies = gate.filter_line((json.dumps([self.call(1, "wb_launch"), {"jsonrpc": "2.0", "id": 2,
                                                                                  "method": "ping"}]) + "\n").encode())
        self.assertEqual([r["id"] for r in replies], [1])
        self.assertEqual(json.loads(fwd), [{"jsonrpc": "2.0", "id": 2, "method": "ping"}])


# --------------------------------------------------------------------------- the wrapper


class WrapperTests(PinCase):
    WRAPPER = os.path.join(EMCP, "enfusion-mcp-wsl.sh")

    def test_server_only_behind_the_gate(self):
        with open(self.WRAPPER) as fh:
            text = fh.read()
        execs = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("exec ")]
        self.assertEqual(execs, ['exec "$PY" -I -B "$GATE" -- "$NODE_BIN" "$DIST" "$@"'])
        self.assertNotIn('export ENFUSION_MCP_DATA_DIR="$ROOT/data"', text)
        self.assertEqual(subprocess.run(["bash", "-n", self.WRAPPER]).returncode, 0)

    def test_print_env_runs_the_gate(self):
        # The wrapper accepts a work root the write boundary allows: RAM here (the synthetic config's drives are
        # mounted nowhere).
        dist = os.path.join(self.root, "build", "enfusion-mcp-BK", "dist")
        os.makedirs(os.path.join(dist, "workbench"))
        os.makedirs(os.path.join(dist, "tools"))
        node = os.path.join(self.tmp, "node")
        with open(node, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(node, 0o755)
        env = {"PATH": "/usr/bin:/bin", "EMCP_ROOT": self.root, "NODE_BIN": node, "ENFUSION_WORKBENCH_HOST": "127.0.0.1",
               "EMCP_PIN_DIR": self.pin_dir, "EMCP_PIN": "1.8.0.13", "HOME": self.tmp,
               "DD_PIN_TOOLS": os.path.join(REPO, "tools", "pin"), "DRIVEDIT_LOCAL_CONFIG": base.LOCAL_CONFIG_FIXTURE}
        r = subprocess.run(["bash", self.WRAPPER, "--print-env"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("REFUSED", r.stderr)
        self.assertIn("has no API index", r.stderr)
        self.assertEqual(r.stdout, "")


class LazyPinToolsTests(unittest.TestCase):
    """Codex round 2 row 15: importing endpoint_handshake (and mcp_gate) in an installed MCP layout reads no config;
    tools/pin is found on first use."""

    def test_import_reads_no_config(self):
        tmp = tempfile.mkdtemp(prefix="dd22-lazy-", dir=base._tmp_base())
        self.addCleanup(shutil.rmtree, tmp, True)
        installed = os.path.join(tmp, "mcp")   # no tools/pin next to it, no local_config.py
        os.makedirs(installed)
        shutil.copyfile(os.path.join(EMCP, "endpoint_handshake.py"), os.path.join(installed, "endpoint_handshake.py"))
        code = ("import importlib.util, sys\n"
                "spec = importlib.util.spec_from_file_location('eh_lazy', sys.argv[1])\n"
                "eh = importlib.util.module_from_spec(spec); spec.loader.exec_module(eh)\n"
                "print('imported')\n"
                "try:\n    eh.pinlib.PinError\nexcept SystemExit as exc:\n    print('first use:', exc)\n")
        env = {k: v for k, v in os.environ.items() if k != "DD_PIN_TOOLS"}
        env["DRIVEDIT_LOCAL_CONFIG"] = os.path.join(tmp, "no-such.local.env")
        r = subprocess.run([sys.executable, "-I", "-B", "-c", code, os.path.join(installed, "endpoint_handshake.py")],
                           capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[0], "imported")
        self.assertIn("first use: endpoint_handshake: cannot find tools/pin", r.stdout)

    def test_the_lazy_modules_are_the_pin_tools(self):
        self.assertIs(eh.pinlib.PinError, pinlib.PinError)
        self.assertIs(gate_mod.pinlib.check_write_path, pinlib.check_write_path)


class ShellGuardTests(unittest.TestCase):
    """The MCP shell scripts write only where the write boundary allows (tools/local_config.py check-write), and
    launch-workbench.sh has no way around the pin preflight (Codex round 2 rows 2, 14)."""

    def run_script(self, name, *args, **env):
        e = {"PATH": "/usr/bin:/bin", "HOME": "/dev/shm", "DRIVEDIT_LOCAL_CONFIG": base.LOCAL_CONFIG_FIXTURE}
        e.update(env)
        return subprocess.run(["bash", os.path.join(EMCP, name)] + list(args), env=e, capture_output=True,
                              text=True, timeout=120)

    def test_work_roots_off_the_boundary_are_refused(self):
        never = "/tmp/__dd22_emcp_never_written__"
        for name in ("enfusion-mcp-wsl.sh", "build-bk.sh"):
            for root in (never, "relative/root", base.SYSTEM_DRIVE + "/mcp", base.DATA_DRIVE + "mcp"):
                r = self.run_script(name, "--print-env" if name.startswith("enfusion") else "--smoke",
                                    EMCP_ROOT=root)
                self.assertEqual(r.returncode, 2, (name, root, r.stderr))
                self.assertIn("refusing EMCP_ROOT=%s" % root, r.stderr)
        self.assertFalse(os.path.exists(never))

    def test_launch_workbench_has_no_preflight_bypass(self):
        with open(os.path.join(EMCP, "launch-workbench.sh")) as fh:
            code = [ln for ln in fh.read().splitlines() if not ln.lstrip().startswith("#")]
        self.assertFalse([ln for ln in code if "SKIP" in ln])
        # a gproj off the boundary (the repo's own, on a drive the synthetic config does not verify) is refused
        # before tasklist, the preflight or Workbench is touched
        gproj = os.path.join(REPO, "addons", "DataCapture", "addon.gproj")
        r = self.run_script("launch-workbench.sh", gproj, WB_SKIP_PREFLIGHT="1")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("refusing gproj", r.stderr)


if __name__ == "__main__":
    unittest.main()
