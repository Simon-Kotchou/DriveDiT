"""Tests for the per-pin MCP index, the index diff, the pin gate and the endpoint handshake (ticket DD-14a, CC-05).
Stdlib unittest only; no Workbench, no MCP server, no Windows process: the Windows probes are stubs, the MCP server
is a stub child that echoes.

    python3 -I -B -m unittest discover -s tools/enfusion-mcp/tests -v

Temp trees go under $DD_PIN_TEST_TMP, else /dev/shm, and are deleted afterwards.
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
gate_mod = _load("dd_emcp_mcp_gate", os.path.join(EMCP, "mcp_gate.py"))
WB_REL = "Workbench/ArmaReforgerWorkbenchSteamDiag.exe"


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a, **kw)
    return rc, out.getvalue(), err.getvalue()


class PinCase(base.TmpCase):
    """A fake Steam layout, pin 1.8.0.13 (dd-pin/1) with its build_key sidecar, and a fake MCP root."""

    def setUp(self):
        super().setUp()
        self.fs.set_auto_update("1")
        self.assertEqual(base.run_main(make_pin.main, self.fs.make_pin_argv(), self.roots)[0], 0)
        self.pin_dir = self.fs.out_dir
        rc, _, err = base.run_main(pin_sidecar.main, ["build-key", "1.8.0.13", "--pin-dir", self.pin_dir,
                                                      "--from-live", "--apply"], self.roots)
        self.assertEqual(rc, 0, err)
        self.src = preflight.load_pin_by_id(self.pin_dir, "1.8.0.13")
        self.ident = preflight.pin_identity(self.src)
        with open(os.path.join(self.pin_dir, "ACTIVE"), "w") as fh:
            fh.write("1.8.0.13\n")
        self.root = os.path.join(self.tmp, "mcp")
        os.makedirs(os.path.join(self.root, "pins"))
        self.project = os.path.join(self.tmp, "addons")
        os.makedirs(os.path.join(self.project, "DataCapture"))
        self.profile = os.path.join(self.tmp, "wb_profile")
        self.gproj = os.path.join(self.project, "DataCapture", "addon.gproj")
        for name, value in (("DEPLOY_ROOTS", {"stable": self.project}),
                            ("WB_LAUNCH", {"stable": {"profile": self.profile, "gproj": self.gproj,
                                                      "launcher": "wb_R1.bat"}})):
            saved = getattr(pinlib, name)
            setattr(pinlib, name, value)
            self.addCleanup(setattr, pinlib, name, saved)

    def build(self, pin_id="1.8.0.13", dump="dump_18", pin_dir=None):
        out = os.path.join(self.root, "pins", pin_id, "data")
        rc, o, err = quiet(api_index.main, ["build", "--dump-dir", os.path.join(FIX, dump, "scripts"), "--pin-id",
                                            pin_id, "--pin-dir", pin_dir or self.pin_dir, "--out", out, "--apply"],
                           write_roots=self.roots)
        self.assertEqual(rc, 0, err)
        return out

    def env(self, **extra):
        e = {"EMCP_ROOT": self.root, "EMCP_PIN_DIR": self.pin_dir, "EMCP_PIN": "1.8.0.13"}
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
        _, classes = api_index_diff.load_index(out)
        vw = classes["VehicleWheeledSimulation"]
        self.assertEqual((vw["parents"], vw["source"], vw["group"]), (["VehicleBaseSimulation"], "arma", "Vehicle"))
        self.assertEqual([m["name"] for m in vw["methods"]], ["GetSteering", "SetSteering", "SetBreak", "GetGear"])
        self.assertEqual(vw["methods"][0]["description"], "Returns current steering input in range <-1,1>")
        self.assertEqual(vw["methods"][2]["params"], [{"name": "in", "type": "float", "defaultValue": ""},
                                                      {"name": "hb", "type": "bool", "defaultValue": ""}])
        cam = classes["ScriptCamera"]
        self.assertEqual(cam["source"], "enfusion")
        self.assertEqual(sorted(m["name"] for m in cam["methods"]), ["EOnFrame", "ScriptCamera"])
        self.assertEqual([p["name"] for p in cam["properties"]], ["FOV"])
        self.assertEqual([p["name"] for p in cam["protectedProperties"]], ["m_fSpeed"])
        self.assertEqual([m["name"] for m in classes["TestHarness"]["staticMethods"]],
                         ["GetNSuites", "Begin", "ActiveSuite"])
        enum = classes["EVehicleDrivingAssistanceMode"]["enums"][0]
        self.assertEqual([(v["name"], v["value"]) for v in enum["values"]], [("NONE", ""), ("PARTIAL", ""),
                                                                             ("FULL", "2")])
        self.assertEqual(enum["values"][0]["description"], "Manual shifting")

    def test_multiline_declaration_and_new_class(self):
        out = self.build("1.8.0.13", "dump_19")
        _, classes = api_index_diff.load_index(out)
        m = [x for x in classes["VehicleWheeledSimulation"]["methods"] if x["name"] == "UpdateTyreConfiguration"][0]
        self.assertEqual(m["signature"], "proto external bool UpdateTyreConfiguration (notnull map<string, float> "
                                         "values, int wheelIdx = -1)")
        self.assertIn("VehicleWheelEffectComponent", classes)

    def test_built_once_and_dry_run(self):
        out = os.path.join(self.root, "pins", "1.8.0.13", "data")
        rc, _, err = quiet(api_index.main, ["build", "--dump-dir", os.path.join(FIX, "dump_18", "scripts"),
                                            "--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir, "--out", out,
                                            "--dry-run"], write_roots=self.roots)
        self.assertEqual(rc, 0, err)
        self.assertFalse(os.path.exists(out))
        self.build()
        rc, _, err = quiet(api_index.main, ["build", "--dump-dir", os.path.join(FIX, "dump_18", "scripts"),
                                            "--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir, "--out", out,
                                            "--apply"], write_roots=self.roots)
        self.assertEqual(rc, 2, err)
        self.assertIn("already exists", err)

    def test_out_must_be_the_pins_dir(self):
        rc, _, err = quiet(api_index.main, ["build", "--dump-dir", os.path.join(FIX, "dump_18", "scripts"),
                                            "--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir, "--out",
                                            os.path.join(self.root, "pins", "1.9.0.37", "data"), "--apply"],
                           write_roots=self.roots)
        self.assertEqual(rc, 2, err)

    def test_reads_a_git_tag_without_checkout(self):
        repo = os.path.join(self.tmp, "sd")
        os.makedirs(repo)
        email = "noreply" + "@" + "localhost"

        def git(*args):
            subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=" + email,
                            "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"] + list(args), check=True,
                           capture_output=True)
        git("init", "-q")
        for dump, tag in (("dump_18", "v1.8.0.13"), ("dump_19", "v1.9.0.37")):
            shutil.rmtree(os.path.join(repo, "scripts"), ignore_errors=True)
            shutil.copytree(os.path.join(FIX, dump, "scripts"), os.path.join(repo, "scripts"))
            git("add", "-A")
            git("commit", "-q", "-m", tag)
            git("tag", tag)
        commit, files = api_index.read_tree_git(repo, "v1.8.0.13")
        self.assertEqual(len(files), 5)
        self.assertNotIn("Game/generated/Vehicles/VehicleWheelEffectComponent.c", dict(files))
        _, files19 = api_index.read_tree_git(repo, "v1.9.0.37")
        self.assertIn("Game/generated/Vehicles/VehicleWheelEffectComponent.c", dict(files19))


class IndexDiffTests(PinCase):
    def setUp(self):
        super().setUp()
        self.old = self.build("1.8.0.13", "dump_18")
        self.new = os.path.join(self.tmp, "new", "pins", "1.8.0.13", "data")
        rc, _, err = quiet(api_index.main, ["build", "--dump-dir", os.path.join(FIX, "dump_19", "scripts"),
                                            "--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir, "--out", self.new,
                                            "--apply"], write_roots=self.roots)
        self.assertEqual(rc, 0, err)

    def diff(self, expect):
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            return quiet(api_index_diff.main, ["--old", self.old, "--new", self.new, "--expect", expect])
        finally:
            os.chdir(cwd)

    def test_fixture_expectations_hold(self):
        rc, out, err = self.diff(os.path.join(FIX, "expect_fixture.txt"))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: OK (7 of 7 assertions hold)", out)
        self.assertIn("planned_fixture.txt: 2 new-in-new, 2 unchanged", out)

    def test_wrong_expectations_fail(self):
        bad = os.path.join(self.tmp, "bad.txt")
        with open(bad, "w") as fh:
            fh.write("class-added ScriptCamera\nunchanged TestHarness Begin\nmember-removed VehicleWheeledSimulation "
                     "GetSteering\nplanned %s\nnonsense x\n" % os.path.join(FIX, "planned_fixture.txt"))
        with open(os.path.join(FIX, "planned_fixture.txt")) as fh:
            planned = fh.read()
        with open(os.path.join(self.tmp, "planned_bad.txt"), "w") as fh:
            fh.write(planned + "TestHarness GetNSuites\n")
        with open(bad, "a") as fh:
            fh.write("planned %s\n" % os.path.join(self.tmp, "planned_bad.txt"))
        rc, out, _ = self.diff(bad)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL line 2 unchanged TestHarness Begin: Begin SIG_CHANGED", out)
        self.assertIn("TestHarness.GetNSuites REMOVED", out)
        self.assertIn("RESULT: MISMATCH (1 of 6 assertions hold)", out)

    def test_tampered_index_refused(self):
        with open(os.path.join(self.new, "api", "arma-classes.json"), "a") as fh:
            fh.write(" ")
        rc, _, err = self.diff(os.path.join(FIX, "expect_fixture.txt"))
        self.assertEqual(rc, 2, err)
        self.assertIn("does not match the sha256", err)


# --------------------------------------------------------------------------- the pin gate (wrapper config)


class GateConfigTests(PinCase):
    def test_accepts_the_pins_own_index(self):
        out = self.build()
        cfg = gate_mod.resolve_config(self.env())
        self.assertEqual((cfg["data_dir"], cfg["project"], cfg["resolved_by"]), (out, self.project, "EMCP_PIN"))
        env = gate_mod.server_env({"X": "1"}, cfg)
        self.assertEqual((env["ENFUSION_MCP_DATA_DIR"], env["ENFUSION_PROJECT_PATH"], env["ENFUSION_GAME_PATH"]),
                         (out, self.project, self.fs.game_root))

    def test_active_pointer(self):
        self.build()
        cfg = gate_mod.resolve_config({"EMCP_ROOT": self.root, "EMCP_PIN_DIR": self.pin_dir})
        self.assertEqual(cfg["resolved_by"], "ACTIVE")

    def assertRefused(self, env, text):
        with self.assertRaises(gate_mod.GateRefusal) as cm:
            gate_mod.resolve_config(env)
        self.assertIn(text, str(cm.exception))

    def test_no_index_no_fallback(self):
        self.assertRefused(self.env(), "has no API index")

    def test_index_of_another_pin_refused(self):
        other = self.build()
        # the index of 1.8.0.13 copied in as if it were another pin's
        rc, _, err = base.run_main(make_pin.main, ["--pin-id", "1.8.0.14", "--out",
                                                   os.path.join(self.pin_dir, "pin_1.8.0.14.json"), "--game-acf",
                                                   self.fs.game_acf, "--tools-acf", self.fs.tools_acf], self.roots)
        self.assertEqual(rc, 0, err)
        shutil.copytree(other, os.path.join(self.root, "pins", "1.8.0.14", "data"))
        self.assertRefused(self.env(EMCP_PIN="1.8.0.14"), "but ACTIVE is 1.8.0.13")
        with open(os.path.join(self.pin_dir, "ACTIVE"), "w") as fh:
            fh.write("1.8.0.14\n")
        self.assertRefused(self.env(EMCP_PIN="1.8.0.14"), "is for pin 1.8.0.13")

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

    def test_main_refuses_before_starting_anything(self):
        rc, _, err = quiet(gate_mod.main, ["--", "/bin/false"], env=self.env())
        self.assertEqual(rc, 2)
        self.assertIn("the MCP server was not started", err)


# --------------------------------------------------------------------------- the endpoint handshake


class StubProbes:
    def __init__(self, netstat, proc):
        self.netstat, self.proc = netstat, proc
        self.calls = []

    def netstat_text(self):
        self.calls.append("netstat")
        if isinstance(self.netstat, Exception):
            raise self.netstat
        return self.netstat

    def process_json(self, pid):
        self.calls.append(("process", pid))
        return json.dumps(self.proc) if not isinstance(self.proc, str) else self.proc


class HandshakeTests(PinCase):
    LID = "20261106T052424Z-wb-addc"

    def setUp(self):
        super().setUp()
        self.exe = os.path.join(self.fs.tools_root, *WB_REL.split("/"))
        self.write_record()
        self.write_log()

    def write_record(self, **over):
        rec = {"schema": "dd-launch/2", "launch_id": self.LID, "status": "ok", "exe": "wb",
               "pin": {"pin_id": "1.8.0.13", "sha256": self.src["sha256"]}, "build_key": self.ident["build_key"],
               "launch_path_id": "stable"}
        rec.update(over)
        os.makedirs(os.path.join(self.pin_dir, "launches"), exist_ok=True)
        with open(os.path.join(self.pin_dir, "launches", "launch_%s.json" % self.LID), "w") as fh:
            json.dump(rec, fh)

    def write_log(self, version="1.8.0.13", lid=None, gproj=None, name="logs_2026-11-06_12-45-17"):
        d = os.path.join(self.profile, "logs", name)
        os.makedirs(d, exist_ok=True)
        lines = ["Log %s/console.log started at 2026-11-06 12:45:17 (2026-11-06 19:45:17 UTC)" % d,
                 "12:45:18.651   ENGINE       : CLI Params: -profile %s -gproj %s -forceUpdate -maxFPS 60 "
                 "-dcLaunchId %s" % (self.profile, gproj or self.gproj, lid or self.LID)]
        if version:
            lines.append("12:45:25.775   INIT         : Creating game instance(ArmaReforgerScripted), version %s "
                         "built 2026-08-21 14:30:18 UTC." % version)
        with open(os.path.join(d, "console.log"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def probes(self, pids=(20960,), exe=None, cmd=None):
        netstat = "\n".join("  TCP    0.0.0.0:5775           0.0.0.0:0              LISTENING       %d" % p for p in pids)
        netstat += "\n  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4\n"
        cmd = cmd if cmd is not None else ('"%s" -profile "%s" -gproj "%s" -addonsDir "%s" -forceUpdate -maxFPS 60 '
                                           '-dcLaunchId %s' % (exe or self.exe, self.profile, self.gproj,
                                                              os.path.join(self.fs.game_root, "addons"), self.LID))
        return StubProbes(netstat, {"ProcessId": pids[0] if pids else 0, "ExecutablePath": exe or self.exe,
                                    "CommandLine": cmd})

    def shake(self, probes, ident=None):
        return eh.handshake(self.src, ident or self.ident, 5775, probes, {})

    def test_the_pins_workbench_passes(self):
        res = self.shake(self.probes())
        self.assertTrue(res["ok"], res["reasons"])
        self.assertEqual((res["facts"]["pid"], res["facts"]["launch_id"], res["facts"]["build_version"]),
                         (20960, self.LID, "1.8.0.13"))
        self.assertIn("not compared", res["facts"]["build_version_note"])

    def refused(self, probes, text, ident=None):
        res = self.shake(probes, ident)
        self.assertFalse(res["ok"])
        self.assertTrue(any(text in r for r in res["reasons"]), res["reasons"])
        return res

    def test_no_listener_and_two_listeners(self):
        self.refused(self.probes(pids=()), "no process listens on port 5775")
        self.refused(self.probes(pids=(1, 2)), "2 processes listen")
        self.refused(StubProbes(OSError("no interop"), {}), "cannot list listening ports")

    def test_another_exe_path(self):
        other = os.path.join(self.tmp, "other", WB_REL)
        os.makedirs(os.path.dirname(other))
        shutil.copyfile(self.exe, other)
        self.refused(self.probes(exe=other), "is not the pin's Workbench")

    def test_another_exe_hash(self):
        base.FakeSteam.write(self.fs.tools_root, WB_REL, b"MZ wc", base.TOOLS_MTIME_NS)
        self.refused(self.probes(), "exe sha256")

    def test_no_exe_hash_without_sidecar(self):
        ident = dict(self.ident, exe_sha256=None)
        self.refused(self.probes(), "no Workbench exe hash", ident)

    def test_another_profile_or_project(self):
        cmd = '"%s" -profile "D:\\OneDrive\\ArmaReforgerWorkbench" -gproj "%s" -dcLaunchId %s' % (
            self.exe, self.gproj, self.LID)
        self.refused(self.probes(cmd=cmd), "-profile is")
        cmd = '"%s" -profile "%s" -gproj "C:\\DataTest\\addon.gproj" -dcLaunchId %s' % (self.exe, self.profile, self.LID)
        self.refused(self.probes(cmd=cmd), "-gproj is")

    def test_started_by_hand(self):
        cmd = '"%s" -profile "%s" -gproj "%s"' % (self.exe, self.profile, self.gproj)
        self.refused(self.probes(cmd=cmd), "no valid -dcLaunchId")

    def test_launch_record_of_another_pin(self):
        self.write_record(pin={"pin_id": "1.9.0.37", "sha256": "0" * 64})
        self.refused(self.probes(), "was made for pin 1.9.0.37")
        self.write_record(build_key="e" * 64)
        self.refused(self.probes(), "has build_key eeee")
        os.unlink(os.path.join(self.pin_dir, "launches", "launch_%s.json" % self.LID))
        self.refused(self.probes(), "has no readable record")

    def test_build_version_against_the_engine_sidecar(self):
        ident = dict(self.ident, engine_version_expected="1.9.0.37")
        self.refused(self.probes(), "reports version 1.8.0.13, the pin expects 1.9.0.37", ident)
        ident = dict(self.ident, engine_version_expected="1.8.0.13")
        self.assertTrue(self.shake(self.probes(), ident)["ok"])

    def test_console_log(self):
        shutil.rmtree(os.path.join(self.profile, "logs"))
        self.refused(self.probes(), "no console.log")
        self.write_log(version=None)
        self.refused(self.probes(), "still starting")
        self.write_log(lid="20261101T000000Z-wb-0000", name="logs_2026-11-06_12-45-17")
        self.refused(self.probes(), "no console.log")
        self.write_log(gproj=os.path.join(self.tmp, "other.gproj"))
        self.refused(self.probes(), "names another -gproj")

    def test_cli_main_reads_only(self):
        before = sorted(os.listdir(self.pin_dir))
        rc, out, _ = quiet(eh.main, ["--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir], probes=self.probes())
        self.assertEqual(rc, 0, out)
        self.assertIn("HANDSHAKE OK (pin 1.8.0.13)", out)
        rc, out, _ = quiet(eh.main, ["--pin-id", "1.8.0.13", "--pin-dir", self.pin_dir], probes=self.probes(pids=()))
        self.assertEqual(rc, 1)
        self.assertEqual(sorted(os.listdir(self.pin_dir)), before)

    def test_cmdline_parsing(self):
        argv = eh.split_cmdline('"C:\\Program Files (x86)\\x.exe" -profile "E:\\a b" -forceUpdate -maxFPS 60 -dcLaunchId X')
        self.assertEqual(argv[0], "C:\\Program Files (x86)\\x.exe")
        self.assertEqual(eh.cli_args(argv[1:]), {"profile": "E:\\a b", "forceupdate": "", "maxfps": "60",
                                                 "dclaunchid": "X"})
        self.assertEqual(eh.listening_pids("  TCP    0.0.0.0:57750   0.0.0.0:0   LISTENING   9\n"
                                           "  TCP    [::]:5775       [::]:0      LISTENING   7\n", 5775), {7})


# --------------------------------------------------------------------------- the stdio proxy


STUB_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"echo": msg.get("method"),
                                     "name": (msg.get("params") or {}).get("name")}}) + "\n")
        sys.stdout.flush()
'''


class ProxyTests(unittest.TestCase):
    def run_proxy(self, msgs, handshake_ok=True):
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
                self.call(2, "api_search", {"query": "VehicleWheelEffectComponent"}),
                self.call(3, "wb_state"),
                self.call(4, "wb_entity_create", {"name": "x"})]
        replies, calls = self.run_proxy(msgs)
        self.assertEqual(replies[1]["result"]["echo"], "initialize")
        self.assertEqual(replies[2]["result"]["name"], "api_search")
        self.assertEqual(replies[3]["result"]["name"], "wb_state")
        self.assertEqual(replies[4]["result"]["name"], "wb_entity_create")
        self.assertEqual(len(calls), 2)  # every wb_* call, none for api_search

    def test_refused_handshake_is_never_forwarded(self):
        replies, calls = self.run_proxy([self.call(3, "wb_state"), self.call(5, "api_search")], handshake_ok=False)
        self.assertTrue(replies[3]["result"]["isError"])
        self.assertIn("stub endpoint mismatch", replies[3]["result"]["content"][0]["text"])
        self.assertNotIn("echo", replies[3]["result"])
        self.assertEqual(replies[5]["result"]["name"], "api_search")

    def test_denied_and_sealed(self):
        replies, calls = self.run_proxy([self.call(1, "wb_launch"), self.call(2, "wb_cleanup", {"all": True}),
                                         self.call(3, "mod_build"),
                                         self.call(4, "game_read", {"path": "worlds/Eden/Eden.ent"}),
                                         self.call(5, "wb_open_resource", {"path": "Everon_route.json"}),
                                         self.call(6, "api_search", {"query": "credentials"})])
        for i in (1, 2, 3, 4, 5):
            self.assertTrue(replies[i]["result"]["isError"], i)
        self.assertIn("sealed", replies[4]["result"]["content"][0]["text"])
        self.assertEqual(replies[6]["result"]["name"], "api_search")
        self.assertEqual(calls, [])

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
        # The wrapper accepts a work root only on a Windows drive other than C:, so this runs only with
        # DD_PIN_TEST_TMP on such a drive.
        if not re.match(r"^/mnt/[abd-z]/", os.path.realpath(self.tmp)):
            self.skipTest("temp dir is not on a /mnt/<d..z> drive")
        wrapper = self.WRAPPER
        dist = os.path.join(self.root, "build", "enfusion-mcp-BK", "dist")
        os.makedirs(os.path.join(dist, "workbench"))
        os.makedirs(os.path.join(dist, "tools"))
        node = os.path.join(self.tmp, "node")
        with open(node, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(node, 0o755)
        env = {"PATH": "/usr/bin:/bin", "EMCP_ROOT": self.root, "NODE_BIN": node, "ENFUSION_WORKBENCH_HOST": "127.0.0.1",
               "EMCP_PIN_DIR": self.pin_dir, "EMCP_PIN": "1.8.0.13", "HOME": self.tmp,
               "DD_PIN_TOOLS": os.path.join(REPO, "tools", "pin")}
        r = subprocess.run(["bash", wrapper, "--print-env"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("REFUSED", r.stderr)          # no index yet; the real DEPLOY_ROOTS is not the temp one
        self.assertIn("has no API index", r.stderr)
        self.assertEqual(r.stdout, "")              # stdout belongs to JSON-RPC


if __name__ == "__main__":
    unittest.main()
