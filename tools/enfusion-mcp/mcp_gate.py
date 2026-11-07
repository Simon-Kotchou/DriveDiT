#!/usr/bin/env python3
"""Per-pin MCP gate: pick the pin's API index and project root, then proxy the MCP stdio session and run the
endpoint handshake before Workbench calls (ticket DD-14a, CC-05; MIGRATION_1_9 5.3 last row, 6.3, 8.2 component 6).

enfusion-mcp-wsl.sh execs this instead of node. Before the server starts:
  - the pin: EMCP_PIN (a pin id, for a second wrapper config) or <pin dir>/ACTIVE (EMCP_PIN_DIR, default
    /mnt/e/DriveDiT/pin); loaded and validated by tools/pin (its sidecars included). EMCP_PIN may name a stable pin
    only if it is the active one (the stable project root and installs are the active pin's); a 1.8 config after the
    re-pin is the cold pin, EMCP_PIN=1.8.0.13-cold;
  - the index: <EMCP_ROOT>/pins/<pin id>/data, whose api/INDEX_PIN.json must name this pin id and pin sha256 (and its
    build_key, when both are known), with every api file matching its recorded hash. An ENFUSION_MCP_DATA_DIR set by
    the caller must be that directory. There is no fallback: the 1.8 scrape never answers for a 1.9 pin;
  - the project root: pinlib.DEPLOY_ROOTS[launch path] (E:\\DriveDiT\\addons for stable, addons_c18 for the cold
    pin). An ENFUSION_PROJECT_PATH set by the caller must be it, and a deploy there stamped for another pin
    (.deploy_state.json target_pin) refuses;
  - ENFUSION_GAME_PATH and ENFUSION_WORKBENCH_PATH are the pin's installs.
Any refusal exits 2 before the server starts, with the reason on stderr.

Also before the server starts: ENFUSION_WORKBENCH_HOST must be loopback or the WSL default gateway (this machine's
Windows host), and the index stamp's source must be the dump tag and commit index_sources.json maps the pin to, with
the tag equal to the pin's engine_version_expected (api_index.verify_source).

During the session it forwards every JSON-RPC line both ways, except tools/call requests it refuses itself (an MCP
error result, the request is never forwarded). Each call is classified by bk_tools.py (the BK tools and actions at the
pinned commit):
  - deny: wb_launch, wb_cleanup, mod build, always (README "Rules");
  - unknown tools are refused;
  - the sealed-map guard on every call: an identifier token eden or everon, or the Everon root world GUID;
  - wb_read: endpoint_handshake.handshake() must pass right before the call;
  - write and wb_write: the handshake, and every destination argument (projectPath, outputPath, path, ...) must
    resolve inside this pin's project root;
  - read: forwarded as is.
The handshake runs before every gated call (an exe hash is cached per path, size and mtime), so a Workbench restarted
by hand in the middle of a session is caught at the next call.

    mcp_gate.py --print-config            resolve and print the config, start nothing
    mcp_gate.py -- NODE DIST [ARGS...]    what the wrapper runs

stdout is the MCP channel: diagnostics go to stderr. Exit codes: the server's; 2 config refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
API_FILES = ("enfusion-classes.json", "arma-classes.json", "groups.json", "hierarchy.json")


def _load(name, path):
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eh = _load("dd_emcp_endpoint_handshake", os.path.join(HERE, "endpoint_handshake.py"))
bk = _load("dd_emcp_bk_tools", os.path.join(HERE, "bk_tools.py"))
api_index = _load("dd_emcp_api_index", os.path.join(HERE, "api_index.py"))
pinlib, preflight = eh.pinlib, eh.preflight


class GateRefusal(Exception):
    pass


def resolve_config(env: dict) -> dict:
    """The pin, data dir, project root and install paths for this session; GateRefusal on any mismatch."""
    root = os.path.realpath(env.get("EMCP_ROOT") or "/mnt/e/drivedit_realign/mcp")
    pin_dir = os.path.abspath(pinlib.win_to_wsl(env.get("EMCP_PIN_DIR") or pinlib.DEFAULT_PIN_DIR))
    try:
        if env.get("EMCP_PIN"):
            src = preflight.load_pin_by_id(pin_dir, pinlib.validate_pin_id(env["EMCP_PIN"]))
            how = "EMCP_PIN"
        else:
            src = preflight.resolve_pin(None, True, pin_dir)
            how = "ACTIVE"
        ident = preflight.pin_identity(src)
    except pinlib.PinError as exc:
        raise GateRefusal("pin: %s" % exc)
    if ident["channel"] == "experimental" and how == "ACTIVE":
        raise GateRefusal("ACTIVE names an experimental pin")
    lp = ident["launch_path_id"]
    if lp == pinlib.STABLE_LAUNCH_PATH and how == "EMCP_PIN":
        # The stable project root and the stable installs belong to the active pin: a config for another stable pin
        # (1.8.0.13 after the re-pin) would let generators write 1.8 code into the 1.9 addon.
        try:
            active = pinlib.read_active(pin_dir)
        except pinlib.PinError as exc:
            raise GateRefusal("EMCP_PIN=%s is a stable pin, and the active pin cannot be read: %s" % (src["pin_id"], exc))
        if active != src["pin_id"]:
            raise GateRefusal("EMCP_PIN=%s is a stable pin, but ACTIVE is %s: the stable project root and installs "
                              "belong to the active pin (a 1.8 config runs as the cold pin, EMCP_PIN=<id>-cold)"
                              % (src["pin_id"], active))
    data_dir = os.path.join(root, "pins", src["pin_id"], "data")
    if env.get("ENFUSION_MCP_DATA_DIR") and os.path.realpath(env["ENFUSION_MCP_DATA_DIR"]) != os.path.realpath(data_dir):
        raise GateRefusal("ENFUSION_MCP_DATA_DIR=%s is not pin %s's index %s"
                          % (env["ENFUSION_MCP_DATA_DIR"], src["pin_id"], data_dir))
    stamp_path = os.path.join(data_dir, "api", "INDEX_PIN.json")
    try:
        with open(stamp_path, encoding="utf-8") as fh:
            stamp = json.load(fh)
    except (OSError, ValueError) as exc:
        raise GateRefusal("pin %s has no API index (%s: %s); build it with api_index.py build --pin-id %s"
                          % (src["pin_id"], stamp_path, exc, src["pin_id"]))
    if stamp.get("schema") != "dd-mcp-index/1" or stamp.get("pin_id") != src["pin_id"] \
            or stamp.get("pin_sha256") != src["sha256"]:
        raise GateRefusal("index %s is for pin %s (sha256 %s), not %s (sha256 %s)"
                          % (data_dir, stamp.get("pin_id"), str(stamp.get("pin_sha256"))[:16], src["pin_id"],
                             src["sha256"][:16]))
    if stamp.get("build_key") and ident["build_key"] and stamp["build_key"] != ident["build_key"]:
        raise GateRefusal("index %s has build_key %s, the pin %s" % (data_dir, stamp["build_key"][:16],
                                                                    ident["build_key"][:16]))
    try:
        api_index.verify_source(src["pin_id"], ident, stamp.get("source"), api_index.load_sources(
            env.get("EMCP_INDEX_SOURCES")))
    except ValueError as exc:
        raise GateRefusal("index %s: %s" % (data_dir, exc))
    for name in API_FILES:
        path = os.path.join(data_dir, "api", name)
        try:
            with open(path, "rb") as fh:
                have = hashlib.sha256(fh.read()).hexdigest()
        except OSError as exc:
            raise GateRefusal("cannot read %s: %s" % (path, exc))
        if have != (stamp.get("files") or {}).get(name):
            raise GateRefusal("%s does not match its INDEX_PIN.json hash" % path)
    project = pinlib.DEPLOY_ROOTS.get(lp)
    if project is None:
        raise GateRefusal("launch path %s has no deploy root (pinlib.DEPLOY_ROOTS)" % lp)
    if env.get("ENFUSION_PROJECT_PATH") and os.path.realpath(env["ENFUSION_PROJECT_PATH"]) != os.path.realpath(project):
        raise GateRefusal("ENFUSION_PROJECT_PATH=%s is not pin %s's project root %s"
                          % (env["ENFUSION_PROJECT_PATH"], src["pin_id"], project))
    state_path = os.path.join(project, "DataCapture", ".deploy_state.json")
    try:
        with open(state_path, encoding="utf-8") as fh:
            target_pin = json.load(fh).get("target_pin")
    except (OSError, ValueError):
        target_pin = None
    if target_pin and target_pin != src["pin_id"]:
        raise GateRefusal("project root %s was deployed for pin %s, not %s" % (project, target_pin, src["pin_id"]))
    gateway = eh.default_gateway(env.get("EMCP_TEST_ROUTE")) if env.get("EMCP_TEST_ROUTE") else eh.default_gateway()
    host = env.get("ENFUSION_WORKBENCH_HOST") or "127.0.0.1"
    why = eh.endpoint_host_reason(host, gateway)
    if why:
        raise GateRefusal(why)
    try:
        port = int(env.get("ENFUSION_WORKBENCH_PORT") or eh.DEFAULT_PORT)
    except ValueError:
        raise GateRefusal("ENFUSION_WORKBENCH_PORT=%r is not a port" % env.get("ENFUSION_WORKBENCH_PORT"))
    return {"pin_src": src, "ident": ident, "resolved_by": how, "data_dir": data_dir, "project": project,
            "game_path": src["pin"]["game"]["install"]["wsl"], "workbench_path": src["pin"]["tools"]["install"]["wsl"],
            "index_stamp": stamp_path, "host": host, "gateway": gateway, "port": port,
            "server_cwd": os.path.join(root, "cwd")}


def server_env(env: dict, cfg: dict) -> dict:
    out = dict(env)
    out.update(ENFUSION_MCP_DATA_DIR=cfg["data_dir"], ENFUSION_PROJECT_PATH=cfg["project"],
               ENFUSION_GAME_PATH=cfg["game_path"], ENFUSION_WORKBENCH_PATH=cfg["workbench_path"],
               ENFUSION_WORKBENCH_HOST=cfg["host"], ENFUSION_WORKBENCH_PORT=str(cfg["port"]),
               EMCP_PIN_RESOLVED=cfg["pin_src"]["pin_id"])
    return out


class Gate:
    """The tools/call policy. handshake_fn() -> {ok, reasons, facts}; project_root and server_cwd for the destination
    check (bk_tools.destination_reason)."""

    def __init__(self, handshake_fn, log=None, project_root=None, server_cwd=None):
        self.handshake_fn = handshake_fn
        self.log = log or (lambda msg: print("[mcp_gate] %s" % msg, file=sys.stderr, flush=True))
        self.project_root = project_root
        self.server_cwd = server_cwd or "/nonexistent-server-cwd"
        self.handshakes = 0

    def decide(self, name: str, arguments) -> str:
        """None to forward, else the refusal text."""
        kind, why = bk.classify(name, arguments)
        if kind in ("deny", "unknown"):
            return "%s is never allowed from this wrapper: %s" % (name, why)
        sealed = bk.sealed_reason(arguments if arguments is not None else {})
        if sealed:
            return sealed + "; the sealed-map guard refuses the call"
        if kind == "read":
            return None
        if kind in ("write", "wb_write"):
            if self.project_root is None:
                return "no project root is configured for writes"
            dest = bk.destination_reason(name, arguments, self.project_root, self.server_cwd, pinlib.win_to_wsl)
            if dest:
                return dest
        self.handshakes += 1
        res = self.handshake_fn()
        if not res.get("ok"):
            return "endpoint handshake refused (MIGRATION_1_9 6.3), nothing was forwarded: " + \
                "; ".join(res.get("reasons") or ["unknown"])
        self.log("handshake ok for %s (%s): pid %s, launch %s" % (name, kind, res["facts"].get("pid"),
                                                                 res["facts"].get("launch_id")))
        return None

    @staticmethod
    def refusal(req_id, text: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": "REFUSED by the DriveDiT MCP gate: " + text}],
                           "isError": True}}

    def filter_line(self, line: bytes):
        """(bytes to forward or None, [responses to send back])."""
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return line, []
        msgs = msg if isinstance(msg, list) else [msg]
        replies, keep = [], []
        for m in msgs:
            if isinstance(m, dict) and m.get("method") == "tools/call":
                params = m.get("params") if isinstance(m.get("params"), dict) else {}
                why = self.decide(str(params.get("name", "")), params.get("arguments"))
                if why is not None:
                    self.log("refused %s: %s" % (params.get("name"), why))
                    if "id" in m:
                        replies.append(self.refusal(m["id"], why))
                    continue
            keep.append(m)
        if not replies:
            return line, []
        if not keep:
            return None, replies
        out = keep if isinstance(msg, list) else keep[0]
        return (json.dumps(out) + "\n").encode("utf-8"), replies


def proxy(gate: Gate, cmd: list, env: dict, cwd=None, stdin=None, stdout=None) -> int:
    """Run cmd and relay newline-delimited JSON-RPC between our stdio and its stdio through gate."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=cwd)
    lock = threading.Lock()

    def send(data: bytes):
        with lock:
            stdout.write(data)
            stdout.flush()

    def pump_out():
        for out_line in iter(child.stdout.readline, b""):
            send(out_line)

    t = threading.Thread(target=pump_out, daemon=True)
    t.start()
    try:
        for line in iter(stdin.readline, b""):
            fwd, replies = gate.filter_line(line)
            for r in replies:
                send((json.dumps(r) + "\n").encode("utf-8"))
            if fwd is not None:
                try:
                    child.stdin.write(fwd if fwd.endswith(b"\n") else fwd + b"\n")
                    child.stdin.flush()
                except BrokenPipeError:
                    break
    finally:
        try:
            child.stdin.close()
        except OSError:
            pass
        rc = child.wait()
        t.join(timeout=5)
        child.stdout.close()
    return rc


def main(argv=None, env=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    env = dict(os.environ if env is None else env)
    try:
        cfg = resolve_config(env)
    except GateRefusal as exc:
        print("[mcp_gate] REFUSED: %s; the MCP server was not started" % exc, file=sys.stderr)
        return 2
    lines = ["pin %s (%s, %s, launch path %s, build_key %s)" % (
        cfg["pin_src"]["pin_id"], cfg["resolved_by"], cfg["ident"]["channel"], cfg["ident"]["launch_path_id"],
        cfg["ident"]["build_key"]), "index %s" % cfg["data_dir"], "project %s" % cfg["project"],
        "game %s" % cfg["game_path"], "workbench %s" % cfg["workbench_path"],
        "endpoint %s:%d (gateway %s)" % (cfg["host"], cfg["port"], cfg["gateway"]),
        "handshake before every wb_* and write call; destinations inside %s; denied %s; sealed-map guard on"
        % (cfg["project"], ", ".join(sorted(bk.DENY)))]
    for ln in lines:
        print("[mcp_gate] %s" % ln, file=sys.stderr)
    if args.print_config:
        return 0
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        print("[mcp_gate] no server command given (mcp_gate.py -- NODE DIST)", file=sys.stderr)
        return 2
    cache = {}
    gate = Gate(lambda: eh.handshake(cfg["pin_src"], cfg["ident"], cfg["port"], None, cache, cfg["host"],
                                     cfg["gateway"]), project_root=cfg["project"], server_cwd=cfg["server_cwd"])
    return proxy(gate, cmd, server_env(env, cfg))


if __name__ == "__main__":
    sys.exit(main())
