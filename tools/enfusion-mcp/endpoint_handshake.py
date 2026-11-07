#!/usr/bin/env python3
"""MCP endpoint handshake: prove which Workbench answers on the NET API port before the MCP touches it (ticket DD-14a,
CC-05; MIGRATION_1_9 6.3, CRIT F-13).

The wrapper's index and project checks do not prove which process answers on the shared port 5775, and BK MCP
auto-installs its handlers into the first addon on its first wb_* call. So mcp_gate.py runs this before the first
wb_* call of a session and before every other wb_* call. Every step is read-only and writes nothing:

  1. From Windows: the PID listening on the port (netstat -ano), its exe path and command line (Win32_Process through
     powershell.exe), and the sha256 of that exe. The exe must be the pin's Workbench exe (its Tools install, the
     launch path's Workbench) with the pin's exe_sha256; -profile and -gproj must be the launch path's (pinlib
     WB_LAUNCH), and the command line must carry -dcLaunchId.
  2. The launch: launch_<id>.json of that -dcLaunchId under <pin dir>/launches must be an ok wb record of this pin
     file (pin id and sha256), with this build_key and launch path when it records them. The build: that launch's own
     console.log (the -profile's logs/logs_*/ whose "CLI Params" line names the launch id) must report the pin's
     engine_version_expected in "Creating game instance(...), version X" once the pin has an engine sidecar (until
     then the version is recorded, not compared), and its CLI line must name the same -gproj.

  Any mismatch refuses, with the reasons. 6.3 step 2 asks for a read-only handler call for GetBuildVersion(); no
  deployed handler answers it (BK has none, and a DataCapture handler is DD-14b's Enforce code), so the build is read
  from the launch's console.log, which the same process writes. A handler check can be added in front later.

    python3 -I -B endpoint_handshake.py --pin-active [--pin-dir DIR] [--port 5775] [--json]
    python3 -I -B endpoint_handshake.py --pin-id 1.8.0.13-cold

Exit codes: 0 the endpoint is the pin's Workbench; 1 refused; 2 config error. Tests replace the Windows probes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys

NETSTAT = "/mnt/c/Windows/System32/NETSTAT.EXE"
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WB_EXE_REL = "Workbench/ArmaReforgerWorkbenchSteamDiag.exe"
DEFAULT_PORT = 5775
LOG_DIRS_SCANNED = 30
GAME_INSTANCE_RE = re.compile(r"Creating game instance\([^)]*\), version (\S+?)(?: built (.*?))?\.?\s*$")


def load_pin_tools():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("DD_PIN_TOOLS"), os.path.join(os.path.dirname(here), "pin"), "/mnt/e/DriveDiT/tools/pin"):
        if cand and os.path.isfile(os.path.join(cand, "pinlib.py")):
            mods = []
            for name in ("pinlib", "preflight"):
                path = os.path.join(cand, name + ".py")
                modname = "dd_pin_" + name
                mod = sys.modules.get(modname)
                if mod is None or getattr(mod, "__file__", None) != path:
                    spec = importlib.util.spec_from_file_location(modname, path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[modname] = mod
                    spec.loader.exec_module(mod)
                mods.append(mod)
            return tuple(mods)
    raise SystemExit("endpoint_handshake: cannot find tools/pin (set DD_PIN_TOOLS)")


pinlib, preflight = load_pin_tools()


# --------------------------------------------------------------------------- Windows probes (read-only)


class WindowsProbes:
    """netstat.exe and powershell.exe from WSL. Both only read; neither writes a file (-NoProfile)."""

    def __init__(self, netstat=NETSTAT, powershell=POWERSHELL, timeout=30):
        self.netstat, self.powershell, self.timeout = netstat, powershell, timeout

    def netstat_text(self) -> str:
        return subprocess.run([self.netstat, "-ano"], capture_output=True, timeout=self.timeout,
                              check=True).stdout.decode("utf-8", errors="replace")

    def process_json(self, pid: int) -> str:
        cmd = ("Get-CimInstance Win32_Process -Filter 'ProcessId=%d' | "
               "Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress" % int(pid))
        return subprocess.run([self.powershell, "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True,
                              timeout=self.timeout, check=True).stdout.decode("utf-8", errors="replace")


def listening_pids(netstat_text: str, port: int) -> set:
    pids = set()
    for line in netstat_text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING" \
                and parts[1].rsplit(":", 1)[-1] == str(port) and parts[4].isdigit():
            pids.add(int(parts[4]))
    return pids


def split_cmdline(cmd: str) -> list:
    """Windows command-line splitting, enough for the launchers' lines (double quotes; no escaped quotes)."""
    out, cur, quoted, have = [], [], False, False
    for ch in cmd or "":
        if ch == '"':
            quoted, have = not quoted, True
        elif ch in " \t" and not quoted:
            if have:
                out.append("".join(cur))
            cur, have = [], False
        else:
            cur.append(ch)
            have = True
    if have:
        out.append("".join(cur))
    return out


def cli_args(argv: list) -> dict:
    out = {}
    for i, a in enumerate(argv):
        if a.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            out.setdefault(a[1:].lower(), argv[i + 1])
        elif a.startswith("-"):
            out.setdefault(a[1:].lower(), "")
    return out


def win_norm(path: str) -> str:
    return (path or "").replace("/", "\\").rstrip("\\").lower()


# --------------------------------------------------------------------------- the handshake


def find_console_log(profile_wsl: str, launch_id: str):
    """(path, cli line, build version, build time) of the console.log under <profile>/logs/logs_* whose CLI Params
    line names -dcLaunchId <launch_id>; the newest LOG_DIRS_SCANNED log dirs are scanned. (None, ...) if none."""
    logs = os.path.join(profile_wsl, "logs")
    try:
        dirs = sorted((d for d in os.listdir(logs) if d.startswith("logs_")), reverse=True)[:LOG_DIRS_SCANNED]
    except OSError:
        return None, None, None, None
    want = "-dcLaunchId %s" % launch_id
    for d in dirs:
        path = os.path.join(logs, d, "console.log")
        try:
            with open(path, "rb") as fh:
                head = fh.read(4 << 20).decode("utf-8", errors="replace")
        except OSError:
            continue
        cli = next((ln for ln in head.splitlines()[:200] if "CLI Params:" in ln), None)
        if cli is None or not re.search(r"%s(\s|$)" % re.escape(want), cli):
            continue
        version = built = None
        for ln in head.splitlines():
            m = GAME_INSTANCE_RE.search(ln)
            if m:
                version, built = m.group(1), m.group(2)
                break
        return path, cli, version, built
    return None, None, None, None


def handshake(pin_src: dict, ident: dict, port: int = DEFAULT_PORT, probes=None, exe_hash_cache=None) -> dict:
    """{ok, reasons, facts} for the endpoint on port against the pin. Reads only."""
    probes = probes or WindowsProbes()
    reasons, facts = [], {"pin_id": pin_src["pin_id"], "port": port}
    lp = ident["launch_path_id"]
    wb_launch = pinlib.WB_LAUNCH.get(lp)
    if wb_launch is None:
        return {"ok": False, "reasons": ["launch path %r has no Workbench launcher (pinlib.WB_LAUNCH)" % lp],
                "facts": facts}
    tools_install = pin_src["pin"]["tools"]["install"]["wsl"]
    want_exe_wsl = os.path.join(tools_install, *WB_EXE_REL.split("/"))
    want_exe = pinlib.wsl_to_win(want_exe_wsl) or want_exe_wsl
    want_sha = ((ident.get("exe_sha256") or {}).get("tools") or {}).get(WB_EXE_REL)
    # 1. who listens
    try:
        pids = listening_pids(probes.netstat_text(), port)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reasons": ["cannot list listening ports (netstat.exe): %s" % exc], "facts": facts}
    facts["pids"] = sorted(pids)
    if len(pids) != 1:
        return {"ok": False, "facts": facts, "reasons": [
            "no process listens on port %d (Workbench closed or NET API off)" % port if not pids else
            "%d processes listen on port %d (%s); only one Workbench may run" % (len(pids), port, sorted(pids))]}
    pid = pids.pop()
    try:
        info = json.loads(probes.process_json(pid) or "null")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"ok": False, "reasons": ["cannot read process %d (powershell.exe): %s" % (pid, exc)], "facts": facts}
    if isinstance(info, list):
        info = info[0] if len(info) == 1 else None
    if not isinstance(info, dict) or info.get("ProcessId") != pid:
        return {"ok": False, "reasons": ["process %d is gone or unreadable" % pid], "facts": facts}
    exe, cmd = info.get("ExecutablePath") or "", info.get("CommandLine") or ""
    args = cli_args(split_cmdline(cmd)[1:])
    facts.update(pid=pid, exe=exe, profile=args.get("profile"), gproj=args.get("gproj"),
                 launch_id=args.get("dclaunchid"))
    # Compared in WSL form, case-insensitively (NTFS): the process reports a Windows path.
    if not exe or pinlib.win_to_wsl(exe).lower() != want_exe_wsl.lower():
        reasons.append("exe %s is not the pin's Workbench %s" % (exe or "(unknown)", want_exe))
    if not want_sha:
        reasons.append("pin %s has no Workbench exe hash (dd-pin/1 without its build_key sidecar: run "
                       "tools/pin/pin_sidecar.py build-key)" % pin_src["pin_id"])
    elif exe:
        exe_wsl = pinlib.win_to_wsl(exe)
        try:
            st = os.stat(exe_wsl)
            key = (exe_wsl, st.st_size, st.st_mtime_ns)
            have = (exe_hash_cache or {}).get(key)
            if have is None:
                have = pinlib.sha256_file(exe_wsl)
                if exe_hash_cache is not None:
                    exe_hash_cache[key] = have
        except OSError as exc:
            have = None
            reasons.append("cannot hash %s: %s" % (exe_wsl, exc))
        facts["exe_sha256"] = have
        if have is not None and have != want_sha:
            reasons.append("exe sha256 %s is not the pin's %s" % (have[:16], want_sha[:16]))
    for key in ("profile", "gproj"):
        if win_norm(args.get(key)) != win_norm(wb_launch[key]):
            reasons.append("-%s is %r, the launch path %s uses %r (started by hand or by another launcher?)"
                           % (key, args.get(key), lp, wb_launch[key]))
    lid = args.get("dclaunchid")
    if not lid or not re.fullmatch(r"[0-9A-Za-z-]{20,64}", lid):
        reasons.append("the command line has no valid -dcLaunchId (not started by %s)" % wb_launch["launcher"])
        return {"ok": False, "reasons": reasons, "facts": facts}
    # 2. the launch record and the launch's own console.log
    rec_path = os.path.join(pin_src["pin_dir"], "launches", "launch_%s.json" % lid)
    try:
        rec = json.loads(pinlib.read_regular_file(rec_path, "launch record", 1 << 22))
    except (pinlib.PinError, ValueError) as exc:
        reasons.append("launch %s has no readable record: %s" % (lid, exc))
        rec = None
    if isinstance(rec, dict):
        pin = rec.get("pin") if isinstance(rec.get("pin"), dict) else {}
        if (rec.get("status"), rec.get("exe"), rec.get("launch_id")) != ("ok", "wb", lid):
            reasons.append("launch record %s is not an ok wb record of %s" % (rec_path, lid))
        if (pin.get("pin_id"), pin.get("sha256")) != (pin_src["pin_id"], pin_src["sha256"]):
            reasons.append("launch %s was made for pin %s (sha256 %s), not %s (sha256 %s)"
                           % (lid, pin.get("pin_id"), str(pin.get("sha256"))[:16], pin_src["pin_id"],
                              pin_src["sha256"][:16]))
        if rec.get("build_key") and ident.get("build_key") and rec["build_key"] != ident["build_key"]:
            reasons.append("launch %s has build_key %s, the pin %s" % (lid, rec["build_key"][:16],
                                                                      ident["build_key"][:16]))
        if rec.get("launch_path_id") and rec["launch_path_id"] != lp:
            reasons.append("launch %s used launch path %s, the pin %s" % (lid, rec["launch_path_id"], lp))
    log, cli, version, built = find_console_log(pinlib.win_to_wsl(args.get("profile") or ""), lid)
    facts.update(console_log=log, build_version=version, build_time=built)
    if log is None:
        reasons.append("no console.log under %s\\logs names launch %s" % (args.get("profile"), lid))
    else:
        if win_norm(cli_args(split_cmdline(cli.split("CLI Params:", 1)[1])).get("gproj")) != win_norm(args.get("gproj")):
            reasons.append("console.log %s names another -gproj than the process" % log)
        expected = ident.get("engine_version_expected")
        facts["engine_version_expected"] = expected
        if version is None:
            reasons.append("console.log %s has no 'Creating game instance ... version' line yet (Workbench still "
                           "starting?)" % log)
        elif expected and version != expected:
            reasons.append("Workbench reports version %s, the pin expects %s" % (version, expected))
        elif not expected:
            facts["build_version_note"] = "not compared: pin %s has no engine sidecar yet" % pin_src["pin_id"]
    return {"ok": not reasons, "reasons": reasons, "facts": facts}


def resolve(pin_id=None, pin_active=False, pin_dir=None) -> tuple:
    pin_dir = os.path.abspath(pinlib.win_to_wsl(pin_dir or pinlib.DEFAULT_PIN_DIR))
    if pin_active:
        src = preflight.resolve_pin(None, True, pin_dir)
    else:
        src = preflight.load_pin_by_id(pin_dir, pinlib.validate_pin_id(pin_id))
    return src, preflight.pin_identity(src)


def main(argv=None, probes=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--pin-id", default=None)
    which.add_argument("--pin-active", action="store_true")
    ap.add_argument("--pin-dir", default=None)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        src, ident = resolve(args.pin_id, args.pin_active, args.pin_dir)
    except pinlib.PinError as exc:
        print("endpoint_handshake: %s" % exc, file=sys.stderr)
        return 2
    res = handshake(src, ident, args.port, probes)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        for k, v in sorted(res["facts"].items()):
            print("  %-24s %s" % (k, v))
        for r in res["reasons"]:
            print("REFUSED: %s" % r)
        print("HANDSHAKE %s (pin %s)" % ("OK" if res["ok"] else "REFUSED", src["pin_id"]))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
