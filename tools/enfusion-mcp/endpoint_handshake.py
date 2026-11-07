#!/usr/bin/env python3
"""MCP endpoint handshake: prove which Workbench answers on the NET API endpoint before the MCP touches it (ticket
DD-14a, CC-05; MIGRATION_1_9 6.3, CRIT F-13).

The wrapper's index and project checks do not prove which process answers on the shared port 5775, and BK MCP
auto-installs its handlers into the first addon on its first wb_* call. So mcp_gate.py runs this before every wb_*
call and every mutating call. Every step is read-only and writes nothing; no PowerShell is used:

  0. The endpoint: BK connects to ENFUSION_WORKBENCH_HOST:PORT. The host must be loopback or the WSL NAT default
     gateway, which is this machine's Windows host (endpoint_host_reason); anything else is refused, so a check of
     the local Windows listener never vouches for another machine.
  1. The listener: netstat.exe -ano must show exactly one PID LISTENING on that port, on 0.0.0.0, [::] or the host's
     own address. Its process (cscript.exe running wb_proc.js, a WMI Win32_Process query; about 0.15 s, measured)
     gives the exe path, the creation time and the command line. The exe must be the pin's Workbench exe (its Tools
     install) with the pin's exe_sha256; -profile and -gproj must be the launch path's (pinlib.WB_LAUNCH), and the
     command line must carry -dcLaunchId.
  2. The launch: launch_<id>.json of that -dcLaunchId under <pin dir>/launches must be a JSON object with the
     dd-launch schema fields, an ok wb record of this pin file (pin id and sha256), with this build_key and launch
     path when it records them. The process must have started 0..300 s after the record was written.
  3. The log: exactly one console.log under the -profile's logs/ names the launch id after its record. Its start
     time must lie within -2..+120 s of the process creation time, its working directory must be the pin's Workbench
     dir, its CLI line the process's -gproj, and its "Creating game instance(...), version X" must equal the pin's
     engine_version_expected once the pin has an engine sidecar (until then the version is recorded, not compared).

  Any mismatch refuses, with the reasons. 6.3 step 2 asks for a read-only handler call for GetBuildVersion(); no
  deployed handler answers it (BK has none, and a DataCapture handler is DD-14b's Enforce code), so the build is read
  from the launch's own console.log, bound to the process by its creation time. Not detected (a documented
  limitation; the runbook forbids both): a project or world opened by hand inside a running Workbench after its
  start.

    python3 -I -B endpoint_handshake.py --pin-active [--pin-dir DIR] [--host H] [--port 5775] [--json]

Exit codes: 0 the endpoint is the pin's Workbench; 1 refused; 2 config error. Tests replace the Windows probes.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import ipaddress
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NETSTAT = "/mnt/c/Windows/System32/NETSTAT.EXE"
CSCRIPT = "/mnt/c/Windows/System32/cscript.exe"
HELPER = os.path.join(HERE, "wb_proc.js")
WB_EXE_REL = "Workbench/ArmaReforgerWorkbenchSteamDiag.exe"
DEFAULT_PORT = 5775
LAUNCH_SCHEMAS = ("dd-launch/1", "dd-launch/2")
START_AFTER_RECORD_S = (-5, 300)     # process creation relative to the record's utc (launcher: preflight, copy, start)
LOG_AFTER_START_S = (-2, 120)        # console.log header time relative to the process creation time
LOOPBACK = ("127.0.0.1", "::1", "localhost")


def load_pin_tools():
    for cand in (os.environ.get("DD_PIN_TOOLS"), os.path.join(os.path.dirname(HERE), "pin"), "/mnt/e/DriveDiT/tools/pin"):
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


# --------------------------------------------------------------------------- the endpoint host


def default_gateway(route_text=None):
    """The IPv4 default gateway of this WSL distro (/proc/net/route), the Windows host under WSL NAT; or None."""
    if route_text is None:
        try:
            with open("/proc/net/route", encoding="ascii", errors="replace") as fh:
                route_text = fh.read()
        except OSError:
            return None
    for line in route_text.splitlines()[1:]:
        f = line.split()
        if len(f) >= 8 and f[1] == "00000000" and f[7] == "00000000" and f[2] != "00000000":
            try:
                return str(ipaddress.IPv4Address(bytes.fromhex(f[2])[::-1]))
            except ValueError:
                return None
    return None


def endpoint_host_reason(host: str, gateway=None):
    """Why BK may not use this host, or None. Loopback, or the WSL NAT default gateway (the local Windows host)."""
    if host in LOOPBACK:
        return None
    if gateway is not None and host == gateway:
        return None
    return ("ENFUSION_WORKBENCH_HOST=%s is neither loopback nor this machine's Windows host (the WSL default gateway "
            "%s): the handshake can only vouch for the local Workbench" % (host, gateway))


# --------------------------------------------------------------------------- Windows probes (read-only)


class WindowsProbes:
    """netstat.exe and cscript.exe (wb_proc.js, WMI) from WSL. Both only read; neither starts PowerShell."""

    def __init__(self, netstat=NETSTAT, cscript=CSCRIPT, helper=HELPER, timeout=15):
        self.netstat, self.cscript, self.helper, self.timeout = netstat, cscript, helper, timeout

    def netstat_text(self) -> str:
        return subprocess.run([self.netstat, "-ano"], capture_output=True, timeout=self.timeout,
                              check=True).stdout.decode("utf-8", errors="replace")

    def process_line(self, pid: int) -> str:
        helper = os.path.realpath(self.helper)
        win = pinlib.wsl_to_win(helper)
        if not win or not helper.startswith("/mnt/e/"):
            raise OSError("the helper %s must lie on E: (it runs as a Windows process)" % self.helper)
        r = subprocess.run([self.cscript, "//NoLogo", win, str(int(pid))], capture_output=True, timeout=self.timeout,
                           cwd=os.path.dirname(helper))
        if r.returncode not in (0, 1):
            raise OSError("wb_proc.js exited %d: %s" % (r.returncode, r.stderr.decode(errors="replace").strip()))
        return r.stdout.decode("utf-8", errors="replace")


def listening_pids(netstat_text: str, port: int, host: str = "127.0.0.1") -> set:
    """PIDs LISTENING on port at an address that serves host: 0.0.0.0, [::], the host itself, or loopback for a
    loopback host."""
    serve = {"0.0.0.0", "[::]", host, "[%s]" % host}
    if host in LOOPBACK:
        serve |= {"127.0.0.1", "[::1]"}
    pids = set()
    for line in netstat_text.splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING" or not parts[4].isdigit():
            continue
        addr, _, p = parts[1].rpartition(":")
        if p == str(port) and addr in serve:
            pids.add(int(parts[4]))
    return pids


_WMI_TIME_RE = re.compile(r"^([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})\.[0-9]{6}([+-][0-9]{3})$")


def wmi_time_to_unix(text: str):
    """WMI CIM_DATETIME 'yyyymmddHHMMSS.ffffff+UUU' (local time, UTC offset in minutes) -> Unix seconds, or None."""
    m = _WMI_TIME_RE.match((text or "").strip())
    if not m:
        return None
    y, mo, d, h, mi, s, off = m.groups()
    tz = datetime.timezone(datetime.timedelta(minutes=int(off)))
    try:
        return int(datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=tz).timestamp())
    except ValueError:
        return None


def parse_process_line(text: str, pid: int):
    """{ProcessId, ExecutablePath, CreationUnix, CommandLine} from wb_proc.js output, or None."""
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) != 1:
        return None
    f = lines[0].split("\t", 3)
    if len(f) != 4 or not f[0].isdigit() or int(f[0]) != pid:
        return None
    return {"ProcessId": pid, "ExecutablePath": f[1], "CreationUnix": wmi_time_to_unix(f[2]), "CommandLine": f[3]}


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


def launch_record_problems(rec, lid: str, pin_src: dict, ident: dict, lp: str, exe: str = "wb") -> list:
    """Why a parsed launch record is not an ok record of this pin for launch lid ([] when it is)."""
    if not isinstance(rec, dict):
        return ["the launch record of %s is a JSON %s, not an object" % (lid, type(rec).__name__)]
    probs = []
    if rec.get("schema") not in LAUNCH_SCHEMAS:
        probs.append("launch record schema %r is not one of %s" % (rec.get("schema"), ", ".join(LAUNCH_SCHEMAS)))
    if (rec.get("status"), rec.get("exe"), rec.get("launch_id")) != ("ok", exe, lid):
        probs.append("launch record of %s is not an ok %s record of that id" % (lid, exe))
    if pinlib.iso_utc_to_unix(rec.get("utc")) is None:
        probs.append("launch record of %s has no utc time" % lid)
    pin = rec.get("pin")
    if not isinstance(pin, dict) or (pin.get("pin_id"), pin.get("sha256")) != (pin_src["pin_id"], pin_src["sha256"]):
        have = pin if isinstance(pin, dict) else {}
        probs.append("launch %s was made for pin %s (sha256 %s), not %s (sha256 %s)"
                     % (lid, have.get("pin_id"), str(have.get("sha256"))[:16], pin_src["pin_id"], pin_src["sha256"][:16]))
    if rec.get("build_key") and ident.get("build_key") and rec["build_key"] != ident["build_key"]:
        probs.append("launch %s has build_key %s, the pin %s" % (lid, str(rec["build_key"])[:16],
                                                                ident["build_key"][:16]))
    if rec.get("launch_path_id") and rec["launch_path_id"] != lp:
        probs.append("launch %s used launch path %s, the pin %s" % (lid, rec["launch_path_id"], lp))
    return probs


def check_log(log: dict, want_cwd: str, gproj, expected, created=None, reasons=None, facts=None) -> list:
    """Append why a launch's console.log does not show this pin's Workbench start; return reasons."""
    reasons = [] if reasons is None else reasons
    facts = {} if facts is None else facts
    facts.update(console_log=log["path"], build_version=log["version"], build_time=log["built"],
                 log_start_unix=log["start_unix"])
    if created is not None and (log["start_unix"] is None or not
                                (LOG_AFTER_START_S[0] <= log["start_unix"] - created <= LOG_AFTER_START_S[1])):
        reasons.append("console.log %s started %s s from the process creation (expected %d..%d s): not this "
                       "process's log" % (log["path"], "?" if log["start_unix"] is None else "%+d" % (
                           log["start_unix"] - created), *LOG_AFTER_START_S))
    if win_norm(log["cwd"]) != win_norm(want_cwd):
        reasons.append("console.log %s has working directory %r, the pin's Workbench is %r"
                       % (log["path"], log["cwd"], want_cwd))
    if gproj is not None and win_norm(cli_args(split_cmdline(log["cli"] or "")).get("gproj")) != win_norm(gproj):
        reasons.append("console.log %s names another -gproj than the process" % log["path"])
    facts["engine_version_expected"] = expected
    if log["version"] is None:
        reasons.append("console.log %s has no 'Creating game instance ... version' line yet (Workbench still "
                       "starting?)" % log["path"])
    elif expected and log["version"] != expected:
        reasons.append("Workbench reports version %s, the pin expects %s" % (log["version"], expected))
    elif not expected:
        facts["build_version_note"] = "not compared: the pin has no engine sidecar yet"
    return reasons


def handshake(pin_src: dict, ident: dict, port: int = DEFAULT_PORT, probes=None, exe_hash_cache=None,
              host: str = "127.0.0.1", gateway=None) -> dict:
    """{ok, reasons, facts} for the endpoint host:port against the pin. Reads only."""
    probes = probes or WindowsProbes()
    reasons, facts = [], {"pin_id": pin_src["pin_id"], "host": host, "port": port}
    why = endpoint_host_reason(host, default_gateway() if gateway is None else gateway)
    if why:
        return {"ok": False, "reasons": [why], "facts": facts}
    lp = ident["launch_path_id"]
    wb_launch = pinlib.WB_LAUNCH.get(lp)
    if wb_launch is None:
        return {"ok": False, "reasons": ["launch path %r has no Workbench launcher (pinlib.WB_LAUNCH)" % lp],
                "facts": facts}
    tools_install = pin_src["pin"]["tools"]["install"]["wsl"]
    want_exe_wsl = os.path.join(tools_install, *WB_EXE_REL.split("/"))
    want_exe = pinlib.wsl_to_win(want_exe_wsl) or want_exe_wsl
    want_cwd = pinlib.wsl_to_win(os.path.join(tools_install, "Workbench")) or os.path.join(tools_install, "Workbench")
    want_sha = ((ident.get("exe_sha256") or {}).get("tools") or {}).get(WB_EXE_REL)
    # 1. the listener and its process
    try:
        pids = listening_pids(probes.netstat_text(), port, host)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reasons": ["cannot list listening ports (netstat.exe): %s" % exc], "facts": facts}
    facts["pids"] = sorted(pids)
    if len(pids) != 1:
        return {"ok": False, "facts": facts, "reasons": [
            "no process listens on %s:%d (Workbench closed or NET API off)" % (host, port) if not pids else
            "%d processes listen on port %d (%s); only one Workbench may run" % (len(pids), port, sorted(pids))]}
    pid = pids.pop()
    try:
        info = parse_process_line(probes.process_line(pid), pid)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reasons": ["cannot read process %d (wb_proc.js): %s" % (pid, exc)], "facts": facts}
    if info is None:
        return {"ok": False, "reasons": ["process %d is gone or unreadable" % pid], "facts": facts}
    exe, cmd, created = info["ExecutablePath"], info["CommandLine"], info["CreationUnix"]
    args = cli_args(split_cmdline(cmd)[1:])
    facts.update(pid=pid, exe=exe, created_unix=created, profile=args.get("profile"), gproj=args.get("gproj"),
                 launch_id=args.get("dclaunchid"))
    if not exe or pinlib.win_to_wsl(exe).lower() != want_exe_wsl.lower():
        reasons.append("exe %s is not the pin's Workbench %s" % (exe or "(unknown)", want_exe))
    if created is None:
        reasons.append("process %d has no readable creation time" % pid)
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
    # 2. the launch record
    rec_path = os.path.join(pin_src["pin_dir"], "launches", "launch_%s.json" % lid)
    rec = None
    try:
        rec = json.loads(pinlib.read_regular_file(rec_path, "launch record", 1 << 22))
        probs = launch_record_problems(rec, lid, pin_src, ident, lp)
    except (pinlib.PinError, ValueError) as exc:
        probs = ["launch %s has no readable record: %s" % (lid, exc)]
    reasons.extend(probs)
    rec_unix = None if probs else pinlib.iso_utc_to_unix(rec["utc"])
    if rec_unix is not None and created is not None and \
            not (START_AFTER_RECORD_S[0] <= created - rec_unix <= START_AFTER_RECORD_S[1]):
        reasons.append("process %d started %+d s from launch record %s (expected %d..%d s): a restart that reuses an "
                       "old launch id?" % (pid, created - rec_unix, lid, *START_AFTER_RECORD_S))
    # 3. the launch's own console.log, exactly one, bound to the process by its creation time
    logs = pinlib.launch_logs(pinlib.win_to_wsl(args.get("profile") or ""), lid, rec_unix)
    facts["console_logs"] = [x["path"] for x in logs]
    if len(logs) != 1:
        reasons.append("%d console.log(s) under %s\\logs name launch %s after its record (expected exactly one)"
                       % (len(logs), args.get("profile"), lid))
    else:
        check_log(logs[0], want_cwd, args.get("gproj"), ident.get("engine_version_expected"), created, reasons, facts)
    return {"ok": not reasons, "reasons": reasons, "facts": facts}


def resolve(pin_id=None, pin_active=False, pin_dir=None) -> tuple:
    pin_dir = os.path.abspath(pinlib.win_to_wsl(pin_dir or pinlib.DEFAULT_PIN_DIR))
    if pin_active:
        src = preflight.resolve_pin(None, True, pin_dir)
    else:
        src = preflight.load_pin_by_id(pin_dir, pinlib.validate_pin_id(pin_id))
    return src, preflight.pin_identity(src)


def main(argv=None, probes=None, gateway=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--pin-id", default=None)
    which.add_argument("--pin-active", action="store_true")
    ap.add_argument("--pin-dir", default=None)
    ap.add_argument("--host", default=None, help="default: ENFUSION_WORKBENCH_HOST, else the WSL default gateway")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        src, ident = resolve(args.pin_id, args.pin_active, args.pin_dir)
    except pinlib.PinError as exc:
        print("endpoint_handshake: %s" % exc, file=sys.stderr)
        return 2
    gw = default_gateway() if gateway is None else gateway
    host = args.host or os.environ.get("ENFUSION_WORKBENCH_HOST") or gw or "127.0.0.1"
    res = handshake(src, ident, args.port, probes, None, host, gw)
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
