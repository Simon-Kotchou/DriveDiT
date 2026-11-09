#!/usr/bin/env python3
"""Write this machine's local config from the paths it used before DD-22, and prove that nothing moved (DD-22).

Before DD-22 the repo held this machine's paths as literals. This tool reads them back from a revision that still
has them (--from-rev REV, read with `git show`; or --from-dir DIR, the same files exported), derives every key of
drivedit.local.env from them, and then compares, for that config, every value the old revision hard-coded with what
the current code makes of the config:
  - the four launchers (tools/launch): the resolved start line, every set variable and the wsl.exe preflight command
    lines, old against new (the new ones read through dd_local_config.cmd's values, computed by
    tools/local_config.py bat_values);
  - the old golden start lines against tools/pin/tests/fixtures/golden_start_lines.txt rendered with the config;
  - every path table of tools/pin/pinlib.py, the roots of tools/deploy_addon.py and the profile root it writes into
    the addon, the Windows tools and helper temp dir of tools/enfusion-mcp/endpoint_handshake.py (the current
    modules, loaded with the config);
  - every old write root and forbidden root (pinlib, deploy, harvester, probes, dataset builder) against the write
    boundary of tools/local_config.py with this config: each old root must still be writable, each old forbidden
    root still refused (check_write on a probe path below it; nothing is written), the data drive must be the one
    the boundary verifies, and the old system drive the advisory SYSTEM_DRIVE;
  - the old literals of the modules that need numpy (harvester, validator, dataset builder, reader, DINOv3 checkout)
    and of the MCP roots against their runtime getters: the current modules run in a child interpreter that has
    the repo's dependencies (--python; default the repo's .venv, else this interpreter) with the config in RAM,
    and the wrapper prints its default work root (enfusion-mcp-wsl.sh --print-root).
Any difference refuses: nothing is written. It reads the repo and the revision only; it never runs a launcher.

    python3 -I -B tools/migrate_local_config.py --from-rev REV --dry-run   # print the config and the comparison
    python3 -I -B tools/migrate_local_config.py --from-rev REV --write     # write <DATA_ROOT>/config/drivedit.local.env

--write writes exactly one new file, <DATA_ROOT>/config/drivedit.local.env (creating config/ when missing), through
the write guard of tools/pin, which is the write boundary of tools/local_config.py (the data drive verified in
/proc/self/mountinfo, never a Windows system volume) narrowed to the data drive. An existing file is never
replaced; an identical one is left alone. Exit codes: 0 ok (or dry run); 1 a difference (nothing written); 2 usage,
read or path error.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import posixpath
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)

LEGACY = {
    "pinlib": "tools/pin/pinlib.py",
    "deploy": "tools/deploy_addon.py",
    "golden": "tools/pin/tests/fixtures/golden_start_lines.txt",
    "wrapper": "tools/enfusion-mcp/enfusion-mcp-wsl.sh",
    "gate": "tools/enfusion-mcp/mcp_gate.py",
    "handshake": "tools/enfusion-mcp/endpoint_handshake.py",
    "harvest": "tools/harvester/harvest.py",
    "validate": "tools/validator/dc_validate.py",
    "build": "src/drivedit/data/build.py",
    "reader": "data/dc2/reader.py",
    "probe": "tools/probes/t0eng/probe_common.py",
    "dinov3": "src/drivedit/codec/dinov3.py",
    "session_c": "src/enfusion/datacapture/DC_Session.c",
    "layer_arland": "addons/DataCapture/worlds/DataCapture/DC_Arland_Layers/default.layer",
    "layer_probes": "addons/DataCapture/worlds/DataCapture/DC_Probes_Arland_Layers/default.layer",
}
LAUNCHERS = ("wb_R1.bat", "client_R1_i01.bat", "wb_C18.bat", "client_C18_i01.bat")
PINLIB_TABLES = ("DEFAULT_ACF", "DEFAULT_ACF_BY_CHANNEL", "LAUNCH_PATHS", "COLD_RUN_ROOTS", "COLD_LAUNCH_PATHS",
                 "EXPERIMENTAL_LAUNCH_PATHS", "WB_LAUNCH", "DEPLOY_ROOTS", "DEFAULT_WRITE_ROOTS", "DEFAULT_PIN_DIR")
# A path below an old root, for the write boundary's verdict on that root (check_write only; nothing is created).
BOUNDARY_PROBE = "__dd22_migrate_boundary_probe__/x"
DEPLOY_TABLES = ("DEFAULT_TARGET", "ALLOWED_TARGET_ROOTS", "ALLOW_ROOT_PREFIXES", "DEFAULT_DEPLOY_ROOTS")


class MigrationError(Exception):
    """A revision, file or value that cannot be migrated (exit 2)."""


def load_path(name, path):
    """A module loaded by path under name (replacing any module of that name loaded from another file)."""
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The loader every current tool shares (pinlib, deploy_addon and the others load this file under this name).
lc = load_path("dd_local_config", os.path.join(HERE, "local_config.py"))


# --------------------------------------------------------------------------- the old revision


class Legacy:
    """The files of the old revision: git show REV:<path> in repo, or <dir>/<path>."""

    def __init__(self, repo=None, rev=None, directory=None):
        self.repo, self.rev, self.directory = repo, rev, directory
        self._cache = {}

    def where(self):
        return "revision %s" % self.rev if self.rev else "directory %s" % self.directory

    def text(self, rel):
        if rel in self._cache:
            return self._cache[rel]
        if self.rev:
            r = subprocess.run(["git", "--no-optional-locks", "-C", self.repo, "show", "%s:%s" % (self.rev, rel)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0:
                raise MigrationError("cannot read %s at %s: %s" % (rel, self.rev, r.stderr.decode("utf-8", "replace")
                                                                    .strip()))
            data = r.stdout
        else:
            try:
                with open(os.path.join(self.directory, *rel.split("/")), "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise MigrationError("cannot read %s in %s: %s" % (rel, self.directory, exc))
        text = data.decode("utf-8").replace("\r\n", "\n")
        self._cache[rel] = text
        return text

    def module(self, rel):
        """The namespace of an old stdlib-only module, executed in memory (its main() never runs)."""
        ns = {"__name__": "dd_legacy_" + posixpath.basename(rel).replace(".", "_"),
              "__file__": "/nonexistent-dd22-legacy/" + rel, "__builtins__": __builtins__}
        exec(compile(self.text(rel), "<%s>/%s" % (self.where(), rel), "exec"), ns)
        return ns


def literal(text, name):
    """The value of a one-line `NAME = <literal>` (or `NAME: Final = Path("...")`) assignment, or MigrationError."""
    m = re.search(r"(?m)^%s(?::\s*Final)?\s*=\s*(.+?)\s*(?:#.*)?$" % re.escape(name), text)
    if not m:
        raise MigrationError("no literal %s in the old file" % name)
    rhs = m.group(1)
    p = re.fullmatch(r'Path\("([^"]+)"\)', rhs)
    try:
        return p.group(1) if p else ast.literal_eval(rhs)
    except (ValueError, SyntaxError):
        raise MigrationError("%s = %s is not a literal" % (name, rhs))


def one(pattern, text, what):
    found = sorted(set(re.findall(pattern, text)))
    if len(found) != 1:
        raise MigrationError("expected one %s in the old file, found %s" % (what, found or "none"))
    return found[0]


def derive(legacy):
    """{key: value} of drivedit.local.env from the old revision's literals."""
    pin = legacy.module(LEGACY["pinlib"])
    if "DEFAULT_PIN_DIR" not in pin or "LAUNCH_PATHS" not in pin:
        raise MigrationError("the old pinlib.py has no DEFAULT_PIN_DIR/LAUNCH_PATHS literals: %s is not a pre-DD-22 "
                             "revision" % legacy.where())
    distros = {one(r"wsl\.exe -d (\S+) -e ", legacy.text("tools/launch/" + n), "wsl.exe -d distro in " + n)
               for n in LAUNCHERS}
    if len(distros) != 1:
        raise MigrationError("the launchers name different WSL distros: %s" % sorted(distros))
    mcp_root = one(r'ROOT="\$\{EMCP_ROOT:-([^}"]+)\}"', legacy.text(LEGACY["wrapper"]), "EMCP_ROOT default")
    dinov3 = literal(legacy.text(LEGACY["dinov3"]), "DEFAULT_DINOV3_DIR")
    if posixpath.basename(mcp_root) != "mcp" or not dinov3.endswith("/third_party/dinov3"):
        raise MigrationError("unexpected MCP root %r or DINOv3 dir %r" % (mcp_root, dinov3))
    return {
        "DATA_ROOT": posixpath.dirname(pin["DEFAULT_PIN_DIR"]),
        "REALIGN_ROOT": posixpath.dirname(mcp_root),
        "CODE_ROOT": dinov3[:-len("/third_party/dinov3")],
        "GAME_INSTALL": pin["LAUNCH_PATHS"]["game"]["install"],
        "TOOLS_INSTALL": pin["LAUNCH_PATHS"]["tools"]["install"],
        "EXPERIMENTAL_GAME_INSTALL": pin["EXPERIMENTAL_LAUNCH_PATHS"]["game"]["install"],
        "EXPERIMENTAL_TOOLS_INSTALL": pin["EXPERIMENTAL_LAUNCH_PATHS"]["tools"]["install"],
        "SYSTEM_DRIVE": pin["FORBIDDEN_WRITE_PREFIXES"][0].rstrip("/"),
        "WSL_DISTRO": distros.pop(),
    }


def config_text(values, source):
    lines = ["# drivedit.local.env of this machine (DD-22). Written by tools/migrate_local_config.py from the paths of",
             "# %s; never commit it. Check it with: python3 -I -B tools/local_config.py check" % source, ""]
    for key in lc.KEYS:
        lines += ["# " + lc.KEYS[key][1], "%s=%s" % (key, values[key])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- old against new


def launcher_rows(legacy, cfg):
    rows = []
    seed = lc.bat_values(cfg)
    for name in LAUNCHERS:
        old = legacy.text("tools/launch/" + name)
        with open(os.path.join(REPO, "tools", "launch", name), "rb") as fh:
            new = fh.read().decode("ascii").replace("\r\n", "\n")
        rows.append(("%s start line" % name, lc.bat_start_line(old), lc.bat_start_line(new, seed)))
        old_env, new_env = lc.bat_env(old), lc.bat_env(new, seed)
        for var in sorted(old_env):
            rows.append(("%s set %s" % (name, var), old_env[var], new_env.get(var, "<not set>")))

        def wsl_lines(text, env):
            out = []
            for line in text.split("\n"):
                if "wsl.exe -d" in line:
                    out.append(re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%", lambda m: env.get(m.group(1), m.group(0)),
                                      line.strip()))
            return out
        rows.append(("%s wsl.exe lines" % name, wsl_lines(old, old_env), wsl_lines(new, new_env)))
    old_golden = [l for l in legacy.text(LEGACY["golden"]).split("\n") if l and not l.startswith("#")]
    with open(os.path.join(REPO, *LEGACY["golden"].split("/")), encoding="ascii") as fh:
        template = [l for l in fh.read().split("\n") if l and not l.startswith("#")]
    rows.append(("golden_start_lines.txt", old_golden, [lc.render(l, cfg) for l in template]))
    return rows


def module_rows(legacy, cfg):
    rows = []
    with lc.override(cfg):
        pinlib = load_path("dd_pin_pinlib", os.path.join(REPO, "tools", "pin", "pinlib.py"))
        old = legacy.module(LEGACY["pinlib"])
        for name in PINLIB_TABLES:
            rows.append(("pinlib.%s" % name, old[name], getattr(pinlib, name)))
        deploy = load_path("dd_current_deploy_addon", os.path.join(REPO, "tools", "deploy_addon.py"))
        old = legacy.module(LEGACY["deploy"])
        for name in DEPLOY_TABLES:
            rows.append(("deploy_addon.%s" % name, old[name], getattr(deploy, name)))
        system = cfg.get("SYSTEM_DRIVE")
        rows.append(("deploy_addon.FORBIDDEN_DEVICE_PATHS (paths; the system drive is the boundary's now)",
                     [p for p, _ in old["FORBIDDEN_DEVICE_PATHS"] if p != system],
                     [p for p, _ in deploy.FORBIDDEN_DEVICE_PATHS]))
        root = deploy.profile_root_value()
        rows.append(("DC_Session.c m_sRequiredProfileRoot default",
                     one(r'\[Attribute\("([^"]+)", UIWidgets\.EditBox, "Launch guard:', legacy.text(LEGACY["session_c"]),
                         "profile root attribute"), root))
        for key in ("layer_arland", "layer_probes"):
            rows.append(("%s m_sRequiredProfileRoot" % posixpath.basename(posixpath.dirname(LEGACY[key])),
                         one(r'm_sRequiredProfileRoot "([^"]+)"', legacy.text(LEGACY[key]), "layer profile root"),
                         root))
        old_hs = legacy.text(LEGACY["handshake"])
        hs = load_path("dd_emcp_endpoint_handshake", os.path.join(REPO, "tools", "enfusion-mcp",
                                                                  "endpoint_handshake.py"))
        for name in ("NETSTAT", "CSCRIPT", "IPCONFIG", "MCP_WIN_TEMP"):
            rows.append(("endpoint_handshake.%s" % name, literal(old_hs, name), getattr(hs, name)))
    return rows


def boundary_rows(legacy, cfg):
    """Every old write root and forbidden root against the write boundary with cfg (tools/local_config.py
    check_write on BOUNDARY_PROBE below the root, through lc.mounted_path; it only checks)."""
    pin = legacy.module(LEGACY["pinlib"])
    deploy = legacy.module(LEGACY["deploy"])
    harvest, probe = legacy.text(LEGACY["harvest"]), legacy.text(LEGACY["probe"])
    allowed = [("pinlib.DEFAULT_WRITE_ROOTS", r) for r in pin["DEFAULT_WRITE_ROOTS"]]
    allowed += [("pinlib.WRITE_ROOT_DRIVES", r) for r in sorted(pin["WRITE_ROOT_DRIVES"])]
    allowed += [("probe_common.WRITE_ROOTS", r) for r in literal(probe, "WRITE_ROOTS")]
    allowed += [("deploy_addon.ALLOW_ROOT_PREFIXES", r) for r in deploy["ALLOW_ROOT_PREFIXES"]]
    allowed += [("harvest.TEST_ROOT_PREFIXES", r) for r in literal(harvest, "TEST_ROOT_PREFIXES")]
    refused = [("pinlib.FORBIDDEN_WRITE_PREFIXES", r) for r in pin["FORBIDDEN_WRITE_PREFIXES"]]
    refused += [("probe_common.FORBIDDEN_ROOTS", r) for r in literal(probe, "FORBIDDEN_ROOTS")]
    refused += [("deploy_addon.FORBIDDEN_DEVICE_PATHS", p) for p, _ in deploy["FORBIDDEN_DEVICE_PATHS"]]
    refused += [("harvest.FORBIDDEN_DEVICE_PATHS", p) for p, _ in literal(harvest, "FORBIDDEN_DEVICE_PATHS")]
    refused += [("build.FORBIDDEN_ROOTS", r) for r in literal(legacy.text(LEGACY["build"]), "FORBIDDEN_ROOTS")]
    rows = []
    with lc.override(cfg):
        for label, root in allowed:
            probe_path = posixpath.join(lc.mounted_path(root.rstrip("/") or "/"), BOUNDARY_PROBE)
            why = lc.write_refusal(probe_path)
            rows.append(("boundary writes under %s (%s)" % (root, label), True, why is None if why is None else why))
        for label, root in refused:
            probe_path = posixpath.join(lc.mounted_path(root.rstrip("/") or "/"), BOUNDARY_PROBE)
            rows.append(("boundary refuses %s (%s)" % (root, label), True, lc.write_refusal(probe_path) is not None))
        rows.append(("pinlib.WRITE_ROOT_DRIVES -> the boundary's data drive", pin["WRITE_ROOT_DRIVES"],
                     dict([lc.data_drive()])))
        rows.append(("pinlib.FORBIDDEN_WRITE_PREFIXES -> SYSTEM_DRIVE (advisory)",
                     [r.rstrip("/") for r in pin["FORBIDDEN_WRITE_PREFIXES"]], [lc.advisory_system_drive()]))
    return rows


# Run in the child interpreter: the runtime getters of the modules that need numpy (and the MCP gate), with the
# config DRIVEDIT_LOCAL_CONFIG names. Prints one JSON object.
RUNTIME_PROBE = r"""
import importlib.util, json, os, sys
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "src"))
def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(repo, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
out = {}
harvest = load("dd_migrate_harvest", "tools/harvester/harvest.py")
out["harvest.DATA_ROOT"] = harvest.DATA_ROOT
out["harvest.TEST_ROOT_PREFIXES"] = sorted(harvest.TEST_ROOT_PREFIXES)
out["harvest.FORBIDDEN_DEVICE_PATHS"] = [p for p, _ in harvest.FORBIDDEN_DEVICE_PATHS]
validate = load("dd_migrate_dc_validate", "tools/validator/dc_validate.py")
out["dc_validate.DEFAULT_REPORT_DIR"] = validate.DEFAULT_REPORT_DIR
out["dc_validate.REPORT_ROOT"] = validate.REPORT_ROOT
from drivedit.data import build
out["build.default_dataset_roots"] = list(build.default_dataset_roots())
out["build.default_capture_roots"] = list(build.default_capture_roots())
reader = load("dd_migrate_reader", "data/dc2/reader.py")
out["reader system drive"] = reader._system_drive()[0]
dinov3 = load("dd_migrate_dinov3", "src/drivedit/codec/dinov3.py")
out["dinov3.default_dinov3_dir"] = str(dinov3.default_dinov3_dir())
gate = load("dd_migrate_mcp_gate", "tools/enfusion-mcp/mcp_gate.py")
out["mcp_gate.default_root"] = gate.default_root({})
print(json.dumps(out))
"""


def default_python():
    venv = os.path.join(REPO, ".venv", "bin", "python")
    return venv if os.path.isfile(venv) else sys.executable


def runtime_values(cfg_text, repo=REPO, python=None):
    """{getter: value} from RUNTIME_PROBE in a child interpreter, with the config written to a new file in RAM (the
    write boundary's scratch, removed afterwards); MigrationError when the child fails."""
    import shutil
    import tempfile
    import json
    tmp = tempfile.mkdtemp(prefix="dd22_migrate_", dir=lc.check_write(lc.SCRATCH_MOUNT))
    try:
        path = lc.write_new(os.path.join(tmp, lc.CONFIG_NAME), cfg_text.encode("ascii"))
        env = dict(os.environ, DRIVEDIT_LOCAL_CONFIG=path, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([python or default_python(), "-I", "-B", "-c", RUNTIME_PROBE, repo], capture_output=True,
                           text=True, env=env, timeout=300)
        if r.returncode != 0:
            raise MigrationError("the runtime getters failed in %s (pass --python an interpreter with the repo's "
                                 "dependencies): %s" % (python or default_python(), r.stderr.strip()[-2000:]))
        out = json.loads(r.stdout.strip().splitlines()[-1])
        wrapper = subprocess.run(["bash", os.path.join(repo, "tools", "enfusion-mcp", "enfusion-mcp-wsl.sh"),
                                  "--print-root"], capture_output=True, text=True, timeout=120,
                                 env={k: v for k, v in env.items() if k != "EMCP_ROOT"})
        out["enfusion-mcp-wsl.sh default root"] = wrapper.stdout.strip() if wrapper.returncode == 0 else (
            "<exit %d: %s>" % (wrapper.returncode, wrapper.stderr.strip()))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def runtime_rows(legacy, cfg, cfg_text, repo=REPO, python=None):
    """Old literals of the numpy-dependent modules and the MCP roots against their runtime getters (runtime_values)."""
    harvest, validate = legacy.text(LEGACY["harvest"]), legacy.text(LEGACY["validate"])
    build, reader = legacy.text(LEGACY["build"]), legacy.text(LEGACY["reader"])
    got = runtime_values(cfg_text, repo, python)
    system = cfg.get("SYSTEM_DRIVE")
    return [
        ("mcp_gate EMCP_ROOT default", one(r'env\.get\("EMCP_ROOT"\) or "([^"]+)"', legacy.text(LEGACY["gate"]),
                                           "gate root"), got["mcp_gate.default_root"]),
        ("enfusion-mcp-wsl.sh EMCP_ROOT default", one(r'ROOT="\$\{EMCP_ROOT:-([^}"]+)\}"', legacy.text(LEGACY["wrapper"]),
                                                      "wrapper root"), got["enfusion-mcp-wsl.sh default root"]),
        ("harvest.DATA_ROOT", literal(harvest, "DATA_ROOT"), got["harvest.DATA_ROOT"]),
        ("harvest.TEST_ROOT_PREFIXES", sorted(literal(harvest, "TEST_ROOT_PREFIXES")), got["harvest.TEST_ROOT_PREFIXES"]),
        ("harvest.FORBIDDEN_DEVICE_PATHS (paths; the system drive is the boundary's now)",
         [p for p, _ in literal(harvest, "FORBIDDEN_DEVICE_PATHS") if p != system], got["harvest.FORBIDDEN_DEVICE_PATHS"]),
        ("dc_validate.DEFAULT_REPORT_DIR", literal(validate, "DEFAULT_REPORT_DIR"), got["dc_validate.DEFAULT_REPORT_DIR"]),
        ("dc_validate.REPORT_ROOT", literal(validate, "REPORT_ROOT"), got["dc_validate.REPORT_ROOT"]),
        ("build.DATASET_ROOTS", list(literal(build, "DATASET_ROOTS")), got["build.default_dataset_roots"]),
        ("build.CAPTURE_ROOTS", list(literal(build, "CAPTURE_ROOTS")), got["build.default_capture_roots"]),
        ("reader.FORBIDDEN_ROOTS", [r.rstrip("/") for r in literal(reader, "FORBIDDEN_ROOTS")],
         [got["reader system drive"]]),
        ("dinov3.DEFAULT_DINOV3_DIR", literal(legacy.text(LEGACY["dinov3"]), "DEFAULT_DINOV3_DIR"),
         got["dinov3.default_dinov3_dir"]),
    ]


def compare(legacy, cfg, cfg_text=None, repo=REPO, python=None):
    """[(label, old, new, same)] of every comparison. cfg_text: the config's text (for the child interpreter of the
    runtime getters; default: rebuilt from cfg)."""
    if cfg_text is None:
        cfg_text = "".join("%s=%s\n" % kv for kv in cfg.values.items())
    rows = (launcher_rows(legacy, cfg) + module_rows(legacy, cfg) + boundary_rows(legacy, cfg)
            + runtime_rows(legacy, cfg, cfg_text, repo, python))
    return [(label, old, new, old == new) for label, old, new in rows]


# --------------------------------------------------------------------------- writing


def write_config(text, cfg):
    """Write <DATA_ROOT>/config/drivedit.local.env once, through the pin tools' guard. Returns (path, written)."""
    with lc.override(cfg):
        pinlib = load_path("dd_pin_pinlib", os.path.join(REPO, "tools", "pin", "pinlib.py"))
        directory = os.path.join(cfg.get("DATA_ROOT"), "config")
        target = os.path.join(directory, lc.CONFIG_NAME)
        try:
            checked_dir = pinlib.check_write_path(directory)
            if os.path.lexists(target):
                with open(target, "rb") as fh:
                    if fh.read() == text.encode("ascii"):
                        return target, False
                raise MigrationError("%s exists and differs; it is never replaced (compare and remove it by hand "
                                     "first)" % target)
            if not os.path.isdir(checked_dir):
                os.mkdir(checked_dir)
            return pinlib.publish_new(target, text.encode("ascii")), True
        except pinlib.PinError as exc:
            raise MigrationError(str(exc))


def short(value):
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-rev", metavar="REV", help="a pre-DD-22 revision of this repo (read with git show)")
    src.add_argument("--from-dir", metavar="DIR", help="the files of such a revision, exported to DIR")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the config and the comparison; write nothing")
    mode.add_argument("--write", action="store_true", help="write <DATA_ROOT>/config/drivedit.local.env")
    ap.add_argument("--quiet", action="store_true", help="print only differences and the summary")
    ap.add_argument("--python", default=None, help="the interpreter for the runtime getters (needs numpy and the "
                    "repo's package; default: the repo's .venv, else this interpreter)")
    args = ap.parse_args(argv)
    legacy = Legacy(repo=REPO, rev=args.from_rev, directory=args.from_dir)
    try:
        values = derive(legacy)
        text = config_text(values, legacy.where())
        cfg = lc.parse(text, "<the config written to %s/config/%s>" % (values["DATA_ROOT"], lc.CONFIG_NAME))
        rows = compare(legacy, cfg, text, python=args.python)
    except (MigrationError, lc.LocalConfigError) as exc:
        print("migrate_local_config: %s" % exc, file=sys.stderr)
        return 2
    print("# --- %s/config/%s" % (values["DATA_ROOT"], lc.CONFIG_NAME))
    print(text, end="")
    print("# --- old (%s) against new (current code with this config)" % legacy.where())
    for label, old, new, same in rows:
        if same and args.quiet:
            continue
        print("%-4s %s" % ("SAME" if same else "DIFF", label) + ("" if same else "\n     old %s\n     new %s"
                                                                   % (short(old), short(new))))
    diffs = sum(1 for r in rows if not r[3])
    print("%d comparison(s): %d same, %d different" % (len(rows), len(rows) - diffs, diffs))
    if diffs:
        print("migrate_local_config: refusing: the current code would not reproduce the old paths; nothing written",
              file=sys.stderr)
        return 1
    if args.dry_run:
        print("DRY RUN: nothing written")
        return 0
    try:
        path, written = write_config(text, cfg)
    except MigrationError as exc:
        print("migrate_local_config: %s" % exc, file=sys.stderr)
        return 2
    print("%s %s" % ("wrote" if written else "already there, identical:", path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
