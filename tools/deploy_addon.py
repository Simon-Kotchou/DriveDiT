#!/usr/bin/env python3
"""Deploy the repo-versioned DataCapture Workbench addon to its E: target, from a manifest.

Run it with the isolated interpreter (stdlib only):

    python3 -I -B tools/deploy_addon.py [--target DIR] [--allow-root DIR] [--dry-run] [--emcp-handlers DIR] [FILE ...]

What it copies (nothing else, ever):
  * the files listed in addons/DataCapture/deploy_manifest.txt:
      addon  <path under addons/DataCapture/>         -> <target>/<same relative path>
      script src/enfusion/datacapture/<Name>.c        -> <target>/Scripts/Game/DataCapture/<Name>.c
      config addons/DataCapture/Configs/<path>.json   -> <target>/Configs/<path>.json (routes, DD-7-min)
      config tools/ident/cards/<pin_id>/<id>.json     -> <target>/Configs/ident/cards/<pin_id>/<id>.json (vehicle
                                                         cards: the repo keeps them where the DD-5-min reader looks)
    Config files are copied byte for byte, so the sha256 llc_v1 and the expert compute in the engine is the
    repo file's.
  * a generated build stamp <target>/Scripts/Game/DataCapture/DC_BuildInfo.c (git sha, branch, dirty flag,
    deploy UTC time, manifest hash, content hash), because Enforce Script cannot run git;
  * with --emcp-handlers DIR, the EMCP_WB_*.c files found directly in DIR
                                                       -> <target>/Scripts/WorkbenchGame/EnfusionMCP/

Safety rules:
  * Refuses (exit 2, before writing anything) when the manifest lists a file outside the allowed set
    (core/, modules/ or any subfolder of src/enfusion/datacapture/, DriveDiT_DataCapture/, src/SCR_CaptureTests.c,
    the auxiliary prototypes, SCR_AIDrivingSimulator, Workbench-generated files; compared without regard to case,
    and every manifest path must match the repo's spelling exactly, because E: is case-insensitive), or when a FILE
    argument is not in the manifest.
  * Target allowlist: the resolved target must lie strictly inside /mnt/e/DriveDiT/addons/ (the default target is
    /mnt/e/DriveDiT/addons/DataCapture). --allow-root DIR adds another root for scratch deploys and the self-test;
    DIR must be an existing directory under /mnt/e/ or /dev/shm/. Targets on the C: drive (/mnt/c, C:\\..., the
    same device as /mnt/c) or on the WSL disk image (the device of /, which lives on C:) are always refused.
  * Never writes through a symlink or junction inside the target: every destination, the state file and every
    backup path are checked component by component before anything is written.
  * Never writes, overwrites or deletes Workbench-generated files (*.meta, resourceDatabase.rdb, UserMaps.desc).
  * Deletes only files it deployed before. They are tracked in <target>/.deploy_state.json together with their
    sha256. A tracked file that changed since it was deployed (a Workbench save, a hand edit) and an existing
    file this tool did not deploy are never overwritten or deleted silently: the run is refused unless --force is
    given, in which case the old content is first copied to <target>/.deploy_backup/<UTC>/<path>.bak.
  * FILE arguments deploy a subset of the manifest (plus the build stamp) and never delete anything.

Exit codes: 0 success (or dry run), 2 refused, 1 unexpected I/O error.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

TOOL_REL = "tools/deploy_addon.py"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DEFAULT_TARGET = "/mnt/e/DriveDiT/addons/DataCapture"
# Deploy targets must resolve strictly inside one of these roots (DD-1: deploys write only under E:\DriveDiT\addons).
ALLOWED_TARGET_ROOTS = ("/mnt/e/DriveDiT/addons",)
# An extra --allow-root must resolve strictly under one of these (E:, or RAM for the self-test).
ALLOW_ROOT_PREFIXES = ("/mnt/e/", "/dev/shm/")
# Paths whose device must never receive a deploy: the C: drive, and / (the WSL disk image, which lives on C:).
FORBIDDEN_DEVICE_PATHS = (("/mnt/c", "the C: drive"), ("/", "the WSL disk image, which lives on the C: drive"))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "addons", "DataCapture", "deploy_manifest.txt")

ADDON_SRC_DIR = "addons/DataCapture/"
SCRIPT_SRC_DIR = "src/enfusion/datacapture/"
# config sources -> deployed folder (DD-7-min): routes under the addon's Configs/, vehicle cards from tools/ident/cards/
CONFIG_ROOTS = (("addons/DataCapture/Configs/", "Configs/"), ("tools/ident/cards/", "Configs/ident/cards/"))
CONFIG_EXT = ".json"
CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SCRIPT_DST_DIR = "Scripts/Game/DataCapture"
EMCP_DST_DIR = "Scripts/WorkbenchGame/EnfusionMCP"
BUILDINFO_DST = SCRIPT_DST_DIR + "/DC_BuildInfo.c"
STATE_NAME = ".deploy_state.json"
BACKUP_DIR = ".deploy_backup"
STATE_SCHEMA = 1

ADDON_EXTS = (".gproj", ".ent", ".layer", ".et", ".conf")
SCRIPT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\.c$")
EMCP_NAME_RE = re.compile(r"^EMCP_WB_[A-Za-z0-9_]*\.c$")

# Files Workbench creates and owns. The tool never deploys, overwrites or deletes them.
WB_GENERATED_NAMES = ("resourcedatabase.rdb", "usermaps.desc")
WB_GENERATED_SUFFIXES = (".meta",)

# Repo paths that must never reach the deployed addon, with the reason printed on refusal.
DENIED_PREFIXES = (
    ("DriveDiT_DataCapture/", "the unloadable in-repo prototype addon is never deployed (review TKT-10)"),
    (SCRIPT_SRC_DIR + "core/", "the modular core is not deployed (compile errors; review TKT-10)"),
    (SCRIPT_SRC_DIR + "modules/", "the modular capture modules are not deployed (1.8 API errors; review TKT-10)"),
    ("src/SCR_CaptureTests.c", "pre-1.8 test file with known compile errors is never deployed (review TKT-11)"),
)
DENIED_SCRIPTS = {
    "SCR_AIDrivingSimulator.c": "superseded by the compiled director; it must never run next to it (review TKT-01)",
    "SCR_BinarySerializer.c": "auxiliary prototype (ENFCAP), never deployed (review TKT-10)",
    "SCR_DepthRaycaster.c": "auxiliary prototype with known compile errors, never deployed (review TKT-10)",
    "SCR_DrivingSimDebugUI.c": "auxiliary prototype bound to SCR_AIDrivingSimulator, never deployed (review TKT-10)",
    "SCR_ModularCaptureExample.c": "example for the modular core, never deployed (review TKT-10)",
    "SCR_MultiCameraRig.c": "auxiliary prototype, never deployed (review TKT-10)",
    "SCR_SceneEnumerator.c": "auxiliary prototype, never deployed (review TKT-10)",
    "DC_BuildInfo.c": "generated by this tool on every deploy; it is never copied from the repo",
}
# E: (NTFS through drvfs) is case-insensitive, so every denial check compares lower-cased paths.
DENIED_SCRIPTS_LOWER = {name.lower(): reason for name, reason in DENIED_SCRIPTS.items()}

GROUP_ADDON = "addon"
GROUP_SCRIPT = "script"
GROUP_CONFIG = "config"
VERSIONED_GROUPS = (GROUP_ADDON, GROUP_SCRIPT, GROUP_CONFIG)
GROUP_BUILDINFO = "buildinfo"
GROUP_EMCP = "emcp"


class Refusal(Exception):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_wb_generated(rel):
    base = rel.rsplit("/", 1)[-1].lower()
    return base in WB_GENERATED_NAMES or base.endswith(WB_GENERATED_SUFFIXES)


def is_reserved_dst(rel):
    first = rel.split("/", 1)[0]
    return rel == STATE_NAME or first == BACKUP_DIR


def clean_rel(rel, what):
    """Validate a relative POSIX path (no backslash, no '..', no absolute path, no empty parts)."""
    if not rel or "\\" in rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise Refusal("%s %r is not a relative POSIX path" % (what, rel))
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise Refusal("%s %r contains an empty, '.' or '..' component" % (what, rel))
    return rel


def denial_reason(src_rel):
    low = src_rel.lower()
    for prefix, reason in DENIED_PREFIXES:
        prefix = prefix.lower()
        if low == prefix.rstrip("/") or low.startswith(prefix):
            return reason
    script_dir = SCRIPT_SRC_DIR.lower()
    if low.startswith(script_dir):
        rest = low[len(script_dir):]
        if "/" in rest:
            return "only top-level scripts of %s are deployable; subfolders are not" % SCRIPT_SRC_DIR
        if rest in DENIED_SCRIPTS_LOWER:
            return DENIED_SCRIPTS_LOWER[rest]
    if is_wb_generated(src_rel):
        return "Workbench-generated files are never versioned or deployed"
    return None


def inside(path, root):
    path = os.path.normcase(path)
    root = os.path.normcase(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def case_mismatch(src_rel):
    """Return the first component of src_rel whose spelling differs from the directory entry on disk, or None.

    On a case-insensitive drive (E:) 'scr_binaryserializer.c' opens SCR_BinarySerializer.c, so only the exact
    spelling is accepted; otherwise a case variant could slip a denied file past the checks."""
    cur = REPO_ROOT
    for part in src_rel.split("/"):
        try:
            names = os.listdir(cur)
        except OSError:
            return None
        if part not in names:
            return part
        cur = os.path.join(cur, part)
    return None


def resolve_source(src_rel):
    src_abs = os.path.join(REPO_ROOT, *src_rel.split("/"))
    if os.path.islink(src_abs):
        raise Refusal("%s is a symlink; only regular files inside the repo are deployed" % src_rel)
    if not os.path.isfile(src_abs):
        raise Refusal("%s does not exist in the repo (%s)" % (src_rel, REPO_ROOT))
    wrong = case_mismatch(src_rel)
    if wrong:
        raise Refusal("%s: %r is not spelled as in the repo; manifest paths must match the repo's exact case"
                      % (src_rel, wrong))
    if not inside(os.path.realpath(src_abs), REPO_ROOT):
        raise Refusal("%s resolves outside the repo" % src_rel)
    return src_abs


def manifest_entry(kind, src_rel):
    """Map one manifest entry to (group, dst_rel); raise Refusal if it is outside the allowed set."""
    clean_rel(src_rel, "manifest path")
    reason = denial_reason(src_rel)
    if reason:
        raise Refusal("%s: %s" % (src_rel, reason))
    if kind == GROUP_ADDON:
        if not src_rel.startswith(ADDON_SRC_DIR):
            raise Refusal("%s: 'addon' entries must live under %s" % (src_rel, ADDON_SRC_DIR))
        if not src_rel.lower().endswith(ADDON_EXTS):
            raise Refusal("%s: 'addon' entries must be one of %s (scripts use 'script')"
                          % (src_rel, " ".join(ADDON_EXTS)))
        return GROUP_ADDON, src_rel[len(ADDON_SRC_DIR):]
    if kind == GROUP_SCRIPT:
        name = src_rel[len(SCRIPT_SRC_DIR):] if src_rel.startswith(SCRIPT_SRC_DIR) else ""
        if not name or "/" in name or not SCRIPT_NAME_RE.match(name):
            raise Refusal("%s: 'script' entries must be %s<Name>.c (top level only)" % (src_rel, SCRIPT_SRC_DIR))
        return GROUP_SCRIPT, SCRIPT_DST_DIR + "/" + name
    if kind == GROUP_CONFIG:
        if not src_rel.endswith(CONFIG_EXT):
            raise Refusal("%s: 'config' entries are %s files" % (src_rel, CONFIG_EXT))
        for src_root, dst_root in CONFIG_ROOTS:
            if src_rel.startswith(src_root):
                rest = src_rel[len(src_root):]
                if not all(CONFIG_NAME_RE.match(part) for part in rest.split("/")):
                    raise Refusal("%s: config path parts must be [A-Za-z0-9._-]" % src_rel)
                return GROUP_CONFIG, dst_root + rest
        raise Refusal("%s: 'config' entries must live under %s" % (src_rel, " or ".join(r for r, _ in CONFIG_ROOTS)))
    raise Refusal("unknown manifest kind %r for %s (expected 'addon', 'script' or 'config')" % (kind, src_rel))


def read_manifest(path, refusals):
    """Return a list of plan items {group, src_rel, src_abs, dst} for every valid manifest line."""
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        refusals.append("cannot read manifest %s: %s" % (path, e))
        return items
    for n, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            refusals.append("%s:%d: expected '<kind> <path>', got %r" % (path, n, raw.strip()))
            continue
        kind, src_rel = fields
        try:
            group, dst = manifest_entry(kind, src_rel)
            src_abs = resolve_source(src_rel)
        except Refusal as e:
            refusals.append("%s:%d: %s" % (os.path.basename(path), n, e))
            continue
        items.append({"group": group, "src_rel": src_rel, "src_abs": src_abs, "dst": dst})
    if not items and not refusals:
        refusals.append("manifest %s lists no files" % path)
    return items


def emcp_items(handler_dir, refusals):
    items = []
    if not os.path.isdir(handler_dir):
        refusals.append("--emcp-handlers %s is not a directory" % handler_dir)
        return items
    skipped = []
    for name in sorted(os.listdir(handler_dir)):
        path = os.path.join(handler_dir, name)
        if EMCP_NAME_RE.match(name) and os.path.isfile(path) and not os.path.islink(path):
            items.append({"group": GROUP_EMCP, "src_rel": os.path.abspath(path), "src_abs": path,
                          "dst": EMCP_DST_DIR + "/" + name})
        else:
            skipped.append(name)
    if not items:
        refusals.append("--emcp-handlers %s contains no EMCP_WB_*.c files" % handler_dir)
    if skipped:
        print("note: --emcp-handlers ignores %d non-handler entries: %s" % (len(skipped), ", ".join(skipped[:8])))
    return items


def existing_ancestor(path):
    """The longest prefix of `path` that exists (the path itself if it exists)."""
    probe = path
    while not os.path.lexists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return probe


def forbidden_device_reason(real, what):
    """Why `real` (or its nearest existing ancestor) is on a forbidden device (C:, the WSL disk image), or None."""
    try:
        dev = os.stat(existing_ancestor(real)).st_dev
    except OSError:
        return None
    for path, label in FORBIDDEN_DEVICE_PATHS:
        try:
            if os.path.isdir(path) and os.stat(path).st_dev == dev:
                return "%s %s is on the same device as %s (%s); deploys go to E:" % (what, real, path, label)
        except OSError:
            pass
    return None


def allow_root_reason(raw):
    """Why `raw` cannot be an extra --allow-root (it must be an existing directory under /mnt/e/ or /dev/shm/)."""
    real = os.path.realpath(os.path.abspath(raw))
    if not any(real.startswith(prefix) and len(real) > len(prefix) for prefix in ALLOW_ROOT_PREFIXES):
        return "--allow-root %s resolves to %s, which is not inside %s" % (raw, real, " or ".join(ALLOW_ROOT_PREFIXES))
    if not os.path.isdir(real):
        return "--allow-root %s is not an existing directory" % raw
    return forbidden_device_reason(real, "--allow-root")


def target_reason(raw, allow_roots=()):
    """Return why `raw` is not an acceptable target, or None.

    Refused: Windows syntax, the C: drive or the WSL disk image (by path or by device), filesystem or drive roots,
    anything overlapping the repo, and everything that does not resolve strictly inside ALLOWED_TARGET_ROOTS or an
    accepted --allow-root."""
    if re.match(r"^[A-Za-z]:", raw) or raw.startswith("\\\\"):
        if raw[:1].lower() == "c":
            return "target %r is on the C: drive; deploys go to E: (/mnt/e/...)" % raw
        return "target %r uses Windows syntax; pass the WSL form, e.g. /mnt/e/DriveDiT/addons/DataCapture" % raw
    real = os.path.realpath(os.path.abspath(raw))
    low = real.lower()
    if low == "/mnt/c" or low.startswith("/mnt/c/"):
        return "target %s resolves to the C: drive (%s); deploys go to E:" % (raw, real)
    reason = forbidden_device_reason(real, "target")
    if reason:
        return reason
    if real == "/" or re.match(r"^/mnt/[A-Za-z]$", real):
        return "target %s is a filesystem or drive root" % raw
    if inside(real, REPO_ROOT) or inside(REPO_ROOT, real):
        return "target %s overlaps the repo %s" % (raw, REPO_ROOT)
    roots = [os.path.realpath(r) for r in tuple(ALLOWED_TARGET_ROOTS) + tuple(allow_roots)]
    if not any(real != root and inside(real, root) for root in roots):
        return ("target %s resolves to %s, which is not inside an allowed deploy root (%s); the capture addon "
                "deploys to %s" % (raw, real, ", ".join(roots), DEFAULT_TARGET))
    return None


def escape_reason(target, rel):
    """Why writing or deleting <target>/<rel> could reach outside the target, or None.

    Every existing component below the target must be a real directory or file, not a symlink or junction, and the
    deepest existing component must resolve inside the target."""
    cur = target
    for part in rel.split("/"):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            return ("%s: %s inside the target is a symlink or junction; refusing to write or delete through it"
                    % (rel, os.path.relpath(cur, target).replace(os.sep, "/")))
        if not os.path.exists(cur):
            break
    probe = existing_ancestor(cur)
    if inside(probe, target) and not inside(os.path.realpath(probe), target):
        return "%s: %s resolves outside the target (%s)" % (rel, probe, os.path.realpath(probe))
    return None


def git(args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    out = subprocess.run(["git", "-C", REPO_ROOT, "--no-optional-locks"] + args, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out.stdout.decode("utf-8", "replace")


def git_info(source_rels):
    """(sha, branch, dirty). dirty: tracked changes anywhere, or a deployed source that is untracked."""
    try:
        sha = git(["rev-parse", "HEAD"]).strip()
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        tracked_changes = git(["status", "--porcelain", "--untracked-files=no"]).strip()
        untracked_sources = git(["status", "--porcelain", "--untracked-files=all", "--"] + source_rels).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        print("warning: git metadata unavailable (%s); stamping GIT_SHA=\"\" and GIT_DIRTY=true" % e)
        return "", "", True
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        sha = ""
    branch = re.sub(r"[^A-Za-z0-9._/\-]", "_", branch)
    return sha, branch, bool(tracked_changes or untracked_sources)


def gproj_identity(items):
    for it in items:
        if it["dst"] == "addon.gproj":
            with open(it["src_abs"], "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            ident = re.search(r'^\s*ID\s+"([A-Za-z0-9_]+)"', text, re.M)
            guid = re.search(r'^\s*GUID\s+"([0-9A-Fa-f]{16})"', text, re.M)
            return (ident.group(1) if ident else ""), (guid.group(1).upper() if guid else "")
    return "", ""


def content_hash(file_shas):
    """sha256 over '<sha256>  <path>' lines (sorted) of the versioned addon and script files."""
    lines = "".join("%s  %s\n" % (file_shas[p], p) for p in sorted(file_shas))
    return sha256_bytes(lines.encode("utf-8")), len(file_shas)


def enforce_bool(value):
    return "true" if value else "false"


def render_buildinfo(info):
    b = enforce_bool
    lines = [
        "// GENERATED by %s on every deploy. Do not edit and do not commit." % TOOL_REL,
        "// Provenance of the deployed DataCapture addon (CAPTURE_PROGRAM TKT-1 build stamping).",
        "// Enforce Script cannot run git, so the deploy step writes these values for the session metadata writers.",
        "class DC_BuildInfo",
        "{",
        '\tstatic const string ADDON_ID = "%s";' % info["addon_id"],
        '\tstatic const string ADDON_GUID = "%s";' % info["addon_guid"],
        '\tstatic const string GIT_SHA = "%s";' % info["git_sha"],
        '\tstatic const string GIT_BRANCH = "%s";' % info["git_branch"],
        "\tstatic const bool GIT_DIRTY = %s;" % b(info["git_dirty"]),
        '\tstatic const string DEPLOY_UTC = "%s";' % info["deploy_utc"],
        '\tstatic const string MANIFEST_SHA256 = "%s";' % info["manifest_sha256"],
        '\tstatic const string CONTENT_SHA256 = "%s";' % info["content_sha256"],
        "\tstatic const int CONTENT_FILE_COUNT = %d;" % info["content_file_count"],
        "\tstatic const bool DEPLOY_PARTIAL = %s;" % b(info["partial"]),
        "}",
        "",
    ]
    return "\n".join(lines).encode("ascii")


def load_state(target, refusals):
    path = os.path.join(target, STATE_NAME)
    if not os.path.exists(path):
        return {"schema": STATE_SCHEMA, "files": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict) or not isinstance(state.get("files"), dict):
            raise ValueError("missing 'files' object")
        for rel, meta in state["files"].items():
            clean_rel(rel, "state path")
            if not isinstance(meta, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256", ""))):
                raise ValueError("bad entry for %s" % rel)
    except (OSError, ValueError, Refusal) as e:
        refusals.append("cannot use %s (%s); fix or remove it by hand" % (path, e))
        return {"schema": STATE_SCHEMA, "files": {}}
    return state


def current_sha(target, rel, refusals):
    path = os.path.join(target, *rel.split("/"))
    if os.path.islink(path) or (os.path.exists(path) and not os.path.isfile(path)):
        refusals.append("%s exists in the target but is not a regular file" % rel)
        return None
    return sha256_file(path) if os.path.isfile(path) else None


def atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), ".%s.deploy-tmp-%d" % (os.path.basename(path), os.getpid()))
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backup_rel(stamp, rel):
    return "%s/%s/%s.bak" % (BACKUP_DIR, stamp, rel)


def backup(target, stamp, rel):
    src = os.path.join(target, *rel.split("/"))
    dst_rel = backup_rel(stamp, rel)
    with open(src, "rb") as f:
        atomic_write(os.path.join(target, *dst_rel.split("/")), f.read())
    return dst_rel


def normalize_request(arg):
    rel = arg.replace("\\", "/")
    if os.path.isabs(rel):
        real = os.path.realpath(rel)
        if inside(real, REPO_ROOT):
            rel = os.path.relpath(real, REPO_ROOT).replace(os.sep, "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", metavar="FILE",
                    help="deploy only these manifest entries (repo-relative source paths); never deletes")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="addon root to deploy into (default %(default)s)")
    ap.add_argument("--allow-root", action="append", default=[], metavar="DIR",
                    help="also accept targets strictly inside DIR, an existing directory under %s (scratch deploys, "
                         "self-test); may be repeated" % " or ".join(ALLOW_ROOT_PREFIXES))
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="deploy manifest (default %(default)s)")
    ap.add_argument("--emcp-handlers", metavar="DIR",
                    help="also deploy DIR/EMCP_WB_*.c into %s/" % EMCP_DST_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite or delete files changed outside this tool, after backing them up")
    ap.add_argument("--require-clean", action="store_true", help="refuse when the git tree is dirty")
    args = ap.parse_args(argv)

    refusals = []
    allow_roots = []
    for root in args.allow_root:
        why = allow_root_reason(root)
        if why:
            refusals.append(why)
        else:
            allow_roots.append(root)
    reason = target_reason(args.target, allow_roots)
    if reason:
        refusals.append(reason)
    target = os.path.realpath(os.path.abspath(args.target))
    if not reason and not os.path.isdir(os.path.dirname(target)):
        refusals.append("the parent of target %s does not exist; create it first" % target)
    if not reason:
        why = escape_reason(target, STATE_NAME)
        if why:
            refusals.append(why)
            reason = why

    manifest_path = os.path.abspath(args.manifest)
    items = read_manifest(manifest_path, refusals)

    all_items = list(items)
    partial = bool(args.files)
    if partial:
        by_src = {it["src_rel"]: it for it in items}
        chosen = []
        for arg in args.files:
            rel = normalize_request(arg)
            if rel in by_src:
                chosen.append(by_src[rel])
                continue
            why = denial_reason(rel)
            refusals.append("%s is not in the deploy manifest%s" % (arg, "; " + why if why else ""))
        items = chosen

    if args.emcp_handlers:
        items += emcp_items(os.path.abspath(args.emcp_handlers), refusals)

    seen = {}
    for it in items:
        dst = it["dst"]
        if is_wb_generated(dst) or is_reserved_dst(dst) or dst == BUILDINFO_DST:
            refusals.append("%s would deploy to the reserved or Workbench-generated path %s" % (it["src_rel"], dst))
        if dst.lower() in seen:
            refusals.append("%s and %s both deploy to %s" % (seen[dst.lower()], it["src_rel"], dst))
        seen[dst.lower()] = it["src_rel"]

    state = load_state(target, refusals) if not reason else {"schema": STATE_SCHEMA, "files": {}}
    old_files = state["files"]
    _, manifest_guid = gproj_identity([it for it in all_items if it["group"] == GROUP_ADDON])
    if state.get("addon_guid") and manifest_guid and state["addon_guid"] != manifest_guid:
        refusals.append("the target was deployed for addon GUID %s, but the manifest's addon.gproj has GUID %s; "
                        "refusing to mix two addons in one folder" % (state["addon_guid"], manifest_guid))

    repo_sources = [it["src_rel"] for it in items if it["group"] in VERSIONED_GROUPS]
    if inside(manifest_path, REPO_ROOT):
        repo_sources.append(os.path.relpath(manifest_path, REPO_ROOT).replace(os.sep, "/"))
    sha, branch, dirty = git_info(repo_sources) if repo_sources else ("", "", True)
    if args.require_clean and dirty:
        refusals.append("--require-clean: the repo has uncommitted changes or untracked deployed sources")

    if refusals:
        return refuse(refusals)

    # Build the new content of every planned file (copies first, then the build stamp).
    payload = {}
    for it in items:
        with open(it["src_abs"], "rb") as f:
            payload[it["dst"]] = f.read()

    versioned = {rel: meta["sha256"] for rel, meta in old_files.items()
                 if meta.get("group") in VERSIONED_GROUPS} if partial else {}
    for it in items:
        if it["group"] in VERSIONED_GROUPS:
            versioned[it["dst"]] = sha256_bytes(payload[it["dst"]])
    content_sha, content_count = content_hash(versioned)
    addon_id, addon_guid = gproj_identity([it for it in all_items if it["group"] == GROUP_ADDON])
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    info = {
        "addon_id": addon_id, "addon_guid": addon_guid, "git_sha": sha, "git_branch": branch,
        "git_dirty": dirty, "deploy_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": sha256_file(manifest_path), "content_sha256": content_sha,
        "content_file_count": content_count, "partial": partial,
    }
    payload[BUILDINFO_DST] = render_buildinfo(info)
    plan = [(it["group"], it["src_rel"], it["dst"]) for it in items]
    plan.append((GROUP_BUILDINFO, "(generated)", BUILDINFO_DST))

    # Decide every action before touching the target.
    actions = []  # (op, rel, group, source, backup_needed)
    for group, src, dst in plan:
        why = escape_reason(target, dst)
        if why:
            refusals.append(why)
            continue
        cur = current_sha(target, dst, refusals)
        new = sha256_bytes(payload[dst])
        if cur is None:
            actions.append(("create", dst, group, src, False))
        elif cur == new:
            actions.append(("same", dst, group, src, False))
        elif dst in old_files and old_files[dst]["sha256"] == cur:
            actions.append(("update", dst, group, src, False))
        elif args.force:
            actions.append(("update", dst, group, src, True))
        else:
            why = "changed since the last deploy" if dst in old_files else "was not deployed by this tool"
            refusals.append("%s exists in the target and %s (Workbench save or hand edit?); refusing to "
                            "overwrite it. Inspect it, then rerun with --force to back it up and replace it" % (dst, why))

    if not partial:
        active = set(VERSIONED_GROUPS) | {GROUP_BUILDINFO}
        if args.emcp_handlers:
            active.add(GROUP_EMCP)
        planned = {dst for _, _, dst in plan}
        for rel in sorted(old_files):
            meta = old_files[rel]
            if rel in planned or meta.get("group") not in active:
                continue
            if is_wb_generated(rel) or is_reserved_dst(rel):
                print("note: state lists %s, which this tool never deletes; dropping it from the state" % rel)
                actions.append(("forget", rel, meta.get("group"), meta.get("source", ""), False))
                continue
            why = escape_reason(target, rel)
            if why:
                refusals.append(why)
                continue
            cur = current_sha(target, rel, refusals)
            if cur is None:
                actions.append(("forget", rel, meta.get("group"), meta.get("source", ""), False))
            elif cur == meta["sha256"]:
                actions.append(("remove", rel, meta.get("group"), meta.get("source", ""), False))
            elif args.force:
                actions.append(("remove", rel, meta.get("group"), meta.get("source", ""), True))
            else:
                refusals.append("%s is no longer in the manifest but changed since it was deployed; refusing to "
                                "delete it. Rerun with --force to back it up and delete it" % rel)

    for op, rel, group, src, needs_backup in actions:
        if needs_backup:
            why = escape_reason(target, backup_rel(stamp, rel))
            if why:
                refusals.append(why)

    if refusals:
        return refuse(refusals)

    print("deploy %s -> %s%s" % (os.path.relpath(manifest_path, REPO_ROOT) if inside(manifest_path, REPO_ROOT)
                                 else manifest_path, target, "  (DRY RUN)" if args.dry_run else ""))
    print("git %s %s%s" % (sha or "<none>", branch or "<none>", " DIRTY" if dirty else ""))
    for op, rel, group, src, needs_backup in actions:
        print("  %-6s %-8s %s%s%s" % (op, group, rel, "  <- " + src if op in ("create", "update", "same") else "",
                                      "  (backup first)" if needs_backup else ""))
    if args.dry_run:
        return 0

    new_files = dict(old_files)
    failed = None
    rel = target
    try:
        os.makedirs(target, exist_ok=True)
        for op, rel, group, src, needs_backup in actions:
            path = os.path.join(target, *rel.split("/"))
            if needs_backup:
                print("  backup %s -> %s" % (rel, backup(target, stamp, rel)))
            if op in ("create", "update"):
                atomic_write(path, payload[rel])
            elif op == "remove":
                os.remove(path)
            if op in ("create", "update", "same"):
                new_files[rel] = {"sha256": sha256_bytes(payload[rel]), "group": group, "source": src}
            else:
                new_files.pop(rel, None)
    except OSError as e:
        failed = "%s: %s" % (rel, e)

    state_out = {
        "schema": STATE_SCHEMA, "tool": TOOL_REL, "addon_id": addon_id, "addon_guid": addon_guid,
        "last_deploy_utc": info["deploy_utc"], "git_sha": sha, "git_branch": branch, "git_dirty": dirty,
        "manifest_sha256": info["manifest_sha256"], "content_sha256": content_sha, "partial": partial,
        "files": new_files,
    }
    try:
        atomic_write(os.path.join(target, STATE_NAME),
                     (json.dumps(state_out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    except OSError as e:
        failed = (failed + "; " if failed else "") + "state: %s" % e
    if failed:
        print("ERROR: deploy stopped: %s" % failed, file=sys.stderr)
        return 1
    counts = {}
    for op, *_ in actions:
        counts[op] = counts.get(op, 0) + 1
    print("done: " + ", ".join("%d %s" % (n, op) for op, n in sorted(counts.items())))
    return 0


def refuse(refusals):
    for r in refusals:
        print("REFUSED: %s" % r, file=sys.stderr)
    print("deploy refused (%d problem%s); nothing was written" % (len(refusals), "" if len(refusals) == 1 else "s"),
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
