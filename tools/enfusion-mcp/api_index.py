#!/usr/bin/env python3
"""Build the Enfusion MCP API index of one pin from the official script dump (ticket DD-14a, CC-05; MIGRATION_1_9 6.3,
8.2 component 6; API_DELTA 6.3 P19-3).

BK enfusion-mcp loads its API index from <data dir>/api/{enfusion-classes,arma-classes,groups,hierarchy}.json. The
only index on the machine is the 1.8 scrape (fma-rc 4cd14dbb), which knows nothing of 1.9. This tool builds an index
in the same JSON shape from Bohemia's Arma-Reforger-Script-Diff dump at the pin's tag, and stamps it with the pin:

    <out>/api/enfusion-classes.json   classes of Core, GameLib, Workbench, WorkbenchCommon  (source "enfusion")
    <out>/api/arma-classes.json       classes of Game, GameCode, WorkbenchGame, ...          (source "arma")
    <out>/api/groups.json, hierarchy.json
    <out>/api/INDEX_PIN.json          pin_id, pin sha256, build_key, the dump tag and commit, file hashes
    <out>/{patterns,recipes,wiki}/    copied from --shared (BK reads them from the same data dir)

The wrapper (enfusion-mcp-wsl.sh, mcp_pin.py) serves <mcp root>/pins/<pin id>/data only when INDEX_PIN.json names
that pin and its build_key, so a 1.8 index can never answer for a 1.9 pin.

    python3 -I -B api_index.py build --dump-git /mnt/e/drivedit_realign/program/1_9/work/dl/sd_stable \
        --pin-id 1.9.0.37 --out /mnt/e/drivedit_realign/mcp/pins/1.9.0.37/data \
        --shared /mnt/e/drivedit_realign/mcp/data --dry-run          # then --apply

Provenance (Codex round 1): index_sources.json maps each pin id to a dump tag and its commit. A build refuses unless
the pin is mapped, its engine sidecar's engine_version_expected gives the same tag (v<version>), and the tag resolves
to the mapped commit in the clone; the stamp records them, and mcp_gate.py checks the stamp against the map again.
So a 1.8 dump can never be stamped as a 1.9 pin. The dump is read with `git ls-tree` and `git cat-file --batch` at
the tag (never a checkout). The index is written to a temp dir and renamed into place. The parser is a declaration scanner, not a compiler: classes (parent from ':' or
'extends'), methods (multi-line declarations joined; public, protected/private, static), member variables and enums,
with their '//!' doc lines. Bodies are skipped by brace depth.

Exit codes: 0 built (or dry run); 2 usage, read or path error. Writes only under /mnt/e (or the test roots).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

SCHEMA = "dd-mcp-index/1"
API_FILES = ("enfusion-classes.json", "arma-classes.json", "groups.json", "hierarchy.json")
SHARED_DIRS = ("patterns", "recipes", "wiki")
ENFUSION_MODULES = ("Core", "GameLib", "Workbench", "WorkbenchCommon")

CLASS_RE = re.compile(r"^\s*(?:sealed\s+|modded\s+)*class\s+(\w+)(?:\s*<[^>]*>)?\s*(?:(?::|\bextends\b)\s*(\w+))?")
ENUM_RE = re.compile(r"^\s*enum\s+(\w+)(?:\s*:\s*\w+)?")
MODIFIERS = {"proto", "external", "native", "static", "protected", "private", "override", "event", "sealed",
             "volatile", "notnull", "ref", "owned", "const", "autoptr", "out", "inout", "local", "reference"}
KEYWORDS = {"if", "for", "while", "switch", "return", "foreach", "new", "delete", "else", "case", "typedef", "class",
            "enum", "break", "continue", "thread"}
GROUP_RE = re.compile(r"\\addtogroup\s+(\w+)")


def _load_pinlib():
    """tools/pin/pinlib.py from the repo layout, else the installed copy (E:\\DriveDiT\\tools\\pin)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("DD_PIN_TOOLS"), os.path.join(os.path.dirname(here), "pin"), "/mnt/e/DriveDiT/tools/pin"):
        if cand and os.path.isfile(os.path.join(cand, "pinlib.py")):
            mods = {}
            for name in ("pinlib", "preflight"):
                path = os.path.join(cand, name + ".py")
                modname = "dd_pin_" + name
                mod = sys.modules.get(modname)
                if mod is None or getattr(mod, "__file__", None) != path:
                    spec = importlib.util.spec_from_file_location(modname, path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[modname] = mod
                    spec.loader.exec_module(mod)
                mods[name] = mod
            return mods["pinlib"], mods["preflight"]
    raise SystemExit("api_index: cannot find tools/pin (set DD_PIN_TOOLS)")


# --------------------------------------------------------------------------- reading the dump


def read_tree_git(repo: str, tag: str, prefix: str = "scripts/"):
    """(commit, [(path relative to prefix, text)]) of every *.c under prefix at tag, read from git objects only."""
    def git(*args, **kw):
        return subprocess.run(["git", "--no-optional-locks", "-C", repo] + list(args), check=True,
                              capture_output=True, **kw).stdout
    commit = git("rev-parse", "--verify", "%s^{commit}" % tag).decode().strip()
    listing = git("ls-tree", "-r", "-z", commit, "--", prefix.rstrip("/")).decode("utf-8")
    entries = []
    for item in listing.split("\0"):
        if not item:
            continue
        meta, path = item.split("\t", 1)
        mode, kind, oid = meta.split()
        if kind == "blob" and path.endswith(".c") and path.startswith(prefix):
            entries.append((path[len(prefix):], oid))
    batch = "".join(oid + "\n" for _, oid in entries).encode("ascii")
    out = git("cat-file", "--batch", input=batch)
    files, pos = [], 0
    for rel, oid in entries:
        nl = out.index(b"\n", pos)
        head = out[pos:nl].decode("ascii").split()
        if head[0] != oid or head[1] != "blob":
            raise ValueError("git cat-file answered %s for %s" % (head, rel))
        size = int(head[2])
        data = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
        files.append((rel, data.decode("utf-8", errors="replace")))
    return commit, files


def read_tree_dir(root: str):
    files = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            if fn.endswith(".c"):
                full = os.path.join(dp, fn)
                with open(full, encoding="utf-8", errors="replace") as fh:
                    files.append((os.path.relpath(full, root).replace(os.sep, "/"), fh.read()))
    return files


# --------------------------------------------------------------------------- parsing


def _blank_strings_and_comments(text: str):
    """(masked, raw, docs). masked: comments and string contents blanked, for the structure (braces, parentheses,
    '=' and ';' inside a string never count). raw: only comments blanked, string literals kept, so declarations and
    their defaults ("0 0 0") are taken from it. Both keep every character position and newline, so a span found in
    masked is the same span in raw. docs maps a line number to its '//!' doc text."""
    masked, raw, docs = [], [], {}
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            if text.startswith("//!", i):
                docs.setdefault(line, []).append(text[i + 3:j].strip())
            masked.append(" " * (j - i))
            raw.append(" " * (j - i))
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            chunk = text[i:j]
            blank = "".join("\n" if ch == "\n" else " " for ch in chunk)
            masked.append(blank)
            raw.append(blank)
            line += chunk.count("\n")
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            masked.append('"' + " " * max(0, j - i - 2) + ('"' if j - i >= 2 else ""))
            raw.append(text[i:j])
            i = j
        else:
            if c == "\n":
                line += 1
            masked.append(c)
            raw.append(c)
            i += 1
    return "".join(masked), "".join(raw), docs


def _doc_before(docs: dict, lineno: int) -> str:
    lines, k = [], lineno - 1
    while k in docs:
        lines[:0] = docs[k]
        k -= 1
    return " ".join(x for x in lines if x)


def _norm(s: str) -> str:
    return " ".join(s.split())


def _split_params(params_m: str, params_r: str):
    """ParamInfo list; commas and '=' are found in the masked text, names, types and defaults taken from the raw."""
    spans, depth, start = [], 0, 0
    for k, ch in enumerate(params_m):
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == "," and depth == 0:
            spans.append((start, k))
            start = k + 1
    spans.append((start, len(params_m)))
    res = []
    for a, b in spans:
        pm, pr = params_m[a:b], params_r[a:b]
        if not pm.strip():
            continue
        e = pm.find("=")
        default = pr[e + 1:].strip() if e >= 0 else ""
        toks = (pm[:e] if e >= 0 else pm).split()
        if not toks:
            continue
        res.append({"name": toks[-1].rstrip("[]"), "type": " ".join(toks[:-1]), "defaultValue": default})
    return res


_METHOD_RE = re.compile(r"^(?P<head>.*?)\b(?P<name>\w+)\s*\((?P<params>.*)\)\s*(?P<tail>[\w\s]*)$", re.S)
_PROPERTY_RE = re.compile(r"^(?P<head>[\w<>,\[\]\s]*?)\b(?P<name>\w+)\s*(?:\[[^\]]*\])?\s*(?P<init>=.*)?;\s*$", re.S)


def _method(decl_m: str, decl_r: str, doc: str):
    """(kind, MethodInfo) from a declaration 'mods type Name(params) [const]' (masked and raw), or (None, None)."""
    m = _METHOD_RE.match(decl_m)
    if not m or m.group("name") in KEYWORDS:
        return None, None
    head = _norm(m.group("head"))
    toks = head.split()
    if any(t in KEYWORDS for t in toks):
        return None, None
    params = _split_params(m.group("params"), decl_r[m.start("params"):m.end("params")])
    plist = ", ".join(_norm("%s %s" % (p["type"], p["name"])) + (" = %s" % p["defaultValue"] if p["defaultValue"]
                                                                  else "") for p in params)
    tail = _norm(m.group("tail"))
    signature = "%s %s (%s)%s" % (head, m.group("name"), plist, (" " + tail) if tail else "")
    kind = "static" if "static" in toks else ("protected" if ("protected" in toks or "private" in toks) else "public")
    return kind, {"name": m.group("name"), "returnType": head, "signature": signature.strip(), "params": params,
                  "description": doc}


def _property(decl_m: str, decl_r: str, doc: str):
    """(kind, PropertyInfo) from a member variable 'mods type name [= init];' (masked and raw), or (None, None).
    The initializer, which may hold parentheses ('= new Color(0, 0, 0, 1)'), is kept as defaultValue."""
    m = _PROPERTY_RE.match(decl_m)
    if not m or not m.group("head").strip() or m.group("name") in KEYWORDS:
        return None, None
    toks = m.group("head").split()
    if any(t in KEYWORDS for t in toks):
        return None, None
    kind = "protected" if ("protected" in toks or "private" in toks) else "public"
    default = decl_r[m.start("init") + 1:m.end("init")].strip() if m.group("init") else ""
    return kind, {"name": m.group("name"), "type": " ".join(t for t in toks if t not in MODIFIERS) or _norm(
        m.group("head")), "defaultValue": default, "description": doc}


def _enum_values(body: str, docs: dict, first_line: int):
    vals = []
    line = first_line
    for part in body.split(","):
        stripped = part.strip()
        line += part.count("\n")
        if not stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if re.fullmatch(r"\w+", name):
            vals.append({"name": name, "value": value.strip(), "description": " ".join(docs.get(line, []))})
    return vals


def parse_file(rel: str, text: str) -> list:
    """Top-level classes and enums of one script file, as BK ClassInfo dicts (enums as classes with one EnumInfo)."""
    module = rel.split("/", 1)[0]
    source = "enfusion" if module in ENFUSION_MODULES else "arma"
    gm = GROUP_RE.search(text)
    group = gm.group(1) if gm else module
    code, raw_code, docs = _blank_strings_and_comments(text)
    lines = code.split("\n")
    raw_lines = raw_code.split("\n")
    out = []
    depth = 0
    cur = None              # the class being read
    pending = None          # a class/enum header seen, waiting for its '{'
    acc, acc_r, acc_line = "", "", 0   # a multi-line member declaration, masked and raw (equal lengths)
    enum_acc = None
    for lineno, (line, line_r) in enumerate(zip(lines, raw_lines), 1):
        lead = len(line) - len(line.lstrip())
        end = len(line.rstrip())
        stripped, stripped_r = line[lead:end], line_r[lead:end]
        if enum_acc is not None:
            enum_acc["body"] += line + "\n"
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                depth = 0
                body = enum_acc["body"]
                body = body[body.index("{") + 1:body.rindex("}")] if "{" in body and "}" in body else ""
                info = enum_acc["info"]
                info["enums"] = [{"name": info["name"], "description": info["description"],
                                  "values": _enum_values(body, docs, enum_acc["line"])}]
                out.append(info)
                enum_acc = None
            continue
        if depth == 0 and cur is None:
            cm = CLASS_RE.match(line)
            em = ENUM_RE.match(line) if not cm else None
            if cm:
                cur = {"name": cm.group(1), "source": source, "brief": _doc_before(docs, lineno),
                       "description": _doc_before(docs, lineno), "parents": [cm.group(2)] if cm.group(2) else [],
                       "children": [], "group": group, "sourceFile": "scripts/" + rel, "methods": [],
                       "protectedMethods": [], "staticMethods": [], "enums": [], "properties": [],
                       "protectedProperties": [], "docsUrl": ""}
                pending = "class"
            elif em:
                info = {"name": em.group(1), "source": source, "brief": _doc_before(docs, lineno),
                        "description": _doc_before(docs, lineno), "parents": [], "children": [], "group": group,
                        "sourceFile": "scripts/" + rel, "methods": [], "protectedMethods": [], "staticMethods": [],
                        "enums": [], "properties": [], "protectedProperties": [], "docsUrl": ""}
                enum_acc = {"info": info, "body": line + "\n", "line": lineno}
                depth += line.count("{") - line.count("}")
                if "{" in line and depth <= 0:  # one-line enum
                    depth = 0
                    body = line[line.index("{") + 1:line.rindex("}")] if "}" in line else ""
                    info["enums"] = [{"name": info["name"], "description": info["description"],
                                      "values": _enum_values(body, docs, lineno)}]
                    out.append(info)
                    enum_acc = None
                continue
        if cur is not None and depth == 1 and pending is None:
            if acc:
                acc += " " + stripped
                acc_r += " " + stripped_r
            elif stripped and not stripped.startswith(("{", "}", "#")):
                acc, acc_r, acc_line = stripped, stripped_r, lineno
            if acc:
                opened = acc.count("(") - acc.count(")") + acc.count("[") - acc.count("]")
                pre = re.match(r"^(?:\[[^\]]*\]\s*)+", acc) if opened <= 0 else None
                cut = pre.end() if pre else 0
                bare, bare_r = acc[cut:], acc_r[cut:]
                paren, eq = bare.find("("), bare.find("=")
                if opened > 0:
                    pass                                    # a declaration or attribute that goes on
                elif not bare.strip():
                    acc = acc_r = ""                        # an attribute line; the member follows
                elif paren >= 0 and (eq < 0 or paren < eq):
                    # a method: its parameter list comes before any '=' (an initializer with parentheses,
                    # 'Color Black = new Color(...)', is a member variable)
                    k = bare.find("{")
                    decl = bare[:k] if k >= 0 else bare
                    endk = len(decl.rstrip().rstrip(";").rstrip())
                    kind, info = _method(decl[:endk], bare_r[:endk], _doc_before(docs, acc_line))
                    if info is not None:
                        cur[{"public": "methods", "protected": "protectedMethods", "static": "staticMethods"}[kind]] \
                            .append(info)
                    acc = acc_r = ""
                elif bare.rstrip().endswith(";"):
                    kind, info = _property(bare, bare_r, _doc_before(docs, acc_line))
                    if info is not None:
                        cur["properties" if kind == "public" else "protectedProperties"].append(info)
                    acc = acc_r = ""
                elif "{" in bare or "}" in bare:
                    acc = acc_r = ""
        opens, closes = line.count("{"), line.count("}")
        if pending and opens:
            pending = None
        depth += opens - closes
        if depth <= 0:
            depth = 0
            if cur is not None and closes and pending is None:
                out.append(cur)
                cur = None
                acc = acc_r = ""
    if cur is not None:
        out.append(cur)
    return out


def build_index(files) -> dict:
    """{"enfusion": [...], "arma": [...], "groups": [...], "hierarchy": [...]} from [(rel, text)]."""
    by_name = {}
    for rel, text in files:
        for c in parse_file(rel, text):
            prev = by_name.get(c["name"])
            if prev is None:
                by_name[c["name"]] = c
            else:  # a class split over several files: merge members
                for key in ("methods", "protectedMethods", "staticMethods", "enums", "properties",
                            "protectedProperties"):
                    prev[key].extend(c[key])
                if not prev["parents"] and c["parents"]:
                    prev["parents"] = c["parents"]
    for c in by_name.values():
        for p in c["parents"]:
            if p in by_name and c["name"] not in by_name[p]["children"]:
                by_name[p]["children"].append(c["name"])
    classes = sorted(by_name.values(), key=lambda c: c["name"])
    for c in classes:
        c["children"].sort()
    groups = {}
    for c in classes:
        groups.setdefault(c["group"], []).append(c["name"])
    return {
        "enfusion": [c for c in classes if c["source"] == "enfusion"],
        "arma": [c for c in classes if c["source"] == "arma"],
        "groups": [{"name": g, "description": "", "classes": names} for g, names in sorted(groups.items())],
        "hierarchy": [{"name": c["name"], "children": c["children"]} for c in classes],
    }


def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=False) + "\n").encode("utf-8")


def render(index: dict) -> dict:
    """{file name: bytes} of the api/ files."""
    return {"enfusion-classes.json": _dump(index["enfusion"]), "arma-classes.json": _dump(index["arma"]),
            "groups.json": _dump(index["groups"]), "hierarchy.json": _dump(index["hierarchy"])}


SOURCES_SCHEMA = "dd-mcp-index-sources/1"
DEFAULT_SOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_sources.json")


def load_sources(path=None) -> dict:
    """index_sources.json: {pin id: {tag, commit}}. ValueError when unreadable."""
    path = path or DEFAULT_SOURCES
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read the index source map %s: %s" % (path, exc))
    if not isinstance(data, dict) or data.get("schema") != SOURCES_SCHEMA or not isinstance(data.get("pins"), dict):
        raise ValueError("%s is not a %s file" % (path, SOURCES_SCHEMA))
    return data["pins"]


def expected_source(pin_id: str, ident: dict, sources: dict) -> dict:
    """{tag, commit} the index of pin_id must be built from. ValueError when the pin is not mapped, or when its
    engine_version_expected is unknown or does not match the mapped tag (a 1.8 dump never stamps a 1.9 pin)."""
    want = sources.get(pin_id)
    if not isinstance(want, dict) or not want.get("tag") or not re.fullmatch(r"[0-9a-f]{40}", str(want.get("commit"))):
        raise ValueError("pin %s has no entry in the index source map (index_sources.json)" % pin_id)
    ver = ident.get("engine_version_expected")
    if not ver:
        raise ValueError("pin %s has no engine_version_expected (apply its engine sidecar, tools/pin/pin_sidecar.py "
                         "engine), so the dump tag cannot be checked against the build" % pin_id)
    if want["tag"] != "v" + ver:
        raise ValueError("the source map names tag %s for pin %s, but the pin's engine reports %s (tag v%s)"
                         % (want["tag"], pin_id, ver, ver))
    return want


def verify_source(pin_id: str, ident: dict, source, sources: dict) -> None:
    """ValueError unless an index stamp's source is the dump tag and commit the map gives for this pin."""
    want = expected_source(pin_id, ident, sources)
    if not isinstance(source, dict) or source.get("kind") != "script-dump-git":
        raise ValueError("the index was not built from the git script dump (source %r)" % (source,))
    if (source.get("tag"), source.get("commit")) != (want["tag"], want["commit"]):
        raise ValueError("the index was built from %s @ %s, but pin %s needs %s @ %s" % (
            source.get("tag"), str(source.get("commit"))[:8], pin_id, want["tag"], want["commit"][:8]))


def write_index(out_dir: str, blobs: dict, stamp: dict, shared, check_write_path, dry_run: bool) -> list:
    """Write <out>/api/* and INDEX_PIN.json, copy the shared dirs. The output dir must not exist yet (an index is
    built once per pin; rebuild into a new dir). Returns the planned paths."""
    if os.path.lexists(out_dir):
        raise ValueError("%s already exists; an index is built once per pin (remove it by hand to rebuild)" % out_dir)
    check_write_path(out_dir)
    plan = [os.path.join(out_dir, "api", n) for n in list(blobs) + ["INDEX_PIN.json"]]
    if shared:
        plan += [os.path.join(out_dir, d) + "/" for d in SHARED_DIRS]
    if dry_run:
        return plan
    # Built in a temp dir next to it and renamed into place: a killed build never leaves a partial index under the
    # name the gate reads, and a leftover temp dir (unique name) never blocks a retry.
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, ".%s.tmp-%d-%s" % (os.path.basename(out_dir), os.getpid(), os.urandom(3).hex()))
    check_write_path(tmp)
    try:
        os.makedirs(os.path.join(tmp, "api"))
        for name, data in blobs.items():
            with open(os.path.join(tmp, "api", name), "xb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        if shared:
            for d in SHARED_DIRS:
                src = os.path.join(shared, d)
                if os.path.isdir(src):
                    shutil.copytree(src, os.path.join(tmp, d), symlinks=False)
        with open(os.path.join(tmp, "api", "INDEX_PIN.json"), "x", encoding="utf-8") as fh:
            json.dump(stamp, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.lexists(out_dir):
            raise ValueError("%s appeared during the build; not replaced" % out_dir)
        os.rename(tmp, out_dir)
    finally:
        if os.path.lexists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return plan


def main(argv=None, write_roots=None) -> int:
    pinlib, preflight = _load_pinlib()
    roots = write_roots or pinlib.DEFAULT_WRITE_ROOTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dump-git", metavar="REPO", required=True, help="a clone of Arma-Reforger-Script-Diff")
    b.add_argument("--tag", help="optional: must equal the tag the source map gives for the pin")
    b.add_argument("--pin-id", required=True)
    b.add_argument("--pin-dir", default=pinlib.DEFAULT_PIN_DIR)
    b.add_argument("--sources", default=None, help="the index source map (default %s)" % DEFAULT_SOURCES)
    b.add_argument("--out", required=True, help="the per-pin data dir, <mcp root>/pins/<pin id>/data")
    b.add_argument("--shared", default=None, help="data dir to copy patterns/, recipes/ and wiki/ from")
    mode = b.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    try:
        pin_src = preflight.load_pin_by_id(os.path.abspath(pinlib.win_to_wsl(args.pin_dir)), args.pin_id)
        ident = preflight.pin_identity(pin_src)
        out = os.path.abspath(args.out)
        if os.path.basename(os.path.dirname(out)) != args.pin_id or os.path.basename(out) != "data":
            raise pinlib.PinError("--out %s must be <mcp root>/pins/%s/data" % (out, args.pin_id))
        sources_path = args.sources or DEFAULT_SOURCES
        want = expected_source(args.pin_id, ident, load_sources(sources_path))
        if args.tag and args.tag != want["tag"]:
            raise ValueError("--tag %s, but pin %s is built from %s (index source map)" % (args.tag, args.pin_id,
                                                                                      want["tag"]))
        commit, files = read_tree_git(args.dump_git, want["tag"])
        if commit != want["commit"]:
            raise ValueError("tag %s resolves to %s in %s, but the source map pins it to %s (a moved tag or another "
                             "repo)" % (want["tag"], commit[:12], args.dump_git, want["commit"][:12]))
        with open(sources_path, "rb") as fh:
            sources_sha = hashlib.sha256(fh.read()).hexdigest()
        source = {"kind": "script-dump-git", "repo": os.path.abspath(args.dump_git), "tag": want["tag"],
                  "commit": commit, "verified": "tag commit equals the index source map; tag equals v<engine version>",
                  "sources_sha256": sources_sha}
    except (pinlib.PinError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print("api_index: %s" % exc, file=sys.stderr)
        return 2
    if not files:
        print("api_index: no *.c files in the dump", file=sys.stderr)
        return 2
    index = build_index(files)
    blobs = render(index)
    source["files"] = len(files)
    source["tree_sha256"] = hashlib.sha256("".join("%s\n%s\n" % (r, hashlib.sha256(t.encode()).hexdigest())
                                                   for r, t in files).encode()).hexdigest()
    stamp = {"schema": SCHEMA, "pin_id": args.pin_id, "pin_sha256": pin_src["sha256"], "build_key": ident["build_key"],
             "source": source, "built_utc": pinlib.utc_iso(pinlib.utc_now()),
             "builder": "tools/enfusion-mcp/api_index.py",
             "counts": {"enfusion": len(index["enfusion"]), "arma": len(index["arma"]),
                        "methods": sum(len(c["methods"]) + len(c["protectedMethods"]) + len(c["staticMethods"])
                                       for c in index["enfusion"] + index["arma"])},
             "files": {n: hashlib.sha256(d).hexdigest() for n, d in blobs.items()}}
    try:
        plan = write_index(out, blobs, stamp, args.shared, lambda p: pinlib.check_write_path(p, roots), args.dry_run)
    except (pinlib.PinError, OSError, ValueError) as exc:
        print("api_index: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in stamp.items()}, indent=2))
    print("%s %d files: %d enfusion + %d arma classes, %d methods; build_key %s"
          % ("DRY RUN, would write" if args.dry_run else "wrote", len(plan), stamp["counts"]["enfusion"],
             stamp["counts"]["arma"], stamp["counts"]["methods"], ident["build_key"]), file=sys.stderr)
    for p in plan:
        print("  %s" % p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
