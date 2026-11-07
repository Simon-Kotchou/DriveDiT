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
        --tag v1.9.0.37 --pin-id 1.9.0.37 --out /mnt/e/drivedit_realign/mcp/pins/1.9.0.37/data \
        --shared /mnt/e/drivedit_realign/mcp/data --dry-run          # then --apply

The dump is read with `git ls-tree` and `git cat-file --batch` at the tag (never a checkout); --dump-dir reads an
extracted scripts/ tree instead. The parser is a declaration scanner, not a compiler: classes (parent from ':' or
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
    """(code, docs): code with comments and string contents blanked (newlines kept, so line numbers hold); docs maps
    a line number to its '//!' doc text."""
    out, docs = [], {}
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            if text.startswith("//!", i):
                docs.setdefault(line, []).append(text[i + 3:j].strip())
            out.append(" " * (j - i))
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            chunk = text[i:j]
            out.append("".join("\n" if ch == "\n" else " " for ch in chunk))
            line += chunk.count("\n")
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append('"' + " " * max(0, j - i - 2) + '"')
            i = j
        else:
            if c == "\n":
                line += 1
            out.append(c)
            i += 1
    return "".join(out), docs


def _doc_before(docs: dict, lineno: int) -> str:
    lines, k = [], lineno - 1
    while k in docs:
        lines[:0] = docs[k]
        k -= 1
    return " ".join(x for x in lines if x)


def _split_params(params: str):
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    res = []
    for p in out:
        p = " ".join(p.split())
        default = ""
        if "=" in p:
            p, default = [x.strip() for x in p.split("=", 1)]
        toks = p.split()
        if not toks:
            continue
        name = toks[-1].rstrip("[]")
        res.append({"name": name, "type": " ".join(toks[:-1]), "defaultValue": default})
    return res


def _method(decl: str, doc: str):
    """MethodInfo from a declaration 'mods type Name(params) [const]', or None."""
    decl = " ".join(decl.split())
    decl = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", decl)
    m = re.match(r"^(?P<head>.*?)\b(?P<name>\w+)\s*\((?P<params>.*)\)\s*(?P<tail>[\w\s]*)$", decl)
    if not m or m.group("name") in KEYWORDS:
        return None, None
    head = m.group("head").strip()
    toks = head.split()
    if any(t in KEYWORDS for t in toks):
        return None, None
    signature = "%s %s (%s)%s" % (head, m.group("name"), m.group("params").strip(),
                                  (" " + m.group("tail").strip()) if m.group("tail").strip() else "")
    kind = "static" if "static" in toks else ("protected" if ("protected" in toks or "private" in toks) else "public")
    return kind, {"name": m.group("name"), "returnType": head, "signature": " ".join(signature.split()),
                  "params": _split_params(m.group("params")), "description": doc}


def _property(decl: str, doc: str):
    decl = " ".join(re.sub(r"^\s*(?:\[[^\]]*\]\s*)+", "", decl).split())
    m = re.match(r"^(?P<head>[\w<>,\[\]\s]*?)\b(?P<name>\w+)\s*(?:\[[^\]]*\])?\s*(?:=.*)?;$", decl)
    if not m or not m.group("head").strip() or m.group("name") in KEYWORDS:
        return None, None
    toks = m.group("head").split()
    if any(t in KEYWORDS for t in toks):
        return None, None
    kind = "protected" if ("protected" in toks or "private" in toks) else "public"
    return kind, {"name": m.group("name"), "type": " ".join(t for t in toks if t not in MODIFIERS) or m.group("head"),
                  "description": doc}


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
    code, docs = _blank_strings_and_comments(text)
    lines = code.split("\n")
    out = []
    depth = 0
    cur = None              # the class being read
    pending = None          # a class/enum header seen, waiting for its '{'
    acc, acc_line = "", 0   # a multi-line member declaration
    enum_acc = None
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
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
            elif stripped and not stripped.startswith(("{", "}", "#")):
                acc, acc_line = stripped, lineno
            if acc:
                opened = acc.count("(") - acc.count(")") + acc.count("[") - acc.count("]")
                bare = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", acc) if opened <= 0 else acc
                if opened > 0:
                    pass                                    # a declaration or attribute that goes on
                elif not bare:
                    acc = ""                                # an attribute line; the member follows
                elif "(" in bare:
                    decl = bare.split("{", 1)[0].rstrip().rstrip(";").rstrip()
                    kind, info = _method(decl, _doc_before(docs, acc_line))
                    if info is not None:
                        cur[{"public": "methods", "protected": "protectedMethods", "static": "staticMethods"}[kind]] \
                            .append(info)
                    acc = ""
                elif bare.endswith(";"):
                    kind, info = _property(bare, _doc_before(docs, acc_line))
                    if info is not None:
                        cur["properties" if kind == "public" else "protectedProperties"].append(info)
                    acc = ""
                elif "{" in bare or "}" in bare:
                    acc = ""
        opens, closes = line.count("{"), line.count("}")
        if pending and opens:
            pending = None
        depth += opens - closes
        if depth <= 0:
            depth = 0
            if cur is not None and closes and pending is None:
                out.append(cur)
                cur = None
                acc = ""
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
    os.makedirs(os.path.join(out_dir, "api"))
    for name, data in blobs.items():
        with open(os.path.join(out_dir, "api", name), "xb") as fh:
            fh.write(data)
    if shared:
        for d in SHARED_DIRS:
            src = os.path.join(shared, d)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(out_dir, d), symlinks=False)
    with open(os.path.join(out_dir, "api", "INDEX_PIN.json"), "x", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=2)
        fh.write("\n")
    return plan


def main(argv=None, write_roots=None) -> int:
    pinlib, preflight = _load_pinlib()
    roots = write_roots or pinlib.DEFAULT_WRITE_ROOTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump-git", metavar="REPO", help="a clone of Arma-Reforger-Script-Diff")
    src.add_argument("--dump-dir", metavar="DIR", help="an extracted scripts/ tree")
    b.add_argument("--tag", help="with --dump-git: the pin's tag, e.g. v1.9.0.37")
    b.add_argument("--pin-id", required=True)
    b.add_argument("--pin-dir", default=pinlib.DEFAULT_PIN_DIR)
    b.add_argument("--out", required=True, help="the per-pin data dir, <mcp root>/pins/<pin id>/data")
    b.add_argument("--shared", default=None, help="data dir to copy patterns/, recipes/ and wiki/ from")
    mode = b.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    if args.dump_git and not args.tag:
        print("api_index: --dump-git needs --tag", file=sys.stderr)
        return 2
    try:
        pin_src = preflight.load_pin_by_id(os.path.abspath(pinlib.win_to_wsl(args.pin_dir)), args.pin_id)
        ident = preflight.pin_identity(pin_src)
        out = os.path.abspath(args.out)
        if os.path.basename(os.path.dirname(out)) != args.pin_id:
            raise pinlib.PinError("--out %s must be <mcp root>/pins/%s/data" % (out, args.pin_id))
        if args.dump_git:
            commit, files = read_tree_git(args.dump_git, args.tag)
            source = {"kind": "script-dump-git", "repo": os.path.abspath(args.dump_git), "tag": args.tag,
                      "commit": commit}
        else:
            files = read_tree_dir(args.dump_dir)
            source = {"kind": "script-dump-dir", "dir": os.path.abspath(args.dump_dir)}
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
