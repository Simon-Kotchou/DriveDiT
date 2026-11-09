#!/usr/bin/env python3
"""Diff two per-pin MCP API indexes and check them against the recorded API delta (ticket DD-14a, CC-05;
MIGRATION_1_9 6.1 row CC-05: "api_index_diff.py agrees with [AD 2-3]").

    python3 -I -B api_index_diff.py --old <mcp root>/pins/1.8.0.13/data --new <mcp root>/pins/1.9.0.37/data \
        --expect tools/enfusion-mcp/expect_1.8.0.13_to_1.9.0.37.txt [--tsv OUT.tsv]

Both directories are data dirs written by api_index.py (api/*.json and api/INDEX_PIN.json). The files must match the
hashes in their INDEX_PIN.json. The expectations file holds one assertion per line ('#' starts a comment):

    class-added    NAME...                  in --new, not in --old
    class-removed  NAME...                  in --old, not in --new
    parent-changed CLASS OLD_PARENT NEW_PARENT
    member-added   CLASS MEMBER...          on the class (or an ancestor) in --new only
    member-removed CLASS MEMBER...          in --old only
    unchanged      CLASS MEMBER...          in both, with the same declaration (whitespace-normalised)
    planned        FILE                     every 'Class Member...' line of FILE (tools/compat19/planned_surface.txt):
                                            each member unchanged or new in --new; REMOVED, SIG_CHANGED and
                                            NOT_FOUND fail (FILE is relative to the repo root, or absolute)

Members are looked up on the class and its ancestors (parents in the index), as a script call resolves them.
Prints PASS/FAIL per assertion, the delta counts and the two INDEX_PIN stamps; --tsv writes the whole class and
member delta (data drive only: the same resolved-path and mount guard as tools/pin). Exit codes: 0 every assertion holds; 1 at least one fails; 2 usage or read error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASS_FILES = ("enfusion-classes.json", "arma-classes.json")


def _pinlib():
    """tools/pin/pinlib.py through api_index's loader (the write guard for --tsv)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_index.py")
    mod = sys.modules.get("dd_emcp_api_index")
    if mod is None or getattr(mod, "__file__", None) != path:
        spec = importlib.util.spec_from_file_location("dd_emcp_api_index", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dd_emcp_api_index"] = mod
        spec.loader.exec_module(mod)
    return mod._load_pinlib()[0]


def open_output(path: str, write_roots=None):
    """Open a new report file for writing, after pinlib's resolved-path and mount guard (data drive only, never the
    system drive, a symlinked parent resolved first); the parent must exist and the file must not."""
    pinlib = _pinlib()
    target = pinlib.check_write_path(path, write_roots)
    if not os.path.isdir(os.path.dirname(target)):
        raise pinlib.PinError("output directory does not exist: %s" % os.path.dirname(target))
    return open(target, "x", encoding="utf-8")


class IndexError_(Exception):
    pass


def load_index(data_dir: str) -> tuple:
    """(stamp, {class name: class}) of a data dir. The api files must match INDEX_PIN.json's hashes."""
    api = os.path.join(data_dir, "api")
    try:
        with open(os.path.join(api, "INDEX_PIN.json"), encoding="utf-8") as fh:
            stamp = json.load(fh)
    except (OSError, ValueError) as exc:
        raise IndexError_("%s: no readable api/INDEX_PIN.json (%s); build it with api_index.py" % (data_dir, exc))
    classes = {}
    for name in CLASS_FILES:
        path = os.path.join(api, name)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            raise IndexError_("cannot read %s: %s" % (path, exc))
        want = (stamp.get("files") or {}).get(name)
        if hashlib.sha256(raw).hexdigest() != want:
            raise IndexError_("%s does not match the sha256 in its INDEX_PIN.json" % path)
        for c in json.loads(raw.decode("utf-8")):
            classes[c["name"]] = c
    return stamp, classes


_LITERAL_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def _norm(s: str) -> str:
    """Whitespace collapsed outside string literals only (Codex round 2): '"a b"' and '"a  b"' stay different."""
    s = s or ""
    out, pos = [], 0
    for m in _LITERAL_RE.finditer(s):
        out.append(re.sub(r"\s+", " ", s[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(re.sub(r"\s+", " ", s[pos:]))
    return "".join(out).strip()


def members(cls: dict) -> dict:
    """{member name: declaration} (overloads joined, sorted) of methods of every kind and member variables. Method
    signatures keep their default literals ("0 0 0") and member variables their initializers."""
    out = {}
    for key in ("methods", "protectedMethods", "staticMethods"):
        for m in cls.get(key) or []:
            out.setdefault(m["name"], set()).add(_norm(m.get("signature")))
    for key in ("properties", "protectedProperties"):
        for p in cls.get(key) or []:
            decl = "%s %s" % (p.get("type"), p["name"])
            if p.get("defaultValue"):
                decl += " = %s" % p["defaultValue"]   # an initializer change is a declaration change
            out.setdefault(p["name"], set()).add(_norm(decl))
    return {k: " | ".join(sorted(v)) for k, v in out.items()}


def find_member(classes: dict, cls: str, member: str):
    """(declaring class, declaration) of member on cls or an ancestor, or (None, None)."""
    seen = set()
    while cls and cls in classes and cls not in seen:
        seen.add(cls)
        mem = members(classes[cls])
        if member in mem:
            return cls, mem[member]
        parents = classes[cls].get("parents") or []
        cls = parents[0] if parents else None
    return None, None


def member_status(old: dict, new: dict, cls: str, member: str) -> tuple:
    oc, od = find_member(old, cls, member)
    nc, nd = find_member(new, cls, member)
    if od is not None and nd is not None:
        return ("unchanged" if od == nd else "SIG_CHANGED"), od, nd
    if od is not None:
        return "REMOVED", od, None
    if nd is not None:
        return "new-in-new", None, nd
    return "NOT_FOUND", None, None


def delta_rows(old: dict, new: dict) -> list:
    rows = []
    for n in sorted(set(old) | set(new)):
        a, b = old.get(n), new.get(n)
        if a is None:
            rows.append(("CLASS_ADDED", n, "", "", ",".join(b.get("parents") or [])))
            continue
        if b is None:
            rows.append(("CLASS_REMOVED", n, "", ",".join(a.get("parents") or []), ""))
            continue
        if (a.get("parents") or []) != (b.get("parents") or []):
            rows.append(("PARENT_CHANGED", n, "", ",".join(a.get("parents") or []), ",".join(b.get("parents") or [])))
        ma, mb = members(a), members(b)
        for k in sorted(set(ma) - set(mb)):
            rows.append(("removed", n, k, ma[k], ""))
        for k in sorted(set(mb) - set(ma)):
            rows.append(("added", n, k, "", mb[k]))
        for k in sorted(set(ma) & set(mb)):
            if ma[k] != mb[k]:
                rows.append(("sig_changed", n, k, ma[k], mb[k]))
    return rows


def read_expectations(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].split()
            if line:
                out.append((n, line[0], line[1:]))
    return out


def check(old: dict, new: dict, expectations: list, base_dir: str = REPO_ROOT) -> list:
    """[(ok, text)] for each assertion."""
    res = []
    for n, kind, args in expectations:
        label = "line %d %s %s" % (n, kind, " ".join(args[:6]) + (" ..." if len(args) > 6 else ""))
        if kind in ("class-added", "class-removed"):
            bad = [a for a in args if (a in new) != (kind == "class-added") or (a in old) != (kind == "class-removed")]
            res.append((not bad, "%s%s" % (label, ": wrong for %s" % bad if bad else "")))
        elif kind == "parent-changed" and len(args) == 3:
            c, po, pn = args
            have = ((old.get(c) or {}).get("parents"), (new.get(c) or {}).get("parents"))
            res.append((have == ([po], [pn]), "%s: %s -> %s" % (label, have[0], have[1])))
        elif kind in ("member-added", "member-removed", "unchanged") and len(args) >= 2:
            want = {"member-added": "new-in-new", "member-removed": "REMOVED", "unchanged": "unchanged"}[kind]
            bad = []
            for m in args[1:]:
                st = member_status(old, new, args[0], m)[0]
                if st != want:
                    bad.append("%s %s" % (m, st))
            res.append((not bad, "%s%s" % (label, ": " + "; ".join(bad) if bad else "")))
        elif kind == "planned" and len(args) == 1:
            path = args[0] if os.path.isabs(args[0]) else os.path.join(base_dir, args[0])
            counts, bad = {}, []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    p = line.split()
                    for m in p[1:]:
                        st = member_status(old, new, p[0], m)[0]
                        counts[st] = counts.get(st, 0) + 1
                        if st not in ("unchanged", "new-in-new"):
                            bad.append("%s.%s %s" % (p[0], m, st))
            res.append((not bad, "%s: %s%s" % (label, ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items())),
                                               "; " + "; ".join(bad[:8]) if bad else "")))
        else:
            res.append((False, "%s: unknown or malformed assertion" % label))
    return res


def main(argv=None, write_roots=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--expect", required=True)
    ap.add_argument("--tsv", default=None, help="write the full delta here (must not exist; data drive only)")
    args = ap.parse_args(argv)
    if args.tsv:
        pinlib = _pinlib()
        try:
            pinlib.check_write_path(args.tsv, write_roots)
        except pinlib.PinError as exc:
            print("api_index_diff: --tsv: %s" % exc, file=sys.stderr)
            return 2
    try:
        so, old = load_index(args.old)
        sn, new = load_index(args.new)
        exp = read_expectations(args.expect)
    except (IndexError_, OSError, ValueError) as exc:
        print("api_index_diff: %s" % exc, file=sys.stderr)
        return 2
    for tag, s in (("OLD", so), ("NEW", sn)):
        src = s.get("source") or {}
        print("%s pin %s build_key %s from %s %s (%s classes)" % (
            tag, s.get("pin_id"), s.get("build_key"), src.get("tag") or src.get("dir"), (src.get("commit") or "")[:8],
            (s.get("counts") or {}).get("enfusion", 0) + (s.get("counts") or {}).get("arma", 0)))
    rows = delta_rows(old, new)
    counts = {}
    for r in rows:
        counts[r[0]] = counts.get(r[0], 0) + 1
    print("DELTA " + ", ".join("%s %d" % kv for kv in sorted(counts.items())))
    results = check(old, new, exp)
    for ok, text in results:
        print("%-4s %s" % ("PASS" if ok else "FAIL", text))
    if args.tsv:
        try:
            with open_output(args.tsv, write_roots) as fh:
                fh.write("kind\tclass\tmember\told\tnew\n")
                for r in rows:
                    fh.write("\t".join(r) + "\n")
        except Exception as exc:  # PinError or OSError: nothing was written
            print("api_index_diff: %s" % exc, file=sys.stderr)
            return 2
    failed = [t for ok, t in results if not ok]
    print("RESULT: %s (%d of %d assertions hold)" % ("OK" if not failed else "MISMATCH", len(results) - len(failed),
                                                     len(results)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
