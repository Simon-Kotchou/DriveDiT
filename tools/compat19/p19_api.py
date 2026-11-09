#!/usr/bin/env python3
"""CC-02 planned-surface check of two script dumps (P19-1; ticket DD-14a; MIGRATION_1_9 6.1 row CC-02).

Ports program/1_9/work/planned_check.py and member_delta.py onto the per-pin index parser of
tools/enfusion-mcp/api_index.py, so CC-02 and CC-05 read the dump the same way:

    python3 -I -B tools/compat19/p19_api.py --dump-git <REALIGN_ROOT>/program/1_9/work/dl/sd_stable \
        --old-tag v1.8.0.13 --new-tag v1.9.0.37 [--planned tools/compat19/planned_surface.txt] [--tsv OUT]

Every member of the planned surface (one 'Class Member...' line each, looked up on the class and its ancestors)
must be unchanged or new in the new dump; REMOVED, SIG_CHANGED and NOT_FOUND fail, unless the member is listed in a
triage file (--triage FILE: 'Class.Member status reason' lines, the accepted differences of the G19 report). --tsv
writes the planned rows (class, member, status, old, new) for the report, under the tools/pin write guard (data drive only). Reads the dump through git only; writes
only --tsv. Exit codes: 0 PASS; 1 FAIL; 2 usage or read error.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EMCP = os.path.join(REPO, "tools", "enfusion-mcp")


def _load(name, path):
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


api_index = _load("dd_emcp_api_index", os.path.join(EMCP, "api_index.py"))
api_index_diff = _load("dd_emcp_api_index_diff", os.path.join(EMCP, "api_index_diff.py"))


def classes_of(files) -> dict:
    idx = api_index.build_index(files)
    return {c["name"]: c for c in idx["enfusion"] + idx["arma"]}


def read_triage(path):
    out = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split("#", 1)[0].split(None, 2)
                if len(parts) >= 2:
                    out[parts[0]] = (parts[1], parts[2].strip() if len(parts) > 2 else "")
    return out


def planned_rows(old: dict, new: dict, planned_path: str) -> list:
    rows = []
    with open(planned_path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            for m in p[1:]:
                st, od, nd = api_index_diff.member_status(old, new, p[0], m)
                rows.append((p[0], m, st, od or "", nd or ""))
    return rows


def main(argv=None, write_roots=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump-git", metavar="REPO")
    src.add_argument("--dump-dirs", nargs=2, metavar=("OLD_SCRIPTS", "NEW_SCRIPTS"))
    ap.add_argument("--old-tag")
    ap.add_argument("--new-tag")
    ap.add_argument("--planned", default=os.path.join(HERE, "planned_surface.txt"))
    ap.add_argument("--triage", default=None)
    ap.add_argument("--tsv", default=None, help="write the planned rows here (must not exist)")
    args = ap.parse_args(argv)
    if args.tsv:
        pinlib = api_index._load_pinlib()[0]
        try:
            pinlib.check_write_path(args.tsv, write_roots)
        except pinlib.PinError as exc:
            print("p19_api: --tsv: %s" % exc, file=sys.stderr)
            return 2
    try:
        if args.dump_git:
            if not (args.old_tag and args.new_tag):
                raise ValueError("--dump-git needs --old-tag and --new-tag")
            co, fo = api_index.read_tree_git(args.dump_git, args.old_tag)
            cn, fn = api_index.read_tree_git(args.dump_git, args.new_tag)
            label = "%s (%s) -> %s (%s)" % (args.old_tag, co[:8], args.new_tag, cn[:8])
        else:
            fo, fn = (api_index.read_tree_dir(d) for d in args.dump_dirs)
            label = "%s -> %s" % tuple(args.dump_dirs)
        old, new = classes_of(fo), classes_of(fn)
        rows = planned_rows(old, new, args.planned)
        triage = read_triage(args.triage)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print("p19_api: %s" % exc, file=sys.stderr)
        return 2
    counts, bad = {}, []
    for cls, mem, st, _, _ in rows:
        counts[st] = counts.get(st, 0) + 1
        key = "%s.%s" % (cls, mem)
        if st not in ("unchanged", "new-in-new") and triage.get(key, ("",))[0] != st:
            bad.append("%s %s" % (key, st))
    print("CC-02 planned surface %s: %d members: %s" % (label, len(rows),
                                                         ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items()))))
    for b in bad:
        print("FAIL  %s" % b)
    if args.tsv:
        try:
            with api_index_diff.open_output(args.tsv, write_roots) as fh:
                fh.write("class\tmember\tstatus\told\tnew\n")
                for r in rows:
                    fh.write("\t".join(r) + "\n")
        except Exception as exc:  # PinError or OSError: nothing was written
            print("p19_api: %s" % exc, file=sys.stderr)
            return 2
    print("CC-02 %s" % ("PASS" if not bad else "FAIL (%d untriaged difference(s))" % len(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
