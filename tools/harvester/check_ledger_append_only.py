#!/usr/bin/env python3
"""check_ledger_append_only: the Everon exposure ledger (docs/program/everon_ledger.md) only grows (DD-6-min;
CAPTURE_PROGRAM r2 L4 and Appendix A).

    python3 -I -B tools/harvester/check_ledger_append_only.py [--ledger FILE] [--base-rev REV | --base-file FILE]
    python3 -I -B tools/harvester/check_ledger_append_only.py --history [REV]
        (both with [--repo DIR], the git repository; default this one)

Checks, in this order:
  1. Format. The file is ASCII with LF line ends and holds exactly one ledger table: the header
     `| Id | Date | Exposure or decision | Effect allowed |`, the separator `|---|---|---|---|`, then one row per
     entry with four non-empty cells. Ids run EL-001, EL-002, ... without a gap, and dates are real YYYY-MM-DD dates
     that never go back from one row to the next. Nothing outside the table may look like a row (a `|` line after
     it, or an `| EL-` line before it, e.g. a row appended after a blank line), and the table may not sit inside an
     HTML comment or a code fence.
  2. Seed. Rows EL-001 to EL-007 are the Appendix A seed, byte for byte. Their sha256 is fixed here (SEED_SHA256),
     so the seed cannot be rewritten even in a commit that also rewrites the base.
  3. Append-only. Every row of the base version (the ledger at git revision --base-rev, default HEAD, or
     --base-file) is still present, unchanged and in the same order, at the start of the table; new rows only follow
     it. A base without the ledger (before the commit that adds it) skips this check.

--history checks the committed history instead, so an edit that was committed (which the HEAD comparison cannot
see) still fails: every commit reachable from REV (default HEAD) that changed the ledger must pass checks 1 and 2,
and check 3 against the ledger of each of its parents; the commit that adds it has no base, and a commit that
deletes it is a violation. The test suite runs it.

Text outside the table may change; the entries may not. Reads only: git show, cat-file and rev-list
(GIT_OPTIONAL_LOCKS=0), no file is written. Exit codes: 0 the ledger is valid and only grew, 1 a violation (each one
is printed), 2 a usage error or an unreadable file.
"""

import argparse
import datetime
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LEDGER_REL = "docs/program/everon_ledger.md"
HEADER = "| Id | Date | Exposure or decision | Effect allowed |"
SEPARATOR = "|---|---|---|---|"
ROW_LIKE_RE = re.compile(r"^\s*\|")                       # a table line: none may follow the ledger table
EL_ROW_RE = re.compile(r"^\s*\|\s*EL-[0-9]")              # an entry row: none may precede it
COMMENT_RE = re.compile(r"<!--.*?(-->|\Z)", re.DOTALL)    # an HTML comment (an unclosed one runs to the end)
ROW_RE = re.compile(r"^\| (EL-[0-9]{3}) \| ([0-9]{4}-[0-9]{2}-[0-9]{2}) \| (.+) \| (.+) \|$")
SEED_IDS = ("EL-001", "EL-002", "EL-003", "EL-004", "EL-005", "EL-006", "EL-007")
# sha256 of the seed rows EL-001..EL-007 (each row line + LF), copied verbatim from CAPTURE_PROGRAM r2.2 Appendix A
# with the section sign written as &sect;.
SEED_SHA256 = "4ffc3b2a54c680c7f4786249a31f213dff4bf4b544cf6bf5f75ba094e9c6a951"

EXIT_OK, EXIT_VIOLATION, EXIT_USAGE = 0, 1, 2


class LedgerError(Exception):
    """The ledger cannot be read as a ledger at all (exit 1 with the reason)."""


def table_rows(text):
    """The row lines of the ledger table (after the header and separator), in file order. LedgerError when the
    table is not the one visible table of the file: an `| EL-` line before it, any table line after it (a row
    appended after a blank line would be outside the checked table), or the table inside an HTML comment or a code
    fence."""
    lines = text.split("\n")
    heads = [i for i, line in enumerate(lines) if line == HEADER]
    if len(heads) != 1:
        raise LedgerError("expected exactly one ledger table header %r, found %d" % (HEADER, len(heads)))
    i = heads[0]
    if i + 1 >= len(lines) or lines[i + 1] != SEPARATOR:
        raise LedgerError("the table header is not followed by %r" % SEPARATOR)
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    after = i + 2 + len(rows)
    stray = [n + 1 for n, line in enumerate(lines[:i]) if EL_ROW_RE.match(line)]
    stray += [n + 1 for n, line in enumerate(lines[after:], after) if ROW_LIKE_RE.match(line)]
    if stray:
        raise LedgerError("line(s) %s outside the ledger table look like rows; every entry goes in the one table, "
                          "with no blank line before it" % ", ".join(str(n) for n in stray[:5]))
    start = sum(len(line) + 1 for line in lines[:i])
    end = sum(len(line) + 1 for line in lines[:after])
    if any(m.start() < end and m.end() > start for m in COMMENT_RE.finditer(text)):
        raise LedgerError("the ledger table lies inside an HTML comment")
    if sum(1 for line in lines[:i] if line.lstrip().startswith("```")) % 2:
        raise LedgerError("the ledger table lies inside a code fence")
    return rows


def check_text(data):
    """(rows, problems) of a ledger file's bytes: the format and seed checks (1 and 2)."""
    problems = []
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return [], ["the ledger is not ASCII"]
    if "\r" in text:
        problems.append("the ledger has CR characters (LF line ends only)")
    if not text.endswith("\n"):
        problems.append("the ledger does not end with a line feed")
    try:
        rows = table_rows(text)
    except LedgerError as e:
        return [], problems + [str(e)]
    prev_date = None
    for n, line in enumerate(rows, 1):
        m = ROW_RE.match(line)
        if not m:
            problems.append("row %d is not '| EL-NNN | YYYY-MM-DD | exposure | effect |': %s" % (n, line[:60]))
            continue
        rid, date, exposure, effect = m.groups()
        if rid != "EL-%03d" % n:
            problems.append("row %d has id %s, expected EL-%03d (ids run without gaps)" % (n, rid, n))
        try:
            day = datetime.date(int(date[0:4]), int(date[5:7]), int(date[8:10]))
        except ValueError:
            problems.append("%s: %s is not a date" % (rid, date))
            continue
        if prev_date is not None and day < prev_date:
            problems.append("%s: date %s is earlier than the row above it (%s)" % (rid, date, prev_date))
        prev_date = day
        if "|" in exposure or "|" in effect or not exposure.strip() or not effect.strip():
            problems.append("%s: needs exactly four non-empty cells" % rid)
    seed = rows[:len(SEED_IDS)]
    if [line.split(" | ", 1)[0][2:] for line in seed] != list(SEED_IDS):
        problems.append("the table does not start with the seed rows %s..%s" % (SEED_IDS[0], SEED_IDS[-1]))
    elif hashlib.sha256(("\n".join(seed) + "\n").encode("ascii")).hexdigest() != SEED_SHA256:
        problems.append("the seed rows %s..%s differ from the Appendix A seed (sha256 %s)" % (
            SEED_IDS[0], SEED_IDS[-1], SEED_SHA256[:16]))
    return rows, problems


def check_append_only(base_rows, rows):
    """Problems when rows does not start with base_rows, unchanged and in order (check 3)."""
    problems = []
    for n, old in enumerate(base_rows):
        if n >= len(rows):
            problems.append("row %s of the base was removed" % old.split(" | ", 1)[0][2:])
        elif rows[n] != old:
            problems.append("row %d changed: base %r, now %r" % (n + 1, old[:48], rows[n][:48]))
    return problems


def git(repo, *args):
    """A read-only git command in repo (no optional locks)."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    return subprocess.run(["git", "-C", repo] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def git_base(rev, rel, repo=REPO):
    """Bytes of rel at git revision rev of repo, or None when the revision has no such file."""
    proc = git(repo, "cat-file", "-e", "%s:%s" % (rev, rel))
    if proc.returncode != 0:
        check = git(repo, "rev-parse", "--verify", "--quiet", rev + "^{commit}")
        if check.returncode != 0:
            raise OSError("git revision %r not found" % rev)
        return None
    proc = git(repo, "show", "%s:%s" % (rev, rel))
    if proc.returncode != 0:
        raise OSError("git show %s:%s failed: %s" % (rev, rel, proc.stderr.decode("utf-8", "replace").strip()))
    return proc.stdout


def check(ledger_bytes, base_bytes):
    """All problems of a ledger against its base (None: no base version)."""
    rows, problems = check_text(ledger_bytes)
    if base_bytes is not None and not problems:
        base_rows, base_problems = check_text(base_bytes)
        if base_problems:
            problems.append("the base version is not a valid ledger (%s); compare against another base" %
                            base_problems[0])
        else:
            problems += check_append_only(base_rows, rows)
    return problems


def check_history(rev, repo=REPO, rel=LEDGER_REL):
    """(commits checked, problems) of the ledger's committed history: every commit reachable from rev that changed
    rel, oldest first, against each of its parents (check() with the parent's version as the base; none for the
    commit that adds it). A commit that deletes the ledger is a violation. OSError when git cannot answer."""
    proc = git(repo, "rev-list", "--full-history", "--reverse", rev, "--", rel)
    if proc.returncode != 0:
        raise OSError("git rev-list %s -- %s failed: %s" % (rev, rel, proc.stderr.decode("utf-8", "replace").strip()))
    commits = proc.stdout.decode("ascii").split()
    if not commits:
        return [], ["no commit reachable from %s adds %s" % (rev, rel)]
    problems = []
    for commit in commits:
        parents = git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.decode("ascii").split()[1:]
        data = git_base(commit, rel, repo)
        if data is None:
            problems.append("%s deletes %s" % (commit[:12], rel))
            continue
        for base in [git_base(p, rel, repo) for p in parents] or [None]:
            problems += ["%s: %s" % (commit[:12], p) for p in check(data, base)]
    return commits, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", default=None, help="the ledger (default %s of --repo)" % LEDGER_REL)
    ap.add_argument("--repo", default=REPO, help="the git repository (default this one)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--base-rev", default=None, help="git revision of the base version (default HEAD)")
    group.add_argument("--base-file", default=None, help="a file holding the base version")
    group.add_argument("--history", nargs="?", const="HEAD", default=None, metavar="REV",
                       help="check every commit that changed the ledger, up to REV (default HEAD)")
    args = ap.parse_args(argv)
    if args.history and args.ledger:
        ap.error("--history checks the committed ledger of --repo; --ledger does not apply")
    if args.history:
        try:
            commits, problems = check_history(args.history, args.repo)
        except OSError as e:
            print("error: %s" % e)
            return EXIT_USAGE
        for p in problems:
            print("violation: %s" % p)
        if problems:
            return EXIT_VIOLATION
        print("ok: %d commit(s) up to %s changed %s; each only appended to its parents' ledger" % (
            len(commits), args.history, LEDGER_REL))
        return EXIT_OK
    try:
        with open(args.ledger or os.path.join(args.repo, LEDGER_REL), "rb") as f:
            ledger = f.read()
        if args.base_file:
            with open(args.base_file, "rb") as f:
                base = f.read()
            base_name = args.base_file
        else:
            rev = args.base_rev or "HEAD"
            base = git_base(rev, LEDGER_REL, args.repo)
            base_name = "%s:%s" % (rev, LEDGER_REL)
    except OSError as e:
        print("error: %s" % e)
        return EXIT_USAGE
    problems = check(ledger, base)
    for p in problems:
        print("violation: %s" % p)
    if problems:
        return EXIT_VIOLATION
    rows, _ = check_text(ledger)
    print("ok: %d entries (%s..%s), seed intact%s" % (
        len(rows), rows[0].split(" | ", 1)[0][2:], rows[-1].split(" | ", 1)[0][2:],
        "" if base is None else "; only appended to %s" % base_name))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
