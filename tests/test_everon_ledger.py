#!/usr/bin/env python3
"""Tests for docs/program/everon_ledger.md and tools/harvester/check_ledger_append_only.py (DD-6-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_everon_ledger.py

  * the seed rows EL-001..EL-007 are CAPTURE_PROGRAM r2.2 Appendix A verbatim (the section sign written &sect;);
    compared with /mnt/e/drivedit_realign/program/CAPTURE_PROGRAM.md when it exists, else that test skips;
  * the ledger is ASCII with LF line ends and passes the checker against itself and against git HEAD, and its
    committed history passes --history;
  * appending a row passes; editing, removing, reordering or inserting a row fails, and so do an id gap, a date
    that goes back and a rewritten seed (even when the base is rewritten the same way);
  * a row outside the table (after a blank line, before the header, or text after the table that looks like a row)
    and a table inside an HTML comment or a code fence fail (review F-6);
  * in a scratch git repository, an edit or a removal that was committed passes the HEAD comparison but fails
    --history (review F-4).
Base versions and the scratch repository are written to a temporary directory under $TMPDIR (on /dev/shm/ or
/mnt/e/, never /mnt/e/DriveDiT nor the C: or WSL-image device; else /dev/shm/).
"""

import contextlib
import html
import importlib.util
import io
import os
import shutil
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEDGER = os.path.join(REPO, "docs", "program", "everon_ledger.md")
PROGRAM_MD = "/mnt/e/drivedit_realign/program/CAPTURE_PROGRAM.md"
NEW_ROW = ("| EL-008 | 2026-10-23 | Synthetic test entry: a declared build migration | build change; no "
           "Everon-driven change |")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load("dd6_ledger_check", "tools/harvester/check_ledger_append_only.py")


def scratch_ok(path):
    """True for an existing, writable directory under /dev/shm/ or /mnt/e/ (symlinks resolved) that is not inside
    /mnt/e/DriveDiT and not on the device of C: or of the WSL disk image (/), as test_harvest.scratch_ok."""
    real = os.path.realpath(path or "/")
    low = real.lower()
    if not (low.startswith("/dev/shm/") or low.startswith("/mnt/e/")) or low == "/mnt/e/drivedit" \
            or low.startswith("/mnt/e/drivedit/"):
        return False
    if not (os.path.isdir(real) and os.access(real, os.W_OK)):
        return False
    dev = os.stat(real).st_dev
    return all(not os.path.isdir(other) or os.stat(other).st_dev != dev for other in ("/mnt/c", "/"))


def temp_dir(prefix):
    base = os.environ.get("TMPDIR")
    if not scratch_ok(base):
        base = "/dev/shm"
    return tempfile.mkdtemp(prefix=prefix, dir=os.path.realpath(base))


def run_main(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = checker.main(argv)
    return code, out.getvalue()


def ledger_text():
    with open(LEDGER, "rb") as f:
        return f.read().decode("ascii")


def seed_only(text):
    """The ledger text with its table cut back to the seed rows EL-001..EL-007 (the mutation tests' base, so they
    keep working after real entries are appended)."""
    lines = text.split("\n")
    i = lines.index(checker.SEPARATOR) + 1
    n = 0
    while i + n < len(lines) and lines[i + n].startswith("|"):
        n += 1
    return "\n".join(lines[:i] + lines[i:i + 7] + lines[i + n:])


class LedgerSeedTest(unittest.TestCase):
    def test_seed_is_appendix_a_verbatim(self):
        if not os.path.isfile(PROGRAM_MD):
            self.skipTest("%s not on this machine" % PROGRAM_MD)
        with open(PROGRAM_MD, "rb") as f:
            lines = f.read().decode("utf-8").split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("## Appendix A. Everon ledger seed"))
        source = []
        for line in lines[start:]:
            if line.startswith("| EL-"):
                source.append(line)
            elif source:
                break
        rows = checker.table_rows(ledger_text())
        self.assertEqual([html.unescape(r) for r in rows[:7]], source)
        self.assertEqual(len(source), 7)

    def test_ascii_lf_and_valid(self):
        with open(LEDGER, "rb") as f:
            data = f.read()
        self.assertTrue(data.isascii())
        self.assertNotIn(b"\r", data)
        rows, problems = checker.check_text(data)
        self.assertEqual(problems, [])
        self.assertEqual([r.split(" | ", 1)[0][2:] for r in rows[:7]], list(checker.SEED_IDS))

    def test_checker_cli_against_git_head(self):
        code, out = run_main(["--base-rev", "HEAD"])
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("ok: "), out)

    def test_committed_history_only_appends(self):
        """--history over this repository: every commit that changed the ledger only appended to it."""
        if checker.git(checker.REPO, "rev-parse", "--verify", "--quiet", "HEAD").returncode != 0:
            self.skipTest("%s is not a git checkout" % checker.REPO)
        code, out = run_main(["--history"])
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("ok: "), out)


class AppendOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = temp_dir("dd6_ledger_")
        self.base = seed_only(ledger_text())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, new_text, base_text=None):
        paths = []
        for name, text in (("new.md", new_text), ("base.md", base_text if base_text is not None else self.base)):
            path = os.path.join(self.tmp, name)
            with open(path, "wb") as f:
                f.write(text.encode("ascii"))
            paths.append(path)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = checker.main(["--ledger", paths[0], "--base-file", paths[1]])
        return code, out.getvalue()

    def edited(self, fn):
        lines = self.base.split("\n")
        i = lines.index(checker.SEPARATOR) + 1
        rows = lines[i:i + 7]
        rows = fn(rows)
        return "\n".join(lines[:i] + rows + lines[i + 7:])

    def test_unchanged_and_appended_pass(self):
        self.assertEqual(self.run_cli(self.base)[0], 0)
        code, out = self.run_cli(self.edited(lambda rows: rows + [NEW_ROW]))
        self.assertEqual(code, 0, out)
        self.assertIn("8 entries (EL-001..EL-008)", out)
        self.assertEqual(self.run_cli(self.base.replace("Rules:", "Rules (text outside the table):"))[0], 0)

    def test_edit_remove_reorder_insert_fail(self):
        cases = {
            "edit": lambda rows: rows[:2] + [rows[2].replace("| none |", "| none at all |")] + rows[3:],
            "remove": lambda rows: rows[:6],
            "reorder": lambda rows: rows[:1] + [rows[2], rows[1]] + rows[3:],
            "insert": lambda rows: rows[:3] + [NEW_ROW] + rows[3:],
        }
        for name, fn in cases.items():
            with self.subTest(name):
                code, out = self.run_cli(self.edited(fn))
                self.assertEqual(code, 1, out)
                self.assertIn("violation:", out)

    def test_append_after_a_valid_base_but_edit_the_new_row_later(self):
        grown = self.edited(lambda rows: rows + [NEW_ROW])
        code, out = self.run_cli(grown.replace("build change; no", "build change, no"), base_text=grown)
        self.assertEqual(code, 1, out)
        self.assertIn("row 8 changed", out)

    def test_bad_new_rows_fail(self):
        cases = {
            "gap": NEW_ROW.replace("EL-008", "EL-009"),
            "date back": NEW_ROW.replace("2026-10-23", "2026-10-19"),
            "not a date": NEW_ROW.replace("2026-10-23", "2026-13-01"),
            "empty cell": "| EL-008 | 2026-10-23 |   | effect |",
        }
        for name, row in cases.items():
            with self.subTest(name):
                code, out = self.run_cli(self.edited(lambda rows: rows + [row]))
                self.assertEqual(code, 1, out)

    def test_rewritten_seed_fails_even_with_a_rewritten_base(self):
        rewritten = self.edited(lambda rows: [rows[0].replace("navmesh streaming |", "navmesh |")] + rows[1:])
        code, out = self.run_cli(rewritten, base_text=rewritten)
        self.assertEqual(code, 1, out)
        self.assertIn("differ from the Appendix A seed", out)

    def test_rows_outside_the_table_fail(self):
        grown = self.edited(lambda rows: rows + [NEW_ROW])
        lines = self.base.split("\n")
        head = lines.index(checker.HEADER)
        fenced = lines[:head] + ["```"] + lines[head:head + 9] + ["```"] + lines[head + 9:]   # header, separator, seed
        cases = {
            "row after a blank line": self.edited(lambda rows: rows + ["", NEW_ROW]),
            "row text after the table": self.base.rstrip("\n") + "\n\nNotes:\n" + NEW_ROW + "\n",
            "row before the header": "\n".join(lines[:head] + [NEW_ROW, ""] + lines[head:]),
            "table in an HTML comment": "\n".join(lines[:head] + ["<!--"] + lines[head:]) + "-->\n",
            "table in a code fence": "\n".join(fenced),
        }
        for name, text in cases.items():
            with self.subTest(name):
                code, out = self.run_cli(text, base_text=self.base)
                self.assertEqual(code, 1, out)
                self.assertIn("violation:", out)
        # the slip F-6 describes: EL-008 after a blank line, then edited; the check must not say ok
        hidden = self.edited(lambda rows: rows + ["", NEW_ROW])
        code, out = self.run_cli(hidden.replace("build change; no", "build change, no"), base_text=hidden)
        self.assertEqual(code, 1, out)
        self.assertEqual(self.run_cli(grown)[0], 0)

    def test_not_ascii_or_crlf_fails(self):
        for name, text in (("section sign", self.base.replace("&sect;", "\u00a7")),
                           ("crlf", self.base.replace("\n", "\r\n"))):
            with self.subTest(name):
                path = os.path.join(self.tmp, "x.md")
                with open(path, "wb") as f:
                    f.write(text.encode("utf-8"))
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = checker.main(["--ledger", path, "--base-file", LEDGER])
                self.assertEqual(code, 1, out.getvalue())


class HistoryTest(unittest.TestCase):
    """--history in a scratch git repository: a committed edit or removal of a row passes the comparison with HEAD
    (the base is the edited file itself) but fails the history check."""

    def setUp(self):
        self.tmp = temp_dir("dd6_ledger_git_")
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(self.repo, "docs", "program"))
        self.env = dict(os.environ, HOME=self.tmp, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@example.invalid", GIT_COMMITTER_NAME="test",
                        GIT_COMMITTER_EMAIL="test@example.invalid")
        self.git("init", "-q")
        self.seed = seed_only(ledger_text())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def git(self, *args):
        subprocess.run(["git", "-C", self.repo, "-c", "init.defaultBranch=main"] + list(args), check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)

    def commit(self, text, message):
        with open(os.path.join(self.repo, checker.LEDGER_REL), "wb") as f:
            f.write(text.encode("ascii"))
        self.git("add", checker.LEDGER_REL)
        self.git("commit", "-q", "-m", message)

    def with_rows(self, extra):
        lines = self.seed.split("\n")
        i = lines.index(checker.SEPARATOR) + 1 + len(checker.SEED_IDS)
        return "\n".join(lines[:i] + extra + lines[i:])

    def appended_then(self, rows, message):
        """Commit the seed, then EL-008 (history ok), then the ledger with rows after the seed instead."""
        self.commit(self.seed, "seed")
        self.commit(self.with_rows([NEW_ROW]), "append EL-008")
        code, out = run_main(["--history", "--repo", self.repo])
        self.assertEqual(code, 0, out)
        self.assertIn("ok: 2 commit(s)", out)
        self.commit(self.with_rows(rows), message)
        code, out = run_main(["--repo", self.repo, "--base-rev", "HEAD"])
        self.assertEqual(code, 0, out)                            # the HEAD comparison cannot see it
        return run_main(["--history", "--repo", self.repo])

    def test_committed_edit_fails_history(self):
        code, out = self.appended_then([NEW_ROW.replace("build change; no", "build change, no")], "edit EL-008")
        self.assertEqual(code, 1, out)
        self.assertIn("row 8 changed", out)

    def test_committed_removal_fails_history(self):
        code, out = self.appended_then([], "remove EL-008")
        self.assertEqual(code, 1, out)
        self.assertIn("row EL-008 of the base was removed", out)

    def test_deleting_the_ledger_fails(self):
        self.commit(self.seed, "seed")
        self.git("rm", "-q", checker.LEDGER_REL)
        self.git("commit", "-q", "-m", "delete")
        code, out = run_main(["--history", "--repo", self.repo])
        self.assertEqual(code, 1, out)
        self.assertIn("deletes %s" % checker.LEDGER_REL, out)


if __name__ == "__main__":
    unittest.main()
