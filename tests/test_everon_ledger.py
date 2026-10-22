#!/usr/bin/env python3
"""Tests for docs/program/everon_ledger.md and tools/harvester/check_ledger_append_only.py (DD-6-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_everon_ledger.py

  * the seed rows EL-001..EL-007 are CAPTURE_PROGRAM r2.2 Appendix A verbatim (the section sign written &sect;);
    compared with /mnt/e/drivedit_realign/program/CAPTURE_PROGRAM.md when it exists, else that test skips;
  * the ledger is ASCII with LF line ends and passes the checker against itself and against git HEAD;
  * appending a row passes; editing, removing, reordering or inserting a row fails, and so do an id gap, a date
    that goes back and a rewritten seed (even when the base is rewritten the same way).
Base versions are written to a temporary directory under $TMPDIR (on /dev/shm/ or /mnt/e/, else /dev/shm/).
"""

import contextlib
import html
import importlib.util
import io
import os
import shutil
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = checker.main(["--base-rev", "HEAD"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertTrue(out.getvalue().startswith("ok: "), out.getvalue())


class AppendOnlyTest(unittest.TestCase):
    def setUp(self):
        base = os.path.realpath(os.environ.get("TMPDIR") or "/")
        low = base.lower()
        if not ((low.startswith("/dev/shm/") or low.startswith("/mnt/e/")) and os.path.isdir(base)) \
                or low == "/mnt/e/drivedit" or low.startswith("/mnt/e/drivedit/"):
            base = "/dev/shm"
        self.tmp = tempfile.mkdtemp(prefix="dd6_ledger_", dir=base)
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


if __name__ == "__main__":
    unittest.main()
