#!/usr/bin/env python3
"""Tests for tools/probes/t0eng/probe_common.py (DD-4-probe).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_common.py

Synthetic plays are written by probe_common.write_synthetic_play into a make_temp_dir directory (under $TMPDIR when
that lies in /dev/shm/ or /mnt/e/, else /dev/shm/; never /tmp, which is on the C: drive) and removed afterwards.
"""

import contextlib
import io
import os
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
T0ENG = os.path.join(REPO, "tools", "probes", "t0eng")
sys.path.insert(0, T0ENG)
import probe_common as pc  # noqa: E402


class ProbeCommonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pc.make_temp_dir("t0eng_common_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def play_dir(self, name="play_01"):
        return os.path.join(self.tmp, "DC_Probes", "e10", "20261022T010203Z-wb-abcd", name)

    def test_temp_dir_is_not_on_c(self):
        real = os.path.realpath(self.tmp)
        self.assertTrue(real.startswith("/dev/shm/") or real.startswith("/mnt/e/"), real)

    def test_read_kv_later_keys_win(self):
        path = os.path.join(self.tmp, "kv.txt")
        with open(path, "w", newline="") as f:
            f.write("a=1\nb=x=y\nno equals sign\na=2\nempty=\n")
        kv = pc.read_kv(path)
        self.assertEqual(kv["a"], "2")
        self.assertEqual(kv["b"], "x=y")
        self.assertEqual(kv["empty"], "")
        self.assertNotIn("no equals sign", kv)

    def test_read_kv_missing_raises(self):
        with self.assertRaises(pc.InputError):
            pc.read_kv(os.path.join(self.tmp, "absent.txt"))

    def test_read_csv_required_columns(self):
        d = pc.write_synthetic_play(self.play_dir(), {"probe": "e10"}, {"t": (["a", "b"], [[1, None], [2, 3.5]])})
        rows = pc.read_csv(os.path.join(d, "t.csv"), ("a", "b"))
        self.assertEqual(pc.ints(rows, "a"), [1, 2])
        self.assertEqual(pc.floats(rows, "b"), [None, 3.5])
        with self.assertRaises(pc.InputError):
            pc.read_csv(os.path.join(d, "t.csv"), ("a", "c"))
        with open(os.path.join(d, "t.csv"), "rb") as f:
            self.assertEqual(f.read(), b"a,b\n1,\n2,3.5\n")

    def test_percentile(self):
        self.assertEqual(pc.percentile([3, 1, 2], 50), 2.0)
        self.assertAlmostEqual(pc.percentile([0, 10], 25), 2.5)
        self.assertEqual(pc.percentile([7], 99), 7.0)
        self.assertIsNone(pc.percentile([], 50))

    def test_play_with_and_without_footer(self):
        header = {"schema": "dc-probe-run/1", "probe": "e10", "params.k": "3"}
        d1 = pc.write_synthetic_play(self.play_dir("play_01"), header, {}, footer={"status": "complete"})
        d2 = pc.write_synthetic_play(self.play_dir("play_02"), header, {})
        p1, p2 = pc.Play(d1), pc.Play(d2)
        self.assertTrue(p1.complete)
        self.assertEqual(p1.param("k"), "3")
        self.assertEqual(p1.probe, "e10")
        self.assertFalse(p2.complete)
        self.assertIsNone(p2.footer)
        v = pc.Verdict("E-10 tick counting", d2)
        v.check_footer(p2)
        self.assertEqual(v.result(), pc.INCOMPLETE)
        with self.assertRaises(pc.InputError):
            pc.Play(os.path.join(self.tmp, "not_a_play"))

    def test_find_plays(self):
        pc.write_synthetic_play(self.play_dir("play_02"), {"probe": "e10"}, {})
        pc.write_synthetic_play(self.play_dir("play_01"), {"probe": "e10"}, {})
        os.makedirs(os.path.join(self.tmp, "DC_Probes", "e10", "20261022T010203Z-wb-abcd", "not_a_play"))
        root = os.path.join(self.tmp, "DC_Probes")
        plays = pc.find_plays(root, "e10")
        self.assertEqual([os.path.basename(p) for p in plays], ["play_01", "play_02"])
        self.assertEqual(pc.find_plays(root), plays)
        self.assertEqual(pc.find_plays(root, "e04"), [])

    def test_verdict_results_and_exit_codes(self):
        cases = [
            ([True, True], pc.PASS, 0),
            ([True, None], pc.INCOMPLETE, 3),
            ([True, None, False], pc.FAIL, 1),
            ([], pc.INCOMPLETE, 3),
        ]
        for checks, result, code in cases:
            v = pc.Verdict("E-99 something", "/dev/shm/x")
            for i, ok in enumerate(checks):
                v.check("c%d" % i, ok, "detail")
            v.info("note", "info only")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(v.report(), code)
            self.assertEqual(v.result(), result)
            text = out.getvalue().splitlines()
            self.assertEqual(text[0], "E-99 something: /dev/shm/x")
            self.assertEqual(text[-1], "RESULT E-99: %s" % result)
            self.assertIn("  [INFO] note: info only", text)

    def test_run_cli_input_error_exits_2(self):
        def main(argv):
            raise pc.InputError("missing file x")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(pc.run_cli(main, []), 2)
        self.assertIn("input error: missing file x", out.getvalue())

    def test_synthetic_play_refuses_c_and_tmp(self):
        for bad in ("/mnt/c/dd4_should_not_exist/play_01", "/tmp/dd4_should_not_exist/play_01", "/home/x/play_01"):
            with self.assertRaises(ValueError):
                pc.write_synthetic_play(bad, {"probe": "e10"}, {})
            self.assertFalse(os.path.exists(os.path.dirname(bad)), bad)
        with self.assertRaises(ValueError):
            pc.write_synthetic_play(self.play_dir(), {"probe": "e10"}, {}, files={"../../../../../../../../tmp/x": b""})

    def test_check_out_path(self):
        self.assertEqual(pc.check_out_path(self.tmp), self.tmp)
        for bad in ("/mnt/c/Users/x/out.json", "/tmp/out.json", "", "/home/x/out.json"):
            with self.assertRaises(pc.InputError):
                pc.check_out_path(bad)


if __name__ == "__main__":
    unittest.main()
