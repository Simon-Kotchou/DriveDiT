#!/usr/bin/env python3
"""Tests for data/dc2/reader.py (DD-5-min) and the legacy guard of data/enfusion_preprocessing.py.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_reader.py

The reader, the fiducial decoder and the fixture generator are loaded by file path (`import data.dc2` would run
data/__init__.py, which imports torch). The legacy-guard tests import data/enfusion_preprocessing.py, which imports
torch and cv2, and skip when either is missing.

Covered: the reader reproduces golden.json (keys exact, integers exact, values within 1e-4, pixels within 1/255;
a NaN never compares equal); the committed golden fixture equals a fresh generation, and the 5 Hz fallback session
reproduces its own golden; committed chunks only; header-based parse (shuffled columns), refused headers naming the
column, whole-row validate-then-append, overflowing numbers refused; the R-01 decision boundary, a held command and
every exec channel; the img -> hf join through the fiducial across the 2^18 wrap, and obs only through hf rows that
join ctl and telemetry; segments at gaps, episode boundaries and masks (the obs of another episode, a teleport
between the obs and the window, react_visible at the label step); the seal (a failed status, its fields, a lost
tail) and salvage; duplicated keys, empty chunks, img paths and files outside the session; the vehicle card hash
and gear mapping; dc-1.0 sessions raise; Everon (incl. an Eden root world), heldout/ and SEALED paths need
sealed_eval; C: paths are refused; enfusion_preprocessing names a missing column. Temporary sessions live under
$TMPDIR when that is on /dev/shm/ or /mnt/e/, else /dev/shm/, and are removed.
"""

import importlib.util
import json
import math
import os
import shutil
import struct
import tempfile
import unittest
import warnings
import zlib

import numpy

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures", "capture_dc2")
GOLDEN_ID = "arland_T0b_20261025T091512Z_wb_i00_5e00"
GOLDEN_DIR = os.path.join(FIXTURES, "golden", GOLDEN_ID)
GOLDEN_JSON = os.path.join(FIXTURES, "golden.json")
CARD = os.path.join(FIXTURES, "cards", "S105_synthetic.json")
VALUE_TOL = 1e-4
PIXEL_TOL = 1.0 / 255.0 + 1e-6


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = load("dc2_reader", "data/dc2/reader.py")
synth = load("dc2_make_synthetic", "tests/fixtures/capture_dc2/make_synthetic.py")


def temp_dir(prefix):
    """A new directory under $TMPDIR when that is on /dev/shm/ or /mnt/e/, else under /dev/shm/ (never /tmp, which
    is on the C: drive here). The caller removes it."""
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not ((base.startswith("/dev/shm/") or base.startswith("/mnt/e/")) and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)) or base.startswith("/mnt/c"):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/ or /mnt/e/")
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def png_content(data):
    """(IHDR, decompressed IDAT) of a PNG: equal for equal images whatever zlib build compressed them."""
    pos, ihdr, idat = 8, None, b""
    while pos < len(data):
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        payload = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            ihdr = payload
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length
    return ihdr, zlib.decompress(idat)


def compare(expected, actual, path, out, pixel=False):
    """Recursive comparison: dict keys exact, list lengths exact, integers exact, other numbers finite and within
    VALUE_TOL (PIXEL_TOL under pixels), everything else equal. Appends (path, expected, actual) to out."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            out.append((path, "keys", sorted(set(expected) ^ set(actual if isinstance(actual, dict) else {}))))
            return
        for k in expected:
            compare(expected[k], actual[k], path + "." + k, out, pixel or k == "pixels")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            out.append((path, "len", len(expected), len(actual) if isinstance(actual, list) else actual))
            return
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare(e, a, "%s[%d]" % (path, i), out, pixel)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool) and \
            isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                out.append((path, expected, actual))
        elif not (math.isfinite(expected) and math.isfinite(actual)) \
                or abs(expected - actual) > (PIXEL_TOL if pixel else VALUE_TOL):
            out.append((path, expected, actual))
    elif expected != actual:
        out.append((path, expected, actual))


_RAM = []


def golden_source():
    """A copy of the golden session in a temporary directory (made once; copies from /mnt/e are slow)."""
    if not _RAM:
        tmp = temp_dir("dc2_golden_src_")
        _RAM.append(tmp)
        shutil.copytree(GOLDEN_DIR, os.path.join(tmp, GOLDEN_ID))
    return os.path.join(_RAM[0], GOLDEN_ID)


def tearDownModule():
    for tmp in _RAM:
        shutil.rmtree(tmp, ignore_errors=True)


def copy_golden(tmp, parent=None):
    dst = os.path.join(tmp, parent, GOLDEN_ID) if parent else os.path.join(tmp, GOLDEN_ID)
    shutil.copytree(golden_source(), dst)
    return dst


def rewrite_chunk(session_dir, stream, chunk_id, transform, keys=False):
    """Rewrite one chunk through transform(lines) -> lines and keep its commit line's n_bytes and n_rows (and with
    keys=True its first and last key) true, so only the intended defect is present."""
    sdir = os.path.join(session_dir, stream)
    path = os.path.join(sdir, "chunk_%05d.csv" % chunk_id)
    with open(path, "rb") as f:
        lines = f.read().decode("ascii").split("\n")[:-1]
    lines = transform(lines)
    data = ("\n".join(lines) + "\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(data)
    commit = os.path.join(sdir, "commit.csv")
    with open(commit, "rb") as f:
        rows = f.read().decode("ascii").split("\n")[:-1]
    parts = rows[chunk_id + 1].split(",")
    parts[3] = str(len(lines) - 1)
    parts[4] = str(len(data))
    if keys and len(lines) > 1:
        k = lines[0].split(",").index(reader.load_schema().key(stream))
        parts[1], parts[2] = lines[1].split(",")[k], lines[-1].split(",")[k]
    rows[chunk_id + 1] = ",".join(parts)
    with open(commit, "wb") as f:
        f.write(("\n".join(rows) + "\n").encode("ascii"))


def set_column(session_dir, stream, chunk_id, column, fn):
    """Rewrite one column of one chunk: fn(key, text) -> text, with key the row's first column (its key)."""
    def transform(lines):
        cols = lines[0].split(",")
        j = cols.index(column)
        out = [lines[0]]
        for line in lines[1:]:
            vals = line.split(",")
            vals[j] = fn(int(vals[0]), vals[j])
            out.append(",".join(vals))
        return out

    rewrite_chunk(session_dir, stream, chunk_id, transform)


def edit_json(path, fn):
    with open(path, "rb") as f:
        obj = json.loads(f.read().decode("utf-8"))
    fn(obj)
    with open(path, "wb") as f:
        f.write((json.dumps(obj, indent=2) + "\n").encode("ascii"))


class GoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN_JSON, "rb") as f:
            cls.golden = json.loads(f.read().decode("ascii"))
        cls.session = reader.open_session(GOLDEN_DIR, vehicle_card=CARD)

    def test_reproduces_golden_json(self):
        slots = [p["slot_id"] for p in self.golden["pixels"]["slots"]]
        view = json.loads(json.dumps(self.session.golden_view(slots, self.golden["pixels"]["stride"])))
        diffs = []
        compare(self.golden, view, "golden", diffs)
        self.assertEqual(diffs, [], "reader differs from golden.json: %s" % diffs[:10])
        self.assertEqual(len(self.golden["decision_steps"]), 130)
        self.assertEqual(len(self.golden["pixels"]["slots"]), 4)

    def test_compare_catches_a_value_off_by_more_than_the_tolerance(self):
        view = json.loads(json.dumps(self.golden))
        view["decision_steps"][30]["state"][0] += 2 * VALUE_TOL
        view["pixels"]["slots"][0]["core"][0][0][0] += 2.0 / 255.0
        view["clocks"]["extra"] = 1
        diffs = []
        compare(self.golden, view, "golden", diffs)
        self.assertEqual(len(diffs), 3, diffs)

    def test_compare_catches_nan_and_integer_drift(self):
        diffs = []
        compare({"a": 1.0, "b": [2.5], "c": 7}, {"a": float("nan"), "b": [float("inf")], "c": 7}, "x", diffs)
        self.assertEqual([d[0] for d in diffs], ["x.a", "x.b[0]"])
        diffs = []
        compare({"slot": 5, "v": 1.0}, {"slot": 6, "v": 1.00005}, "x", diffs)    # an integer key never "within 1e-4"
        self.assertEqual([d[0] for d in diffs], ["x.slot"])

    def test_committed_fixture_equals_a_fresh_generation(self):
        tmp = temp_dir("dc2_regen_")
        try:
            synth.generate(os.path.join(tmp, "golden"), golden_json=os.path.join(tmp, "golden.json"),
                           routes_dir=os.path.join(tmp, "routes"))
            fresh = os.path.join(tmp, "golden", GOLDEN_ID)
            for root, _, files in os.walk(GOLDEN_DIR):
                for name in files:
                    rel = os.path.relpath(os.path.join(root, name), GOLDEN_DIR)
                    with open(os.path.join(GOLDEN_DIR, rel), "rb") as a, open(os.path.join(fresh, rel), "rb") as b:
                        if rel.endswith(".png"):
                            self.assertEqual(png_content(a.read()), png_content(b.read()), rel)
                        else:
                            self.assertEqual(a.read(), b.read(), rel)
            n_fresh = sum(len(files) for _, _, files in os.walk(fresh))
            n_committed = sum(len(files) for _, _, files in os.walk(GOLDEN_DIR))
            self.assertEqual(n_fresh, n_committed)
            for rel in ("golden.json", os.path.join("routes", "golden_s_curve_v1.json")):
                with open(os.path.join(FIXTURES, rel), "rb") as a, open(os.path.join(tmp, rel), "rb") as b:
                    self.assertEqual(a.read(), b.read(), rel)
        finally:
            shutil.rmtree(tmp)

    def test_fixture_is_small(self):
        total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(FIXTURES) for f in fs)
        self.assertLess(total, 2 * 1024 * 1024)

    def test_committed_streams_and_chunks(self):
        s = self.session
        self.assertEqual(s.problems, [])
        self.assertTrue(s.sealed)
        for name in ("ctl", "telemetry", "hf", "img", "events"):
            self.assertEqual(len(s.streams[name].commits), 2, name)     # the 10 s chunk rule
        self.assertEqual(len(s.streams["episodes"].commits), 2)         # chunk_max_rows 1
        self.assertEqual(s.streams["ctl"].col("physics_step"), list(range(780)))
        self.assertEqual(s.streams["telemetry"].col("physics_step"), list(range(0, 780, 3)))
        self.assertNotIn("traffic", [n for n, st in s.streams.items() if st.present])

    def test_timing_from_session_json(self):
        c = self.session.clocks
        self.assertEqual((c["k"], c["N"], c["image_hz"], c["img_lead_ms"]), (3, 6, 10.0, 0))
        self.assertAlmostEqual(c["image_period_ms"], 100.0)
        self.assertEqual(reader.tbd1_rule(c["slice_s"]), (6, 3))
        self.assertEqual(reader.tbd1_rule(1 / 50.0), (5, 1))
        self.assertEqual(reader.tbd1_rule(1 / 30.0), (3, 1))
        self.assertEqual(reader.tbd1_rule(1 / 60.0, 5.0), (12, 3))      # TBD-2 5 Hz fallback: N 12 at 60 Hz

    def test_fiducial_join_across_the_wrap(self):
        s = self.session
        hf = s.streams["hf"]
        frames = [r["captured_render_frame"] for r in s.images]
        self.assertTrue(all(r["fid_ok"] == 1 for r in s.images))
        self.assertTrue(any(f >= reader.fiducial.CODE_MODULUS for f in frames))
        self.assertTrue(any(f < reader.fiducial.CODE_MODULUS for f in frames))
        for r in s.images:
            self.assertEqual(r["code"], r["captured_render_frame"] % reader.fiducial.CODE_MODULUS)
            self.assertEqual(hf.col("render_frame")[r["hf_index"]], r["captured_render_frame"])
            self.assertEqual(r["captured_render_frame"] - r["issue_render_frame"], 1)

    def test_hf_joins(self):
        s = self.session
        hf, ctl, tel = s.streams["hf"], s.streams["ctl"], s.streams["telemetry"]
        for i, lps in enumerate(hf.col("last_physics_step")):
            if lps < 0:
                self.assertIsNone(s.hf_ctl[i])
                self.assertIsNone(s.hf_tel[i])
                continue
            self.assertEqual(ctl.col("physics_step")[s.hf_ctl[i]], lps)
            t = tel.col("physics_step")[s.hf_tel[i]]
            self.assertTrue(0 <= lps - t < s.clocks["k"])

    def test_decision_boundary_r01(self):
        s = self.session
        ctl = s.streams["ctl"]
        idx = ctl.index()
        n = s.clocks["N"]
        by_step = {d["decision_step"]: d for d in s.decisions}
        checked = 0
        for d in s.decisions:
            p = d["decision_step"]
            if d["masks"]["wm_valid"] != 1:
                continue
            nxt = ctl.row(idx[p + 1])
            cur = ctl.row(idx[p])
            # action_cmd_t is the command applied from P_t + 1; in mode E it equals the label decided after P_t
            self.assertEqual(d["action_cmd"], [nxt["hl_kappa"], nxt["hl_accel"], nxt["hl_dir"]])
            self.assertEqual(d["action_cmd"], d["action_label"])
            # the row of P_t still holds the previous decision's command; hl_seq increments at P_t + 1 only
            prev = by_step.get(p - n)
            if prev is not None and prev["masks"]["wm_valid"] == 1:
                self.assertEqual([cur["hl_kappa"], cur["hl_accel"], cur["hl_dir"]], prev["action_cmd"])
                self.assertNotEqual(cur["hl_kappa"], nxt["hl_kappa"])
            self.assertEqual(nxt["hl_seq"], cur["hl_seq"] + 1)
            for j in range(2, n + 1):
                self.assertEqual(ctl.row(idx[p + j])["hl_seq"], nxt["hl_seq"])
            self.assertEqual(len(d["action_exec"]), n)
            self.assertLessEqual(d["obs_last_physics_step"], p)
            checked += 1
        self.assertEqual(checked, 98)       # 100 in-episode steps less each episode's first (obs before it)

    def test_segments_and_masks(self):
        s = self.session
        self.assertEqual(s.segments, [
            {"episode_id": 0, "first_decision_step": 126, "last_decision_step": 414, "n_steps": 49},
            {"episode_id": 1, "first_decision_step": 456, "last_decision_step": 744, "n_steps": 49}])
        by_step = {d["decision_step"]: d for d in s.decisions}
        self.assertEqual(by_step[114]["masks"]["warmup"], 1)
        self.assertEqual(by_step[120]["masks"]["warmup"], 0)
        # the first decision of an episode sees an image from before the episode started: masked
        self.assertEqual((by_step[120]["obs_valid"], by_step[120]["masks"]["wm_valid"]), (1, 0))
        self.assertEqual(by_step[432]["masks"]["teleport"], 1)
        self.assertEqual(by_step[438]["masks"]["teleport"], 1)      # its obs (step 433) predates the teleport (434)
        self.assertEqual(by_step[420]["masks"]["wm_valid"], 0)     # the window leaves the episode
        self.assertEqual(by_step[120]["state"][3], 1)               # forward gear through the card (TBD-7)
        self.assertEqual(by_step[0]["state"][3], 0)

    def test_exports_contract_sign(self):
        s = self.session
        ctl = s.streams["ctl"]
        for i in range(len(ctl)):
            raw = ctl.col("input_post_steer")[i]
            self.assertEqual(s.ctl_export["steer_input"][i], -raw)   # steer_sign -1 (TBD-5 default)
        tel = s.streams["telemetry"]
        heading = s.telemetry_export["heading_rad"]
        self.assertTrue(numpy.allclose(heading, numpy.arctan2(tel.array("fwd_x"), tel.array("fwd_z"))))
        k = s.telemetry_export["kappa_realised"]
        v = s.telemetry_export["speed_fwd_mps"]
        self.assertTrue(numpy.all(numpy.isnan(k[numpy.abs(v) <= 3.0])))

    def test_no_card_gives_null_dir_state(self):
        s = reader.open_session(golden_source(), decode_images=False)
        self.assertTrue(all(d is None for d in s.telemetry_export["dir_state"]))
        self.assertIn("no vehicle card: dir_state is null (TBD-7)", s.notes)

    def test_dir_state_through_the_card(self):
        t0min = {"gear_indices_forward": [3], "gear_index_neutral": None, "gear_index_reverse": None}
        self.assertEqual([reader.dir_state_of(t0min, g) for g in (3, 1, 2, None)], [1, 0, 0, None])
        measured = {"gear_indices_forward": [3, 4], "gear_index_neutral": 1, "gear_index_reverse": 0}
        self.assertEqual([reader.dir_state_of(measured, g) for g in (4, 1, 0, 2)], [1, 0, -1, None])


class FiveHzTest(unittest.TestCase):
    """The TBD-2 5 Hz fallback: image_hz = decision_hz = 5, N = 12, k = 3."""

    def test_reproduces_its_golden(self):
        tmp = temp_dir("dc2_5hz_")
        try:
            out = synth.generate(os.path.join(tmp, "s"), kind="golden_5hz", golden_json=os.path.join(tmp, "g.json"))
            with open(os.path.join(tmp, "g.json"), "rb") as f:
                golden = json.loads(f.read().decode("ascii"))
            s = reader.open_session(out["session_dir"], vehicle_card=CARD)
            self.assertEqual((s.clocks["N"], s.clocks["k"], s.clocks["image_hz"]), (12, 3, 5.0))
            slots = [p["slot_id"] for p in golden["pixels"]["slots"]]
            view = json.loads(json.dumps(s.golden_view(slots, golden["pixels"]["stride"])))
            diffs = []
            compare(golden, view, "golden", diffs)
            self.assertEqual(diffs, [], diffs[:10])
            self.assertEqual(len(s.decisions[50]["action_exec"]), 12)
            self.assertTrue(any(d["masks"]["wm_valid"] for d in s.decisions))
        finally:
            shutil.rmtree(tmp)


class DefectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = temp_dir("dc2_reader_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_parse_by_header_name(self):
        session = copy_golden(self.tmp)

        def shuffle(lines):
            cols = lines[0].split(",")
            order = list(range(len(cols)))[::-1]
            return [",".join(line.split(",")[i] for i in order) for line in lines]

        rewrite_chunk(session, "telemetry", 0, shuffle)
        a = reader.open_session(golden_source(), decode_images=False)
        b = reader.open_session(session, decode_images=False)
        for col in a.streams["telemetry"].columns:
            self.assertEqual(a.streams["telemetry"].col(col), b.streams["telemetry"].col(col), col)

    def test_missing_required_column_refuses_the_chunk_and_names_it(self):
        session = copy_golden(self.tmp)

        def drop(lines):
            cols = lines[0].split(",")
            j = cols.index("vel_local_z")
            return [",".join(v for i, v in enumerate(line.split(",")) if i != j) for line in lines]

        rewrite_chunk(session, "telemetry", 0, drop)
        with self.assertRaisesRegex(reader.ReaderError, "vel_local_z"):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, strict=False, decode_images=False)
        c05 = s.problems_for("C05")
        self.assertEqual(len(c05), 1)
        self.assertIn("missing required column(s) vel_local_z", c05[0].message)
        self.assertEqual(s.streams["telemetry"].col("physics_step")[0], 600)    # chunk 0 refused, chunk 1 read

    def test_unknown_column_refuses_the_chunk(self):
        session = copy_golden(self.tmp)
        rewrite_chunk(session, "hf", 1, lambda lines: [lines[0] + ",bogus"] + [x + ",1" for x in lines[1:]])
        s = reader.open_session(session, strict=False, decode_images=False)
        self.assertTrue(any("unknown column(s) bogus" in p.message for p in s.problems_for("C05")))

    def test_whole_row_validate_then_append(self):
        session = copy_golden(self.tmp)

        def corrupt(lines):
            cols = lines[0].split(",")
            j = cols.index("acc_local_z")
            vals = lines[5].split(",")
            vals[j] = "nan"
            lines[5] = ",".join(vals)
            vals = lines[9].split(",")
            lines[9] = ",".join(vals[:20])          # a truncated row
            return lines

        rewrite_chunk(session, "telemetry", 0, corrupt)
        s = reader.open_session(session, strict=False, decode_images=False)
        tel = s.streams["telemetry"]
        lengths = set(len(tel.col(c)) for c in tel.columns)
        self.assertEqual(lengths, {258})
        msgs = [p.message for p in s.problems_for("C05")]
        self.assertTrue(any("column acc_local_z" in m for m in msgs), msgs)
        self.assertTrue(any("20 fields" in m for m in msgs), msgs)
        self.assertNotIn(12, tel.col("physics_step"))
        with self.assertRaises(reader.ReaderError):
            reader.open_session(session, decode_images=False)

    def test_exponent_and_crlf_are_refused(self):
        session = copy_golden(self.tmp)

        def bad(lines):
            vals = lines[3].split(",")
            vals[7] = "1e-3"                        # pos_z with an exponent
            lines[3] = ",".join(vals)
            lines[4] = lines[4] + "\r"
            return lines

        rewrite_chunk(session, "telemetry", 1, bad)
        s = reader.open_session(session, strict=False, decode_images=False)
        msgs = [p.message for p in s.problems_for("C05")]
        self.assertTrue(any("exponent" in m for m in msgs), msgs)
        self.assertTrue(any("CR" in m for m in msgs), msgs)

    def test_committed_chunks_only(self):
        session = copy_golden(self.tmp)
        src = os.path.join(session, "ctl", "chunk_00001.csv")  # a copy of a chunk, never committed
        shutil.copyfile(src, os.path.join(session, "ctl", "chunk_00002.csv"))
        s = reader.open_session(session, decode_images=False)
        self.assertEqual(s.streams["ctl"].uncommitted, ["chunk_00002.csv"])
        self.assertEqual(len(s.streams["ctl"]), 780)

    def test_commit_mismatch_raises_in_strict_mode(self):
        session = copy_golden(self.tmp)
        commit = os.path.join(session, "img", "commit.csv")
        with open(commit, "rb") as f:
            text = f.read().decode("ascii")
        self.assertIn("\n0,0,99,100,", text)
        with open(commit, "wb") as f:
            f.write(text.replace("\n0,0,99,100,", "\n0,0,99,99,", 1).encode("ascii"))
        with self.assertRaisesRegex(reader.ReaderError, "C34"):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, strict=False, decode_images=False)
        self.assertTrue(any("commit line says 99" in p.message for p in s.problems_for("C34")))

    def test_gap_variant_splits_the_segment(self):
        out = synth.generate(os.path.join(self.tmp, "gap"), variant="gap")
        s = reader.open_session(out["session_dir"], vehicle_card=CARD)
        self.assertNotIn(304, s.streams["ctl"].col("physics_step"))
        by_step = {d["decision_step"]: d for d in s.decisions}
        self.assertEqual(by_step[300]["masks"]["wm_valid"], 0)
        self.assertIsNone(by_step[300]["action_exec"][3])          # step 304
        self.assertEqual(s.segments[:2], [
            {"episode_id": 0, "first_decision_step": 126, "last_decision_step": 294, "n_steps": 29},
            {"episode_id": 0, "first_decision_step": 306, "last_decision_step": 414, "n_steps": 19}])

    def test_missing_commit_variant_discards_the_tail(self):
        out = synth.generate(os.path.join(self.tmp, "mc"), variant="missing_commit")
        # the seal says the session ran to step 779; the committed ctl rows stop at 599: a builder must refuse it
        with self.assertRaisesRegex(reader.ReaderError, "C34 ctl/: committed rows end at physics_step 599"):
            reader.open_session(out["session_dir"], decode_images=False)
        for s in (reader.open_session(out["session_dir"], decode_images=False, salvage=True),
                  reader.open_session(out["session_dir"], decode_images=False, strict=False)):
            self.assertEqual(s.streams["ctl"].uncommitted, ["chunk_00001.csv"])
            self.assertEqual(s.streams["ctl"].col("physics_step")[-1], 599)

    def test_unsealed_session(self):
        session = copy_golden(self.tmp)
        os.remove(os.path.join(session, "session_end.json"))
        with self.assertRaisesRegex(reader.ReaderError, "not sealed"):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, decode_images=False, salvage=True)
        self.assertFalse(s.sealed)

    def test_failed_seal_raises_unless_salvage(self):
        session = copy_golden(self.tmp)
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o.update({"status": "failed", "end_reason": "script_error:write_fail"}))
        with self.assertRaisesRegex(reader.ReaderError, "status 'failed'"):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, decode_images=False, salvage=True)
        self.assertEqual(len(s.streams["ctl"]), 780)
        s = reader.open_session(session, decode_images=False, strict=False)
        self.assertEqual(len(s.problems_for("C01")), 1)
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o.update({"status": "ok", "end_reason": "ego_lost"}))     # an error ending, status ok
        with self.assertRaisesRegex(reader.ReaderError, "end_reason 'ego_lost'"):
            reader.open_session(session, decode_images=False)

    def test_seal_fields_against_the_data(self):
        session = copy_golden(self.tmp)

        def bad(o):
            o["contract_version"] = "dc-1.0"
            o["episodes"] = 9
            o["end"]["physics_step"] = 99999
            o["end"]["render_frame"] = 1

        edit_json(os.path.join(session, "session_end.json"), bad)
        s = reader.open_session(session, decode_images=False, strict=False)
        msgs = sorted("%s %s" % (p.check, p.message) for p in s.problems)
        self.assertEqual(len(msgs), 5, msgs)
        self.assertTrue(any("C01 contract_version 'dc-1.0'" in m for m in msgs), msgs)
        self.assertTrue(any("C01 episodes 9, 2 committed" in m for m in msgs), msgs)
        self.assertEqual(sum(1 for m in msgs if m.startswith("C34") and "truncated tail" in m), 3, msgs)
        with self.assertRaises(reader.ReaderError):
            reader.open_session(session, decode_images=False, salvage=True)  # salvage excuses only the seal tail

    def test_truncated_telemetry_tail_with_honest_counts(self):
        session = copy_golden(self.tmp)
        os.remove(os.path.join(session, "telemetry", "chunk_00001.csv"))
        commit = os.path.join(session, "telemetry", "commit.csv")
        with open(commit, "rb") as f:
            lines = f.read().decode("ascii").split("\n")
        with open(commit, "wb") as f:
            f.write(("\n".join(lines[:2]) + "\n").encode("ascii"))
        edit_json(os.path.join(session, "session_end.json"),
                  lambda o: o["counts"]["telemetry"].update({"rows": 200, "chunks": 1, "last_key": 597}))
        with self.assertRaisesRegex(reader.ReaderError, "telemetry/: committed rows end at physics_step 597"):
            reader.open_session(session, decode_images=False)

    def test_overflowing_number_is_refused(self):
        session = copy_golden(self.tmp)
        set_column(session, "telemetry", 0, "pos_x", lambda key, v: "9" * 400 if key == 30 else v)
        s = reader.open_session(session, decode_images=False, strict=False)
        msgs = [p.message for p in s.problems_for("C05")]
        self.assertTrue(any("column pos_x" in m and "non-finite" in m for m in msgs), msgs)
        self.assertNotIn(30, s.streams["telemetry"].col("physics_step"))
        self.assertTrue(all(math.isfinite(v) for v in s.streams["telemetry"].col("pos_x")))

    def test_duplicated_key_raises_in_strict_mode(self):
        session = copy_golden(self.tmp)

        def dup(lines):
            row = lines[302].split(",")                 # physics_step 301 (header is line 0)
            self.assertEqual(row[0], "301")
            return lines[:303] + [lines[302]] + lines[303:]

        rewrite_chunk(session, "ctl", 0, dup)
        with self.assertRaisesRegex(reader.ReaderError, "C07 ctl/: key physics_step not strictly increasing"):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, decode_images=False, strict=False)
        self.assertEqual(len(s.problems_for("C07")), 1)

    def test_empty_committed_chunk(self):
        session = copy_golden(self.tmp)
        path = os.path.join(session, "episodes", "chunk_00002.csv")
        with open(os.path.join(session, "episodes", "chunk_00001.csv"), "rb") as f:
            header = f.read().decode("ascii").split("\n")[0] + "\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
        with open(os.path.join(session, "episodes", "commit.csv"), "ab") as f:
            f.write(("2,1,1,0,%d\n" % len(header)).encode("ascii"))
        s = reader.open_session(session, decode_images=False, strict=False)
        self.assertTrue(any("committed with 0 rows" in p.message for p in s.problems_for("C34")),
                        s.problems_for("C34"))

    def test_img_paths_and_files_outside_the_session(self):
        session = copy_golden(self.tmp, "a")
        other = os.path.join(self.tmp, "elsewhere", "frames")
        os.makedirs(other)
        shutil.copyfile(os.path.join(session, "frames", "f_00000005.png"), os.path.join(other, "f_00000005.png"))
        paths = {5: os.path.join(other, "f_00000005"), 7: "../../elsewhere/frames/f_00000005"}
        set_column(session, "img", 0, "path", lambda key, v: paths.get(key, v))
        s = reader.open_session(session, strict=False)
        for slot in (5, 7):
            self.assertIsNone(s.images[slot]["file"])
            self.assertEqual(s.images[slot]["fid_ok"], 0)
        self.assertEqual(len(s.problems_for("C11")), 2)
        # a symlinked chunk that points out of the session is never read, and strict mode refuses the session
        chunk = os.path.join(session, "hf", "chunk_00001.csv")
        outside = os.path.join(self.tmp, "elsewhere", "hf_chunk.csv")
        shutil.move(chunk, outside)
        os.symlink(outside, chunk)
        s = reader.open_session(session, strict=False, decode_images=False)
        self.assertTrue(any(p.where == "hf/chunk_00001.csv" and "outside" in p.message
                            for p in s.problems_for("C01")))
        with self.assertRaisesRegex(reader.ReaderError, "outside the session directory"):
            reader.open_session(session, decode_images=False)

    def test_obs_needs_the_hf_joins(self):
        base = reader.open_session(golden_source(), vehicle_card=CARD)
        d = next(x for x in base.decisions if x["decision_step"] == 300)
        lps = d["obs_last_physics_step"]          # 295: its as-of telemetry row is step 294
        session = copy_golden(self.tmp)
        rewrite_chunk(session, "telemetry", 0, lambda lines: [ln for ln in lines if not ln.startswith("%d," % (
            lps - lps % 3))])
        s = reader.open_session(session, vehicle_card=CARD, strict=False)
        d2 = next(x for x in s.decisions if x["decision_step"] == 300)
        self.assertNotEqual(d2["obs_slot_id"], d["obs_slot_id"])
        self.assertEqual(d2["masks"]["wm_valid"], 0)

    def test_teleport_between_obs_and_decision_masks_it(self):
        session = copy_golden(self.tmp)

        def add_teleport(lines):
            end = lines[1].split(",")                   # 6,1,750,...,episode_end
            tele = ["6", "1", "597", str(int(end[3]) - 153), str(int(end[4]) - 2550), "teleport", "1", "reason=test"]
            end[0] = "7"
            return [lines[0], ",".join(tele), ",".join(end)]

        rewrite_chunk(session, "events", 1, add_teleport, keys=True)
        s = reader.open_session(session, vehicle_card=CARD)
        d = next(x for x in s.decisions if x["decision_step"] == 600)
        self.assertEqual(d["obs_last_physics_step"], 595)        # pre-teleport image, post-teleport state
        self.assertEqual((d["masks"]["teleport"], d["masks"]["wm_valid"], d["masks"]["policy_valid"]), (1, 0, 0))

    def test_command_held_and_exec_channels_complete(self):
        for column, value in (("hl_kappa", "0.012345"), ("llc_throttle", "")):
            session = copy_golden(self.tmp, column)
            set_column(session, "ctl", 0, column, lambda key, v: value if key == 303 else v)
            s = reader.open_session(session, vehicle_card=CARD, strict=False)
            d = next(x for x in s.decisions if x["decision_step"] == 300)
            self.assertEqual(d["masks"]["wm_valid"], 0, column)
            self.assertEqual(d["masks"]["policy_valid"], 1, column)    # the label at P_t is unaffected

    def test_react_visible_at_the_label_step(self):
        session = copy_golden(self.tmp)
        set_column(session, "ctl", 0, "react_visible", lambda key, v: "0" if key == 300 else v)
        s = reader.open_session(session, vehicle_card=CARD)
        d = next(x for x in s.decisions if x["decision_step"] == 300)
        self.assertEqual((d["masks"]["policy_valid"], d["masks"]["wm_valid"]), (0, 1))

    def test_vehicle_card_hash(self):
        bad_card = os.path.join(self.tmp, "card.json")
        with open(CARD, "rb") as f:
            card = json.loads(f.read().decode("ascii"))
        card.update({"gear_indices_forward": [], "gear_index_reverse": 3})      # forward gear moved to reverse
        with open(bad_card, "w", encoding="ascii") as f:
            json.dump(card, f)
        with self.assertRaisesRegex(reader.ReaderError, "C02 control.vehicle_card_sha"):
            reader.open_session(golden_source(), vehicle_card=bad_card, decode_images=False)
        s = reader.open_session(golden_source(), vehicle_card=bad_card, decode_images=False, strict=False)
        self.assertIsNone(s.vehicle_card)
        self.assertTrue(all(d is None for d in s.telemetry_export["dir_state"]))
        s = reader.open_session(golden_source(), vehicle_card=card, decode_images=False, strict=False)
        self.assertIsNone(s.vehicle_card)                       # a dict cannot be checked against the sha
        self.assertEqual(len(s.problems_for("C02")), 1)

    def test_legacy_sessions_raise(self):
        legacy = synth.write_legacy_session(self.tmp)
        with self.assertRaisesRegex(reader.LegacySessionError, "dc-1.0"):
            reader.open_session(legacy)
        session = copy_golden(self.tmp)
        edit_json(os.path.join(session, "session.json"), lambda o: o.update({"contract_version": "dc-1.0"}))
        with self.assertRaises(reader.LegacySessionError):
            reader.open_session(session, decode_images=False)

    def test_everon_needs_sealed_eval(self):
        everon_id = GOLDEN_ID.replace("arland_", "everon_")
        session = os.path.join(self.tmp, everon_id)
        shutil.copytree(golden_source(), session)

        def to_everon(obj):
            obj["session_id"] = everon_id
            if "map" in obj:
                obj["map"].update({"map_id": "everon", "terrain": "Eden",
                                   "root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"})

        edit_json(os.path.join(session, "session.json"), to_everon)
        edit_json(os.path.join(session, "session_end.json"), to_everon)
        with self.assertRaises(reader.SealedMapError):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, sealed_eval=True, decode_images=False)
        self.assertEqual(s.map_id, "everon")
        self.assertEqual(s.problems, [])
        # map_id everon under a name that does not say so is refused as well
        renamed = copy_golden(self.tmp, "renamed")
        edit_json(os.path.join(renamed, "session.json"),
                  lambda o: o["map"].update({"map_id": "everon", "terrain": "Eden"}))
        with self.assertRaises(reader.SealedMapError):
            reader.open_session(renamed, strict=False, decode_images=False)
        # and so is every Eden-rooted world, whatever map_id and terrain say
        eden = copy_golden(self.tmp, "eden")
        edit_json(os.path.join(eden, "session.json"),
                  lambda o: o["map"].update({"root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"}))
        with self.assertRaisesRegex(reader.SealedMapError, "Eden-rooted"):
            reader.open_session(eden, strict=False, decode_images=False)

    def test_sealed_paths_need_sealed_eval(self):
        for part in ("heldout", "SEALED"):
            session = copy_golden(self.tmp, part)
            with self.assertRaises(reader.SealedMapError):
                reader.open_session(session, decode_images=False)
            reader.open_session(session, sealed_eval=True, decode_images=False)

    def test_c_drive_paths_are_refused(self):
        with self.assertRaisesRegex(reader.ReaderError, "C: drive"):
            reader.open_session("/mnt/c/__dd5_no_such_dir__/" + GOLDEN_ID)


class LegacyGuardTest(unittest.TestCase):
    """data/enfusion_preprocessing.py raises naming the missing column instead of reading zeros (CR-081)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            import cv2  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest("enfusion_preprocessing needs torch and cv2: %s" % e)
        cls.pre = load("enfusion_preprocessing_legacy", "data/enfusion_preprocessing.py")

    def setUp(self):
        self.tmp = temp_dir("dc2_guard_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dc2_session_raises_naming_the_column(self):
        parser = self.pre.EnfusionTelemetryParser()
        chunk = os.path.join(GOLDEN_DIR, "telemetry", "chunk_00000.csv")
        with self.assertRaises(self.pre.MissingTelemetryColumnError) as cm:
            parser.parse_csv(chunk)
        self.assertIn("'timestamp'", str(cm.exception))
        self.assertIn("rot_pitch", cm.exception.columns)

    def test_legacy_session_raises_naming_the_column(self):
        legacy = synth.write_legacy_session(self.tmp)
        parser = self.pre.EnfusionTelemetryParser()
        with self.assertRaises(self.pre.MissingTelemetryColumnError) as cm:
            parser.parse_csv(os.path.join(legacy, "telemetry.csv"))
        self.assertIn("'timestamp'", str(cm.exception))
        self.assertNotIn("timestamp_ms", cm.exception.columns)

    def test_bad_row_never_desynchronises_the_arrays(self):
        parser = self.pre.EnfusionTelemetryParser()
        cols = parser.required_columns()
        path = os.path.join(self.tmp, "telemetry.csv")
        good = ",".join("1" for _ in cols)
        short = ",".join("1" for _ in cols[:5])
        with open(path, "w", encoding="ascii", newline="") as f:
            f.write(",".join(cols) + "\n" + good + "\n" + short + "\n" + good + "\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = parser.parse_csv(path)
        self.assertEqual(set(len(v) for v in data.values()), {2})


if __name__ == "__main__":
    unittest.main()
