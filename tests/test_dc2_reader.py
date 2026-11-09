#!/usr/bin/env python3
"""Tests for data/dc2/reader.py (DD-5-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_reader.py

The reader, the fiducial decoder and the fixture generator are loaded by file path, as the tools load them.

Covered: the reader reproduces golden.json (keys exact, integers exact, values within 1e-4, pixels within 1/255;
a NaN never compares equal); the committed golden fixture equals a fresh generation, and the 5 Hz fallback session
reproduces its own golden; committed chunks only; header-based parse (shuffled columns), refused headers naming the
column, whole-row validate-then-append, overflowing numbers refused; the R-01 decision boundary, a held command and
every exec channel; the img -> hf join through the fiducial across the 2^18 wrap, and obs only through hf rows that
join ctl and telemetry; segments at gaps, episode boundaries and masks (the obs of another episode, a teleport
between the obs and the window, react_visible at the label step); the seal (a failed status, its fields, a lost
tail) and salvage; duplicated keys, empty chunks, img paths and files outside the session; the vehicle card hash
and gear mapping; dc-1.0 sessions raise; Everon (incl. an Eden root world), heldout/ and SEALED paths need
sealed_eval; system-drive paths are refused; the crash events of contract 9.5.1 (payload parsing, the first terminal ego event,
policy_valid 0 from the crash on, ego_destroyed as a normal seal). Temporary sessions live under $TMPDIR when that is
on /dev/shm/, else /dev/shm/, and are removed.
"""

import importlib.util
import json
import math
import os
import shutil
import struct
import tempfile
import types
import unittest
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


# DD-22: the synthetic local config (also set by tests/conftest.py under pytest), so the repo's fixtures on any
# drive are readable and no real drive is the system drive here.
os.environ["DRIVEDIT_LOCAL_CONFIG"] = os.path.join(REPO, "tests", "fixtures", "local_config", "placeholder.local.env")

def temp_dir(prefix):
    """A new directory under $TMPDIR when that is on /dev/shm/, else under /dev/shm/ (never /tmp, which
    is on the system drive here). The caller removes it."""
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not (base.startswith("/dev/shm/") and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/")
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
    """A copy of the golden session in a temporary directory (made once; copies from drvfs are slow)."""
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

    def test_bridge_wait_ends_for_every_decision(self):
        """N3: the bridge wait (current frame >= C_t + 1 and obs_t done by then) ends for all 52 policy-valid golden
        decisions; for 17 of them obs_t completed before C_t + 1, where the round-1 rule never held."""
        self.assertEqual(bridge_waits(self, self.session), (52, 17))

    def test_reproduces_golden_json(self):
        slots = [p["slot_id"] for p in self.golden["pixels"]["slots"]]
        view = json.loads(json.dumps(self.session.golden_view(slots, self.golden["pixels"]["stride"])))
        diffs = []
        compare(self.golden, view, "golden", diffs)
        self.assertEqual(diffs, [], "reader differs from golden.json: %s" % diffs[:10])
        self.assertEqual(len(self.golden["decision_steps"]), 68)
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
            self.assertEqual(len(s.streams[name].commits), 2, name)     # the 10 s and chunk_max_rows rules
        self.assertEqual(len(s.streams["episodes"].commits), 2)         # chunk_max_rows 1
        # frame-bound rows (D-A): one ctl and one telemetry row per callback, physics_step + covered_steps per row
        ctl, tel = s.streams["ctl"], s.streams["telemetry"]
        steps, covered = ctl.col("physics_step"), ctl.col("covered_steps")
        self.assertEqual(len(steps), 310)
        self.assertEqual(steps[0], covered[0] - 1)
        self.assertTrue(all(b - a == c for a, b, c in zip(steps, steps[1:], covered[1:])))
        self.assertEqual(set(covered), {1, 2, 3})
        self.assertEqual(tel.col("physics_step"), steps)
        self.assertEqual(tel.col("covered_steps"), covered)
        frames = ctl.col("render_frame")
        self.assertTrue(all(b > a for a, b in zip(frames, frames[1:])))     # at most one callback per frame
        self.assertLess(len(steps), len(s.streams["hf"]))                   # frames without a fixed step
        self.assertNotIn("traffic", [n for n, st in s.streams.items() if st.present])

    def test_timing_from_session_json(self):
        c = self.session.clocks
        self.assertEqual((c["N"], c["decision_hz"], c["image_hz"], c["image_hz_setting"], c["img_lead_ms"]),
                         (6, 5.0, 5.0, "default", 0))
        self.assertAlmostEqual(c["image_period_ms"], 200.0)
        self.assertEqual(c["decision_period_ms"], 200)
        self.assertAlmostEqual(c["step_dt_s"], 1 / 30.0, places=5)
        self.assertNotIn("k", c)
        self.assertEqual(reader.steps_per_decision(5, c["step_dt_s"]), 6)
        self.assertEqual(reader.steps_per_decision(10, 1 / 30.0), 3)
        self.assertEqual(reader.steps_per_decision(5, 1 / 60.0), 12)
        self.assertEqual(reader.grid_period_ms(5), 200)
        self.assertIsNone(reader.grid_period_ms(3))           # 1000 / 3 ms is not an integer grid
        world = self.session.streams["ctl"].col("world_time_ms")
        calls, skipped = reader.decision_callbacks(world, 5)
        self.assertEqual(skipped, 0)
        self.assertEqual([g for _, g in calls], list(range(len(calls))))
        for i, g in calls:
            self.assertGreaterEqual(world[i], world[0] + 200 * g)
            self.assertTrue(i == 0 or world[i - 1] < world[0] + 200 * g)
        self.assertEqual(reader.decision_callbacks([0, 50, 450, 500], 5), ([(0, 0), (2, 2)], 1))

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
            self.assertEqual(r["captured_render_frame"], r["issue_render_frame"])
            self.assertEqual(r["captured_render_frame"], r["done_render_frame"] - 1)     # Workbench 1.8, run 4
            self.assertEqual(hf.col("render_frame")[r["pose_hf_index"]], r["captured_render_frame"] + 1)

    def test_hf_joins(self):
        s = self.session
        hf, ctl, tel = s.streams["hf"], s.streams["ctl"], s.streams["telemetry"]
        cb_frame = dict(zip(ctl.col("physics_step"), ctl.col("render_frame")))
        for i, lps in enumerate(hf.col("last_physics_step")):
            if lps < 0:
                self.assertIsNone(s.hf_ctl[i])
                self.assertIsNone(s.hf_tel[i])
                continue
            self.assertEqual(ctl.col("physics_step")[s.hf_ctl[i]], lps)
            self.assertEqual(tel.col("physics_step")[s.hf_tel[i]], lps)          # exact (a row per callback)
            self.assertLess(cb_frame[lps], hf.col("render_frame")[i])         # the callback of an earlier frame
        self.assertEqual(s.grid_anchor_ms, ctl.col("world_time_ms")[0])

    def test_decision_boundary_r01(self):
        s = self.session
        ctl = s.streams["ctl"]
        idx = ctl.index()
        steps = ctl.col("physics_step")
        n = s.clocks["N"]
        checked = 0
        for t, d in enumerate(s.decisions):
            p = d["decision_step"]
            if d["masks"]["wm_valid"] != 1:
                continue
            i = idx[p]
            nxt, cur = ctl.row(i + 1), ctl.row(i)
            # action_cmd_t is the command of the row after C_t; in mode E it equals the label decided at C_t
            self.assertEqual(d["action_cmd"], [nxt["hl_kappa"], nxt["hl_accel"], nxt["hl_dir"]])
            self.assertEqual(d["action_cmd"], d["action_label"])
            # the row of C_t still holds the previous decision's command; hl_seq increments on the next row only
            prev = s.decisions[t - 1]
            if prev["masks"]["wm_valid"] == 1:
                self.assertEqual([cur["hl_kappa"], cur["hl_accel"], cur["hl_dir"]], prev["action_cmd"])
            self.assertEqual(nxt["hl_seq"], cur["hl_seq"] + 1)
            p_next = s.decisions[t + 1]["decision_step"]
            self.assertEqual(d["interval_steps"], p_next - p)
            for j in range(i + 1, idx[p_next] + 1):                 # held over the decision interval
                self.assertEqual(ctl.row(j)["hl_seq"], nxt["hl_seq"])
            self.assertEqual(len(d["action_exec"]), n)
            inside = min(n, p_next - p)
            self.assertTrue(all(e is not None for e in d["action_exec"][:inside]))
            self.assertTrue(all(e is None for e in d["action_exec"][inside:]))
            # each step of the exec window comes from the ctl row that covers it
            for q, e in zip(range(p + 1, p + inside + 1), d["action_exec"]):
                j = reader.covering_row(steps, ctl.col("covered_steps"), q)
                self.assertEqual(e[1], ctl.col("llc_throttle")[j])
            self.assertLessEqual(d["obs_last_physics_step"], p)
            checked += 1
        self.assertEqual(checked, 52)
        self.assertGreater(len(set(d["interval_steps"] for d in s.decisions if d["interval_steps"])), 2)

    def test_segments_and_masks(self):
        s = self.session
        self.assertEqual(s.segments, [
            {"episode_id": 0, "first_decision_step": 61, "last_decision_step": 211, "n_steps": 26},
            {"episode_id": 1, "first_decision_step": 235, "last_decision_step": 385, "n_steps": 26}])
        by_step = {d["decision_step"]: d for d in s.decisions}
        self.assertEqual(by_step[55]["masks"]["warmup"], 1)
        self.assertEqual(by_step[61]["masks"]["warmup"], 0)
        # the first decision of an episode sees the image of its own frame: the state after its callback (pose row)
        self.assertEqual((by_step[61]["obs_last_physics_step"], by_step[61]["obs_age_ms"]), (61, 0))
        self.assertEqual(by_step[61]["masks"]["wm_valid"], 1)
        # an image whose frame had no callback shows the state of an earlier callback (80 ms older here)
        self.assertEqual((by_step[75]["obs_last_physics_step"], by_step[75]["obs_age_ms"]), (72, 80))
        self.assertEqual(by_step[217]["masks"]["teleport"], 1)          # the teleport (step 222) is in its interval
        self.assertEqual(by_step[225]["masks"]["teleport"], 1)          # its obs (step 222) is at the teleport
        self.assertEqual(by_step[217]["masks"]["wm_valid"], 0)          # the interval leaves the episode
        self.assertEqual(by_step[61]["state"][3], 1)                    # forward gear through the card (TBD-7)
        self.assertEqual(by_step[0]["state"][3], 0)                     # neutral
        self.assertIsNone(s.decisions[-1]["interval_steps"])           # the last decision has no interval

    def test_exports_contract_sign(self):
        s = self.session
        ctl = s.streams["ctl"]
        for i in range(len(ctl)):
            raw = ctl.col("input_post_steer")[i]
            self.assertEqual(s.ctl_export["steer_input"][i], raw)    # steer_sign +1 (TBD-5 decided)
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
        s105 = {"gear_indices_forward": [2, 3, 4, 5], "gear_index_neutral": 1, "gear_index_reverse": 0}  # TBD-7
        self.assertEqual([reader.dir_state_of(s105, g) for g in (2, 5, 1, 0, 6)], [1, 1, 0, -1, None])


def bridge_waits(test, s):
    """Review round 2, N3: the closed-loop bridge holds the next physics step after the rows of C_t until the current
    render frame is at least C_t's frame + 1 and obs_t's file has completed (done_render_frame set and at or before
    the current frame): the wait ends in frame max(C_t + 1, done) for every policy-valid decision, and the pose row of
    obs_t (C + 1 <= done) exists by then. Returns (policy-valid decisions, those whose obs_t completed before C_t + 1,
    for which the round-1 predicate done >= C_t + 1 could never hold)."""
    rows = {r["slot_id"]: r for r in s.images}
    hf_frames = set(s.streams["hf"].col("render_frame"))
    valid = [d for d in s.decisions if d["masks"]["policy_valid"]]
    older = 0
    for d in valid:
        obs = rows[d["obs_slot_id"]]
        done = obs["done_render_frame"]
        test.assertIsNotNone(done, d["decision_step"])
        released = max(d["render_frame"] + 1, done)
        test.assertTrue(done <= released and d["render_frame"] + 1 <= released)
        test.assertLessEqual(obs["captured_render_frame"] + 1, released)
        test.assertIn(obs["captured_render_frame"] + 1, hf_frames)
        older += done < d["render_frame"] + 1
    return len(valid), older


class TenHzTest(unittest.TestCase):
    """The non-default 10 Hz setting (TBD-2): image_hz = decision_hz = 10, N = 3, clocks.image_hz_setting trial_conf."""

    def test_reproduces_its_golden(self):
        tmp = temp_dir("dc2_10hz_")
        try:
            out = synth.generate(os.path.join(tmp, "s"), kind="golden_10hz", golden_json=os.path.join(tmp, "g.json"))
            with open(os.path.join(tmp, "g.json"), "rb") as f:
                golden = json.loads(f.read().decode("ascii"))
            s = reader.open_session(out["session_dir"], vehicle_card=CARD)
            self.assertEqual((s.clocks["N"], s.clocks["image_hz"], s.clocks["image_hz_setting"]),
                             (3, 10.0, "trial_conf"))
            self.assertEqual(s.meta_get("trial.conf")["image_hz"], 10)
            slots = [p["slot_id"] for p in golden["pixels"]["slots"]]
            view = json.loads(json.dumps(s.golden_view(slots, golden["pixels"]["stride"])))
            diffs = []
            compare(golden, view, "golden", diffs)
            self.assertEqual(diffs, [], diffs[:10])
            self.assertEqual(len(s.decisions[50]["action_exec"]), 3)
            self.assertTrue(any(d["masks"]["wm_valid"] for d in s.decisions))
            self.assertEqual(bridge_waits(self, s), (100, 22))
        finally:
            shutil.rmtree(tmp)


def repose_slot(session_dir, slot, captured, keep_done=False):
    """Codex's X1 mutation: slot's file shows render frame `captured` (its fiducial and the state of hf row captured
    + 1, as the engine renders it, or the last state past the simulation) and was first seen in frame captured + 1
    (keep_done: its done_render_frame is left as it was)."""
    spec = dict(synth.SPECS["golden"])
    sim = synth.simulate(spec, synth.drivedit_nav.route_from_dict(synth.ROUTES[spec["route"]]()))
    state = sim.frames[min(captured + 1 - spec["render_frame0"], len(sim.frames) - 1)]["state"]
    img = synth.render_image(spec, state, synth.fiducial.code_for(captured))
    with open(os.path.join(session_dir, "frames", "f_%08d.png" % slot), "wb") as f:
        f.write(synth.png_bytes(img))
    if keep_done:
        return
    for name in sorted(os.listdir(os.path.join(session_dir, "img"))):
        if name.startswith("chunk_"):
            set_column(session_dir, "img", int(name[6:11]), "done_render_frame",
                       lambda key, v: str(captured + 1) if key == slot else v)


class DefectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = temp_dir("dc2_reader_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_a_capture_past_the_seal_is_not_exempt(self):
        """Review round 2, N1: Codex's slot 67 re-encoded to frame 262296, past the seal (end.render_frame 262291);
        the slot completed at 262287 and nothing was pending: no exemption, its captured frame does not join."""
        session = copy_golden(self.tmp)
        g = reader.open_session(golden_source(), vehicle_card=CARD)
        self.assertEqual((g.end_get("end.render_frame"), g.end_get("images.pending_at_end")), (262291, 0))
        self.assertEqual([r["done_render_frame"] for r in g.images if r["slot_id"] == 67], [262287])
        repose_slot(session, 67, 262296, keep_done=True)
        s = reader.open_session(session, vehicle_card=CARD)
        row = [r for r in s.images if r["slot_id"] == 67][0]
        self.assertEqual((row["fid_ok"], row["captured_render_frame"], row["pending_at_seal"], row["pose_after_seal"],
                          row["hf_index"]), (1, 262296, 0, 0, None))
        self.assertEqual(sum(r["pending_at_seal"] for r in s.images), 0)

    def test_obs_t_is_never_a_future_image(self):
        """Review round 1, X1: decision 93 is taken in frame 261989 (world 3139 ms); frame 261990 (3166 ms) has no
        callback, so its pose row 261991 still has last_physics_step 93. Slot 15 re-posed to frame 261990 meets the
        step bound but lies in the decision's future: it must not be obs_t, and no obs_age_ms is negative."""
        session = copy_golden(self.tmp)
        g = reader.open_session(golden_source(), vehicle_card=CARD)
        hf = g.streams["hf"]
        at = {f: (w, lps) for f, w, lps in zip(hf.col("render_frame"), hf.col("world_time_ms"),
                                                hf.col("last_physics_step"))}
        before = [d for d in g.decisions if d["decision_step"] == 93][0]
        self.assertEqual((before["render_frame"], at[261989][0], at[261990][0], at[261991][1]),
                         (261989, 3139, 3166, 93))
        self.assertEqual(before["obs_slot_id"], 15)
        repose_slot(session, 15, 261990)
        s = reader.open_session(session, vehicle_card=CARD)
        self.assertEqual([(r["captured_render_frame"], r["done_render_frame"]) for r in s.images if r["slot_id"] == 15],
                         [(261990, 261991)])
        d = [d for d in s.decisions if d["decision_step"] == 93][0]
        self.assertEqual((d["obs_slot_id"], d["obs_captured_render_frame"]), (14, 261983))
        self.assertGreaterEqual(d["obs_age_ms"], 0)
        self.assertTrue(all(x["obs_age_ms"] is None or x["obs_age_ms"] >= 0 for x in s.decisions))
        self.assertTrue(all(x["obs_captured_render_frame"] is None or
                            x["obs_captured_render_frame"] <= x["render_frame"] for x in s.decisions))
        self.assertTrue(all(x["render_frame"] >= 261990 for x in s.decisions if x["obs_slot_id"] == 15))

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
        self.assertEqual(s.streams["telemetry"].col("physics_step")[0], 263)    # chunk 0 refused, chunk 1 read

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
        self.assertEqual(lengths, {308})
        msgs = [p.message for p in s.problems_for("C05")]
        self.assertTrue(any("column acc_local_z" in m for m in msgs), msgs)
        self.assertTrue(any("20 fields" in m for m in msgs), msgs)
        self.assertNotIn(11, tel.col("physics_step"))           # the truncated row (line 9)
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
        self.assertEqual(len(s.streams["ctl"]), 310)

    def test_commit_mismatch_raises_in_strict_mode(self):
        session = copy_golden(self.tmp)
        commit = os.path.join(session, "img", "commit.csv")
        with open(commit, "rb") as f:
            text = f.read().decode("ascii")
        self.assertIn("\n0,0,49,50,", text)
        with open(commit, "wb") as f:
            f.write(text.replace("\n0,0,49,50,", "\n0,0,48,50,", 1).encode("ascii"))
        with self.assertRaisesRegex(reader.ReaderError, "C34"):
            reader.open_session(session, decode_images=False)
        s = reader.open_session(session, strict=False, decode_images=False)
        self.assertTrue(any("commit line says 0..48" in p.message for p in s.problems_for("C34")))

    def test_gap_variant_splits_the_segment(self):
        out = synth.generate(os.path.join(self.tmp, "gap"), variant="gap")
        s = reader.open_session(out["session_dir"], vehicle_card=CARD)
        self.assertNotIn(64, s.streams["ctl"].col("physics_step"))
        self.assertIn(64, s.streams["telemetry"].col("physics_step"))
        by_step = {d["decision_step"]: d for d in s.decisions}
        self.assertEqual(by_step[61]["masks"]["wm_valid"], 0)          # its interval (61, 66] lost two rows
        self.assertIsNone(by_step[61]["action_exec"][2])                 # step 64
        self.assertEqual(s.segments[0], {"episode_id": 0, "first_decision_step": 66, "last_decision_step": 211,
                                         "n_steps": 25})

    def test_missing_commit_variant_discards_the_tail(self):
        out = synth.generate(os.path.join(self.tmp, "mc"), variant="missing_commit")
        # the seal says the session ran to step 407; the committed ctl rows stop at 300: a builder must refuse it
        with self.assertRaisesRegex(reader.ReaderError, "C34 ctl/: committed rows end at physics_step 300"):
            reader.open_session(out["session_dir"], decode_images=False)
        for s in (reader.open_session(out["session_dir"], decode_images=False, salvage=True),
                  reader.open_session(out["session_dir"], decode_images=False, strict=False)):
            self.assertEqual(s.streams["ctl"].uncommitted, ["chunk_00001.csv"])
            self.assertEqual(s.streams["ctl"].col("physics_step")[-1], 300)

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
        self.assertEqual(len(s.streams["ctl"]), 310)
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
                  lambda o: o["counts"]["telemetry"].update({"rows": 200, "chunks": 1, "last_key": 262}))
        with self.assertRaisesRegex(reader.ReaderError, "telemetry/: committed rows end at physics_step 262"):
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
            row = lines[200].split(",")                 # the 200th row (header is line 0)
            self.assertEqual(row[0], "262")
            return lines[:201] + [lines[200]] + lines[201:]

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
        d = next(x for x in base.decisions if x["decision_step"] == 151)
        lps = d["obs_last_physics_step"]          # 151: the image's pose row joins telemetry at that step
        session = copy_golden(self.tmp)
        rewrite_chunk(session, "telemetry", 0, lambda lines: [ln for ln in lines if not ln.startswith("%d," % lps)])
        s = reader.open_session(session, vehicle_card=CARD, strict=False)
        d2 = next(x for x in s.decisions if x["decision_step"] == 151)
        self.assertNotEqual(d2["obs_slot_id"], d["obs_slot_id"])
        self.assertEqual(d2["masks"]["wm_valid"], 0)

    def test_teleport_between_obs_and_decision_masks_it(self):
        session = copy_golden(self.tmp)
        base = reader.open_session(golden_source(), vehicle_card=CARD)
        d = next(x for x in base.decisions if x["decision_step"] == 301)
        self.assertEqual((d["obs_last_physics_step"], d["masks"]["teleport"]), (301, 0))

        def add_teleport(lines):
            end = lines[1].split(",")                   # 6,1,391,...,episode_end
            tele = ["6", "1", "298", str(int(end[3]) - 80), str(int(end[4]) - 2700), "teleport", "1", "reason=test"]
            end[0] = "7"
            return [lines[0], ",".join(tele), ",".join(end)]

        rewrite_chunk(session, "events", 1, add_teleport, keys=True)
        s = reader.open_session(session, vehicle_card=CARD)
        d = next(x for x in s.decisions if x["decision_step"] == 301)
        self.assertEqual((d["masks"]["teleport"], d["masks"]["wm_valid"], d["masks"]["policy_valid"]), (0, 1, 1))
        d = next(x for x in s.decisions if x["decision_step"] == 295)        # the teleport lies in its interval
        self.assertEqual((d["masks"]["teleport"], d["masks"]["wm_valid"], d["masks"]["policy_valid"]), (1, 0, 0))

    def test_command_held_and_exec_channels_complete(self):
        for column, value in (("hl_kappa", "0.012345"), ("llc_throttle", "")):
            session = copy_golden(self.tmp, column)
            set_column(session, "ctl", 0, column, lambda key, v: value if key == 154 else v)
            s = reader.open_session(session, vehicle_card=CARD, strict=False)
            d = next(x for x in s.decisions if x["decision_step"] == 151)
            self.assertEqual(d["masks"]["wm_valid"], 0, column)
            self.assertEqual(d["masks"]["policy_valid"], 1, column)    # the label at C_t is unaffected

    def test_react_visible_at_the_label_step(self):
        session = copy_golden(self.tmp)
        set_column(session, "ctl", 0, "react_visible", lambda key, v: "0" if key == 151 else v)
        s = reader.open_session(session, vehicle_card=CARD)
        d = next(x for x in s.decisions if x["decision_step"] == 151)
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
        with self.assertRaisesRegex(reader.ReaderError, "system drive"):
            reader.open_session("/mnt/z/Synthetic Steam (x86)/__dd5_no_such_dir__/" + GOLDEN_ID)


crash_dv = load("dc2_crash_dv", "data/dc2/crash_dv.py")
E01_RUNS = os.path.join(FIXTURES, "e01_s105_runs.csv")
STEP_MS = 1000.0 / 30.0           # the 1/30 s physics step of 1.8 (probe run 4, E-10)


def e01_runs():
    """{run: [(t_ms, vx, vz)]} of the E-01 S105 per-step probe (full brakes, steering, full throttle; resets
    excluded): the physics time of each engine step and its horizontal world-frame velocity."""
    runs = {}
    with open(E01_RUNS, "rb") as f:
        lines = f.read().decode("ascii").split("\n")[1:-1]
    for line in lines:
        run, _, step, _, vx, vz = line.split(",")
        runs.setdefault(int(run), []).append((int(step) * STEP_MS, float(vx), float(vz)))
    return runs


class CrashDetectorTest(unittest.TestCase):
    """The reference detector of contract 9.5.1 (data/dc2/crash_dv.py) on frame-bound callbacks of 1 to 3 merged
    steps of 1/30 s (freeze decision D-A)."""

    PATTERNS = ((1,), (2,), (3,), (1, 2, 1, 3, 1, 1, 2, 1, 3, 2), (3, 1, 3, 3, 2, 3, 1, 3))

    def test_e01_braking_never_crosses_the_threshold(self):
        runs = e01_runs()
        self.assertEqual(len(runs), 6)
        for pattern in self.PATTERNS:
            worst = 0.0
            for samples in runs.values():
                series, det = crash_dv.dv_series(crash_dv.frame_bound(samples, pattern))
                self.assertEqual(det.onsets, [], pattern)
                worst = max([worst] + [dv for _, dv in series if dv is not None])
            # 2.01 m/s per 100 ms at full brake (E-01), whatever the merging; interpolation may shave or add a little
            self.assertLess(worst, 2.3, pattern)
            self.assertGreater(worst, 1.5, pattern)

    def test_a_window_of_callbacks_would_fire_on_braking(self):
        # the review's H1: 3 callbacks of 3 merged steps span 300 ms, and full braking gives 5.09 m/s over 300 ms
        worst = 0.0
        for run in e01_runs().values():
            samples = crash_dv.frame_bound(run, (3,))
            worst = max([worst] + [math.hypot(y[1] - x[1], y[2] - x[2]) for x, y in zip(samples, samples[3:])])
        self.assertGreater(worst, 4.0)

    def test_an_impact_fires_at_any_merging(self):
        for pattern in self.PATTERNS:
            samples = [(i * STEP_MS, 0.0, 8.0 if i < 60 else 0.0) for i in range(120)]
            series, det = crash_dv.dv_series(crash_dv.frame_bound(samples, pattern))
            self.assertEqual(len(det.onsets), 1, pattern)
            self.assertGreaterEqual(det.onsets[0]["peak"], 7.9, pattern)
            self.assertAlmostEqual(math.hypot(*det.onsets[0]["ref"]), 8.0, 4)

    def test_a_hitch_is_not_a_change(self):
        det = crash_dv.Detector()
        for i in range(10):
            dv = det.step(i * 34, 0.0, 8.0)
            if i < 3:
                self.assertIsNone(dv)                           # history shorter than the window
        # 300 ms without a callback, then a very different velocity: the samples around t - 100 ms are 300 ms apart
        self.assertIsNone(det.step(9 * 34 + 300, 0.0, 0.0))
        self.assertEqual(det.onsets, [])
        self.assertEqual(len(det.samples), 1)                  # the history restarted at the current sample
        t0 = 9 * 34 + 300
        self.assertIsNone(det.step(t0 + 50, 0.0, 0.0))         # history shorter than the window
        self.assertEqual(det.step(t0 + 100, 0.0, 0.0), 0.0)

    def test_a_reposition_clears_the_history_and_equal_times_replace(self):
        det = crash_dv.Detector()
        det.watch(0)
        for i in range(8):
            det.step(i * 33, 0.0, 8.0)
        det.reposition(260)
        self.assertIsNone(det.step(300, 0.0, 0.0))
        self.assertIsNone(det.step(300, 0.0, 5.0))              # no world-time advance: replaces the sample
        self.assertEqual(det.samples, [(300, 0.0, 5.0)])
        self.assertEqual(det.onsets, [])

    def test_the_start_up_suppression(self):
        # DC_CrashWatch suppresses onsets for 1000 ms after a slot is watched; dv_h is still measured
        det = crash_dv.Detector()
        det.watch(0)
        series = [det.step(t, 0.0, 8.0 if t < 300 else 0.0) for t in range(0, 600, 33)]
        self.assertEqual(det.onsets, [])
        self.assertGreater(max(v for v in series if v is not None), 7.9)
        for t in range(600, 2400, 33):
            det.step(t, 0.0, 8.0 if t < 1800 else 0.0)
        self.assertEqual([round(o["peak"], 3) for o in det.onsets], [8.0])

    def test_a_reposition_finalises_the_pending_onset(self):
        # the round-2 reproduction: 8 -> 0 is an 8 m/s onset; a reset and 30 m/s inside the old peak window must not
        # grow it (DC_CrashWatch.PrepareReposition writes the row before the move)
        det = crash_dv.Detector()
        det.watch(0)
        t = 0
        while t < 1500:
            det.step(t, 0.0, 8.0)
            t += 33
        det.step(t, 0.0, 0.0)
        self.assertEqual(len(det.onsets), 1)
        det.reposition(t + 10)
        det.step(t + 33, 0.0, 30.0)
        det.step(t + 66, 0.0, 30.0)
        self.assertEqual(len(det.onsets), 1)
        self.assertAlmostEqual(det.onsets[0]["peak"], 8.0, 3)
        self.assertIsNone(det.pending)


class CrashTest(unittest.TestCase):
    """Contract 9.5.1 on the crash variant of make_synthetic, in the shape DD-4-min writes it (frame-bound rows, D-A):
    in episode 1 the ego stops dead against a rock and is destroyed at least two callbacks after the detector's onset,
    in a frame that issued an image request; the crash rows, the episode end and the seal share the terminal step L;
    the session ends with ego_destroyed, status ok; the seal (no drain) leaves that request pending while the engine
    still writes its file, whose pose row never exists (contract section 6, review round 1)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temp_dir("dc2_reader_crash_")
        cls.out = synth.generate(os.path.join(cls.tmp, "crash"), variant="crash")
        cls.session = reader.open_session(cls.out["session_dir"], vehicle_card=CARD)
        cls.golden = reader.open_session(GOLDEN_DIR, vehicle_card=CARD)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_ego_destroyed_is_a_normal_seal(self):
        s = self.session                    # strict=True raised nothing
        self.assertEqual((s.end_get("status"), s.end_get("end_reason")), ("ok", "ego_destroyed"))
        self.assertEqual(s.problems_for("C01"), [])

    def test_crash_rows_at_the_terminal_step(self):
        s = self.session
        self.assertEqual([c["kind"] for c in s.crash_events], ["teleport", "collision", "veh_damage", "veh_fire_start"])
        self.assertTrue(all(c["problems"] == [] for c in s.crash_events), s.crash_events)
        last = s.streams["ctl"].col("physics_step")[-1]
        ep = {e: i for i, e in enumerate(s.streams["episodes"].col("episode_id"))}[1]
        self.assertEqual(s.streams["episodes"].col("end_physics_step")[ep], last)
        self.assertEqual(s.end_get("end.physics_step"), last)
        self.assertEqual(s.streams["episodes"].col("termination_reason")[ep], "collision")
        crash = s.crash_events[1:]
        self.assertEqual([(c["physics_step"], c["episode_id"], c["in_episode"]) for c in crash], [(last, 1, True)] * 3)
        collision = crash[0]
        steps = s.streams["ctl"].col("physics_step")
        onset = collision["values"]["onset_step"]
        self.assertIn(onset, steps[:-2])                                   # at least two callbacks before L
        self.assertEqual((collision["values"]["src"], collision["values"]["other"]), ("dv_contact", "rock"))
        self.assertGreaterEqual(collision["values"]["dv_mps"], 4.0)
        # telemetry.dv_max_mps carries the same quantity at the impact (one telemetry row per callback)
        tel = s.streams["telemetry"]
        dv = dict(zip(tel.col("physics_step"), tel.col("dv_max_mps")))
        self.assertGreaterEqual(dv[onset], 4.0)
        self.assertLess(max(v for p, v in dv.items() if v is not None and p < onset), 4.0)
        self.assertEqual(crash[1]["values"]["state"], "destroyed")

    def test_the_seal_leaves_an_image_in_flight(self):
        """C-H1: the request of L's frame is pending at the seal; its file exists (the engine writes it) and decodes
        to the last render frame, which has no pose row: not joined, never obs_t."""
        s = self.session
        end_frame = s.end_get("end.render_frame")
        self.assertEqual(s.end_get("images.pending_at_end"), 1)
        last = s.images[-1]
        self.assertEqual((last["issue_status"], last["done_render_frame"]), ("issued", None))
        self.assertTrue(last["fid_ok"])
        self.assertEqual((last["captured_render_frame"], last["pose_after_seal"]), (end_frame, 1))
        self.assertIsNone(last["pose_hf_index"])
        self.assertEqual(sum(r["pose_after_seal"] for r in s.images), 1)
        self.assertNotIn(last["slot_id"], [d["obs_slot_id"] for d in s.decisions])
        with open(os.path.join(s.path, "_engine_logs", "console.log"), encoding="ascii") as f:
            self.assertIn("f_%08d" % last["slot_id"], f.read())

    def test_pending_at_end_must_be_the_integer_0_or_1(self):
        """Review round 4: images.pending_at_end is the JSON integer 0 or 1; a bool, float, string or null never grants
        the exemption (the predicate C13a shares)."""
        for value, want in ((0, (0, True)), (1, (1, True)), (1.0, (0, False)), ("1", (0, False)), (True, (0, False)),
                            (False, (0, False)), (None, (0, False)), (3.0, (0, False)), (2, (0, False)),
                            (-1, (0, False))):
            end = {"images.pending_at_end": value}
            fake = types.SimpleNamespace(end=end, end_get=lambda k, d=None, end=end: end.get(k, d))
            self.assertEqual(reader.pending_at_end(fake), want, repr(value))
        tmp = temp_dir("dc2_reader_pending_type_")
        try:
            for name, value in (("float", 1.0), ("str", "1"), ("bool", True), ("int", 1)):
                dst = os.path.join(tmp, name, self.out["session_id"])
                shutil.copytree(self.out["session_dir"], dst)
                edit_json(os.path.join(dst, "session_end.json"),
                          lambda o, value=value: o["images"].update({"pending_at_end": value}))
                s = reader.open_session(dst, vehicle_card=CARD)
                self.assertEqual(sum(r["pose_after_seal"] for r in s.images), int(type(value) is int), name)
        finally:
            shutil.rmtree(tmp)

    def test_the_exemption_needs_exactly_one_pending(self):
        """Review round 3, N6: the exemption is the unique newest issued slot with images.pending_at_end 1; a session
        declaring 2 pending exempts nothing."""
        tmp = temp_dir("dc2_reader_pending_")
        try:
            dst = os.path.join(tmp, self.out["session_id"])
            shutil.copytree(self.out["session_dir"], dst)
            edit_json(os.path.join(dst, "session_end.json"), lambda o: o["images"].update({"pending_at_end": 2}))
            s = reader.open_session(dst, vehicle_card=CARD)
            self.assertEqual(sum(r["pose_after_seal"] for r in s.images), 0)
            self.assertEqual(sum(r["pose_after_seal"] for r in self.session.images), 1)
            self.assertEqual([r["slot_id"] for r in self.session.images if r["pose_after_seal"]],
                             [max(r["slot_id"] for r in self.session.images if r["issue_status"] == "issued")])
        finally:
            shutil.rmtree(tmp)

    def test_first_terminal_ego_event(self):
        s = self.session
        self.assertEqual(sorted(s.terminal_ego), [1])
        self.assertEqual(s.terminal_ego[1]["kind"], "collision")         # its onset precedes the destroyed row
        self.assertEqual(s.terminal_ego_confirmed[1]["kind"], "collision")

    def test_policy_valid_is_0_from_the_crash_on(self):
        n = self.session.clocks["N"]
        onset = self.session.terminal_ego[1]["step"]
        golden = {d["decision_step"]: d["masks"] for d in self.golden.decisions}
        before = after = 0
        for d in self.session.decisions:
            if d["decision_step"] + (d["interval_steps"] or n) < onset:
                self.assertEqual(d["masks"], golden[d["decision_step"]], d["decision_step"])
                before += 1
            elif d["episode_id"] == 1:
                self.assertEqual(d["masks"]["policy_valid"], 0, d["decision_step"])
                after += 1
        self.assertGreater(before, 50)
        self.assertGreater(after, 0)

    def test_an_onset_outside_its_episode_is_not_terminal(self):
        tmp = temp_dir("dc2_reader_crash_out_")
        try:
            dst = os.path.join(tmp, self.out["session_id"])
            shutil.copytree(self.out["session_dir"], dst)

            def early(lines):                # the review's probe: onset_step -100 claimed for episode 1
                return [l.replace("onset_step=%d;" % (self.session.terminal_ego[1]["step"]), "onset_step=-100;")
                        for l in lines]
            names = sorted(n for n in os.listdir(os.path.join(dst, "events")) if n.startswith("chunk_"))
            rewrite_chunk(dst, "events", int(names[-1][6:11]), early)
            s = reader.open_session(dst, vehicle_card=CARD)
            collision = [c for c in s.crash_events if c["kind"] == "collision"][0]
            self.assertEqual((collision["step"], collision["in_episode"]), (-100, False))
            self.assertEqual(s.terminal_ego[1]["kind"], "veh_damage")     # the destroyed row still counts
        finally:
            shutil.rmtree(tmp)

    def test_chronology_and_confirmation_decide_terminal_events(self):
        s = reader.open_session(GOLDEN_DIR, vehicle_card=CARD, decode_images=False)
        ev = s.streams["events"]
        episodes = s.streams["episodes"]
        start, end = episodes.col("start_physics_step")[0], episodes.col("end_physics_step")[0]
        ego = s.streams["telemetry"].col("ego_slot_id")[0]
        row = {c: None for c in ev.columns}
        row.update({"event_seq": len(ev), "episode_id": 0, "render_frame": 0, "world_time_ms": 0, "kind": "rollover",
                    "veh_slot_id": ego})
        # confirmed at start + 20 with an onset at start + 40: chronologically invalid, never terminal
        ev.append_rows([dict(row, physics_step=start + 20, payload="onset_step=%d;up_y=0.20" % (start + 40))])
        reader._read_crash_events(s)
        bad = [c for c in s.crash_events if c["kind"] == "rollover"][0]
        self.assertEqual((bad["in_episode"], bad["chrono_ok"]), (True, False))
        self.assertEqual(s.terminal_ego, {})
        # confirmed after the episode ended (end + 30): the core could not latch it, so it is not the termination
        ev.append_rows([dict(row, event_seq=len(ev), physics_step=end + 30,
                             payload="onset_step=%d;up_y=0.20" % (end - 3))])
        reader._read_crash_events(s)
        self.assertEqual(s.terminal_ego[0]["step"], end - 3)              # still masks policy labels from its onset
        self.assertFalse(s.terminal_ego_confirmed[0]["confirmed_inside"])  # the core could not have latched it
        # confirmed inside the episode
        ev.append_rows([dict(row, event_seq=len(ev), physics_step=end - 5,
                             payload="onset_step=%d;up_y=0.20" % (end - 35))])
        reader._read_crash_events(s)
        self.assertEqual(s.terminal_ego_confirmed[0]["confirm_step"], end - 5)
        self.assertTrue(s.terminal_ego_confirmed[0]["confirmed_inside"])

    def test_a_traffic_reset_never_masks_the_ego(self):
        s = reader.open_session(GOLDEN_DIR, vehicle_card=CARD)
        before = [dict(d["masks"]) for d in s.decisions]
        ev = s.streams["events"]
        mid = [d["decision_step"] for d in s.decisions if d["masks"]["wm_valid"] == 1][40]
        row = {c: None for c in ev.columns}
        row.update({"event_seq": len(ev), "episode_id": 0, "physics_step": mid, "render_frame": 0,
                    "world_time_ms": 0, "kind": "teleport", "veh_slot_id": 9, "payload": "reason=stuck_reset"})
        ev.append_rows([row])
        reader._build_decisions(s)
        self.assertEqual([d["masks"] for d in s.decisions], before)       # the review's reproduction, fixed
        ev.append_rows([dict(row, event_seq=len(ev), veh_slot_id=s.streams["telemetry"].col("ego_slot_id")[0])])
        reader._build_decisions(s)
        masked = [d for d in s.decisions if d["masks"]["teleport"] == 1]
        self.assertTrue(masked and all(d["masks"]["wm_valid"] == 0 for d in masked))

    def test_payload_parsing(self):
        payloads = self.session.schema.payloads
        self.assertEqual(reader.parse_payload(""), [])
        self.assertEqual(reader.parse_payload(None), [])
        self.assertEqual(reader.parse_payload("a=1;b=x"), [("a", "1"), ("b", "x")])
        for bad in ("a", "a=", "=1", "a=1;;b=2"):
            with self.assertRaises(ValueError, msg=bad):
                reader.parse_payload(bad)
        values, problems = reader.check_payload("driver_state", "state=dead;in_seat=0", payloads)
        self.assertEqual((values, problems), ({"state": "dead", "in_seat": 0}, []))
        for text, word in (("state=asleep;in_seat=0", "not one of"), ("state=dead", "required key in_seat"),
                           ("state=dead;in_seat=0;mood=bad", "not a driver_state key"),
                           ("state=dead;in_seat=0;state=alive", "repeated"), ("state=dead;in_seat=2", "0 or 1")):
            _, problems = reader.check_payload("driver_state", text, payloads)
            self.assertTrue(any(word in p for p in problems), (text, problems))
        _, problems = reader.check_payload("rollover", "onset_step=4;up_y=1e-3", payloads)
        self.assertTrue(problems)                                        # no exponent (section 8)
        self.assertEqual(reader.check_payload("veh_fire_end", "", payloads), ({}, []))
        self.assertTrue(reader.check_payload("veh_fire_end", "fire_state=burning", payloads)[1])
        _, problems = reader.check_payload("collision", "onset_step=4;dv_mps=5.00;v_mps=3.00;other=rock;src=dv;"
                                           "other_prefab=a=b", payloads)
        self.assertTrue(problems)
        # payload ranges and minimums (the review's probe: health=2.00;movement_damage=-4.00)
        _, problems = reader.check_payload("veh_damage", "state=intermediary;health=2.00;movement_damage=-4.00",
                                           payloads)
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any("outside [0, 1]" in p for p in problems))
        for kind, text in (("rollover", "onset_step=4;up_y=1.50"),
                           ("collision", "onset_step=4;dv_mps=-1.00;v_mps=3.00;other=rock;src=dv"),
                           ("stuck", "stuck_s=-2.00")):
            self.assertTrue(reader.check_payload(kind, text, payloads)[1], text)
        self.assertEqual(reader.check_payload("rollover", "onset_step=4;up_y=-1.00", payloads)[1], [])

    def test_terminal_rules(self):
        self.assertEqual(reader.terminal_reason("collision", {"src": "dv"}), "collision")
        self.assertIsNone(reader.terminal_reason("collision", {"src": "contact"}))
        self.assertIsNone(reader.terminal_reason("veh_damage", {"state": "undamaged", "movement_damage": 0.3}))
        self.assertEqual(reader.terminal_reason("veh_damage", {"state": "undamaged", "movement_damage": 1.0}),
                         "collision")
        self.assertEqual(reader.terminal_reason("veh_fire_start", {}), "collision")
        self.assertEqual(reader.terminal_reason("rollover", {"up_y": 0.2}), "rollover")
        for kind in ("driver_state", "dismount", "remount", "veh_fire_end", "stuck", "teleport"):
            self.assertIsNone(reader.terminal_reason(kind, {}), kind)
        # onset and confirmation (9.5.1): a rollover begins at its first tilted step and is confirmed at its row
        self.assertEqual(reader.onset_and_confirmation("rollover", {"onset_step": 40}, 55), (40, 55))
        self.assertEqual(reader.onset_and_confirmation("collision", {"onset_step": 40}, 52), (40, 40))
        self.assertEqual(reader.onset_and_confirmation("veh_damage", {}, 52), (52, 52))


if __name__ == "__main__":
    unittest.main()
