#!/usr/bin/env python3
"""Tests of the Arland T0 loop route (DD-7-min): addons/DataCapture/Configs/routes/arland_t0_loop.json and its
generator tools/routes/make_t0_loop.py.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0_loop_route.py

  - the route passes the DD-11-min route schema (drivedit_nav.validate_route / load_route): arland, train, closed,
    right lane, one route_end stop at the end, sorted maneuvers with the loop's three turns;
  - it lies inside TRAIN: no point in the VAL-L box x in [1536, 2560), z in [512, 1536) or its 150 m buffer
    (dc-2.0.md 4; splits_v2, still PROPOSED);
  - C17 content (DD-5-min): left and right turning each >= 0.2 rad by C17's route curvature (heading change over
    +-5 m), and the left share of its mirror image differs by more than C17's 0.25; turning radius >= 8 m;
  - the file is canonical JSON plus one LF, so sha256 of the file without its LF (DC_ExpertHL's episodes.route_sha256)
    is drivedit_nav's Route.sha256; it keeps the form DC_Route reads (one "polyline" key, numbers without exponent,
    source.n_points, ASCII) and holds every key DC_Route.Load asks for;
  - with the decoder inputs on this machine the generator rebuilds the committed file byte for byte (skipped
    elsewhere); the generator refuses Everon paths and outputs outside E: or RAM.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import drivedit_nav  # noqa: E402

ROUTE_FILE = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes", "arland_t0_loop.json")
GENERATOR = os.path.join(REPO, "tools", "routes", "make_t0_loop.py")
PROGRAM = "/mnt/e/drivedit_realign/program"
POLYLINES = os.path.join(PROGRAM, "evidence", "maps", "Arland_road_polylines.tsv")
SPLITS = os.path.join(PROGRAM, "maps", "splits_v2_PROPOSED.json")
VAL_L_BOX = (1536.0, 2560.0, 512.0, 1536.0)     # dc-2.0.md 4: x in [1536, 2560), z in [512, 1536)
VAL_L_BUFFER_M = 150.0
C17_MIN_TURN_RAD = 0.2
C17_LR_DEV_MAX = 0.25


def load_generator():
    spec = importlib.util.spec_from_file_location("make_t0_loop", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def route_curvature(route, s):
    """C17's route curvature (DD-5-min checks/behaviour.py): tangent heading change over [s - 5, s + 5] / length."""
    def tangent(t):
        a = route.point_at(max(t - 0.5, 0.0))
        b = route.point_at(min(t + 0.5, route.length_m))
        return math.atan2(b[0] - a[0], b[1] - a[1])
    lo = max(s - 5.0, 0.0)
    hi = min(s + 5.0, route.length_m)
    return wrap(tangent(hi) - tangent(lo)) / (hi - lo) if hi - lo > 1e-6 else 0.0


def turning(route, step=1.0):
    """(left, right) absolute turning [rad] of kappa x ds along the route."""
    left = right = 0.0
    s = 0.0
    while s < route.length_m:
        turn = route_curvature(route, s) * step
        if turn > 0:
            right += turn
        else:
            left -= turn
        s += step
    return left, right


class LoopRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ROUTE_FILE, "rb") as f:
            cls.raw = f.read()
        cls.obj = json.loads(cls.raw.decode("ascii"))
        cls.route = drivedit_nav.load_route(ROUTE_FILE)

    def test_route_schema(self):
        self.assertEqual(drivedit_nav.validate_route(self.obj), [])
        obj = self.obj
        self.assertEqual((obj["route_id"], obj["map_id"], obj["split"], obj["lane_side"], obj["closed"]),
                         ("arland_t0_loop", "arland", "train", "right", True))
        self.assertEqual(obj["route_id"] + ".json", os.path.basename(ROUTE_FILE))
        self.assertEqual(obj["stops"], [{"s_m": obj["length_m"], "kind": "route_end"}])
        self.assertEqual(obj["polyline"][0], obj["polyline"][-1])

    def test_inside_train(self):
        x0, x1, z0, z1 = VAL_L_BOX
        b = VAL_L_BUFFER_M
        for x, _, z in self.obj["polyline"]:
            self.assertFalse(x0 - b <= x < x1 + b and z0 - b <= z < z1 + b, (x, z))
        self.assertEqual(self.obj["source"]["train_rule"], {"val_l_box_xz": list(VAL_L_BOX), "buffer_m": b})
        self.assertEqual(self.obj["source"]["inputs"]["splits"]["status"], "PROPOSED")

    def test_length_and_turning_radius(self):
        self.assertTrue(1000.0 <= self.route.length_m <= 3000.0, self.route.length_m)
        s = 0.0
        while s <= self.route.length_m:
            self.assertLessEqual(abs(route_curvature(self.route, s)), 1.0 / 8.0, s)
            s += 1.0

    def test_c17_turning_content_and_mirror(self):
        left, right = turning(self.route)
        self.assertGreaterEqual(left, C17_MIN_TURN_RAD)
        self.assertGreaterEqual(right, C17_MIN_TURN_RAD)
        share = left / (left + right)
        # the mirror image swaps left and right: its share is 1 - share, so C17 fails it when |1 - 2 share| > 0.25
        self.assertGreater(abs(1.0 - 2.0 * share), C17_LR_DEV_MAX + 0.1)
        self.assertAlmostEqual(left - right, 2.0 * math.pi, delta=0.3)    # counter-clockwise loop: one left turn net

    def test_maneuvers(self):
        maneuvers = self.obj["maneuvers"]
        positions = [m["s_m"] for m in maneuvers]
        self.assertEqual(positions, sorted(set(positions)))
        self.assertTrue(all(0.0 < s < self.route.length_m for s in positions))
        turns = [(m["junction_id"], m["type"]) for m in maneuvers if m["type"] != "straight"]
        self.assertEqual(turns, [("j452_461", "right"), ("j461_93", "left"), ("j451_452", "left")])

    def test_canonical_form_and_route_sha(self):
        text = self.raw.decode("ascii")
        self.assertEqual(text, drivedit_nav.canonical_json(self.obj) + "\n")
        self.assertNotIn("\r", text)
        self.assertEqual(hashlib.sha256(self.raw[:-1]).hexdigest(), self.route.sha256)

    def test_dc_route_reads_it(self):
        text = self.raw.decode("ascii")
        self.assertEqual(text.count('"polyline":'), 1)
        self.assertEqual(self.obj["source"]["n_points"], len(self.obj["polyline"]))
        self.assertIsNone(re.search(r"[0-9]e[-+]?[0-9]", text.split('"polyline":', 1)[1].split("]]", 1)[0]))
        self.assertAlmostEqual(self.obj["length_m"], self.route.length_m, delta=0.05)

    def test_enforce_reader_keys_exist(self):
        with open(os.path.join(REPO, "src", "enfusion", "datacapture", "DC_ExpertHL.c"), "r", encoding="ascii") as f:
            source = f.read()
        load = source.split("class DC_Route", 1)[1].split("protected bool ReadPolyline", 1)[0]
        keys = re.findall(r'ReadValue\("(\w+)"', load) + re.findall(r'ReadContextNumber\(context, "(\w+)"', load)
        self.assertEqual(sorted(keys), sorted(["schema", "route_id", "map_id", "split", "lane_side", "closed", "length_m",
                                                "n_points"]))
        for key in keys:
            self.assertTrue(key in self.obj or key in self.obj["source"], key)
        self.assertIn('FindValue(text, "polyline")', source)

    def test_rebuilt_from_the_decoder_inputs(self):
        if not (os.path.isfile(POLYLINES) and os.path.isfile(SPLITS)):
            self.skipTest("decoder polylines / splits_v2 not on this machine")
        generator = load_generator()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = generator.main(["--polylines", POLYLINES, "--splits", SPLITS, "--check", ROUTE_FILE])
        self.assertEqual(code, 0, out.getvalue())

    def test_generator_refuses_everon_and_outputs_outside_e(self):
        generator = load_generator()
        self.assertIsNotNone(generator.sealed_reason("/mnt/e/x/Eden_road_polylines.tsv"))
        self.assertIsNotNone(generator.sealed_reason("/mnt/e/x/everon/routes.tsv"))
        self.assertIsNone(generator.sealed_reason(POLYLINES))
        with self.assertRaises(generator.Refusal):
            generator.check_out_path("/tmp/arland_t0_loop.json")
        with self.assertRaises(generator.Refusal):
            generator.check_out_path("/mnt/c/arland_t0_loop.json")
        base = os.path.realpath(os.environ.get("TMPDIR") or "/dev/shm")
        if not (base.startswith("/mnt/e/") or base.startswith("/dev/shm")):
            base = "/dev/shm"
        tmp = tempfile.mkdtemp(prefix="t0_loop_", dir=base)
        try:
            sealed = os.path.join(tmp, "Eden_road_polylines.tsv")
            with open(sealed, "w", encoding="ascii") as f:
                f.write("road_id\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = generator.main(["--polylines", sealed, "--splits", SPLITS])
            self.assertEqual(code, 2, out.getvalue())
            self.assertIn("sealed", out.getvalue())
        finally:
            shutil.rmtree(tmp, True)


if __name__ == "__main__":
    unittest.main()
