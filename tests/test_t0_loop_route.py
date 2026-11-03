#!/usr/bin/env python3
"""Tests of the Arland T0 loop route (DD-7-min): addons/DataCapture/Configs/routes/arland_t0_loop.json, its generator
tools/routes/make_t0_loop.py and the engine road dump it is built from (tools/routes/data/e08a_<launch_id>/).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0_loop_route.py

  - the route passes the DD-11-min route schema (drivedit_nav.validate_route / load_route): arland, train, closed,
    right lane, one route_end stop at the end, sorted maneuvers;
  - it is built from the ENGINE road graph: the versioned E-08a dump of probe run 1 (its files are the ones source.txt
    names by sha256, the footer says complete, and they equal the play's files when the play is on this machine);
    source.road_graph names that dump, every loop road is an engine road of it with its road_key, every maneuver's
    junction_id names engine roads only (no decoder road id such as the old j452_458), and the turns are the engine
    junctions' (e165_167_17, e136_133_48, e133_132, e161_176, all right on this clockwise loop);
  - every point lies on the engine's road surface (centreline within half the engine width, plus a junction allowance;
    a 1 m connector counts with the width of the roads it joins), checked here on the engine's own points with
    straight segments, independently of the generator's spline; away from junctions the line is in the right lane:
    right of its engine road's centreline by about a quarter of that road's width;
  - it lies inside TRAIN: no point in the VAL-L box x in [1536, 2560), z in [512, 1536) or its 150 m buffer
    (dc-2.0.md 4; splits_v2, still PROPOSED);
  - C17 content (DD-5-min): left and right turning each >= 0.2 rad by C17's route curvature (heading change over
    +-5 m), and the left share of its mirror image differs by more than C17's 0.25; turning radius >= 6.5 m;
  - the file is canonical JSON plus one LF, so sha256 of the file without its LF (DC_ExpertHL's episodes.route_sha256)
    is drivedit_nav's Route.sha256; it keeps the form DC_Route reads (one "polyline" key, numbers without exponent,
    source.n_points, ASCII) and holds every key DC_Route.Load asks for;
  - with splits_v2 on this machine the generator rebuilds the committed file byte for byte from the versioned dump
    (skipped elsewhere), and with the decoder polylines there no decoder trail crosses the loop (trail 414, which
    crossed the old loop, is far away); the trail check tells a crossing from a trail ending on the road; the
    generator refuses Everon paths, an incomplete or altered dump, a dump whose roads have other keys, and outputs
    outside E: or RAM.
"""

import contextlib
import csv
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
DUMP = os.path.join(REPO, "tools", "routes", "data", "e08a_20261023T021038Z-wb-776e")
DUMP_FILES = ("arland_roads.csv", "arland_road_points.csv", "arland_roads_footer.txt")
LAUNCH_ID = "20261023T021038Z-wb-776e"
PLAY = "/mnt/e/DriveDiT/wb_profile/profile/DC_Probes/e08a/%s/play_01" % LAUNCH_ID       # read only, when present
PROGRAM = "/mnt/e/drivedit_realign/program"
POLYLINES = os.path.join(PROGRAM, "evidence", "maps", "Arland_road_polylines.tsv")
SPLITS = os.path.join(PROGRAM, "maps", "splits_v2_PROPOSED.json")
VAL_L_BOX = (1536.0, 2560.0, 512.0, 1536.0)     # dc-2.0.md 4: x in [1536, 2560), z in [512, 1536)
VAL_L_BUFFER_M = 150.0
C17_MIN_TURN_RAD = 0.2
C17_LR_DEV_MAX = 0.25
SURFACE_TOL_M = 0.6         # beyond half the engine width: the generator's junction allowance and the spline's bulge
LANE_TOL_M = 0.75           # the right lane, away from junctions (the generator checks 0.6 on its spline)
JUNCTION_M = 20.0


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


def read_dump(path):
    """{road_idx: (road_key, width_m, [(x, z)])} of an E-08a dump, read with csv only."""
    shapes = {}
    with open(os.path.join(path, "arland_road_points.csv"), "r", encoding="ascii", newline="") as f:
        for row in csv.DictReader(f):
            shapes.setdefault(int(row["road_idx"]), []).append((int(row["pt_idx"]), float(row["x"]), float(row["z"])))
    roads = {}
    with open(os.path.join(path, "arland_roads.csv"), "r", encoding="ascii", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["road_idx"])
            roads[idx] = (row["road_key"], float(row["width_m"]), [(x, z) for _, x, z in sorted(shapes[idx])])
    return roads


def distance_to(line, x, z):
    best = None
    for (ax, az), (bx, bz) in zip(line, line[1:]):
        dx, dz = bx - ax, bz - az
        seg2 = dx * dx + dz * dz
        t = 0.0 if seg2 <= 0 else max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / seg2))
        fx, fz = ax + t * dx, az + t * dz
        d = math.hypot(x - fx, z - fz)
        if best is None or d < best[0]:
            best = (d, fx, fz)
    return best


def read_kv(path):
    out = {}
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            if not line.startswith("#") and "=" in line:
                key, _, value = line.rstrip("\n").partition("=")
                out[key] = value
    return out


class LoopRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ROUTE_FILE, "rb") as f:
            cls.raw = f.read()
        cls.obj = json.loads(cls.raw.decode("ascii"))
        cls.route = drivedit_nav.load_route(ROUTE_FILE)
        cls.roads = read_dump(DUMP)

    def test_route_schema(self):
        self.assertEqual(drivedit_nav.validate_route(self.obj), [])
        obj = self.obj
        self.assertEqual((obj["route_id"], obj["map_id"], obj["split"], obj["lane_side"], obj["closed"]),
                         ("arland_t0_loop", "arland", "train", "right", True))
        self.assertEqual(obj["route_id"] + ".json", os.path.basename(ROUTE_FILE))
        self.assertEqual(obj["stops"], [{"s_m": obj["length_m"], "kind": "route_end"}])
        self.assertEqual(obj["polyline"][0], obj["polyline"][-1])

    def test_versioned_dump(self):
        # the three files are the ones source.txt names, complete, and (on this machine) the play's own files
        source = read_kv(os.path.join(DUMP, "source.txt"))
        self.assertEqual((source["probe"], source["launch_id"], source["play"]), ("e08a", LAUNCH_ID, "play_01"))
        for name in DUMP_FILES:
            with open(os.path.join(DUMP, name), "rb") as f:
                data = f.read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256." + name], name)
            self.assertNotIn(b"\r", data)
            data.decode("ascii")
            if os.path.isdir(PLAY):
                with open(os.path.join(PLAY, name), "rb") as f:
                    self.assertEqual(f.read(), data, name)
        footer = read_kv(os.path.join(DUMP, "arland_roads_footer.txt"))
        self.assertEqual((footer["map_id"], footer["complete"]), ("arland", "1"))
        self.assertEqual(int(footer["roads_written"]), len(self.roads))
        self.assertEqual(int(footer["points_total"]), sum(len(r[2]) for r in self.roads.values()))
        graph = self.obj["source"]["road_graph"]
        self.assertEqual((graph["probe"], graph["launch_id"]), ("e08a", LAUNCH_ID))
        self.assertEqual(graph["files"], {name: source["sha256." + name] for name in DUMP_FILES})

    def test_engine_roads_and_junctions(self):
        source = self.obj["source"]
        self.assertEqual(source["generator_version"], 2)
        self.assertNotIn("polylines", source.get("inputs", {}))           # no decoder input
        for leg in source["legs"]:
            key, width, _ = self.roads[leg["road"]]
            self.assertEqual((leg["key"], leg["width_m"]), (key, width))
        maneuvers = self.obj["maneuvers"]
        positions = [m["s_m"] for m in maneuvers]
        self.assertEqual(positions, sorted(set(positions)))
        self.assertTrue(all(0.0 < s < self.route.length_m for s in positions))
        for m in maneuvers:
            self.assertRegex(m["junction_id"], r"^e\d+(_\d+)+$")
            ids = [int(v) for v in m["junction_id"][1:].split("_")]
            self.assertTrue(all(i in self.roads and self.roads[i][1] > 1.5 for i in ids), m)
        turns = [(m["junction_id"], m["type"]) for m in maneuvers if m["type"] != "straight"]
        self.assertEqual(turns, [("e165_167_17", "right"), ("e136_133_48", "right"), ("e133_132", "right"),
                                 ("e161_176", "right")])
        # each maneuver lies where another engine road ends at the loop (the engine graph), not at a decoder road
        legs = set(leg["road"] for leg in source["legs"])
        for m in maneuvers:
            x, z = self.route.point_at(m["s_m"])
            arms = [i for i in (int(v) for v in m["junction_id"][1:].split("_")) if i not in legs]
            for arm in arms:
                line = self.roads[arm][2]
                self.assertLess(min(math.hypot(x - p[0], z - p[1]) for p in (line[0], line[-1])), 30.0, m)

    def test_on_the_engine_road_surface_and_right_lane(self):
        wide = {i: r for i, r in self.roads.items() if r[1] > 1.5}
        surface = {}
        for idx, (_, width, line) in self.roads.items():
            if width > 1.5:
                surface[idx] = width
                continue
            joined = [w for i, (_, w, other) in wide.items()
                      if min(distance_to(other, p[0], p[1])[0] for p in (line[0], line[-1])) <= 4.0]
            if joined:
                surface[idx] = max(joined)              # a connector carries the roads it joins
        junctions = [self.route.point_at(m["s_m"]) for m in self.obj["maneuvers"]]
        junctions += [(j["x"], j["z"]) for j in self.obj["source"]["junctions"]]
        legs = set(leg["road"] for leg in self.obj["source"]["legs"])
        points = self.obj["polyline"]
        lane = []
        for i, (x, _, z) in enumerate(points[:-1]):
            near = [(distance_to(self.roads[idx][2], x, z), idx) for idx in surface
                    if any(abs(px - x) < 60 and abs(pz - z) < 60 for px, pz in self.roads[idx][2])]
            excess = min(d[0] - surface[idx] / 2.0 for d, idx in near)
            self.assertLessEqual(excess, SURFACE_TOL_M, (i, x, z))
            if any(math.hypot(x - jx, z - jz) <= JUNCTION_M for jx, jz in junctions):
                continue
            (d, fx, fz), idx = min((d, idx) for d, idx in near if idx in legs and self.roads[idx][1] > 1.5)
            a, b = points[i - 1], points[i + 1]
            norm = math.hypot(b[0] - a[0], b[2] - a[2])
            right = ((b[2] - a[2]) / norm, -(b[0] - a[0]) / norm)
            offset = (x - fx) * right[0] + (z - fz) * right[1]
            lane.append(abs(offset - self.roads[idx][1] / 4.0))
        self.assertGreater(len(lane), 0.7 * len(points))
        self.assertLessEqual(max(lane), LANE_TOL_M)

    def test_inside_train(self):
        x0, x1, z0, z1 = VAL_L_BOX
        b = VAL_L_BUFFER_M
        for x, _, z in self.obj["polyline"]:
            self.assertFalse(x0 - b <= x < x1 + b and z0 - b <= z < z1 + b, (x, z))
        self.assertEqual(self.obj["source"]["train_rule"], {"val_l_box_xz": list(VAL_L_BOX), "buffer_m": b})
        self.assertEqual(self.obj["source"]["inputs"]["splits"]["status"], "PROPOSED")

    def test_length_and_turning_radius(self):
        self.assertTrue(2000.0 <= self.route.length_m <= 2700.0, self.route.length_m)
        s = 0.0
        while s <= self.route.length_m:
            self.assertLessEqual(abs(route_curvature(self.route, s)), 1.0 / 6.5, s)
            s += 1.0

    def test_c17_turning_content_and_mirror(self):
        left, right = turning(self.route)
        self.assertGreaterEqual(left, C17_MIN_TURN_RAD)
        self.assertGreaterEqual(right, C17_MIN_TURN_RAD)
        share = left / (left + right)
        # the mirror image swaps left and right: its share is 1 - share, so C17 fails it when |1 - 2 share| > 0.25
        self.assertGreater(abs(1.0 - 2.0 * share), C17_LR_DEV_MAX + 0.1)
        self.assertAlmostEqual(right - left, 2.0 * math.pi, delta=0.3)    # clockwise loop: one right turn net

    def test_canonical_form_and_route_sha(self):
        text = self.raw.decode("ascii")
        self.assertEqual(text, drivedit_nav.canonical_json(self.obj) + "\n")
        self.assertNotIn("\r", text)
        self.assertEqual(hashlib.sha256(self.raw[:-1]).hexdigest(), self.route.sha256)

    def test_dc_route_reads_it(self):
        text = self.raw.decode("ascii")
        self.assertEqual(text.count('"polyline"'), 1)          # DC_JsonScan.FindValue needs the key once
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

    def test_rebuilt_from_the_engine_dump(self):
        if not os.path.isfile(SPLITS):
            self.skipTest("splits_v2 not on this machine")
        generator = load_generator()
        argv = ["--splits", SPLITS, "--check", ROUTE_FILE]
        if os.path.isfile(POLYLINES):
            argv += ["--trails", POLYLINES]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = generator.main(argv)
        self.assertEqual(code, 0, out.getvalue())
        if os.path.isfile(POLYLINES):
            self.assertIn("decoder trails: none crosses the loop", out.getvalue())
            # trail 414 crossed the old loop on engine road 121; it is nowhere near this one
            trail = dict(generator.read_trails(POLYLINES))[414]
            self.assertGreater(min(math.hypot(x - p[0], z - p[2]) for x, z in trail for p in self.obj["polyline"]),
                               100.0)

    def test_trail_check_tells_a_crossing_from_an_end(self):
        generator = load_generator()
        route = [(0.0, 0.0, float(z)) for z in range(0, 41, 2)]
        crossing = (7, [(-20.0, 20.0), (20.0, 21.0)])               # goes on 20 m on both sides
        ending = (8, [(-30.0, 11.0), (1.5, 11.0)])                   # ends 1.5 m past the line: meets the road
        near = (9, [(-30.0, 31.0), (-4.0, 31.0)])                    # ends 4 m short of it
        crossings, touches = generator.check_trails(route, [crossing, ending, near])
        self.assertEqual([c[0] for c in crossings], [7])
        self.assertEqual(touches, [(8, 0.0), (9, 4.0)])

    def test_generator_refuses_altered_or_other_dumps(self):
        generator = load_generator()
        tmp = tempfile.mkdtemp(prefix="t0_loop_dump_", dir=self.tmp_base())
        try:
            dump = os.path.join(tmp, "e08a_copy")
            shutil.copytree(DUMP, dump)
            generator.read_dump(dump)
            with open(os.path.join(dump, "arland_roads_footer.txt"), "a", encoding="ascii") as f:
                f.write("x=1\n")
            with self.assertRaises(generator.Refusal) as caught:                  # not the file source.txt names
                generator.read_dump(dump)
            self.assertIn("sha256", str(caught.exception))
            shutil.copy(os.path.join(DUMP, "arland_roads_footer.txt"), dump)
            source = os.path.join(dump, "source.txt")
            with open(source, "r", encoding="ascii") as f:
                kept = [line for line in f if not line.startswith("sha256.")]
            with open(source, "w", encoding="ascii") as f:
                f.writelines(kept)
            with open(os.path.join(dump, "arland_roads.csv"), "r", encoding="ascii") as f:
                rows = f.read().replace("86/1682.48:3384.40/2521.30:3088.66", "87/1682.48:3384.40/2521.30:3088.66")
            with open(os.path.join(dump, "arland_roads.csv"), "w", encoding="ascii") as f:
                f.write(rows)
            roads, _ = generator.read_dump(dump)
            with self.assertRaises(generator.Refusal) as caught:                  # another dump: other keys
                generator.loop_roads(roads)
            self.assertIn("derived again", str(caught.exception))
            with open(os.path.join(dump, "arland_roads_footer.txt"), "w", encoding="ascii") as f:
                f.write("map_id=arland\ncomplete=0\n")
            with self.assertRaises(generator.Refusal) as caught:
                generator.read_dump(dump)
            self.assertIn("complete", str(caught.exception))
        finally:
            shutil.rmtree(tmp, True)

    def test_generator_refuses_everon_and_outputs_outside_e(self):
        generator = load_generator()
        self.assertIsNotNone(generator.sealed_reason("/mnt/e/x/Eden_road_dump"))
        self.assertIsNotNone(generator.sealed_reason("/mnt/e/x/everon/e08a"))
        self.assertIsNone(generator.sealed_reason(DUMP))
        with self.assertRaises(generator.Refusal):
            generator.check_out_path("/tmp/arland_t0_loop.json")
        with self.assertRaises(generator.Refusal):
            generator.check_out_path("/mnt/c/arland_t0_loop.json")
        tmp = tempfile.mkdtemp(prefix="t0_loop_", dir=self.tmp_base())
        try:
            sealed = os.path.join(tmp, "Eden_e08a")
            os.makedirs(sealed)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = generator.main(["--dump", sealed, "--splits", SPLITS])
            self.assertEqual(code, 2, out.getvalue())
            self.assertIn("sealed", out.getvalue())
        finally:
            shutil.rmtree(tmp, True)

    @staticmethod
    def tmp_base():
        base = os.path.realpath(os.environ.get("TMPDIR") or "/dev/shm")
        if not (base.startswith("/mnt/e/") or base.startswith("/dev/shm")):
            base = "/dev/shm"
        return base


if __name__ == "__main__":
    unittest.main()
