#!/usr/bin/env python3
"""Tests of the Arland T0 loop route (DD-7-min): addons/DataCapture/Configs/routes/arland_t0_loop.json, its generator
tools/routes/make_t0_loop.py and the engine road dump it is built from (tools/routes/data/e08a_<launch_id>/); and of
its counter-clockwise twin arland_t0_loop_rev.json (DD-7-ident pilot, --direction ccw, ReverseLoopRouteTest: the same
roads the other way round, the clockwise start in the other lane, the turns swapped, left and right balanced with
the clockwise loop, the same form and rules, rebuilt byte for byte).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0_loop_route.py

  - the route passes the DD-11-min route schema (drivedit_nav.validate_route / load_route): arland, train, closed,
    right lane, one route_end stop at the end, sorted maneuvers;
  - it is built from the ENGINE road graph: the versioned E-08a dump of probe run 1 (its files are the ones source.txt
    names by sha256, the footer says complete, and they equal the play's files when DD_PROBE_PROFILE names the
    Workbench profile folder that holds the play); source.road_graph names that dump, every loop road is an engine
    road of it with its road_key, every maneuver's junction_id names engine roads only (no decoder road id such as the
    old j452_458), and the turns are the engine junctions' (e165_167_17, e136_133_48, e161_176, all right on this
    clockwise loop);
  - the same-side rule (dc-2.0 freeze decision D-E, R-3, 2026-11-03): at e133_132 the route follows 133, which bends
    right less than side road 132, so it is straight (geometrically right); every recorded type follows the rule from
    the recorded branch angles; on synthetic forks +31/+63 deg gives straight/right, +40/+63 right/right, a single
    right turn (a left side road only) stays right, and a branch straighter than the route keeps the route's type;
  - the maneuvers are complete and local: every engine road wider than a connector that is not on the loop and comes
    within ARM_REACH_M of the line is named by the maneuver where it comes closest, and every road a junction_id names
    runs within ID_REACH_M of that maneuver's point;
  - every point lies on the engine's road surface at the REPORTED widths (connectors 1 m), checked here on the
    engine's own points with straight segments, independently of the generator's spline: away from junctions within
    half the width minus 0.5 m of a centreline, the generator's limit; beyond every strip only at the two connector
    gaps listed in CONNECTOR_GAPS (by at most CONNECTOR_ALLOW_M), whose decoder road covers those points when the
    decoder polylines are on this machine; away from junctions the line is in the right lane: right of its engine
    road's centreline by about a quarter of that road's width;
  - the lane corridor (source.lane_corridor_*, DC_ExpertHL's offroad termination) is the right lane of each road: a
    quarter of the engine width of the nearest loop road away from junctions, never more, so on the 4.5 m dirt roads
    and the 3 m road 136 a car leaves it before it leaves the road;
  - it lies inside TRAIN: no point in the VAL-L box x in [1536, 2560), z in [512, 1536) or its 150 m buffer
    (dc-2.0.md 4; splits_v2, still PROPOSED);
  - C17 content (DD-5-min): left and right turning each >= 0.2 rad by C17's route curvature (heading change over
    +-5 m), and the left share of its mirror image differs by more than C17's 0.25; turning radius >= 6.5 m;
  - the file is canonical JSON plus one LF, so sha256 of the file without its LF (DC_ExpertHL's episodes.route_sha256)
    is drivedit_nav's Route.sha256; it keeps the form DC_Route reads (one "polyline" key and one key of each lane
    corridor array, numbers without exponent, source.n_points, ASCII) and holds every key DC_Route.Load asks for;
  - with splits_v2 on this machine the generator rebuilds the committed file byte for byte from the versioned dump
    (skipped elsewhere), and with the decoder polylines there no decoder trail crosses the loop (trail 414, which
    crossed the old loop, is far away); the trail check tells a crossing from a trail ending on the road, also for a
    trail through a route point and by the trail length at the actual intersection; the generator refuses Everon
    paths, an incomplete or altered dump, a source.txt without the sha256 of every file, a dump whose roads have other
    keys, and outputs off the data drive and RAM.
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
# the play the dump was copied from ($profile:DC_Probes/e08a/<launch>/play_01, read only), compared byte for byte when
# DD_PROBE_PROFILE names the Workbench profile folder that holds DC_Probes
PROBE_PROFILE = os.environ.get("DD_PROBE_PROFILE", "")
PLAY = os.path.join(PROBE_PROFILE, "DC_Probes", "e08a", LAUNCH_ID, "play_01") if PROBE_PROFILE else ""
# DD-22: documents of a real machine are compared only when DD_TEST_LOCAL_CONFIG names its local config (opt-in).
_evidence_spec = importlib.util.spec_from_file_location(
    "dd_test_evidence", os.path.join(REPO, "tests", "fixtures", "local_config", "evidence.py"))
evidence = importlib.util.module_from_spec(_evidence_spec)
_evidence_spec.loader.exec_module(evidence)
PROGRAM = evidence.evidence_path("REALIGN_ROOT", "program") or "/nonexistent-dd22-evidence"
POLYLINES = os.path.join(PROGRAM, "evidence", "maps", "Arland_road_polylines.tsv")
SPLITS = os.path.join(PROGRAM, "maps", "splits_v2_PROPOSED.json")
VAL_L_BOX = (1536.0, 2560.0, 512.0, 1536.0)     # dc-2.0.md 4: x in [1536, 2560), z in [512, 1536)
VAL_L_BUFFER_M = 150.0
C17_MIN_TURN_RAD = 0.2
C17_LR_DEV_MAX = 0.25
SURFACE_MARGIN_M = 0.5      # away from junctions: within half the reported width minus this (the generator's limit)
# The engine's 1 m connectors whose gap the line crosses beyond every reported strip, with the decoder road (the prefab
# road of the decoder's polylines) that covers it: 166 bridges 167 and 137 (decoder dirt road 401, 4.5 m), 179
# bridges 161 and 163 (decoder asphalt road 95, 8 m). Points beyond every reported strip lie within
# CONNECTOR_REACH_M of such a connector and at most CONNECTOR_ALLOW_M beyond the nearest strip (0.07, 0.49 and
# 0.56 m on this route)
CONNECTOR_GAPS = {166: 401, 179: 95}
CONNECTOR_REACH_M = 3.0
CONNECTOR_ALLOW_M = 0.6
LANE_TOL_M = 0.75           # the right lane, away from junctions (the generator checks 0.6 on its spline)
JUNCTION_M = 20.0
ARM_REACH_M = 15.0          # an engine road this close to the line meets the loop: a maneuver names it ...
ARM_AT_M = 30.0             # ... at most this far along the route from where it comes closest
ID_REACH_M = 10.0           # every road a junction_id names runs this close to the maneuver's point


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


def read_decoder_roads(path, ids):
    """{road_id: (width_m, [(x, z)])} of the decoder's polylines TSV (prefab roads, read with csv only)."""
    out = {}
    with open(path, "r", encoding="ascii", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["road_id"]) in ids:
                points = [p.split(",") for p in row["points_xyz"].split(";") if p]
                out[int(row["road_id"])] = (float(row["width_m"]), [(float(p[0]), float(p[2])) for p in points])
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
        # the three files are the ones source.txt names, complete, and (with DD_PROBE_PROFILE) the play's own files
        source = read_kv(os.path.join(DUMP, "source.txt"))
        self.assertEqual((source["probe"], source["launch_id"], source["play"]), ("e08a", LAUNCH_ID, "play_01"))
        for name in DUMP_FILES:
            with open(os.path.join(DUMP, name), "rb") as f:
                data = f.read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256." + name], name)
            self.assertNotIn(b"\r", data)
            data.decode("ascii")
            if PLAY and os.path.isdir(PLAY):
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
        self.assertEqual(source["generator_version"], 4)
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
        self.assertEqual(turns, [("e165_167_17", "right"), ("e136_133_48", "right"), ("e161_176", "right")])
        # each maneuver lies where another engine road ends at the loop (the engine graph), not at a decoder road
        legs = set(leg["road"] for leg in source["legs"])
        for m in maneuvers:
            x, z = self.route.point_at(m["s_m"])
            arms = [i for i in (int(v) for v in m["junction_id"][1:].split("_")) if i not in legs]
            for arm in arms:
                line = self.roads[arm][2]
                self.assertLess(min(math.hypot(x - p[0], z - p[1]) for p in (line[0], line[-1])), 30.0, m)

    def test_maneuvers_are_complete_and_local(self):
        # independently of the generator's junction search: every engine road wider than a connector and not on the
        # loop that comes within ARM_REACH_M of the line is named by a maneuver within ARM_AT_M of where it comes
        # closest (a dropped maneuver fails), and every road a junction_id names, loop road or arm, runs within
        # ID_REACH_M of the maneuver's point (an id naming a road of another junction fails)
        legs = set(leg["road"] for leg in self.obj["source"]["legs"])
        line = [(p[0], p[2]) for p in self.obj["polyline"]]
        xs, zs = [p[0] for p in line], [p[1] for p in line]
        reach = {}
        for idx, (_, width, road) in self.roads.items():
            if width <= 1.5 or idx in legs:
                continue
            rx, rz = [p[0] for p in road], [p[1] for p in road]
            if max(rx) < min(xs) - ARM_REACH_M or min(rx) > max(xs) + ARM_REACH_M \
                    or max(rz) < min(zs) - ARM_REACH_M or min(rz) > max(zs) + ARM_REACH_M:
                continue
            best = min([(distance_to(line, x, z)[0], (x, z)) for x, z in road]
                       + [(distance_to(road, x, z)[0], (x, z)) for x, z in line])
            if best[0] <= ARM_REACH_M:
                reach[idx] = self.route.project(best[1][0], best[1][1])[0]
        named = {}
        for m in self.obj["maneuvers"]:
            x, z = self.route.point_at(m["s_m"])
            for idx in (int(v) for v in m["junction_id"][1:].split("_")):
                self.assertLessEqual(distance_to(self.roads[idx][2], x, z)[0], ID_REACH_M, (m, idx))
                if idx not in legs:
                    named.setdefault(idx, []).append(m["s_m"])
        self.assertEqual(sorted(reach), sorted(named))
        self.assertEqual(sorted(reach), [17, 48, 94, 132, 134, 135, 139, 162, 176, 180])
        for idx, s_near in reach.items():
            self.assertTrue(any(abs(s_m - s_near) <= ARM_AT_M for s_m in named[idx]), (idx, s_near, named[idx]))

    def test_same_side_rule_on_the_route(self):
        # D-E (R-3): at e133_132 the route follows 133 at about +22 deg while side road 132 turns about +54 deg (both
        # right): 133 is the branch closest to straight on and within 35 deg, so the maneuver is straight although the
        # route's own turn (+42 deg) says right. Every other type is the geometric one, and each recorded type follows
        # from the recorded branch angles
        generator = load_generator()
        source = self.obj["source"]
        self.assertEqual(generator.SAME_SIDE_STRAIGHT_DEG, 35.0)
        self.assertEqual(source["maneuver_rule"]["same_side_straight_within_deg"], 35.0)
        types = {m["junction_id"]: m["type"] for m in self.obj["maneuvers"]}
        facts = {f["junction_id"]: f for f in source["maneuver_branches"]}
        self.assertEqual(sorted(facts), sorted(types))
        changed = sorted(j for j, f in facts.items() if f["geometric_type"] != types[j])
        self.assertEqual(changed, ["e133_132"])
        fact = facts["e133_132"]
        self.assertEqual((fact["geometric_type"], types["e133_132"]), ("right", "straight"))
        self.assertEqual([idx for idx, _ in fact["branches"]], [132])
        self.assertTrue(0.0 < fact["route_branch_deg"] <= 35.0 < fact["branches"][0][1], fact)
        for junction_id, f in facts.items():
            self.assertEqual(generator.same_side_type(f["geometric_type"], f["route_branch_deg"],
                                                      [deg for _, deg in f["branches"]]), types[junction_id], f)

    def test_same_side_rule_on_synthetic_forks(self):
        generator = load_generator()

        def fork(route_deg, other_degs):
            """(type, fact) of the one maneuver where engine road 1 runs north for 100 m to (0, 100) and the route goes
            on along road 2 at route_deg (right positive); roads 3, 4, ... leave (0, 100) at other_degs."""
            def road(idx, z0, deg):
                a = math.radians(deg)
                return generator.Road(idx, "k%d" % idx, 8.0, [(10.0 * k * math.sin(a), 0.0, z0 + 10.0 * k * math.cos(a))
                                                              for k in range(11)])
            first, second = road(1, 0.0, 0.0), road(2, 100.0, route_deg)
            roads = {1: first, 2: second}
            for k, deg in enumerate(other_degs):
                roads[3 + k] = road(3 + k, 100.0, deg)
            end = second.line[-1]
            junctions = [generator.Junction(first, second, first.length, 0.0, 0.0, 100.0, 0.0),
                         generator.Junction(second, first, second.length, 0.0, end[0], end[2], 0.0)]
            points = first.line + second.line[1:]
            events = generator.junction_events(points, [first, second], junctions, roads)
            maneuvers, facts = generator.maneuvers_of(points, events)
            self.assertEqual(len(maneuvers), 1, maneuvers)
            return maneuvers[0]["type"], facts[0]

        kind, fact = fork(31.0, [63.0])                 # e133_132's case: the less sharp of two right branches
        self.assertEqual((kind, fact["geometric_type"]), ("straight", "right"))
        self.assertAlmostEqual(fact["route_branch_deg"], 31.0, delta=0.1)
        self.assertEqual(fact["branches"][0][0], 3)
        self.assertAlmostEqual(fact["branches"][0][1], 63.0, delta=0.1)
        self.assertEqual(fork(63.0, [31.0])[0], "right")            # the sharper one stays right
        self.assertEqual(fork(-31.0, [-63.0])[0], "straight")       # the same on the left
        self.assertEqual(fork(-63.0, [-31.0])[0], "left")
        self.assertEqual(fork(40.0, [63.0])[0], "right")            # beyond 35 deg: right / right, d_man tells
        self.assertEqual(fork(63.0, [40.0])[0], "right")
        self.assertEqual(fork(31.0, [-60.0])[0], "right")           # a single right turn: unchanged
        self.assertEqual(fork(31.0, [63.0, -10.0])[0], "right")     # 3 is straighter than the route: unchanged
        self.assertEqual(fork(20.0, [63.0])[0], "straight")         # geometrically straight already
        self.assertEqual(generator.same_side_type("right", 31.0, []), "right")
        self.assertEqual(generator.same_side_type("right", 35.0, [63.0]), "straight")      # within +-35 inclusive
        self.assertEqual(generator.same_side_type("right", 35.1, [63.0]), "right")
        self.assertEqual(generator.same_side_type("right", 31.0, [31.0]), "right")         # a tie: not the closest

    def test_on_the_engine_road_surface_and_right_lane(self):
        # every engine road at its reported width, connectors 1 m (no width inherited from the roads they join)
        junctions = [self.route.point_at(m["s_m"]) for m in self.obj["maneuvers"]]
        junctions += [(j["x"], j["z"]) for j in self.obj["source"]["junctions"]]
        legs = set(leg["road"] for leg in self.obj["source"]["legs"])
        decoder = {}
        if os.path.isfile(POLYLINES):
            decoder = read_decoder_roads(POLYLINES, set(CONNECTOR_GAPS.values()))
        points = self.obj["polyline"]
        lane = []
        beyond = []
        for i, (x, _, z) in enumerate(points[:-1]):
            near = [(distance_to(line, x, z), idx) for idx, (_, _, line) in self.roads.items()
                    if any(abs(px - x) < 60 and abs(pz - z) < 60 for px, pz in line)]
            excess, idx = min((d[0] - self.roads[idx][1] / 2.0, idx) for d, idx in near)
            in_junction = any(math.hypot(x - jx, z - jz) <= JUNCTION_M for jx, jz in junctions)
            if excess > 0.0:
                # beyond every reported strip: only at a listed connector gap, and on its decoder road
                gaps = [c for c in CONNECTOR_GAPS if distance_to(self.roads[c][2], x, z)[0] <= CONNECTOR_REACH_M]
                self.assertTrue(gaps and excess <= CONNECTOR_ALLOW_M, (i, x, z, excess, idx))
                beyond.append(i)
                if decoder:
                    width, road = decoder[CONNECTOR_GAPS[gaps[0]]]
                    self.assertLessEqual(distance_to(road, x, z)[0], width / 2.0 - SURFACE_MARGIN_M, (i, gaps))
            elif not in_junction:
                self.assertLessEqual(excess, -SURFACE_MARGIN_M, (i, x, z, idx))
            if in_junction:
                continue
            (d, fx, fz), idx = min((d, idx) for d, idx in near if idx in legs and self.roads[idx][1] > 1.5)
            a, b = points[i - 1], points[i + 1]
            norm = math.hypot(b[0] - a[0], b[2] - a[2])
            right = ((b[2] - a[2]) / norm, -(b[0] - a[0]) / norm)
            offset = (x - fx) * right[0] + (z - fz) * right[1]
            lane.append(abs(offset - self.roads[idx][1] / 4.0))
        self.assertGreater(len(lane), 0.7 * len(points))
        self.assertLessEqual(max(lane), LANE_TOL_M)
        self.assertEqual(beyond, [512, 513, 1235])     # 166's gap (s 1024-1026) and 179's (s 2470)

    def test_lane_corridor_is_the_right_lane_of_each_road(self):
        # DC_ExpertHL terminates offroad beyond the lane corridor half width w/2 + 0.5 m (r2 1.3). The corridor is the
        # right lane, half the road: away from junctions w/2 is a quarter of the engine width of the nearest loop road,
        # and nowhere more, so on the 4.5 m dirt roads (1.125 m) and the 3 m road 136 (0.75 m) the threshold lies at
        # most 0.5 m beyond the road edge, as on the 8 m roads (2 m)
        source = self.obj["source"]
        starts, halves = source["lane_corridor_s_m"], source["lane_corridor_half_width_m"]
        self.assertEqual(len(starts), len(halves))
        self.assertEqual(starts[0], 0.0)                        # as DC_Route.ReadCorridor checks them
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])), starts)
        self.assertLess(starts[-1], self.obj["length_m"])
        self.assertTrue(all(h > 0 for h in halves), halves)
        self.assertEqual(sorted(set(halves)), [0.75, 1.125, 2.0])
        junctions = [self.route.point_at(m["s_m"]) for m in self.obj["maneuvers"]]
        junctions += [(j["x"], j["z"]) for j in source["junctions"]]
        legs = [leg["road"] for leg in source["legs"] if leg["width_m"] > 1.5]
        dirt = 0
        for (x, _, z), s in zip(self.obj["polyline"][:-1], self.route.s[:-1]):
            half = halves[max(k for k, start in enumerate(starts) if start <= s + 1e-9)]
            _, idx = min((distance_to(self.roads[i][2], x, z)[0], i) for i in legs)
            quarter = self.roads[idx][1] / 4.0
            self.assertLessEqual(half, quarter + 1e-9, (s, idx))
            if not any(math.hypot(x - jx, z - jz) <= JUNCTION_M for jx, jz in junctions):
                self.assertEqual(half, quarter, (s, idx))
                dirt += self.roads[idx][1] < 5.0
        self.assertGreater(dirt, 300)                           # the dirt roads are a large part of the loop

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
        # the lane corridor arrays DC_Route.ReadCorridor reads by DC_JsonScan.ReadNumbers: each key once, flat number
        # arrays without exponent, in source
        corridor = dict(re.findall(r'static const string (CORRIDOR_\w+_KEY) = "(\w+)";', load))
        self.assertEqual(sorted(corridor.values()), ["lane_corridor_half_width_m", "lane_corridor_s_m"])
        text = self.raw.decode("ascii")
        for name, key in corridor.items():
            self.assertIn("DC_JsonScan.ReadNumbers(text, %s, " % name, load)
            self.assertEqual(text.count('"%s"' % key), 1, key)
            self.assertIn(key, self.obj["source"])
            value = text.split('"%s":' % key, 1)[1].split("]", 1)[0]
            self.assertRegex(value, r"^\[-?[0-9]+(\.[0-9]+)?(,-?[0-9]+(\.[0-9]+)?)*$", key)

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
        vertex = (10, [(-20.0, 20.0), (20.0, 20.0)])                 # through the route point (0, 20): one crossing
        # meets the line 8.02 m along the trail (at z 18.72): a crossing; the start of the route segment there,
        # (0, 18), projects to 7.54 m. Its start lies 6 m from the line: also a touch at 6 m
        cut = (11, [(-6.0, 13.4), (24.0, 40.0)])
        on_point = (12, [(-30.0, 30.0), (0.0, 30.0)])               # ends on the route point (0, 30): a touch
        crossings, touches = generator.check_trails(route, [crossing, ending, near, vertex, cut, on_point])
        self.assertEqual([c[0] for c in crossings], [7, 10, 11])
        self.assertEqual(crossings[1:], [(10, 0.0, 20.0), (11, 0.0, 18.7)])
        self.assertEqual(touches, [(8, 0.0), (9, 4.0), (11, 6.0), (12, 0.0)])
        self.assertEqual(generator.segment_cuts((-6.0, 0.0, 13.4), (24.0, 0.0, 40.0), (0.0, 0.0, 18.0),
                                                (0.0, 0.0, 20.0)), [0.2])
        self.assertEqual(generator.segment_cuts((0.0, 0.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, 2.0),
                                                (0.0, 0.0, 6.0)), [0.5, 1.0])        # on one line: the overlap

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
                lines = f.readlines()
            with open(source, "w", encoding="ascii") as f:                        # a copy without its hashes
                f.writelines(line for line in lines if not line.startswith("sha256.arland_roads.csv="))
            with self.assertRaises(generator.Refusal) as caught:
                generator.read_dump(dump)
            self.assertIn("names no sha256 of arland_roads.csv", str(caught.exception))
            # another dump (other road keys) whose source.txt names its files: read, but its loop roads are refused
            with open(os.path.join(dump, "arland_roads.csv"), "r", encoding="ascii") as f:
                rows = f.read().replace("86/1682.48:3384.40/2521.30:3088.66", "87/1682.48:3384.40/2521.30:3088.66")
            with open(os.path.join(dump, "arland_roads.csv"), "w", encoding="ascii") as f:
                f.write(rows)
            digest = hashlib.sha256(rows.encode("ascii")).hexdigest()
            with open(source, "w", encoding="ascii") as f:
                f.writelines("sha256.arland_roads.csv=%s\n" % digest if line.startswith("sha256.arland_roads.csv=")
                             else line for line in lines)
            roads, _ = generator.read_dump(dump)
            with self.assertRaises(generator.Refusal) as caught:                  # another dump: other keys
                generator.loop_roads(roads)
            self.assertIn("derived again", str(caught.exception))
            # a play directory (run_header.txt, no source.txt) has no hashes to check: the files as the probe wrote them
            os.remove(source)
            with open(os.path.join(dump, "run_header.txt"), "w", encoding="ascii") as f:
                f.write("probe=e08a\nlaunch.id=%s\nengine.build_version=1.8.0.13\n" % LAUNCH_ID)
            _, provenance = generator.read_dump(dump)
            self.assertEqual((provenance["launch_id"], provenance["play"]), (LAUNCH_ID, "e08a_copy"))
            with open(os.path.join(dump, "arland_roads_footer.txt"), "w", encoding="ascii") as f:
                f.write("map_id=arland\ncomplete=0\n")
            with self.assertRaises(generator.Refusal) as caught:
                generator.read_dump(dump)
            self.assertIn("complete", str(caught.exception))
        finally:
            shutil.rmtree(tmp, True)

    def test_generator_refuses_everon_and_outputs_outside_e(self):
        generator = load_generator()
        self.assertIsNotNone(generator.sealed_reason("/mnt/x/DataRoot/Eden_road_dump"))
        self.assertIsNotNone(generator.sealed_reason("/mnt/x/DataRoot/everon/e08a"))
        self.assertIsNone(generator.sealed_reason(DUMP))
        with self.assertRaises(generator.Refusal):
            generator.check_out_path("/tmp/arland_t0_loop.json")
        with self.assertRaises(generator.Refusal):
            generator.check_out_path("/mnt/z/Synthetic Steam (x86)/arland_t0_loop.json")
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
        if not base.startswith("/dev/shm"):
            base = "/dev/shm"
        return base


ROUTE_REV_FILE = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes", "arland_t0_loop_rev.json")


class ReverseLoopRouteTest(unittest.TestCase):
    """arland_t0_loop_rev (DD-7-ident pilot, make_t0_loop.py --direction ccw): the T0 loop's engine roads in the
    other driving order, in the right lane of that direction, starting at the clockwise start in the other lane, so
    the pilot's left and right turning balance; every rule of the clockwise loop holds for it too."""

    @classmethod
    def setUpClass(cls):
        with open(ROUTE_REV_FILE, "rb") as f:
            cls.raw = f.read()
        cls.obj = json.loads(cls.raw.decode("ascii"))
        cls.route = drivedit_nav.load_route(ROUTE_REV_FILE)
        with open(ROUTE_FILE, "rb") as f:
            cls.cw = json.loads(f.read().decode("ascii"))
        cls.cw_route = drivedit_nav.load_route(ROUTE_FILE)

    def test_route_schema(self):
        obj = self.obj
        self.assertEqual(drivedit_nav.validate_route(obj), [])
        self.assertEqual((obj["route_id"], obj["map_id"], obj["split"], obj["lane_side"], obj["closed"]),
                         ("arland_t0_loop_rev", "arland", "train", "right", True))
        self.assertEqual(obj["route_id"] + ".json", os.path.basename(ROUTE_REV_FILE))
        self.assertEqual(obj["stops"], [{"s_m": obj["length_m"], "kind": "route_end"}])
        self.assertEqual(obj["polyline"][0], obj["polyline"][-1])
        self.assertEqual(obj["source"]["reverse_of"], "arland_t0_loop")
        self.assertIn("counter-clockwise", obj["description"])

    def test_same_roads_other_way_round(self):
        generator = load_generator()
        cw = [leg["road"] for leg in self.cw["source"]["legs"]]
        ccw = [leg["road"] for leg in self.obj["source"]["legs"]]
        self.assertEqual(ccw, [cw[0]] + cw[1:][::-1])
        self.assertEqual(tuple((leg["road"], leg["key"]) for leg in self.obj["source"]["legs"]), generator.LOOP_REV)
        self.assertEqual(self.obj["source"]["road_graph"], self.cw["source"]["road_graph"])
        # the same junctions, the turns the other way: every left / right of one loop is the other's right / left
        cw_turns = {frozenset(m["junction_id"][1:].split("_")): m["type"] for m in self.cw["maneuvers"]}
        ccw_turns = {frozenset(m["junction_id"][1:].split("_")): m["type"] for m in self.obj["maneuvers"]}
        swap = {"left": "right", "right": "left"}
        self.assertEqual(sorted(t for t in cw_turns.values() if t != "straight"), ["right"] * 3)
        for roads, kind in cw_turns.items():
            if kind != "straight":
                self.assertEqual(ccw_turns.get(roads), swap[kind], sorted(roads))
        # the start: the clockwise start in the other lane (one lane width of the 8 m road 163), facing the other way
        p, q = self.obj["polyline"][0], self.cw["polyline"][0]
        self.assertAlmostEqual(math.hypot(p[0] - q[0], p[2] - q[2]), 4.0, delta=0.5)
        a, b = self.obj["polyline"][0], self.obj["polyline"][1]
        c, d = self.cw["polyline"][0], self.cw["polyline"][1]
        turn = wrap(math.atan2(b[0] - a[0], b[2] - a[2]) - math.atan2(d[0] - c[0], d[2] - c[2]))
        self.assertGreater(abs(turn), math.radians(170.0))

    def test_turning_balances_the_clockwise_loop(self):
        left, right = turning(self.route)
        cw_left, cw_right = turning(self.cw_route)
        self.assertGreaterEqual(min(left, right), C17_MIN_TURN_RAD)
        self.assertGreater(abs(1.0 - 2.0 * left / (left + right)), C17_LR_DEV_MAX + 0.1)   # C17 tells it from its mirror
        self.assertAlmostEqual(left - right, 2.0 * math.pi, delta=0.3)                     # one left turn net
        self.assertGreater(cw_right / cw_left, 1.5)
        both_left, both_right = left + cw_left, right + cw_right
        self.assertLessEqual(max(both_left, both_right) / min(both_left, both_right), 1.2)

    def test_inside_train_and_radius(self):
        x0, x1, z0, z1 = VAL_L_BOX
        b = VAL_L_BUFFER_M
        for x, _, z in self.obj["polyline"]:
            self.assertFalse(x0 - b <= x < x1 + b and z0 - b <= z < z1 + b, (x, z))
        self.assertTrue(2000.0 <= self.route.length_m <= 2700.0, self.route.length_m)
        s = 0.0
        while s <= self.route.length_m:
            self.assertLessEqual(abs(route_curvature(self.route, s)), 1.0 / 6.5, s)
            s += 1.0

    def test_canonical_form_and_the_form_dc_route_reads(self):
        text = self.raw.decode("ascii")
        self.assertEqual(text, drivedit_nav.canonical_json(self.obj) + "\n")
        self.assertEqual(hashlib.sha256(self.raw[:-1]).hexdigest(), self.route.sha256)
        self.assertGreater(len(text), 8191)        # one line longer than string.Substring returns: DD-7's fix applies
        self.assertEqual(text.count('"polyline"'), 1)
        self.assertEqual(self.obj["source"]["n_points"], len(self.obj["polyline"]))
        self.assertIsNone(re.search(r"[0-9]e[-+]?[0-9]", text.split('"polyline":', 1)[1].split("]]", 1)[0]))
        for key in ("lane_corridor_s_m", "lane_corridor_half_width_m"):
            self.assertEqual(text.count('"%s"' % key), 1, key)
            value = text.split('"%s":' % key, 1)[1].split("]", 1)[0]
            self.assertRegex(value, r"^\[-?[0-9]+(\.[0-9]+)?(,-?[0-9]+(\.[0-9]+)?)*$", key)
        self.assertEqual(sorted(set(self.obj["source"]["lane_corridor_half_width_m"])),
                         sorted(set(self.cw["source"]["lane_corridor_half_width_m"])))

    def test_rebuilt_from_the_engine_dump(self):
        if not os.path.isfile(SPLITS):
            self.skipTest("splits_v2 not on this machine")
        generator = load_generator()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = generator.main(["--direction", "ccw", "--splits", SPLITS, "--check", ROUTE_REV_FILE])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("route arland_t0_loop_rev", out.getvalue())


if __name__ == "__main__":
    unittest.main()
