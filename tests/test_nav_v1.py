#!/usr/bin/env python3
"""Tests for drivedit_nav.nav_v1 and the route schema (DD-11-min), stdlib unittest only.

    python3 -I -B tests/test_nav_v1.py -v
    python3 -I -B -m pytest -p no:cacheprovider tests/test_nav_v1.py

Acceptance (TICKETS_UNIFIED DD-11-min): on a straight route p20 lies 20 m ahead on the forward axis (+-0.05 m) with
zero lateral; signs follow the right-positive convention on curves; ordered progress handles a loop crossing;
d_stop = 200 with no stop; deterministic. Everything is synthetic geometry; no map data is read.

The one test that writes a file (load_route on a real path) uses $TMPDIR only when it lies under /dev/shm or /mnt/e,
else /dev/shm, and is skipped otherwise; it never falls back to the system temp directory.
"""

import copy
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import drivedit_nav  # noqa: E402
from drivedit_nav import (  # noqa: E402
    D_MAX_M, NAV_CHANNELS, Pose, RouteError, SealedMapError, load_route, nav_sequence, nav_v1, nav_vector,
    pose_from_transform, route_from_dict, to_legacy_6d, validate_route)

NAVMOD = importlib.import_module("drivedit_nav.nav_v1")  # the package name nav_v1 is the function
FEATURES_JSON = os.path.join(REPO, "docs", "capture_contract", "features_v1.json")
TOL_P20 = 0.05  # acceptance tolerance on p20 (m)


def tmp_base():
    tmpdir = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if (tmpdir.startswith("/dev/shm/") or tmpdir.startswith("/mnt/e/")) and os.path.isdir(tmpdir):
        return tmpdir
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    raise unittest.SkipTest("no TMPDIR on /dev/shm or /mnt/e and no writable /dev/shm; not using the system temp dir")


def route_dict(xz, map_id="arland", split="train", maneuvers=(), stops=None, closed=False, route_id="test_route",
               y=12.5):
    """A route dict; stops None = the one route_end stop at the end of the polyline (nav_v1 has terminal stops only)."""
    if stops is None:
        length = sum(math.hypot(xz[i][0] - xz[i - 1][0], xz[i][1] - xz[i - 1][1]) for i in range(1, len(xz)))
        stops = [(length, "route_end")]
    return {
        "schema": "drivedit.route/1",
        "route_id": route_id,
        "map_id": map_id,
        "split": split,
        "nav_version": "nav_v1",
        "lane_side": "right",
        "closed": closed,
        "polyline": [[float(x), y, float(z)] for x, z in xz],
        "maneuvers": [{"s_m": float(s), "type": t} for s, t in maneuvers],
        "stops": [{"s_m": float(s), "kind": k} for s, k in stops],
    }


def straight_xz(x0=100.0, z0=0.0, heading=0.0, length=1000.0, step=5.0):
    """A straight polyline from (x0, z0) along the given heading (0 = north, clockwise positive)."""
    n = int(round(length / step))
    sh, ch = math.sin(heading), math.cos(heading)
    return [(x0 + i * step * sh, z0 + i * step * ch) for i in range(n + 1)]


def curve_xz(turn, radius=50.0, lead=200.0, tail=200.0, step=1.0):
    """North for `lead` m from (0, 0), a 90 degree arc turning right (+1, towards +x) or left (-1), then straight."""
    pts = [(0.0, i * step) for i in range(int(lead / step) + 1)]
    cx = turn * radius  # arc centre on the side of the turn
    n_arc = int(round(0.5 * math.pi * radius / step))
    for i in range(1, n_arc + 1):
        a = 0.5 * math.pi * i / n_arc
        pts.append((cx - turn * radius * math.cos(a), lead + radius * math.sin(a)))
    ex, ez = pts[-1]
    for i in range(1, int(tail / step) + 1):
        pts.append((ex + turn * i * step, ez))
    return pts


def figure_eight_xz(a=100.0, n=4000):
    """Lemniscate of Gerono x = a sin t, z = a sin t cos t, started at t = pi/2 and closed. It crosses itself at the
    origin (t = pi and t = 2 pi) at a right angle, half a loop apart."""
    pts = []
    for i in range(n + 1):
        t = 0.5 * math.pi + 2.0 * math.pi * i / n
        pts.append((a * math.sin(t), a * math.sin(t) * math.cos(t)))
    return pts


def heading_of(x0, z0, x1, z1):
    return math.atan2(x1 - x0, z1 - z0)


def route_pose(route, s, lateral=0.0):
    """Pose on the route at arc length s, facing along it, offset `lateral` m to the right."""
    x0, z0 = route.point_at(max(s - 0.5, 0.0))
    x1, z1 = route.point_at(min(s + 0.5, route.length_m))
    h = heading_of(x0, z0, x1, z1)
    x, z = route.point_at(s)
    return Pose(x + lateral * math.cos(h), z - lateral * math.sin(h), h)


class StraightRouteTest(unittest.TestCase):
    def test_p20_on_forward_axis(self):
        route = route_from_dict(route_dict(straight_xz()))
        for s in (0.0, 100.3, 512.75, 900.0):
            nav = nav_v1(route, route_pose(route, s), progress_m=max(s - 1.0, 0.0))
            self.assertAlmostEqual(nav["p20_x"], 20.0, delta=TOL_P20)
            self.assertAlmostEqual(nav["p20_y"], 0.0, delta=TOL_P20)
            self.assertAlmostEqual(nav["p40_x"], 40.0, delta=TOL_P20)
            self.assertAlmostEqual(nav["p40_y"], 0.0, delta=TOL_P20)
            self.assertAlmostEqual(nav["p10_x"], 10.0, delta=TOL_P20)
            self.assertAlmostEqual(nav["progress_m"], s, delta=1e-6)

    def test_any_heading(self):
        for heading_deg in (-135.0, -30.0, 45.0, 90.0, 180.0):
            h = math.radians(heading_deg)
            route = route_from_dict(route_dict(straight_xz(x0=1500.0, z0=900.0, heading=h, length=400.0)))
            nav = nav_v1(route, route_pose(route, 150.0), progress_m=149.0)
            self.assertAlmostEqual(nav["p20_x"], 20.0, delta=TOL_P20, msg=heading_deg)
            self.assertAlmostEqual(nav["p20_y"], 0.0, delta=TOL_P20, msg=heading_deg)
            self.assertAlmostEqual(nav["p40_x"], 40.0, delta=TOL_P20, msg=heading_deg)

    def test_lateral_offset_sign(self):
        route = route_from_dict(route_dict(straight_xz()))
        right = nav_v1(route, route_pose(route, 300.0, lateral=1.5), progress_m=299.0)
        left = nav_v1(route, route_pose(route, 300.0, lateral=-1.5), progress_m=299.0)
        # ego right of the line: the route lies to its left (negative y), and vice versa
        self.assertAlmostEqual(right["p20_y"], -1.5, delta=1e-9)
        self.assertAlmostEqual(left["p20_y"], 1.5, delta=1e-9)
        self.assertAlmostEqual(right["p20_x"], 20.0, delta=1e-9)
        self.assertAlmostEqual(right["dist_to_route_m"], 1.5, delta=1e-9)

    def test_progress_window_is_30_m(self):
        self.assertEqual(drivedit_nav.PROGRESS_WINDOW_M, 30.0)
        self.assertEqual(drivedit_nav.ROUTE_DEVIATION_M, 30.0)
        route = route_from_dict(route_dict(straight_xz()))
        # a pose 50 m ahead of the previous progress: progress may move by the window only, and the pose is 20 m
        # from that projection (still valid); 100 m ahead it is 70 m away (nav_valid 0)
        nav = nav_v1(route, route_pose(route, 250.0), progress_m=200.0)
        self.assertAlmostEqual(nav["progress_m"], 230.0, delta=1e-9)
        self.assertAlmostEqual(nav["dist_to_route_m"], 20.0, delta=1e-9)
        self.assertEqual(nav["nav_valid"], 1)
        far = nav_v1(route, route_pose(route, 300.0), progress_m=200.0)
        self.assertAlmostEqual(far["progress_m"], 230.0, delta=1e-9)
        self.assertEqual(far["nav_valid"], 0)

    def test_far_from_the_route_is_not_valid(self):
        route = route_from_dict(route_dict(straight_xz()))
        first = nav_v1(route, route_pose(route, 500.0))     # first call of an episode far from the route start
        self.assertEqual(first["nav_valid"], 0)
        self.assertAlmostEqual(first["dist_to_route_m"], 470.0, delta=1e-6)
        aside = nav_v1(route, route_pose(route, 300.0, lateral=31.0), progress_m=299.0)
        self.assertEqual(aside["nav_valid"], 0)
        near = nav_v1(route, route_pose(route, 300.0, lateral=29.0), progress_m=299.0)
        self.assertEqual(near["nav_valid"], 1)

    def test_non_finite_pose_raises(self):
        route = route_from_dict(route_dict(straight_xz()))
        for bad in (float("nan"), float("inf"), -float("inf")):
            for pose in ((bad, 10.0, 0.0), (100.0, bad, 0.0), (100.0, 10.0, bad), Pose(bad, 10.0, 0.0),
                         {"x": 100.0, "z": 10.0, "heading_rad": bad}):
                with self.assertRaises(ValueError, msg=repr(pose)):
                    nav_v1(route, pose, progress_m=0.0)

    def test_lookahead_never_wraps_on_a_closed_route(self):
        square = [(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0), (0.0, 0.0)]
        route = route_from_dict(route_dict(square, closed=True, route_id="test_square"))
        nav = nav_v1(route, route_pose(route, 395.0), progress_m=394.0)
        # 5 m before the end, heading west: p20 clamps at the end point 5 m ahead instead of wrapping 15 m north
        self.assertAlmostEqual(nav["p20_x"], 5.0, delta=1e-9)
        self.assertAlmostEqual(nav["p20_y"], 0.0, delta=1e-9)
        self.assertAlmostEqual(nav["p40_x"], 5.0, delta=1e-9)

    def test_lookahead_clamped_at_route_end(self):
        route = route_from_dict(route_dict(straight_xz(length=100.0)))
        nav = nav_v1(route, route_pose(route, 90.0), progress_m=89.0)
        self.assertAlmostEqual(nav["p20_x"], 10.0, delta=1e-9)
        self.assertAlmostEqual(nav["p40_x"], 10.0, delta=1e-9)

    def test_progress_never_decreases(self):
        route = route_from_dict(route_dict(straight_xz()))
        nav = nav_v1(route, route_pose(route, 400.0), progress_m=399.0)
        back = nav_v1(route, route_pose(route, 390.0), progress_m=nav["progress_m"])
        self.assertEqual(back["progress_m"], nav["progress_m"])
        with self.assertRaises(ValueError):
            nav_v1(route, route_pose(route, 10.0), progress_m=-1.0)
        with self.assertRaises(ValueError):
            nav_v1(route, route_pose(route, 10.0), progress_m=route.length_m + 1.0)

    def test_pose_forms(self):
        route = route_from_dict(route_dict(straight_xz()))
        p = route_pose(route, 200.0)
        a = nav_v1(route, p, progress_m=199.0)
        b = nav_v1(route, (p.x, p.z, p.heading_rad), progress_m=199.0)
        c = nav_v1(route, {"x": p.x, "z": p.z, "heading_rad": p.heading_rad}, progress_m=199.0)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(pose_from_transform([3.0, 7.0, 4.0], [0.0, 0.0, 1.0]), Pose(3.0, 4.0, 0.0))
        self.assertAlmostEqual(pose_from_transform([0, 0, 0], [1.0, 0.0, 0.0]).heading_rad, 0.5 * math.pi)
        self.assertAlmostEqual(abs(pose_from_transform([0, 0, 0], [0.0, 0.0, -1.0]).heading_rad), math.pi)


class CurveSignTest(unittest.TestCase):
    def test_right_curve_is_positive(self):
        right = route_from_dict(route_dict(curve_xz(+1)))
        left = route_from_dict(route_dict(curve_xz(-1), route_id="test_route_left"))
        for s in (175.0, 190.0, 210.0, 240.0):
            nr = nav_v1(right, route_pose(right, s), progress_m=s - 1.0)
            nl = nav_v1(left, route_pose(left, s), progress_m=s - 1.0)
            self.assertGreater(nr["p40_y"], 0.5, s)
            self.assertLess(nl["p40_y"], -0.5, s)
            # mirror symmetry: the left route is the right route mirrored about the start's north axis
            self.assertAlmostEqual(nr["p40_y"], -nl["p40_y"], delta=1e-9)
            self.assertAlmostEqual(nr["p20_y"], -nl["p20_y"], delta=1e-9)
            self.assertAlmostEqual(nr["p40_x"], nl["p40_x"], delta=1e-9)
        # entering the right turn, the route bends right: p20_y > 0 and p40_y > p20_y
        nr = nav_v1(right, route_pose(right, 190.0), progress_m=189.0)
        self.assertGreater(nr["p20_y"], 0.0)
        self.assertGreater(nr["p40_y"], nr["p20_y"])

    def test_right_turn_heading_is_clockwise(self):
        right = route_from_dict(route_dict(curve_xz(+1)))
        end = route_pose(right, right.length_m - 1.0)
        self.assertAlmostEqual(end.heading_rad, 0.5 * math.pi, delta=1e-6)  # north -> east is +90 deg


class LoopCrossingTest(unittest.TestCase):
    def setUp(self):
        self.route = route_from_dict(route_dict(figure_eight_xz(), closed=True, route_id="test_figure_eight"))
        # arc lengths at the two passes through the crossing (the origin)
        hits = [(math.hypot(x, z), i) for i, (x, z) in enumerate(self.route.xz)]
        near = sorted(i for d, i in hits if d < 1e-6)
        self.assertEqual(len(near), 2)
        self.s1, self.s2 = self.route.s[near[0]], self.route.s[near[1]]
        self.assertGreater(self.s2 - self.s1, 2 * drivedit_nav.PROGRESS_WINDOW_M)

    def test_ordered_progress_through_crossing(self):
        route = self.route
        poses, truth = [], []
        s, k = 0.0, 0
        while s < route.length_m - 1.0:
            lateral = 0.4 * math.sin(0.7 * k)  # deterministic wander around the line
            poses.append(route_pose(route, s, lateral=lateral))
            truth.append(s)
            s += 2.0
            k += 1
        navs = nav_sequence(route, poses)
        last = -1.0
        for nav, s_true in zip(navs, truth):
            self.assertGreaterEqual(nav["progress_m"], last)
            last = nav["progress_m"]
            self.assertAlmostEqual(nav["progress_m"], s_true, delta=0.6)
            if s_true < route.length_m - 45.0:  # the lookahead is clamped at the route end
                self.assertGreater(nav["p20_x"], 15.0)
                self.assertLess(abs(nav["p20_y"]), 12.0)

    def test_crossing_point_keeps_the_branch(self):
        route = self.route
        for s_cross in (self.s1, self.s2):
            pose = route_pose(route, s_cross)
            self.assertLess(math.hypot(pose.x, pose.z), 1e-6)
            nav = nav_v1(route, pose, progress_m=s_cross - 2.0)
            self.assertAlmostEqual(nav["progress_m"], s_cross, delta=1e-6)
            self.assertAlmostEqual(nav["p20_x"], 20.0, delta=0.5)
            self.assertAlmostEqual(nav["p20_y"], 0.0, delta=0.5)
        # a global projection cannot tell the branches apart (both are 0 m away); ordered progress can
        s_glob, d_glob = route.project(0.0, 0.0)
        self.assertLess(d_glob, 1e-6)
        self.assertAlmostEqual(s_glob, self.s1, delta=1e-6)
        wrong = nav_v1(route, route_pose(route, self.s2), progress_m=self.s1)  # the other branch's progress
        right = nav_v1(route, route_pose(route, self.s2), progress_m=self.s2 - 2.0)
        self.assertGreater(abs(wrong["p20_y"]), 10.0)  # 20 m along the wrong branch is far off the forward axis
        self.assertLess(abs(right["p20_y"]), 0.5)


class CommandsTest(unittest.TestCase):
    def test_d_stop_is_200_while_the_route_end_is_far(self):
        route = route_from_dict(route_dict(straight_xz()))
        for s in (0.0, 300.0, 799.0):
            nav = nav_v1(route, route_pose(route, s), progress_m=max(s - 1.0, 0.0))
            self.assertEqual(nav["d_stop"], D_MAX_M)
            self.assertEqual(nav["d_stop"], 200.0)

    def test_d_stop_counts_down_to_the_route_end(self):
        route = route_from_dict(route_dict(straight_xz()))
        cases = [(300.0, 200.0), (800.0, 200.0), (850.0, 150.0), (999.5, 0.5), (1000.0, 0.0)]
        for s, want in cases:
            nav = nav_v1(route, route_pose(route, s), progress_m=max(s - 1.0, 0.0))
            self.assertAlmostEqual(nav["d_stop"], want, delta=1e-6, msg=s)
            self.assertLessEqual(nav["d_stop"], 200.0)

    def test_only_terminal_stops(self):
        # a stop the ego must leave again has no release in nav_v1: commanded stops and dwell are nav_v2
        self.assertEqual(NAVMOD.STOP_KINDS, ("route_end",))
        mid = route_dict(straight_xz(), stops=[(450.0, "commanded"), (1000.0, "route_end")])
        self.assertTrue(validate_route(mid))
        dwell = route_dict(straight_xz())
        dwell["stops"][0]["dwell_s"] = 5.0
        self.assertTrue(validate_route(dwell))
        with self.assertRaises(RouteError):
            route_from_dict(mid)

    def test_maneuvers(self):
        route = route_from_dict(route_dict(straight_xz(), maneuvers=[(380.0, "right"), (700.0, "straight"),
                                                                     (720.0, "left")]))
        cases = [(100.0, "follow", 200.0), (180.0, "right", 200.0), (300.0, "right", 80.0), (379.5, "right", 0.5),
                 (381.0, "follow", 200.0), (600.0, "straight", 100.0), (705.0, "left", 15.0),
                 (721.0, "follow", 200.0)]
        for s, kind, d in cases:
            nav = nav_v1(route, route_pose(route, s), progress_m=max(s - 1.0, 0.0))
            onehot = [nav["man_left"], nav["man_straight"], nav["man_right"], nav["man_follow"]]
            self.assertEqual(sum(onehot), 1, s)
            self.assertEqual(nav["man_" + kind], 1, s)
            self.assertAlmostEqual(nav["d_man"], d, delta=1e-6, msg=s)

    def test_vector_order_matches_features_v1(self):
        with open(FEATURES_JSON, encoding="ascii") as f:
            reg = json.load(f)
        nav_t = [t for t in reg["tensors"] if t["name"] == "nav"][0]
        self.assertEqual(tuple(c["name"] for c in nav_t["channels"]), NAV_CHANNELS)
        self.assertEqual(nav_t["shape"], ["T", len(NAV_CHANNELS)])
        route = route_from_dict(route_dict(straight_xz(), maneuvers=[(250.0, "left")]))
        nav = nav_v1(route, route_pose(route, 200.0), progress_m=199.0)
        vec = nav_vector(nav)
        self.assertEqual(len(vec), 10)
        self.assertEqual(vec, tuple(float(nav[k]) for k in NAV_CHANNELS))
        self.assertEqual(vec[4], 1.0)  # man_left


class DeterminismTest(unittest.TestCase):
    def test_bit_identical(self):
        obj = route_dict(figure_eight_xz(), closed=True, maneuvers=[(150.0, "left"), (600.0, "right")],
                         route_id="test_det")
        reordered = dict(reversed(list(copy.deepcopy(obj).items())))
        round_trip = json.loads(json.dumps(obj))
        routes = [route_from_dict(obj), route_from_dict(reordered), route_from_dict(round_trip)]
        self.assertEqual(len({r.sha256 for r in routes}), 1)
        base = routes[0]
        poses = [route_pose(base, s, lateral=0.3 * math.cos(s)) for s in [i * 3.7 for i in range(150)]]
        results = [[tuple(sorted(n.items())) for n in nav_sequence(r, poses)] for r in routes]
        results.append([tuple(sorted(n.items())) for n in nav_sequence(base, poses)])
        for other in results[1:]:
            self.assertEqual(other, results[0])
        bits = [repr(v) for n in results[0] for _, v in n]
        self.assertEqual(bits, [repr(v) for n in results[-1] for _, v in n])

    def test_pure_python(self):
        with open(os.path.join(REPO, "drivedit_nav", "nav_v1.py"), encoding="ascii") as f:
            src = f.read()
        for mod in ("numpy", "torch", "random", "time"):
            self.assertNotIn("import %s" % mod, src)


class RouteSchemaTest(unittest.TestCase):
    def good(self):
        return route_dict(straight_xz(length=200.0), maneuvers=[(150.0, "right")], stops=[(200.0, "route_end")])

    def test_valid(self):
        self.assertEqual(validate_route(self.good()), [])
        obj = self.good()
        obj["length_m"] = 200.0
        obj["source"] = {"generator": "test"}
        self.assertEqual(validate_route(obj), [])

    MUTATIONS = {
        "dwell_s on a stop": lambda o: o["stops"][0].update(dwell_s=3.0),
        "commanded stop": lambda o: o["stops"].insert(0, {"s_m": 100.0, "kind": "commanded"}),
        "junction_id on a stop": lambda o: o["stops"][0].update(junction_id="j1"),
        "dwell_s on a maneuver": lambda o: o["maneuvers"][0].update(dwell_s=1.0),
        "junction_id not a string": lambda o: o["maneuvers"][0].update(junction_id=7),
        "no stop": lambda o: o.update(stops=[]),
        "two route_end stops": lambda o: o["stops"].append({"s_m": 200.0, "kind": "route_end"}),
        "list as map_id": lambda o: o.update(map_id=["arland"]),
        "unknown key": lambda o: o.update(goal_x=3.0),
        "bad maneuver": lambda o: o.update(maneuvers=[{"s_m": 50.0, "type": "u_turn"}]),
        "terrain name as map_id": lambda o: o.update(map_id="cain"),
    }

    def test_validators_agree(self):
        """Every mutation the JSON schema can express is refused by both validate_route and the schema."""
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed; validate_route alone is checked in test_invalid")
        with open(drivedit_nav.ROUTE_SCHEMA_FILE, encoding="ascii") as f:
            v = jsonschema.Draft202012Validator(json.load(f))
        for label, mutate in self.MUTATIONS.items():
            obj = self.good()
            mutate(obj)
            self.assertTrue(validate_route(obj), label)
            self.assertTrue(list(v.iter_errors(obj)), label)

    def test_route_end_must_be_at_the_end(self):
        obj = self.good()
        obj["stops"] = [{"s_m": 150.0, "kind": "route_end"}]
        self.assertTrue(validate_route(obj))
        obj["stops"] = [{"s_m": 200.03, "kind": "route_end"}]
        self.assertEqual(validate_route(obj), [])

    def test_invalid(self):
        mutations = dict(self.MUTATIONS)
        mutations.update({
            "split of another map": lambda o: o.update(split="k_val"),
            "unsorted maneuvers": lambda o: o.update(maneuvers=[{"s_m": 150.0, "type": "right"},
                                                                {"s_m": 100.0, "type": "left"}]),
            "repeated point": lambda o: o["polyline"].insert(3, list(o["polyline"][2])),
            "closed with a 200 m gap": lambda o: o.update(closed=True),
            "wrong length": lambda o: o.update(length_m=210.0),
            "stop beyond the end": lambda o: o.update(stops=[{"s_m": 250.0, "kind": "route_end"}]),
            "one point": lambda o: o.update(polyline=[[0.0, 0.0, 0.0]]),
            "nav version": lambda o: o.update(nav_version="nav_v0"),
            "route_end before the end": lambda o: o.update(stops=[{"s_m": 150.0, "kind": "route_end"}]),
        })
        cases = {}
        for label, mutate in mutations.items():
            obj = self.good()
            mutate(obj)
            cases[label] = obj
        for label, obj in cases.items():
            self.assertTrue(validate_route(obj), label)
            with self.assertRaises(RouteError, msg=label):
                route_from_dict(obj)

    def test_json_schema_file(self):
        with open(drivedit_nav.ROUTE_SCHEMA_FILE, encoding="ascii") as f:
            schema = json.load(f)
        self.assertEqual(schema["$id"], drivedit_nav.ROUTE_SCHEMA_ID)
        self.assertEqual(schema["properties"]["map_id"]["enum"], list(drivedit_nav.MAP_SPLITS))
        splits = [s for m in drivedit_nav.MAP_SPLITS.values() for s in m]
        self.assertEqual(schema["properties"]["split"]["enum"], splits)
        for rule in schema["allOf"]:
            m = rule["if"]["properties"]["map_id"]["const"]
            self.assertEqual(rule["then"]["properties"]["split"]["enum"], list(drivedit_nav.MAP_SPLITS[m]))
        self.assertEqual(sorted(schema["properties"]), sorted(NAVMOD.TOP_KEYS))
        self.assertEqual(schema["required"], list(NAVMOD.REQUIRED_KEYS))
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed (the structural checks above ran)")
        jsonschema.Draft202012Validator.check_schema(schema)
        v = jsonschema.Draft202012Validator(schema)
        self.assertEqual(list(v.iter_errors(self.good())), [])
        for mutate in (lambda o: o.update(map_id="eden"), lambda o: o.update(split="everon_test"),
                       lambda o: o.update(extra=1), lambda o: o["stops"].append({"s_m": 1.0, "kind": "maybe"}),
                       lambda o: o["stops"][0].update(kind="commanded")):
            obj = self.good()
            mutate(obj)
            self.assertTrue(list(v.iter_errors(obj)))


class SealedMapTest(unittest.TestCase):
    def everon(self):
        return route_dict(straight_xz(length=100.0), map_id="everon", split="everon_test", route_id="everon_t")

    def test_everon_needs_the_sealed_path(self):
        with self.assertRaises(SealedMapError):
            route_from_dict(self.everon())
        route = route_from_dict(self.everon(), sealed_eval=True)
        self.assertEqual(route.map_id, "everon")

    def test_sealed_directories_refused_before_reading(self):
        for path in ("/mnt/e/DriveDiT/heldout/everon/SEALED/harness/routes/r.json", "x/SEALED/r.json",
                     "/nonexistent/heldout/r.json"):
            with self.assertRaises(SealedMapError):
                load_route(path)

    def test_load_route_file(self):
        base = tmp_base()
        d = tempfile.mkdtemp(prefix="dd11_nav_", dir=base)
        try:
            path = os.path.join(d, "r.json")
            with open(path, "w", encoding="ascii") as f:
                json.dump(route_dict(straight_xz(length=100.0)), f)
            route = load_route(path)
            self.assertAlmostEqual(route.length_m, 100.0, delta=1e-9)
            with open(path, "w", encoding="ascii") as f:
                json.dump(self.everon(), f)
            with self.assertRaises(SealedMapError):
                load_route(path)
            self.assertEqual(load_route(path, sealed_eval=True).split, "everon_test")
        finally:
            shutil.rmtree(d)


class LegacyViewTest(unittest.TestCase):
    def test_to_legacy_6d(self):
        route = route_from_dict(route_dict(straight_xz()))
        nav = nav_v1(route, route_pose(route, 100.0, lateral=0.5), progress_m=99.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            six = to_legacy_6d((0.01, -0.5, 1), nav, 12.0, 0.12, lambda k, v: 25.0 * k)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertEqual(six, (0.25, -0.5, nav["p20_x"], nav["p20_y"], 12.0, 0.12))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                to_legacy_6d((0.01, -0.5, 1), nav, 12.0, 0.12, None)


if __name__ == "__main__":
    unittest.main()
