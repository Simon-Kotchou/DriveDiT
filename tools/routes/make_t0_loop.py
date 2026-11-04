#!/usr/bin/env python3
"""Build the Arland T0 loop (DD-7-min) on the ENGINE road graph as a drivedit_nav route (drivedit.route/1).

Run it with the isolated interpreter (stdlib only, plus the repo's drivedit_nav package):

    python3 -I -B tools/routes/make_t0_loop.py --splits <splits_v2.json> [--dump DIR]
        [--layer addons/DataCapture/worlds/DataCapture/DC_Arland_Layers/default.layer] [--trails <decoder TSV>]
        [--out addons/DataCapture/Configs/routes/arland_t0_loop.json | --check FILE]

INPUTS (read only)
  --dump    the engine's road network as probe E-08a dumped it (DC_RoadDump): arland_roads.csv (road_idx, road_key,
            width_m, n_pts, ...), arland_road_points.csv (road_idx, pt_idx, x, y, z along each road) and
            arland_roads_footer.txt, which must say complete=1 for map arland with the row counts of both files.
            Default: the versioned copy DEFAULT_DUMP (T0-ENG probe run 1, 1.8.0.13; its source.txt names the play and
            the sha256 of each of the three files: a copy whose source.txt lacks one, or whose file differs, is
            refused).
            An E-08a play directory works too (launch.id from its run_header.txt). The loop's roads are named by the
            road_idx of that dump and checked against their road_key, so another dump (another launch or build) is
            refused until LOOP is derived again on it.
  --splits  program/maps/splits_v2_PROPOSED.json: the Arland VAL-L box and its buffer (TRAIN = everything outside).
  --layer   the DC_Arland layer: the parked vehicles the loop keeps clear of.
  --trails  optional, a CHECK only: the decoder's road polylines (Arland_road_polylines.tsv). Its trails (class
            trail, prefab Type Path) are not in the engine's road network, so no engine junction marks them: the build
            is refused when a trail crosses the loop (at a point where it meets the line, end points of the segments
            included, it goes on for TRAIL_END_M or more along the trail on both sides, as 414 across engine road
            121), and the trails that end on the loop's road or within TRAIL_TOUCH_M of it are listed.
            The route never depends on it (--check gives the same bytes with and without it).
--out writes the route; --check regenerates it and compares it byte for byte with FILE (exit 1 when they differ);
without either the route is only built, checked and summarised.

THE LOOP (LOOP below; about 2.5 km, clockwise seen from above, north-east of the main airfield): engine road 163
(8 m asphalt) east, then south-east, from its junction with 161 to the junction with 17; through that junction (165,
the 4.5 m pieces 164 and 167 and the connector 166) onto 137 (4.5 m dirt) south-west; over 136 (3 m) onto 133
(4.5 m dirt) west-north-west to its junction with 93; 93 (8 m asphalt) north-west to 161; 161 north and through the
connector 179 back onto 163. It replaces the decoder-derived loop of generator version 1 (decoder roads 452, 461,
93, 451), which crossed the decoder-only trail 414 and named junctions the engine does not have (t0/run1/ROADS.md
section 5). Around the airfield every loop of engine roads crosses or touches a decoder trail (414 on engine roads
121 and 129, 413 on 63); this ring crosses none.

How the driving line is made (the route polyline is the driving line; drivedit_nav never offsets it):
  1. each engine road's points (about every 10 m on its spline) are interpolated by a centripetal Catmull-Rom
     spline;
  2. consecutive loop roads meet where an end of one comes closest to the other (at most JOIN_TOL_M): the junction
     points; each road is used between its junctions with the previous and the next loop road;
  3. each piece is offset to the right lane centre, right of the driving direction by LANE_WIDTH_FRACTION of the
     ENGINE road width; a junction connector (width <= CONNECTOR_MAX_W, the engine's 1 m pieces) takes the width of
     the next loop road;
  4. junction points closer than CLUSTER_M along the line form one junction. A junction that turns by CORNER_MIN_DEG
     or more gets a corner: the lane lines before and after it meet at an apex, and the line is cut where a circle of
     radius CORNER_RIGHT_M (right turns, which cut the inner corner of the right lane) or CORNER_LEFT_M touches them;
     the cuts lie at least CORNER_MIN_CUT_M before and after every junction point that turns or changes the lane
     offset (roads of different widths), which alone also gets a join. The cut is joined by a cubic Bezier curve
     tangent to the lane line at both cuts, with near-circular handles;
  5. the closed line starts START_ALONG_M along the first road's lane line, is resampled every 1 m and smoothed, each
     point staying within SMOOTH_MAX_SHIFT_M of its lane position, then resampled every STEP_M with the first and
     last point identical.
Heights are the engine heights of the nearest loop-road centreline point (nav_v1 ignores y; DC_EgoSession puts the
spawn on the surface).

The lane corridor (source.lane_corridor_s_m and source.lane_corridor_half_width_m, read by DC_Route for DC_ExpertHL's
offroad termination, r2 1.3: dist_lane_corridor > w/2 + 0.5 m for more than 3 s): the corridor around the driving
line is the right lane of the nearest loop road that is not a connector, half its engine width wide, so its half
width w/2 is LANE_CORRIDOR_FRACTION of that width (2 m on the 8 m roads, 1.125 m on the 4.5 m dirt roads, 0.75 m on
the 3 m road 136). Within CORRIDOR_SPAN_M along the route of a point the narrowest one holds, so a junction between
roads of different widths has the narrower corridor on both sides. Piece k covers the arc lengths [s_k, s_k+1), the
first one starts at 0 and the last one runs to the route end.

Checks before anything is written (exit 2 and nothing written when one fails):
  * every point lies in TRAIN: outside the VAL-L box of the splits file dilated by its buffer;
  * every point lies on the engine's road surface: within half the width minus ON_ROAD_MARGIN_M of the nearest engine
    road centreline, or half the width plus JUNCTION_ALLOW_M within JUNCTION_ZONE_M of a junction of the engine graph
    (a connector counts with the width of the widest road it joins, inherit_widths);
  * right lane: farther than JUNCTION_ZONE_M from every junction, the signed offset of each point right of its loop
    road's centreline lies within LANE_TOL_M of LANE_WIDTH_FRACTION of that road's engine width;
  * every parked vehicle of the DC_Arland layer is at least CLEARANCE_M away;
  * the turning radius (heading change over CURV_WINDOW_M) is at least MIN_RADIUS_M everywhere;
  * the line does not cross itself; its length lies in LENGTH_RANGE_M;
  * C17 content (DD-5-min): left and right turning each at least MIN_TURN_RAD, and the left share of the absolute
    turning differs from its mirror image by more than MIRROR_DEV_MIN (so a mirrored route fails C17);
  * drivedit_nav.validate_route accepts it; with --trails, no decoder trail crosses it.

Maneuvers come from the ENGINE graph only (decoder-only roads such as Type 255 pieces and trails never appear): a
junction is where another engine road wider than a connector has an end within ARM_TOL_M of a loop road's used
piece, or where a loop road continues past a junction point with the next loop road; junction events closer than
MANEUVER_MERGE_M along the route are one junction. Its geometric type is the route's own heading change from
TURN_SPAN_M before to TURN_SPAN_M after it: left or right from MANEUVER_STRAIGHT_DEG on, else straight. Then the
same-side rule (dc-2.0 freeze decision D-E, review finding R-3, adopted 2026-11-03): every branch leaving the junction
gets a turn angle, the heading of its chord over its first TURN_SPAN_M from the junction point (shorter when the road
is) relative to the arrival, the route's chord over the TURN_SPAN_M before its first junction event. The route's own
branch is measured on the route from its last junction event; an arm along its engine road centreline from its end
at the loop; a loop road going on past a junction point from that point, the way the route does not take. When
another branch turns to the same side as the route's branch, the route's branch is the one closest to straight on
(strictly, among all the branches of the junction) and it turns by SAME_SIDE_STRAIGHT_DEG or less, the maneuver is
straight; otherwise the geometric type stays (left / right: d_man disambiguates). The rule only ever changes left or
right into straight. At e133_132 the route follows 133 (+22 deg by these chords) while side road 132 turns +54 deg:
straight (it was right). junction_id is "e" and the dump's road_idx joined by "_": the loop road arriving, the loop
road leaving (when it differs) and the other engine roads meeting there, ascending. Stops: the one route_end at the
route end (nav_v1).

Output: the route's canonical JSON (drivedit_nav.canonical_json: sorted keys, no whitespace, ASCII) and one LF, so
the sha256 of the file without its final LF is drivedit_nav's Route.sha256; DC_ExpertHL computes episodes.route_sha256
that way.

Sealed map: any input path naming eden or everon is refused with exit 2 before anything is read (Everon is the
zero-shot test map). Outputs only under /mnt/e/ or /dev/shm/ (never /mnt/c or /tmp).
Exit codes: 0 built (and written or equal), 1 --check found a difference, 2 refused (input, check or path problem).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import drivedit_nav  # noqa: E402  (the route schema and its validator, DD-11-min)

GENERATOR = "tools/routes/make_t0_loop.py"
GENERATOR_VERSION = 4              # 3: the lane corridor per road (source.lane_corridor_*); 4: the same-side rule (D-E)
ROUTE_ID = "arland_t0_loop"
MAP_ID = "arland"
SPLIT = "train"
DEFAULT_DUMP = os.path.join(HERE, "data", "e08a_20261023T021038Z-wb-776e")
DEFAULT_LAYER = os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Arland_Layers",
                             "default.layer")
DUMP_FILES = ("arland_roads.csv", "arland_road_points.csv", "arland_roads_footer.txt")

# The loop in driving order: (road_idx, road_key) of the E-08a dump DEFAULT_DUMP; each road meets the next one
LOOP = (
    (163, "86/1682.48:3384.40/2521.30:3088.66"),
    (165, "2/2521.19:3088.73/2547.19:3088.39"),
    (164, "4/2549.41:3088.22/2554.53:3079.16"),
    (167, "8/2553.32:3061.89/2565.08:3086.58"),
    (166, "2/2553.50:3063.64/2551.97:3052.74"),
    (137, "34/2411.44:2776.81/2552.54:3054.43"),
    (136, "2/2390.07:2772.22/2411.57:2776.79"),
    (133, "70/1847.38:3052.87/2391.09:2772.88"),
    (93, "80/1686.88:3174.76/1549.74:2617.21"),
    (161, "39/1639.04:3111.89/1674.97:3381.00"),
    (179, "2/1674.15:3380.38/1684.04:3385.20"),
)
START_ALONG_M = 100.0           # the start (and end) of the closed route, along the first road's lane line
LANE_WIDTH_FRACTION = 0.25      # right lane centre = a quarter of the engine road width right of the centreline
CONNECTOR_MAX_W = 1.5           # the engine's junction connectors are 1 m wide pieces of about 11 m
LANE_CORRIDOR_FRACTION = 0.25   # lane corridor half width w/2: the right lane is half the engine road width
CORRIDOR_SPAN_M = 20.0          # the narrowest corridor within this far along the route holds (junctions, corners)
CORRIDOR_DECIMALS = 3

SPLINE_STEP_M = 0.5             # Catmull-Rom sampling of the road centrelines
JOIN_TOL_M = 4.0                # a loop road's end must lie this close to the next loop road
TANGENT_SPAN_M = 5.0            # the lane direction at a cut: over this much of the lane line
CLUSTER_M = 30.0                # junction points closer than this along the lane line are one junction
CORNER_RIGHT_M = 6.5            # right turns cut the inner corner of the right lane: a tight corner stays on the road
CORNER_LEFT_M = 15.0
CORNER_MIN_CUT_M = 6.0
CORNER_MIN_DEG = 10.0           # a junction turning less is joined over CORNER_MIN_CUT_M each side, without an apex
SMOOTH_STEP_M = 1.0
SMOOTH_ITERATIONS = 200
SMOOTH_GAIN = 0.5
SMOOTH_MAX_SHIFT_M = 0.5        # the smoothed line stays within this of the lane line
STEP_M = 2.0                    # spacing of the route polyline
COORD_DECIMALS = 3
HEIGHT_DECIMALS = 2

ON_ROAD_MARGIN_M = 0.5
JUNCTION_ZONE_M = 20.0
JUNCTION_ALLOW_M = 0.5
LANE_TOL_M = 0.6
CLEARANCE_M = 10.0
CURV_WINDOW_M = 5.0             # as C17's route curvature: heading change over +-5 m of arc length
MIN_RADIUS_M = 6.5              # a guard against corner construction errors: the right lane of engine road 161's
                                # right-hand bend (centreline radius 8.9 m) itself turns at about 7 m
LENGTH_RANGE_M = (1000.0, 3000.0)
MIN_TURN_RAD = 0.2              # DD-5-min C17: below this much turning the L/R comparison has no signal
MIRROR_DEV_MIN = 0.35           # C17 fails a route whose left share differs by more than 0.25; keep a margin
ARM_TOL_M = 6.0                 # an engine road end this close to a loop road's piece is a junction arm
MANEUVER_MERGE_M = 15.0
TURN_SPAN_M = 20.0              # the route's turn: over this far each side; a branch's heading: over its first this far
MANEUVER_STRAIGHT_DEG = 30.0
# Same-side rule, dc-2.0 freeze decision D-E (R-3 of the DD-7-min review), adopted by the user on 2026-11-03: when
# another branch of a junction turns to the same side as the route's branch, the route's branch is typed straight if it
# is the branch closest to straight on and its turn angle is within +-SAME_SIDE_STRAIGHT_DEG (module docstring)
SAME_SIDE_STRAIGHT_DEG = 35.0
SAME_SIDE_RULE = ("dc-2.0 freeze D-E (2026-11-03): branch turn angles are chords over turn_span_m from the junction "
                  "point against the route's arrival chord; when another branch turns to the route's side, the route's "
                  "branch is straight if it is the closest to straight on and within same_side_straight_within_deg")
TRAIL_TOUCH_M = 6.0
TRAIL_END_M = 8.0               # a trail ending within this of the route ends on its road: a touch, not a crossing
CUT_EPS_M = 1e-6                # segments this close meet; meeting points this close along a trail are one
GRID_M = 32.0

WRITE_ROOTS = ("/mnt/e/", "/dev/shm/")
FORBIDDEN_ROOTS = ("/mnt/c/", "/tmp/")
SEALED_RE = re.compile(r"eden|everon", re.IGNORECASE)
VEHICLE_RE = re.compile(r'^Vehicle\s+(\w+)\s*:')
COORDS_RE = re.compile(r'^\s*coords\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$')


class Refusal(Exception):
    """An input, path or check problem: exit 2, nothing written."""


# ---------------------------------------------------------------------------------------------------------------
# inputs


def sealed_reason(path):
    for candidate in (path, os.path.realpath(path)):
        if SEALED_RE.search(candidate):
            return "refusing %s: it names the sealed map (eden / everon); Everon is the zero-shot test map" % path
    return None


def check_out_path(path):
    real = os.path.realpath(os.path.abspath(path))
    probe = real if real.endswith("/") else real + "/"
    if any(probe.startswith(root) for root in FORBIDDEN_ROOTS) or not any(probe.startswith(r) for r in WRITE_ROOTS):
        raise Refusal("refusing output %s: only /mnt/e/... or /dev/shm/... (never /mnt/c or /tmp)" % path)
    return real


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_kv(path):
    out = {}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            key, sep, value = line.rstrip("\r\n").partition("=")
            if sep:
                out[key.strip()] = value
    return out


class Road:
    """One engine road of the dump: its centreline (Catmull-Rom through the engine's points), width and key."""

    def __init__(self, idx, key, width, points):
        self.idx = idx
        self.key = key
        self.width = width
        self.points = points                        # [(x, y, z)] as the engine reported them
        self.connector = width <= CONNECTOR_MAX_W
        self.surface_width = width                  # a connector: the width of the roads it joins (inherit_widths)
        self.line = catmull_rom(points, SPLINE_STEP_M) if len(points) >= 3 else list(points)
        self.s = arc_lengths(self.line)
        self.length = self.s[-1]


def read_dump(path):
    """({road_idx: Road}, provenance) of an E-08a dump directory (the repo copy or a play directory)."""
    reason = sealed_reason(path)
    if reason:
        raise Refusal(reason)
    for name in DUMP_FILES:
        if not os.path.isfile(os.path.join(path, name)):
            raise Refusal("the road dump %s has no %s" % (path, name))
    footer = read_kv(os.path.join(path, "arland_roads_footer.txt"))
    if footer.get("map_id") != MAP_ID or footer.get("complete") != "1":
        raise Refusal("the road dump %s is not a complete dump of %s (footer map_id %s, complete %s)"
                      % (path, MAP_ID, footer.get("map_id"), footer.get("complete")))
    shapes = {}
    with open(os.path.join(path, "arland_road_points.csv"), "r", encoding="ascii", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in sorted(rows, key=lambda r: (int(r["road_idx"]), int(r["pt_idx"]))):
        shapes.setdefault(int(row["road_idx"]), []).append((float(row["x"]), float(row["y"]), float(row["z"])))
    roads = {}
    with open(os.path.join(path, "arland_roads.csv"), "r", encoding="ascii", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["road_idx"])
            points = shapes.get(idx, [])
            if len(points) != int(row["n_pts"]) or len(points) < 2:
                raise Refusal("road %d of %s has %d points, its row says %s" % (idx, path, len(points), row["n_pts"]))
            roads[idx] = Road(idx, row["road_key"], float(row["width_m"]), points)
    inherit_widths(roads)
    if str(len(roads)) != footer.get("roads_written") or str(len(rows)) != footer.get("points_total"):
        raise Refusal("the road dump %s has %d roads and %d points, its footer says %s and %s"
                      % (path, len(roads), len(rows), footer.get("roads_written"), footer.get("points_total")))
    meta = {}
    hashed = os.path.isfile(os.path.join(path, "source.txt"))      # a copy: source.txt must name every file's sha256
    if hashed:
        meta = read_kv(os.path.join(path, "source.txt"))
        missing = [name for name in DUMP_FILES if not meta.get("sha256." + name)]
        if missing:
            raise Refusal("the source.txt of %s names no sha256 of %s" % (path, ", ".join(missing)))
    elif os.path.isfile(os.path.join(path, "run_header.txt")):     # a play directory: the files as the probe wrote them
        header = read_kv(os.path.join(path, "run_header.txt"))
        meta = {"probe": header.get("probe", ""), "launch_id": header.get("launch.id", ""),
                "play": os.path.basename(os.path.normpath(path)),
                "engine_build_version": header.get("engine.build_version", "")}
    if meta.get("probe") != "e08a" or not meta.get("launch_id"):
        raise Refusal("the road dump %s names no E-08a launch (source.txt or run_header.txt)" % path)
    files = {name: sha256_file(os.path.join(path, name)) for name in DUMP_FILES}
    for name, digest in sorted(files.items()):
        if hashed and meta["sha256." + name] != digest:
            raise Refusal("%s in %s is not the file its source.txt names (sha256 %s, source.txt %s)"
                          % (name, path, digest, meta["sha256." + name]))
    provenance = {"probe": "e08a", "launch_id": meta["launch_id"], "play": meta.get("play", ""),
                  "engine_build_version": meta.get("engine_build_version", ""), "files": files,
                  "road_ids": "road_idx of this dump"}
    return roads, provenance


def inherit_widths(roads):
    """A junction connector (the engine's 1 m pieces) carries the surface of the roads it joins (t0/run1/ROADS.md 6.5:
    connectivity edges that inherit their neighbours' class): its surface_width is the largest width of the roads
    wider than a connector with a point within JOIN_TOL_M of either of its ends (0 without one)."""
    wide = [road for road in roads.values() if not road.connector]
    for road in roads.values():
        if not road.connector:
            continue
        widths = [0.0]
        for end in (road.line[0], road.line[-1]):
            for other in wide:
                xs = [p[0] for p in other.line]
                zs = [p[2] for p in other.line]
                if min(xs) - JOIN_TOL_M <= end[0] <= max(xs) + JOIN_TOL_M \
                        and min(zs) - JOIN_TOL_M <= end[2] <= max(zs) + JOIN_TOL_M \
                        and project(other.line, other.s, end[0], end[2])[1] <= JOIN_TOL_M:
                    widths.append(other.width)
        road.surface_width = max(widths)


def read_train_rule(path):
    """The Arland VAL-L box and buffer of the splits file: (box (x0, x1, z0, z1), buffer_m, status)."""
    with open(path, "rb") as f:
        splits = json.loads(f.read().decode("utf-8"))
    try:
        regions = splits["maps"]["arland"]["regions"]
        box = regions["val_L_south_rural"]["box_xz"]
        buffer_m = float(regions["buffer_m"])
    except (KeyError, TypeError) as e:
        raise Refusal("%s has no Arland VAL-L box and buffer (%s)" % (path, e))
    if len(box) != 4:
        raise Refusal("%s: the VAL-L box_xz must be [x_min, x_max, z_min, z_max]" % path)
    return tuple(float(v) for v in box), buffer_m, str(splits.get("status", ""))


def read_parked_vehicles(path):
    """[(name, x, z)] of the Vehicle entities of a layer file (their first coords line)."""
    out = []
    name = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = VEHICLE_RE.match(line)
            if match:
                name = match.group(1)
                continue
            match = COORDS_RE.match(line)
            if match and name:
                out.append((name, float(match.group(1)), float(match.group(3))))
                name = None
    return out


def read_trails(path):
    """[(road_id, [(x, z)])] of the decoder's trails (class trail) in its polylines TSV."""
    out = []
    with open(path, "r", encoding="ascii", errors="replace", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if not all(name in header for name in ("road_id", "class", "points_xyz")):
            raise Refusal("%s is not the decoder's polylines TSV" % path)
        ix = {name: i for i, name in enumerate(header)}
        for line in f:
            cells = line.rstrip("\r\n").split("\t")
            if len(cells) < len(header) or cells[ix["class"]] != "trail":
                continue
            points = [tuple(float(v) for v in p.split(",")) for p in cells[ix["points_xyz"]].split(";") if p]
            out.append((int(cells[ix["road_id"]]), [(p[0], p[2]) for p in points]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# geometry (x, z horizontal; points carry y as their second value)


def catmull_rom(points, step):
    """Centripetal Catmull-Rom spline through the points (x, y, z), sampled about every step metres."""
    padded = [points[0]] + list(points) + [points[-1]]
    out = [points[0]]
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        t0 = 0.0
        t1 = t0 + max(math.hypot(p1[0] - p0[0], p1[2] - p0[2]) ** 0.5, 1e-6)
        t2 = t1 + max(math.hypot(p2[0] - p1[0], p2[2] - p1[2]) ** 0.5, 1e-6)
        t3 = t2 + max(math.hypot(p3[0] - p2[0], p3[2] - p2[2]) ** 0.5, 1e-6)
        pieces = max(1, int(math.ceil(math.hypot(p2[0] - p1[0], p2[2] - p1[2]) / step)))
        for k in range(1, pieces + 1):
            t = t1 + (t2 - t1) * k / pieces
            a1 = blend(p0, p1, t0, t1, t)
            a2 = blend(p1, p2, t1, t2, t)
            a3 = blend(p2, p3, t2, t3, t)
            b1 = blend(a1, a2, t0, t2, t)
            b2 = blend(a2, a3, t1, t3, t)
            out.append(blend(b1, b2, t1, t2, t))
    return out


def blend(a, b, ta, tb, t):
    if tb - ta < 1e-9:
        return a
    return tuple(((tb - t) * a[j] + (t - ta) * b[j]) / (tb - ta) for j in range(3))


def arc_lengths(line):
    s = [0.0]
    for a, b in zip(line, line[1:]):
        s.append(s[-1] + math.hypot(b[0] - a[0], b[2] - a[2]))
    return s


def project(line, s_values, x, z):
    """(s, distance, index) of the closest point of a polyline to (x, z)."""
    best = None
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        dx, dz = b[0] - a[0], b[2] - a[2]
        seg2 = dx * dx + dz * dz
        t = 0.0 if seg2 <= 0 else max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[2]) * dz) / seg2))
        d = math.hypot(x - (a[0] + t * dx), z - (a[2] + t * dz))
        if best is None or d < best[1]:
            best = (s_values[i] + t * math.sqrt(seg2), d, i)
    return best


def point_at(line, s_values, s):
    """(x, y, z) at arc length s, clamped to the line."""
    if s <= 0:
        return line[0]
    if s >= s_values[-1]:
        return line[-1]
    lo, hi = 0, len(s_values) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if s_values[mid] <= s:
            lo = mid
        else:
            hi = mid
    seg = s_values[hi] - s_values[lo]
    t = 0.0 if seg <= 0 else (s - s_values[lo]) / seg
    a, b = line[lo], line[hi]
    return tuple(a[j] + t * (b[j] - a[j]) for j in range(3))


def sub_line(road, s_from, s_to):
    """The road centreline from arc length s_from to s_to (reversed when s_to < s_from), every SPLINE_STEP_M."""
    pieces = max(1, int(math.ceil(abs(s_to - s_from) / SPLINE_STEP_M)))
    return [point_at(road.line, road.s, s_from + (s_to - s_from) * k / pieces) for k in range(pieces + 1)]


def unit(dx, dz):
    norm = math.hypot(dx, dz)
    if norm <= 0:
        raise Refusal("zero-length direction")
    return dx / norm, dz / norm


def right_of(direction):
    """The right normal of a horizontal direction (x east, z north, seen from above): (dz, -dx)."""
    return direction[1], -direction[0]


def heading(direction):
    """Heading of a direction: atan2(dx, dz), 0 = north, clockwise positive."""
    return math.atan2(direction[0], direction[1])


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def offset_line(line, offset):
    """The polyline shifted right of its direction by offset metres (per-vertex normal of the neighbours)."""
    out = []
    for i, p in enumerate(line):
        a = line[max(i - 1, 0)]
        b = line[min(i + 1, len(line) - 1)]
        nx, nz = right_of(unit(b[0] - a[0], b[2] - a[2]))
        out.append((p[0] + offset * nx, p[1], p[2] + offset * nz))
    return out


def direction_at(line, s_values, s, ahead):
    """Unit direction of a polyline over TANGENT_SPAN_M after (ahead) or before arc length s."""
    if ahead:
        a, b = point_at(line, s_values, s), point_at(line, s_values, s + TANGENT_SPAN_M)
    else:
        a, b = point_at(line, s_values, s - TANGENT_SPAN_M), point_at(line, s_values, s)
    return unit(b[0] - a[0], b[2] - a[2])


def chord_heading(line, s_values, s_from, s_to):
    """Heading of the chord of a polyline from arc length s_from to s_to (each clamped to the line)."""
    a, b = point_at(line, s_values, s_from), point_at(line, s_values, s_to)
    return heading(unit(b[0] - a[0], b[2] - a[2]))


def branch_heading(road, s_from, forward):
    """Heading of the branch of an engine road that leaves arc length s_from towards larger arc lengths (forward) or
    smaller ones: the chord of its centreline over its first TURN_SPAN_M, less where the road ends sooner."""
    s_to = min(s_from + TURN_SPAN_M, road.length) if forward else max(s_from - TURN_SPAN_M, 0.0)
    return chord_heading(road.line, road.s, s_from, s_to)


def resample_closed(points, step):
    """A closed polyline (first point = last point) resampled every ~step metres; the first point is kept and the
    last point equals the first."""
    s = arc_lengths(points)
    count = max(3, int(round(s[-1] / step)))
    out = [point_at(points, s, s[-1] * k / count) for k in range(count)]
    out.append(out[0])
    return out


def smooth_closed(points, anchors):
    """Laplacian smoothing of a closed polyline (last point = first), each point kept within SMOOTH_MAX_SHIFT_M of its
    anchor (its position before smoothing). Heights are not smoothed."""
    pts = [list(p) for p in points[:-1]]
    n = len(pts)
    for _ in range(SMOOTH_ITERATIONS):
        new = []
        for i in range(n):
            prev, nxt = pts[i - 1], pts[(i + 1) % n]
            x = pts[i][0] + SMOOTH_GAIN * (0.5 * (prev[0] + nxt[0]) - pts[i][0])
            z = pts[i][2] + SMOOTH_GAIN * (0.5 * (prev[2] + nxt[2]) - pts[i][2])
            ax, az = anchors[i][0], anchors[i][2]
            shift = math.hypot(x - ax, z - az)
            if shift > SMOOTH_MAX_SHIFT_M:
                x = ax + (x - ax) * SMOOTH_MAX_SHIFT_M / shift
                z = az + (z - az) * SMOOTH_MAX_SHIFT_M / shift
            new.append([x, pts[i][1], z])
        pts = new
    out = [tuple(p) for p in pts]
    out.append(out[0])
    return out


def segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def segment_cuts(a, b, c, d):
    """The fractions t in [0, 1] along segment ab (horizontal) where it meets segment cd, end points included: the
    intersection point, or both ends of the overlap when the two lie on one line; [] when they do not meet."""
    if max(a[0], b[0]) < min(c[0], d[0]) - CUT_EPS_M or min(a[0], b[0]) > max(c[0], d[0]) + CUT_EPS_M \
            or max(a[2], b[2]) < min(c[2], d[2]) - CUT_EPS_M or min(a[2], b[2]) > max(c[2], d[2]) + CUT_EPS_M:
        return []
    rx, rz = b[0] - a[0], b[2] - a[2]
    sx, sz = d[0] - c[0], d[2] - c[2]
    qx, qz = c[0] - a[0], c[2] - a[2]
    r2 = rx * rx + rz * rz
    scale = max(r2, sx * sx + sz * sz, 1e-12)
    den = rx * sz - rz * sx
    if abs(den) > 1e-12 * scale:
        t = (qx * sz - qz * sx) / den
        u = (qx * rz - qz * rx) / den
        slack_t = CUT_EPS_M / math.sqrt(max(r2, 1e-12))
        slack_u = CUT_EPS_M / math.sqrt(max(sx * sx + sz * sz, 1e-12))
        if -slack_t <= t <= 1.0 + slack_t and -slack_u <= u <= 1.0 + slack_u:
            return [min(max(t, 0.0), 1.0)]
        return []
    if r2 <= 0 or abs(qx * rz - qz * rx) > CUT_EPS_M * math.sqrt(r2):     # parallel, not on one line
        return []
    t0 = (qx * rx + qz * rz) / r2
    t1 = ((d[0] - a[0]) * rx + (d[2] - a[2]) * rz) / r2
    lo, hi = max(min(t0, t1), 0.0), min(max(t0, t1), 1.0)
    return [lo, hi] if lo <= hi else []


class SegmentGrid:
    """The segments of polylines [(tag, [(x, y, z)])] by GRID_M cell, for nearest-segment queries."""

    def __init__(self, lines):
        self.lines = lines
        self.cells = {}
        for li, (_, line) in enumerate(lines):
            for si in range(len(line) - 1):
                a, b = line[si], line[si + 1]
                for cx in range(int(math.floor(min(a[0], b[0]) / GRID_M)),
                                int(math.floor(max(a[0], b[0]) / GRID_M)) + 1):
                    for cz in range(int(math.floor(min(a[2], b[2]) / GRID_M)),
                                    int(math.floor(max(a[2], b[2]) / GRID_M)) + 1):
                        self.cells.setdefault((cx, cz), []).append((li, si))

    def near(self, x, z, radius):
        """{line index: (distance, segment index, t)} of the lines with a segment within radius of (x, z)."""
        out = {}
        reach = int(math.ceil(radius / GRID_M))
        cx, cz = int(math.floor(x / GRID_M)), int(math.floor(z / GRID_M))
        for gx in range(cx - reach, cx + reach + 1):
            for gz in range(cz - reach, cz + reach + 1):
                for li, si in self.cells.get((gx, gz), ()):
                    line = self.lines[li][1]
                    a, b = line[si], line[si + 1]
                    dx, dz = b[0] - a[0], b[2] - a[2]
                    seg2 = dx * dx + dz * dz
                    t = 0.0 if seg2 <= 0 else max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[2]) * dz) / seg2))
                    d = math.hypot(x - (a[0] + t * dx), z - (a[2] + t * dz))
                    if d <= radius and (li not in out or d < out[li][0]):
                        out[li] = (d, si, t)
        return out


# ---------------------------------------------------------------------------------------------------------------
# the loop


class Junction:
    def __init__(self, road_a, road_b, s_a, s_b, x, z, gap):
        self.road_a = road_a        # incoming loop road
        self.road_b = road_b        # outgoing loop road
        self.s_a = s_a              # arc length of the junction on road a
        self.s_b = s_b              # ... on road b
        self.x = x
        self.z = z
        self.gap = gap              # distance between the two roads at the junction (m)
        self.turn = 0.0             # signed direction change of the lane line (rad, right positive)


def find_junction(road_a, road_b):
    """Where road a meets road b: the endpoint of either road closest to the other road (at most JOIN_TOL_M)."""
    best = None
    for end_road, other in ((road_a, road_b), (road_b, road_a)):
        for point in (end_road.line[0], end_road.line[-1]):
            s_other, distance, _ = project(other.line, other.s, point[0], point[2])
            if best is None or distance < best[0]:
                best = (distance, end_road, point, s_other)
    distance, end_road, point, s_other = best
    if distance > JOIN_TOL_M:
        raise Refusal("engine roads %d and %d do not meet (closest end %.1f m away, needs <= %.1f m)"
                      % (road_a.idx, road_b.idx, distance, JOIN_TOL_M))
    s_end, _, _ = project(end_road.line, end_road.s, point[0], point[2])
    if end_road is road_a:
        return Junction(road_a, road_b, s_end, s_other, point[0], point[2], distance)
    return Junction(road_a, road_b, s_other, s_end, point[0], point[2], distance)


def loop_roads(roads):
    """The LOOP roads of the dump, each checked against its road_key."""
    legs = []
    for idx, key in LOOP:
        road = roads.get(idx)
        if road is None:
            raise Refusal("engine road %d is not in the dump" % idx)
        if road.key != key:
            raise Refusal("engine road %d of this dump has key %s, the loop was derived on %s: another dump needs the "
                          "loop derived again" % (idx, road.key, key))
        legs.append(road)
    if legs[0].connector:
        raise Refusal("the loop must start on a road, not on a connector")
    return legs


def lane_offsets(legs):
    """The lane offset of each loop road: LANE_WIDTH_FRACTION of its engine width; a connector takes the next road's."""
    out = []
    for i, road in enumerate(legs):
        width = road.width
        k = i
        while legs[k % len(legs)].connector and k < i + len(legs):
            k += 1
            width = legs[k % len(legs)].width
        out.append(LANE_WIDTH_FRACTION * width)
    return out


def line_intersection(p, d, q, e):
    """Intersection of the lines p + t d and q + u e (horizontal); None when they are parallel."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p[0] + t * d[0], p[1] + t * d[1]


def nearest_s(line, s_values, point, s_lo, s_hi):
    """The arc length within [s_lo, s_hi] of the polyline point closest to point (horizontal)."""
    best = None
    for k in range(len(line) - 1):
        if s_values[k + 1] < s_lo or s_values[k] > s_hi:
            continue
        a, b = line[k], line[k + 1]
        dx, dz = b[0] - a[0], b[2] - a[2]
        seg2 = dx * dx + dz * dz
        t = 0.0 if seg2 <= 0 else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[2] - a[2]) * dz) / seg2))
        d = math.hypot(point[0] - (a[0] + t * dx), point[2] - (a[2] + t * dz))
        if best is None or d < best[0]:
            best = (d, s_values[k] + t * math.sqrt(seg2))
    return min(max(best[1], s_lo), s_hi)


def bezier_join(p0, d0, p3, d3):
    """A cubic Bezier from p0 (direction d0) to p3 (direction d3) whose handles make a near-circular arc for the turn
    between d0 and d3 (handle = chord x 2/3 x tan(turn / 4) / sin(turn / 2), a third of the chord without a turn),
    sampled every SPLINE_STEP_M (p0 and p3 excluded)."""
    chord = math.hypot(p3[0] - p0[0], p3[2] - p0[2])
    turn = abs(wrap(heading(d3) - heading(d0)))
    handle = chord / 3.0 if turn < 1e-3 else chord * (2.0 / 3.0) * math.tan(turn / 4.0) / math.sin(turn / 2.0)
    c1 = (p0[0] + handle * d0[0], p0[2] + handle * d0[1])
    c2 = (p3[0] - handle * d3[0], p3[2] - handle * d3[1])
    pieces = max(2, int(math.ceil(chord * 1.2 / SPLINE_STEP_M)))
    out = []
    for k in range(1, pieces):
        t = float(k) / pieces
        u = 1.0 - t
        x = u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p3[0]
        z = u ** 3 * p0[2] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p3[2]
        out.append((x, p0[1] + t * (p3[1] - p0[1]), z))
    return out


def build_lane_line(legs):
    """(closed lane line starting START_ALONG_M along the first road, [Junction] between consecutive loop roads,
    [corner dict])."""
    junctions = [find_junction(legs[i], legs[(i + 1) % len(legs)]) for i in range(len(legs))]
    offsets = lane_offsets(legs)
    lanes = []
    for i, road in enumerate(legs):
        s_from, s_to = junctions[i - 1].s_b, junctions[i].s_a
        if abs(s_to - s_from) < 0.5:
            raise Refusal("engine road %d is used over %.2f m only" % (road.idx, abs(s_to - s_from)))
        lanes.append(offset_line(sub_line(road, s_from, s_to), offsets[i]))
    first = lanes[0]
    first_s = arc_lengths(first)
    if not 0.0 < START_ALONG_M < first_s[-1]:
        raise Refusal("START_ALONG_M %.1f m is not on the first road's lane line (%.1f m)" % (START_ALONG_M,
                                                                                         first_s[-1]))
    split = next(k for k, s in enumerate(first_s) if s > START_ALONG_M)
    start = point_at(first, first_s, START_ALONG_M)
    loop = [start] + first[split:]
    marks = []                                      # lane-line arc length of each junction point
    for i in range(1, len(lanes)):
        marks.append(arc_lengths(loop)[-1])
        loop += lanes[i][1:]
    marks.append(arc_lengths(loop)[-1])
    loop += first[1:split] + [start]
    s = arc_lengths(loop)
    for i, junction in enumerate(junctions):
        junction.turn = wrap(heading(direction_at(loop, s, marks[i], True))
                             - heading(direction_at(loop, s, marks[i], False)))
    # clusters of junction points, joined from the last one back (earlier arc lengths stay valid)
    clusters = []
    for i, mark in enumerate(marks):
        if clusters and mark - marks[clusters[-1][-1]] < CLUSTER_M:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    # a junction point needs a join when the lane turns there or changes its offset (roads of different widths)
    core = [abs(math.degrees(junctions[i].turn)) >= CORNER_MIN_DEG
            or abs(offsets[i] - offsets[(i + 1) % len(offsets)]) > 0.01 for i in range(len(junctions))]
    corners = []
    for cluster in reversed(clusters):
        s_first, s_last = marks[cluster[0]], marks[cluster[-1]]
        core_marks = [marks[k] for k in cluster if core[k]]
        p_in = point_at(loop, s, s_first - TANGENT_SPAN_M)
        p_out = point_at(loop, s, s_last + TANGENT_SPAN_M)
        d_in = direction_at(loop, s, s_first - TANGENT_SPAN_M, False)
        d_out = direction_at(loop, s, s_last + TANGENT_SPAN_M, True)
        turn = wrap(heading(d_out) - heading(d_in))
        apex = None
        if abs(math.degrees(turn)) >= CORNER_MIN_DEG:
            apex = line_intersection((p_in[0], p_in[2]), d_in, (p_out[0], p_out[2]), d_out)
        if apex is None and not core_marks:
            continue                                # straight on between roads of one width: nothing to join
        if core_marks:
            s0, s1 = min(core_marks) - CORNER_MIN_CUT_M, max(core_marks) + CORNER_MIN_CUT_M
        if apex is not None:
            radius = CORNER_RIGHT_M if turn > 0 else CORNER_LEFT_M
            cut = radius * math.tan(abs(turn) / 2.0)
            a = (apex[0] - cut * d_in[0], 0.0, apex[1] - cut * d_in[1])
            b = (apex[0] + cut * d_out[0], 0.0, apex[1] + cut * d_out[1])
            s0 = nearest_s(loop, s, a, s_first - cut - CLUSTER_M, s_last)
            s1 = nearest_s(loop, s, b, s_first, s_last + cut + CLUSTER_M)
            if core_marks:
                s0, s1 = min(s0, min(core_marks) - CORNER_MIN_CUT_M), max(s1, max(core_marks) + CORNER_MIN_CUT_M)
        if s0 <= TANGENT_SPAN_M or s1 >= s[-1] - TANGENT_SPAN_M:
            raise Refusal("the corner at the junction of roads %d and %d reaches the route start" % (
                junctions[cluster[0]].road_a.idx, junctions[cluster[-1]].road_b.idx))
        p0, p3 = point_at(loop, s, s0), point_at(loop, s, s1)
        arc = bezier_join(p0, direction_at(loop, s, s0, False), p3, direction_at(loop, s, s1, True))
        i0 = next(k for k, v in enumerate(s) if v > s0)
        i1 = next(k for k, v in enumerate(s) if v >= s1)
        loop = loop[:i0] + [p0] + arc + [p3] + loop[i1:]
        s = arc_lengths(loop)
        corners.append({"roads": [junctions[cluster[0]].road_a.idx] + [junctions[k].road_b.idx for k in cluster],
                        "turn_deg": round(math.degrees(turn), 1), "apex": apex is not None,
                        "length_m": round(s1 - s0, 2)})
    corners.reverse()
    return loop, junctions, corners


def heights_from(points, legs):
    """y of the nearest loop-road centreline point for every point."""
    grid = SegmentGrid([(road.idx, road.line) for road in legs])
    out = []
    for p in points:
        near = grid.near(p[0], p[2], 2.0 * GRID_M)
        if not near:
            raise Refusal("point (%.1f, %.1f) is far from every loop road" % (p[0], p[2]))
        li, (_, si, t) = min(near.items(), key=lambda item: item[1][0])
        line = grid.lines[li][1]
        out.append((p[0], line[si][1] + t * (line[si + 1][1] - line[si][1]), p[2]))
    return out


def lane_corridor(points, legs):
    """(arc lengths [s_k], half widths [w_k / 2]) of the lane corridor of a closed route (module docstring): piece k
    covers [s_k, s_k+1), the last one to the route end. Each point takes LANE_CORRIDOR_FRACTION of the engine width of
    its nearest loop road that is not a connector, then the smallest of those within CORRIDOR_SPAN_M along the route
    (around the closed loop)."""
    grid = SegmentGrid([(road.idx, road.line) for road in legs if not road.connector])
    widths = {road.idx: road.width for road in legs}
    s = arc_lengths(points)
    length = s[-1]
    own = []
    for p in points[:-1]:
        near = grid.near(p[0], p[2], 2.0 * GRID_M)
        if not near:
            raise Refusal("point (%.1f, %.1f) is far from every loop road" % (p[0], p[2]))
        li = min(near.items(), key=lambda item: item[1][0])[0]
        own.append(LANE_CORRIDOR_FRACTION * widths[grid.lines[li][0]])
    halves = []
    for i in range(len(own)):
        nearby = [own[j] for j in range(len(own))
                  if min(abs(s[j] - s[i]), length - abs(s[j] - s[i])) <= CORRIDOR_SPAN_M]
        halves.append(round(min(nearby), CORRIDOR_DECIMALS))
    starts, values = [], []
    scale = 10 ** CORRIDOR_DECIMALS
    for i, half in enumerate(halves):
        if not values or half != values[-1]:
            starts.append(math.floor(s[i] * scale) / scale)       # rounded down: the piece holds its first point
            values.append(half)
    if starts[0] != 0.0 or any(b <= a for a, b in zip(starts, starts[1:])) or min(values) <= 0.0:
        raise Refusal("the lane corridor pieces %s / %s are not ordered from 0 with positive half widths"
                      % (starts, values))
    return starts, values


def route_curvature(points, s, at):
    """Signed curvature (1/m, right positive) at arc length at: the tangent heading change over +-CURV_WINDOW_M
    (C17's route_curvature, DD-5-min)."""
    def tangent(t):
        a = point_at(points, s, max(t - 0.5, 0.0))
        b = point_at(points, s, min(t + 0.5, s[-1]))
        return math.atan2(b[0] - a[0], b[2] - a[2])
    lo = max(at - CURV_WINDOW_M, 0.0)
    hi = min(at + CURV_WINDOW_M, s[-1])
    if hi - lo < 1e-6:
        return 0.0
    return wrap(tangent(hi) - tangent(lo)) / (hi - lo)


def turning(points):
    """(left, right) absolute turning of the polyline in rad, and the left share."""
    left = right = 0.0
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        turn = wrap(math.atan2(c[0] - b[0], c[2] - b[2]) - math.atan2(b[0] - a[0], b[2] - a[2]))
        if turn > 0:
            right += turn
        else:
            left -= turn
    total = left + right
    return left, right, (left / total if total > 0 else None)


# ---------------------------------------------------------------------------------------------------------------
# junctions of the engine graph and the maneuvers


def used_ranges(legs, junctions):
    """{road_idx: [(s_lo, s_hi)]} of the pieces of each loop road the loop drives on."""
    used = {}
    for i, road in enumerate(legs):
        lo, hi = sorted((junctions[i - 1].s_b, junctions[i].s_a))
        used.setdefault(road.idx, []).append((lo, hi))
    return used


def junction_events(points, legs, junctions, roads):
    """The junction events of the loop in the engine graph, in route order: dicts with s (on the route), x, z, the
    loop roads in (arriving) and out (leaving), arms (other engine roads meeting there), branches ([(road_idx,
    heading)] of the branches the route does not take, branch_heading) and leg (True for a junction point between
    consecutive loop roads). Arms: an engine road wider than a connector, not a loop road, with an end within ARM_TOL_M
    of a loop road's used piece (its branch leaves from that end); at a junction point, a loop road that continues past
    it (the point farther than ARM_TOL_M from both of its ends; its branch is the part the route does not drive: the
    arriving road on beyond the point, the leaving road back from it)."""
    s = arc_lengths(points)
    used = used_ranges(legs, junctions)
    events = []
    for i, junction in enumerate(junctions):
        arms = set()
        branches = []
        before, after = junctions[i - 1], junctions[(i + 1) % len(junctions)]
        # road a is driven from before.s_b to s_a and goes on the same way; road b is driven from s_b to after.s_a
        for road, s_j, forward in ((junction.road_a, junction.s_a, junction.s_a > before.s_b),
                                   (junction.road_b, junction.s_b, junction.s_b > after.s_a)):
            if not road.connector and ARM_TOL_M < s_j < road.length - ARM_TOL_M:
                arms.add(road.idx)
                branches.append((road.idx, branch_heading(road, s_j, forward)))
        s_r, _, _ = project(points, s, junction.x, junction.z)
        events.append({"s": s_r, "x": junction.x, "z": junction.z, "in": junction.road_a, "out": junction.road_b,
                       "arms": arms, "branches": branches, "leg": True})
    grid = SegmentGrid([(idx, roads[idx].line) for idx in sorted(used)])
    for road in sorted(roads.values(), key=lambda r: r.idx):
        if road.connector or road.idx in used:
            continue
        for end, s_end in ((road.line[0], 0.0), (road.line[-1], road.length)):
            near = []
            for li, (d, si, t) in grid.near(end[0], end[2], ARM_TOL_M).items():
                leg = roads[grid.lines[li][0]]
                s_leg = leg.s[si] + t * (leg.s[si + 1] - leg.s[si])
                if any(lo - 1.0 <= s_leg <= hi + 1.0 for lo, hi in used[leg.idx]):
                    near.append((d, leg.idx))
            if near:                                # the arm meets the nearest loop road it touches
                leg = roads[min(near)[1]]
                s_r, _, _ = project(points, s, end[0], end[2])
                events.append({"s": s_r, "x": end[0], "z": end[2], "in": leg, "out": leg, "arms": {road.idx},
                               "branches": [(road.idx, branch_heading(road, s_end, s_end == 0.0))], "leg": False})
    events.sort(key=lambda e: (e["s"], not e["leg"]))
    return events


def same_side_type(kind, route_deg, branch_degs):
    """The maneuver type after the same-side rule (D-E, SAME_SIDE_STRAIGHT_DEG): kind is the geometric type, route_deg
    the turn angle of the route's branch and branch_degs those of the junction's other branches (right positive).
    A left or right maneuver becomes straight when another branch turns to the same side as the route's branch, the
    route's branch is strictly the closest to straight on of all of them and it turns by SAME_SIDE_STRAIGHT_DEG or
    less; every other maneuver keeps its geometric type."""
    if kind == "straight" or abs(route_deg) > SAME_SIDE_STRAIGHT_DEG:
        return kind
    if not any(deg * route_deg > 0 for deg in branch_degs):
        return kind                                 # alone on its side
    if all(abs(route_deg) < abs(deg) for deg in branch_degs):
        return "straight"
    return kind


def maneuvers_of(points, events):
    """(maneuvers, junction facts): events closer than MANEUVER_MERGE_M along the route are one junction; a junction
    with at least one arm is a maneuver at the middle of its events, typed by the route's heading change from
    TURN_SPAN_M before its first event to TURN_SPAN_M after its last one, then by same_side_type on the turn angles of
    its branches (module docstring); its roads in and out are those of its junction points between loop roads
    (connectors skipped), else the loop road its arm meets."""
    s = arc_lengths(points)
    clusters = []
    for event in events:
        if clusters and event["s"] - clusters[-1][-1]["s"] < MANEUVER_MERGE_M:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    maneuvers, facts = [], []
    for cluster in clusters:
        arms = set().union(*(e["arms"] for e in cluster))
        if not arms:
            continue                                # two loop roads joined end to end: no junction in the engine
        s_first, s_last = cluster[0]["s"], cluster[-1]["s"]
        before = direction_at(points, s, max(s_first - TURN_SPAN_M + TANGENT_SPAN_M, TANGENT_SPAN_M), False)
        after = direction_at(points, s, min(s_last + TURN_SPAN_M - TANGENT_SPAN_M, s[-1] - TANGENT_SPAN_M), True)
        turn = math.degrees(wrap(heading(after) - heading(before)))
        geometric = "straight"
        if turn >= MANEUVER_STRAIGHT_DEG:
            geometric = "right"
        elif turn <= -MANEUVER_STRAIGHT_DEG:
            geometric = "left"
        # the turn angles of the branches: chords over TURN_SPAN_M from the junction against the arrival chord
        arrival = chord_heading(points, s, max(s_first - TURN_SPAN_M, 0.0), s_first)
        route_deg = math.degrees(wrap(chord_heading(points, s, s_last, min(s_last + TURN_SPAN_M, s[-1])) - arrival))
        branches = [(idx, math.degrees(wrap(h - arrival))) for e in cluster for idx, h in e["branches"]]
        kind = same_side_type(geometric, route_deg, [deg for _, deg in branches])
        joins = [e for e in cluster if e["leg"]] or cluster       # the loop roads the junction joins, else the arm's
        road_in = next((e["in"] for e in joins if not e["in"].connector), joins[0]["in"]).idx
        road_out = next((e["out"] for e in reversed(joins) if not e["out"].connector), joins[-1]["out"]).idx
        ids = [road_in] + ([road_out] if road_out != road_in else []) + sorted(arms - {road_in, road_out})
        s_m = 0.5 * (s_first + s_last)
        if not 0.0 < s_m < s[-1]:
            continue
        junction_id = "e" + "_".join(str(i) for i in ids)
        maneuvers.append({"s_m": round(s_m, 3), "type": kind, "junction_id": junction_id})
        facts.append({"junction_id": junction_id, "s_m": round(s_m, 1), "turn_deg": round(turn, 1),
                      "geometric_type": geometric, "route_branch_deg": round(route_deg, 1),
                      "branches": [[idx, round(deg, 1)] for idx, deg in branches],
                      "x": round(cluster[0]["x"], 1), "z": round(cluster[0]["z"], 1)})
    return maneuvers, facts


def check_route(points, legs, junctions, events, roads, train_rule, vehicles):
    """Every check of the module docstring; returns the facts for the summary. Raises Refusal on a failure."""
    box, buffer_m, _ = train_rule
    x0, x1, z0, z1 = box
    s = arc_lengths(points)
    facts = {"length_m": s[-1], "points": len(points)}
    for p in points:
        if x0 - buffer_m <= p[0] < x1 + buffer_m and z0 - buffer_m <= p[2] < z1 + buffer_m:
            raise Refusal("point (%.1f, %.1f) lies in the VAL-L box or its %.0f m buffer: not TRAIN"
                          % (p[0], p[2], buffer_m))
    if not LENGTH_RANGE_M[0] <= s[-1] <= LENGTH_RANGE_M[1]:
        raise Refusal("length %.1f m outside %s" % (s[-1], LENGTH_RANGE_M))
    surface = SegmentGrid([(road.idx, road.line) for road in roads.values() if road.surface_width > 0])
    centres = [(e["x"], e["z"]) for e in events]
    loop_ids = set(road.idx for road in legs)
    worst = (None, None)
    lane_dev = []
    for i, p in enumerate(points[:-1]):
        in_junction = any(math.hypot(p[0] - jx, p[2] - jz) <= JUNCTION_ZONE_M for jx, jz in centres)
        best = None
        for li, (distance, _, _) in surface.near(p[0], p[2], 40.0).items():
            road = roads[surface.lines[li][0]]
            half = road.surface_width / 2.0
            allowed = half + JUNCTION_ALLOW_M if in_junction else half - ON_ROAD_MARGIN_M
            if best is None or distance - allowed < best[0]:
                best = (distance - allowed, road.idx, distance)
        if best is None or best[0] > 0:
            raise Refusal("point (%.1f, %.1f) is off the engine road surface (%s)" % (
                p[0], p[2], "no engine road nearby" if best is None else "%.2f m from road %d" % (best[2], best[1])))
        if worst[0] is None or best[0] > worst[0]:
            worst = (best[0], best[1])
        if in_junction:
            continue
        own = [(d, li, si, t) for li, (d, si, t) in surface.near(p[0], p[2], 10.0).items()
               if surface.lines[li][0] in loop_ids and not roads[surface.lines[li][0]].connector]
        if not own:
            raise Refusal("point (%.1f, %.1f) is not on a loop road" % (p[0], p[2]))
        d, li, si, t = min(own)
        road = roads[surface.lines[li][0]]
        a, b = road.line[si], road.line[si + 1]
        foot = (a[0] + t * (b[0] - a[0]), a[2] + t * (b[2] - a[2]))
        q, r = points[i - 1], points[i + 1]
        nx, nz = right_of(unit(r[0] - q[0], r[2] - q[2]))
        offset = (p[0] - foot[0]) * nx + (p[2] - foot[1]) * nz    # right of the route's own direction
        lane_dev.append((abs(offset - road.width * LANE_WIDTH_FRACTION), road.idx, p, offset))
    facts["on_road_worst_excess_m"] = worst[0]
    facts["lane_points"] = len(lane_dev)
    facts["lane_dev_max_m"] = max(v[0] for v in lane_dev) if lane_dev else None
    bad = [v for v in lane_dev if v[0] > LANE_TOL_M]
    if bad:
        v = bad[0]
        raise Refusal("point (%.1f, %.1f) lies %+.2f m right of engine road %d's centreline, its lane at %+.2f m (%d "
                      "points off by more than %.1f m)" % (v[2][0], v[2][2], v[3], v[1], roads[v[1]].width *
                                                           LANE_WIDTH_FRACTION, len(bad), LANE_TOL_M))
    clearance = None
    for name, vx, vz in vehicles:
        d = min(math.hypot(p[0] - vx, p[2] - vz) for p in points)
        if d < CLEARANCE_M:
            raise Refusal("parked vehicle %s at (%.1f, %.1f) is %.1f m from the loop (needs >= %.0f m)"
                          % (name, vx, vz, d, CLEARANCE_M))
        if clearance is None or d < clearance[1]:
            clearance = (name, d)
    facts["closest_parked_vehicle"] = clearance
    kappa = [route_curvature(points, s, s_i) for s_i in s]
    facts["max_abs_kappa"] = max(abs(k) for k in kappa)
    if facts["max_abs_kappa"] > 1.0 / MIN_RADIUS_M:
        at = kappa.index(max(kappa, key=abs))
        raise Refusal("turning radius %.1f m at s %.1f m (needs >= %.0f m)" % (1.0 / abs(kappa[at]), s[at],
                                                                                 MIN_RADIUS_M))
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            if i == 0 and j == len(points) - 2:
                continue
            if segments_cross(points[i], points[i + 1], points[j], points[j + 1]):
                raise Refusal("the loop crosses itself near (%.1f, %.1f)" % (points[i][0], points[i][2]))
    left, right, share = turning(points)
    facts.update({"left_turn_rad": left, "right_turn_rad": right, "left_share": share})
    if left < MIN_TURN_RAD or right < MIN_TURN_RAD:
        raise Refusal("turning content left %.2f rad, right %.2f rad (each needs >= %.1f rad for C17)"
                      % (left, right, MIN_TURN_RAD))
    if abs(1.0 - 2.0 * share) <= MIRROR_DEV_MIN:
        raise Refusal("left share %.3f: its mirror image differs by only %.3f (needs > %.2f)"
                      % (share, abs(1.0 - 2.0 * share), MIRROR_DEV_MIN))
    return facts


def check_trails(points, trails):
    """(crossings [(trail id, x, z)], touches [(trail id, distance)]) of the decoder trails against the route. A trail
    meets the route at every point where one of its segments meets a route segment, end points included (a trail
    through a route point, or with a point on the line, meets it there; the same point found from two adjacent
    segments counts once). A meeting point with at least TRAIL_END_M of the trail on both sides of it, measured along
    the trail from that point, is a crossing (414 across engine road 121); a trail that meets the line only nearer to
    one of its ends ends on the loop's road, a touch at distance 0, and so is every trail with an end within
    TRAIL_TOUCH_M of the route (at that distance)."""
    crossings, touches = [], []
    route_s = arc_lengths(points)
    for trail_id, line in trails:
        pts = [(x, 0.0, z) for x, z in line]
        along = arc_lengths(pts)
        cuts = []
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            for c, d in zip(points, points[1:]):
                for t in segment_cuts(a, b, c, d):
                    cuts.append((along[k] + t * (along[k + 1] - along[k]),
                                 a[0] + t * (b[0] - a[0]), a[2] + t * (b[2] - a[2])))
        touched = None
        last = None
        for s_cut, x, z in sorted(cuts):
            if last is not None and s_cut - last <= CUT_EPS_M:
                continue                            # one point, found from two segments sharing a vertex
            last = s_cut
            if min(s_cut, along[-1] - s_cut) >= TRAIL_END_M:
                crossings.append((trail_id, round(x, 1), round(z, 1)))
            else:
                touched = 0.0
        for end in (pts[0], pts[-1]):
            distance = project(points, route_s, end[0], end[2])[1]
            if distance <= TRAIL_TOUCH_M and (touched is None or distance < touched):
                touched = distance
        if touched is not None:
            touches.append((trail_id, round(touched, 2)))
    return crossings, touches


def build_route(dump_path, splits_path, layer_path):
    """(route dict, facts) or Refusal."""
    for path in (dump_path, splits_path, layer_path):
        reason = sealed_reason(path)
        if reason:
            raise Refusal(reason)
        if not os.path.exists(path):
            raise Refusal("missing input %s" % path)
    roads, provenance = read_dump(dump_path)
    train_rule = read_train_rule(splits_path)
    vehicles = read_parked_vehicles(layer_path)
    legs = loop_roads(roads)
    lane, junctions, corners = build_lane_line(legs)
    dense = resample_closed(lane, SMOOTH_STEP_M)
    smooth = smooth_closed(dense, dense[:-1])
    points = resample_closed(smooth, STEP_M)
    points = heights_from(points, legs)
    points[-1] = points[0]
    rounded = [[round(p[0], COORD_DECIMALS), round(p[1], HEIGHT_DECIMALS), round(p[2], COORD_DECIMALS)]
               for p in points]
    rounded[-1] = list(rounded[0])
    pts = [tuple(p) for p in rounded]
    events = junction_events(pts, legs, junctions, roads)
    facts = check_route(pts, legs, junctions, events, roads, train_rule, vehicles)
    maneuvers, junction_facts = maneuvers_of(pts, events)
    facts["junctions"] = junction_facts
    corridor_s, corridor_half = lane_corridor(pts, legs)
    facts["lane_corridor"] = list(zip(corridor_s, corridor_half))
    length = arc_lengths(pts)[-1]
    offsets = lane_offsets(legs)
    route = {
        "schema": drivedit_nav.ROUTE_SCHEMA_ID,
        "route_id": ROUTE_ID,
        "map_id": MAP_ID,
        "split": SPLIT,
        "nav_version": drivedit_nav.NAV_VERSION,
        "lane_side": "right",
        "closed": True,
        "polyline": rounded,
        "maneuvers": maneuvers,
        "stops": [{"s_m": round(length, 6), "kind": "route_end"}],
        "length_m": round(length, 6),
        "source": {
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "n_points": len(rounded),
            "road_graph": provenance,
            "legs": [{"road": road.idx, "key": road.key, "width_m": road.width, "lane_offset_m": round(offsets[i], 3)}
                     for i, road in enumerate(legs)],
            "junctions": [{"roads": [j.road_a.idx, j.road_b.idx], "x": round(j.x, 2), "z": round(j.z, 2),
                           "gap_m": round(j.gap, 2), "turn_deg": round(math.degrees(j.turn), 1)} for j in junctions],
            "corners": corners,
            "corner_radius_m": {"right": CORNER_RIGHT_M, "left": CORNER_LEFT_M, "min_cut_m": CORNER_MIN_CUT_M},
            "lane_width_fraction": LANE_WIDTH_FRACTION,
            "lane_corridor_s_m": corridor_s,
            "lane_corridor_half_width_m": corridor_half,
            "lane_corridor_rule": {"half_width_fraction": LANE_CORRIDOR_FRACTION, "narrowest_within_m": CORRIDOR_SPAN_M,
                                   "road": "nearest loop road that is not a connector"},
            "start_along_m": START_ALONG_M,
            "smoothing": {"step_m": SMOOTH_STEP_M, "max_shift_m": SMOOTH_MAX_SHIFT_M},
            "maneuver_rule": {"arm_tol_m": ARM_TOL_M, "merge_m": MANEUVER_MERGE_M, "turn_span_m": TURN_SPAN_M,
                              "straight_below_deg": MANEUVER_STRAIGHT_DEG,
                              "same_side_straight_within_deg": SAME_SIDE_STRAIGHT_DEG,
                              "same_side_rule": SAME_SIDE_RULE},
            "maneuver_branches": [{"junction_id": f["junction_id"], "turn_deg": f["turn_deg"],
                                   "geometric_type": f["geometric_type"], "route_branch_deg": f["route_branch_deg"],
                                   "branches": f["branches"]} for f in junction_facts],
            "inputs": {
                "splits": {"name": os.path.basename(splits_path), "sha256": sha256_file(splits_path),
                           "status": (train_rule[2].split() or ["unknown"])[0]},
            },
            "parked_vehicles_checked": [[name, round(vx, 2), round(vz, 2)] for name, vx, vz in vehicles],
            "train_rule": {"val_l_box_xz": list(train_rule[0]), "buffer_m": train_rule[1]},
            "geometry": "engine road dump (E-08a %s): engine road centrelines, right lane at %g of each engine road "
                        "width; junctions from the engine graph" % (provenance["launch_id"], LANE_WIDTH_FRACTION),
        },
        "description": "Arland T0 loop (DD-7-min) on the engine road graph: engine roads %s (E-08a %s road_idx), "
                       "clockwise, right-lane line, inside TRAIN of splits_v2 (PROPOSED, not frozen)."
                       % (", ".join(str(road.idx) for road in legs), provenance["launch_id"]),
    }
    errors = drivedit_nav.validate_route(route)
    if errors:
        raise Refusal("the route fails drivedit_nav.validate_route: %s" % "; ".join(errors))
    drivedit_nav.route_from_dict(route)
    return route, facts


def route_text(route):
    """The file content: canonical JSON and one LF (sha256 of the file without its LF = Route.sha256)."""
    return drivedit_nav.canonical_json(route) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Arland T0 loop on the engine road graph (DD-7-min)")
    ap.add_argument("--dump", default=DEFAULT_DUMP, help="E-08a road dump directory (default: the versioned copy)")
    ap.add_argument("--splits", required=True, help="map splits JSON (splits_v2, PROPOSED)")
    ap.add_argument("--layer", default=DEFAULT_LAYER, help="DC_Arland layer with the parked vehicles")
    ap.add_argument("--trails", help="decoder road polylines TSV: check that no decoder trail crosses the loop")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--out", help="write the route file here")
    group.add_argument("--check", help="regenerate and compare with this route file")
    args = ap.parse_args(argv)
    try:
        route, facts = build_route(args.dump, args.splits, args.layer)
        trail_facts = None
        if args.trails:
            reason = sealed_reason(args.trails)
            if reason:
                raise Refusal(reason)
            pts = [tuple(p) for p in route["polyline"]]
            crossings, touches = check_trails(pts, read_trails(args.trails))
            if crossings:
                raise Refusal("decoder trails cross the loop (not in the engine's road network): %s" % crossings)
            trail_facts = touches
        text = route_text(route)
        if args.out:
            out = check_out_path(args.out)
            with open(out, "w", encoding="ascii", newline="") as f:
                f.write(text)
    except Refusal as e:
        print("REFUSED: %s" % e)
        return 2
    sha = drivedit_nav.route_from_dict(route).sha256
    print("route %s: %.1f m, %d points, %d maneuvers, sha256 %s" % (route["route_id"], facts["length_m"],
                                                                    facts["points"], len(route["maneuvers"]), sha))
    print("  engine roads %s of E-08a %s" % (" ".join(str(leg["road"]) for leg in route["source"]["legs"]),
                                            route["source"]["road_graph"]["launch_id"]))
    print("  turning: left %.2f rad, right %.2f rad, left share %.3f (mirror %.3f)" % (
        facts["left_turn_rad"], facts["right_turn_rad"], facts["left_share"], 1.0 - facts["left_share"]))
    print("  min turning radius %.1f m; worst on-road margin %.2f m; right lane: %d points away from junctions, max "
          "|offset - w/4| %.2f m; closest parked vehicle %s at %.1f m" % (
              1.0 / facts["max_abs_kappa"], -facts["on_road_worst_excess_m"], facts["lane_points"],
              facts["lane_dev_max_m"], facts["closest_parked_vehicle"][0], facts["closest_parked_vehicle"][1]))
    print("  lane corridor half width from s: %s" % ", ".join("%.1f m: %.3f m" % piece
                                                             for piece in facts["lane_corridor"]))
    print("  TRAIN of %s (status %s)" % (os.path.basename(args.splits), route["source"]["inputs"]["splits"]["status"]))
    for fact in facts["junctions"]:
        kind = next(m["type"] for m in route["maneuvers"] if m["junction_id"] == fact["junction_id"])
        print("  maneuver s %.1f m %s %s (route turn %+.1f deg%s; branches: route %+.1f, %s)" % (
            fact["s_m"], kind, fact["junction_id"], fact["turn_deg"],
            "" if kind == fact["geometric_type"] else ": geometric %s, %s by the same-side rule" % (
                fact["geometric_type"], kind),
            fact["route_branch_deg"], ", ".join("%d %+.1f" % (idx, deg) for idx, deg in fact["branches"])))
    if trail_facts is not None:
        print("  decoder trails: none crosses the loop; trails ending on its roads or within %.0f m: %s" % (
            TRAIL_TOUCH_M, ", ".join("%d at %.2f m" % t for t in trail_facts) or "none"))
    if args.check:
        with open(args.check, "r", encoding="ascii", newline="") as f:
            if f.read() != text:
                print("DIFFERS: %s is not what the generator builds from these inputs" % args.check)
                return 1
        print("check: %s is identical" % args.check)
    elif args.out:
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
