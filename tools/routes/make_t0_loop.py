#!/usr/bin/env python3
"""Build the hand-picked Arland T0 loop (DD-7-min) as a drivedit_nav route (drivedit.route/1).

Run it with the isolated interpreter (stdlib only, plus the repo's drivedit_nav package):

    python3 -I -B tools/routes/make_t0_loop.py --polylines <Arland_road_polylines.tsv> --splits <splits_v2.json>
        [--layer addons/DataCapture/worlds/DataCapture/DC_Arland_Layers/default.layer]
        [--out addons/DataCapture/Configs/routes/arland_t0_loop.json | --check FILE]

Inputs (read only): the decoder's road polylines of Arland (road_id, class, generator, width_m, n_pts, length_m,
has_angles, points_xyz "x,y,z;..."; program/evidence/maps/Arland_road_polylines.tsv), the map splits
(program/maps/splits_v2_PROPOSED.json: the VAL-L box of Arland and its buffer) and the DC_Arland layer (the parked
vehicles the loop must keep clear of). --out writes the route; --check regenerates it and compares it byte for byte
with FILE (exit 1 when they differ); without either the route is only built, checked and summarised.

THE LOOP (counter-clockwise seen from above, about 1.5 km, all of it asphalt E02/E03, 8 m wide, north-west of and
across the main runway): road 452 east from its junction with 451, across the runway and north to the junction with
453 and 461; right onto 461 north-east to road 93; left onto 93 west to its end, where 451 continues it; 451 south
back to 452. The start lies START_ALONG_M along 452 (a straight in the right lane); the loop is closed there.

How the driving line is made (the route polyline is the driving line, drivedit_nav never offsets it):
  1. each road's control points (the decoder's SplinePoints) are interpolated by a centripetal Catmull-Rom spline;
     a road whose decoder record carries angles is refused (the decoder does not apply them, so its points are not
     trusted);
  2. each leg runs between its junctions with the previous and the next leg (the closest approach of an endpoint of
     one road to the other road, at most JOIN_TOL_M);
  3. the leg is offset to the right lane centre: right of the driving direction by a quarter of the road width;
  4. each junction gets a circular corner (CORNER_RIGHT_M for right turns, which cut the inner corner of the right
     lane, CORNER_LEFT_M for left turns); a continuation (turn below CORNER_MIN_DEG) is joined as it is;
  5. the closed line is resampled every 1 m and smoothed, each point staying within SMOOTH_MAX_SHIFT_M of its lane
     position, then resampled every STEP_M with the first and last point identical.
Heights are the decoder heights of the nearest leg centreline point (nav_v1 ignores y).

Checks before anything is written (exit 2 and nothing written when one fails):
  * every point lies in TRAIN: outside the VAL-L box of the splits file dilated by its buffer (a conservative
    rectangle); splits_v2 is PROPOSED, so the result is recorded with that status;
  * every point lies on a drivable road surface (asphalt, dirt or trail of the decoder): within half the width minus
    ON_ROAD_MARGIN_M of the nearest centreline, or half the width plus JUNCTION_ALLOW_M within JUNCTION_ZONE_M of a
    junction (junction areas are wider than the two roads);
  * every parked vehicle of the DC_Arland layer is at least CLEARANCE_M away;
  * the turning radius (heading change over CURV_WINDOW_M) is at least MIN_RADIUS_M everywhere;
  * the line does not cross itself; its length lies in LENGTH_RANGE_M;
  * C17 content (DD-5-min): left and right turning each at least MIN_TURN_RAD, and the left share of the absolute
    turning differs from its mirror image by more than MIRROR_DEV_MIN (so a mirrored route fails C17);
  * drivedit_nav.validate_route accepts it.

Maneuvers: one per junction, at the route position of the junction point: the leg junctions with their turn
(left / right, or straight below MANEUVER_STRAIGHT_DEG) and every side road (decoder asphalt, dirt or trail) whose
end touches the loop within SIDE_ROAD_M (straight on). Stops: the one route_end at the route end (nav_v1).

Output: the route's canonical JSON (drivedit_nav.canonical_json: sorted keys, no whitespace, ASCII) and one LF, so
the sha256 of the file without its final LF is drivedit_nav's Route.sha256; DC_ExpertHL computes episodes.route_sha256
that way. The geometry is decoder-derived and unverified in the engine: E-08a / E-08b (DD-4-probe, DD-7) check the
roads against the engine's road dump; regenerate the loop then.

Sealed map: any input path naming eden or everon is refused with exit 2 before anything is read (Everon is the
zero-shot test map). Outputs only under /mnt/e/ or /dev/shm/ (never /mnt/c or /tmp).
Exit codes: 0 built (and written or equal), 1 --check found a difference, 2 refused (input, check or path problem).
"""

import argparse
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
GENERATOR_VERSION = 1
ROUTE_ID = "arland_t0_loop"
MAP_ID = "arland"
SPLIT = "train"
DEFAULT_LAYER = os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Arland_Layers",
                             "default.layer")

# The loop: the roads in driving order (decoder road ids); leg i ends where leg i + 1 begins
LEGS = (452, 461, 93, 451)
START_ALONG_M = 100.0           # the start (and end) of the closed route, along the first leg's lane line
LANE_WIDTH_FRACTION = 0.25      # right lane centre = a quarter of the road width right of the centreline

DRIVABLE_CLASSES = ("asphalt_E02_E03_secondary", "dirt_road", "trail")
LEG_CLASSES = ("asphalt_E02_E03_secondary",)
SPLINE_STEP_M = 0.5             # Catmull-Rom sampling of the road centrelines
JOIN_TOL_M = 4.0                # a road end must lie this close to the next road
TANGENT_SPAN_M = 5.0            # the leg direction at a junction: over this much of the lane line
CORNER_RIGHT_M = 10.0           # right turns cut the inner corner of the right lane: a tighter corner stays on asphalt
CORNER_LEFT_M = 15.0
CORNER_MIN_DEG = 10.0           # smaller direction changes at a junction are joined without a corner
SMOOTH_STEP_M = 1.0
SMOOTH_ITERATIONS = 200
SMOOTH_GAIN = 0.5
SMOOTH_MAX_SHIFT_M = 0.5        # the smoothed line stays within this of the lane line (the lane is 4 m wide)
STEP_M = 2.0                    # spacing of the route polyline
COORD_DECIMALS = 3
HEIGHT_DECIMALS = 2

ON_ROAD_MARGIN_M = 0.5
JUNCTION_ZONE_M = 20.0
JUNCTION_ALLOW_M = 1.0
CLEARANCE_M = 10.0
CURV_WINDOW_M = 5.0             # as C17's route curvature: heading change over +-5 m of arc length
MIN_RADIUS_M = 8.0
LENGTH_RANGE_M = (1000.0, 3000.0)
MIN_TURN_RAD = 0.2              # DD-5-min C17: below this much turning the L/R comparison has no signal
MIRROR_DEV_MIN = 0.35           # C17 fails a route whose left share differs by more than 0.25; keep a margin
MANEUVER_STRAIGHT_DEG = 30.0
SIDE_ROAD_M = 5.0
MANEUVER_MERGE_M = 10.0

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


class Road:
    def __init__(self, road_id, cls, generator, width, has_angles, points):
        self.road_id = road_id
        self.cls = cls
        self.generator = generator
        self.width = width
        self.has_angles = has_angles
        self.control = points                       # [(x, y, z)]
        self.drivable = cls in DRIVABLE_CLASSES and "/Decals/" not in generator
        self.line = catmull_rom(points, SPLINE_STEP_M) if len(points) >= 2 else list(points)
        self.s = arc_lengths(self.line)
        xs = [p[0] for p in self.line] or [0.0]
        zs = [p[2] for p in self.line] or [0.0]
        self.bounds = (min(xs), max(xs), min(zs), max(zs))

    def near(self, x, z, pad):
        """True when (x, z) lies within pad of the bounding box of the centreline."""
        x0, x1, z0, z1 = self.bounds
        return x0 - pad <= x <= x1 + pad and z0 - pad <= z <= z1 + pad


def read_roads(path):
    """{road_id: Road} of the decoder TSV."""
    roads = {}
    with open(path, "r", encoding="ascii", errors="replace", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        need = ("road_id", "class", "generator", "width_m", "has_angles", "points_xyz")
        missing = [name for name in need if name not in header]
        if missing:
            raise Refusal("%s lacks column(s) %s" % (path, ", ".join(missing)))
        ix = {name: i for i, name in enumerate(header)}
        for line in f:
            cells = line.rstrip("\r\n").split("\t")
            if len(cells) < len(header):
                continue
            points = [tuple(float(v) for v in p.split(",")) for p in cells[ix["points_xyz"]].split(";") if p]
            road = Road(int(cells[ix["road_id"]]), cells[ix["class"]], cells[ix["generator"]],
                        float(cells[ix["width_m"]]), cells[ix["has_angles"]].strip() not in ("", "0"), points)
            roads[road.road_id] = road
    return roads


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


# ---------------------------------------------------------------------------------------------------------------
# geometry (x, z horizontal; points carry y as a third value where needed)


def catmull_rom(points, step):
    """Centripetal Catmull-Rom spline through the control points (x, y, z), sampled about every step metres."""
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
    length = abs(s_to - s_from)
    pieces = max(1, int(math.ceil(length / SPLINE_STEP_M)))
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


def end_direction(line, at_start):
    """Unit direction of a lane line over its first (at_start) or last TANGENT_SPAN_M."""
    if at_start:
        a = line[0]
        b = next((p for p in line if math.hypot(p[0] - a[0], p[2] - a[2]) >= TANGENT_SPAN_M), line[-1])
    else:
        b = line[-1]
        a = next((p for p in reversed(line) if math.hypot(b[0] - p[0], b[2] - p[2]) >= TANGENT_SPAN_M), line[0])
    return unit(b[0] - a[0], b[2] - a[2])


def line_intersection(p, d, q, e):
    """Intersection of the lines p + t d and q + u e (horizontal); None when they are parallel."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p[0] + t * d[0], p[1] + t * d[1]


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


# ---------------------------------------------------------------------------------------------------------------
# the loop


class Junction:
    def __init__(self, road_a, road_b, s_a, s_b, x, z, gap):
        self.road_a = road_a        # incoming leg road
        self.road_b = road_b        # outgoing leg road
        self.s_a = s_a              # arc length of the junction on road a
        self.s_b = s_b              # ... on road b
        self.x = x
        self.z = z
        self.gap = gap              # distance between the two roads at the junction (m)
        self.turn = 0.0             # signed direction change (rad, right positive), set by build_lane_line


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
        raise Refusal("roads %d and %d do not meet (closest end %.1f m away, needs <= %.1f m)"
                      % (road_a.road_id, road_b.road_id, distance, JOIN_TOL_M))
    s_end, _, _ = project(end_road.line, end_road.s, point[0], point[2])
    if end_road is road_a:
        return Junction(road_a, road_b, s_end, s_other, point[0], point[2], distance)
    return Junction(road_a, road_b, s_other, s_end, point[0], point[2], distance)


def corner(line_in, line_out, turn):
    """A circular corner joining the end of line_in to the start of line_out: (trimmed line_in, corner points,
    trimmed line_out). Below CORNER_MIN_DEG the two lines are joined as they are."""
    if abs(math.degrees(turn)) < CORNER_MIN_DEG:
        return line_in, [], line_out
    d_in = end_direction(line_in, False)
    d_out = end_direction(line_out, True)
    apex = line_intersection((line_in[-1][0], line_in[-1][2]), d_in, (line_out[0][0], line_out[0][2]), d_out)
    if apex is None:
        return line_in, [], line_out
    radius = CORNER_RIGHT_M if turn > 0 else CORNER_LEFT_M
    tangent = radius * math.tan(abs(turn) / 2.0)
    kept_in = [p for p in line_in if (apex[0] - p[0]) * d_in[0] + (apex[1] - p[2]) * d_in[1] > tangent]
    kept_out = [p for p in line_out if (p[0] - apex[0]) * d_out[0] + (p[2] - apex[1]) * d_out[1] > tangent]
    if len(kept_in) < 2 or len(kept_out) < 2:
        raise Refusal("the %.0f m corner does not fit between the legs (tangent %.1f m)" % (radius, tangent))
    start = (apex[0] - tangent * d_in[0], apex[1] - tangent * d_in[1])
    side = right_of(d_in) if turn > 0 else (-right_of(d_in)[0], -right_of(d_in)[1])
    centre = (start[0] + radius * side[0], start[1] + radius * side[1])
    y = 0.5 * (kept_in[-1][1] + kept_out[0][1])
    a0 = math.atan2(start[0] - centre[0], start[1] - centre[1])
    sweep = turn                                    # clockwise positive, as the heading
    pieces = max(2, int(math.ceil(abs(sweep) * radius / SPLINE_STEP_M)))
    arc = []
    for k in range(pieces + 1):
        a = a0 + sweep * k / pieces
        arc.append((centre[0] + radius * math.sin(a), y, centre[1] + radius * math.cos(a)))
    return kept_in, arc, kept_out


def build_lane_line(roads):
    """(closed lane line starting at the route start, [Junction] of the leg junctions)."""
    legs = []
    for road_id in LEGS:
        road = roads.get(road_id)
        if road is None:
            raise Refusal("road %d is not in the polylines file" % road_id)
        if road.cls not in LEG_CLASSES or not road.drivable:
            raise Refusal("road %d is %s (%s), not a leg class %s" % (road_id, road.cls, road.generator, LEG_CLASSES))
        if road.has_angles:
            raise Refusal("road %d carries angles the decoder does not apply: its points are not trusted" % road_id)
        legs.append(road)
    junctions = [find_junction(legs[i], legs[(i + 1) % len(legs)]) for i in range(len(legs))]
    lanes = []
    for i, road in enumerate(legs):
        s_from = junctions[i - 1].s_b
        s_to = junctions[i].s_a
        lanes.append(offset_line(sub_line(road, s_from, s_to), LANE_WIDTH_FRACTION * road.width))
    for i, junction in enumerate(junctions):
        junction.turn = wrap(heading(end_direction(lanes[(i + 1) % len(lanes)], True))
                             - heading(end_direction(lanes[i], False)))
    # corners: junction i joins lane i to lane i + 1; lane 0 is cut at the route start afterwards
    pieces = [list(lane) for lane in lanes]
    arcs = []
    for i, junction in enumerate(junctions):
        j = (i + 1) % len(pieces)
        pieces[i], arc, pieces[j] = corner(pieces[i], pieces[j], junction.turn)
        arcs.append(arc)
    loop = []
    for i in range(len(pieces)):
        loop += pieces[i] + arcs[i]
    loop.append(loop[0])
    # rotate to the start: START_ALONG_M along the first lane line
    s = arc_lengths(loop)
    start = point_at(loop, s, START_ALONG_M)
    s_start, _, index = project(loop, s, start[0], start[2])
    rotated = [start] + loop[index + 1:-1] + loop[:index + 1] + [start]
    return rotated, junctions


def heights_from(points, roads):
    """y of the nearest leg centreline point for every point."""
    lines = [roads[road_id].line for road_id in LEGS]
    out = []
    for p in points:
        best = None
        for line in lines:
            for q in line:
                d = (q[0] - p[0]) ** 2 + (q[2] - p[2]) ** 2
                if best is None or d < best[0]:
                    best = (d, q[1])
        out.append((p[0], best[1], p[2]))
    return out


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


def road_pad(road):
    """How far outside a road's bounding box a point can still be on (or in a junction of) that road."""
    return road.width / 2.0 + JUNCTION_ALLOW_M


def segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


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


def maneuvers_of(points, junctions, roads):
    """Maneuvers (s_m, type, junction_id) of the leg junctions and of the side roads touching the loop."""
    s = arc_lengths(points)
    found = []
    for junction in junctions:
        if abs(math.degrees(junction.turn)) < MANEUVER_STRAIGHT_DEG:
            continue                                # a continuation of the road; a side road there is found below
        s_j, _, _ = project(points, s, junction.x, junction.z)
        kind = "right" if junction.turn > 0 else "left"
        found.append((s_j, kind, "j%d_%d" % (junction.road_a.road_id, junction.road_b.road_id), True))
    legs = [roads[road_id] for road_id in LEGS]
    for road in sorted(roads.values(), key=lambda r: r.road_id):
        if road.road_id in LEGS or not road.drivable or len(road.line) < 2:
            continue
        for end in (road.line[0], road.line[-1]):
            for leg in [r for r in legs if r.near(end[0], end[2], SIDE_ROAD_M)]:
                _, distance, _ = project(leg.line, leg.s, end[0], end[2])
                if distance <= SIDE_ROAD_M:
                    s_j, _, _ = project(points, s, end[0], end[2])
                    found.append((s_j, "straight", "j%d_%d" % (leg.road_id, road.road_id), False))
                    break
    found.sort()
    merged = []
    for item in found:
        if merged and item[0] - merged[-1][0] < MANEUVER_MERGE_M:
            if item[3] and not merged[-1][3]:
                merged[-1] = item
            continue
        merged.append(item)
    out = []
    for s_m, kind, junction_id, _ in merged:
        if 0.0 < s_m < s[-1]:
            out.append({"s_m": round(s_m, 3), "type": kind, "junction_id": junction_id})
    return out


def check_route(points, junctions, roads, train_rule, vehicles):
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
    drivable = [r for r in roads.values() if r.drivable and len(r.line) >= 2 and not r.has_angles]
    worst = (0.0, None)
    for p in points:
        best = None
        for road in [r for r in drivable if r.near(p[0], p[2], road_pad(r))]:
            _, distance, _ = project(road.line, road.s, p[0], p[2])
            allowed = road.width / 2.0 - ON_ROAD_MARGIN_M
            if any(math.hypot(p[0] - j.x, p[2] - j.z) <= JUNCTION_ZONE_M for j in junctions):
                allowed = road.width / 2.0 + JUNCTION_ALLOW_M
            excess = distance - allowed
            if best is None or excess < best[0]:
                best = (excess, road.road_id, distance)
        if best is None or best[0] > 0:
            raise Refusal("point (%.1f, %.1f) is off the drivable surface (%s)" % (
                p[0], p[2], "no drivable road nearby" if best is None else "%.2f m from road %d" % (best[2], best[1])))
        if worst[1] is None or best[0] > worst[0]:
            worst = (best[0], best[1])
    facts["on_road_worst_excess_m"] = worst[0]
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


def build_route(polylines_path, splits_path, layer_path):
    """(route dict, facts) or Refusal."""
    for path in (polylines_path, splits_path, layer_path):
        reason = sealed_reason(path)
        if reason:
            raise Refusal(reason)
        if not os.path.isfile(path):
            raise Refusal("missing input %s" % path)
    roads = read_roads(polylines_path)
    train_rule = read_train_rule(splits_path)
    vehicles = read_parked_vehicles(layer_path)
    lane, junctions = build_lane_line(roads)
    dense = resample_closed(lane, SMOOTH_STEP_M)
    smooth = smooth_closed(dense, dense[:-1])
    points = resample_closed(smooth, STEP_M)
    points = heights_from(points, roads)
    points[-1] = points[0]
    rounded = [[round(p[0], COORD_DECIMALS), round(p[1], HEIGHT_DECIMALS), round(p[2], COORD_DECIMALS)]
               for p in points]
    rounded[-1] = list(rounded[0])
    pts = [tuple(p) for p in rounded]
    facts = check_route(pts, junctions, roads, train_rule, vehicles)
    length = arc_lengths(pts)[-1]
    route = {
        "schema": drivedit_nav.ROUTE_SCHEMA_ID,
        "route_id": ROUTE_ID,
        "map_id": MAP_ID,
        "split": SPLIT,
        "nav_version": drivedit_nav.NAV_VERSION,
        "lane_side": "right",
        "closed": True,
        "polyline": rounded,
        "maneuvers": maneuvers_of(pts, junctions, roads),
        "stops": [{"s_m": round(length, 6), "kind": "route_end"}],
        "length_m": round(length, 6),
        "source": {
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "n_points": len(rounded),
            "legs": list(LEGS),
            "junctions": [{"roads": [j.road_a.road_id, j.road_b.road_id], "x": round(j.x, 2), "z": round(j.z, 2),
                           "gap_m": round(j.gap, 2), "turn_deg": round(math.degrees(j.turn), 1)} for j in junctions],
            "lane_offset_m": round(LANE_WIDTH_FRACTION * roads[LEGS[0]].width, 3),
            "start_along_m": START_ALONG_M,
            "corner_radius_m": {"right": CORNER_RIGHT_M, "left": CORNER_LEFT_M},
            "smoothing": {"step_m": SMOOTH_STEP_M, "max_shift_m": SMOOTH_MAX_SHIFT_M},
            "inputs": {
                "polylines": {"name": os.path.basename(polylines_path), "sha256": sha256_file(polylines_path)},
                "splits": {"name": os.path.basename(splits_path), "sha256": sha256_file(splits_path),
                           "status": (train_rule[2].split() or ["unknown"])[0]},
            },
            "parked_vehicles_checked": [[name, round(vx, 2), round(vz, 2)] for name, vx, vz in vehicles],
            "train_rule": {"val_l_box_xz": list(train_rule[0]), "buffer_m": train_rule[1]},
            "geometry": "decoder-derived (unofficial EBIN decode), unverified in the engine until E-08a/E-08b",
        },
        "description": "Hand-picked Arland T0 loop (DD-7-min): roads 452, 461, 93, 451 (asphalt E02/E03), right-lane "
                       "centreline, counter-clockwise, inside TRAIN of splits_v2 (PROPOSED, not frozen).",
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
    ap = argparse.ArgumentParser(description="Build the hand-picked Arland T0 loop (DD-7-min)")
    ap.add_argument("--polylines", required=True, help="decoder road polylines of Arland (TSV)")
    ap.add_argument("--splits", required=True, help="map splits JSON (splits_v2, PROPOSED)")
    ap.add_argument("--layer", default=DEFAULT_LAYER, help="DC_Arland layer with the parked vehicles")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--out", help="write the route file here")
    group.add_argument("--check", help="regenerate and compare with this route file")
    args = ap.parse_args(argv)
    try:
        route, facts = build_route(args.polylines, args.splits, args.layer)
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
    print("  turning: left %.2f rad, right %.2f rad, left share %.3f (mirror %.3f)" % (
        facts["left_turn_rad"], facts["right_turn_rad"], facts["left_share"], 1.0 - facts["left_share"]))
    print("  min turning radius %.1f m; worst on-road margin %.2f m; closest parked vehicle %s at %.1f m" % (
        1.0 / facts["max_abs_kappa"], -facts["on_road_worst_excess_m"], facts["closest_parked_vehicle"][0],
        facts["closest_parked_vehicle"][1]))
    print("  TRAIN of %s (status %s)" % (os.path.basename(args.splits), route["source"]["inputs"]["splits"]["status"]))
    for m in route["maneuvers"]:
        print("  maneuver s %.1f m %s (%s)" % (m["s_m"], m["type"], m["junction_id"]))
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
