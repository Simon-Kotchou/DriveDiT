"""nav_v1: the shared route-point function of capture contract dc-2.0 (DD-11-min).

DRAFT with dc-2.0 (docs/capture_contract/dc-2.0.md, status line): nav_v1 freezes with the contract. After the freeze
any change of behaviour (progress window, lookahead, maneuver window, stop clearance, route-deviation limit) or of
route_schema.json is nav_v2 with its own nav_version; nav_v1 output never changes in place.

    from drivedit_nav import load_route, nav_v1
    route = load_route("routes/arland_t0_loop.json")
    nav = nav_v1(route, (x, z, heading_rad))                       # first call of an episode: progress 0
    nav = nav_v1(route, (x, z, heading_rad), progress_m=nav["progress_m"])   # every later call

One function, three callers (CAPTURE_PROGRAM r2 3.3): the harvester fills the logged nav fields offline, the dataset
builder builds the `nav` tensor, and the closed-loop bridge computes it at test time. C22 cross-checks them on the
golden fixture, so the arithmetic must not depend on the caller: plain Python floats (IEEE double), no randomness, no
numpy, no dependence on dict order. The same route and the same pose sequence always give bit-identical output.

Conventions (docs/capture_contract/dc-2.0.md section 7):
  world    Enfusion metres, x east, y up, z north. nav works in the horizontal (x, z) plane; y is ignored.
  heading  psi = atan2(fwd_x, fwd_z): 0 = north (+z), positive clockwise seen from above (towards +x, east).
  ego      x forward, y right. Right is positive, as for steering, curvature and yaw rate in dc-2.0.
  route    s is the horizontal arc length along the route polyline from its first point. The polyline is the
           driving line (already offset to the lane); nav_v1 never offsets it.

Output channels (features_v1 tensor `nav`, in this order): p20_x, p20_y, p40_x, p40_y, man_left, man_straight,
man_right, man_follow, d_man, d_stop. Also returned, outside the tensor: p10_x, p10_y (ablation only), progress_m and
dist_to_route_m (eval-only bookkeeping), and nav_valid: 0 when dist_to_route_m exceeds the 30 m route-deviation limit
(r2 1.3), where the route points mean nothing (the builder then masks the step). A non-finite pose raises.

Stops: nav_v1 knows terminal stops only. Every route has exactly one stop, of kind route_end, at the route end. A
stop the ego must leave again needs a release every caller sees (a logged stop_release nav command); that, the
commanded stop kind and dwell times are nav_v2.

Ordered progress (r2 1.3): the pose is projected onto the route only inside a window of +-30 m along the route around
the previous progress, and progress never decreases. That is what keeps a figure-eight or a loop crossing on the
branch being driven. Callers carry progress_m from one call to the next; None means the episode start (0 m).
"""

import bisect
import hashlib
import json
import math
import os
import warnings
from collections import namedtuple

NAV_VERSION = "nav_v1"
ROUTE_SCHEMA_ID = "drivedit.route/1"
ROUTE_SCHEMA_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "route_schema.json")

D_MAX_M = 200.0                 # d_man and d_stop saturate here; 200 = nothing within range
LOOKAHEAD_M = (20.0, 40.0)      # p20, p40 (the tensor)
ABLATION_LOOKAHEAD_M = 10.0     # p10 (logged for ablations only)
PROGRESS_WINDOW_M = 30.0        # ordered-progress search window, each side of the previous progress
ROUTE_DEVIATION_M = 30.0        # r2 1.3: beyond this distance from the route the nav output is not valid
STOP_CLEAR_M = 2.0              # a stop stays pending until progress passes it by this much
MIN_SEGMENT_M = 0.01            # consecutive polyline points closer than this are rejected
CLOSED_TOL_M = 1.0              # first and last point of a closed route lie within this distance
LENGTH_TOL_M = 0.05             # a declared length_m must match the computed length within this

NAV_CHANNELS = ("p20_x", "p20_y", "p40_x", "p40_y", "man_left", "man_straight", "man_right", "man_follow",
                "d_man", "d_stop")
MANEUVER_TYPES = ("left", "straight", "right")
STOP_KINDS = ("route_end",)    # terminal stops only; commanded stops with a release are nav_v2
MARK_KEYS = {"maneuvers": ("s_m", "type", "junction_id"), "stops": ("s_m", "kind")}
LANE_SIDES = ("right", "center", "mixed")
MAP_SPLITS = {
    "arland": ("train", "val_l", "a0", "a_stress"),
    "kolguyev": ("k_val", "k_rep"),
    "everon": ("everon_harness", "everon_test"),
}
SEALED_MAP_IDS = ("everon",)
SEALED_PATH_PARTS = ("heldout", "sealed")
TOP_KEYS = ("schema", "route_id", "map_id", "split", "nav_version", "lane_side", "closed", "polyline", "maneuvers",
            "stops", "length_m", "source", "created", "description")
REQUIRED_KEYS = ("schema", "route_id", "map_id", "split", "nav_version", "lane_side", "closed", "polyline",
                 "maneuvers", "stops")

Pose = namedtuple("Pose", "x z heading_rad")


class RouteError(ValueError):
    pass


class SealedMapError(PermissionError):
    """An Everon route or a sealed path was opened outside the sealed evaluation path (L1, L7)."""


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _route_id_ok(v):
    if not isinstance(v, str) or not 1 <= len(v) <= 64 or not ("a" <= v[0] <= "z" or "0" <= v[0] <= "9"):
        return False
    return all("a" <= c <= "z" or "0" <= c <= "9" or c in "_.-" for c in v)


def validate_route(obj):
    """Return the list of problems with a route dict (empty when it is a valid drivedit.route/1 route).

    Checks everything route_schema.json states plus what JSON Schema cannot: map/split consistency, segment lengths,
    maneuver and stop positions inside [0, length], their order, the closed-route tolerance and a declared length.
    """
    errors = []
    if not isinstance(obj, dict):
        return ["route is not a JSON object"]
    for k in REQUIRED_KEYS:
        if k not in obj:
            errors.append("missing key %r" % k)
    for k in obj:
        if k not in TOP_KEYS:
            errors.append("unknown key %r" % k)
    if errors:
        return errors
    if obj["schema"] != ROUTE_SCHEMA_ID:
        errors.append("schema must be %r" % ROUTE_SCHEMA_ID)
    if not _route_id_ok(obj["route_id"]):
        errors.append("route_id must match [a-z0-9][a-z0-9_.-]{0,63}")
    if not isinstance(obj["map_id"], str) or obj["map_id"] not in MAP_SPLITS:
        errors.append("map_id must be one of %s (canonical ids, not terrain names)" % ", ".join(MAP_SPLITS))
    elif not isinstance(obj["split"], str) or obj["split"] not in MAP_SPLITS[obj["map_id"]]:
        errors.append("split %r is not a split of map %r (allowed: %s)"
                      % (obj["split"], obj["map_id"], ", ".join(MAP_SPLITS[obj["map_id"]])))
    if obj["nav_version"] != NAV_VERSION:
        errors.append("nav_version must be %r" % NAV_VERSION)
    if not isinstance(obj["lane_side"], str) or obj["lane_side"] not in LANE_SIDES:
        errors.append("lane_side must be one of %s" % ", ".join(LANE_SIDES))
    if not isinstance(obj["closed"], bool):
        errors.append("closed must be true or false")
    pts = obj["polyline"]
    if not isinstance(pts, list) or len(pts) < 2:
        return errors + ["polyline needs at least 2 points"]
    for i, p in enumerate(pts):
        if not isinstance(p, list) or len(p) != 3 or not all(_is_number(c) for c in p):
            return errors + ["polyline[%d] must be [x, y, z] with finite numbers" % i]
    length = 0.0
    for i in range(1, len(pts)):
        seg = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][2] - pts[i - 1][2])
        if seg < MIN_SEGMENT_M:
            errors.append("polyline[%d] repeats polyline[%d] (horizontal step %.4f m < %.2f m)"
                          % (i, i - 1, seg, MIN_SEGMENT_M))
        length += seg
    if obj["closed"] is True:
        gap = math.hypot(pts[-1][0] - pts[0][0], pts[-1][2] - pts[0][2])
        if gap > CLOSED_TOL_M:
            errors.append("closed route: first and last point are %.2f m apart (> %.1f m)" % (gap, CLOSED_TOL_M))
    if "length_m" in obj:
        if not _is_number(obj["length_m"]) or abs(obj["length_m"] - length) > LENGTH_TOL_M:
            errors.append("length_m %r differs from the polyline length %.3f m" % (obj["length_m"], length))
    errors += _check_marks(obj["maneuvers"], "maneuvers", "type", MANEUVER_TYPES, length, strict=True)
    errors += _check_marks(obj["stops"], "stops", "kind", STOP_KINDS, length, strict=False)
    if isinstance(obj["maneuvers"], list):
        for i, m in enumerate(obj["maneuvers"]):
            if isinstance(m, dict) and "junction_id" in m and not isinstance(m["junction_id"], str):
                errors.append("maneuvers[%d].junction_id must be a string" % i)
    if isinstance(obj["stops"], list):
        # nav_v1 has terminal stops only: exactly one route_end stop, at the route end
        ends = [st for st in obj["stops"] if isinstance(st, dict) and st.get("kind") == "route_end"]
        if len(obj["stops"]) != 1 or len(ends) != 1:
            errors.append("stops must hold exactly one stop, of kind route_end (nav_v1 has terminal stops only)")
        elif not _is_number(ends[0].get("s_m")) or abs(ends[0]["s_m"] - length) > LENGTH_TOL_M:
            errors.append("the route_end stop must lie at the route end (s_m %r, length %.3f m)"
                          % (ends[0].get("s_m"), length))
    for k in ("source",):
        if k in obj and not isinstance(obj[k], dict):
            errors.append("%s must be an object" % k)
    for k in ("created", "description"):
        if k in obj and not isinstance(obj[k], str):
            errors.append("%s must be a string" % k)
    return errors


def _check_marks(marks, name, kind_key, kinds, length, strict):
    errors = []
    if not isinstance(marks, list):
        return ["%s must be a list" % name]
    last = None
    for i, m in enumerate(marks):
        if not isinstance(m, dict) or "s_m" not in m or kind_key not in m:
            errors.append("%s[%d] needs s_m and %s" % (name, i, kind_key))
            continue
        extra = set(m) - set(MARK_KEYS[name])
        if extra:
            errors.append("%s[%d]: unknown keys %s" % (name, i, ", ".join(sorted(extra))))
        if not isinstance(m[kind_key], str) or m[kind_key] not in kinds:
            errors.append("%s[%d].%s must be one of %s" % (name, i, kind_key, ", ".join(kinds)))
        s = m["s_m"]
        if not _is_number(s) or s < 0 or s > length + LENGTH_TOL_M:
            errors.append("%s[%d].s_m must lie in [0, %.3f]" % (name, i, length))
            continue
        if last is not None and (s <= last if strict else s < last):
            errors.append("%s must be sorted by s_m%s" % (name, " with distinct positions" if strict else ""))
        last = s
    return errors


class Route(object):
    """A validated route: the horizontal polyline with its arc lengths, maneuvers and stops (treat as read-only)."""

    def __init__(self, obj):
        self.route_id = obj["route_id"]
        self.map_id = obj["map_id"]
        self.split = obj["split"]
        self.lane_side = obj["lane_side"]
        self.closed = obj["closed"]
        self.xz = tuple((float(p[0]), float(p[2])) for p in obj["polyline"])
        s = [0.0]
        for i in range(1, len(self.xz)):
            s.append(s[-1] + math.hypot(self.xz[i][0] - self.xz[i - 1][0], self.xz[i][1] - self.xz[i - 1][1]))
        self.s = tuple(s)
        self.length_m = s[-1]
        self.maneuvers = tuple((float(m["s_m"]), m["type"]) for m in obj["maneuvers"])
        self.stops = tuple((float(m["s_m"]), m["kind"]) for m in obj["stops"])
        self.sha256 = hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()

    def point_at(self, s):
        """(x, z) at arc length s, clamped to [0, length_m]."""
        if s <= 0.0:
            return self.xz[0]
        if s >= self.length_m:
            return self.xz[-1]
        i = bisect.bisect_right(self.s, s) - 1
        i = min(max(i, 0), len(self.xz) - 2)
        a, b = self.xz[i], self.xz[i + 1]
        t = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    def project(self, x, z, s_lo=0.0, s_hi=None):
        """Closest point of the route to (x, z) among arc lengths [s_lo, s_hi]: returns (s, distance).

        Segments are scanned in increasing s and a later one wins only if strictly closer, so ties go to the
        smaller s. Deterministic for a given input.
        """
        if s_hi is None:
            s_hi = self.length_m
        s_lo = max(0.0, s_lo)
        s_hi = min(self.length_m, s_hi)
        best_s, best_d = None, None
        i = max(bisect.bisect_right(self.s, s_lo) - 1, 0)
        while i < len(self.xz) - 1 and self.s[i] <= s_hi:
            a, b = self.xz[i], self.xz[i + 1]
            seg = self.s[i + 1] - self.s[i]
            t_lo = max(0.0, (s_lo - self.s[i]) / seg)
            t_hi = min(1.0, (s_hi - self.s[i]) / seg)
            if t_lo <= t_hi:
                dx, dz = b[0] - a[0], b[1] - a[1]
                t = ((x - a[0]) * dx + (z - a[1]) * dz) / (seg * seg)
                t = min(max(t, t_lo), t_hi)
                px, pz = a[0] + t * dx, a[1] + t * dz
                d = math.hypot(x - px, z - pz)
                if best_d is None or d < best_d:
                    best_s, best_d = self.s[i] + t * seg, d
            i += 1
        return best_s, best_d


def canonical_json(obj):
    """The route's canonical JSON text (sorted keys, no whitespace, ASCII): Route.sha256 hashes it."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sealed_path(path):
    parts = os.path.normpath(os.path.abspath(path)).replace("\\", "/").lower().split("/")
    return any(p in SEALED_PATH_PARTS for p in parts)


def route_from_dict(obj, sealed_eval=False):
    """Validate a route dict and build a Route. Everon routes need sealed_eval=True (the sealed evaluation path)."""
    errors = validate_route(obj)
    if errors:
        raise RouteError("invalid route %r: %s" % (obj.get("route_id") if isinstance(obj, dict) else None,
                                                   "; ".join(errors)))
    if (obj["map_id"] in SEALED_MAP_IDS or obj["split"].startswith("everon")) and not sealed_eval:
        raise SealedMapError("route %r is on the sealed test map %r; it is opened only on the sealed evaluation "
                             "path (sealed_eval=True)" % (obj["route_id"], obj["map_id"]))
    return Route(obj)


def load_route(path, sealed_eval=False):
    """Read and validate a route file. A path under heldout/ or a SEALED directory needs sealed_eval=True."""
    if _sealed_path(path) and not sealed_eval:
        raise SealedMapError("%s lies under a sealed directory; it is opened only on the sealed evaluation path "
                             "(sealed_eval=True)" % path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise RouteError("%s: not JSON (%s)" % (path, e))
    return route_from_dict(obj, sealed_eval=sealed_eval)


def pose_from_transform(pos, fwd):
    """Pose from an engine position [x, y, z] and forward axis [fx, fy, fz] (GetWorldTransform rows 3 and 2)."""
    return Pose(float(pos[0]), float(pos[2]), math.atan2(float(fwd[0]), float(fwd[2])))


def _as_pose(pose):
    """Pose of any accepted form; ValueError when x, z or the heading is not finite (NaN or inf)."""
    if isinstance(pose, Pose):
        p = pose
    elif isinstance(pose, dict):
        p = Pose(float(pose["x"]), float(pose["z"]), float(pose["heading_rad"]))
    else:
        x, z, h = pose
        p = Pose(float(x), float(z), float(h))
    if not (math.isfinite(p.x) and math.isfinite(p.z) and math.isfinite(p.heading_rad)):
        raise ValueError("pose %r is not finite" % (pose,))
    return p


def to_ego(pose, x, z):
    """World (x, z) -> ego (forward, right) for a Pose."""
    dx, dz = x - pose.x, z - pose.z
    sh, ch = math.sin(pose.heading_rad), math.cos(pose.heading_rad)
    return (dx * sh + dz * ch, dx * ch - dz * sh)


def nav_v1(route, pose, progress_m=None):
    """Compute the nav_v1 channels for one pose.

    route       a Route (route_from_dict / load_route)
    pose        Pose, (x, z, heading_rad) or {"x", "z", "heading_rad"}; heading as in the module docstring
    progress_m  the progress returned by the previous call of this episode; None = episode start (0 m)

    Returns a dict with NAV_CHANNELS plus p10_x, p10_y, progress_m, dist_to_route_m and nav_valid (0 when the pose is
    more than ROUTE_DEVIATION_M from its projection inside the progress window). ValueError for a non-finite pose.
    """
    p = _as_pose(pose)
    prev = 0.0 if progress_m is None else float(progress_m)
    if not (math.isfinite(prev) and 0.0 <= prev <= route.length_m):
        raise ValueError("progress_m %r outside [0, %.3f]" % (progress_m, route.length_m))
    s_proj, dist = route.project(p.x, p.z, prev - PROGRESS_WINDOW_M, prev + PROGRESS_WINDOW_M)
    s = max(prev, s_proj)
    out = {}
    for tag, ahead in (("p10", ABLATION_LOOKAHEAD_M), ("p20", LOOKAHEAD_M[0]), ("p40", LOOKAHEAD_M[1])):
        wx, wz = route.point_at(s + ahead)
        out[tag + "_x"], out[tag + "_y"] = to_ego(p, wx, wz)
    man = None
    for m_s, m_type in route.maneuvers:
        if m_s >= s:
            man = (m_s, m_type)
            break
    for t in MANEUVER_TYPES:
        out["man_" + t] = 0
    if man is not None and man[0] - s <= D_MAX_M:
        out["man_" + man[1]] = 1
        out["man_follow"] = 0
        out["d_man"] = man[0] - s
    else:
        out["man_follow"] = 1
        out["d_man"] = D_MAX_M
    d_stop = D_MAX_M
    for st_s, _ in route.stops:
        if s < st_s + STOP_CLEAR_M:
            d_stop = min(max(st_s - s, 0.0), D_MAX_M)
            break
    out["d_stop"] = d_stop
    out["progress_m"] = s
    out["dist_to_route_m"] = dist
    out["nav_valid"] = 1 if dist <= ROUTE_DEVIATION_M else 0
    return out


def nav_vector(nav):
    """The 10 tensor channels of a nav_v1 result, in NAV_CHANNELS order, as floats."""
    return tuple(float(nav[k]) for k in NAV_CHANNELS)


def nav_sequence(route, poses):
    """nav_v1 over one episode's poses in order, carrying progress from call to call (harvester and builder)."""
    out, progress = [], None
    for pose in poses:
        nav = nav_v1(route, pose, progress_m=progress)
        progress = nav["progress_m"]
        out.append(nav)
    return out


def to_legacy_6d(action_cmd, nav, speed_mps, yaw_rate_rps, steer_of_kappa):
    """Deprecated dc-1.0 view (CLAUDE.md 6D control): (steering, accel, goal_x, goal_y, speed, heading_rate).

    = [steer-equivalent of kappa_target, a_target, p20_x, p20_y, speed(t), yaw_rate(t)] (r2 3.3). action_cmd is
    (kappa_target, a_target, dir_cmd); steer_of_kappa(kappa, speed) is the vehicle card's inverse steering map in the
    contract sign. There is no default map: guessing one would silently mix units. The state fields are history at t.
    """
    warnings.warn("to_legacy_6d is deprecated: use the dc-2.0 tensors of features_v1", DeprecationWarning,
                  stacklevel=2)
    if steer_of_kappa is None:
        raise ValueError("to_legacy_6d needs the vehicle card's steer_of_kappa(kappa, speed)")
    kappa, a_target = float(action_cmd[0]), float(action_cmd[1])
    return (float(steer_of_kappa(kappa, float(speed_mps))), a_target, float(nav["p20_x"]), float(nav["p20_y"]),
            float(speed_mps), float(yaw_rate_rps))
