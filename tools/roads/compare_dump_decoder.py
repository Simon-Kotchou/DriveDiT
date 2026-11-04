#!/usr/bin/env python3
"""E-08a: compare the in-engine road dump (DC_RoadDump, DD-4-probe) with the decoder's road polylines.

Run it with the isolated interpreter (stdlib only):

    python3 -I -B tools/roads/compare_dump_decoder.py <e08a play_dir> \\
        --decoder /mnt/e/drivedit_realign/program/evidence/maps/Arland_road_polylines.tsv \\
        [--match-m 2.0] [--expect-km 46.4] [--tol 0.05] [--sample-m 1.0] [--list 20]

Inputs (read only): <play_dir>/run_header.txt, run_footer.txt, <map_id>_roads.csv, <map_id>_road_points.csv,
<map_id>_tiles.csv, <map_id>_roads_footer.txt (map_id from the run header, world.map_id), and the decoder TSV
(road_id, class, generator, width_m, n_pts, length_m, has_angles, points_xyz "x,y,z;x,y,z;...").

Sealed map: any input path containing "eden" or "everon" (any case, also after resolving links) is refused with
exit 2 before anything is read (Everon is the zero-shot test map).

Matching (2D, x and z; both polylines resampled at --sample-m): an engine road matches a decoder polyline when their
symmetric Hausdorff distance is <= --match-m, and then inherits its class. The engine and the decoder may split roads
at different places, so an engine road without such a partner still gets the class that covers most of its length
("contained" when all of it lies within --match-m of decoder polylines, "partial" when at least half does), else it
is "unmatched" (input for E-08b). Kilometres per class are summed per resampled piece of engine road (each piece takes
the class of the nearest decoder polyline within --match-m), so a different split does not change them.

Prints the r2 4.4 E-08a criterion ("count, km and width histogram vs decoder; unparented/decal-like widths flagged"):
  run_footer      run_footer.txt status complete (failed: FAIL; stopped or missing: INCOMPLETE)
  roads_footer    <map>_roads_footer.txt complete=1 (missing or complete=0: INCOMPLETE)
  no_duplicates   no two rows share a road_key and every dup_content_of is -1 (repeats of one BaseRoad object across
                  tiles are expected: n_tiles_seen, roads_dup_identity, reported only)
  no_cap          cap_confirmed 0 and every subdivided tile's sub-tile union <= its own result
  drivable_km     km of unique engine road on decoder drivable classes (dirt_road, trail, asphalt_E02_E03_secondary,
                  decal generators excluded) within --tol of --expect-km [design tolerance; r2 says "vs decoder"]
  INFO            count, km and width histogram (0-2, 2-4, 4-6, 6-8, 8-10, 10-20, >20 m) per class for the engine and
                  the decoder; flagged roads (class unparented_road_entity, decal_or_other or runway_concrete_panel,
                  or width < 2 m or > 20 m); unmatched engine roads; decoder drivable km covered by the engine dump.

Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error (and every sealed-map path).
"""

import argparse
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools", "probes", "t0eng"))
import probe_common  # noqa: E402  (shared reader and verdict printer, PROBE_DESIGN.md 4.7)

SEALED_RE = re.compile(r"eden|everon", re.IGNORECASE)
DRIVABLE_CLASSES = ("dirt_road", "trail", "asphalt_E02_E03_secondary")
FLAGGED_CLASSES = ("unparented_road_entity", "decal_or_other", "runway_concrete_panel")
FLAG_MIN_WIDTH_M = 2.0
FLAG_MAX_WIDTH_M = 20.0
WIDTH_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 20.0), (20.0, None))
UNMATCHED = "unmatched"
GRID_CELL_M = 16.0
ROADS_COLUMNS = ("road_idx", "road_key", "width_m", "n_pts", "length_m", "n_tiles_seen", "dup_content_of")
POINT_COLUMNS = ("road_idx", "pt_idx", "x", "z")
TILE_COLUMNS = ("tile_ix", "tile_iz", "n_returned", "subdivided", "n_sub_union")


class Polyline:
    def __init__(self, ident, cls, width, points, generator="", length_m=None):
        self.ident = ident
        self.cls = cls
        self.width = width
        self.points = points              # [(x, z)]
        self.generator = generator
        self.length_m = polyline_length(points) if length_m is None else length_m
        self.drivable = cls in DRIVABLE_CLASSES and "/Decals/" not in generator


def sealed_reason(path):
    for candidate in (path, os.path.realpath(path)):
        if SEALED_RE.search(candidate):
            return "refusing %s: it names the sealed map (eden / everon); Everon is the zero-shot test map" % path
    return None


def polyline_length(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def resample(points, step):
    """Points every <= step metres along the polyline (vertices kept) and the length each point stands for."""
    if not points:
        return [], []
    if len(points) == 1:
        return [points[0]], [0.0]
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        pieces = max(1, int(math.ceil(length / step)))
        for i in range(1, pieces + 1):
            t = i / pieces
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    weights = []
    for i in range(len(out)):
        left = math.hypot(out[i][0] - out[i - 1][0], out[i][1] - out[i - 1][1]) if i > 0 else 0.0
        right = math.hypot(out[i + 1][0] - out[i][0], out[i + 1][1] - out[i][1]) if i + 1 < len(out) else 0.0
        weights.append(0.5 * (left + right))
    return out, weights


def point_segment_distance(p, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dz * dz
    if length2 <= 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dz))


class SegmentGrid:
    """Uniform grid over the segments of some polylines: near(p) returns {polyline index: distance} for every
    polyline with a segment within pad of p."""

    def __init__(self, polylines, pad, cell=GRID_CELL_M):
        self.cell = cell
        self.pad = pad
        self.polylines = polylines
        self.cells = {}
        step = cell / 4.0
        reach = pad + step
        for index, poly in enumerate(polylines):
            for seg, (a, b) in enumerate(zip(poly.points, poly.points[1:])):
                keys = set()
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                count = max(1, int(math.ceil(length / step)))
                for i in range(count + 1):
                    t = i / count
                    x, z = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                    for cx in range(int(math.floor((x - reach) / cell)), int(math.floor((x + reach) / cell)) + 1):
                        for cz in range(int(math.floor((z - reach) / cell)), int(math.floor((z + reach) / cell)) + 1):
                            keys.add((cx, cz))
                for key in keys:
                    self.cells.setdefault(key, []).append((index, seg))

    def near(self, p):
        found = {}
        for index, seg in self.cells.get((int(math.floor(p[0] / self.cell)), int(math.floor(p[1] / self.cell))), ()):
            pts = self.polylines[index].points
            d = point_segment_distance(p, pts[seg], pts[seg + 1])
            if d <= self.pad and d < found.get(index, float("inf")):
                found[index] = d
        return found


def directed_hausdorff(samples, grid, index):
    """max over samples of the distance to polyline `index` of grid, or None when one sample is farther than pad."""
    worst = 0.0
    for p in samples:
        d = grid.near(p).get(index)
        if d is None:
            return None
        worst = max(worst, d)
    return worst


def load_decoder(path):
    csv.field_size_limit(1 << 30)
    polylines = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        missing = [c for c in ("road_id", "class", "width_m", "length_m", "points_xyz") if c not in columns]
        if missing:
            raise probe_common.InputError("%s lacks the columns %s" % (path, ", ".join(missing)))
        for row in reader:
            points = []
            for triple in (row["points_xyz"] or "").split(";"):
                if triple.strip():
                    x, _, z = (float(v) for v in triple.split(","))
                    points.append((x, z))
            polylines.append(Polyline(row["road_id"], row["class"], float(row["width_m"] or 0), points,
                                      row.get("generator") or "", float(row["length_m"] or 0)))
    return polylines


def load_engine(play, map_id):
    """(roads rows, {road_idx: [(x, z)]}) of the dump; InputError when the CSVs are missing."""
    roads = play.csv(map_id + "_roads", required=ROADS_COLUMNS)
    points = {}
    for row in play.csv(map_id + "_road_points", required=POINT_COLUMNS):
        points.setdefault(int(row["road_idx"]), []).append((int(row["pt_idx"]), float(row["x"]), float(row["z"])))
    shapes = {idx: [(x, z) for _, x, z in sorted(pts)] for idx, pts in points.items()}
    return roads, shapes


def width_histogram(items):
    """items: [(width, km)] -> "0-2:n/km ..." text."""
    parts = []
    for low, high in WIDTH_BINS:
        inside = [km for w, km in items if w >= low and (high is None or w < high)]
        label = "%g-%g" % (low, high) if high is not None else ">%g" % low
        parts.append("%s:%d/%.2fkm" % (label, len(inside), sum(inside)))
    return " ".join(parts)


def classify(engine, decoder, match_m, sample_m):
    """Per engine road: (class, kind, hausdorff) and per class the km of engine road pieces."""
    grid = SegmentGrid(decoder, match_m)
    per_road = []
    class_km = {}
    for road in engine:
        samples, weights = resample(road.points, sample_m)
        candidates = None
        worst = {}
        class_weight = {}
        covered = 0.0
        for p, w in zip(samples, weights):
            near = grid.near(p)
            if near:
                best = min(near, key=near.get)
                cls = decoder[best].cls
                if decoder[best].cls in DRIVABLE_CLASSES and not decoder[best].drivable:
                    cls = "decal_or_other"
                class_weight[cls] = class_weight.get(cls, 0.0) + w
                covered += w
            else:
                class_weight[UNMATCHED] = class_weight.get(UNMATCHED, 0.0) + w
            keys = set(near)
            candidates = keys if candidates is None else candidates & keys
            for key in candidates:
                worst[key] = max(worst.get(key, 0.0), near[key])
        for cls, metres in class_weight.items():
            class_km[cls] = class_km.get(cls, 0.0) + metres / 1000.0
        match = None
        if candidates:
            road_grid = SegmentGrid([road], match_m)
            for key in sorted(candidates):
                reverse = directed_hausdorff(resample(decoder[key].points, sample_m)[0], road_grid, 0)
                if reverse is not None:
                    h = max(worst[key], reverse)
                    if match is None or h < match[1]:
                        match = (key, h)
        total = sum(weights)
        if match is not None:
            cls = decoder[match[0]].cls
            if cls in DRIVABLE_CLASSES and not decoder[match[0]].drivable:
                cls = "decal_or_other"
            per_road.append((cls, "hausdorff", match[1]))
        elif total > 0 and covered >= 0.5 * total:
            matched = {c: m for c, m in class_weight.items() if c != UNMATCHED}
            cls = max(matched, key=matched.get)
            per_road.append((cls, "contained" if covered >= total - 1e-9 else "partial", None))
        else:
            per_road.append((UNMATCHED, UNMATCHED, None))
    return per_road, class_km


def decoder_coverage(decoder, engine, match_m, sample_m):
    """km of drivable decoder polylines within match_m of any engine road, and their total km."""
    if not engine:
        return 0.0, sum(d.length_m for d in decoder if d.drivable) / 1000.0
    grid = SegmentGrid(engine, match_m)
    covered = total = 0.0
    for poly in decoder:
        if not poly.drivable:
            continue
        samples, weights = resample(poly.points, sample_m)
        for p, w in zip(samples, weights):
            total += w
            if grid.near(p):
                covered += w
    return covered / 1000.0, total / 1000.0


def analyse(play_dir, decoder_path, match_m, expect_km, tol, sample_m, list_max):
    play = probe_common.Play(play_dir)
    map_id = play.header.get("world.map_id") or "arland"
    verdict = probe_common.Verdict("E-08a road dump vs decoder", play.dir)
    decoder = load_decoder(decoder_path)

    verdict.check_footer(play)
    footer_path = play.path(map_id + "_roads_footer.txt")
    roads_footer = probe_common.read_kv(footer_path) if os.path.isfile(footer_path) else None
    if roads_footer is None:
        verdict.check("roads_footer", None, "no %s_roads_footer.txt (the dump did not finish)" % map_id)
    elif roads_footer.get("complete") != "1":
        verdict.check("roads_footer", None, "%s_roads_footer.txt complete=%s after %s of %s tiles" % (
            map_id, roads_footer.get("complete"), roads_footer.get("tiles_done", "?"), roads_footer.get("tiles", "?")))
    else:
        verdict.check("roads_footer", True, "complete=1 (%s tiles of %s m, %s roads written, %s points, %s km)" % (
            roads_footer.get("tiles"), roads_footer.get("tile_m"), roads_footer.get("roads_written"),
            roads_footer.get("points_total"), roads_footer.get("km_total")))

    roads, shapes = load_engine(play, map_id)
    keys = {}
    for row in roads:
        keys.setdefault(row["road_key"], []).append(row["road_idx"])
    shared = {k: v for k, v in keys.items() if len(v) > 1}
    content_dups = [r["road_idx"] for r in roads if int(r["dup_content_of"]) != -1]
    repeats = sum(max(0, int(r["n_tiles_seen"]) - 1) for r in roads)
    verdict.check("no_duplicates", not shared and not content_dups,
                  "%d road_key values shared by several rows%s, %d rows with dup_content_of != -1; %d repeats of one "
                  "object across tiles (expected, written once)" % (
                      len(shared), (" (" + "; ".join("%s: rows %s" % (k, ",".join(v)) for k, v in
                                                    list(shared.items())[:5]) + ")") if shared else "",
                      len(content_dups), repeats))

    tiles_path = play.path(map_id + "_tiles.csv")
    if not os.path.isfile(tiles_path) or roads_footer is None:
        verdict.check("no_cap", None, "no %s_tiles.csv or no roads footer" % map_id)
    else:
        tiles = play.csv(map_id + "_tiles", required=TILE_COLUMNS)
        over = [t for t in tiles if t["subdivided"] == "1" and t["n_sub_union"] != ""
                and int(t["n_sub_union"]) > int(t["n_returned"])]
        suspects = sum(1 for t in tiles if t["subdivided"] == "1")
        confirmed = int(roads_footer.get("cap_confirmed", "0") or 0)
        verdict.check("no_cap", confirmed == 0 and not over,
                      "cap_confirmed %d, %d suspect tiles (>= %s roads), %d with a larger sub-tile union, max %s roads "
                      "per tile" % (confirmed, suspects, roads_footer.get("cap_suspect", "?"), len(over),
                                    roads_footer.get("max_returned_per_tile", "?")))

    unique = [r for r in roads if int(r["dup_content_of"]) == -1]
    engine = [Polyline(r["road_idx"], "", float(r["width_m"] or 0), shapes.get(int(r["road_idx"]), []))
              for r in unique]
    per_road, class_km = classify(engine, decoder, match_m, sample_m)
    drivable_km = sum(class_km.get(c, 0.0) for c in DRIVABLE_CLASSES)
    detail = "%.2f km of engine road on decoder drivable classes (%s), expected %.2f +-%.0f%%" % (
        drivable_km, ", ".join(DRIVABLE_CLASSES), expect_km, 100 * tol)
    if roads_footer is None or roads_footer.get("complete") != "1":
        verdict.check("drivable_km", None, detail + " (the dump did not finish: not judged)")
    else:
        verdict.check("drivable_km", abs(drivable_km - expect_km) <= tol * expect_km, detail)

    kinds = {}
    for cls, kind, _ in per_road:
        kinds[kind] = kinds.get(kind, 0) + 1
    engine_km = sum(p.length_m for p in engine) / 1000.0
    verdict.info("engine", "%d rows, %d unique roads, %.2f km (2D), %d points; matched hausdorff %d, contained %d, "
                 "partial %d, unmatched %d (--match-m %.1f)" % (
                     len(roads), len(unique), engine_km, sum(len(p.points) for p in engine), kinds.get("hausdorff", 0),
                     kinds.get("contained", 0), kinds.get("partial", 0), kinds.get(UNMATCHED, 0), match_m))
    decoder_drivable = sum(d.length_m for d in decoder if d.drivable) / 1000.0
    decoder_drivable_2d = sum(polyline_length(d.points) for d in decoder if d.drivable) / 1000.0
    verdict.info("decoder", "%d polylines, %.2f km; drivable %.2f km by its length_m column, %.2f km 2D (decal "
                 "generators excluded)" % (len(decoder), sum(d.length_m for d in decoder) / 1000.0, decoder_drivable,
                                           decoder_drivable_2d))
    classes = sorted({c for c, _, _ in per_road} | {d.cls for d in decoder} | set(class_km))
    for cls in classes:
        roads_of = [(p.width, p.length_m / 1000.0) for p, (c, _, _) in zip(engine, per_road) if c == cls]
        dec_of = [(d.width, d.length_m / 1000.0) for d in decoder if d.cls == cls]
        verdict.info("class %s" % cls, "engine n=%d km=%.2f [%s] | decoder n=%d km=%.2f [%s]" % (
            len(roads_of), class_km.get(cls, 0.0), width_histogram(roads_of), len(dec_of),
            sum(km for _, km in dec_of), width_histogram(dec_of)))

    flagged = [(p, c) for p, (c, _, _) in zip(engine, per_road)
               if c in FLAGGED_CLASSES or p.width < FLAG_MIN_WIDTH_M or p.width > FLAG_MAX_WIDTH_M]
    verdict.info("flagged", "%d roads, %.2f km (class %s, or width < %.0f m or > %.0f m)%s" % (
        len(flagged), sum(p.length_m for p, _ in flagged) / 1000.0, "/".join(FLAGGED_CLASSES), FLAG_MIN_WIDTH_M,
        FLAG_MAX_WIDTH_M, describe([p for p, _ in flagged], list_max, [c for _, c in flagged])))
    unmatched = [p for p, (c, _, _) in zip(engine, per_road) if c == UNMATCHED]
    verdict.info("unmatched", "%d engine roads, %.2f km with no decoder polyline within %.1f m (input for E-08b)%s" % (
        len(unmatched), sum(p.length_m for p in unmatched) / 1000.0, match_m, describe(unmatched, list_max)))
    covered, total = decoder_coverage(decoder, engine, match_m, sample_m)
    verdict.info("decoder_coverage", "%.2f of %.2f km of drivable decoder road lie within %.1f m of an engine road" % (
        covered, total, match_m))
    return verdict


def describe(polylines, list_max, classes=None):
    if not polylines or list_max <= 0:
        return ""
    parts = []
    for i, p in enumerate(polylines[:list_max]):
        first = p.points[0] if p.points else (float("nan"), float("nan"))
        cls = (" " + classes[i]) if classes else ""
        parts.append("road %s%s w=%.2f len=%.1f at %.1f,%.1f" % (p.ident, cls, p.width, p.length_m, first[0], first[1]))
    more = " ..." if len(polylines) > list_max else ""
    return ": " + "; ".join(parts) + more


def main(argv=None):
    parser = argparse.ArgumentParser(description="E-08a: engine road dump vs decoder polylines (DD-4-probe)")
    parser.add_argument("play_dir", help="the e08a play_NN directory")
    parser.add_argument("--decoder", required=True, help="decoder <Map>_road_polylines.tsv")
    parser.add_argument("--match-m", type=float, default=2.0, help="symmetric Hausdorff match distance [m]")
    parser.add_argument("--expect-km", type=float, default=46.4, help="expected drivable km (Arland: ~46.4)")
    parser.add_argument("--tol", type=float, default=0.05, help="relative km tolerance (default 0.05)")
    parser.add_argument("--sample-m", type=float, default=1.0, help="resampling step [m]")
    parser.add_argument("--list", type=int, default=20, help="roads listed per INFO line")
    args = parser.parse_args(argv)
    for path in (args.play_dir, args.decoder):
        reason = sealed_reason(path)
        if reason:
            print("ERROR: " + reason, file=sys.stderr)
            return 2
    if args.match_m <= 0 or args.sample_m <= 0 or args.tol < 0:
        print("ERROR: --match-m and --sample-m must be positive, --tol >= 0", file=sys.stderr)
        return 2
    try:
        if not os.path.isfile(args.decoder):
            raise probe_common.InputError("no decoder file %s" % args.decoder)
        verdict = analyse(args.play_dir, args.decoder, args.match_m, args.expect_km, args.tol, args.sample_m,
                          args.list)
    except probe_common.InputError as exc:
        print("input error: %s" % exc)
        return probe_common.EXIT_INPUT_ERROR
    return verdict.report()


if __name__ == "__main__":
    sys.exit(main())
