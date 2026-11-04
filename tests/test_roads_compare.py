#!/usr/bin/env python3
"""Synthetic-fixture tests for tools/roads/compare_dump_decoder.py (E-08a, DD-4-probe).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_roads_compare.py

Each case writes a small decoder TSV and an e08a play directory (run header and footer, arland_roads.csv,
arland_road_points.csv, arland_tiles.csv, arland_roads_footer.txt; probe_common.write_synthetic_play) into a fresh
probe_common.make_temp_dir() directory
(RAM or E:, never /tmp) and removes it afterwards:

  - 3 engine roads on the 3 drivable decoder polylines (offset, resampled, one reversed) -> PASS;
  - the same with one decoder road split into two engine roads -> still PASS (contained), same km;
  - one content duplicate row -> FAIL; a capped tile -> FAIL;
  - no arland_roads_footer.txt -> INCOMPLETE;
  - a decoder or play path naming Eden / Everon -> exit 2 before anything is read.
"""

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "probes", "t0eng"))
import probe_common  # noqa: E402


def load_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_tool("compare_dump_decoder", os.path.join(REPO, "tools", "roads", "compare_dump_decoder.py"))

# Decoder polylines (x, z): three drivable roads, a flower-bed decal and an asphalt patch-stripe decal
DECODER = [
    ("0", "dirt_road", "Prefabs/WEGenerators/Roads/Dirt/RG_Road_Dirt_01.et", 4.0,
     [(100.0, 100.0 + 50.0 * i) for i in range(11)]),
    ("1", "asphalt_E02_E03_secondary", "Prefabs/WEGenerators/Roads/Asphalt/RG_Road_Asphalt_E_02.et", 8.0,
     [(200.0, 100.0), (200.0, 400.0), (500.0, 400.0)]),
    ("2", "trail", "Prefabs/WEGenerators/Roads/Dirt/RG_TrailDirt_01.et", 1.75, [(700.0, 100.0), (700.0, 400.0)]),
    ("3", "decal_or_other", "Prefabs/WEGenerators/Decals/FlowerBed/RG_FlowerBed_01_TEMP.et", 4.0,
     [(900.0, 900.0), (950.0, 900.0)]),
    ("4", "asphalt_E02_E03_secondary", "Prefabs/WEGenerators/Decals/Asphalt/RG_Asphalt_Patch_Stripe_01.et", 1.6,
     [(1200.0, 100.0), (1220.0, 100.0)]),
]
DRIVABLE_KM = 1.4


def dense(points, step):
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, int(round(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / step)))
        out.extend((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(1, n + 1))
    return out


def engine_roads(split=False):
    """(width, [(x, z)]) of the engine dump: road 0 offset 0.5 m with 10 m points, road 1 reversed (split in two when
    split), road 2 offset 1 m."""
    roads = [(4.0, [(x + 0.5, z) for x, z in dense(DECODER[0][4], 10.0)])]
    asphalt = list(reversed(dense(DECODER[1][4], 25.0)))
    if split:
        middle = len(asphalt) // 2
        roads.append((8.0, asphalt[:middle + 1]))
        roads.append((8.0, asphalt[middle:]))
    else:
        roads.append((8.0, asphalt))
    roads.append((1.75, [(x - 1.0, z) for x, z in dense(DECODER[2][4], 20.0)]))
    return roads


def write_text(path, lines):
    real = os.path.realpath(path)
    if not (real.startswith("/dev/shm/") or real.startswith("/mnt/e/")):
        raise AssertionError("refusing to write a fixture outside /dev/shm/ and /mnt/e/: %s" % real)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


def write_decoder(path):
    lines = ["road_id\tclass\tgenerator\twidth_m\tn_pts\tlength_m\thas_angles\tpoints_xyz"]
    for ident, cls, generator, width, points in DECODER:
        length = compare.polyline_length(points)
        xyz = ";".join("%.1f,39.0,%.1f" % p for p in points)
        lines.append("%s\t%s\t%s\t%.2f\t%d\t%.1f\t0\t%s" % (ident, cls, generator, width, len(points), length, xyz))
    write_text(path, lines)


ROAD_COLUMNS = ["road_idx", "road_key", "width_m", "n_pts", "length_m", "x0", "y0", "z0", "x1", "y1", "z1",
                "first_tile_ix", "first_tile_iz", "n_tiles_seen", "dup_content_of"]
POINT_COLUMNS = ["road_idx", "pt_idx", "x", "y", "z"]
TILE_COLUMNS = ["tile_ix", "tile_iz", "min_x", "min_z", "max_x", "max_z", "ret_value", "n_returned", "n_new",
                "n_dup_identity", "n_dup_content", "subdivided", "n_sub_union"]


def write_dump(play_dir, roads, duplicate=False, capped=False, footer=True):
    """An e08a play as DC_RoadDump writes it (probe_common.write_synthetic_play)."""
    rows, points, keys = [], [], []
    if duplicate:
        roads = roads + [roads[0]]
    for idx, (width, pts) in enumerate(roads):
        key = "%d/%.2f:%.2f/%.2f:%.2f" % (len(pts), pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
        dup_of = keys.index(key) if key in keys else -1
        keys.append(key)
        rows.append([idx, key, "%.3f" % width, len(pts), "%.3f" % compare.polyline_length(pts), "%.3f" % pts[0][0],
                     "40.000", "%.3f" % pts[0][1], "%.3f" % pts[-1][0], "40.000", "%.3f" % pts[-1][1], 0, 0, 2, dup_of])
        points.extend([idx, i, "%.3f" % x, "40.000", "%.3f" % z] for i, (x, z) in enumerate(pts))
    tiles = [[0, 0, "0.000", "0.000", "256.000", "256.000", 2, 2, 2, 0, 0, 0, None],
             [1, 0, "256.000", "0.000", "512.000", "256.000", 64, 64, 1, 0, 0, 1, 65 if capped else 64]]
    files = {}
    if footer:
        lines = ["map_id=arland", "tile_m=256.000", "tiles=2", "tiles_done=2", "roads_written=%d" % len(roads),
                 "points_total=%d" % len(points), "km_total=1.400", "max_returned_per_tile=64", "cap_suspect=64",
                 "cap_suspect_tiles=1", "cap_confirmed=%d" % (1 if capped else 0), "complete=1"]
        files["arland_roads_footer.txt"] = ("\n".join(lines) + "\n").encode("ascii")
    probe_common.write_synthetic_play(
        play_dir, {"schema": "dc-probe-run/1", "probe": "e08a", "world.map_id": "arland", "params.m_fTileM": "256.000"},
        {"arland_roads": (ROAD_COLUMNS, rows), "arland_road_points": (POINT_COLUMNS, points),
         "arland_tiles": (TILE_COLUMNS, tiles)},
        {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}, files)


def run_tool(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = compare.main(argv)
    return code, out.getvalue() + err.getvalue()


class RoadsCompareTests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_roads_")
        self.addCleanup(shutil.rmtree, self.base, True)
        self.decoder = os.path.join(self.base, "Arland_road_polylines.tsv")
        write_decoder(self.decoder)

    def run_case(self, name, **kwargs):
        play = os.path.join(self.base, name)
        write_dump(play, engine_roads(kwargs.pop("split", False)), **kwargs)
        return run_tool([play, "--decoder", self.decoder, "--expect-km", str(DRIVABLE_KM)])

    def test_matching_dump_passes(self):
        code, text = self.run_case("ok")
        self.assertEqual(code, 0, text)
        self.assertIn("[PASS] drivable_km: 1.40 km", text)
        self.assertIn("matched hausdorff 3", text)
        self.assertIn("class trail: engine n=1", text)
        self.assertIn("unmatched: 0 engine roads", text)
        self.assertIn("1.40 of 1.40 km of drivable decoder road", text)
        self.assertIn("flagged: 1 roads", text)        # the 1.75 m trail is narrower than 2 m

    def test_split_roads_still_pass(self):
        code, text = self.run_case("split", split=True)
        self.assertEqual(code, 0, text)
        self.assertIn("matched hausdorff 2, contained 2", text)
        self.assertIn("[PASS] drivable_km: 1.40 km", text)

    def test_duplicate_fails(self):
        code, text = self.run_case("dup", duplicate=True)
        self.assertEqual(code, 1, text)
        self.assertIn("[FAIL] no_duplicates", text)

    def test_capped_tile_fails(self):
        code, text = self.run_case("capped", capped=True)
        self.assertEqual(code, 1, text)
        self.assertIn("[FAIL] no_cap", text)

    def test_missing_footer_is_incomplete(self):
        code, text = self.run_case("nofooter", footer=False)
        self.assertEqual(code, 3, text)
        self.assertIn("[INCOMPLETE] roads_footer", text)

    def test_sealed_map_paths_are_refused(self):
        eden = os.path.join(self.base, "Eden_road_polylines.tsv")
        shutil.copy(self.decoder, eden)
        play = os.path.join(self.base, "ok")
        write_dump(play, engine_roads())
        code, text = run_tool([play, "--decoder", eden])
        self.assertEqual(code, 2, text)
        self.assertIn("sealed map", text)
        code, text = run_tool([os.path.join(self.base, "EVERON_play"), "--decoder", self.decoder])
        self.assertEqual(code, 2, text)

    def test_missing_dump_is_input_error(self):
        play = os.path.join(self.base, "empty")
        probe_common.write_synthetic_play(play, {"world.map_id": "arland"}, {})
        code, _ = run_tool([play, "--decoder", self.decoder])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
