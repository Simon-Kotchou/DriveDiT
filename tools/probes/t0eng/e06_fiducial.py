#!/usr/bin/env python3
"""E-06 analysis: fiducial v2 decode over the e05 screenshots (r2 4.4 T0-ENG core; DD-4-probe).

    python3 -I -B tools/probes/t0eng/e06_fiducial.py <e05 play_dir> [--min-frames 600] [--header-geometry]

Decodes every screenshot the engine wrote for an issued request of the play (data/dc2/fiducial.py, loaded by file
path) and prints the r2 4.4 criterion of E-06:
  * decode_ge_99_5pct   decoded (both bands decode and agree) / files present >= 99.5%, over >= --min-frames
                        files (fewer: INCOMPLETE)
  * no_negative_offset  offset = unwrap(code, issue_render_frame) - issue_render_frame is never negative (an image
                        cannot show a frame before the request)
and as INFO the offset p50 / p99 (frames from request to capture) and the per-status counts.

The patch origins come from the image's own size (data/dc2/fiducial.layout). --header-geometry uses the
fiducial.left_x, fiducial.right_x and fiducial.y0..y5 keys of run_header.txt instead (for a screenshot whose size
differs from the workspace). The same play layout works for an e06a play (e06a_requests.csv).
Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error. The script only reads.
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402  (shared reader and verdict printer)
import e05_images  # noqa: E402  (FrameFiles: request path -> engine-written file)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIDUCIAL_PATH = os.path.join(REPO, "data", "dc2", "fiducial.py")
LABEL = "E-06 fiducial v2 decode"
PROBES = ("e05", "e06a")
REQUEST_COLUMNS = ("slot_id", "issue_status", "issue_render_frame", "path", "found_suffix")
DECODE_MIN_PCT = 99.5
DEFAULT_MIN_FRAMES = 600


def load_fiducial():
    """data/dc2/fiducial.py by file path (importing the data package would load torch)."""
    spec = importlib.util.spec_from_file_location("dc2_fiducial", FIDUCIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def header_int(header, key, default=None):
    value = header.get(key, "")
    if value == "":
        if default is None:
            raise probe_common.InputError("run_header.txt has no %s" % key)
        return default
    try:
        return int(float(value))
    except ValueError:
        raise probe_common.InputError("run_header.txt %s=%r is not a number" % (key, value))


def geometry_from_header(header, fiducial):
    """(left origins, right origins, patch px) from the fiducial.* keys appended by DC_Fiducial.AppendHeaderLines."""
    patch = header_int(header, "fiducial.patch_px")
    left_x = header_int(header, "fiducial.left_x")
    right_x = header_int(header, "fiducial.right_x")
    ys = [header_int(header, "fiducial.y%d" % i) for i in range(fiducial.PATCH_COUNT)]
    return [(left_x, y) for y in ys], [(right_x, y) for y in ys], patch


def decode_file(fiducial, path, geometry, header):
    """FiducialResult of one file; status "unreadable" when the image cannot be loaded."""
    try:
        image = fiducial.load_rgb(path)
    except ValueError:
        return fiducial.FiducialResult(None, None, None, "unreadable")
    if geometry is not None:
        left, right, patch = geometry
        return fiducial.decode_origins(image, left, right, patch=patch)
    patch = header_int(header, "fiducial.patch_px", fiducial.PATCH_PX)
    margin = header_int(header, "fiducial.margin_px", fiducial.MARGIN_PX)
    gap = header_int(header, "fiducial.gap_px", fiducial.GAP_PX)
    return fiducial.decode(image, patch=patch, margin=margin, gap=gap)


def evaluate(play_dir, min_frames=DEFAULT_MIN_FRAMES, header_geometry=False):
    """Builds the E-06 verdict of one e05 (or e06a) play directory."""
    fiducial = load_fiducial()
    play = probe_common.Play(play_dir)
    probe = play.header.get("probe", "")
    if probe not in PROBES:
        raise probe_common.InputError("%s is a '%s' play, not one of %s" % (play_dir, probe, ", ".join(PROBES)))
    requests = play.csv(probe + "_requests", required=REQUEST_COLUMNS)
    files = e05_images.FrameFiles(play.dir)
    geometry = None
    if header_geometry:
        geometry = geometry_from_header(play.header, fiducial)

    verdict = probe_common.Verdict(LABEL, play.dir)

    counts = {}
    offsets = []
    present = 0
    for row in requests:
        if row["issue_status"] != "issued":
            continue
        path, _ = files.resolve(row["path"], row.get("found_suffix", ""))
        if not path:
            continue
        present += 1
        result = decode_file(fiducial, path, geometry, play.header)
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status == fiducial.STATUS_OK and row["issue_render_frame"]:
            issue_frame = int(row["issue_render_frame"])
            offsets.append(fiducial.unwrap(result.code, issue_frame) - issue_frame)

    decoded = counts.get(fiducial.STATUS_OK, 0)
    if present == 0:
        verdict.check("decode_ge_99_5pct", None, "no screenshot file present")
    else:
        rate = 100.0 * decoded / present
        detail = "%d of %d files decoded (%.2f%%, limit %.1f%%)" % (decoded, present, rate, DECODE_MIN_PCT)
        if present < min_frames:
            verdict.check("decode_ge_99_5pct", None, "%s; fewer than %d files" % (detail, min_frames))
        else:
            verdict.check("decode_ge_99_5pct", rate >= DECODE_MIN_PCT, detail)

    if not offsets:
        verdict.check("no_negative_offset", None, "no decoded file")
    else:
        negative = sum(1 for value in offsets if value < 0)
        verdict.check("no_negative_offset", negative == 0, "%d negative of %d (min %d, max %d frames)" % (
            negative, len(offsets), min(offsets), max(offsets)))
        verdict.info("offset_frames", "p50 %.1f, p99 %.1f (request frame to captured frame)" % (
            probe_common.percentile(offsets, 50), probe_common.percentile(offsets, 99)))
    verdict.info("status_counts", ", ".join("%s %d" % item for item in sorted(counts.items())) or "none")
    verdict.info("geometry", "run header fiducial.*" if header_geometry else "layout() of each image's size")
    verdict.check_footer(play)
    return verdict


def parse_args(argv):
    parser = argparse.ArgumentParser(description="E-06 fiducial decode over one e05 play directory (r2 4.4).")
    parser.add_argument("play_dir", help="$profile:DC_Probes/e05/<launch_id>/play_NN as a WSL path")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
                        help="files needed for a PASS/FAIL verdict (fewer: INCOMPLETE); default 600")
    parser.add_argument("--header-geometry", action="store_true",
                        help="patch origins from the run header's fiducial.* keys instead of the image size")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return evaluate(args.play_dir, args.min_frames, args.header_geometry).report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
