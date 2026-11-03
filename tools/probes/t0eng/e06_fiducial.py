#!/usr/bin/env python3
"""E-06 analysis: fiducial v2 decode over the e05 screenshots, per carrier (r2 4.4 T0-ENG core; DD-4-probe).

    python3 -I -B tools/probes/t0eng/e06_fiducial.py <e05 play_dir> [--min-frames 600] [--header-geometry]
        [--max-offset 2]

Decodes every screenshot the engine wrote for an issued request of the play (data/dc2/fiducial.py, loaded by file
path) once per carrier the play drew (rig.fiducial.carrier: widget, shape or both; tools/probes/t0eng/
fiducial_carriers.py), with the presence gate (a band of scene texture is "absent", not code 0) and the plausibility
gate (the captured frame lies in [issue, issue + --max-offset] and <= done_render_frame - 1, else "implausible").
It prints the r2 4.4 criterion of E-06:
  * decode_ge_99_5pct   plausible decodes / files present >= 99.5% for at least one carrier, over >= --min-frames
                        files (fewer: INCOMPLETE); the detail names every carrier's rate
  * offset_in_window    no carrier decoded a present band to a frame outside the window (an image cannot show a frame
                        before its request, nor one after the frame that saw its file)
and as INFO, per carrier ("carrier_widget", "carrier_shape"): its own PASS/FAIL word for the 99.5% rule, the status
counts (ok, absent, ambiguous_left / _right, mirror_mismatch, out_of_bounds, implausible, unreadable), the offset
p50 / p99 of the plausible decodes, how often captured = done_render_frame - 1 on single-flight requests, the
smallest colour margin and the files with at least one present patch; "carriers_agree" when both decoded the same
file; "widget_controls" (the control widgets found in the files), "widget_readback" (fiducial.diag.*) and
"shape_placement" (fiducial.shape.* placement and residual totals, footer clock.postframe_calls) when the play has
them.

The patch origins come from the image's own size (data/dc2/fiducial.layout, with each carrier's patch, margin and
gap). --header-geometry uses the carriers' left_x / right_x / y0..y5 keys of run_header.txt instead (for a
screenshot whose size differs from the workspace). The same play layout works for an e06a play (e06a_requests.csv).
Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error. The script only reads.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402  (shared reader and verdict printer)
import e05_images  # noqa: E402  (FrameFiles: request path -> engine-written file)
import fiducial_carriers  # noqa: E402  (carriers, geometry, per-carrier statistics)

LABEL = "E-06 fiducial v2 decode"
PROBES = ("e05", "e06a")
REQUEST_COLUMNS = ("slot_id", "issue_status", "issue_render_frame", "path", "found_suffix")
DECODE_MIN_PCT = 99.5
DEFAULT_MIN_FRAMES = 600

load_fiducial = fiducial_carriers.load_fiducial
header_int = fiducial_carriers.header_int


def geometry_from_header(header, fiducial, carrier=fiducial_carriers.WIDGET):
    """(left origins, right origins, patch px) from a carrier's run-header keys (DC_Fiducial / DC_FiducialShape
    AppendHeaderLines)."""
    return fiducial_carriers.geometry_from_header(header, fiducial, carrier)


def evaluate(play_dir, min_frames=DEFAULT_MIN_FRAMES, header_geometry=False, max_offset=None):
    """Builds the E-06 verdict of one e05 (or e06a) play directory."""
    fiducial = load_fiducial()
    if max_offset is None:
        max_offset = fiducial.MAX_CAPTURE_OFFSET
    play = probe_common.Play(play_dir)
    probe = play.header.get("probe", "")
    if probe not in PROBES:
        raise probe_common.InputError("%s is a '%s' play, not one of %s" % (play_dir, probe, ", ".join(PROBES)))
    requests = play.csv(probe + "_requests", required=REQUEST_COLUMNS)
    files = e05_images.FrameFiles(play.dir)
    carrier_list = fiducial_carriers.carriers(play.header)

    verdict = probe_common.Verdict(LABEL, play.dir)
    verdict.info("carriers", "%s (rig.fiducial.carrier=%s)" % (
        ", ".join(carrier_list) or "none",
        play.header.get("rig.fiducial.carrier", "absent: widget unless fiducial.created=0")))
    stats = {carrier: fiducial_carriers.CarrierStats(carrier) for carrier in carrier_list}
    controls = fiducial_carriers.ControlStats(play.header)
    agree = 0
    disagree = 0
    present = 0
    for row in requests:
        if row["issue_status"] != "issued":
            continue
        path, _ = files.resolve(row["path"], row.get("found_suffix", ""))
        if not path:
            continue
        present += 1
        if not carrier_list:
            continue
        try:
            image = fiducial.load_rgb(path)
        except ValueError:
            for carrier in carrier_list:
                stats[carrier].add_unreadable()
            continue
        issue = probe_common.to_int(row["issue_render_frame"])
        if issue is None:
            raise probe_common.InputError("issued request %s has no issue_render_frame" % row["slot_id"])
        done = probe_common.to_int(row.get("done_render_frame", ""))
        decodes = fiducial_carriers.decode_all(fiducial, image, play.header, carrier_list, issue, done, max_offset,
                                               header_geometry)
        single = fiducial_carriers.is_single_flight(row)
        for carrier, (result, capture) in decodes.items():
            stats[carrier].add(result, capture, done, single)
        if len(decodes) > 1:
            _, reason = fiducial_carriers.agreed_frame(fiducial, decodes)
            plausible = [c for c, (_, capture) in decodes.items() if capture.status == fiducial.STATUS_OK]
            if reason == "ok" and len(plausible) > 1:
                agree += 1
            elif reason == "disagree":
                disagree += 1
        if controls.active():
            controls.add(image)

    if not carrier_list:
        verdict.check("decode_ge_99_5pct", None, "the play drew no fiducial carrier")
    elif present == 0:
        verdict.check("decode_ge_99_5pct", None, "no screenshot file present")
    else:
        rates = ", ".join("%s %d of %d (%.2f%%)" % (c, stats[c].decoded(), stats[c].files, stats[c].rate())
                          for c in carrier_list)
        best = max(carrier_list, key=lambda c: stats[c].rate())
        detail = "best carrier %s; %s; limit %.1f%%" % (best, rates, DECODE_MIN_PCT)
        if present < min_frames:
            verdict.check("decode_ge_99_5pct", None, "%s; fewer than %d files" % (detail, min_frames))
        else:
            verdict.check("decode_ge_99_5pct", stats[best].rate() >= DECODE_MIN_PCT, detail)

    implausible = sum(len(stats[c].implausible) for c in carrier_list)
    decoded = sum(stats[c].decoded() for c in carrier_list)
    if implausible + decoded == 0:
        verdict.check("offset_in_window", None, "no carrier decoded a file")
    else:
        verdict.check("offset_in_window", implausible == 0, "%d implausible of %d decodes (%s; window [issue, "
                      "min(issue + %d, done - 1)])" % (
                          implausible, implausible + decoded,
                          ", ".join("%s %d" % (c, len(stats[c].implausible)) for c in carrier_list), max_offset))

    for carrier in carrier_list:
        word = "n/a"
        if stats[carrier].files:
            word = probe_common.PASS if stats[carrier].rate() >= DECODE_MIN_PCT else probe_common.FAIL
        verdict.info("carrier_" + carrier, "%s %s" % (word, stats[carrier].summary(DECODE_MIN_PCT)))
    if len(carrier_list) > 1:
        verdict.info("carriers_agree", "%d files decoded plausibly by both carriers to the same frame, %d to "
                     "different frames" % (agree, disagree))
    if controls.active():
        verdict.info("widget_controls", controls.summary())
    readback = fiducial_carriers.readback_text(play.header)
    if readback:
        verdict.info("widget_readback", readback)
    placement = fiducial_carriers.placement_text(play.header, play.footer)
    if placement:
        verdict.info("shape_placement", placement)
    verdict.info("geometry", "run header fiducial.* / fiducial.shape.*" if header_geometry
                 else "layout() of each image's size")
    verdict.check_footer(play)
    return verdict


def parse_args(argv):
    parser = argparse.ArgumentParser(description="E-06 fiducial decode over one e05 play directory (r2 4.4).")
    parser.add_argument("play_dir", help="$profile:DC_Probes/e05/<launch_id>/play_NN as a WSL path")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
                        help="files needed for a PASS/FAIL verdict (fewer: INCOMPLETE); default 600")
    parser.add_argument("--header-geometry", action="store_true",
                        help="patch origins from the run header's fiducial.* keys instead of the image size")
    parser.add_argument("--max-offset", type=int, default=None,
                        help="largest captured - issue offset a decode may have [frames]; default 2")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return evaluate(args.play_dir, args.min_frames, args.header_geometry, args.max_offset).report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
