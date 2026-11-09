#!/usr/bin/env python3
"""Tests for tools/probes/t0eng/e05_images.py (E-05) and e06_fiducial.py (E-06) on synthetic e05 plays.

The fiducial cases draw the widget carrier (margin 16) and / or the shape carrier (margin 64) of
tools/probes/t0eng/fiducial_carriers.py, or dark textured bands with no carrier (probe run 1); ProducerTests check
that the run-header and footer keys the analyses read are written by the Enforce sources.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_images.py

Every play is written with probe_common.write_synthetic_play into a probe_common.make_temp_dir directory (under
$TMPDIR when it lies in /dev/shm/ or on the data drive, else /dev/shm/; never /tmp) and removed afterwards. The column lists
mirror DC_ImageSlotScheduler.AddRequestColumns and DC_ProbeImages.AddFrameColumns.
"""

import os
import re
import shutil
import subprocess
import sys
import unittest

import cv2
import numpy

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
T0ENG = os.path.join(REPO, "tools", "probes", "t0eng")
sys.path.insert(0, T0ENG)
import probe_common  # noqa: E402
import e05_images  # noqa: E402
import e06_fiducial  # noqa: E402
import fiducial_carriers  # noqa: E402

ENFORCE = os.path.join(REPO, "src", "enfusion", "datacapture")

fiducial = e06_fiducial.load_fiducial()

LAUNCH_ID = "20261022T080000Z-wb-0a1b"
PLAY_VALUE = "$profile:DC_Probes/e05/%s/play_01" % LAUNCH_ID
WIDTH, HEIGHT = 128, 256          # smallest convenient canvas that holds both bands (y0 = 12)
WIDE = 320                        # holds the widget bands (x 16, 272) and the shape bands (x 64, 224)
SHAPE_MARGIN = 64
CONTROL = (255, 128, 0)

REQUEST_COLUMNS = [
    "slot_id", "requested_world_ms", "issue_status", "issue_world_ms", "issue_render_frame", "issue_physics_step",
    "issue_tick_ms", "path", "make_ok", "cam_current", "cam_attached", "inflight_at_issue", "backlog", "done_status",
    "done_render_frame", "done_world_ms", "done_tick_ms", "found_suffix",
]


def mat_columns(prefix):
    return ["%s_%s_%s" % (prefix, part, c) for part in ("pos", "right", "up", "fwd") for c in "xyz"]


FRAME_COLUMNS = ([
    "render_frame", "world_time_ms", "tick_ms", "frame_timeslice_s", "fps", "fps_avg", "last_physics_step", "paused",
    "cam_current", "cam_attached", "fiducial_code"] + mat_columns("cam") + mat_columns("veh")
    + ["veh_speed_kmh", "inflight", "segment"])

IDENTITY = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]


def png_bytes(rgb):
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return data.tobytes()


def frame_image(code, seed, undecodable=False, width=WIDTH, widget=True, shape=False, scene=None,
                black_widget=False):
    """A scene (random 60..200, or scene: "bark" = dark texture 25..60 like the run-1 forest edges) with the code
    drawn by the widget carrier (margin 16) and / or the shape carrier (margin 64); black_widget: flat black where the
    widget bands would be (a carrier that is not captured, over a dark flat scene: the decoder reads code 0)."""
    rng = numpy.random.default_rng(seed)
    if scene == "bark":
        image = rng.integers(25, 60, size=(HEIGHT, width, 3), dtype=numpy.uint8)
    else:
        image = rng.integers(60, 200, size=(HEIGHT, width, 3), dtype=numpy.uint8)
    if widget:
        fiducial.render(image, code)
    elif black_widget:
        fiducial.render(image, 0)
    if shape:
        fiducial.render(image, code, margin=SHAPE_MARGIN)
    if undecodable:
        x0, y0 = fiducial.layout(width, HEIGHT)["left"][3]
        image[y0:y0 + 32, x0:x0 + 32, :] = 128
    return image


def carrier_header(header, carrier, width, controls=False):
    """The rig.fiducial.* / fiducial.shape.* / fiducial.control.* keys DC_FiducialCarriers appends."""
    header["rig.fiducial.carrier"] = carrier
    header["rig.fiducial.depth_m"] = "1.0000"
    if carrier in ("shape", "both"):
        geometry = fiducial.layout(width, HEIGHT, margin=SHAPE_MARGIN)
        header.update({"fiducial.shape.created": "1", "fiducial.shape.patch_px": "32",
                       "fiducial.shape.margin_px": str(SHAPE_MARGIN), "fiducial.shape.gap_px": "8",
                       "fiducial.shape.left_x": str(geometry["left"][0][0]),
                       "fiducial.shape.right_x": str(geometry["right"][0][0])})
        for i, (_, y) in enumerate(geometry["left"]):
            header["fiducial.shape.y%d" % i] = str(y)
    else:
        header["fiducial.shape.created"] = "0"
    if carrier == "shape":
        header["fiducial.created"] = "0"
    if controls:
        # above the bands (y0 = 12) and between the shape bands (x 96..224), as the engine's top-centre row
        header.update({"fiducial.control.color": "255,128,0", "fiducial.control.flat": "100 0 30 10",
                       "fiducial.control.textured": "135 0 30 10", "fiducial.control.text": "170 0 50 10"})


def draw_controls(image):
    """The three control widgets as a capture that includes the UI would show them (fiducial.control.* above)."""
    image[0:10, 100:130] = CONTROL
    image[0:10, 135:165] = (200, 90, 0)             # a texture tinted (255, 128, 0): blue 0, dark texels too
    image[0:10:2, 135:165:2] = (0, 0, 0)
    image[2:8, 172:218:3] = CONTROL                 # glyph strokes


def build_e05_play(base, slots=20, missing=(), undecodable=(), offsets=None, footer=True, saved=None,
                   probe="e05", fps=60.0, not_current=(), vfov_read="50.000000", width=WIDTH, carrier=None,
                   widget=True, shape=False, scene=None, controls=False, extra_header=None, black_widget=()):
    """Writes a synthetic e05 play and its console log. Returns (play_dir, log_path).

    missing: slots without a file; undecodable: slots whose left band is grey; offsets: {slot: frames from the
    request to the captured frame} (default 1); saved: number of "Screenshot saved" lines (default: one per file);
    carrier: the rig.fiducial.carrier value (None: a play from before the shape carrier, no such key); widget /
    shape: which carriers the images show; scene "bark": dark textured background; controls: the header keys and
    the drawn control widgets; extra_header: more run header keys; black_widget: slots whose widget bands are flat
    black (with widget False).
    """
    play_dir = os.path.join(base, "DC_Probes", probe, LAUNCH_ID, "play_01")
    first_frame = 5000
    requests = []
    files = {}
    for slot in range(slots):
        issue_frame = first_frame + 6 * slot
        rel = "frames/f_%08d" % slot
        offset = 1
        if offsets and slot in offsets:
            offset = offsets[slot]
        captured = issue_frame + offset
        present = slot not in missing
        if present:
            image = frame_image(fiducial.code_for(captured), slot, slot in undecodable, width, widget, shape, scene,
                                slot in black_widget)
            if controls:
                draw_controls(image)
            files[rel + ".png"] = png_bytes(image)
        issue_ms = 100.0 * slot
        tick = 20000 + 100 * slot
        if present:
            done = ["done", issue_frame + 3, issue_ms + 50, tick + 50, ".png"]
        else:
            done = ["timeout", None, None, None, None]
        current = 0 if slot in not_current else 1
        requests.append([slot, issue_ms, "issued", issue_ms, issue_frame, 3 * issue_frame, tick, rel, 1, current, 1,
                         0, 0] + done)
    frames = []
    for frame in range(first_frame - 5, first_frame + 6 * slots + 5):
        frames.append([frame, (frame - first_frame) * 1000.0 / fps, 20000 + (frame - first_frame) * 17,
                       1.0 / fps, fps, fps, frame * 3, 0, 1, 1, fiducial.code_for(frame)] + IDENTITY + IDENTITY
                      + [30.0, 0, "drive"])
    header = {
        "schema": "dc-probe-run/1", "probe": probe, "play_dir": PLAY_VALUE.replace("e05", probe, 1),
        "launch.id": LAUNCH_ID, "params.image_hz": "10.000", "fiducial.patch_px": "32", "fiducial.margin_px": "16",
        "fiducial.gap_px": "8", "params.vertical_fov_deg": "50.000", "rig.vfov_read": vfov_read,
    }
    geometry = fiducial.layout(width, HEIGHT)
    header["fiducial.left_x"] = str(geometry["left"][0][0])
    header["fiducial.right_x"] = str(geometry["right"][0][0])
    for i, (_, y) in enumerate(geometry["left"]):
        header["fiducial.y%d" % i] = str(y)
    if carrier is not None:
        carrier_header(header, carrier, width, controls)
    header.update(extra_header or {})
    footer_kv = None
    if footer:
        footer_kv = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}
    csv_name = probe + "_requests"
    probe_common.write_synthetic_play(play_dir, header, {
        csv_name: (REQUEST_COLUMNS, requests),
        probe + "_frames": (FRAME_COLUMNS, frames),
    }, footer=footer_kv, files=files)

    if saved is None:
        saved = len(files)
    log_path = os.path.join(base, "console.log")
    lines = ["08:00:00.000 RESOURCES    : Screenshot saved to $profile:old/f_x.png (an earlier play, not counted)",
             "08:00:01.000 SCRIPT       : [DC_Probe %s] START play_dir=%s" % (probe, header["play_dir"])]
    for slot in range(saved):
        lines.append("08:00:02.000 RESOURCES    : Screenshot saved to %s/frames/f_%08d.png"
                     % (header["play_dir"], slot))
    lines.append("08:00:09.000 SCRIPT       : [DC_Probe %s] END reason=done status=complete" % probe)
    lines.append("08:00:10.000 RESOURCES    : Screenshot saved to $profile:later/f_y.png (after END, not counted)")
    with open(log_path, "w", encoding="ascii", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return play_dir, log_path


def items(verdict):
    return {name: (tag, detail) for tag, name, detail in verdict.items}


class TempCase(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_images_")
        self.addCleanup(shutil.rmtree, self.base, True)


class E05Tests(TempCase):
    def test_complete_play_passes(self):
        play_dir, log_path = build_e05_play(self.base)
        verdict = e05_images.evaluate(play_dir, log_path)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertEqual(found["rig_camera_current"][0], "PASS")
        self.assertEqual(found["rig_vfov"][0], "PASS")
        self.assertEqual(found["png_dims"][0], "PASS")
        self.assertIn("%dx%d" % (WIDTH, HEIGHT), found["png_dims"][1])
        self.assertEqual(found["saved_count_eq_files"][0], "PASS")
        self.assertIn("20 'Screenshot saved' lines", found["saved_count_eq_files"][1])
        self.assertIn("image_hz=10", found["DECISION"][1])
        self.assertIn(".png", found["engine_suffix"][1])
        self.assertIn("p50 50.0", found["latency"][1])

    def test_one_missing_file_in_20_fails(self):
        play_dir, log_path = build_e05_play(self.base, missing=(7,))
        verdict = e05_images.evaluate(play_dir, log_path)
        found = items(verdict)
        self.assertEqual(verdict.result(), "FAIL")
        self.assertEqual(found["missing_le_1pct"][0], "FAIL")
        self.assertIn("5.00%", found["missing_le_1pct"][1])
        self.assertEqual(found["saved_count_eq_files"][0], "PASS")   # 19 lines, 19 files
        self.assertIn("image_hz=5", found["DECISION"][1])

    def test_saved_count_mismatch_fails(self):
        play_dir, log_path = build_e05_play(self.base, saved=18)
        verdict = e05_images.evaluate(play_dir, log_path)
        found = items(verdict)
        self.assertEqual(verdict.result(), "FAIL")
        self.assertEqual(found["saved_count_eq_files"][0], "FAIL")
        self.assertEqual(found["missing_le_1pct"][0], "PASS")

    def test_rig_camera_not_current_fails(self):
        play_dir, log_path = build_e05_play(self.base, not_current=(4,))
        verdict = e05_images.evaluate(play_dir, log_path)
        found = items(verdict)
        self.assertEqual(found["rig_camera_current"][0], "FAIL")
        self.assertIn("1 of 20", found["rig_camera_current"][1])
        self.assertEqual(verdict.result(), "FAIL")

    def test_rig_vfov_off_fails_and_missing_is_incomplete(self):
        play_dir, log_path = build_e05_play(self.base, vfov_read="52.000000")
        found = items(e05_images.evaluate(play_dir, log_path))
        self.assertEqual(found["rig_vfov"][0], "FAIL")
        other = os.path.join(self.base, "second")
        play_dir, log_path = build_e05_play(other, vfov_read="")
        found = items(e05_images.evaluate(play_dir, log_path))
        self.assertEqual(found["rig_vfov"][0], "INCOMPLETE")

    def test_low_fps_fails(self):
        play_dir, log_path = build_e05_play(self.base, fps=25.0)
        found = items(e05_images.evaluate(play_dir, log_path))
        self.assertEqual(found["fps_p5_ge_30"][0], "FAIL")

    def test_without_log_or_footer_is_incomplete(self):
        play_dir, _ = build_e05_play(self.base, footer=False)
        verdict = e05_images.evaluate(play_dir, None)
        found = items(verdict)
        self.assertEqual(verdict.result(), "INCOMPLETE")
        self.assertEqual(found["saved_count_eq_files"][0], "INCOMPLETE")
        self.assertEqual(found["run_footer"][0], "INCOMPLETE")

    def test_cli_exit_codes(self):
        play_dir, log_path = build_e05_play(self.base)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        script = os.path.join(T0ENG, "e05_images.py")
        done = subprocess.run([sys.executable, "-I", "-B", script, play_dir, "--log", log_path],
                              capture_output=True, text=True, env=env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("RESULT E-05: PASS", done.stdout)
        self.assertTrue(done.stdout.startswith("E-05 R1 camera and 10 Hz screenshots: "))
        wrong = subprocess.run([sys.executable, "-I", "-B", script, os.path.join(self.base, "nowhere")],
                               capture_output=True, text=True, env=env)
        self.assertEqual(wrong.returncode, 2)

    def test_wrong_probe_is_an_input_error(self):
        play_dir, _ = build_e05_play(self.base, probe="e06a")
        with self.assertRaises(probe_common.InputError):
            e05_images.evaluate(play_dir, None)

    def test_shuttle_drive_info(self):
        # The gearbox fix (probe run 2: the manual S105 sat in neutral) adds the drive lines to the e05 run header;
        # the shuttle's own speed comes from e05_frames.csv (veh_speed_kmh of the drive frames, 30.0 here)
        play_dir, log_path = build_e05_play(self.base)
        found = items(e05_images.evaluate(play_dir, log_path))
        self.assertNotIn("drive", found, "a play without params.drive=1 is a static capture")
        with open(os.path.join(play_dir, "run_header.txt"), "a", encoding="ascii", newline="") as handle:
            handle.write("params.drive=1\nvehicle.gearbox=manual\ndrive.force_sim=1\ndrive.force_sim_e01=needed\n"
                         "gear_policy.shifts=4\nphysics_wake.calls=7\nphysics_wake.asleep_before=7\n")
        verdict = e05_images.evaluate(play_dir, log_path)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertEqual(found["drive"][0], "INFO")
        self.assertIn("max veh_speed_kmh 30.00 over 130 drive frames; gearbox manual; force_sim 1 (e01 decision "
                      "needed); gear policy shifts 4; physics wake 7 calls, 7 with the body asleep before",
                      found["drive"][1])
        self.assertNotIn("sleep re-wakes", found["drive"][1], "a play before the review fix has no such keys")

    def test_shuttle_drive_info_with_the_stuck_guard(self):
        # Review R-4: e01 said not_needed, the body kept falling asleep and the stuck guard latched the force at the
        # stop; the drive.force_sim appended at the stop wins over the start value
        play_dir, log_path = build_e05_play(self.base)
        with open(os.path.join(play_dir, "run_header.txt"), "a", encoding="ascii", newline="") as handle:
            handle.write("params.drive=1\nvehicle.gearbox=manual\ndrive.force_sim=0\ndrive.force_sim_e01=not_needed\n"
                         "gear_policy.shifts=4\nphysics_wake.calls=7\nphysics_wake.asleep_before=0\n"
                         "physics_wake.sleep_wakes=12\ndrive.stuck_forces=1\ndrive.force_sim=1\n")
        found = items(e05_images.evaluate(play_dir, log_path))
        self.assertIn("force_sim 1 (e01 decision not_needed); gear policy shifts 4; physics wake 7 calls, 0 with the "
                      "body asleep before; 12 sleep re-wakes while driving; stuck force latches 1", found["drive"][1])


class E06Tests(TempCase):
    def test_one_undecodable_in_700_passes(self):
        play_dir, _ = build_e05_play(self.base, slots=700, undecodable=(350,))
        verdict = e06_fiducial.evaluate(play_dir)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertIn("best carrier widget; widget 699 of 700", found["decode_ge_99_5pct"][1])
        self.assertIn("widget (rig.fiducial.carrier=absent", found["carriers"][1])
        self.assertEqual(found["carrier_widget"][0], "INFO")
        self.assertTrue(found["carrier_widget"][1].startswith("PASS decoded 699 of 700 files (99.86%"))
        self.assertIn("ambiguous_left 1, ok 699", found["carrier_widget"][1])
        self.assertIn("offset p50 1.0, p99 1.0", found["carrier_widget"][1])
        self.assertIn("done - 1 = captured on 0 of 699 single-flight decodes", found["carrier_widget"][1])
        self.assertEqual(found["offset_in_window"][0], "PASS")

    def test_100_frames_is_incomplete(self):
        play_dir, _ = build_e05_play(self.base, slots=100, undecodable=(5,))
        verdict = e06_fiducial.evaluate(play_dir)
        self.assertEqual(verdict.result(), "INCOMPLETE")
        self.assertEqual(items(verdict)["decode_ge_99_5pct"][0], "INCOMPLETE")
        self.assertEqual(e06_fiducial.evaluate(play_dir, min_frames=50).result(), "FAIL")   # 99/100 < 99.5%

    def test_negative_offset_is_implausible(self):
        play_dir, _ = build_e05_play(self.base, slots=30, offsets={4: -1})
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10)
        found = items(verdict)
        self.assertEqual(found["offset_in_window"][0], "FAIL")
        self.assertIn("carrier widget: 1 implausible of 30 decodes (window [issue, min(issue + 2, done - 1)])",
                      found["offset_in_window"][1])
        self.assertIn("implausible 1", found["carrier_widget"][1])
        self.assertIn("implausible offsets -1..-1", found["carrier_widget"][1])
        self.assertEqual(found["decode_ge_99_5pct"][0], "FAIL")       # 29 of 30
        self.assertEqual(verdict.result(), "FAIL")

    def test_a_frame_after_done_minus_1_is_implausible(self):
        # done_render_frame = issue + 3 in these plays: offset 3 shows a frame after the one that saw the file
        play_dir, _ = build_e05_play(self.base, slots=30, offsets={7: 3})
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertEqual(found["offset_in_window"][0], "FAIL")
        self.assertIn("implausible offsets 3..3", found["carrier_widget"][1])

    def test_header_geometry(self):
        play_dir, _ = build_e05_play(self.base, slots=12)
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10, header_geometry=True)
        self.assertEqual(verdict.result(), "PASS", verdict.items)

    def test_run1_dark_bands_no_longer_pass(self):
        # Run 1: no carrier in the files, dark bark in the bands: the old decoder said 100% "ok" with code 0
        play_dir, _ = build_e05_play(self.base, slots=40, widget=False, scene="bark")
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10)
        found = items(verdict)
        self.assertEqual(found["decode_ge_99_5pct"][0], "FAIL")
        self.assertIn("widget 0 of 40 (0.00%)", found["decode_ge_99_5pct"][1])
        self.assertIn("FAIL decoded 0 of 40 files (0.00%, limit 99.5%); absent 40", found["carrier_widget"][1])
        self.assertIn("files with >= 1 present patch 0", found["carrier_widget"][1])
        self.assertEqual(found["offset_in_window"][0], "INCOMPLETE")
        self.assertEqual(verdict.result(), "FAIL")

    def test_both_carriers_present(self):
        play_dir, _ = build_e05_play(self.base, slots=20, width=WIDE, carrier="both", shape=True)
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertIn("widget, shape (rig.fiducial.carrier=both)", found["carriers"][1])
        self.assertIn("widget 20 of 20 (100.00%), shape 20 of 20 (100.00%)", found["decode_ge_99_5pct"][1])
        self.assertTrue(found["carrier_shape"][1].startswith("PASS decoded 20 of 20"))
        self.assertIn("20 files decoded plausibly by both carriers to the same frame, 0 to different frames",
                      found["carriers_agree"][1])
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10, header_geometry=True)
        self.assertEqual(verdict.result(), "PASS", verdict.items)

    def test_shape_carrier_only_in_the_files(self):
        # The expected Workbench outcome: the widget bands are not captured, the shape bands are
        play_dir, _ = build_e05_play(self.base, slots=20, width=WIDE, carrier="both", widget=False, shape=True)
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertIn("best carrier shape; widget 0 of 20 (0.00%), shape 20 of 20 (100.00%)",
                      found["decode_ge_99_5pct"][1])
        self.assertTrue(found["carrier_widget"][1].startswith("FAIL decoded 0 of 20"))
        self.assertIn("absent 20", found["carrier_widget"][1])
        self.assertIn("0 files decoded plausibly by both carriers", found["carriers_agree"][1])

    def test_a_false_decode_on_the_absent_carrier_is_not_judged(self):
        # Review R-3: the widget carrier is not in the files, and in two of them its bands are flat black (code 0,
        # unwrapped to frame 0: implausible). The shape carrier qualifies; E-06 judges the carrier it selects.
        play_dir, _ = build_e05_play(self.base, slots=20, width=WIDE, carrier="both", widget=False, shape=True,
                                     black_widget=(3, 11))
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertIn("best carrier shape; widget 0 of 20 (0.00%), shape 20 of 20 (100.00%)",
                      found["decode_ge_99_5pct"][1])
        self.assertEqual(found["offset_in_window"][0], "PASS")
        self.assertIn("carrier shape: 0 implausible of 20 decodes", found["offset_in_window"][1])
        self.assertIn("implausible on the other carriers, not judged: widget 2", found["offset_in_window"][1])
        self.assertIn("absent 18, implausible 2", found["carrier_widget"][1])
        # the same false decodes on the selected carrier still fail
        play_dir, _ = build_e05_play(os.path.join(self.base, "w"), slots=20, width=WIDE, carrier="widget",
                                     widget=False, black_widget=(3, 11))
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertEqual(found["offset_in_window"][0], "FAIL")
        self.assertIn("carrier widget: 2 implausible of 2 decodes", found["offset_in_window"][1])

    def test_select_carrier_breaks_ties_on_implausible_decodes(self):
        stats = {name: fiducial_carriers.CarrierStats(name) for name in fiducial_carriers.CARRIERS}
        for name in fiducial_carriers.CARRIERS:
            stats[name].files = 10
            stats[name].offsets = [1] * 10
        stats["widget"].implausible = [-5]
        self.assertEqual(e06_fiducial.select_carrier(["widget", "shape"], stats), "shape")
        stats["widget"].implausible = []
        self.assertEqual(e06_fiducial.select_carrier(["widget", "shape"], stats), "widget")

    def test_shape_only_play(self):
        play_dir, _ = build_e05_play(self.base, slots=12, width=WIDE, carrier="shape", widget=False, shape=True)
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertIn("shape (rig.fiducial.carrier=shape)", found["carriers"][1])
        self.assertNotIn("carrier_widget", found)
        self.assertEqual(found["decode_ge_99_5pct"][0], "PASS")

    def test_no_carrier(self):
        play_dir, _ = build_e05_play(self.base, slots=12, extra_header={"fiducial.created": "0"})
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertEqual(found["decode_ge_99_5pct"][0], "INCOMPLETE")
        self.assertIn("drew no fiducial carrier", found["decode_ge_99_5pct"][1])
        with self.assertRaises(probe_common.InputError):
            play_dir, _ = build_e05_play(os.path.join(self.base, "x"), slots=2, carrier="sideways")
            e06_fiducial.evaluate(play_dir, min_frames=1)

    def test_widget_controls_and_header_summaries(self):
        summary = {
            "fiducial.diag.readbacks": "600", "fiducial.diag.patches_visible_min": "12",
            "fiducial.diag.patches_visible_max": "12", "fiducial.diag.patches_at_layout_min": "12",
            "fiducial.diag.patches_at_layout_max": "12", "fiducial.diag.patch_offset_max_px": "0.000",
            "fiducial.diag.control_text_visible": "600", "fiducial.diag.control_textured_visible": "600",
            "fiducial.diag.control_flat_visible": "600", "fiducial.diag.control_replacements": "0",
            "fiducial.diag.last": "workspace 1839x595 visible 1",
            "fiducial.shape.residual_frames": "20000", "fiducial.shape.last_postframe": "19990",
            "fiducial.shape.last_poststep": "0", "fiducial.shape.last_frame": "10",
            "fiducial.shape.residual_max_px": "0.040", "fiducial.shape.residual_max_px_postframe": "0.040",
            "fiducial.shape.residual_max_px_poststep": "-", "fiducial.shape.residual_max_px_frame": "0.000",
            "fiducial.shape.residual_mean_px": "0.010", "fiducial.shape.residual_over_1px": "0",
            "fiducial.shape.residual_rig_max_px": "0.040", "fiducial.shape.place_frame": "20000",
            "fiducial.shape.place_poststep": "17000", "fiducial.shape.place_postframe": "19990",
            "fiducial.shape.place_teleport": "12", "fiducial.shape.teleport_frames": "12",
            "fiducial.shape.residual_teleport_max_px": "851.200", "fiducial.shape.depth_test": "0",
            "fiducial.shape.drawing_at_stop": "1",
        }
        play_dir, _ = build_e05_play(self.base, slots=12, width=WIDE, carrier="both", shape=True, controls=True,
                                     extra_header=summary)
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertIn("in flat 12, textured 12, text 12 of 12 files", found["widget_controls"][1])
        self.assertIn("600 readbacks: patches visible_in_hierarchy 12..12 of 12, at layout 12..12, max offset 0.000 "
                      "px; controls visible text 600, textured 600, flat 600 (replaced 0); last: workspace 1839x595 "
                      "visible 1", found["widget_readback"][1])
        self.assertIn("20000 frames, last placed by postframe 19990, poststep 0, frame 10; residual vs the world "
                      "camera max 0.040 px (postframe 0.040, poststep -, frame 0.000), mean 0.010 px, over 1 px 0; "
                      "vs the rig camera max 0.040 px; placements frame 20000, poststep 17000, postframe 19990, "
                      "teleport 12; teleport frames 12 (not in the residual above), max 851.200 px; depth 1.0000 m "
                      "(depth test 0); runner clock.postframe_calls n/a", found["shape_placement"][1])
        stopped = dict(summary, **{"fiducial.shape.drawing_at_stop": "0", "fiducial.shape.layout_failures": "1",
                                   "fiducial.shape.failure": "shape fiducial canvas 100x40 px is too small"})
        play_dir, _ = build_e05_play(os.path.join(self.base, "small"), slots=12, width=WIDE, carrier="both",
                                     shape=True, extra_header=stopped)
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertIn("; not drawing at the stop (layout failures 1): shape fiducial canvas 100x40 px is too small; "
                      "depth", found["shape_placement"][1])
        play_dir, _ = build_e05_play(os.path.join(self.base, "plain"), slots=12, width=WIDE, carrier="both",
                                     shape=True, controls=False)
        found = items(e06_fiducial.evaluate(play_dir, min_frames=10))
        self.assertNotIn("widget_controls", found)
        self.assertNotIn("widget_readback", found)
        self.assertNotIn("shape_placement", found)


class ControlStatsTests(unittest.TestCase):
    def test_rectangles_follow_the_canvas_centre(self):
        # Review R-4: the engine centres the controls again on a new canvas; the detector shifts the start rectangles
        header = {"fiducial.control.canvas_w": "300", "fiducial.control.flat": "100 0 30 10"}
        stats = fiducial_carriers.ControlStats(header)
        self.assertEqual(stats.shift(300), 0)
        self.assertEqual(stats.shift(WIDE), 10)
        image = numpy.zeros((HEIGHT, WIDE, 3), dtype=numpy.uint8)
        image[0:10, 110:140] = CONTROL
        self.assertEqual(stats.fractions(image)["flat"], 1.0)
        old = fiducial_carriers.ControlStats({"fiducial.control.flat": "100 0 30 10"})   # no canvas_w: as written
        self.assertEqual(old.shift(WIDE), 0)
        self.assertAlmostEqual(old.fractions(image)["flat"], 20 / 30)


class E05FiducialTests(TempCase):
    def test_per_carrier_info_on_a_sample(self):
        play_dir, log_path = build_e05_play(self.base, slots=40, width=WIDE, carrier="both", widget=False,
                                            shape=True)
        verdict = e05_images.evaluate(play_dir, log_path)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertIn("every 10-th present file: decoded 0 of 4 files (0.00%); absent 4", found["fiducial_widget"][1])
        self.assertIn("every 10-th present file: decoded 4 of 4 files (100.00%); ok 4; offset p50 1.0",
                      found["fiducial_shape"][1])
        found = items(e05_images.evaluate(play_dir, log_path, fiducial_every=1))
        self.assertIn("decoded 40 of 40 files", found["fiducial_shape"][1])
        found = items(e05_images.evaluate(play_dir, log_path, fiducial_every=0))
        self.assertIn("carriers widget, shape; decode skipped", found["fiducial"][1])
        self.assertNotIn("fiducial_shape", found)

    def test_old_play_reads_as_widget(self):
        play_dir, log_path = build_e05_play(self.base, slots=20)
        found = items(e05_images.evaluate(play_dir, log_path))
        self.assertIn("decoded 2 of 2 files (100.00%); ok 2", found["fiducial_widget"][1])
        self.assertNotIn("fiducial_shape", found)



def enforce_source(name):
    with open(os.path.join(ENFORCE, name), "r", encoding="ascii") as handle:
        return handle.read()


class ProducerTests(unittest.TestCase):
    """The run-header and footer keys the fiducial analyses read are written by the Enforce producers."""

    def test_carrier_and_shape_keys(self):
        carriers = enforce_source("DC_ProbeFiducial.c")
        shape = enforce_source("DC_FiducialShape.c")
        for key in ("rig.fiducial.carrier", "rig.fiducial.depth_m", "fiducial.diag.enabled"):
            self.assertIn('"%s="' % key, carriers)
        # a carrier that is not drawn writes its created=0 key (fiducial_carriers.carriers reads fiducial.created)
        for line in ('"fiducial.created=0"', '"fiducial.shape.created=0"'):
            self.assertIn(line, carriers)
        self.assertIn('"fiducial.created=" + BoolText(m_bCreated)', enforce_source("DC_Fiducial.c"))
        self.assertIn('"fiducial.shape.created=" + BoolText(m_bCreated)', shape)
        for key in ("patch_px", "margin_px", "gap_px", "left_x", "right_x", "depth_m", "focal_px", "canvas_w",
                    "depth_test"):
            self.assertIn('"fiducial.shape.%s="' % key, shape)
        self.assertIn('"fiducial.shape.y"', shape)
        for value in fiducial_carriers.CARRIERS + ("both",):
            self.assertIn('return "%s";' % value, carriers)
        self.assertIn('"rig.fiducial.carrier=none"', enforce_source("DC_ProbeImages.c"))

    def test_summary_and_control_keys(self):
        carriers = enforce_source("DC_ProbeFiducial.c")
        for key in ("place_frame", "place_poststep", "place_postframe", "last_frame", "last_poststep",
                    "last_postframe", "residual_frames", "residual_max_px", "residual_mean_px", "residual_over_1px",
                    "residual_max_px_postframe", "residual_max_px_poststep", "residual_max_px_frame",
                    "residual_rig_max_px", "place_teleport", "teleport_frames", "residual_teleport_max_px",
                    "layout_failures", "drawing_at_stop", "failure"):
            self.assertIn('"fiducial.shape.%s="' % key, carriers)
        for key in ("readbacks", "patches_visible_min", "patches_visible_max", "patches_at_layout_min",
                    "patches_at_layout_max", "patch_offset_max_px", "control_flat_visible",
                    "control_textured_visible", "control_text_visible", "control_replacements", "last"):
            self.assertIn('"fiducial.diag.%s="' % key, carriers)
        for name in fiducial_carriers.CONTROLS + ("color", "canvas_w"):
            self.assertIn('"fiducial.control.%s="' % name, carriers)
        rgb = tuple(int(re.search(r"static const int COLOR_%s = (\d+);" % c, carriers).group(1)) for c in "RGB")
        self.assertEqual(rgb, fiducial_carriers.CONTROL_COLOR)
        self.assertIn('"clock.postframe_calls="', enforce_source("DC_ProbeRunner.c"))

    def test_attribute_defaults_match_the_analysis_defaults(self):
        for name in ("DC_ProbeImages.c", "DC_ProbeMarkers.c"):
            source = enforce_source(name)
            margin = re.search(r'\[Attribute\("(\d+)"[^\n]*\]\s*protected int m_iShapeMarginPx;', source)
            self.assertEqual(int(margin.group(1)), fiducial_carriers.DEFAULT_MARGIN_PX[fiducial_carriers.SHAPE], name)
            carrier = re.search(r'\[Attribute\("(\d+)"[^\n]*\]\s*protected DC_EFiducialCarrier m_eFiducialCarrier;',
                                source)
            self.assertEqual(carrier.group(1), "2", "%s: default BOTH (enum index 2)" % name)
            depth_test = re.search(r'\[Attribute\("(\d+)"[^\n]*\]\s*protected bool m_bShapeDepthTest;', source)
            self.assertEqual(depth_test.group(1), "0", "%s: the overlay flags by default" % name)
        enum = re.search(r"enum DC_EFiducialCarrier\s*\{([^}]*)\}", enforce_source("DC_FiducialShape.c")).group(1)
        self.assertEqual(re.findall(r"^\s*(\w+),?", enum, re.M), ["WIDGET", "SHAPE", "BOTH"])


if __name__ == "__main__":
    unittest.main()
