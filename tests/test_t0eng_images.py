#!/usr/bin/env python3
"""Tests for tools/probes/t0eng/e05_images.py (E-05) and e06_fiducial.py (E-06) on synthetic e05 plays.

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_images.py

Every play is written with probe_common.write_synthetic_play into a probe_common.make_temp_dir directory (under
$TMPDIR when it lies in /dev/shm/ or /mnt/e/, else /dev/shm/; never /tmp) and removed afterwards. The column lists
mirror DC_ImageSlotScheduler.AddRequestColumns and DC_ProbeImages.AddFrameColumns.
"""

import os
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

fiducial = e06_fiducial.load_fiducial()

LAUNCH_ID = "20261022T080000Z-wb-0a1b"
PLAY_VALUE = "$profile:DC_Probes/e05/%s/play_01" % LAUNCH_ID
WIDTH, HEIGHT = 128, 256          # smallest convenient canvas that holds both bands (y0 = 12)

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


def frame_image(code, seed, undecodable=False):
    rng = numpy.random.default_rng(seed)
    image = rng.integers(60, 200, size=(HEIGHT, WIDTH, 3), dtype=numpy.uint8)
    fiducial.render(image, code)
    if undecodable:
        x0, y0 = fiducial.layout(WIDTH, HEIGHT)["left"][3]
        image[y0:y0 + 32, x0:x0 + 32, :] = 128
    return image


def build_e05_play(base, slots=20, missing=(), undecodable=(), offsets=None, footer=True, saved=None,
                   probe="e05", fps=60.0, not_current=(), vfov_read="50.000000"):
    """Writes a synthetic e05 play and its console log. Returns (play_dir, log_path).

    missing: slots without a file; undecodable: slots whose left band is grey; offsets: {slot: frames from the
    request to the captured frame} (default 1); saved: number of "Screenshot saved" lines (default: one per file).
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
            files[rel + ".png"] = png_bytes(frame_image(fiducial.code_for(captured), slot, slot in undecodable))
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
    geometry = fiducial.layout(WIDTH, HEIGHT)
    header["fiducial.left_x"] = str(geometry["left"][0][0])
    header["fiducial.right_x"] = str(geometry["right"][0][0])
    for i, (_, y) in enumerate(geometry["left"]):
        header["fiducial.y%d" % i] = str(y)
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


class E06Tests(TempCase):
    def test_one_undecodable_in_700_passes(self):
        play_dir, _ = build_e05_play(self.base, slots=700, undecodable=(350,))
        verdict = e06_fiducial.evaluate(play_dir)
        found = items(verdict)
        self.assertEqual(verdict.result(), "PASS", verdict.items)
        self.assertIn("699 of 700", found["decode_ge_99_5pct"][1])
        self.assertIn("ambiguous_left 1", found["status_counts"][1])
        self.assertIn("p50 1.0", found["offset_frames"][1])

    def test_100_frames_is_incomplete(self):
        play_dir, _ = build_e05_play(self.base, slots=100, undecodable=(5,))
        verdict = e06_fiducial.evaluate(play_dir)
        self.assertEqual(verdict.result(), "INCOMPLETE")
        self.assertEqual(items(verdict)["decode_ge_99_5pct"][0], "INCOMPLETE")
        self.assertEqual(e06_fiducial.evaluate(play_dir, min_frames=50).result(), "FAIL")   # 99/100 < 99.5%

    def test_negative_offset_fails(self):
        play_dir, _ = build_e05_play(self.base, slots=30, offsets={4: -1})
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10)
        self.assertEqual(items(verdict)["no_negative_offset"][0], "FAIL")
        self.assertEqual(verdict.result(), "FAIL")

    def test_header_geometry(self):
        play_dir, _ = build_e05_play(self.base, slots=12)
        verdict = e06_fiducial.evaluate(play_dir, min_frames=10, header_geometry=True)
        self.assertEqual(verdict.result(), "PASS", verdict.items)


if __name__ == "__main__":
    unittest.main()
