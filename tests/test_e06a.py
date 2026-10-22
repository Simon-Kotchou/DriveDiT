#!/usr/bin/env python3
"""Tests for tools/validator/e06a_reproject.py (E-06A marker reprojection alignment, r2 3.5).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_e06a.py

Synthetic world: magenta markers on two lines beside the path, a camera moving forward at 40 km/h with a yaw
oscillation. Images are rendered in memory: discs at the pinhole projections of one frame's camera, plus fiducial v2
of the frame the image claims to show. Rendering frame F and claiming F must PASS (RMS ~0 at k = 0); rendering
F - 1 and claiming F (a known one-frame lag) must give argmin k = -1 and FAIL; a static camera is INCOMPLETE. The
end-to-end cases write 12 PNGs into a probe_common.make_temp_dir play (never /tmp) and run the CLI.
"""

import math
import os
import shutil
import subprocess
import sys
import unittest

import cv2
import numpy

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
VALIDATOR = os.path.join(REPO, "tools", "validator")
sys.path.insert(0, VALIDATOR)
import e06a_reproject as rp  # noqa: E402  (puts tools/probes/t0eng on sys.path itself)
import probe_common  # noqa: E402

fiducial = rp.load_fiducial()

WIDTH, HEIGHT, VFOV = 640, 360, 50.0
MARKER_RADIUS_M = 0.25
SPEED_MS = 40 / 3.6
YAW_AMPLITUDE_DEG = 10.0
YAW_PERIOD_S = 2.0
MAGENTA = (255, 0, 255)
LAUNCH_ID = "20261022T090000Z-wb-0c2d"


def marker_field():
    """Two lines 10 m to each side, every 10 m from 10 to 120 m, heights alternating 0.6 / 1.6 m."""
    points = []
    for i, z in enumerate(range(10, 130, 10)):
        height = (0.6, 1.6)[i % 2]
        points.append((-10.0, height, float(z)))
        points.append((10.0, height, float(z)))
    return numpy.array(points)


MARKERS = marker_field()


def camera(t, moving=True):
    """(4, 3) rows right, up, fwd, pos (Enforce mat order); heading 0 = +Z, clockwise seen from above."""
    if not moving:
        t = 0.0
    yaw = math.radians(YAW_AMPLITUDE_DEG * math.sin(2 * math.pi * t / YAW_PERIOD_S))
    return numpy.array([
        [math.cos(yaw), 0.0, -math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [math.sin(yaw), 0.0, math.cos(yaw)],
        [0.0, 1.25, SPEED_MS * t],
    ])


def render(cam, code, seed):
    """Grey noisy background, a disc per marker 3..100 m in front, then the fiducial of code."""
    rng = numpy.random.default_rng(seed)
    image = rng.integers(80, 140, size=(HEIGHT, WIDTH, 3), dtype=numpy.uint8)
    uv, depth = rp.project(MARKERS, cam, VFOV, WIDTH, HEIGHT)
    f = rp.focal_px(VFOV, HEIGHT)
    ys, xs = numpy.mgrid[0:HEIGHT, 0:WIDTH]
    for (u, v), z in zip(uv, depth):
        if not (3.0 <= z <= 100.0) or not numpy.isfinite(u):
            continue
        radius = max(f * MARKER_RADIUS_M / z, 2.0)
        inside = (xs + 0.5 - u) ** 2 + (ys + 0.5 - v) ** 2 <= radius ** 2
        image[inside] = MAGENTA
    fiducial.render(image, fiducial.code_for(code))
    return image


def cameras(first_frame, count, fps, moving=True):
    return {first_frame + i: camera(i / fps, moving) for i in range(count)}


def observations_for(cams, frames, lag=0, condition=60):
    """Renders frame F - lag, claims F (fiducial and unwrap), and detects the blobs as the CLI does."""
    exclude = rp.band_exclusion(WIDTH, {}, fiducial)
    out = []
    for frame in frames:
        image = render(cams[frame - lag], frame, frame)
        decoded = fiducial.decode(image)
        assert decoded.status == "ok"
        claimed = fiducial.unwrap(decoded.code, frame - 2)
        blobs = rp.detect_blobs(image, MAGENTA, 60, exclude, 3)
        out.append({"condition": condition, "frame": claimed, "blobs": blobs})
    return out


class ModelTests(unittest.TestCase):
    def test_projection_axes(self):
        cam = camera(0.0)
        points = numpy.array([[0.0, 1.25, 20.0], [2.0, 1.25, 20.0], [0.0, 3.25, 20.0], [0.0, 1.25, -5.0]])
        uv, depth = rp.project(points, cam, VFOV, WIDTH, HEIGHT)
        self.assertAlmostEqual(uv[0, 0], WIDTH / 2)
        self.assertAlmostEqual(uv[0, 1], HEIGHT / 2)
        self.assertGreater(uv[1, 0], WIDTH / 2)       # +x right of the camera -> right in the image
        self.assertLess(uv[2, 1], HEIGHT / 2)         # +y above the camera -> up in the image
        self.assertTrue(numpy.isnan(uv[3, 0]))        # behind the camera
        self.assertAlmostEqual(depth[0], 20.0)
        self.assertAlmostEqual(rp.focal_px(VFOV, HEIGHT), 180 / math.tan(math.radians(25)))

    def test_detect_blobs_skips_the_bands(self):
        image = numpy.full((HEIGHT, WIDTH, 3), 100, dtype=numpy.uint8)
        image[100:110, 300:310] = MAGENTA
        fiducial.render(image, 5 * (1 + 8 + 64))      # value 5 (magenta) in the bands
        blobs = rp.detect_blobs(image, MAGENTA, 60, rp.band_exclusion(WIDTH, {}, fiducial), 3)
        self.assertEqual(blobs.shape, (1, 2))
        self.assertAlmostEqual(blobs[0, 0], 305.0)
        self.assertAlmostEqual(blobs[0, 1], 105.0)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.cams = cameras(1000, 140, 60.0)
        self.frames = list(range(1010, 1130, 8))

    def test_aligned_images_pass(self):
        result = rp.evaluate(observations_for(self.cams, self.frames), self.cams, MARKERS, VFOV, WIDTH, HEIGHT,
                             3, 20.0)[60]
        ok, detail = rp.judge(result)
        self.assertTrue(ok, detail)
        self.assertLess(result["rms"][0], 0.5)
        self.assertEqual(min(result["rms"], key=lambda k: result["rms"][k]), 0)
        self.assertGreater(result["blobs"], 50)

    def test_one_frame_lag_fails_at_k_minus_1(self):
        result = rp.evaluate(observations_for(self.cams, self.frames, lag=1), self.cams, MARKERS, VFOV, WIDTH,
                             HEIGHT, 3, 20.0)[60]
        ok, detail = rp.judge(result)
        self.assertFalse(ok)
        self.assertEqual(min(result["rms"], key=lambda k: result["rms"][k]), -1, detail)
        self.assertIn("argmin k=-1", detail)
        self.assertLess(result["rms"][-1], 0.5)

    def test_static_camera_is_incomplete(self):
        cams = cameras(1000, 140, 60.0, moving=False)
        result = rp.evaluate(observations_for(cams, self.frames), cams, MARKERS, VFOV, WIDTH, HEIGHT, 3, 20.0)[60]
        ok, detail = rp.judge(result)
        self.assertIsNone(ok, detail)
        self.assertEqual(result["static"], len(self.frames))

    def test_missing_neighbour_frames_are_edge_cases(self):
        result = rp.evaluate(observations_for(self.cams, [1001, 1137]), self.cams, MARKERS, VFOV, WIDTH, HEIGHT,
                             3, 20.0)[60]
        self.assertEqual(result["edge"], 2)
        self.assertIsNone(rp.judge(result)[0])


def mat_columns(prefix):
    return ["%s_%s_%s" % (prefix, part, c) for part in ("pos", "right", "up", "fwd") for c in "xyz"]


REQUEST_COLUMNS = [
    "slot_id", "requested_world_ms", "issue_status", "issue_world_ms", "issue_render_frame", "issue_physics_step",
    "issue_tick_ms", "path", "make_ok", "cam_current", "cam_attached", "inflight_at_issue", "backlog", "done_status",
    "done_render_frame", "done_world_ms", "done_tick_ms", "found_suffix", "fps_setting", "segment",
]
FRAME_COLUMNS = ([
    "render_frame", "world_time_ms", "tick_ms", "frame_timeslice_s", "fps", "fps_avg", "last_physics_step", "paused",
    "cam_current", "cam_attached", "fiducial_code"] + mat_columns("cam") + mat_columns("veh")
    + ["veh_speed_kmh", "inflight", "segment", "fps_setting"])


def cam_values(cam):
    return [round(v, 6) for v in list(cam[3]) + list(cam[0]) + list(cam[1]) + list(cam[2])]


def build_e06a_play(base, lag=0, measured_fps_30=30.0):
    """6 images at fps_setting 60 (frames 3000+) and 6 at 30 (frames 7000+), one slalom/backlog each."""
    play_dir = os.path.join(base, "DC_Probes", "e06a", LAUNCH_ID, "play_01")
    frames, requests, proj, files = [], [], [], {}
    slot = 0
    for setting, first, fps, segment in ((60, 3000, 60.0, "slalom"), (30, 7000, 30.0, "backlog")):
        cams = cameras(first, 80, fps)
        measured = fps
        if setting == 30:
            measured = measured_fps_30
        for frame, cam in sorted(cams.items()):
            frames.append([frame, (frame - first) * 1000.0 / fps, 1000 + frame, 1.0 / fps, measured, measured,
                           frame * 2, 0, 1, 1, fiducial.code_for(frame)] + cam_values(cam) + cam_values(cam)
                          + [40.0, 0, segment, setting])
        for frame in range(first + 10, first + 70, 10):
            rel = "frames/f_%08d" % slot
            files[rel + ".png"] = cv2.imencode(".png", cv2.cvtColor(render(cams[frame - lag], frame, slot),
                                                                    cv2.COLOR_RGB2BGR))[1].tobytes()
            requests.append([slot, 100.0 * slot, "issued", 100.0 * slot, frame - 1, frame * 2, 5000 + slot, rel,
                             1, 1, 1, 0, int(segment == "backlog"), "done", frame + 2, 100.0 * slot + 30,
                             5030 + slot, ".png", setting, segment])
            uv, depth = rp.project(MARKERS, cams[frame], VFOV, WIDTH, HEIGHT)
            for marker_id in range(len(MARKERS)):
                if 0.1 < depth[marker_id] <= 150 and numpy.isfinite(uv[marker_id, 0]):
                    proj.append([frame, marker_id, round(uv[marker_id, 0], 3), round(uv[marker_id, 1], 3),
                                 round(depth[marker_id], 3)])
            slot += 1
    markers = [[i, "straight", x, y, z, MARKER_RADIUS_M, "FF00FF", "shape"] for i, (x, y, z) in enumerate(MARKERS)]
    markers_end = [[i, 3, 0, 0.0] for i in range(len(MARKERS))]
    geometry = fiducial.layout(WIDTH, HEIGHT)
    header = {
        "schema": "dc-probe-run/1", "probe": "e06a", "play_dir": "$profile:DC_Probes/e06a/%s/play_01" % LAUNCH_ID,
        "params.fps_phases": "60,30", "rig.vfov_read": "%.6f" % VFOV, "fiducial.patch_px": "32",
        "fiducial.margin_px": "16", "fiducial.gap_px": "8", "fiducial.physical_w": str(WIDTH),
        "fiducial.physical_h": str(HEIGHT), "fiducial.left_x": str(geometry["left"][0][0]),
        "fiducial.right_x": str(geometry["right"][0][0]),
    }
    footer = {"schema": "dc-probe-run/1", "status": "complete", "reason": "done"}
    probe_common.write_synthetic_play(play_dir, header, {
        "e06a_frames": (FRAME_COLUMNS, frames),
        "e06a_requests": (REQUEST_COLUMNS, requests),
        "e06a_markers": (["marker_id", "set", "x", "y", "z", "radius_m", "color_hex", "kind"], markers),
        "e06a_markers_end": (["marker_id", "phase_idx", "moved", "moved_m"], markers_end),
        "e06a_proj": (["render_frame", "marker_id", "screen_x", "screen_y", "depth"], proj),
    }, footer=footer, files=files)
    return play_dir


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("e06a_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def run_cli(self, play_dir):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-I", "-B", os.path.join(VALIDATOR, "e06a_reproject.py"), play_dir],
                              capture_output=True, text=True, env=env)

    def test_12_pngs_pass(self):
        play_dir = build_e06a_play(self.base)
        self.assertEqual(len(os.listdir(os.path.join(play_dir, "frames"))), 12)
        done = self.run_cli(play_dir)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("[PASS] alignment_60fps", done.stdout)
        self.assertIn("[PASS] alignment_30fps", done.stdout)
        self.assertIn("[PASS] camera_model", done.stdout)
        self.assertIn("RESULT E-06A: PASS", done.stdout)

    def test_12_pngs_with_a_one_frame_lag_fail(self):
        done = self.run_cli(build_e06a_play(self.base, lag=1))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("[FAIL] alignment_60fps", done.stdout)
        self.assertIn("argmin k=-1", done.stdout)
        self.assertIn("RESULT E-06A: FAIL", done.stdout)

    def test_fps_setting_not_taken_is_incomplete(self):
        verdict = rp.analyse(build_e06a_play(self.base, measured_fps_30=58.0))
        found = {name: tag for tag, name, _ in verdict.items}
        self.assertEqual(found["alignment_60fps"], "PASS")
        self.assertEqual(found["alignment_30fps"], "INCOMPLETE")
        self.assertEqual(verdict.result(), "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
