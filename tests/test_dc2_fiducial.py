#!/usr/bin/env python3
"""Tests for data/dc2/fiducial.py (fiducial v2 encoder and decoder, r2 3.4).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_fiducial.py

The module is loaded by file path, as the probe tools load it. Images are
synthetic and built in memory; the one file round trip (load_rgb) writes into a temporary directory under $TMPDIR
when that lies in /dev/shm/, else under /dev/shm/ (never /tmp: the WSL root is on the system drive), and removes it.
"""

import importlib.util
import os
import random
import shutil
import tempfile
import unittest

import numpy

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FIDUCIAL_PATH = os.path.join(REPO, "data", "dc2", "fiducial.py")


def load_fiducial():
    spec = importlib.util.spec_from_file_location("dc2_fiducial", FIDUCIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fiducial = load_fiducial()
M = fiducial.CODE_MODULUS


def temp_dir(prefix):
    """The make_temp_dir rule of tools/probes/t0eng/probe_common.py, kept local so this test needs no other tool."""
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not (base.startswith("/dev/shm/") and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/")
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def scene(width, height, seed=0):
    """Grey-ish random background, so the decoder never sees an easy all-black frame."""
    rng = numpy.random.default_rng(seed)
    return rng.integers(60, 200, size=(height, width, 3), dtype=numpy.uint8)


def encoded(code, width=1408, height=720, seed=0):
    image = scene(width, height, seed)
    fiducial.render(image, code)
    return image


class CodeTests(unittest.TestCase):
    def test_constants_match_the_engine_side(self):
        self.assertEqual(fiducial.CODE_BITS, 18)
        self.assertEqual(M, 262144)
        self.assertEqual(fiducial.PATCH_COUNT * fiducial.BITS_PER_PATCH, fiducial.CODE_BITS)

    def test_code_for_wraps(self):
        self.assertEqual(fiducial.code_for(0), 0)
        self.assertEqual(fiducial.code_for(M + 5), 5)
        self.assertEqual(fiducial.code_for(M - 1), M - 1)
        self.assertEqual(fiducial.code_for(-1), M - 1)

    def test_patch_values_order_and_inverse(self):
        self.assertEqual(fiducial.patch_values(0), [0] * 6)
        self.assertEqual(fiducial.patch_values(1), [1, 0, 0, 0, 0, 0])
        self.assertEqual(fiducial.patch_values(7 << 15), [0, 0, 0, 0, 0, 7])
        self.assertEqual(fiducial.patch_values(M - 1), [7] * 6)
        for code in (0, 1, 12345, M - 1):
            self.assertEqual(fiducial.code_from_values(fiducial.patch_values(code)), code)
        with self.assertRaises(ValueError):
            fiducial.patch_values(M)

    def test_the_8_colours(self):
        expected = {
            0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0),
            4: (0, 0, 255), 5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 255, 255),
        }
        for value, rgb in expected.items():
            self.assertEqual(fiducial.patch_color(value), rgb)
        with self.assertRaises(ValueError):
            fiducial.patch_color(8)

    def test_unwrap(self):
        self.assertEqual(fiducial.unwrap(5, 5), 5)
        self.assertEqual(fiducial.unwrap(fiducial.code_for(1000 + 2), 1000), 1002)
        self.assertEqual(fiducial.unwrap(fiducial.code_for(1000 - 3), 1000), 997)
        # across the wrap, both directions
        self.assertEqual(fiducial.unwrap(3, M - 2), M + 3)
        self.assertEqual(fiducial.unwrap(M - 2, M + 1), M - 2)
        for reference in (0, 7, M - 4, M, 3 * M + 1, 10 * M - 1):
            for k in range(-3, 4):
                self.assertEqual(fiducial.unwrap(fiducial.code_for(reference + k), reference), reference + k)


class LayoutTests(unittest.TestCase):
    def test_client_class_1408x720(self):
        geometry = fiducial.layout(1408, 720)
        self.assertEqual(geometry["left"][0], (16, 244))
        self.assertEqual(geometry["right"][0], (1360, 244))
        self.assertEqual([y for _, y in geometry["left"]], [244, 284, 324, 364, 404, 444])
        self.assertEqual(geometry["left"], [(16, y) for _, y in geometry["right"]])
        # inside the 64 px bands of the 1408 = 64 + 1280 + 64 client class
        self.assertLessEqual(16 + 32, 64)
        self.assertGreaterEqual(1360, 1408 - 64)

    def test_odd_size(self):
        geometry = fiducial.layout(1919, 1007)
        self.assertEqual(geometry["left"][0], (16, 387))
        self.assertEqual(geometry["right"][0], (1871, 387))
        self.assertEqual(geometry["right"][5], (1871, 387 + 5 * 40))

    def test_truncating_division_like_enforce(self):
        # 231 px is one pixel short of 6 * 32 + 5 * 8: Enforce truncates -1 / 2 to 0, Python would floor to -1
        self.assertEqual(fiducial.layout(200, 231)["left"][0], (16, 0))


class DecodeTests(unittest.TestCase):
    def assertDecodes(self, image, code):
        result = fiducial.decode(image)
        self.assertEqual(result.status, "ok", result)
        self.assertEqual(result.code, code)
        self.assertEqual(result.left, code)
        self.assertEqual(result.right, code)

    def test_round_trip_edges(self):
        for code in (0, 1, M - 1):
            self.assertDecodes(encoded(code), code)

    def test_round_trip_1000_random_codes(self):
        rng = random.Random(4)
        base = scene(1408, 720, seed=1)
        for _ in range(1000):
            code = rng.randrange(M)
            image = base.copy()
            fiducial.render(image, code)
            self.assertDecodes(image, code)

    def test_small_noise_still_decodes(self):
        # The capture is lossless and debug shapes kept their colour within 2 levels (run 1); +-6 is a wide margin
        rng = numpy.random.default_rng(2)
        for code in (0, 0x2AAAA, M - 1, 123456):
            image = encoded(code, seed=3).astype(numpy.int16)
            image += rng.integers(-6, 7, size=image.shape, dtype=numpy.int16)
            noisy = numpy.clip(image, 0, 255).astype(numpy.uint8)
            self.assertDecodes(noisy, code)

    def test_heavy_noise_is_not_a_carrier(self):
        # +-40 noise was "ok" under the old rule (margin 80, no presence gate); now no patch is flat enough
        rng = numpy.random.default_rng(2)
        image = encoded(0x2AAAA, seed=3).astype(numpy.int16)
        image += rng.integers(-40, 41, size=image.shape, dtype=numpy.int16)
        result = fiducial.decode(numpy.clip(image, 0, 255).astype(numpy.uint8))
        self.assertNotEqual(result.status, "ok")
        self.assertIsNone(result.code)

    def test_channel_at_128_is_ambiguous(self):
        image = encoded(0x15555)
        x0, y0 = fiducial.layout(1408, 720)["left"][2]
        image[y0:y0 + 32, x0:x0 + 32, 0] = 128
        result = fiducial.decode(image)
        self.assertEqual(result.status, "ambiguous_left")
        self.assertIsNone(result.code)
        self.assertIsNone(result.left)
        self.assertEqual(result.right, 0x15555)

        image = encoded(0x15555)
        x0, y0 = fiducial.layout(1408, 720)["right"][5]
        image[y0:y0 + 32, x0:x0 + 32, 2] = 128
        self.assertEqual(fiducial.decode(image).status, "ambiguous_right")

    def test_edge_of_the_margin(self):
        margin = fiducial.CHANNEL_MARGIN
        self.assertEqual(margin, 32)
        image = encoded(0)
        x0, y0 = fiducial.layout(1408, 720)["left"][0]
        image[y0:y0 + 32, x0:x0 + 32, 1] = margin          # exactly channel_margin: still 0
        result = fiducial.decode(image)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.margin, 0.0)
        image[y0:y0 + 32, x0:x0 + 32, 1] = margin + 1      # just above: no value, so not present
        result = fiducial.decode(image)
        self.assertEqual(result.status, "ambiguous_left")
        self.assertEqual((result.present_left, result.present_right), (5, 6))
        image[y0:y0 + 32, x0:x0 + 32, 1] = 255 - margin    # exactly 255 - channel_margin: 1
        result = fiducial.decode(image)
        self.assertEqual(result.status, "mirror_mismatch")
        self.assertEqual(result.left, 2)
        self.assertEqual(result.right, 0)

    def test_exact_patches_have_the_full_margin(self):
        result = fiducial.decode(encoded(4242))
        self.assertEqual(result.margin, float(fiducial.CHANNEL_MARGIN))
        self.assertEqual((result.present_left, result.present_right), (6, 6))
        image = encoded(7)                                  # patch 0 white
        x0, y0 = fiducial.layout(1408, 720)["right"][0]
        image[y0:y0 + 32, x0:x0 + 32, 0] = 250
        self.assertEqual(fiducial.decode(image).margin, float(fiducial.CHANNEL_MARGIN) - 5.0)

    def test_left_right_disagreement_is_mirror_mismatch(self):
        image = encoded(1000)
        right = scene(1408, 720)
        fiducial.render(right, 1001)
        image[:, 1408 - 64:, :] = right[:, 1408 - 64:, :]
        result = fiducial.decode(image)
        self.assertEqual(result.status, "mirror_mismatch")
        self.assertIsNone(result.code)
        self.assertEqual((result.left, result.right), (1000, 1001))

    def test_out_of_bounds(self):
        tiny = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        self.assertEqual(fiducial.decode(tiny).status, "out_of_bounds")
        with self.assertRaises(ValueError):
            fiducial.render(tiny, 5)

    def test_odd_size_round_trip(self):
        image = scene(1919, 1007, seed=5)
        fiducial.render(image, 77777)
        self.assertDecodes(image, 77777)

    def test_decode_origins_with_header_geometry(self):
        # a screenshot whose size differs from the workspace: the origins come from the run header instead
        image = scene(1500, 800, seed=6)
        geometry = fiducial.layout(1408, 720)
        for band in ("left", "right"):
            for (x0, y0), value in zip(geometry[band], fiducial.patch_values(4242)):
                image[y0:y0 + 32, x0:x0 + 32] = fiducial.patch_color(value)
        self.assertNotEqual(fiducial.decode(image).status, "ok")
        result = fiducial.decode_origins(image, geometry["left"], geometry["right"])
        self.assertEqual((result.status, result.code), ("ok", 4242))

    def test_result_keeps_the_four_positional_fields(self):
        # e06_fiducial and older callers build FiducialResult(code, left, right, status)
        result = fiducial.FiducialResult(None, None, None, "unreadable")
        self.assertEqual((result.present_left, result.present_right, result.margin), (0, 0, None))
        self.assertEqual(fiducial.decode_band(encoded(9), fiducial.layout(1408, 720)["left"]), (9, "ok"))

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            fiducial.decode(numpy.zeros((720, 1408), dtype=numpy.uint8))
        with self.assertRaises(ValueError):
            fiducial.decode(numpy.zeros((720, 1408, 3), dtype=numpy.float32))


def bark(width=1408, height=720, seed=11, low=25, high=60):
    """Dark, textured scene like the run-1 forest edges: medians ~40, per-channel std ~10."""
    rng = numpy.random.default_rng(seed)
    return rng.integers(low, high, size=(height, width, 3), dtype=numpy.uint8)


class PresenceGateTests(unittest.TestCase):
    """Run 1: dark tree bark in both bands decoded as six black patches, code 0 "ok", on every e05 frame."""

    def test_dark_textured_bands_are_absent(self):
        image = bark()
        result = fiducial.decode(image)
        self.assertEqual(result.status, "absent")
        self.assertIsNone(result.code)
        self.assertEqual((result.present_left, result.present_right), (0, 0))
        # the rule before the gates (margin 80, no presence test) read the same image as code 0 "ok"
        old = fiducial.decode(image, channel_margin=80, pixel_tolerance=255, uniform_std_max=1e9,
                              min_corner_fraction=0.0)
        self.assertEqual((old.status, old.code), ("ok", 0))

    def test_textured_patch_with_a_corner_median_is_not_present(self):
        image = encoded(0)
        x0, y0 = fiducial.layout(1408, 720)["left"][1]
        rng = numpy.random.default_rng(5)
        # median stays near 0, but the block is texture (std ~9 > 6)
        image[y0:y0 + 32, x0:x0 + 32] = rng.integers(0, 32, size=(32, 32, 3), dtype=numpy.uint8)
        result = fiducial.decode(image)
        self.assertEqual(result.status, "ambiguous_left")
        self.assertEqual(result.present_left, 5)

    def test_one_band_absent_is_ambiguous(self):
        image = encoded(77)
        image[:, :64] = bark(64, 720)
        result = fiducial.decode(image)
        self.assertEqual(result.status, "ambiguous_left")
        self.assertEqual((result.present_left, result.present_right), (0, 6))
        self.assertEqual(result.right, 77)

    def test_band_details(self):
        band = fiducial.band_details(bark(), fiducial.layout(1408, 720)["left"])
        self.assertEqual(band, fiducial.BandResult(None, "absent", 0, None))
        band = fiducial.band_details(encoded(5), fiducial.layout(1408, 720)["right"])
        self.assertEqual((band.code, band.reason, band.present), (5, "ok", 6))
        tiny = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        self.assertEqual(fiducial.band_details(tiny, fiducial.layout(100, 100)["left"]).reason, "out_of_bounds")

    def test_the_code_domain_stays_v2(self):
        # a flat black scene band is still the valid codeword 0: only the plausibility gate can reject it
        image = numpy.zeros((720, 1408, 3), dtype=numpy.uint8)
        self.assertEqual((fiducial.decode(image).status, fiducial.decode(image).code), ("ok", 0))
        self.assertDecodesCodeword(M - 1)

    def assertDecodesCodeword(self, code):
        image = scene(1408, 720, seed=4)
        fiducial.render(image, code)
        self.assertEqual(fiducial.decode(image).code, code)


class PlausibilityGateTests(unittest.TestCase):
    def test_capture_window(self):
        self.assertEqual(fiducial.MAX_CAPTURE_OFFSET, 2)
        self.assertEqual(fiducial.capture_window(1000), (1000, 1002))
        self.assertEqual(fiducial.capture_window(1000, done_frame=1002), (1000, 1001))
        self.assertEqual(fiducial.capture_window(1000, done_frame=1001), (1000, 1000))
        self.assertEqual(fiducial.capture_window(1000, done_frame=1077), (1000, 1002))   # backlog: late file
        self.assertEqual(fiducial.capture_window(1000, max_offset=5), (1000, 1005))

    def test_resolve_inside_the_window(self):
        result = fiducial.decode(encoded(fiducial.code_for(20001)))
        capture = fiducial.resolve(result, 20000, done_frame=20002)
        self.assertEqual(capture, fiducial.Capture(20001, 1, "ok", (20000, 20001)))

    def test_resolve_across_the_wrap(self):
        result = fiducial.decode(encoded(fiducial.code_for(M + 1)))
        capture = fiducial.resolve(result, M - 1)
        self.assertEqual((capture.frame, capture.offset, capture.status), (M + 1, 2, "ok"))

    def test_resolve_outside_is_implausible(self):
        before = fiducial.resolve(fiducial.decode(encoded(fiducial.code_for(19999))), 20000)
        self.assertEqual((before.frame, before.offset, before.status), (19999, -1, "implausible"))
        late = fiducial.resolve(fiducial.decode(encoded(fiducial.code_for(20002))), 20000, done_frame=20002)
        self.assertEqual((late.offset, late.status), (2, "implausible"))
        # run 1: a black band read as code 0 against request frames 737..11793
        black = fiducial.resolve(fiducial.decode(numpy.zeros((720, 1408, 3), dtype=numpy.uint8)), 737)
        self.assertEqual((black.offset, black.status), (-737, "implausible"))

    def test_resolve_passes_other_statuses_through(self):
        capture = fiducial.resolve(fiducial.decode(bark()), 500)
        self.assertEqual(capture, fiducial.Capture(None, None, "absent", (500, 502)))


class LoadRgbTests(unittest.TestCase):
    def test_png_round_trip_keeps_rgb_order(self):
        import cv2

        work = temp_dir("dc2_fiducial_")
        try:
            image = encoded(54321, width=640, height=360, seed=7)
            path = os.path.join(work, "f_00000001.png")
            self.assertTrue(cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
            loaded = fiducial.load_rgb(path)
            self.assertTrue(numpy.array_equal(loaded, image))
            self.assertEqual(fiducial.decode(loaded).code, 54321)
            with self.assertRaises(ValueError):
                fiducial.load_rgb(os.path.join(work, "missing.png"))
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
