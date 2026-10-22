#!/usr/bin/env python3
"""Tests for data/dc2/fiducial.py (fiducial v2 encoder and decoder, r2 3.4).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_dc2_fiducial.py

The module is loaded by file path: `import data.dc2` would run data/__init__.py, which imports torch. Images are
synthetic and built in memory; the one file round trip (load_rgb) writes into a temporary directory under $TMPDIR
when that lies in /dev/shm/ or /mnt/e/, else under /dev/shm/ (never /tmp: the WSL root is on C:), and removes it.
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
    if not ((base.startswith("/dev/shm/") or base.startswith("/mnt/e/")) and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)) or base.startswith("/mnt/c"):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/ or /mnt/e/")
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

    def test_noise_40_still_decodes(self):
        rng = numpy.random.default_rng(2)
        for code in (0, 0x2AAAA, M - 1, 123456):
            image = encoded(code, seed=3).astype(numpy.int16)
            image += rng.integers(-40, 41, size=image.shape, dtype=numpy.int16)
            noisy = numpy.clip(image, 0, 255).astype(numpy.uint8)
            self.assertDecodes(noisy, code)

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
        image = encoded(0)
        x0, y0 = fiducial.layout(1408, 720)["left"][0]
        image[y0:y0 + 32, x0:x0 + 32, 1] = 80     # exactly channel_margin: still 0
        self.assertEqual(fiducial.decode(image).status, "ok")
        image[y0:y0 + 32, x0:x0 + 32, 1] = 81     # just above: ambiguous
        self.assertEqual(fiducial.decode(image).status, "ambiguous_left")
        image[y0:y0 + 32, x0:x0 + 32, 1] = 175    # exactly 255 - channel_margin: 1
        result = fiducial.decode(image)
        self.assertEqual(result.status, "mirror_mismatch")
        self.assertEqual(result.left, 2)
        self.assertEqual(result.right, 0)

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

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            fiducial.decode(numpy.zeros((720, 1408), dtype=numpy.uint8))
        with self.assertRaises(ValueError):
            fiducial.decode(numpy.zeros((720, 1408, 3), dtype=numpy.float32))


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
