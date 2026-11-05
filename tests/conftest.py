"""Pytest configuration for the DriveDiT tests.

torch is optional here: the capture tests need only the stdlib, numpy and OpenCV, and the capture-tests CI job runs
them with no torch installed. Test modules that need torch (the `model` extra) are listed in TORCH_TEST_MODULES; they
get the `torch` marker, and without torch they are not collected at all. A new test module that imports torch must be
added to that list. Single tests that need torch inside an otherwise torch-free module are listed in TORCH_TEST_IDS:
they get the marker too, so the model job (`-m torch`) runs them and the capture job (`-m "not torch"`) does not.

This file imports neither torch nor any project module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Paths relative to this directory.
TORCH_TEST_MODULES = [
    "test_torch_smoke.py",
]

# Node ids (relative to the repo root) of torch tests in torch-free modules. Without torch they skip themselves.
TORCH_TEST_IDS = [
    # data/enfusion_preprocessing.py (dc-1.0) imports torch; the rest of test_dc2_reader.py does not.
    "tests/test_dc2_reader.py::LegacyGuardTest::test_dc2_session_raises_naming_the_column",
    "tests/test_dc2_reader.py::LegacyGuardTest::test_legacy_session_raises_naming_the_column",
    "tests/test_dc2_reader.py::LegacyGuardTest::test_bad_row_never_desynchronises_the_arrays",
]

_HERE = Path(__file__).parent
_TORCH_PATHS = {(_HERE / p).resolve() for p in TORCH_TEST_MODULES}
_TORCH_IDS = set(TORCH_TEST_IDS)

if importlib.util.find_spec("torch") is None:
    collect_ignore = list(TORCH_TEST_MODULES)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in _TORCH_IDS or item.path.resolve() in _TORCH_PATHS:
            item.add_marker(pytest.mark.torch)
