"""Pytest configuration for the DriveDiT tests.

torch is optional here: the capture tests need only the stdlib, numpy and OpenCV, and the capture-tests CI job runs
them with no torch installed. Test modules that need torch (the `model` extra) are listed in TORCH_TEST_MODULES; they
get the `torch` marker, and without torch they are not collected at all. A new test module that imports torch must be
added to that list.

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

_HERE = Path(__file__).parent
_TORCH_PATHS = {(_HERE / p).resolve() for p in TORCH_TEST_MODULES}

if importlib.util.find_spec("torch") is None:
    collect_ignore = list(TORCH_TEST_MODULES)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.resolve() in _TORCH_PATHS:
            item.add_marker(pytest.mark.torch)
