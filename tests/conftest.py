"""Pytest configuration for the DriveDiT tests.

torch is optional here: the capture tests need only the stdlib, numpy and OpenCV, and the capture-tests CI job runs
them with no torch installed. Test modules that need torch (the `model` extra) are listed in TORCH_TEST_MODULES; they
get the `torch` marker, and without torch they are not collected at all. Whole directories of torch tests (the drivedit
codec, model and train tests) are listed in TORCH_TEST_DIRS and treated the same way. A new test module that imports
torch must be added to one of the two lists.

This file imports neither torch nor any project module, and sets DRIVEDIT_LOCAL_CONFIG (below). It also replaces pytest's
tmp_path and tmp_path_factory with directories in validated RAM (tests/fixtures/local_config/scratch.py, DD-22): no test falls back to the
system temp dir.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Paths relative to this directory.
TORCH_TEST_MODULES = [
    "test_torch_smoke.py",
    "drivedit/data/test_torch_dataset.py",
    "drivedit/eval/test_idm.py",
]

# Directories, relative to this one, in which every test needs torch.
TORCH_TEST_DIRS = [
    "drivedit/codec",
    "drivedit/model",
    "drivedit/train",
]

_HERE = Path(__file__).parent

# DD-22: every test reads the synthetic local config (placeholder drives X:, Y:, Z:, none of them mounted), never this
# machine's, so no default root names a real drive. Tests that compare with files of a real machine opt in with
# DD_TEST_LOCAL_CONFIG (tests/fixtures/local_config/evidence.py).
os.environ["DRIVEDIT_LOCAL_CONFIG"] = str(_HERE / "fixtures" / "local_config" / "placeholder.local.env")
_TORCH_PATHS = {(_HERE / p).resolve() for p in TORCH_TEST_MODULES}
_TORCH_DIRS = [(_HERE / d).resolve() for d in TORCH_TEST_DIRS]

_scratch_spec = importlib.util.spec_from_file_location("dd22_scratch", _HERE / "fixtures" / "local_config" / "scratch.py")
assert _scratch_spec is not None
assert _scratch_spec.loader is not None
_scratch = importlib.util.module_from_spec(_scratch_spec)
_scratch_spec.loader.exec_module(_scratch)


class _RamTempPathFactory:
    """The part of pytest's TempPathFactory the tests use (mktemp, getbasetemp), in validated RAM."""

    def __init__(self, base: Path) -> None:
        self._base = base

    def getbasetemp(self) -> Path:
        return self._base

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        return Path(tempfile.mkdtemp(prefix=basename + "_", dir=self._base))


@pytest.fixture(scope="session")
def tmp_path_factory() -> Iterator[_RamTempPathFactory]:
    """pytest's tmp_path_factory in validated RAM (scratch.ram_base(); never the system temp dir)."""
    base = Path(tempfile.mkdtemp(prefix="dd22_pytest_", dir=_scratch.ram_base()))
    yield _RamTempPathFactory(base)
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """pytest's tmp_path in validated RAM (scratch.ram_base(); never the system temp dir), removed afterwards. The test
    skips, with the reason, where there is no usable RAM scratch."""
    base = _scratch.ram_base()
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", request.node.name)[:40]
    path = Path(tempfile.mkdtemp(prefix=prefix + "_", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)


if importlib.util.find_spec("torch") is None:
    collect_ignore = [*TORCH_TEST_MODULES, *TORCH_TEST_DIRS]


def _needs_torch(path: Path) -> bool:
    path = path.resolve()
    return path in _TORCH_PATHS or any(path.is_relative_to(d) for d in _TORCH_DIRS)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _needs_torch(item.path):
            item.add_marker(pytest.mark.torch)
