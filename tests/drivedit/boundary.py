"""Helpers for the tests of the model-side writers and of the process paths (DD-23). Torch-free.

- ``world(state)``: a fresh DD-22 fake world (``tests/fixtures/local_config/boundary_vectors.py``: a fake mountinfo, a
  data drive ``<mnt>/x``, a Windows system drive ``<mnt>/z``, an unmounted drive ``<mnt>/y``, another tmpfs
  ``<base>/elsewhere``, RAM ``<base>/shm``) that the package's write boundary (``drivedit.local_paths``) sees for the
  block, under the config ``state``. Policy tests run there: they need no real mount and never skip.
- ``require_real_boundary(path)``: ``path``, or a skip naming the boundary's reason, for tests that write through the
  real boundary into RAM scratch (a sandbox may hide /dev/shm's device; the boundary then refuses RAM, fail-closed).
- ``set_process_env(monkeypatch, base)``: every managed variable of ``drivedit.process_paths.PROCESS_PATHS`` pointed
  into ``base`` and every other one unset, through ``monkeypatch``, so an entry point's ``prepare_process`` finds them
  set and changes nothing, and the test leaves the environment as it found it.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from drivedit import local_paths
from drivedit.process_paths import PROCESS_PATHS

REPO = Path(__file__).resolve().parents[2]
NEVER = "/tmp/dd23-never-written"  # a system temp dir path the tests pass as a destination; it must never appear


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bv = _load("dd23_boundary_vectors", REPO / "tests" / "fixtures" / "local_config" / "boundary_vectors.py")
LC = local_paths._loader()


@contextmanager
def world(state: str = "real_like") -> Iterator[Any]:
    """A fresh fake world, seen by ``drivedit.local_paths`` under the config ``state`` for the block."""
    with bv.world("dd23_world_") as w, w.use(LC, state):
        yield w


def require_real_boundary(path: Path) -> Path:
    """``path`` when the real write boundary allows writing in it, else a skip with the reason."""
    try:
        local_paths.check_dir(path)
    except local_paths.WriteRefusedError as exc:
        pytest.skip(f"the write boundary refuses RAM scratch here, fail-closed ({exc}); this test writes through it")
    return path


def set_process_env(monkeypatch: pytest.MonkeyPatch, base: Path) -> dict[str, str]:
    """Every managed process path set to a directory (or file prefix) under ``base``, ``TMPDIR`` created; every
    unmanaged one and every legacy name unset; Python's cached temp dir restored afterwards. Returns the values set."""
    values: dict[str, str] = {}
    for entry in PROCESS_PATHS:
        for name in (entry.name, entry.legacy):
            if name is not None:
                monkeypatch.delenv(name, raising=False)
        if entry.managed:
            values[entry.name] = str(base / entry.name) + ("_" if entry.kind == "prefix" else "")
    (base / "TMPDIR").mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    return values


def assert_never_written() -> None:
    assert not os.path.exists(NEVER), f"{NEVER} was created"
