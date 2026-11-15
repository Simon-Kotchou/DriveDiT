"""Fixtures shared by the drivedit tests. Torch-free.

``process_env``: the process paths of ``drivedit.process_paths`` (caches, ``TMPDIR``) pointed into the test's own
scratch (``tests/drivedit/boundary.py`` ``set_process_env``), for tests that run an entry point's ``main``, which calls
``prepare_process`` first. Without it ``main`` would find the variables unset and refuse their defaults: the tests'
synthetic local config names a data drive that is mounted nowhere (DD-22).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.drivedit.boundary import set_process_env


@pytest.fixture
def process_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return set_process_env(monkeypatch, tmp_path / "process")
