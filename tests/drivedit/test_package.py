"""The drivedit package installs, has a version, and imports without torch (DD-18)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
TORCH_FREE = ("drivedit", "drivedit.data", "drivedit.eval")


def _run(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )


def test_torch_free_packages_import_without_torch(tmp_path: Path) -> None:
    imports = "; ".join(f"import {name}" for name in TORCH_FREE)
    report = "print(sorted(m for m in sys.modules if m.split('.')[0] == 'torch'))"
    code = f"import sys; {imports}; {report}"
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_version_is_set(tmp_path: Path) -> None:
    result = _run("import drivedit; print(drivedit.__version__)", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
