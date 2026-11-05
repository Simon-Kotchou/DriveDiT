"""drivedit.eval and its numpy metric modules import without torch; only drivedit.eval.idm needs it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
NUMPY_MODULES = (
    "drivedit.eval",
    "drivedit.eval.linear",
    "drivedit.eval.strata",
    "drivedit.eval.e0",
    "drivedit.eval.control",
    "drivedit.eval.probes",
)
# Makes any torch import fail, so a module that needs torch cannot hide behind an installed one.
BLOCK_TORCH = """
import importlib.abc, sys
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked")
        return None
sys.meta_path.insert(0, _NoTorch())
"""


def _run(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )


def test_numpy_metric_modules_import_with_torch_blocked(tmp_path: Path) -> None:
    imports = "; ".join(f"import {name}" for name in NUMPY_MODULES)
    report = "print(sorted(m for m in sys.modules if m.split('.')[0] == 'torch' or m == 'drivedit.eval.idm'))"
    result = _run(f"{BLOCK_TORCH}\n{imports}\n{report}", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_idm_module_is_the_one_that_needs_torch(tmp_path: Path) -> None:
    result = _run(f"{BLOCK_TORCH}\nimport drivedit.eval.idm", tmp_path)
    assert result.returncode != 0
    assert "torch is blocked" in result.stderr
