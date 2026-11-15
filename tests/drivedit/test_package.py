"""The drivedit package installs, has a version, and imports without torch (DD-18)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
TORCH_FREE = ("drivedit", "drivedit.data", "drivedit.eval")
# Import nothing heavy either (DD-23): the launcher and the lazily loading packages of the command modules.
LIGHT = ("drivedit.process_paths", "drivedit.commands", "drivedit.train", "drivedit.codec")


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


def test_the_launcher_and_the_command_packages_import_without_torch(tmp_path: Path) -> None:
    imports = "; ".join(f"import {name}" for name in LIGHT)
    report = "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'huggingface_hub')))"
    result = _run(f"import sys; {imports}; {report}", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_lazy_exports_are_listed_and_resolve(tmp_path: Path) -> None:
    """dir() lists every export before first use; each resolves to its defining module's object (needs torch, so in a
    subprocess only where torch is installed)."""
    code = (
        "import importlib.util, drivedit.train as t, drivedit.codec as c\n"
        "print(set(t.__all__) <= set(dir(t)), set(c.__all__) <= set(dir(c)))\n"
        "if importlib.util.find_spec('torch'):\n"
        "    from drivedit.train.trainer import Trainer; from drivedit.codec.encode import encode_session\n"
        "    assert t.Trainer is Trainer and c.encode_session is encode_session\n"
        "try:\n"
        "    t.NoSuchName\n"
        "except AttributeError as exc:\n"
        "    print('NoSuchName' in str(exc))\n"
    )
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True", "True"]
