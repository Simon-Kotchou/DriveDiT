"""``drivedit.commands``: the console scripts route the process paths before torch is imported (DD-23), because
importing torch already writes when ``TORCH_LOGS_OUT`` is set. Torch-free: a refused command never imports torch."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from drivedit import commands
from drivedit.process_paths import PROCESS_PATHS
from tests.drivedit.boundary import NEVER, assert_never_written

REPO = Path(__file__).resolve().parents[2]
PROBE = (
    "import sys; sys.path.insert(0, sys.argv[1]); from drivedit import commands\n"
    "try:\n"
    "    commands.main(sys.argv[2:])\n"
    "except SystemExit as exc:\n"
    "    print(exc)\n"
    "print('torch imported' if 'torch' in sys.modules else 'torch not imported')\n"
)


def test_the_console_scripts_go_through_the_bootstrap() -> None:
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {"dd-encode": "drivedit.commands:dd_encode", "dd-train": "drivedit.commands:dd_train"}
    assert set(commands.COMMANDS) == {"dd-train", "dd-encode", "bench"}


def test_an_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit, match="usage"):
        commands.main(["train"])
    with pytest.raises(SystemExit, match="usage"):
        commands.main([])


@pytest.mark.parametrize("command", sorted(commands.COMMANDS))
def test_a_refusal_comes_before_torch_is_imported(tmp_path: Path, command: str) -> None:
    """Even for --help: torch would open TORCH_LOGS_OUT at import. No config or cache variable is needed for it."""
    names = {n for e in PROCESS_PATHS for n in (e.name, e.legacy) if n is not None}
    env = {k: v for k, v in os.environ.items() if k not in names}
    env["TMPDIR"] = str(tmp_path)
    env["TORCH_LOGS_OUT"] = NEVER + "/torch.log"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROBE, str(REPO / "src"), command, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    first, last = result.stdout.strip().splitlines()[0], result.stdout.strip().splitlines()[-1]
    assert first.startswith(f"{command}: TORCH_LOGS_OUT={NEVER}/"), result.stdout
    assert last == "torch not imported"
    assert_never_written()


MODULE_PROBE = (
    "import runpy, sys; sys.path.insert(0, sys.argv[1]); target = sys.argv[2]; sys.argv = [target, '--help']\n"
    "try:\n"
    "    if target.endswith('.py'):\n"
    "        runpy.run_path(target, run_name='__main__')\n"
    "    else:\n"
    "        runpy.run_module(target, run_name='__main__', alter_sys=True)\n"
    "except SystemExit as exc:\n"
    "    print(exc)\n"
    "print('torch imported' if 'torch' in sys.modules else 'torch not imported')\n"
)
MODULES = {
    "drivedit.train.cli": "dd-train",
    "drivedit.train.bench": "bench",
    "drivedit.codec.encode": "dd-encode",
    str(REPO / "tools" / "codec" / "measure_codecs.py"): "measure_codecs",
}


@pytest.mark.parametrize("target", sorted(MODULES))
@pytest.mark.parametrize("value", [NEVER + "/torch.log", " "])
def test_the_command_modules_check_before_torch_too(tmp_path: Path, target: str, value: str) -> None:
    """python -m drivedit.train.cli (and the others): the packages load lazily, so the module's own guard runs before
    torch is imported. A blank TORCH_LOGS_OUT is removed (torch would open a file named by it), an unsafe one
    refused."""
    names = {n for e in PROCESS_PATHS for n in (e.name, e.legacy) if n is not None}
    env = {k: v for k, v in os.environ.items() if k not in names}
    env |= {"TMPDIR": str(tmp_path), "TORCH_LOGS_OUT": value}
    if value.strip():
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", MODULE_PROBE, str(REPO / "src"), target],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert lines[0].startswith(f"{MODULES[target]}: TORCH_LOGS_OUT={NEVER}/"), result.stdout
        assert lines[-1] == "torch not imported"
    else:  # removed before torch reads it: no file named " " in the working directory
        code = "import os, sys; sys.path.insert(0, sys.argv[1]); from drivedit import process_paths\n"
        code += "process_paths.guard_module_main('__main__', 'x'); print(repr(os.environ.get('TORCH_LOGS_OUT')))\n"
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(REPO / "src")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "None"
        assert list(tmp_path.iterdir()) == []
    assert_never_written()
