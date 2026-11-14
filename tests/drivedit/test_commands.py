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
from tests.drivedit.boundary import NEVER, assert_never_written, require_real_boundary

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
    base = require_real_boundary(tmp_path)  # the caches below lie in RAM scratch the real boundary must allow
    names = {n for e in PROCESS_PATHS for n in (e.name, e.legacy) if n is not None}
    env = {k: v for k, v in os.environ.items() if k not in names}
    (base / "tmp").mkdir()
    env |= {"TMPDIR": str(base / "tmp"), "XDG_CACHE_HOME": str(base / "cache")}
    env["TORCH_LOGS_OUT"] = NEVER + "/torch.log"  # torch would open it at import
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROBE, str(REPO / "src"), command, "--help"],
        cwd=base,
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
