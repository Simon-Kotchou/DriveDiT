"""``--help`` and usage errors of the model commands work without a local config and without any cache variable: only
work that writes needs ``prepare_process`` (DD-23). Each command runs in a subprocess, as a user would start it
(``drivedit.commands``), with no config and every process path unset except a RAM ``TMPDIR``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from drivedit.commands import COMMANDS
from drivedit.process_paths import PROCESS_PATHS
from drivedit.train.cli import main
from tests.drivedit.boundary import clear_process_env, world

REPO = Path(__file__).resolve().parents[3]
CALL = "import sys; sys.path.insert(0, sys.argv[1]); from drivedit.commands import main; sys.exit(main(sys.argv[2:]))"


def _run(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    names = {n for e in PROCESS_PATHS for n in (e.name, e.legacy) if n is not None}
    env = {k: v for k, v in os.environ.items() if k not in names}
    env["TMPDIR"] = str(tmp_path)  # RAM; nothing here should use it
    env["DRIVEDIT_LOCAL_CONFIG"] = str(tmp_path / "no-such.local.env")  # no config
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", CALL, str(REPO / "src"), *argv],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_help_works_without_a_config(tmp_path: Path, command: str) -> None:
    result = _run(tmp_path, command, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage:"), result.stdout
    assert "XDG_CACHE_HOME" not in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_usage_error_is_argparse_not_a_refusal(tmp_path: Path, command: str) -> None:
    result = _run(tmp_path, command, "--no-such-option")
    assert result.returncode == 2, result.stderr
    assert ": error: " in result.stderr  # argparse: unrecognised option, or dd-encode's missing shard_dir first
    assert "refusing" not in result.stderr
    assert "XDG_CACHE_HOME" not in result.stderr


def test_print_config_writes_nothing_and_needs_no_config(tmp_path: Path) -> None:
    result = _run(tmp_path, "dd-train", "--print-config", "probe_50m")
    assert result.returncode == 0, result.stderr
    assert '"run_id": "probe_50m"' in result.stdout


def test_work_without_a_config_is_refused_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the fake world (no skip where a sandbox hides /dev/shm's device): no config, no cache variable."""
    with world("missing") as w:
        clear_process_env(monkeypatch)
        run_dir = Path(w.shm, "run")
        refusal = r"^dd-train: XDG_CACHE_HOME is not set, and its default <DATA_ROOT>/cache"
        with pytest.raises(SystemExit, match=refusal):
            main(["--config", str(Path(w.shm, "run.json")), "--run-dir", str(run_dir)])
        assert not run_dir.exists()
