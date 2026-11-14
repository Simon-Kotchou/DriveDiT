"""``python -m drivedit.train.bench`` keeps its trainer's run directory in checked scratch, never the system temp dir,
and routes the process's caches into the write boundary first (DD-23). Runs in a DD-22 fake world on a tiny model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drivedit.train import bench
from drivedit.train.trainer import Trainer
from tests.drivedit.boundary import NEVER, assert_never_written, set_process_env, world
from tests.drivedit.model.toy import tiny_config

TINY = tiny_config()
ARGV = [
    "--preset",
    "tiny",
    "--latent",
    *(str(n) for n in TINY.latent_shape),
    "--window",
    str(TINY.window),
    "--batch",
    "2",
    "--steps",
    "1",
    "--warmup",
    "0",
    "--precision",
    "fp32",
    "--device",
    "cpu",
]


@pytest.fixture
def tiny_preset(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """A ``tiny`` preset, and the run directories the bench hands its trainer."""
    monkeypatch.setitem(bench.PRESETS, "tiny", tiny_config)
    seen: list[Path] = []

    def spy(*args: object, **kwargs: object) -> Trainer:
        seen.append(Path(str(args[2])))
        return Trainer(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(bench, "Trainer", spy)
    return seen


def test_the_run_dir_is_temporary_scratch_under_tmpdir(
    tiny_preset: list[Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with world() as w:
        env = set_process_env(monkeypatch, Path(w.shm, "process"))
        assert bench.main(ARGV) == 0
        (run_dir,) = tiny_preset
        assert run_dir.parent == Path(env["TMPDIR"])
        assert run_dir.name.startswith("dd_bench_")
        assert not run_dir.exists()  # removed after the measurement
    assert json.loads(capsys.readouterr().out)["preset"] == "tiny"
    assert_never_written()


def test_a_tmpdir_outside_the_boundary_stops_it_first(tiny_preset: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    with world() as w:
        set_process_env(monkeypatch, Path(w.shm, "process"))
        monkeypatch.setenv("TMPDIR", NEVER)
        with pytest.raises(SystemExit, match=r"^bench: TMPDIR=/tmp/"):
            bench.main(ARGV)
        monkeypatch.setenv("TMPDIR", str(Path(w.mnt, "z")))
        with pytest.raises(SystemExit, match=r"^bench: TMPDIR=.*Windows system drive"):
            bench.main(ARGV)
    assert tiny_preset == []
    assert_never_written()
