"""Where a process's caches and temporary files go (DD-23, ``drivedit.process_paths``), and the directory checks of
``drivedit.local_paths`` it builds on. Torch-free. Every test runs in a DD-22 fake world
(``tests/drivedit/boundary.py``), so nothing depends on, or is written to, a real drive, and nothing skips."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from drivedit import local_paths
from drivedit.local_paths import LocalPathsError, WriteRefusedError
from drivedit.process_paths import DATA, PROCESS_PATHS, RAM, command_line, prepare_process, resolve, scratch_dir
from tests.drivedit.boundary import NEVER, assert_never_written, world

SRC = Path(__file__).resolve().parents[2] / "src"
UID = os.getuid()
MANAGED = [entry.name for entry in PROCESS_PATHS if entry.managed]


def _expected_defaults(w: Any) -> dict[str, str]:
    cache = f"{w.mnt}/x/DataRoot/cache"
    tmp = f"{w.shm}/drivedit-tmp-{UID}"
    return {
        "TMPDIR": tmp,
        "XDG_CACHE_HOME": cache,
        "HF_HOME": f"{cache}/huggingface",
        "HF_HUB_CACHE": f"{cache}/huggingface/hub",
        "HF_XET_CACHE": f"{cache}/huggingface/xet",
        "TORCH_HOME": f"{cache}/torch",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor",
        "TRITON_HOME": f"{cache}/triton",
        "CUDA_CACHE_PATH": f"{cache}/nv/ComputeCache",
        "TORCH_FR_DUMP_TEMP_FILE": f"{tmp}/nccl_trace_rank_",
    }


def test_the_table_lists_each_base_before_what_follows_it() -> None:
    seen: set[str] = set()
    for entry in PROCESS_PATHS:
        assert entry.base in (DATA, RAM) or entry.base in seen, entry.name
        if entry.managed:
            seen.add(entry.name)
    assert len({e.name for e in PROCESS_PATHS}) == len(PROCESS_PATHS)


def test_defaults_go_to_the_data_drive_and_ram() -> None:
    with world() as w:
        environ: dict[str, str] = {}
        values = prepare_process(environ)
        assert values == environ == _expected_defaults(w)
        assert sorted(values) == sorted(MANAGED)
        tmp = Path(values["TMPDIR"])
        assert tmp.is_dir()
        assert stat.S_IMODE(tmp.stat().st_mode) == 0o700
        assert not Path(w.mnt, "x", "DataRoot", "cache").exists()  # caches are created by their libraries, on use


def test_a_set_variable_is_kept_and_what_follows_it_follows_it() -> None:
    with world() as w:
        hf, xdg = f"{w.mnt}/x/DataRoot/hf", f"{w.shm}/xdg"
        environ = {"HF_HOME": hf, "XDG_CACHE_HOME": xdg}
        prepare_process(environ)
        assert environ["HF_HOME"] == hf
        assert environ["XDG_CACHE_HOME"] == xdg
        assert environ["HF_HUB_CACHE"] == f"{hf}/hub"
        assert environ["HF_XET_CACHE"] == f"{hf}/xet"
        assert environ["TORCH_HOME"] == f"{xdg}/torch"
        assert environ["TRITON_HOME"] == f"{xdg}/triton"


def test_legacy_names_keep_their_effect() -> None:
    with world() as w:
        hub, fr = f"{w.mnt}/x/DataRoot/legacy_hub", f"{w.shm}/fr/nccl_"
        environ = {"HUGGINGFACE_HUB_CACHE": hub, "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": fr}
        prepare_process(environ)
        assert environ["HF_HUB_CACHE"] == hub
        assert environ["TORCH_FR_DUMP_TEMP_FILE"] == fr


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_value_counts_as_unset(value: str) -> None:
    with world() as w:
        environ = {"HF_HOME": value}
        prepare_process(environ)
        assert environ["HF_HOME"] == _expected_defaults(w)["HF_HOME"]


@pytest.mark.parametrize(
    ("name", "value", "fragment"),
    [
        ("HF_HOME", NEVER + "/hf", "refusing to write"),
        ("HF_HUB_CACHE", "{M}/z/hub", "Windows system drive"),
        ("HUGGINGFACE_HUB_CACHE", "{M}/z/hub", "Windows system drive"),
        ("XDG_CACHE_HOME", "{M}/y/cache", "refusing to write"),
        ("TORCH_HOME", "{B}/elsewhere/torch", "refusing to write"),
        ("TORCHINDUCTOR_CACHE_DIR", "~/.cache/inductor", "not an absolute path"),
        ("TRITON_HOME", "relative/triton", "not an absolute path"),
        ("CUDA_CACHE_PATH", "{B}/shm/$HOME/nv", "not an absolute path"),
        ("TRITON_CACHE_DIR", "{M}/z/triton", "Windows system drive"),
        ("TORCH_EXTENSIONS_DIR", NEVER + "/ext", "refusing to write"),
        ("TORCH_FR_DUMP_TEMP_FILE", "{M}/z/nccl_trace_rank_", "Windows system drive"),
        ("TORCHELASTIC_ERROR_FILE", NEVER + "/torchelastic_x/error.json", "refusing to write"),
        ("TMPDIR", NEVER, "refusing to write"),
        ("TMPDIR", "{B}/shm/no_such_dir", "not an existing writable directory"),
        ("HF_XET_CACHE", "{B}/shm/../mnt/z/xet", "'..'"),
    ],
)
def test_a_value_outside_the_boundary_is_refused_and_nothing_changes(name: str, value: str, fragment: str) -> None:
    with world() as w:
        environ = {name: w.path(value)}
        before = dict(environ)
        with pytest.raises(WriteRefusedError, match=name) as info:
            prepare_process(environ)
        assert fragment in str(info.value)
        assert environ == before
        assert not Path(_expected_defaults(w)["TMPDIR"]).exists()
    assert_never_written()


@pytest.mark.parametrize(
    ("state", "fragment"),
    [
        ("placeholder", "is not mounted"),  # the synthetic config: its data drive is mounted nowhere
        ("swapped", "Windows system drive"),  # DATA_ROOT on the system volume
        ("partial", "not the whole drive"),
    ],
)
def test_a_default_on_an_unverified_data_drive_is_refused(state: str, fragment: str) -> None:
    with world(state):
        environ: dict[str, str] = {}
        with pytest.raises(WriteRefusedError, match=r"XDG_CACHE_HOME=.* \(its default\)") as info:
            prepare_process(environ)
        assert fragment in str(info.value)
        assert environ == {}


def test_without_a_config_the_caches_must_be_set() -> None:
    with world("missing") as w:
        environ: dict[str, str] = {}
        with pytest.raises(LocalPathsError, match="XDG_CACHE_HOME is not set, and its default <DATA_ROOT>/cache"):
            prepare_process(environ)
        assert environ == {}
        environ = {"XDG_CACHE_HOME": f"{w.shm}/xdg"}  # RAM needs no config
        values = prepare_process(environ)
        assert values["HF_HUB_CACHE"] == f"{w.shm}/xdg/huggingface/hub"
        assert values["TMPDIR"] == f"{w.shm}/drivedit-tmp-{UID}"


def test_prepare_process_points_os_environ_and_tempfile_into_the_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    for entry in PROCESS_PATHS:
        for name in (entry.name, entry.legacy):
            if name is not None:
                monkeypatch.delenv(name, raising=False)  # undone afterwards, also for what prepare_process sets
    monkeypatch.setattr(tempfile, "tempdir", "/never/used/temp/dir")
    with world() as w:
        values = prepare_process()
        assert {name: os.environ[name] for name in values} == _expected_defaults(w)
        assert tempfile.gettempdir() == values["TMPDIR"]
        with tempfile.TemporaryDirectory() as made:
            assert os.path.dirname(made) == values["TMPDIR"]


def test_scratch_dir_is_tmpdir_or_its_ram_default_never_the_system_temp_dir() -> None:
    with world() as w:
        default = Path(f"{w.shm}/drivedit-tmp-{UID}")
        assert scratch_dir({}) == default
        assert default.is_dir()
        assert scratch_dir({"TMPDIR": f"{w.shm}/real"}) == Path(f"{w.shm}/real")
        with pytest.raises(WriteRefusedError, match="TMPDIR"):
            scratch_dir({"TMPDIR": NEVER})
        with pytest.raises(WriteRefusedError, match="TMPDIR"):
            scratch_dir({"TMPDIR": f"{w.mnt}/z"})
    assert_never_written()


def test_resolve_changes_nothing() -> None:
    with world() as w:
        environ: dict[str, str] = {}
        assert resolve(environ) == _expected_defaults(w)
        assert environ == {}
        assert not Path(_expected_defaults(w)["TMPDIR"]).exists()


def test_command_line_turns_a_refusal_into_one_line() -> None:
    with pytest.raises(SystemExit, match=r"^prog: refusing it$"), command_line("prog"):
        raise WriteRefusedError("refusing it")
    with pytest.raises(SystemExit, match=r"^prog: no config$"), command_line("prog"):
        raise LocalPathsError("no config")
    with pytest.raises(ValueError, match="other"), command_line("prog"):
        raise ValueError("other")


# ---------------------------------------------------------------------------------------- drivedit.local_paths


def test_check_dir_judges_a_directory_by_where_its_files_land() -> None:
    with world() as w:
        assert local_paths.check_dir(f"{w.mnt}/x/DataRoot/new/dir") == Path(f"{w.mnt}/x/DataRoot/new/dir")
        assert local_paths.check_dir(f"{w.shm}/dir_in_ram") == Path(f"{w.shm}/real")  # a symlink into RAM
        for refused in (f"{w.shm}/dir_to_system", f"{w.mnt}/z/out", f"{w.shm}/dir_to_tmp", NEVER, ""):
            with pytest.raises(WriteRefusedError):
                local_paths.check_dir(refused)
        assert not Path(w.mnt, "x", "DataRoot", "new").exists()


def test_data_root_and_ram_root_are_where_the_boundary_looks() -> None:
    with world() as w:
        assert local_paths.data_root() == Path(f"{w.mnt}/x/DataRoot")
        assert local_paths.ram_root() == Path(w.shm)
    with world("missing"), pytest.raises(LocalPathsError, match=r"DATA_ROOT|no local config|not a file"):
        local_paths.data_root()


def test_importing_the_package_changes_no_environment(tmp_path: Path) -> None:
    """No import-time side effects: importing every torch-free drivedit module leaves os.environ and Python's temp
    dir alone (only an entry point calls prepare_process)."""
    code = (
        "import os, sys, tempfile; sys.path.insert(0, sys.argv[1]); before = dict(os.environ); "
        "import drivedit, drivedit.local_paths, drivedit.process_paths, drivedit.data, drivedit.eval; "
        "print(dict(os.environ) == before, tempfile.tempdir is None)"
    )
    env = {k: v for k, v in os.environ.items() if k not in MANAGED}
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(SRC)], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]
