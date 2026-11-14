"""This machine's paths for the drivedit package (DD-22).

The repository names no machine path. The data root, the code root and the drives live in the untracked local
config, ``drivedit.local.env``, read by the repository's ``tools/local_config.py``: the one loader every capture tool
uses too (its lookup order and format are documented there). The package runs from the source tree, so the loader is
found by path, and nothing is read before a value is asked for. A missing config or key raises ``LocalPathsError``,
which names the file and the key; no path is ever guessed.

The package writes through the same write boundary as every tool: ``check_write`` is the loader's ``check_write``
(RAM, or the verified drive of DATA_ROOT; never a volume whose root holds Windows/System32; caller roots only narrow
it), so there is one implementation, driven by one table of test vectors
(``tests/fixtures/local_config/boundary_vectors.py``) through both entry points. ``check_dir`` is the same check for a
directory that files are written into; ``data_root`` and ``ram_root`` are the two places the boundary allows (DD-23).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

__all__ = [
    "DIR_PROBE",
    "LOADER",
    "LocalPathsError",
    "WriteRefusedError",
    "check_dir",
    "check_write",
    "data_root",
    "drive",
    "ram_root",
    "value",
]

# src/drivedit/local_paths.py -> the repository root, two levels above the package.
LOADER = Path(__file__).resolve().parents[2] / "tools" / "local_config.py"
_MODULE = "dd_local_config"


class LocalPathsError(RuntimeError):
    """The local config is missing or malformed, or lacks the key asked for."""


class WriteRefusedError(LocalPathsError):
    """The write boundary refuses a destination (the message says why)."""


def _loader() -> ModuleType:
    module = sys.modules.get(_MODULE)
    if module is not None and getattr(module, "__file__", None) == str(LOADER):
        return module
    if not LOADER.is_file():
        raise LocalPathsError(f"{LOADER} is missing: the package reads this machine's paths through it")
    spec = importlib.util.spec_from_file_location(_MODULE, LOADER)
    if spec is None or spec.loader is None:
        raise LocalPathsError(f"cannot load {LOADER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _call(form: str, key: str) -> str:
    module = _loader()
    try:
        return str(getattr(module.load(), form)(key))
    except module.LocalConfigError as exc:
        raise LocalPathsError(str(exc)) from exc


def value(key: str) -> str:
    """A key of the local config; a path in its WSL form (``/mnt/<x>/...``)."""
    return _call("get", key)


def drive(key: str) -> str:
    """The drive mount (``/mnt/<x>``) of a path key of the local config."""
    return _call("drive", key)


def check_write(path: str | Path, roots: Sequence[str | Path] | None = None) -> str:
    """The resolved destination when the write boundary allows writing to ``path``; ``WriteRefusedError`` otherwise.
    ``roots`` (the caller's own roots) can only narrow the boundary."""
    module = _loader()
    try:
        return str(module.check_write(str(path), None if roots is None else [str(r) for r in roots]))
    except module.WriteRefused as exc:
        raise WriteRefusedError(str(exc)) from exc


#: The file name ``check_dir`` checks inside a directory; nothing is ever created under that name.
DIR_PROBE = ".drivedit-write-check"


def check_dir(path: str | Path, roots: Sequence[str | Path] | None = None) -> Path:
    """The resolved directory when the write boundary allows writing files inside ``path`` (existing or not);
    ``WriteRefusedError`` otherwise. A file name inside it is checked, so a directory that is a symlink is judged by
    where it leads, as the files written through it are (``check_write`` refuses a symlink as the last component of a
    file path: a write would follow it)."""
    text = os.fspath(path)
    if not text:
        raise WriteRefusedError("refusing to write into '': not a directory path")
    return Path(check_write(os.path.join(text, DIR_PROBE), roots)).parent


def data_root() -> Path:
    """DATA_ROOT of the local config where it is mounted (the loader's ``mounted_path``: the same path, except in the
    fake worlds of the tests); ``LocalPathsError`` without a config or the key."""
    return Path(_loader().mounted_path(value("DATA_ROOT")))


def ram_root() -> Path:
    """The RAM mount the write boundary allows with or without a config (the loader's ``SCRATCH_MOUNT``, /dev/shm)."""
    return Path(_loader().SCRATCH_MOUNT)
