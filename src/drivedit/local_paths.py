"""This machine's paths for the drivedit package (DD-22).

The repository names no machine path. The data root, the code root and the drives live in the untracked local
config, ``drivedit.local.env``, read by the repository's ``tools/local_config.py``: the one loader every capture tool
uses too (its lookup order and format are documented there). The package runs from the source tree, so the loader is
found by path, and nothing is read before a value is asked for. A missing config or key raises ``LocalPathsError``,
which names the file and the key; no path is ever guessed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["LOADER", "LocalPathsError", "drive", "value"]

# src/drivedit/local_paths.py -> the repository root, two levels above the package.
LOADER = Path(__file__).resolve().parents[2] / "tools" / "local_config.py"
_MODULE = "dd_local_config"


class LocalPathsError(RuntimeError):
    """The local config is missing or malformed, or lacks the key asked for."""


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
