"""A static scan of the model-side code (``src/drivedit``, ``tools/codec``) for writers that could leave the write
boundary unnoticed (DD-23). Torch-free: it parses, never imports.

- No ``/tmp`` fallback: every ``tempfile`` call that creates something names its ``dir=``, nothing asks for the system
  temp dir (``gettempdir``; only ``process_paths.prepare_process`` reads it back, to verify ``TMPDIR``), and no string
  is a system temp path (``/tmp``, ``/var/tmp``) or a path under the home directory (``expanduser``, ``Path.home``).
- No environment changes in library code: ``os.environ`` is never assigned, deleted or updated, ``os.putenv`` and
  ``tempfile.tempdir`` never set, except in ``drivedit/process_paths.py``, whose ``prepare_process`` entry points call.
- Every command line (a module with ``main`` and a ``__main__`` guard, or a ``[project.scripts]`` target) calls
  ``prepare_process``, or is listed in ``NO_PROCESS_PATHS`` with the reason it needs none.
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "src" / "drivedit", REPO / "tools" / "codec")
TEMPFILE_MAKERS = {
    "mkdtemp",
    "mkstemp",
    "TemporaryDirectory",
    "NamedTemporaryFile",
    "TemporaryFile",
    "SpooledTemporaryFile",
}
TEMPFILE_SYSTEM = {"gettempdir", "gettempdirb"}
ENV_MUTATORS = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__"}
PROCESS_PATHS = REPO / "src" / "drivedit" / "process_paths.py"
# Command lines that need no prepare_process, and why.
NO_PROCESS_PATHS = {
    "src/drivedit/data/build.py": "numpy and OpenCV only (no library cache); build_session checks every output path",
    "src/drivedit/eval/e0.py": "writes nothing: its report goes to stdout",
}


def _sources() -> Iterator[Path]:
    for root in ROOTS:
        yield from sorted(root.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _docstrings(tree: ast.AST) -> set[int]:
    """ids of the docstring constants (module, class, function), which may name /tmp in prose."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


def _tempfile_names(tree: ast.AST) -> dict[str, str]:
    """Local names bound to tempfile functions by ``from tempfile import ...``."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tempfile":
            for alias in node.names:
                out[alias.asname or alias.name] = alias.name
    return out


def _tempfile_call(node: ast.Call, imported: dict[str, str]) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "tempfile":
        return func.attr
    if isinstance(func, ast.Name) and func.id in imported:
        return imported[func.id]
    return None


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _call_findings(node: ast.Call, imported: dict[str, str], allowed_env: bool) -> list[str]:
    found = []
    name = _tempfile_call(node, imported)
    if name in TEMPFILE_MAKERS and not any(k.arg == "dir" for k in node.keywords):
        found.append(f"tempfile.{name}() without dir= falls back to the system temp dir")
    if name in TEMPFILE_SYSTEM and not allowed_env:
        found.append(f"tempfile.{name}() is the system temp dir; use process_paths.scratch_dir()")
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in ENV_MUTATORS and _is_os_environ(func.value) and not allowed_env:
            found.append(f"os.environ.{func.attr}() in library code")
        if func.attr in ("putenv", "unsetenv") and isinstance(func.value, ast.Name) and func.value.id == "os":
            found.append(f"os.{func.attr}()")
        if func.attr in ("expanduser", "home"):
            found.append(f"{func.attr}(): a path under the home directory (the root filesystem)")
    return found


def _target_findings(target: ast.expr, allowed_env: bool) -> list[str]:
    if allowed_env:
        return []
    if isinstance(target, ast.Subscript) and _is_os_environ(target.value):
        return ["os.environ[...] changed in library code"]
    if (
        isinstance(target, ast.Attribute)
        and target.attr == "tempdir"
        and isinstance(target.value, ast.Name)
        and target.value.id == "tempfile"
    ):
        return ["tempfile.tempdir set in library code"]
    return []


def _import_findings(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Aliases that would hide a call from this scan: ``import tempfile as t``, ``import os as o``,
    ``from os import environ`` (or ``putenv``, ``unsetenv``)."""
    if isinstance(node, ast.Import):
        return [
            f"import {a.name} as {a.asname}: an alias hides its calls from this scan"
            for a in node.names
            if a.name in ("os", "tempfile") and a.asname not in (None, a.name)
        ]
    if node.module == "os":
        return [
            f"from os import {a.name}: use os.{a.name}, which this scan sees"
            for a in node.names
            if a.name in ("environ", "putenv", "unsetenv")
        ]
    return []


def _findings(path: Path, label: str | None = None) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docs = _docstrings(tree)
    imported = _tempfile_names(tree)
    allowed_env = path == PROCESS_PATHS
    found = []
    for node in ast.walk(tree):
        what: list[str] = []
        if isinstance(node, ast.Call):
            what = _call_findings(node, imported, allowed_env)
        elif isinstance(node, (ast.Assign, ast.Delete)):
            what = [f for target in node.targets for f in _target_findings(target, allowed_env)]
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            what = _target_findings(node.target, allowed_env)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            what = _import_findings(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            text = node.value
            if text in ("/tmp", "/var/tmp") or text.startswith(("/tmp/", "/var/tmp/", "~/")):
                what = [f"{text!r} names the system temp dir or the home directory"]
        found += [f"{label or _rel(path)}:{getattr(node, 'lineno', 0)}: {f}" for f in what]
    return found


def test_the_scan_covers_the_model_side() -> None:
    names = {_rel(p) for p in _sources()}
    for need in (
        "src/drivedit/train/trainer.py",
        "src/drivedit/train/bench.py",
        "src/drivedit/codec/encode.py",
        "src/drivedit/codec/base.py",
        "tools/codec/measure_codecs.py",
    ):
        assert need in names


@pytest.mark.parametrize("path", list(_sources()), ids=_rel)
def test_no_writer_falls_back_to_the_system_temp_dir_or_changes_the_environment(path: Path) -> None:
    assert _findings(path) == []


def test_the_scan_finds_what_it_looks_for(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""Docstrings may say /tmp."""\n'
        "import os, tempfile\n"
        "from tempfile import mkdtemp as md\n"
        "tempfile.TemporaryDirectory()\n"
        "md(prefix='x')\n"
        "tempfile.mkdtemp(dir=base)\n"
        "tempfile.gettempdir()\n"
        "os.environ['HF_HOME'] = 'x'\n"
        "os.environ.setdefault('A', 'b')\n"
        "del os.environ['A']\n"
        "tempfile.tempdir = 'x'\n"
        "open('/tmp/x', 'w')\n"
        "os.path.expanduser('~/x')\n"
        "import tempfile as tf\n"
        "from os import environ\n",
        encoding="utf-8",
    )
    found = _findings(sample, "sample.py")
    lines = sorted(int(f.split(":")[1]) for f in found)
    assert lines == [4, 5, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15], found  # line 6 names its dir=; 13 holds two


def _commands() -> Iterator[Path]:
    """Every command line of the model side: modules with a ``main`` and a ``__main__`` guard, and script targets."""
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        if "def main(" in text and '__name__ == "__main__"' in text:
            yield path
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    for target in scripts.values():
        module = target.split(":")[0]
        yield REPO / "src" / Path(*module.split(".")).with_suffix(".py")


def _calls_prepare_process(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name == "prepare_process":
                return True
    return False


@pytest.mark.parametrize("path", sorted(set(_commands())), ids=_rel)
def test_every_command_line_routes_its_process_paths(path: Path) -> None:
    rel = _rel(path)
    assert _calls_prepare_process(path) != (rel in NO_PROCESS_PATHS), (
        f"{rel}: call drivedit.process_paths.prepare_process() in main, or list it in NO_PROCESS_PATHS with a reason"
    )
