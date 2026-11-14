"""Where a drivedit process's caches and temporary files go: inside the write boundary, never onto the Linux root
filesystem (DD-23).

The libraries of the model side pick their own write locations, outside every directory drivedit passes them:

| variable (``PROCESS_PATHS``) | written by | default without it |
|---|---|---|
| ``TMPDIR`` | ``tempfile``: torch.compile and Triton builds, multiprocessing, torchrun | ``/tmp`` |
| ``XDG_CACHE_HOME`` | torch (kernel cache, extensions); every default below follows it | ``~/.cache`` |
| ``HF_HOME``, ``HF_HUB_CACHE``, ``HF_XET_CACHE`` | huggingface_hub: the codec weights | ``~/.cache/huggingface`` |
| ``TORCH_HOME`` | torch.hub | ``~/.cache/torch`` |
| ``TORCHINDUCTOR_CACHE_DIR`` | torch.compile (``RunConfig.compile``, ``bench --compile``) | ``/tmp/torchinductor_*`` |
| ``TRITON_HOME`` | Triton, under ``.triton/`` (cache, dump, override) | ``~`` |
| ``CUDA_CACHE_PATH`` | the CUDA driver's JIT cache | ``~/.nv/ComputeCache`` |
| ``TORCH_FR_DUMP_TEMP_FILE`` | the NCCL flight recorder's dump on a DDP watchdog timeout | ``/tmp/nccl_trace_rank_`` |

On WSL ``~`` and ``/tmp`` lie on the Linux root filesystem, whose virtual disk is on the Windows system drive, so each
of them is an accident waiting for a full system drive.

``prepare_process()`` is for entry points (``dd-encode``, ``dd-train``, ``python -m drivedit.train.bench``,
``tools/codec/measure_codecs.py``), called before any work, and its values pass to child processes (DataLoader
workers, torchrun ranks, compile workers). For each variable of ``PROCESS_PATHS``:

- **Set** (non-empty): validated, never changed. It must be an absolute path (no ``~``, no ``$``) that the write
  boundary allows (``drivedit.local_paths.check_dir``: RAM or the verified data drive of the local config, never a
  Windows system volume); ``TMPDIR`` must also be an existing directory (else ``tempfile`` falls back to ``/tmp``
  without a word). The variables marked ``managed=False`` are only validated when set: their library default lies
  under a managed one.
- **Unset or empty**: set to its default, derived the way the library itself derives it from the variable it follows
  (``HF_HUB_CACHE`` = ``$HF_HOME/hub``, ...), so a variable the user did set keeps its effect on everything below it;
  a legacy name the library still honours (``HUGGINGFACE_HUB_CACHE``) is carried over. The two roots:
  ``XDG_CACHE_HOME`` = ``<DATA_ROOT>/cache`` (persistent caches go to the data drive; without a local config there
  is no default, and the refusal says to set it or add the config), and ``TMPDIR`` = ``<RAM>/drivedit-tmp-<uid>``
  (short-lived scratch in RAM, ``/dev/shm``, which the boundary allows without a config; created mode 0700). RAM,
  not the data drive, also because the drvfs data volume cannot hold the Unix sockets multiprocessing puts there.

Every value is computed and checked before anything changes: a refusal (``WriteRefusedError``, or
``LocalPathsError`` for a missing config) leaves ``os.environ`` as it was. Python caches its temp dir on first use, so
``prepare_process`` drops that cache and checks that ``tempfile.gettempdir()`` is the ``TMPDIR`` it validated.

Library code never calls ``prepare_process`` (no import-time side effects): it gets a checked directory for temporary
files from ``scratch_dir()``, which has no ``/tmp`` fallback, and it checks its own outputs with
``drivedit.local_paths.check_dir`` / ``check_write``. ``drivedit.codec.base.fetch_weights`` checks the Hugging Face
cache it is about to download into, so a process that skipped ``prepare_process`` is refused too.

Threat model: the write boundary's (``tools/local_config.py``): accidents (a wrong default, a forgotten variable), not
a hostile process. A check is a snapshot (no time-of-check to time-of-use guarantee); nested mounts and Windows
junctions are out of scope.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from drivedit import local_paths
from drivedit.local_paths import LocalPathsError, WriteRefusedError

__all__ = [
    "CACHE_DIR_REL",
    "DATA",
    "PROCESS_PATHS",
    "RAM",
    "ProcessPath",
    "command_line",
    "prepare_process",
    "resolve",
    "scratch_dir",
]

DATA: Final = "<DATA_ROOT>"  # a root default under DATA_ROOT of the local config (drivedit.local_paths.data_root)
RAM: Final = "<RAM>"  # a root default under the RAM mount (drivedit.local_paths.ram_root)
CACHE_DIR_REL: Final = "cache"  # XDG_CACHE_HOME defaults to <DATA_ROOT>/cache


@dataclass(frozen=True)
class ProcessPath:
    """An environment variable naming where a library writes.

    ``base`` is the variable the library derives its default from (``DATA`` or ``RAM`` for the roots) and ``child``
    what it appends (``{uid}`` becomes the user id). ``legacy`` is an older name the library reads when ``name`` is
    unset. ``kind`` is what the value names: a directory, a file, or a file-name prefix (the library appends a rank).
    ``managed=False``: validated when set, never set.
    """

    name: str
    base: str
    child: str
    kind: Literal["dir", "file", "prefix"] = "dir"
    legacy: str | None = None
    managed: bool = True


# In dependency order: a base comes before the variables that follow it. The derivations are the libraries' own
# (huggingface_hub.constants, torch.hub.get_dir, torch._appdirs, torch._inductor.runtime.cache_dir_utils,
# triton.knobs.cache_knobs, the CUDA driver, the c10d flight recorder); tests/drivedit/codec/test_process_paths_libs.py
# reads them back from the installed huggingface_hub, torch and Triton.
PROCESS_PATHS: Final = (
    ProcessPath("TMPDIR", RAM, "drivedit-tmp-{uid}"),
    ProcessPath("XDG_CACHE_HOME", DATA, CACHE_DIR_REL),
    ProcessPath("HF_HOME", "XDG_CACHE_HOME", "huggingface"),
    ProcessPath("HF_HUB_CACHE", "HF_HOME", "hub", legacy="HUGGINGFACE_HUB_CACHE"),
    ProcessPath("HF_XET_CACHE", "HF_HOME", "xet"),
    ProcessPath("HF_ASSETS_CACHE", "HF_HOME", "assets", legacy="HUGGINGFACE_ASSETS_CACHE", managed=False),
    ProcessPath("HF_MODULES_CACHE", "HF_HOME", "modules", managed=False),
    ProcessPath("TORCH_HOME", "XDG_CACHE_HOME", "torch"),
    ProcessPath("PYTORCH_KERNEL_CACHE_PATH", "XDG_CACHE_HOME", "torch/kernels", managed=False),
    ProcessPath("TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "torch_extensions", managed=False),
    ProcessPath("TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "torchinductor"),
    ProcessPath("TRITON_HOME", "XDG_CACHE_HOME", "triton"),
    ProcessPath("TRITON_CACHE_DIR", "TRITON_HOME", ".triton/cache", managed=False),
    ProcessPath("TRITON_DUMP_DIR", "TRITON_HOME", ".triton/dump", managed=False),
    ProcessPath("TRITON_OVERRIDE_DIR", "TRITON_HOME", ".triton/override", managed=False),
    ProcessPath("CUDA_CACHE_PATH", "XDG_CACHE_HOME", "nv/ComputeCache"),
    ProcessPath(
        "TORCH_FR_DUMP_TEMP_FILE", "TMPDIR", "nccl_trace_rank_", kind="prefix", legacy="TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
    ),
    # Set by torchrun for its ranks, in torchrun's own temp dir (TMPDIR of the shell that started it, or --log-dir).
    ProcessPath("TORCHELASTIC_ERROR_FILE", "TMPDIR", "error.json", kind="file", managed=False),
)
_BY_NAME: Final = {entry.name: entry for entry in PROCESS_PATHS}


def _given(environ: Mapping[str, str], name: str | None) -> str | None:
    """The value of ``name`` in ``environ``; None when it is unset, empty or blank (which count as unset)."""
    if name is None:
        return None
    value = environ.get(name, "")
    return value if value.strip() else None


def _check(entry: ProcessPath, value: str, origin: str) -> None:
    """WriteRefusedError, naming the variable, unless ``value`` is an absolute path the write boundary allows."""
    if not os.path.isabs(value) or "$" in value:
        raise WriteRefusedError(
            f"{entry.name}={value} ({origin}) is not an absolute path; set it to a directory in RAM (/dev/shm) or on "
            "the data drive of the local config, written out in full (no '~' or '$')"
        )
    try:
        if entry.kind == "dir":
            local_paths.check_dir(value)
        elif entry.kind == "file":
            local_paths.check_write(value)
        else:  # a prefix: the library writes <value><rank>
            local_paths.check_write(value + "0")
    except WriteRefusedError as exc:
        raise WriteRefusedError(f"{entry.name}={value} ({origin}): {exc}") from exc


def _default(entry: ProcessPath, resolved: Mapping[str, str]) -> str:
    child = entry.child.format(uid=os.getuid())
    if entry.base == RAM:
        return os.path.join(local_paths.ram_root(), child)
    if entry.base == DATA:
        try:
            return os.path.join(local_paths.data_root(), child)
        except LocalPathsError as exc:
            raise LocalPathsError(
                f"{entry.name} is not set, and its default <DATA_ROOT>/{child} needs the local config: {exc}. Set "
                f"{entry.name} to a directory in RAM (/dev/shm) or on the data drive, or add the local config"
            ) from exc
    return os.path.join(resolved[entry.base], child)


def _resolve_one(entry: ProcessPath, environ: Mapping[str, str], resolved: Mapping[str, str]) -> str | None:
    """The checked value ``entry.name`` has after prepare_process; None for an unset unmanaged variable."""
    given = _given(environ, entry.name)
    if given is not None:
        _check(entry, given, "set in the environment")
        if entry.name == "TMPDIR" and not (os.path.isdir(given) and os.access(given, os.W_OK | os.X_OK)):
            raise WriteRefusedError(
                f"TMPDIR={given} (set in the environment) is not an existing writable directory, so Python's tempfile "
                "would fall back to /tmp; create it or unset TMPDIR"
            )
        return given
    if not entry.managed:
        return None
    legacy = _given(environ, entry.legacy)
    if legacy is not None:
        _check(entry, legacy, f"from {entry.legacy}, which the library reads while {entry.name} is unset")
        return legacy
    value = _default(entry, resolved)
    _check(entry, value, "its default")
    return value


def resolve(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{name: value}`` that ``prepare_process`` gives ``environ`` (default ``os.environ``): every managed variable
    and every set one, each checked; nothing is changed or created. ``WriteRefusedError`` naming the variable when a
    value lies outside the write boundary, ``LocalPathsError`` when a default needs the missing local config."""
    env = os.environ if environ is None else environ
    resolved: dict[str, str] = {}
    for entry in PROCESS_PATHS:
        value = _resolve_one(entry, env, resolved)
        if value is not None:
            resolved[entry.name] = value
    return resolved


def _make_scratch(path: str) -> None:
    """Create the default TMPDIR (0700); WriteRefusedError if something else than a directory is in the way."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise WriteRefusedError(f"cannot create the scratch directory {path}: {exc}") from exc
    if not os.path.isdir(path):
        raise WriteRefusedError(f"the scratch directory {path} is not a directory")


def scratch_dir(environ: Mapping[str, str] | None = None) -> Path:
    """A checked directory for temporary files, for library code (``tempfile.mkdtemp(dir=scratch_dir())``): ``TMPDIR``
    of ``environ`` (default ``os.environ``) when set, validated like ``prepare_process`` does, else its RAM default,
    created. Never the system temp dir: ``WriteRefusedError`` instead."""
    env = os.environ if environ is None else environ
    entry = _BY_NAME["TMPDIR"]
    value = _resolve_one(entry, env, {})
    if value is None:  # TMPDIR is managed, so it always has a value
        raise WriteRefusedError("no scratch directory")
    if _given(env, "TMPDIR") is None:
        _make_scratch(value)
    return Path(value)


def prepare_process(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Route this process's caches and temporary files into the write boundary (the module docstring): resolve every
    variable, create the default ``TMPDIR``, then set each managed variable that was unset or empty. Returns the
    values (``resolve``). With ``environ=None`` this is ``os.environ``, and Python's cached temp dir is dropped and
    checked against ``TMPDIR``. For entry points only; ``command_line`` turns its refusals into an exit message."""
    env: MutableMapping[str, str] = os.environ if environ is None else environ
    values = resolve(env)
    if _given(env, "TMPDIR") is None:
        _make_scratch(values["TMPDIR"])
    before = {name: env.get(name) for name in values}
    for name, value in values.items():
        if _given(env, name) is None:
            env[name] = value
    if environ is None:
        tempfile.tempdir = None  # computed again from TMPDIR on the next use
        got = tempfile.gettempdir()
        if os.path.realpath(got) != os.path.realpath(values["TMPDIR"]):
            for name, old in before.items():  # leave the environment as it was
                if old is None:
                    env.pop(name, None)
                else:
                    env[name] = old
            tempfile.tempdir = None
            raise WriteRefusedError(
                f"Python's temp dir would be {got}, not TMPDIR={values['TMPDIR']} (is it writable?); refusing"
            )
    return values


@contextmanager
def command_line(prog: str) -> Iterator[None]:
    """For a command line's ``main``: a ``LocalPathsError`` (a refusal of the write boundary, a missing local config)
    inside the block becomes ``SystemExit("<prog>: <reason>")``, one line instead of a traceback."""
    try:
        yield
    except LocalPathsError as exc:
        raise SystemExit(f"{prog}: {exc}") from exc
