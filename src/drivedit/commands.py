"""The model-side commands, started so that the process paths are routed before torch is imported (DD-23).

    dd-train ...                                   # console scripts of pyproject.toml
    dd-encode ...
    python -m drivedit.commands bench ...          # the step-time bench
    torchrun --nproc-per-node 2 -m drivedit.commands dd-train --config run.json --run-dir <run dir>

Each command's own module imports torch through its package (``drivedit.train``, ``drivedit.codec``), and importing
torch already writes when ``TORCH_LOGS_OUT`` is set. So this module, which imports nothing heavy, first calls
``drivedit.process_paths.prepare_process`` (validating every set variable, routing the unset caches into the write
boundary; a refusal exits with one line) and only then imports the command and runs its ``main``, which checks again.
``python -m drivedit.train.cli`` and the other modules still work, with their check after their imports.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Final

from drivedit.process_paths import command_line, prepare_process

__all__ = ["COMMANDS", "bench", "dd_encode", "dd_train", "main", "run"]

#: command name -> the module whose ``main(argv)`` runs it.
COMMANDS: Final = {
    "dd-train": "drivedit.train.cli",
    "dd-encode": "drivedit.codec.encode",
    "bench": "drivedit.train.bench",
}


def run(command: str, argv: Sequence[str] | None = None) -> int:
    """``prepare_process``, then import ``COMMANDS[command]`` and return its ``main(argv)``."""
    with command_line(command):
        prepare_process()
    module = importlib.import_module(COMMANDS[command])
    result: int = module.main(argv)
    return result


def dd_train() -> int:
    return run("dd-train")


def dd_encode() -> int:
    return run("dd-encode")


def bench() -> int:
    return run("bench")


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m drivedit.commands <command> [args]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS:
        raise SystemExit(f"usage: python -m drivedit.commands {{{','.join(COMMANDS)}}} [args]")
    return run(args[0], args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
