"""Training: the loop, noise schedules, few-step objectives, EMA, checkpoints and run configs.

The names below load on first use (PEP 562), not with the package, so ``python -m drivedit.train.cli`` (or ``.bench``)
can check what importing torch would write before anything imports torch (DD-23, ``drivedit.process_paths``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drivedit.train.config import DataConfig, OptimConfig, RunConfig
    from drivedit.train.losses import diffusion_forcing_loss
    from drivedit.train.schedule import ForcingSchedule, sampling_times
    from drivedit.train.trainer import Trainer, load_world_model

#: exported name -> the module that defines it.
_EXPORTS = {
    "DataConfig": "drivedit.train.config",
    "OptimConfig": "drivedit.train.config",
    "RunConfig": "drivedit.train.config",
    "diffusion_forcing_loss": "drivedit.train.losses",
    "ForcingSchedule": "drivedit.train.schedule",
    "sampling_times": "drivedit.train.schedule",
    "Trainer": "drivedit.train.trainer",
    "load_world_model": "drivedit.train.trainer",
}

__all__ = [
    "DataConfig",
    "ForcingSchedule",
    "OptimConfig",
    "RunConfig",
    "Trainer",
    "diffusion_forcing_loss",
    "load_world_model",
    "sampling_times",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    """The module's own names and the lazy exports (they are not in globals() until first used)."""
    return sorted({*globals(), *_EXPORTS})
