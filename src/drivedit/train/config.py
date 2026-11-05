"""Run configuration: frozen dataclasses that round-trip through JSON.

The optimiser defaults are MODELING_PLAN section 3: AdamW (0.9, 0.95), weight decay 0.1, lr 1e-4 constant after a 1k
warm-up, gradient clip 1.0 (MIRA / CADRE, arch 3.4), bf16 autocast, Power-EMA.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from drivedit.model.config import WorldModelConfig, probe_50m
from drivedit.train.schedule import ForcingSchedule

T = TypeVar("T")


@dataclass(frozen=True)
class OptimConfig:
    """AdamW, linear warm-up then constant lr, global-norm clipping, and the Power-EMA exponent.

    ``ema_gamma`` is the power-function EMA exponent of EDM2 (Karras et al. 2024, arXiv 2312.02696): the average
    weights the update at step tau of t by ``tau ** gamma``; gamma 6.94 is a relative width sigma_rel of 0.10
    (16.97 would be 0.05).
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ema_gamma: float = 6.94


@dataclass(frozen=True)
class DataConfig:
    """The features_v1 latent windows of the training split (DRIVEDIT_INTERFACES sections 2-4).

    Shards are ``<root>/<session_id>/features_v1/`` and latents ``<root>/<session_id>/latents/<codec_id>/``. Only
    training-split sessions belong here: normalisation statistics come from them (section 3), and Everon never does.
    """

    root: str
    sessions: tuple[str, ...]
    codec_id: str
    stride: int = 1
    num_workers: int = 4


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines a training run, except where it writes (``Trainer(run_dir=...)``)."""

    run_id: str
    model: WorldModelConfig = field(default_factory=probe_50m)
    schedule: ForcingSchedule = field(default_factory=ForcingSchedule)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig | None = None
    #: Windows per process and step (the global batch is batch_size x world size).
    batch_size: int = 8
    steps: int = 100_000
    seed: int = 0
    precision: Literal["bf16", "fp32"] = "bf16"
    activation_checkpointing: bool = False
    #: Compile each block with torch.compile (regional compile, efficiency 3.4 item 4).
    compile: bool = False
    log_every: int = 50
    checkpoint_every: int = 1000
    keep_checkpoints: int = 2

    def __post_init__(self) -> None:
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if min(self.batch_size, self.steps, self.log_every, self.checkpoint_every, self.keep_checkpoints) < 1:
            raise ValueError("batch_size, steps, log_every, checkpoint_every and keep_checkpoints must be >= 1")
        if self.schedule.shortcut and self.model.shortcut_levels < 2:
            raise ValueError("schedule.shortcut needs model.shortcut_levels >= 2")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunConfig:
        return from_dict(cls, json.loads(text))


def from_dict(cls: type[T], data: Any) -> T:
    """Build the dataclass ``cls`` from ``dataclasses.asdict`` output (lists back to tuples, nested dataclasses).

    Unknown keys raise; missing keys take the field default.
    """
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__}: expected an object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if unknown := set(data) - names:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    return cls(**{name: _coerce(hints[name], value) for name, value in data.items()})


def _coerce(tp: Any, value: Any) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    out: Any
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        out = from_dict(tp, value)
    elif origin in (types.UnionType, typing.Union):
        (inner,) = (a for a in args if a is not type(None))
        out = None if value is None else _coerce(inner, value)
    elif origin is tuple:
        elements = (args[0],) * len(value) if len(args) == 2 and args[1] is Ellipsis else args
        out = tuple(_coerce(a, v) for a, v in zip(elements, value, strict=True))
    elif origin is Literal:
        if value not in args:
            raise ValueError(f"{value!r} is not one of {args}")
        out = value
    else:
        out = float(value) if tp is float and isinstance(value, int) else value
    return out
