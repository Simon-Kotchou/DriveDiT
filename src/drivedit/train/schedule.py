"""The diffusion-forcing schedule: what is drawn for every frame of a training window, and how its loss is weighted.

Convention: rectified flow with t = 0 data and t = 1 noise (arch 3.4). Every frame draws its own t (diffusion
forcing; MIRA DF vs TF gFID 10.7 vs 32.5, causal-memory 1.1). A random-length prefix of each window is clean context:
modulated at t = 0 so its keys and values are those a rollout caches (Krea 3, arch TL;DR 9), with a light noise
augmentation (Dreamer 4 tau_ctx 0.1, causal-memory 1.1) that the model is not told about.

Timestep density: logit-normal(mean, std), then the SD3 / RAE shift ``alpha t / (1 + (alpha - 1) t)`` with
``alpha = sqrt(h * w * C / 4096)`` of one latent frame (arch 3.2: per-frame dimension; about 1.06 at 144 x 32).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from drivedit.model.config import COND_GROUPS


@dataclass(frozen=True)
class ForcingSchedule:
    """Per-frame draws of the diffusion-forcing objective and its loss weights (MODELING_PLAN section 3).

    - ``logit_mean``, ``logit_std``: the logit-normal density of t before the shift (SD3; LightningDiT 16.61 -> 13.99).
    - ``shift``: alpha; ``None`` uses the latent's resolution shift ``sqrt(h * w * C / 4096)`` (arch 3.2). The A/B
      arms are alpha0 x {2, 4}.
    - ``t_clamp``: the velocity denominator floor, ``v = (z_t - x) / max(t, t_clamp)`` (V-RAE / JiT, arch 3.1).
    - ``context_noise``: context frames mix in noise of level U(0, context_noise) while modulated at t = 0.
    - ``min_context`` / ``max_context``: the clean-prefix length k ~ U{min..max} (``None``: T - 1); frames >= k
      are targets. k = 0 is pure diffusion forcing, k = T - 1 teacher forcing (causal-memory 5.5 item 5: random
      context length, including single-frame contexts).
    - ``drop_action_cmd``, ``drop_state``: per-frame, per-group dropout to the learned null (p 0.1; the
      unconditional CFG branch drops only the current action_cmd, which per-frame dropout trains; causal-memory 5.5).
    - ``weighting``: ``"velocity"`` is the x-prediction velocity loss ``||x0_hat - x||^2 / max(t, t_clamp)^2``
      (v1); ``"x_ramp"`` is the x-loss with Dreamer 4's ramp ``0.9 (1 - t) + 0.1`` (its x-loss + ramp cut FVD
      329 -> 102; the arm to pair with shortcut forcing, arch 3.3).
    - ``aux_weight``: lambda of every auxiliary head (MODELING_PLAN section 3: 0.01-0.05).
    - ``shortcut``: Dreamer 4 shortcut forcing; ``shortcut_fraction`` of the samples train an Euler jump of size
      d = 2**-k > d_min against a two-half-step bootstrap target, the rest the flow loss at d_min (shortcut models
      train the bootstrap on 1/4 of the batch, causal-memory 2.1).
    """

    logit_mean: float = 0.0
    logit_std: float = 1.0
    shift: float | None = None
    t_clamp: float = 0.05
    context_noise: float = 0.1
    min_context: int = 0
    max_context: int | None = None
    drop_action_cmd: float = 0.1
    drop_state: float = 0.1
    weighting: Literal["velocity", "x_ramp"] = "velocity"
    aux_weight: float = 0.02
    shortcut: bool = False
    shortcut_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.logit_std <= 0 or (self.shift is not None and self.shift <= 0):
            raise ValueError("logit_std and shift must be positive")
        if not 0 < self.t_clamp < 1 or not 0 <= self.context_noise < 1:
            raise ValueError("t_clamp must be in (0, 1) and context_noise in [0, 1)")
        if self.min_context < 0 or (self.max_context is not None and self.max_context < self.min_context):
            raise ValueError(f"bad context range [{self.min_context}, {self.max_context}]")
        for p in (self.drop_action_cmd, self.drop_state, self.shortcut_fraction):
            if not 0 <= p <= 1:
                raise ValueError(f"probability {p} outside [0, 1]")
        if self.weighting not in ("velocity", "x_ramp"):
            raise ValueError(f"unknown weighting {self.weighting!r}")


@dataclass(frozen=True)
class FrameDraw:
    """One window's draws, all ``[B, T]`` unless noted.

    ``t``: the model's flow-time input (0 on context). ``noise_level``: the level actually mixed into the input
    (t on targets, U(0, context_noise) on context). ``target``: frames that carry the loss. ``drop_cond [B, T, G]``.
    ``step_size``: shortcut step sizes (``None`` without step conditioning). ``bootstrap [B]``: samples trained on
    the shortcut bootstrap target.
    """

    t: Tensor
    noise_level: Tensor
    target: Tensor
    drop_cond: Tensor
    step_size: Tensor | None
    bootstrap: Tensor | None


def shift_time(t: Tensor, alpha: float) -> Tensor:
    """The SD3 / RAE timestep shift ``alpha t / (1 + (alpha - 1) t)``: monotone, fixes 0 and 1, alpha > 1 moves mass
    toward noise, and ``shift_time(shift_time(t, a), 1 / a) == t``."""
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def resolution_shift(channels: int, height: int, width: int) -> float:
    """alpha0 = sqrt(h * w * C / 4096) for one latent frame (arch 3.2)."""
    return math.sqrt(channels * height * width / 4096)


def sample_t(
    shape: tuple[int, ...], schedule: ForcingSchedule, alpha: float, *, generator: torch.Generator, device: torch.device
) -> Tensor:
    """Shifted logit-normal flow times in (0, 1)."""
    u = torch.randn(shape, generator=generator, device=device) * schedule.logit_std + schedule.logit_mean
    return shift_time(torch.sigmoid(u), alpha)


def loss_weight(t: Tensor, schedule: ForcingSchedule) -> Tensor:
    """Per-frame weight of ``||x0_hat - x_target||^2`` (see ``ForcingSchedule.weighting``)."""
    if schedule.weighting == "velocity":
        return torch.clamp(t, min=schedule.t_clamp).pow(-2)
    return 0.9 * (1.0 - t) + 0.1


def draw_frames(
    schedule: ForcingSchedule,
    batch: int,
    frames: int,
    *,
    alpha: float,
    shortcut_levels: int,
    generator: torch.Generator,
    device: torch.device,
) -> FrameDraw:
    """Draw per-frame times, the context prefix, conditioning dropout and shortcut step sizes for one window."""
    max_context = frames - 1 if schedule.max_context is None else min(schedule.max_context, frames - 1)
    if schedule.min_context > max_context:
        raise ValueError(f"min_context {schedule.min_context} leaves no target frame in a {frames}-frame window")
    if schedule.shortcut and shortcut_levels < 2:
        raise ValueError("shortcut forcing needs a model with shortcut_levels >= 2")

    def uniform(*shape: int) -> Tensor:
        return torch.rand(shape, generator=generator, device=device)

    k = torch.randint(schedule.min_context, max_context + 1, (batch, 1), generator=generator, device=device)
    target = torch.arange(frames, device=device)[None, :] >= k
    t = sample_t((batch, frames), schedule, alpha, generator=generator, device=device)

    step_size, bootstrap = None, None
    if shortcut_levels:
        d_min = 2.0 ** -(shortcut_levels - 1)
        step_size = torch.full((batch, frames), d_min, device=device)
        if schedule.shortcut:
            bootstrap = uniform(batch) < schedule.shortcut_fraction
            level = torch.randint(0, shortcut_levels - 1, (batch, frames), generator=generator, device=device)
            d = torch.exp2(-level.float())
            jump = bootstrap[:, None] & target
            step_size = torch.where(jump, d, step_size)
            t = torch.where(jump, torch.ceil(t / d) * d, t)  # on the d-grid {d, 2d, ..., 1}, so t - d >= 0

    context_level = uniform(batch, frames) * schedule.context_noise
    drop_p = torch.tensor([schedule.drop_action_cmd, schedule.drop_state], device=device)
    drop_cond = uniform(batch, frames, len(COND_GROUPS)) < drop_p
    return FrameDraw(
        t=torch.where(target, t, 0.0),
        noise_level=torch.where(target, t, context_level),
        target=target,
        drop_cond=drop_cond,
        step_size=step_size,
        bootstrap=bootstrap,
    )


def sampling_times(steps: int, *, shift: float = 1.0) -> list[float]:
    """The Euler grid ``[1, ..., 0]`` of ``steps`` steps, shifted by ``shift`` (BFL: sample shift >= train shift).

    A shortcut model samples on the unshifted uniform grid, whose step 1 / steps it was trained on.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return [float(v) for v in shift_time(grid, shift)]
