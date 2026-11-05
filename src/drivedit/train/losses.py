"""The diffusion-forcing loss: x-prediction in velocity space, optional shortcut forcing, and the auxiliary heads.

``batch`` is the item dict of DD-11's ``drivedit.data.torch.LatentWindowDataset`` (DRIVEDIT_INTERFACES section 4),
collated:

- ``latents [B, T, C, h, w]`` float32, normalised, 0 where ``valid`` is False; ``valid [B, T]`` bool;
- ``action_cmd [B, T, 3]`` and ``state [B, T, 4]`` float32, aligned at index t (the model shifts them);
- ``masks [B, T, 5]`` uint8: policy_valid, wm_valid, warmup, env_transition, teleport (features_v1 order);
- the targets of the configured aux heads: ``future_pose [B, T, H, 3]`` (NaN = no target) and
  ``contact_events [B, T, 2]`` float32 (contact_event, contact_class).

A frame carries loss only where ``valid & wm_valid``; non-finite aux targets are masked. Frame f's aux heads describe
the transition t = f - 1 -> f (``future_pose[f-1]``, ``contact_events[f-1]``), so frame 0 has none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as nnf
from torch.nn.parallel import DistributedDataParallel

from drivedit.model.config import WorldModelConfig
from drivedit.model.layers import symlog
from drivedit.model.world_model import WorldModel, WorldModelOutput
from drivedit.train.schedule import ForcingSchedule, FrameDraw, draw_frames, loss_weight, resolution_shift

Batch = Mapping[str, Tensor]
WM_VALID = 1  # index of wm_valid in features_v1 masks (policy_valid, wm_valid, warmup, env_transition, teleport)
Predict = Callable[[Tensor, Tensor, Tensor], Tensor]


def unwrap(model: WorldModel | DistributedDataParallel) -> WorldModel:
    net = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(net, WorldModel):
        raise TypeError(f"expected a WorldModel, got {type(net).__name__}")
    return net


def _require(batch: Batch, name: str) -> Tensor:
    if name not in batch:
        raise KeyError(f"batch has no {name!r} (keys: {sorted(batch)})")
    return batch[name]


def frame_valid(batch: Batch) -> Tensor:
    """``[B, T]`` frames that may carry a loss: ``valid`` (a latent exists) and ``wm_valid`` of ``masks``."""
    return _require(batch, "valid").bool() & _require(batch, "masks")[..., WM_VALID].bool()


@torch.no_grad()
def shortcut_target(predict: Predict, z: Tensor, t: Tensor, step: Tensor, target: Tensor) -> Tensor:
    """The x-space bootstrap target of an Euler jump of size ``step`` from ``z`` at ``t`` (Dreamer 4, arch 3.3).

    Two half steps of the model itself, each ``z <- z - (d / 2) v`` with ``v = (z - x0_hat) / t``, give the target
    velocity ``v* = (v1 + v2) / 2``; the target clean latent is the one whose straight line through ``z`` has that
    velocity, ``x* = z - t v*``. ``predict(z, t, step)`` returns x0; shapes ``z [B, T, C, h, w]``, ``t, step,
    target [B, T]``. Non-target (context) frames keep their input and t.
    """
    half = torch.where(target, step / 2, step)
    t_safe = torch.where(target, t, torch.ones_like(t))[..., None, None, None]
    move = (target[..., None, None, None] * half[..., None, None, None]).to(z.dtype)
    v1 = (z - predict(z, t, half)) / t_safe
    z_mid = z - move * v1
    t_mid = torch.where(target, t - half, t)
    v2 = (z_mid - predict(z_mid, t_mid, half)) / torch.where(target, t_mid, torch.ones_like(t))[..., None, None, None]
    return z - t[..., None, None, None] * (v1 + v2) / 2


def aux_losses(aux: Mapping[str, Tensor], batch: Batch, cfg: WorldModelConfig) -> dict[str, Tensor]:
    """Loss per aux head over frames f >= 1 with a valid latent and finite targets.

    ``future_pose`` is a squared error in ``symlog(x / future_pose_scale)`` space; ``contact_events`` a binary
    cross-entropy on the ``contact_event`` channel (``contact_class`` is not trained in v1).
    """
    frame_ok = frame_valid(batch)[:, 1:]
    losses: dict[str, Tensor] = {}
    for name, pred in aux.items():
        p = pred[:, 1:].float()
        if name == "future_pose":
            scale = torch.tensor(cfg.future_pose_scale, device=p.device)
            target = symlog(_require(batch, "future_pose")[:, :-1].float() / scale)
        else:  # contact_events
            target = _require(batch, "contact_events")[:, :-1, 0].float()
        mask = torch.isfinite(target) & frame_ok.view(*frame_ok.shape, *([1] * (target.dim() - 2)))
        target = torch.where(mask, target, torch.zeros_like(target))
        if name == "contact_events":
            err = nnf.binary_cross_entropy_with_logits(p, target, reduction="none")
        else:
            err = (p - target).pow(2)
        losses[name] = torch.where(mask, err, 0.0).sum() / mask.sum().clamp_min(1)
    return losses


def diffusion_forcing_loss(
    model: WorldModel | DistributedDataParallel, batch: Batch, *, schedule: ForcingSchedule, rng: torch.Generator
) -> tuple[Tensor, dict[str, Tensor]]:
    """The training loss of one window batch and its detached components.

    Each frame draws t (``draw_frames``); ``z = (1 - level) x + level eps``; the model predicts x0 and the loss is
    ``loss_weight(t) * ||x0_hat - x_target||^2`` averaged over valid target frames, where ``x_target`` is the data
    (flow samples) or the shortcut bootstrap target. Plus ``schedule.aux_weight`` times each aux head's loss.
    ``rng`` must live on the batch's device.
    """
    cfg = unwrap(model).cfg
    x = _require(batch, "latents").float()
    action_cmd, state = _require(batch, "action_cmd").float(), _require(batch, "state").float()
    valid = frame_valid(batch)
    b, frames = x.shape[:2]
    alpha = schedule.shift if schedule.shift is not None else resolution_shift(*cfg.latent_shape)
    draw = draw_frames(
        schedule, b, frames, alpha=alpha, shortcut_levels=cfg.shortcut_levels, generator=rng, device=x.device
    )
    eps = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype)
    level = draw.noise_level[..., None, None, None]
    z = (1.0 - level) * x + level * eps
    # The bootstrap passes (unwrapped, no grad) run before the graded pass, the one DDP forward of the update.
    x_target = _bootstrapped(model, x, z, draw, action_cmd=action_cmd, state=state)
    out: WorldModelOutput = model(z, draw.t, action_cmd, state, drop_cond=draw.drop_cond, step_size=draw.step_size)
    weight = loss_weight(draw.t, schedule) * (draw.target & valid)
    err = (out.x0.float() - x_target).pow(2).mean(dim=(2, 3, 4))
    flow = (weight * err).sum() / (draw.target & valid).sum().clamp_min(1)

    logs = {"flow": flow.detach()}
    loss = flow
    for name, value in aux_losses(out.aux, batch, cfg).items():
        loss = loss + schedule.aux_weight * value
        logs[f"aux/{name}"] = value.detach()
    logs["loss"] = loss.detach()
    return loss, logs


def _bootstrapped(
    model: WorldModel | DistributedDataParallel,
    x: Tensor,
    z: Tensor,
    draw: FrameDraw,
    *,
    action_cmd: Tensor,
    state: Tensor,
) -> Tensor:
    """``x`` with the shortcut bootstrap target substituted on the bootstrap samples (no-op without shortcut).

    The bootstrap predictions run on the unwrapped module, never through DistributedDataParallel: which samples
    bootstrap differs by rank, and every DDP forward issues collectives (buffer broadcast, bucket rebuild), so a
    rank-dependent number of DDP forwards would desynchronise them. The graded pass stays the one DDP forward.
    """
    if draw.bootstrap is None or draw.step_size is None:
        return x
    rows = draw.bootstrap.nonzero().squeeze(1)
    if rows.numel() == 0:
        return x
    net = unwrap(model)
    a, s, drop = action_cmd[rows], state[rows], draw.drop_cond[rows]

    def predict(z_in: Tensor, t_in: Tensor, step: Tensor) -> Tensor:
        out: WorldModelOutput = net(z_in, t_in, a, s, drop_cond=drop, step_size=step)
        return out.x0.float()

    target = shortcut_target(predict, z[rows], draw.t[rows], draw.step_size[rows], draw.target[rows])
    return x.index_copy(0, rows, target)
