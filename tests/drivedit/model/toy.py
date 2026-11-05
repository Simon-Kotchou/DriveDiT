"""Tiny configs, random-weight models, window inputs and toy latent dynamics for the CPU world-model tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from drivedit.model import WorldModel, WorldModelConfig

KAPPA = 0.02  # |kappa_target| of a turning toy command [1/m]

#: The keys and dtypes of one DD-11 ``LatentWindowDataset`` item for the world model (channels action_cmd and state,
#: aux future_pose and contact_events).
WINDOW_KEYS = {
    "latents": torch.float32,
    "valid": torch.bool,
    "action_cmd": torch.float32,
    "state": torch.float32,
    "future_pose": torch.float32,
    "contact_events": torch.float32,
    "masks": torch.uint8,
}


def tiny_config(**overrides: Any) -> WorldModelConfig:
    """A 2-block, d32 model on a 4 x 3 x 4 latent with a 4-frame window."""
    base: dict[str, Any] = {
        "latent_channels": 4,
        "latent_height": 3,
        "latent_width": 4,
        "dim": 32,
        "depth": 2,
        "heads": 2,
        "kv_heads": 1,
        "head_dim": 16,
        "rope_split": (4, 6, 6),
        "window": 4,
        "time_freq_dim": 32,
        "future_pose_horizon": 3,
    }
    return WorldModelConfig(**{**base, **overrides})


def randomised(model: WorldModel, *, seed: int = 0, std: float = 0.2) -> WorldModel:
    """Perturb every parameter. The DiT init zeroes the AdaLN projections and the output, which would make an untrained
    model pass any causality or sensitivity check trivially."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
    return model


def window_inputs(
    cfg: WorldModelConfig, batch: int, frames: int, *, seed: int = 0, dtype: torch.dtype = torch.float32
) -> dict[str, Tensor]:
    """Random ``x [B, T, C, h, w]``, ``t [B, T]``, ``action_cmd [B, T, 3]`` and ``state [B, T, 4]`` in range."""
    g = torch.Generator().manual_seed(seed)
    action_cmd = torch.randn(batch, frames, 3, generator=g, dtype=dtype) * torch.tensor([KAPPA, 1.0, 0.0], dtype=dtype)
    action_cmd[..., 2] = torch.randint(-1, 2, (batch, frames), generator=g).to(dtype)
    state = torch.randn(batch, frames, 4, generator=g, dtype=dtype) * torch.tensor([3.0, 0.2, 1.0, 0.0], dtype=dtype)
    state[..., 3] = torch.randint(-1, 2, (batch, frames), generator=g).to(dtype)
    return {
        "x": torch.randn(batch, frames, *cfg.latent_shape, generator=g, dtype=dtype),
        "t": torch.rand(batch, frames, generator=g, dtype=dtype),
        "action_cmd": action_cmd,
        "state": state,
    }


class ShiftDynamics(Dataset[dict[str, Tensor]]):
    """Toy latent dynamics driven by the command: ``x[f + 1] = roll(x[f], sign(kappa_target[f]))`` along the width.

    Frame 0 is an N(0, 1) pattern; each step's ``kappa_target`` is -KAPPA, 0 or +KAPPA with equal probability, so the
    next latent is a deterministic function of the current one and the command. ``state`` is constant (it carries no
    information about the command). Items are DD-11's ``LatentWindowDataset`` dict (``WINDOW_KEYS``, dtypes as
    there): ``future_pose[t, k]`` is a toy plan (2 m per step forward, sideways and heading with the summed turns),
    ``contact_events`` has no contact (class -1), and ``masks`` marks every frame policy- and wm-valid.
    """

    def __init__(self, cfg: WorldModelConfig, size: int, *, seed: int) -> None:
        self.cfg, self.size, self.seed = cfg, size, seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        frames, horizon = self.cfg.window, self.cfg.future_pose_horizon
        turn = torch.randint(-1, 2, (frames + horizon,), generator=g)
        x = [torch.randn(self.cfg.latent_shape, generator=g)]
        for f in range(frames - 1):
            x.append(torch.roll(x[-1], shifts=int(turn[f]), dims=-1))
        action_cmd = torch.zeros(frames, 3)
        action_cmd[:, 0] = KAPPA * turn[:frames]
        action_cmd[:, 2] = 1.0
        summed = torch.stack([turn[t : t + horizon].cumsum(0) for t in range(frames)]).float()  # [L, H]
        steps = torch.arange(1, horizon + 1).float().expand(frames, horizon)
        contact_events = torch.zeros(frames, 2)
        contact_events[:, 1] = -1.0
        masks = torch.zeros(frames, 5, dtype=torch.uint8)
        masks[:, :2] = 1
        return {
            "latents": torch.stack(x),
            "valid": torch.ones(frames, dtype=torch.bool),
            "action_cmd": action_cmd,
            "state": torch.tensor([[10.0, 0.0, 0.0, 1.0]]).expand(frames, 4).clone(),
            "future_pose": torch.stack((2.0 * steps, 0.1 * summed, 0.02 * summed), dim=-1),
            "contact_events": contact_events,
            "masks": masks,
        }


def collate(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {k: torch.stack([item[k] for item in items]) for k in items[0]}
