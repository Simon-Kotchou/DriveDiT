"""Inverse-dynamics model (IDM) over frame pairs, and CADRE-style action recoverability with a donor-action control
(torch).

**The model.** ``InverseDynamicsModel`` reads a pair of frames (t, t+1), latents ``[C, h, w]`` or images
``[3, H, W]``, and predicts ``action_cmd[t]``, the command applied over the decision interval between them
(features_v1 ``offsets``: ``action_cmd[t]`` applies over (P_t, P_t+1], ``obs[t+1]`` is after it). It is a small CNN on
the two frames and their difference, pooled to a coarse grid so that where the motion happens still counts (rotation
moves the whole view sideways, speed expands it from the centre), then an MLP. MODELING_PLAN §5 E1 asks for a "small
CNN / IDM", and gap-4 §8.5 for an evaluation IDM of a different architecture family from any reward IDM. Training
follows CADRE's probe recipe (arXiv 2610.30129 App. E): smooth-L1 on standardised actions, AdamW at 3e-4, cosine decay.

**ARR with a donor control** (``arr_r2_with_donor_gap``; CADRE App. E eq. 7 and Table 5; research delta-2026-11 §5.2
M2). The IDM reads actions back from frames, per action dimension d:

- ``R²_d(real)``: on the real held-out continuation against its commands, the ceiling;
- ``R²_d``: on the generated frames against the commands the rollout received;
- ``ARR = mean_d max(R²_d, -1) / R²_d(real)``, over the dimensions the IDM can read on real data
  (``R²_d(real) >= min_real_r2``; gap-4 §3.1 scores a ratio only where the ceiling is at least 0.5);
- **control**: the same recovered actions scored against the commands of a *donor* window, which the rollout never
  received. A model that ignores its actions scores alike on both, so ``gap = mean_d max(R²_d, -1) - mean_d
  max(R²_d(donor), -1)`` (raw R², as CADRE Table 5) measures action following: swap the donor's actions in and the
  score must drop. CADRE uses the clip's own commands shifted by 48 frames; at our near-constant speed that is a weak
  control, so the donor is another window with a matched initial speed (delta §5.2 M2, ±5 km/h; ``match_donors``).
  The same gap on the real continuation is reported beside it as the ceiling.

Read the gap together with the counterfactual tests of ``drivedit.eval.control``: under the *logged* commands, a
model that ignores its actions but predicts the expert's usual continuation also follows those commands, and so also
shows a gap. Only rollouts under commands the context does not predict (κ offsets, donor actions as conditioning)
separate the two.

**Use.** The IDM is an evaluation instrument. Train it only on real sessions held out from world-model training (by
session; MODELING_PLAN §5 E3, delta §5.2 M2). **The evaluation IDM is never used for a reward, for hyperparameter or
checkpoint selection, or for early stopping**: a model optimised against its own judge reaches the judge's ceiling by
exploiting it (gap-4 §0 item 4 and §8.5: RLIR; EVA's separate cross-architecture evaluation IDM). A reward IDM, if one
is needed, is a separate model of another architecture trained on disjoint sessions. Before any IDM number is
reported, its R² on held-out real data must pass the validity gate (delta §5.2 M2: yaw rate R² >= 0.8).

Inputs are float arrays normalised by the caller (latents with the training split's statistics, images in [0, 1]);
batches are converted to float32 tensors on ``device``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional

from drivedit.eval.linear import FloatArray, IntArray, constant_columns, r2_score

ACTION_CHANNELS = ("kappa_target", "a_target", "dir_cmd")  # features_v1 action_cmd, verbatim
_KMH_PER_MPS = 3.6


@dataclass(frozen=True)
class IdmConfig:
    """Architecture of the IDM. ``in_channels`` is the channel count of one frame (latent C, or 3 for RGB);
    ``n_down`` stride-2 stages of width ``width`` (a multiple of 8, for GroupNorm), then pooling to ``pool_hw``."""

    in_channels: int
    out_dim: int = len(ACTION_CHANNELS)
    width: int = 64
    n_down: int = 3
    pool_hw: tuple[int, int] = (2, 4)
    hidden: int = 256

    def __post_init__(self) -> None:
        if self.in_channels < 1 or self.out_dim < 1 or self.n_down < 0 or self.hidden < 1:
            raise ValueError(f"invalid IDM config {self}")
        if self.width < 8 or self.width % 8:
            raise ValueError(f"width must be a positive multiple of 8, got {self.width}")


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class InverseDynamicsModel(nn.Module):
    """``(x_t, x_next)`` ``[B, C, H, W]`` -> ``action_cmd[t]`` ``[B, out_dim]``.

    ``forward`` returns standardised actions (the training target); ``predict`` returns them in action units, through
    the per-channel ``target_mean`` / ``target_std`` buffers that ``train_idm`` sets from the training actions.
    Parameters are initialised from ``seed`` without touching the global random state.
    """

    target_mean: Tensor
    target_std: Tensor

    def __init__(self, cfg: IdmConfig, *, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers = [_block(3 * cfg.in_channels, cfg.width, 1)]
            for _ in range(cfg.n_down):
                layers += [_block(cfg.width, cfg.width, 2), _block(cfg.width, cfg.width, 1)]
            self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(cfg.pool_hw))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cfg.width * cfg.pool_hw[0] * cfg.pool_hw[1], cfg.hidden),
                nn.SiLU(),
                nn.Linear(cfg.hidden, cfg.out_dim),
            )
        self.register_buffer("target_mean", torch.zeros(cfg.out_dim))
        self.register_buffer("target_std", torch.ones(cfg.out_dim))

    def forward(self, x_t: Tensor, x_next: Tensor) -> Tensor:
        if x_t.shape != x_next.shape or x_t.ndim != 4 or x_t.shape[1] != self.cfg.in_channels:
            raise ValueError(f"expected two [B, {self.cfg.in_channels}, H, W] frames, got {x_t.shape}, {x_next.shape}")
        out: Tensor = self.head(self.encoder(torch.cat([x_t, x_next, x_next - x_t], dim=1)))
        return out

    def predict(self, x_t: Tensor, x_next: Tensor) -> Tensor:
        return self.forward(x_t, x_next) * self.target_std + self.target_mean


@dataclass(frozen=True)
class IdmTrainConfig:
    """Training of the IDM: CADRE App. E's optimiser (AdamW, lr 3e-4, cosine decay) and loss (smooth-L1 on
    standardised actions); ``weight_decay`` is AdamW's default. ``seed`` drives the batch sampling."""

    steps: int
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or not self.lr > 0.0:
            raise ValueError(f"invalid IDM training config {self}")


def _check_pairs(x_t: npt.ArrayLike, x_next: npt.ArrayLike, actions: npt.ArrayLike | None) -> int:
    a, b = np.shape(x_t), np.shape(x_next)
    if a != b or len(a) != 4:
        raise ValueError(f"x_t and x_next must be equal [N, C, H, W], got {a} and {b}")
    if actions is not None and np.shape(actions)[0] != a[0]:
        raise ValueError(f"{np.shape(actions)[0]} actions for {a[0]} pairs")
    return int(a[0])


def _batch(x: npt.ArrayLike, idx: IntArray, device: torch.device) -> Tensor:
    return torch.as_tensor(np.asarray(x)[idx], dtype=torch.float32, device=device)


def train_idm(
    model: InverseDynamicsModel,
    x_t: npt.ArrayLike,
    x_next: npt.ArrayLike,
    actions: npt.ArrayLike,
    *,
    cfg: IdmTrainConfig,
    device: torch.device | str,
) -> list[float]:
    """Fit ``model`` on pairs ``x_t``, ``x_next`` ``[N, C, H, W]`` and their ``actions`` ``[N, out_dim]``
    (``action_cmd`` of real held-out sessions only; see the module docstring). Sets the target statistics (a constant
    channel gets std 0, so ``predict`` returns its value and nothing is learnt for it), moves the model to ``device``
    and returns the loss of every step."""
    n = _check_pairs(x_t, x_next, actions)
    acts = np.asarray(actions, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] != model.cfg.out_dim or not np.all(np.isfinite(acts)):
        raise ValueError(f"actions must be finite [N, {model.cfg.out_dim}], got {acts.shape}")
    dev = torch.device(device)
    model.to(dev)
    std = np.where(constant_columns(acts), 0.0, acts.std(axis=0))  # 0 for a constant channel: predict its mean
    model.target_mean.copy_(torch.as_tensor(acts.mean(axis=0), dtype=torch.float32))
    model.target_std.copy_(torch.as_tensor(std, dtype=torch.float32))
    targets = torch.as_tensor((acts - acts.mean(axis=0)) / np.where(std > 0.0, std, 1.0), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    gen = np.random.default_rng(cfg.seed)
    losses = []
    model.train()
    for _ in range(cfg.steps):
        idx = np.sort(gen.integers(0, n, size=min(cfg.batch_size, n)))
        pred = model(_batch(x_t, idx, dev), _batch(x_next, idx, dev))
        loss = functional.smooth_l1_loss(pred, targets[idx].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


@torch.no_grad()
def predict_actions(
    model: InverseDynamicsModel,
    x_t: npt.ArrayLike,
    x_next: npt.ArrayLike,
    *,
    batch_size: int,
    device: torch.device | str,
) -> FloatArray:
    """The IDM's actions ``[N, out_dim]`` (action units) for pairs ``[N, C, H, W]``, in eval mode."""
    n = _check_pairs(x_t, x_next, None)
    dev = torch.device(device)
    model.to(dev).eval()
    out = np.empty((n, model.cfg.out_dim), dtype=np.float64)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n), dtype=np.int64)
        out[idx] = model.predict(_batch(x_t, idx, dev), _batch(x_next, idx, dev)).double().cpu().numpy()
    return out


def evaluate_idm(
    model: InverseDynamicsModel,
    x_t: npt.ArrayLike,
    x_next: npt.ArrayLike,
    actions: npt.ArrayLike,
    *,
    batch_size: int,
    device: torch.device | str,
) -> FloatArray:
    """Per-channel R² ``[out_dim]`` of the IDM on held-out pairs (NaN for a constant channel): the validity check."""
    _check_pairs(x_t, x_next, actions)
    return r2_score(actions, predict_actions(model, x_t, x_next, batch_size=batch_size, device=device))


def match_donors(
    initial_speed_mps: npt.ArrayLike,
    *,
    groups: Sequence[Hashable],
    rng: np.random.Generator,
    max_speed_diff_kmh: float = 5.0,
) -> IntArray:
    """For each window, a donor window: another clip (``groups``, one label per window; windows of one clip share
    their motion) whose initial speed is within ``max_speed_diff_kmh`` (delta §5.2 M2: ±5 km/h), drawn uniformly
    with ``rng``. Raises when a window has no donor."""
    v = np.asarray(initial_speed_mps, dtype=np.float64).ravel()
    if len(groups) != v.size:
        raise ValueError(f"{len(groups)} group labels for {v.size} windows")
    keys = {key: i for i, key in enumerate(dict.fromkeys(groups))}
    g = np.asarray([keys[key] for key in groups], dtype=np.int64)
    tol = max_speed_diff_kmh / _KMH_PER_MPS
    donors = np.empty(v.size, dtype=np.int64)
    for i in range(v.size):
        cand = np.flatnonzero((g != g[i]) & (np.abs(v - v[i]) <= tol))
        if cand.size == 0:
            raise ValueError(f"window {i} (speed {v[i]:.2f} m/s) has no donor within {max_speed_diff_kmh} km/h")
        donors[i] = cand[rng.integers(cand.size)]
    return donors


@dataclass(frozen=True)
class PairSet:
    """Frame pairs ``x_t``, ``x_next`` ``[N, C, H, W]`` and the ``action_cmd`` ``[N, K]`` they were produced under
    (logged for real frames, the conditioning for a rollout)."""

    x_t: npt.ArrayLike
    x_next: npt.ArrayLike
    action_cmd: npt.ArrayLike


@dataclass(frozen=True)
class ArrReport:
    """Action recoverability of a rollout and its donor control, with the real-data ceiling (module docstring).

    Per dimension (``names``): ``r2_real`` and ``r2_real_donor`` on the real continuation against its own and the
    donors' commands, ``r2_generated`` and ``r2_generated_donor`` likewise on the rollout. ``scored`` lists the
    dimensions the IDM reads on real data (``r2_real >= min_real_r2``); ``arr``, ``control``, ``gap`` and
    ``real_gap`` average over them and are NaN (``defined`` False) when none is.
    """

    names: tuple[str, ...]
    r2_real: tuple[float, ...]
    r2_real_donor: tuple[float, ...]
    r2_generated: tuple[float, ...]
    r2_generated_donor: tuple[float, ...]
    scored: tuple[str, ...]
    min_real_r2: float
    arr: float
    control: float
    gap: float
    real_gap: float
    defined: bool


def arr_r2_with_donor_gap(
    model: InverseDynamicsModel,
    real: PairSet,
    generated: PairSet,
    *,
    donors: npt.ArrayLike,
    batch_size: int,
    device: torch.device | str,
    min_real_r2: float = 0.5,
    names: Sequence[str] = ACTION_CHANNELS,
) -> ArrReport:
    """CADRE-style ARR and the donor-action control gap of ``generated`` (a rollout per real window, from the same
    context and under the commands in its ``action_cmd``), against the real continuation ``real``.

    ``donors`` ``[N]`` gives each window's donor window (``match_donors``): its commands, which this window's rollout
    never received, are swapped in as the control target. ``model`` is the evaluation IDM, trained on real held-out
    sessions only and never used for reward or model selection.
    """
    n = _check_pairs(real.x_t, real.x_next, real.action_cmd)
    if _check_pairs(generated.x_t, generated.x_next, generated.action_cmd) != n:
        raise ValueError("real and generated must hold the same windows")
    donor = np.asarray(donors, dtype=np.int64).ravel()
    if donor.size != n or np.any(donor < 0) or np.any(donor >= n) or np.any(donor == np.arange(n)):
        raise ValueError("donors must give every window another window's index")
    channel_names = tuple(names)
    a_real = np.asarray(real.action_cmd, dtype=np.float64)
    a_gen = np.asarray(generated.action_cmd, dtype=np.float64)
    if a_real.shape != (n, len(channel_names)) or a_gen.shape != a_real.shape:
        raise ValueError(f"action_cmd must be [N, {len(channel_names)}] for both sets")
    p_real = predict_actions(model, real.x_t, real.x_next, batch_size=batch_size, device=device)
    p_gen = predict_actions(model, generated.x_t, generated.x_next, batch_size=batch_size, device=device)
    r2_real = r2_score(a_real, p_real)
    r2_real_donor = r2_score(a_real[donor], p_real)
    r2_gen = r2_score(a_gen, p_gen)
    r2_gen_donor = r2_score(a_gen[donor], p_gen)
    keep = np.isfinite(r2_real) & (r2_real >= min_real_r2) & np.isfinite(r2_gen) & np.isfinite(r2_gen_donor)
    defined = bool(keep.any())

    def clipped(r2: FloatArray) -> FloatArray:
        out: FloatArray = np.maximum(r2[keep], -1.0)
        return out

    nan = math.nan
    arr = float(np.mean(clipped(r2_gen) / r2_real[keep])) if defined else nan
    control = float(np.mean(clipped(r2_gen_donor))) if defined else nan
    gap = float(np.mean(clipped(r2_gen))) - control if defined else nan
    real_gap = float(np.mean(clipped(r2_real)) - np.mean(clipped(r2_real_donor))) if defined else nan
    return ArrReport(
        names=channel_names,
        r2_real=tuple(float(v) for v in r2_real),
        r2_real_donor=tuple(float(v) for v in r2_real_donor),
        r2_generated=tuple(float(v) for v in r2_gen),
        r2_generated_donor=tuple(float(v) for v in r2_gen_donor),
        scored=tuple(name for name, k in zip(channel_names, keep, strict=True) if k),
        min_real_r2=min_real_r2,
        arr=arr,
        control=control,
        gap=gap,
        real_gap=real_gap,
        defined=defined,
    )
