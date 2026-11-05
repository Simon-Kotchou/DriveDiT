"""The causal latent DiT world model and its KV-cached rollout.

Time base (DRIVEDIT_INTERFACES section 1): index t is the 5 Hz decision step, and a transition is (obs_t,
``action_cmd[t]`` applied over (P_t, P_t+1], obs_t+1). Inputs arrive aligned at index t, unshifted, exactly as the
dataset stores them; the model shifts them itself: latent frame f is conditioned on ``action_cmd[f-1]`` and
``state[f-1]``, and frame 0 gets the learned nulls. ``action_cmd[:, -1]`` and ``state[:, -1]`` would condition a frame
after the window and are not used.

Flow convention: rectified flow with t = 0 data and t = 1 noise, ``z_t = (1 - t) x + t eps`` (arch 3.4: the SD3 / RAE
shift formulas then copy over directly). The network predicts the clean latent x0 (x-prediction, JiT / JLT / V-RAE,
arch 3.3); the loss in ``drivedit.train.losses`` is taken in velocity space.

Conditioning (AdaLN-single, arch 4.3 and causal-memory 5.5): ``c_f = SiLU(MLP_t(t_f) + E_a(action_cmd[f-1]) +
E_s(state[f-1]) [+ E_d(step size)])`` feeds one shared ``Linear(D, 6D)``; every block adds a zero-initialised bias.
The direction channels (``dir_cmd``, ``dir_state``, -1/0/+1) are lookup embeddings (Dreamer 4 style, causal-memory
5.5); the continuous channels enter as ``symlog(x / scale)``. Each group has a learned null, used for frame 0 and for
classifier-free-guidance dropout. Clean context frames are modulated at t = 0, so their keys and values do not depend
on any later denoising step and the KV cache stays valid (Krea 3, arch TL;DR 9).

Rollout: ``begin_rollout`` commits context frames at t = 0 in one pass; ``step`` evaluates the denoiser for the next
frame against the cache and, with ``commit=True``, appends that frame's keys and values. The cache keeps the last
``window - 1`` frames, so every query sees at most ``window`` frames, as in training (causal-memory 7).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf
from torch.utils.checkpoint import checkpoint

from drivedit.model.config import COND_GROUPS, WorldModelConfig
from drivedit.model.layers import (
    DiTBlock,
    RMSNorm,
    frame_causal_mask,
    frame_positions,
    modulate,
    rope_angles,
    symlog,
    timestep_features,
)


@dataclass(frozen=True)
class WorldModelOutput:
    """``x0 [B, T, C, h, w]``: the predicted clean latents. ``aux``: per-frame auxiliary predictions, keyed by head.

    ``aux`` shapes: ``future_pose [B, T, H, 3]`` in ``symlog(x / future_pose_scale)`` (the target space of
    ``drivedit.train.losses``) and ``contact_events [B, T]`` (the logit of ``contact_event``). A single rollout step
    drops the T axis.
    """

    x0: Tensor
    aux: dict[str, Tensor]


@dataclass
class RolloutCache:
    """Per-block un-rotated keys and values ``[B, kv_heads, n * N, Dh]`` of the last n <= window - 1 committed frames.

    ``frames`` counts every frame committed so far: it is the index of the next frame. ``step(..., commit=True)``
    appends to the cache in place.
    """

    keys: list[Tensor]
    values: list[Tensor]
    frames: int
    tokens_per_frame: int

    @property
    def cached_frames(self) -> int:
        return self.keys[0].shape[2] // self.tokens_per_frame

    def append(self, kv: Sequence[tuple[Tensor, Tensor]], keep_frames: int) -> None:
        """Add one call's keys and values per block, then keep only the newest ``keep_frames`` frames."""
        keep = keep_frames * self.tokens_per_frame
        for i, (k, v) in enumerate(kv):
            self.keys[i] = torch.cat((self.keys[i], k), dim=2)[:, :, -keep:] if keep else k[:, :, :0]
            self.values[i] = torch.cat((self.values[i], v), dim=2)[:, :, -keep:] if keep else v[:, :, :0]


class ConditionEncoder(nn.Module):
    """Per-frame conditioning vector ``c [B, T, D]`` from flow time, the previous step's command and state.

    The inputs are already shifted: ``prev_action [B, T, 3]`` and ``prev_state [B, T, 4]`` hold, for frame f, the
    values at index f - 1. ``null [B, T, G]`` (groups in ``COND_GROUPS`` order) replaces a group by its learned null.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        d = cfg.dim
        self.time_freq_dim = cfg.time_freq_dim
        self.time_mlp = nn.Sequential(nn.Linear(cfg.time_freq_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.step_embed = nn.Embedding(cfg.shortcut_levels, d) if cfg.shortcut_levels else None
        self.action_in = nn.Linear(cfg.action_dim - 1, d)
        self.dir_cmd = nn.Embedding(3, d)
        self.action_out = nn.Linear(d, d)
        self.state_in = nn.Linear(cfg.state_dim - 1, d)
        self.dir_state = nn.Embedding(3, d)
        self.state_out = nn.Linear(d, d)
        self.null = nn.Parameter(torch.zeros(len(COND_GROUPS), d))
        self.register_buffer("action_scale", torch.tensor(cfg.action_scale), persistent=False)
        self.register_buffer("state_scale", torch.tensor(cfg.state_scale), persistent=False)
        self.action_scale: Tensor
        self.state_scale: Tensor

    @staticmethod
    def _direction(x: Tensor) -> Tensor:
        """-1/0/+1 direction channel to a lookup index 0/1/2 (an invalid value fails in the embedding lookup)."""
        return (torch.round(x) + 1).long()

    def forward(
        self, t: Tensor, prev_action: Tensor, prev_state: Tensor, null: Tensor, step_level: Tensor | None
    ) -> Tensor:
        e_t: Tensor = self.time_mlp(timestep_features(t, self.time_freq_dim).to(self.null.dtype))
        if self.step_embed is not None:
            if step_level is None:
                raise ValueError("this model is step-size conditioned (shortcut_levels > 0); pass step_size")
            e_t = e_t + self.step_embed(step_level)
        elif step_level is not None:
            raise ValueError("step_size given to a model without shortcut levels")
        a = symlog(prev_action[..., :2] / self.action_scale.to(prev_action.dtype))
        e_a = self.action_out(nnf.silu(self.action_in(a) + self.dir_cmd(self._direction(prev_action[..., 2]))))
        s = symlog(prev_state[..., :3] / self.state_scale.to(prev_state.dtype))
        e_s = self.state_out(nnf.silu(self.state_in(s) + self.dir_state(self._direction(prev_state[..., 3]))))
        e_a = torch.where(null[..., 0, None], self.null[0], e_a)
        e_s = torch.where(null[..., 1, None], self.null[1], e_s)
        return nnf.silu(e_t + e_a + e_s)


class AuxHeads(nn.Module):
    """Linear heads on the token-mean of each frame's final hidden state (arch 4.3 aux heads; P5: loss side only)."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        shapes: dict[str, tuple[int, ...]] = {"future_pose": (cfg.future_pose_horizon, 3), "contact_events": ()}
        self.shapes = {name: shapes[name] for name in cfg.aux_heads}
        self.norm = RMSNorm(cfg.dim)
        self.heads = nn.ModuleDict({name: nn.Linear(cfg.dim, math.prod(s)) for name, s in self.shapes.items()})

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        """``h [B, T, N, D]`` -> ``{name: [B, T, *shape]}``."""
        pooled = self.norm(h.mean(dim=2))
        return {name: head(pooled).view(*pooled.shape[:2], *self.shapes[name]) for name, head in self.heads.items()}


class WorldModel(nn.Module):
    """Frame-block-causal latent DiT: ``forward`` for training, ``begin_rollout`` / ``step`` for KV-cached rollout."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.patch_embed = nn.Linear(cfg.latent_channels, d)
        self.cond = ConditionEncoder(cfg)
        self.adaln = nn.Linear(d, 6 * d)
        self.blocks = nn.ModuleList(DiTBlock(cfg) for _ in range(cfg.depth))
        self.final_norm = RMSNorm(d, affine=False)
        self.final_adaln = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, cfg.latent_channels)
        self.aux_heads = AuxHeads(cfg) if cfg.aux_heads else None
        #: Recompute each block in the backward pass (set by the trainer; needed for base_250m on 12 GB cards).
        self.activation_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """DiT initialisation: xavier linears, N(0, 0.02) embeddings and nulls, zero AdaLN projections and output, so
        every block starts as the identity and the model starts by predicting the mean latent (Peebles & Xie)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.cond.null, std=0.02)
        for linear in self.cond.time_mlp:
            if isinstance(linear, nn.Linear):
                nn.init.normal_(linear.weight, std=0.02)
        for zero in (self.adaln, self.final_adaln, self.out):
            nn.init.zeros_(zero.weight)
            nn.init.zeros_(zero.bias)

    # ----------------------------------------------------------------------------------------------------- training

    def forward(
        self,
        x_noisy: Tensor,
        t: Tensor,
        action_cmd: Tensor,
        state: Tensor,
        *,
        drop_cond: Tensor | None = None,
        step_size: Tensor | None = None,
    ) -> WorldModelOutput:
        """Denoise a window of latent frames.

        ``x_noisy [B, T, C, h, w]``; ``t [B, T]`` in [0, 1] (t = 1 noise; 0 for clean context); ``action_cmd [B, T, 3]``
        and ``state [B, T, 4]`` aligned at index t (unshifted). ``drop_cond [B, T, G]`` bool replaces group g of frame
        f (the value at index f - 1) by its null; a ``[B, T]`` mask drops every group. ``step_size [B, T]`` (powers of
        two) is required by step-conditioned models (``shortcut_levels > 0``). Frames longer than ``window`` back are
        masked, as in the rollout.
        """
        self._check_window(x_noisy, t, action_cmd, state)
        c = self._window_cond(t, action_cmd, state, drop_cond, step_size)
        x, _ = self._window_blocks(self._embed(x_noisy), c, keep_kv=False)
        return self._readout(x, c)

    # ------------------------------------------------------------------------------------------------------ rollout

    def begin_rollout(self, latents: Tensor, action_cmd: Tensor, state: Tensor) -> RolloutCache:
        """Commit K >= 1 clean context frames ``latents [B, K, C, h, w]`` at t = 0 in one frame-causal pass.

        ``action_cmd [B, K, 3]`` and ``state [B, K, 4]`` are aligned at index t as in ``forward``; their last entries
        condition the next frame and go to the first ``step`` call instead.
        """
        b, k = latents.shape[:2]
        if k < 1:
            raise ValueError("begin_rollout needs at least one context frame")
        t = torch.zeros(b, k, dtype=latents.dtype, device=latents.device)
        self._check_window(latents, t, action_cmd, state)
        c = self._window_cond(t, action_cmd, state, None, self._finest_step((b, k), latents.device))
        _, kvs = self._window_blocks(self._embed(latents), c, keep_kv=True)
        cache = RolloutCache(
            keys=[kv[0][:, :, :0] for kv in kvs],
            values=[kv[1][:, :, :0] for kv in kvs],
            frames=k,
            tokens_per_frame=self.cfg.tokens_per_frame,
        )
        cache.append(kvs, keep_frames=self.cfg.window - 1)
        return cache

    def step(
        self,
        cache: RolloutCache,
        x_noisy: Tensor,
        t: Tensor,
        action_cmd: Tensor,
        state: Tensor,
        *,
        drop_cond: Tensor | None = None,
        step_size: Tensor | None = None,
        commit: bool = False,
    ) -> WorldModelOutput:
        """Denoise the next frame f = ``cache.frames`` against the cached window.

        ``x_noisy [B, C, h, w]``, ``t [B]``; ``action_cmd [B, 3]`` and ``state [B, 4]`` are the values at index f - 1
        (the command applied after the last committed frame, and the state there). ``drop_cond [B, G]``. With
        ``commit=True`` this frame's keys and values are appended to ``cache``; commit the final sample at t = 0.
        Returns ``x0 [B, C, h, w]`` and aux predictions without the T axis.
        """
        b = x_noisy.shape[0]
        n_past = cache.cached_frames
        null = torch.zeros(b, 1, len(COND_GROUPS), dtype=torch.bool, device=x_noisy.device)
        if drop_cond is not None:
            null = null | drop_cond[:, None].bool()
        size = step_size[:, None] if step_size is not None else None
        c = self.cond(t[:, None], action_cmd[:, None], state[:, None], null, self._step_level(size))
        index = torch.arange(n_past + 1, dtype=self._angle_dtype(), device=x_noisy.device)
        k_angles = rope_angles(frame_positions(index, self.cfg), self.cfg)
        q_angles = k_angles[n_past * self.cfg.tokens_per_frame :]
        x, mod = self._embed(x_noisy[:, None]), self.adaln(c)
        kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, mod, q_angles, k_angles, mask=None, prefix=(cache.keys[i], cache.values[i]))
            kvs.append(kv)
        if commit:
            cache.append(kvs, keep_frames=self.cfg.window - 1)
            cache.frames += 1
        out = self._readout(x, c)
        return WorldModelOutput(x0=out.x0[:, 0], aux={name: v[:, 0] for name, v in out.aux.items()})

    def sample_next(
        self,
        cache: RolloutCache,
        action_cmd: Tensor,
        state: Tensor,
        *,
        times: Sequence[float],
        generator: torch.Generator,
        guidance: float = 1.0,
    ) -> WorldModelOutput:
        """Sample frame f = ``cache.frames`` by Euler steps on the predicted velocity and commit it at t = 0.

        ``times`` is the decreasing grid from 1 to 0 (``drivedit.train.schedule.sampling_times``). For a
        step-conditioned model each step's size must be one of its powers of two. ``guidance`` w mixes
        ``x0_u + w (x0_c - x0_u)``, where the unconditional branch drops only the current ``action_cmd`` (MODELING_PLAN
        section 3, action CFG). Returns the sample as ``x0`` and the aux predictions of its t = 0 commit pass.
        """
        if len(times) < 2 or times[0] != 1.0 or times[-1] != 0.0 or any(a <= b for a, b in pairwise(times)):
            raise ValueError(f"times must decrease from 1.0 to 0.0, got {list(times)}")
        sizes = {2.0**-k for k in range(self.cfg.shortcut_levels)}
        if self.cfg.shortcut_levels and any(a - b not in sizes for a, b in pairwise(times)):
            raise ValueError(f"a step-conditioned model samples with step sizes in {sorted(sizes)}, got {list(times)}")
        b, device, dtype = action_cmd.shape[0], action_cmd.device, self.out.weight.dtype
        z = torch.randn((b, *self.cfg.latent_shape), generator=generator, device=device, dtype=dtype)
        uncond = torch.zeros(b, len(COND_GROUPS), dtype=torch.bool, device=device)
        uncond[:, COND_GROUPS.index("action_cmd")] = True
        for t_now, t_next in pairwise(times):
            t = torch.full((b,), t_now, device=device, dtype=dtype)
            size = torch.full((b,), t_now - t_next, device=device, dtype=dtype) if self.cfg.shortcut_levels else None
            x0 = self.step(cache, z, t, action_cmd, state, step_size=size).x0
            if guidance != 1.0:
                x0_u = self.step(cache, z, t, action_cmd, state, drop_cond=uncond, step_size=size).x0
                x0 = x0_u + guidance * (x0 - x0_u)
            z = z + (t_next - t_now) * (z - x0) / t_now
        t0 = torch.zeros(b, device=device, dtype=dtype)
        committed = self.step(cache, z, t0, action_cmd, state, step_size=self._finest_step((b,), device), commit=True)
        return WorldModelOutput(x0=z, aux=committed.aux)

    # ------------------------------------------------------------------------------------------------------ helpers

    def _check_window(self, x: Tensor, t: Tensor, action_cmd: Tensor, state: Tensor) -> None:
        b, frames = x.shape[:2]
        expected = {
            "x": (x.shape, (b, frames, *self.cfg.latent_shape)),
            "t": (t.shape, (b, frames)),
            "action_cmd": (action_cmd.shape, (b, frames, self.cfg.action_dim)),
            "state": (state.shape, (b, frames, self.cfg.state_dim)),
        }
        for name, (got, want) in expected.items():
            if tuple(got) != want:
                raise ValueError(f"{name}: shape {tuple(got)}, expected {want}")

    def _angle_dtype(self) -> torch.dtype:
        return torch.float64 if self.out.weight.dtype == torch.float64 else torch.float32

    def _finest_step(self, shape: tuple[int, ...], device: torch.device) -> Tensor | None:
        """The flow-matching step size d_min of a step-conditioned model (used for clean commits), else None."""
        if not self.cfg.shortcut_levels:
            return None
        return torch.full(shape, 2.0 ** -(self.cfg.shortcut_levels - 1), device=device)

    def _window_cond(
        self, t: Tensor, action_cmd: Tensor, state: Tensor, drop_cond: Tensor | None, step_size: Tensor | None
    ) -> Tensor:
        """Conditioning ``c [B, T, D]`` of a window aligned at index t: frame f gets index f - 1, frame 0 the nulls."""
        b, frames = t.shape
        prev_action = torch.cat((torch.zeros_like(action_cmd[:, :1]), action_cmd[:, :-1]), dim=1)
        prev_state = torch.cat((torch.zeros_like(state[:, :1]), state[:, :-1]), dim=1)
        null = torch.zeros(b, frames, len(COND_GROUPS), dtype=torch.bool, device=t.device)
        null[:, 0] = True
        if drop_cond is not None:
            null = null | (drop_cond if drop_cond.dim() == 3 else drop_cond[..., None]).bool()
        c: Tensor = self.cond(t, prev_action, prev_state, null, self._step_level(step_size))
        return c

    def _window_blocks(self, x: Tensor, c: Tensor, *, keep_kv: bool) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run the blocks over a window ``x [B, T, N, D]`` with the frame-causal mask; optionally keep each block's
        ``(k, v)``. Blocks are recomputed in backward when ``activation_checkpointing`` is on and training."""
        frames = x.shape[1]
        index = torch.arange(frames, dtype=self._angle_dtype(), device=x.device)
        angles = rope_angles(frame_positions(index, self.cfg), self.cfg)
        mask = frame_causal_mask(frames, self.cfg.tokens_per_frame, self.cfg.window, x.device)
        mod = self.adaln(c)
        kvs = []
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x, kv = checkpoint(block, x, mod, angles, angles, mask=mask, use_reentrant=False)
            else:
                x, kv = block(x, mod, angles, angles, mask=mask)
            if keep_kv:
                kvs.append(kv)
        return x, kvs

    def _step_level(self, step_size: Tensor | None) -> Tensor | None:
        """Step size d = 2**-k to level k (an invalid level fails in the embedding lookup)."""
        if step_size is None:
            return None
        return torch.round(-torch.log2(step_size.float())).long()

    def _embed(self, latents: Tensor) -> Tensor:
        """``[B, T, C, h, w]`` -> tokens ``[B, T, N, D]`` (patch 1, row-major over the latent grid)."""
        tokens = latents.permute(0, 1, 3, 4, 2).flatten(2, 3)
        y: Tensor = self.patch_embed(tokens.to(self.patch_embed.weight.dtype))
        return y

    def _readout(self, x: Tensor, c: Tensor) -> WorldModelOutput:
        """Final AdaLN and an fp32 output head (comma, arch 3.4) -> x0 ``[B, T, C, h, w]``; aux heads on ``x``."""
        b, frames = x.shape[:2]
        shift, scale = self.final_adaln(c).unsqueeze(2).chunk(2, dim=-1)
        h = modulate(self.final_norm(x), shift, scale)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y: Tensor = self.out(h.to(self.out.weight.dtype))
        x0 = y.view(b, frames, self.cfg.latent_height, self.cfg.latent_width, -1).permute(0, 1, 4, 2, 3)
        aux = self.aux_heads(x) if self.aux_heads is not None else {}
        return WorldModelOutput(x0=x0, aux=aux)
