"""Building blocks of the world model: norms, 3D axial RoPE, frame-causal attention with a KV prefix, SwiGLU, AdaLN.

The block is the arch-objectives section 4.3 v1 block: pre-norm, zero-centred RMSNorm, QK-norm, an elementwise sigmoid
output gate, GQA and SwiGLU (Krea 2, MIRA, CADRE, LightningDiT; arch 4.2 (f)), modulated by one shared AdaLN
projection plus a zero-initialised per-block bias (AdaLN-single; comma, PixArt-alpha, arch 4.2 (b)).

Tensor layout inside the network: tokens ``[B, T, N, D]`` (N = h * w tokens per latent frame), flattened to
``[B, S, D]`` with S = T * N, frame-major, for attention. Attention heads are ``[B, H, S, Dh]``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf

from drivedit.model.config import WorldModelConfig


def symlog(x: Tensor) -> Tensor:
    """sign(x) * log(1 + |x|): linear near 0, logarithmic in the tails (GAIA-2's action encoding, arch 4.1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class RMSNorm(nn.Module):
    """Zero-centred RMSNorm over the last axis: ``x / rms(x) * (1 + gamma)`` with gamma initialised to 0 (Krea 2).

    ``affine=False`` drops gamma. The AdaLN-modulated norms of a block use that form: their zero-centred gain is the
    block's AdaLN scale ``1 + scale``, whose per-block bias starts at 0, so a learned gamma there would be redundant.
    Statistics are computed in float32 and the result is cast back to the input dtype.
    """

    def __init__(self, dim: int, *, affine: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.gamma is not None:
            y = y * (1.0 + self.gamma.to(y.dtype))
        return y.to(x.dtype)


def rope_angles(positions: Tensor, cfg: WorldModelConfig) -> Tensor:
    """Rotation angles ``[S, head_dim / 2]`` of 3D axial RoPE for token positions ``[S, 3]`` = (time, row, column).

    Each axis owns ``rope_split[i]`` head dims with frequencies ``theta ** (-2j / n_i)`` (Krea 2's 3D axial RoPE;
    the time axis carries physical time in rotary units, arch 4.2 (c)).
    """
    parts = []
    for axis, n in enumerate(cfg.rope_split):
        exponent = torch.arange(0, n, 2, dtype=positions.dtype, device=positions.device) / n
        freqs = cfg.rope_theta ** (-exponent)
        parts.append(positions[:, axis, None] * freqs[None, :])
    return torch.cat(parts, dim=-1)


def frame_positions(frame_index: Tensor, cfg: WorldModelConfig) -> Tensor:
    """Token positions ``[F * N, 3]`` for latent frames with (possibly fractional) indices ``[F]``.

    Time is ``frame_index / decision_hz`` seconds times ``rope_time_scale``; rows and columns are the latent grid
    indices. Tokens are frame-major, then row-major, matching ``x.flatten``.
    """
    h, w = cfg.latent_height, cfg.latent_width
    dtype, device = frame_index.dtype, frame_index.device
    time = frame_index * (cfg.rope_time_scale / cfg.decision_hz)
    rows = torch.arange(h, dtype=dtype, device=device).repeat_interleave(w)
    cols = torch.arange(w, dtype=dtype, device=device).repeat(h)
    n = h * w
    return torch.stack(
        (time.repeat_interleave(n), rows.repeat(frame_index.numel()), cols.repeat(frame_index.numel())), dim=-1
    )


def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Rotate consecutive channel pairs of ``x [B, H, S, Dh]`` by ``angles [S, Dh / 2]``; computed in angle dtype."""
    xr = x.to(angles.dtype).reshape(*x.shape[:-1], -1, 2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = xr[..., 0], xr[..., 1]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


def frame_causal_mask(frames: int, tokens_per_frame: int, window: int, device: torch.device) -> Tensor:
    """Frame-block-causal attention mask ``[S, S]`` (True = attend), S = frames * tokens_per_frame.

    A token of frame f sees every token of frames f - window + 1 .. f: bidirectional within a frame, causal across
    frames (arch 4.2 (a), SETTLED), limited to the rollout window so that the full pass and the KV-cached rollout see
    the same keys.
    """
    frame = torch.arange(frames, device=device).repeat_interleave(tokens_per_frame)
    lag = frame[:, None] - frame[None, :]
    return (lag >= 0) & (lag < window)


class Attention(nn.Module):
    """GQA self-attention with QK-RMSNorm, 3D RoPE, an elementwise sigmoid output gate and an optional KV prefix.

    One fused projection gives q (``heads``), k and v (``kv_heads``) and the gate (``heads * head_dim``); the output is
    ``Wo(attn * sigmoid(gate))`` (Krea 2 / MIRA gate, arch 4.3). Keys are cached after QK-norm but before RoPE
    ("keys stored un-rotated", causal-memory 3.6), so a rollout assigns positions inside the trained range at attention
    time.

    Training uses PyTorch SDPA with the explicit boolean frame-causal mask; a single-frame rollout step needs no mask.
    The KV heads are repeated to the query heads before SDPA (the cache keeps ``kv_heads``): with a mask,
    ``enable_gqa=True`` falls back to the math kernel, measured on a local CUDA GPU (torch 2.9.1, B 8, 8/2 heads, 1440
    tokens, fwd + bwd): 1.23 ms and 2156 MiB, against 0.62 ms and 161 MiB for the memory-efficient kernel after the
    repeat.
    FlexAttention (efficiency 2.2, ADOPT-NOW for training masks) is not used in v1: at 144 tokens per frame a 10-frame
    clip is 1440 tokens, attention is about 17% of the FLOPs (efficiency 2.1), frames do not align with Flex's
    128-token blocks (efficiency 2.3 item 3), and Flex needs torch.compile to be fast. SDPA keeps one code path that
    runs on CPU, in float64 and under bf16 autocast.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.heads, cfg.kv_heads, cfg.head_dim
        q_dim, kv_dim = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        self.split = (q_dim, kv_dim, kv_dim, q_dim)
        self.qkvg = nn.Linear(cfg.dim, sum(self.split), bias=False)
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)
        self.out = nn.Linear(q_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: Tensor,
        q_angles: Tensor,
        k_angles: Tensor,
        *,
        mask: Tensor | None,
        prefix: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend ``x [B, S, D]`` over ``prefix`` keys and its own.

        ``q_angles [S, Dh/2]`` rotate the queries, ``k_angles [P + S, Dh/2]`` the prefix and new keys. ``prefix`` is
        the cached ``(k, v)`` ``[B, kv_heads, P, Dh]``. Returns the output ``[B, S, D]`` and this call's un-rotated
        ``(k, v)`` ``[B, kv_heads, S, Dh]``.
        """
        b, s, _ = x.shape
        q, k, v, gate = self.qkvg(x).split(self.split, dim=-1)
        q = self.q_norm(q.view(b, s, self.heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(k.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2))
        v = v.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        new_kv = (k, v)
        if prefix is not None:
            k, v = torch.cat((prefix[0], k), dim=2), torch.cat((prefix[1], v), dim=2)
        q, k = apply_rope(q, q_angles), apply_rope(k, k_angles)
        if self.heads != self.kv_heads:
            group = self.heads // self.kv_heads
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        o = nnf.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(b, s, -1) * torch.sigmoid(gate)
        y: Tensor = self.out(o)
        return y, new_kv


class SwiGLU(nn.Module):
    """``W2(silu(W1 x) * W3 x)`` with a fused W1/W3 (LightningDiT 12.52 -> 10.10 FID, arch 3.1)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w13(x).chunk(2, dim=-1)
        y: Tensor = self.w2(nnf.silu(a) * b)
        return y


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN: ``x * (1 + scale) + shift``."""
    return x * (1.0 + scale) + shift


def timestep_features(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal features ``[..., dim]`` of flow time ``t [...]`` in [0, 1], scaled by 1000 as in DiT."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    angles = 1000.0 * t.float()[..., None] * freqs
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class DiTBlock(nn.Module):
    """Pre-norm block ``x += g1 * Attn(mod1(norm x)); x += g2 * SwiGLU(mod2(norm x))``.

    The modulation ``[B, T, 6D]`` (shift, scale, gate for attention, then for the MLP) is the shared AdaLN-single
    projection of the frame's conditioning; this block adds its own zero-initialised bias (arch 4.2 (b): per-block
    AdaLN would add about 52% of the core parameters). It is broadcast over the N tokens of each frame.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim, affine=False)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.dim, affine=False)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_hidden)
        self.mod_bias = nn.Parameter(torch.zeros(6 * cfg.dim))

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
        q_angles: Tensor,
        k_angles: Tensor,
        *,
        mask: Tensor | None,
        prefix: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """``x [B, T, N, D]``, ``mod [B, T, 6D]``; returns the new ``x`` and this call's ``(k, v)``."""
        shift1, scale1, gate1, shift2, scale2, gate2 = (mod + self.mod_bias).unsqueeze(2).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1).flatten(1, 2)
        a, kv = self.attn(h, q_angles, k_angles, mask=mask, prefix=prefix)
        x = x + gate1 * a.view_as(x)
        x = x + gate2 * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x, kv
