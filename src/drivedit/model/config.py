"""World-model configuration: one frozen dataclass and the three size presets.

The recipe is MODELING_PLAN section 3 (v1), built from the research notes ``arch-objectives-2026-11.md`` (sections 3,
4.3 and the v1 recipe table) and ``causal-memory-2026-11.md`` (sections 5.5 and 7). The latent shape is a config value
because the codec bake-off (E2) decides it; the default is the 144-token, 32-channel grid the plan assumes
(16 x 9 tokens at /32 of 512 x 288, MIRA Table 23 via arch section 4.2 (d)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

#: Auxiliary heads the model can carry, named as the features_v1 tensors they predict. Both are world-model targets
#: in the features_v1 access table (``target: true``) and loss-side only (MODELING_PLAN P5, section 3). Each
#: describes the transition t = f - 1 -> f that produced latent frame f: ``future_pose[t]`` ([H, 3], whose first step
#: is that transition's motion) and the ``contact_event`` channel of ``contact_events[t]``. The other world-model
#: targets are left out: ``hindsight_anchors`` repeats future_pose at +2/+4/+6 s, and ``road_context`` is not in v1.
AUX_HEADS: Final = ("future_pose", "contact_events")

#: Conditioning groups, in the order of the last axis of ``drop_cond``.
COND_GROUPS: Final = ("action_cmd", "state")

ACTION_DIM: Final = 3  # features_v1 action_cmd: kappa_target, a_target, dir_cmd
STATE_DIM: Final = 4  # features_v1 state: speed_mps, yaw_rate_rps, acc_long_achieved_mps2, dir_state


@dataclass(frozen=True)
class WorldModelConfig:
    """Shape and conditioning of the causal latent DiT.

    Latent frames are ``[C, h, w]`` = (``latent_channels``, ``latent_height``, ``latent_width``), one per 5 Hz decision
    step, tokenised at patch 1 (arch 4.2 (d)). ``window`` is the number of frames a frame may attend to, itself
    included: the training clip length and the rollout KV window are the same (causal-memory 7, "the same window in
    training and inference").

    Continuous conditioning channels enter as ``symlog(x / scale)`` (GAIA-2 encodes speed and curvature with symlog,
    arch 4.1). The default scales are the typical magnitudes measured on the first dc-2.0 Arland session (2026-11-05,
    12 002 ctl rows): ``|kappa_target|`` p95 0.03 1/m (max 0.16), ``a_target`` p1..p99 -1.85..1.5 m/s2, cruise
    about 11 m/s, so typical values map to about 1 and outliers are compressed.
    """

    latent_channels: int = 32
    latent_height: int = 9
    latent_width: int = 16
    dim: int = 1024
    depth: int = 20
    heads: int = 16
    kv_heads: int = 4
    head_dim: int = 64
    #: SwiGLU hidden width as a multiple of ``dim``, rounded up to a multiple of 64 (8/3 keeps the parameter count of a
    #: 4x GELU MLP; LightningDiT, arch 3.1).
    mlp_ratio: float = 8 / 3
    window: int = 10
    #: Widths of features_v1 ``action_cmd`` and ``state``; fixed by the contract and checked.
    action_dim: int = ACTION_DIM
    state_dim: int = STATE_DIM
    #: Latent frame rate. The RoPE time axis is physical time in seconds (MIRA, arch 4.2 (c)), so 5 Hz and 10 Hz
    #: sessions share one axis.
    decision_hz: float = 5.0
    #: Rotary positions per second on the time axis: 5 makes one 5 Hz decision step one rotary unit, the same unit as
    #: one latent cell on the spatial axes.
    rope_time_scale: float = 5.0
    #: Head-dim split for (time, height, width) of the 3D axial RoPE (arch 4.2 (c): 16 / 24 / 24 of head dim 64).
    rope_split: tuple[int, int, int] = (16, 24, 24)
    #: RoPE base (Krea 2 uses 1e3; arch 4.2 (c)).
    rope_theta: float = 1000.0
    #: Width of the sinusoidal flow-time features fed to the t-embedding MLP (DiT).
    time_freq_dim: int = 256
    action_scale: tuple[float, float] = (0.01, 1.0)  # kappa_target [1/m], a_target [m/s2]
    state_scale: tuple[float, float, float] = (5.0, 0.1, 1.0)  # speed_mps, yaw_rate_rps, acc_long_achieved_mps2
    future_pose_scale: tuple[float, float, float] = (5.0, 1.0, 0.1)  # fut_dx [m], fut_dy [m], fut_dpsi [rad]
    #: Auxiliary heads, a subset of ``AUX_HEADS``. Off by default; v1 runs use both (MODELING_PLAN section 3).
    aux_heads: tuple[str, ...] = ()
    #: H of features_v1 ``future_pose [T, H, 3]``, set by the dataset builder: DD-11 uses 6 s x 5 Hz = 30, the
    #: longest hindsight anchor.
    future_pose_horizon: int = 30
    #: Shortcut step-size levels; 0 disables step conditioning. Level k is the step size d = 2**-k, so 7 levels cover
    #: 1 to 64 sampling steps (Dreamer 4 shortcut forcing; causal-memory 2.5 evaluates 64 / 8 / 4 / 2 / 1 steps).
    shortcut_levels: int = 0

    def __post_init__(self) -> None:
        if self.dim != self.heads * self.head_dim:
            raise ValueError(f"dim {self.dim} != heads {self.heads} x head_dim {self.head_dim}")
        if self.heads % self.kv_heads:
            raise ValueError(f"heads {self.heads} is not a multiple of kv_heads {self.kv_heads}")
        if sum(self.rope_split) != self.head_dim or any(n % 2 for n in self.rope_split):
            raise ValueError(f"rope_split {self.rope_split} must be even sizes summing to head_dim {self.head_dim}")
        if unknown := set(self.aux_heads) - set(AUX_HEADS):
            raise ValueError(f"unknown aux heads {sorted(unknown)}; known: {AUX_HEADS}")
        if len(set(self.aux_heads)) != len(self.aux_heads):
            raise ValueError(f"duplicate aux heads in {self.aux_heads}")
        if min(self.latent_channels, self.latent_height, self.latent_width, self.depth, self.window) < 1:
            raise ValueError("latent shape, depth and window must be positive")
        if (self.action_dim, self.state_dim) != (ACTION_DIM, STATE_DIM):
            raise ValueError(f"features_v1 fixes action_dim {ACTION_DIM} and state_dim {STATE_DIM}")
        if self.shortcut_levels < 0:
            raise ValueError("shortcut_levels must be >= 0")
        if min(*self.action_scale, *self.state_scale, *self.future_pose_scale) <= 0:
            raise ValueError("conditioning and target scales must be positive")

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """``(C, h, w)`` of one latent frame."""
        return (self.latent_channels, self.latent_height, self.latent_width)

    @property
    def tokens_per_frame(self) -> int:
        return self.latent_height * self.latent_width

    @property
    def mlp_hidden(self) -> int:
        return 64 * math.ceil(self.mlp_ratio * self.dim / 64)

    def with_latent(self, channels: int, height: int, width: int) -> WorldModelConfig:
        """The same network on another codec's latent grid."""
        return replace(self, latent_channels=channels, latent_height=height, latent_width=width)


def probe_50m() -> WorldModelConfig:
    """d512 x L16, GQA 8/2: the codec bake-off probe (MODELING_PLAN E2, "a 50M probe model per codec")."""
    return WorldModelConfig(dim=512, depth=16, heads=8, kv_heads=2)


def small_110m() -> WorldModelConfig:
    """d768 x L16, GQA 12/3: the ablation size (arch 4.3, "ablations d768 x L16 (~110M)")."""
    return WorldModelConfig(dim=768, depth=16, heads=12, kv_heads=3)


def base_250m() -> WorldModelConfig:
    """d1024 x L20, GQA 16/4: the v1 world model (arch 4.3; deeper over wider, the seriality-gap result)."""
    return WorldModelConfig(dim=1024, depth=20, heads=16, kv_heads=4)


PRESETS: Final = {"probe_50m": probe_50m, "small_110m": small_110m, "base_250m": base_250m}
