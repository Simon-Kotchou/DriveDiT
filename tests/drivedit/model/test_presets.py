"""Preset sizes on the default 144-token latent, the latent shape as a config value, and config validation."""

from __future__ import annotations

import pytest
import torch

from drivedit.model import PRESETS, WorldModel, WorldModelConfig

#: Parameter counts in millions on the default 32 x 9 x 16 latent (meta-device count; the report quotes them).
EXPECTED_M = {"probe_50m": (50, 55), "small_110m": (110, 120), "base_250m": (245, 260)}


def _count(cfg: WorldModelConfig) -> int:
    with torch.device("meta"):
        return sum(p.numel() for p in WorldModel(cfg).parameters())


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_parameter_count(name: str) -> None:
    low, high = EXPECTED_M[name]
    assert low * 1e6 <= _count(PRESETS[name]()) <= high * 1e6


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_share_the_v1_block(name: str) -> None:
    cfg = PRESETS[name]()
    assert cfg.latent_shape == (32, 9, 16)
    assert cfg.tokens_per_frame == 144
    assert (cfg.head_dim, cfg.heads // cfg.kv_heads, cfg.window) == (64, 4, 10)
    assert cfg.aux_heads == ()


def test_base_is_d1024_l20_gqa_16_4() -> None:
    cfg = PRESETS["base_250m"]()
    assert (cfg.dim, cfg.depth, cfg.heads, cfg.kv_heads) == (1024, 20, 16, 4)


def test_the_latent_shape_only_changes_the_input_and_output_layers() -> None:
    probe = PRESETS["probe_50m"]()
    wide = probe.with_latent(16, 18, 32)  # a 576-token, 16-channel codec
    assert wide.tokens_per_frame == 576
    delta = _count(probe) - _count(wide)
    assert delta == (32 - 16) * probe.dim * 2 + (32 - 16)  # patch_embed weight, out weight and out bias


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dim": 500}, "head_dim"),
        ({"kv_heads": 3}, "kv_heads"),
        ({"rope_split": (16, 16, 16)}, "rope_split"),
        ({"aux_heads": ("depth",)}, "unknown aux heads"),
        ({"aux_heads": ("action_exec",)}, "unknown aux heads"),  # features_v1: not a world-model target
        ({"aux_heads": ("future_pose", "future_pose")}, "duplicate"),
        ({"action_dim": 2}, "features_v1"),
        ({"window": 0}, "positive"),
    ],
)
def test_bad_configs_raise(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        WorldModelConfig(**{"dim": 512, "heads": 8, "kv_heads": 2, **overrides})  # type: ignore[arg-type]
