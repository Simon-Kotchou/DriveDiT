"""The KV-cached rollout (begin_rollout, step, sample_next) equals the full frame-causal forward pass."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from drivedit.model import AUX_HEADS, WorldModel, WorldModelConfig, WorldModelOutput
from drivedit.train.schedule import sampling_times
from tests.drivedit.model.toy import randomised, tiny_config, window_inputs

WINDOW = 4
TOL = {"rtol": 0.0, "atol": 1e-10}


def _model(cfg: WorldModelConfig) -> WorldModel:
    return randomised(WorldModel(cfg).double(), seed=7).eval()


def _assert_frame(full: WorldModelOutput, step: WorldModelOutput, frame: int) -> None:
    torch.testing.assert_close(step.x0, full.x0[:, frame], **TOL)
    assert step.aux.keys() == full.aux.keys()
    for name, value in step.aux.items():
        torch.testing.assert_close(value, full.aux[name][:, frame], **TOL)


def _inputs(cfg: WorldModelConfig, frames: int, context: int) -> dict[str, Tensor]:
    inputs = window_inputs(cfg, 3, frames, seed=11, dtype=torch.float64)
    inputs["t"][:, :context] = 0.0  # begin_rollout commits its context at t = 0
    return inputs


@pytest.mark.parametrize("frames", [WINDOW, 2 * WINDOW + 1])
@pytest.mark.parametrize("context", [1, 3])
def test_committed_steps_equal_the_full_pass(frames: int, context: int) -> None:
    """Every frame, inside and past the window, after a one- or multi-frame prefill."""
    cfg = tiny_config(window=WINDOW, aux_heads=AUX_HEADS)
    model = _model(cfg)
    x, t, a, s = _inputs(cfg, frames, context).values()
    with torch.no_grad():
        full = model(x, t, a, s)
        cache = model.begin_rollout(x[:, :context], a[:, :context], s[:, :context])
        assert cache.frames == context
        for f in range(context, frames):
            _assert_frame(full, model.step(cache, x[:, f], t[:, f], a[:, f - 1], s[:, f - 1], commit=True), f)
            assert cache.frames == f + 1
            assert cache.cached_frames == min(f + 1, WINDOW - 1)


def test_uncommitted_steps_leave_the_cache_unchanged() -> None:
    cfg = tiny_config(window=WINDOW)
    model = _model(cfg)
    x, t, a, s = _inputs(cfg, 3, 2).values()
    with torch.no_grad():
        cache = model.begin_rollout(x[:, :2], a[:, :2], s[:, :2])
        keys = [k.clone() for k in cache.keys]
        first = model.step(cache, x[:, 2], t[:, 2], a[:, 1], s[:, 1])
        again = model.step(cache, x[:, 2], t[:, 2], a[:, 1], s[:, 1])
    torch.testing.assert_close(first.x0, again.x0, rtol=0, atol=0)
    assert cache.frames == 2
    assert all(torch.equal(k0, k1) for k0, k1 in zip(keys, cache.keys, strict=True))


def test_dropout_and_step_size_match_the_full_pass() -> None:
    cfg = tiny_config(window=WINDOW, shortcut_levels=4)
    model = _model(cfg)
    x, t, a, s = _inputs(cfg, WINDOW, 1).values()
    g = torch.Generator().manual_seed(5)
    drop = torch.rand(3, WINDOW, 2, generator=g) < 0.5
    step_size = 2.0 ** -torch.randint(0, 4, (3, WINDOW), generator=g).double()
    step_size[:, 0] = 2.0**-3  # begin_rollout commits its context at the finest step, d_min
    with torch.no_grad():
        full = model(x, t, a, s, drop_cond=drop, step_size=step_size)
        cache = model.begin_rollout(x[:, :1], a[:, :1], s[:, :1])
        for f in range(1, WINDOW):
            out = model.step(
                cache, x[:, f], t[:, f], a[:, f - 1], s[:, f - 1], drop_cond=drop[:, f], step_size=step_size[:, f],
                commit=True,
            )  # fmt: skip
            _assert_frame(full, out, f)


@pytest.mark.parametrize("shortcut_levels", [0, 4])
def test_sample_next_is_seeded_and_commits_one_frame(shortcut_levels: int) -> None:
    cfg = tiny_config(window=WINDOW, shortcut_levels=shortcut_levels, aux_heads=("contact_events",))
    model = _model(cfg)
    x, _, a, s = _inputs(cfg, 3, 3).values()
    times = sampling_times(4)

    def sample(guidance: float) -> tuple[WorldModelOutput, int]:
        with torch.no_grad():
            cache = model.begin_rollout(x, a, s)
            out = model.sample_next(
                cache, a[:, -1], s[:, -1], times=times, generator=torch.Generator().manual_seed(0), guidance=guidance
            )
        return out, cache.frames

    first, frames = sample(1.0)
    second, _ = sample(1.0)
    guided, _ = sample(2.0)
    assert frames == 4
    assert first.x0.shape == (3, *cfg.latent_shape)
    assert first.aux["contact_events"].shape == (3,)
    assert torch.isfinite(first.x0).all()
    torch.testing.assert_close(first.x0, second.x0, rtol=0, atol=0)
    assert not torch.allclose(first.x0, guided.x0)


def test_sample_next_rejects_a_bad_grid() -> None:
    cfg = tiny_config(window=WINDOW)
    model = _model(cfg)
    x, _, a, s = _inputs(cfg, 2, 2).values()
    with torch.no_grad():
        cache = model.begin_rollout(x, a, s)
        with pytest.raises(ValueError, match="decrease"):
            model.sample_next(cache, a[:, -1], s[:, -1], times=[1.0, 0.5], generator=torch.Generator())
    shortcut = _model(tiny_config(window=WINDOW, shortcut_levels=4))
    with torch.no_grad():
        cache = shortcut.begin_rollout(x, a, s)
        with pytest.raises(ValueError, match="step sizes"):
            shortcut.sample_next(cache, a[:, -1], s[:, -1], times=sampling_times(3), generator=torch.Generator())
