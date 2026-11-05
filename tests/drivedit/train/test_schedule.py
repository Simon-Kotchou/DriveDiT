"""Timestep density, shift, clamp and the per-frame draws of the diffusion-forcing schedule."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from drivedit.train.schedule import (
    ForcingSchedule,
    draw_frames,
    loss_weight,
    resolution_shift,
    sample_t,
    sampling_times,
    shift_time,
)

CPU = torch.device("cpu")


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_unshifted_density_is_logit_normal() -> None:
    t = sample_t((200_000,), ForcingSchedule(logit_mean=0.3, logit_std=0.8), 1.0, generator=_gen(), device=CPU)
    assert ((t > 0) & (t < 1)).all()
    logit = torch.logit(t.double())
    assert logit.mean().item() == pytest.approx(0.3, abs=0.01)
    assert logit.std().item() == pytest.approx(0.8, abs=0.01)


def test_shift_fixes_the_ends_is_monotone_and_inverts() -> None:
    t = torch.linspace(0, 1, 1001, dtype=torch.float64)
    for alpha in (0.5, 1.0, 1.06, 3.0):
        s = shift_time(t, alpha)
        assert s[0].item() == 0.0
        assert s[-1].item() == 1.0
        assert (s.diff() > 0).all()
        torch.testing.assert_close(shift_time(s, 1 / alpha), t)
    torch.testing.assert_close(shift_time(t, 1.0), t)
    assert (shift_time(t[1:-1], 3.0) > t[1:-1]).all()  # alpha > 1 moves mass toward noise (t = 1)


def test_shifted_median_moves_to_alpha_over_one_plus_alpha() -> None:
    alpha = 3.0
    t = sample_t((200_000,), ForcingSchedule(), alpha, generator=_gen(1), device=CPU)
    assert t.median().item() == pytest.approx(alpha / (1 + alpha), abs=0.005)


def test_resolution_shift_of_the_v1_latent() -> None:
    assert resolution_shift(32, 9, 16) == pytest.approx(math.sqrt(144 * 32 / 4096))
    assert resolution_shift(32, 9, 16) == pytest.approx(1.0607, abs=1e-4)
    assert resolution_shift(64, 18, 32) == pytest.approx(3.0)  # 576 x 64 (arch 3.2)


def test_velocity_weight_is_clamped_at_t_clamp() -> None:
    schedule = ForcingSchedule(t_clamp=0.05)
    t = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    torch.testing.assert_close(loss_weight(t, schedule), torch.tensor([400.0, 400.0, 400.0, 100.0, 4.0, 1.0]))


def test_x_ramp_weight_is_dreamer4_ramp() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(loss_weight(t, ForcingSchedule(weighting="x_ramp")), torch.tensor([1.0, 0.55, 0.1]))


def test_draws_split_each_window_into_clean_context_and_targets() -> None:
    schedule = ForcingSchedule(context_noise=0.1, drop_action_cmd=0.1, drop_state=0.3)
    draw = draw_frames(schedule, 4000, 10, alpha=1.06, shortcut_levels=0, generator=_gen(2), device=CPU)
    context = ~draw.target
    assert (draw.t[context] == 0).all()
    assert (draw.noise_level[context] <= 0.1).all()
    assert (draw.t[draw.target] > 0).all()
    torch.testing.assert_close(draw.noise_level[draw.target], draw.t[draw.target])
    lengths = context.sum(1)
    assert set(lengths.tolist()) == set(range(10))  # k ~ U{0..T-1}: pure DF to teacher forcing
    assert (context.cumsum(1) == torch.arange(1, 11).clamp(max=lengths[:, None])).all()  # a prefix
    rates = draw.drop_cond.float().mean(dim=(0, 1))
    assert rates[0].item() == pytest.approx(0.1, abs=0.01)
    assert rates[1].item() == pytest.approx(0.3, abs=0.01)
    assert draw.step_size is None
    assert draw.bootstrap is None


def test_context_range_is_respected() -> None:
    draw = draw_frames(
        ForcingSchedule(min_context=2, max_context=3), 500, 8, alpha=1.0, shortcut_levels=0, generator=_gen(3),
        device=CPU,
    )  # fmt: skip
    assert set((~draw.target).sum(1).tolist()) == {2, 3}
    with pytest.raises(ValueError, match="no target"):
        draw_frames(ForcingSchedule(min_context=8), 2, 8, alpha=1.0, shortcut_levels=0, generator=_gen(), device=CPU)


def test_shortcut_draws_put_jumps_on_their_grid() -> None:
    levels = 5
    d_min = 2.0**-4
    schedule = ForcingSchedule(shortcut=True, shortcut_fraction=0.25)
    draw = draw_frames(schedule, 4000, 6, alpha=1.06, shortcut_levels=levels, generator=_gen(4), device=CPU)
    assert draw.step_size is not None
    assert draw.bootstrap is not None
    assert draw.bootstrap.float().mean().item() == pytest.approx(0.25, abs=0.02)
    jump = draw.bootstrap[:, None] & draw.target
    d = draw.step_size[jump]
    assert set(d.unique().tolist()) == {1.0, 0.5, 0.25, 0.125}  # every level coarser than d_min
    t = draw.t[jump]
    assert (t >= d).all()
    torch.testing.assert_close(t / d, torch.round(t / d))
    assert (draw.step_size[~jump] == d_min).all()


def test_sampling_times() -> None:
    assert sampling_times(4) == [1.0, 0.75, 0.5, 0.25, 0.0]
    shifted = sampling_times(8, shift=3.0)
    assert shifted[0] == 1.0
    assert shifted[-1] == 0.0
    assert all(a > b for a, b in pairwise(shifted))
    assert shifted[4] == pytest.approx(0.75)  # shift_time(0.5, 3)
    with pytest.raises(ValueError, match="steps"):
        sampling_times(0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"logit_std": 0.0}, "positive"),
        ({"t_clamp": 0.0}, "t_clamp"),
        ({"min_context": 3, "max_context": 2}, "context range"),
        ({"drop_state": 1.5}, "probability"),
    ],
)
def test_bad_schedules_raise(overrides: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ForcingSchedule(**overrides)  # type: ignore[arg-type]
