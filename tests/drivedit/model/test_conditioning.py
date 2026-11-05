"""Conditioning: frame 0 and dropped groups use the learned nulls; directions are lookups; shapes are checked."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from drivedit.model import COND_GROUPS, WorldModel
from tests.drivedit.model.toy import randomised, tiny_config, window_inputs

FRAMES = 4
CFG = tiny_config(window=FRAMES)
ACTION, STATE = COND_GROUPS.index("action_cmd"), COND_GROUPS.index("state")


@pytest.fixture(scope="module")
def model() -> WorldModel:
    return randomised(WorldModel(CFG).double(), seed=3).eval()


def _x0(model: WorldModel, inputs: dict[str, Tensor], drop: Tensor | None = None) -> Tensor:
    with torch.no_grad():
        out = model(inputs["x"], inputs["t"], inputs["action_cmd"], inputs["state"], drop_cond=drop)
    return out.x0


def _inputs(seed: int) -> dict[str, Tensor]:
    return window_inputs(CFG, 2, FRAMES, seed=seed, dtype=torch.float64)


def test_frame_zero_uses_the_nulls(model: WorldModel) -> None:
    base, other = _inputs(1), _inputs(2)
    other = {**other, "x": base["x"], "t": base["t"]}  # same latents and times, different commands and states
    frame0 = _x0(model, base)[:, 0]
    torch.testing.assert_close(_x0(model, other)[:, 0], frame0, rtol=0, atol=0)
    everything_dropped = torch.ones(2, FRAMES, dtype=torch.bool)
    torch.testing.assert_close(_x0(model, base, everything_dropped)[:, 0], frame0, rtol=0, atol=0)


@pytest.mark.parametrize(("dropped", "kept"), [("action_cmd", "state"), ("state", "action_cmd")])
def test_a_dropped_group_is_ignored_and_the_other_is_not(model: WorldModel, dropped: str, kept: str) -> None:
    f = 2
    drop = torch.zeros(2, FRAMES, len(COND_GROUPS), dtype=torch.bool)
    drop[:, f, COND_GROUPS.index(dropped)] = True
    base = _inputs(4)
    for name, should_change in ((dropped, False), (kept, True)):
        changed = {**base, name: base[name].clone()}
        changed[name][:, f - 1, :2] += torch.tensor([0.05, 2.0], dtype=torch.float64)
        same = torch.allclose(_x0(model, changed, drop)[:, f], _x0(model, base, drop)[:, f], rtol=0, atol=1e-12)
        assert same != should_change, f"{name} at step {f - 1} with {dropped} dropped"


def test_a_frame_mask_drops_every_group(model: WorldModel) -> None:
    base = _inputs(5)
    per_frame = torch.zeros(2, FRAMES, dtype=torch.bool)
    per_frame[:, 1:] = True
    every_group = per_frame[..., None].expand(-1, -1, len(COND_GROUPS))
    torch.testing.assert_close(_x0(model, base, per_frame), _x0(model, base, every_group), rtol=0, atol=0)


def test_dropped_frames_share_the_null_of_frame_zero(model: WorldModel) -> None:
    """With every group dropped, the conditioning vector is the same for any command and state."""
    t = torch.full((1, 3), 0.4, dtype=torch.float64)
    null = torch.ones(1, 3, len(COND_GROUPS), dtype=torch.bool)
    inputs = _inputs(6)
    c = model.cond(t, inputs["action_cmd"][:1, :3], inputs["state"][:1, :3], null, None)
    torch.testing.assert_close(c, c[:, :1].expand_as(c), rtol=0, atol=0)


def test_direction_is_a_lookup_embedding(model: WorldModel) -> None:
    base = _inputs(7)
    outputs = []
    for direction in (-1.0, 0.0, 1.0):
        inputs = {**base, "action_cmd": base["action_cmd"].clone()}
        inputs["action_cmd"][:, 0, 2] = direction
        outputs.append(_x0(model, inputs)[:, 1])
    assert all(not torch.allclose(outputs[i], outputs[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    bad = {**base, "action_cmd": base["action_cmd"].clone()}
    bad["action_cmd"][:, 0, 2] = 2.0
    with pytest.raises(IndexError):
        _x0(model, bad)


def test_shapes_are_checked(model: WorldModel) -> None:
    base = _inputs(8)
    with pytest.raises(ValueError, match="action_cmd"):
        model(base["x"], base["t"], base["action_cmd"][..., :2], base["state"])
    with pytest.raises(ValueError, match="x: shape"):
        model(base["x"][..., :2], base["t"], base["action_cmd"], base["state"])
