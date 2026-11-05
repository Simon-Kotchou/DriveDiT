"""Frame-block causality: a frame's outputs never depend on later frames, and frame f's command conditions frame f+1."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from drivedit.model import AUX_HEADS, WorldModel, WorldModelOutput
from tests.drivedit.model.toy import randomised, tiny_config, window_inputs

FRAMES = 5
CFG = tiny_config(window=FRAMES, aux_heads=AUX_HEADS)


@pytest.fixture(scope="module")
def model() -> WorldModel:
    return randomised(WorldModel(CFG).double()).eval()


def _run(model: WorldModel, inputs: dict[str, Tensor]) -> list[Tensor]:
    with torch.no_grad():
        out: WorldModelOutput = model(inputs["x"], inputs["t"], inputs["action_cmd"], inputs["state"])
    return [out.x0, *out.aux.values()]


def _base() -> dict[str, Tensor]:
    return window_inputs(CFG, 2, FRAMES, seed=1, dtype=torch.float64)


def _assert_same_before(a: list[Tensor], b: list[Tensor], frame: int) -> None:
    for pa, pb in zip(a, b, strict=True):
        torch.testing.assert_close(pa[:, :frame], pb[:, :frame], rtol=0, atol=1e-12)


def _assert_changed_at(a: list[Tensor], b: list[Tensor], frame: int) -> None:
    assert not torch.allclose(a[0][:, frame], b[0][:, frame]), f"frame {frame} ignored the change"


@pytest.mark.parametrize("k", range(FRAMES))
def test_changing_frame_k_never_changes_earlier_frames(model: WorldModel, k: int) -> None:
    base = _base()
    changed = {**base, "x": base["x"].clone(), "t": base["t"].clone()}
    changed["x"][:, k] += 1.0
    changed["t"][:, k] = 1.0 - changed["t"][:, k]
    a, b = _run(model, base), _run(model, changed)
    _assert_same_before(a, b, k)
    for frame in range(k, FRAMES):
        _assert_changed_at(a, b, frame)


@pytest.mark.parametrize("name", ["action_cmd", "state"])
@pytest.mark.parametrize("k", range(FRAMES - 1))
def test_step_k_conditioning_reaches_frame_k_plus_1_first(model: WorldModel, name: str, k: int) -> None:
    base = _base()
    changed = {**base, name: base[name].clone()}
    changed[name][:, k, :2] += torch.tensor([0.05, 2.0], dtype=torch.float64)
    a, b = _run(model, base), _run(model, changed)
    _assert_same_before(a, b, k + 1)
    _assert_changed_at(a, b, k + 1)


def test_the_last_command_and_state_are_unused(model: WorldModel) -> None:
    base = _base()
    changed = {**base, "action_cmd": base["action_cmd"].clone(), "state": base["state"].clone()}
    changed["action_cmd"][:, -1] += 1.0
    changed["state"][:, -1] += 1.0
    _assert_same_before(_run(model, base), _run(model, changed), FRAMES)
