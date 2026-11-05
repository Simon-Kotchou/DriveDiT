"""The inverse-dynamics model and the donor-gap ARR (drivedit.eval.idm) on toy frame pairs (torch, CPU, fast).

Toy dynamics: a frame is a few random 2-D sinusoids; the next frame is the frame shifted sideways by
2 x kappa_target pixels (sub-pixel, through the FFT) with 0.5 x a_target added to channel 0; dir_cmd is constant.
An IDM must compare the two frames to read the commands back.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from drivedit.eval.idm import (
    IdmConfig,
    IdmTrainConfig,
    InverseDynamicsModel,
    PairSet,
    arr_r2_with_donor_gap,
    evaluate_idm,
    match_donors,
    predict_actions,
    train_idm,
)

C, H, W = 2, 8, 16
CFG = IdmConfig(in_channels=C, width=32, n_down=2, pool_hw=(2, 4), hidden=64)
TRAIN = IdmTrainConfig(steps=300, batch_size=64, lr=1e-3, seed=0)


def _frames(rng: np.random.Generator, n: int) -> np.ndarray:
    ky, kx = np.arange(H)[:, None], np.arange(W)[None, :]
    x = np.zeros((n, C, H, W))
    for _ in range(4):
        fx = rng.integers(1, 4, size=(n, C, 1, 1))
        fy = rng.integers(0, 3, size=(n, C, 1, 1))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, C, 1, 1))
        x += np.sin(2.0 * np.pi * (fx * kx / W + fy * ky / H) + phase)
    return (x / 2.0).astype(np.float32)


def _step(x: np.ndarray, action: np.ndarray) -> np.ndarray:
    k = np.fft.fftfreq(W) * W
    shift = np.exp(-2j * np.pi * k[None, None, None, :] * (2.0 * action[:, 0])[:, None, None, None] / W)
    y = np.real(np.fft.ifft(np.fft.fft(x, axis=-1) * shift, axis=-1))
    y[:, 0] += 0.5 * action[:, 1][:, None, None]
    return y.astype(np.float32)


def _actions(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.stack([rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), np.ones(n)], axis=1)


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x0, a = _frames(rng, 768), _actions(rng, 768)
    x1 = _step(x0, a) + 0.01 * rng.normal(size=x0.shape).astype(np.float32)
    t0, ta = _frames(rng, 256), _actions(rng, 256)
    return {"x0": x0, "x1": x1, "a": a, "t0": t0, "t1": _step(t0, ta), "ta": ta}


@pytest.fixture(scope="module")
def trained(data: dict[str, np.ndarray]) -> InverseDynamicsModel:
    model = InverseDynamicsModel(CFG, seed=0)
    train_idm(model, data["x0"], data["x1"], data["a"], cfg=TRAIN, device="cpu")
    return model


def test_idm_learns_the_commands_from_frame_pairs(trained: InverseDynamicsModel, data: dict[str, np.ndarray]) -> None:
    r2 = evaluate_idm(trained, data["t0"], data["t1"], data["ta"], batch_size=128, device="cpu")
    assert r2[0] > 0.9  # kappa_target: the sideways shift
    assert r2[1] > 0.9  # a_target: the brightness step
    assert np.isnan(r2[2])  # dir_cmd is constant: R² undefined
    pred = predict_actions(trained, data["t0"], data["t1"], batch_size=128, device="cpu")
    np.testing.assert_allclose(pred[:, 2], 1.0, atol=1e-5)  # a constant channel is predicted at its value


def test_idm_trained_on_shuffled_commands_reads_nothing(data: dict[str, np.ndarray]) -> None:
    model = InverseDynamicsModel(CFG, seed=0)
    shuffled = data["a"][np.random.default_rng(1).permutation(len(data["a"]))]
    train_idm(model, data["x0"], data["x1"], shuffled, cfg=TRAIN, device="cpu")
    r2 = evaluate_idm(model, data["t0"], data["t1"], data["ta"], batch_size=128, device="cpu")
    assert r2[0] < 0.2
    assert r2[1] < 0.2


def test_training_is_deterministic_and_leaves_the_global_rng_alone(data: dict[str, np.ndarray]) -> None:
    cfg = IdmTrainConfig(steps=5, batch_size=16, seed=3)
    before = torch.random.get_rng_state()
    runs = []
    for _ in range(2):
        model = InverseDynamicsModel(CFG, seed=7)
        runs.append(train_idm(model, data["x0"][:64], data["x1"][:64], data["a"][:64], cfg=cfg, device="cpu"))
    assert runs[0] == runs[1]
    assert torch.equal(before, torch.random.get_rng_state())


def test_donor_gap_separates_a_following_model_from_an_ignoring_one(
    trained: InverseDynamicsModel, data: dict[str, np.ndarray]
) -> None:
    rng = np.random.default_rng(2)
    n = len(data["ta"])
    donors = match_donors(rng.uniform(5.0, 12.0, n), groups=list(range(n)), rng=rng)
    real = PairSet(data["t0"], data["t1"], data["ta"])
    follows = PairSet(data["t0"], _step(data["t0"], data["ta"]), data["ta"])
    ignores = PairSet(data["t0"], _step(data["t0"], _actions(rng, n)), data["ta"])  # its own commands
    good = arr_r2_with_donor_gap(trained, real, follows, donors=donors, batch_size=128, device="cpu")
    bad = arr_r2_with_donor_gap(trained, real, ignores, donors=donors, batch_size=128, device="cpu")
    assert good.scored == ("kappa_target", "a_target")
    assert good.arr > 0.95
    assert good.gap > 1.0  # swapping the donor's commands in drops R² from about 1 to about -1
    assert good.real_gap > 1.0
    assert bad.arr < 0.0
    assert abs(bad.gap) < 0.3
    assert bad.real_gap == pytest.approx(good.real_gap)  # the ceiling does not depend on the rollout


def test_match_donors_never_picks_the_same_clip_and_respects_speed() -> None:
    rng = np.random.default_rng(3)
    speed = rng.uniform(5.0, 12.0, 200)
    groups = list(np.repeat(np.arange(40), 5))
    donors = match_donors(speed, groups=groups, rng=rng, max_speed_diff_kmh=5.0)
    assert all(groups[d] != groups[i] for i, d in enumerate(donors))
    assert np.all(np.abs(speed[donors] - speed) <= 5.0 / 3.6)
    with pytest.raises(ValueError, match="no donor"):
        match_donors([1.0, 20.0], groups=["a", "b"], rng=rng)


def test_bad_inputs_raise(trained: InverseDynamicsModel, data: dict[str, np.ndarray]) -> None:
    real = PairSet(data["t0"][:8], data["t1"][:8], data["ta"][:8])
    with pytest.raises(ValueError, match="another window"):
        arr_r2_with_donor_gap(trained, real, real, donors=np.arange(8), batch_size=8, device="cpu")
    with pytest.raises(ValueError, match="frames"):
        trained(torch.zeros(1, C + 1, H, W), torch.zeros(1, C + 1, H, W))
    with pytest.raises(ValueError, match="multiple of 8"):
        IdmConfig(in_channels=3, width=12)
