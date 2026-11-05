"""The trainer: config round trips, warm-up, checkpoint save / load / resume determinism, the non-finite stop, weight
decay groups, latent statistics and the CLI. DDP is in test_ddp.py."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from drivedit.model import AUX_HEADS, WorldModel, WorldModelConfig
from drivedit.train import DataConfig, ForcingSchedule, OptimConfig, RunConfig, Trainer, load_world_model
from drivedit.train.cli import main
from drivedit.train.config import from_dict
from drivedit.train.data import channel_stats
from tests.drivedit.model.toy import ShiftDynamics, tiny_config

CPU = torch.device("cpu")


def _cfg(**overrides: object) -> RunConfig:
    base = RunConfig(
        run_id="test",
        model=tiny_config(aux_heads=AUX_HEADS),
        optim=OptimConfig(lr=1e-3, warmup_steps=2),
        batch_size=4,
        steps=6,
        precision="fp32",
        log_every=3,
        checkpoint_every=3,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _params(model: WorldModel) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def _assert_equal(a: dict[str, Tensor], b: dict[str, Tensor]) -> None:
    assert a.keys() == b.keys()
    for name, value in a.items():
        assert torch.equal(value, b[name]), name


def test_run_config_round_trips_through_json() -> None:
    cfg = _cfg(
        model=tiny_config(aux_heads=("future_pose",), shortcut_levels=4),
        schedule=ForcingSchedule(shift=2.0, max_context=3, weighting="x_ramp", shortcut=True),
        data=DataConfig(root="datasets", sessions=("s1", "s2"), codec_id="cadre-1x16x16"),
    )
    assert RunConfig.from_json(cfg.to_json()) == cfg
    assert RunConfig.from_json(_cfg().to_json()) == _cfg()


def test_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        from_dict(OptimConfig, {"lr": 1.0, "momentum": 0.9})


def test_shortcut_schedule_needs_a_step_conditioned_model() -> None:
    with pytest.raises(ValueError, match="shortcut_levels"):
        _cfg(schedule=ForcingSchedule(shortcut=True))


def test_warm_up_is_linear_then_constant(tmp_path: Path) -> None:
    trainer = Trainer(
        _cfg(optim=OptimConfig(lr=1e-3, warmup_steps=10)), ShiftDynamics(tiny_config(), 8, seed=0), tmp_path
    )
    assert trainer.lr_at(0) == pytest.approx(1e-4)
    assert trainer.lr_at(9) == pytest.approx(1e-3)
    assert trainer.lr_at(5000) == pytest.approx(1e-3)


def test_resume_continues_exactly(tmp_path: Path) -> None:
    cfg = _cfg()
    data = ShiftDynamics(cfg.model, 32, seed=1)
    straight = Trainer(cfg, data, tmp_path / "straight", device=CPU)
    straight.fit()
    assert straight.step == 6
    checkpoints = sorted((tmp_path / "straight" / "checkpoints").glob("step_*.pt"))
    assert [p.name for p in checkpoints] == ["step_00000003.pt", "step_00000006.pt"]

    resumed_dir = tmp_path / "resumed" / "checkpoints"
    resumed_dir.mkdir(parents=True)
    shutil.copy(checkpoints[0], resumed_dir / checkpoints[0].name)
    resumed = Trainer(cfg, data, tmp_path / "resumed", device=CPU)
    assert resumed.resume()
    assert resumed.step == 3
    resumed.fit()
    _assert_equal(_params(resumed.model), _params(straight.model))
    _assert_equal(resumed.ema.state_dict(), straight.ema.state_dict())
    for sa, sb in zip(
        resumed.optimizer.state_dict()["state"].values(), straight.optimizer.state_dict()["state"].values(), strict=True
    ):
        assert torch.equal(sa["exp_avg"], sb["exp_avg"])
        assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])
    log = [json.loads(line) for line in (tmp_path / "straight" / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [3, 6]
    assert {"loss", "flow", "aux/future_pose", "aux/contact_events", "lr", "grad_norm"} <= log[0].keys()


def test_load_world_model_gives_the_ema_or_the_raw_weights(tmp_path: Path) -> None:
    cfg = _cfg(steps=3)
    trainer = Trainer(cfg, ShiftDynamics(cfg.model, 16, seed=2), tmp_path, device=CPU)
    trainer.fit()
    path = tmp_path / "checkpoints" / "step_00000003.pt"
    ema = load_world_model(path)
    assert not ema.training
    _assert_equal(_params(ema), trainer.ema.state_dict())
    _assert_equal(_params(load_world_model(path, ema=False)), _params(trainer.model))
    assert not torch.equal(ema.out.weight, trainer.model.out.weight)


def test_a_run_dir_belongs_to_one_config(tmp_path: Path) -> None:
    data = ShiftDynamics(tiny_config(), 8, seed=0)
    Trainer(_cfg(), data, tmp_path)
    with pytest.raises(ValueError, match="different run config"):
        Trainer(_cfg(seed=1), data, tmp_path)


class _BadWindows(ShiftDynamics):
    """NaN in the last frame, or finite latents of alternating sign at 1e20, whose squared error overflows to inf
    while the gradients stay finite (the Codex DD-20 reproduction: loss inf, grad_norm 0, a step taken)."""

    def __init__(self, cfg: WorldModelConfig, size: int, *, value: float) -> None:
        super().__init__(cfg, size, seed=0)
        self.value = value

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        item = super().__getitem__(i)
        if math.isnan(self.value):
            item["latents"][-1] = self.value
        else:
            sign = torch.ones_like(item["latents"]).flatten()
            sign[1::2] = -1.0
            item["latents"] = self.value * sign.view_as(item["latents"])
        return item


@pytest.mark.parametrize("value", [float("nan"), 1e20])
def test_a_non_finite_loss_stops_before_the_update(tmp_path: Path, value: float) -> None:
    cfg = _cfg(model=tiny_config(), schedule=ForcingSchedule(min_context=3))
    trainer = Trainer(cfg, _BadWindows(cfg.model, 8, value=value), tmp_path, device=CPU)
    before = _params(trainer.model)
    with pytest.raises(FloatingPointError, match="non-finite loss at step 0"):
        trainer.fit()
    assert trainer.step == 0
    assert not trainer.optimizer.state
    _assert_equal(_params(trainer.model), before)


def test_weight_decay_spares_biases_gains_and_the_nulls(tmp_path: Path) -> None:
    trainer = Trainer(_cfg(), ShiftDynamics(tiny_config(), 8, seed=0), tmp_path, device=CPU)
    decay = {id(p): group["weight_decay"] for group in trainer.optimizer.param_groups for p in group["params"]}
    model = trainer.model
    assert len(decay) == len(list(model.parameters()))
    assert decay[id(model.cond.null)] == 0.0  # [G, D]: matrix-shaped, but not a weight
    assert decay[id(model.adaln.bias)] == 0.0
    assert decay[id(model.blocks[0].mod_bias)] == 0.0
    assert decay[id(model.blocks[0].attn.q_norm.gamma)] == 0.0
    assert decay[id(model.adaln.weight)] == 0.1
    assert decay[id(model.cond.dir_cmd.weight)] == 0.1  # an embedding table


def test_channel_stats_pool_every_row() -> None:
    g = np.random.default_rng(0)
    chunks = [g.normal(2.0, 3.0, size=(n, 3, 2, 4)).astype(np.float16) for n in (5, 1, 7)]
    mean, std = channel_stats(chunks)
    pooled = np.concatenate(chunks).astype(np.float64).transpose(1, 0, 2, 3).reshape(3, -1)
    np.testing.assert_allclose(mean, pooled.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(std, pooled.std(axis=1), rtol=1e-9)
    with pytest.raises(ValueError, match="no valid"):
        channel_stats([])


def test_cli_prints_a_preset_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--print-config", "small_110m"]) == 0
    cfg = RunConfig.from_json(capsys.readouterr().out)
    assert (cfg.model.dim, cfg.model.depth) == (768, 16)


def test_cli_needs_data_and_the_data_layer(tmp_path: Path) -> None:
    config = tmp_path / "run.json"
    config.write_text(_cfg().to_json())
    with pytest.raises(ValueError, match=r"RunConfig\.data"):
        main(["--config", str(config), "--run-dir", str(tmp_path / "run")])
