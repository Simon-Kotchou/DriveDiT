"""Action sensitivity on toy dynamics: the command shifts the latent pattern (``ShiftDynamics``), and a tiny model
trained for 400 CPU steps with the v1 loss predicts the commanded next frame. With the commands zeroed or shuffled
across windows, the same model's prediction of that frame falls apart, so the prediction depends on ``action_cmd``.

The toy latent has 64 dims per frame, so the resolution shift sqrt(64 / 4096) = 0.125 would put almost every draw
near t = 0, where the frame is visible in its own input and the command is not needed. The test sets alpha = 4 (the
alpha0 x 4 high-noise arm, arch 3.2 / Krea 3) and head dim 64 (the v1 head dim). Measured on this toy (4 seeds, 400
steps): last-frame MSE 0.05-0.14 with the true commands against 0.6-1.6 zeroed or shuffled. The gap is about 3x at
alpha 1 or at head dim 16, and at alpha 0.125 the model does not use the command within 800 steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from drivedit.model import WorldModel, WorldModelConfig
from drivedit.train import ForcingSchedule, OptimConfig, RunConfig, Trainer
from tests.drivedit.model.toy import ShiftDynamics, collate, tiny_config

STEPS = 400
CFG = tiny_config(latent_height=2, latent_width=8, dim=64, heads=1, kv_heads=1, head_dim=64, rope_split=(16, 24, 24))


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> WorldModel:
    run = RunConfig(
        run_id="toy-shift",
        model=CFG,
        schedule=ForcingSchedule(shift=4.0),
        optim=OptimConfig(lr=3e-3, warmup_steps=20),
        batch_size=32,
        steps=STEPS,
        precision="fp32",
        log_every=STEPS,
        checkpoint_every=STEPS,
    )
    trainer = Trainer(
        run, ShiftDynamics(CFG, 4096, seed=1), Path(tmp_path_factory.mktemp("toy")), device=torch.device("cpu")
    )
    trainer.fit()
    return trainer.model.eval()


def _last_frame_error(model: WorldModel, batch: dict[str, Tensor], action_cmd: Tensor, t_last: float) -> float:
    """MSE of the predicted last frame, noised to ``t_last``, given clean context frames and ``action_cmd``."""
    x = batch["latents"]
    eps = torch.randn(x[:, -1].shape, generator=torch.Generator().manual_seed(0))
    z = x.clone()
    z[:, -1] = (1 - t_last) * x[:, -1] + t_last * eps
    t = torch.zeros(x.shape[:2])
    t[:, -1] = t_last
    with torch.no_grad():
        x0 = model(z, t, action_cmd, batch["state"]).x0
    return float((x0[:, -1] - x[:, -1]).pow(2).mean())


def _held_out(cfg: WorldModelConfig) -> dict[str, Tensor]:
    data = ShiftDynamics(cfg, 256, seed=2)
    return collate([data[i] for i in range(len(data))])


@pytest.mark.parametrize("t_last", [0.5, 1.0])
def test_the_prediction_follows_the_command(trained: WorldModel, t_last: float) -> None:
    batch = _held_out(CFG)
    action_cmd = batch["action_cmd"]
    zeroed = action_cmd.clone()
    zeroed[..., 0] = 0.0
    shuffled = action_cmd[torch.randperm(len(action_cmd), generator=torch.Generator().manual_seed(3))]
    true = _last_frame_error(trained, batch, action_cmd, t_last)
    errors = {"zeroed": _last_frame_error(trained, batch, zeroed, t_last)}
    errors["shuffled"] = _last_frame_error(trained, batch, shuffled, t_last)
    for name, err in errors.items():
        assert true < 0.3 * err, f"t={t_last}: true commands {true:.3f}, {name} {err:.3f}"
