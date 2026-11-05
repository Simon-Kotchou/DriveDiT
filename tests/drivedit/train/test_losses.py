"""The diffusion-forcing loss on DD-11's window dict: it can overfit one batch, the shortcut target is exact for an
exact denoiser, the aux heads follow features_v1 and are aligned to the transition into each frame, and masked
entries carry no loss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from drivedit.model import AUX_HEADS, WorldModel, WorldModelConfig
from drivedit.model.layers import symlog
from drivedit.train.losses import WM_VALID, aux_losses, diffusion_forcing_loss, shortcut_target
from drivedit.train.schedule import ForcingSchedule
from tests.drivedit.model.toy import WINDOW_KEYS, ShiftDynamics, collate, randomised, tiny_config

FEATURES_V1 = Path(__file__).resolve().parents[3] / "docs" / "capture_contract" / "features_v1.json"


def _batch(cfg_window: int = 4, size: int = 8, seed: int = 0) -> dict[str, Tensor]:
    data = ShiftDynamics(tiny_config(window=cfg_window), size, seed=seed)
    return collate([data[i] for i in range(size)])


def test_overfits_one_batch() -> None:
    torch.manual_seed(0)
    model = WorldModel(tiny_config(dim=64, heads=4, kv_heads=2))
    batch = _batch()
    schedule = ForcingSchedule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    losses = []
    for _ in range(150):
        loss, _ = diffusion_forcing_loss(model, batch, schedule=schedule, rng=torch.Generator().manual_seed(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 20, (losses[0], losses[-1])


def test_shortcut_target_is_exact_for_an_exact_denoiser() -> None:
    """On a point mass x*, the exact denoiser always predicts x*, so two half steps land on the straight path and the
    bootstrap target is x* itself; context frames keep their input."""
    g = torch.Generator().manual_seed(1)
    x_star = torch.randn(2, 3, 4, 3, 4, generator=g, dtype=torch.float64)
    eps = torch.randn(x_star.shape, generator=g, dtype=torch.float64)
    target = torch.tensor([[False, True, True], [False, False, True]])
    step = torch.tensor([[0.125, 0.5, 1.0], [0.125, 0.125, 0.25]], dtype=torch.float64)
    t = torch.where(target, torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.0, 0.75]], dtype=torch.float64), 0.0)
    z = (1 - t[..., None, None, None]) * x_star + t[..., None, None, None] * eps

    def exact(z_in: Tensor, t_in: Tensor, d: Tensor) -> Tensor:
        return x_star

    got = shortcut_target(exact, z, t, step, target)
    torch.testing.assert_close(got[target], x_star[target])
    torch.testing.assert_close(got[~target], z[~target])


def test_shortcut_forcing_trains_the_step_embedding() -> None:
    model = randomised(WorldModel(tiny_config(shortcut_levels=4)))  # the zero-init AdaLN would block the gradient
    schedule = ForcingSchedule(shortcut=True, shortcut_fraction=0.5)
    loss, logs = diffusion_forcing_loss(model, _batch(), schedule=schedule, rng=torch.Generator().manual_seed(2))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.cond.step_embed is not None
    assert model.cond.step_embed.weight.grad is not None
    assert model.cond.step_embed.weight.grad.abs().sum() > 0
    assert set(logs) == {"flow", "loss"}


def test_aux_heads_and_masks_follow_features_v1() -> None:
    """Every aux head is a world-model target in the contract, and WM_VALID indexes wm_valid in masks."""
    spec = json.loads(FEATURES_V1.read_text())
    tensors = {t["name"]: t for t in spec["tensors"]}
    for name in AUX_HEADS:
        assert tensors[name]["access"]["world_model"]["target"] is True, name
    for name in ("state", "action_exec"):
        assert tensors[name]["access"]["world_model"]["target"] is False, name
    masks = [c["name"] for c in tensors["masks"]["channels"]]
    assert masks == ["policy_valid", "wm_valid", "warmup", "env_transition", "teleport"]
    assert masks[WM_VALID] == "wm_valid"


def test_aux_heads_predict_the_transition_into_each_frame() -> None:
    """Frame f's heads describe t = f - 1: future_pose[f-1] and contact_events[f-1]."""
    cfg = tiny_config(window=4, aux_heads=AUX_HEADS)
    batch = _batch(cfg_window=4, size=2)
    batch["future_pose"][0, 0, -1] = float("nan")  # no target at the far horizon of step 0
    batch["contact_events"][1, 1] = torch.tensor([1.0, 3.0])  # a contact of class 3 in the transition into frame 2
    scale = torch.tensor(cfg.future_pose_scale)
    target_pose = symlog(batch["future_pose"] / scale).nan_to_num()
    logit = 40.0 * (batch["contact_events"][..., 0] - 0.5)
    shifted = {
        "future_pose": torch.cat((torch.zeros_like(target_pose[:, :1]), target_pose[:, :-1]), dim=1),
        "contact_events": torch.cat((torch.zeros_like(logit[:, :1]), logit[:, :-1]), dim=1),
    }
    losses = aux_losses(shifted, batch, cfg)
    assert set(losses) == set(AUX_HEADS)
    assert losses["future_pose"].item() == pytest.approx(0.0, abs=1e-9)
    assert losses["contact_events"].item() < 1e-6
    unshifted = aux_losses({"future_pose": target_pose, "contact_events": logit}, batch, cfg)
    assert unshifted["future_pose"].item() > 0.01
    assert unshifted["contact_events"].item() > 0.01


class _DD11Windows(Dataset[dict[str, Tensor]]):
    """Items with exactly the keys and dtypes of DD-11's LatentWindowDataset: NaN future_pose entries, a contact, a
    frame that is not wm-valid and a last frame without a latent (zeroed, valid False, wm_valid 0)."""

    def __init__(self, cfg: WorldModelConfig, size: int) -> None:
        self.toy = ShiftDynamics(cfg, size, seed=5)

    def __len__(self) -> int:
        return len(self.toy)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        item = self.toy[i]
        item["future_pose"][0, -2:] = float("nan")  # frame 1's target, past the end of the episode
        item["contact_events"][0] = torch.tensor([1.0, 3.0])  # frame 1's target: a contact of class 3
        item["masks"][2, 1] = 0  # frame 2 is not wm-valid
        item["latents"][-1] = 0.0
        item["valid"][-1] = False
        item["masks"][-1, 1] = 0
        return item


def _dd11_batch(cfg: WorldModelConfig) -> dict[str, Tensor]:
    batch: dict[str, Tensor] = next(iter(DataLoader(_DD11Windows(cfg, 4), batch_size=4)))
    return batch


def test_the_dd11_window_dict_trains_and_its_masked_entries_carry_no_loss() -> None:
    cfg = tiny_config(window=5, aux_heads=AUX_HEADS)
    model = randomised(WorldModel(cfg))
    batch = _dd11_batch(cfg)
    assert {k: v.dtype for k, v in batch.items()} == WINDOW_KEYS
    assert batch["future_pose"].shape == (4, 5, cfg.future_pose_horizon, 3)
    assert batch["contact_events"].shape == (4, 5, 2)
    assert batch["masks"].shape == (4, 5, 5)

    def loss_of(b: dict[str, Tensor]) -> Tensor:
        loss, logs = diffusion_forcing_loss(model, b, schedule=ForcingSchedule(), rng=torch.Generator().manual_seed(6))
        assert all(torch.isfinite(v) for v in logs.values()), logs
        return loss.detach()

    base = loss_of(batch)
    assert torch.isfinite(base)
    assert base > 0

    def changed(name: str, index: tuple[slice | int, ...], value: float) -> dict[str, Tensor]:
        out = {k: v.clone() for k, v in batch.items()}
        out[name][index] = value
        return out

    ignored = {
        "another non-finite future_pose entry": changed("future_pose", (slice(None), 0, slice(-2, None)), float("inf")),
        "the latent of the frame without one": changed("latents", (slice(None), -1), 3.0),
        "the targets of the frame without a latent": changed("future_pose", (slice(None), -2), 7.0),
        "the targets of the frame that is not wm-valid": changed("contact_events", (slice(None), 1, 0), 1.0),
    }
    for what, b in ignored.items():
        assert torch.equal(loss_of(b), base), what
    counted = changed("future_pose", (slice(None), 0, slice(-2, None)), 0.0)  # the same entries, now finite
    assert not torch.equal(loss_of(counted), base)


def test_invalid_frames_and_wm_invalid_frames_carry_no_loss() -> None:
    torch.manual_seed(0)
    model = WorldModel(tiny_config(aux_heads=AUX_HEADS))
    batch = _batch()
    rng = torch.Generator().manual_seed(4)
    loss, _ = diffusion_forcing_loss(model, batch, schedule=ForcingSchedule(), rng=rng)
    assert loss.item() > 0
    masked = {**batch, "masks": torch.zeros_like(batch["masks"])}
    loss, logs = diffusion_forcing_loss(model, masked, schedule=ForcingSchedule(), rng=rng)
    assert loss.item() == 0.0
    assert logs["aux/future_pose"].item() == 0.0


def test_a_missing_key_raises() -> None:
    model = WorldModel(tiny_config(aux_heads=("future_pose",)))
    batch = _batch()
    for name in ("future_pose", "masks"):
        with pytest.raises(KeyError, match=name):
            diffusion_forcing_loss(
                model, {k: v for k, v in batch.items() if k != name}, schedule=ForcingSchedule(), rng=torch.Generator()
            )
