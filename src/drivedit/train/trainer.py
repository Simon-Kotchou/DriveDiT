"""The training loop: bf16 autocast, AdamW, warm-up, clipping, Power-EMA, checkpoints, and DDP under torchrun.

Determinism: the data order and every random draw of the loss are functions of ``(seed, rank, step)``, so a run resumed
from a checkpoint (model, EMA, optimiser, step) continues exactly as the uninterrupted run would. A process started
by ``torchrun`` (``WORLD_SIZE`` > 1) joins the process group, wraps the model in DistributedDataParallel and takes its
own slice of every global batch (efficiency 3.4 item 1: DDP, not FSDP, at <= 400M).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from drivedit.model.world_model import WorldModel
from drivedit.train.config import RunConfig
from drivedit.train.losses import diffusion_forcing_loss

CHECKPOINT_FORMAT = "drivedit.checkpoint/1"


@dataclass(frozen=True)
class DistInfo:
    """This process's place in a torchrun job (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``; 0, 1, 0 when run alone)."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @classmethod
    def from_env(cls) -> DistInfo:
        env = os.environ
        return cls(int(env.get("RANK", 0)), int(env.get("WORLD_SIZE", 1)), int(env.get("LOCAL_RANK", 0)))


def _mix(*values: int) -> int:
    """A deterministic 63-bit seed from integers (splitmix-style)."""
    h = 0x9E3779B97F4A7C15
    for v in values:
        h = ((h ^ (v & 0xFFFFFFFFFFFFFFFF)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
    return h & 0x7FFFFFFFFFFFFFFF


class WindowBatchSampler(Sampler[list[int]]):
    """Endless batches of dataset indices from ``start_step``: epoch e is a permutation seeded by (seed, e), cut into
    global batches of ``batch_size * world_size``; rank r takes its contiguous ``batch_size`` slice. The last partial
    global batch of an epoch is dropped."""

    def __init__(self, size: int, batch_size: int, *, seed: int, rank: int, world_size: int, start_step: int) -> None:
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.rank, self.world_size, self.start_step = rank, world_size, start_step
        self.per_epoch = size // (batch_size * world_size)
        if self.per_epoch < 1:
            raise ValueError(f"{size} windows cannot fill one global batch of {batch_size} x {world_size}")

    def __iter__(self) -> Iterator[list[int]]:
        step = self.start_step
        while True:
            epoch, i = divmod(step, self.per_epoch)
            perm = torch.randperm(self.size, generator=torch.Generator().manual_seed(_mix(self.seed, epoch)))
            start = (i * self.world_size + self.rank) * self.batch_size
            yield perm[start : start + self.batch_size].tolist()
            step += 1


class PowerEMA:
    """EDM2 power-function EMA of the parameters: ``beta_t = (1 - 1/t) ** (gamma + 1)``, kept in float32."""

    def __init__(self, model: nn.Module, gamma: float) -> None:
        self.gamma = gamma
        self.shadow = {name: p.detach().float().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Fold in the parameters after optimiser step ``step`` (1-based)."""
        beta = (1.0 - 1.0 / step) ** (self.gamma + 1.0)
        names = list(self.shadow)
        params = dict(model.named_parameters())
        torch._foreach_lerp_([self.shadow[n] for n in names], [params[n].detach().float() for n in names], 1.0 - beta)

    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != set(self.shadow):
            raise KeyError("EMA state does not match the model's parameters")
        for name, value in state.items():
            self.shadow[name].copy_(value)


#: Matrix-shaped parameters that are not weights: the learned conditioning nulls ``[G, D]``.
NO_DECAY: tuple[str, ...] = ("cond.null",)


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay on weight matrices and embedding tables only, never on biases, norm gains, AdaLN biases or the
    learned nulls (``NO_DECAY``)."""
    params = list(model.named_parameters())
    decay = [p for name, p in params if p.ndim >= 2 and name not in NO_DECAY]
    no_decay = [p for name, p in params if p.ndim < 2 or name in NO_DECAY]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_model(cfg: RunConfig) -> WorldModel:
    """The run's model, initialised from ``cfg.seed`` without touching the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return WorldModel(cfg.model)


class Trainer:
    """A plain loop over ``dataset`` windows (the DRIVEDIT_INTERFACES section 4 dict) for ``cfg.steps`` steps.

    ``run_dir`` receives ``run_config.json``, ``log.jsonl`` (rank 0) and ``checkpoints/step_<step>.pt``.
    """

    def __init__(
        self, cfg: RunConfig, dataset: Dataset[Any], run_dir: Path, *, device: torch.device | None = None
    ) -> None:
        self.cfg, self.dataset, self.run_dir = cfg, dataset, run_dir
        self.dist = DistInfo.from_env()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cuda":  # no index: this process's GPU
            device = torch.device("cuda", self.dist.local_rank)
        self.device = device
        if self.dist.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        self.model = build_model(cfg).to(device)
        self.model.activation_checkpointing = cfg.activation_checkpointing
        if cfg.compile:
            for block in self.model.blocks:
                block.compile()
        self.net: WorldModel | DistributedDataParallel = self.model
        if self.dist.world_size > 1:
            self.net = DistributedDataParallel(self.model, device_ids=[device.index] if device.type == "cuda" else None)
        self.ema = PowerEMA(self.model, cfg.optim.ema_gamma)
        self.optimizer = torch.optim.AdamW(
            _param_groups(self.model, cfg.optim.weight_decay),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            fused=device.type == "cuda",
        )
        self.rng = torch.Generator(device=device)
        self.step = 0
        if self.is_main:
            self._write_run_config()

    @property
    def is_main(self) -> bool:
        return self.dist.rank == 0

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def lr_at(self, step: int) -> float:
        """Linear warm-up over ``warmup_steps``, then constant (MODELING_PLAN section 3)."""
        return self.cfg.optim.lr * min(1.0, (step + 1) / max(1, self.cfg.optim.warmup_steps))

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """One optimiser step on one batch; returns the detached loss components plus ``lr`` and ``grad_norm``."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, Tensor)}
        lr = self.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.rng.manual_seed(_mix(self.cfg.seed, self.dist.rank, self.step))
        self.model.train()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.precision == "bf16"):
            loss, logs = diffusion_forcing_loss(self.net, batch, schedule=self.cfg.schedule, rng=self.rng)
        self._stop_unless_finite(loss)
        self.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.ema.update(self.model, self.step)
        return {**logs, "lr": torch.tensor(lr), "grad_norm": grad_norm.detach()}

    def fit(self) -> None:
        """Train from the current step to ``cfg.steps``, logging and checkpointing on the configured cadence."""
        if self.step >= self.cfg.steps:
            return
        sampler = WindowBatchSampler(
            len(cast(Sized, self.dataset)),
            self.cfg.batch_size,
            seed=self.cfg.seed,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            start_step=self.step,
        )
        workers = self.cfg.data.num_workers if self.cfg.data is not None else 0
        loader = DataLoader(
            self.dataset, batch_sampler=sampler, num_workers=workers, pin_memory=self.device.type == "cuda"
        )
        tick = time.perf_counter()
        for batch in loader:
            logs = self.train_step(batch)
            if self.step % self.cfg.log_every == 0 or self.step == self.cfg.steps:
                now = time.perf_counter()
                self._log(logs, seconds=(now - tick) / self.cfg.log_every)
                tick = now
            if self.step % self.cfg.checkpoint_every == 0 or self.step == self.cfg.steps:
                self.save_checkpoint()
            if self.step >= self.cfg.steps:
                break

    def _stop_unless_finite(self, loss: Tensor) -> None:
        """Raise ``FloatingPointError`` on every rank, before any backward or update, if the loss is non-finite on any
        rank. A squared error can overflow to inf while the gradients stay finite or cancel, so the gradient-norm check
        alone does not catch it; the decision is all-reduced so no rank is left waiting in backward."""
        bad = (~torch.isfinite(loss.detach())).float().reshape(1)
        if dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item() > 0:
            raise FloatingPointError(
                f"non-finite loss at step {self.step} (this rank: {loss.item()}); stopping before the update"
            )

    # -------------------------------------------------------------------------------------------------- checkpoints

    def save_checkpoint(self) -> Path | None:
        """Write ``checkpoints/step_<step>.pt`` atomically (rank 0) and keep the newest ``keep_checkpoints``."""
        path = self.checkpoint_dir / f"step_{self.step:08d}.pt"
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "format": CHECKPOINT_FORMAT,
                "run_config": self.cfg.to_json(),
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            tmp = path.with_suffix(".tmp")
            torch.save(state, tmp)
            tmp.replace(path)
            for old in sorted(self.checkpoint_dir.glob("step_*.pt"))[: -self.cfg.keep_checkpoints]:
                old.unlink()
        if dist.is_initialized():
            dist.barrier()
        return path if self.is_main else None

    def load_checkpoint(self, path: Path) -> None:
        state = _read_checkpoint(path, self.device)
        if RunConfig.from_json(state["run_config"]) != self.cfg:
            raise ValueError(f"{path} was written by a different run config")
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])

    def resume(self) -> bool:
        """Load the newest checkpoint in ``run_dir`` if there is one."""
        found = sorted(self.checkpoint_dir.glob("step_*.pt"))
        if not found:
            return False
        self.load_checkpoint(found[-1])
        return True

    # ------------------------------------------------------------------------------------------------------ helpers

    def _write_run_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "run_config.json"
        text = self.cfg.to_json()
        if path.exists() and RunConfig.from_json(path.read_text()) != self.cfg:
            raise ValueError(f"{path} holds a different run config; use a new run_dir")
        path.write_text(text + "\n")

    def _log(self, logs: Mapping[str, Tensor], *, seconds: float) -> None:
        if not self.is_main:
            return
        record = {"step": self.step, "step_s": round(seconds, 4), **{k: float(v) for k, v in logs.items()}}
        line = json.dumps(record)
        with (self.run_dir / "log.jsonl").open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def _read_checkpoint(path: Path, device: torch.device | str) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return state


def load_world_model(path: Path, *, ema: bool = True, device: torch.device | str = "cpu") -> WorldModel:
    """The model of a checkpoint, with its Power-EMA weights by default, in eval mode."""
    state = _read_checkpoint(path, device)
    model = WorldModel(RunConfig.from_json(state["run_config"]).model).to(device)
    if ema:
        model.load_state_dict(state["ema"], strict=True)
    else:
        model.load_state_dict(state["model"])
    return model.eval()
