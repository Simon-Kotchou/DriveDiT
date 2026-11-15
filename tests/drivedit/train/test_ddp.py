"""DDP as torchrun runs it, on CPU (gloo, 2 processes): replicas stay identical, a rank-dependent share of shortcut
bootstrap samples keeps the collectives in step, and a non-finite loss on one rank stops every rank before the update.

A desynchronised collective shows up as a worker error or a hang; the ranks are joined against a deadline, so a hang
fails the test instead of blocking the suite.
"""

from __future__ import annotations

import errno
import math
import os
import socket
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch import distributed as dist
from torch import multiprocessing as mp

from drivedit import local_paths
from drivedit.model import AUX_HEADS
from drivedit.train import ForcingSchedule, OptimConfig, RunConfig, Trainer
from tests.drivedit.boundary import require_real_boundary
from tests.drivedit.model.toy import ShiftDynamics, tiny_config

CPU = torch.device("cpu")
DEADLINE_S = 180.0


@pytest.fixture
def tmp_path(tmp_path: Path) -> Path:
    """RAM scratch the real write boundary allows (every rank's trainer writes through it, DD-23), or the test skips."""
    return require_real_boundary(tmp_path)


def _cfg(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_id": "ddp",
        "model": tiny_config(aux_heads=AUX_HEADS),
        "optim": OptimConfig(lr=1e-3, warmup_steps=2),
        "batch_size": 4,
        "steps": 3,
        "precision": "fp32",
        "log_every": 3,
        "checkpoint_every": 3,
    }
    return RunConfig(**{**base, **overrides})


def _join_group(rank: int, port: int) -> None:
    os.environ.update(
        {"RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank), "MASTER_ADDR": "127.0.0.1"}
        | {"MASTER_PORT": str(port)}
    )
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60))


def _save_params(trainer: Trainer, run_dir: str, rank: int) -> None:
    flat = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
    torch.save(flat, Path(run_dir) / f"params_rank{rank}.pt")


def _replicas(rank: int, port: int, run_dir: str) -> None:
    _join_group(rank, port)
    cfg = _cfg()
    trainer = Trainer(cfg, ShiftDynamics(cfg.model, 32, seed=3), Path(run_dir), device=CPU)
    trainer.fit()
    _save_params(trainer, run_dir, rank)
    dist.destroy_process_group()


def _rank_dependent_bootstrap(rank: int, port: int, run_dir: str) -> None:
    """Rank 0 draws no bootstrap sample (fraction 0), rank 1 bootstraps every sample (fraction 1)."""
    _join_group(rank, port)
    cfg = _cfg(
        model=tiny_config(aux_heads=AUX_HEADS, shortcut_levels=7),
        schedule=ForcingSchedule(shortcut=True, shortcut_fraction=float(rank)),
    )
    trainer = Trainer(cfg, ShiftDynamics(cfg.model, 32, seed=3), Path(run_dir), device=CPU)
    trainer.fit()
    _save_params(trainer, run_dir, rank)
    dist.destroy_process_group()


class _HugeWindows(ShiftDynamics):
    """Finite latents of alternating sign at 1e20: the squared error overflows to inf (the Codex DD-20 reproduction)."""

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        item = super().__getitem__(i)
        sign = torch.ones_like(item["latents"]).flatten()
        sign[1::2] = -1.0
        item["latents"] = 1e20 * sign.view_as(item["latents"])
        return item


def _non_finite_on_rank_1(rank: int, port: int, run_dir: str) -> None:
    _join_group(rank, port)
    cfg = _cfg()
    data = (_HugeWindows if rank == 1 else ShiftDynamics)(cfg.model, 32, seed=3)
    trainer = Trainer(cfg, data, Path(run_dir), device=CPU)
    try:
        trainer.fit()
        outcome = f"finished {trainer.step}"
    except FloatingPointError as err:
        outcome = f"stopped {trainer.step}: {err}"
    (Path(run_dir) / f"outcome_rank{rank}.txt").write_text(outcome)
    dist.destroy_process_group()


def _checkpoint_refused_on_rank_0(rank: int, port: int, run_dir: str) -> None:
    """Rank 0's write boundary refuses the checkpoint (as for a link planted after the start); rank 1's would not."""
    _join_group(rank, port)
    trainer = Trainer(_cfg(), ShiftDynamics(_cfg().model, 32, seed=3), Path(run_dir), device=CPU)
    if rank == 0:
        allowed = local_paths.check_write

        def refuse_checkpoints(path: str | Path, roots: object = None) -> str:
            if "checkpoints" in str(path):
                raise local_paths.WriteRefusedError(f"refusing to write {path}: test refusal")
            return allowed(path)

        local_paths.check_write = refuse_checkpoints  # type: ignore[assignment]
    try:
        trainer.fit()
        outcome = f"finished {trainer.step}"
    except local_paths.WriteRefusedError as err:
        outcome = f"refused {trainer.step}: {err}"
    (Path(run_dir) / f"outcome_rank{rank}.txt").write_text(outcome)
    dist.destroy_process_group()


def _checkpoint_disk_full_on_rank_0(rank: int, port: int, run_dir: str) -> None:
    """Rank 0's torch.save fails (a full disk) after the boundary allowed the write."""
    _join_group(rank, port)
    trainer = Trainer(_cfg(), ShiftDynamics(_cfg().model, 32, seed=3), Path(run_dir), device=CPU)
    if rank == 0:

        def full(*_: object, **__: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        torch.save = full  # type: ignore[assignment]
    try:
        trainer.fit()
        outcome = f"finished {trainer.step}"
    except (OSError, RuntimeError) as err:
        outcome = f"failed {trainer.step}: {type(err).__name__}: {err}"
    (Path(run_dir) / f"outcome_rank{rank}.txt").write_text(outcome)
    dist.destroy_process_group()


def _run_two_ranks(worker: Callable[[int, int, str], None], run_dir: Path) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.start_processes(worker, args=(port, str(run_dir)), nprocs=2, join=False, start_method="spawn")
    deadline = time.monotonic() + DEADLINE_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
            pytest.fail(f"the DDP ranks did not finish within {DEADLINE_S} s (desynchronised collectives)")


def _assert_identical_replicas(run_dir: Path) -> None:
    rank0, rank1 = (torch.load(run_dir / f"params_rank{r}.pt") for r in (0, 1))
    assert torch.equal(rank0, rank1)
    assert (run_dir / "checkpoints" / "step_00000003.pt").exists()


def test_ddp_replicas_stay_identical(tmp_path: Path) -> None:
    _run_two_ranks(_replicas, tmp_path)
    _assert_identical_replicas(tmp_path)


def test_rank_dependent_shortcut_bootstrap_keeps_ddp_in_step(tmp_path: Path) -> None:
    _run_two_ranks(_rank_dependent_bootstrap, tmp_path)
    _assert_identical_replicas(tmp_path)


def test_a_non_finite_loss_on_one_rank_stops_every_rank(tmp_path: Path) -> None:
    _run_two_ranks(_non_finite_on_rank_1, tmp_path)
    outcomes = [(tmp_path / f"outcome_rank{r}.txt").read_text() for r in (0, 1)]
    assert all(o.startswith("stopped 0: non-finite loss at step 0") for o in outcomes), outcomes
    own_loss = [float(o.split("this rank: ")[1].split(")")[0]) for o in outcomes]
    assert math.isfinite(own_loss[0])  # rank 0 stopped because rank 1's loss was not finite
    assert not math.isfinite(own_loss[1])


def test_a_refused_checkpoint_on_rank_0_stops_every_rank(tmp_path: Path) -> None:
    """DD-23: rank 0's verdict is all-reduced, so rank 1 raises too instead of waiting at the checkpoint barrier."""
    _run_two_ranks(_checkpoint_refused_on_rank_0, tmp_path)
    outcomes = [(tmp_path / f"outcome_rank{r}.txt").read_text() for r in (0, 1)]
    assert outcomes[0].startswith("refused 3: refusing to write"), outcomes
    assert "test refusal" in outcomes[0], outcomes
    assert outcomes[1].startswith("refused 3: rank 0 refused to write"), outcomes
    assert not (tmp_path / "checkpoints").exists()


def test_a_failed_checkpoint_write_on_rank_0_stops_every_rank(tmp_path: Path) -> None:
    """Not only a boundary refusal: any failure of a rank-0 write is all-reduced too, so rank 1 does not wait."""
    _run_two_ranks(_checkpoint_disk_full_on_rank_0, tmp_path)
    outcomes = [(tmp_path / f"outcome_rank{r}.txt").read_text() for r in (0, 1)]
    assert outcomes[0].startswith("failed 3: OSError: [Errno 28]"), outcomes
    assert outcomes[1].startswith("failed 3: RuntimeError: rank 0 failed to write"), outcomes
