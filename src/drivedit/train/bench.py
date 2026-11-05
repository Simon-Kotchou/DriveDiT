"""Step time and peak memory of training steps of a model preset, on random latents: a speed measurement only.

    python -m drivedit.train.bench --preset probe_50m --latent 32 9 16 --window 10 --batch 8 --steps 30

Prints one JSON record: parameter count, shapes, precision, the median and p90 milliseconds per optimiser step after
warm-up, and the peak CUDA memory allocated and reserved. The batch is drawn once and reused, so data loading is
excluded. This is the reproducible measurement behind any step-time number quoted for the world model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from drivedit.model.config import PRESETS, WorldModelConfig
from drivedit.train.config import RunConfig
from drivedit.train.trainer import Trainer


class RandomWindows(Dataset[dict[str, Tensor]]):
    """Windows of N(0, 1) latents with in-range ``action_cmd`` and ``state`` (timing input, not data)."""

    def __init__(self, cfg: WorldModelConfig, size: int, seed: int) -> None:
        self.cfg, self.size, self.seed = cfg, size, seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        frames = self.cfg.window
        action_cmd = torch.randn(frames, 3, generator=g) * torch.tensor([0.02, 1.0, 0.0])
        action_cmd[:, 2] = 1.0
        state = torch.randn(frames, 4, generator=g) * torch.tensor([2.0, 0.2, 1.0, 0.0]) + torch.tensor([11, 0, 0, 1.0])
        masks = torch.zeros(frames, 5, dtype=torch.uint8)
        masks[:, :2] = 1  # policy_valid, wm_valid
        return {
            "latents": torch.randn(frames, *self.cfg.latent_shape, generator=g),
            "valid": torch.ones(frames, dtype=torch.bool),
            "action_cmd": action_cmd,
            "state": state,
            "masks": masks,
        }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m drivedit.train.bench", description=__doc__.split("\n\n")[0])
    p.add_argument("--preset", choices=sorted(PRESETS), default="probe_50m")
    p.add_argument("--latent", type=int, nargs=3, metavar=("C", "H", "W"), default=(32, 9, 16))
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    model = replace(PRESETS[args.preset]().with_latent(*args.latent), window=args.window)
    cfg = RunConfig(
        run_id=f"bench-{args.preset}",
        model=model,
        batch_size=args.batch,
        precision=args.precision,
        activation_checkpointing=args.activation_checkpointing,
        compile=args.compile,
    )
    device = torch.device(args.device)
    data = RandomWindows(model, size=args.batch, seed=0)
    batch = {k: torch.stack([data[i][k] for i in range(args.batch)]) for k in data[0]}
    with tempfile.TemporaryDirectory() as run_dir:
        trainer = Trainer(cfg, data, Path(run_dir), device=device)
        times = []
        for i in range(args.warmup + args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                if i == args.warmup:
                    torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            logs = trainer.train_step(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if i >= args.warmup:
                times.append(1000 * (time.perf_counter() - tick))
    cuda = device.type == "cuda"
    record = {
        "preset": args.preset,
        "params_m": round(sum(q.numel() for q in trainer.model.parameters()) / 1e6, 2),
        "latent": list(args.latent),
        "tokens_per_frame": model.tokens_per_frame,
        "window": args.window,
        "batch": args.batch,
        "precision": args.precision,
        "activation_checkpointing": args.activation_checkpointing,
        "compile": args.compile,
        "device": torch.cuda.get_device_name(device) if cuda else str(device),
        "torch": torch.__version__,
        "ms_per_step_median": round(statistics.median(times), 1),
        "ms_per_step_p90": round(sorted(times)[math.ceil(0.9 * len(times)) - 1], 1),  # nearest rank
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20) if cuda else None,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20) if cuda else None,
        "final_loss": round(float(logs["loss"]), 4),
    }
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
