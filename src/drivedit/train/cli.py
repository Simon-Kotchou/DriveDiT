"""``dd-train``: train the world model from a RunConfig JSON.

    dd-train --config run.json --run-dir runs/<run_id> [--resume]
    torchrun --nproc-per-node 2 -m drivedit.train.cli --config run.json --run-dir runs/<run_id> --resume

``--print-config PRESET`` writes a default RunConfig for a model preset to stdout, as a starting point.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from drivedit.model.config import PRESETS
from drivedit.train.config import RunConfig
from drivedit.train.data import open_training_windows
from drivedit.train.trainer import Trainer


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dd-train", description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--config", type=Path, help="RunConfig JSON")
    p.add_argument("--run-dir", type=Path, help="where checkpoints, run_config.json and log.jsonl go")
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --run-dir")
    p.add_argument("--print-config", choices=sorted(PRESETS), help="print a default RunConfig for a preset and exit")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_config:
        print(RunConfig(run_id=args.print_config, model=PRESETS[args.print_config]()).to_json())
        return 0
    if args.config is None or args.run_dir is None:
        _parser().error("--config and --run-dir are required")
    cfg = RunConfig.from_json(args.config.read_text())
    trainer = Trainer(cfg, open_training_windows(cfg), args.run_dir)
    if args.resume and trainer.resume():
        print(f"resumed at step {trainer.step}", file=sys.stderr)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
