"""``dd-train``: train the world model from a RunConfig JSON.

    dd-train --config run.json --run-dir runs/<run_id> [--resume]
    torchrun --nproc-per-node 2 -m drivedit.train.cli --config run.json --run-dir runs/<run_id> --resume

``--print-config PRESET`` writes a default RunConfig for a model preset to stdout, as a starting point.

Before anything else a training run routes the process's caches and temporary files into the write boundary
(``drivedit.process_paths.prepare_process``: torch.compile, Triton and CUDA caches under ``<DATA_ROOT>/cache``,
temporary files in RAM) and checks ``--run-dir`` (``drivedit.train.trainer.check_run_dir``: RAM or the verified data
drive of the local config, DD-23); a refusal exits with one line naming the reason. torchrun itself runs before this
module: start it with ``TMPDIR`` (or ``--log-dir``) in RAM or on the data drive, or its per-run log directory goes to
the system temp dir and every rank refuses the ``TORCHELASTIC_ERROR_FILE`` it was given there.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from drivedit.model.config import PRESETS
from drivedit.process_paths import command_line, prepare_process
from drivedit.train.config import RunConfig
from drivedit.train.data import open_training_windows
from drivedit.train.trainer import Trainer, check_run_dir


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
    with command_line("dd-train"):
        prepare_process()
        check_run_dir(args.run_dir)  # before the data is opened
        cfg = RunConfig.from_json(args.config.read_text())
        trainer = Trainer(cfg, open_training_windows(cfg), args.run_dir)
        if args.resume and trainer.resume():
            print(f"resumed at step {trainer.step}", file=sys.stderr)
        trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
