# DriveDiT

A small-scale attempt at the comma.ai recipe inside Arma Reforger:

1. Learn an action-conditioned world model from captured driving.
2. Train a driving policy inside that world model.
3. Test the policy in the real engine.

Arland provides the training data. Everon is the sealed zero-shot test map.

## Status

- **Capture works.** The repository holds the capture contract, the Workbench capture addon and its tools. The
  contract is [dc-2.0](docs/capture_contract/dc-2.0.md), frozen; features_v1 is its model-boundary registry.
- **The model is being rebuilt.** The plan is a small latent DiT on a frozen pretrained tokenizer, reading
  features_v1. There is no model, trainer or evaluator in the tree yet.
- **Earlier code is archived in tags.** The March 2026 model stack never trained a world model and is kept in tag
  `archive/model-2026-03`. The dc-1.0 capture path and the Rust workspace are kept in tag `archive/dc1-rust`.
  Restore a path with `git checkout <tag> -- <path>`.

## What is here

| Path | Contents |
|---|---|
| `docs/capture_contract/` | dc-2.0 contract, its streams JSON and the features_v1 registry |
| `src/enfusion/datacapture/`, `addons/DataCapture/` | Enforce capture scripts (session writer, expert driver, probes) and the Workbench addon ([README](addons/DataCapture/README.md)) |
| `tools/` | deploy, validator, harvester and retention, build pins and launchers, probe analyses, route builder, secret scan |
| `drivedit_nav/` | nav_v1 and the route schema |
| `data/dc2/` | dc-2.0 session reader and fiducial decoder |
| `tests/` | contract, reader, validator and tooling tests, with a synthetic golden session |

## Quick start

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync                                    # capture tooling: numpy + OpenCV, no torch
uv run pytest -m "not torch"               # capture tests
uv run ruff check .
python3 -I -B tools/ci/secret_scan.py      # privacy scan of tracked files

uv sync --extra model --extra cpu          # the model extra with CPU torch
uv run --extra model --extra cpu pytest -m torch
```

[CLAUDE.md](CLAUDE.md) holds the repository rules: the frozen contract, the sealed test map, the model-input rules
and the Enforce Script pitfalls.

## License

Apache-2.0, see [LICENSE](LICENSE).
