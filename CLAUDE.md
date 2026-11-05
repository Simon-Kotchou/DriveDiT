# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## Goal

Reproduce, at small scale inside Arma Reforger, the recipe comma.ai ships:

1. Learn an action-conditioned world model from captured driving.
2. Train a driving policy inside that world model.
3. Test the policy in the real engine.

Training data comes from the Arland map. Everon is the sealed zero-shot test map. The world model is judged by
controllability (does a prediction follow the commanded action?) and by whether a policy trained in it drives in the
engine, not by visual quality.

## Status

- **Capture side: works and is tested.** It has the capture contract, the Enforce writer and expert driver, deploy,
  the validator, the harvester and retention, nav_v1 and the dc-2.0 reader.
- **Model side: being rebuilt.** There is no world model, trainer or evaluator in the tree today. The plan is a small
  latent DiT on a frozen pretrained tokenizer, reading features_v1 (contract section 12).
- **The March 2026 model stack never trained a world model.** DD-16 removed it from main: `layers/`, `blocks/`,
  `core/`, `models/`, `training/`, `inference/`, `evaluation/`, `config/`, `utils/`, `scripts/`, the pre-dc-2.0
  `data/` pipeline and their tests. It is kept in tag `archive/model-2026-03`.
- **The dc-1.0 capture path and the Rust workspace** are kept in tag `archive/dc1-rust`.
- Restore an archived path with `git checkout <tag> -- <path>`. Treat archived code as reference only: none of it
  reads dc-2.0 data.

## Layout

| Path | What it is |
|---|---|
| `docs/capture_contract/` | Capture contract dc-2.0: `dc-2.0.md` (meaning) and its normative JSON twins `dc-2.0.streams.json` (streams, session files) and `features_v1.json` (the model-boundary registry) |
| `src/enfusion/datacapture/DC_*.c` | Enforce scripts of the capture addon: `DC_Session` and its stream, image, crash and ego modules; the expert driver (`DC_ExpertHL`, `DC_LLCv1`); the T0-ENG probe components (`DC_Probe*`, `DC_RoadDump`) |
| `addons/DataCapture/` | Workbench addon source: project file, capture and probe worlds, routes, `deploy_manifest.txt`. Its `README.md` documents the worlds, deploy and probes |
| `tools/deploy_addon.py` | Deploys exactly the files in the manifest into the addon target, and refuses anything else |
| `tools/validator/` | `dc_validate.py`: the contract checks (C01 ...) on one session; `e06a_reproject.py`: probe E-06A |
| `tools/harvester/` | `harvest.py` (validate, decode fiducials, build the derivative, move); `retention.py` (raw-screenshot retention, dry run unless `--apply`); `check_ledger_append_only.py` (the Everon ledger only grows) |
| `tools/pin/`, `tools/launch/` | Reforger build pins, preflight and launchers (`tools/pin/README.md`) |
| `tools/probes/t0eng/`, `tools/routes/`, `tools/ident/`, `tools/roads/` | T0-ENG probe analyses, the route builder, steering identification with vehicle cards, the road-dump comparison |
| `tools/enfusion-mcp/` | Wrapper scripts and patches for the Enfusion MCP server |
| `tools/ci/secret_scan.py` | Privacy scan of tracked files (stdlib only; a CI job) |
| `drivedit_nav/` | nav_v1 and the route schema (pure Python). The validator and the route builder use it. The planned dataset builder and closed-loop bridge must use the same code, so their nav values agree (C22) |
| `data/dc2/` | dc-2.0 reader (`reader.py`: rows, joins, decisions, masks, segments), fiducial decoder, reference crash detector. Stdlib and numpy; cv2 only for images |
| `tests/` | Contract, reader, validator, harvester, tooling and Enforce-source tests. `tests/fixtures/capture_dc2/` holds the synthetic golden session |
| `docs/trials/`, `docs/program/` | T0-ENG probe results; the Everon exposure ledger |

`docs/WORLD_MODEL_DATA_CAPTURE_ARCHITECTURE.md` is the historical dc-1.0 design, not the current capture path.

## Commands

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. The base install has no torch.

```bash
uv sync                                    # capture tooling: numpy + OpenCV, plus pytest and ruff
uv run pytest -m "not torch"               # capture tests (the CI capture job)
uv run ruff check .                        # lint (CI gate: correctness rules, see pyproject.toml)
python3 -I -B tools/ci/secret_scan.py      # privacy scan; run it before every push

uv sync --extra model --extra cpu          # adds the model extra with CPU torch (--extra cuda: PyPI torch)
uv run --extra model --extra cpu pytest -m torch   # torch tests (today only tests/test_torch_smoke.py)

python3 -I -B tools/deploy_addon.py --dry-run       # deploy plan: review it before any real deploy
uv run python -I -B tools/validator/dc_validate.py --session DIR
```

Each tool's module docstring documents its usage, and most tools also take `--help`. The tools load `data/dc2` and
their helpers by file path, so they need no package install. Tools that read sessions or images need numpy (and
OpenCV), so run them in the project environment (`uv run python -I -B tools/...`). The stdlib-only tools also run
with a plain `python3 -I -B`: deploy, the secret scan, the pin tools, the route builder, the steering fit and the
road-dump comparison.

CI (`.github/workflows/ci.yml`) has four jobs:

- ruff;
- the capture tests, with no torch installed;
- the torch tests, with CPU torch;
- the secret scan.

## Rules

- **The contract is frozen.** dc-2.0 has been frozen since 2026-11-04. That covers `docs/capture_contract/`, nav_v1
  and the route schema. Any change of a field, unit, sign, key, cadence or registry rule needs a new version dc-2.x,
  and any change of nav behaviour needs nav_v2. `tests/test_contract_dc2.py` checks the document against its JSON
  twins.
- **Everon is sealed.** No Everon data (frames, latents, statistics, model outputs) may go into any training,
  validation, normalisation or model-selection set (contract section 3.2). The reader and `drivedit_nav` refuse
  Everon, `heldout/` and `SEALED` paths unless they are called with `sealed_eval=True`. Every Everon exposure is
  recorded in `docs/program/everon_ledger.md` first.
- **Model inputs follow features_v1** (contract section 12):
  - The world model conditions on `action_cmd` at offset 0 plus state history, never on realised future state.
  - The policy sees actions only as history (offset <= -1).
  - The C24 inputs are forbidden: map id, absolute pose, future-derived values, hindsight anchors, split and session
    identity. The full list is `c24_forbidden_inputs` in `features_v1.json`.
  - Training augmentation never flips images.
- **The capture side stays torch-free.** `data/`, `drivedit_nav/` and `tools/` need only the stdlib, numpy and
  OpenCV, and the capture CI job runs without torch. A test module that imports torch must be added to
  `TORCH_TEST_MODULES` in `tests/conftest.py`.
- **Deploy only through `tools/deploy_addon.py`.** It copies exactly `addons/DataCapture/deploy_manifest.txt`, so a
  new Enforce file needs a manifest line. Run `--dry-run` first.
- **The repository is public.** Never commit any of these:
  - account or Steam ids;
  - e-mail addresses other than noreply ones;
  - host names;
  - absolute profile or home paths;
  - credentials.

  Run the secret scan before every push. Check commit messages, tag messages and PR text by hand as well: the scan
  reads only tracked files.

### Enforce Script pitfalls (`src/enfusion/datacapture/`)

- **No ternary operator.** `a ? b : c` is a compile error; use if/else.
- **Class names are global** across every loaded addon. Never define a class twice. Never load DataCapture together
  with an addon that defines the same `DC_` classes (`addons/DataCapture/README.md`).
- **Never create a `ScriptComponent` subclass with `new`.** Components are attached to entities in the world or
  prefab, and code finds them with `FindComponent`.
- After editing a world layer by hand, run `python3 -I -B tools/check_layer.py <file>`. Numeric vectors are never
  quoted.
