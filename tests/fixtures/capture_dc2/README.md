# capture_dc2 fixtures (DD-5-min)

**SYNTHETIC.** No T0b session exists yet (DD-4-min has not run), so the golden fixture here is written by
`make_synthetic.py`. The real golden fixture is cut from the first clean T0b session and **replaces this one** (ticket
DD-5-min component 3); the negative set and the tests stay.

| Path | What |
|---|---|
| `golden/arland_T0b_20261025T091512Z_wb_i00_5e00/` | A 13 s dc-2.0 session as DD-4-min's writer will write it (every stream of `docs/capture_contract/dc-2.0.streams.json`, chunks committed by `commit.csv`, `session.json`, `session_end.json`, `heartbeat.csv`, 130 PNGs with fiducial v2 bands), plus the harvester sidecars `harvest.json` (build_key of a dd-launch/1 session), `_engine_logs/console.log` (the sliced engine log) and `_launch/launch_<launch_id>.json` (the session's dd-launch/1 preflight record, which C02 joins). |
| `golden.json` | The expected reader output: counts, clocks, `img_resolved`, every decision sample, segments and a pixel subsample. Computed by `make_synthetic.py` from its own simulation, not by the reader: `action_cmd`, `hl_seq` and the held-command rule come from the simulation's decision events (the command decided after P_t and where it was applied), not from the ctl rows at P_t + 1. The masks and the obs rule repeat contract section 6, so C22-min checks the parse, the joins and the R-01 offsets; the real T0b golden adds what the synthetic one lacks (decode failures, skipped slots, missing hf frames, paused rows). |
| `routes/golden_s_curve_v1.json` | The golden episodes' route (drivedit_nav route schema). |
| `cards/S105_synthetic.json` | A synthetic vehicle card with the TBD-7 gear fields only (`dir_state`). |
| `negative/{gap,missing_commit,reused_slot}/mutation.json` | The negative set. Each is generated in the test with `make_synthetic.py --variant <name>` (the repo keeps only the spec, about 1 MB less per fixture) and must fail exactly the check it names. |

Regenerate (writes only under /mnt/e or /dev/shm):

```bash
python3 -I -B tests/fixtures/capture_dc2/make_synthetic.py --out tests/fixtures/capture_dc2/golden \
    --golden-json tests/fixtures/capture_dc2/golden.json --routes-dir tests/fixtures/capture_dc2/routes
```

`tests/test_dc2_reader.py` checks that the committed files equal a fresh generation, so a generator change must come
with the regenerated fixture. `--kind golden_5hz` writes the same session at the TBD-2 fallback (image_hz =
decision_hz = 5, N = 12); the tests generate it, it is not committed. `--kind client_1408` (DD-6-min) writes the
golden drive as a packaged client captures it: exe `cl`, instance `i01`, source class `client_1408x720`, 1408x720
images (a 1280x720 core between two 64 px bands) and a dd-launch/1 record of exe `client`; `tests/test_harvest.py`
generates it, it is not committed.

The synthetic session uses the TBD defaults of contract dc-2.0: 60 Hz physics, k = 3, N = 6, image_hz 10,
img_lead_ms 0, `frames/f_<slot:08d>.png`, steer_sign -1, yaw_axis_sign +1. Its images are 256x240 `wb_viewport`
captures (64 px bands, 128x240 core), far smaller than the client's 1408x720. The fiducial bands are real fiducial
v2 patches (`data/dc2/fiducial.py` layout); the render frame crosses 2^18 during the session, so the decoder's unwrap
is exercised.

Legacy self-test: `tests/test_dc2_validate.py` runs `dc_validate --legacy` on sessions 4811, 2702 and 2929 under
`/mnt/e/DriveDiT/captures/_legacy_DrivingData_C/` when that copy exists. It does not exist yet (TICKETS_UNIFIED
decision 13 is open), so that test skips. The synthetic dc-1.0 session the other legacy tests use takes its columns
from the writer source (`SCR_MLDataCollector.c` `WriteTelemetryHeader`), never from an invented schema (CR-109).
