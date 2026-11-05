# capture_dc2 fixtures (DD-5-min)

**SYNTHETIC.** No T0b session exists yet (DD-4-min has not run), so the golden fixture here is written by
`make_synthetic.py`. The real golden fixture is cut from the first clean T0b session and **replaces this one** (ticket
DD-5-min component 3); the negative set and the tests stay.

| Path | What |
|---|---|
| `golden/arland_T0b_20261025T091512Z_wb_i00_5e00/` | A 13 s dc-2.0 session in the shape DD-4-min's writer writes it at the freeze (every stream of `docs/capture_contract/dc-2.0.streams.json`, frame-bound ctl and telemetry rows of 1 to 3 steps at 30 Hz, chunks committed by `commit.csv`, `session.json`, `session_end.json`, `heartbeat.csv`, 68 PNGs at 5 Hz with fiducial v2 bands), plus the harvester sidecars `harvest.json` (build_key of a dd-launch/1 session), `_engine_logs/console.log` (the sliced engine log) and `_launch/launch_<launch_id>.json` (the session's dd-launch/1 preflight record, which C02 joins). |
| `golden.json` | The expected reader output: counts, clocks, `img_resolved`, every decision sample, segments and a pixel subsample. Computed by `make_synthetic.py` from its own simulation, not by the reader: `action_cmd`, `hl_seq`, the held-command rule and the per-step `action_exec` come from the simulation's decision events (the command decided at the decision callback and where it was applied) and its step-to-callback map, not from the ctl rows. The masks and the obs rule repeat contract section 6, so C22-min checks the parse, the joins and the R-01 offsets; the real T0b golden adds what the synthetic one lacks (decode failures, skipped slots, missing hf frames, paused rows). |
| `routes/golden_s_curve_v1.json` | The golden episodes' route (drivedit_nav route schema). |
| `cards/S105_synthetic.json` | A synthetic vehicle card with the TBD-7 gear fields only (`dir_state`): the S105 map decided at the freeze, 0 reverse, 1 neutral, 2..5 forward. |
| `negative/{gap,missing_commit,reused_slot,crash_unexplained,destroyed_unexplained}/mutation.json` | The negative set. Each is generated in the test with `make_synthetic.py --variant <name>` (the repo keeps only the spec, about 1 MB less per fixture) and must fail exactly the check it names. |

Regenerate (writes only under /mnt/e or /dev/shm):

```bash
python3 -I -B tests/fixtures/capture_dc2/make_synthetic.py --out tests/fixtures/capture_dc2/golden \
    --golden-json tests/fixtures/capture_dc2/golden.json --routes-dir tests/fixtures/capture_dc2/routes
```

`tests/test_dc2_reader.py` checks that the committed files equal a fresh generation, so a generator change must come
with the regenerated fixture. `--kind golden_10hz` writes the same session at the non-default 10 Hz setting (TBD-2:
image_hz = decision_hz = 10, N = 3, `clocks.image_hz_setting` `trial_conf`); the tests generate it, it is not
committed. `--kind client_1408` (DD-6-min) writes the golden drive as a packaged client captures it: exe `cl`,
instance `i01`, source class `client_1408x720`, 1408x720 images (a 1280x720 core between two 64 px bands) and a
dd-launch/1 record of exe `client`; `tests/test_harvest.py` generates it, it is not committed. `--kind t0_loop_noise`
(DD-7-ident pilot) writes the t0_loop lap as a mode-N episode of trial T0P (trial conf `"perturb": "pilot_v1"`) with
one disturbance of each family DC_Noise draws (k_pulse, a_pulse, k_step, stop, speed), with DC_Noise's amplitude
semantics (k_* a fraction of kappa_max(v), pulses half-sines sampled per decision interval): ctl `hl_*` the applied
command, `exp_*_now` the clean expert, `noise_active` / `noise_id` on every row that holds a disturbed command, and
`perturb_start` / `perturb_end` events with payload `noise_id;kind;amplitude;duration_s`; `tests/test_dc2_validate.py`
generates it (every check passes, C17 N/A: it judges mode-E episodes only), it is not committed.

The clock model is the dc-2.0 freeze's (decision D-A, probe runs 3-4): render frames of about 39 FPS whose world time
is the frame clock, a capture stall of 80 ms in the frame a screenshot is taken in, engine fixed steps of 1/30 s run
after each frame's EOnFrame (at most 3 per frame), one callback per frame with a step and one ctl and one telemetry row
per callback (`physics_step` the last step it covers, `covered_steps` 1 to 3, `render_frame`, `world_time_ms`).
Decisions and images lie on the 200 ms world-time grid anchored at the first ctl row; the image of render frame C
(captured = `done_render_frame` - 1) shows the state of hf row C + 1. The vehicle is integrated over every frame's
world time, so the logged pose follows the frame clock.

Crash events (contract 9.5.1): `session.json` carries the `events.*` detector parameters (the TBD-10 defaults) and
telemetry the optional `dv_max_mps`, computed by the reference detector `data/dc2/crash_dv.py` (100 ms of world time,
interpolated, gap rule; watched at the session open and repositioned before the teleport, as DC_CrashWatch does) on
the simulated velocities. The crash fixtures are generated in the tests, not committed,
in the shape DD-4-min writes them:
- `--variant crash`: the golden session in which the ego stops dead against a rock 4 s into episode 1 and is
  destroyed two callbacks after the detector's onset. At that terminal step L: the `collision` row (`onset_step` = the impact step, `dv_mps`
  and `v_mps` from the reference detector, `src=dv_contact`, `other=rock`), `veh_damage` `destroyed` and
  `veh_fire_start`; then the seal ends the open episode at L with termination `collision`, and the session ends with
  `ego_destroyed`, status ok. Nothing is written after L.
- `--variant crash_unexplained` drops the crash rows (termination `collision` stays, end `stop`) and
  `--variant destroyed_unexplained` the destroyed row; both must fail C36 only.

`e01_s105_runs.csv` is the E-01 S105 per-step probe of 2026-11-03 (launch `20261103T061122Z-wb-86bd`, play 01):
the full-brake, steering and full-throttle phases without their resets, as physics step and horizontal world-frame
velocity (from `vel_l_*` and `fwd_*`). The detector tests merge its steps 1 to 3 per callback: full braking gives
2.01 m/s per 100 ms, below the 4.0 m/s threshold, at any merging.

The synthetic session uses the values decided at the dc-2.0 freeze: frame-bound rows with 1/30 s steps, N = 6,
image_hz 5, img_lead_ms 0, `frames/f_<slot:08d>.png`, steer_sign +1, yaw_axis_sign +1, the Shape fiducial carrier.
Its images are 256x240 `wb_viewport` captures (64 px bands, 128x240 core), far smaller than the client's 1408x720.
The fiducial bands are real fiducial v2 patches (`data/dc2/fiducial.py` layout); the render frame crosses 2^18 during
the session, so the decoder's unwrap is exercised.

Legacy self-test: `tests/test_dc2_validate.py` runs `dc_validate --legacy` on sessions 4811, 2702 and 2929 under
`/mnt/e/DriveDiT/captures/_legacy_DrivingData_C/` when that copy exists. It does not exist yet (TICKETS_UNIFIED
decision 13 is open), so that test skips. The synthetic dc-1.0 session the other legacy tests use takes its columns
from the writer source, never from an invented schema (CR-109): `make_synthetic.LEGACY_HEADER` is copied verbatim
from `WriteTelemetryHeader` of the dc-1.0 writer `SCR_MLDataCollector.c`. DD-12 deleted that writer from main; it is
in tag `archive/dc1-rust`.
