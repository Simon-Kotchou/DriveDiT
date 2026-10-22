# T0-ENG core results (DD-4-probe)

Results of the T0-ENG core probe set (capture program r2 4.4). DD-11-min cites this table when it freezes dc-2.0;
DD-4-min and DD-7-min take their parameters from it. Every result is per build: a re-run on 1.9 gets its own rows
and never overwrites the 1.8 ones.

How the probes run and what each analysis prints: `addons/DataCapture/README.md`, section "Probe world
DC_Probes_Arland (DD-4-probe)". Each analysis ends with `RESULT <probe>: PASS|FAIL|INCOMPLETE`; copy that word and
the key numbers here. Never paste the E-11 console line or a `$profile` path: with Workbench started from Steam they
hold the Windows user name.

## Run identity

| Field | Value |
|---|---|
| Engine build (`engine.build_version` in the run header) | PENDING |
| Pin id (`launch.pin_id`) | PENDING |
| Build key (`launch.build_key`, empty on dd-launch/1) | PENDING |
| Branch and commit (`addon.git_sha`, `addon.git_dirty`) | PENDING |
| Launch mode (`launch.mode`: cli or wb_escape) | PENDING |
| Fixed time slice (`engine.fixed_time_slice_s`) | PENDING |
| Plays (date, plan, launch id) | PENDING |

## Results

| Probe | r2 4.4 criterion | Analysis | Play dir | Result | Numbers |
|---|---|---|---|---|---|
| E-01 | The car moves; `input_pre` vs `input_post` vs command recorded; setter-to-response step delay identified; `steer_sign` measured; auto-gear works or a scripted gearbox is needed; whether it works without `ForceEnableSimulation` | `tools/probes/t0eng/e01_actuation.py e01 <play_dir>` | PENDING | PENDING | delay steps, steer_sign, gearbox, ForceEnableSimulation: PENDING |
| E-02a | First-pass steer to kappa fit usable by `llc_v1` | `tools/probes/t0eng/e01_actuation.py e02a <play_dir>` | PENDING | PENDING | per speed a, b, R2: PENDING |
| E-04 | S105 component classes and `HasAutomaticGearbox()` read from the prefab | `tools/probes/t0eng/e04_components.py <play_dir>` | PENDING | PENDING | PENDING |
| E-05 | PNG dims logged; missing <= 1%; `Screenshot saved` count = files; FPS p5 >= 30; request-to-file latency distribution; raw MB/frame; engine-written file name | `tools/probes/t0eng/e05_images.py <play_dir> --log <console.log>` | PENDING | PENDING | image_hz decision, latency p50/p95, suffix: PENDING |
| E-06 | Fiducial v2 decode >= 99.5% over >= 600 frames; offset p50/p99 | `tools/probes/t0eng/e06_fiducial.py <play_dir>` | PENDING | PENDING | PENDING |
| E-06A | RMS <= 2 px at k = 0, unique minimum with margin, at 60 and 30 FPS with backlog | `tools/validator/e06a_reproject.py <play_dir>` | PENDING | PENDING | RMS(k) at 60 and 30 FPS: PENDING |
| E-08a | Count, km and width histogram vs the decoder (~46.4 km); 0 duplicates; footer; no cap; unparented and decal-like widths flagged | `tools/roads/compare_dump_decoder.py <play_dir> --decoder <Arland_road_polylines.tsv>` | PENDING | PENDING | PENDING |
| E-10 | `GetFixedTimeSlice` logged; `EOnPostSimulate` count per world-second = 1/dt +-1; telemetry every k steps, 0 missed | `tools/probes/t0eng/e10_ticks.py <play_dir>` | PENDING | PENDING | dt, k, step source: PENDING |
| E-11 | `$profile` under `E:\DriveDiT\wb_profile`; nothing new under OneDrive; preflight record joined (whether Workbench Play sees its command line) | console line `[DC_Probe] E-11: ...` and the run header `launch.*` keys (paste neither) | PENDING | PENDING | `-profile` visible, `-dcLaunchId` visible, launch mode: PENDING |
| E-19 | Short or failed writes detected; the commit log stays consistent after a kill | `tools/probes/t0eng/e19_write.py <play_dir> [--kill <e19kill play_dir>]` | PENDING | PENDING | PENDING |
| E-20 | `VideoUserSettings` and `DisplayUserSettings` dumped; Workbench and client differences listed | `tools/probes/t0eng/e20_settings.py <play_dir> [--compare <client play_dir>]` | PENDING | PENDING | PENDING |

E-15 (unfocused Workbench) stays with DD-0; the bridge set (E-03, E-07, E-12..E-14, E-16..E-18) with DD-8.

## Values DD-11-min takes from these results

TBD-1 to TBD-5 block the dc-2.0 freeze; TBD-6 to TBD-9 may be settled after it (dc-2.0 draft, section TBD).

| dc-2.0 item | Source probe | Value |
|---|---|---|
| TBD-1 telemetry every k physics steps | E-10 | PENDING |
| TBD-2 image_hz (10, or 5 as the fallback) | E-05 | PENDING |
| TBD-3 writer semantics (`FileHandle.Write` return) | E-19 | PENDING |
| TBD-4 engine-written screenshot file name | E-05 | PENDING |
| TBD-5 steer_sign normalisation | E-01 | PENDING |
| TBD-6 launch id visible in Workbench Play | E-11 | PENDING |
| TBD-7 gearbox and gear index semantics | E-04, E-01 | PENDING |
| TBD-8 render-settings dump keys | E-20 | PENDING |
| TBD-9 road key of the in-engine road dump | E-08a | PENDING |
