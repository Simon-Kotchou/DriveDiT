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
| Engine build (`engine.build_version` in the run header) | Workbench 1.8.0.13 (runs 1-4; 1.9 gets its own rows) |
| Pin id (`launch.pin_id`) | 1.8.0.13 |
| Build key (`launch.build_key`, empty on dd-launch/1) | empty (dd-launch/1) |
| Branch and commit (`addon.git_sha`, `addon.git_dirty`) | run 3: `699d8e2`, dirty 0; run 4: `f96dbf9` |
| Launch mode (`launch.mode`: cli or wb_escape) | cli |
| Fixed time slice (`engine.fixed_time_slice_s`) | 0.1 s in a 3-step frame: `GetFixedTimeSlice()` reports n/30 per frame and is never the step dt (the step is 1/30 s, E-10) |
| Plays (date, plan, launch id) | run 3: 2026-11-03, plan e10,e04,e20,e19,e08a,e01,e02a,e05,e06a, launch `20261103T061122Z-wb-86bd` (and its e19kill Play); run 4: 2026-11-03, plan e10,e05 (with E-06), launch `20261103T191010Z-wb-04b6`; E-19 kill analysis re-run 2026-11-04 |

## Results

Frozen into contract dc-2.0 on 2026-11-04 (section 1 TBD table and changelog section 16). Numbers per probe from the
program's run notes (run 3 and run 4 summaries); "FAIL" in the Result column is the tool's word against its
pre-registered rule, the verdict follows it.

| Probe | r2 4.4 criterion | Analysis | Play dir | Result | Numbers |
|---|---|---|---|---|---|
| E-01 | The car moves; `input_pre` vs `input_post` vs command recorded; setter-to-response step delay identified; `steer_sign` measured; auto-gear works or a scripted gearbox is needed; whether it works without `ForceEnableSimulation` | `tools/probes/t0eng/e01_actuation.py e01 <play_dir>` | run 3 | PASS | moves without ForceEnableSimulation (31.92 km/h; answer taken with `Physics.SetActive` at each handbrake release and on every driving step that finds the body asleep, as `DC_ProbeActuator` does; a driver without both wakes cannot use the answer); `steer_sign` +1 (corr(command, heading rate) 0.954 at lag 2), `yaw_axis_sign` +1, `local_axes` x right / y up / z forward (vel_l_z sign on 2089 of 2089 steps); inputs echo; delays steer 1 step plus a lag, throttle 2, brake 1; manual gearbox, scripted shifts 2-3-4-3-2 honoured |
| E-02a | First-pass steer to kappa fit usable by `llc_v1` | `tools/probes/t0eng/e01_actuation.py e02a <play_dir>` | run 3 | PASS | kappa / steer 0.2303 / 0.2284 / 0.2299 1/m at 15 / 30 / 50 km/h, R2 >= 0.99999; 8 of 15 cells `aborted_area` |
| E-04 | S105 component classes and `HasAutomaticGearbox()` read from the prefab | `tools/probes/t0eng/e04_components.py <play_dir>` | runs 2-3 | PASS | `HasAutomaticGearbox()` 0; 6 gears, 4 forward; assistance FULL; engine off and handbrake on at spawn |
| E-05 | R1 camera attached, current, vFOV; PNG dims logged; missing <= 1%; `Screenshot saved` count = files; FPS p5 >= 30; request-to-file latency distribution; raw MB/frame; engine-written file name | `tools/probes/t0eng/e05_images.py <play_dir> --log <console.log>` | runs 3-4 | FAIL (fps_p5 only) | run 3: 5992 of 6000 at 10 Hz on the runway, latency p50 80 / p95 101 ms; run 4: 5999 of 6000, p50 77 / p95 99 ms; run 1 forest 7.85% missing at 10 Hz; fps_p5 11.4-11.7 is the screenshot-frame cost (structural); 1.25 MB/PNG; suffix `.png` appended 5992/5992 and 5999/5999. image_hz decision: 5 Hz default, 10 Hz only as a trial-conf setting (D-B) |
| E-06 | Fiducial v2 decode >= 99.5% over >= 600 frames on the selected carrier (highest rate of widget, shape; presence and plausibility gates), no decode of that carrier outside [issue, min(issue + 2, done - 1)]; offset p50/p99 per carrier | `tools/probes/t0eng/e06_fiducial.py <play_dir>` | run 4 | PASS | shape carrier at depth 1.0 m (near plane 0.1 m): 5999 of 5999 decoded, 0 implausible, offset p50 0 / p99 1 frame, captured = `done_render_frame` - 1 on 5999 of 5999; widget carrier 0 of 5999 (widgets never reach MakeScreenshot files). Production depth 0.05 m with near plane 0.03 m awaits the first capture Play's E-06 check; 1.0 m / 0.1 m is the recorded fallback |
| E-06A | RMS <= 2 px at k = +1 (the image of frame C shows hf row C + 1), unique minimum with margin, at 60 and 30 FPS with backlog (grid and backlog images judged separately; slalom, backlog and brake must each have data) | `tools/validator/e06a_reproject.py <play_dir>` | run 3 | INCOMPLETE (no fiducial in run 3); diagnostic decisive | anchored at `done - 1`: argmin k = +1 in every condition, 628 of 640 moving images; RMS(0) / RMS(+1) px: slalom 60 4.89 / 1.44, slalom 30 5.57 / 1.11, curve 60 5.00 / 1.87, curve 30 5.35 / 1.54, brake 60 2.58 / 0.15, backlog 30 3.66 / 0.11; camera model p95 0.082 px |
| E-08a | Count, km and width histogram vs the decoder (34.5 km engine-network drivable road, E-08b, D-D); 0 duplicates; footer; no cap; unparented and decal-like widths flagged | `tools/roads/compare_dump_decoder.py <play_dir> --decoder <Arland_road_polylines_v2.tsv>` | runs 1-3 | PASS against the E-08b network rows | 181 roads, 39.83 km dumped (4.7 km runway, taxiway and panels are not a drivable class), 0 duplicates, 0 capped; road keys not stable across plays (n_pts +-1 on 10-12 roads, 13 of 181 keys differ), endpoints unique only for exact equality (4 twin pairs within 2 m) |
| E-10 | `GetFixedTimeSlice` logged; fixed-step callbacks per world-second; rows per callback, 0 missed | `tools/probes/t0eng/e10_ticks.py <play_dir>` | runs 3-4 | FAIL against the per-step rule; decides TBD-1 | step dt 1/30 s; run 3: 21296 of 21296 step rows with `EOnFixedFrame` slice = covered / 30, 30.6% of 30828 steps merged; run 4: 18113 steps in 10215 callbacks, 44% merged, covered_steps > 1 on 5988 rows; no `EOnSimulate` on the game mode: frame-bound rows (D-A), telemetry one row per callback, `world_timestamp` is the frame clock |
| E-11 | `$profile` under `E:\DriveDiT\wb_profile`; nothing new under OneDrive; preflight record joined (whether Workbench Play sees its command line) | console line `[DC_Probe] E-11: ...` and the run header `launch.*` keys (paste neither) | run 3 | PASS | `-profile` visible, `-dcLaunchId` visible=1 (uptime 35 s from the process start), launch mode cli |
| E-19 | Short or failed writes detected (including the read-only target, c09); the commit log stays consistent after a kill (both commit-log patterns) | `tools/probes/t0eng/e19_write.py <e19 play_dir> --kill <e19kill play_dir>` (INCOMPLETE without either) | run 3 and its e19kill Play | PASS | exact byte counts (1 B to 1 MiB), 0 through a READ handle, null open on a missing directory, a directory and a read-only target; c10 WRITE truncates; c11 append-reopen adds exactly the line; 212 of 212 committed chunks in both commit logs after the kill (re-run 2026-11-04) |
| E-20 | `VideoUserSettings` and `DisplayUserSettings` dumped; Workbench and client differences listed | `tools/probes/t0eng/e20_settings.py <play_dir> [--compare <client play_dir>]` | runs 2-3 | PASS | 33 + 2 engine variables; capture = the Workbench workspace; client compare and the PPQuality recursion at the first client session |

E-15 (unfocused Workbench) stays with DD-0; the bridge set (E-03, E-07, E-12..E-14, E-16..E-18) with DD-8.

## Values DD-11-min took from these results

All decided at the dc-2.0 freeze of 2026-11-04 (TBD-10 stays provisional; contract section 1).

| dc-2.0 item | Source probe | Value |
|---|---|---|
| TBD-1 row cadence (was: telemetry every k physics steps); whether WorldTimestamp advances per step | E-10 | frame-bound rows (D-A): one ctl and one telemetry row per fixed-frame callback with `covered_steps` and `render_frame`; step dt measured, 1/30 s; N = 6 at 5 Hz |
| TBD-2 image_hz (10, or 5 as the fallback) | E-05 | 5 Hz default (D-B); 10 Hz only as `clocks.image_hz_setting` trial_conf, with C37 asking for the session's utilisation evidence |
| TBD-3 writer semantics (`FileHandle.Write` return, c10 overwrite, c11 append-reopen, both kill commit logs) | E-19 | the r2 default stands (exact byte counts, WRITE truncates, append per line, both commit logs consistent after a kill) |
| TBD-4 engine-written screenshot file name | E-05 | `frames/f_<slot_id:08d>.png` (the engine appends `.png`) |
| TBD-5 steer_sign normalisation and vehicle-local axes (`local_axes`) | E-01 | `steer_sign` +1, `yaw_axis_sign` +1, x right / y up / z forward |
| TBD-6 launch id visible in Workbench Play | E-11 | visible: launch mode cli |
| TBD-7 gearbox and gear index semantics (run header `vehicle.gearbox`, `vehicle.gear_index_*`: derived from the gear counts with the vanilla rule, neutral = gears - forward - 1, first = gears - forward; e01 reads them back; reverse is not driven) | E-04, E-01 | S105 manual gearbox: 0 reverse, 1 neutral, 2..5 first..fourth (e04 counts 6 / 4, vanilla `SCR_ToggleHandbrakeAction`); gears 2-4 driven and honoured in run 3 |
| TBD-8 render-settings dump keys | E-20 | `VideoUserSettings` and `DisplayUserSettings` as E-20 dumps them |
| TBD-9 road key of the in-engine road dump | E-08a | the key text is per session; cross-session joins by the unordered endpoint pair within 2 m, one-to-one by segment Hausdorff (D-C) |
