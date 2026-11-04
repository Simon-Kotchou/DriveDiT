# Capture contract dc-2.0

Status: FROZEN 2026-11-04 (freeze decisions D-A to D-E adopted 2026-11-03, review round 1 corrections folded in on 2026-11-04 before any capture, changelog in section 16); a change needs a new version dc-2.x; user decision D15 adopted 2026-10-22

| | |
|---|---|
| Ticket | DD-11-min, branch `feat/contract-dc2-spec` (program ticket TKT-11, spec part) |
| Replaces | dc-1.0 (one 5 Hz CSV row per vehicle per tick). The dc-1.0 spec is never written (R27); dc-1.0 sessions are legacy and are never mixed with dc-2.0 data |
| Program of record | `CAPTURE_PROGRAM.md` r2.1: 3.3 (contract), 3.5 (clocks), 3.9 (storage, layout, writer), 2.2-2.3 (splits, leakage); trial-ladder [TL] 3-4 for what r2 inherits from r1 |
| Plan | `TICKETS_UNIFIED.md` r2: DD-11-min, conflict resolutions R01-R03, R06-R08, R10-R13, R26, R35 |
| Build | `program/1_9/MIGRATION_1_9.md` 2.2, 3.3 and 8.3: build stamps, `build_key`, steering channels, reserved fields |
| Machine-readable twins | `docs/capture_contract/dc-2.0.streams.json` (streams and session files), `docs/capture_contract/features_v1.json` (model-boundary registry), `drivedit_nav/route_schema.json` (routes) |
| Probe results | `docs/trials/T0-ENG.md` (written by DD-4-probe; does not exist yet) |
| Consumers | DD-4-min (writer), DD-5-min (reader, validator), DD-7-min (llc_v1, expert, T0 loop), DD-6-min (harvester), DD-11 (builder, nav adapters), DD-8 (bridge) |

## 0. How to read this

- MUST, MUST NOT and "never" are contract rules: a session that breaks one fails validation.
- **TBD-n** marks a value that a T0-ENG core probe decides. Section 1 lists every TBD with its r2 default, the value
  decided at the freeze and the evidence. DD-4-min, DD-5-min and DD-7-min keep each value in one constant or one
  session.json field. TBD-10 stays PROVISIONAL after the freeze: its values are session.json parameters, so measuring
  them changes a value, never a field.
- The JSON twins are normative for field names, order, types, units and keys. This document is normative for
  meaning. `tests/test_contract_dc2.py` checks that the field tables below equal the JSON (name, type, unit, Req,
  description, Range, Values and the bracketed tags), so a disagreement is a test failure, not a judgement call.
- nav_v1 (`drivedit_nav` and `drivedit_nav/route_schema.json`) is part of this contract and froze with it. Any
  change of nav behaviour (progress window, lookahead, maneuver window, stop clearance, route-deviation
  limit) or of the route schema is a new `nav_version` (nav_v2), stamped in `episodes.nav_version` and
  `session.json versions.nav_version`; nav_v1 output never changes in place.
- Change control. dc-2.0 is FROZEN since 2026-11-04 (section 16 lists every freeze decision with its date and
  evidence): any change of a field, unit, sign, key, cadence or registry rule is a new version dc-2.x and re-runs
  the L-gate. TBD-10 values may still be measured, because they are session.json values, never fields. A build change (1.8 to 1.9, a 1.9.x hotfix) also re-runs the L-gate
  but does not change the contract version: per-build values live in session.json and in files keyed by `build_key`,
  never in this contract.

## 1. Probe-dependent values (TBD-by-probe)

Each item is decided by the named probe of the DD-4-probe kit (results: `docs/trials/T0-ENG.md` and the program's
probe-run notes named in the Evidence column). The freeze copied each decided value into this document,
`dc-2.0.streams.json` (`tbd_by_probe[].decided`, `status`, `evidence`) and `features_v1.json`. TBD-1 to TBD-5 blocked
the freeze and are decided; TBD-6 to TBD-9 are decided; TBD-10 stays PROVISIONAL (a value, never a field). The
Default (r2) column is kept as history.

<!-- dc2:tbd -->
| Id | Item | Default (r2) | Decided (freeze) | Evidence | Decided by | Blocks freeze |
|---|---|---|---|---|---|---|
| TBD-1 | k rule: telemetry cadence and decision steps | physics_steps_per_decision = floor(100 ms / GetFixedTimeSlice() + 0.5), decided first (= 6 at a 60 Hz fixed step); telemetry_every_k = the largest divisor k of physics_steps_per_decision with k x GetFixedTimeSlice() <= 50 ms, at least 1 (= 3 at 60 Hz), so k always divides it; E-10 also logs the WorldTimestamp clock next to GetWorldTime per step | Frame-bound rows (D-A, adopted 2026-11-03): one ctl and one telemetry row per fixed-frame callback (EOnFixedFrame / EOnPostFixedFrame; EOnSimulate never reaches the game mode on 1.8) with physics_step = the index of the last engine fixed step it covers, covered_steps >= 1, render_frame and world_time_ms; the engine step dt is measured (1/30 s on 1.8), never GetFixedTimeSlice (it reports n/30 per frame); physics_steps_per_decision N = floor(1000 / decision_hz / step_dt_ms + 0.5) (6 at 5 Hz); decisions on a world-time grid every 1000 / decision_hz ms, taken at the first callback at or after the grid point, applied from the next physics step (R-01); no telemetry_every_k | probe run 4 E-10 (launch 20261103T191010Z-wb-04b6): 18113 engine steps of 1/30 s in 10215 EOnFixedFrame calls over 600 s, 44% merged, covered_steps > 1 on 5988 rows, 0 missed rows; run 3 E-10 and E-06A (20261103T061122Z-wb-86bd): GetFixedTimeSlice 0.1 s in a 3-step frame, world time per frame, pose on the frame clock (program notes t0/run4/RUN4_SUMMARY.md, t0/run3/RUN3_SUMMARY.md 5 and 8; decision memo t0/freeze/DECISIONS_dc2_freeze.md D-A) | E-10 | yes |
| TBD-2 | image_hz (10 or 5) and the one-request-in-flight rule | image_hz = 10 (world-time slot grid phase-locked to the decision steps, img_lead_ms = 0, one MakeScreenshot request in flight, in-flight timeout 2000 ms); fall back to 5 Hz first if E-05 shows misses or request-to-file latency that 10 Hz cannot sustain; at 5 Hz the HL decision rate drops to 5 Hz as well (the decision interval is the training interval, r2 3.5); img_lead_ms is set from the E-05/E-06 capture offsets | image_hz = decision_hz = 5 as the production default (D-B, adopted 2026-11-03): a 200 ms grid, N = 6 at 1/30 s steps, img_lead_ms 0, one request in flight, in-flight timeout 2000 ms; 10 Hz only as an explicit non-default session setting (trial conf key image_hz, recorded as clocks.image_hz_setting trial_conf) with decision_hz 10 and N = 3, and the validator warns (C37, SOFT) unless the session's own capture utilisation u95 <= 0.85 is shown | E-05 run 3 (latency p95 101 ms at 10 Hz on the runway, 0.13% missing) and run 4 (p95 99 ms, 5999 of 6000 files); run 1 forest 7.85% missing at 10 Hz; cost model 35 ms + 31 ms/MB caps forest scenes at about 9.2 Hz (decision memo D-B) | E-05 | yes |
| TBD-3 | writer semantics (FileHandle.Write return values, Close durability, taskkill mid-commit) | FileHandle.Write returns the number of bytes written (FileHandle.c:93-103; WriteLine returns void and is never used); any return other than the expected byte count is a short or failed write; a closed file survives a process kill; one open-APPEND-write-close cycle per line adds exactly that line (commit.csv and heartbeat.csv; E-19 c11 and the e19kill e19_commit_append.csv log); no file is rewritten in place (E-19 c10 only records whether WRITE truncates) | The r2 default stands: Write returns the exact byte count (1 B to 1 MiB) and 0 through a READ handle; a missing directory, a directory used as a file and a read-only target give a null open; WRITE truncates; one open-APPEND-write-close cycle per line adds exactly that line, and after a process kill during commits both commit logs list every committed chunk; commit.csv and heartbeat.csv stay append-per-line (no keep-open fallback) | E-19 run 3 play_01 and e19kill play_01 (launch 20261103T061122Z-wb-86bd): e19_write.py --kill PASS, 212 of 212 committed chunks in both commit logs (re-run 2026-11-04, t0/freeze/e19_kill_2026-11-04.txt) | E-19 | yes |
| TBD-4 | engine-written screenshot file name | the request path has no extension (frames/f_<slot_id:08d>) and the engine appends .png, so the file is frames/f_<slot_id:08d>.png (1.8 evidence: a request ending .bmp produced *.bmp.png) | The engine appends .png to the extension-less request: the file is frames/f_<slot_id:08d>.png (Workbench 1.8.0.13; the client exe is checked by C11 at its first session) | E-05 run 3: .png appended on 5992/5992 files, E-06A 821/821; run 4 5999/5999 | E-05 | yes |
| TBD-5 | steer_sign normalisation (and the yaw-axis and local-axes checks) | steer_sign = -1: CarSteering is +1 for left, the contract is right-positive, so steer_contract = steer_sign * steer_engine; yaw_axis_sign = +1 [inference: Enfusion is left-handed with Y up, so +angvel_local_y is clockwise seen from above]; vehicle-local axes x right, y up, z forward; checked by tools/probes/t0eng/e01_actuation.py: steer_sign, yaw_axis_sign and local_axes (z forward: sign of vel_l_z on forward motion; x right: sign of corr(acc_l_x, v x heading rate)) | steer_sign = +1: a positive SetSteering turns right (the r2 default -1 is refuted); yaw_axis_sign = +1; vehicle-local axes x right, y up, z forward | E-01 run 3: corr(command, heading rate) 0.954 at lag 2, all 15 E-02a cells give kappa > 0 and acc_l_x > 0 for a positive steer; corr(angvel_l_y, heading rate) 0.980; vel_l_z sign agrees with forward travel on 2089/2089 steps | E-01 | yes |
| TBD-6 | launch id visible in Workbench Play | System.GetCLIParam("dcLaunchId") returns Workbench's own -dcLaunchId argument in Play mode (launch_guard.launch_mode cli). Answered by DD-4-probe's E-11 console line (-dcLaunchId visible=0 or 1); DD-1's 'Capture output allowed' or 'capture output refused' line tests GetCLIParam("profile") and is only an inference about it. If the id is not visible, Workbench sessions pass through the escape (launch_guard.launch_mode wb_escape, section 5.1); the session.json fields exist either way | Visible: System.GetCLIParam sees Workbench's own -dcLaunchId in a Play (launch_guard.launch_mode cli); wb_escape stays for a Play without a visible command line | E-11 run 3: -dcLaunchId visible=1, uptime_s=35 (counted from the process start), mode cli | E-11 | no |
| TBD-7 | gearbox and gear index semantics | HasAutomaticGearbox() is true on the S105 and the automatic gearbox works driverless; gear indices map to reverse/neutral/forward through the vehicle card (VehicleCinematicTrack defaults to index 2, so the mapping is read back, never assumed). T0-MIN is forward-only: the card's gear_indices_forward come from E-01 (gear_indices seen while moving forward); gear_index_neutral and gear_index_reverse stay null until a probe phase holds neutral and drives in reverse (an E-01 extension or E-02Q, DD-7-ident); no session commands hl_dir -1 before that | Manual gearbox (HasAutomaticGearbox() 0); the S105 has 6 gears, 4 forward: index 0 reverse, 1 neutral, 2..5 first to fourth. General rule (vanilla gear computation): neutral = GearboxGearsCount() - GearboxForwardGearsCount() - 1, first = neutral + 1, top = GearboxGearsCount() - 1, reverse below neutral. The S105 card names gear_indices_forward [2, 3, 4, 5], gear_index_neutral 1, gear_index_reverse 0; reverse is not driven yet, so no session commands hl_dir -1 before a probe drives it (DD-7-ident) | E-04 runs 2-3 (HasAutomaticGearbox 0, 6 gears, 4 forward); E-01 run 3 (gears 2-4 driven and honoured, scripted 2-3-4-3-2, ratio steps 3.8 / 2.12 / 1.41); vanilla 1.8.0.13 SCR_ToggleHandbrakeAction.c:100-108, SCR_VehicleGearCondition.c:32-44 | E-04, E-01 | no |
| TBD-8 | render-settings dump keys | GetGame().GetEngineUserSettings() modules VideoUserSettings and DisplayUserSettings, keys from SCR_SettingsManagerVideoModule.c:12-56 | VideoUserSettings and DisplayUserSettings as E-20 dumps them (33 + 2 engine variables; the capture is the Workbench workspace); the client compare and the PPQuality recursion are checked at the first client session (a value, never a field) | E-20 runs 2-3 | E-20 | no |
| TBD-9 | road key of the in-engine road dump | road_key = <n_pts>/<x0>:<z0>/<x1>:<z1> of the road's GetPoints: n_pts in decimal, the first and last point's x and z each formatted by Enforce float.ToString(-1, 2) (fixed point, 2 decimals), 0 for a road without points (DD-4-probe DC_RoadDump.MakeKey); E-08a dedupes by it and telemetry.road_key writes the same text, so offline matching (E-08b) joins on the logged text and never re-formats coordinates | road_key text unchanged (<n_pts>/<x0>:<z0>/<x1>:<z1>) and valid within a session only; cross-session joins match by the unordered endpoint pair, both endpoints within 2 m (n_pts excluded), one-to-one: among the candidates the smallest E-08b segment Hausdorff distance wins (a global greedy assignment, so a twin road never takes another's partner) and twins are reported (D-C, adopted 2026-11-03) | E-08a runs 1-3: n_pts changes by +-1 on 10-12 of 181 roads between launches, 3 roads move an endpoint 0.1-0.9 m, 13 of 181 keys differ between runs 2 and 3; the endpoints are unique for exact equality (181/181) but not within 2 m: 4 twin pairs (8 of 181 roads) lie 0.49-1.56 m apart (review round 1), so the join is one-to-one (t0d/decoder/E08B.md 7, decision memo D-C) | E-08a | no |
| TBD-10 | collision detector and crash-event parameters (session.json events.*) | collision_dv_mps = 4.0 over collision_window_ms = 100 of world time (gap tolerance collision_window_tol_ms 50), PROVISIONAL: 2.0 x the E-01 S105 full-braking maximum (2.01 m/s per 100 ms) and 0.56 x the weakest of the 7 vehicle impacts of the dc-1.0 session_10423 (7.2 m/s); collision_peak_ms 200, collision_refractory_ms 1000, episode_end_wait_ms 250, crash_cooldown_ms 3000, episode_block_max_ms 60000, rollover_up_y 0.5 held rollover_hold_ms 500, state_poll_ms 100. Rows are frame-bound (D-A: one callback per fixed frame, 1 to 3 physics steps of 1/30 s on 1.8), so every window is world time, never a count of steps or callbacks. The TBD-10 probe (an E-01 extension: full brake from 60 km/h on asphalt and dirt with 10 Hz images on, so frames merge up to 3 steps, a curb strike, a 1 m drop, scripted 10 and 30 km/h wall hits) logs dv_h per callback and sets the threshold at 2 x its non-collision p99.99; measured again for every build_key, on 1.9 before production data and after any veh_cfg_override | PROVISIONAL, not decided by the freeze: the defaults stand until the TBD-10 probe measures them (a value, never a field; per build_key) | crash-event design and its two review rounds (t0d/crash_events/IMPL.md) | E-01 extension | no |

Decisions applied at the freeze (what changed where):
- TBD-1 (D-A, frame-bound rows): `ctl` and `telemetry` gain `covered_steps` and `render_frame`; `physics_step` is the
  engine's fixed-step clock and advances by `covered_steps` per row; `clocks.telemetry_every_k` is withdrawn (one
  telemetry row per callback); `session.json engine.step_dt_s` / `engine.step_dt_source` record the measured step;
  decisions follow the world-time grid of section 6; every check that assumed rows on every k-th step (C06, C07, C08,
  C15, C17, C28, C31) is re-specified on callbacks, world time and covered steps (DD-5-min).
- TBD-2 (D-B, 5 Hz): `clocks.image_hz` and `clocks.decision_hz` default to 5 and N = 6 at the 1/30 s step, so
  `action_exec` is [T,6,4]; the 10 Hz override is `clocks.image_hz_setting` `trial_conf` with N = 3 and `action_exec`
  [T,3,4] (`features_v1.json` `offsets.override_10hz`); C37 (SOFT) asks for the session's capture utilisation.
- TBD-3: no change (append per line; the writer's keep-open and length re-check fallbacks stay off).
- TBD-4: the harvester's and reader's image name is `frames/f_<slot_id:08d>.png`; the img stream keeps the request
  path.
- TBD-5: `control.steer_sign` +1 (the r2 draft had -1); `steer_sign` and `yaw_axis_sign` stay per session and per
  vehicle card, and export always normalises to the contract sign (section 7).
- TBD-7: the vehicle card's gear indices of section 7.
- TBD-9 (D-C): the cross-session road join of 9.8 `road` (one-to-one, twins reported); no field changes.
- Folded facts (no choice needed): the logged pose follows the frame clock and the image of render frame C shows the
  state of hf row C + 1 (section 6); the fiducial carrier is the debug-Shape carrier (`rig.fiducial` `carrier`,
  `depth_m`) at 0.05 m with the rig near plane at 0.03 m (`rig.near_plane_m`), awaiting the first capture Play's
  E-06 check (fallback 1.0 m and 0.1 m, validated by run 4), the widget carrier is a diagnostic only;
  `img.done_render_frame` records the frame the file was first seen in (captured frame = done_render_frame - 1 in
  Workbench).
- TBD-10: the values live in `session.json` `events.*`, so a measured threshold never changes a field; the optional
  `dv_max_mps` columns (9.2, 9.7) let a new threshold be re-applied offline without capturing again.

## 2. Scope and invariants

dc-2.0 is what the capture code writes in-engine (rig R1, Mode A, 1.8.0.13 now, the first qualified 1.9 stable build
later), what the harvester, reader and validator read, and what the dataset builder hands to a model.

- Streams logged in-engine: `ctl`, `telemetry`, `hf`, `img`, `events`, `episodes` (and `traffic`, reserved for T2).
  Each has its own key and its own clock (section 6).
- Images are written by `System.MakeScreenshot` to `frames/`; completion and the captured render frame are decoded
  offline from fiducial v2 (R11), never inferred from timing.
- The model boundary is the named tensors of `features_v1.json` (section 12), not the 6D vector of dc-1.0. The 6D view
  survives only as the deprecated `drivedit_nav.to_legacy_6d` (section 14).
- One session has one ego entity for its whole life. Every `ctl` and `telemetry` row describes a live ego; a step
  without a live ego is a C05 failure, so any later ego replacement (DD-4) must switch atomically between episodes.
- The capture code is build-agnostic: no 1.8-only constant appears in code or in this contract. The fixed step dt,
  the steps a callback covers, `steer_sign`, gear maps and actuator dynamics are measured per build and recorded per
  session and per vehicle card (the 1.8 values in section 1 are the decided values of build 1.8.0.13).

## 3. Map identity, splits and the sealed test map

### 3.1 Canonical map ids (R13)

`map_id` is one of `arland`, `kolguyev`, `everon`. `terrain` is the engine name of the root world and is a separate
field. They are never interchanged: `cain` and `eden` are not map ids.

| map_id | terrain | Root world | Capture world | Splits | Role | Storage root |
|---|---|---|---|---|---|---|
| `arland` | `Arland` | `{A9806AF617972E97}worlds/Arland/Arland.ent` | worlds/DataCapture/DC_Arland.ent | `train`, `val_c`, `val_l`, `a0`, `a_stress` | TRAIN + VAL-C + VAL-L layout diagnostic + A0/A-stress references | `E:\DriveDiT\captures\arland\` |
| `kolguyev` | `Cain` | `{1EA95DAE3230BEB0}worlds/Cain/Cain.ent` | worlds/DataCapture/DC_Kolguyev.ent (DD-2) | `k_val`, `k_rep` | OOD-VAL: K-VAL selection within its budget, K-REP report only (pending D4); never training | `E:\DriveDiT\captures\kolguyev\` |
| `everon` | `Eden` | `{853E92315D1D9EFE}worlds/Eden/Eden.ent` | worlds/DataCapture/DC_Everon.ent (DD-2-everon, just before T4-E) | `everon_harness`, `everon_test` | SEALED TEST (zero-shot); every Eden-rooted world counts as Everon | `E:\DriveDiT\heldout\everon\SEALED\` |

- `map_id` is declared by the capture world (DC_Arland gives `arland`) and verified at session start by the L8
  fingerprint: world size and latitude must match the map card (`program/maps/map_cards_v1.json`). A mismatch logs an
  ERROR and aborts before the session directory exists (`end_reason` is never written because no session exists).
- `map_id` is never parsed from `GetWorldFile()`, which returns the SubScene (`worlds/DataCapture/DC_Arland.ent`;
  CR-100). `map.world_file` records it for provenance.
- Every Eden-rooted world counts as Everon, including SubScenes and the user's ArmaDrive projects.

### 3.2 Split rules by map_id

- **arland.** Splits `train`, `val_c`, `val_l`, `a0`, `a_stress` (splits_v2, PROPOSED until E-08b; D5).
  - A session's split comes from its route's split tag (`drivedit_nav` route schema), never typed by hand, and is
    checked against `road.region` per frame (C24): a `train` route has no frame inside the VAL-L box
    x in [1536, 2560), z in [512, 1536) or its 150 m buffer; a `val_l` route has every frame inside the box.
  - `val_c` is not a route tag: it is 10% of `train` sessions, chosen by
    `int(sha256(session_id)[:8], 16) mod 10 == 0` (proposed; DD-6 owns the manifests).
  - `train` fits every learned part, including normalisation statistics. `val_c` is in-distribution debugging only.
    `a0`, `val_l` and `a_stress` are evaluation references and are never used for selection (L2).
- **kolguyev.** Splits `k_val` (checkpoint selection and early stopping within the logged K-VAL budget) and `k_rep`
  (report only), boundary from the Kolguyev dump before any run (DD-2, pending D4). Kolguyev data never enters a
  training manifest or normalisation statistics.
- **everon.** Splits `everon_harness` and `everon_test` only. Everon is the sealed zero-shot test map:
  - no Everon data (frames, latents, statistics, model outputs) in any training, pre-training, validation,
    normalisation or model-selection set, curriculum statistic or FD reference (L1, L2);
  - before the sealed run only non-learned drivers run on Everon (L5), and behavioural harness metrics go to the sealed
    report store unseen (L12);
  - storage only under `E:\DriveDiT\heldout\everon\SEALED\` (L7);
  - **loaders refuse Everon outside the sealed evaluation path.** Every reader, dataset builder, manifest builder and
    route loader (DD-5-min, DD-6, DD-11, `drivedit_nav.load_route`) raises unless it is called on the sealed
    evaluation path (`sealed_eval=True` in Python, `--sealed-eval` on a command line) when it meets `map_id`
    `everon`, `terrain` `Eden`, a route split `everon_*`, or a path under `heldout/` or a `SEALED` directory.
- **One build.** Train, selection and test sets of an Everon-facing model share one `build_key`, the target build's
  (MIGRATION_1_9 2.2). Manifests are single-`build_key`; 1.8 sessions keep their 1.8 `pin_id` and `build_key` and stay
  out of those sets except the declared appearance-pretraining stage.

## 4. Session id, layout and path policy

### 4.1 Session id

`<map_id>_<trial_id>_<YYYYMMDDTHHMMSSZ>_<exe>_<inst>_<rand4>`, for example
`arland_T0b_20261025T091512Z_wb_i00_7f3a`. Regular expression (also in the streams JSON):

```
^(arland|kolguyev|everon)_[A-Za-z0-9-]{1,16}_[0-9]{8}T[0-9]{6}Z_(wb|cl|cld)_i[0-9]{2}_[0-9a-f]{4}$
```

- The time is UTC from `System.GetYearMonthDayUTC` and `System.GetHourMinuteSecondUTC`; `rand4` is 4 hex digits from
  a generator seeded with `System.GetTickCount()`. It is never built from world time (dc-1.0 ids collided).
- `exe` is `wb` (Workbench), `cl` (ArmaReforgerSteam client) or `cld` (Diag client); `inst` is `i00` in Workbench and
  the client instance `iNN` otherwise. `trial_id` has no underscore.
- The capture code refuses an existing session directory (ERROR, nothing written).

### 4.2 Layout

```
$profile:DC_Sessions/<session_id>/
  session.json              written once, after every start check passed (section 5)
  session_end.json          written last; its presence seals the session
  heartbeat.csv             header at session start, one row appended every 5 s
  ctl/chunk_00000.csv ...   ctl/commit.csv
  telemetry/ ...            same pattern for every stream
  hf/ ...
  img/ ...                  the img stream (rows), not the pictures
  events/ ...
  episodes/ ...
  traffic/ ...              reserved: written from DD-4 (T2); absent before
  frames/f_00000000.png     screenshots: the request path has no extension, the engine appends .png (TBD-4)
```

Stream summary:

| Stream | Status | Written in | Cadence | Key | Key step | Chunk rows |
|---|---|---|---|---|---|---|
| `ctl` | required | EOnPostFixedFrame | one row per fixed-frame callback (1 to 3 physics steps on 1.8, TBD-1) | `physics_step` | `covered_steps` | 600 |
| `telemetry` | required | EOnPostFixedFrame | one row per fixed-frame callback, the same rows as ctl (TBD-1) | `physics_step` | `covered_steps` | 200 |
| `hf` | required | EOnFrame | every render frame | `render_frame` | 1 | 600 |
| `img` | required | EOnFrame | image_hz world-time grid (TBD-2: 5 Hz), each row written when its slot resolves | `slot_id` | 1 | 100 |
| `events` | required | as they occur | irregular | `event_seq` | 1 | 100 |
| `episodes` | required | at episode end | one row per episode | `episode_id` | 1 | 1 |
| `traffic` | reserved | EOnPostFixedFrame | the telemetry callbacks, one row per live traffic vehicle (from DD-4, T2; absent before) | `physics_step`, `veh_slot_id` | - | 2000 |

The camera description of the review's camera.json lives in `session.json` `rig` (R12). A stream directory holds only
chunks with at least one row: a stream with no row in a chunk period (events, episodes) writes no chunk and no
commit line for it.

### 4.3 Path policy

- The game writes only under `$profile:DC_Sessions/<session_id>/`. There is no capture-root override such as
  `-dcCaptureRoot` (R26), so the FileIO authorisation prompt and `-scriptAuthorizeAll` are never needed.
- `$profile:DC_Sessions/` holds dc-2.0 sessions only. DD-1's dc-1.0 `$profile:DrivingData/session_<n>/` folders and
  DD-4-probe's `$profile:DC_Probes/` lie beside it and are never read by the dc-2.0 harvester, reader or validator,
  so a Play that runs a dc-1.0 producer and a dc-2.0 session never mixes their files.
- `$profile` must lie under `E:\DriveDiT\` (the launch guard, section 5, and DD-1's profile guard). The C: drive is
  never a capture, scratch or harvest target.
- The harvester moves a sealed session, by rename on E:, to:
  - `E:\DriveDiT\captures\arland\<YYYY-MM-DD>\<trial_id>\<session_id>\` or
    `E:\DriveDiT\captures\kolguyev\<YYYY-MM-DD>\<trial_id>\<session_id>\` on PASS;
  - `E:\DriveDiT\captures\_rejected\<session_id>\` on FAIL (Arland and Kolguyev only);
  - `E:\DriveDiT\heldout\everon\SEALED\{harness,harness_reports_sealed,openloop_test,closedloop_test}\...` for every
    Everon session, PASS or FAIL; Everon never lands in `captures\`;
  - `E:\DriveDiT\compat19\<pin_id>\` for any session from an experimental (`-exp`) pin; never `captures\`.
- C03 checks the map-to-root mapping; C24 checks that no `heldout\` path appears outside the sealed test manifest.
- `$profile:DC_STOP` is the stop file of r2 3.9 (supervisor, DD-9): when it exists at a heartbeat, the session ends the
  episode, writes `session_end.json` (`end_reason` `dc_stop`) and requests close.

## 5. Launch guard, trial parameters and session lifecycle

### 5.1 Launch guard (fail closed)

Nothing is written, not even a directory, unless the guard passes in one of two modes (DD-4-probe
`DC_LaunchGuard`, reused by DD-4-min). `session.json` `launch_guard.*` records which mode passed and how.

Mode `cli` (every client session; Workbench when TBD-6 is positive):
1. `System.GetCLIParam("dcLaunchId", id)` succeeds and `id` is a launch id (20-64 characters of `[0-9A-Za-z-]`);
2. `-profile` is visible and lies inside `E:\DriveDiT`, and in Workbench the `$profile` resolved by
   `Workbench.GetAbsolutePath` lies there too; a `$profile` Workbench cannot resolve refuses;
3. `FileIO.FileExists("$profile:dc_launch/launch_" + id + ".json")` is true. The launchers are meant to copy the
   preflight record there, so a Workbench or client started any other way never writes. As of DD-0 `6f045b1`
   neither does so yet: `wb_R1.bat` passes `-dcLaunchId` and copies the record only after DD-0 component 4 plus the
   follow-up patch `t0/dd0_followup/wb_R1_dclaunch.patch` is applied and installed; `client_R1_iNN.bat` passes the
   id but copies nothing, which DD-9-min must add before the first client session (S1);
4. the record parses, has `status` `ok`, names the same launch id and this exe (`wb` or `client`), and yields
   `pin_id`, `pin_sha256`, the buildids, `channel`, `use_class` and, from dd-launch/2 on, `build_key` (dd-launch/1
   records: `channel` stable, `use_class` dev, `build_key` null and supplied by the harvester's `harvest.json`).

Mode `wb_escape` (Workbench only, for a Play that cannot see the command line at all, TBD-6 negative):
1. no command-line parameter is visible (a visible command line without `-dcLaunchId` always refuses: Workbench was
   then started by hand), and the escape is enabled on the capture component;
2. the resolved `$profile` lies inside `E:\DriveDiT` (refused when it cannot be resolved);
3. the newest `-wb-` record in `$profile:dc_launch` is valid (as in cli item 4), younger than the deploy
   (`DC_BuildInfo.DEPLOY_UTC`), and was written by the launch that started this Workbench process: at most 300 s
   before the process start, estimated as `System.GetUnixTime() - System.GetTickCount() / 1000`, and not after it
   (`launch_guard.escape_binding` `process_start`). Only the newest record is considered; when it belongs to another
   Workbench start the guard refuses and never falls back to an older record. If E-11 shows that `GetTickCount`
   does not count from the process start, the binding can be switched off for a Play
   (`escape_binding` `newest_after_deploy`).
4. Escape sessions are development and qualification data: C02 accepts `wb_escape` with `use_class` dev, probe and
   appearance and refuses it for `g19` and `production` (proposed; Workbench sessions are dev-only unless C30 parity
   passes anyway). The client has no escape.

A refusal logs one ERROR line naming the failed condition. Edit mode never captures: every DC component checks
`SCR_Global.IsEditMode` first and queues nothing.

### 5.2 Trial parameters

- `-dcTrial <trial_id>` names `$profile:dc_trials/<trial_id>.json`, copied there by the launcher or the supervisor.
- Without `-dcTrial`, the session reads the one-line pointer `$profile:dc_trials/ACTIVE` at each session start, so one
  Workbench process can run T0a, T0b and T0d in turn (proposed; the agent rewrites ACTIVE between Plays).
- Neither present, or the conf missing or unparsable: ERROR, no session.
- The conf holds at least `trial_id`, `duration_s`, `warmup_s`, `seed` and the route file; DD-4-min verifies the 1.8 JSON
  loader API before use. The parsed conf is copied into `session.json` `trial.conf`.

### 5.3 Lifecycle

1. Edit-mode guard, launch guard, trial resolution.
2. L8 fingerprint and build check: `GetBuildVersion()` must equal the pin's expected engine version once the pin
   records one (MIGRATION_1_9 8.3); a mismatch aborts.
3. Readiness: the ego is spawned driverless (Mode A), settles 3 s, is live, and the rig camera is current and
   attached. A readiness timeout is an ERROR and creates no directory.
4. The directory is created (refused if it exists) and `session.json` is written. The streams start; the warm-up is
   recorded and masked (`events` `warmup_end`).
5. Episodes run (`events` `episode_start` ... `episode_end`, one `episodes` row each).
6. End (duration, Play stop, `DC_STOP`, or an error): no new image request is issued; `hf` rows continue until no
   request is in flight or `clocks.img_inflight_timeout_ms` has passed (requests still unresolved count as
   `images.pending_at_end`); then every open chunk with at least one row is closed and committed, and
   `session_end.json` is written last. Status `ok` only if every commit succeeded; otherwise `failed`.
7. Shutdown is idempotent and removes every owned callback; nothing is captured after `session_end.json`
   (`ScriptCallQueue.Dump()` shows no DC_* entry, T0d).

Checks that fail before step 4 (guard, trial, fingerprint, build, readiness) create no directory, so they have no
`end_reason`; they log one ERROR line. After step 4: an ego lost before the first row ends the session `failed`
with `end_reason` `zero_ego`, 0 rows and 0 images; an ego or camera lost later ends it `failed` with `ego_lost` or
`camera_lost` (a camera that fails to initialise fails readiness). An ego destroyed after the first row
(`DamageManagerComponent.IsDestroyed()`) ends the session with `end_reason` `ego_destroyed`, which is not an error:
status stays `ok`, because a crash is valid data. `ego_lost` stays for an ego that vanishes or changes identity. An
ego that, between episodes, stays tilted, immobile or burning without being destroyed for
`events.episode_block_max_ms` (9.5.1) ends the session through the planned end (5.3 step 6, with its drain) with
`end_reason` `ego_disabled`, also not an error; the session logs one WARNING line naming the condition.
- One terminal step L: the last step whose rows are written. Destruction seen after a step's simulation lets that
  step's ctl and telemetry rows be written and ends after it (L is that step); seen at a frame start or before a
  step, L is the previous step.
- At L the core writes the ego's pending `collision` row, `veh_damage` `state=destroyed` and, when it burns,
  `veh_fire_start` (9.5.1), all with `physics_step` L; no ctl or telemetry row follows L.
- The open episode ends at the seal with `end_physics_step` L, the `world_time_ms` of the row of L as `end_world_ms`,
  and the termination already latched (`collision`); `session_end.json` `end.physics_step` is L. Its `end.world_ms`
  stays the time of the seal, as for every session.

## 6. Clocks, keys and monotonicity (r2 3.5; R01, R03; freeze decisions D-A and D-B)

| Counter | Stream | Rule | Checked by |
|---|---|---|---|
| `physics_step` | `ctl`, `telemetry` | The engine's fixed-step clock: the 0-based index of the last engine fixed step the row's callback covers, counted from the session open (the first row has `physics_step` = `covered_steps` - 1); each row advances it by its own `covered_steps`; ctl and telemetry hold the same keys, and per key the same `render_frame` and `world_time_ms` | C06, C07, C08 |
| `covered_steps` | `ctl`, `telemetry` | Engine fixed steps the callback covers, >= 1: `BaseWorld.GetFixedSteps()` of its render frame (1 to 3 on 1.8); 1 per call when the engine delivers `EOnSimulate` | C06 |
| `render_frame` | `ctl`, `telemetry` | `GetFrameNumber()` at the callback; non-decreasing (one fixed-frame callback per render frame at most, so strictly increasing on 1.8) | C07 |
| `render_frame` | `hf` | `GetFrameNumber()`, one row per render frame, +1 per row | C06, C07 |
| `slot_id` | `img` | 0-based, +1 per row; advances only while the ego is live and the camera current | C06, C07, C10 |
| `event_seq` | `events` | 0-based, +1 per row | C06, C19 |
| `episode_id` | `episodes` | 0-based, +1 per episode; -1 in other streams outside an episode | C06 |
| `world_time_ms` | all | `WorldTimestamp.DiffMilliseconds` against the session start, integer ms, non-decreasing. It is the frame clock: one value per render frame, shared by the hf row of frame F and the ctl and telemetry row of the callback of frame F (E-10 run 3: 21296 of 21296 rows); equal values in consecutive hf rows only while paused or within one frame's world-time resolution | C06 (ctl = telemetry per key), C07 (ctl = hf of the same frame) |
| `last_physics_step` | `hf` | `physics_step` of the newest callback of an earlier render frame (the callbacks of frame F run after its `EOnFrame`), -1 before the first: exactly the newest ctl key whose `render_frame` < F | C06, C07 |

- **Callbacks (D-A).** On 1.8 the game mode entity gets no `EOnSimulate` / `EOnPostSimulate`. It gets one
  `EOnFixedFrame` / `EOnPostFixedFrame` pair per render frame that runs at least one engine fixed step, after that
  frame's `EOnFrame`, and the pair covers all of the frame's steps: probe run 4 (600 s with 10 Hz screenshots) counted
  18113 engine steps of 1/30 s in 10215 callbacks, 44% of the steps merged, `covered_steps` > 1 on 5988 rows, almost
  exactly the 5999 screenshot frames. Per-step rows from the game mode are therefore impossible while capturing, and
  `ctl` and `telemetry` are frame-bound: one ctl and one telemetry row per callback with `physics_step`,
  `covered_steps`, `render_frame` and `world_time_ms`. The core applies the held command once per callback (in
  `EOnFixedFrame`), so all covered steps run with the same inputs; the rows are written in `EOnPostFixedFrame`, after
  the covered steps. An engine build that delivers `EOnSimulate` to the game mode gives one callback per step
  (`covered_steps` 1) and every rule below holds unchanged.
- **Step dt.** The engine fixed step dt is measured during readiness, per frame with at least one fixed step: the step
  callback's time slice divided by the steps it covers (`EOnFixedFrame` reports n x the step for a frame of n steps:
  covered_steps / 30 on 21296 of 21296 rows of run 3 E-10, so the ratio is the step even in merged frames; 1 per
  `EOnSimulate`). From 30 fixed steps and 30 ratios on, every readiness frame decides afresh, never locking in a
  guess: the step dt is the median of the ratios once 90% of them lie within 0.5% of it (`step_dt_source` engine);
  while they disagree, the measurement goes on, and the measured fixed step (`measured`: the EOnFrame time slices
  after the first frame with a step through the last one, over the steps run in them) is taken only once its span is
  long (at least 90 steps), stable (its two halves, split at half its steps, agree within 0.5%) and confirms the ratio
  median within 0.5%, half of C28's 1% tolerance. The span alone is not bounded: the fixed-step accumulator's
  remainder and backlog at both of its ends stay in it (review round 2: 34 ms frames with one step each measure 0.034
  s, 2% high). Without a trustworthy value by the readiness timeout readiness is refused with the reason, so no
  session opens with an untrustworthy step dt. `session.json engine.step_dt_s`, `step_dt_source`; `session.json
  observed` keeps `readiness_step_ratio_median_s`, `readiness_step_ratios`, `readiness_step_ratio_agree_frac`,
  `readiness_measured_step_s`, `readiness_measured_half_a_s`, `readiness_measured_half_b_s` and, as a cross-check,
  `readiness_slice_ratio_median_s` (`GetFixedTimeSlice()` / `GetFixedSteps()` read in the same callbacks).
  `BaseWorld.GetFixedTimeSlice()` alone reports n/30 for a frame of n steps (0.1 s in a 3-step frame, runs 3-4) and is
  never the step dt; `engine.fixed_time_slice_s` keeps its raw value at the open for provenance only. A module that
  needs the physics time of a callback uses `covered_steps` x `step_dt`.
- **Decision grid (D-A, D-B).** `physics_steps_per_decision` N = floor(1000 / `decision_hz` / step_dt_ms + 0.5) (an
  explicit half-up rounding, the same in Python and Enforce): 6 at the 5 Hz default and 3 at the 10 Hz override with
  1/30 s steps. N is the nominal length of a decision interval and of `action_exec`; the real interval varies.
  Decisions lie on a world-time grid: grid point g = 0, 1, 2, ... is due at anchor + g x 1000 / `decision_hz` ms
  (integer ms: `decision_hz` is 5 or 10), anchor = the `world_time_ms` of the session's first ctl row. The decision
  callback of grid point g is the first callback whose `world_time_ms` is at or past its due time. A callback that
  reaches several grid points (a hitch longer than one grid period) takes one decision, for the newest of them; the
  skipped grid points are counted (`session_end.json observed.decision_grid_points_skipped`). P_t is the
  `physics_step` of decision callback t.
- **Decision boundary** (R-01, machine-readable in the streams JSON `timing`). In decision callback C_t the held
  command is applied (`EOnFixedFrame`), the covered steps run, the ctl and telemetry rows of C_t are written
  (`EOnPostFixedFrame`) and then, still in that callback, the HL driver decides on the state of that row. The new
  command applies from physics step P_t + 1, the first step of the next callback, and is held through the next
  decision callback C_t+1, so:
  - `ctl.hl_*` of a row is the command applied in that row's covered steps. `hl_seq` is -1, the no-command sentinel
    (`hl_*` empty), until the first command; it advances by 1 exactly on the row after every decision callback of an
    episode that starts or goes on (a driver that keeps its command, `Decide` false, gets the held command numbered
    again with the same values) and on no other row, so no number exists before the first episode; a command changes
    only together with an `hl_seq` increment, or at an episode boundary (C07);
  - an episode is admitted at a decision callback only after the driver's admission handshake: it confirms that this
    callback's decision will give the first command, or it supplies an initial command with the episode start; a
    driver that cannot decide yet defers the admission to the next grid point and nothing is committed, so a driver
    that is not ready never fails a session. Only a first decision without a command after an admission without an
    initial command fails the session (`script_error`), which a well-behaved driver never reaches;
  - the decision interval of t is the steps (P_t, P_t+1] of n_t = P_t+1 - P_t steps: nominally N, but the fixed-step
    clock lags world time by up to about 250-310 ms while capturing (runs 3-4) and catches up, so n_t varies around N;
  - `ctl.exp_*_now` of a row is the expert re-evaluated after its callback; at C_t it is the label of decision t;
  - `action_cmd_t` = `ctl.hl_*` of the row after C_t; `action_label_t` = `ctl.exp_*_now` of C_t; `state_t` = the
    telemetry row of C_t, which holds no step of `action_cmd_t`; `action_exec_t` = the executed llc values of steps
    P_t + 1 .. P_t + N, step q taken from the ctl row that covers it (a row covers `physics_step` - `covered_steps` + 1
    .. `physics_step`), an entry left empty when q lies past P_t+1 (the interval was shorter than N; steps of a longer
    interval past P_t + N are not in the tensor); obs_t+1 at C_t+1 holds exactly the n_t steps of `action_cmd_t`;
  - the closed-loop bridge (DD-8) holds the next physics step after the rows of C_t and waits until the current
    render frame is at least the render frame of C_t + 1 and obs_t's file has completed (its `done_render_frame` set
    and at or before the current frame; by then its pose row C + 1 exists). obs_t is often the image of C_t's own
    frame, which is rendered after the callback and completes in the next frame; an older obs_t completed earlier
    (17 of 52 policy-valid golden decisions, 22 of 100 at 10 Hz), so the wait compares the current frame, never the
    image's fixed completion frame, with C_t + 1. The frames go on, the physics does not; then the bridge sends
    state_t and obs_t and applies the returned command from step P_t + 1, so training and test see the same
    boundary.
- **Clock of the logged state (frame clock).** The logged pose follows world time, not the fixed-step clock: the pose
  in hf row n is the pose at the world time of frame n - 1 (run 3 E-06A: between consecutive hf rows the displacement
  equals speed x world dt of the frame, corr 1.0000 over 15,304 pairs, |difference| p95 0.1 ms; against the fixed
  steps run after `EOnFrame` the corr is 0.91), and the poses of the step rows follow world time too (e01 and e02a,
  corr 0.998-1.000). Time differences between rows are therefore `world_time_ms` differences; `covered_steps` x
  step_dt is the fixed-step clock, which lags. That velocities and accelerations, read in the same callback, follow
  the same clock is an inference [unverified; the C15 tolerances cover one step].
- **Images.** The slot grid runs in world time at `image_hz` (TBD-2: 5 Hz; 10 Hz only as the session setting
  `clocks.image_hz_setting` `trial_conf`), on the anchor of the decision grid: grid point g is due at anchor + g x 1000
  / `image_hz` - `clocks.img_lead_ms` (default 0; at `image_hz` = `decision_hz` grid point g is due `img_lead_ms` before
  decision grid point g). Each grid point reached while the ego is live and the camera current becomes the next
  `slot_id`. The request is issued in `EOnFrame` of the first frame whose world time is at or past the due time once the
  anchor exists: grid point 0 is due at the anchor itself, the world time of the first callback, which runs after its
  frame's `EOnFrame`, so slot 0 is issued in the next frame; every later grid point in the first frame at or past its
  due time (C08: jitter <= 1 render frame at p99). One request is in flight at a time: a slot that arrives while one is
  in flight is written `skipped_busy`, a request that `MakeScreenshot` refuses is written `make_failed`, and both count
  as missing (C10). A request whose file is not seen within `clocks.img_inflight_timeout_ms` (real time) stops blocking
  the next slot; it is logged as an `events` row `img_timeout` (payload `img_slot_id=<n>`) and in heartbeat `img_fail`.
  The request path is extension-less (R06). The img row of a slot is written once the slot resolves (its file seen,
  timed out, skipped, refused, or the session sealed), in slot order, with `done_render_frame` the render frame in which
  the writer first saw the file. A seal that does not drain (`ego_destroyed`, a Play stop, an error) writes the row of a
  request still in flight with `done_render_frame` empty (`images.pending_at_end`); the engine may still write its file.
- **Pairing.** The harvester decodes fiducial v2 (`render_frame mod 2^18`, drawn by the debug-Shape carrier) from each
  image and joins `hf` at that `captured_render_frame` C (C13a, exact). In Workbench 1.8 C = `img.done_render_frame`
  - 1 (run 4 E-06: 5999 of 5999 single-flight decodes); `done_render_frame` is the cross-check, the fiducial the truth.
  C37 measures each issued slot's capture stall at its decoded frame C (a dropped `done_render_frame` cannot remove a
  slow capture), refuses the evidence where `done_render_frame` contradicts C + 1, and assumes the worst case (u = 1)
  for an issued slot it cannot measure. Writer integrity (C13a): the issued rows with `done_render_frame` empty and
  no `img_timeout` event are exactly `images.pending_at_end` of them, the newest issued slots; any other such row is
  a completion the writer dropped.
  The image of render frame C shows the state logged in hf row C + 1 (run 3 E-06A: argmin k = +1 on 628 of 640 moving
  images, RMS 0.11-1.87 px against 2.6-5.6 px at k = 0): that row is the image's pose row. Its `cam_*` and `veh_*` are
  the pose at capture, its `last_physics_step` joins `ctl` and `telemetry` exactly (both have a row for every
  callback), and the pose belongs to the world time of hf row C. The one exemption from the exact joins is the request
  a seal without drain caught in flight: its row is issued with `done_render_frame` empty and no `img_timeout` event,
  it is counted in `images.pending_at_end`, its image shows the terminal render frame (C = `end.render_frame`, whose
  hf row exists) and only its pose row C + 1 is missing; it is not joined and its decision masks are 0, like a
  pending slot, never a join failure. A later frame cannot carry a decodable code (the seal releases the fiducial
  carrier in the same frame), so any other capture without its hf rows is a join failure (C13a). These joins do not
  prove the scene's age; E-06A (marker reprojection per rig x exe x build, at k = +1) and C13b do: C13b requires the
  best lag to be +1 whenever the yaw signal suffices.
- **Sampling at the model boundary.** Decision step t at P_t uses the samples of the decision boundary above; obs_t is
  the newest image whose pose row (hf row C + 1) has `last_physics_step` <= P_t and whose captured frame C is at or
  before the render frame of C_t, so its world time (hf row C) is at or before that of C_t: a frame without a callback
  keeps `last_physics_step` while world time advances, so the step bound alone would admit an image of a later frame
  (a future observation). `obs_age_ms` (`world_time_ms` of C_t minus that of hf row C) is therefore never negative;
  obs_t is dropped (masks 0) when it is older than one image period; nav comes from `drivedit_nav.nav_v1` at the
  pose of P_t. The builder writes the realised image age and the interval length of every decision step
  (`decision_index`, section 9.8).
- Sessions are at most 30 min (the fiducial code is unique for 73 min at 60 FPS).

## 7. Units, axes and sign convention

- **Units.** SI unless the unit column says otherwise: m, s, m/s, m/s2, rad, rad/s, 1/m; `world_time_ms` in integer
  ms; sun elevation, vFOV and latitude in degrees; rpm; FPS in 1/s.
- **World frame.** Enfusion metres, Y up: x east, y up (altitude), z north. Positions, transforms and route polylines
  are in this frame.
- **Vehicle-local frame.** x right, y up, z forward, as returned by `GetVelocityLocal`, `GetAngularVelocityLocal` and
  `GetAccelerationLocal` (checked by the E-01 analysis `local_axes`: z forward from the sign of the local forward
  velocity while the car advances, x right from the sign of the lateral acceleration in turns; TBD-5). Rows 0-2 of
  `GetWorldTransform` are the right, up and forward axes in world coordinates (`right_*`, `up_*`, `fwd_*`).
- **Heading.** `heading_rad = atan2(fwd_x, fwd_z)`: 0 = north (+z), increasing clockwise seen from above (towards
  +x, east), wrapped to (-pi, pi]. World-frame heading is eval-only.
- **Ego frame** (nav, lane errors, anchors). Origin at the ego entity origin, horizontal: x forward along the heading,
  y to the right. A point 20 m straight ahead is (20, 0); a point on the right is y > 0.
- **Right-positive rule** (unchanged from dc-1.0). Positive steering, positive curvature (`kappa_*`), positive yaw
  rate, positive lateral and heading errors and positive `p*_y` all mean right (clockwise seen from above).
- **Engine-native steering.** `llc_steer_*` and `input_*_steer` are logged exactly as passed to `SetSteering` or
  returned by `GetSteering`. On 1.8 a positive `SetSteering` turns right (TBD-5 decided, E-01 run 3), so
  `control.steer_sign` is +1; the field stays because a build or vehicle may differ. Export normalises: `steer_cmd =
  steer_sign * llc_steer_lim`, `steer_input = steer_sign * input_post_steer`, with `steer_sign` from `session.json`
  `control.steer_sign` (TBD-5: +1). `yaw_rate_rps = yaw_axis_sign * angvel_local_y` (TBD-5: +1). The vehicle card
  records both per build.
- **Steering channels** (MIGRATION_1_9 8.3, CRIT F-09). `steer_cmd` is the command in [-1, 1]; `steer_input` the input
  readback (`input_post`); `wheel_angle_l_rad` and `wheel_angle_r_rad` the applied road-wheel angles; `delta_eff_rad`
  their bicycle equivalent; `kappa_realised` = yaw rate / forward speed above 3 m/s. On 1.8 the wheel angles are
  unknown: the fields are reserved and empty, `control.steering_geometry` is `unknown_on_1.8`, and they become labels
  only after CC-07 step 0 validates a source. C15's `steer_eff` is `delta_eff` once it exists, else the identified lag
  of `input_post_steer`. The bicycle relation kappa = tan(delta_eff) / L is used only for abs(a_lat) <= 1 m/s2 [target].
- **Getters mean inputs.** `GetSteering`, `GetThrottle`, `GetBrake` return simulation inputs, not wheel angle or force
  (`VehicleWheeledSimulation.c:18-37`): they are `input_*`, never "applied".
- **Gear.** `gear` and `llc_gear` are engine indices; the vehicle card maps them to reverse, neutral and forward
  (TBD-7) for `dir_state`. The S105 has a manual gearbox with 6 gears, 4 forward: 0 reverse, 1 neutral, 2..5 first to
  fourth (E-04, E-01 run 3). The general rule is vanilla's gear computation (1.8.0.13
  `SCR_ToggleHandbrakeAction.c:100-108`, `SCR_VehicleGearCondition.c:32-44`): neutral = `GearboxGearsCount()` -
  `GearboxForwardGearsCount()` - 1, first = neutral + 1, top = `GearboxGearsCount()` - 1, reverse below neutral. The
  card names `gear_indices_forward`, `gear_index_neutral` and `gear_index_reverse`, so `dir_state` is +1 in a forward
  gear, 0 in neutral and -1 in reverse. T0-MIN stays forward-only: reverse is not driven yet, and no session commands
  `hl_dir` -1 before a probe phase drives it (DD-7-ident).
- **Sun.** `sun_elev_deg` comes from `GetCurrentSunMoonDirAndPhase`; the direction convention of `sun_dir` is checked
  at T0-MIN (a noon session gives a positive elevation).

## 8. CSV format and acknowledged writes (r2 3.9; R10)

- ASCII (a subset of UTF-8), LF line ends, `,` delimiter, no quoting. Every chunk file starts with the header row;
  column order is the field order of section 9. Readers parse by header name (C05) and refuse a missing required or an
  unknown column. Text fields never contain `,`, `"`, CR or LF.
- The empty string is null and is allowed only in fields with Req = no. Numbers are finite (no NaN or inf), with `.`
  as the decimal point and no exponent; at least 3 decimals for m, m/s and m/s2 and 6 for rad, rad/s, 1/m and unit
  vectors. Booleans are 0 or 1.
- **Chunks.** Rows are buffered per stream. A chunk is closed at the stream's `chunk_max_rows`, after 10 s of world
  time (`clocks.chunk_max_s`), or at session end, whichever comes first; a chunk is written only when it holds at
  least one row, so `first_key` and `last_key` are always defined. It is written whole, header plus rows, as
  `<stream>/chunk_NNNNN.csv` (5 digits, from 00000) by one `FileHandle.Write`, and closed. `FileHandle` has no flush
  and `FileIO` no rename, so a crash loses at most the chunk in memory.
- **Commit.** After the chunk is closed, one line `chunk_id,first_key,last_key,n_rows,n_bytes` is appended to
  `<stream>/commit.csv` (header written when the file is created) with one open-APPEND-write-close cycle. A row is
  **acknowledged only when its chunk's commit line was written successfully.**
- **Write checks.** Every `Write` return is compared with the byte length of the text (ASCII, so one byte per
  character). `WriteLine` returns void and is never used. No file is rewritten in place: chunk files, session.json
  and session_end.json are new files, and commit.csv and heartbeat.csv only grow by appended lines (so whether
  `FileMode.WRITE` truncates an existing file, E-19 c10, never matters). Semantics on failure, read-only targets,
  the append cycle and process kill: TBD-3 (E-19 c01-c11 and both e19kill commit logs).
- **Failure.** Any short or failed write (chunk, commit line, heartbeat row, `session.json`) stops the session: an
  `events` row `script_error` with payload `reason=write_fail` is attempted, `session_end.json` with status `failed`
  and `end_reason` `script_error:write_fail` is attempted, and no further rows are acknowledged. The session never
  claims success.
- **Readers** read committed chunks only and discard uncommitted tails. C34 re-reads every committed chunk and checks
  `n_rows` and `n_bytes`; a missing commit line is a negative fixture (DD-5-min).

Commit log (`<stream>/commit.csv`):

<!-- dc2:fields file=commit.csv -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `chunk_id` | int (key) | - | yes | Chunk number, 0-based, +1 per commit. |
| `first_key` | int | - | yes | Key of the first row in the chunk (first key column for composite keys). |
| `last_key` | int | - | yes | Key of the last row in the chunk. |
| `n_rows` | int | - | yes | Data rows in the chunk (header excluded). |
| `n_bytes` | int | B | yes | Bytes of the chunk file, header included. |

Heartbeat (`heartbeat.csv`: the header at session start, then one row appended every 5 s with one
open-APPEND-write-close cycle; readers and the supervisor take the last complete row):

<!-- dc2:fields file=heartbeat.csv -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `unix_s` | int | s | yes | Wall time. |
| `world_ms` | int | ms | yes | world_time_ms. |
| `physics_step` | int | step | yes | Current physics_step. |
| `render_frame` | int | frame | yes | Current render_frame. |
| `fps` | float | 1/s | yes | FPS. |
| `mem_kb` | int | KiB | yes | Script-visible memory use. |
| `img_ok` | int | - | yes | Requests whose file was seen (online check; the fiducial is the truth). |
| `img_fail` | int | - | yes | Requests whose file was not seen in time. |

## 9. Streams

Field order in each table is the CSV column order. Tags in brackets: `right+` right-positive, `engine-native` raw
engine sign, frame names, `build-scoped` (raw engine input, never pooled across `build_key`s), `reserved` (always
empty until the named step fills it; null on 1.8), `eval-only` (never a model input), and the TBD that decides a value.

### 9.1 Stream `ctl` (one row per fixed-frame callback)

<!-- dc2:fields stream=ctl -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | Engine fixed-step clock: the 0-based index of the last engine fixed step this callback covers, counted from the session open (the first row has covered_steps - 1); each row adds its own covered_steps (section 6). The row is written in EOnPostFixedFrame, after the covered steps. [TBD-1] |
| `covered_steps` | int | step | yes | Engine fixed steps this callback covers, >= 1: BaseWorld.GetFixedSteps() of its render frame (1 to 3 on 1.8); 1 per call when the engine delivers EOnSimulate (section 6). [TBD-1] |
| `render_frame` | int | frame | yes | GetFrameNumber() at the callback: the render frame whose fixed steps this row covers; its EOnFrame (hf row) precedes the callback (section 6). |
| `world_time_ms` | int | ms | yes | World time of the callback's render frame since the session start: WorldTimestamp.DiffMilliseconds against the session-start timestamp (R03), the frame clock of section 6. Non-decreasing. |
| `episode_id` | int | - | yes | Current episode; -1 outside an episode (warm-up, settle, between episodes). |
| `mode` | enum | - | yes | Capture mode of this callback (r2 3.1): E clean expert, N noise injection, R recovery start, X exploration, D DAgger, H human, A AI ego, I idle (no HL command held: warm-up, settle). Values: E, N, R, X, D, H, A, I. |
| `driver` | enum | - | yes | Who produced the executed HL command of this step (C18, C27). Values: expert, policy, human, ai, none. |
| `hl_seq` | int | - | yes | Counter of the HL decision applied in this row's covered steps: -1, the no-command sentinel (hl_* empty), until the first command; it advances by 1 exactly on the row after every decision callback of an episode that starts or goes on (the decision of C_t is taken after the rows of C_t and applies from physics step P_t + 1; a kept command is numbered again) and on no other row; a command changes only with an increment or at an episode boundary (section 6, TBD-1). [TBD-1] |
| `hl_kappa` | float | 1/m | no | Held HL curvature target applied in this callback's covered steps (the action_cmd.kappa_target of decision hl_seq, decided at the previous decision callback), right-positive. Empty in mode I. [right+] |
| `hl_accel` | float | m/s2 | no | Held HL acceleration target applied in this callback (action_cmd.a_target, desired). |
| `hl_dir` | int | - | no | Held HL direction command applied in this callback: -1 reverse, 0 stop, +1 forward. Values: -1, 0, 1. |
| `exp_kappa_now` | float | 1/m | no | Expert HL curvature re-evaluated on the true state after this callback (r2 3.2): a per-callback diagnostic; at a decision callback C_t it is the label of decision t, which in mode E is the command applied from P_t + 1. Empty when no expert is attached. [right+] |
| `exp_accel_now` | float | m/s2 | no | Expert HL acceleration re-evaluated after this callback. |
| `exp_dir_now` | int | - | no | Expert HL direction re-evaluated after this callback (added to r2's exp_{kappa,accel}_now so the label has all three action channels). Values: -1, 0, 1. |
| `llc_steer_raw` | float | 1 | no | llc_v1 steering before its rate limiter and lag, engine-native sign (contract sign: multiply by session.json control.steer_sign, TBD-5). Empty when llc_v1 is not driving. [engine-native, build-scoped, TBD-5] |
| `llc_steer_lim` | float | 1 | no | llc_v1 steering after the explicit rate limiter and first-order lag, as passed to SetSteering, engine-native sign. Range [-1, 1]. [engine-native, build-scoped, TBD-5] |
| `llc_throttle` | float | 1 | no | llc_v1 throttle as passed to SetThrottle; never non-zero together with llc_brake. Range [0, 1]. [build-scoped] |
| `llc_brake` | float | 1 | no | llc_v1 brake as passed to SetBreak. Range [0, 1]. [build-scoped] |
| `llc_handbrake` | bool | - | no | Handbrake flag passed to SetBreak. [build-scoped] |
| `llc_gear` | int | - | no | Gear index passed to SetGear (engine index; mapping in the vehicle card, TBD-7). Empty when no gear is set this step. [build-scoped, TBD-7] |
| `llc_clutch` | float | 1 | no | Clutch passed to SetClutch; empty unless a scripted gearbox sets it (TBD-7). Range [0, 1]. [build-scoped, TBD-7] |
| `input_pre_steer` | float | 1 | yes | GetSteering() read in EOnFixedFrame right after the setters, before the covered steps (an input readback, not a wheel angle; r2 3.2), engine-native sign. Range [-1, 1]. [engine-native, build-scoped, TBD-5] |
| `input_pre_throttle` | float | 1 | yes | GetThrottle() read in EOnFixedFrame right after the setters, before the covered steps (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_pre_brake` | float | 1 | yes | GetBrake() read in EOnFixedFrame right after the setters, before the covered steps (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_pre_clutch` | float | 1 | yes | GetClutch() read in EOnFixedFrame right after the setters, before the covered steps (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_pre_gear` | int | - | yes | GetGear() read in EOnFixedFrame right after the setters, before the covered steps (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `input_pre_handbrake` | bool | - | yes | IsHandbrakeOn() read in EOnFixedFrame right after the setters, before the covered steps (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `input_post_steer` | float | 1 | yes | GetSteering() read in EOnPostFixedFrame after the covered steps (an input readback, not a wheel angle; r2 3.2), engine-native sign. Range [-1, 1]. [engine-native, build-scoped, TBD-5] |
| `input_post_throttle` | float | 1 | yes | GetThrottle() read in EOnPostFixedFrame after the covered steps (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_post_brake` | float | 1 | yes | GetBrake() read in EOnPostFixedFrame after the covered steps (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_post_clutch` | float | 1 | yes | GetClutch() read in EOnPostFixedFrame after the covered steps (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_post_gear` | int | - | yes | GetGear() read in EOnPostFixedFrame after the covered steps (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `input_post_handbrake` | bool | - | yes | IsHandbrakeOn() read in EOnPostFixedFrame after the covered steps (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `noise_active` | bool | - | yes | 1 while an N-mode disturbance is applied to the executed command. |
| `noise_id` | int | - | yes | Disturbance id (events perturb_start/perturb_end); -1 when none. |
| `steer_signal_raw` | float | 1 | no | Reserved, post-GO (DD-4-probe (d)): the simulation's steering0 signal, a raw candidate for the applied steering. Empty until CC-07 step 0 validates it. [build-scoped, reserved] |
| `wheel_angle_l_rad` | float | rad | no | Reserved: applied left road-wheel angle, right-positive. Unknown on 1.8; filled only after CC-07 step 0 validates a source (MIGRATION_1_9 8.3). [right+, build-scoped, reserved] |
| `wheel_angle_r_rad` | float | rad | no | Reserved: applied right road-wheel angle, right-positive. Unknown on 1.8. [right+, build-scoped, reserved] |
| `react_veh_slot_id` | int | - | no | Reserved until DD-4/DD-7 (empty before): vehicle slot of the actor the expert reacts to in this step (perception-limited reactions, r2 3.1); empty when it reacts to no actor. [reserved] |
| `react_visible` | bool | - | no | Reserved until DD-4/DD-7: 1 when that actor passes the in-engine visibility test of r2 3.1 (camera FOV + 10 deg, raycast line of sight, or within 15 m), 0 when not; the builder sets policy_valid 0 while a reaction to an invisible actor lasts. [reserved] |

### 9.2 Stream `telemetry` (one row per fixed-frame callback, the keys of ctl)

<!-- dc2:fields stream=telemetry -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | physics_step of the callback, as in ctl: one telemetry row per callback (the same keys as ctl), the state after its covered steps, the pose on the frame clock (section 6). [TBD-1] |
| `covered_steps` | int | step | yes | As in ctl. [TBD-1] |
| `render_frame` | int | frame | yes | As in ctl. |
| `world_time_ms` | int | ms | yes | As in ctl. |
| `episode_id` | int | - | yes | As in ctl. |
| `ego_slot_id` | int | - | yes | Slot id of the ego (monotonic, never reused; r2 3.9). |
| `ego_entity_id` | str | - | yes | IEntity.GetID() of the ego, as an opaque ASCII token (C35). |
| `pos_x` | float | m | yes | Ego position (GetWorldTransform row 3; pos_y is altitude), x component. [world, eval-only] |
| `pos_y` | float | m | yes | Ego position (GetWorldTransform row 3; pos_y is altitude), y component. [world, eval-only] |
| `pos_z` | float | m | yes | Ego position (GetWorldTransform row 3; pos_y is altitude), z component. [world, eval-only] |
| `right_x` | float | 1 | yes | Ego right (+x local) unit axis, world x component (GetWorldTransform rows 0-2). [world, eval-only] |
| `right_y` | float | 1 | yes | Ego right (+x local) unit axis, world y component (GetWorldTransform rows 0-2). [world, eval-only] |
| `right_z` | float | 1 | yes | Ego right (+x local) unit axis, world z component (GetWorldTransform rows 0-2). [world, eval-only] |
| `up_x` | float | 1 | yes | Ego up (+y local) unit axis, world x component (GetWorldTransform rows 0-2). [world, eval-only] |
| `up_y` | float | 1 | yes | Ego up (+y local) unit axis, world y component (GetWorldTransform rows 0-2). [world, eval-only] |
| `up_z` | float | 1 | yes | Ego up (+y local) unit axis, world z component (GetWorldTransform rows 0-2). [world, eval-only] |
| `fwd_x` | float | 1 | yes | Ego forward (+z local) unit axis, world x component (GetWorldTransform rows 0-2). [world, eval-only] |
| `fwd_y` | float | 1 | yes | Ego forward (+z local) unit axis, world y component (GetWorldTransform rows 0-2). [world, eval-only] |
| `fwd_z` | float | 1 | yes | Ego forward (+z local) unit axis, world z component (GetWorldTransform rows 0-2). [world, eval-only] |
| `vel_local_x` | float | m/s | yes | Ego velocity in the vehicle-local frame (GetVelocityLocal; x right, y up, z forward, TBD-5), x component. [vehicle_local] |
| `vel_local_y` | float | m/s | yes | Ego velocity in the vehicle-local frame (GetVelocityLocal; x right, y up, z forward, TBD-5), y component. [vehicle_local] |
| `vel_local_z` | float | m/s | yes | Ego velocity in the vehicle-local frame (GetVelocityLocal; x right, y up, z forward, TBD-5), z component. [vehicle_local] |
| `angvel_local_x` | float | rad/s | yes | Ego angular velocity in the vehicle-local frame (GetAngularVelocityLocal), raw engine sign (yaw_rate_rps = yaw_axis_sign * angvel_local_y, TBD-5), x component. [vehicle_local] |
| `angvel_local_y` | float | rad/s | yes | Ego angular velocity in the vehicle-local frame (GetAngularVelocityLocal), raw engine sign (yaw_rate_rps = yaw_axis_sign * angvel_local_y, TBD-5), y component. [vehicle_local] |
| `angvel_local_z` | float | rad/s | yes | Ego angular velocity in the vehicle-local frame (GetAngularVelocityLocal), raw engine sign (yaw_rate_rps = yaw_axis_sign * angvel_local_y, TBD-5), z component. [vehicle_local] |
| `acc_local_x` | float | m/s2 | yes | Ego acceleration in the vehicle-local frame (GetAccelerationLocal), x component. [vehicle_local] |
| `acc_local_y` | float | m/s2 | yes | Ego acceleration in the vehicle-local frame (GetAccelerationLocal), y component. [vehicle_local] |
| `acc_local_z` | float | m/s2 | yes | Ego acceleration in the vehicle-local frame (GetAccelerationLocal), z component. [vehicle_local] |
| `speed_engine_mps` | float | m/s | yes | GetSpeedKmh() / 3.6 as reported by the engine (unsigned; the signed forward speed is telemetry_export.speed_fwd_mps, the registry channel speed_mps). |
| `dv_max_mps` | float | m/s | no | Largest dv_h of the ego (the collision detector's quantity of section 9.5.1: horizontal world-frame delta-v over events.collision_window_ms of world time, with the same interpolation and gap rule) over the callbacks since the previous telemetry row, which with one row per callback is this callback's dv_h, so the TBD-10 threshold can be re-applied offline per build_key; empty when it was not evaluated (history shorter than the window, a gap, a reset). [eval-only, build-scoped, TBD-10] |
| `gear` | int | - | yes | GetGear() (engine index, TBD-7). [TBD-7] |
| `rpm` | float | rpm | yes | EngineGetRPM(). |
| `engine_on` | bool | - | yes | EngineIsOn(). |
| `handbrake` | bool | - | yes | IsHandbrakeOn(). |
| `ground_contact` | bool | - | yes | HasAnyGroundContact(). |
| `wheel_count` | int | - | yes | WheelCount(); the w0..w3 columns hold the first four wheels. |
| `w0_contact` | bool | - | no | Wheel 0 (engine index; its axle and side come from the vehicle card, built from the E-04 wheel names: 1.8 exposes only WheelGetName) has ground contact (WheelHasContact). Empty when i >= wheel_count. |
| `w0_slip_long` | float | 1 | no | Wheel 0 longitudinal slip (WheelGetLongitudinalSlip), raw engine value. [build-scoped] |
| `w0_slip_lat` | float | 1 | no | Wheel 0 lateral slip (WheelGetLateralSlip), raw engine value. [build-scoped] |
| `w0_material` | str | - | no | Wheel 0 contact material name (WheelGetContactMaterial); empty without contact or without a 1.8 name getter. |
| `w0_temp_c` | float | degC | no | Reserved, null on 1.8: wheel 0 tyre temperature (MIGRATION_1_9 8.3). [reserved] |
| `w1_contact` | bool | - | no | Wheel 1 (engine index; its axle and side come from the vehicle card, built from the E-04 wheel names: 1.8 exposes only WheelGetName) has ground contact (WheelHasContact). Empty when i >= wheel_count. |
| `w1_slip_long` | float | 1 | no | Wheel 1 longitudinal slip (WheelGetLongitudinalSlip), raw engine value. [build-scoped] |
| `w1_slip_lat` | float | 1 | no | Wheel 1 lateral slip (WheelGetLateralSlip), raw engine value. [build-scoped] |
| `w1_material` | str | - | no | Wheel 1 contact material name (WheelGetContactMaterial); empty without contact or without a 1.8 name getter. |
| `w1_temp_c` | float | degC | no | Reserved, null on 1.8: wheel 1 tyre temperature (MIGRATION_1_9 8.3). [reserved] |
| `w2_contact` | bool | - | no | Wheel 2 (engine index; its axle and side come from the vehicle card, built from the E-04 wheel names: 1.8 exposes only WheelGetName) has ground contact (WheelHasContact). Empty when i >= wheel_count. |
| `w2_slip_long` | float | 1 | no | Wheel 2 longitudinal slip (WheelGetLongitudinalSlip), raw engine value. [build-scoped] |
| `w2_slip_lat` | float | 1 | no | Wheel 2 lateral slip (WheelGetLateralSlip), raw engine value. [build-scoped] |
| `w2_material` | str | - | no | Wheel 2 contact material name (WheelGetContactMaterial); empty without contact or without a 1.8 name getter. |
| `w2_temp_c` | float | degC | no | Reserved, null on 1.8: wheel 2 tyre temperature (MIGRATION_1_9 8.3). [reserved] |
| `w3_contact` | bool | - | no | Wheel 3 (engine index; its axle and side come from the vehicle card, built from the E-04 wheel names: 1.8 exposes only WheelGetName) has ground contact (WheelHasContact). Empty when i >= wheel_count. |
| `w3_slip_long` | float | 1 | no | Wheel 3 longitudinal slip (WheelGetLongitudinalSlip), raw engine value. [build-scoped] |
| `w3_slip_lat` | float | 1 | no | Wheel 3 lateral slip (WheelGetLateralSlip), raw engine value. [build-scoped] |
| `w3_material` | str | - | no | Wheel 3 contact material name (WheelGetContactMaterial); empty without contact or without a 1.8 name getter. |
| `w3_temp_c` | float | degC | no | Reserved, null on 1.8: wheel 3 tyre temperature (MIGRATION_1_9 8.3). [reserved] |
| `road_key` | str | - | no | Key of the closest engine road, <n_pts>/<x0>:<z0>/<x1>:<z1> exactly as the E-08a dump writes it; valid within the session only: a join across sessions matches the unordered endpoint pair within 2 m one-to-one, never this text (TBD-9, section 9.8 road); empty when no road is found. [TBD-9] |
| `road_dist_m` | float | m | no | Distance to the closest road (RoadNetworkManager.GetClosestRoad). |
| `road_width_m` | float | m | no | BaseRoad.GetWidth() of the closest road. |
| `lane_lat_err_m` | float | m | no | Lane-relative signed lateral error: ego offset from the active route's driving line, positive = ego right of the line. Computed by the driver interface; empty without an active route. [right+] |
| `lane_head_err_rad` | float | rad | no | Lane-relative heading error: ego heading minus route tangent heading, wrapped to (-pi, pi], positive = ego pointing right of the route. [right+] |
| `route_progress_m` | float | m | no | Ordered progress along the active route as tracked in-engine (nav_v1 rule); empty without a route. The harvester recomputes it offline with drivedit_nav. [eval-only] |
| `env_weather_state` | str | - | yes | Current weather state name (names from WeatherStates.conf). |
| `env_rain` | float | 1 | yes | GetRainIntensity(). |
| `env_fog` | float | 1 | yes | GetFogAmount(). |
| `env_fog_height_density` | float | 1 | yes | GetFogHeightDensity(). |
| `env_wetness` | float | 1 | yes | GetCurrentWetness(). |
| `env_tod_h` | float | h | yes | Time of day in hours [0, 24). |
| `sun_dir_x` | float | 1 | yes | Raw sun direction from GetCurrentSunMoonDirAndPhase, x component. [world] |
| `sun_dir_y` | float | 1 | yes | Raw sun direction from GetCurrentSunMoonDirAndPhase, y component. [world] |
| `sun_dir_z` | float | 1 | yes | Raw sun direction from GetCurrentSunMoonDirAndPhase, z component. [world] |
| `moon_phase01` | float | 1 | yes | Moon phase (0 new, 1 full) from GetCurrentSunMoonDirAndPhase. |
| `sun_elev_deg` | float | deg | yes | Sun elevation above the horizon from sun_dir (the sign convention of sun_dir is checked at T0-MIN: a noon session must give a positive elevation). |
| `traffic_n` | int | - | yes | Number of live traffic vehicles (0 until T2). |
| `traffic_slots` | str | - | no | Traffic vehicles as 'veh_slot_id:entity_id' pairs joined by ';' (no commas); empty when traffic_n = 0. |

### 9.3 Stream `hf` (every render frame)

<!-- dc2:fields stream=hf -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `render_frame` | int (key) | frame | yes | BaseWorld.GetFrameNumber() of this render frame; written every frame in EOnFrame, so +1 per row. |
| `world_time_ms` | int | ms | yes | As in ctl (the frame clock); equal in consecutive rows only while paused or within one ms. |
| `last_physics_step` | int | step | yes | physics_step of the newest fixed-frame callback of an earlier render frame (a frame's callbacks run after its EOnFrame), so exactly the newest ctl key with render_frame below this row's; -1 before the first. Joins ctl and telemetry exactly; the image of render frame C is paired with hf row C + 1 (section 6). |
| `paused` | bool | - | yes | 1 while the game time is paused. |
| `cam_pos_x` | float | m | yes | Camera position (from the camera entity's world transform), x component. [world, eval-only] |
| `cam_pos_y` | float | m | yes | Camera position (from the camera entity's world transform), y component. [world, eval-only] |
| `cam_pos_z` | float | m | yes | Camera position (from the camera entity's world transform), z component. [world, eval-only] |
| `cam_right_x` | float | 1 | yes | Camera right (+x local) unit axis, world x component (camera entity world transform). [world, eval-only] |
| `cam_right_y` | float | 1 | yes | Camera right (+x local) unit axis, world y component (camera entity world transform). [world, eval-only] |
| `cam_right_z` | float | 1 | yes | Camera right (+x local) unit axis, world z component (camera entity world transform). [world, eval-only] |
| `cam_up_x` | float | 1 | yes | Camera up (+y local) unit axis, world x component (camera entity world transform). [world, eval-only] |
| `cam_up_y` | float | 1 | yes | Camera up (+y local) unit axis, world y component (camera entity world transform). [world, eval-only] |
| `cam_up_z` | float | 1 | yes | Camera up (+y local) unit axis, world z component (camera entity world transform). [world, eval-only] |
| `cam_fwd_x` | float | 1 | yes | Camera forward (+z local) unit axis, world x component (camera entity world transform). [world, eval-only] |
| `cam_fwd_y` | float | 1 | yes | Camera forward (+z local) unit axis, world y component (camera entity world transform). [world, eval-only] |
| `cam_fwd_z` | float | 1 | yes | Camera forward (+z local) unit axis, world z component (camera entity world transform). [world, eval-only] |
| `veh_pos_x` | float | m | yes | Ego position at this frame, x component. [world, eval-only] |
| `veh_pos_y` | float | m | yes | Ego position at this frame, y component. [world, eval-only] |
| `veh_pos_z` | float | m | yes | Ego position at this frame, z component. [world, eval-only] |
| `veh_right_x` | float | 1 | yes | Ego right (+x local) unit axis, world x component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_right_y` | float | 1 | yes | Ego right (+x local) unit axis, world y component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_right_z` | float | 1 | yes | Ego right (+x local) unit axis, world z component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_up_x` | float | 1 | yes | Ego up (+y local) unit axis, world x component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_up_y` | float | 1 | yes | Ego up (+y local) unit axis, world y component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_up_z` | float | 1 | yes | Ego up (+y local) unit axis, world z component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_fwd_x` | float | 1 | yes | Ego forward (+z local) unit axis, world x component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_fwd_y` | float | 1 | yes | Ego forward (+z local) unit axis, world y component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `veh_fwd_z` | float | 1 | yes | Ego forward (+z local) unit axis, world z component (GetWorldTransform rows 0-2 at this frame). [world, eval-only] |
| `fps` | float | 1/s | yes | Frames per second as reported by the engine at this frame. |

### 9.4 Stream `img` (image slots)

<!-- dc2:fields stream=img -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `slot_id` | int (key) | slot | yes | Image slot, 0-based, +1 per row. Slots advance only while the ego is live and the camera current. |
| `episode_id` | int | - | yes | As in ctl. |
| `ego_slot_id` | int | - | yes | As in telemetry. |
| `requested_world_ms` | int | ms | yes | Due time of the slot's grid point: anchor + g x 1000 / image_hz - clocks.img_lead_ms, anchor = world_time_ms of the first ctl row (section 6, TBD-2). [TBD-2] |
| `issue_status` | enum | - | yes | issued; skipped_busy when the previous request was still in flight and younger than clocks.img_inflight_timeout_ms at the slot; make_failed when System.MakeScreenshot returned false (the issue fields are kept). Skipped and failed slots count as missing in C10 (TBD-2). Values: issued, skipped_busy, make_failed. [TBD-2] |
| `issue_world_ms` | int | ms | no | World time of the frame the request was issued in (the first frame at or after the slot). Empty when skipped. |
| `issue_render_frame` | int | frame | no | render_frame of that frame. Empty when skipped. |
| `issue_physics_step` | int | step | no | last_physics_step at issue. Empty when skipped. |
| `path` | str | - | no | Request path relative to the session directory, without extension: frames/f_<slot_id:08d>; the engine writes frames/f_<slot_id:08d>.png (TBD-4). Empty when skipped. [TBD-4] |
| `cam_current` | bool | - | yes | 1 when the rig camera is the current camera at issue (C18). |
| `cam_attached` | bool | - | yes | 1 when the rig camera is attached to the ego at issue (C18). |
| `done_render_frame` | int | frame | no | render_frame in which the writer first saw the engine-written file (the request resolved); empty when the slot was not issued, timed out or was still pending at the seal. In Workbench 1.8 the captured frame is done_render_frame - 1 (run 4 E-06: 5999 of 5999); the fiducial stays the truth (section 6). |

### 9.5 Stream `events`

Event kinds: `session_start`, `warmup_end`, `episode_start`, `settle_end`, `episode_end`, `perturb_start`,
`perturb_end`, `nav_command`, `style_set`, `teleport`, `respawn`, `env_change`, `pause`, `resume`, `stuck`,
`collision`, the crash events `veh_damage`, `veh_fire_start`, `veh_fire_end`, `driver_state`, `dismount`, `remount`
and `rollover` (section 9.5.1, which also defines the `collision`, `stuck` and `teleport` payloads), `camera_reset`,
`img_timeout` (an image request in flight longer than `clocks.img_inflight_timeout_ms`; payload `img_slot_id=<n>`),
`dc_stop`, `script_error`, and `veh_cfg_override` (reserved, null on 1.8; a 1.9 `Update*Configuration` call must be
logged with this kind).

<!-- dc2:fields stream=events -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `event_seq` | int (key) | - | yes | Event counter, 0-based, +1 per row. |
| `episode_id` | int | - | yes | Episode the event belongs to; -1 outside episodes. |
| `physics_step` | int | step | yes | physics_step of the last completed callback when the event was recorded. |
| `render_frame` | int | frame | yes | render_frame when the event was recorded. |
| `world_time_ms` | int | ms | yes | As in ctl. |
| `kind` | enum | - | yes | Event kind (veh_cfg_override is reserved, null on 1.8; the crash events veh_damage, veh_fire_start, veh_fire_end, driver_state, dismount, remount and rollover are section 9.5.1). Values: session_start, warmup_end, episode_start, settle_end, episode_end, perturb_start, perturb_end, nav_command, style_set, teleport, respawn, env_change, pause, resume, stuck, collision, veh_damage, veh_fire_start, veh_fire_end, driver_state, dismount, remount, rollover, camera_reset, img_timeout, dc_stop, script_error, veh_cfg_override. |
| `veh_slot_id` | int | - | yes | Subject vehicle slot (ego_slot_id or a traffic slot), -1 when none; named veh_slot_id so a join by column name never meets img.slot_id. |
| `payload` | str | - | no | Kind-specific 'key=value' pairs joined by ';' (ASCII, no ',' or '='-in-values): termination reasons, perturbation parameters (noise_id, kind, amplitude, duration_s), nav commands (route_id, maneuver, s_m, stop s_m), style_v1 values, script_error reason, img_timeout img_slot_id, and the keys of section 9.5.1 (event_payloads) for collision, veh_damage, veh_fire_start, driver_state, dismount, remount, rollover, stuck and teleport. |

### 9.5.1 Crash events

Collisions, vehicle damage, fire, the driver's life state, the pilot seat and rollovers are discrete state changes,
so they are `events` rows, not telemetry columns: their value at any step is the last event of the kind for the slot.
Every crash row names its subject vehicle in `veh_slot_id` (>= 0), the ego or a traffic slot; a vehicle-vehicle
collision gives two `collision` rows, one per slot, each naming the other in `other_slot`. The writer is DD-4-min's
`DC_CrashWatch` (a capture module); its values are session parameters in `session.json` `events.*` (TBD-10).

Payload rules, in addition to 9.5: a key without a value is left out (no empty values); floats have 2 decimals;
speeds are horizontal, in m/s; prefab paths are written without their `{GUID}` prefix; no value holds `,`, `=` or
`;`. The keys of each kind are machine-readable in the streams JSON `event_payloads`; C36 checks them (an unknown or
missing required key fails).

<!-- dc2:payloads -->
| Kind | Key | Type | Unit | Req | Description |
|---|---|---|---|---|---|
| `collision` | `onset_step` | int | step | yes | physics_step of the callback at which the detector first crossed events.collision_dv_mps (src dv or dv_contact); for src contact the last completed physics_step when ProcessCollision reported the contact. It lies in the row's episode (episode_id) and never after the row's physics_step. |
| `collision` | `dv_mps` | float | m/s | yes | Horizontal world-frame delta-v: the peak of the norm of v_h(t) - v_ref over events.collision_peak_ms after the onset, v_ref the interpolated velocity at onset - events.collision_window_ms; >= events.collision_dv_mps unless src is contact; for src contact the vanilla momentum change of the vehicle over its mass (SCR_CollisionDamageContainer.m_fTotalMomentumOwner / Physics.GetMass). At least 0. |
| `collision` | `dv_lon_mps` | float | m/s | no | That delta-v along the horizontal forward axis of the vehicle at the onset (local z, forward-positive: a frontal hit is negative); empty for src contact. |
| `collision` | `dv_lat_mps` | float | m/s | no | That delta-v along the horizontal right axis at the onset (local x, right-positive); empty for src contact. [right+] |
| `collision` | `v_mps` | float | m/s | yes | Horizontal speed of v_ref (collision_window_ms before the onset); for src contact the speed before the contact (SCR_CollisionDamageContainer.GetOwnerVelocityBefore). At least 0. |
| `collision` | `other` | enum | - | yes | What was hit, classified in the order of section 9.5.1 from the vanilla contact (src dv_contact or contact) or from a registered vehicle with an onset within 3 steps and 10 m; unknown without either. Values: vehicle, character, tree, building, rock, terrain, static, dynamic, unknown. |
| `collision` | `other_slot` | int | - | no | veh_slot_id of the other vehicle when it is a registered slot (C35). |
| `collision` | `other_prefab` | str | - | no | Prefab of the other (root) entity when it is not a registered vehicle, without its {GUID} prefix. |
| `collision` | `impulse_ns` | float | Ns | no | Largest vanilla contact impulse of the matched contacts (src dv_contact or contact). At least 0. |
| `collision` | `src` | enum | - | yes | dv: the delta-v detector alone; dv_contact: the detector and a vanilla contact; contact: a vanilla contact without a detector onset (a hit below the threshold, never terminal). Values: dv, dv_contact, contact. |
| `veh_damage` | `state` | enum | - | yes | Damage state of the vehicle (DamageManagerComponent.GetState; the EDamageState values STATE1 to STATE3 are written intermediary). Values: undamaged, intermediary, destroyed. |
| `veh_damage` | `health` | float | 1 | yes | GetHealthScaled() of the vehicle. Range [0, 1]. |
| `veh_damage` | `movement_damage` | float | 1 | yes | GetMovementDamage() of the vehicle; 1.0 is the AI's cannot-move limit. Range [0, 1]. |
| `veh_fire_start` | `fire_state` | enum | - | yes | Highest SCR_EBurningState over the flammable hit zones of the vehicle (SCR_FlammableHitZone.GetFireState) when the first of them reaches the prefab's minimum burning state (SCR_DamageManagerComponent.IsOnFire). Values: smoking_light, smoking_heavy, smoking_igniting, burning. |
| `veh_fire_end` | - | - | - | - | No payload (empty). |
| `driver_state` | `state` | enum | - | yes | Life state of the slot's driver (CharacterControllerComponent.GetLifeState): the pilot-seat occupant, or the last one after the seat emptied. Values: alive, incapacitated, dead. |
| `driver_state` | `in_seat` | bool | - | yes | 1 while that driver occupies the pilot seat. |
| `dismount` | `driver` | enum | - | yes | Life state of the driver who left the pilot seat. Values: alive, incapacitated, dead. |
| `dismount` | `v_mps` | float | m/s | yes | Horizontal speed of the vehicle at the dismount. |
| `remount` | `driver` | enum | - | yes | Life state of the new pilot-seat occupant. Values: alive, incapacitated, dead. |
| `remount` | `by` | enum | - | no | Who seated the driver, when the writer knows it: ai (the vanilla AI got in), reseat (a director teleported the same character back), spawn (a director seated a new character). Values: ai, reseat, spawn. |
| `rollover` | `onset_step` | int | step | yes | First physics_step of the hold: the first callback with up_y below events.rollover_up_y; never after the row's physics_step. The row is written once the hold of events.rollover_hold_ms is confirmed and belongs to the episode of this step, even when that episode has ended meanwhile. |
| `rollover` | `up_y` | float | 1 | yes | up_y of the vehicle (GetWorldTransform row 1, y) when the hold was confirmed. Range [-1, 1]. |
| `stuck` | `stuck_s` | float | s | no | How long the vehicle had been stuck when the expert or a director reported it. At least 0. |
| `stuck` | `termination_reason` | enum | - | no | Present when the stuck ends the episode (the expert's stuck termination). Values: stuck. |
| `teleport` | `reason` | str | - | yes | Why the slot was moved by a scripted SetWorldTransform (for example episode_reset, stuck_reset, probe_reset). |

Collision detector (every registered slot, in every fixed-frame callback: `EOnPostFixedFrame`, or `EOnPostSimulate`
when the engine delivers it). Rows are frame-bound (dc-2.0 freeze decision D-A): one callback covers 1 to 3 merged
physics steps of 1/30 s on 1.8, so every window below is world time, never a count of steps or callbacks.
- Each callback stores a sample `(world_time_ms, v_h)` per slot, `v_h` the horizontal (world x, z) part of
  `Physics.GetVelocity()`. The world frame, never the local one: a spin turns the local velocity vector with no force
  acting.
- `dv_h(t) = |v_h(t) - v_ref|`, where `v_ref` is the velocity at `t - events.collision_window_ms`, linearly
  interpolated between the newest sample at or before that instant and the sample after it. When these two samples
  are more than `events.collision_window_ms + events.collision_window_tol_ms` apart (a frame hitch; one callback of 3
  merged steps spans only 100 ms), the slot is not evaluated and its history restarts at the current sample; a
  history that does not reach back to that instant is not evaluated either. `telemetry.dv_max_mps` uses exactly this
  quantity.
- Onset: the first evaluated callback with `dv_h >= events.collision_dv_mps` while the slot is neither in refractory
  nor suppressed; `onset_step` is its `physics_step`. `dv_mps` is the peak of `|v_h(t) - v_ref|` (the onset's
  `v_ref`) over `events.collision_peak_ms` of world time after the onset; further crossings within
  `events.collision_refractory_ms` belong to the same impact. Suppressed for 1000 ms after a slot is registered and
  for 500 ms after a reposition of it.
- Repositioning: whoever moves a slot by `SetWorldTransform` (a `teleport` or `respawn`) tells the core first
  (DD-4-min `DC_CrashWatch.PrepareReposition`, `DC_DriverContext.PrepareReposition` for the ego): the pending
  collision row is written with its peak so far, the held contacts are written, the velocity history is cleared and
  the 500 ms suppression starts before the move; the `teleport` or `respawn` row then clears the history again.
- The row is written later than its onset (`physics_step >= onset_step`): once a contact matched it and the peak
  window passed, else once `events.episode_end_wait_ms` and 2 render frames have passed since the onset (the vanilla
  contact arrives 1 to 2 render frames after its physics step; below about 8 frames per second it can miss the wait,
  and the row is then written with `src` `dv`).
- Contact source (`events.contact_source` `vanilla_process_collision`): vanilla's
  `SCR_VehicleDamageManagerComponent.ProcessCollision`, called once per new (vehicle, other entity) contact after
  vanilla's impulse filters and its continued-contact filter, forwarded by a modded class that changes nothing else.
  Each contact keeps the `physics_step`, episode and world time at which it was reported. Contacts whose world time
  lies in [onset - 100 ms, onset + 250 ms] are kept until the row is written; then the largest impulse names `other`,
  the contacts with that other entity merge into the row (`src` `dv_contact`), and every contact with a different
  other entity becomes its own row (`src` `contact`). A contact without an onset becomes its own row unless the
  slot's last row, within the refractory, has the same other entity (or none). With `none`, no contact rows exist and
  `other` comes from the registry match only.
- `other`, tested on the contact's other entity and then on each of its parents up to the root, in this order at
  each level: a registered slot (`vehicle` with `other_slot`), `Vehicle` (`vehicle`), `ChimeraCharacter`
  (`character`), `BaseTree` (`tree`, bushes included), `Building` (`building`, `SCR_DestructibleBuildingEntity`
  included), `SCR_IndestructibleEnvironmentalEntity` (`rock`: granite rocks, breakwaters), `GenericTerrainEntity`
  (`terrain`; vanilla's water entry is a terrain contact); when no level matches, `dynamic` or `static` by the other
  entity's `Physics.IsDynamic()`, and `unknown` without a contact or physics. Without a contact, another slot with an
  onset within 3 steps whose origin is within 10 m gives `vehicle` and `other_slot`.

Episode ownership: a crash row belongs to the episode in which it occurred, `episode_id` is that episode (-1 between
episodes) even when the row is written after that episode's `episode_end`: the onset for a `collision` (its
`onset_step`, or the step at which a contact was reported), the first tilted step for a `rollover` (its
`onset_step`), the poll for the other kinds. `event_seq` and `world_time_ms` stay in write order.
- Steps and episodes are taken as one pair. Whatever happens inside a physics callback (an onset, a tilt, a poll in
  that callback) takes that callback's step and the episode of its ctl row. A report that arrives between callbacks
  (a vanilla contact, a destruction or damage poll at a frame start or after an episode end) takes the last
  completed step and the episode that owns that step (its ctl `episode_id`), even when that episode has just ended:
  a contact reported after `episode_end` of an episode that ended at step P carries `onset_step` P and that episode.
- A crash row whose onset lies outside its episode (a -1 row inside any episode), or whose `onset_step` lies after
  its own `physics_step`, fails C36 and is never a terminal event.

Polled state (every slot, every `events.state_poll_ms` of world time, so its timing is +- one period):
- `veh_damage`: the damage state changes, or `movement_damage` crosses 1.0 in either direction;
- `veh_fire_start` / `veh_fire_end`: the start and the end of the interval in which any flammable hit zone is on fire
  (at or above the prefab's minimum burning state);
- `driver_state`: the life state of the slot's driver changes. The driver is the pilot-seat occupant; after the seat
  empties the slot keeps the last one until someone else sits in, so a death after an ejection is still seen;
- `dismount`: the pilot seat goes from occupied to empty, whatever the driver's state; `remount`: it is occupied again
  after a dismount. The first occupancy of a seat that was empty at registration writes no row (a Mode A ego has no
  driver, so these kinds and `driver_state` never occur for it);
- at registration the default is undamaged, not burning, and a driver (when seated) alive and in the seat; the first
  poll writes a row only where the vehicle differs from that default.

`rollover`: up_y below `events.rollover_up_y` for `events.rollover_hold_ms` of world time; the row is written when the
hold is confirmed and carries the first tilted step as `onset_step`. Re-armed once up_y has stayed above 0.8 for 2 s.
The confirmed rollover row is the only source of the termination `rollover`: a tilt that rights itself before the hold
ends ends nothing.

Pairs: per slot `veh_fire_start` and `veh_fire_end` alternate, start first, and so do `dismount` and `remount`; the
last of a slot may stay open at the session end (C36). Payload numbers lie in the `range` and at or above the `min`
of their key (event_payloads; C36).

Ego crashes (9.6). Terminal ego events: a `collision` with `src` `dv` or `dv_contact`, a `veh_damage` with a state
other than `undamaged` or `movement_damage` >= 1, a `veh_fire_start`, a `rollover`; they give `collision`, except
`rollover`, which gives `rollover`. Each has an onset (`onset_step` of a collision or rollover, else its
`physics_step`) and a confirmation (when the core can latch it: the onset of a collision, the end of the hold of a
rollover, the poll of the others).
- The capture core is the only source of the terminations `collision` and `rollover`: a driver never reports them
  (an expert that sees `up_y` below a limit must not end the episode itself). The core latches the reason in the
  callback of the confirmation, before the driver's own termination of that callback, and only into the event's own
  episode; the episode then ends at the next decision step. That end waits until the ego's pending `collision` row is
  written, at most `events.episode_end_wait_ms` after its onset, so the row carries the episode id and precedes
  `episode_end`.
- So the `termination_reason` `collision` or `rollover` of an episode must be the reason of its first terminal ego
  event by confirmation, confirmed inside the episode (C36).
- No new episode begins while the ego's `collision` row is pending, within `events.crash_cooldown_ms` after its last
  terminal crash event (in or between episodes; why 3000 ms: section 10.1), while it is tilted (up_y below
  `events.rollover_up_y`), immobile (`movement_damage` >= 1) or burning. The last three can last forever without a
  destruction: after `events.episode_block_max_ms` (60 s) of them between episodes the session ends with
  `ego_disabled` (5.3).
- A destroyed ego ends the session with `ego_destroyed` (5.3).
- The builder sets `policy_valid` 0 for every decision step whose decision interval (P_t, P_t+1] contains or
  follows the earliest onset of the episode's terminal ego events; `wm_valid` is not changed, because crash dynamics are
  world-model data. Its `teleport` mask uses only the `teleport` and `respawn` rows of the ego slot and of no vehicle
  (`veh_slot_id` -1): a traffic reset never masks ego samples.

### 9.6 Stream `episodes` (one row per episode, written at its end)

<!-- dc2:fields stream=episodes -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `episode_id` | int (key) | - | yes | Episode id, 0-based, +1 per episode; one row written at episode end. |
| `start_physics_step` | int | step | yes | physics_step of the episode's first ctl row (a decision callback after the settle). |
| `end_physics_step` | int | step | yes | physics_step of the episode's last ctl row. |
| `start_world_ms` | int | ms | yes | world_time_ms at start. |
| `end_world_ms` | int | ms | yes | world_time_ms at end. |
| `ego_slot_id` | int | - | yes | Ego slot, constant within the episode (C35). |
| `ego_entity_id` | str | - | yes | Ego IEntity.GetID() token. |
| `ego_prefab` | str | - | yes | Ego prefab resource name. |
| `vehicle_card_id` | str | - | yes | Vehicle card used by llc_v1 (keyed by build_key, DD-7-min). |
| `route_id` | str | - | no | Route followed (drivedit_nav route schema); empty in X/A modes. |
| `route_sha256` | str | - | no | sha256 of the route's canonical JSON (drivedit_nav.Route.sha256); may be filled by the harvester. |
| `mode` | enum | - | yes | Episode mode (r2 3.1). Values: E, N, R, X, D, H, A. |
| `driver` | enum | - | yes | Episode driver. Values: expert, policy, human, ai, none. |
| `style_speed_factor` | float | 1 | yes | style_v1 speed factor (nominal 1.0). |
| `style_lane_offset_m` | float | m | yes | style_v1 lane offset, right-positive (nominal 0.0). [right+] |
| `style_alat_max_mps2` | float | m/s2 | yes | style_v1 a_lat,max (nominal 2.75). |
| `env_weather_state` | str | - | yes | Weather state at the episode start (fixed within the episode). |
| `env_tod_h_start` | float | h | yes | Time of day at the episode start. |
| `env_rain` | float | 1 | yes | Rain at the episode start. |
| `env_fog` | float | 1 | yes | Fog at the episode start. |
| `env_wetness` | float | 1 | yes | Wetness at the episode start. |
| `seed` | int | - | yes | Episode seed. |
| `termination_reason` | enum | - | yes | First terminal event (r2 1.3); route_complete on success. Values: route_complete, timeout, offroad, route_deviation, stuck, rollover, collision, engine_off, script_error, session_stop. |
| `expert_sha` | str | - | no | Expert source hash; empty without an expert. |
| `llc_sha` | str | - | no | llc_v1 source hash; empty when llc_v1 did not drive. |
| `nav_version` | str | - | yes | nav_v1. |
| `noise_steer_sensitivity` | float | 1 | yes | GetNoiseSteerSensitivity() (engine default, logged; X6). |
| `roughness_sensitivity` | float | 1 | yes | GetRoughnessSensitivity() (engine default, logged; X6). |
| `veh_dirt_body` | float | 1 | no | Reserved, null on 1.8 (MIGRATION_1_9 8.3). [reserved] |
| `veh_dirt_glass` | float | 1 | no | Reserved, null on 1.8 (MIGRATION_1_9 8.3). [reserved] |

Crash terminations: `collision` and `rollover` come from the first terminal ego crash event of the episode as
defined in 9.5.1 (a collision row with `src` `dv` or `dv_contact`, a `veh_damage` with a state other than `undamaged`
or `movement_damage` >= 1, a `veh_fire_start`; a `rollover`), the first by confirmation: the core latches the reason
when it confirms the event and is their only source (a driver never reports `collision` or `rollover`). A termination
the driver reported for an earlier event of another kind wins (first terminal event, r2 1.3). No new reason value
exists for crashes; a destroyed ego still ends its episode with `collision`.

### 9.7 Stream `traffic` (reserved for T2)

Written from DD-4 on, at the telemetry cadence, one row per live traffic vehicle; key (`physics_step`,
`veh_slot_id`).
Absent in T0-MIN to T1 sessions; readers accept its absence.

<!-- dc2:fields stream=traffic -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | physics_step of the callback (the telemetry rows). |
| `veh_slot_id` | int (key) | - | yes | Traffic vehicle slot (never reused). |
| `entity_id` | str | - | yes | IEntity.GetID() token, constant per slot (C35). |
| `prefab` | str | - | yes | Prefab resource name. |
| `pos_x` | float | m | yes | Vehicle position, x component. [world, eval-only] |
| `pos_y` | float | m | yes | Vehicle position, y component. [world, eval-only] |
| `pos_z` | float | m | yes | Vehicle position, z component. [world, eval-only] |
| `fwd_x` | float | 1 | yes | Vehicle forward axis, x component. [world, eval-only] |
| `fwd_y` | float | 1 | yes | Vehicle forward axis, y component. [world, eval-only] |
| `fwd_z` | float | 1 | yes | Vehicle forward axis, z component. [world, eval-only] |
| `speed_engine_mps` | float | m/s | yes | GetSpeedKmh() / 3.6 (unsigned). |
| `dv_max_mps` | float | m/s | no | As telemetry.dv_max_mps, for this traffic vehicle. [eval-only, build-scoped, TBD-10] |

### 9.8 Offline-derived columns

These are never written in-engine. The tool named in each heading computes them; the dataset builder and C22 use them.

Harvester, per img row (`img_resolved`; completion and `captured_render_frame` from the fiducial, R11):

<!-- dc2:fields derived=img_resolved -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `slot_id` | int (key) | slot | yes | img.slot_id. |
| `file` | str | - | no | Engine-written file relative to the session directory (TBD-4); empty when missing. [TBD-4] |
| `file_exists` | bool | - | yes | 1 when the file exists and decodes. |
| `png_w` | int | px | no | Image width (C11: exactly 1408 for the client class). |
| `png_h` | int | px | no | Image height (C11: exactly 720 for the client class). |
| `fid_ok` | bool | - | yes | 1 when fiducial v2 decodes in both bands and they agree (C13a). |
| `captured_render_frame` | int | frame | no | Decoded render_frame mod 2^18 unwrapped against issue_render_frame (C): joins hf exactly; the image shows the state of hf row C + 1, its pose row (section 6; r2 3.3: completion and captured_render_frame come from the fiducial). |
| `fid_margin` | float | 1 | no | Smallest nearest-colour margin over the decoded patches. |
| `deriv_path` | str | - | no | 512x288 JPEG q95 derivative path, once built. |

Reader, per ctl row (`ctl_export`; contract-sign steering):

<!-- dc2:fields derived=ctl_export -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | ctl.physics_step. |
| `steer_cmd` | float | 1 | no | steer_sign * llc_steer_lim: commanded steering in the contract sign (right-positive). [right+, build-scoped, TBD-5] |
| `steer_input` | float | 1 | yes | steer_sign * input_post_steer: steering input readback in the contract sign. [right+, build-scoped, TBD-5] |

Reader, per telemetry row (`telemetry_export`):

<!-- dc2:fields derived=telemetry_export -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | telemetry.physics_step. |
| `heading_rad` | float | rad | yes | atan2(fwd_x, fwd_z): 0 = north (+z), positive clockwise seen from above. World frame, eval_only. [right+, world, eval-only] |
| `speed_fwd_mps` | float | m/s | yes | vel_local_z: signed forward speed (negative in reverse). |
| `yaw_rate_rps` | float | rad/s | yes | yaw_axis_sign * angvel_local_y, right-positive (TBD-5). [right+, TBD-5] |
| `acc_long_mps2` | float | m/s2 | yes | acc_local_z: achieved longitudinal acceleration. |
| `acc_lat_mps2` | float | m/s2 | yes | acc_local_x: achieved lateral acceleration, right-positive. [right+] |
| `kappa_realised` | float | 1/m | no | yaw_rate_rps / speed_fwd_mps when abs(speed_fwd_mps) > 3 m/s, else empty. [right+] |
| `dir_state` | int | - | yes | Drive direction from the gear through the vehicle card: +1 forward gear, 0 neutral, -1 reverse (TBD-7); forward-only in T0-MIN (section 7). Values: -1, 0, 1. [TBD-7] |
| `delta_eff_rad` | float | rad | no | Reserved: bicycle-equivalent road-wheel angle from wheel_angle_l/r; C15's steer_eff once it exists. kappa = tan(delta_eff)/L holds only for abs(a_lat) <= 1 m/s2 [target]. [right+, build-scoped, reserved] |

Harvester `nav_fill` through `drivedit_nav.nav_v1`, per telemetry row of an episode with a route (`nav`):

<!-- dc2:fields derived=nav -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | telemetry.physics_step. |
| `route_id` | str | - | yes | Route of the episode. |
| `nav_version` | str | - | yes | nav_v1. |
| `progress_m` | float | m | yes | Ordered progress (drivedit_nav). [eval-only] |
| `p10_x` | float | m | yes | Route point 10 m ahead, ego x (forward); ablation only (r2 3.3). [ego] |
| `p10_y` | float | m | yes | Route point 10 m ahead, ego y (right). [right+, ego] |
| `p20_x` | float | m | yes | Route point 20 m ahead, ego x (forward). [ego] |
| `p20_y` | float | m | yes | Route point 20 m ahead, ego y (right). [right+, ego] |
| `p40_x` | float | m | yes | Route point 40 m ahead, ego x (forward). [ego] |
| `p40_y` | float | m | yes | Route point 40 m ahead, ego y (right). [right+, ego] |
| `man_left` | int | - | yes | Next maneuver within 200 m is a left turn. Values: 0, 1. |
| `man_straight` | int | - | yes | Next maneuver within 200 m is straight through a junction. Values: 0, 1. |
| `man_right` | int | - | yes | Next maneuver within 200 m is a right turn. Values: 0, 1. |
| `man_follow` | int | - | yes | No maneuver within 200 m (follow the road). Values: 0, 1. |
| `d_man` | float | m | yes | Distance to the next maneuver, 0-200; 200 with man_follow. Range [0, 200]. |
| `d_stop` | float | m | yes | Distance to the next stop (nav_v1: route_end only), 0-200; 200 = none. Range [0, 200]. |
| `dist_to_route_m` | float | m | yes | Horizontal distance from the pose to its projection inside the progress window (drivedit_nav). [eval-only] |
| `nav_valid` | int | - | yes | 0 when dist_to_route_m > 30 m (the r2 1.3 route-deviation limit): the route points are then meaningless and the builder sets policy_valid and wm_valid to 0. Values: 0, 1. |

Road matching (E-08b) and region (C24), per telemetry row (`road`):

<!-- dc2:fields derived=road -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | telemetry.physics_step. |
| `matched_road_id` | str | - | no | Generator polyline matched to road_key by E-08b (per 1 m piece, Hausdorff <= 2 m); across sessions roads are joined one-to-one by the unordered endpoint pair of road_key within 2 m (n_pts excluded), the segment Hausdorff distance deciding between candidates (4 twin pairs on Arland lie within 2 m of each other), never by the key text (TBD-9, D-C); empty when unmatched. [TBD-9] |
| `road_class` | enum | - | yes | Class inherited from the matched generator; unmatched roads are not drivable. Values: asphalt_e01, asphalt_e02, asphalt_e03, dirt, trail, cobblestone, concrete, unmatched. |
| `region` | enum | - | yes | Region of the ego position per splits_v2 (C24). Values: train, val_l, buffer, k_val, k_rep, everon. [eval-only] |

Dataset builder, per decision step (`decision_index`; the decision boundary and image rule of section 6):

<!-- dc2:fields derived=decision_index -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `decision_step` | int (key) | step | yes | P_t: physics_step of decision callback t (the first callback at or past its world-time grid point, section 6). |
| `hl_seq` | int | - | no | ctl.hl_seq of the row after C_t: the decision applied over the steps (P_t, P_t+1]; empty when C_t is the session's last ctl row. |
| `interval_steps` | int | step | no | n_t = P_t+1 - P_t, the steps of the decision interval (nominally physics_steps_per_decision; it varies with the frame clock); empty for the session's last decision callback. |
| `obs_slot_id` | int | slot | no | img.slot_id of obs_t; empty when no image qualifies (pose row last_physics_step <= P_t and captured frame at or before the render frame of C_t). |
| `obs_captured_render_frame` | int | frame | no | img_resolved.captured_render_frame of obs_t. |
| `obs_last_physics_step` | int | step | no | hf.last_physics_step of the image's pose row (captured_render_frame + 1), <= P_t. |
| `obs_age_steps` | int | step | no | P_t - obs_last_physics_step: the realised image age in physics steps. |
| `obs_age_ms` | int | ms | no | world_time_ms of C_t minus hf.world_time_ms of the captured frame (the world time of the image's pose); never negative (section 6). |
| `obs_valid` | bool | - | yes | 1 when an image qualifies and is at most one image period old; 0 sets wm_valid and policy_valid to 0. |

## 10. Session files

### 10.1 `session.json`

Written once, after every check of section 5 passed. Dotted names are nested JSON keys. In the JSON files Req = yes
means the key is always present; "yes, null ok" allows the value null; Req = no marks an optional key.

<!-- dc2:fields file=session.json -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `contract_version` | str | - | yes | dc-2.0. |
| `session_id` | str | - | yes | Session id (format in the spec, section 4). |
| `created_utc` | str | - | yes | ISO 8601 UTC time the session directory was created. |
| `start.unix_s` | int | s | yes | Wall time at start (System time APIs). |
| `start.local_hms` | str | - | yes | Local hh:mm:ss at start, for console.log slicing. |
| `start.engine_world_ms` | int | ms | yes | BaseWorld.GetWorldTime() at the session start (reference only). |
| `start.render_frame` | int | frame | yes | GetFrameNumber() at the session start. |
| `map.map_id` | enum | - | yes | Canonical map id, declared by the capture world and verified by the L8 fingerprint. Values: arland, kolguyev, everon. |
| `map.terrain` | enum | - | yes | Engine terrain name of the root world. Values: Arland, Cain, Eden. |
| `map.world_file` | str | - | yes | GetGame().GetWorldFile() (the SubScene, e.g. worlds/DataCapture/DC_Arland.ent; CR-100). |
| `map.root_world` | str | - | yes | Root world resource of the terrain (section 3 table). |
| `map.fingerprint.size_m` | float | m | yes | World size used by the L8 check. |
| `map.fingerprint.latitude_deg` | float | deg | yes | Latitude used by the L8 check. |
| `map.fingerprint.ok` | bool | - | yes | Always true in a written session (a mismatch aborts before the directory exists). |
| `trial.trial_id` | str | - | yes | Trial id (-dcTrial, or the ACTIVE pointer in Workbench). |
| `trial.source` | enum | - | yes | Where the trial id came from. Values: cli, active_pointer. |
| `trial.conf` | object | - | yes | The trial conf object as parsed ($profile:dc_trials/<trial_id>.json). |
| `trial.seed` | int | - | yes | Session seed. |
| `trial.warmup_s` | float | s | yes | Warm-up length (masked). |
| `trial.duration_s` | float | s | yes | Planned capture length after warm-up. |
| `exe.kind` | enum | - | yes | wb (Workbench), cl (ArmaReforgerSteam client), cld (Diag client). Values: wb, cl, cld. |
| `exe.instance` | str | - | yes | Instance tag i00..i99 (Workbench: i00). |
| `exe.cli` | object | - | yes | The named -dc* parameters read with System.GetCLIParam, name -> value: dcLaunchId, dcTrial (and any later named -dc* parameter). 1.8 cannot list command-line parameters, so unnamed ones are never recorded. |
| `engine.build_version` | str | - | yes | GetGame().GetBuildVersion() at start (repeated at end). |
| `engine.build_time` | str | - | yes | GetGame().GetBuildTime(). |
| `engine.fixed_time_slice_s` | float | s | yes | BaseWorld.GetFixedTimeSlice() at the open, for provenance only: it reports n/30 for a frame of n fixed steps on 1.8 and is never the step dt (D-A, TBD-1). [TBD-1] |
| `engine.step_dt_s` | float | s | yes | Engine fixed step dt measured during readiness (section 6): 1/30 s on 1.8. [TBD-1] |
| `engine.step_dt_source` | enum | - | yes | engine: the EOnFixedFrame time slice of a one-step frame, within 5% of the measurement; measured: the sum of EOnFrame time slices / the sum of GetFixedSteps() over the readiness. Values: engine, measured. [TBD-1] |
| `pin.pin_id` | str | - | yes | Pin id from the launch record (e.g. 1.8.0.13; 1.9 later). |
| `pin.pin_sha256` | str | - | yes | sha256 of the pin file, from the launch record. |
| `pin.build_key` | str | - | yes, null ok | Build identity (game and Tools appid+buildid, exe sha256s; computed by pinlib, DD-14a) from a dd-launch/2 record; null while launch records are dd-launch/1. session.json is never edited after it is written: the harvester then writes build_key (from the pin) into its sidecar harvest.json, and C02 reads it from there and refuses a session that has neither. |
| `pin.channel` | enum | - | yes | stable or experimental (dd-launch/1 records: stable). Values: stable, experimental. |
| `pin.launch_id` | str | - | yes | -dcLaunchId (launch_mode cli) or the id of the matched record (wb_escape); joins the launch record in E:\DriveDiT\pin\launches. |
| `pin.launch_schema` | str | - | yes | Schema of the launch record read (dd-launch/1, later /2). |
| `pin.game_buildid` | int | - | yes | Game Steam buildid from the launch record. |
| `pin.tools_buildid` | int | - | yes | Tools Steam buildid from the launch record. |
| `use_class` | enum | - | yes | Set at launch time, copied from the launch record; dd-launch/1 records give dev (MIGRATION_1_9 3.3). Values: dev, appearance, probe, compat-preview, g19, production. |
| `addon.addon_id` | str | - | yes | DC_BuildInfo ADDON_ID. |
| `addon.addon_guid` | str | - | yes | DC_BuildInfo ADDON_GUID. |
| `addon.git_sha` | str | - | yes | DC_BuildInfo GIT_SHA (40 hex). |
| `addon.git_branch` | str | - | yes | DC_BuildInfo GIT_BRANCH. |
| `addon.git_dirty` | bool | - | yes | DC_BuildInfo GIT_DIRTY (must be 0 from T1 on, C02). |
| `addon.content_sha256` | str | - | yes | DC_BuildInfo CONTENT_SHA256. |
| `addon.manifest_sha256` | str | - | yes | DC_BuildInfo MANIFEST_SHA256. |
| `addon.deploy_utc` | str | - | yes | DC_BuildInfo DEPLOY_UTC. |
| `addon.target_pin` | str | - | yes, null ok | Pin the deploy targeted (per-pin deploy, DD-14a); null before. |
| `control.actuation_mode` | enum | - | yes | A (driverless, the setters in every callback) or B (seated character). Values: A, B. |
| `control.llc_sha` | str | - | yes, null ok | llc_v1 source hash; null when llc_v1 is not attached. |
| `control.expert_sha` | str | - | yes, null ok | Expert source hash; null without an expert. |
| `control.vehicle_card_id` | str | - | yes, null ok | tools/ident/cards/<pin_id>/<vehicle>.json id; null before the card exists (probe sessions). |
| `control.vehicle_card_sha` | str | - | yes, null ok | sha256 of that card; null with it. |
| `control.steer_sign` | int | - | yes | Engine-to-contract steering sign (TBD-5 decided: +1, a positive SetSteering turns right on 1.8). Values: -1, 1. [TBD-5] |
| `control.yaw_axis_sign` | int | - | yes | Engine-to-contract yaw-rate sign (TBD-5 decided: +1). Values: -1, 1. [TBD-5] |
| `control.steering_geometry` | str | - | yes | unknown_on_1.8 until CC-07 validates wheel angles. |
| `authority.standalone` | bool | - | yes | Standalone authority stamp (MIGRATION_1_9 8.3). |
| `authority.rpl_ok` | bool | - | yes | Replication check stamp. |
| `versions.features_version` | str | - | yes | features_v1. |
| `versions.nav_version` | str | - | yes | nav_v1. |
| `versions.streams_schema_sha256` | str | - | no | sha256 of dc-2.0.streams.json, stamped by the deploy tool when it supports it. |
| `clocks.physics_steps_per_decision` | int | step | yes | N = floor(1000 / decision_hz / step_dt_ms + 0.5) with engine.step_dt_s: the nominal steps of a decision interval and the length of action_exec (6 at 5 Hz, 3 at 10 Hz with 1/30 s steps; section 6, TBD-1). [TBD-1] |
| `clocks.decision_hz` | float | 1/s | yes | HL decision rate, equal to image_hz (TBD-2: 5; 10 only with clocks.image_hz_setting trial_conf). [TBD-2] |
| `clocks.image_hz` | float | 1/s | yes | Image slot rate (TBD-2 decided: 5; 10 only as the non-default session setting of clocks.image_hz_setting). [TBD-2] |
| `clocks.image_hz_setting` | enum | - | yes | default: the production default 5 Hz; trial_conf: the trial conf's key image_hz set the non-default 10 Hz (C08 refuses 10 Hz without it; C37 warns unless the session's capture utilisation u95 <= 0.85). Values: default, trial_conf. [TBD-2] |
| `clocks.img_inflight_timeout_ms` | int | ms | yes | Real-time limit after which a request in flight stops blocking the next slot (TBD-2: 2000). [TBD-2] |
| `clocks.img_lead_ms` | int | ms | yes | Image grid lead: grid point g is due img_lead_ms before decision grid point g at image_hz = decision_hz (TBD-2: 0). [TBD-2] |
| `clocks.chunk_max_s` | float | s | yes | World-time bound on a chunk (10). |
| `events.collision_dv_mps` | float | m/s | yes | Collision onset threshold on dv_h, the horizontal world-frame delta-v over events.collision_window_ms of world time (section 9.5.1; default 4.0, PROVISIONAL until the TBD-10 probe, measured per build_key and again on 1.9). [TBD-10] |
| `events.collision_window_ms` | int | ms | yes | Delta-v window in world time: dv_h at a sample is measured against the velocity at t - collision_window_ms, linearly interpolated between the two samples around that instant (section 9.5.1; default 100). Rows are frame-bound (one callback covers 1 to 3 physics steps of 1/30 s on 1.8), so the window is never a count of steps or callbacks. [TBD-10] |
| `events.collision_window_tol_ms` | int | ms | yes | Gap tolerance of the delta-v window: the two samples around t - collision_window_ms must be at most collision_window_ms + this apart, else the slot is not evaluated and its history restarts (default 50: a callback that merges 3 steps of 1/30 s spans 100 ms and is evaluated, a frame hitch of more than 150 ms never counts as a 100 ms change). [TBD-10] |
| `events.collision_peak_ms` | int | ms | yes | World time after the onset over which the collision dv_mps is the peak of the velocity change against the onset's reference velocity (default 200). [TBD-10] |
| `events.collision_refractory_ms` | int | ms | yes | Per-slot refractory after an onset: a further crossing or a contact with the same other entity belongs to the same impact (default 1000). [TBD-10] |
| `events.episode_end_wait_ms` | int | ms | yes | Longest wait, from the onset, of an episode end for the ego's pending collision row (default 250). [TBD-10] |
| `events.crash_cooldown_ms` | int | ms | yes | No new episode begins until this long after the ego's last terminal crash event (default 3000: longer than the refractory 1000 ms plus the rollover hold 500 ms plus a poll period and the 250 ms row wait, so the late polls of one impact, a rollover after it and a merged second contact land before the next episode; 3 s also lets the vehicle come to rest, 20 m/s at 7 m/s2). [TBD-10] |
| `events.episode_block_max_ms` | int | ms | yes | Longest time, between episodes, for which the ego may stay tilted, immobile or burning without being destroyed: the session then ends through the planned end with end_reason ego_disabled (default 60000; the cool-down and a pending row do not count). [TBD-10] |
| `events.rollover_up_y` | float | 1 | yes | rollover when up_y (GetWorldTransform row 1, y) stays below this (default 0.5, a tilt of 60 deg). [TBD-10] |
| `events.rollover_hold_ms` | int | ms | yes | How long up_y must stay below rollover_up_y (default 500). [TBD-10] |
| `events.state_poll_ms` | int | ms | yes | Poll period of the damage, fire, driver and seat state of every slot (default 100). [TBD-10] |
| `events.contact_source` | enum | - | yes | Where collision rows get other, other_slot, other_prefab and impulse_ns: vanilla_process_collision (DD-4-min's modded SCR_VehicleDamageManagerComponent.ProcessCollision forwarder) or none (the registry match only). Values: vanilla_process_collision, none. |
| `rig.rig_id` | str | - | yes | vehicle + mount + vFOV + near plane + writer + source class + render-settings hash (r2 3.4). |
| `rig.rig_name` | str | - | yes | e.g. R1-S105. |
| `rig.ego_prefab` | str | - | yes | Ego prefab resource name. |
| `rig.mount.fwd_m` | float | m | yes | Camera mount offset forward of the vehicle origin. |
| `rig.mount.up_m` | float | m | yes | Camera mount offset up. |
| `rig.mount.right_m` | float | m | yes | Camera mount offset right. |
| `rig.mount.pitch_deg` | float | deg | yes | Camera pitch (negative = down; r2: -3 to -5). |
| `rig.vfov_deg` | float | deg | yes | Vertical FOV (50). |
| `rig.near_plane_m` | float | m | yes | Near clip plane of the rig camera (part of rig_id). 0.03 m by default so the Shape fiducial at rig.fiducial.depth_m 0.05 lies beyond it, a setting awaiting the first capture Play's E-06 check; 0.1 m with depth_m 1.0 (validated by probe run 4) is the documented fallback and an allowed recorded setting. |
| `rig.writer` | str | - | yes | MakeScreenshot. |
| `rig.source_class` | enum | - | yes | Image source class (r2 3.4). Values: wb_viewport, client_1408x720. |
| `rig.capture_wh` | array | px | yes | [w, h] of the captured image (client: [1408, 720]). |
| `rig.core_wh` | array | px | yes | [1280, 720]. |
| `rig.band_px` | int | px | yes | Side band width (64). |
| `rig.derivative_wh` | array | px | yes | [512, 288]. |
| `rig.derivative_codec` | str | - | yes | jpeg_q95. |
| `rig.fiducial` | object | - | yes | {version: v2, bits: 18, patches: 6, bits_per_patch: 3, patch_px: 32, margin_px: 16, gap_px: 8, mirrored: true, carrier: shape, depth_m: 0.05}: the debug-Shape carrier (12 flat boxes depth_m in front of the rig camera, beyond rig.near_plane_m) is the production carrier. depth_m 0.05 (near plane 0.03) awaits the first capture Play's E-06 check; depth_m 1.0 with near plane 0.1, validated by probe run 4 (5999 of 5999 decoded), is the documented fallback and an allowed recorded setting. The widget carrier never reaches MakeScreenshot files (run 4: 0 of 5999) and is a diagnostic only (carrier widget, depth_m null). |
| `rig.dpi_scale` | float | 1 | yes | User DPI scale forced by the fiducial (1.0). |
| `rig.render_settings` | object | - | yes | VideoUserSettings + DisplayUserSettings dump (TBD-8). [TBD-8] |
| `rig.render_settings_hash` | str | - | yes | Hash of that dump (C30). |
| `env_initial` | object | - | yes | {weather_state, rain, fog, wetness, tod_h, date, day_auto_advance}. |
| `launch_guard.launch_mode` | enum | - | yes | How the launch guard passed: cli (-dcLaunchId visible) or wb_escape (Workbench Play without a visible command line; section 5.1). Values: cli, wb_escape. |
| `launch_guard.record_path` | str | - | yes | The matched record, $profile:dc_launch/launch_<id>.json. |
| `launch_guard.record_launch_id` | str | - | yes | launch_id inside the matched record (equals pin.launch_id). |
| `launch_guard.cli_seen` | bool | - | yes | 1 when System.GetCLIParam saw -dcLaunchId or -profile (E-11). |
| `launch_guard.profile_root_ok` | bool | - | yes | 1: -profile (when visible) and the resolved $profile lie inside E:\DriveDiT. |
| `launch_guard.escape_binding` | str | - | yes, null ok | wb_escape only: process_start (the record was written by the launch that started this Workbench; the default) or newest_after_deploy (binding switched off for one Play); null in cli mode. |

### 10.2 `session_end.json`

Written last. Its presence seals the session; a session without it is unsealed (crash) and is salvaged or rejected by
the harvester, never treated as complete.

<!-- dc2:fields file=session_end.json -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `contract_version` | str | - | yes | dc-2.0. |
| `session_id` | str | - | yes | As in session.json. |
| `status` | enum | - | yes | ok only when every commit succeeded and no error ended the session (ego_destroyed and ego_disabled are not errors); failed otherwise. Values: ok, failed. |
| `end_reason` | enum | - | yes | Why the session ended (ego_destroyed: the ego was destroyed after the first row; ego_disabled: between episodes it stayed tilted, immobile or burning for events.episode_block_max_ms; both normal ends with status ok, section 5.3). Values: duration, stop, dc_stop, ego_destroyed, ego_disabled, script_error:write_fail, script_error:other, zero_ego, ego_lost, camera_lost. |
| `end.unix_s` | int | s | yes | Wall time at end. |
| `end.utc` | str | - | yes | ISO 8601 UTC at end. |
| `end.world_ms` | int | ms | yes | world_time_ms at end. |
| `end.physics_step` | int | step | yes | Last physics_step. |
| `end.render_frame` | int | frame | yes | Last render_frame. |
| `pin.pin_id` | str | - | yes | Repeated from session.json; C02 requires equality. |
| `pin.build_key` | str | - | yes, null ok | Repeated from session.json (same null rule; the harvester's harvest.json then carries it). |
| `engine.build_version` | str | - | yes | GetBuildVersion() read again at the end; C02 requires it to equal the start value. |
| `counts` | object | - | yes | Per stream: {rows, chunks, last_key} of committed data. |
| `images` | object | - | yes | {requested, issued, skipped_busy, make_failed, timed_out, pending_at_end}: image slot counts; pending_at_end = requests still unresolved when the session sealed (section 5.3). |
| `episodes` | int | - | yes | Number of episode rows committed. |
| `heartbeats` | int | - | yes | Number of heartbeat rows appended. |
| `observed.decision_grid_points_skipped` | int | - | no | Decision grid points a hitch skipped, as the writer counted them (section 6); C08 compares it with its own rebuild of the grid. The observed object holds further run-time facts that are not contract fields. |

## 11. Version stamps and the one-build rule

Every session carries, in `session.json`: `contract_version`; `versions.features_version`; `versions.nav_version`;
`addon.*` (DC_BuildInfo: git SHA, dirty flag, content and manifest hashes); `engine.build_version` and
`engine.build_time`; `pin.pin_id`, `pin.pin_sha256`, `pin.build_key`, `pin.channel`, `pin.launch_id`; `use_class`;
`control.expert_sha`, `control.llc_sha`, `control.vehicle_card_id` and `_sha`; `rig.rig_id` and
`rig.render_settings_hash`. `pin.pin_id`, `pin.build_key` and `engine.build_version` are repeated in
`session_end.json`, and C02 requires start and end to agree and to match the launch record.

- 1.8.0.13 is the interim pin; the target build is the first 1.9 stable build that passes G19 (D2 r2.1). The contract
  is the same on both; only the stamps and the per-build values differ.
- `build_key` (game and Tools appid and buildid, sha256 of the client and Workbench exes) is the build identity; pins
  with equal `build_key` are one build (`1.8.0.13`, `1.8.0.13-r1`, `1.8.0.13-cold`). With dd-launch/1 records the
  session's `pin.build_key` is null and the harvester supplies it in its sidecar `harvest.json` next to the sealed
  session; `session.json` and `session_end.json` are never edited after they are written. C02 and every manifest
  take `build_key` from `session.json` when it is set, else from `harvest.json`.
- `use_class` is set at launch (`dev`, `appearance`, `probe`, `compat-preview`, `g19`, `production`); C02 refuses
  `production` without a qualified G19 record for the session's `build_key`.
- `pin_id` and `build_key` are `eval_only` in `features_v1`: there is no model-visible build token.

## 12. Model boundary (features_v1)

`features_v1.json` is the registry: every channel has one class, a unit, a source in the streams, an `absolute` flag,
a `build_scoped` flag, and per-role access (input with a maximum offset in decision steps, or not an input; target or
not). The builder (DD-11) enforces it, C24 (DD-5) checks it, and `tests/test_contract_dc2.py` checks its rules.

Classes: `student_input` (may reach a deployed world model or policy; never absolute), `teacher_input` (privileged,
teacher only), `aux_target` (loss side only: targets, auxiliary targets, masks; never an input), `eval_only` (never an
input or a target).

<!-- dc2:tensors -->
| Tensor | Shape | Class | WM input | Policy input | Target of | Channels |
|---|---|---|---|---|---|---|
| `image` | [T,3,288,512] | student_input | <= 0 | <= 0 | world_model | `image_r`, `image_g`, `image_b` |
| `state` | [T,4] | student_input | <= 0 | <= 0 | - | `speed_mps`, `yaw_rate_rps`, `acc_long_achieved_mps2`, `dir_state` |
| `action_cmd` | [T,3] | student_input | <= 0 | <= -1 | - | `kappa_target`, `a_target`, `dir_cmd` |
| `action_label` | [T,3] | aux_target | no | no | policy | `label_kappa_target`, `label_a_target`, `label_dir_cmd` |
| `action_exec` | [T,6,4] | student_input | <= 0 | <= -1 | - | `exec_steer_cmd`, `exec_throttle`, `exec_brake`, `exec_steer_input` |
| `nav` | [T,10] | student_input | <= 0 | <= 0 | - | `p20_x`, `p20_y`, `p40_x`, `p40_y`, `man_left`, `man_straight`, `man_right`, `man_follow`, `d_man`, `d_stop` |
| `style` | [T,3] | student_input | <= 0 | <= 0 | - | `style_speed_factor`, `style_lane_offset_m`, `style_alat_max_mps2` |
| `masks` | [T,5] | aux_target | no | no | - | `policy_valid`, `wm_valid`, `warmup`, `env_transition`, `teleport` |
| `hindsight_anchors` | [T,6] | aux_target | no | no | world_model, policy | `anchor_2s_x`, `anchor_2s_y`, `anchor_4s_x`, `anchor_4s_y`, `anchor_6s_x`, `anchor_6s_y` |
| `future_pose` | [T,H,3] | aux_target | no | no | world_model, policy | `fut_dx`, `fut_dy`, `fut_dpsi` |
| `road_context` | [T,4] | aux_target | no | no | world_model, policy | `road_class`, `road_width_m`, `lane_lat_err_m`, `lane_head_err_rad` |
| `contact_events` | [T,2] | aux_target | no | no | world_model, policy | `contact_event`, `contact_class` |
| `privileged_route` | [T,5] | teacher_input | no | no | - | `path_kappa_0m`, `path_kappa_10m`, `path_kappa_20m`, `path_kappa_40m`, `lead_gap_m` |
| `eval_only` | [T,-] | eval_only | no | no | - | `pos_x`, `pos_y`, `pos_z`, `heading_world_rad`, `vel_world_x`, `vel_world_y`, `vel_world_z`, `distance_total_m`, `route_progress_m`, `map_id`, `terrain`, `region`, `split`, `session_id`, `pin_id`, `build_key`, `exp_now_60hz` |

Rules (r2 3.3, R35):
- Offsets are decision steps and follow the decision boundary of section 6: offset 0 of `state`, `nav` and `image`
  is the state of decision callback C_t; offset 0 of `action_cmd` is the command applied over its decision interval
  (P_t, P_t+1], and `action_exec` holds the executed values of the steps P_t + 1 .. P_t + N (N =
  `physics_steps_per_decision`; entries past P_t+1 empty). At the 5 Hz default N = 6 and `action_exec` is [T,6,4];
  at the 10 Hz override (TBD-2) N = 3 and `action_exec` is [T,3,4] (`features_v1.json` `offsets.override_10hz`).
- The world model conditions on `action_cmd` at offset 0, the command executed over [t, t+dt) through `llc_v1`, plus
  state history. It is never conditioned on realised future state (acceleration, speed, yaw at t+1 ...).
- The policy predicts `action_label` (the expert's HL command at t) and sees `action_cmd` and `action_exec` only as
  history (offset <= -1). State, nav and style are inputs at offset <= 0.
- Desired (`a_target`) and achieved (`acc_long_achieved_mps2`) acceleration are separate channels; reverse is the
  explicit `dir_cmd` channel.
- C24 forbidden inputs: `map_id`, absolute pose and altitude, `distance_total_m`, world-frame heading and velocity,
  hindsight anchors and planned-time goals (+2/+4/+6 s), future poses, anything derived from future telemetry, region,
  split, session and build identity. None is a `student_input` or `teacher_input`.
- Images: photometric augmentation on uint8 before normalisation, no clamp after normalisation, no flips (flips appear
  only in the L3 diagnostic).

## 13. Navigation (nav_v1) and the route schema

- `drivedit_nav.nav_v1(route, pose)` is the one nav function: the harvester fills the logged nav fields with it, the
  builder builds the `nav` tensor with it and the bridge calls it at test time. C22 cross-checks the three callers on
  the golden fixture. It is pure Python and deterministic.
- Output (ego frame): `p20_x, p20_y, p40_x, p40_y` (route points 20 and 40 m ahead along the route's driving line),
  the one-hot `man_left, man_straight, man_right, man_follow` (the next maneuver within 200 m, else follow),
  `d_man` (0-200 m; 200 with follow) and `d_stop` (0-200 m; 200 = no commanded stop within 200 m). `p10` is computed
  for ablations only and is not in the tensor.
- Progress is ordered (r2 1.3): the ego is projected onto the route inside a +-30 m along-route window around the last
  progress (`PROGRESS_WINDOW_M`), and progress never decreases, which handles loop crossings. Along-route distances
  are horizontal arc length. The lookahead clamps at the route end and never wraps, also on a closed route.
- `dist_to_route_m` is the distance to that projection. Above 30 m (the r2 1.3 route-deviation limit) nav_v1 returns
  `nav_valid` 0: the route points are then meaningless (for example a first call far from the route start) and the
  builder sets `policy_valid` and `wm_valid` to 0. A non-finite pose (NaN or inf) raises.
- Stops are explicit nav commands. nav_v1 has terminal stops only: every route has exactly one stop, of kind
  `route_end`, at the route end (`s_m` = length). A stop the ego must leave again needs a release that every caller
  sees (a logged `stop_release` nav command, consumed by the harvester, builder and bridge alike); that, the
  `commanded` stop kind and any dwell time are nav_v2 (DD-7). Hindsight goals are never nav inputs (RT-C03).
- `drivedit_nav/route_schema.json` is the one route schema (`route_id`, `map_id`, split tag, lane side, driving-line
  polyline in world metres, maneuvers, stops) shared by nav_v1, the T0 loop (DD-7-min), the sampler (DD-7) and C17
  (DD-5-min). `load_route` refuses Everon routes outside the sealed evaluation path.

## 14. Legacy and deprecated views

- `drivedit_nav.to_legacy_6d` gives the CLAUDE.md 6D view `[steer-equivalent of kappa_target, a_target, p20_x, p20_y,
  speed(t), yaw_rate(t)]`. It is deprecated, needs the vehicle card's inverse steering map, and its state fields are
  history at t. The CLAUDE.md tensor-contract update is DD-11 (D15).
- dc-1.0 sessions (`$profile:DrivingData/session_<n>/`) live outside the dc-2.0 root `$profile:DC_Sessions/` and never
  match the dc-2.0 session id; the dc-2.0 reader raises on them, and they are never mixed with dc-2.0 data. Until DD-11
  and DD-12 merge, CLAUDE.md's Tensor Contracts, ENFCAP, BMP, Rust streaming and navigation sections are stale; this
  contract governs.

## 15. Freeze items

- TBD-1 to TBD-9 are decided (section 1); TBD-10 stays PROVISIONAL (session.json values, measured per build_key).
- User decision D15 (adopt dc-2.0 and the CLAUDE.md tensor-contract change): adopted 2026-10-22. The CLAUDE.md
  update stays with DD-11.
- Freeze decisions D-A to D-E adopted by the user on 2026-11-03 (D-D and D-E concern routes and road-km, not this
  contract's fields).
- Sign-off by the DD-4-min, DD-5-min and DD-7-min authors.
- Proposed in the draft and confirmed at the freeze (none of these is r2 text):
  - trial selection: the `ACTIVE` trial pointer when `-dcTrial` is absent (5.2);
  - splits: the VAL-C hash rule `int(sha256(session_id)[:8], 16) mod 10 == 0` (3.2);
  - session root `$profile:DC_Sessions/` beside the dc-1.0 `DrivingData` and the probes' `DC_Probes` (4.3);
  - one ego per session; every ctl and telemetry row describes a live ego (2);
  - `end_reason` values duration, stop, dc_stop, ego_destroyed and ego_disabled (not errors),
    script_error:write_fail, script_error:other, zero_ego, ego_lost, camera_lost; failures before the directory
    exists have none; status ok or failed only (5.3, 10.2);
  - crash events (9.5.1): the kinds veh_damage, veh_fire_start, veh_fire_end, driver_state, dismount, remount and
    rollover, the collision, stuck and teleport payloads, the world-frame delta-v detector timed in world ms (frame-
    bound rows, D-A) with its gap rule and TBD-10 values, the vanilla ProcessCollision contact source (the addon's
    first modded class), episode ownership by onset with step/episode pairs for reports between callbacks, the core
    as the only source of collision and rollover terminations (first by confirmation), the bounded block ending in
    `ego_disabled`, the episode end waiting up to `events.episode_end_wait_ms` for the ego's collision row, the
    crash cool-down before the next episode, the reposition protocol, the `ego_destroyed` end at one terminal step,
    the optional `dv_max_mps` columns and the builder's `policy_valid` and ego-only `teleport` rules after a crash;
  - the launch-guard escape `wb_escape`, its process-start binding and its C02 use-class rule (5.1);
  - the decision boundary: decide after the rows of decision callback C_t, apply from physics step P_t + 1 (6);
  - frame-bound rows with `covered_steps`, the measured step dt, the world-time decision grid on the first ctl row's
    world time and the image grid on the same anchor with `img_lead_ms` (6, D-A);
  - the obs_t rule: newest image whose pose row (hf row C + 1) has `last_physics_step <= P_t` and whose captured frame
    is at or before the render frame of C_t (never a future image), dropped when older than one image period (6);
  - the `skipped_busy`, `make_failed` and `img_timeout` image rules and the end-of-session drain (5.3, 6);
  - 5 Hz images and decisions as the default, 10 Hz only as a trial-conf session setting (TBD-2, D-B);
  - chunk rules: close at `chunk_max_rows` or 10 s of world time, at least one row per chunk; commit.csv and
    heartbeat.csv grow only by open-APPEND-write-close (8);
  - the ego frame x forward, y right, and the heading convention `atan2(fwd_x, fwd_z)`, clockwise positive (7);
  - T0-MIN forward-only driving with the full S105 gear map (7, TBD-7);
  - `exp_dir_now` (9.1), the reserved `react_veh_slot_id` and `react_visible` (9.1), `veh_slot_id` (9.5, 9.7), the
    reserved `traffic` stream (9.7) and the `privileged_route` teacher inputs (12);
  - nav_v1 rules: progress None = episode start (0 m), the 30 m progress window, horizontal arc length, the lookahead
    clamped at the end and never wrapping, a maneuver counted while progress <= its `s_m` and reported within 200 m,
    a stop pending until progress passes it by 2 m, terminal `route_end` stops only, `nav_valid` above 30 m (13);
  - the route schema: `lane_side` in {right, center, mixed}, the polyline is the driving line, split tags per map
    (`val_c` is a session property, not a route tag), exactly one `route_end` stop at the route end (13).

## 16. Changelog

dc-2.0 was frozen on 2026-11-04 after the user adopted the freeze decisions D-A to D-E on 2026-11-03
(`t0/freeze/DECISIONS_dc2_freeze.md`); the reader, validator and capture core that follow it are DD-5-min and
DD-4-min (`t0/freeze/FREEZE_IMPL.md`). Each row is one decision or folded fact, the date it was decided and its
evidence; both JSON twins carry the same rows (`changelog` in `dc-2.0.streams.json` and `features_v1.json`). The
rows R1-* are the pre-capture corrections of the freeze review round 1 (Codex and Claude, 2026-11-04): no dc-2.0
session existed and nothing was pushed, so they were folded into dc-2.0 itself instead of a dc-2.x.

<!-- dc2:changelog -->
| Id | Date | Decision | Evidence |
|---|---|---|---|
| D-A | 2026-11-03 | TBD-1: frame-bound rows. One ctl and one telemetry row per fixed-frame callback with physics_step (the engine fixed-step clock, + covered_steps per row), covered_steps and render_frame; the step dt is measured, never GetFixedTimeSlice; physics_steps_per_decision from the measured step; decisions on a world-time grid, taken at the first callback at or past the grid point and applied from the next physics step; clocks.telemetry_every_k withdrawn, engine.step_dt_s and step_dt_source added (sections 6, 9.1, 9.2, 10.1) | probe run 4 E-10 (18113 steps in 10215 callbacks, 44% merged); run 3 E-10 (t0/run3/RUN3_SUMMARY.md 8); decision memo t0/freeze/DECISIONS_dc2_freeze.md |
| D-B | 2026-11-03 | TBD-2: image_hz = decision_hz = 5 as the production default (N = 6); 10 Hz only as the trial-conf session setting clocks.image_hz_setting trial_conf (N = 3), with the validator's SOFT C37 unless the session's capture utilisation u95 <= 0.85 (sections 6, 10.1, 12) | E-05 runs 1, 3 and 4 (p95 latency 101 / 99 ms at 10 Hz on the runway, 7.85% missing in the forest); decision memo |
| D-C | 2026-11-03 | TBD-9: road_key text unchanged and session-local; cross-session joins by the unordered endpoint pair within 2 m (n_pts excluded), the E-08b per-piece Hausdorff distance as the tie-break; no field change (9.2 road_key, 9.8 road) | E-08a runs 1-3: 13 of 181 keys differ between runs 2 and 3; endpoints unique for exact equality only, 4 twin pairs lie within 2 m (corrected in review round 1) (t0d/decoder/E08B.md 7) |
| D-D | 2026-11-03 | Trails (Type Path, 11.1 km on Arland) dropped from routes and road-km: no field change here; the validator thresholds lose the C32 trail key (CAPTURE_PROGRAM 5.2) and the E-08a tool expects the 34.5 km engine network | E-08b (t0d/decoder/E08B.md); t0/freeze/DD_DE_APPLIED.md |
| D-E | 2026-11-03 | R-3 maneuver typing (the straighter of two same-side branches within 35 deg is straight): a nav_v1 route-generator rule, no field change here | t0/freeze/DD_DE_APPLIED.md (DD-7-min c7024d2) |
| TBD-3 | 2026-11-04 | Writer semantics confirmed as the r2 default (append per line, exact Write counts, null open on read-only targets, WRITE truncates); commit logs consistent after a process kill | E-19 run 3 with the e19kill play: PASS, 212 of 212 commit lines (t0/freeze/e19_kill_2026-11-04.txt) |
| TBD-4 | 2026-11-03 | The engine appends .png: frames/f_<slot_id:08d>.png | E-05 runs 3-4 (5992/5992, 5999/5999) |
| TBD-5 | 2026-11-03 | steer_sign +1 (a positive SetSteering turns right; the draft -1 is refuted), yaw_axis_sign +1, local axes x right, y up, z forward | E-01 run 3 (corr 0.954; 15/15 E-02a cells) |
| TBD-6 | 2026-11-03 | The launch id is visible to a Workbench Play (launch_mode cli) | E-11 run 3 (-dcLaunchId visible=1) |
| TBD-7 | 2026-11-03 | Manual gearbox; S105 6 gears, 4 forward: 0 reverse, 1 neutral, 2..5 forward; general rule from the vanilla gear computation | E-04 runs 2-3, E-01 run 3; SCR_ToggleHandbrakeAction.c:100-108, SCR_VehicleGearCondition.c:32-44 (1.8.0.13) |
| TBD-8 | 2026-11-03 | Render-settings dump keys of E-20 (VideoUserSettings, DisplayUserSettings); the client compare follows at the first client session | E-20 runs 2-3 |
| FIDUCIAL | 2026-11-03 | rig.fiducial carrier = the debug-Shape carrier at depth_m 0.05 (the widget carrier is a diagnostic only, it never reaches screenshots); rig.fiducial gains carrier and depth_m | probe run 4 E-06 at depth 1.0 m with near plane 0.1 m (t0/run4/e06.txt): shape 5999 of 5999 decoded, widget 0 of 5999 (t0/run4/RUN4_SUMMARY.md); depth 0.05 m with near plane 0.03 m is not yet measured and awaits the first capture Play's E-06 check, 1.0 m / 0.1 m being the recorded fallback (corrected in review round 1) |
| CAPTURE-PHASE | 2026-11-03 | The logged pose follows the frame clock and the image of render frame C shows the state of hf row C + 1; img.done_render_frame added (captured = done_render_frame - 1 in Workbench); hf joins ctl and telemetry exactly (section 6, 9.4) | run 3 E-06A (argmin k = +1 on 628 of 640 moving images, pose corr 1.0000); run 4 E-06 (5999 of 5999) |
| TBD-10 | 2026-11-04 | Stays PROVISIONAL: the crash-detector values are session.json parameters measured by the TBD-10 probe (a value, never a field) | t0d/crash_events/IMPL.md |
| FREEZE | 2026-11-04 | Status FROZEN: every field, unit, sign, key, cadence or registry rule change from now on is a new version dc-2.x and re-runs the L-gate; per-build values stay in session.json and in files keyed by build_key | this changelog; t0/freeze/FREEZE_IMPL.md |
| R1-OBS | 2026-11-04 | obs_t also needs its captured frame at or before the render frame of C_t, so obs_age_ms is never negative (a frame without a callback keeps last_physics_step, so the step bound alone admitted a future image); C13a fails a decision sample with a future obs_t (sections 6, 9.8; features_v1 image rule) | Codex review round 1 X1: golden slot 15 re-posed to frame 261990 (3166 ms) against decision 93 in frame 261989 (3139 ms) became obs_t with obs_age_ms -27 (t0/freeze/review/CODEX.md); reader and validator mutation tests |
| R1-SEAL | 2026-11-04 | An image captured in the session's last render frame (a seal without drain caught its request in flight) has no pose row: pending, not joined, masks 0, never a C13a join failure; a world-teardown seal writes the held img rows too (section 6) | Claude review round 1 H1 and L2 (t0/freeze/review/CLAUDE_REVIEW.md); the crash fixture now ends with an image in flight |
| R1-STEPDT | 2026-11-04 | Step dt = the median of the readiness step callbacks' time slice / covered steps when 90% of at least 30 ratios lie within 0.5% of it (engine), else the measured span without the start remainder (measured); both kept in session.json observed (section 6) | Codex X2, Claude L1: 15 frames of 68 ms with 2 steps each measured 1.020/30 under the former rule; run 3 E-10: the EOnFixedFrame slice is covered / 30 on 21296 of 21296 rows |
| R1-HZ | 2026-11-04 | clocks.image_hz_setting names the rate: 5 Hz is default also when a trial conf sets it, only 10 Hz is trial_conf (sections 6, 10.1) | Codex X6 = Claude M1; C08 and writer tests |
| R1-CLOCK | 2026-11-04 | Frame-clock joins are HARD checks: ctl and telemetry share render_frame and world_time_ms per key (C06); ctl world_time_ms is the hf world time of its frame and hf.last_physics_step the newest callback of an earlier frame (C07) (section 6) | Codex X3 = Claude M3: the mutations render_frame +10000, world time +1000, a stale last_physics_step and P1-P4 passed before; validator tests |
| R1-HLSEQ | 2026-11-04 | hl_seq advances by 1 on the row after every decision callback of an episode that goes on (a kept command is numbered again) and on no other row; a command changes only with it (section 6, ctl.hl_seq; C07) | Codex X7: a session with every hl_seq -1 passed C07; validator test |
| R1-C37 | 2026-11-04 | C37 takes done_render_frame as utilisation evidence only where the decoded fiducial confirms it (captured = done_render_frame - 1); a contradiction refuses the evidence (section 6) | Codex X5: a redirected completion passed C37; validator test |
| R1-LAG | 2026-11-04 | C13b requires the best lag +1 whenever the yaw signal suffices; lags 0 and -1 fail (a code rule, the thresholds file is unchanged) (section 6) | Claude M2; validator test with the hf poses shifted by one row |
| R1-ROAD | 2026-11-04 | The D-C road join is one-to-one: candidate pairs within 2 m are ranked by the segment Hausdorff distance and assigned greedily, ambiguous twins reported; the endpoints are unique for exact equality only (section 1 TBD-9, 9.8 road) | Codex X4, Claude L7: A [(0,0),(10,0)] picked the parallel B2 [(0,0.1),(10,0.1)] over B1 [(0,0),(0.5,0),(10,0)]; E-08a: 4 twin pairs 0.49-1.56 m apart; tools tests |
| R1-NEAR | 2026-11-04 | rig.near_plane_m added to session.json and to rig_id; the Shape fiducial at 0.05 m with near plane 0.03 m awaits the first capture Play's E-06 check, 1.0 m with 0.1 m being the allowed recorded fallback (10.1 rig.near_plane_m, rig.fiducial) | Claude M4: probe run 4 E-06 ran at depth 1.0 m with near plane 0.1 m (t0/run4/e06.txt) |
| R1-BRIDGE | 2026-11-04 | The closed-loop bridge holds the next physics step until the current render frame is at least C_t's frame + 1 and obs_t's file has completed (done_render_frame set and at or before the current frame) before it sends state_t and obs_t (section 6; the predicate was corrected in review round 2, R2-BRIDGE) | Claude M5: obs_t is the image of C_t's own render frame in 35 of 52 policy-valid golden decisions |
| R1-TEXT | 2026-11-04 | decision_index.hl_seq may be empty for a decision at the last ctl row; slot 0 is issued in the frame after the anchor's; session_end.json observed.decision_grid_points_skipped declared (int, cross-checked by C08); the contract test compares descriptions, Range, Values and tags; nav_v1 and the route schema no longer say DRAFT; actuation mode A applies the setters in every callback | Claude L4, L5, L6, L8 (t0/freeze/review/CLAUDE_REVIEW.md) |
| R1-REFREEZE | 2026-11-04 | Review round 1 corrections folded into dc-2.0 itself: no dc-2.0 session was captured and nothing was pushed, so no dc-2.x; the FROZEN date stays 2026-11-04 | coordinator decision 2026-11-04; t0/freeze/FREEZE_IMPL.md (Review round 1 fixes) |
