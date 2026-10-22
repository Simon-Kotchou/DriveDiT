# Capture contract dc-2.0

Status: DRAFT 2026-10-22, freezes at T0-ENG core GO after E-01/E-05/E-10/E-19; pending user decision D15

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
- **TBD-n** marks a value that a T0-ENG core probe decides. The default shown is the r2 value. DD-4-min, DD-5-min and
  DD-7-min code against the default and keep the value in one constant or one session.json field, so the freeze changes
  a value, never a field name. Section 1 lists every TBD.
- The JSON twins are normative for field names, order, types, units and keys. This document is normative for
  meaning. `tests/test_contract_dc2.py` checks that the field tables below equal the JSON, so a disagreement is a test
  failure, not a judgement call.
- nav_v1 (`drivedit_nav` and `drivedit_nav/route_schema.json`) is part of this DRAFT and freezes with it. After the
  freeze any change of nav behaviour (progress window, lookahead, maneuver window, stop clearance, route-deviation
  limit) or of the route schema is a new `nav_version` (nav_v2), stamped in `episodes.nav_version` and
  `session.json versions.nav_version`; nav_v1 output never changes in place.
- Change control. Until the freeze, edits land on this branch. The freeze (T0-ENG core GO, target 2026-10-24) replaces
  every blocking TBD with the measured value, sets the status line to FROZEN and needs sign-off from the DD-4-min,
  DD-5-min and DD-7-min authors. After the freeze any change of a field, unit, sign, key, cadence or registry rule is
  a new version dc-2.x and re-runs the L-gate. A build change (1.8 to 1.9, a 1.9.x hotfix) also re-runs the L-gate
  but does not change the contract version: per-build values live in session.json and in files keyed by `build_key`,
  never in this contract.

## 1. Probe-dependent values (TBD-by-probe)

Each item is decided by the named probe of the DD-4-probe kit. The results table goes in `docs/trials/T0-ENG.md`;
the freeze copies the decided value into this document, `dc-2.0.streams.json` and `features_v1.json`. TBD-1 to TBD-5
block the freeze. TBD-6 to TBD-9 change a value or a mechanism only, never a field, and may be settled after it.

<!-- dc2:tbd -->
| Id | Item | Default (r2) | Decided by | Blocks freeze |
|---|---|---|---|---|
| TBD-1 | k rule: telemetry cadence and decision steps | physics_steps_per_decision = floor(100 ms / GetFixedTimeSlice() + 0.5), decided first (= 6 at a 60 Hz fixed step); telemetry_every_k = the largest divisor k of physics_steps_per_decision with k x GetFixedTimeSlice() <= 50 ms, at least 1 (= 3 at 60 Hz), so k always divides it; E-10 also logs the WorldTimestamp clock next to GetWorldTime per step | E-10 | yes |
| TBD-2 | image_hz (10 or 5) and the one-request-in-flight rule | image_hz = 10 (world-time slot grid phase-locked to the decision steps, img_lead_ms = 0, one MakeScreenshot request in flight, in-flight timeout 2000 ms); fall back to 5 Hz first if E-05 shows misses or request-to-file latency that 10 Hz cannot sustain; at 5 Hz the HL decision rate drops to 5 Hz as well (the decision interval is the training interval, r2 3.5); img_lead_ms is set from the E-05/E-06 capture offsets | E-05 | yes |
| TBD-3 | writer semantics (FileHandle.Write return values, Close durability, taskkill mid-commit) | FileHandle.Write returns the number of bytes written (FileHandle.c:93-103; WriteLine returns void and is never used); any return other than the expected byte count is a short or failed write; a closed file survives a process kill; one open-APPEND-write-close cycle per line adds exactly that line (commit.csv and heartbeat.csv; E-19 c11 and the e19kill e19_commit_append.csv log); no file is rewritten in place (E-19 c10 only records whether WRITE truncates) | E-19 | yes |
| TBD-4 | engine-written screenshot file name | the request path has no extension (frames/f_<slot_id:08d>) and the engine appends .png, so the file is frames/f_<slot_id:08d>.png (1.8 evidence: a request ending .bmp produced *.bmp.png) | E-05 | yes |
| TBD-5 | steer_sign normalisation (and the yaw-axis and local-axes checks) | steer_sign = -1: CarSteering is +1 for left, the contract is right-positive, so steer_contract = steer_sign * steer_engine; yaw_axis_sign = +1 [inference: Enfusion is left-handed with Y up, so +angvel_local_y is clockwise seen from above]; vehicle-local axes x right, y up, z forward; checked by tools/probes/t0eng/e01_actuation.py: steer_sign, yaw_axis_sign and local_axes (z forward: sign of vel_l_z on forward motion; x right: sign of corr(acc_l_x, v x heading rate)) | E-01 | yes |
| TBD-6 | launch id visible in Workbench Play | System.GetCLIParam("dcLaunchId") returns Workbench's own -dcLaunchId argument in Play mode (launch_guard.launch_mode cli). Answered by DD-4-probe's E-11 console line (-dcLaunchId visible=0 or 1); DD-1's 'Capture output allowed' or 'capture output refused' line tests GetCLIParam("profile") and is only an inference about it. If the id is not visible, Workbench sessions pass through the escape (launch_guard.launch_mode wb_escape, section 5.1); the session.json fields exist either way | E-11 | no |
| TBD-7 | gearbox and gear index semantics | HasAutomaticGearbox() is true on the S105 and the automatic gearbox works driverless; gear indices map to reverse/neutral/forward through the vehicle card (VehicleCinematicTrack defaults to index 2, so the mapping is read back, never assumed). T0-MIN is forward-only: the card's gear_indices_forward come from E-01 (gear_indices seen while moving forward); gear_index_neutral and gear_index_reverse stay null until a probe phase holds neutral and drives in reverse (an E-01 extension or E-02Q, DD-7-ident); no session commands hl_dir -1 before that | E-04, E-01 | no |
| TBD-8 | render-settings dump keys | GetGame().GetEngineUserSettings() modules VideoUserSettings and DisplayUserSettings, keys from SCR_SettingsManagerVideoModule.c:12-56 | E-20 | no |
| TBD-9 | road key of the in-engine road dump | road_key = <n_pts>/<x0>:<z0>/<x1>:<z1> of the road's GetPoints: n_pts in decimal, the first and last point's x and z each formatted by Enforce float.ToString(-1, 2) (fixed point, 2 decimals), 0 for a road without points (DD-4-probe DC_RoadDump.MakeKey); E-08a dedupes by it and telemetry.road_key writes the same text, so offline matching (E-08b) joins on the logged text and never re-formats coordinates | E-08a | no |

Consequences the freeze must apply:
- TBD-1: the rule is applied in this order: `physics_steps_per_decision` N = floor(100 ms / slice + 0.5) first (an
  explicit half-up rounding, the same in Python and Enforce), then k = the largest divisor of N with k x slice <= 50 ms
  (compared with a 1 us tolerance; at least 1). k therefore always divides N: (N, k) = (6, 3) at a 60 Hz slice,
  (5, 1) at 50 Hz and (3, 1) at 30 Hz. If k x slice is not 50 ms, the telemetry period is k x slice and is recorded
  through `clocks.telemetry_every_k`; C08 and C28 keep counting steps. E-10 logs the WorldTimestamp clock (the
  `world_time_ms` source) next to `GetWorldTime`, so "world time advances per step" is answered for the right clock.
- TBD-2: at 5 Hz the HL decision rate drops to 5 Hz as well, because the decision interval is the training interval
  (r2 3.5): `clocks.decision_hz` 5, `physics_steps_per_decision` 12 at a 60 Hz slice. The tensors keep their names;
  `action_cmd` stays [T,3] and `action_exec` becomes [T,12,4] (`features_v1.json` `offsets.fallback_5hz`).
- TBD-3: if `Write` does not report short writes reliably, the writer re-checks each closed chunk's size with the
  1.8 file API chosen at E-19 before writing the commit line. If E-19 c11 or the e19kill append log shows that the
  open-APPEND-write-close cycle loses or duplicates lines, commit.csv and heartbeat.csv are kept open for the session
  instead (the e19kill `e19_commit.csv` pattern).
- TBD-4: the img stream keeps the request path; only the harvester's `img_resolved.file` rule changes.
- TBD-5: `steer_sign` and `yaw_axis_sign` are recorded per session in `control.*` and per vehicle card; export always
  normalises to the contract sign (section 7).

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
- The capture code is build-agnostic: no 1.8-only constant appears in code or in this contract. Physics rate, k,
  `steer_sign`, gear maps and actuator dynamics are measured per build and recorded per session and per vehicle card.

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
  frames/f_00000000         screenshot requests; engine-written name TBD-4 (default frames/f_00000000.png)
```

Stream summary:

| Stream | Status | Written in | Cadence | Key | Key step | Chunk rows |
|---|---|---|---|---|---|---|
| `ctl` | required | EOnPostSimulate | every physics step | `physics_step` | 1 | 600 |
| `telemetry` | required | EOnPostSimulate | every k-th physics step (k = telemetry_every_k, TBD-1) | `physics_step` | k | 200 |
| `hf` | required | EOnFrame | every render frame | `render_frame` | 1 | 600 |
| `img` | required | EOnFrame | image_hz world-time grid (TBD-2) | `slot_id` | 1 | 100 |
| `events` | required | as they occur | irregular | `event_seq` | 1 | 100 |
| `episodes` | required | at episode end | one row per episode | `episode_id` | 1 | 1 |
| `traffic` | reserved | EOnPostSimulate | telemetry steps, one row per live traffic vehicle (from DD-4, T2; absent before) | `physics_step`, `veh_slot_id` | - | 2000 |

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
`camera_lost` (a camera that fails to initialise fails readiness).

## 6. Clocks, keys and monotonicity (r2 3.5; R01, R03)

| Counter | Stream | Rule | Checked by |
|---|---|---|---|
| `physics_step` | `ctl` | own counter; -1 at open, +1 at the top of `EOnSimulate`; first step 0; row written in `EOnPostSimulate` of the same step; +1 per row | C06, C07 |
| `physics_step` | `telemetry` | rows on steps with `physics_step mod k == 0` (TBD-1); +k per row, 0 missed rows | C06, C07, C08 |
| `render_frame` | `hf` | `GetFrameNumber()`, one row per render frame, +1 per row | C06, C07 |
| `slot_id` | `img` | 0-based, +1 per row; advances only while the ego is live and the camera current | C06, C07, C10 |
| `event_seq` | `events` | 0-based, +1 per row | C06, C19 |
| `episode_id` | `episodes` | 0-based, +1 per episode; -1 in other streams outside an episode | C06 |
| `world_time_ms` | all | `WorldTimestamp.DiffMilliseconds` against the session start, integer ms, non-decreasing; equal values only across frames with no physics step or while `paused` = 1 (r2 3.5). E-10 also shows whether world time advances per fixed step or per frame; if per frame, ctl rows of one frame share a value and `physics_step` stays the clock (TBD-1) | C07 |

- **No CallLater clock.** ctl and telemetry are written from `EOnPostSimulate`, which fires once per fixed step and may
  fire several times per engine update (VERIFIED-r2 `IEntity.c:53-64`). Telemetry therefore sits on an exact
  k x slice grid (50 ms at 60 Hz) whatever the FPS (R01).
- **Decision steps.** Decision step t is the physics step P_t with `P_t mod physics_steps_per_decision == 0` (N = 6
  at 60 Hz, TBD-1). Because k divides N, every decision step is also a telemetry step.
- **Decision boundary** (machine-readable in the streams JSON `timing`). In step P_t, `EOnSimulate` applies the
  command held so far and `EOnPostSimulate` writes the ctl and telemetry rows of P_t: they describe the state after
  step P_t. Then, still in `EOnPostSimulate` of P_t, the HL driver decides on that state. The new command applies in
  `EOnSimulate` of steps P_t + 1 .. P_t + N, so:
  - `ctl.hl_*` of a step is the command applied in that step, and `hl_seq` increments at P_t + 1;
  - `ctl.exp_*_now` of a step is the expert re-evaluated after that step; at P_t it is the label of decision t;
  - `action_cmd_t` = `ctl.hl_*` at P_t + 1 (applied over steps (P_t, P_t + N]); `action_label_t` = `ctl.exp_*_now`
    at P_t; `action_exec_t` = the ctl rows P_t + 1 .. P_t + N; `state_t` = telemetry at P_t, which holds no step of
    `action_cmd_t`; obs_t+1 at P_t + N holds exactly N steps of it;
  - the closed-loop bridge (DD-8) pauses after `EOnPostSimulate` of P_t, sends state_t and obs_t, and applies the
    returned command from step P_t + 1, so training and test see the same boundary.
- **Images.** The slot grid runs in world time at `image_hz` (TBD-2), phase-locked to the decision steps: grid point
  g = 0, 1, 2, ... is due at the `world_time_ms` of physics_step 0 + g x 1000 / image_hz - `clocks.img_lead_ms`
  (default 0; at image_hz = decision_hz grid point g is due `img_lead_ms` before decision step g). Each grid point
  reached while the ego is live and the camera current becomes the next `slot_id`. The request is issued in the
  first frame whose world time is at or past the due time (C08: jitter <= 1 render frame at p99). One request is in
  flight at a time: a slot that arrives while one is in flight is written `skipped_busy`, a request that
  `MakeScreenshot` refuses is written `make_failed`, and both count as missing (C10). A request whose file is not
  seen within `clocks.img_inflight_timeout_ms` (real time) stops blocking the next slot; it is logged as an `events`
  row `img_timeout` (payload `img_slot_id=<n>`) and in heartbeat `img_fail`. The request path is extension-less (R06).
- **Pairing.** The harvester decodes fiducial v2 (`render_frame mod 2^18`) from each image and joins `hf` at that
  `captured_render_frame` (C13a, exact). `hf.last_physics_step` joins `ctl` exactly (ctl has every step). Telemetry
  exists only on every k-th step, so `hf` joins `telemetry` as-of: the newest telemetry step <= `last_physics_step`,
  whose age is below k steps when no telemetry row is missing (otherwise the frame is not joined and its masks are
  0). The pose at capture is `hf`'s own `cam_*` and `veh_*`. These joins do not prove the scene's age; E-06A (marker
  reprojection per rig x exe x build) and C13b do.
- **Sampling at the model boundary.** Decision step t at physics step P_t uses the samples of the decision boundary
  above; obs_t is the newest image whose captured frame has `last_physics_step <= P_t`, dropped (masks 0) when it is
  older than one image period; nav comes from `drivedit_nav.nav_v1` at the pose of P_t. The builder writes the
  realised image age of every decision step (`decision_index`, section 9.8).
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
- **Engine-native steering.** `CarSteering` is +1 for left. `llc_steer_*` and `input_*_steer` are logged exactly as
  passed to `SetSteering` or returned by `GetSteering`. Export normalises: `steer_cmd = steer_sign * llc_steer_lim`,
  `steer_input = steer_sign * input_post_steer`, with `steer_sign` from `session.json` `control.steer_sign` (TBD-5,
  default -1). `yaw_rate_rps = yaw_axis_sign * angvel_local_y` (TBD-5, default +1). The vehicle card records both
  per build.
- **Steering channels** (MIGRATION_1_9 8.3, CRIT F-09). `steer_cmd` is the command in [-1, 1]; `steer_input` the input
  readback (`input_post`); `wheel_angle_l_rad` and `wheel_angle_r_rad` the applied road-wheel angles; `delta_eff_rad`
  their bicycle equivalent; `kappa_realised` = yaw rate / forward speed above 3 m/s. On 1.8 the wheel angles are
  unknown: the fields are reserved and empty, `control.steering_geometry` is `unknown_on_1.8`, and they become labels
  only after CC-07 step 0 validates a source. C15's `steer_eff` is `delta_eff` once it exists, else the identified lag
  of `input_post_steer`. The bicycle relation kappa = tan(delta_eff) / L is used only for abs(a_lat) <= 1 m/s2 [target].
- **Getters mean inputs.** `GetSteering`, `GetThrottle`, `GetBrake` return simulation inputs, not wheel angle or force
  (`VehicleWheeledSimulation.c:18-37`): they are `input_*`, never "applied".
- **Gear.** `gear` and `llc_gear` are engine indices; the vehicle card maps them to reverse, neutral and forward
  (TBD-7) for `dir_state`. T0-MIN is forward-only: the card names only `gear_indices_forward` (from E-01), so
  `dir_state` is +1 in those gears and 0 otherwise (standstill in an unmapped gear), never -1; `gear_index_neutral`
  and `gear_index_reverse` stay null until a probe phase measures them (DD-7-ident), and no session commands
  `hl_dir` -1 before that.
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

### 9.1 Stream `ctl` (every physics step)

<!-- dc2:fields stream=ctl -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | Own physics-step counter: -1 when the session opens, +1 at the top of EOnSimulate, so the first step is 0. The row is written in EOnPostSimulate of the same step. |
| `world_time_ms` | int | ms | yes | World time since the session start: WorldTimestamp.DiffMilliseconds against the session-start timestamp (R03). Non-decreasing. |
| `episode_id` | int | - | yes | Current episode; -1 outside an episode (warm-up, settle, between episodes). |
| `mode` | enum | - | yes | Capture mode of this step (r2 3.1): E clean expert, N noise injection, R recovery start, X exploration, D DAgger, H human, A AI ego, I idle (no HL command held: warm-up, settle). Values: E, N, R, X, D, H, A, I. |
| `driver` | enum | - | yes | Who produced the executed HL command of this step (C18, C27). Values: expert, policy, human, ai, none. |
| `hl_seq` | int | - | yes | Counter of the HL decision applied in this step: -1 before the first decision; it increments on the first step after each decision step (physics_step mod physics_steps_per_decision == 1), because the decision of P_t is taken in EOnPostSimulate of P_t and applies from step P_t + 1 (section 6, TBD-1). [TBD-1] |
| `hl_kappa` | float | 1/m | no | Held HL curvature target applied in this step's EOnSimulate (the action_cmd.kappa_target of decision hl_seq, decided at the previous decision step), right-positive. Empty in mode I. [right+] |
| `hl_accel` | float | m/s2 | no | Held HL acceleration target applied in this step (action_cmd.a_target, desired). |
| `hl_dir` | int | - | no | Held HL direction command applied in this step: -1 reverse, 0 stop, +1 forward. Values: -1, 0, 1. |
| `exp_kappa_now` | float | 1/m | no | Expert HL curvature re-evaluated on the true state after this step (EOnPostSimulate, r2 3.2): a 60 Hz diagnostic; at a decision step P_t it is the label of decision t, which in mode E is the command applied from P_t + 1. Empty when no expert is attached. [right+] |
| `exp_accel_now` | float | m/s2 | no | Expert HL acceleration re-evaluated after this step. |
| `exp_dir_now` | int | - | no | Expert HL direction re-evaluated after this step (added to r2's exp_{kappa,accel}_now so the label has all three action channels). Values: -1, 0, 1. |
| `llc_steer_raw` | float | 1 | no | llc_v1 steering before its rate limiter and lag, engine-native sign (contract sign: multiply by session.json control.steer_sign, TBD-5). Empty when llc_v1 is not driving. [engine-native, build-scoped, TBD-5] |
| `llc_steer_lim` | float | 1 | no | llc_v1 steering after the explicit rate limiter and first-order lag, as passed to SetSteering, engine-native sign. Range [-1, 1]. [engine-native, build-scoped, TBD-5] |
| `llc_throttle` | float | 1 | no | llc_v1 throttle as passed to SetThrottle; never non-zero together with llc_brake. Range [0, 1]. [build-scoped] |
| `llc_brake` | float | 1 | no | llc_v1 brake as passed to SetBreak. Range [0, 1]. [build-scoped] |
| `llc_handbrake` | bool | - | no | Handbrake flag passed to SetBreak. [build-scoped] |
| `llc_gear` | int | - | no | Gear index passed to SetGear (engine index; mapping in the vehicle card, TBD-7). Empty when no gear is set this step. [build-scoped, TBD-7] |
| `llc_clutch` | float | 1 | no | Clutch passed to SetClutch; empty unless a scripted gearbox sets it (TBD-7). Range [0, 1]. [build-scoped, TBD-7] |
| `input_pre_steer` | float | 1 | yes | GetSteering() read in EOnSimulate right after the setters (an input readback, not a wheel angle; r2 3.2), engine-native sign. Range [-1, 1]. [engine-native, build-scoped, TBD-5] |
| `input_pre_throttle` | float | 1 | yes | GetThrottle() read in EOnSimulate right after the setters (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_pre_brake` | float | 1 | yes | GetBrake() read in EOnSimulate right after the setters (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_pre_clutch` | float | 1 | yes | GetClutch() read in EOnSimulate right after the setters (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_pre_gear` | int | - | yes | GetGear() read in EOnSimulate right after the setters (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `input_pre_handbrake` | bool | - | yes | IsHandbrakeOn() read in EOnSimulate right after the setters (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `input_post_steer` | float | 1 | yes | GetSteering() read in EOnPostSimulate after integration (an input readback, not a wheel angle; r2 3.2), engine-native sign. Range [-1, 1]. [engine-native, build-scoped, TBD-5] |
| `input_post_throttle` | float | 1 | yes | GetThrottle() read in EOnPostSimulate after integration (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_post_brake` | float | 1 | yes | GetBrake() read in EOnPostSimulate after integration (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_post_clutch` | float | 1 | yes | GetClutch() read in EOnPostSimulate after integration (an input readback, not a wheel angle; r2 3.2). Range [0, 1]. [build-scoped] |
| `input_post_gear` | int | - | yes | GetGear() read in EOnPostSimulate after integration (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `input_post_handbrake` | bool | - | yes | IsHandbrakeOn() read in EOnPostSimulate after integration (an input readback, not a wheel angle; r2 3.2). [build-scoped] |
| `noise_active` | bool | - | yes | 1 while an N-mode disturbance is applied to the executed command. |
| `noise_id` | int | - | yes | Disturbance id (events perturb_start/perturb_end); -1 when none. |
| `steer_signal_raw` | float | 1 | no | Reserved, post-GO (DD-4-probe (d)): the simulation's steering0 signal, a raw candidate for the applied steering. Empty until CC-07 step 0 validates it. [build-scoped, reserved] |
| `wheel_angle_l_rad` | float | rad | no | Reserved: applied left road-wheel angle, right-positive. Unknown on 1.8; filled only after CC-07 step 0 validates a source (MIGRATION_1_9 8.3). [right+, build-scoped, reserved] |
| `wheel_angle_r_rad` | float | rad | no | Reserved: applied right road-wheel angle, right-positive. Unknown on 1.8. [right+, build-scoped, reserved] |
| `react_veh_slot_id` | int | - | no | Reserved until DD-4/DD-7 (empty before): vehicle slot of the actor the expert reacts to in this step (perception-limited reactions, r2 3.1); empty when it reacts to no actor. [reserved] |
| `react_visible` | bool | - | no | Reserved until DD-4/DD-7: 1 when that actor passes the in-engine visibility test of r2 3.1 (camera FOV + 10 deg, raycast line of sight, or within 15 m), 0 when not; the builder sets policy_valid 0 while a reaction to an invisible actor lasts. [reserved] |

### 9.2 Stream `telemetry` (every k-th physics step)

<!-- dc2:fields stream=telemetry -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | Physics step of the row; rows are written in EOnPostSimulate on every step with physics_step mod k == 0 (k = telemetry_every_k, TBD-1), so consecutive rows differ by exactly k. [TBD-1] |
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
| `road_key` | str | - | no | Key of the closest engine road, <n_pts>/<x0>:<z0>/<x1>:<z1> exactly as the E-08a dump writes it (TBD-9); empty when no road is found. [TBD-9] |
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
| `world_time_ms` | int | ms | yes | As in ctl; may repeat across frames without a physics step or while paused. |
| `last_physics_step` | int | step | yes | physics_step of the most recent EOnPostSimulate; -1 before the first. |
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
| `requested_world_ms` | int | ms | yes | Due time of the slot's grid point: world_time_ms of physics_step 0 + g x 1000 / image_hz - clocks.img_lead_ms (section 6, TBD-2). [TBD-2] |
| `issue_status` | enum | - | yes | issued; skipped_busy when the previous request was still in flight and younger than clocks.img_inflight_timeout_ms at the slot; make_failed when System.MakeScreenshot returned false (the issue fields are kept). Skipped and failed slots count as missing in C10 (TBD-2). Values: issued, skipped_busy, make_failed. [TBD-2] |
| `issue_world_ms` | int | ms | no | World time of the frame the request was issued in (the first frame at or after the slot). Empty when skipped. |
| `issue_render_frame` | int | frame | no | render_frame of that frame. Empty when skipped. |
| `issue_physics_step` | int | step | no | last_physics_step at issue. Empty when skipped. |
| `path` | str | - | no | Request path relative to the session directory, without extension: frames/f_<slot_id:08d> (TBD-4). Empty when skipped. [TBD-4] |
| `cam_current` | bool | - | yes | 1 when the rig camera is the current camera at issue (C18). |
| `cam_attached` | bool | - | yes | 1 when the rig camera is attached to the ego at issue (C18). |

### 9.5 Stream `events`

Event kinds: `session_start`, `warmup_end`, `episode_start`, `settle_end`, `episode_end`, `perturb_start`,
`perturb_end`, `nav_command`, `style_set`, `teleport`, `respawn`, `env_change`, `pause`, `resume`, `stuck`,
`collision`, `camera_reset`, `img_timeout` (an image request in flight longer than `clocks.img_inflight_timeout_ms`;
payload `img_slot_id=<n>`), `dc_stop`, `script_error`, and `veh_cfg_override` (reserved, null on 1.8; a 1.9
`Update*Configuration` call must be logged with this kind).

<!-- dc2:fields stream=events -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `event_seq` | int (key) | - | yes | Event counter, 0-based, +1 per row. |
| `episode_id` | int | - | yes | Episode the event belongs to; -1 outside episodes. |
| `physics_step` | int | step | yes | physics_step of the last completed step when the event was recorded. |
| `render_frame` | int | frame | yes | render_frame when the event was recorded. |
| `world_time_ms` | int | ms | yes | As in ctl. |
| `kind` | enum | - | yes | Event kind (veh_cfg_override is reserved, null on 1.8). Values: session_start, warmup_end, episode_start, settle_end, episode_end, perturb_start, perturb_end, nav_command, style_set, teleport, respawn, env_change, pause, resume, stuck, collision, camera_reset, img_timeout, dc_stop, script_error, veh_cfg_override. |
| `veh_slot_id` | int | - | yes | Subject vehicle slot (ego_slot_id or a traffic slot), -1 when none; named veh_slot_id so a join by column name never meets img.slot_id. |
| `payload` | str | - | no | Kind-specific 'key=value' pairs joined by ';' (ASCII, no ',' or '='-in-values): termination reasons, perturbation parameters (noise_id, kind, amplitude, duration_s), nav commands (route_id, maneuver, s_m, stop s_m), style_v1 values, script_error reason, img_timeout img_slot_id. |

### 9.6 Stream `episodes` (one row per episode, written at its end)

<!-- dc2:fields stream=episodes -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `episode_id` | int (key) | - | yes | Episode id, 0-based, +1 per episode; one row written at episode end. |
| `start_physics_step` | int | step | yes | First physics step of the episode (after the settle). |
| `end_physics_step` | int | step | yes | Last physics step of the episode. |
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

### 9.7 Stream `traffic` (reserved for T2)

Written from DD-4 on, at the telemetry cadence, one row per live traffic vehicle; key (`physics_step`,
`veh_slot_id`).
Absent in T0-MIN to T1 sessions; readers accept its absence.

<!-- dc2:fields stream=traffic -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `physics_step` | int (key) | step | yes | Telemetry step of the row (same cadence as telemetry). |
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
| `captured_render_frame` | int | frame | no | Decoded render_frame mod 2^18 unwrapped against issue_render_frame; joins hf (r2 3.3: completion and captured_render_frame come from the fiducial). |
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
| `matched_road_id` | str | - | no | Generator polyline matched to road_key by E-08b (Hausdorff <= 2 m); empty when unmatched. [TBD-9] |
| `road_class` | enum | - | yes | Class inherited from the matched generator; unmatched roads are not drivable. Values: asphalt_e01, asphalt_e02, asphalt_e03, dirt, trail, cobblestone, concrete, unmatched. |
| `region` | enum | - | yes | Region of the ego position per splits_v2 (C24). Values: train, val_l, buffer, k_val, k_rep, everon. [eval-only] |

Dataset builder, per decision step (`decision_index`; the decision boundary and image rule of section 6):

<!-- dc2:fields derived=decision_index -->
| Field | Type | Unit | Req | Description |
|---|---|---|---|---|
| `decision_step` | int (key) | step | yes | P_t: physics_step of decision step t (P_t mod physics_steps_per_decision == 0). |
| `hl_seq` | int | - | yes | ctl.hl_seq at P_t + 1: the decision applied over steps (P_t, P_t + physics_steps_per_decision]. |
| `obs_slot_id` | int | slot | no | img.slot_id of obs_t; empty when no image qualifies. |
| `obs_captured_render_frame` | int | frame | no | img_resolved.captured_render_frame of obs_t. |
| `obs_last_physics_step` | int | step | no | hf.last_physics_step of that frame (<= P_t). |
| `obs_age_steps` | int | step | no | P_t - obs_last_physics_step: the realised image age in physics steps. |
| `obs_age_ms` | int | ms | no | world_time_ms of P_t minus hf.world_time_ms of the captured frame. |
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
| `engine.fixed_time_slice_s` | float | s | yes | BaseWorld.GetFixedTimeSlice() (TBD-1). [TBD-1] |
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
| `control.actuation_mode` | enum | - | yes | A (driverless, setters every step) or B (seated character). Values: A, B. |
| `control.llc_sha` | str | - | yes, null ok | llc_v1 source hash; null when llc_v1 is not attached. |
| `control.expert_sha` | str | - | yes, null ok | Expert source hash; null without an expert. |
| `control.vehicle_card_id` | str | - | yes, null ok | tools/ident/cards/<pin_id>/<vehicle>.json id; null before the card exists (probe sessions). |
| `control.vehicle_card_sha` | str | - | yes, null ok | sha256 of that card; null with it. |
| `control.steer_sign` | int | - | yes | Engine-to-contract steering sign (TBD-5; default -1). Values: -1, 1. [TBD-5] |
| `control.yaw_axis_sign` | int | - | yes | Engine-to-contract yaw-rate sign (TBD-5; default +1). Values: -1, 1. [TBD-5] |
| `control.steering_geometry` | str | - | yes | unknown_on_1.8 until CC-07 validates wheel angles. |
| `authority.standalone` | bool | - | yes | Standalone authority stamp (MIGRATION_1_9 8.3). |
| `authority.rpl_ok` | bool | - | yes | Replication check stamp. |
| `versions.features_version` | str | - | yes | features_v1. |
| `versions.nav_version` | str | - | yes | nav_v1. |
| `versions.streams_schema_sha256` | str | - | no | sha256 of dc-2.0.streams.json, stamped by the deploy tool when it supports it. |
| `clocks.telemetry_every_k` | int | step | yes | k (TBD-1). [TBD-1] |
| `clocks.physics_steps_per_decision` | int | step | yes | Physics steps per HL decision (TBD-1). [TBD-1] |
| `clocks.decision_hz` | float | 1/s | yes | HL decision rate (10). |
| `clocks.image_hz` | float | 1/s | yes | Image slot rate (TBD-2; default 10). [TBD-2] |
| `clocks.img_inflight_timeout_ms` | int | ms | yes | Real-time limit after which a request in flight stops blocking the next slot (TBD-2; default 2000). [TBD-2] |
| `clocks.img_lead_ms` | int | ms | yes | Image grid lead: grid point g is due img_lead_ms before the world time of decision step g at image_hz = decision_hz (TBD-2; default 0, set from the E-05/E-06 capture offsets). [TBD-2] |
| `clocks.chunk_max_s` | float | s | yes | World-time bound on a chunk (10). |
| `rig.rig_id` | str | - | yes | vehicle + mount + vFOV + writer + source class + render-settings hash (r2 3.4). |
| `rig.rig_name` | str | - | yes | e.g. R1-S105. |
| `rig.ego_prefab` | str | - | yes | Ego prefab resource name. |
| `rig.mount.fwd_m` | float | m | yes | Camera mount offset forward of the vehicle origin. |
| `rig.mount.up_m` | float | m | yes | Camera mount offset up. |
| `rig.mount.right_m` | float | m | yes | Camera mount offset right. |
| `rig.mount.pitch_deg` | float | deg | yes | Camera pitch (negative = down; r2: -3 to -5). |
| `rig.vfov_deg` | float | deg | yes | Vertical FOV (50). |
| `rig.writer` | str | - | yes | MakeScreenshot. |
| `rig.source_class` | enum | - | yes | Image source class (r2 3.4). Values: wb_viewport, client_1408x720. |
| `rig.capture_wh` | array | px | yes | [w, h] of the captured image (client: [1408, 720]). |
| `rig.core_wh` | array | px | yes | [1280, 720]. |
| `rig.band_px` | int | px | yes | Side band width (64). |
| `rig.derivative_wh` | array | px | yes | [512, 288]. |
| `rig.derivative_codec` | str | - | yes | jpeg_q95. |
| `rig.fiducial` | object | - | yes | {version: v2, bits: 18, patches: 6, bits_per_patch: 3, patch_px: 32, mirrored: true}. |
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
| `status` | enum | - | yes | ok only when every commit succeeded and no error ended the session; failed otherwise. Values: ok, failed. |
| `end_reason` | enum | - | yes | Why the session ended. Values: duration, stop, dc_stop, script_error:write_fail, script_error:other, zero_ego, ego_lost, camera_lost. |
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
  is the state after step P_t; offset 0 of `action_cmd` and `action_exec` is the command applied over steps
  (P_t, P_t + N]. At the 5 Hz fallback (TBD-2) the decision rate drops to 5 Hz; `action_exec` is then [T,12,4].
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

## 15. Open items before the freeze

- TBD-1 to TBD-5 from `docs/trials/T0-ENG.md` (E-10, E-05, E-19, E-01).
- User decision D15 (adopt dc-2.0 and the CLAUDE.md tensor-contract change).
- Sign-off by the DD-4-min, DD-5-min and DD-7-min authors.
- Proposed here and to be confirmed at the freeze (none of these is r2 text):
  - trial selection: the `ACTIVE` trial pointer when `-dcTrial` is absent (5.2);
  - splits: the VAL-C hash rule `int(sha256(session_id)[:8], 16) mod 10 == 0` (3.2);
  - session root `$profile:DC_Sessions/` beside the dc-1.0 `DrivingData` and the probes' `DC_Probes` (4.3);
  - one ego per session; every ctl and telemetry row describes a live ego (2);
  - `end_reason` values duration, stop, dc_stop, script_error:write_fail, script_error:other, zero_ego, ego_lost,
    camera_lost; failures before the directory exists have none; status ok or failed only (5.3, 10.2);
  - the launch-guard escape `wb_escape`, its process-start binding and its C02 use-class rule (5.1);
  - the decision boundary: decide in `EOnPostSimulate` of P_t, apply from P_t + 1 (6);
  - the decision phase lock `physics_step mod physics_steps_per_decision == 0`, the TBD-1 resolution order, and the
    image grid phase lock with `img_lead_ms` (6);
  - the obs_t rule: newest image with `last_physics_step <= P_t`, dropped when older than one image period (6);
  - the `skipped_busy`, `make_failed` and `img_timeout` image rules and the end-of-session drain (5.3, 6);
  - the 5 Hz fallback dropping the decision rate to 5 Hz (TBD-2);
  - chunk rules: close at `chunk_max_rows` or 10 s of world time, at least one row per chunk; commit.csv and
    heartbeat.csv grow only by open-APPEND-write-close (8);
  - the ego frame x forward, y right, and the heading convention `atan2(fwd_x, fwd_z)`, clockwise positive (7);
  - T0-MIN forward-only `dir_state` (7, TBD-7);
  - `exp_dir_now` (9.1), the reserved `react_veh_slot_id` and `react_visible` (9.1), `veh_slot_id` (9.5, 9.7), the
    reserved `traffic` stream (9.7) and the `privileged_route` teacher inputs (12);
  - nav_v1 rules: progress None = episode start (0 m), the 30 m progress window, horizontal arc length, the lookahead
    clamped at the end and never wrapping, a maneuver counted while progress <= its `s_m` and reported within 200 m,
    a stop pending until progress passes it by 2 m, terminal `route_end` stops only, `nav_valid` above 30 m (13);
  - the route schema: `lane_side` in {right, center, mixed}, the polyline is the driving line, split tags per map
    (`val_c` is a session property, not a route tag), exactly one `route_end` stop at the route end (13).
