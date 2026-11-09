# DataCapture Workbench addon (versioned source)

This folder is the git-tracked source of the Arma Reforger 1.8 capture addon `DataCapture`
(GUID `88BEF4187E0D2DFC`, depends only on vanilla ArmaReforger `58D0FB3206B6F859`). It replaces the untracked
`DataTest` addon (GUID `68F21EE03DB916A2`). Never load both in one Workbench session: they define the same script
classes.

| File | What it is |
|---|---|
| `addon.gproj` | Project file (same GameProject syntax as DataTest; new ID and GUID) |
| `worlds/DataCapture/DC_Arland.ent` | Arland capture world: a SubScene of vanilla `worlds/Arland/Arland.ent` |
| `worlds/DataCapture/DC_Arland_Layers/default.layer` | Its layer (game mode, AI world, cars, spawn points, drive targets) |
| `worlds/DataCapture/DC_Probes_Arland.ent` | T0-ENG probe world (DD-4-probe): a SubScene of vanilla Arland, see below |
| `worlds/DataCapture/DC_Probes_Arland_Layers/default.layer` | Its layer (probe game mode with the runner and the probe components, AI world, `DC_Ego`) |
| `deploy_manifest.txt` | The only files `tools/deploy_addon.py` may copy into the deployed addon |

Not versioned, because Workbench generates and owns them: `*.meta`, `resourceDatabase.rdb`, `UserMaps.desc`.
The Enforce scripts live in `src/enfusion/datacapture/` and are deployed to `Scripts/Game/DataCapture/` by the
manifest.

In Workbench the world is `$DataCapture:worlds/DataCapture/DC_Arland.ent`. Workbench gives it a GUID in a `.meta`
file the first time it registers the world. The client launcher passes `-world {that GUID}worlds/DataCapture/DC_Arland.ent`
and the MCP runbook opens `worlds/DataCapture/DC_Arland.ent`. Keep this path stable: moving the world after it has
been registered changes the path that every launcher refers to.

## Deploy

```bash
python3 -I -B tools/deploy_addon.py --dry-run     # show the plan
python3 -I -B tools/deploy_addon.py               # deploy to <DATA_ROOT>/addons/DataCapture
python3 -I -B tools/deploy_addon.py --emcp-handlers <BK build>/mod/Scripts/WorkbenchGame/EnfusionMCP
python3 -I -B tools/test_deploy_addon.py -v       # self-test (temp targets only)
```

The tool also writes `Scripts/Game/DataCapture/DC_BuildInfo.c` (git sha, branch, dirty flag, deploy time, manifest
and content hashes) and tracks what it deployed in `<target>/.deploy_state.json`. It deploys only into a folder inside
`<DATA_ROOT>/addons/` (`--allow-root DIR` adds a scratch root on the data drive or under `/dev/shm/`, for static checks
and the self-test) and never onto the system drive or the WSL disk image. It refuses files outside the manifest (case variants
included), writing through a symlink or junction inside the target, and overwriting or deleting anything that
changed in the target since it was deployed (for example a layer saved by Workbench) unless `--force` is given,
which backs the old file up first.

## Editing the world

- Edit the layer here, deploy, then reload the world in Workbench. If you change the world inside Workbench, copy the
  saved file back here before the next deploy (the deploy refuses to overwrite it otherwise).
- After any hand edit, check that the world loads without `Unexpected data at offset` or `Unknown keyword` in
  `console.log` and that all entities are created. Vectors are never quoted: `m_vHoodOffset 2.2 0.95 0`.
- Never save the world from a Workbench session that loaded it truncated: saving overwrites the layer with what was
  loaded.
- `python3 -I -B tools/check_layer.py addons/DataCapture/worlds/DataCapture/DC_Arland_Layers/default.layer` checks a
  hand-edited layer statically (quoted vectors, braces, userScript, entity list) before you deploy it.

## What a Play does

A Play on `DC_Arland` runs the dc-2.0 capture session: `DC_Session` and its modules on `GameMode_DataCapture`
(DD-4-min), described in the next sections.

Before DD-4-min, `DC_Arland` ran the dc-1.0 director and collector (`SCR_MLDataCaptureDirector`,
`SCR_MLDataCollector`, `SCR_MLScreenshotCapture`), which wrote `$profile:DrivingData/session_<n>/`. DD-4-min
stopped deploying them and DD-12 deleted their sources from main. The sources, and this section's former
description of their Play, console lines and output, are in tag `archive/dc1-rust`
(`git show archive/dc1-rust:addons/DataCapture/README.md`).

## The dc-2.0 capture core at the contract freeze (2026-11-04)

`DC_Session` and its modules on `DC_Arland` write contract dc-2.0 as frozen (decisions D-A, D-B and the folded facts
of `t0/freeze/DECISIONS_dc2_freeze.md`):

- **Frame-bound rows (D-A).** The game mode gets one `EOnFixedFrame` / `EOnPostFixedFrame` pair per render frame
  with a fixed step, covering 1 to 3 merged steps of 1/30 s on 1.8. Every callback writes one `ctl` and one
  `telemetry` row: `physics_step` (the last step it covers), `covered_steps`, `render_frame`, `world_time_ms`. The
  step dt is measured during readiness (`[DC_Session] clocks: step dt ...` line; `session.json engine.step_dt_s`):
  the median of the step callbacks' time slice / covered steps (`engine`, exact also in merged frames, once 90% of
  at least 30 ratios agree within 0.5%); when they disagree the measurement goes on, and the measured span is taken
  only once it is long (90 steps), its halves agree within 0.5% and it confirms the median within 0.5% (`measured`).
  Without a trustworthy value by the readiness timeout readiness is refused with the reason. All values go to
  `session.json observed.readiness_*`. Never `GetFixedTimeSlice()` alone (it reports n/30 per frame). Drivers get
  `timeSlice` = covered steps x step dt.
- **Numbered decisions and admission.** `hl_seq` advances on the callback after every decision callback of an
  episode; a driver whose `Decide` returns false keeps the held command, which the core numbers again (C07). An
  episode is admitted only after the handshake `CanDecide(physicsStep)` (base default true): a driver that cannot
  give the first command in this callback returns false and is asked again at the next grid point, or it fills
  `DC_EpisodeInfo.m_InitialCommand` in `BeginEpisode`. Only a first `Decide` false after an admission without an
  initial command fails the session.
- **Decisions and images on one world-time grid.** Every 200 ms at the default 5 Hz from the first callback's world
  time; a decision is taken in the first callback at or past a grid point and applies from the next callback. 10 Hz
  only through the trial conf key `"image_hz": 10` (`clocks.image_hz_setting` `trial_conf`; a trial conf with
  `"image_hz": 5` runs the default and is stamped `default`); the layer must keep `m_fImageHz 5`, any other value
  refuses the session.
- **Images.** The fiducial is the debug-Shape carrier (`m_eFiducialCarrier 1` = SHAPE) 0.05 m in front of the rig
  camera (`m_fShapeDepthM`), so the rig's near plane is 0.03 m (`DC_HfRecorder m_fNearPlaneM`; the probes drew the
  bands at 1 m with a 0.1 m near plane, the documented fallback). The near plane is written to `session.json
  rig.near_plane_m` and is part of `rig_id`. The widget carrier (0) is a diagnostic only: widgets never reach
  `MakeScreenshot` files. Each `img` row is written when its slot resolves, with `done_render_frame` (the captured
  frame is `done_render_frame - 1` in Workbench); the held rows belong to the session, so a world-teardown seal
  writes them too (a request in flight with `done_render_frame` empty). The Shape carrier's relayouts and layout
  failures go to `session_end.json observed.fiducial_shape_*`.
- **Signs.** `DC_StreamRecorder m_iSteerSign 1` (a positive `SetSteering` turns right, E-01 run 3).

What only a Play confirms: the step-dt line (`engine` source, 0.033333 s), rows with `covered_steps` 2-3 on screenshot
frames, the Shape bands decoded at 0.05 m with the 0.03 m near plane (E-06 tool on a capture session), and C08 / C28
passing on the first T0a/T0b session.

## Crash events and the addon's first modded class (dc-2.0 9.5.1)

`DC_CrashWatch` (`src/enfusion/datacapture/DC_CrashWatch.c`, a capture module on `GameMode_DataCapture` in
`DC_Arland`) writes the crash events of contract dc-2.0 section 9.5.1 for every watched vehicle slot (the ego now,
traffic slots from DD-4/T2): `collision`, `veh_damage`, `veh_fire_start`, `veh_fire_end`, `driver_state`,
`dismount`, `remount` and `rollover`, and the detector values in `session.json` `events.*`.

- Collisions come from the horizontal world-frame velocity change over 100 ms of **world time**, interpolated
  between the samples of the physics callbacks. On 1.8 one callback covers 1 to 3 merged steps of 1/30 s (frame-bound
  rows, freeze decision D-A), so the window is never a count of callbacks; two samples more than 150 ms apart (a
  hitch) restart the history instead of counting as a change. `data/dc2/crash_dv.py` is the same rule in Python.
  The threshold `m_fCollisionDvMps` 4.0 m/s is **provisional** (TBD-10): it must be measured again by the TBD-10
  probe, for every `build_key` and on 1.9, with 10 Hz images on so that frames merge steps. `telemetry.dv_max_mps`
  keeps the per-row maximum, so a new value can be applied offline.
- Crash rows belong to the episode in which they occurred, also when written after its end. After an ego crash no
  new episode begins for `m_iCrashCooldownMs` (3 s), nor while the ego is tilted, immobile or burning.
- Whoever moves a vehicle by `SetWorldTransform` calls `DC_CrashWatch.PrepareReposition(slot)` first (the driver:
  `DC_DriverContext.PrepareReposition()`), then logs the `teleport` row: the pending collision row is finished
  and the velocity history cleared, so the jump is never measured.
- The episode end waits at most 250 ms for the ego's collision row. Below about 8 frames per second the vanilla
  contact (1 to 2 frames after its step) can miss that wait; the row is then written with `src=dv` and without
  `other`.
- A destroyed ego ends the session with `end_reason` `ego_destroyed`, status `ok`: a crash is data. The crash rows,
  the end of the open episode and `session_end.json` share one terminal step, the last step with rows.

`src/enfusion/datacapture/DC_CollisionHook.c` is the **first `modded class` of this addon**:
`modded class SCR_VehicleDamageManagerComponent` overrides `ProcessCollision`. While the addon is loaded this runs for
every vehicle in every world, so:

- it calls `super.ProcessCollision` first and changes nothing: vanilla's damage, instigators and ejections stay as
  they are;
- it then hands the contact to `DC_CrashWatch.NoteContact`, which returns at once unless a dc-2.0 session is
  capturing and watches that vehicle (no allocation, no string);
- it is a manifest entry like every other script. To switch the contacts off, set `m_bUseContactHook 0` on
  `DC_CrashWatch` in the layer; the session then records `events.contact_source` `none` and names `other` from the
  registry only. To remove the hook completely, drop the manifest line as well.

The hook's signature is the same in the 1.9 index. The 1.9 method bodies around it (the impulse gates, the
continued-contact filter) must be checked again once the 1.9 scripts are extracted.

## Probe world DC_Probes_Arland (DD-4-probe)

`worlds/DataCapture/DC_Probes_Arland.ent` is a second world in the same addon, for the T0-ENG core engine probes
(capture program r2 4.4). Like DC_Arland it is a SubScene of vanilla Arland. One deploy carries both worlds;
DC_Arland and its capture producers are unchanged. In Workbench open
`$DataCapture:worlds/DataCapture/DC_Probes_Arland.ent` and press Play.

Its layer `DC_Probes_Arland_Layers/default.layer` holds:
- `GameMode_DCProbes` (the GameMode_Plain set of DC_Arland, no auto-spawn) with `DC_ProbeRunner` and the probe
  components `DC_ProbeEngine`, `DC_ProbeSequences`, `DC_ProbeImages`, `DC_ProbeMarkers` and `DC_RoadDump`;
- DC_Arland's `SCR_AIWorld`: its road network is what the road dump (e08a) reads;
- DC_Arland's chat, faction, radio and loadout managers and its two spawn points (inert without auto-spawn);
- the `ManualCamera`, behind the south end of the main runway, which is the test area of the driving probes;
- one S105 named `DC_Ego` at AI_Car_1's spot. The driving probes teleport it to the runway.

It has no `SCR_MLDataCollector`, no `SCR_MLScreenshotCapture`, no director and no AI groups, so a Play in this
world writes no `DrivingData` session.

Before the first probe Play:
- `<DATA_ROOT_WIN>\tools\launch\wb_R1.bat` must be the patched launcher (DD-0 follow-up). It copies the launch record
  to `<DATA_ROOT_WIN>\wb_profile\profile\dc_launch\` and passes `-dcLaunchId`. Without it every probe Play refuses and
  writes nothing.
- New script classes and a new world need a Workbench restart from `wb_R1.bat` after the deploy.

### Launch guard

`DC_LaunchGuard` decides at the start of every Play whether anything may be written. It writes nothing, not even
a folder, unless one of these holds:
- `-dcLaunchId <id>` is on the command line, `-profile` lies inside `<DATA_ROOT_WIN>`, in Workbench the resolved
  `$profile` lies there too (a `$profile` Workbench cannot resolve refuses), and `$profile:dc_launch/launch_<id>.json`
  is a valid launch record of this exe (`wb` in Workbench);
- the Workbench escape (`m_bAllowWorkbenchWithoutLaunchId 1` in this layer): the game cannot see the command line
  at all, `$profile` lies inside `<DATA_ROOT_WIN>`, and the newest `-wb-` record in `$profile:dc_launch` is valid, newer
  than the deploy and this Workbench start's own record: written at most 300 s before the process start, which the
  guard estimates as `GetUnixTime - GetTickCount / 1000` (`m_bEscapeBindToProcessStart 1`). Only the newest record
  is considered: when it belongs to another Workbench start the Play is refused, never joined to an older record.
  Every run header of such a Play says `launch.mode=wb_escape` and `launch.escape_binding=process_start`.
  - If the first Play shows that `GetTickCount` does not count from the Workbench start (the E-11 line's
    `uptime_s` is only the seconds since Play, and the refusal says the record was written long before "this
    Workbench started"), set `m_bEscapeBindToProcessStart` to 0 in the Object Properties for that Play (do not save
    the layer). The escape then takes the newest record younger than the deploy and marks
    `launch.escape_binding=newest_after_deploy`.

A visible command line without `-dcLaunchId` always refuses: Workbench was then not started by the patched launcher.

### Expected probe lines in console.log

1. `[DC_Probe] E-11: -profile visible=<0|1> -dcLaunchId visible=<0|1> $profile=<path or n/a> exe=wb uptime_s=<n>`.
   This line answers E-11; `uptime_s` (GetTickCount / 1000) shows whether the tick count runs from the Workbench
   start. It shows the profile path, which holds the Windows user name when Workbench was started from
   Steam: never paste it into a PR, a commit or a report.
2. Then either `[DC_Probe] launch guard: mode=cli id=<launch id> pin=<pin id>` (a WARNING with `mode=wb_escape`), or
   `[DC_Probe] ERROR: launch guard refused: <reason> - no probe runs, nothing written`. The usual reasons:
   - `started without -dcLaunchId: start Workbench with <DATA_ROOT_WIN>\tools\launch\wb_R1.bat`;
   - `-profile '...' is not inside <DATA_ROOT_WIN>` or `$profile '...' is not inside <DATA_ROOT_WIN>` (Workbench started
     from Steam);
   - `$profile could not be resolved ...`;
   - CLI mode: `no launch record $profile:dc_launch/launch_<id>.json ...` (the unpatched launcher, or the record was
     removed);
   - escape: `no launch record under $profile:dc_launch ...` (no `-wb-` record at all), `no launch record of this
     Workbench start: the newest, <id>, was written <n> s before this Workbench started ...` (this start's record was
     removed, or Workbench was not started by `wb_R1.bat`), or `... written <n> s after this Workbench started
     (another Workbench started later?)`;
   - `newest launch record <id> predates the deploy <utc>: restart Workbench with wb_R1.bat`.
   - Guard test (optional, no restart): rename this start's record
     `<DATA_ROOT_WIN>\wb_profile\profile\dc_launch\launch_<id>.json`, press Play, then Stop. In CLI mode the expected
     refusal is `no launch record $profile:dc_launch/launch_<id>.json ...`; in escape mode it is `no launch record
     under $profile:dc_launch ...` when no other `-wb-` record exists, else `no launch record of this Workbench
     start: ...` or `... predates the deploy ...`. No `DC_Probes` folder may appear. Rename the record back.
3. `[DC_Probe] plan from m_sPlan: '...' -> N probe(s) with a component: ...`, with a WARNING for every plan id that
   has no probe component in this deploy.
4. `[DC_Probe] run started (OnGameModeStart): plan=..., warm-up 3.0 s, gap 1.0 s`.
5. `[DC_Probe] runner heartbeat #<n> via=frame: world_ms=... step_source=undecided ...` about once per world second
   until the step source is decided, then every 30 world seconds while the plan runs. `via=fixedframe` or
   `via=watchdog` means that EOnFrame missed the heartbeat; `plan_calls` greater than `plan_entries` means that EOnFrame
   calls the sequencer (AdvancePlan) but the call does not reach it. The heartbeat runs on world time: while the world
   is paused it is silent, and after about 10 s the WARNING `runner: world time has not advanced for 10 watchdog
   ticks ...` appears instead (and `runner: world time advances again ...` once it moves). A WARNING `watchdog:
   AdvancePlan has not run for <s> world s (...): it stops <id> (timeout|write_fail)` means that a probe callback fails
   on every frame: send the log folder.
6. After the warm-up `[DC_Probe] step source: simulate (...)`, or on the probe world in 1.8 the WARNING
   `step source: fixedframe - no EOnSimulate on the game mode entity ...`, each with `decided by frame at world <ms> ms;
   step dt <s> s (<engine|measured>) ...`. An ERROR `step source: none` means that neither EOnSimulate nor
   EOnFixedFrame arrived. A WARNING `decision path: the frame path (EOnFrame -> AdvancePlan) did not decide the step
   source; the fixedframe|watchdog path decides it ...` means that the sequencer did not run: send the log folder.
7. For each exclusive probe: `[DC_Probe] plan: starting exclusive <i>/<n> <id> ...`, then
   `[DC_Probe <id>] START play_dir=$profile:DC_Probes/<id>/<launch id>/play_NN`, the probe's own `[DC_Probe <id>]`
   lines, `[DC_Probe <id>] END reason=done status=complete` and `[DC_Probe] plan: <id> ended (done, complete) after
   <s> world s; next exclusive ...`. Other ends:
   - `reason=timeout`;
   - `reason=write_fail`, with the requested and the returned byte counts of the failed write;
   - `reason=start_failed`, for example without `DC_Ego`, followed by `[DC_Probe] plan: <id> did not start ...`.
8. `[DC_Probe] plan complete: ...`. Stop the Play after this line. A Stop before it ends the running probes with
   `reason=play_end`.

### Where the probe files are

Every probe writes to `$profile:DC_Probes/<probe>/<launch id>/play_NN/`. NN = 01..99 is the first free number:
every Play of one Workbench start shares the launch id.
- With `wb_R1.bat` this is **`<DATA_ROOT_WIN>\wb_profile\profile\DC_Probes\`**, which is
  `<DATA_ROOT>/wb_profile/profile/DC_Probes/` in WSL.
- Each play folder holds:
  - `run_header.txt`: build stamp (git sha, dirty flag), launch id and pin, build version, fixed time slice,
    world, clocks (including the run's fixed and merged step counts at the probe start), and every probe parameter
    as `params.*`;
  - the probe's CSV files, each with one header row;
  - `run_footer.txt`, written when the probe ends: status, reason, rows and bytes per file, and the run's step
    timing (`clock.step_source`, `clock.step_dt_s` and the engine's raw values, see the DC_ProbeRunner header). A play
    folder without it was ended by a crash, a kill or a Stop that tore the world down.
- A refused Play creates no `DC_Probes` folder at all.

### Choosing the probes

The plan is `m_sPlan` on `GameMode_DCProbes` > `DC_ProbeRunner` (default `e10,e04,e20,e19,e08a,e01,e02a,e05,e06a`).
e10 runs alongside the others for the whole plan; the others run one after another, 1 s apart.
`-dcProbePlan <ids>` on the Workbench command line overrides `m_sPlan`, when the game can see the command line.
The probe parameters are attributes of the probe components, set in this layer, and every run header repeats them.

`e19kill` is never in the default plan. Run it on its own, last in a session: it waits for a
`taskkill /F /IM ArmaReforgerWorkbenchSteamDiag.exe`, after which Workbench must be restarted. For E-19's read-only
case, run `attrib +R <DATA_ROOT_WIN>\wb_profile\profile\DC_Probes\e19\ro_target.txt` (cmd) between two Plays with e19.
E-19 stays INCOMPLETE until both exist: an e19 Play whose c09 refused the write to the read-only target, and an
analysed e19kill Play (`e19_write.py <e19 play_dir> --kill <e19kill play_dir>`). e19 also measures whether WRITE
truncates a longer file (c10, the dc-2.0 heartbeat) and the open-APPEND-write-close cycle per line (c11); e19kill
writes its commit lines both through one open handle (`e19_commit.csv`) and with that cycle
(`e19_commit_append.csv`).

Each analysis prints one `[PASS]`, `[FAIL]`, `[INCOMPLETE]` or `[INFO]` line per item and ends with
`RESULT <probe>: PASS|FAIL|INCOMPLETE` (exit code 0, 1 or 3; 2 = usage or input error). Run them from the repo
with `python3 -I -B`; scripts without a folder are in `tools/probes/t0eng/`.

| id | measures | Play time | analysis |
|---|---|---|---|
| e10 | ticks per world-second, k | whole plan | `e10_ticks.py <play_dir>` |
| e04 | S105 components, gearbox | < 1 s | `e04_components.py <play_dir>` |
| e20 | render-settings dump | < 1 s | `e20_settings.py <play_dir> [--compare <client play_dir>] [--profile-dir <dir>]` |
| e19 | writer return values | < 5 s | `e19_write.py <play_dir>` |
| e19kill | taskkill consistency (opt-in) | until killed, max 300 s | `e19_write.py [<e19 play_dir>] --kill <play_dir>` |
| e08a | engine road dump | ~10-60 s | `tools/roads/compare_dump_decoder.py <play_dir> --decoder <Arland_road_polylines_v2.tsv>` |
| e01 | Mode A actuation identification | ~6 min | `e01_actuation.py e01 <play_dir>` |
| e02a | first-pass steer -> kappa | ~10-15 min | `e01_actuation.py e02a <play_dir>` |
| e05 | R1 camera, 10 Hz screenshots (E-05, E-06) | 10 min | `e05_images.py <play_dir> --log <console.log>`, `e06_fiducial.py <play_dir>` |
| e06a | marker reprojection alignment (60 and 30 FPS, backlog) | ~3-5 min | `tools/validator/e06a_reproject.py <play_dir> [--header-geometry]` |

Notes on the driving probes (e01, e02a, e05, e06a; all drive `DC_Ego` through `DC_ProbeActuator`):
- The S105 has a manual gearbox (`HasAutomaticGearbox` 0, 6 gear indices, 4 forward). Probe run 2 left it in
  neutral, so the throttle only revved the engine. The actuator now derives the indices once per probe start as
  vanilla does (`neutral = gears - forward - 1`, `first = gears - forward`, `top = gears - 1`; S105: 0 reverse,
  1 neutral, 2..5 first..fourth) and writes them to the run header (`vehicle.gearbox`, `vehicle.gear_index_*`).
- Without an explicit phase gear its shift policy picks the gear: neutral while the handbrake is on, first gear when
  it is released (before the throttle), then up at >= 4500 rpm and down at <= 1800 rpm, at least 1.5 s apart, first
  gear below 8 km/h, never neutral or reverse while driving (`DC_ProbeSequences` category Gearbox; e05 and e06a
  use the same values as defaults). `cmd_gear` in the step CSVs is the gear the policy chose. The move checks hold
  first gear, `gear_scripted` still calls `SetGear` every step, and `gear_observe` is skipped with the note
  `manual_gearbox`. The park/drive choice is the probe's own: vanilla `SCR_ToggleHandbrakeAction` releases into
  neutral and parks in first gear. After an explicit gear the policy continues from that gear, and every probe stop
  sets neutral on a manual gearbox.
- Each release of the handbrake also wakes the car's physics as vanilla does (`Physics.SetActive(ActiveState.ACTIVE)`),
  and every later step with the handbrake released wakes a body it finds asleep (`Physics.IsActive` false), as
  DD-4-min Mode A does. The phase notes (`wakes`, `asleep_at_wake`, `shifts`, `sleep_wakes`) and the run header lines
  appended at the stop (`gear_policy.*`, `physics_wake.*`) count these wakes and the gear changes. E-01's
  ForceEnableSimulation answer holds with both wakes; `e01_actuation.py` prints that condition after the answer.
- e05 and e06a latch `ForceEnableSimulation` unless e01 of the same Play found it not needed (`drive.force_sim`,
  `drive.force_sim_e01` in their run headers). If e01 found it not needed and a drive start stays below 3 km/h for
  10 s, the actuator's stuck guard latches it anyway and logs a WARNING (`drive.stuck_forces` at the stop).

Notes on the image probes:
- e05 and e06a drive from 2 x `m_fAreaMarginM` along the test area (6 m with the layer's margin), inside the area
  their drive check uses; a start point outside it refuses the probe at its start.
- At the end of the capture time (e05) or after the last FPS phase (e06a) no new screenshot is requested, and the
  requests in flight are drained with the rig camera and the bands still in place before the probe ends, so every
  `Screenshot saved` line of the play comes before its END line.
- E-05 also checks that the rig camera was current and attached on every request, and the vFOV read back.
- E-06A judges the grid images (slalom, brake, curve) and the backlog images separately at each FPS setting, and
  is INCOMPLETE when slalom, backlog or brake has no counted marker or its `e06a_phases.csv` row is not `ok`.
- The fiducial is laid out on `WorkspaceWidget.GetWidth() x GetHeight()`, which is the window's render target in
  physical pixels; the run header logs `fiducial.dpi_scale`.
- Fiducial carriers (`m_eFiducialCarrier`, e05 and e06a, default BOTH): WIDGET is the r2 3.4 ImageWidget bands,
  SHAPE draws the same code as debug Shapes 1 m (`m_fShapeDepthM`) in front of the rig camera at their own margin
  (`m_iShapeMarginPx`, 64 px: beside the widget bands), BOTH draws both. Probe runs 1 and 2 found no widget patch in
  any Workbench `MakeScreenshot` file, while debug shapes keep their exact colour there. The shapes are placed in
  EOnFrame, after the physics step and in EOnPostFrame (the last placement of a frame wins; the camera applies its
  transform in EOnPostFrame); the residual against the camera the frame was rendered with is logged once per second
  (`fiducial shapes:` lines) and summed up at the stop (`fiducial.shape.*` in the run header, `clock.postframe_calls`
  in the footer). The run header records `rig.fiducial.carrier` and `rig.fiducial.depth_m`. A shuttle or reset
  teleport places the shapes once more at once; teleport frames are counted apart (`fiducial.shape.teleport_*`). A
  canvas too small for the bands stops the drawing until the next size change (`fiducial.shape.layout_failures`,
  `drawing_at_stop`). `m_bShapeDepthTest` (default 0) drops `NOZBUFFER`/`NOZWRITE` for a Play that found the shapes
  absent; use it with `m_fShapeDepthM` just beyond the near plane (e.g. 0.11 m), since at 1 m the car body hides
  the bands.
- With the widget carrier, `m_bWidgetDiagnostics` reads the patch widgets back once per second (`fiducial widgets:`
  lines: workspace size and DPI, `IsVisibleInHierarchy`, `GetScreenPos` / `GetScreenSize`) and adds three orange
  control widgets at the top centre (a text, a textured and a flat image), so a Play tells "never drawn" from
  "drawn but not captured".
- `data/dc2/fiducial.py` gates every decode: a band of scene texture is `absent` (run 1 read dark bark as code 0
  `ok`), and a code outside [issue, min(issue + 2, done - 1)] is `implausible`. `e06_fiducial.py` and `e05_images.py`
  report each carrier on its own; E-06 selects the carrier with the highest decode rate and passes when it decodes
  >= 99.5% with no implausible decode (the other carriers' implausible decodes are listed, not judged).
