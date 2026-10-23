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
python3 -I -B tools/deploy_addon.py               # deploy to /mnt/e/DriveDiT/addons/DataCapture
python3 -I -B tools/deploy_addon.py --emcp-handlers <BK build>/mod/Scripts/WorkbenchGame/EnfusionMCP
python3 -I -B tools/test_deploy_addon.py -v       # self-test (temp targets only)
```

The tool also writes `Scripts/Game/DataCapture/DC_BuildInfo.c` (git sha, branch, dirty flag, deploy time, manifest
and content hashes) and tracks what it deployed in `<target>/.deploy_state.json`. It deploys only into a folder inside
`/mnt/e/DriveDiT/addons/` (`--allow-root DIR` adds a scratch root under `/mnt/e/` or `/dev/shm/`, for static checks and
the self-test) and never onto C: or the WSL disk image. It refuses files outside the manifest (case variants
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

`SCR_MLDataCaptureDirector` and `SCR_MLDataCollector` sit on `GameMode_DataCapture`. On Play, the director does this
for each group k = 1..5:

1. It spawns an **empty** group (`Group_US_Base`) at `AI_Spawn_k`.
2. It puts **one** driver (`Character_US_Rifleman`) straight into `AI_Car_k`'s driver seat. This is the vanilla Game
   Master "crewed" call `BaseCompartmentSlot.SpawnCharacterInCompartment`. Nobody walks to the car, so it works at
   any distance from the camera.
   - A failed attempt is retried up to 3 times, 1 s apart.
   - Only after that does the driver spawn on foot at `AI_Spawn_k` and walk to the car (walk-in fallback).
3. It keeps the group simulated wherever the camera is. 1.8 deactivates AI about 1 km or more from every observer.
   - The group is pinned to full AI LOD: `SetPermanentLOD(0)`, `PreventMaxLOD`, `CRITICAL` importance.
   - A temporary observer is put on the car (`ObserversSystem.InsertObserverSP`).
   - Both are undone when the director stops.
4. It sends the car between the `AI_Drive_Target_N` markers, starting once the driver exists.

A car counts as **ready** when a living driver sits in its driver seat **and** that driver's AI is active below max
LOD. A seated driver alone is not enough: direct seating puts the body in the seat even when its AI is switched off.

The collector session starts:
- as soon as all 5 drivers are ready (the normal case, about 10-11 s after the director start);
- or, if some groups failed to start or got no driver at all, as soon as every group that still can is ready;
- or when the 60 s boarding window ends, with the cars that are ready by then (at least `m_iMinSeatedGroups`).

A session with fewer than 5 cars does not meet the TKT-01 acceptance criteria. It still starts, as smoke data, but
logs `ERROR: TKT-01 acceptance NOT met` and one line per missing car. With no ready driver, or fewer than the
minimum, it logs an ERROR and starts no session.

60 s after the session start, the director checks that every car in the session has moved at least 20 m. Each car
that has not is logged as `ERROR: group k: AI_Car_k is in the collector session but not driving`.

After the boarding window the director also watches the driver seats. The vanilla AI sometimes brakes mid-route, sets
the handbrake and orders the driver out, then usually walks it back in 10-45 s later. With an obstructed door the
driver never gets back in. The most likely trigger, still unconfirmed, is a failed group move in the car (for example
`UNREACHABLE`); a car that cannot move or is on fire is another. The director logs each failed group move, each
dismount (with the last move failure and the car's movement damage and fire state) and each return. After
`m_fReseatAfterS` (30 s) with no living driver in the seat, it teleports the driver back in, or seats a new one if the
old driver is gone or dead. A dead driver is moved out of the seat first, and the group is kept when it becomes empty.
It does not reseat into a car that cannot move or is on fire, and it stops after 3 reseats in a row that did not get
the car moving 5 m. While the driver is out, the stuck clock pauses; it restarts when the vanilla AI gets back in by
itself. The statistics line counts `dismounts=`, `reseats=` and `move_failures=` per group.

Vehicles are registered in group order right before the session starts, so `vehicle_id` is k-1 when all 5 are
ready. Otherwise the ids are listed in the `starting the collector session with N vehicles: V0=AI_Car_1 ...` line and
in `session_info.txt`. A car whose driver becomes ready later keeps driving but is not captured in that session.

The collector writes a session only when Workbench was started with `-profile` inside `E:\DriveDiT` (the
`wb_R1.bat` launcher does this). Otherwise it logs `ERROR: capture output refused` at Play start and writes nothing,
so captures never land on C:.

Director attributes in the layer (the defaults need no layer change):

| Attribute | Default | Meaning |
|---|---|---|
| `m_sGroupPrefab` | `Group_US_Base` | Empty group per car. A prefab with unit slots logs a WARNING (extra soldiers on foot) |
| `m_sDriverPrefab` | `Character_US_Rifleman` | The driver |
| `m_iSeatAttempts` / `m_iSeatRetryMs` | 3 / 1000 | Direct seating attempts before walk-in boarding |
| `m_bPinDriverLOD` | 1 | Full AI LOD pin |
| `m_bVehicleObservers` | 1 | One temporary observer per car (SP keys 4400 + k) |
| `m_iBoardingWindowMs` | 60000 | Window after `Spawned N/M groups`. The session starts when all are ready, or when the window ends |
| `m_iMinSeatedGroups` | 1 | Ready cars needed when the window ends (limited to 1..`m_iMaxAIGroups`, with a WARNING). Set it to `m_iMaxAIGroups` for the old all-or-nothing gate |
| `m_iDrivingCheckMs` / `m_fDrivingCheckMinDistanceM` | 60000 / 20 | When the "is every car driving" check runs after the session start, and the distance each car must have moved. 0 turns the check off |
| `m_fReseatAfterS` | 30 | Seat watchdog: seconds without a living driver in the driver seat before the director puts one back, repeated while the seat stays empty. Never into a car that cannot move or is on fire, and at most 3 times in a row without the car moving `m_fStuckDistanceM` (5 m). 0 turns the reseat off; dismounts are still logged |

Collector attribute: `m_sRequiredProfileRoot` (default `E:/DriveDiT`). Sessions are written only when `-profile` lies
inside it. Empty turns the guard off; use that only deliberately.

### Expected lines in console.log

Booleans may print as `1`/`0` or as `true`/`false`.

1. `[MLDataCollector] Capture output allowed: -profile 'E:\DriveDiT\wb_profile' is inside E:/DriveDiT; sessions go to ...`.
   An `ERROR: capture output refused` line here means no capture will be written: relaunch with `wb_R1.bat`.
2. `[DataCapture] Director started (OnGameModeStart)`, then `AI LOD: max LOD <n> ...; driver LOD pin 1, vehicle observers 1`.
3. For each group, about 2 s apart:
   - `Group k spawned at AI_Spawn_k for vehicle AI_Car_k with 20 waypoints`;
   - `Group k: lod_pin=1 (permanent LOD 0, max LOD prevented, importance CRITICAL), observer=1 on AI_Car_k (key 440k)`;
   - `Group k: driver spawned in the pilot seat of AI_Car_k (direct seating, attempt 1/3) boarding=DIRECT agents=1 ... driver=alive pilot=1 ...`;
   - `Group k waypoint none -> N`.
4. `Spawned 5/5 groups - boarding window 60 s: the session starts once the drivers of all 5 started groups are ready ...`.
5. Five `Group k: driver ready in AI_Car_k (seated X s, AI active below max LOD Y s after the director start) ...`
   lines, each with `[lod=0/<max> perm=0 active=1]` and `driver=alive pilot=1`.
6. `All 5 drivers ready after X s - starting the collector session with 5 vehicles: V0=AI_Car_1 ... V4=AI_Car_5`. Then:
   - five `[MLDataCollector] Registered vehicle: AI_Car_k (index k-1)` lines;
   - `[MLDataCollector] Session initialized: $profile:DrivingData/session_<n>`;
   - `Session started. Tracking 5 vehicles`.

   With fewer than 5 cars ready, this line is instead:
   - `ERROR: TKT-01 acceptance NOT met: boarding window of 60 s ended with S/5 drivers ready (minimum 1) - starting a partial session without groups: ...`
     (or `... S/5 drivers ready after X s and no other group can get one ...` when groups failed outright);
   - one line per missing group: `group k (AI_Car_k): no seated driver, ...` or
     `group k (AI_Car_k): driver seated but its AI is not active below max LOD, ...`;
   - then `S/5 drivers ready after X s - starting the collector session with S vehicles: ...`.
7. About 60 s after the session start: `Driving check 60 s after the session start: all 5 registered vehicles moved at least 20 m: AI_Car_1=...m ...`.
8. Signs of trouble:
   - `direct seating attempt i/3 ... failed (pilot seat occupant=... accessible=...)`;
   - `falling back to walk-in boarding`;
   - `driver=dead` in a group line;
   - `ERROR: group k: AI_Car_k is in the collector session but not driving`;
   - `ERROR: boarding window of 60 s ended with no ready driver ... - collector session NOT started`;
   - `WARNING: group k: no living driver in the pilot seat of AI_Car_k (dismount n ...)` with no
     `Group k: driver back in the pilot seat of AI_Car_k after X s` line within about 30 s;
   - `reseating the driver ... failed`, `cannot reseat the driver`, `not reseating the driver` or
     `ERROR: group k: no more reseats into AI_Car_k`.

### Where the captures are

The collector writes to `$profile:DrivingData/session_<n>/`, where `<n>` is the world time in ms at the session
start. If that folder already exists from an earlier Play, the session uses `session_<n>_<unix time>` instead, so
nothing is overwritten.
- With the E: profile (`wb_R1.bat`, `-profile E:\DriveDiT\wb_profile`) this is
  **`E:\DriveDiT\wb_profile\profile\DrivingData\session_<n>\`**, which is
  `/mnt/e/DriveDiT/wb_profile/profile/DrivingData/` in WSL.
- Each session holds:
  - `session_info.txt`, written at the start: `data_class=smoke_bootstrap`, `harvest=never`, the build stamp and
    `vehicle_<id>=AI_Car_k`;
  - `telemetry.csv`, one row per car per 200 ms;
  - `frames/frame_NNNNNN.bmp.png`, one screenshot per 200 ms from the first car in the session (V0, normally
    `AI_Car_1`). The layer enables screenshots; each is about 0.5-1.6 MB, so roughly 10-30 GB per hour;
  - after Stop, `summary.txt`.
- No session means no `DrivingData` folder at all.
- Without `-profile` inside `E:\DriveDiT` (for example Workbench started from Steam), the collector refuses to write
  and logs `ERROR: capture output refused`. Relaunch with `wb_R1.bat`.

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
- `E:\DriveDiT\tools\launch\wb_R1.bat` must be the patched launcher (DD-0 follow-up). It copies the launch record
  to `E:\DriveDiT\wb_profile\profile\dc_launch\` and passes `-dcLaunchId`. Without it every probe Play refuses and
  writes nothing.
- New script classes and a new world need a Workbench restart from `wb_R1.bat` after the deploy.

### Launch guard

`DC_LaunchGuard` decides at the start of every Play whether anything may be written. It writes nothing, not even
a folder, unless one of these holds:
- `-dcLaunchId <id>` is on the command line, `-profile` lies inside `E:/DriveDiT`, in Workbench the resolved
  `$profile` lies there too (a `$profile` Workbench cannot resolve refuses), and `$profile:dc_launch/launch_<id>.json`
  is a valid launch record of this exe (`wb` in Workbench);
- the Workbench escape (`m_bAllowWorkbenchWithoutLaunchId 1` in this layer): the game cannot see the command line
  at all, `$profile` lies inside `E:/DriveDiT`, and the newest `-wb-` record in `$profile:dc_launch` is valid, newer
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
   - `started without -dcLaunchId: start Workbench with E:\DriveDiT\tools\launch\wb_R1.bat`;
   - `-profile '...' is not inside E:/DriveDiT` or `$profile '...' is not inside E:/DriveDiT` (Workbench started
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
     `E:\DriveDiT\wb_profile\profile\dc_launch\launch_<id>.json`, press Play, then Stop. In CLI mode the expected
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
- With `wb_R1.bat` this is **`E:\DriveDiT\wb_profile\profile\DC_Probes\`**, which is
  `/mnt/e/DriveDiT/wb_profile/profile/DC_Probes/` in WSL.
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
case, run `attrib +R E:\DriveDiT\wb_profile\profile\DC_Probes\e19\ro_target.txt` (cmd) between two Plays with e19.
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
| e08a | engine road dump | ~10-60 s | `tools/roads/compare_dump_decoder.py <play_dir> --decoder <Arland_road_polylines.tsv>` |
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
