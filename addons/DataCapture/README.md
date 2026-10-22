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
   - `ERROR: boarding window of 60 s ended with no ready driver ... - collector session NOT started`.

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
- `-dcLaunchId <id>` is on the command line, `-profile` lies inside `E:/DriveDiT`, and
  `$profile:dc_launch/launch_<id>.json` is a valid launch record of this exe (`wb` in Workbench);
- the Workbench escape (`m_bAllowWorkbenchWithoutLaunchId 1` in this layer): the game cannot see the command line
  at all, `$profile` lies inside `E:/DriveDiT`, and the newest `-wb-` record in `$profile:dc_launch` is valid and
  newer than the deploy. Every run header of such a Play says `launch.mode=wb_escape`.

A visible command line without `-dcLaunchId` always refuses: Workbench was then not started by the patched launcher.

### Expected probe lines in console.log

1. `[DC_Probe] E-11: -profile visible=<0|1> -dcLaunchId visible=<0|1> $profile=<path or n/a> exe=wb`. This line
   answers E-11. It shows the profile path, which holds the Windows user name when Workbench was started from
   Steam: never paste it into a PR, a commit or a report.
2. Then either `[DC_Probe] launch guard: mode=cli id=<launch id> pin=<pin id>` (a WARNING with `mode=wb_escape`), or
   `[DC_Probe] ERROR: launch guard refused: <reason> - no probe runs, nothing written`. The usual reasons:
   - `started without -dcLaunchId: start Workbench with E:\DriveDiT\tools\launch\wb_R1.bat`;
   - `-profile '...' is not inside E:/DriveDiT` or `$profile '...' is not inside E:/DriveDiT` (Workbench started
     from Steam);
   - `no launch record $profile:dc_launch/launch_<id>.json ...` or `no launch record under $profile:dc_launch` (the
     unpatched launcher, or the record was removed);
   - `newest launch record <id> predates the deploy <utc>: restart Workbench with wb_R1.bat`.
3. `[DC_Probe] plan from m_sPlan: '...' -> N probe(s) with a component: ...`, with a WARNING for every plan id that
   has no probe component in this deploy.
4. `[DC_Probe] run started (OnGameModeStart): plan=..., warm-up 3.0 s, gap 1.0 s`.
5. After the warm-up `[DC_Probe] step source: simulate (...)`. A WARNING `step source: fixedframe` or an ERROR
   `step source: none` means that the game mode entity gets no EOnSimulate.
6. For each probe: `[DC_Probe <id>] START play_dir=$profile:DC_Probes/<id>/<launch id>/play_NN`, the probe's own
   `[DC_Probe <id>]` lines, then `[DC_Probe <id>] END reason=done status=complete`. Other ends:
   - `reason=timeout`;
   - `reason=write_fail`, with the requested and the returned byte counts of the failed write;
   - `reason=start_failed`, for example without `DC_Ego`.
7. `[DC_Probe] plan complete: ...`. Stop the Play after this line. A Stop before it ends the running probes with
   `reason=play_end`.

### Where the probe files are

Every probe writes to `$profile:DC_Probes/<probe>/<launch id>/play_NN/`. NN = 01..99 is the first free number:
every Play of one Workbench start shares the launch id.
- With `wb_R1.bat` this is **`E:\DriveDiT\wb_profile\profile\DC_Probes\`**, which is
  `/mnt/e/DriveDiT/wb_profile/profile/DC_Probes/` in WSL.
- Each play folder holds:
  - `run_header.txt`: build stamp (git sha, dirty flag), launch id and pin, build version, fixed time slice,
    world, clocks, and every probe parameter as `params.*`;
  - the probe's CSV files, each with one header row;
  - `run_footer.txt`, written when the probe ends: status, reason, rows and bytes per file. A play folder without
    it was ended by a crash, a kill or a Stop that tore the world down.
- A refused Play creates no `DC_Probes` folder at all.

### Choosing the probes

The plan is `m_sPlan` on `GameMode_DCProbes` > `DC_ProbeRunner` (default `e10,e04,e20,e19,e08a,e01,e02a,e05,e06a`).
e10 runs alongside the others for the whole plan; the others run one after another, 1 s apart.
`-dcProbePlan <ids>` on the Workbench command line overrides `m_sPlan`, when the game can see the command line.
The probe parameters are attributes of the probe components, set in this layer, and every run header repeats them.

`e19kill` is never in the default plan. Run it on its own, last in a session: it waits for a
`taskkill /F /IM ArmaReforgerWorkbenchSteamDiag.exe`, after which Workbench must be restarted. For E-19's read-only
case, run `attrib +R E:\DriveDiT\wb_profile\profile\DC_Probes\e19\ro_target.txt` (cmd) between two Plays with e19.

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
| e06a | marker reprojection alignment (60 and 30 FPS, backlog) | ~3-5 min | `tools/validator/e06a_reproject.py <play_dir>` |
