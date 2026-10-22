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

The collector session starts as soon as all 5 drivers are seated. Otherwise it starts when the boarding window
ends, with the cars that have a driver by then. With no driver, or fewer than the minimum, it logs an ERROR and starts
no session.

Vehicles are registered in group order right before the session starts, so `vehicle_id` is k-1 when all 5 are
seated. Otherwise the ids are listed in the `starting the collector session with N vehicles: V0=AI_Car_1 ...` line and
in `session_info.txt`. A car that gets its driver later keeps driving but is not captured in that session.

Director attributes in the layer (the defaults need no layer change):

| Attribute | Default | Meaning |
|---|---|---|
| `m_sGroupPrefab` | `Group_US_Base` | Empty group per car. A prefab with unit slots logs a WARNING (extra soldiers on foot) |
| `m_sDriverPrefab` | `Character_US_Rifleman` | The driver |
| `m_iSeatAttempts` / `m_iSeatRetryMs` | 3 / 1000 | Direct seating attempts before walk-in boarding |
| `m_bPinDriverLOD` | 1 | Full AI LOD pin |
| `m_bVehicleObservers` | 1 | One temporary observer per car (SP keys 4400 + k) |
| `m_iBoardingWindowMs` | 60000 | Window after `Spawned N/M groups`. The session starts when all are seated, or when the window ends |
| `m_iMinSeatedGroups` | 1 | Seated cars needed when the window ends. Set it to `m_iMaxAIGroups` for the old all-or-nothing gate |

### Expected `[DataCapture]` lines in console.log

Booleans may print as `1`/`0` or as `true`/`false`.

1. `Director started (OnGameModeStart)`, then `AI LOD: max LOD <n> ...; driver LOD pin 1, vehicle observers 1`.
2. For each group, about 2 s apart:
   - `Group k spawned at AI_Spawn_k for vehicle AI_Car_k with 20 waypoints`;
   - `Group k: lod_pin=1 ..., observer=1 on AI_Car_k (key 440k)`;
   - `Group k: driver spawned in the pilot seat of AI_Car_k (direct seating, attempt 1/3) boarding=DIRECT agents=1 ... [lod=<l>/<max> perm=0 active=1] pilot=1 ...`;
   - `Group k waypoint none -> N`.
3. `Spawned 5/5 groups - boarding window 60 s: ...`.
4. Five `Group k: driver seated in AI_Car_k (first seen X s after the director start) ...` lines, each with `pilot=1`
   and `[lod=0/<max> perm=0 active=1]`. The statistics block every 60 s repeats this per group, and groups 3-5 must
   show `distance` > 0.
5. `All 5 drivers seated after X s - starting the collector session with 5 vehicles: V0=AI_Car_1 ... V4=AI_Car_5`. Then:
   - five `[MLDataCollector] Registered vehicle: AI_Car_k (index k-1)` lines;
   - `[MLDataCollector] Session initialized: $profile:DrivingData/session_<n>`;
   - `Session started. Tracking 5 vehicles`.

   With fewer than 5 cars seated, this line is instead:
   - `WARNING: boarding window of 60 s ended with S/5 drivers seated (minimum 1) - starting the session without groups: ...`,
     followed by one line per missing group;
   - then `S/5 drivers seated after X s - starting the collector session with S vehicles: ...`.
6. Signs of trouble:
   - `direct seating attempt i/3 ... failed (pilot seat occupant=... accessible=...)`;
   - `falling back to walk-in boarding`;
   - `ERROR: boarding window of 60 s ended with no seated driver ... - collector session NOT started`.

### Where the captures are

The collector writes to `$profile:DrivingData/session_<n>/`, where `<n>` is the world time in ms at the session
start.
- With the E: profile (`wb_R1.bat`, `-profile E:\DriveDiT\wb_profile`) this is
  **`E:\DriveDiT\wb_profile\profile\DrivingData\session_<n>\`**, which is
  `/mnt/e/DriveDiT/wb_profile/profile/DrivingData/` in WSL.
- Each session holds `session_info.txt` (written at the start: `data_class=smoke_bootstrap`, `harvest=never`, the
  build stamp and `vehicle_<id>=AI_Car_k`), `telemetry.csv` (one row per car per 200 ms) and, after Stop,
  `summary.txt`.
- No session means no `DrivingData` folder at all.
- If console.log shows the profile under OneDrive, sessions land on C: instead. Stop and relaunch with `wb_R1.bat`.
