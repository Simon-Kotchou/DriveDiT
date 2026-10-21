# Build pin and launchers (DD-0)

Ticket DD-0 is capture program r2 TKT-0 (`CAPTURE_PROGRAM.md` section 4.2 step 3 and Appendix B). It covers
decisions D1 and D2.

The campaign runs on **Arma Reforger 1.8.0.13**:

| App | appid | buildid | depot: manifest |
|---|---|---|---|
| Arma Reforger | 1874880 | 24903726 | 1874881: 7768706146966449237, 1874882: 1650233779375073015 |
| Arma Reforger Tools | 1874910 | 24870687 | 1874911: 6937477522608233045 |

Both apps have `AutoUpdateBehavior "0"`, so Steam may install 1.9 at any time. 1.9 rewrites the vehicle simulation,
which would split the data and the Everon test across two physics versions. This directory makes that update visible
and stops every launch until the build is back on the pin.

## Files

| Path | What it does |
|---|---|
| `tools/pin/make_pin.py` | Writes the pin JSON. It reads both appmanifests and both installs read-only. |
| `tools/pin/preflight.py` | Checks the pin before every launch. Every launcher calls it. |
| `tools/pin/pinlib.py` | Shared code: KeyValues (`.acf`) parser, path guard, file stats. |
| `tools/pin/cold_copy.sh` | Copies both installs to E: and writes `SHA256SUMS`. It is a dry run unless `--run` is passed. |
| `tools/pin/tests/` | Unit tests (stdlib `unittest`) with sanitized fixture appmanifests. |
| `tools/launch/wb_R1.bat` | Pinned Workbench launcher. |
| `tools/launch/client_R1_i01.bat` | Pinned game-client launcher, instance i01. |

The installed copies live on E:, and the launchers call those paths:

```
E:\DriveDiT\tools\pin\{pinlib,make_pin,preflight}.py, cold_copy.sh
E:\DriveDiT\tools\launch\wb_R1.bat, client_R1_i01.bat
E:\DriveDiT\pin\pin_1.8.0.13.json                     the pin
E:\DriveDiT\pin\launches\launch_<id>.json             one record per allowed launch
E:\DriveDiT\pin\launches\refused_<id>.json            one record per refused launch
E:\DriveDiT\pin\1.8.0.13\{game,tools,meta,SHA256SUMS}  cold copy (only after cold_copy.sh --run)
```

To update the installed copies, copy the files from the repo. Do not edit them on E:.

## Write safety

Every write goes through `pinlib.check_write_path`, which resolves symlinks first. It accepts only paths under
`/mnt/e/`, and it always refuses `/mnt/c/` (C: is nearly full and holds the live Workbench addon). `make_pin.py`
never creates directories and never overwrites a pin unless `--force` is given. `cold_copy.sh` has the same guard.
The installs and appmanifests on C: and D: are only read.

## 1. Make the pin

```bash
mkdir -p /mnt/e/DriveDiT/pin
python3 -I -B /mnt/e/DriveDiT/tools/pin/make_pin.py --pin-id 1.8.0.13 \
  --out /mnt/e/DriveDiT/pin/pin_1.8.0.13.json \
  --expect-game-buildid 24903726 --expect-tools-buildid 24870687
```

The pin records the following for each app:
- appid, buildid, TargetBuildID and StateFlags;
- the AutoUpdateBehavior at pin time;
- depot manifests;
- the appmanifest and install paths, in both WSL and Windows form;
- `steam_appid.txt`;
- the launched exes;
- every `*.pak`, keyed by relative path, with `size`, `mtime_ns`, `mtime_utc` and `mtime_local`.

It leaves out `LastOwner`, which is a personal Steam id.

`make_pin.py` refuses to pin a moving target. It exits 1 if a buildid differs from `--expect-*`, if
TargetBuildID != buildid (an update is pending), if StateFlags != 4, or if there are no paks. With the default
`AutoUpdateBehavior "0"` it still writes the pin but warns.

## 2. Preflight (every launch)

```
python3 -I -B preflight.py --pin PIN --exe wb|client [--instance i01]     # launch mode
python3 -I -B preflight.py --pin PIN --check                              # report only, writes nothing
```

The preflight refuses unless all of these hold, for **both** apps on every launch. Workbench also reads the game's
data paks.

| Check | Refuses when |
|---|---|
| `buildid` | buildid != pin (Steam installed another build) |
| `target_buildid` | TargetBuildID != buildid (an update is pending or downloading) |
| `auto_update_behavior` | AutoUpdateBehavior is not `"1"` ("Only update this game when I launch it"). `"0"` is refused even if the pin allows it |
| `state_flags` | StateFlags != 4 (not settled) |
| `depots` | any installed depot manifest != pin |
| `paks` | any pinned pak is missing, or its size or `mtime_ns` differs, or a pak exists that is not in the pin |
| `binaries` | a launched exe differs in size or mtime |
| `steam_appid_txt` | `steam_appid.txt` changed or is missing (direct launches depend on it) |
| `acf`, `appid` | the appmanifest is unreadable or belongs to another app |

**Launch mode.** On success it prints only the launch id (for example `20261021T063012Z-client-3f9a`) on stdout and
writes `launch_<id>.json` next to the pin, under `launches/`. On refusal it prints nothing on stdout, explains on
stderr, and writes `refused_<id>.json`.

**Exit codes.** 0 means OK. 1 means refused. 2 means a usage or config error, such as an unreadable pin or a bad
path. 3 means the launch record could not be written, so the launch is refused.

**Record contents.** Each record (`dd-launch/1`) holds the status, exe, instance, the pin path, id and sha256, the
observed appmanifest fields, every check with its detail, the host, the WSL distro and the Python version. The
client passes the id to the game as `-dcLaunchId`, which joins a session to its record (validator check C02).

**Check-only options.** `--allow-fail CHECK` and `--mtime-resolution-s N` are accepted only with `--check`. A launch
can never waive a check. `cold_copy.sh` uses both.

**Today's expected result.** Until the Steam setting below is changed, the preflight refuses with
`game.auto_update_behavior` and `tools.auto_update_behavior`.

### User action that unblocks launches (P0 step 1)

In Steam, open the library, then Arma Reforger, Properties, Updates. Set Automatic updates to **"Only update this
game when I launch it"**. Do the same for Arma Reforger Tools. Then check that both appmanifests show
`"AutoUpdateBehavior" "1"`:

```bash
grep AutoUpdateBehavior /mnt/d/SteamLibrary/steamapps/appmanifest_1874880.acf \
  "/mnt/c/Program Files (x86)/Steam/steamapps/appmanifest_1874910.acf"
python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin /mnt/e/DriveDiT/pin/pin_1.8.0.13.json --check
```

The value mapping (0 = always, 1 = on launch, 2 = high priority) is community-documented; the Valve docs are
UNVERIFIED. Steam has no "never update" option. For overnight runs, also put Steam in **offline mode** (D2c).

## 3. Launchers

Both launchers start the exe directly, never through `steam://`. They set the exe's own directory as the working
directory, as a Steam launch does, and that directory holds `steam_appid.txt`. Before starting, they run the
preflight through `wsl.exe -d Ubuntu-22.04 -e python3 -I -B ...`. The distro is named because a second one,
`Ubuntu-22.04-E`, exists. They start the exe only when the preflight printed an id **and**
`launch_<id>.json` exists. The second check rejects a stray stdout line, such as a wsl.exe warning.

**`wb_R1.bat`** runs:

```
ArmaReforgerWorkbenchSteamDiag.exe -profile "E:\DriveDiT\wb_profile" -gproj "E:\DriveDiT\addons\DataCapture\addon.gproj" -forceUpdate -maxFPS 60
```

It refuses while a Workbench is already running, and while the gproj is not deployed (DD-1).

**`client_R1_i01.bat`** runs:

```
ArmaReforgerSteam.exe -profile "E:\DriveDiT\client_profiles\i01" -logsDir "E:\DriveDiT\logs\i01" -cfg "E:\DriveDiT\tools\cfg\engine_R1.conf" -window -screenWidth 1408 -screenHeight 720 -posX 0 -posY 0 -forceUpdate -noFocus -maxFPS 60 -GPUAdapter 0 -freezeCheck 120 -freezeCheckMode kill -noThrow -dcLaunchId <id> -addonsDir "E:\DriveDiT\addons" -addons 88BEF4187E0D2DFC -world "<DC_WORLD>"
```

It refuses until `DC_WORLD` is filled in (`{GUID}worlds/DataCapture/DC_Arland.ent`, from TKT-2-min) and
`engine_R1.conf` exists (probe E-20).

Every engine flag comes from the Startup Parameters wiki revision cited in `trial-ladder.md` 8.2. The flags are
UNVERIFIED on 1.8.0.13 until probe S1, and each one is annotated in the `.bat`. The probes involved are P-02
(unpacked addon), P-03 (PNG size), P-06 (concurrent instances), P-12 (`-world` starts the game mode) and P-13
(`-forceUpdate` keeps frames fresh). `-noBackend` is left out: add it only for capture-only runs.

It is UNVERIFIED (capture program 4.2 item 2) that a direct exe launch never makes Steam update or relaunch the app.
Steam must be running.

## 4. Cold copy (decision D2b)

```bash
bash /mnt/e/DriveDiT/tools/pin/cold_copy.sh          # dry run: checks the installs against the pin and prints the plan
bash /mnt/e/DriveDiT/tools/pin/cold_copy.sh --run    # copies about 27.9 GB to E:\DriveDiT\pin\1.8.0.13
```

`--run` does the following, in order:
1. Checks the live installs against the pin. AutoUpdateBehavior is waived; everything else must pass.
2. Runs `rsync -rt` into `game/` and `tools/`. A marker file allows an interrupted copy to be resumed.
3. Checks the live installs again, to show that nothing changed during the copy.
4. Checks the copy against the pin at 1 s mtime resolution. WSL drvfs truncates the mtimes it *sets* to whole
   seconds; this was measured on E:.
5. Writes `SHA256SUMS` for every copied file.
6. Copies the pin and both appmanifests into `meta/`.

Verify the copy later with `cd /mnt/e/DriveDiT/pin/1.8.0.13 && sha256sum -c --quiet SHA256SUMS`. The depot manifest
ids in the pin also allow a third-party DepotDownloader rollback. Running that tool is the user's decision (D2d).

## Tests

```bash
python3 -I -B -m unittest discover -s tools/pin/tests -v
```

Temp trees go to `$DD_PIN_TEST_TMP`, otherwise `/dev/shm`, and are deleted afterwards. Run the tests with
`/usr/bin/python3` (3.10) as well, because `wsl.exe -e python3` resolves to it: `-e` starts no login shell.

## Differences from the Appendix B sketch

- The preflight accepts only AutoUpdateBehavior `"1"`. Appendix B forbids only `"0"`, but `"2"` (high priority) also
  auto-updates.
- The preflight also checks StateFlags, depot manifests, exes and `steam_appid.txt`. It also refuses a pak that is
  not in the pin.
- In the pin, `depots` maps each depot to `{manifest, size}`, and `paks` is keyed by relative path, because there are
  two `data.pak` files: `addons/core` and `addons/data`. Mtimes are stored as `mtime_ns` and compared exactly.
- The launchers name the WSL distro, set the working directory to the exe's directory (`start /D`), require the
  launch record to exist, and refuse while placeholders are unset. `wb_R1.bat` refuses a second Workbench.
- The preflight writes a record for refused launches as well.
