# Build pin and launchers (DD-0)

Ticket DD-0 is capture program r2 TKT-0 (`CAPTURE_PROGRAM.md` section 4.2 step 3 and Appendix B). It covers
decisions D1 and D2.

The rig runs on **Arma Reforger 1.8.0.13**, as an **interim** pin:

| App | appid | buildid | depot: manifest |
|---|---|---|---|
| Arma Reforger | 1874880 | 24903726 | 1874881: 7768706146966449237, 1874882: 1650233779375073015 |
| Arma Reforger Tools | 1874910 | 24870687 | 1874911: 6937477522608233045 |

1.9 rewrites the vehicle simulation. The 1.9 Experimental notes of 2026-09-15 and 2026-09-18 list Ackermann steering,
a Pacejka tyre model and a powertrain rewrite. The target is to run the system on 1.9's vehicle physics once 1.9
reaches the stable branch and compatibility is checked (see "1.8 is an interim pin; moving to 1.9" below). Until
then, an update that nobody chose would split the data across two physics versions.

This directory does three things:
- it makes any build change visible;
- it stops every launch until the installs match the **active pin**;
- it makes the move to a new build a deliberate, checked switch (`pin_activate.py`), never a side effect of Steam.

## Files

| Path | What it does |
|---|---|
| `tools/pin/make_pin.py` | Writes a pin JSON. It reads both appmanifests and both installs read-only. |
| `tools/pin/pin_activate.py` | Switches the active pin (`E:\DriveDiT\pin\ACTIVE`), after a full check against the live installs. |
| `tools/pin/preflight.py` | Checks the active pin before every launch. Every launcher calls it with `--pin-active`. |
| `tools/pin/pinlib.py` | Shared code: KeyValues (`.acf`) parser, path guard, file stats, the ACTIVE pointer. |
| `tools/pin/cold_copy.sh` | Copies both installs of the active pin to E: and writes `SHA256SUMS`. It is a dry run unless `--run` is passed. |
| `tools/pin/pin_sidecar.py` | DD-14a. Writes the write-once sidecars of a pin: `pin_<id>.build_key.json` (the build_key of a dd-pin/1 pin) and `pin_<id>.engine.json` (`engine_version_expected`). Dry run or `--apply`. |
| `tools/pin/freeze_check.py` | DD-14a. The direct-launch freeze check of MIGRATION_1_9 5.5 (CC-01): `before`, a user launch, `after`. |
| `tools/pin/tests/` | Unit tests (stdlib `unittest`) with sanitized fixture appmanifests. |
| `tools/launch/wb_R1.bat` | Pinned Workbench launcher. |
| `tools/launch/client_R1_i01.bat` | Pinned game-client launcher, instance i01. |
| `tools/launch/wb_C18.bat` | DD-14a. Cold-run Workbench launcher: 1.8.0.13 from the working copy `E:\DriveDiT\pin\1.8.0.13-run`, explicit pin `1.8.0.13-cold`. |
| `tools/launch/client_C18_i01.bat` | DD-14a. Cold-run client launcher, instance c18_i01, always `-noBackend`. |

The installed copies live on E:, and the launchers call those paths:

```
E:\DriveDiT\tools\pin\{pinlib,make_pin,preflight,pin_activate}.py, cold_copy.sh
E:\DriveDiT\tools\launch\wb_R1.bat, client_R1_i01.bat
E:\DriveDiT\pin\pin_1.8.0.13.json                     a pin (pins are immutable; one file per build)
E:\DriveDiT\pin\pin_1.9.0.37.build_key.json           (DD-14a) build_key sidecar of a dd-pin/1 pin (write-once)
E:\DriveDiT\pin\pin_1.9.0.37.engine.json              (DD-14a) engine_version_expected sidecar (write-once)
E:\DriveDiT\pin\freeze\freeze_<id>_<UTC>.{before,after}.json   (DD-14a) freeze-check records
E:\DriveDiT\pin\pin_1.8.0.13-r1.json                  (only after a rollback) a re-pin of restored 1.8 installs; see "Rollback"
E:\DriveDiT\pin\ACTIVE                                one line: the active pin id (written only by pin_activate.py)
E:\DriveDiT\pin\ACTIVE.log                            one line per switch: <UTC time> <old> -> <new> <user>@<host>
E:\DriveDiT\pin\ACTIVE.lock                           exists only while pin_activate.py runs
E:\DriveDiT\pin\launches\launch_<id>.json             one record per allowed launch
E:\DriveDiT\pin\launches\refused_<id>.json            one record per refused launch
E:\DriveDiT\pin\1.8.0.13\{game,tools,meta,SHA256SUMS}  cold copy (only after cold_copy.sh --run)
```

To update the installed copies, copy the files from the repo. Do not edit them on E:.

## Threat model (DD-14a)

These tools defend against accidental misuse, stale state, wrong-build or wrong-root mistakes and partial writes: a
launcher started against the wrong install, a pin or index of another build, a forgotten sidecar, an update Steam
slipped in, a deploy into the wrong addon root, a write cut short by a full disk or a killed process. They do **not**
defend against a malicious local user with write access to E:: such a user can rewrite a pin, its sidecars, a
SHA256SUMS file or the tools themselves. Hashes and bindings here catch accidents (a changed exe, a moved tag, a
different launch), not a forger. One known limit of that kind: the trust in a cold copy's SHA256SUMS starts when it
is first used. A copy whose exe was replaced and whose SHA256SUMS was regenerated before the first sidecar or cold pin
is accepted; from then on the recorded sha256 of that SHA256SUMS catches any change (Codex round 2, LOW,
forgery-only).

## Write safety

Every write goes through `pinlib.check_write_path`, which resolves symlinks first. It accepts only paths under
`/mnt/e/`, and it always refuses `/mnt/c/` (C: is never written, a project rule, and it holds the live Workbench addon). `/mnt/e` is only a
name, so the guard also reads `/proc/self/mountinfo`: the path must lie on the mount at `/mnt/e` whose source is the
`E:` drive (`pinlib.WRITE_ROOT_DRIVES`). An unmounted `/mnt/e` (an empty directory on the WSL disk, which is on C:),
a redirected one, or another filesystem mounted below it is refused.

`make_pin.py` never creates directories and never overwrites a pin unless `--force` is given. `pin_activate.py`
writes only `ACTIVE`, a temp file next to it, `ACTIVE.log` and `ACTIVE.lock`, in the pin directory. It opens the pin
directory once (`pinlib.open_guarded_dir`: the guard runs again on the directory the fd really refers to, which must
be on the E: mount) and makes every create, rename and unlink relative to that fd. A pin directory swapped for a
symlink after the check therefore cannot redirect a write. On E: (drvfs), a directory renamed under the open fd makes
every such call fail (ENXIO, measured), so the switch fails closed. `ACTIVE` is replaced by rename, and the log and
the lock are opened with `O_NOFOLLOW`, so a symlink at any of these names is never written through. `cold_copy.sh`
checks its destination with the same `check_write_path`. The installs and appmanifests on C: and D: are only read.

## 1. Make a pin

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
TargetBuildID != buildid (an update is pending), if StateFlags != 4, or if there are no paks. If AutoUpdateBehavior
is not `"1"`, it still writes the pin but warns.

Make launch pins from the default appmanifests (`--game-acf`/`--tools-acf` left out). A pin made from any other tree
fails the `launch_path` check (section 3), so it can never be activated or launched; such a pin is for `--check`
only. `--restore-of PIN_ID` re-pins restored installs after a rollback; see "Rollback" below.

**Pin ids.** `--pin-id` must be 2 to 4 dot-separated numbers with an optional lowercase suffix, such as `1.8.0.13`,
`1.9.0.12` or `1.9.0.9-exp`. Anything else is refused (exit 2): the id names `pin_<id>.json` and the cold-copy
directory, so it may hold no path separator, no `..` and no whitespace. Name the file `pin_<id>.json` in the pin
directory; `make_pin.py` warns otherwise, because `pin_activate.py` and `--pin-active` look only there.

Making a pin does **not** change what the launchers check. That takes step 2.

## 2. Activate a pin (the ACTIVE pointer)

```bash
python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_activate.py 1.8.0.13
```

`E:\DriveDiT\pin\ACTIVE` holds one line, the active pin id. The launchers run `preflight.py --pin-active`, which
checks `E:\DriveDiT\pin\pin_<id>.json`. No launcher names a game version, so moving to a new build never needs a
launcher edit, as long as Steam keeps the installs where the launchers start them (`pinlib.LAUNCH_PATHS`; see
section 4). `pin_activate.py` is the only supported way to change `ACTIVE`; never edit it by hand.

`pin_activate.py <id>` refuses, and changes nothing, unless all of these hold:
- the id is valid;
- `pin_<id>.json` is a regular file, not a symlink, and its `pin_id` is that id;
- no other activation is running: it creates `ACTIVE.lock` with `O_CREAT|O_EXCL` and holds it from reading the old
  `ACTIVE` to the read-back, so two activations are serialized (the second exits 2). A lock left by a killed run is
  never removed automatically: if no `pin_activate.py` is running, check `ACTIVE` and `ACTIVE.log`, then delete
  `ACTIVE.lock` by hand;
- the **full preflight check** of that pin passes against the live installs it names, with no waivers. That includes
  `launch_path`: the pin must have been made from the installs the launchers start. A pin that does not match what
  Steam has installed cannot be activated. This applies both ways: a switch back to 1.8.0.13 is refused while the
  installs are on another build, and also after a restore (see "Rollback");
- an existing `ACTIVE` or `ACTIVE.log` is a regular file. A symlink is left for a human to inspect;
- `ACTIVE.log` can be opened for appending. It is opened **before** `ACTIVE` changes, so a log that cannot be written
  to (an ACL, a read-only attribute) refuses the switch with `ACTIVE` unchanged (exit 3).

On success it replaces `ACTIVE` atomically (a temp file, then a rename in the same directory). It then appends
`<UTC ISO time> <old> -> <new> <user>@<host>` to `ACTIVE.log`, for example
`2026-10-21T08:15:02Z none -> 1.8.0.13 user@host`. `<old>` is `none` when there was no `ACTIVE`, and
`invalid` when it held no valid id. If the id is already active, it re-checks and writes nothing, unless the last log
line does not record a switch to that id. That happens when a run switched `ACTIVE` but could not write its log
line (a disk error after the log was opened), or when `ACTIVE` was edited by hand. It then appends
`<UTC ISO time> unlogged -> <id> <user>@<host>`, so every switch reaches the log.

Exit codes: 0 activated, already active, or the missing log line appended; 1 refused (a check failed); 2 usage,
config, path or lock error; 3 a write failed (the message says whether `ACTIVE` changed).

**How the preflight reads the pointer.** `--pin-active` refuses (exit 2, nothing on stdout, no launch record) when:
- `ACTIVE` is missing, empty or not plain ASCII;
- it holds more than one line, a stray carriage return, or an invalid id (one trailing LF or CRLF is allowed);
- `ACTIVE` is a symlink or not a regular file;
- `pin_<id>.json` is missing or a symlink, or holds another `pin_id`.

`--pin-dir DIR` overrides the default pin directory `/mnt/e/DriveDiT/pin` (tests and the harness use it). `--pin PATH`
still checks an explicit pin file, and `--pin` and `--pin-active` are mutually exclusive.

## 3. Preflight (every launch)

```
python3 -I -B preflight.py --pin-active --exe wb|client [--instance i01]     # launch mode (what the launchers run)
python3 -I -B preflight.py --pin-active --check                              # report only, writes nothing
python3 -I -B preflight.py --pin PIN --check                                 # an explicit pin file instead
```

The preflight refuses unless all of these hold, for **both** apps on every launch. Workbench also reads the game's
data paks.

| Check | Refuses when |
|---|---|
| `buildid` | buildid != pin (Steam installed another build) |
| `target_buildid` | TargetBuildID != buildid (an update is pending or downloading) |
| `auto_update_behavior` | AutoUpdateBehavior is not `"1"` ("Only update this game when I launch it"). A pin can only narrow this: `"0"` and `"2"` are refused even if the pin lists them |
| `state_flags` | StateFlags != 4 (not settled) |
| `depots` | any installed depot manifest != pin |
| `paks` | any pinned pak is missing, or its size or `mtime_ns` differs, or a pak exists that is not in the pin |
| `binaries` | a launched exe differs in size or mtime |
| `steam_appid_txt` | `steam_appid.txt` changed or is missing (direct launches depend on it) |
| `acf`, `appid` | the appmanifest is unreadable or belongs to another app |
| `launch_path` | the appmanifest or install dir stored in the pin is not the one the launchers start (`pinlib.LAUNCH_PATHS`). Every other check reads the paths stored in the pin, so without this one a pin made from another tree could vouch for an exe it never checked |

**Launch mode.** On success it prints only the launch id (for example `20261021T063012Z-client-3f9a`) on stdout and
writes `launch_<id>.json` next to the pin, under `launches/`. On refusal it prints nothing on stdout, explains on
stderr, and writes `refused_<id>.json`.

**Check mode.** The first line names the pin and how it was found, for example
`PIN    pin 1.8.0.13 via active pointer /mnt/e/DriveDiT/pin/ACTIVE -> /mnt/e/DriveDiT/pin/pin_1.8.0.13.json (sha256 ...)`.
Then come the PASS/FAIL lines and `RESULT: OK (pin <id>)` or `RESULT: REFUSED ...`.

**Exit codes.** 0 means OK. 1 means refused. 2 means a usage or config error, such as an unreadable pin, a bad
`ACTIVE` or a bad path. 3 means the launch record could not be written, so the launch is refused.

**Record contents.** Each record (`dd-launch/1`) holds:
- the status, exe and instance;
- the pin: `path`, `pin_id`, `sha256` (of the exact bytes checked), `resolved_by` (`active-pointer` or
  `explicit-path`) and `pointer` (the `ACTIVE` file, or null);
- the observed appmanifest fields and every check with its detail;
- the host, the WSL distro and the Python version.

The client passes the id to the game as `-dcLaunchId`, which joins a session to its record and therefore to its build
(validator check C02).

**Check-only options.** `--allow-fail CHECK`, `--mtime-resolution-s N` and the path overrides `--game-acf`,
`--tools-acf`, `--game-root` and `--tools-root` are accepted only with `--check`. A launch can never waive a check,
and is always checked against the pinned appmanifests and install dirs, so its record describes the files the
launcher starts. `cold_copy.sh` uses these options to check the copy.

**Today's result.** Since 2026-10-21 both apps have `AutoUpdateBehavior "1"`, and `--pin-active --check` on pin
1.8.0.13 passes all 22 checks.

### User action that unblocks launches (P0 step 1, done 2026-10-21)

In Steam, open the library, then Arma Reforger, Properties, Updates. Set Automatic updates to **"Only update this
game when I launch it"**. Do the same for Arma Reforger Tools. Then check that both appmanifests show
`"AutoUpdateBehavior" "1"`:

```bash
grep AutoUpdateBehavior /mnt/d/SteamLibrary/steamapps/appmanifest_1874880.acf \
  "/mnt/c/Program Files (x86)/Steam/steamapps/appmanifest_1874910.acf"
python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --check
```

The value mapping (0 = always, 1 = on launch, 2 = high priority) is community-documented; the Valve docs are
UNVERIFIED. Steam has no "never update" option. For overnight runs, also put Steam in **offline mode** (D2c). Keep
the setting at "1" for good, including across the move to 1.9: a build change should happen only when you start it.

## 4. Launchers

Both launchers start the exe directly, never through `steam://`. They set the exe's own directory as the working
directory, as a Steam launch does, and that directory holds `steam_appid.txt`. Before starting, they run the
preflight through `wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B .../preflight.py --pin-active ...`. The launch
logic names no game version: the build it accepts is whatever pin `ACTIVE` names. The distro is named because a second
one, `Ubuntu-22.04-E`, exists. The interpreter is named because `-e` starts no shell and the PATH it gets also holds
Windows directories. They start the exe only when the preflight printed an id **and** `launch_<id>.json` exists.

**Launch paths.** The launchers start fixed exes: `wb_R1.bat` `WB_DIR` (the Tools install plus `\Workbench`),
`client_R1_i01.bat` `GAME_DIR`, and DD-3's `launch-workbench.sh` `WB_DIR`. `pinlib.LAUNCH_PATHS` names the same
installs and appmanifests, and the preflight's `launch_path` check requires the active pin to have been made from
exactly them. So the exe a launcher starts is always the one its pin checked, and every launch record that credits a
pin is true. A unit test compares `LAUNCH_PATHS` with the `.bat` files. If an install ever moves (another Steam
library, another drive), change the launchers and `LAUNCH_PATHS` together, then make and activate a new pin.

How the id is read (review fix). wsl.exe writes its own messages, such as an unknown distro, to **stdout**, in UTF-16
unless `WSL_UTF8=1` is set (both measured). The launchers set `WSL_UTF8=1` and pass the preflight's stdout through
two `findstr` filters: the id pattern anchored at the start of the line, then `/V` on any character outside the id
set. `findstr /X` and `$` never match the LF-only lines that WSL prints (measured), so an exact-line match is not
available. Only a well-formed id can reach `LID`: a stray line before or after the id no longer blocks a good launch,
and no cmd metacharacter (`&`, `|`, `<`, `>`, `"`, `%`, `^`) can reach the `echo`, `if exist` and `start` lines.
Before this fix, a stdout line such as `x&ver` ran `ver`, and `x>file` created a file in the current directory.

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
(`-forceUpdate` keeps frames fresh). `-noBackend` is left out: add it only for capture-only runs. The flags must be
verified again on 1.9 before any capture on it.

It is UNVERIFIED (capture program 4.2 item 2) that a direct exe launch never makes Steam update or relaunch the app.
Steam must be running.

## 5. Cold copy (decision D2b)

```bash
bash /mnt/e/DriveDiT/tools/pin/cold_copy.sh          # dry run: checks the installs against the active pin and prints the plan
bash /mnt/e/DriveDiT/tools/pin/cold_copy.sh --run    # copies the whole install to E:\DriveDiT\pin\<active pin id>
```

By default it copies the **active** pin, resolved exactly as `preflight --pin-active` does. `--pin PIN_JSON` names a
pin file instead, and `--pin-dir DIR` moves the pointer lookup; the two are mutually exclusive. The destination
defaults to `<pin dir>/<pin id>`, for example `E:\DriveDiT\pin\1.8.0.13`.

`--run` does the following, in order:
1. Checks the live installs against the pin. AutoUpdateBehavior is waived; everything else must pass.
2. Runs `rsync -rt` into `game/` and `tools/`. A marker file allows an interrupted copy to be resumed. A
   destination that contains a symlink is refused before anything is written (`rsync -rt` never makes one, and one
   could lead a write off E:).
3. Checks the live installs again, to show that nothing changed during the copy.
4. Checks the copy against the pin at 1 s mtime resolution. WSL drvfs truncates the mtimes it *sets* to whole
   seconds; this was measured on E:. Then each copy must hold exactly the regular files of its source, with none
   missing and none extra (such as an rsync temp file left by a killed run).
5. Writes `SHA256SUMS` for every copied file.
6. Copies the pin and both appmanifests into `meta/`.

The script never deletes anything except its own marker file.

Verify the copy later with `cd /mnt/e/DriveDiT/pin/1.8.0.13 && sha256sum -c --quiet SHA256SUMS`. The depot manifest
ids in the pin also allow a third-party DepotDownloader rollback. Running that tool is the user's decision (D2d).

The copy keeps file contents and whole-second mtimes, not the sub-second mtimes of three Tools files in
`pin_1.8.0.13.json` (the Workbench exe, `core/data.pak` and `shaders-dx12.pak`). The pin copied to `meta/` still
records every exact `mtime_ns`. A restore is therefore re-pinned under a new id; see "Rollback".

## 1.8 is an interim pin; moving to 1.9

**Direction (user, 2026-10-21).** Pin 1.8.0.13 for now. The system's target is 1.9's new vehicle physics, once 1.9
reaches the stable branch and compatibility is checked. This supersedes decision D2 of `CAPTURE_PROGRAM.md` r2
("1.8.0.13 for the whole campaign, Everon test included, even if 1.9 ships"); updating the program text is the
coordinator's job. The migration plan will be `E:\drivedit_realign\program\1_9\MIGRATION_1_9.md` (not written yet).
It owns the go/no-go and the list of compatibility probes.

**The re-pin procedure.** Run it only when MIGRATION_1_9.md says go. Close Workbench and every client first.

1. **Keep Automatic updates at "1"** ("Only update this game when I launch it") on both apps. Do not switch to
   "Always keep this game updated" to get 1.9: the update should happen when you start it, not overnight.
2. **Optional: cold-copy 1.8 first**, while 1.8.0.13 is still the active pin, if a rollback to 1.8 should stay
   possible: `bash /mnt/e/DriveDiT/tools/pin/cold_copy.sh` (dry run), then `--run` (the whole install, to
   `E:\DriveDiT\pin\1.8.0.13`).
3. **Update both apps via Steam**, by hand. UNVERIFIED: the exact Steam UI for a manual update under this setting;
   record what you did in MIGRATION_1_9.md. From now until step 5, every launch is refused (`buildid != pin`). That is
   intended. Both apps must be on the new build, and Arma Reforger Tools may ship later than the game. Wait until
   both appmanifests show TargetBuildID == buildid and StateFlags 4.
4. **Make the new pin** (read the new buildids from the appmanifests):

   ```bash
   python3 -I -B /mnt/e/DriveDiT/tools/pin/make_pin.py --pin-id <new id, e.g. 1.9.0.12> \
     --out /mnt/e/DriveDiT/pin/pin_<new id>.json \
     --expect-game-buildid <game buildid> --expect-tools-buildid <tools buildid>
   ```

5. **Activate it**: `python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_activate.py <new id>`. This is refused unless the
   new pin passes every check against the installs. The switch is logged in `E:\DriveDiT\pin\ACTIVE.log`.
6. **Preflight**: `python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --check` must end with
   `RESULT: OK (pin <new id>)`.
7. **Compatibility probes** from MIGRATION_1_9.md, before any capture on the new build. The request for this change
   names them DD-12, but `TICKETS_UNIFIED.md` uses DD-12 for the consolidation ticket, so the ticket id is to be
   confirmed there. At least these depend on the build and need a re-check:
   - the DataCapture addon must compile in the new Workbench (DD-1);
   - the MCP's script API index is the 1.8 one (DD-3 `data/api`);
   - every client flag (the S1 probes);
   - the actuator identification behind `llc_v1` (E-01, E-02Q), since steering and the powertrain change.

**Activating a pin also opens capture on it.** Once `ACTIVE` names a 1.9 pin, `client_R1_i01.bat` starts sessions on
1.9. That is necessary, because the compatibility probes themselves need Workbench and client runs. Step 7 and the
proposed G19 gate (`program\1_9\SURFACE.md` section 6) are procedure, not code: nothing in these tools stops a
capture session on a pin that has not passed them. Data stays traceable, because every launch record and session
names its pin. The admission rule belongs in validator C02 (DD-5-min; SURFACE E-02): per-pin allow-lists, read from
the manifest or `ACTIVE`, so a session on a pin that has not passed G19 fails validation even though its launch was
allowed. MIGRATION_1_9.md should own that rule. Until C02 enforces it, run nothing on a new pin except the probes.

**Data from two builds.** Every launch record names its pin id, and the client passes the launch id into the session,
so each session can be traced to its build. Whether data from 1.8 and 1.9 may be mixed, and on which build the sealed
Everon test runs, is a decision for MIGRATION_1_9.md, not for these tools.

**Rollback.** `pin_activate.py 1.8.0.13` stays refused even after the 1.8 files are back. Three Tools files in
`pin_1.8.0.13.json` have sub-second mtimes (`...758506855800`), and activation compares `mtime_ns` exactly. A
restore from the cold copy brings back whole seconds (drvfs sets whole seconds), and a DepotDownloader restore brings
back download times. The checks are not weakened for this, and `pin_1.8.0.13.json` is never overwritten. Instead, the
restored installs get a new pin id, checked against the original. Every step that writes to C: or D: is a user action
(D2d). Close Workbench, every client and Steam first.

1. Verify the cold copy: `cd /mnt/e/DriveDiT/pin/1.8.0.13 && sha256sum -c --quiet SHA256SUMS`.
2. Restore the files of both installs from `E:\DriveDiT\pin\1.8.0.13\game` and `...\tools` (for example
   `robocopy <copy> <install> /MIR` from Windows), or with DepotDownloader to the depot manifests in the pin.
3. Restore both appmanifests from `E:\DriveDiT\pin\1.8.0.13\meta` into the two `steamapps` directories, because
   Steam still records the 1.9 buildid. UNVERIFIED: how Steam treats a restored appmanifest; keep Automatic updates at
   "1" and Steam offline. Without a cold copy there is no saved appmanifest, and editing one by hand is UNVERIFIED and
   the user's call.
4. Check the restored contents against the copy (read-only), for example for the game:
   `cd /mnt/e/DriveDiT/pin/1.8.0.13 && sed -n 's#^\([0-9a-f]\{64\}\)  game/#\1  /mnt/d/SteamLibrary/steamapps/common/Arma Reforger/#p' SHA256SUMS | sha256sum -c --quiet`,
   and the same for `tools/` with the Tools install.
5. Re-pin under a new id. `--restore-of` refuses (exit 1) unless the restored installs equal `pin_1.8.0.13.json` in
   appid, buildid, depot manifests, appmanifest and install paths, `steam_appid.txt`, and the set and size of every
   pak and exe; only mtimes may differ. The new pin records `restore_of` (the original id and its sha256), so a
   validator can treat both ids as one build:

   ```bash
   python3 -I -B /mnt/e/DriveDiT/tools/pin/make_pin.py --pin-id 1.8.0.13-r1 --restore-of 1.8.0.13 \
     --out /mnt/e/DriveDiT/pin/pin_1.8.0.13-r1.json --expect-game-buildid 24903726 --expect-tools-buildid 24870687
   ```

6. `python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_activate.py 1.8.0.13-r1`, then
   `preflight.py --pin-active --check` must end with `RESULT: OK (pin 1.8.0.13-r1)`.

Never delete or overwrite `pin_1.8.0.13.json`. Restoring the exact mtimes from Windows (NTFS keeps 100 ns, and the pin
in `meta/` records every `mtime_ns`) would let `1.8.0.13` itself pass again, but these tools do not do that.

**Side-by-side Experimental install.** 1.9 Experimental is not the stable app. DD-14a added the experimental channel
(app ids 1890860 and 1890880 on the E: library; `make_pin.py --channel experimental`); see section 6. Such a pin is
never activated. Since 1.9 stable shipped (2026-11-05) the Experimental apps are unused.

## 6. Pin channels, build_key and use_class (DD-14a, dd-pin/2)

MIGRATION_1_9 2.2, 5.3, 5.5 and 8.2 component 1. Since 2026-11-05 the active pin is `1.9.0.37` (dd-pin/1, made before
DD-14a); 1.8.0.13 survives only as the cold copy `E:\DriveDiT\pin\1.8.0.13`.

**build_key.** `build_key/1` names a build, not a pin: the sha256 of the ASCII text

```
build_key/1
game.appid=<int>
game.buildid=<int>
game.exe_sha256=<sha256 of the client exe, ArmaReforgerSteam.exe>
tools.appid=<int>
tools.buildid=<int>
tools.exe_sha256=<sha256 of Workbench/ArmaReforgerWorkbenchSteamDiag.exe>
```

(LF after each line). It is the same function as the harvester's `build_key_v1`, so the pin, its sidecar, the launch
record and the harvester agree. Pins with equal `build_key` are one build (`1.8.0.13`, `1.8.0.13-r1`, `1.8.0.13-cold`).

**dd-pin/2.** `make_pin.py --channel stable|experimental|cold` writes schema `dd-pin/2`: `channel`, `release_channel`
(what sessions carry as `pin.channel`: stable or experimental; a cold pin runs a stable build), `launch_path_id`, the
sha256 of every launched exe (`<app>.exe_sha256`), `build_key` with its `build_key_inputs`, and `physics_hz_expected`
(default 30, the measured fixed slice of 1/30 s on 1.8.0.13 and 1.9.0.37; MIGRATION_1_9 8.2 planned 60; `--physics-hz-expected` states another rate). Without `--channel`, make_pin writes DD-0's dd-pin/1 pin, unchanged. Use
`--channel stable` for the next re-pin. `--dry-run` prints the pin and writes nothing. With `--restore-of`, a dd-pin/2
re-pin must also have the reference pin's build_key.

| Channel | App ids | Exes (build_key exe first) | Launch path | Activation |
|---|---|---|---|---|
| `stable` | 1874880 / 1874910 | `ArmaReforgerSteam.exe`, `ArmaReforgerSteamDiag.exe`; Workbench | `stable` (`LAUNCH_PATHS`) | yes |
| `experimental` | 1890860 / 1890880 (E: library) | `ArmaReforgerSteam_Experimental.exe`, `ArmaReforgerSteamDiag.exe`; Workbench | `experimental` | **never**: `pin_activate.py` exits 1 and writes nothing; `--pin-active` refuses an ACTIVE naming one (exit 2). Explicit `--pin` only, always `use_class compat-preview` |
| `cold` | as the reference stable pin | as stable, run from a verified copy | `cold-<ref id>` (`COLD_LAUNCH_PATHS`) | yes (the preferred rollback); see section 7 |

No launcher starts an experimental pin: the side-by-side `wb_X19.bat` of the plan is moot since 1.9 stable shipped.

**The dd-pin/1 migration path: sidecars.** A pin file is immutable, so a dd-pin/1 pin gets its build_key from a
write-once sidecar next to it, bound to the exact pin bytes (`pin_sha256`). The harvester reads the same file.

```bash
python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_sidecar.py build-key 1.9.0.37 --from-live --dry-run   # then --apply
python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_sidecar.py build-key 1.8.0.13 --from-sums --dry-run   # the cold copy
python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_sidecar.py engine 1.9.0.37 --run-header <run_header.txt> --dry-run
```

- `build-key --from-live` hashes the live exes only after the full preflight check of the pin passes against the live
  installs (so the hashed exes are the pinned ones), and stats them again afterwards. It reads about 147 MB (0.9 s on
  2026-11-07 for 1.9.0.37).
- `build-key --from-sums` takes the exe hashes from `<pin dir>/<id>/SHA256SUMS` and hashes the copy's exes to confirm
  them: for a pin whose installs have moved on (1.8.0.13).
- `engine` writes `pin_<id>.engine.json` with `engine_version_expected`, the `engine.build_version` of a probe run
  header that names this pin id and pin sha256 (SURFACE 5 row 2: the first E-11). It also records the measured
  `engine.fixed_time_slice_s`. Apply the build-key sidecar first, so the engine sidecar names the build_key.
- A sidecar is written atomically and never replaced (`pinlib.publish_new`: a uniquely named temp file, fsync, then
  a hard link to the final name, or `renameat2(RENAME_NOREPLACE)`, and a directory fsync), so a reader sees no file
  or the whole file, and a write cut short never blocks a retry. A filesystem with neither refuses (nothing
  written): there is no check-then-rename fallback. E: supports `link()` (uv's hard-linked venvs on E: show
  `st_nlink > 1`). One that exists but does not validate (another
  pin's bytes, a build_key that does not match its inputs, a `--from-sums` sidecar whose SHA256SUMS changed since)
  is a config error everywhere (exit 2), never ignored.

**What the preflight does with it.** When the exe hashes are known (dd-pin/2, or a dd-pin/1 pin with its build_key
sidecar), every launch, `--check` and activation also runs `<app>.exe_sha256`: each pinned exe is hashed (about 147 MB,
under a second) and compared, so an exe replaced with the same size and mtime is refused. A launcher names its launch
path (`--launch-path`, default `stable`); `<app>.launch_path` refuses a pin made for another one. `--use-class` sets the
launch-time `use_class` (dev, appearance, probe, compat-preview, g19; default dev). An experimental pin always
records `compat-preview`; `production` is refused until DD-14b's G19 admission record exists.

**Launch record dd-launch/2.** When there is anything to add (a known build_key, a dd-pin/2 pin, a non-stable channel
or `--use-class`), the record is `dd-launch/2` with top-level `build_key` and `build_key_source`, `channel` (the
release channel), `pin_channel`, `launch_path_id`, `use_class`, `physics_hz_expected` and, from the engine sidecar,
`engine_version_expected`. DC_LaunchGuard reads the first five; DC_Session compares `engine_version_expected` with
`GetBuildVersion()`. `build_key` is left out, never null, while it is unknown. A dd-pin/1 pin without sidecars keeps
DD-0's dd-launch/1 record and its 22 checks.

**Freeze check (MIGRATION_1_9 5.5, CC-01).** Once per app pair, with Steam running online:

```bash
python3 -I -B /mnt/e/DriveDiT/tools/pin/freeze_check.py before --pin-active --steam online   # --dry-run first
#   the user starts Workbench with wb_R1.bat, lets it load, closes it
python3 -I -B /mnt/e/DriveDiT/tools/pin/freeze_check.py after --before /mnt/e/DriveDiT/pin/freeze/freeze_<id>_<UTC>.before.json
```

`before` runs the full preflight check, then records both appmanifests (identity fields, file size, mtime and
sha256; never `LastOwner`) and the pinned exes. `after` re-reads them and lists the launch records of this pin written
in between. PASS: no identity field changed (buildid, TargetBuildID, StateFlags, AutoUpdateBehavior, depot manifests,
LastUpdated), the full preflight check passes again, and at least one **real** launch lies in between: a launch
record of this pin (a JSON object with the dd-launch fields) and that launch's own `console.log` under the launch
path's Workbench profile, naming its `-dcLaunchId`, started after the before-record and reporting the build (the
pin's `engine_version_expected` once it has one). A preflight run alone never passes. Only `--steam online`
qualifies for 5.5: an offline run ends `PASS_UNQUALIFIED`, which CC-01 does not accept. An appmanifest
rewrite that changes none of them (Steam updating `LastPlayed` on a launch, measured 2026-11-07) is a note, not a
failure. Steam's online state cannot be read from WSL, so `--steam` records what the user says. The records live in
`<pin dir>/freeze/`; `tools/compat19/p19_build.py` reads them for CC-01.

## 7. The cold-run pin: 1.8.0.13 without Steam (DD-14a, MIGRATION_1_9 7.1)

Both stable apps are single Steam installs, and they are on 1.9.0.37 since 2026-11-05. 1.8.0.13 runs again only from
a copy. The verified reference copy `E:\DriveDiT\pin\1.8.0.13` (cold_copy.sh, 2026-10-22) stays a pristine restore
source; runs use a separate **working copy** `E:\DriveDiT\pin\1.8.0.13-run` (decision 16), a second full copy of the install.

| Piece | Value |
|---|---|
| Pin | `pin_1.8.0.13-cold.json`, dd-pin/2, channel `cold`, release channel `stable`, launch path `cold-1.8.0.13` |
| Launch paths | `pinlib.COLD_LAUNCH_PATHS["cold-1.8.0.13"]`: installs `<run>/game`, `<run>/tools`; appmanifests `<run>/meta/appmanifest_<appid>.acf` (the saved ones) |
| Launchers | `wb_C18.bat` (profile `E:\DriveDiT\wb_profile_c18`, gproj `E:\DriveDiT\addons_c18\DataCapture\addon.gproj`, `-addonsDir` the copy's game addons) and `client_C18_i01.bat` (always `-noBackend`). Both pass the explicit pin and `--launch-path cold-1.8.0.13`; neither reads ACTIVE, so a 1.8 baseline never switches the active 1.9 pin. Both set TEMP and TMP to `E:\DriveDiT\tmp\wb_c18` or `...\c18_i01` (created if missing, refused unless on E:), and both copy the launch record into their profile's `dc_launch\` and byte-check it |
| Addon | its own copy, `mkdir /mnt/e/DriveDiT/addons_c18`, then `deploy_addon.py --target /mnt/e/DriveDiT/addons_c18/DataCapture --target-pin 1.8.0.13-cold --require-clean`, so a 1.8 Workbench never re-saves the 1.9 addon |
| build_key | equal to 1.8.0.13's (`0d31b417…` from the reference copy's SHA256SUMS); `make_pin.py` refuses otherwise |

**Make it** (agent, E: only; reads the whole copy):

```bash
# 1. the working copy, from the verified reference (the user's call: a second full copy on E:)
cd /mnt/e/DriveDiT/pin/1.8.0.13 && sha256sum -c --quiet SHA256SUMS            # verify the reference first
mkdir /mnt/e/DriveDiT/pin/1.8.0.13-run
rsync -rt /mnt/e/DriveDiT/pin/1.8.0.13/{game,tools,meta} /mnt/e/DriveDiT/pin/1.8.0.13-run/
# 2. the reference pin's build_key sidecar (so the cold pin and the harvester read one key)
python3 -I -B /mnt/e/DriveDiT/tools/pin/pin_sidecar.py build-key 1.8.0.13 --from-sums --apply
# 3. the cold pin: every pak and exe of the working copy is hashed against the reference SHA256SUMS
python3 -I -B /mnt/e/DriveDiT/tools/pin/make_pin.py --channel cold --pin-id 1.8.0.13-cold --cold-of 1.8.0.13 \
  --from-copy /mnt/e/DriveDiT/pin/1.8.0.13-run --out /mnt/e/DriveDiT/pin/pin_1.8.0.13-cold.json --dry-run   # then without
python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin /mnt/e/DriveDiT/pin/pin_1.8.0.13-cold.json \
  --launch-path cold-1.8.0.13 --check [--verify-hashes]
```

`make_pin.py --channel cold` refuses (exit 1, nothing written) unless the copy's saved appmanifests name the reference
pin's app ids, buildids and depot manifests, the copy holds exactly the reference pin's paks and exes with their sizes,
every one of them matches SHA256SUMS, `steam_appid.txt` is unchanged, and the build_key equals the reference pin's. It
refuses the reference copy itself as `--from-copy`, and any directory that is not a `COLD_RUN_ROOTS` entry (a pin from
there could never launch). The pin records each pak's sha256, and every launch, `--check` and activation of a cold pin re-hashes every pak
against it (`<app>.pak_sha256`, mandatory), so a pak changed with its size and mtime kept is refused. There is no
cache (one keyed by size and mtime could not see that change): every cold launch reads every pak of the install,
which takes minutes. The pin also records the sha256 of the reference SHA256SUMS
(`cold_of.sums_sha256`), checked at every launch. The acf checks read the saved appmanifests: on a
copy nothing moves them, so they catch tampering only. `pin_activate.py 1.8.0.13-cold` works (the preferred rollback,
MIGRATION_1_9 7 step 16); while it is active, `wb_R1.bat` and `client_R1_i01.bat` refuse (launch path mismatch) and
the cold launchers are the ones to use.

**User verification, once (decision 16; never by an agent).** Steam closed or in offline mode. Run `wb_C18.bat`; check
that `E:\DriveDiT\wb_profile_c18\logs\...\console.log` shows `version 1.8.0.13` and the c18 profile and gproj, then
close it. Later, once the client launcher can run (DC_WORLD set, engine_R1.conf built), `client_C18_i01.bat`.
UNVERIFIED until then: that the exes run outside their Steam library at all, and that a 1.8 client is not refused by
the production backend with `-noBackend`. If it fails, the fallback is the DD-0 restore procedure ("Rollback").

## Tests

```bash
mkdir -p /dev/shm/dd0t && TMPDIR=/dev/shm/dd0t /usr/bin/python3 -B tools/pin/tests/test_pin.py; rm -rf /dev/shm/dd0t
python3 -I -B -m unittest discover -s tools/pin/tests -v
```

`test_pin_dd14a.py` covers the DD-14a additions (channels, build_key, dd-pin/2, sidecars, use_class, the freeze check); `test_cold_pin.py` the cold-run pin, the cold launchers and the golden start lines of the stable launchers (`fixtures/golden_start_lines.txt`, recorded at DD-0 HEAD). Both reuse the fake Steam layout of `test_pin.py`, which is unchanged.

Temp trees go to `$DD_PIN_TEST_TMP`, otherwise `/dev/shm`, and are deleted afterwards. Run the tests with
`/usr/bin/python3` (3.10) as well, because the launchers run the preflight with it. No test hands a real `/mnt/c`
path to code that writes. Tests point `pinlib.LAUNCH_PATHS` at their fake Steam layout; a script run in a subprocess
sees the real launch paths, so its fake pin fails `launch_path` there (the tests expect that). Run the suite once
with `DD_PIN_TEST_TMP` on E: as well: drvfs differs from tmpfs (sub-second mtimes, no FIFOs, renamed directories).

## Differences from the Appendix B sketch

- The preflight accepts only AutoUpdateBehavior `"1"`. Appendix B forbids only `"0"`, but `"2"` (high priority) also
  auto-updates.
- The preflight also checks StateFlags, depot manifests, exes and `steam_appid.txt`. It also refuses a pak that is
  not in the pin.
- In the pin, `depots` maps each depot to `{manifest, size}`, and `paks` is keyed by relative path, because there are
  two `data.pak` files: `addons/core` and `addons/data`. Mtimes are stored as `mtime_ns` and compared exactly.
- The launchers name the WSL distro and the interpreter, set the working directory to the exe's directory
  (`start /D`), accept only a well-formed id line (`WSL_UTF8=1` plus two `findstr` filters), require the launch
  record to exist, and refuse while placeholders are unset. `wb_R1.bat` refuses a second Workbench.
- The launchers check the pin named by the `ACTIVE` pointer (`--pin-active`), not a fixed pin path. The pointer is
  switched only by `pin_activate.py`, which checks the new pin first, so moving to 1.9 is a checked, logged step.
- The preflight also checks that the pin was made from the installs the launchers start (`launch_path`).
- The preflight writes a record for refused launches as well.
