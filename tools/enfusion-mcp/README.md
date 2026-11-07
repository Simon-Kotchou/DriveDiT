# Enfusion MCP on WSL: setup runbook (DD-3)

This directory builds and runs [`steffenbk/enfusion-mcp-BK`](https://github.com/steffenbk/enfusion-mcp-BK) 0.12.0 as a
stdio MCP server for Claude Code in WSL, driving Arma Reforger Workbench over its NET API (port 5775).
It replaces the abandoned npm `enfusion-mcp` 0.6.1 entry: its upstream repo is gone, it defaults to `127.0.0.1` (refused
under WSL NAT), and it has no WSL path handling.

Every write goes to E:. C: is nearly full, and the WSL disk (`~`, `/tmp`, `/var`, ...) lives on C: too.
The scripts accept a work root, project path, `.gproj` or Workbench profile only as an absolute path that resolves onto a
drive other than C: (`/mnt/<d..z>/...`, or `X:\...` for the profile). Relative paths are refused as well: Claude Code starts
the wrapper with its own cwd, which is the C: repo.

Evidence and background: `/mnt/e/drivedit_realign/enfusion/mcp-setup.md`, `FIX_PLAN.md` §5, `program/CAPTURE_PROGRAM.md` §7.
Review record: `/mnt/e/drivedit_realign/p0/dd3-review.md`.

## Pinned versions

| Piece | Pin |
|---|---|
| MCP server | BK `0acfa884228477043c6b4c2b8c7f0c270d648398` (main, 2026-08-05), `package.json` 0.12.0, 54 tools |
| API index (legacy) | Reforger 1.8 index from [`fma-rc/enfusion-mcp`](https://github.com/fma-rc/enfusion-mcp) `4cd14dbbfb8a3b2832b0ec2480a5dcc8fe1b2c49` (`data/api/*.json`: 887 Enfusion + 8,085 Arma classes, scraped from Tools buildid 24870687). BK's own index predates 1.7 |
| Node | nvm `v24.2.0` (BK needs `>=20`), resolved from the real `~/.nvm` (read-only) |
| Engine | Reforger 1.8.0.13 (game buildid 24903726, Tools buildid 24870687), the **interim** pin; see DD-0 for the build pin. The target is 1.9 once it is stable and compatibility is checked; a move to 1.9 also needs a 1.9 API index here |

## Files

| File | Purpose |
|---|---|
| `build-bk.sh` | build at the pin, apply both guard patches, `npm ci --ignore-scripts`, `tsc`, check the guards are in `dist/`, prune devDependencies. `--data18` also composes the 1.8 data dir from the pinned fma-rc commit. `--smoke` runs `initialize` + `tools/list` and exits 1 unless it gets serverInfo and 54 tools |
| `enfusion-mcp-wsl.sh` | stdio wrapper that Claude Code runs. Forces the three guards to `1`, refuses a `dist/` without the guard code, finds the Windows host, runs the server from `$EMCP_ROOT/cwd`. `--print-env` shows the resolved config and creates nothing; `--check` also opens one TCP connection to the NET API port |
| `launch-workbench.sh` | WSL fallback for DD-0's `wb_R1.bat`: same arguments (`-profile "E:\DriveDiT\wb_profile"`, DataCapture gproj, `-forceUpdate -maxFPS 60`), runs the DD-0 pin preflight against the **active** pin (`preflight.py --pin-active`: the pin that `E:\DriveDiT\pin\ACTIVE` names, switched only by DD-0's `pin_activate.py`; `WB_PIN_DIR` overrides the pin dir, `WB_PIN` is refused), names no game version, and starts Workbench only if the preflight (run on `/usr/bin/python3`) printed exactly one launch id and `launch_<id>.json` exists, as `wb_R1.bat` does. Its `WB_DIR` is the Tools install in DD-0's `pinlib.LAUNCH_PATHS`, and the preflight's `launch_path` check refuses any pin not made from that install. It refuses if Workbench already runs or that cannot be checked (it reads `tasklist /FO CSV`: the default table format truncates the image name to `ArmaReforgerWorkbenchStea`, so a full-name match there never fires) |
| `claude-mcp-entry.json` | the user-scope MCP entry (same as the `claude mcp add` command below) |
| `mcp_gate.py` | DD-14a. What the wrapper execs: picks the pin (`EMCP_PIN`, else ACTIVE), serves only that pin's index and project root, binds the endpoint host, and proxies the stdio session through the `bk_tools.py` policy (deny by default: reads, and wb reads after the handshake; every write denied; sealed-map guard) |
| `endpoint_handshake.py`, `wb_proc.js` | DD-14a. The read-only endpoint handshake of MIGRATION_1_9 6.3 (also a CLI); `wb_proc.js` is its WMI process query, run by cscript.exe (no PowerShell) |
| `bk_tools.py` | DD-14a. The gate's deny-by-default allow-list of BK read tools and wb read actions, the read-path check and the sealed-map guard |
| `index_sources.json` | DD-14a. Pin id to dump tag and commit, for the index provenance check |
| `api_index.py` | DD-14a. Builds a per-pin API index in BK's JSON shape from the official script dump at the pin's tag, stamped with the pin (`api/INDEX_PIN.json`) |
| `api_index_diff.py`, `expect_1.8.0.13_to_1.9.0.37.txt` | DD-14a, CC-05. Diffs two per-pin indexes and checks them against the API delta (API_DELTA 2-3, G19-0) and the planned surface (`tools/compat19/planned_surface.txt`) |
| `tests/test_handshake.py`, `tests/fixtures/` | DD-14a tests: index, provenance, diff, gate config, policy (every write tool and action denied, reads allowed), handshake (stub Windows probes), proxy (stub server). The fixture dumps are synthetic classes, not game scripts |
| `patches/bk-wsl-guard.patch` | env-gated guards in `src/workbench/client.ts` (sha256 `457d5e46…1c92`) |
| `patches/20-bk-mod-build-guard.patch` | env-gated guard in `src/tools/mod.ts` (sha256 `09803bae…dc9ba`) |

The live install is `EMCP_ROOT=/mnt/e/drivedit_realign/mcp`. Its copies of these files are byte-identical to the ones here
(scripts at the top level, patches in `patches/`). After changing a file here, copy it there.
No BK source, `node_modules` or build output is committed.

## E: layout (`EMCP_ROOT`, default `/mnt/e/drivedit_realign/mcp`)

```
src/enfusion-mcp-BK/            pristine clone (untrusted content, never built in place)
src/others/fma-rc_enfusion-mcp/ pristine clone, source of the 1.8 API index
build/enfusion-mcp-BK/          build copy at the pin + guard patches; dist/ (2.0 MB) and prod node_modules/ (27 MB)
data/                           api/ (fma-rc 1.8) + BK patterns/ recipes/ wiki/; api/SOURCE_fma-rc_commit.txt
home/ tmp/ xdg/ npm-cache/ npm-logs/   redirected HOME, TMPDIR, XDG_*, npm cache and logs
cwd/                            the server's working directory (relative tool paths land here, not on C:)
```

## Build (agent or user, E: only)

Prerequisite: the two pristine clones exist (one-time; needs GitHub):

```bash
R=/mnt/e/drivedit_realign/mcp
git clone https://github.com/steffenbk/enfusion-mcp-BK "$R/src/enfusion-mcp-BK"
git clone https://github.com/fma-rc/enfusion-mcp "$R/src/others/fma-rc_enfusion-mcp"
```

Then (needs the npm registry; the build reads the fma-rc index from commit `4cd14db` whatever is checked out):

```bash
bash /mnt/e/drivedit_realign/mcp/build-bk.sh --data18   # pin, guard patches, npm ci, tsc, guard check, prune, 1.8 API data
bash /mnt/e/drivedit_realign/mcp/build-bk.sh --smoke    # expect serverInfo {name: enfusion-mcp, version: 0.12.0} and 54 tools; exit 0
```

Re-running either command is safe. To move the pin, see the last section.

What the build redirects, and where to:

| Default target (C:) | Redirected to |
|---|---|
| `~` (`.npmrc`, `.gitconfig`, `.enfusion-mcp/`) | `$EMCP_ROOT/home` |
| `~/.npm` (cache, `_logs`) | `$EMCP_ROOT/npm-cache`, `$EMCP_ROOT/npm-logs` |
| `~/.cache`, `~/.config`, `~/.local/{state,share}` | `$EMCP_ROOT/xdg/{cache,config,state,data}` |
| `/tmp`, `/tmp/node-compile-cache` (npm 11 and tsc 5.9 enable it) | `$EMCP_ROOT/tmp`, `$EMCP_ROOT/tmp/node-compile-cache` |
| the caller's cwd | `$EMCP_ROOT` (build), `$EMCP_ROOT/cwd` (server) |
| playwright browsers (devDependency, scraper only) | download skipped; path `$EMCP_ROOT/pw-browsers` |

Both scripts also drop inherited `NODE_OPTIONS` and `NODE_V8_COVERAGE`, which could write reports elsewhere.
Lifecycle scripts never run (`--ignore-scripts`). Nothing is installed globally, and `npx` is not used.
After a build, `find ~ -maxdepth 3 -newer <stamp>` should list only Claude Code's own `~/.claude.json` and
`~/.claude/backups/`.

## Guards (env-gated; the wrapper always sets all of them to `1`)

| Variable | Patch | Blocks |
|---|---|---|
| `ENFUSION_MCP_NO_AUTO_INSTALL=1` | `bk-wsl-guard.patch` | the silent copy of the 21 handler files (101,273 B) into the alphabetically first addon under `ENFUSION_PROJECT_PATH` when Workbench answers `Undefined API func` |
| `ENFUSION_MCP_NO_AUTO_LAUNCH=1` | `bk-wsl-guard.patch` | `wb_launch` and the auto-launch on `ECONNREFUSED`. BK would copy handlers and spawn the Windows exe with `/mnt/...` paths, the game-dir CWD and no `-profile` |
| `ENFUSION_MCP_NO_AUTO_LAUNCH=1` | `20-bk-mod-build-guard.patch` | `mod` action=`build`. It would spawn a second Workbench with Linux paths, defaulting its output to `Program Files` on C: |
| `ENFUSION_MCP_WSL_PATHS=1` | `bk-wsl-guard.patch` | whole-string params `/mnt/<x>/...` are sent to Workbench as `<X>:/...` (for example `RegisterResourceFile(absFilePath)`). Resource paths (`Prefabs/...`, `{GUID}...`) are untouched |

The patched code tests for exactly `"1"`, so `0`, `true` or `yes` would each turn a guard off. The wrapper therefore ignores
any incoming value (it logs `ignoring ...` on stderr) and exports `1`. It also refuses to start a `dist/` whose
`workbench/client.js` and `tools/mod.js` lack the guard code, so an unpatched build can never run behind it.

Other wrapper defaults (each can be overridden by exporting it, or through `env` in the MCP entry):
- `ENFUSION_PROJECT_PATH`: set by `mcp_gate.py` to the pin's project root (`/mnt/e/DriveDiT/addons` for a stable pin); generators (`project`, `prefab`, `mod` create, `script_create`, ...) write there. A value on C: or the WSL disk is refused (exit 2), and so is any value other than the pin's root.
- `ENFUSION_DEFAULT_MOD=DataCapture`: if anything ever injects handlers, it targets the open addon, not `ArmaDrive` on C:.
- `ENFUSION_MCP_DATA_DIR`: set by `mcp_gate.py` to the pin's index (see "Per-pin index"); BK silently falls back to an empty index when it is empty, so the gate checks every api file first.
- `ENFUSION_GAME_PATH` (D:) and `ENFUSION_WORKBENCH_PATH` (C:) are only read. Everything BK does with the Workbench path that could write (launch, `mod build`) is guarded.
- `HOME`, `TMPDIR` and `XDG_*` point at `$EMCP_ROOT`, and the server runs in `$EMCP_ROOT/cwd`. stdout carries only JSON-RPC.

## Per-pin index and the endpoint handshake (DD-14a, CC-05)

Since 2026-11-05 the active pin is 1.9.0.37, and the fma-rc index in `data/` is the 1.8 scrape: it has no
`VehicleWheelEffectComponent`, `GetHandbrake` or `Update*Configuration`. The wrapper therefore no longer serves
`data/`. It execs `mcp_gate.py`, which serves `$EMCP_ROOT/pins/<pin id>/data` only:

- **The pin.** `EMCP_PIN=<id>` in the MCP entry's `env` (a second config), else `E:\DriveDiT\pin\ACTIVE`. A stable
  `EMCP_PIN` must be the active pin: the stable project root and installs are the active pin's. A 1.8 config after
  the re-pin is the cold pin, `EMCP_PIN=1.8.0.13-cold` (project root `E:\DriveDiT\addons_c18`).
- **The index.** `pins/<id>/data/api/INDEX_PIN.json` must name that pin id and pin file sha256 (and its build_key,
  when both are known), and every api file must match its hash. Its source must be the dump tag and commit that
  `index_sources.json` maps the pin to, and that tag must be `v<engine_version_expected>` of the pin's engine
  sidecar, so a 1.8 dump can never stamp a 1.9 pin. There is no fallback: a missing or foreign index refuses
  (exit 2) before the server starts, and `--print-env` says why.
- **The project root** is `pinlib.DEPLOY_ROOTS[launch path]`; an `ENFUSION_PROJECT_PATH` set by hand must be it, and
  a deploy stamped for another pin (`.deploy_state.json` `target_pin`) refuses. `ENFUSION_GAME_PATH` and
  `ENFUSION_WORKBENCH_PATH` are the pin's installs.
- **The endpoint.** `ENFUSION_WORKBENCH_HOST` must be loopback, or the WSL NAT gateway (the Windows host,
  `172.19.208.1` today) when NAT mode is established: exactly one IPv4 default route, via `eth0`, and
  `/etc/resolv.conf` carrying WSL's generated header with that gateway as its only nameserver. Anything else (a LAN or
  VPN gateway such as `192.168.1.1`) refuses: the handshake can only vouch for the local Workbench. The gate passes the
  checked host and port to the server, and the listener must serve that address (`0.0.0.0`, `[::]` or the host).
- **Deny by default** (coordinator decision, Codex round 2: the runbook never uses MCP mutations; agents never call
  MCP tools to change anything, the user presses Play). `bk_tools.py` allows only these, and denies every other tool,
  every other action, unknown tools, unknown actions and a missing action:

  | Class | Tools and actions |
  |---|---|
  | read (no handshake) | `api_search`, `wiki_search`, `wiki_read`, `asset_search`, `component_search`, `game_browse`, `game_read`, `workshop_info`, `wb_knowledge`, `project` with `browse` or `read` |
  | wb read (after the handshake) | `wb_connect`, `wb_diagnose`, `wb_state`, `wb_entity_list`, `wb_entity_inspect`, `wb_read_props`; `wb_validate` material/texture; `wb_terrain` getHeight/getBounds; `wb_resources` getInfo; `wb_projects` list/locate; `wb_layers` list/isVisible/getInfo; `wb_prefabs` getGuid/locate/getAncestor; `wb_localization` getTable/listLanguages; `wb_script_editor` getCurrentFile/getLine/getLinesCount; `wb_entity_select` getSelected; `wb_entity_modify` getProperty/listProperties/listArrayItems/getWorldTransform; `wb_component` list |

  Denied, among others: every file writer (`project` write, `script_create`, `prefab`, `config_create`,
  `layout_create`, `mod` create/build/validate, ...), `wb_projects` open, `wb_play`, `wb_stop`, `wb_save`, every
  entity, layer, component, clipboard, localization and script-editor write, `wb_undo_redo`, `wb_execute_action`,
  `wb_reload`, `wb_open_resource`, `wb_launch` and `wb_cleanup`. Read tools that take a path pass one small check: no
  absolute, UNC, drive-relative or `..` path outside the pin's project root and game install (case-insensitive on
  `/mnt/<drive>`), and no relative `projectPath`. **Enabling any write tool later needs its own reviewed per-tool
  destination policy** (where each argument and each generated subdirectory really lands, as BK computes it).
- **The sealed-map guard** runs on every call: an identifier token `eden` or `everon` (split on non-alphanumerics and
  camelCase, so `EdenTerrain` is caught and `foreverOnline` is not) or the Everon root world GUID
  `853E92315D1D9EFE` refuses. Not detected: a read on a sealed world already open in Workbench without naming it; the
  runbook rule "never open Eden/Everon in an MCP-attached Workbench" covers it (writes are denied anyway).
- **The endpoint handshake** (MIGRATION_1_9 6.3) runs before every wb read; a refused call is answered with an MCP
  error and never forwarded. All read-only and without PowerShell: `netstat.exe -ano` gives the one PID listening on
  the port; `cscript.exe` running `wb_proc.js` (a WMI `Win32_Process` query, E:-resident) gives its exe path,
  creation time and command line. The exe must be the pin's Workbench with the pin's exe sha256; `-profile` and
  `-gproj` must be the launch path's (`pinlib.WB_LAUNCH`); `-dcLaunchId` must name a launch record that is a JSON
  object with the dd-launch fields, an ok wb record of this pin file (and build_key and launch path), and the process
  must have been created -5..+300 s from the record's utc. Exactly one `console.log` under the profile names that
  launch id after the record, and it must be the newest log of the profile; it must start 0..+120 s after the
  process creation (whole seconds), show the pin's Workbench as its working directory and the process's `-gproj`,
  and report `Creating game instance(...), version X` equal to the pin's `engine_version_expected`. 6.3 asks for a
  read-only handler call there; no deployed handler returns `GetBuildVersion()`, so the log of the same process,
  bound by its creation time, is read instead. A Workbench started by hand (no `-dcLaunchId`, a OneDrive profile, a
  C: project) fails by design. Residual, documented: a replacement process that writes no log of its own within the
  same second, and a project changed inside a running Workbench, are not detected (we never do either).
- **R1.** Both Windows helpers run with Windows `TEMP` and `TMP` set to `E:\DriveDiT\tmp\mcp` (`WSLENV=TEMP/p:TMP/p`
  path translation; the gate creates the dir through the tools/pin write guard), and `wb_proc.js --env` proves it once
  per session before any query (measured: it reports `E:\...`). The claim is **no tool-owned C: file writes**. WMI's
  own repository and event-log activity is Windows service behaviour, outside these tools and outside R1; it is not
  claimed to be zero.
- **Cost, measured 2026-11-07:** `netstat.exe -ano` 0.03 s, `wb_proc.js --env` 0.08 s once, a process query 0.16 s;
  the exe hash (68 MB) once per path, size and mtime; the log heads (256 KiB each, only dirs touched since the
  record). About 0.2 s per wb read.

Build the indexes (agent, E: only, about 15 s each; `--dry-run` first). Apply the build_key and engine sidecars of
both pins first: the build needs `engine_version_expected`, and the stamps then carry the build_key:

```bash
SD=/mnt/e/drivedit_realign/program/1_9/work/dl/sd_stable R=/mnt/e/drivedit_realign/mcp
python3 -I -B tools/enfusion-mcp/api_index.py build --dump-git $SD --pin-id 1.9.0.37 \
  --out $R/pins/1.9.0.37/data --shared $R/data --apply
python3 -I -B tools/enfusion-mcp/api_index.py build --dump-git $SD --pin-id 1.8.0.13 \
  --out $R/pins/1.8.0.13/data --shared $R/data --apply
python3 -I -B tools/enfusion-mcp/api_index_diff.py --old $R/pins/1.8.0.13/data --new $R/pins/1.9.0.37/data \
  --expect tools/enfusion-mcp/expect_1.8.0.13_to_1.9.0.37.txt     # from the repo root; RESULT: OK
```

Then copy `mcp_gate.py`, `endpoint_handshake.py`, `bk_tools.py`, `wb_proc.js`, `index_sources.json`, `api_index.py`,
`api_index_diff.py` and the wrapper to `$EMCP_ROOT` (the gate finds `tools/pin` at `E:\DriveDiT\tools\pin`, or
`DD_PIN_TOOLS`), and check with `enfusion-mcp-wsl.sh --print-env` (`pin gate: ok`). `endpoint_handshake.py
--pin-active` runs the handshake by hand against a running Workbench.

## Network: Windows host discovery under WSL NAT

- This machine is Windows 10 22H2 (19045) with WSL 2.3.26 in **NAT** mode. Mirrored mode needs Windows 11 22H2+.
- At every start the wrapper reads `wslinfo --networking-mode`:
  - `mirrored`: the host is `127.0.0.1`;
  - otherwise: the host is the `via` address of `ip -4 route show default` (today `172.19.208.1`), or the default
    route's gateway in `/proc/net/route` when `ip` is missing or fails. It must be an IPv4 address.
  - If there is none (for example a device-only default route), the wrapper exits 1 with
    `cannot determine the Windows host IP ...`. Then set `ENFUSION_WORKBENCH_HOST` in the MCP entry's `env` to the IPv4
    address of the `vEthernet (WSL)` adapter (Windows `ipconfig`).
  - It does this per start because the NAT subnet can change after a reboot. Keep `env` in the MCP entry empty unless discovery fails.
- Workbench's NET API listens on `0.0.0.0:5775`, so no `netsh portproxy` is needed. `enfusion-mcp-wsl.sh --check` tests TCP reachability without sending a request.
- Security: `0.0.0.0:5775` exposes editor actions to the LAN, as far as Windows Firewall allows. Consider scoping the inbound rule for 5775 to the WSL vEthernet subnet (`172.16.0.0/12`). That is a Windows change for the user.

## Handlers

- BK's 21 `EMCP_WB_*.c` NET API handler scripts are in the build at
  `/mnt/e/drivedit_realign/mcp/build/enfusion-mcp-BK/mod/Scripts/WorkbenchGame/EnfusionMCP/` (101,273 B, identical to the pin).
- The deploy step (not this directory) copies them into
  `E:\DriveDiT\addons\DataCapture\Scripts\WorkbenchGame\EnfusionMCP\`.
- Keep **exactly one** copy in the loaded project set. Do not load DataTest (C:) in the same session, and do not create `DriveDiT_EMCP` as well: duplicate `EMCP_WB_*` classes break the whole WorkbenchGame module.
- Workbench builds its NET API dispatch table at process start (BK `d6909d2`), so new handlers need a **Workbench restart**.

## User steps (in order, after the DataCapture deploy)

1. **Close Workbench.** Never save the truncated `DataDrive` world in the old session.
2. **Relaunch Workbench on E:**, with `-profile "E:\DriveDiT\wb_profile"` and `-gproj "E:\DriveDiT\addons\DataCapture\addon.gproj"`. Use one of:
   - **preferred:** DD-0's `E:\DriveDiT\tools\launch\wb_R1.bat` (pin preflight against the active pin + pinned arguments);
   - **from WSL (Steam running):** `bash /mnt/e/drivedit_realign/mcp/launch-workbench.sh`. It runs the same preflight
     (`--pin-active`, the pin named by `E:\DriveDiT\pin\ACTIVE`) and refuses if DD-0 is not deployed yet or no pin is
     active. Only before DD-0 is deployed, and knowing the build is unchecked:
     `WB_SKIP_PREFLIGHT=1 bash /mnt/e/drivedit_realign/mcp/launch-workbench.sh`.

   Do **not** launch from the Steam library or set Steam launch options: the capture program forbids library launches for
   the whole campaign (`CAPTURE_PROGRAM.md` §4.2 item 2, §7.1), and Steam would write its config on C:.

   Then check the new `console.log` under `E:\DriveDiT\wb_profile\logs\`:
   - `Adding relative directory 'E:\DriveDiT\wb_profile\profile'`;
   - `Loaded addons` lists DataCapture and **not** DataTest;
   - `Module: WorkbenchGame; loaded 191x files` (170 base + 21 handlers);
   - no `SCRIPT (E)` line naming `EMCP_WB_`.

   If the profile still points at OneDrive, stop, because captures would land on C:.
3. **Check the wrapper, then swap the MCP entry.**
   - `bash /mnt/e/drivedit_realign/mcp/enfusion-mcp-wsl.sh --check` (prints to stderr; opens one TCP connection to port 5775
     and sends nothing). Expect the three guards at `1`, `guards compiled into dist: yes`, the host `172.19.208.1` (or the
     current gateway) and `TCP ... reachable`.
   - The swap writes a few hundred bytes to `~/.claude.json` on C:, and only the user runs it:
     ```bash
     claude mcp remove enfusion-mcp -s user && claude mcp add -s user enfusion-mcp -- /mnt/e/drivedit_realign/mcp/enfusion-mcp-wsl.sh
     ```
   - This is the same as `claude-mcp-entry.json`. Keeping the name `enfusion-mcp` keeps the `mcp__enfusion-mcp__*` prefix.
   - In the C: repo's `.claude/settings.local.json`, delete the stale allowlist names `mcp__enfusion-mcp__project_browse` and `mcp__enfusion-mcp__prefab_create`.
   - Do **not** allowlist `…__project`, `…__prefab`, `…__wb_cleanup` or any other tool with write actions: the permission prompt is the only check on their arguments (see Rules).
4. **Restart Claude Code, run `/mcp`, and probe in this order:**
   1. `wb_diagnose`: it never launches or copies anything. It may report `error` instead of `up_no_handlers` (a BK classifier bug, see below).
   2. `wb_connect`: expect `Workbench Connected` with `Info: EnfusionMCP Workbench bridge active`.
   3. `wb_state`.

   Then follow the capture runbook R0-R4 in `CAPTURE_PROGRAM.md` §7.2.

## Rules

- **Never use `wb_launch` or `mod build` from WSL.** Both are blocked by the guards, so don't try to work around them. Launch Workbench only with `wb_R1.bat` or `launch-workbench.sh`.
- **Never use `wb_cleanup`.** With `all: true` it deletes the handlers from DataCapture (Workbench then loses the bridge until they are redeployed and Workbench restarts), and with `modDir` it deletes from any directory it is given, C: included.
- **Tool arguments bypass the wrapper's path defaults.** 13 BK tools (`project`, `prefab`, `mod`, `script_create`, `config_create`, `server_config`, `layout_create`, `layout_recipe`, `menu_create`, `gamemode_scaffold`, `scenario_create_conflict`, `animation_graph`, `workshop_info`) take a `projectPath` argument that replaces `ENFUSION_PROJECT_PATH`; `animation_graph` and `mod` also take an `outputPath`. Never pass a `/mnt/c/...` path (or a Windows `C:` path) in any of them, and approve such calls only after reading the arguments. Relative paths resolve under `$EMCP_ROOT/cwd`.
- Before handlers are loaded, use only `wb_diagnose` and `wb_connect` as probes. With the guards on, other `wb_*` calls fail with `LAUNCH_FAILED` instead of copying files.
- Never point `ENFUSION_PROJECT_PATH` at the OneDrive addons dir on C: (the wrapper refuses it).

## Known quirks

- `wb_diagnose` matches only `not existing Net API function`, but 1.8 replies `Undefined API func`. So it reports `error` rather than `up_no_handlers`.
- `wb_connect` reports `Connection Failed` both when Workbench is down and when the handlers are not loaded; `wb_diagnose` tells them apart (`refused` vs `error`).
- `wb_projects list` prints "No projects loaded". BK reads `result.projects`, but Workbench returns `{"Loaded Projects": [...]}`.
- The `NO_AUTO_INSTALL` error text still says "Load the DriveDiT_EMCP handler addon". After the re-home, read it as "deploy the handlers into DataCapture and restart Workbench". The patch is kept byte-identical to the reviewed version (sha256 above).
- `npm prune` rewrites the root `version` field in the build copy's `package-lock.json` (BK ships 0.7.0 there). No dependency changes.
- Nobody has yet compiled BK's exact handler set on 1.8. All 108 method names exist in the 1.8 docs, and a sibling handler set compiled on 1.8.0.13. The `Module: WorkbenchGame` line in step 2 settles it.
- BK cannot read Workbench logs or report compile errors. Use `console.log` and `script.log` under the E: profile.

## Verification record (2026-10-20)

- `build-bk.sh --data18`: exit 0. HEAD is `0acfa88`. Both guard patches are applied (`git apply --reverse --check`) and present in `dist/`. 173 packages installed, then 58 pruned. `node_modules` is 27 MB and `dist` 2.0 MB.
- `build-bk.sh --smoke` (run from both the live copy and this directory): serverInfo `{name: enfusion-mcp, version: 0.12.0}`, 54 tools.
- One local `api_search` call (no Workbench I/O) logged `Loaded index: 887 enfusion + 8085 arma classes (8972 total)`.
- `find ~ -maxdepth 3 -newer <start stamp>` listed only Claude Code's own `~/.claude.json` and `~/.claude/backups/`. Its `enfusion-mcp` entry was unchanged.
- `/tmp/node-compile-cache` was untouched.
- Review (`p0/dd3-review.md`): the path guards, forced guards, host discovery, launcher refusals and the smoke assertion were
  tested with stubs and E:-only roots; a second build run in a scratch E: root converged (idempotent). Workbench was not launched.

## Updating the pin

1. `git -C $EMCP_ROOT/src/enfusion-mcp-BK fetch`, then review the diff from `0acfa88`, especially `src/workbench/client.ts`, `src/tools/mod.ts` and `mod/Scripts/`.
2. Change `PIN` (and `EXPECT_TOOLS` if the tool count changes) in `build-bk.sh`, and confirm `git apply --check` of both patches against the new commit.
3. Delete `$EMCP_ROOT/build`, run `--data18` and `--smoke`, and record the new tool count here.
4. Redeploy the handlers if `mod/Scripts/` changed, then restart Workbench.
