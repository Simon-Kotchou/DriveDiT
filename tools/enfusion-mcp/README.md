# Enfusion MCP on WSL: setup runbook (DD-3)

This directory builds and runs [`steffenbk/enfusion-mcp-BK`](https://github.com/steffenbk/enfusion-mcp-BK) 0.12.0 as a
stdio MCP server for Claude Code in WSL, driving Arma Reforger Workbench over its NET API (port 5775).
It replaces the abandoned npm `enfusion-mcp` 0.6.1 entry: its upstream repo is gone, it defaults to `127.0.0.1` (refused
under WSL NAT), and it has no WSL path handling.

Every write goes to the data drive. The system drive is never written (a project rule), and the WSL disk (`~`, `/tmp`,
`/var`, ...) lives on it by default.
The scripts accept a work root, project path, `.gproj`, pin dir or Workbench profile only where the write boundary of
`tools/local_config.py` allows (`local_config.py check-write PATH`, the same check every Python writer uses): the
data drive of the local config, verified in `/proc/self/mountinfo` as that drive's own mount, or RAM (`/dev/shm`);
never a volume whose root holds `Windows/System32`, whatever the config says (`SYSTEM_DRIVE` only words the refusal).
Relative paths are refused as well: Claude Code starts the wrapper with its own cwd, its project folder.

**Local paths (DD-22).** `<REALIGN_ROOT>`, `<DATA_ROOT>`, `<GAME_INSTALL>`, `<TOOLS_INSTALL>` and the system drive are
keys of the machine's untracked local config `drivedit.local.env` (`<KEY_WIN>`: the Windows form), read through
`tools/local_config.py`. The installed MCP root holds a copy of `local_config.py` next to the scripts, and the MCP
entry tells it the file: `"env": {"DRIVEDIT_LOCAL_CONFIG": "<DATA_ROOT>/config/drivedit.local.env"}`
(`claude-mcp-entry.json`; `python3 -I -B tools/local_config.py render tools/enfusion-mcp/claude-mcp-entry.json`
prints it with this machine's values). Without the file the wrapper and the gate refuse (exit 2) and name the key.

Evidence and background: `<REALIGN_ROOT>/enfusion/mcp-setup.md`, `FIX_PLAN.md` §5, `program/CAPTURE_PROGRAM.md` §7.
Review record: `<REALIGN_ROOT>/p0/dd3-review.md`.

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
| `launch-workbench.sh` | WSL fallback for DD-0's `wb_R1.bat`: same arguments (`-profile "<DATA_ROOT_WIN>\wb_profile"`, DataCapture gproj, `-forceUpdate -maxFPS 60`), runs the DD-0 pin preflight against the **active** pin (`preflight.py --pin-active`: the pin that `<DATA_ROOT_WIN>\pin\ACTIVE` names, switched only by DD-0's `pin_activate.py`; `WB_PIN_DIR` overrides the pin dir, `WB_PIN` is refused), names no game version, and starts Workbench only if the preflight (run on `/usr/bin/python3`) printed exactly one launch id and `launch_<id>.json` exists, as `wb_R1.bat` does. Its `WB_DIR` is the Tools install in DD-0's `pinlib.LAUNCH_PATHS`, and the preflight's `launch_path` check refuses any pin not made from that install. It refuses if Workbench already runs or that cannot be checked (it reads `tasklist /FO CSV`: the default table format truncates the image name to `ArmaReforgerWorkbenchStea`, so a full-name match there never fires) |
| `claude-mcp-entry.json` | the user-scope MCP entry (same as the `claude mcp add` command below) |
| `mcp_gate.py` | DD-14a. What the wrapper execs: picks the pin (`EMCP_PIN`, else ACTIVE), serves only that pin's index and project root, binds the endpoint host, and proxies the stdio session through the `bk_tools.py` policy (deny by default: reads, and wb reads after the handshake; every write denied; sealed-map guard) |
| `endpoint_handshake.py`, `wb_proc.js` | DD-14a. The read-only endpoint handshake of MIGRATION_1_9 6.3 (also a CLI); `wb_proc.js` is its WMI process query, run by cscript.exe (no PowerShell) |
| `bk_tools.py` | DD-14a. The gate's deny-by-default allow-list (path-free BK read tools and wb getters, each with its allowed argument keys) and the sealed-map guard |
| `index_sources.json` | DD-14a. Pin id to dump tag and commit, for the index provenance check |
| `api_index.py` | DD-14a. Builds a per-pin API index in BK's JSON shape from the official script dump at the pin's tag, stamped with the pin (`api/INDEX_PIN.json`) |
| `api_index_diff.py`, `expect_1.8.0.13_to_1.9.0.37.txt` | DD-14a, CC-05. Diffs two per-pin indexes and checks them against the API delta (API_DELTA 2-3, G19-0) and the planned surface (`tools/compat19/planned_surface.txt`) |
| `tests/test_handshake.py`, `tests/fixtures/` | DD-14a tests: index, provenance, diff, gate config, policy (every write and path-taking tool denied, the path-free reads allowed), handshake (stub Windows probes), proxy (stub server). The fixture dumps are synthetic classes, not game scripts |
| `patches/bk-wsl-guard.patch` | env-gated guards in `src/workbench/client.ts` (sha256 `4eb60058…d6eb`; DD-22 changed comment and message text only) |
| `patches/20-bk-mod-build-guard.patch` | env-gated guard in `src/tools/mod.ts` (sha256 `58de5533…0b7e`; DD-22 changed comment and message text only) |

The live install is `EMCP_ROOT=<REALIGN_ROOT>/mcp`. Its copies of these files are byte-identical to the ones here
(scripts at the top level, patches in `patches/`). After changing a file here, copy it there.
No BK source, `node_modules` or build output is committed.

## MCP root layout (`EMCP_ROOT`, default `<REALIGN_ROOT>/mcp`)

```
src/enfusion-mcp-BK/            pristine clone (untrusted content, never built in place)
src/others/fma-rc_enfusion-mcp/ pristine clone, source of the 1.8 API index
build/enfusion-mcp-BK/          build copy at the pin + guard patches; dist/ (2.0 MB) and prod node_modules/ (27 MB)
data/                           api/ (fma-rc 1.8) + BK patterns/ recipes/ wiki/; api/SOURCE_fma-rc_commit.txt
home/ tmp/ xdg/ npm-cache/ npm-logs/   redirected HOME, TMPDIR, XDG_*, npm cache and logs
cwd/                            the server's working directory (relative tool paths land here, not on the system drive)
local_config.py                 (DD-22) the loader of the local config, a copy of the repo's tools/local_config.py
```

## Build (agent or user, data drive only)

Prerequisite: the two pristine clones exist (one-time; needs GitHub):

```bash
R=<REALIGN_ROOT>/mcp
git clone https://github.com/steffenbk/enfusion-mcp-BK "$R/src/enfusion-mcp-BK"
git clone https://github.com/fma-rc/enfusion-mcp "$R/src/others/fma-rc_enfusion-mcp"
```

Then (needs the npm registry; the build reads the fma-rc index from commit `4cd14db` whatever is checked out):

```bash
bash <REALIGN_ROOT>/mcp/build-bk.sh --data18   # pin, guard patches, npm ci, tsc, guard check, prune, 1.8 API data
bash <REALIGN_ROOT>/mcp/build-bk.sh --smoke    # expect serverInfo {name: enfusion-mcp, version: 0.12.0} and 54 tools; exit 0
```

Re-running either command is safe. To move the pin, see the last section.

**When the build copy is reset (DD-22).** `build-bk.sh` never discards edits it cannot prove are its own. Before it
applies the guard patches it compares the build copy's tracked files with three trees, each built in a scratch index
under `$EMCP_ROOT/tmp`: the pinned commit, the pinned commit plus the current patches, and the pinned commit plus
the *previous* patch set (the patches the last build applied: copies in `build/applied-patches/`, named by
`build/applied-patches.list`, which every build writes). A clean copy gets the patches; a copy that already holds them
is left alone; a copy that holds exactly the previous patch set is reset to the pin (`checkout -f`, after
`git apply -R --check` of those patches and the tree comparison prove that the difference is theirs; untracked
`node_modules/` and `dist/` stay). Anything else refuses and touches nothing: HEAD at another commit, staged
changes, or any other edit. A build copy made before the record existed (its patches in the old wording) needs the
old patch files once: `EMCP_PREVIOUS_PATCHES=<old bk-wsl-guard.patch>:<old 20-bk-mod-build-guard.patch>`, for example
exported with `git show <last pre-DD-22 main commit>:tools/enfusion-mcp/patches/<name>` into the MCP root's `tmp/`.

What the build redirects, and where to:

| Default target (system drive) | Redirected to |
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
| `ENFUSION_MCP_NO_AUTO_LAUNCH=1` | `20-bk-mod-build-guard.patch` | `mod` action=`build`. It would spawn a second Workbench with Linux paths, defaulting its output to the Tools install |
| `ENFUSION_MCP_WSL_PATHS=1` | `bk-wsl-guard.patch` | whole-string params `/mnt/<x>/...` are sent to Workbench as `<X>:/...` (for example `RegisterResourceFile(absFilePath)`). Resource paths (`Prefabs/...`, `{GUID}...`) are untouched |

The patched code tests for exactly `"1"`, so `0`, `true` or `yes` would each turn a guard off. The wrapper therefore ignores
any incoming value (it logs `ignoring ...` on stderr) and exports `1`. It also refuses to start a `dist/` whose
`workbench/client.js` and `tools/mod.js` lack the guard code, so an unpatched build can never run behind it.

Other wrapper defaults (each can be overridden by exporting it, or through `env` in the MCP entry):
- `ENFUSION_PROJECT_PATH`: set by `mcp_gate.py` to the pin's project root (`<DATA_ROOT>/addons` for a stable pin); generators (`project`, `prefab`, `mod` create, `script_create`, ...) write there. A value on the system drive or the WSL disk is refused (exit 2), and so is any value other than the pin's root.
- `ENFUSION_DEFAULT_MOD=DataCapture`: if anything ever injects handlers, it targets the open addon, not `ArmaDrive` on the system drive.
- `ENFUSION_MCP_DATA_DIR`: set by `mcp_gate.py` to the pin's index (see "Per-pin index"); BK silently falls back to an empty index when it is empty, so the gate checks every api file first.
- `ENFUSION_GAME_PATH` (`GAME_INSTALL`) and `ENFUSION_WORKBENCH_PATH` (`TOOLS_INSTALL`) are only read. Everything BK does with the Workbench path that could write (launch, `mod build`) is guarded.
- `HOME`, `TMPDIR` and `XDG_*` point at `$EMCP_ROOT`, and the server runs in `$EMCP_ROOT/cwd`. stdout carries only JSON-RPC.

## Per-pin index and the endpoint handshake (DD-14a, CC-05)

Since 2026-11-05 the active pin is 1.9.0.37, and the fma-rc index in `data/` is the 1.8 scrape: it has no
`VehicleWheelEffectComponent`, `GetHandbrake` or `Update*Configuration`. The wrapper therefore no longer serves
`data/`. It execs `mcp_gate.py`, which serves `$EMCP_ROOT/pins/<pin id>/data` only:

- **The pin.** `EMCP_PIN=<id>` in the MCP entry's `env` (a second config), else `<DATA_ROOT_WIN>\pin\ACTIVE`. A stable
  `EMCP_PIN` must be the active pin: the stable project root and installs are the active pin's. A 1.8 config after
  the re-pin is the cold pin, `EMCP_PIN=1.8.0.13-cold` (project root `<DATA_ROOT_WIN>\addons_c18`).
- **The index.** `pins/<id>/data/api/INDEX_PIN.json` must name that pin id and pin file sha256 (and its build_key,
  when both are known), and every api file must match its hash. Its source must be the dump tag and commit that
  `index_sources.json` maps the pin to, and that tag must be `v<engine_version_expected>` of the pin's engine
  sidecar, so a 1.8 dump can never stamp a 1.9 pin. There is no fallback: a missing or foreign index refuses
  (exit 2) before the server starts, and `--print-env` says why.
- **The project root** is `pinlib.DEPLOY_ROOTS[launch path]`; an `ENFUSION_PROJECT_PATH` set by hand must be it, and
  a deploy stamped for another pin (`.deploy_state.json` `target_pin`) refuses. `ENFUSION_GAME_PATH` and
  `ENFUSION_WORKBENCH_PATH` are the pin's installs.
- **The endpoint.** `ENFUSION_WORKBENCH_HOST` must be loopback, or the WSL NAT gateway proven to be the local
  Windows host (the address of its `vEthernet (WSL)` adapter). All three are required: exactly one up IPv4 default route of any kind (a
  device-only `tun0` default route counts, Codex round 3), with a gateway, via `eth0`; `/etc/resolv.conf` carrying
  WSL's generated header with that gateway as its only nameserver; and `ipconfig.exe` (read-only, Windows `TEMP`/`TMP`
  on the data drive as for the helpers below) showing a `vEthernet (WSL...)` adapter whose IPv4 address is that gateway. The route
  and resolver files alone can describe a stale or bridged setup; the adapter address is the Windows side of the WSL
  NAT switch. Anything else (a LAN or VPN gateway such as `198.51.100.1`, or no such adapter) refuses, and only
  loopback is accepted: the handshake can only vouch for the local Workbench. `ipconfig.exe` runs only for a
  non-loopback host. The gate passes the
  checked host and port to the server, and the listener must serve that address (`0.0.0.0`, `[::]` or the host).
- **Deny by default** (coordinator decisions, Codex rounds 2-3: the runbook never uses MCP mutations; agents never
  call MCP tools to change anything, the user presses Play; and the allow-list holds only what we need). `bk_tools.py`
  allows only these, each with an explicit set of argument keys, and denies every other tool, every other action,
  unknown tools, unknown actions, a missing action and any other argument key:

  | Class | Tools and actions |
  |---|---|
  | read (no handshake) | `api_search`, `component_search`, `wiki_search`, `wiki_read`, `wb_knowledge` |
  | wb read (after the handshake) | `wb_connect`, `wb_state`, `wb_entity_list`, `wb_entity_inspect`; `wb_entity_modify` getProperty/listProperties/listArrayItems/getWorldTransform; `wb_entity_select` getSelected; `wb_script_editor` getCurrentFile/getLine/getLinesCount (no `path`); `wb_layers` list/isVisible/getInfo; `wb_localization` getTable/listLanguages; `wb_component` list; `wb_terrain` getHeight/getBounds |

  Denied, among others: every file writer (`project` write, `script_create`, `prefab`, `config_create`,
  `layout_create`, `mod` create/build/validate, ...), `wb_projects` open, `wb_play`, `wb_stop`, `wb_save`, every
  entity, layer, component, clipboard, localization and script-editor write, `wb_undo_redo`, `wb_execute_action`,
  `wb_reload`, `wb_open_resource`, `wb_launch` and `wb_cleanup`. Also denied (Codex round 3): `wb_validate`
  (`MaterialValidator.CheckTextures()` can remove an item from Workbench's shared texture queue, and BK's request
  fields do not match the handlers'), and every tool that takes a filesystem, resource or mod path: `project`,
  `workshop_info`, `wb_read_props`, `asset_search`, `game_browse`, `game_read` (the game install holds the sealed
  Everon world), `wb_resources`, `wb_prefabs` and `wb_projects`; and (Codex round 4) `wb_diagnose`, which lists each
  addon's `Scripts/WorkbenchGame/EnfusionMCP` directory without resolving symlinks (BK `client.ts:288`), an indirect
  filesystem read. `wb_connect` stays: it is a ping. There is no read-path check any more: no allowed
  entry takes a path. **Enabling any write tool, or re-enabling any path-taking read tool, needs its own reviewed
  per-tool policy first** (for writes, where each argument and generated subdirectory really lands, as BK computes it;
  for reads, every read-location field per action, resolved against its real base, symlinks followed, rooted and
  resource namespaces handled).
- **The sealed-map guard** runs on every call: an identifier token `eden` or `everon` (split on non-alphanumerics and
  camelCase, so `EdenTerrain` is caught and `foreverOnline` is not) or the Everon root world GUID
  `853E92315D1D9EFE` refuses. Not detected: a read on a sealed world already open in Workbench without naming it; the
  runbook rule "never open Eden/Everon in an MCP-attached Workbench" covers it (writes are denied anyway).
- **The endpoint handshake** (MIGRATION_1_9 6.3) runs before every wb read; a refused call is answered with an MCP
  error and never forwarded. All read-only and without PowerShell: `netstat.exe -ano` gives the one PID listening on
  the port; `cscript.exe` running `wb_proc.js` (a WMI `Win32_Process` query, resident on the data drive) gives its exe path,
  creation time and command line. The exe must be the pin's Workbench with the pin's exe sha256; `-profile` and
  `-gproj` must be the launch path's (`pinlib.WB_LAUNCH`); `-dcLaunchId` must name a launch record that is a JSON
  object with the dd-launch fields, an ok wb record of this pin file (and build_key and launch path), and the process
  must have been created -5..+300 s from the record's utc. Exactly one `console.log` under the profile names that
  launch id after the record, and it must be the newest log of the profile; it must start 0..+120 s after the
  process creation (whole seconds), show the pin's Workbench as its working directory and the process's `-gproj`,
  and report `Creating game instance(...), version X` equal to the pin's `engine_version_expected`. 6.3 asks for a
  read-only handler call there; no deployed handler returns `GetBuildVersion()`, so the log of the same process,
  bound by its creation time, is read instead. A Workbench started by hand (no `-dcLaunchId`, a OneDrive profile, a
  system-drive project) fails by design. Residual, documented: a replacement process that writes no log of its own within the
  same second, and a project changed inside a running Workbench, are not detected (we never do either).
- **R1.** The Windows helpers (`ipconfig.exe`, `netstat.exe`, `cscript.exe`) run with Windows `TEMP` and `TMP` set
  to `<DATA_ROOT_WIN>\tmp\mcp` (`WSLENV=TEMP/p:TMP/p` path translation; the gate creates the dir through the tools/pin
  write guard), and `wb_proc.js --env` proves it once per session before any query (measured: it reports the data-drive dir). The claim is **no tool-owned system-drive file writes**. WMI's
  own repository and event-log activity is Windows service behaviour, outside these tools and outside R1; it is not
  claimed to be zero.
- **Cost, measured 2026-11-07:** `ipconfig.exe` 0.04 s once per session (non-loopback host only), `netstat.exe -ano`
  0.03 s, `wb_proc.js --env` 0.08 s once, a process query 0.16 s;
  the exe hash (68 MB) once per path, size and mtime; the log heads (256 KiB each, only dirs touched since the
  record). About 0.2 s per wb read.

Build the indexes (agent, data drive only, about 15 s each; `--dry-run` first). Apply the build_key and engine sidecars of
both pins first: the build needs `engine_version_expected`, and the stamps then carry the build_key:

```bash
SD=<REALIGN_ROOT>/program/1_9/work/dl/sd_stable R=<REALIGN_ROOT>/mcp
python3 -I -B tools/enfusion-mcp/api_index.py build --dump-git $SD --pin-id 1.9.0.37 \
  --out $R/pins/1.9.0.37/data --shared $R/data --apply
python3 -I -B tools/enfusion-mcp/api_index.py build --dump-git $SD --pin-id 1.8.0.13 \
  --out $R/pins/1.8.0.13/data --shared $R/data --apply
python3 -I -B tools/enfusion-mcp/api_index_diff.py --old $R/pins/1.8.0.13/data --new $R/pins/1.9.0.37/data \
  --expect tools/enfusion-mcp/expect_1.8.0.13_to_1.9.0.37.txt     # from the repo root; RESULT: OK
```

Then copy `mcp_gate.py`, `endpoint_handshake.py`, `bk_tools.py`, `wb_proc.js`, `index_sources.json`, `api_index.py`,
`api_index_diff.py`, the wrapper and `tools/local_config.py` (DD-22) to `$EMCP_ROOT` (the gate finds `tools/pin` at
`<DATA_ROOT>/tools/pin` of the local config `$DRIVEDIT_LOCAL_CONFIG` names, or at `DD_PIN_TOOLS`), and check with `enfusion-mcp-wsl.sh --print-env` (`pin gate: ok`). `endpoint_handshake.py
--pin-active` runs the handshake by hand against a running Workbench.

## Network: Windows host discovery under WSL NAT

- The tools support WSL in **NAT** mode (the Windows host is the default gateway, verified against the Windows
  `vEthernet (WSL)` adapter) and in mirrored mode (the host is `127.0.0.1`; mirrored mode needs a recent Windows 11).
- At every start the wrapper reads `wslinfo --networking-mode`:
  - `mirrored`: the host is `127.0.0.1`;
  - otherwise: the host is the `via` address of `ip -4 route show default`, or the default
    route's gateway in `/proc/net/route` when `ip` is missing or fails. It must be an IPv4 address.
  - If there is none (for example a device-only default route), the wrapper exits 1 with
    `cannot determine the Windows host IP ...`. Then set `ENFUSION_WORKBENCH_HOST` in the MCP entry's `env` to the IPv4
    address of the `vEthernet (WSL)` adapter (Windows `ipconfig`).
  - It does this per start because the NAT subnet can change after a reboot. Keep `env` in the MCP entry empty unless discovery fails.
  - The gate then accepts that host only if `endpoint_handshake.nat_gateway` proves it is the Windows side of the WSL
    NAT switch (one default route, WSL's resolver, and the `vEthernet (WSL...)` adapter address from `ipconfig.exe`;
    see "The endpoint" above); otherwise it refuses before the server starts.
- Workbench's NET API listens on `0.0.0.0:5775`, so no `netsh portproxy` is needed. `enfusion-mcp-wsl.sh --check` tests TCP reachability without sending a request.
- Security: `0.0.0.0:5775` exposes editor actions to the LAN, as far as Windows Firewall allows. Consider scoping the inbound rule for 5775 to the subnet of the `vEthernet (WSL)` adapter (Windows `ipconfig`). That is a Windows change for the user.

## Handlers

- BK's 21 `EMCP_WB_*.c` NET API handler scripts are in the build at
  `<REALIGN_ROOT>/mcp/build/enfusion-mcp-BK/mod/Scripts/WorkbenchGame/EnfusionMCP/` (101,273 B, identical to the pin).
- The deploy step (not this directory) copies them into
  `<DATA_ROOT_WIN>\addons\DataCapture\Scripts\WorkbenchGame\EnfusionMCP\`.
- Keep **exactly one** copy in the loaded project set. Do not load DataTest (on the system drive) in the same session, and do not create `DriveDiT_EMCP` as well: duplicate `EMCP_WB_*` classes break the whole WorkbenchGame module.
- Workbench builds its NET API dispatch table at process start (BK `d6909d2`), so new handlers need a **Workbench restart**.

## User steps (in order, after the DataCapture deploy)

1. **Close Workbench.** Never save the truncated `DataDrive` world in the old session.
2. **Relaunch Workbench on the data drive**, with `-profile "<DATA_ROOT_WIN>\wb_profile"` and `-gproj "<DATA_ROOT_WIN>\addons\DataCapture\addon.gproj"`. Use one of:
   - **preferred:** DD-0's `<DATA_ROOT_WIN>\tools\launch\wb_R1.bat` (pin preflight against the active pin + pinned arguments);
   - **from WSL (Steam running):** `bash <REALIGN_ROOT>/mcp/launch-workbench.sh`. It runs the same preflight
     (`--pin-active`, the pin named by `<DATA_ROOT_WIN>\pin\ACTIVE`) and refuses if DD-0 is not deployed yet or no pin is
     active. There is no launch without the preflight and its launch record (DD-22 removed `WB_SKIP_PREFLIGHT`).

   Do **not** launch from the Steam library or set Steam launch options: the capture program forbids library launches for
   the whole campaign (`CAPTURE_PROGRAM.md` §4.2 item 2, §7.1), and Steam would write its config on the system drive.

   Then check the new `console.log` under `<DATA_ROOT_WIN>\wb_profile\logs\`:
   - `Adding relative directory '<DATA_ROOT_WIN>\wb_profile\profile'`;
   - `Loaded addons` lists DataCapture and **not** DataTest;
   - `Module: WorkbenchGame; loaded 191x files` (170 base + 21 handlers);
   - no `SCRIPT (E)` line naming `EMCP_WB_`.

   If the profile still points at OneDrive, stop, because captures would land on the system drive.
3. **Check the wrapper, then swap the MCP entry.**
   - `bash <REALIGN_ROOT>/mcp/enfusion-mcp-wsl.sh --check` (prints to stderr; opens one TCP connection to port 5775
     and sends nothing). Expect the three guards at `1`, `guards compiled into dist: yes`, the host (the current WSL
     gateway) and `TCP ... reachable`.
   - The swap writes a few hundred bytes to `~/.claude.json` on the system drive, and only the user runs it:
     ```bash
     claude mcp remove enfusion-mcp -s user && claude mcp add -s user enfusion-mcp -- <REALIGN_ROOT>/mcp/enfusion-mcp-wsl.sh
     ```
   - This is the same as `claude-mcp-entry.json`. Keeping the name `enfusion-mcp` keeps the `mcp__enfusion-mcp__*` prefix.
   - In the Claude Code project folder's `.claude/settings.local.json`, delete the stale allowlist names `mcp__enfusion-mcp__project_browse` and `mcp__enfusion-mcp__prefab_create`.
   - Do **not** allowlist `…__project`, `…__prefab`, `…__wb_cleanup` or any other tool with write actions: the permission prompt is the only check on their arguments (see Rules).
4. **Restart Claude Code, run `/mcp`, and probe in this order:**
   1. `wb_connect`: expect `Workbench Connected` with `Info: EnfusionMCP Workbench bridge active`.
   2. `wb_state`.

   (`wb_diagnose` is denied by the gate since DD-14a, Codex round 4; `enfusion-mcp-wsl.sh --check` tests TCP reachability.)

   Then follow the capture runbook R0-R4 in `CAPTURE_PROGRAM.md` §7.2.

## Rules

- **Never use `wb_launch` or `mod build` from WSL.** Both are blocked by the guards, so don't try to work around them. Launch Workbench only with `wb_R1.bat` or `launch-workbench.sh`.
- **Never use `wb_cleanup`.** With `all: true` it deletes the handlers from DataCapture (Workbench then loses the bridge until they are redeployed and Workbench restarts), and with `modDir` it deletes from any directory it is given, the system drive included.
- **Tool arguments bypass the wrapper's path defaults.** 13 BK tools (`project`, `prefab`, `mod`, `script_create`, `config_create`, `server_config`, `layout_create`, `layout_recipe`, `menu_create`, `gamemode_scaffold`, `scenario_create_conflict`, `animation_graph`, `workshop_info`) take a `projectPath` argument that replaces `ENFUSION_PROJECT_PATH`; `animation_graph` and `mod` also take an `outputPath`. Never pass a system-drive path (`<SYSTEM_DRIVE>/...`, or its Windows form) in any of them, and approve such calls only after reading the arguments. Relative paths resolve under `$EMCP_ROOT/cwd`.
- Before handlers are loaded, use only `wb_connect` (and `enfusion-mcp-wsl.sh --check`) as probes. With the guards on, other `wb_*` calls fail with `LAUNCH_FAILED` instead of copying files.
- Never point `ENFUSION_PROJECT_PATH` at the OneDrive addons dir on the system drive (the wrapper refuses it).

## Known quirks

- `wb_diagnose` (denied by the gate) matches only `not existing Net API function`, but 1.8 replies `Undefined API func`. So it reports `error` rather than `up_no_handlers`.
- `wb_connect` reports `Connection Failed` both when Workbench is down and when the handlers are not loaded; `enfusion-mcp-wsl.sh --check` tells a closed port apart (`NOT reachable`).
- `wb_projects list` prints "No projects loaded". BK reads `result.projects`, but Workbench returns `{"Loaded Projects": [...]}`.
- The `NO_AUTO_INSTALL` error text still says "Load the DriveDiT_EMCP handler addon". After the re-home, read it as "deploy the handlers into DataCapture and restart Workbench". The patch is kept byte-identical to the reviewed version (sha256 above).
- `npm prune` rewrites the root `version` field in the build copy's `package-lock.json` (BK ships 0.7.0 there). No dependency changes.
- Nobody has yet compiled BK's exact handler set on 1.8. All 108 method names exist in the 1.8 docs, and a sibling handler set compiled on 1.8.0.13. The `Module: WorkbenchGame` line in step 2 settles it.
- BK cannot read Workbench logs or report compile errors. Use `console.log` and `script.log` under the data-drive profile.

## Verification record (2026-10-20)

- `build-bk.sh --data18`: exit 0. HEAD is `0acfa88`. Both guard patches are applied (`git apply --reverse --check`) and present in `dist/`. 173 packages installed, then 58 pruned. `node_modules` is 27 MB and `dist` 2.0 MB.
- `build-bk.sh --smoke` (run from both the live copy and this directory): serverInfo `{name: enfusion-mcp, version: 0.12.0}`, 54 tools.
- One local `api_search` call (no Workbench I/O) logged `Loaded index: 887 enfusion + 8085 arma classes (8972 total)`.
- `find ~ -maxdepth 3 -newer <start stamp>` listed only Claude Code's own `~/.claude.json` and `~/.claude/backups/`. Its `enfusion-mcp` entry was unchanged.
- `/tmp/node-compile-cache` was untouched.
- Review (`p0/dd3-review.md`): the path guards, forced guards, host discovery, launcher refusals and the smoke assertion were
  tested with stubs and data-drive-only roots; a second build run in a scratch data-drive root converged (idempotent). Workbench was not launched.

## Updating the pin

1. `git -C $EMCP_ROOT/src/enfusion-mcp-BK fetch`, then review the diff from `0acfa88`, especially `src/workbench/client.ts`, `src/tools/mod.ts` and `mod/Scripts/`.
2. Change `PIN` (and `EXPECT_TOOLS` if the tool count changes) in `build-bk.sh`, and confirm `git apply --check` of both patches against the new commit.
3. Delete `$EMCP_ROOT/build`, run `--data18` and `--smoke`, and record the new tool count here.
4. Redeploy the handlers if `mod/Scripts/` changed, then restart Workbench.
