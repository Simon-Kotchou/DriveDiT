# DataCapture Workbench addon (versioned source)

This folder is the git-tracked source of the Arma Reforger 1.8 capture addon `DataCapture`
(GUID `88BEF4187E0D2DFC`, depends only on vanilla ArmaReforger `58D0FB3206B6F859`). It replaces the untracked
`DataTest` addon (GUID `68F21EE03DB916A2`). Never load both in one Workbench session: they define the same script
classes.

| File | What it is |
|---|---|
| `addon.gproj` | Project file (same GameProject syntax as DataTest; new ID and GUID) |
| `DataDrive.ent` | Capture world: a SubScene of vanilla `worlds/Arland/Arland.ent` |
| `DataDrive_Layers/default.layer` | The capture layer (game mode, AI world, cars, spawn points, drive targets) |
| `deploy_manifest.txt` | The only files `tools/deploy_addon.py` may copy into the deployed addon |

Not versioned, because Workbench generates and owns them: `*.meta`, `resourceDatabase.rdb`, `UserMaps.desc`.
The Enforce scripts live in `src/enfusion/datacapture/` and are deployed to `Scripts/Game/DataCapture/` by the
manifest.

## Deploy

```bash
python3 -I -B tools/deploy_addon.py --dry-run     # show the plan
python3 -I -B tools/deploy_addon.py               # deploy to /mnt/e/DriveDiT/addons/DataCapture
python3 -I -B tools/deploy_addon.py --emcp-handlers <BK build>/mod/Scripts/WorkbenchGame/EnfusionMCP
python3 -I -B tools/test_deploy_addon.py -v       # self-test (temp targets only)
```

The tool also writes `Scripts/Game/DataCapture/DC_BuildInfo.c` (git sha, branch, dirty flag, deploy time, manifest
and content hashes) and tracks what it deployed in `<target>/.deploy_state.json`. It refuses C: targets, files
outside the manifest, and overwriting or deleting anything that changed in the target since it was deployed (for
example a layer saved by Workbench) unless `--force` is given, which backs the old file up first.

## Editing the world

- Edit the layer here, deploy, then reload the world in Workbench. If you change the world inside Workbench, copy the
  saved file back here before the next deploy (the deploy refuses to overwrite it otherwise).
- After any hand edit, check that the world loads without `Unexpected data at offset` or `Unknown keyword` in
  `console.log` and that all entities are created. Vectors are never quoted: `m_vHoodOffset 2.2 0.95 0`.
- Never save `DataDrive` from a Workbench session that loaded a truncated world: it overwrites the layer with what
  was loaded.
