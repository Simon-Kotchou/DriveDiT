"""What each BK enfusion-mcp tool does, for the MCP gate (ticket DD-14a, CC-05; Codex round 1).

Classified from the BK source at the pinned commit (src/tools/*.ts, registerTool names and their action enums and
path arguments; tools/enfusion-mcp/README.md "Pinned versions"). A BK update must re-check this table; an unknown
tool is refused, and an unknown action of a known tool counts as a write.

Classes:
  deny      never forwarded (README "Rules")
  read      reads the index, the wiki or game/project files only: forwarded as is
  write     writes files under its project path (or a destination argument): handshake + destination check
  wb_read   talks to Workbench without changing anything: handshake
  wb_write  changes Workbench state or its files: handshake + destination check

Destination check (every write and wb_write): the effective base is the call's projectPath, or the pin's project
root. projectPath must resolve inside the pin's project root. Every other destination argument (DEST_KEYS) that is
absolute (/..., X:\\...) must resolve inside it too, and a relative one must stay inside the base; building_setup's
outputDir is resolved against the server's working directory, as BK does.

Sealed-map guard (every call): the arguments, keys and values, are split into identifier tokens (non-alphanumerics
and camelCase); a token eden or everon refuses, and so does the Everon root world GUID 853E92315D1D9EFE
({853E92315D1D9EFE}worlds/Eden/Eden.ent, docs/capture_contract/dc-2.0.md 3.1). So EdenTerrain is caught and
foreverOnline is not. Not detected: a call that acts on a sealed world already open in Workbench without naming it;
the runbook rule (never open Eden/Everon in Workbench) covers that.
"""
from __future__ import annotations

import os
import re

BK_COMMIT = "0acfa884228477043c6b4c2b8c7f0c270d648398"

DENY = {
    "wb_launch": "launches Workbench with Linux paths (README Rules: launch only through the launchers)",
    "wb_cleanup": "deletes handler files, from any directory it is given (README Rules)",
    "mod_build": "spawns a second Workbench (README Rules)",
}
DENY_ACTIONS = {("mod", "build"): "spawns a second Workbench with Linux paths (README Rules)"}

READ = {"api_search", "asset_search", "component_search", "game_browse", "game_read", "wiki_read", "wiki_search",
        "workshop_info"}
# Local tools with an action argument: these actions only read; every other action writes.
READ_ACTIONS = {"project": {"browse", "read"}, "prefab": {"inspect"}, "mod": {"validate"},
                "animation_graph": {"inspect"}, "gamemode_scaffold": {"list"}, "layout_recipe": {"list"},
                "menu_create": {"list"}}
WRITE = {"script_create", "config_create", "layout_create", "scenario_create_conflict", "server_config",
         "building_setup", "game_duplicate"}
WB_READ = {"wb_connect", "wb_diagnose", "wb_state", "wb_entity_list", "wb_entity_inspect", "wb_read_props",
           "wb_validate", "wb_terrain", "wb_knowledge"}
WB_READ_ACTIONS = {"wb_projects": {"list", "locate"}, "wb_entity_select": {"getSelected"},
                   "wb_entity_modify": {"getProperty", "listProperties", "listArrayItems", "getWorldTransform"},
                   "wb_layers": {"list", "isVisible", "getInfo"}, "wb_localization": {"getTable", "listLanguages"},
                   "wb_prefabs": {"getGuid", "locate", "getAncestor"}, "wb_resources": {"getInfo"},
                   "wb_script_editor": {"getCurrentFile", "getLine", "getLinesCount"}, "wb_clipboard": {"hasCopied"},
                   "wb_component": {"list"}}
WB_WRITE = {"wb_play", "wb_stop", "wb_save", "wb_undo_redo", "wb_open_resource", "wb_entity_create",
            "wb_entity_delete", "wb_entity_duplicate", "wb_execute_action", "wb_reload", "scenario_create"}
DEST_KEYS = ("outputPath", "outputDir", "destPath", "path", "templatePath", "layerPath", "gprojPath", "modDir",
             "agrPath", "asiPath", "agfPath")
SEALED_TOKENS = {"eden", "everon"}
SEALED_GUIDS = ("853E92315D1D9EFE",)


def known_tools() -> set:
    return (set(DENY) | READ | set(READ_ACTIONS) | WRITE | WB_READ | set(WB_READ_ACTIONS) | WB_WRITE
            | {t for t, _ in DENY_ACTIONS})


def classify(name: str, arguments) -> tuple:
    """(class, reason) for a tools/call. class: deny, read, write, wb_read, wb_write or unknown."""
    args = arguments if isinstance(arguments, dict) else {}
    action = args.get("action")
    if name in DENY:
        return "deny", DENY[name]
    if (name, action) in DENY_ACTIONS:
        return "deny", "%s action %s %s" % (name, action, DENY_ACTIONS[(name, action)])
    if name in READ:
        return "read", None
    if name in READ_ACTIONS:
        return ("read", None) if action in READ_ACTIONS[name] else ("write", None)
    if name in WRITE:
        return "write", None
    if name in WB_READ:
        return "wb_read", None
    if name in WB_READ_ACTIONS:
        return ("wb_read", None) if action in WB_READ_ACTIONS[name] else ("wb_write", None)
    if name in WB_WRITE:
        return "wb_write", None
    return "unknown", "tool %r is not in the gate's table of BK %s tools" % (name, BK_COMMIT[:7])


# --------------------------------------------------------------------------- sealed map


def _strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _strings(v, out)
    return out


def identifier_tokens(text: str) -> list:
    """'worlds/EdenTerrain_v2' -> ['worlds', 'eden', 'terrain', 'v', '2']: split on non-alphanumerics, camelCase and
    digit runs, lower-cased."""
    toks = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        chunk = re.sub(r"([a-z])([A-Z])", r"\1 \2", chunk)
        chunk = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", chunk)
        chunk = re.sub(r"([A-Za-z])([0-9])|([0-9])([A-Za-z])", lambda m: " ".join(g for g in m.groups() if g), chunk)
        toks.extend(t.lower() for t in chunk.split() if t)
    return toks


def sealed_reason(arguments):
    """Why the arguments name the sealed test map, or None."""
    for s in _strings(arguments, []):
        up = s.upper()
        for guid in SEALED_GUIDS:
            if guid in up:
                return "the arguments name the Everon root world GUID %s" % guid
        hit = SEALED_TOKENS.intersection(identifier_tokens(s))
        if hit:
            return "the arguments name the sealed test map (%s)" % ", ".join(sorted(hit))
    return None


# --------------------------------------------------------------------------- destinations


_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def _to_wsl(path: str, win_to_wsl) -> str:
    return win_to_wsl(path.replace("/", "\\")) if _WIN_ABS.match(path) else path


def _is_abs(path: str) -> bool:
    return path.startswith("/") or path.startswith("\\\\") or bool(_WIN_ABS.match(path))


def _inside(path: str, root: str) -> bool:
    path, root = os.path.normcase(path), os.path.normcase(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def destination_reason(name: str, arguments, project_root: str, server_cwd: str, win_to_wsl) -> str:
    """Why a writing call's destination lies outside the pin's project root, or None."""
    args = arguments if isinstance(arguments, dict) else {}
    root = os.path.realpath(project_root)
    base = root
    pp = args.get("projectPath")
    if pp is not None:
        if not isinstance(pp, str) or not pp.strip():
            return "%s: projectPath %r is not a path" % (name, pp)
        p = _to_wsl(pp, win_to_wsl)
        base = os.path.realpath(p if _is_abs(p) else os.path.join(server_cwd, p))
        if not _inside(base, root):
            return "%s: projectPath %s resolves to %s, outside this pin's project root %s" % (name, pp, base, root)
    for key in DEST_KEYS:
        v = args.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            return "%s: %s %r is not a path" % (name, key, v)
        if not v.strip():
            continue
        p = _to_wsl(v, win_to_wsl)
        if _is_abs(p):
            target = os.path.realpath(p)
            where = root
        elif key == "outputDir":       # BK resolves it against its working directory, not the project
            target = os.path.realpath(os.path.join(server_cwd, p))
            where = root
        else:
            rel = re.sub(r"^\{[0-9A-Fa-f]{16}\}", "", p).replace("\\", "/")
            target = os.path.realpath(os.path.join(base, rel))
            where = base
        if not _inside(target, where):
            return "%s: %s %s resolves to %s, outside %s" % (name, key, v, target, where)
    return None
