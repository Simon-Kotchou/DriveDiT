"""What the MCP gate lets through to BK enfusion-mcp: an explicit allow-list (ticket DD-14a, CC-05; Codex rounds 1-2).

Coordinator decision (round 2): the runbook never uses MCP mutations. Agents never call MCP tools to change files or
Workbench, and the user presses Play. CC-05 needs only the per-pin API index, api_search and a handshake that refuses a
mismatched Workbench. So the gate is DENY BY DEFAULT: every tool and action that is not listed below is refused,
including every write-class tool and action, unknown tools, unknown actions and a missing action on a tool that has
actions. Enabling any write tool later needs its own reviewed, per-tool destination policy (where each argument and
each generated subdirectory really lands, as BK computes it); none is implemented here.

Classes of what is allowed:
  read     index, wiki and file reads; forwarded without the handshake (BK reads, never writes)
  wb_read  Workbench reads; forwarded only after the endpoint handshake passes

Read tools that take a path pass one small check (read_path_reason): no absolute, UNC, drive-relative or '..' path
outside the allowed read roots (the pin's project root and the pin's game install), compared case-insensitively on
/mnt/<drive> (drvfs is case-insensitive).

The table is BK at the pinned commit (README "Pinned versions"); a BK update must re-check it.

Sealed map (Everon): writes are denied, and every call's arguments are also checked (sealed_reason): an identifier
token eden or everon (non-alphanumerics and camelCase split) or the Everon root world GUID 853E92315D1D9EFE
({853E92315D1D9EFE}worlds/Eden/Eden.ent, docs/capture_contract/dc-2.0.md 3.1) refuses. Not detected: a read on a
sealed world already open in Workbench without naming it; the runbook rule "never open Eden/Everon in an MCP-attached
Workbench" covers that.
"""
from __future__ import annotations

import os
import re

BK_COMMIT = "0acfa884228477043c6b4c2b8c7f0c270d648398"

# tool -> None (the tool has no action argument) or the set of allowed actions (any other or a missing one: denied)
READ = {
    "api_search": None, "wiki_search": None, "wiki_read": None, "asset_search": None, "component_search": None,
    "game_browse": None, "game_read": None, "workshop_info": None, "wb_knowledge": None,
    "project": {"browse", "read"},
}
WB_READ = {
    "wb_connect": None, "wb_diagnose": None, "wb_state": None, "wb_entity_list": None, "wb_entity_inspect": None,
    "wb_read_props": None,
    "wb_validate": {"material", "texture"},
    "wb_terrain": {"getHeight", "getBounds"},
    "wb_resources": {"getInfo"},
    "wb_projects": {"list", "locate"},
    "wb_layers": {"list", "isVisible", "getInfo"},
    "wb_prefabs": {"getGuid", "locate", "getAncestor"},
    "wb_localization": {"getTable", "listLanguages"},
    "wb_script_editor": {"getCurrentFile", "getLine", "getLinesCount"},
    "wb_entity_select": {"getSelected"},
    "wb_entity_modify": {"getProperty", "listProperties", "listArrayItems", "getWorldTransform"},
    "wb_component": {"list"},
}
# Every other BK tool at the pinned commit, for the record and the tests; all denied (the gate denies any unlisted
# name anyway). Actions of the tools above that are not allowed are denied too.
DENIED = {
    "script_create": "writes files", "prefab": "create writes files", "config_create": "writes files",
    "layout_create": "writes files", "layout_recipe": "create writes files", "menu_create": "create writes files",
    "gamemode_scaffold": "create writes files", "mod": "create and build write; validate is not needed",
    "scenario_create_conflict": "writes files", "server_config": "writes server.json",
    "building_setup": "writes files", "game_duplicate": "writes files", "animation_graph": "author/setup write files",
    "wb_launch": "launches Workbench (README Rules)", "wb_cleanup": "deletes handler files (README Rules)",
    "mod_build": "spawns a second Workbench (an old npm name)", "wb_play": "the user presses Play",
    "wb_stop": "the user stops Play", "wb_save": "saves the world", "wb_undo_redo": "changes the world",
    "wb_open_resource": "opens a resource", "wb_entity_create": "changes the world",
    "wb_entity_delete": "changes the world", "wb_entity_duplicate": "writes files and changes the world",
    "wb_execute_action": "runs any editor action", "wb_reload": "reloads scripts", "wb_clipboard": "changes the world",
    "scenario_create": "changes the world",
}
READ_PATH_KEYS = ("projectPath", "path")
SEALED_TOKENS = {"eden", "everon"}
SEALED_GUIDS = ("853E92315D1D9EFE",)


def known_tools() -> set:
    return set(READ) | set(WB_READ) | set(DENIED)


def classify(name: str, arguments) -> tuple:
    """(class, reason): read or wb_read (allowed, reason None), or deny with the reason."""
    args = arguments if isinstance(arguments, dict) else {}
    for table, kind in ((READ, "read"), (WB_READ, "wb_read")):
        if name in table:
            allowed = table[name]
            if allowed is None:
                return kind, None
            action = args.get("action")
            if action is None:
                return "deny", "%s needs an action; a missing action is denied" % name
            if action not in allowed:
                return "deny", "%s action %r is not on the read allow-list (%s); every write is denied" % (
                    name, action, ", ".join(sorted(allowed)))
            return kind, None
    if name in DENIED:
        return "deny", "%s is a write-class tool (%s); the gate denies every write" % (name, DENIED[name])
    return "deny", "tool %r is not on the gate's allow-list (BK %s)" % (name, BK_COMMIT[:7])


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


# --------------------------------------------------------------------------- the read-path check


_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_DRIVE_REL = re.compile(r"^[A-Za-z]:(?![\\/])")


def fold(path: str) -> str:
    """Case-folded on /mnt/<drive> (drvfs is case-insensitive), unchanged elsewhere."""
    return path.lower() if re.match(r"^/mnt/[A-Za-z](/|$)", path) else path


def inside(path: str, root: str) -> bool:
    p, r = fold(os.path.normpath(path)), fold(os.path.normpath(root))
    return p == r or p.startswith(r.rstrip("/") + "/")


def read_path_reason(name: str, arguments, roots, win_to_wsl):
    """Why a read tool's path argument could read outside the allowed roots, or None. Relative resource paths
    ('Prefabs/x.et', '{GUID}x.et') are BK's to resolve inside the project or the game data; '..' is refused in any
    form, and so are UNC and drive-relative paths. An absolute path (POSIX or X:\\) must lie inside a root; a relative
    projectPath is refused (BK would resolve it against its own cwd)."""
    args = arguments if isinstance(arguments, dict) else {}
    for key in READ_PATH_KEYS:
        v = args.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            return "%s: %s %r is not a path" % (name, key, v)
        s = v.strip()
        if not s:
            continue
        if s.startswith("\\\\") or s.startswith("//") or _DRIVE_REL.match(s):
            return "%s: %s %r is a UNC or drive-relative path" % (name, key, v)
        if ".." in re.split(r"[\\/]+", s):
            return "%s: %s %r contains '..'" % (name, key, v)
        if s.startswith("/") or _WIN_ABS.match(s):
            p = win_to_wsl(s.replace("/", "\\")) if _WIN_ABS.match(s) else s
            real = os.path.realpath(p)
            if not any(inside(real, os.path.realpath(r)) for r in roots if r):
                return "%s: %s %s resolves to %s, outside the allowed read roots %s" % (
                    name, key, v, real, ", ".join(r for r in roots if r))
        elif key == "projectPath":
            return "%s: a relative projectPath %r is refused (BK would resolve it against its cwd)" % (name, v)
    return None
