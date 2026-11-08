"""What the MCP gate lets through to BK enfusion-mcp: an explicit allow-list (ticket DD-14a, CC-05; Codex rounds 1-2).

Coordinator decision (round 2): the runbook never uses MCP mutations. Agents never call MCP tools to change files or
Workbench, and the user presses Play. CC-05 needs only the per-pin API index, api_search and a handshake that refuses a
mismatched Workbench. So the gate is DENY BY DEFAULT: every tool and action that is not listed below is refused,
including every write-class tool and action, unknown tools, unknown actions and a missing action on a tool that has
actions. Enabling any write tool later needs its own reviewed, per-tool destination policy (where each argument and
each generated subdirectory really lands, as BK computes it); none is implemented here.

Classes of what is allowed:
  read     index, wiki and knowledge-base reads; forwarded without the handshake
  wb_read  Workbench getters; forwarded only after the endpoint handshake passes

Coordinator decision (Codex round 3): no allowed entry takes a filesystem, resource or mod path. project, workshop_info,
wb_read_props, asset_search, game_browse and game_read (the game install holds the sealed Everon world),
wb_resources, wb_prefabs and wb_projects are denied, and so is wb_validate (material validation can change shared
Workbench state). Each allowed tool and action also has an explicit set of argument keys; any other key is denied.
Re-enabling any path-taking tool needs a reviewed per-field read policy first.

The table is BK at the pinned commit (README "Pinned versions"); a BK update must re-check it.

Sealed map (Everon): writes are denied, and every call's arguments are also checked (sealed_reason): an identifier
token eden or everon (non-alphanumerics and camelCase split) or the Everon root world GUID 853E92315D1D9EFE
({853E92315D1D9EFE}worlds/Eden/Eden.ent, docs/capture_contract/dc-2.0.md 3.1) refuses. Not detected: a read on a
sealed world already open in Workbench without naming it; the runbook rule "never open Eden/Everon in an MCP-attached
Workbench" covers that.
"""
from __future__ import annotations

import re

BK_COMMIT = "0acfa884228477043c6b4c2b8c7f0c270d648398"

# tool -> (class, {action or None: allowed argument keys}). None as the action: the tool has no action argument. Any
# other tool, action or argument key is denied. No allowed entry takes a filesystem, resource or mod path (Codex
# round 3: re-enabling any path-taking tool needs a reviewed per-field read policy first).
_GETTER_KEYS = {"action", "name", "propertyPath", "propertyKey", "memberIndex"}
ALLOW = {
    "api_search": ("read", {None: {"query", "type", "source", "limit", "format"}}),
    "component_search": ("read", {None: {"query", "category", "event", "source", "limit"}}),
    "wiki_search": ("read", {None: {"query", "limit"}}),
    "wiki_read": ("read", {None: {"title"}}),
    "wb_knowledge": ("read", {None: {"query", "max_files"}}),
    "wb_connect": ("wb_read", {None: set()}),
    "wb_state": ("wb_read", {None: set()}),
    "wb_entity_list": ("wb_read", {None: {"offset", "limit", "nameFilter"}}),
    "wb_entity_inspect": ("wb_read", {None: {"name", "index"}}),
    "wb_entity_modify": ("wb_read", {a: _GETTER_KEYS for a in ("getProperty", "listProperties", "listArrayItems",
                                                                 "getWorldTransform")}),
    "wb_entity_select": ("wb_read", {"getSelected": {"action"}}),
    "wb_script_editor": ("wb_read", {"getCurrentFile": {"action"}, "getLine": {"action", "line"},
                                     "getLinesCount": {"action"}}),
    "wb_layers": ("wb_read", {"list": {"action", "subScene"}, "isVisible": {"action", "subScene", "layerPath"},
                              "getInfo": {"action", "subScene", "layerPath"}}),
    "wb_localization": ("wb_read", {"getTable": {"action"}, "listLanguages": {"action"}}),
    "wb_component": ("wb_read", {"list": {"action", "entityName"}}),
    "wb_terrain": ("wb_read", {"getHeight": {"action", "x", "z"}, "getBounds": {"action"}}),
}
READ = {k: v[1] for k, v in ALLOW.items() if v[0] == "read"}
WB_READ = {k: v[1] for k, v in ALLOW.items() if v[0] == "wb_read"}
# Every other BK tool at the pinned commit, for the record and the tests; all denied (any unlisted name is denied
# anyway), with the reason.
_PATH = "takes a filesystem, resource or mod path (no reviewed per-field read policy)"
DENIED = {
    "project": _PATH, "workshop_info": _PATH, "wb_read_props": _PATH, "asset_search": _PATH + "; the game install "
    "holds the sealed Everon world", "game_browse": _PATH + "; the game install holds the sealed Everon world",
    "game_read": _PATH + "; the game install holds the sealed Everon world",
    "wb_diagnose": "enumerates each addon's Scripts/WorkbenchGame/EnfusionMCP directory without resolving symlinks "
                   "(BK client.ts:288), an indirect filesystem read (Codex round 4)",
    "wb_validate": "material validation can change shared Workbench state (Codex round 3), texture validation's "
                   "contract is unverified",
    "wb_resources": _PATH, "wb_prefabs": _PATH, "wb_projects": "locate takes a resource name; open switches projects",
    "script_create": "writes files", "prefab": "writes files", "config_create": "writes files",
    "layout_create": "writes files", "layout_recipe": "writes files", "menu_create": "writes files",
    "gamemode_scaffold": "writes files", "mod": "create and build write", "scenario_create_conflict": "writes files",
    "server_config": "writes server.json", "building_setup": "writes files", "game_duplicate": "writes files",
    "animation_graph": "writes files", "wb_launch": "launches Workbench (README Rules)",
    "wb_cleanup": "deletes handler files (README Rules)", "mod_build": "spawns a second Workbench (an old npm name)",
    "wb_play": "the user presses Play", "wb_stop": "the user stops Play", "wb_save": "saves the world",
    "wb_undo_redo": "changes the world", "wb_open_resource": "opens a resource", "wb_entity_create": "changes the world",
    "wb_entity_delete": "changes the world", "wb_entity_duplicate": "writes files and changes the world",
    "wb_execute_action": "runs any editor action", "wb_reload": "reloads scripts", "wb_clipboard": "changes the world",
    "scenario_create": "changes the world",
}
SEALED_TOKENS = {"eden", "everon"}
SEALED_GUIDS = ("853E92315D1D9EFE",)


def known_tools() -> set:
    return set(ALLOW) | set(DENIED)


def classify(name: str, arguments) -> tuple:
    """(class, reason): read or wb_read (allowed, reason None), or deny with the reason."""
    if name not in ALLOW:
        if name in DENIED:
            return "deny", "%s is not allowed: %s" % (name, DENIED[name])
        return "deny", "tool %r is not on the gate's allow-list (BK %s)" % (name, BK_COMMIT[:7])
    if arguments is not None and not isinstance(arguments, dict):
        return "deny", "%s arguments are not an object" % name
    args = arguments or {}
    kind, actions = ALLOW[name]
    if None in actions:
        allowed_keys = actions[None]
    else:
        action = args.get("action")
        if action is None:
            return "deny", "%s needs an action; a missing action is denied" % name
        if action not in actions:
            return "deny", "%s action %r is not on the read allow-list (%s); every write is denied" % (
                name, action, ", ".join(sorted(actions)))
        allowed_keys = actions[action]
    extra = sorted(set(args) - set(allowed_keys))
    if extra:
        return "deny", "%s does not take %s here (allowed arguments: %s)" % (
            name, ", ".join(extra), ", ".join(sorted(allowed_keys)) or "none")
    return kind, None


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
