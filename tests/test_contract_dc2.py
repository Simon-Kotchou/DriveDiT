#!/usr/bin/env python3
"""Schema tests for contract dc-2.0 (DD-11-min), stdlib unittest only.

    python3 -I -B tests/test_contract_dc2.py -v
    python3 -I -B -m pytest -p no:cacheprovider tests/test_contract_dc2.py

Checks, without touching any session data:
  * docs/capture_contract/dc-2.0.streams.json is structurally valid (keys, types, units, enums, TBD references);
  * every stream field and tensor channel that CAPTURE_PROGRAM r2 3.3 lists, and every addition MIGRATION_1_9 8.3
    asks of DD-11-min, appears in the JSON files;
  * the features_v1 registry rules: no student_input channel is absolute, the policy sees action_cmd and action_exec
    only as history (offset <= -1), aux_target and eval_only channels are never inputs, C24's forbidden list;
  * the field tables in docs/capture_contract/dc-2.0.md equal the JSON, row by row, including every "Values:" list;
  * the crash-event payload table of section 9.5.1 equals the JSON event_payloads, and the crash events, the
    ego_destroyed end reason, the session.json events.* parameters (TBD-10) and dv_max_mps are where the contract
    puts them;
  * the probe-dependent values are TBD entries that cite docs/trials/T0-ENG.md and carry the value decided at the
    freeze with its status and evidence, and the spec's TBD and tensor tables equal the JSON in every column;
  * the freeze decisions: frame-bound ctl and telemetry rows (D-A: covered_steps, render_frame, no telemetry_every_k,
    the measured step dt), 5 Hz images and decisions with the 10 Hz trial-conf override (D-B), the endpoint road
    join (D-C) and the folded facts (steer_sign +1, the S105 gear map, the Shape fiducial carrier at 0.05 m,
    img.done_render_frame, the .png name);
  * the timing rules: the decision boundary on a synthetic frame-bound timeline whose command changes, the exact hf
    joins, the decision-rate rule and the shared world-time anchor of the decision and image grids;
  * the session id rule and the canonical map ids;
  * the committed contract files are ASCII with LF line ends.
The checks are plain functions that return a list of violations, so the tests can also show that a broken copy of a
file is caught.
"""

import copy
import fnmatch
import json
import math
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONTRACT_DIR = os.path.join(REPO, "docs", "capture_contract")
SPEC_MD = os.path.join(CONTRACT_DIR, "dc-2.0.md")
STREAMS_JSON = os.path.join(CONTRACT_DIR, "dc-2.0.streams.json")
FEATURES_JSON = os.path.join(CONTRACT_DIR, "features_v1.json")

STATUS_LINE = ("Status: FROZEN 2026-11-04 (freeze decisions D-A to D-E adopted 2026-11-03, review rounds 1 and 2 "
               "corrections folded in on 2026-11-04 before any capture, changelog in section "
               "16); a change needs a new version dc-2.x; user decision D15 adopted 2026-10-22")
T0ENG = "docs/trials/T0-ENG.md"

FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FILE_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
TYPES = ("int", "float", "bool", "str", "enum")
FILE_TYPES = TYPES + ("object", "array")
STREAM_STATUS = ("required", "reserved")
CANONICAL_MAP_IDS = ["arland", "kolguyev", "everon"]
TERRAINS = ["Arland", "Cain", "Eden"]

# The probe-dependent values the ticket names, with the probe that decides each (TICKETS_UNIFIED DD-11-min).
BLOCKING_TBD = {"TBD-1": "E-10", "TBD-2": "E-05", "TBD-3": "E-19", "TBD-4": "E-05", "TBD-5": "E-01"}
TBD_STATUS = ("decided", "provisional")
# The only TBD still open after the freeze (a value, never a field)
PROVISIONAL_TBD = ("TBD-10",)

# CAPTURE_PROGRAM r2 3.3, stream table, item by item -> where dc-2.0 carries it ("<group>.<field>", or
# "<group>.<field>=<enum value>" for an event kind). Groups are streams, derived groups or session files.
R2_STREAM_ITEMS = {
    "ctl: physics_step (own counter)": ["ctl.physics_step"],
    "ctl: world_time_ms": ["ctl.world_time_ms"],
    "ctl: hl_seq": ["ctl.hl_seq"],
    "ctl: held hl_{kappa,accel,dir}": ["ctl.hl_kappa", "ctl.hl_accel", "ctl.hl_dir"],
    "ctl: exp_{kappa,accel}_now": ["ctl.exp_kappa_now", "ctl.exp_accel_now"],
    "ctl: llc_{steer_raw,steer_lim,throttle,brake,gear}": [
        "ctl.llc_steer_raw", "ctl.llc_steer_lim", "ctl.llc_throttle", "ctl.llc_brake", "ctl.llc_gear"],
    "ctl: input_pre_*": ["ctl.input_pre_steer", "ctl.input_pre_throttle", "ctl.input_pre_brake"],
    "ctl: input_post_*": ["ctl.input_post_steer", "ctl.input_post_throttle", "ctl.input_post_brake"],
    "ctl: noise_active": ["ctl.noise_active"],
    "ctl: noise_id": ["ctl.noise_id"],
    "ctl: mode": ["ctl.mode"],
    "telemetry: physics_step key": ["telemetry.physics_step"],
    "telemetry: pose": ["telemetry.pos_x", "telemetry.pos_y", "telemetry.pos_z", "telemetry.right_x",
                        "telemetry.up_y", "telemetry.fwd_z"],
    "telemetry: local velocity": ["telemetry.vel_local_x", "telemetry.vel_local_y", "telemetry.vel_local_z"],
    "telemetry: angular velocity": ["telemetry.angvel_local_x", "telemetry.angvel_local_y",
                                    "telemetry.angvel_local_z"],
    "telemetry: acceleration": ["telemetry.acc_local_x", "telemetry.acc_local_y", "telemetry.acc_local_z"],
    "telemetry: speed": ["telemetry.speed_engine_mps"],
    "telemetry: gear": ["telemetry.gear"],
    "telemetry: RPM": ["telemetry.rpm"],
    "telemetry: per-wheel slip": ["telemetry.w%d_slip_long" % i for i in range(4)] +
                                 ["telemetry.w%d_slip_lat" % i for i in range(4)],
    "telemetry: per-wheel contact": ["telemetry.w%d_contact" % i for i in range(4)],
    "telemetry: per-wheel material": ["telemetry.w%d_material" % i for i in range(4)],
    "telemetry: road context, matched road id": ["telemetry.road_key", "road.matched_road_id"],
    "telemetry: road context, class": ["road.road_class"],
    "telemetry: road context, width": ["telemetry.road_width_m"],
    "telemetry: lane-relative signed lateral error": ["telemetry.lane_lat_err_m"],
    "telemetry: lane-relative heading error": ["telemetry.lane_head_err_rad"],
    "telemetry: weather": ["telemetry.env_weather_state", "telemetry.env_rain", "telemetry.env_fog",
                           "telemetry.env_wetness"],
    "telemetry: time": ["telemetry.env_tod_h"],
    "telemetry: sun elevation": ["telemetry.sun_elev_deg"],
    "telemetry: traffic slots as (slot_id, entity_id)": ["telemetry.traffic_slots", "telemetry.ego_slot_id",
                                                         "telemetry.ego_entity_id"],
    "hf: render_frame key": ["hf.render_frame"],
    "hf: world_time_ms": ["hf.world_time_ms"],
    "hf: last_physics_step": ["hf.last_physics_step"],
    "hf: paused": ["hf.paused"],
    "hf: camera world transform": ["hf.cam_pos_x", "hf.cam_pos_y", "hf.cam_pos_z", "hf.cam_right_x",
                                   "hf.cam_up_y", "hf.cam_fwd_z"],
    "hf: vehicle pose": ["hf.veh_pos_x", "hf.veh_pos_y", "hf.veh_pos_z", "hf.veh_fwd_z"],
    "hf: fps": ["hf.fps"],
    "img: slot_id key": ["img.slot_id"],
    "img: requested_world_ms": ["img.requested_world_ms"],
    "img: issue_world_ms": ["img.issue_world_ms"],
    "img: issue_render_frame": ["img.issue_render_frame"],
    "img: path": ["img.path"],
    "img: completion (harvester, fiducial)": ["img_resolved.file_exists", "img_resolved.fid_ok"],
    "img: captured_render_frame (harvester, fiducial)": ["img_resolved.captured_render_frame"],
    "events/episodes: episode_id key": ["episodes.episode_id", "events.episode_id"],
    "events/episodes: termination reasons": ["episodes.termination_reason"],
    "events/episodes: perturbations": ["events.kind=perturb_start", "events.kind=perturb_end"],
    "events/episodes: nav commands": ["events.kind=nav_command"],
    "events/episodes: style_v1": ["episodes.style_speed_factor", "episodes.style_lane_offset_m",
                                  "episodes.style_alat_max_mps2"],
    "events/episodes: expert_sha": ["episodes.expert_sha"],
    "events/episodes: llc_sha": ["episodes.llc_sha"],
    "r2 3.9 version stamps": ["session.json.control.expert_sha", "session.json.control.llc_sha",
                              "session.json.versions.nav_version", "session.json.versions.features_version",
                              "session.json.rig.render_settings_hash", "session.json.pin.pin_id",
                              "session.json.pin.launch_id"],
}

# MIGRATION_1_9 8.3, row DD-11-min.
MIGRATION_ITEMS = {
    "session.json engine": ["session.json.engine.build_version", "session.json.engine.build_time",
                            "session.json.engine.fixed_time_slice_s"],
    "session.json pin": ["session.json.pin.pin_id", "session.json.pin.pin_sha256", "session.json.pin.build_key",
                         "session.json.pin.channel", "session.json.pin.launch_id", "session.json.pin.game_buildid",
                         "session.json.pin.tools_buildid"],
    "session.json use_class": ["session.json.use_class"],
    "session.json addon": ["session.json.addon.git_sha", "session.json.addon.git_dirty",
                           "session.json.addon.content_sha256", "session.json.addon.target_pin"],
    "session.json control": ["session.json.control.llc_sha", "session.json.control.expert_sha",
                             "session.json.control.vehicle_card_id", "session.json.control.vehicle_card_sha",
                             "session.json.control.actuation_mode"],
    "session.json authority": ["session.json.authority.standalone", "session.json.authority.rpl_ok"],
    "session_end.json repeats": ["session_end.json.pin.pin_id", "session_end.json.pin.build_key",
                                 "session_end.json.engine.build_version"],
    "steering channels": ["ctl_export.steer_cmd", "ctl_export.steer_input", "ctl.wheel_angle_l_rad",
                          "ctl.wheel_angle_r_rad", "telemetry_export.delta_eff_rad",
                          "telemetry_export.kappa_realised"],
    "reserved, null on 1.8": ["events.kind=veh_cfg_override", "episodes.veh_dirt_body", "episodes.veh_dirt_glass",
                              "telemetry.w0_temp_c", "telemetry.w1_temp_c", "telemetry.w2_temp_c",
                              "telemetry.w3_temp_c"],
}

# Fields every session file must carry with a non-null value (MIGRATION_1_9 8.3 lists them as required).
MIGRATION_NULLABLE_OK = {"session.json.pin.build_key", "session_end.json.pin.build_key",
                         "session.json.addon.target_pin", "session.json.control.llc_sha",
                         "session.json.control.expert_sha", "session.json.control.vehicle_card_id",
                         "session.json.control.vehicle_card_sha"}


def load_json(path):
    with open(path, encoding="ascii") as f:
        return json.load(f)


def read_text(path):
    with open(path, "rb") as f:
        return f.read().decode("ascii")


def field_groups(schema):
    """{group name: [field dicts]} for streams, derived groups and session files."""
    groups = {}
    for name, s in schema["streams"].items():
        groups[name] = s["fields"]
    for name, d in schema["derived"].items():
        groups[name] = d["fields"]
    for name, d in schema["files"].items():
        groups[name] = d["fields"]
    return groups


def resolve_ref(schema, ref):
    """Return the field dict a '<group>.<field>' or '<group>.<field>=<value>' reference names, or None."""
    groups = field_groups(schema)
    value = None
    if "=" in ref:
        ref, value = ref.split("=", 1)
    for group in sorted(groups, key=len, reverse=True):  # 'session_end.json' before 'session.json'-like prefixes
        if ref.startswith(group + "."):
            name = ref[len(group) + 1:]
            for f in groups[group]:
                if f["name"] == name:
                    if value is not None and value not in [str(v) for v in f.get("enum", [])]:
                        return None
                    return f
            return None
    return None


def check_field(f, where, allowed_types, name_re, units, tbd_ids, errors):
    for k in ("name", "type", "unit", "required", "desc"):
        if k not in f:
            errors.append("%s: field without %r: %r" % (where, k, f))
            return
    if not name_re.match(f["name"]):
        errors.append("%s.%s: bad field name" % (where, f["name"]))
    if f["type"] not in allowed_types:
        errors.append("%s.%s: unknown type %r" % (where, f["name"], f["type"]))
    if f["unit"] not in units:
        errors.append("%s.%s: unit %r not declared in 'units'" % (where, f["name"], f["unit"]))
    if not isinstance(f["required"], bool):
        errors.append("%s.%s: required is not a bool" % (where, f["name"]))
    if f["type"] == "enum" and not f.get("enum"):
        errors.append("%s.%s: enum without values" % (where, f["name"]))
    if "enum" in f and len(set(map(str, f["enum"]))) != len(f["enum"]):
        errors.append("%s.%s: duplicate enum values" % (where, f["name"]))
    if "range" in f and not (len(f["range"]) == 2 and f["range"][0] < f["range"][1]):
        errors.append("%s.%s: bad range %r" % (where, f["name"], f["range"]))
    if f.get("tbd") and f["tbd"] not in tbd_ids:
        errors.append("%s.%s: unknown TBD %r" % (where, f["name"], f["tbd"]))
    if f.get("reserved") and f["required"]:
        errors.append("%s.%s: a reserved field cannot be required (it is empty until filled)" % (where, f["name"]))
    if f.get("nullable") and not f["required"]:
        errors.append("%s.%s: nullable only makes sense on a required key" % (where, f["name"]))
    if not f["desc"].strip() or "|" in f["desc"]:
        errors.append("%s.%s: empty description or '|' in it" % (where, f["name"]))


def stream_schema_violations(schema):
    """Structural rules of dc-2.0.streams.json. Returns a list of human-readable violations."""
    errors = []
    for k in ("schema", "contract_version", "status", "tbd_by_probe", "maps", "session", "csv", "writer", "enums",
              "types", "units", "streams", "derived", "files"):
        if k not in schema:
            errors.append("top level: missing %r" % k)
    if errors:
        return errors
    if schema["schema"] != "drivedit.dc-streams/1" or schema["contract_version"] != "dc-2.0":
        errors.append("top level: wrong schema id or contract version")
    units = schema["units"]
    tbd_ids = [t.get("id") for t in schema["tbd_by_probe"]]
    if len(set(tbd_ids)) != len(tbd_ids):
        errors.append("tbd_by_probe: duplicate ids")
    for t in schema["tbd_by_probe"]:
        for k in ("id", "item", "default_r2", "default", "decided_by", "blocks_freeze", "results", "status",
                  "decided", "decided_text", "evidence", "affects"):
            if k not in t:
                errors.append("tbd_by_probe %s: missing %r" % (t.get("id"), k))
        if t.get("status") not in TBD_STATUS:
            errors.append("tbd_by_probe %s: status %r" % (t.get("id"), t.get("status")))
        elif (t["status"] == "provisional") != (t.get("id") in PROVISIONAL_TBD):
            errors.append("tbd_by_probe %s: only %s may stay provisional" % (t.get("id"), ", ".join(PROVISIONAL_TBD)))
        if not str(t.get("evidence", "")).strip() or not str(t.get("decided_text", "")).strip():
            errors.append("tbd_by_probe %s: empty decided_text or evidence" % t.get("id"))
        if t.get("results") != T0ENG:
            errors.append("tbd_by_probe %s: results must cite %s" % (t.get("id"), T0ENG))
    for tid, probe in BLOCKING_TBD.items():
        match = [t for t in schema["tbd_by_probe"] if t.get("id") == tid]
        if not match or not match[0].get("blocks_freeze") or probe not in match[0].get("decided_by", ""):
            errors.append("tbd_by_probe: %s must block the freeze and be decided by %s" % (tid, probe))
    for name, s in schema["streams"].items():
        where = "stream " + name
        for k in ("status", "written_in", "cadence", "key", "key_step", "dir", "chunk_max_rows", "fields"):
            if k not in s:
                errors.append("%s: missing %r" % (where, k))
        if any(k not in s for k in ("status", "key", "dir", "chunk_max_rows", "fields")):
            continue
        if s["status"] not in STREAM_STATUS:
            errors.append("%s: bad status %r" % (where, s["status"]))
        if s["dir"] != name:
            errors.append("%s: dir must equal the stream name" % where)
        if not isinstance(s["chunk_max_rows"], int) or s["chunk_max_rows"] < 1:
            errors.append("%s: chunk_max_rows must be a positive int" % where)
        names = [f.get("name") for f in s["fields"]]
        if len(set(names)) != len(names):
            errors.append("%s: duplicate field names" % where)
        for f in s["fields"]:
            check_field(f, where, TYPES, FIELD_NAME_RE, units, tbd_ids, errors)
        if not s["key"]:
            errors.append("%s: empty key" % where)
        if names[:len(s["key"])] != s["key"]:
            errors.append("%s: key columns must come first, in key order" % where)
        for kname in s["key"]:
            kf = [f for f in s["fields"] if f.get("name") == kname]
            if not kf:
                errors.append("%s: key %r is not a field" % (where, kname))
                continue
            if kf[0].get("type") != "int" or not kf[0].get("required") or not kf[0].get("key"):
                errors.append("%s: key %r must be a required int flagged key" % (where, kname))
        flagged = [f["name"] for f in s["fields"] if f.get("key")]
        if flagged != s["key"]:
            errors.append("%s: fields flagged key %r differ from key %r" % (where, flagged, s["key"]))
    for name, d in schema["derived"].items():
        where = "derived " + name
        names = [f.get("name") for f in d.get("fields", [])]
        if len(set(names)) != len(names):
            errors.append("%s: duplicate field names" % where)
        if names[:len(d.get("key", []))] != d.get("key", []):
            errors.append("%s: key columns must come first" % where)
        for f in d.get("fields", []):
            check_field(f, where, TYPES, FIELD_NAME_RE, units, tbd_ids, errors)
    for name, d in schema["files"].items():
        where = "file " + name
        names = [f.get("name") for f in d.get("fields", [])]
        if len(set(names)) != len(names):
            errors.append("%s: duplicate field names" % where)
        csv_file = d.get("format") == "csv"
        for f in d.get("fields", []):
            check_field(f, where, TYPES if csv_file else FILE_TYPES, FIELD_NAME_RE if csv_file else FILE_FIELD_NAME_RE,
                        units, tbd_ids, errors)
    for gname, fields in field_groups(schema).items():
        for f in fields:
            values = schema["enums"].get(f.get("name", "").split(".")[-1])
            if f.get("type") == "enum" and values is not None and not set(map(str, f["enum"])) <= set(map(str, values)):
                errors.append("%s.%s: enum values outside enums.%s" % (gname, f["name"], f["name"].split(".")[-1]))
    if schema["enums"].get("map_id") != CANONICAL_MAP_IDS:
        errors.append("enums.map_id must be exactly %r" % CANONICAL_MAP_IDS)
    if schema["enums"].get("terrain") != TERRAINS:
        errors.append("enums.terrain must be exactly %r" % TERRAINS)
    return errors


def md_tables(text):
    """{(kind, name): [row cells]} for every table that follows a '<!-- dc2:... -->' marker in the spec."""
    tables = {}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^<!-- dc2:(fields (stream|derived|file)=(\S+)|tbd|tensors|payloads|changelog) -->$",
                     line.strip())
        if not m:
            continue
        key = (m.group(2), m.group(3)) if m.group(2) else (m.group(1), "")
        rows = []
        j = i + 1
        while j < len(lines) and lines[j].startswith("|"):
            cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
            rows.append(cells)
            j += 1
        tables[key] = rows[2:]  # drop the header and the separator
    return tables


def md_agreement_violations(schema, text):
    errors = []
    tables = md_tables(text)
    groups = (("stream", schema["streams"]), ("derived", schema["derived"]), ("file", schema["files"]))
    for kind, entries in groups:
        for name, entry in entries.items():
            rows = tables.get((kind, name))
            if rows is None:
                errors.append("spec: no field table for %s %s" % (kind, name))
                continue
            md_names = [r[0].strip("`") for r in rows]
            js_names = [f["name"] for f in entry["fields"]]
            if md_names != js_names:
                errors.append("spec: %s %s fields differ from the JSON: md %r json %r" % (kind, name, md_names,
                                                                                        js_names))
                continue
            for r, f in zip(rows, entry["fields"]):
                want_type = f["type"] + (" (key)" if f.get("key") else "")
                want_req = "yes" if f["required"] else "no"
                if f["required"] and f.get("nullable"):
                    want_req = "yes, null ok"
                if (r[1], r[2], r[3]) != (want_type, f["unit"], want_req):
                    errors.append("spec: %s %s.%s type/unit/req differ from the JSON" % (kind, name, f["name"]))
                if r[4] != desc_cell(f):
                    errors.append("spec: %s %s.%s description, Range, Values or tags differ from the JSON"
                                  % (kind, name, f["name"]))
    known = {(k, n) for k, entries in groups for n in entries}
    for key in tables:
        if key[0] in ("stream", "derived", "file") and key not in known:
            errors.append("spec: field table %r has no JSON counterpart" % (key,))
    tbd_rows = tables.get(("tbd", ""))
    if tbd_rows is None or [r[0] for r in tbd_rows] != [t["id"] for t in schema["tbd_by_probe"]]:
        errors.append("spec: the TBD table differs from tbd_by_probe")
    else:
        for r, t in zip(tbd_rows, schema["tbd_by_probe"]):
            want = [t["id"], t["item"], t["default_r2"], t["decided_text"], t["evidence"], t["decided_by"],
                    "yes" if t["blocks_freeze"] else "no"]
            if r != want:
                errors.append("spec: TBD row %s differs from tbd_by_probe (item, default, probe or blocking)" % t["id"])
    return errors


def field_tags(f):
    """The bracketed tags of a field row, in the spec's order: sign, frame, eval-only, build-scoped, reserved, tbd."""
    tags = [f[k] for k in ("sign", "frame") if f.get(k)]
    tags += [name for key, name in (("eval_only", "eval-only"), ("build_scoped", "build-scoped"),
                                    ("reserved", "reserved")) if f.get(key)]
    return tags + ([f["tbd"]] if f.get("tbd") else [])


def desc_cell(f):
    """The spec's description cell of a JSON field: desc, then Range, Values and the tags."""
    text = f["desc"]
    if "range" in f:
        text += " Range [%s, %s]." % (f["range"][0], f["range"][1])
    if "enum" in f:
        text += " Values: " + ", ".join(str(v) for v in f["enum"]) + "."
    tags = field_tags(f)
    return text + (" [" + ", ".join(tags) + "]" if tags else "")


VALUES_RE = re.compile(r"Values: (.*?)\.(?: \[[^\]]*\])?$")
PAYLOAD_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Crash events of section 9.5.1 (t0d crash-events design, coordinator decisions of 2026-11-03)
CRASH_KINDS = ["veh_damage", "veh_fire_start", "veh_fire_end", "driver_state", "dismount", "remount", "rollover"]
PAYLOAD_KINDS = ["collision"] + CRASH_KINDS + ["stuck", "teleport"]
SESSION_EVENT_PARAMS = {"events.collision_dv_mps": 4.0, "events.collision_window_ms": 100,
                        "events.collision_window_tol_ms": 50, "events.collision_peak_ms": 200,
                        "events.collision_refractory_ms": 1000, "events.episode_end_wait_ms": 250,
                        "events.crash_cooldown_ms": 3000, "events.episode_block_max_ms": 60000,
                        "events.rollover_up_y": 0.5,
                        "events.rollover_hold_ms": 500, "events.state_poll_ms": 100}


def md_values_violations(schema, text):
    """Every enum field's description in the spec ends its text with 'Values: a, b, c.' equal to the JSON enum, and
    no field without an enum has such a list."""
    errors = []
    tables = md_tables(text)
    for kind, entries in (("stream", schema["streams"]), ("derived", schema["derived"]), ("file", schema["files"])):
        for name, entry in entries.items():
            for r, f in zip(tables.get((kind, name), []), entry["fields"]):
                m = VALUES_RE.search(r[4])
                listed = [v.strip() for v in m.group(1).split(",")] if m else None
                if "enum" in f and listed != [str(v) for v in f["enum"]]:
                    errors.append("spec: %s %s.%s Values %r differ from the JSON enum %r" % (
                        kind, name, f["name"], listed, f["enum"]))
                if "enum" not in f and listed is not None:
                    errors.append("spec: %s %s.%s lists Values but the JSON has no enum" % (kind, name, f["name"]))
    return errors


def payload_violations(schema):
    """Structure of event_payloads: known kinds, key names, CSV types, declared units, enums, bool required."""
    errors = []
    ep = schema.get("event_payloads")
    if not isinstance(ep, dict) or not isinstance(ep.get("kinds"), dict):
        return ["event_payloads: missing or without kinds"]
    kinds = schema["enums"]["event_kind"]
    for kind, keys in ep["kinds"].items():
        if kind not in kinds:
            errors.append("event_payloads.%s: not an event kind" % kind)
        names = [k.get("key") for k in keys]
        if len(set(names)) != len(names):
            errors.append("event_payloads.%s: duplicate keys" % kind)
        for k in keys:
            for field in ("key", "type", "unit", "required", "desc"):
                if field not in k:
                    errors.append("event_payloads.%s: key without %r: %r" % (kind, field, k))
            if any(field not in k for field in ("key", "type", "unit", "required", "desc")):
                continue
            where = "event_payloads.%s.%s" % (kind, k["key"])
            if not PAYLOAD_KEY_RE.match(k["key"]):
                errors.append("%s: bad key name" % where)
            if k["type"] not in TYPES:
                errors.append("%s: unknown type %r" % (where, k["type"]))
            if k["unit"] not in schema["units"]:
                errors.append("%s: unit %r not declared" % (where, k["unit"]))
            if not isinstance(k["required"], bool):
                errors.append("%s: required is not a bool" % where)
            if k["type"] == "enum" and not k.get("enum"):
                errors.append("%s: enum without values" % where)
            if "enum" in k and k["type"] != "enum":
                errors.append("%s: values on a %s key" % (where, k["type"]))
            if ("range" in k or "min" in k) and k["type"] not in ("int", "float"):
                errors.append("%s: a range or min on a %s key" % (where, k["type"]))
            if "range" in k and not (isinstance(k["range"], list) and len(k["range"]) == 2
                                     and k["range"][0] < k["range"][1]):
                errors.append("%s: bad range %r" % (where, k["range"]))
            if "min" in k and not isinstance(k["min"], (int, float)):
                errors.append("%s: min is not a number" % where)
            if not k["desc"].strip() or "|" in k["desc"]:
                errors.append("%s: empty description or '|' in it" % where)
    return errors


def payload_row(kind, k):
    """The spec's payload-table cells (kind, key, type, unit, req) of one JSON payload key."""
    return ["`%s`" % kind, "`%s`" % k["key"], k["type"], k["unit"], "yes" if k["required"] else "no"]


def md_payload_violations(schema, text):
    """The section 9.5.1 table equals event_payloads row by row (a kind without keys is one row with key '-'), and
    every enum key lists its JSON values."""
    rows = md_tables(text).get(("payloads", ""))
    if rows is None:
        return ["spec: no payload table (<!-- dc2:payloads -->)"]
    want = []
    for kind, keys in schema["event_payloads"]["kinds"].items():
        if not keys:
            want.append((["`%s`" % kind, "-", "-", "-", "-"], None))
        for k in keys:
            want.append((payload_row(kind, k), k))
    errors = []
    if [r[:5] for r in rows] != [w for w, _ in want]:
        errors.append("spec: payload table differs from event_payloads: md %r json %r" % (
            [r[:2] for r in rows], [w[:2] for w, _ in want]))
        return errors
    for r, (_, k) in zip(rows, want):
        m = VALUES_RE.search(r[5])
        listed = [v.strip() for v in m.group(1).split(",")] if m else None
        if k is not None and "enum" in k and listed != [str(v) for v in k["enum"]]:
            errors.append("spec: payload %s Values %r differ from %r" % (r[1], listed, k["enum"]))
    return errors


def offset_text(access):
    return "<= %d" % access["max_offset"] if access["input"] else "no"


def tensor_row(t):
    """The spec's tensor-table row of a tensor: name, shape, class, WM input, policy input, targets, channels."""
    shape = "[" + ",".join(str(x) for x in t["shape"]) + "]"
    targets = ", ".join(r for r in ("world_model", "policy") if t["access"][r]["target"]) or "-"
    return ["`%s`" % t["name"], shape, t["class"], offset_text(t["access"]["world_model"]),
            offset_text(t["access"]["policy"]), targets, ", ".join("`%s`" % c["name"] for c in t["channels"])]


class StreamSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_json(STREAMS_JSON)
        cls.spec = read_text(SPEC_MD)

    def test_structure(self):
        self.assertEqual(stream_schema_violations(self.schema), [])

    def test_broken_copies_are_caught(self):
        bad = copy.deepcopy(self.schema)
        bad["streams"]["ctl"]["fields"][0]["type"] = "float"  # physics_step no longer an int key
        self.assertTrue(stream_schema_violations(bad))
        bad = copy.deepcopy(self.schema)
        bad["streams"]["telemetry"]["fields"].append(dict(bad["streams"]["telemetry"]["fields"][-1]))
        self.assertTrue(stream_schema_violations(bad))
        bad = copy.deepcopy(self.schema)
        bad["streams"]["img"]["fields"][4]["tbd"] = "TBD-99"
        self.assertTrue(stream_schema_violations(bad))
        bad = copy.deepcopy(self.schema)
        bad["enums"]["map_id"] = ["arland", "cain", "eden"]
        self.assertTrue(stream_schema_violations(bad))
        bad = copy.deepcopy(self.schema)
        bad["tbd_by_probe"][2]["results"] = "somewhere else"
        self.assertTrue(stream_schema_violations(bad))

    def test_r2_stream_items_present(self):
        missing = [(item, ref) for item, refs in R2_STREAM_ITEMS.items() for ref in refs
                   if resolve_ref(self.schema, ref) is None]
        self.assertEqual(missing, [])

    def test_r2_streams_present(self):
        for name in ("ctl", "telemetry", "hf", "img", "events", "episodes"):
            self.assertEqual(self.schema["streams"][name]["status"], "required", name)
        keys = {n: s["key"] for n, s in self.schema["streams"].items()}
        self.assertEqual(keys["ctl"], ["physics_step"])
        self.assertEqual(keys["telemetry"], ["physics_step"])
        self.assertEqual(keys["hf"], ["render_frame"])
        self.assertEqual(keys["img"], ["slot_id"])
        self.assertEqual(keys["episodes"], ["episode_id"])
        self.assertEqual(self.schema["streams"]["ctl"]["key_step"], "covered_steps")
        self.assertEqual(self.schema["streams"]["telemetry"]["key_step"], "covered_steps")

    def test_migration_items_present(self):
        missing = []
        for item, refs in MIGRATION_ITEMS.items():
            for ref in refs:
                f = resolve_ref(self.schema, ref)
                if f is None:
                    missing.append((item, ref))
                elif ref.startswith("session") and not f["required"]:
                    missing.append((item, ref + " (not required)"))
                elif ref.startswith("session") and f.get("nullable") and ref not in MIGRATION_NULLABLE_OK:
                    missing.append((item, ref + " (nullable)"))
        self.assertEqual(missing, [])

    def test_reserved_fields_are_empty_on_18(self):
        for ref in ("ctl.wheel_angle_l_rad", "ctl.wheel_angle_r_rad", "ctl.steer_signal_raw",
                    "telemetry_export.delta_eff_rad", "episodes.veh_dirt_body", "telemetry.w0_temp_c"):
            f = resolve_ref(self.schema, ref)
            self.assertTrue(f.get("reserved"), ref)
            self.assertFalse(f["required"], ref)
        self.assertEqual(self.schema["streams"]["traffic"]["status"], "reserved")

    def test_raw_inputs_are_build_scoped(self):
        for name in ("llc_steer_raw", "llc_steer_lim", "llc_throttle", "llc_brake", "input_pre_steer",
                     "input_post_steer", "input_post_throttle"):
            self.assertTrue(resolve_ref(self.schema, "ctl." + name).get("build_scoped"), name)
        self.assertTrue(resolve_ref(self.schema, "ctl_export.steer_cmd").get("build_scoped"))

    def test_sign_conventions(self):
        for ref in ("ctl.hl_kappa", "ctl.exp_kappa_now", "telemetry.lane_lat_err_m", "telemetry.lane_head_err_rad",
                    "telemetry_export.yaw_rate_rps", "ctl_export.steer_cmd", "ctl_export.steer_input",
                    "nav.p20_y", "nav.p40_y", "episodes.style_lane_offset_m"):
            self.assertEqual(resolve_ref(self.schema, ref).get("sign"), "right+", ref)
        for ref in ("ctl.llc_steer_lim", "ctl.input_pre_steer", "ctl.input_post_steer"):
            f = resolve_ref(self.schema, ref)
            self.assertEqual(f.get("sign"), "engine-native", ref)
            self.assertEqual(f.get("tbd"), "TBD-5", ref)
        steer = [t for t in self.schema["tbd_by_probe"] if t["id"] == "TBD-5"][0]
        self.assertEqual(steer["default"]["steer_sign"], -1)         # the r2 default, refuted by E-01 run 3
        self.assertEqual(steer["decided"]["steer_sign"], 1)
        self.assertEqual(steer["decided"]["yaw_axis_sign"], 1)
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        self.assertIn("+1", session["control.steer_sign"]["desc"])
        self.assertIn("positive `SetSteering` turns right", self.spec)
        self.assertNotIn("default -1)", self.spec)

    def test_probe_dependent_defaults(self):
        """The r2 defaults stay as history; the decided values are tested in test_freeze_decisions."""
        tbd = {t["id"]: t for t in self.schema["tbd_by_probe"]}
        self.assertEqual(tbd["TBD-1"]["default"]["telemetry_every_k"], 3)
        self.assertEqual(tbd["TBD-1"]["default"]["physics_steps_per_decision"], 6)
        self.assertEqual(tbd["TBD-2"]["default"]["image_hz"], 10)
        self.assertEqual(tbd["TBD-2"]["default"]["fallback_image_hz"], 5)
        self.assertTrue(tbd["TBD-3"]["default"]["write_returns_bytes"])
        self.assertEqual(tbd["TBD-4"]["default"]["engine_file"] % 7, "frames/f_00000007.png")
        self.assertEqual(tbd["TBD-4"]["default"]["request"] % 7, "frames/f_00000007")
        for t in tbd.values():
            self.assertIn(t["id"], self.spec)
        self.assertIn(T0ENG, self.spec)

    def test_spec_tables_equal_json(self):
        self.assertEqual(md_agreement_violations(self.schema, self.spec), [])

    def test_review_round1_rules(self):
        """Pre-capture review corrections (2026-11-04): no future obs_t, the frame-clock joins, the hl_seq rule, the
        step-dt ratio, the bridge wait, the seal-pending image, the near plane, the one-to-one road join."""
        timing = self.schema["timing"]
        self.assertIn("at or before the render frame of C_t", timing["sample"]["obs"]["rule"])
        self.assertIn("obs_age_ms >= 0", timing["sample"]["obs"]["rule"])
        self.assertIn("current render frame >= the render frame of C_t + 1", timing["decision"]["bridge_pause"])
        self.assertIn("done_render_frame set and <= the current frame", timing["decision"]["bridge_pause"])
        self.assertIn("+1 exactly on the row after every decision callback", timing["decision"]["hl_seq"])
        self.assertIn("newest ctl key with a smaller render_frame", timing["joins"]["frame_clock"])
        self.assertIn("only its pose row is missing, it is not joined", timing["joins"]["capture_pose"])
        self.assertIn("median", timing["clock"]["step_dt"])
        section6 = " ".join(self.spec[self.spec.index("## 6. Clocks"):self.spec.index("## 7. Units")].split())
        for phrase in ("whose captured frame C is at or before the render frame of C_t", "never negative",
                       "best lag to be +1", "slot 0 is issued in the next frame", "never a join failure",
                       "covered_steps / 30 on 21296 of 21296 rows", "the ratio is the step even in merged frames",
                       "every decision callback of an episode that starts or goes on"):
            self.assertIn(phrase, section6, phrase)
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        self.assertEqual((session["rig.near_plane_m"]["unit"], session["rig.near_plane_m"]["required"]), ("m", True))
        self.assertIn("near plane", session["rig.rig_id"]["desc"])
        self.assertIn("awaits the first capture Play's E-06 check", session["rig.fiducial"]["desc"])
        end = {f["name"]: f for f in self.schema["files"]["session_end.json"]["fields"]}
        self.assertFalse(end["observed.decision_grid_points_skipped"]["required"])
        index = {f["name"]: f for f in self.schema["derived"]["decision_index"]["fields"]}
        self.assertFalse(index["hl_seq"]["required"])
        tbd9 = [x for x in self.schema["tbd_by_probe"] if x["id"] == "TBD-9"][0]
        self.assertIn("one-to-one", tbd9["decided_text"])
        self.assertIn("4 twin pairs", tbd9["evidence"])
        self.assertNotIn("DRAFT", self.spec[:self.spec.index("## 1. ")])
        with open(os.path.join(REPO, "drivedit_nav", "route_schema.json"), encoding="ascii") as f:
            self.assertNotIn("DRAFT", json.load(f)["description"])

    def test_review_round2_rules(self):
        """Review round 2 (Codex N1-N5, the admission decision): the seal exemption is the one pending request at the
        terminal frame; dropped completions are writer-integrity failures and C37 measures at the decoded frame; the
        bridge waits on the current frame; the step dt never locks in a guess; hl_seq has a sentinel and increments
        only when due; episodes are admitted through a handshake."""
        timing = self.schema["timing"]
        section6 = " ".join(self.spec[self.spec.index("## 6. Clocks"):self.spec.index("## 7. Units")].split())
        for phrase in ("its image shows the terminal render frame (C = `end.render_frame`, whose hf row exists)",
                       "any other capture without its hf rows is a join failure",
                       "a completion the writer dropped", "measures each issued slot's capture stall at its decoded "
                       "frame C", "assumes the worst case (u = 1)",
                       "compares the current frame, never the image's fixed completion frame",
                       "never locking in a guess", "two halves, split at half its steps, agree within 0.5%",
                       "readiness is refused with the reason", "The span alone is not bounded",
                       "the no-command sentinel", "on no other row, so no number exists before the first episode",
                       "admission handshake", "defers the admission to the next grid point"):
            self.assertIn(phrase, section6, phrase)
        self.assertNotIn("leaves out the not yet stepped remainder", section6)
        self.assertIn("at least 90 steps", timing["clock"]["step_dt"])
        self.assertIn("readiness is refused", timing["clock"]["step_dt"])
        self.assertIn("defers to the next grid point", timing["decision"]["admission"])
        self.assertIn("no-command sentinel", timing["decision"]["hl_seq"])
        self.assertIn("every other capture without its hf rows is a join failure", timing["joins"]["capture_pose"])

    def test_review_round3_rules(self):
        """Review round 3 (Codex N6): images.pending_at_end is 0 or 1 and the seal exemption belongs to the unique
        newest issued request; the step-dt reason is the readiness refusal's reason."""
        section6 = " ".join(self.spec[self.spec.index("## 6. Clocks"):self.spec.index("## 7. Units")].split())
        for phrase in ("`images.pending_at_end` is 0 or 1; more fails C13a (HARD)",
                       "the unique newest issued slot", "only the one request the seal caught in flight (below) is "
                       "left out", "rows after it may be `skipped_busy` or `make_failed`"):
            self.assertIn(phrase, section6, phrase)
        self.assertIn("images.pending_at_end 1, never more", self.schema["timing"]["joins"]["capture_pose"])
        end = {f["name"]: f for f in self.schema["files"]["session_end.json"]["fields"]}
        self.assertIn("0 or 1, as one request is in flight at a time", end["images"]["desc"])

    def test_description_drift_is_caught(self):
        drifted = self.spec.replace("Joins ctl and telemetry exactly; the image of render frame C", "Joins ctl", 1)
        self.assertNotEqual(drifted, self.spec)
        self.assertTrue(md_agreement_violations(self.schema, drifted))
        drifted = self.spec.replace("Empty in mode I. [right+]", "Empty in mode I.", 1)
        self.assertNotEqual(drifted, self.spec)
        self.assertTrue(md_agreement_violations(self.schema, drifted))

    def test_spec_table_drift_is_caught(self):
        drifted = self.spec.replace("| `hl_accel` | float | m/s2 |", "| `hl_accel` | float | m/s |", 1)
        self.assertNotEqual(drifted, self.spec)
        self.assertTrue(md_agreement_violations(self.schema, drifted))
        bad = copy.deepcopy(self.schema)
        bad["streams"]["hf"]["fields"].append(
            {"name": "extra", "type": "int", "unit": "-", "required": False, "desc": "x"})
        self.assertTrue(md_agreement_violations(bad, self.spec))

    def test_spec_values_lists_equal_json(self):
        self.assertEqual(md_values_violations(self.schema, self.spec), [])
        drifted = self.spec.replace("ego_destroyed, ego_disabled, script_error:write_fail",
                                    "script_error:write_fail", 1)
        self.assertNotEqual(drifted, self.spec)
        self.assertTrue(md_values_violations(self.schema, drifted))

    def test_event_payloads_structure(self):
        self.assertEqual(payload_violations(self.schema), [])
        self.assertEqual(list(self.schema["event_payloads"]["kinds"]), PAYLOAD_KINDS)
        bad = copy.deepcopy(self.schema)
        bad["event_payloads"]["kinds"]["collision"][0]["unit"] = "furlong"
        self.assertTrue(payload_violations(bad))
        bad = copy.deepcopy(self.schema)
        bad["event_payloads"]["kinds"]["crash"] = []
        self.assertTrue(payload_violations(bad))
        bad = copy.deepcopy(self.schema)
        bad["event_payloads"]["kinds"]["veh_damage"][1]["range"] = [1, 0]
        self.assertTrue(payload_violations(bad))

    def test_spec_payload_table_equals_json(self):
        self.assertEqual(md_payload_violations(self.schema, self.spec), [])
        drifted = self.spec.replace("| `collision` | `src` | enum | - | yes |",
                                    "| `collision` | `src` | enum | - | no |", 1)
        self.assertNotEqual(drifted, self.spec)
        self.assertTrue(md_payload_violations(self.schema, drifted))
        bad = copy.deepcopy(self.schema)
        bad["event_payloads"]["kinds"]["rollover"][0]["enum"] = ["x"]
        bad["event_payloads"]["kinds"]["rollover"][0]["type"] = "enum"
        self.assertTrue(md_payload_violations(bad, self.spec))

    def test_crash_events(self):
        kinds = {f["name"]: f for f in self.schema["streams"]["events"]["fields"]}["kind"]["enum"]
        self.assertEqual(kinds, self.schema["enums"]["event_kind"])
        for kind in CRASH_KINDS + ["collision", "stuck", "teleport"]:
            self.assertIn(kind, kinds)
        collision = {k["key"]: k for k in self.schema["event_payloads"]["kinds"]["collision"]}
        self.assertEqual(collision["src"]["enum"], ["dv", "dv_contact", "contact"])
        self.assertEqual(collision["other"]["enum"], ["vehicle", "character", "tree", "building", "rock", "terrain",
                                                      "static", "dynamic", "unknown"])
        for key in ("onset_step", "dv_mps", "v_mps", "other", "src"):
            self.assertTrue(collision[key]["required"], key)
        self.assertEqual(self.schema["event_payloads"]["kinds"]["veh_fire_end"], [])
        rollover = {k["key"]: k for k in self.schema["event_payloads"]["kinds"]["rollover"]}
        self.assertTrue(rollover["onset_step"]["required"])
        self.assertEqual(list(rollover), ["onset_step", "up_y"])
        damage = {k["key"]: k for k in self.schema["event_payloads"]["kinds"]["veh_damage"]}
        self.assertEqual((damage["health"]["range"], damage["movement_damage"]["range"]), ([0, 1], [0, 1]))
        for key in ("dv_mps", "v_mps", "impulse_ns"):
            self.assertEqual(collision[key]["min"], 0, key)
        # frame-bound rows (D-A): the window is world time, never a step count
        crash = self.spec[self.spec.index("### 9.5.1 Crash events"):self.spec.index("### 9.6 Stream")]
        for phrase in ("world time, never a count of steps or callbacks", "collision_window_tol_ms",
                       "Episode ownership", "PrepareReposition", "crash_cooldown_ms",
                       "first terminal ego\n  event by confirmation", "the only source of the terminations",
                       "the episode that owns that step", "episode_block_max_ms", "of the ego slot and of no vehicle"):
            self.assertIn(phrase, crash, phrase)
        self.assertNotIn("6 at 60 Hz", crash)
        self.assertIn("### 9.5.1 Crash events", self.spec)
        terminations = {f["name"]: f for f in self.schema["streams"]["episodes"]["fields"]}["termination_reason"]
        self.assertIn("collision", terminations["enum"])
        self.assertIn("rollover", terminations["enum"])

    def test_ego_destroyed_is_a_normal_end(self):
        end = {f["name"]: f for f in self.schema["files"]["session_end.json"]["fields"]}
        self.assertIn("ego_destroyed", end["end_reason"]["enum"])
        self.assertIn("ego_disabled", end["end_reason"]["enum"])
        self.assertEqual(end["end_reason"]["enum"], self.schema["enums"]["end_reason"])
        self.assertIn("ego_destroyed and ego_disabled are not errors", end["status"]["desc"])
        self.assertIn("ego_disabled", self.spec[self.spec.index("### 5.3 Lifecycle"):self.spec.index("## 6. ")])
        self.assertIn("ego_destroyed", self.spec[self.spec.index("### 5.3 Lifecycle"):self.spec.index("## 6. ")])
        self.assertIn("ego_destroyed", self.spec[self.spec.index("## 15. Freeze items"):])

    def test_crash_parameters_are_session_values(self):
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        tbd = {t["id"]: t for t in self.schema["tbd_by_probe"]}["TBD-10"]
        self.assertFalse(tbd["blocks_freeze"])
        self.assertTrue(tbd["default"]["provisional"])
        for name, default in SESSION_EVENT_PARAMS.items():
            self.assertTrue(session[name]["required"], name)
            self.assertEqual(session[name].get("tbd"), "TBD-10", name)
            self.assertEqual(tbd["default"][name.split(".", 1)[1]], default, name)
        self.assertEqual(session["events.contact_source"]["enum"], ["vanilla_process_collision", "none"])
        self.assertEqual(self.schema["enums"]["contact_source"], session["events.contact_source"]["enum"])
        for stream in ("telemetry", "traffic"):
            dv = {f["name"]: f for f in self.schema["streams"][stream]["fields"]}["dv_max_mps"]
            self.assertFalse(dv["required"], stream)
            self.assertEqual((dv["unit"], dv.get("tbd")), ("m/s", "TBD-10"), stream)
        self.assertIn("PROVISIONAL", tbd["default_r2"])
        self.assertIn("1.9", tbd["default_r2"])

    def test_status_line(self):
        self.assertEqual(self.spec.split("\n")[2], STATUS_LINE)
        self.assertEqual("Status: " + self.schema["status"], STATUS_LINE)
        self.assertTrue(self.schema["status"].startswith("FROZEN"))         # data/dc2/reader.py Schema.frozen

    def test_changelog(self):
        """The freeze changelog (section 16): every decision with its date and evidence; the JSON twin carries it."""
        log = self.spec[self.spec.index("## 16. Changelog"):]
        rows = md_tables(self.spec).get(("changelog", ""))
        self.assertIsNotNone(rows)
        self.assertEqual([r[0] for r in rows], [c["id"] for c in self.schema["changelog"]])
        for r, c in zip(rows, self.schema["changelog"]):
            self.assertEqual(r[1:4], [c["date"], c["decision"], c["evidence"]], c["id"])
            self.assertTrue(c["evidence"].strip() and c["date"].startswith("2026-"), c["id"])
        ids = [c["id"] for c in self.schema["changelog"]]
        for want in ("D-A", "D-B", "D-C", "D-D", "D-E", "TBD-3", "TBD-4", "TBD-5", "TBD-6", "TBD-7", "FIDUCIAL",
                     "CAPTURE-PHASE", "FREEZE"):
            self.assertIn(want, ids)
        # review rounds 1 and 2: the pre-capture corrections, in the md and in both JSON twins, after the freeze row
        review = [i for i in ids if i.startswith("R1-") or i.startswith("R2-")]
        self.assertEqual(review, ["R1-OBS", "R1-SEAL", "R1-STEPDT", "R1-HZ", "R1-CLOCK", "R1-HLSEQ", "R1-C37", "R1-LAG",
                                  "R1-ROAD", "R1-NEAR", "R1-BRIDGE", "R1-TEXT", "R1-REFREEZE", "R2-SEAL", "R2-C37",
                                  "R2-BRIDGE", "R2-STEPDT", "R2-HLSEQ", "R2-ADMIT", "R2-REFREEZE"])
        self.assertEqual(ids[ids.index("FREEZE") + 1:], review)
        self.assertTrue(all(c["date"] == "2026-11-04" for c in self.schema["changelog"] if c["id"] in review))
        with open(FEATURES_JSON, encoding="ascii") as f:
            features = json.load(f)
        self.assertEqual(features["changelog"], self.schema["changelog"])
        self.assertEqual(features["status"], self.schema["status"])
        self.assertIn("2026-11-04", log)
        self.assertIn("dc-2.x", self.spec[self.spec.index("## 0. How to read this"):self.spec.index("## 1. ")])

    def test_session_id_rule(self):
        rx = re.compile(self.schema["session"]["id_regex"])
        self.assertTrue(rx.match(self.schema["session"]["id_example"]))
        for good in ("arland_T0b_20261025T091512Z_wb_i00_7f3a", "kolguyev_T4-K_20261118T000000Z_cl_i02_0000",
                     "everon_E-OL_20261229T235959Z_cld_i01_ffff"):
            self.assertTrue(rx.match(good), good)
        for bad in ("cain_T0b_20261025T091512Z_wb_i00_7f3a", "eden_T0b_20261025T091512Z_wb_i00_7f3a",
                    "arland_T0_b_20261025T091512Z_wb_i00_7f3a", "arland_T0b_20261025T091512_wb_i00_7f3a",
                    "arland_T0b_20261025T091512Z_client_i00_7f3a", "arland_T0b_20261025T091512Z_wb_i00_7F3A",
                    "session_2702"):
            self.assertIsNone(rx.match(bad), bad)

    def test_maps_and_sealed_test_map(self):
        maps = {m["map_id"]: m for m in self.schema["maps"]}
        self.assertEqual(sorted(maps), sorted(CANONICAL_MAP_IDS))
        self.assertEqual({m: maps[m]["terrain"] for m in maps},
                         {"arland": "Arland", "kolguyev": "Cain", "everon": "Eden"})
        self.assertEqual(maps["everon"]["splits"], ["everon_harness", "everon_test"])
        self.assertIn("SEALED", maps["everon"]["storage"])
        for m in ("arland", "kolguyev"):
            self.assertFalse(set(maps[m]["splits"]) & set(maps["everon"]["splits"]))
            self.assertNotIn("heldout", maps[m]["storage"])
        self.assertNotIn("train", maps["kolguyev"]["splits"])
        self.assertIn("loaders refuse Everon outside the sealed evaluation path", self.spec)
        self.assertEqual(self.schema["enums"]["split"],
                         [s for m in CANONICAL_MAP_IDS for s in maps[m]["splits"]])

    def test_csv_headers(self):
        for name, s in self.schema["streams"].items():
            header = ",".join(f["name"] for f in s["fields"])
            self.assertTrue(all(ord(c) < 127 for c in header), name)
            self.assertNotIn('"', header)
        commit = [f["name"] for f in self.schema["files"]["commit.csv"]["fields"]]
        self.assertEqual(commit, ["chunk_id", "first_key", "last_key", "n_rows", "n_bytes"])
        self.assertEqual(self.schema["session"]["chunk_name"] % 3, "chunk_00003.csv")

    def test_steps_per_decision_rule(self):
        """D-A: N = floor(1000 / decision_hz / step_dt_ms + 0.5) from the measured step dt, never GetFixedTimeSlice."""
        def steps_per_decision(decision_hz, step_dt_s):
            return int(math.floor(1000.0 / decision_hz / (step_dt_s * 1000.0) + 0.5))
        expect = {(5, 1 / 30.0): 6, (10, 1 / 30.0): 3, (5, 1 / 60.0): 12, (10, 1 / 60.0): 6}
        for (hz, dt), want in expect.items():
            self.assertEqual(steps_per_decision(hz, dt), want, (hz, dt))
        tbd = {t["id"]: t for t in self.schema["tbd_by_probe"]}
        d1, d2 = tbd["TBD-1"]["decided"], tbd["TBD-2"]["decided"]
        self.assertEqual(steps_per_decision(d2["decision_hz"], d1["step_dt_s"]), d1["physics_steps_per_decision"])
        self.assertEqual(steps_per_decision(d2["override_image_hz"], d1["step_dt_s"]),
                         d2["override_physics_steps_per_decision"])
        self.assertIn("never GetFixedTimeSlice", d1["step_dt_rule"])
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        self.assertIn("never the step dt", session["engine.fixed_time_slice_s"]["desc"])
        self.assertIn("step_dt_ms", session["clocks.physics_steps_per_decision"]["desc"])

    def test_decision_boundary_on_a_changing_command(self):
        """Synthetic frame-bound timeline (D-A): render frames of varying world time, each with 0 to 3 fixed steps of
        1/30 s and one callback when it has a step; decisions on the world-time grid of decision_hz, taken after the
        rows of the first callback at or past the grid point, applied from the next physics step. A builder that
        follows the JSON timing (row offsets, the covering row of a step) must give action_cmd_t = decision t,
        label_t = decision t (mode E), state_t free of decision t, the command held over the decision interval
        (P_t, P_t+1], and the per-step action_exec of decision t."""
        timing = self.schema["timing"]
        sample = timing["sample"]
        self.assertEqual(timing["decision"]["hl_seq_increments_at_row_offset"], 1)
        self.assertEqual(sample["action_cmd"]["row_offset"], 1)
        self.assertEqual(sample["state"]["row_offset"], 0)
        self.assertEqual(sample["action_label"]["row_offset"], 0)
        self.assertEqual(sample["action_exec"]["last_step_offset"], "physics_steps_per_decision")
        decision_hz, step_ms = 5, 100.0 / 3.0
        n_dec = int(math.floor(1000.0 / decision_hz / step_ms + 0.5))
        self.assertEqual(n_dec, 6)
        frame_ms = [13, 13, 14, 80, 13, 40, 95, 13, 13, 27, 100, 13, 14, 13, 66, 13] * 12

        def simulate(decision_of):
            """Rows by callback: physics_step, covered_steps, world_time_ms, hl_*, exp_*; the speed integrates the
            held acceleration over each covered step."""
            world, clock3, step = 0, 0, -1
            speed, held, seq, pending = 0.0, None, -1, None
            anchor, grid = None, 0
            rows = []
            for dt in frame_ms:
                world += dt
                covered = min(3, (3 * world - clock3) // 100)
                if covered < 1:
                    continue
                clock3 += 100 * covered
                if pending is not None:          # EOnFixedFrame: a decision of the previous callback applies
                    held, seq = pending
                    pending = None
                accel = held if held is not None else 0.0
                speed += accel * covered * step_ms / 1000.0
                step += covered
                if anchor is None:
                    anchor = world
                label = None
                if world >= anchor + grid * 1000 // decision_hz:      # rows first, then the decision
                    label = decision_of(grid, speed)
                    while anchor + grid * 1000 // decision_hz <= world:
                        grid += 1
                rows.append({"physics_step": step, "covered_steps": covered, "world_time_ms": world,
                             "hl_accel": held, "hl_seq": seq, "exp_accel_now": label, "speed": speed,
                             "llc_throttle": accel})
                if label is not None:
                    pending = (label, len([r for r in rows if r["exp_accel_now"] is not None]) - 1)
            return rows

        def covering(rows, q):
            for r in rows:
                if r["physics_step"] - r["covered_steps"] + 1 <= q <= r["physics_step"]:
                    return r
            return None

        def build(rows):
            out = []
            deciders = [i for i, r in enumerate(rows) if r["exp_accel_now"] is not None]
            for t, i in enumerate(deciders[:-1]):
                p, p_next = rows[i]["physics_step"], rows[deciders[t + 1]]["physics_step"]
                exec_rows = []
                for q in range(p + sample["action_exec"]["first_step_offset"], p + n_dec + 1):
                    r = covering(rows, q)
                    exec_rows.append(None if q > p_next or r is None else r["llc_throttle"])
                out.append({"state": rows[i + sample["state"]["row_offset"]]["speed"],
                            "action_cmd": rows[i + sample["action_cmd"]["row_offset"]]["hl_accel"],
                            "label": rows[i + sample["action_label"]["row_offset"]]["exp_accel_now"],
                            "interval": [r["hl_seq"] for r in rows if p < r["physics_step"] <= p_next],
                            "n_t": p_next - p, "exec": exec_rows})
            return out

        plan = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 3.0, -2.0, 1.0, 0.25, 0.75, -0.25, 1.25, 0.6, -0.6]
        rows = simulate(lambda t, speed: plan[t % len(plan)])
        self.assertTrue(any(r["covered_steps"] > 1 for r in rows))       # merged steps are exercised
        self.assertTrue(all(b["physics_step"] - a["physics_step"] == b["covered_steps"]
                            for a, b in zip(rows, rows[1:])))
        self.assertEqual(rows[0]["physics_step"], rows[0]["covered_steps"] - 1)
        built = build(rows)
        self.assertGreater(len(built), 10)
        lengths = set()
        for t, r in enumerate(built):
            want = plan[t % len(plan)]
            self.assertEqual(r["action_cmd"], want, t)
            self.assertEqual(r["label"], want, t)
            self.assertEqual(r["interval"], [t] * len(r["interval"]), t)    # held over (P_t, P_t+1]
            inside = [e for e in r["exec"] if e is not None]
            self.assertEqual(inside, [want] * len(inside), t)
            self.assertEqual(len(inside), min(n_dec, r["n_t"]), t)
            lengths.add(r["n_t"])
        self.assertGreater(len(lengths), 1)            # the interval varies around N with the frame clock
        # state_t never contains a step of decision t: changing decision 4 alone leaves state_4 unchanged
        changed = [plan[t % len(plan)] for t in range(len(built) + 2)]
        changed[4] = -3.0
        built2 = build(simulate(lambda t, speed: changed[t]))
        self.assertEqual(built2[4]["state"], built[4]["state"])
        self.assertNotEqual(built2[5]["state"], built[5]["state"])

    def test_hf_joins_are_exact(self):
        joins = self.schema["timing"]["joins"]
        for key in ("img_to_hf", "hf_to_ctl", "hf_to_telemetry"):
            self.assertTrue(joins[key].startswith("exact"), key)
        self.assertIn("captured_render_frame + 1", joins["capture_pose"])
        self.assertIn("done_render_frame - 1", joins["img_to_hf"])
        self.assertIn("pose row (hf row captured_render_frame + 1)", self.schema["timing"]["sample"]["obs"]["rule"])
        self.assertIn("the frame clock", self.schema["timing"]["clock"]["world_time"])

    def test_decision_and_image_grids_share_the_anchor(self):
        grid = self.schema["timing"]["image_grid"]
        self.assertIn("first ctl row", grid["anchor"])
        self.assertIn("first ctl row", self.schema["timing"]["decision"]["grid"])
        self.assertIn("img_lead_ms", grid["due_ms"])
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        for name in ("clocks.img_lead_ms", "clocks.img_inflight_timeout_ms", "clocks.image_hz_setting"):
            self.assertTrue(session[name]["required"], name)
            self.assertEqual(session[name]["tbd"], "TBD-2", name)
        # 5 Hz images and decisions, lead 0: image grid point g is due at decision grid point g
        anchor, image_hz, decision_hz, lead = 3111, 5, 5, 0
        for g in range(20):
            self.assertEqual(anchor + g * 1000 // image_hz - lead, anchor + g * 1000 // decision_hz)
        derived = {f["name"] for f in self.schema["derived"]["decision_index"]["fields"]}
        self.assertTrue({"obs_age_steps", "obs_age_ms", "obs_valid", "interval_steps"} <= derived)

    def test_freeze_decisions(self):
        """D-A, D-B, D-C and the folded facts, where the contract puts them."""
        tbd = {t["id"]: t for t in self.schema["tbd_by_probe"]}
        self.assertEqual([t for t, v in tbd.items() if v["status"] != "decided"], list(PROVISIONAL_TBD))
        # D-A
        for stream in ("ctl", "telemetry"):
            names = [f["name"] for f in self.schema["streams"][stream]["fields"]]
            self.assertEqual(names[:4], ["physics_step", "covered_steps", "render_frame", "world_time_ms"], stream)
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        self.assertNotIn("clocks.telemetry_every_k", session)
        for name in ("engine.step_dt_s", "engine.step_dt_source", "engine.fixed_time_slice_s",
                     "clocks.physics_steps_per_decision", "clocks.decision_hz"):
            self.assertTrue(session[name]["required"], name)
        self.assertEqual(session["engine.step_dt_source"]["enum"], ["engine", "measured"])
        features = json.loads(read_text(FEATURES_JSON))
        features.pop("changelog")                         # the changelog records the withdrawal
        self.assertNotIn("telemetry_every_k", json.dumps(features))
        self.assertAlmostEqual(tbd["TBD-1"]["decided"]["step_dt_s"], 1 / 30.0)
        self.assertEqual(tbd["TBD-1"]["decided"]["rows"], "frame_bound")
        section6 = " ".join(self.spec[self.spec.index("## 6. Clocks"):self.spec.index("## 7. Units")].split())
        for phrase in ("covered_steps", "EOnFixedFrame", "GetFixedSteps()", "is never the step dt", "world-time grid",
                       "frame clock", "hf row C + 1", "done_render_frame"):
            self.assertIn(phrase, section6, phrase)
        self.assertNotIn("mod physics_steps_per_decision", self.spec)
        # D-B
        d2 = tbd["TBD-2"]["decided"]
        self.assertEqual((d2["image_hz"], d2["decision_hz"], d2["physics_steps_per_decision"]), (5, 5, 6))
        self.assertEqual((d2["override_image_hz"], d2["override_setting"]), (10, "trial_conf"))
        self.assertEqual(d2["override_utilisation_u95_max"], 0.85)
        self.assertEqual(session["clocks.image_hz_setting"]["enum"], ["default", "trial_conf"])
        self.assertEqual(self.schema["enums"]["image_hz_setting"], ["default", "trial_conf"])
        # D-C
        d9 = tbd["TBD-9"]["decided"]
        self.assertEqual((d9["endpoint_tol_m"], d9["n_pts_in_join"]), (2.0, False))
        self.assertEqual(d9["road_key_format"], tbd["TBD-9"]["default"]["road_key_format"])
        tel = {f["name"]: f for f in self.schema["streams"]["telemetry"]["fields"]}
        self.assertIn("unordered endpoint pair within 2 m", tel["road_key"]["desc"])
        road = {f["name"]: f for f in self.schema["derived"]["road"]["fields"]}
        self.assertIn("unordered endpoint pair", road["matched_road_id"]["desc"])
        # folded facts
        self.assertEqual(tbd["TBD-4"]["decided"]["engine_file"] % 7, "frames/f_00000007.png")
        d7 = tbd["TBD-7"]["decided"]
        self.assertEqual((d7["gear_index_reverse"], d7["gear_index_neutral"], d7["gear_indices_forward"]),
                         (0, 1, [2, 3, 4, 5]))
        self.assertEqual(d7["gears"] - d7["forward_gears"] - 1, d7["gear_index_neutral"])
        self.assertTrue(tbd["TBD-6"]["decided"]["launch_id_visible"])
        img = {f["name"]: f for f in self.schema["streams"]["img"]["fields"]}
        self.assertEqual((img["done_render_frame"]["type"], img["done_render_frame"]["required"]), ("int", False))
        self.assertIn("carrier: shape, depth_m: 0.05", session["rig.fiducial"]["desc"])
        self.assertIn("diagnostic only", session["rig.fiducial"]["desc"])

    def test_launch_guard_modes(self):
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        self.assertNotIn("launch_guard", session)
        self.assertEqual(session["launch_guard.launch_mode"]["enum"], ["cli", "wb_escape"])
        self.assertEqual(self.schema["enums"]["launch_mode"], ["cli", "wb_escape"])
        for name in ("launch_guard.record_path", "launch_guard.record_launch_id", "launch_guard.cli_seen",
                     "launch_guard.profile_root_ok"):
            self.assertTrue(session[name]["required"], name)
        self.assertTrue(session["launch_guard.escape_binding"].get("nullable"))
        self.assertIn("Mode `wb_escape`", self.spec)
        self.assertIn("wb_R1_dclaunch.patch", self.spec)
        self.assertIn("only an inference", self.spec)

    def test_image_failure_states(self):
        img = {f["name"]: f for f in self.schema["streams"]["img"]["fields"]}
        self.assertEqual(img["issue_status"]["enum"], ["issued", "skipped_busy", "make_failed"])
        kinds = {f["name"]: f for f in self.schema["streams"]["events"]["fields"]}["kind"]["enum"]
        self.assertIn("img_timeout", kinds)
        self.assertIn("img_timeout", self.schema["enums"]["event_kind"])
        images = {f["name"]: f for f in self.schema["files"]["session_end.json"]["fields"]}["images"]["desc"]
        for word in ("make_failed", "timed_out", "pending_at_end"):
            self.assertIn(word, images)
        self.assertIn("pending_at_end", self.schema["timing"]["image_grid"]["session_end"])

    def test_vehicle_slots_never_named_slot_id(self):
        for name, s in self.schema["streams"].items():
            names = [f["name"] for f in s["fields"]]
            if name != "img":
                self.assertNotIn("slot_id", names, name)
        self.assertEqual(self.schema["streams"]["traffic"]["key"], ["physics_step", "veh_slot_id"])
        self.assertIn("veh_slot_id", [f["name"] for f in self.schema["streams"]["events"]["fields"]])

    def test_speed_names_are_distinct(self):
        for name, s in self.schema["streams"].items():
            self.assertNotIn("speed_mps", [f["name"] for f in s["fields"]], name)
        self.assertIn("speed_engine_mps", [f["name"] for f in self.schema["streams"]["telemetry"]["fields"]])

    def test_writer_patterns_and_heartbeat(self):
        self.assertIn("heartbeat.csv", self.schema["files"])
        self.assertNotIn("heartbeat.txt", self.schema["files"])
        self.assertIn("APPEND", self.schema["writer"]["heartbeat"])
        self.assertIn("APPEND", self.schema["writer"]["commit"])
        self.assertIn("at least one row", self.schema["writer"]["chunk"])
        tbd3 = [t for t in self.schema["tbd_by_probe"] if t["id"] == "TBD-3"][0]
        self.assertTrue(tbd3["default"]["append_reopen_per_line"])
        self.assertFalse(tbd3["default"]["rewrite_in_place"])
        self.assertIn("c11", tbd3["default_r2"])

    def test_session_root_and_identity_rules(self):
        self.assertTrue(self.schema["session"]["root"].startswith("$profile:DC_Sessions/"))
        self.assertTrue(any("DrivingData" in r for r in self.schema["session"]["never_read_roots"]))
        session = {f["name"]: f for f in self.schema["files"]["session.json"]["fields"]}
        self.assertIn("harvest.json", session["pin.build_key"]["desc"])
        self.assertIn("never edited", session["pin.build_key"]["desc"])
        self.assertIn("dcTrial", session["exe.cli"]["desc"])
        self.assertNotIn("Every -dc*", session["exe.cli"]["desc"])
        tbd9 = [t for t in self.schema["tbd_by_probe"] if t["id"] == "TBD-9"][0]
        self.assertEqual(tbd9["default"]["road_key_format"], "<n_pts>/<x0>:<z0>/<x1>:<z1>")
        self.assertNotIn(",", tbd9["default"]["road_key_format"])

    def test_reserved_reaction_fields(self):
        for name in ("react_veh_slot_id", "react_visible"):
            f = resolve_ref(self.schema, "ctl." + name)
            self.assertIsNotNone(f, name)
            self.assertTrue(f.get("reserved"), name)

    def test_open_items_list_every_proposal(self):
        section15 = self.spec[self.spec.index("## 15. Freeze items"):]
        for phrase in ("decision boundary", "world-time decision grid", "obs_t rule", "ego frame", "`end_reason` values",
                       "chunk rules", "nav_v1 rules", "route schema", "`wb_escape`", "DC_Sessions/",
                       "forward-only", "crash events", "modded class"):
            self.assertIn(phrase, section15, phrase)
        self.assertIn("nav_v2", self.spec[:self.spec.index("## 1. ")])

    def test_files_ascii_lf(self):
        for path in (SPEC_MD, STREAMS_JSON, FEATURES_JSON):
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            self.assertTrue(all(b < 127 for b in data), path)
            self.assertNotIn(b"\r", data, path)
            self.assertTrue(data.endswith(b"\n"), path)


# ---------------------------------------------------------------------------------------------------------------
# features_v1 registry

CLASSES = ("student_input", "teacher_input", "aux_target", "eval_only")
ROLES = ("world_model", "policy", "teacher")
DEPLOYED_ROLES = ("world_model", "policy")

# CAPTURE_PROGRAM r2 3.3, tensor table: name -> (shape or None, channel names or None, class).
R2_TENSORS = {
    "image": (["T", 3, 288, 512], None, "student_input"),
    "state": (["T", 4], ["speed_mps", "yaw_rate_rps", "acc_long_achieved_mps2", "dir_state"], "student_input"),
    "action_cmd": (["T", 3], ["kappa_target", "a_target", "dir_cmd"], "student_input"),
    "action_label": (["T", 3], None, "aux_target"),
    "action_exec": (["T", 6, 4], None, "student_input"),
    "nav": (["T", 10], ["p20_x", "p20_y", "p40_x", "p40_y", "man_left", "man_straight", "man_right", "man_follow",
                        "d_man", "d_stop"], "student_input"),
    "style": (["T", 3], None, "student_input"),
    "masks": (None, ["policy_valid", "wm_valid", "warmup", "env_transition", "teleport"], "aux_target"),
}
# r2 3.3 rows 'aux targets' and 'eval-only': item -> channel names and their required class.
R2_CLASSED_ITEMS = {
    "hindsight anchors at +2/+4/+6 s": (["anchor_2s_x", "anchor_2s_y", "anchor_4s_x", "anchor_4s_y", "anchor_6s_x",
                                         "anchor_6s_y"], "aux_target"),
    "future poses": (["fut_dx", "fut_dy", "fut_dpsi"], "aux_target"),
    "road context": (["road_class", "road_width_m", "lane_lat_err_m", "lane_head_err_rad"], "aux_target"),
    "contact events": (["contact_event", "contact_class"], "aux_target"),
    "absolute pose": (["pos_x", "pos_y", "pos_z"], "eval_only"),
    "route progress": (["route_progress_m"], "eval_only"),
    "map_id": (["map_id"], "eval_only"),
    "region": (["region"], "eval_only"),
    "pin_id and build_key (MIGRATION_1_9 8.3)": (["pin_id", "build_key"], "eval_only"),
}


def registry_channels(reg):
    for t in reg["tensors"]:
        for c in t["channels"]:
            yield t, c


def source_exists(schema, source):
    """True when a 'group.field' or 'session.json dotted.key' source names a field of the streams schema; None when
    the source is not a field reference (builder, route geometry, ...)."""
    m = re.match(r"^(session\.json|session_end\.json) ([a-z0-9_.]+)$", source)
    if m:
        return resolve_ref(schema, m.group(1) + "." + m.group(2)) is not None
    m = re.match(r"^([a-z_]+)\.([a-z0-9_]+)(=[a-z0-9_]+)?$", source)
    if m and (m.group(1) in schema["streams"] or m.group(1) in schema["derived"]):
        return resolve_ref(schema, source) is not None
    return None


def registry_violations(reg, schema=None):
    """Rules of features_v1.json (the DD-11-min acceptance rules plus consistency). Returns violations."""
    errors = []
    for k in ("schema", "features_version", "contract_version", "status", "classes", "roles", "offsets", "rules",
              "c24_forbidden_inputs", "tensors"):
        if k not in reg:
            errors.append("top level: missing %r" % k)
    if errors:
        return errors
    if reg["schema"] != "drivedit.features/1" or reg["features_version"] != "features_v1":
        errors.append("top level: wrong schema id or version")
    if tuple(reg["classes"]) != CLASSES:
        errors.append("classes must be exactly %r" % (CLASSES,))
    if tuple(reg["roles"]) != ROLES:
        errors.append("roles must be exactly %r" % (ROLES,))
    tnames = [t["name"] for t in reg["tensors"]]
    if len(set(tnames)) != len(tnames):
        errors.append("duplicate tensor names")
    cnames = [c["name"] for _, c in registry_channels(reg)]
    dup = sorted({n for n in cnames if cnames.count(n) > 1})
    if dup:
        errors.append("channel names used twice: %r" % dup)
    by_name = {c["name"]: (t, c) for t, c in registry_channels(reg)}
    for t in reg["tensors"]:
        where = "tensor " + t["name"]
        cls = t.get("class")
        if cls not in CLASSES:
            errors.append("%s: class %r" % (where, cls))
        shape = t.get("shape", [])
        if not shape or shape[0] != "T":
            errors.append("%s: shape must start with T" % where)
        else:
            n_axis = shape[t.get("channel_axis", -1)]
            if isinstance(n_axis, int) and n_axis != len(t["channels"]):
                errors.append("%s: channel dimension %d != %d channels" % (where, n_axis, len(t["channels"])))
        acc = t.get("access", {})
        if tuple(acc) != ROLES:
            errors.append("%s: access must list exactly %r" % (where, ROLES))
            continue
        for role, a in acc.items():
            if set(a) != {"input", "max_offset", "target"}:
                errors.append("%s/%s: access keys %r" % (where, role, sorted(a)))
                continue
            if not a["input"] and a["max_offset"] is not None:
                errors.append("%s/%s: max_offset without input" % (where, role))
            if a["input"] and a["max_offset"] is None and role in DEPLOYED_ROLES:
                errors.append("%s/%s: a deployed role needs a bounded max_offset" % (where, role))
        for c in t["channels"]:
            for k in ("name", "unit", "source", "absolute", "build_scoped", "desc"):
                if k not in c:
                    errors.append("%s.%s: missing %r" % (where, c.get("name"), k))
            if cls == "student_input" and c.get("absolute"):
                errors.append("%s.%s: a student_input channel is absolute" % (where, c["name"]))
            if schema is not None and "source" in c:
                ok = source_exists(schema, c["source"])
                if ok is False:
                    errors.append("%s.%s: source %r is not in the streams schema" % (where, c["name"], c["source"]))
                if ok:
                    src = c["source"].replace("session.json ", "session.json.").replace("session_end.json ",
                                                                                         "session_end.json.")
                    f = resolve_ref(schema, src)
                    if f.get("build_scoped") and not c.get("build_scoped") and "=" not in src:
                        errors.append("%s.%s: built from a build-scoped field but not build_scoped" % (where,
                                                                                                    c["name"]))
        if cls == "teacher_input" and any(acc[r]["input"] for r in DEPLOYED_ROLES):
            errors.append("%s: teacher_input reaches a deployed role" % where)
        if cls == "aux_target" and any(acc[r]["input"] for r in ROLES):
            errors.append("%s: aux_target used as an input" % where)
        if cls == "eval_only" and any(acc[r]["input"] or acc[r]["target"] for r in ROLES):
            errors.append("%s: eval_only used as an input or target" % where)
        if t["name"] in ("action_cmd", "action_exec"):
            pol = acc["policy"]
            if pol["input"] and (pol["max_offset"] is None or pol["max_offset"] >= 0):
                errors.append("%s: the policy may see it only as history (offset <= -1)" % where)
            wm = acc["world_model"]
            if t["name"] == "action_cmd" and (not wm["input"] or wm["max_offset"] != 0):
                errors.append("%s: the world model conditions on it at offset 0" % where)
        if t["name"] in ("image", "state", "nav", "style"):
            for r in DEPLOYED_ROLES:
                if acc[r]["input"] and acc[r]["max_offset"] > 0:
                    errors.append("%s/%s: future values as input" % (where, r))
        if t["name"] == "action_label" and not acc["policy"]["target"]:
            errors.append("%s: must be the policy target" % where)
    forb = reg["c24_forbidden_inputs"]
    for n in forb.get("names", []):
        if n not in by_name:
            errors.append("C24: forbidden channel %r is not in the registry" % n)
        elif by_name[n][0]["class"] in ("student_input", "teacher_input"):
            errors.append("C24: forbidden channel %r is an input class" % n)
    for t, c in registry_channels(reg):
        if t["class"] in ("student_input", "teacher_input"):
            for pat in forb.get("patterns", []):
                if fnmatch.fnmatchcase(c["name"], pat):
                    errors.append("C24: input channel %r matches forbidden pattern %r" % (c["name"], pat))
    return errors


class FeaturesRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = load_json(FEATURES_JSON)
        cls.schema = load_json(STREAMS_JSON)
        cls.spec = read_text(SPEC_MD)
        cls.tensors = {t["name"]: t for t in cls.reg["tensors"]}
        cls.channels = {c["name"]: (t, c) for t, c in registry_channels(cls.reg)}

    def test_rules(self):
        self.assertEqual(registry_violations(self.reg, self.schema), [])

    def test_no_student_input_is_absolute(self):
        for t, c in registry_channels(self.reg):
            if t["class"] == "student_input":
                self.assertFalse(c["absolute"], c["name"])

    def test_policy_sees_actions_only_as_history(self):
        for name in ("action_cmd", "action_exec"):
            pol = self.tensors[name]["access"]["policy"]
            self.assertTrue(pol["input"])
            self.assertIsNotNone(pol["max_offset"])
            self.assertLessEqual(pol["max_offset"], -1, name)
        self.assertEqual(self.tensors["action_cmd"]["access"]["world_model"]["max_offset"], 0)
        for name in ("state", "nav", "style"):
            for role in DEPLOYED_ROLES:
                self.assertLessEqual(self.tensors[name]["access"][role]["max_offset"], 0, (name, role))

    def test_broken_registries_are_caught(self):
        cases = []
        bad = copy.deepcopy(self.reg)
        [t for t in bad["tensors"] if t["name"] == "action_cmd"][0]["access"]["policy"]["max_offset"] = 0
        cases.append(("policy action_cmd at offset 0", bad))
        bad = copy.deepcopy(self.reg)
        [t for t in bad["tensors"] if t["name"] == "action_exec"][0]["access"]["policy"]["max_offset"] = 0
        cases.append(("policy action_exec at offset 0", bad))
        bad = copy.deepcopy(self.reg)
        [t for t in bad["tensors"] if t["name"] == "state"][0]["channels"][0]["absolute"] = True
        cases.append(("absolute student input", bad))
        bad = copy.deepcopy(self.reg)
        anchors = [t for t in bad["tensors"] if t["name"] == "hindsight_anchors"][0]
        anchors["access"]["policy"] = {"input": True, "max_offset": 0, "target": False}
        cases.append(("aux_target as input", bad))
        bad = copy.deepcopy(self.reg)
        ev = [t for t in bad["tensors"] if t["name"] == "eval_only"][0]
        moved = [c for c in ev["channels"] if c["name"] == "map_id"][0]
        ev["channels"].remove(moved)
        moved = dict(moved, absolute=False)
        [t for t in bad["tensors"] if t["name"] == "style"][0]["channels"].append(moved)
        cases.append(("map_id as a student input", bad))
        bad = copy.deepcopy(self.reg)
        [t for t in bad["tensors"] if t["name"] == "nav"][0]["channels"].append(
            {"name": "goal_2s_x", "unit": "m", "source": "builder", "absolute": False, "build_scoped": False,
             "desc": "planned-time goal"})
        cases.append(("planned-time goal as input", bad))
        bad = copy.deepcopy(self.reg)
        [t for t in bad["tensors"] if t["name"] == "nav"][0]["access"]["world_model"]["max_offset"] = 1
        cases.append(("future nav for the world model", bad))
        bad = copy.deepcopy(self.reg)
        [t for t in bad["tensors"] if t["name"] == "action_exec"][0]["channels"][1]["build_scoped"] = False
        cases.append(("raw input not build_scoped", bad))
        for label, reg in cases:
            self.assertTrue(registry_violations(reg, self.schema), label)

    def test_r2_tensors_present(self):
        for name, (shape, channels, cls) in R2_TENSORS.items():
            self.assertIn(name, self.tensors)
            t = self.tensors[name]
            self.assertEqual(t["class"], cls, name)
            if shape is not None:
                self.assertEqual(t["shape"], shape, name)
            if channels is not None:
                self.assertEqual([c["name"] for c in t["channels"]], channels, name)
        self.assertTrue(self.tensors["action_label"]["access"]["policy"]["target"])
        self.assertTrue(self.tensors["action_exec"].get("optional"))

    def test_r2_classed_items_present(self):
        for item, (names, cls) in R2_CLASSED_ITEMS.items():
            for n in names:
                self.assertIn(n, self.channels, item)
                self.assertEqual(self.channels[n][0]["class"], cls, (item, n))

    def test_c24_forbidden_list(self):
        forb = self.reg["c24_forbidden_inputs"]
        for n in ("map_id", "pos_x", "pos_y", "pos_z", "distance_total_m", "heading_world_rad", "vel_world_x",
                  "anchor_2s_x", "anchor_6s_y", "fut_dx"):
            self.assertIn(n, forb["names"])
        self.assertIn("anchor_*", forb["patterns"])
        self.assertIn("goal_*", forb["patterns"])

    def test_raw_inputs_build_scoped(self):
        for c in self.tensors["action_exec"]["channels"]:
            self.assertTrue(c["build_scoped"], c["name"])
        for n in ("kappa_target", "speed_mps", "p20_x"):
            self.assertFalse(self.channels[n][1]["build_scoped"], n)

    def test_offsets_block(self):
        off = self.reg["offsets"]
        self.assertEqual(off["decision_hz"], 5)
        self.assertEqual(off["physics_steps_per_decision"], 6)
        self.assertAlmostEqual(off["step_dt_s"], 1 / 30.0)
        self.assertEqual(off["tbd"], "TBD-1")
        self.assertEqual(self.reg["contract_version"], self.schema["contract_version"])
        self.assertEqual(self.reg["status"], self.schema["status"])

    def test_spec_tensor_table_equals_json(self):
        rows = md_tables(self.spec).get(("tensors", ""))
        self.assertIsNotNone(rows)
        self.assertEqual([r[0].strip("`") for r in rows], [t["name"] for t in self.reg["tensors"]])
        for r, t in zip(rows, self.reg["tensors"]):
            self.assertEqual(r, tensor_row(t), t["name"])

    def test_spec_tensor_drift_is_caught(self):
        drifted = self.spec.replace("| `action_cmd` | [T,3] | student_input | <= 0 | <= -1 |",
                                    "| `action_cmd` | [T,3] | student_input | <= 0 | <= 0 |", 1)
        self.assertNotEqual(drifted, self.spec)
        rows = md_tables(drifted)[("tensors", "")]
        bad = [t["name"] for r, t in zip(rows, self.reg["tensors"]) if r != tensor_row(t)]
        self.assertEqual(bad, ["action_cmd"])
        drifted = self.spec.replace("| TBD-7 | gearbox and gear index semantics |", "| TBD-7 | gear map |", 1)
        self.assertTrue(md_agreement_violations(self.schema, drifted))

    def test_10hz_override_shapes(self):
        override = self.reg["offsets"]["override_10hz"]
        self.assertNotIn("fallback_5hz", self.reg["offsets"])
        self.assertEqual(override["decision_hz"], 10)
        self.assertEqual(override["shapes"]["action_exec"], ["T", override["physics_steps_per_decision"], 4])
        self.assertEqual(self.tensors["action_exec"]["shape"][1], self.reg["offsets"]["physics_steps_per_decision"])
        tbd2 = [t for t in self.schema["tbd_by_probe"] if t["id"] == "TBD-2"][0]
        self.assertEqual(tbd2["decided"]["override_physics_steps_per_decision"], override["physics_steps_per_decision"])
        self.assertEqual(tbd2["decided"]["physics_steps_per_decision"],
                         self.reg["offsets"]["physics_steps_per_decision"])
        self.assertEqual(self.tensors["action_cmd"]["shape"], ["T", 3])

    def test_policy_valid_has_a_recorded_source(self):
        pv = self.channels["policy_valid"][1]
        self.assertEqual(pv["source"], "ctl.react_visible")
        self.assertTrue(resolve_ref(self.schema, pv["source"]).get("reserved"))
        self.assertIn("(P_t, P_t+1]", self.tensors["action_cmd"]["rule"])
        self.assertIn("exp_*_now of C_t", self.tensors["action_label"]["rule"])


if __name__ == "__main__":
    unittest.main()
