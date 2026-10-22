#!/usr/bin/env python3
"""Schema tests for contract dc-2.0 (DD-11-min), stdlib unittest only.

    python3 -I -B tests/test_contract_dc2.py -v
    python3 -I -B -m pytest -p no:cacheprovider tests/test_contract_dc2.py

Checks, without touching any session data:
  * docs/capture_contract/dc-2.0.streams.json is structurally valid (keys, types, units, enums, TBD references);
  * every stream field that CAPTURE_PROGRAM r2 3.3 lists, and every addition MIGRATION_1_9 8.3 asks of DD-11-min,
    appears in the JSON files;
  * the field tables in docs/capture_contract/dc-2.0.md equal the JSON, row by row;
  * the probe-dependent values are TBD entries that cite docs/trials/T0-ENG.md;
  * the session id rule and the canonical map ids;
  * the committed contract files are ASCII with LF line ends.
The checks are plain functions that return a list of violations, so the tests can also show that a broken copy of a
file is caught.
"""

import copy
import fnmatch
import json
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONTRACT_DIR = os.path.join(REPO, "docs", "capture_contract")
SPEC_MD = os.path.join(CONTRACT_DIR, "dc-2.0.md")
STREAMS_JSON = os.path.join(CONTRACT_DIR, "dc-2.0.streams.json")
FEATURES_JSON = os.path.join(CONTRACT_DIR, "features_v1.json")

STATUS_LINE = ("Status: DRAFT 2026-10-22, freezes at T0-ENG core GO after E-01/E-05/E-10/E-19; pending user "
               "decision D15")
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
    "telemetry: speed": ["telemetry.speed_mps"],
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
        for k in ("id", "item", "default_r2", "default", "decided_by", "blocks_freeze", "results", "affects"):
            if k not in t:
                errors.append("tbd_by_probe %s: missing %r" % (t.get("id"), k))
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
        m = re.match(r"^<!-- dc2:(fields (stream|derived|file)=(\S+)|tbd|tensors) -->$", line.strip())
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
    known = {(k, n) for k, entries in groups for n in entries}
    for key in tables:
        if key[0] in ("stream", "derived", "file") and key not in known:
            errors.append("spec: field table %r has no JSON counterpart" % (key,))
    tbd_rows = tables.get(("tbd", ""))
    if tbd_rows is None or [r[0] for r in tbd_rows] != [t["id"] for t in schema["tbd_by_probe"]]:
        errors.append("spec: the TBD table differs from tbd_by_probe")
    return errors


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
        self.assertEqual(self.schema["streams"]["ctl"]["key_step"], 1)
        self.assertEqual(self.schema["streams"]["telemetry"]["key_step"], "k")

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
        self.assertEqual(steer["default"]["steer_sign"], -1)

    def test_probe_dependent_defaults(self):
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

    def test_spec_table_drift_is_caught(self):
        drifted = self.spec.replace("| `hl_accel` | float | m/s2 |", "| `hl_accel` | float | m/s |", 1)
        self.assertNotEqual(drifted, self.spec)
        self.assertTrue(md_agreement_violations(self.schema, drifted))
        bad = copy.deepcopy(self.schema)
        bad["streams"]["hf"]["fields"].append(
            {"name": "extra", "type": "int", "unit": "-", "required": False, "desc": "x"})
        self.assertTrue(md_agreement_violations(bad, self.spec))

    def test_status_line(self):
        self.assertEqual(self.spec.split("\n")[2], STATUS_LINE)
        self.assertEqual("Status: " + self.schema["status"], STATUS_LINE)

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

    def test_files_ascii_lf(self):
        for path in (SPEC_MD, STREAMS_JSON, FEATURES_JSON):
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            self.assertTrue(all(b < 127 for b in data), path)
            self.assertNotIn(b"\r", data, path)
            self.assertTrue(data.endswith(b"\n"), path)


if __name__ == "__main__":
    unittest.main()
