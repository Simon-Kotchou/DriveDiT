#!/usr/bin/env python3
"""Static checks of the dc-2.0 writer (DD-4-min Enforce sources) against docs/capture_contract/dc-2.0.streams.json.

    TMPDIR=/dev/shm/<x> python3 -I -B -m pytest --noconftest -p no:cacheprovider -q tests/test_capture_writer_contract.py

The Enforce code cannot run here, so these tests read it: the event kinds, the end reasons, the column order of
ctl, telemetry, events and episodes, the session.json events.* keys, the crash payload keys and the crash watch's
TBD-10 defaults (code and DC_Arland layer) must equal the contract, and the crash detector must be the world-time rule
of contract 9.5.1 that data/dc2/crash_dv.py specifies (no step or callback window). The fixtures test the reader and
the validator; this file ties the writer to the same contract (review round 1 of the crash events, L7).
"""

import importlib.util
import json
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC = os.path.join(REPO, "src", "enfusion", "datacapture")
STREAMS_JSON = os.path.join(REPO, "docs", "capture_contract", "dc-2.0.streams.json")
LAYER = os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Arland_Layers", "default.layer")
INPUT_SUFFIXES = ("steer", "throttle", "brake", "clutch", "gear", "handbrake")
WHEELS = 4
# DC_CrashWatch attribute -> session.json events.* key (contract 10.1, TBD-10)
# The dc-2.0 capture core (DC_Arland's modules): every out parameter is assigned on entry (review round 2)
CORE_FILES = ("DC_Session.c", "DC_EgoSession.c", "DC_StreamRecorder.c", "DC_CrashWatch.c", "DC_HfRecorder.c",
              "DC_ImageScheduler.c", "DC_ChunkWriter.c", "DC_VehicleRegistry.c")
ATTRIBUTE_KEYS = {"m_fCollisionDvMps": "collision_dv_mps", "m_iCollisionWindowMs": "collision_window_ms",
                  "m_iCollisionWindowTolMs": "collision_window_tol_ms", "m_iCollisionPeakMs": "collision_peak_ms",
                  "m_iCollisionRefractoryMs": "collision_refractory_ms", "m_iEpisodeEndWaitMs": "episode_end_wait_ms",
                  "m_iCrashCooldownMs": "crash_cooldown_ms", "m_iEpisodeBlockMaxMs": "episode_block_max_ms",
                  "m_fRolloverUpY": "rollover_up_y",
                  "m_iRolloverHoldMs": "rollover_hold_ms", "m_iStatePollMs": "state_poll_ms"}


def source(name):
    """An Enforce file without comments (string literals kept)."""
    with open(os.path.join(SRC, name), encoding="ascii") as f:
        text = f.read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)//.*$", "", text)


def body(text, signature):
    """The brace block after the first line that contains signature."""
    start = text.index(signature)
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError(signature)


def strings(text):
    return re.findall(r'"([^"\\]*)"', text)


def columns_of(block):
    """Column order of an Add*Columns method: Insert("x"), string arrays inserted by foreach, AddVectorColumns,
    AddInputColumns and the wheel loop (prefix "w" + i + "_")."""
    out, arrays = [], {}
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"array<string> (\w+) = \{(.*)\};", line)
        if m:
            arrays[m.group(1)] = strings(m.group(2))
        elif line.startswith("foreach (string"):
            name = re.match(r"foreach \(string \w+ : (\w+)\)", line).group(1)
            out += arrays[name]
        elif line.startswith("for (int i = 0; i < WHEEL_COLUMNS"):
            depth, inner = 0, []
            i += 1
            while True:
                part = lines[i].strip()
                depth += part.count("{") - part.count("}")
                inner.append(part)
                if depth == 0 and part == "}":
                    break
                i += 1
            names = re.findall(r'columns\.Insert\(prefix \+ "(\w+)"\)', "\n".join(inner))
            out += ["w%d_%s" % (w, n) for w in range(WHEELS) for n in names]
        else:
            m = re.search(r'AddVectorColumns\(columns, "(\w+)"\)', line)
            if m:
                out += [m.group(1) + "_" + a for a in "xyz"]
            m2 = re.search(r'AddInputColumns\(columns, "(\w+)"\)', line)
            if m2:
                out += [m2.group(1) + s for s in INPUT_SUFFIXES]
            m3 = re.search(r'columns\.Insert\("(\w+)"\)', line)
            if m3:
                out.append(m3.group(1))
        i += 1
    return out


class WriterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(STREAMS_JSON, encoding="ascii") as f:
            cls.schema = json.load(f)
        cls.session = source("DC_Session.c")
        cls.recorder = source("DC_StreamRecorder.c")
        cls.ego = source("DC_EgoSession.c")
        cls.crash = source("DC_CrashWatch.c")

    def fields(self, stream):
        return [f["name"] for f in self.schema["streams"][stream]["fields"]]

    def test_event_kinds(self):
        kinds = strings(re.search(r"s_aEventKinds = \{(.*?)\};", self.session).group(1))
        self.assertEqual(kinds, self.schema["enums"]["event_kind"])
        used = set()
        for name in os.listdir(SRC):
            if name.startswith("DC_") and name.endswith(".c"):
                text = source(name)
                used |= set(re.findall(r'LogEvent\("(\w+)"', text))
                used |= set(re.findall(r'LogRowIn\(slot, [^,]+, "(\w+)"', text))
        self.assertTrue(used <= set(kinds), used - set(kinds))

    def test_end_reasons(self):
        values = dict(re.findall(r'static const string (END_\w+) = "([^"]+)";', self.session))
        self.assertTrue(set(values.values()) <= set(self.schema["enums"]["end_reason"]))
        normal = re.findall(r"endReason != (END_\w+)", body(self.session, "static bool IsErrorEndReason("))
        self.assertEqual(sorted(values[n] for n in normal),
                         ["dc_stop", "duration", "ego_destroyed", "ego_disabled", "stop"])

    def test_stream_columns(self):
        self.assertEqual(columns_of(body(self.recorder, "static void AddCtlColumns(")), self.fields("ctl"))
        self.assertEqual(columns_of(body(self.recorder, "static void AddTelemetryColumns(")), self.fields("telemetry"))
        self.assertEqual(columns_of(body(self.session, "static void AddEventColumns(")), self.fields("events"))
        self.assertEqual(columns_of(body(self.ego, "static void AddEpisodeColumns(")), self.fields("episodes"))

    def test_session_events_keys(self):
        block = body(self.crash, "override void WriteSessionJson(")
        events = block[:block.index("return;")]
        keys = ["events." + k for k in re.findall(r'json\.Add\w+\("(\w+)"', events)]
        want = [f["name"] for f in self.schema["files"]["session.json"]["fields"] if f["name"].startswith("events.")]
        self.assertEqual(keys, want)
        self.assertIn('json.BeginObject("events")', self.session)

    def test_crash_payload_keys(self):
        payloads = self.schema["event_payloads"]["kinds"]
        written = set(re.findall(r'LogRowIn\(slot, [^,]+, "(\w+)"', self.crash))
        self.assertEqual(written, {"collision", "rollover", "veh_damage", "veh_fire_start", "veh_fire_end",
                                   "driver_state", "dismount", "remount"})
        literal_keys = set(re.findall(r'"(?:;)?([a-z_]+)="', self.crash))
        known = set(k["key"] for kind in written for k in payloads[kind])
        self.assertTrue(literal_keys <= known, literal_keys - known)
        for kind in written:
            required = set(k["key"] for k in payloads[kind] if k["required"])
            self.assertTrue(required <= literal_keys, (kind, required - literal_keys))
        other = [k for k in payloads["collision"] if k["key"] == "other"][0]["enum"]
        names = strings(re.search(r"s_aOtherNames = \{(.*?)\};", self.crash).group(1))
        self.assertEqual(sorted(names), sorted(other))
        src = [k for k in payloads["collision"] if k["key"] == "src"][0]["enum"]
        self.assertTrue(set(re.findall(r'(?:src = "|;src=)(\w+)', self.crash)) <= set(src))

    def test_crash_defaults_equal_the_contract(self):
        default = [t for t in self.schema["tbd_by_probe"] if t["id"] == "TBD-10"][0]["default"]
        attrs = dict((m.group(2), m.group(1)) for m in re.finditer(
            r'\[Attribute\("([^"]*)"[^\n]*\]\s*\n\s*protected \w+ (m_\w+);', self.crash))
        with open(LAYER, encoding="ascii") as f:
            layer = f.read()
        block = re.search(r'DC_CrashWatch "\{[0-9A-F]{16}\}" \{\n(.*?)\n  \}', layer, re.S).group(1)
        values = dict(re.findall(r"^   (m_\w+) (\S+)$", block, re.M))
        for attr, key in ATTRIBUTE_KEYS.items():
            self.assertEqual(float(attrs[attr]), float(default[key]), attr)
            self.assertEqual(float(values[attr]), float(default[key]), attr)
        self.assertEqual(values["m_bUseContactHook"], "1")         # DC_Arland records vanilla_process_collision

    def test_detector_is_the_world_time_rule(self):
        spec = importlib.util.spec_from_file_location("dc2_crash_dv", os.path.join(REPO, "data", "dc2", "crash_dv.py"))
        crash_dv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crash_dv)
        self.assertEqual(int(re.search(r"RING_CAPACITY = (\d+);", self.crash).group(1)), crash_dv.CAPACITY)
        self.assertNotIn("GetFixedTimeSlice", self.crash)          # never a window of steps
        sample = body(self.crash, "protected bool SampleVelocity(")
        self.assertIn("worldMs - m_iCollisionWindowMs", sample)      # the reference instant is world time
        self.assertIn("newerMs - olderMs > m_iCollisionWindowMs + m_iCollisionWindowTolMs", sample)   # span rule
        self.assertIn("f = part / span", sample)                    # interpolation
        step = body(self.crash, "protected void StepSlot(")
        self.assertIn("worldMs <= slot.m_iOnsetMs + m_iCollisionPeakMs", step)
        self.assertIn("slot.m_iOnsetEpisode = m_Session.GetLastStepEpisodeId()", step)


    def test_round2_rules(self):
        # reports between callbacks pair the last completed step with the episode that owns it (contract 9.5.1)
        self.assertIn("m_iLastStepEpisodeId = m_iEpisodeId;", body(self.session, "protected void HandleStepPost("))
        self.assertIn("record.m_iEpisode = m_Session.GetLastStepEpisodeId()", self.crash)
        self.assertIn("int episodeId = m_Session.GetLastStepEpisodeId()", body(self.crash, "protected void PollSlot("))
        self.assertNotIn("m_Session.GetEpisodeId();", self.crash.replace("episodeId == m_Session.GetEpisodeId()", ""))
        # the crash watch is the only source of collision and rollover terminations
        crash_terms = strings(re.search(r"s_aCrashTerminations = \{(.*?)\};", self.recorder).group(1))
        self.assertEqual(sorted(crash_terms), ["collision", "rollover"])
        self.assertIn("NoteIgnoredDriverTermination", body(self.recorder, "override void OnStepPost("))
        # a disabled ego blocks episodes at most episode_block_max_ms, then the session ends with ego_disabled
        begin = body(self.ego, "protected void TryBeginEpisode(")
        self.assertIn("m_CrashWatch.GetEpisodeBlockMaxMs()", begin)
        self.assertIn("m_Session.EndPlanned(DC_Session.END_EGO_DISABLED", begin)
        self.assertIn("BeginEnd(endReason);", body(self.session, "void EndPlanned("))

    def test_out_parameters_are_assigned_on_entry(self):
        sig = re.compile(r"^\t(?:[a-z]+ )*[\w<>]+ (\w+)\(([^)]*\bout\b[^)]*)\)\n\t\{", re.M)
        checked = 0
        for name in CORE_FILES:
            text = source(name)
            for m in sig.finditer(text):
                outs = re.findall(r"\bout [\w<>]+ (\w+)", m.group(2))
                head = body(text, m.group(0)).strip().split(";")[:len(outs)]
                assigned = [re.match(r"\s*(\w+) = ", s).group(1) for s in head if re.match(r"\s*(\w+) = ", s)]
                self.assertEqual(sorted(assigned), sorted(outs), "%s %s" % (name, m.group(1)))
                checked += 1
        self.assertGreaterEqual(checked, 20)


if __name__ == "__main__":
    unittest.main()
