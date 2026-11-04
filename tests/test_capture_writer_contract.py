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

    def test_freeze_frame_bound_rows(self):
        """D-A: one ctl and one telemetry row per callback, physics_step + covered_steps, N from the measured step dt
        (never GetFixedTimeSlice), decisions on the world-time grid, the image grid on the same anchor."""
        pre = body(self.session, "protected void HandleStepPre(")
        self.assertIn("m_iPhysicsStep += covered;", pre)
        self.assertIn("m_iGridAnchorMs = m_iCallbackWorldMs;", pre)
        self.assertIn("m_bDecisionStep = m_iCallbackWorldMs >= m_iGridAnchorMs + m_iNextDecisionGrid * m_iDecisionPeriodMs;", pre)
        self.assertIn("module.OnStepPre(m_iPhysicsStep, sliceS);", pre)
        self.assertIn("int worldMs = m_iCallbackWorldMs;", body(self.session, "protected void HandleStepPost("))
        clocks = body(self.session, "protected bool ComputeClocks(")
        self.assertNotIn("GetFixedTimeSlice", re.sub(r'"[^"]*"', '""', clocks))     # the log line may name it
        self.assertIn("m_iStepsPerDecision = Math.Floor(m_iDecisionPeriodMs / (m_fStepDtS * 1000) + 0.5);", clocks)
        self.assertIn("world.GetFixedSteps()", body(self.session, "protected void NoteFrameSteps("))
        self.assertIn("EntityEvent.POSTFRAME", body(self.session, "protected int GetDispatchEventMask("))
        for text in (self.session, self.recorder, self.ego):
            self.assertNotIn("% m_Session.GetStepsPerDecision()", text)
            self.assertNotIn("GetTelemetryEveryK()", text.replace("int GetTelemetryEveryK()", ""))
        self.assertIn("!m_Session.IsDecisionStep()", body(self.recorder, "override void OnStepDecide("))
        self.assertIn("!m_Session.IsDecisionStep()", body(self.ego, "override void OnStepPre("))
        self.assertIn("!m_Session.IsDecisionStep()", body(self.ego, "override void OnStepDecide("))
        post = body(self.recorder, "override void OnStepPost(")
        self.assertIn("WriteTelemetryRow(physicsStep, worldMs, ego, simulation);", post)
        self.assertNotIn("% ", post)
        session_json = body(self.session, "protected bool WriteSessionJson(")
        self.assertIn('json.AddFloat("step_dt_s", m_fStepDtS, 9);', session_json)
        self.assertIn('json.AddString("image_hz_setting", m_sImageHzSetting);', session_json)
        self.assertNotIn("telemetry_every_k", session_json)
        images = source("DC_ImageScheduler.c")
        self.assertIn("int anchorMs = m_Session.GetGridAnchorWorldMs();", body(images, "override void OnFrame("))
        cols = strings(re.search(r'array<string> columns = \{(.*?)\};', images).group(1))
        self.assertEqual(cols, self.fields("img"))
        self.assertIn("row.m_iDoneFrame = renderFrame;", body(images, "protected void ResolveInFlight("))

    def test_freeze_defaults(self):
        """D-B 5 Hz (10 Hz only through the trial conf), steer_sign +1, the Shape carrier at 0.05 m with the rig's
        near plane below it, in code and in the DC_Arland layer."""
        with open(LAYER, encoding="ascii") as f:
            layer = f.read()

        def attr(text, name):
            return re.search(r'\[Attribute\("([^"]*)"[^\n]*\]\s*\n\s*protected \w+ %s;' % name, text).group(1)

        def layer_value(component, name):
            block = re.search(r'%s "\{[0-9A-F]{16}\}" \{\n(.*?)\n  \}' % component, layer, re.S).group(1)
            return dict(re.findall(r"^   (m_\w+) (\S+)$", block, re.M)).get(name)

        tbd = {x["id"]: x for x in self.schema["tbd_by_probe"]}
        self.assertEqual(float(attr(self.session, "m_fImageHz")), tbd["TBD-2"]["decided"]["image_hz"])
        self.assertEqual(float(layer_value("DC_Session", "m_fImageHz")), 5.0)
        self.assertIsNone(layer_value("DC_Session", "m_fTelemetryMaxPeriodS"))
        self.assertIn('ReadNumber(context, "image_hz", imageHz)', self.session)
        self.assertEqual(int(attr(self.recorder, "m_iSteerSign")), tbd["TBD-5"]["decided"]["steer_sign"])
        self.assertEqual(int(layer_value("DC_StreamRecorder", "m_iSteerSign")), 1)
        images = source("DC_ImageScheduler.c")
        hf = source("DC_HfRecorder.c")
        self.assertEqual(attr(images, "m_eFiducialCarrier"), "1")            # DC_EFiducialCarrier.SHAPE
        self.assertEqual(layer_value("DC_ImageScheduler", "m_eFiducialCarrier"), "1")
        depth = float(attr(images, "m_fShapeDepthM"))
        self.assertEqual(depth, 0.05)
        self.assertEqual(float(layer_value("DC_ImageScheduler", "m_fShapeDepthM")), depth)
        self.assertLess(float(attr(hf, "m_fNearPlaneM")), depth)
        self.assertLess(float(layer_value("DC_HfRecorder", "m_fNearPlaneM")), depth)
        enum = re.search(r"enum DC_EFiducialCarrier\s*\{(.*?)\}", source("DC_FiducialShape.c"), re.S).group(1)
        self.assertEqual(re.findall(r"^\s*(\w+),?", enum, re.M)[:3], ["WIDGET", "SHAPE", "BOTH"])

    def test_step_dt_from_the_callback_ratio(self):
        """Review round 1 (X2, C-L1): the step dt is the median of the readiness step callbacks' time slice / covered
        steps when they agree, else the measured span; the reference model below is the Enforce code's rule."""
        clocks = body(self.session, "protected bool ComputeClocks(")
        self.assertIn("m_fStepRatioMedianS = MedianOf(m_aStepRatios);", clocks)
        self.assertIn("m_fMeasuredStepS = m_fSpanS / m_iSpanSteps;", clocks)
        self.assertIn("ratios >= STEP_DT_MIN_RATIOS && m_fStepRatioMedianS > 0 && "
                      "m_fStepRatioAgreeFrac >= STEP_DT_AGREE_FRAC", clocks)
        self.assertIn("m_aStepRatios.Insert(timeSlice / coveredSteps);",
                      body(self.session, "protected void NoteStepRatio("))
        self.assertIn("NoteStepRatio(timeSlice, GetFrameFixedSteps());",
                      body(self.session, "override void EOnFixedFrame("))
        self.assertIn("NoteStepRatio(timeSlice, 1);", body(self.session, "override void EOnSimulate("))
        consts = dict(re.findall(r"static const (?:int|float) (STEP_DT_\w+) = ([\d.]+);", self.session))
        self.assertEqual((consts["STEP_DT_AGREEMENT"], consts["STEP_DT_AGREE_FRAC"], consts["STEP_DT_MIN_RATIOS"]),
                         ("0.005", "0.9", "30"))
        session_json = body(self.session, "protected bool WriteSessionJson(")
        for key in ("readiness_step_ratio_median_s", "readiness_slice_ratio_median_s", "readiness_measured_step_s"):
            self.assertIn('"%s"' % key, session_json)

        def engine(frames_ms, step_ms=1000.0 / 30, cap=3):
            """(frame time slices [s], fixed steps per frame, callback slices [s]) of an accumulator engine."""
            acc, slices, steps, cb = 0.0, [], [], []
            for ms in frames_ms:
                acc += ms
                n = min(cap, int(acc // step_ms))
                acc -= n * step_ms
                slices.append(ms / 1000.0)
                steps.append(n)
                if n:
                    cb.append(n * step_ms / 1000.0)
            return slices, steps, cb

        def old_rule(slices, steps, cb_slices):
            measured = sum(slices) / sum(steps)
            one = [s for s, n in zip(cb_slices, [n for n in steps if n]) if n == 1]
            if one and abs(one[-1] - measured) <= 0.05 * measured:
                return one[-1], "engine"
            return measured, "measured"

        def new_rule(slices, steps, cb_slices, min_ratios=30, agreement=0.005, agree_frac=0.9):
            ratios = sorted(s / n for s, n in zip(cb_slices, [n for n in steps if n]))
            span_s = span_steps = 0
            seen, total_s, after = False, 0.0, 0
            for s, n in zip(slices, steps):
                if not seen:
                    seen = n > 0
                    continue
                total_s += s
                after += n
                if n:
                    span_s, span_steps = total_s, after
            measured = span_s / span_steps if span_steps else 0.0
            if ratios:
                mid = len(ratios) // 2
                median = ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
                agree = sum(1 for r in ratios if abs(r - median) <= agreement * median) / float(len(ratios))
                if len(ratios) >= min_ratios and agree >= agree_frac:
                    return median, "engine"
            return measured, "measured"

        # Codex's case: 15 frames of 68 ms, 2 steps each (no one-step frame): the old rule measured 1.020 / 30 s
        slices, steps, cb = engine([68.0] * 15)
        self.assertEqual(steps, [2] * 15)
        dt, src = old_rule(slices, steps, cb)
        self.assertEqual(src, "measured")
        self.assertAlmostEqual(dt * 30, 1.020, places=9)
        dt, src = new_rule(slices, steps, cb, min_ratios=15)
        self.assertEqual(src, "engine")
        self.assertAlmostEqual(dt, 1.0 / 30, places=12)
        # capture frames merging 1-3 steps, 60 FPS idle frames: the ratio is the step on every frame
        frames = ([16.7] * 40 + [27.0, 80.0, 25.0, 70.0] * 15) * 2
        slices, steps, cb = engine(frames)
        self.assertGreater(sum(1 for n in steps if n > 1), 20)
        dt, src = new_rule(slices, steps, cb)
        self.assertEqual(src, "engine")
        self.assertAlmostEqual(dt, 1.0 / 30, places=12)
        # disagreeing ratios fall back to the measured span, which leaves out the remainder before the first step
        noisy = [s * f for s, f in zip(cb, [1.0, 1.1, 0.9, 1.05] * len(cb))]
        dt, src = new_rule(slices, steps, noisy)
        self.assertEqual(src, "measured")
        self.assertLess(abs(dt * 30 - 1.0), 1.0 / sum(steps[1:]) + 1e-9)

    def test_review_round1_capture_rules(self):
        """X6 (a trial conf at 5 Hz is the default setting), C-M4 (near plane in session.json and rig_id), C-L2
        (the held img rows survive a teardown seal), C-L3 (the Shape carrier's layout in observed), X7 (a kept
        command is numbered again)."""
        rates = body(self.session, "protected bool CheckRates(")
        self.assertRegex(rates, r"if \(isDefault\)\s*\{\s*m_fImageHz = DEFAULT_IMAGE_HZ;\s*"
                                r"m_sImageHzSetting = IMAGE_HZ_DEFAULT;")
        self.assertIn("m_sImageHzSetting = IMAGE_HZ_TRIAL_CONF;", rates)
        hf = source("DC_HfRecorder.c")
        self.assertIn('json.AddFloat("near_plane_m", GetNearPlaneRead(), 4);',
                      body(hf, "override void WriteSessionJson("))
        self.assertIn('"|n" + GetNearPlaneRead().ToString(-1, 3)', body(hf, "override string GetRigIdPart("))
        self.assertIn("return camera.GetNearPlane();", body(hf, "protected float GetNearPlaneRead("))
        seal = body(self.session, "protected void Seal(")
        self.assertIn("m_iImagesPending = m_ImageQueue.CloseAll(GetWorldMs());", seal)
        self.assertLess(seal.index("m_ImageQueue.CloseAll"), seal.index("CloseStreams(endReason);"))
        images = source("DC_ImageScheduler.c")
        self.assertIn("session.RegisterImageQueue(rowQueue);", body(images, "override bool OnSessionOpen("))
        self.assertIn("pending = m_Queue.CloseAll(m_Session.GetWorldMs());",
                      body(images, "override void OnSessionClosing("))
        self.assertNotIn("m_aRows", images[images.index("class DC_ImageScheduler : DC_SessionModule"):])
        layout = body(images, "protected void NoteShapeLayout(")
        for key in ("fiducial_shape_relayouts", "fiducial_shape_layout_failures"):
            self.assertIn('"%s"' % key, layout)
        self.assertIn('"fiducial_shape_drawing_at_end"', body(images, "override void OnSessionClosing("))
        # C-L4: the declared field is an int, not one of the observed strings
        end = body(self.session, "protected bool WriteSessionEnd(")
        self.assertIn('json.AddInt("decision_grid_points_skipped", m_iDecisionGridSkipped);', end)
        self.assertNotIn('NoteObserved("decision_grid_points_skipped"', self.session)
        decide = body(self.recorder, "override void OnStepDecide(")
        self.assertIn("m_Pending.CopyFrom(m_Held);", decide)
        self.assertIn("if (!m_bHasHeld)", decide)

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
