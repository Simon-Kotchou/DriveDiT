"""DD-7-min's driver (llc_v1 and the minimal expert) against the frozen dc-2.0 capture core (DD-4-min).

The core (src/enfusion/datacapture/DC_StreamRecorder.c, DC_Session.c, DC_EgoSession.c) calls the driver once per
fixed-frame callback; a callback covers 1 to 3 engine steps of the measured step dt (freeze decision D-A, 1/30 s on
1.8), and decisions lie on the 200 ms world-time grid (DC_DriverContext.IsDecisionStep). These tests read the Enforce
sources (the statements the Python models mirror) and check, per adaptation of t0/freeze/FREEZE_IMPL.md section 5:
  - llc_v1 step time: every integrator advances by the callback's timeSlice (covered_steps x step dt), the clutch dip by
    covered steps; GetFixedTimeSlice() is never a step dt and a callback is never counted as one step;
  - decisions on the grid: the expert's decision is the evaluation of a decision callback's PostStep
    (ctx.IsDecisionStep), never keyed on physics_step mod N; exp_*_now is evaluated in every callback;
  - admission: DC_ExpertDriver.CanDecide is true only when this callback's PostStep will take the decision (the one
    CanEvaluate test both make), so an admitted episode's first Decide never returns false (which ends the session);
  - crash events: no expert rollover (or collision) termination, the stuck event carries stuck_s (the DD-5-min reader's
    check_payload accepts it against the contract's event_payloads), no ego SetWorldTransform without PrepareReposition
    and no spawned vehicle without DC_CrashWatch.WatchSlot.
"""

import importlib.util
import math
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SOURCES = os.path.join(REPO, "src", "enfusion", "datacapture")
LLC_SOURCE = os.path.join(SOURCES, "DC_LLCv1.c")
EXPERT_SOURCE = os.path.join(SOURCES, "DC_ExpertHL.c")
SESSION_SOURCE = os.path.join(SOURCES, "DC_Session.c")
EGO_SOURCE = os.path.join(SOURCES, "DC_EgoSession.c")
RECORDER_SOURCE = os.path.join(SOURCES, "DC_StreamRecorder.c")
DECISION_PERIOD_MS = 200                # 1000 / decision_hz at the frozen 5 Hz
LAYER = os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Arland_Layers", "default.layer")
DD7_SOURCES = (LLC_SOURCE, EXPERT_SOURCE)  # the Enforce files of DD-7-min
STEP_DT = 1.0 / 30.0                    # the measured step dt on 1.8 (dc-2.0 section 6, D-A)
RUN4_STEPS, RUN4_CALLBACKS = 18113, 10215   # T0-ENG run 4 E-10: steps and fixed-frame callbacks of one Play


def read(path):
    with open(path, "r", encoding="ascii", newline="") as f:
        return f.read()


def code_only(source):
    """The source without its // comments (a statement test must not match a comment)."""
    return "\n".join(line.split("//", 1)[0] for line in source.split("\n"))


def enforce_body(source, cls, method):
    """One method of an Enforce class, comments dropped and whitespace collapsed."""
    found = re.search(r"\nclass %s(?:\s*:\s*\w+)?\s*\n\{" % cls, source)
    body = source[found.end():]
    head = re.search(r"\n\t(?:(?:override|protected|static) )*[\w<>]+ %s\(" % method, body)
    body = body[head.start():].split("\n\t}\n", 1)[0]
    return " ".join(code_only(body).split())


class LlcStepModel:
    """Python mirror of the time-dependent statements of DC_LLCv1.Step and Longitudinal (rate limiter, lag, speed
    reference, speed integral, clutch dip), with the statements test_llc_step_takes_the_callback_slice asserts."""

    def __init__(self, rate_per_s, tau_s, dip_steps, step_dt=STEP_DT, window=100.0, kp=0.0, ki=0.0):
        self.rate, self.tau, self.dip_steps, self.step_dt = rate_per_s, tau_s, dip_steps, step_dt
        self.window, self.kp, self.ki = window, kp, ki
        self.limited = self.out = self.speed_ref = self.integral = 0.0
        self.dip_left, self.last_gear = 0, -1

    def step(self, steer_raw, accel, speed, gear, time_slice, covered):
        steps = covered if covered >= 1 else 1
        slice_s = time_slice if time_slice > 0 else self.step_dt * steps
        if self.rate > 0:
            max_delta = self.rate * slice_s
            self.limited += min(max(steer_raw - self.limited, -max_delta), max_delta)
        else:
            self.limited = steer_raw
        lag_alpha = 1.0 - math.exp(-slice_s / self.tau) if self.tau > 0 else 1.0
        self.out = min(max(self.out + (self.limited - self.out) * lag_alpha, -1.0), 1.0)
        self.speed_ref = min(max(self.speed_ref + accel * slice_s, max(0.0, speed - self.window)), speed + self.window)
        self.integral += (self.speed_ref - speed) * slice_s
        clutch = 1.0
        if gear != self.last_gear:
            self.dip_left = self.dip_steps
        self.last_gear = gear
        if self.dip_left > 0:
            clutch = 0.0
            self.dip_left -= steps
            if self.dip_left < 0:
                self.dip_left = 0
        return clutch


def merged_pattern(total_steps):
    """Callbacks covering 1, 2 and 3 steps (the D-A shapes of a capture session) up to total_steps."""
    pattern, covered = [], [1, 1, 2, 1, 3, 1, 2, 3, 1, 1]
    while sum(pattern) < total_steps:
        pattern.append(min(covered[len(pattern) % len(covered)], total_steps - sum(pattern)))
    return pattern


def capture_callbacks(seconds, fps=39.0, stall_every=25, stall_ms=80.0):
    """Fixed-frame callbacks of a capture session on the frame clock: (covered steps, physics_step, world_ms) per
    frame. Frames last 1000 / fps ms with a capture stall every stall_every frames (the D-A shape: 1 to 3 steps of
    STEP_DT per callback); world_ms is the frame's world time, read at the callback's top."""
    callbacks, t, frame, physics_step = [], 0.0, 0, -1
    while t < seconds * 1000.0:
        length = stall_ms if frame % stall_every == stall_every - 1 else 1000.0 / fps
        covered = int((t + length) / (STEP_DT * 1000.0)) - int(t / (STEP_DT * 1000.0))
        if covered > 0:                     # a frame without a step has no fixed-frame callback
            physics_step += covered
            callbacks.append((covered, physics_step, int(t)))
        t += length
        frame += 1
    return callbacks


class CoreGrid:
    """Mirror of DC_Session's decision grid (HandleStepPre / HandleStepPost, asserted in
    test_the_model_follows_the_core): a callback is a decision callback when its world time reached the next grid
    point; the grid then moves past it."""

    def __init__(self, period_ms=DECISION_PERIOD_MS):
        self.period, self.anchor, self.next_grid, self.skipped = period_ms, -1, 0, 0

    def is_decision(self, world_ms):
        if self.anchor < 0:
            self.anchor = world_ms
        return world_ms >= self.anchor + self.next_grid * self.period

    def after_post(self, world_ms, decision):
        if decision:
            reached = (world_ms - self.anchor) // self.period
            self.skipped += reached - self.next_grid
            self.next_grid = reached + 1


class ExpertDecisions:
    """Mirror of DC_ExpertDriver.PostStep / Decide: exp_*_now every callback, the decision only in a decision
    callback."""

    def __init__(self):
        self.now, self.decision, self.decision_step = None, None, -1

    def post_step(self, physics_step, evaluation, is_decision):
        self.now = evaluation
        if is_decision:
            self.decision = self.now
            self.decision_step = physics_step

    def decide(self, physics_step, is_decision):
        if not is_decision or self.decision_step != physics_step:
            return None
        return self.decision


class LlcStepTimeTest(unittest.TestCase):
    """(a) llc_v1 integrates over the callback's timeSlice, never over a fixed slice per call."""

    @classmethod
    def setUpClass(cls):
        cls.llc = read(LLC_SOURCE)
        cls.expert = read(EXPERT_SOURCE)

    def test_llc_step_takes_the_callback_slice(self):
        code = code_only(self.llc)
        self.assertIn("void Step(notnull DC_HLCommand held, notnull DC_LLCOutput llc, float timeSlice, "
                      "int coveredSteps)", code)
        for gone in ("m_fTimeSlice", "m_fLagAlpha"):        # the fixed slice of Init and its precomputed lag
            self.assertNotIn(gone, code)
        step = enforce_body(self.llc, "DC_LLCv1", "Step")
        for statement in ("int steps = coveredSteps; if (steps < 1) steps = 1;",
                          "float sliceS = timeSlice; if (sliceS <= 0) sliceS = m_fStepDtS * steps;",
                          "m_fKappaTrim + m_Card.m_fKappaTrimGain * kappaError * sliceS",
                          "float maxDelta = m_Card.m_fSteerRatePerS * sliceS;",
                          "if (m_Card.m_fSteerLagTauS > 0) lagAlpha = 1 - Math.Pow(Math.E, -sliceS / "
                          "m_Card.m_fSteerLagTauS);",
                          "m_fSteerOut = Math.Clamp(m_fSteerOut + (m_fSteerLimited - m_fSteerOut) * lagAlpha, -1, 1);",
                          "Longitudinal(held.m_fAccel, speed, sliceS, throttle, brake);",
                          "if (m_iDipStepsLeft > 0) { clutch = 0; m_iDipStepsLeft -= steps; if (m_iDipStepsLeft < 0) "
                          "m_iDipStepsLeft = 0; }"):
            self.assertIn(statement, step)
        self.assertNotIn("m_iDipStepsLeft--", step)           # a callback is not one step
        longitudinal = enforce_body(self.llc, "DC_LLCv1", "Longitudinal")
        self.assertIn("m_fSpeedRef = Math.Clamp(m_fSpeedRef + accel * sliceS,", longitudinal)
        self.assertIn("m_fIntegral += error * sliceS;", longitudinal)
        init = enforce_body(self.llc, "DC_LLCv1", "Init")
        self.assertIn("int dipSteps = Math.Round(card.m_fClutchDipS / m_fStepDtS);", init)

    def test_expert_passes_the_slice_and_the_covered_steps(self):
        apply_step = enforce_body(self.expert, "DC_ExpertDriver", "ApplyStep")
        self.assertIn("m_LLC.Step(held, llc, timeSlice, m_Context.GetCoveredSteps());", apply_step)
        begin = enforce_body(self.expert, "DC_ExpertDriver", "BeginDriver")
        self.assertIn("m_LLC.Init(m_Card, ctx.GetActuator(), ctx.GetStepDtS(), failure)", begin)
        for source in (self.llc, self.expert):
            code = code_only(source)
            # deprecated core getters: GetFixedTimeSliceS is the step dt only by alias, GetTelemetryEveryK is always 1,
            # GetStepsPerDecision is nominal (decisions lie on the world-time grid)
            for deprecated in ("GetFixedTimeSliceS", "GetTelemetryEveryK", "GetStepsPerDecision", "GetFixedTimeSlice("):
                self.assertNotIn(deprecated, code)

    def test_merged_callbacks_integrate_world_time(self):
        # one second of world time in callbacks of 1, 2 and 3 steps: the speed reference rises by accel x 1 s
        pattern = merged_pattern(30)
        model = LlcStepModel(rate_per_s=0.0, tau_s=0.0, dip_steps=0)
        for covered in pattern:
            model.step(0.0, 1.5, 0.0, 2, covered * STEP_DT, covered)
        self.assertAlmostEqual(model.speed_ref, 1.5 * 30 * STEP_DT, places=9)
        # the old rule (one fixed slice per call) integrated callbacks, not time: run 4's ratio of 10215 callbacks to
        # 18113 steps would have given 56 % of the commanded speed change
        old = 1.5 * len(pattern) * STEP_DT
        self.assertLess(old, 0.8 * model.speed_ref)
        self.assertAlmostEqual(RUN4_CALLBACKS / RUN4_STEPS, 0.564, places=3)

    def test_a_merged_callback_equals_its_single_steps(self):
        # the lag over n steps at a held input is the per-step lag applied n times; the rate limiter allows n steps'
        # worth
        for covered in (1, 2, 3):
            merged = LlcStepModel(rate_per_s=2.0, tau_s=0.08, dip_steps=0)
            single = LlcStepModel(rate_per_s=2.0, tau_s=0.08, dip_steps=0)
            merged.limited = single.limited = 0.5            # a target the limiter has reached: the lag alone acts
            merged.step(0.5, 0.0, 0.0, 2, covered * STEP_DT, covered)
            for _ in range(covered):
                single.step(0.5, 0.0, 0.0, 2, STEP_DT, 1)
            self.assertAlmostEqual(merged.out, single.out, places=12)
            limited = LlcStepModel(rate_per_s=2.0, tau_s=0.0, dip_steps=0)
            limited.step(1.0, 0.0, 0.0, 2, covered * STEP_DT, covered)
            self.assertAlmostEqual(limited.limited, 2.0 * covered * STEP_DT, places=12)

    def test_clutch_dip_counts_covered_steps(self):
        # a 0.2 s dip is 6 steps of 1/30 s: clutch 0 while any of them is left, so at least 6 and fewer than 6 + 3 steps
        for pattern in (merged_pattern(60), [1] * 60, [3] * 20, [2, 3] * 12):
            model = LlcStepModel(rate_per_s=0.0, tau_s=0.0, dip_steps=int(math.floor(0.2 / STEP_DT + 0.5)))
            clutches = [(covered, model.step(0.0, 0.0, 5.0, 3, covered * STEP_DT, covered)) for covered in pattern]
            dipped = sum(covered for covered, clutch in clutches if clutch == 0.0)
            self.assertGreaterEqual(dipped, 6, pattern)
            self.assertLess(dipped, 9, pattern)
        # a zero or missing slice falls back to covered x step dt (never 0 time)
        model = LlcStepModel(rate_per_s=0.0, tau_s=0.0, dip_steps=0)
        model.step(0.0, 3.0, 0.0, 2, 0.0, 2)
        self.assertAlmostEqual(model.speed_ref, 3.0 * 2 * STEP_DT, places=12)


class DecisionGridTest(unittest.TestCase):
    """(b) the expert decides only in decision callbacks of the world-time grid."""

    @classmethod
    def setUpClass(cls):
        cls.expert = read(EXPERT_SOURCE)
        cls.session = read(SESSION_SOURCE)

    def test_the_model_follows_the_core(self):
        pre = enforce_body(self.session, "DC_Session", "HandleStepPre")
        self.assertIn("if (m_iGridAnchorMs < 0) m_iGridAnchorMs = m_iCallbackWorldMs; m_bDecisionStep = "
                      "m_iCallbackWorldMs >= m_iGridAnchorMs + m_iNextDecisionGrid * m_iDecisionPeriodMs;", pre)
        post = enforce_body(self.session, "DC_Session", "HandleStepPost")
        self.assertIn("if (m_bDecisionStep) { int reached = (worldMs - m_iGridAnchorMs) / m_iDecisionPeriodMs; "
                      "m_iDecisionGridSkipped += reached - m_iNextDecisionGrid; m_iNextDecisionGrid = reached + 1; "
                      "m_bDecisionStep = false; }", post)
        # IsDecisionStep holds from OnStepPre through every OnStepDecide of the callback: the reset follows the deciders
        self.assertLess(post.index("decider.OnStepDecide(m_iPhysicsStep);"), post.index("m_bDecisionStep = false;"))

    def test_post_step_takes_the_decision_only_in_a_decision_callback(self):
        post_step = enforce_body(self.expert, "DC_ExpertDriver", "PostStep")
        self.assertIn("m_Expert.Evaluate(position, headingRad, speed, m_Now); if (m_Context.IsDecisionStep()) { "
                      "m_Decision.CopyFrom(m_Now); m_iDecisionStep = physicsStep; }", post_step)
        for field in ("m_fExpKappa = m_Now.m_fKappa;", "m_fExpAccel = m_Now.m_fAccel;", "m_iExpDir = m_Now.m_iDir;"):
            self.assertIn(field, post_step)                 # exp_*_now: every callback's own evaluation
        decide = enforce_body(self.expert, "DC_ExpertDriver", "Decide")
        self.assertIn("if (!m_bInEpisode || !m_Context.IsDecisionStep() || m_iDecisionStep != physicsStep) return "
                      "false; command.CopyFrom(m_Decision); return true;", decide)
        code = code_only(self.expert)
        self.assertIsNone(re.search(r"physicsStep\s*%|%\s*(?:N\b|m_i\w*Steps|\w*GetStepsPerDecision)", code))

    def test_one_decision_per_grid_callback_equal_to_its_exp_now(self):
        callbacks = capture_callbacks(30.0)
        grid, expert = CoreGrid(), ExpertDecisions()
        decided = []
        for index, (covered, physics_step, world_ms) in enumerate(callbacks):
            is_decision = grid.is_decision(world_ms)
            evaluation = ("exp_now", index)                 # stands for the expert on this callback's state
            expert.post_step(physics_step, evaluation, is_decision)
            if is_decision:                                 # the core calls Decide only here
                command = expert.decide(physics_step, is_decision)
                self.assertEqual(command, evaluation)       # mode E: action_cmd of decision t = exp_*_now at P_t
                decided.append(physics_step)
            else:
                self.assertIsNone(expert.decide(physics_step, is_decision))
            grid.after_post(world_ms, is_decision)
        # one decision per 200 ms of world time (nothing skipped at 39 FPS with 80 ms stalls)
        self.assertEqual(grid.skipped, 0)
        self.assertEqual(len(decided), callbacks[-1][2] // DECISION_PERIOD_MS + 1)
        # merged steps: decision intervals vary around N = 6 and physics_step mod 6 does not find the grid callbacks
        intervals = {b - a for a, b in zip(decided, decided[1:])}
        self.assertTrue(len(intervals) > 1 and min(intervals) <= 6 <= max(intervals), intervals)
        by_mod = [p for _, p, _ in callbacks if p % 6 == 0]
        self.assertNotEqual(by_mod, decided)


class AdmissionModel:
    """One decision callback of the core with a driver (DC_EgoSession.TryBeginEpisode, DC_StreamRecorder.OnStepPost
    and OnStepDecide, asserted in test_the_model_follows_the_core_admission): CanDecide, BeginEpisode, PostStep,
    Decide; a first Decide that gives nothing after an admission without an initial command fails the session."""

    def __init__(self, driver):
        self.driver, self.episode, self.failed, self.deferred, self.held = driver, False, False, 0, None

    def decision_callback(self, physics_step, evaluable):
        self.driver.evaluable = evaluable
        if not self.episode:
            if not self.driver.can_decide(physics_step, True):
                self.deferred += 1
                return
            self.episode = self.driver.begin_episode()
        if self.episode:
            self.driver.post_step(physics_step, True)
            command = self.driver.decide(physics_step, True)
            if command is None and self.held is None:
                self.failed = True
            elif command is not None:
                self.held = command


class ExpertDriverMirror:
    """DC_ExpertDriver's CanDecide / CanEvaluate / PostStep / Decide (asserted in test_can_decide_is_post_steps_test);
    with_handshake False is the base DC_DriverInterface.CanDecide (always true), as before this adaptation."""

    def __init__(self, with_handshake=True):
        self.with_handshake, self.evaluable, self.in_episode = with_handshake, True, False
        self.decisions = ExpertDecisions()

    def can_evaluate(self):
        return self.evaluable

    def can_decide(self, physics_step, is_decision):
        if not self.with_handshake:
            return True
        return is_decision and self.can_evaluate()

    def begin_episode(self):
        self.in_episode = True
        return True

    def post_step(self, physics_step, is_decision):
        if not self.in_episode or not self.can_evaluate():
            return
        self.decisions.post_step(physics_step, ("exp_now", physics_step), is_decision)

    def decide(self, physics_step, is_decision):
        if not self.in_episode:
            return None
        return self.decisions.decide(physics_step, is_decision)


class AdmissionTest(unittest.TestCase):
    """(c) the admission handshake: an admitted episode's first Decide never returns false."""

    @classmethod
    def setUpClass(cls):
        cls.expert = read(EXPERT_SOURCE)

    def test_the_model_follows_the_core_admission(self):
        begin = enforce_body(read(EGO_SOURCE), "DC_EgoSession", "TryBeginEpisode")
        self.assertLess(begin.index("if (!driver.CanDecide(physicsStep)) { m_iAdmissionsDeferred++;"),
                        begin.index("if (!driver.BeginEpisode(m_iNextEpisodeId, physicsStep, info)) return;"))
        decide = enforce_body(read(RECORDER_SOURCE), "DC_StreamRecorder", "OnStepDecide")
        self.assertIn("if (!driver.Decide(physicsStep, m_Pending)) {", decide)
        self.assertIn("if (m_bHasHeld) m_Pending.CopyFrom(m_Held); else if (m_bHasInitial) "
                      "m_Pending.CopyFrom(m_Initial); else { m_Session.Fail(DC_Session.END_SCRIPT_ERROR,", decide)

    def test_can_decide_is_post_steps_test(self):
        can_decide = enforce_body(self.expert, "DC_ExpertDriver", "CanDecide")
        self.assertIn("override bool CanDecide(int physicsStep) { if (!m_Context || !m_Context.IsDecisionStep()) "
                      "return false; return CanEvaluate();", can_decide)
        post_step = enforce_body(self.expert, "DC_ExpertDriver", "PostStep")
        self.assertIn("if (!m_bInEpisode || !CanEvaluate()) return;", post_step)
        self.assertEqual(code_only(self.expert).count("CanEvaluate()"), 3)    # the method and its two callers
        evaluate = enforce_body(self.expert, "DC_ExpertDriver", "CanEvaluate")
        self.assertIn("if (!m_bReady || !m_Context || !m_Expert || !m_Route) return false;", evaluate)
        self.assertIn("if (!m_Context.GetEgo() || !actuator || !actuator.GetSimulation()) return false; return true;",
                      evaluate)
        # nothing in PostStep between CanEvaluate and the decision can return early
        tail = post_step.split("if (!m_bInEpisode || !CanEvaluate()) return;", 1)[1]
        self.assertNotIn("return", tail.split("m_iDecisionStep = physicsStep;", 1)[0])

    def test_an_unevaluable_callback_defers_instead_of_failing(self):
        # the ego simulation is missing at the first grid point (evaluable False), there at the second
        core = AdmissionModel(ExpertDriverMirror())
        core.decision_callback(6, evaluable=False)
        self.assertEqual((core.deferred, core.episode, core.failed), (1, False, False))
        core.decision_callback(12, evaluable=True)
        self.assertEqual((core.deferred, core.episode, core.failed), (1, True, False))
        self.assertEqual(core.held, ("exp_now", 12))
        # the base CanDecide (always true) admits the same driver and its first Decide ends the session
        base = AdmissionModel(ExpertDriverMirror(with_handshake=False))
        base.decision_callback(6, evaluable=False)
        self.assertEqual((base.episode, base.failed), (True, True))


def load_reader():
    spec = importlib.util.spec_from_file_location("dc2_reader_dd7", os.path.join(REPO, "data", "dc2", "reader.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stuck_payload(world_ms, slow_since_ms):
    """Mirror of DC_ExpertDriver.StuckPayload (float.ToString(-1, 2): fixed point, 2 decimals)."""
    stuck_s = 0.0
    if slow_since_ms >= 0:
        stuck_s = max((world_ms - slow_since_ms) / 1000.0, 0.0)
    return "stuck_s=%.2f;termination_reason=stuck" % stuck_s


class CrashEventsTest(unittest.TestCase):
    """(e) the crash watch is the only rollover source; stuck_s; repositioning and traffic rules."""

    @classmethod
    def setUpClass(cls):
        cls.expert = read(EXPERT_SOURCE)

    def test_no_expert_rollover_or_collision(self):
        code = code_only(self.expert)
        for gone in ('"rollover"', '"collision"', "m_fRolloverDeg", "up[1]"):
            self.assertNotIn(gone, code)
        self.assertNotIn("m_fRolloverDeg", read(LAYER))
        check = enforce_body(self.expert, "DC_ExpertDriver", "CheckTermination")
        self.assertTrue(check.startswith("protected string CheckTermination(float speed, notnull "
                                         "VehicleWheeledSimulation simulation)"), check[:120])
        returned = set(re.findall(r'return "(\w+)";', check)) | {"stuck"}
        self.assertEqual(returned, {"route_complete", "route_deviation", "offroad", "engine_off", "stuck", "timeout"})

    def test_stuck_event_carries_stuck_s(self):
        post_step = enforce_body(self.expert, "DC_ExpertDriver", "PostStep")
        self.assertIn("if (m_sReported == TERMINATION_STUCK && !m_Context.LogEvent(TERMINATION_STUCK, StuckPayload()))",
                      post_step)
        payload = enforce_body(self.expert, "DC_ExpertDriver", "StuckPayload")
        self.assertIn("float stuckS = 0; if (m_iSlowSinceMs >= 0) { float stuckMs = m_Context.GetWorldMs() - "
                      "m_iSlowSinceMs; stuckS = Math.Max(stuckMs / 1000, 0); } return \"stuck_s=\" + "
                      "stuckS.ToString(-1, 2) + \";termination_reason=\" + TERMINATION_STUCK;", payload)
        reader = load_reader()
        payloads = reader.load_schema().payloads
        self.assertEqual([k["key"] for k in payloads["stuck"]], ["stuck_s", "termination_reason"])
        for world_ms, since in ((65432, 45400), (20000, 0), (100, -1)):
            text = stuck_payload(world_ms, since)
            values, problems = reader.check_payload("stuck", text, payloads)
            self.assertEqual(problems, [], text)
            self.assertEqual(values["termination_reason"], "stuck")
            self.assertGreaterEqual(values["stuck_s"], 0.0)
        self.assertEqual(stuck_payload(65432, 45400), "stuck_s=20.03;termination_reason=stuck")

    def test_no_reposition_without_prepare_and_no_unwatched_traffic(self):
        for path in DD7_SOURCES:
            code = code_only(read(path))
            # every SetWorldTransform / SetOrigin of an entity is preceded by ctx.PrepareReposition() in the same method
            for match in re.finditer(r"\.(SetWorldTransform|SetOrigin|SetTransform)\(", code):
                before = code[:match.start()].rsplit("\n\t//---", 1)[-1]
                self.assertIn("PrepareReposition()", before, "%s: %s" % (os.path.basename(path), match.group(1)))
            # a spawned vehicle (traffic) is watched by the crash watch
            for match in re.finditer(r"SpawnEntityPrefab\(|SpawnEntity\(", code):
                after = code[match.start():].split("\n\t//---", 1)[0]
                self.assertIn("DC_CrashWatch", after, os.path.basename(path))
        # today the driver moves and spawns nothing: the ego is the core's (spawned and parked by DC_EgoSession)
        for path in DD7_SOURCES:
            self.assertIsNone(re.search(r"SetWorldTransform|SetOrigin\(|SpawnEntityPrefab|SpawnEntity\(",
                                        code_only(read(path))))


if __name__ == "__main__":
    unittest.main()
