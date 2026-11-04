"""DD-7-min's driver (llc_v1 and the minimal expert) against the frozen dc-2.0 capture core (DD-4-min).

The core (src/enfusion/datacapture/DC_StreamRecorder.c, DC_Session.c, DC_EgoSession.c) calls the driver once per
fixed-frame callback; a callback covers 1 to 3 engine steps of the measured step dt (freeze decision D-A, 1/30 s on
1.8), and decisions lie on the 200 ms world-time grid (DC_DriverContext.IsDecisionStep). These tests read the Enforce
sources (the statements the Python models mirror) and check, per adaptation of t0/freeze/FREEZE_IMPL.md section 5:
  - llc_v1 step time: every integrator advances by the callback's timeSlice (covered_steps x step dt), the clutch dip by
    covered steps; GetFixedTimeSlice() is never a step dt and a callback is never counted as one step.
"""

import math
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SOURCES = os.path.join(REPO, "src", "enfusion", "datacapture")
LLC_SOURCE = os.path.join(SOURCES, "DC_LLCv1.c")
EXPERT_SOURCE = os.path.join(SOURCES, "DC_ExpertHL.c")
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


if __name__ == "__main__":
    unittest.main()
