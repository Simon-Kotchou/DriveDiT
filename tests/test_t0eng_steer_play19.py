#!/usr/bin/env python3
"""Regression tests from the first 1.9.0.37 steering Play (DD-14e): small game-data excerpts of its e21 / e22 / e23
plays in tests/fixtures/steer_play19 (no launch ids, no paths; extracted by hand from the play directories).

    PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_t0eng_steer_play19.py

What that Play showed, one test group each:
  - e22: every phase skipped (no_site) under a footer that said complete -> no evidence (analysis and producer);
  - the S105's wheel-pivot bones: Animation.GetBoneIndex ids are name hashes, half of them negative; the probe took
    every negative id for a missing bone (rot_r, bone_l, bone_r 'missing 168 of 168');
  - e22's site: the runway is exactly flat within +-15 m, not at +-22 m; the new e22 geometry fits +-15 m;
  - e23: every doublet started +A, shifted the car the same way and every phase ended aborted_area at +-7 m; the
    new schedule (the translated DC_SteerTrainMath) completes the default grid on a car calibrated to that Play;
  - the zero-command readback deviations: engine-side (input_pre_steer = cmd, single-step dispatches, only under a
    hard brake, a teleport landing or a launch below 2 km/h, with assistance FULL and NONE); the CC-07 (a) check
    keeps them unexplained;
  - the site search: every refusal counted by reason, the zone-only rms and the zone share (translated
    DC_SteerTerrainMath), the e21 fallback keeps the held distance the analysis needs.
"""

import csv
import importlib.util
import math
import os
import random
import re
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "steer_play19")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ts = _load("test_t0eng_steer_helpers", os.path.join(REPO, "tests", "test_t0eng_steer.py"))
probe_common = ts.probe_common
steer_common = ts.steer_common
e21 = ts.e21
e23 = ts.e23
enforce_subset = ts.enforce_subset
read = ts.read
DT = 1.0 / 30.0


def fixture_kv(name):
    out = {}
    with open(os.path.join(FIX, name), "r", encoding="ascii") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if "=" in line:
                key, value = line.split("=", 1)
                out[key] = value
    return out


def fixture_rows(name):
    with open(os.path.join(FIX, name), "r", encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


def num(row, key):
    text = row.get(key, "")
    return float(text) if text not in ("", None) else None


def write_e22_play(play_dir):
    """The Play's e22: its phases file, its header-only steps file, its footer, its site and channel lines, with the
    fixtures' common identity (so it joins the synthetic e21 / e23 plays)"""
    header = ts.common_header("e22")
    for key, value in fixture_kv("e22_header.txt").items():
        if key not in ("probe",) and not key.startswith("params.dt_s"):
            header[key] = value
    header["site.flat.found"] = 0
    os.makedirs(play_dir, exist_ok=True)
    probe_common.write_synthetic_play(play_dir, header, {}, fixture_kv("e22_footer.txt"))
    for name in ("e22_phases.csv", "e22_steps.csv"):
        shutil.copyfile(os.path.join(FIX, name), os.path.join(play_dir, name))
    return play_dir


# ------------------------------------------------------------------------------------------------------------------
# the producer's train arithmetic, translated, and a car calibrated to the Play


def train_math():
    source = read(ts.STEPS_SOURCE)
    return enforce_subset.translate(source, ["DC_SteerTrainMath"])["DC_SteerTrainMath"]()


def steps_constant(name):
    match = re.search(r"static const (?:float|int) %s = ([0-9.]+);" % name, read(ts.STEPS_SOURCE))
    return float(match.group(1))


def attribute_default(source, member):
    match = re.search(r'\[Attribute\("([^"]*)"(?:"[^"]*"|[^"\]])*\]\s*protected \w+ %s;' % member, source)
    return match.group(1)


class PlanCar:
    """A kinematic S105 for the schedule: cmd -> steer input (first order 13 ms, CC-08 M2 on this Play), the wheel
    with a speed-dependent rate limit, kappa = 0.215 x wheel (e02a 15..50 km/h) through a 0.1 s lag; constant speed
    on the speed hold, per-event gain noise (the doublets' asymmetry). Calibrated to the Play's per-event heading
    changes within 20% (test_the_car_is_calibrated_to_the_play)"""

    K_SS = 0.215
    TAU_IN = 0.013
    TAU_K = 0.10

    def __init__(self, rng=None, noise=0.0):
        self.rng, self.noise = rng or random.Random(0), noise
        self.reset()

    @staticmethod
    def rate(kmh):
        return 8.0 / (1.0 + kmh / 10.0)

    def reset(self):
        self.x, self.y, self.psi, self.v = 6.0, 0.0, 0.0, 0.0
        self.inp = self.w = self.k = 0.0
        self.gain, self.last = 1.0, 0.0

    def step(self, cmd, v_target, accel=None):
        if cmd != self.last and cmd != 0.0:
            self.gain = 1.0 + self.rng.uniform(-self.noise, self.noise)
        self.last = cmd
        if accel is None:
            self.v = v_target
        elif accel > 0:
            self.v = min(v_target, self.v + accel * DT)
        else:
            self.v = max(0.0, self.v + accel * DT)
        self.inp += (1.0 - math.exp(-DT / self.TAU_IN)) * (cmd - self.inp)
        limit = self.rate(self.v * 3.6) * DT
        self.w += max(-limit, min(limit, self.inp - self.w))
        self.k += (1.0 - math.exp(-DT / self.TAU_K)) * (self.K_SS * self.gain * self.w - self.k)
        self.psi += self.k * self.v * DT
        self.x += self.v * math.cos(self.psi) * DT
        self.y += self.v * math.sin(self.psi) * DT


def pulse_heading_deg(kmh, amplitude, hold, gap=24):
    car = PlanCar()
    v = kmh / 3.6
    for i in range(hold + gap):
        car.step(amplitude if i < hold else 0.0, v)
    return math.degrees(car.psi)


class Lfsr:
    TAPS = {5: (5, 3), 10: (10, 7)}

    def __init__(self, order, seed):
        self.order, self.state = order, seed & ((1 << order) - 1)
        self.shifts = [order - t for t in self.TAPS[order]]

    def bit(self):
        out = self.state & 1
        feedback = 0
        for shift in self.shifts:
            feedback ^= (self.state >> shift) & 1
        self.state = (self.state >> 1) | (feedback << (self.order - 1))
        return out


def run_schedule(grid, new, seed=1, noise=0.05, max_runs=4):
    """The e23 train / PRBS state machine of DC_ProbeSteerSteps on PlanCar: reset at 6 m, settle, accel, cruise, the
    events (gap of 18 + 0..9 jitter steps, onset, hold, release), the room check, trims and doublet order (new), the
    PRBS, the base's guards (|lateral| > 7 m aborted_area, heading > 30 deg aborted_heading), stop. new: the translated
    DC_SteerTrainMath; old: every doublet +A first, holds cut by the heading budget only. Returns (seconds, phases,
    worst |lateral|, worst |heading|); a phase: (kmh, result, runs, events ok, prbs done)"""
    m = train_math()
    rng = random.Random(seed)
    kappa, heading_budget = steps_constant("KAPPA_PER_STEER"), steps_constant("HEADING_BUDGET_DEG")
    lateral_budget, trim_min = steps_constant("LATERAL_BUDGET_M"), steps_constant("TRIM_MIN_DEG")
    trim_amp, trim_max = steps_constant("TRIM_AMPLITUDE"), int(steps_constant("TRIM_MAX_STEPS"))
    on, off, jit, half, margin = 12, 18, 9, 10.0, 3.0
    prbs = Lfsr(5, 11)
    bits = [prbs.bit() for _ in range(31)]
    total, top, drift, top_drift = 0, 1, 0, 1
    for b in bits:
        total += 1 if b else -1
        top = max(top, abs(total))
        drift += total
        top_drift = max(top_drift, abs(drift))
    amps = {10.0: 0.5, 20.0: 0.4, 30.0: 0.3, 40.0: 0.25, 60.0: 0.2}
    jitter = Lfsr(10, 77)
    clock = [0.0]
    worst = [0.0, 0.0]
    phases = []
    for group in grid.split(";"):
        speed, specs = group.split(":")
        kmh, v = float(speed), float(speed) / 3.6
        counts = [[float(a), int(n)] for a, n in (spec.split("x") for spec in specs.split(","))]
        events = []
        while any(n > 0 for _, n in counts):
            for item in counts:
                if item[1] > 0:
                    events += [item[0], -item[0]]
                    item[1] -= 1
        amp = amps.get(kmh, 0.0)
        if new:
            bit_steps = m.PrbsBitSteps(amp, kmh, 5, top, top_drift, DT, kappa, heading_budget, lateral_budget)
        else:
            bit_steps = max(1, min(5, int(math.floor(math.radians(heading_budget) /
                                                     (kappa * amp * max(kmh, 1) / 3.6 * DT * top)))))
        car = PlanCar(rng, noise)
        cursor, runs, ok_events, prbs_done, result = 0, 0, 0, False, "ok"

        def drive(cmd, n):
            for _ in range(n):
                car.step(cmd, v)
                clock[0] += DT
                worst[0] = max(worst[0], abs(car.y))
                worst[1] = max(worst[1], abs(math.degrees(car.psi)))
                if abs(car.y) > half - margin or car.x > 1000.0 - margin or abs(math.degrees(car.psi)) > 30.0:
                    return False
            return True

        def gap():
            value = 0
            for _ in range(8):
                value = value * 2 + jitter.bit()
            return off + int(math.floor(value / 255.0 * jit + 0.5))

        def trim():
            off_deg = math.degrees(car.psi)
            if abs(off_deg) <= trim_min:
                return True
            n = m.TrimSteps(off_deg, kmh, DT, kappa, trim_amp, trim_max)
            return drive(float(m.TrimSteer(off_deg, 1, trim_amp)), n) and drive(0.0, gap())

        while (cursor < len(events) or not prbs_done) and runs < max_runs:
            runs += 1
            car.reset()
            clock[0] += 91 * DT
            while car.v < v - 1 / 3.6:
                car.step(0.0, v, accel=1.55)
                clock[0] += DT
            drive(0.0, 30)
            aborted = not drive(0.0, gap())
            while not aborted and cursor < len(events):
                if 1000.0 - margin - car.x < v * ((on + off + jit) * DT + 2.5) + v * v / 8.0:
                    break
                if new and cursor % 2 == 0:
                    if not trim():
                        aborted = True
                        break
                    if cursor + 1 < len(events):
                        sign = m.DoubletFirstSign(car.y, math.degrees(car.psi), kmh, (on + off) * DT, 1)
                        if events[cursor] * sign < 0:
                            events[cursor], events[cursor + 1] = events[cursor + 1], events[cursor]
                a = events[cursor]
                cursor += 1
                if new:
                    hold = m.HoldSteps(a, kmh, on, off, jit, DT, kappa, heading_budget, lateral_budget)
                else:
                    hold = max(1, min(on, int(math.floor(math.radians(heading_budget) /
                                                         (kappa * abs(a) * kmh / 3.6 * DT)))))
                if drive(a, hold) and drive(0.0, gap()):
                    ok_events += 1
                else:
                    aborted = True
            if not aborted and cursor >= len(events) and not prbs_done:
                if new and not trim():
                    aborted = True
                elif 1000.0 - margin - car.x >= v * (31 * bit_steps * DT + 2.5) + v * v / 8.0:
                    prbs_done = all(drive(amp if b else -amp, bit_steps) for b in bits)
                    aborted = not prbs_done
            if aborted:
                result = "aborted_area"
            while car.v > 0.3:
                car.step(0.0, 0.0, accel=-10.0)
                clock[0] += DT
        phases.append((kmh, result, runs, ok_events, prbs_done, len(events)))
    return clock[0], phases, worst[0], worst[1]


DEFAULT_GRID = attribute_default(read(ts.STEPS_SOURCE), "m_sStepGrid")
FULL_GRID = "10:0.5x50,1.0x50;20:0.5x4;30:0.5x50,1.0x50;40:0.5x4;60:0.5x50,1.0x50"


class FirstPlay19Tests(unittest.TestCase):
    def setUp(self):
        self.base = probe_common.make_temp_dir("t0eng_steer19_")
        self.addCleanup(shutil.rmtree, self.base, True)

    def path(self, *parts):
        return os.path.join(self.base, *parts)

    # --- e22: every phase skipped under a complete footer ------------------------------------------------------
    def test_an_e22_whose_every_phase_was_skipped_is_no_evidence(self):
        phases = fixture_rows("e22_phases.csv")
        self.assertEqual([p["result"] for p in phases], ["skipped"] * 3)
        self.assertEqual(fixture_kv("e22_footer.txt")["status"], "complete")
        load = write_e22_play(self.path("e22", "play_01"))
        ok, why = steer_common.play_evidence(probe_common.Play(load), "e22")
        self.assertIsNone(ok)
        self.assertIn("footer complete but no phase ended ok (skipped 3)", why)
        play = ts.build_e21(self.path("e21", "play_01"))
        code, text = ts.run_main(e21, [play, "--e22", load])
        self.assertEqual(code, 0, text)
        for name in ("evidence", "step0_channel", "a_readback", "d_load"):
            self.assertEqual(ts.item(text, name), "INCOMPLETE", text)
        self.assertIn("no phase ended ok", ts.detail(text, "evidence"))
        self.assertIn("no rows (phase skipped)", ts.detail(text, "a_readback"))
        self.assertIn("RESULT CC-07: INCOMPLETE", text)
        steps = ts.build_e23(self.path("e23", "play_01"), grid="10:0.5x6")
        code, text = ts.run_main(e23, [steps, "--e22", load])
        self.assertEqual(ts.item(text, "evidence"), "INCOMPLETE", text)
        self.assertIn("e22: footer complete but no phase ended ok", ts.detail(text, "evidence"))

    def test_the_producer_ends_a_probe_without_an_ok_phase_as_stopped(self):
        base = read(ts.BASE_SOURCE)
        runner = read(os.path.join(ts.SOURCES, "DC_ProbeRunner.c"))
        self.assertIn('static const string REASON_NO_EVIDENCE = "no_evidence";', runner)
        self.assertIn("StopProbe(m_CurrentRun, DoneReason(m_CurrentRun));", runner)
        stop = runner[runner.index("protected void StopProbe("):]
        stop = stop[:stop.index("FinishProbe(state, status, reason, note);")]
        # only done is complete: no_evidence keeps the default status stopped
        self.assertRegex(stop, r'string status = STATUS_STOPPED;\s*if \(reason == REASON_DONE\)\s*status = '
                               r'STATUS_COMPLETE;')
        done = base[base.index("override string GetProbeDoneReason"):]
        self.assertRegex(done[:400], r"m_iPhasesOk <= 0\)\s*return DC_ProbeRunner\.REASON_NO_EVIDENCE;")
        record = base[base.index("protected void FinishPhaseRecord"):]
        self.assertRegex(record[:600], r'if \(phase\.m_sResult == "ok"\)\s*m_iPhasesOk\+\+;')
        self.assertNotIn("m_iPhasesOk++", base[base.index("protected void BeginPhase()"):
                                               base.index("protected void BeginRun(")])
        # and the analyses take such a footer as no evidence
        load = write_e22_play(self.path("e22", "play_01"))
        footer = fixture_kv("e22_footer.txt")
        footer.update({"status": "stopped", "reason": "no_evidence"})
        probe_common.write_synthetic_play(load, probe_common.Play(load).header, {}, footer)
        ok, why = steer_common.play_evidence(probe_common.Play(load), "e22")
        self.assertIsNone(ok)
        self.assertIn("footer status stopped reason no_evidence", why)

    # --- wheel-angle candidates ---------------------------------------------------------------------------------
    def test_bone_ids_are_name_hashes_and_negative_ones_are_bones(self):
        header = fixture_kv("e22_header.txt")
        ids = {k.split(".")[2]: int(v) for k, v in header.items() if k.endswith(".bone_index")}
        self.assertEqual(sorted(k for k, v in ids.items() if v < 0), ["bone_l", "bone_r", "rot_r"])
        self.assertGreater(ids["rot_l"], 0)
        self.assertNotIn(-1, ids.values())                              # GetBoneIndex's "no such bone"
        source = read(ts.CHANNEL_SOURCE)
        code = "\n".join(line.split("//", 1)[0] for line in source.split("\n"))
        self.assertIn("static const int NO_BONE = -1;", code)
        self.assertNotRegex(code, r"m_aBones\[source\]\s*(?:>=|<)\s*0")
        self.assertNotRegex(code, r"\bbone\s*(?:<|>=)\s*0")
        self.assertEqual(len(re.findall(r"m_aBones\[source\] != NO_BONE", code)), 2)    # AxlesSettled, Describe
        self.assertIn("if (bone == NO_BONE)", code)                                      # BoneYaw
        self.assertIn('".bone_found="', code)

    # --- e22's site -----------------------------------------------------------------------------------------
    def test_e22_fits_the_runway_band_that_is_flat(self):
        e21_site = fixture_kv("e21_site_header.txt")
        self.assertEqual(e21_site["site.flat.found"], "1")
        self.assertEqual(e21_site["site.flat.half_width_m"], "15.00")
        self.assertEqual(e21_site["site.flat.min_y"], e21_site["site.flat.max_y"])      # 39.000 at every sample
        e22_site = fixture_kv("e22_header.txt")
        self.assertEqual((e22_site["site.flat.crit.half_width_m"], e22_site["site.flat.search.first_terrain_reject"]),
                         ("22.00", "cross_all"))
        load = read(ts.LOAD_SOURCE)
        half = float(re.search(r"override float GetFlatHalfWidthM\(\)\s*\{\s*return ([0-9.]+);", load).group(1))
        self.assertLessEqual(half, float(e21_site["site.flat.half_width_m"]))
        corridor = half - 3.0
        cap = float(re.search(r"SPEED_CAP_KMH = ([0-9.]+);", load).group(1))
        layer = read(os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Probes_Arland_Layers",
                                  "default.layer"))
        stair = re.search(r'm_sStairGrid "([^"]+)"', layer).group(1)
        self.assertEqual(stair, attribute_default(load, "m_sStairGrid"))
        commands = [float(c) for c in re.search(r'm_sLoadSteers "([^"]+)"', layer).group(1).split(",")]
        self.assertEqual(sorted(c > 0 for c in commands), [False, True])

        def stair_excursion(levels, kmh=10.5, hold=3.0):
            v, psi, y, worst = kmh / 3.6, 0.0, 0.0, 0.0
            for level in levels:
                for _ in range(int(round(hold / DT))):
                    psi += PlanCar.K_SS * level * v * DT
                    y += v * math.sin(psi) * DT
                    worst = max(worst, abs(y))
            return worst

        levels = [float(x) for x in stair.split(":")[1].split(",")]
        old = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.05, -0.05, 0.15, -0.15, 0.25, -0.25]
        self.assertEqual(sorted(levels), sorted(old))
        self.assertLess(stair_excursion(levels) * 1.2, corridor)
        self.assertGreater(stair_excursion(old), corridor)              # the old order left a +-12 m corridor
        edge = float(re.search(r"m_fCircleEdgeM (\S+)", layer).group(1))
        for command in commands:
            diameter = 2.0 / (PlanCar.K_SS * abs(command))
            self.assertLess(-(half - edge) + 1.15 * diameter, corridor - 1.0, command)    # 15% more radius
            radius = 1.0 / (PlanCar.K_SS * abs(command))
            self.assertLess(math.sqrt(3.5 * radius) * 3.6 * 1.15, cap)  # a_lat 3.5 under the cap with 15% to adapt
        # the swept envelope stays inside what e23's sweep of this Play cleared (+-10 m + its 60 km/h envelope)
        v = cap / 3.6
        stop = v * v / (2 * 4.0) + v * 0.2
        e23_v = 60.0 / 3.6
        e23_lat = 10.0 + (e23_v * e23_v / 8.0 + e23_v * 0.2) * math.sin(math.radians(30))
        self.assertLess(half + stop, e23_lat)

    # --- e23: the doublet drift and the new schedule ----------------------------------------------------------
    def test_every_doublet_moved_the_car_the_same_way_to_the_area_edge(self):
        rows = [r for r in fixture_rows("e23_rows.csv") if r["phase"] == "steps_30" and r["run_idx"] == "1"]
        events = [e for e in fixture_rows("e23_events.csv") if e["phase"] == "steps_30" and e["run_idx"] == "1"]
        by_step = {int(r["physics_step"]): r for r in rows}
        self.assertTrue(all(float(e["amplitude"]) > 0 for e in events[0::2]))           # every doublet +A first
        shifts = []
        for first, second in zip(events[0::2], events[1::2]):
            a = by_step[int(first["onset_step"]) - 1]
            b = by_step[int(second["end_step"])]
            shifts.append(num(b, "lateral_m") - num(a, "lateral_m"))
        self.assertTrue(all(s > 0.5 for s in shifts), shifts)
        self.assertGreater(min(s for s, e in zip(shifts, events[0::2]) if abs(float(e["amplitude"])) == 0.5), 1.8)
        train = [r for r in rows if r["segment"] == "train"]
        self.assertEqual(train[-1]["in_area"], "0")
        self.assertGreaterEqual(num(train[-1], "lateral_m"), 7.0)
        self.assertEqual(events[-1]["result"], "aborted_area")
        self.assertLess(num(train[-1], "along_m"), 200.0)                # the strip length was never the limit
        phases = {p["phase"]: p["result"] for p in fixture_rows("e23_phases.csv")}
        self.assertEqual(phases, {"steps_10": "aborted_area", "steps_20": "aborted_area",
                                  "steps_30": "aborted_area", "steps_40": "aborted_area", "steps_60": "stopped"})

    def test_the_car_is_calibrated_to_the_play(self):
        rows = {int(r["physics_step"]): r for r in fixture_rows("e23_rows.csv")}
        measured = {}
        for e in fixture_rows("e23_events.csv"):
            if e["result"] != "ok" or not e["release_step"]:
                continue
            a, b = rows.get(int(e["onset_step"]) - 1), rows.get(int(e["end_step"]))
            key = (float(e["target_kmh"]), abs(float(e["amplitude"])), int(e["hold_steps"]))
            measured.setdefault(key, []).append(abs(num(b, "heading_deg") - num(a, "heading_deg")))
        self.assertGreaterEqual(len(measured), 6)
        for (kmh, amplitude, hold), values in sorted(measured.items()):
            observed = sorted(values)[len(values) // 2]
            model = pulse_heading_deg(kmh, amplitude, hold)
            self.assertLess(abs(model - observed) / observed, 0.20, (kmh, amplitude, hold, observed, model))

    def test_the_new_schedule_completes_the_default_grid_and_the_old_one_does_not(self):
        timeout = float(attribute_default(read(ts.STEPS_SOURCE), "m_fTimeoutS"))
        layer = read(os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Probes_Arland_Layers",
                                  "default.layer"))
        self.assertIn('m_sStepGrid "%s"' % DEFAULT_GRID, layer)
        self.assertIn("m_fTimeoutS %g" % timeout, layer)
        seconds, phases, lateral, heading = run_schedule(DEFAULT_GRID, new=False)
        self.assertTrue(any(result == "aborted_area" for _, result, _, _, _, _ in phases), phases)
        for seed in (1, 2, 3):
            seconds, phases, lateral, heading = run_schedule(DEFAULT_GRID, new=True, seed=seed)
            print("default grid, seed %d: %.0f s, |lateral| <= %.2f m, |heading| <= %.1f deg" % (
                seed, seconds, lateral, heading))
            for kmh, result, runs, ok, prbs, planned in phases:
                self.assertEqual((result, ok, prbs), ("ok", planned, True), (kmh, phases))
            self.assertLess(lateral, 6.0)
            self.assertLess(heading, 25.0)
            self.assertLess(seconds, 0.9 * timeout)
        # the full CC-08 grid: about 13 min, a Play of its own (the split the producer header proposes)
        seconds, phases, lateral, heading = run_schedule(FULL_GRID, new=True, max_runs=10)
        print("full CC-08 grid: %.0f s, runs %s" % (seconds, [p[2] for p in phases]))
        self.assertTrue(all(p[1] == "ok" and p[3] == p[5] and p[4] for p in phases), phases)
        self.assertGreater(seconds, 600.0)
        self.assertLess(seconds, 1000.0)
        self.assertLessEqual(sum(int(n) * 2 for n in re.findall(r"x(\d+)", FULL_GRID)),
                             steps_constant("MAX_EVENTS"))

    def test_the_train_math_budgets(self):
        m = train_math()
        kappa, budget = steps_constant("KAPPA_PER_STEER"), steps_constant("HEADING_BUDGET_DEG")
        lateral = steps_constant("LATERAL_BUDGET_M")
        holds = {(kmh, a): m.HoldSteps(a, kmh, 12, 18, 9, DT, kappa, budget, lateral)
                 for kmh in (10.0, 20.0, 30.0, 40.0, 60.0) for a in (0.5, 1.0)}
        self.assertEqual(holds, {(10.0, 0.5): 12, (10.0, 1.0): 12, (20.0, 0.5): 12, (20.0, 1.0): 7,
                                 (30.0, 0.5): 7, (30.0, 1.0): 4, (40.0, 0.5): 4, (40.0, 1.0): 2,
                                 (60.0, 0.5): 2, (60.0, 1.0): 1})
        for (kmh, a), hold in holds.items():
            v = kmh / 3.6
            change = kappa * a * v * hold * DT
            self.assertLessEqual(math.degrees(change), budget + 1e-6)
            self.assertLessEqual(v * change * (hold + 18 + 4.5) * DT, lateral + 1e-6)
        # the first step turns away from the side the car is (going to be) on; a negative steer sign mirrors it
        self.assertEqual(m.DoubletFirstSign(2.0, 0.0, 30.0, 1.0, 1), -1)
        self.assertEqual(m.DoubletFirstSign(-2.0, 0.0, 30.0, 1.0, 1), 1)
        self.assertEqual(m.DoubletFirstSign(-0.5, 5.0, 30.0, 1.0, 1), -1)     # heading right: going right
        self.assertEqual(m.DoubletFirstSign(2.0, 0.0, 30.0, 1.0, -1), 1)
        # a trim turns the heading back at kappa x amplitude per metre
        steps = m.TrimSteps(3.0, 30.0, DT, kappa, 0.1, 45)
        self.assertAlmostEqual(steps * kappa * 0.1 * 30.0 / 3.6 * DT, math.radians(3.0), delta=kappa * 0.1 * 8.4 * DT)
        self.assertEqual(m.TrimSteps(90.0, 10.0, DT, kappa, 0.1, 45), 45)
        self.assertEqual(float(m.TrimSteer(3.0, 1, 0.1)), -float(enforce_subset.f32(0.1)))
        self.assertEqual(float(m.TrimSteer(-3.0, 1, 0.1)), float(enforce_subset.f32(0.1)))
        self.assertEqual(float(m.TrimSteer(3.0, -1, 0.1)), float(enforce_subset.f32(0.1)))

    def test_trims_stay_outside_every_event_window(self):
        steps = read(ts.STEPS_SOURCE)
        base = read(ts.BASE_SOURCE)
        self.assertIn('case DC_ESteerSegment.TRIM: return "trim";', base)
        train = steps[steps.index("protected void AdvanceTrain"):]
        train = train[:train.index("protected int HoldStepsFor")]
        # the open event is closed before a trim starts, and the next onset follows a full gap after it
        self.assertLess(train.index('CloseEvent(phase, "ok");'), train.index("StartTrimIfDue(phase)"))
        advance = steps[steps.index("override void AdvanceSegment"):]
        self.assertRegex(advance, r"DC_ESteerSegment\.TRIM\)\s*\{\s*if \(m_iSegStep >= m_iTrimSteps\)\s*"
                                  r"StartTrain\(\);")
        off = float(attribute_default(steps, "m_fStepOffS"))
        self.assertGreater(off / DT, e23.D_MAX)                          # the gap covers the analysis's history
        command = steps[steps.index("override void CommandSegment"):]
        self.assertRegex(command, r"DC_ESteerSegment\.TRIM\)\s*\{\s*m_bSegFlag = false;\s*m_iEventIdx = -1;")

    # --- the zero-command readback deviations --------------------------------------------------------------------
    def test_zero_command_readback_deviations_are_engine_side_and_stay_unexplained(self):
        rows = fixture_rows("e21_zero_rows.csv")
        devs = [r for r in rows if abs(num(r, "input_post_steer")) > 1e-3]
        self.assertGreaterEqual(len(devs), 8)
        for r in devs:
            self.assertEqual(num(r, "cmd_steer"), 0.0)
            self.assertEqual(num(r, "input_pre_steer"), 0.0)          # the setter got exactly the command
            self.assertAlmostEqual(num(r, "timeslice_s"), DT, places=5)  # one engine step: no merged dispatch
            hard = abs(num(r, "acc_l_z")) > 8.0
            self.assertTrue(r["segment"] in ("stop", "settle") or num(r, "speed_kmh") < 2.0, r)
            self.assertTrue(hard or r["segment"] == "settle" or num(r, "speed_kmh") < 2.0, r)
        self.assertEqual({r["assist_mode"] for r in devs}, {"0", "2"})  # FULL and NONE alike: not the assistance
        # with a perfect CC-08 input model whose cells cover these speeds they stay unexplained (a FAIL, not a pass)
        steps = []
        for k, r in enumerate(rows):
            raw = {c: "" for c in steer_common.STEP_COLUMNS}
            raw.update(r)
            steps.append(steer_common.Step(raw, k))
        model = {"structure": "M2", "d": 0, "g": 1.0, "tau": 0.013, "rate": None}
        scope = {"cells": [(0.0, 60.0, 1.0)], "text": "test"}
        out = e21.model_residuals(model, scope, steps, DT, 1e-3, 2e-4)
        self.assertEqual(out["unexplained"] + out["no_state"], len(devs))
        self.assertGreater(out["unexplained"], 0)

    # --- the site search ------------------------------------------------------------------------------------
    def test_every_refusal_is_counted_and_the_first_one_of_any_candidate_kept(self):
        sites = read(ts.SITE_SOURCE)
        code = "\n".join(line.split("//", 1)[0] for line in sites.split("\n"))
        self.assertNotIn("m_bSamplingExplicit && m_sFirstReject.IsEmpty()", code)
        sampling = code[code.index("protected bool ContinueSampling"):code.index("protected void AddRanked")]
        self.assertEqual(sampling.count("CountReject("), 2)               # the pre-check and the full check
        for key in ('"first_terrain_reject="', '"first_explicit_reject="', '"rejects="', '"near_miss="',
                    '"grid_total="', '"grid_complete="'):
            self.assertIn(key, code)
        # the pre-check is the full check's own test on the centre tyre pair: it refuses nothing the full check
        # would accept
        pre = code[code.index("protected bool ContinuePreCheck"):code.index("protected void CountReject")]
        self.assertIn("float cross = (m_fPreLeftY - y) / (2 * c.m_fTrackHalfM);", pre)
        self.assertIn("DC_SteerTerrainMath.ZoneCrossOk(c, m_Sampling.m_iSide, cross)", pre)
        self.assertIn("c.OffsetAt(3 + 2 * centre)", pre)
        self.assertIn("if (ZoneCrossOk(c, site.m_iSide, cross))", code)
        crit = ts.CriteriaStub("cross", half_width=8.0, corridor_half=5.0, corridor_points=ts.corridor_points(5.0))
        offsets = crit.offsets()
        centre = (crit.m_iCorridorPoints - 1) // 2
        self.assertEqual((round(offsets[3 + 2 * centre], 6), round(offsets[4 + 2 * centre], 6)), (-0.65, 0.65))
        # the strip cap covers the cross grid (3 strips per point) instead of ending it at 2591 of 3300 points
        cap = int(attribute_default(read(ts.BASE_SOURCE), "m_iSiteMaxStrips"))
        spacing = float(attribute_default(read(ts.SELF_SOURCE), "m_fCrossGridSpacingM"))
        points = (int((3550 - 600) // spacing) + 1) * (int((3520 - 780) // spacing) + 1)
        self.assertGreaterEqual(cap, 3 * 0.4 * points)                   # 39% of the points had a usable gradient
        cross = fixture_kv("e21_site_header.txt")
        self.assertEqual((cross["site.cross.search.grid_points"], cross["site.cross.search.strips"]), ("2591", "3000"))

    def test_zone_rms_and_zone_share(self):
        math_ = ts.translated_terrain_math()
        tests = ts.TerrainMathTests("test_flat_runway_passes")
        tests.math = math_
        crit = tests.cross_criteria()

        def curved(a, lat):                      # 6% to the right; the run-up bends 2 m, the stop 0.5 m off the plane
            out = 0.0
            if a < 75:
                out = 2.0 * ((75 - a) / 75.0) ** 2
            elif a > 275:
                out = 0.5 * ((a - 275) / 35.0) ** 2
            return 50.0 - 0.06 * lat + out

        site = tests.apply(ts.SiteStub(310, 8, 75, 275, side=1), crit, curved)
        self.assertTrue(site.m_bTerrainOk, site.m_sTerrainReason)
        self.assertLess(float(site.m_fRmsM), 0.01)
        self.assertAlmostEqual(float(site.m_fZoneShare), 1.0, places=6)
        inside = tests.apply(ts.SiteStub(310, 8, 75, 275, side=1), crit,
                             lambda a, lat: 50.0 - 0.06 * lat + 2.0 * ((a - 175) / 100.0) ** 2)
        self.assertFalse(inside.m_bTerrainOk)
        self.assertIn(inside.m_sTerrainReason, ("rms", "grade_window"))
        dip = tests.apply(ts.SiteStub(310, 8, 75, 275, side=1), crit,
                          lambda a, lat: 50.0 - (0.03 if 100 <= a <= 120 else 0.06) * lat)
        self.assertEqual(dip.m_sTerrainReason, "cross_zone")
        self.assertAlmostEqual(float(dip.m_fZoneShare), 1.0 - 5.0 / 41.0, places=4)
        for value in (0.049, 0.05, 0.08, 0.12, 0.121, -0.06):
            self.assertEqual(bool(math_.ZoneCrossOk(crit, 1, enforce_subset.f32(value))), 0.05 <= value <= 0.12)
        flat = ts.CriteriaStub("flat")
        self.assertTrue(math_.ZoneCrossOk(flat, 1, enforce_subset.f32(-0.009)))
        self.assertFalse(math_.ZoneCrossOk(flat, 1, enforce_subset.f32(0.011)))

    def test_the_cross_fallback_keeps_the_held_distance_the_analysis_needs(self):
        selfsteer = read(ts.SELF_SOURCE)
        self.assertIn("BuildCrossSearch(m_fCrossHeldM, m_fCrossFallbackM)", selfsteer)
        self.assertNotIn("phase.m_fHeldM = m_fCrossFallbackM;", selfsteer)
        layer = read(os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Probes_Arland_Layers",
                                  "default.layer"))
        held = float(re.search(r"m_fCrossHeldM (\S+)", layer).group(1))
        self.assertGreaterEqual(held, e21.MIN_CROSS_M)
        fallback = float(re.search(r"m_fCrossFallbackM (\S+)", layer).group(1))
        self.assertLess(fallback, float(re.search(r"m_fCrossNocallM (\S+)", layer).group(1)))
        # the full-length search's lines stay in the header when the fallback replaces it
        self.assertIn('full.AppendHeaderLines(lines, "site.cross_full");', selfsteer)


    # --- Codex review of PR #29: interrupted searches, the cap, the search time, trims in a history -----------
    def test_a_search_cut_short_still_writes_what_it_saw(self):
        base = read(ts.BASE_SOURCE)
        stop = base[base.index("override void OnProbeStop("):]
        stop = stop[:stop.index("m_Ctx = null;")]
        # every stop reason, before the context goes: the snapshot of every search
        self.assertRegex(stop, r"if \(m_eSegment == DC_ESteerSegment\.SEARCH && !m_bProbeDone\)\s*"
                               r"AppendSearchSnapshot\(reason\);")
        self.assertLess(stop.index("AppendSearchSnapshot(reason);"), stop.index("if (reason != DC_ProbeRunner.REASON_DELETE)"))
        snap = base[base.index("protected void AppendSearchSnapshot("):]
        snap = snap[:snap.index("protected bool RefuseStart(")]
        for needle in ('"site.search_stopped=" + reason', "search.NoteProbeStop(reason);",
                       'search.AppendHeaderLines(lines, "site." + kind);', "m_Ctx.AppendHeaderLines(lines)"):
            self.assertIn(needle, snap)
        sites = read(ts.SITE_SOURCE)
        header = sites[sites.index("void AppendHeaderLines(notnull array<string> lines, string prefix)\n\t{\n\t\t"
                                   "lines.Insert(prefix + \".kind=\""):]
        for key in ('"rejects="', '"first_terrain_reject="', '"first_explicit_reject="', '"near_miss="',
                    '"grid_total="', '"strips_capped="', '"grid_complete="', '"status="', '"stage="', '"pending="'):
            self.assertIn(key, header)
        status = sites[sites.index("string GetStatus()"):]
        self.assertRegex(status[:300], r"if \(!m_sStopReason\.IsEmpty\(\)\)\s*return m_sStopReason;")

    def test_grid_complete_needs_every_heading_under_the_cap(self):
        sites = read(ts.SITE_SOURCE)
        add = sites[sites.index("protected void AddCentredStrip("):]
        self.assertRegex(add[:300], r"if \(m_iStrips >= m_iMaxStrips\)\s*\{\s*m_iStripsCapped\+\+;\s*return;")
        complete = sites[sites.index("bool IsGridComplete()"):]
        self.assertIn("m_bGrid && m_bGridDone && m_iGridCursor >= total && m_iStripsCapped == 0 && "
                      "m_sStopReason.IsEmpty()", complete[:400])
        self.assertIn('"grid_complete=" + DC_LaunchGuard.BoolText(IsGridComplete())', sites)

    def test_the_search_time_is_capped_inside_e21s_timeout(self):
        base = read(ts.BASE_SOURCE)
        selfsteer = read(ts.SELF_SOURCE)
        cap = float(attribute_default(base, "m_fSiteSearchMaxS"))
        steps = base[base.index("protected void StepSearches()"):]
        steps = steps[:steps.index("OnSearchesDone()")]
        self.assertLess(steps.index('capped.StopSearch("time_cap");'), steps.index("m_Budget.Reset("))
        self.assertIn("now - m_fSearchStartMs > m_fSiteSearchMaxS * 1000", steps)
        self.assertRegex(selfsteer, r"if \(m_bSearchTimeCapped\)\s*return false;")
        timeout = float(re.search(r"TIMEOUT_E21_S = (\d+)", selfsteer).group(1))
        # the Play's four flat phases, measured; the two cross phases at the full lengths, estimated from the
        # distances (run-up + held + no-call), a 3 s settle, the accel at the Play's 1.55 m/s2 and the stop
        flat = sum(num(p, "end_world_ms") - num(p, "start_world_ms") for p in fixture_rows("e21_phases.csv")
                   if p["kind"] == "flat") / 1000.0
        layer = read(os.path.join(REPO, "addons", "DataCapture", "worlds", "DataCapture", "DC_Probes_Arland_Layers",
                                  "default.layer"))
        distance = sum(float(re.search(r"%s (\S+)" % key, layer).group(1))
                       for key in ("m_fCrossRunupM", "m_fCrossHeldM", "m_fCrossNocallM"))
        cross = sum(3.0 + (v / 3.6) / 1.55 + distance / (v / 3.6) + 1.0 for v in (20.0, 40.0))
        print("e21: search cap %.0f s + flat %.1f s + cross %.1f s = %.1f s of %.0f" % (
            cap, flat, cross, cap + flat + cross, timeout))
        self.assertLess(cap + flat + cross + 10.0, timeout)
        # the budget is per render frame: the Play's fallback cross search made 497 frames in about 9 s
        cross_header = fixture_kv("e21_site_header.txt")
        self.assertEqual(cross_header["site.cross.search.calls"], "497")
        self.assertIn("RENDER frames (the probe's OnProbeFrame, EOnFrame)", read(ts.SITE_SOURCE))

    def test_an_event_with_a_trim_in_its_history_is_left_out(self):
        play = ts.build_e23(self.path("e23", "play_01"), grid="10:0.5x6")
        steps_path = os.path.join(play, "e23_steps.csv")
        with open(steps_path, "r", encoding="ascii", newline="") as handle:
            rows = list(csv.reader(handle))
        events = fixture_rows_from(os.path.join(play, "e23_events.csv"))
        onset = int(events[2]["onset_step"])
        col_step, col_segment = rows[0].index("physics_step"), rows[0].index("segment")
        for row in rows[1:]:
            if onset - 4 <= int(row[col_step]) <= onset - 2:      # inside the D_MAX history of event 2
                row[col_segment] = "trim"
        with open(steps_path, "w", encoding="ascii", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        p = probe_common.Play(play)
        steps = steer_common.load_steps(p, "e23")
        bins, dropped, gaps = e23.event_windows(steps, events, "input", None, None, DT)
        # event 2 (the trim in its D_MAX history) and event 1 (the trim in its scored gap) are both left out
        self.assertEqual(dropped.get("trim_history"), 2, dropped)
        self.assertEqual(sum(len(w) for w in bins.values()), len(events) - 2)
        self.assertGreater(float(attribute_default(read(ts.STEPS_SOURCE), "m_fStepOffS")) / DT, e23.D_MAX)


def fixture_rows_from(path):
    with open(path, "r", encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
