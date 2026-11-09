// ============================================================================
// DC_ProbeSteerSteps - probe e23: S2 steering steps and a held-out PRBS per speed (DD-14b; CC-08)
// ============================================================================
//
// MIGRATION_1_9.md 6.1 CC-08: steering steps 0 -> +-0.5 and 0 -> +1.0 at 10/30/60 km/h (+80 where the area allows),
// repeated with random phase, plus a held-out PRBS; competing actuator structures are fitted per speed and amplitude
// bin for the raw path (steer_cmd -> steer_input -> wheel_angle) and the kappa path. Every physics step records
// cmd_steer, input_pre/post_steer, the wheel-angle candidates (DC_SteerChannels) and the yaw rate.
//
// One phase per group of m_sStepGrid ('<km/h>:<amplitude>x<doublets>,...;...'), on the flat site (the runway):
//   reset on the centre line, settle, accel, hold (m_fCruiseS at the speed, steering 0), train, prbs, stop.
//   train  the events of the group: doublets of +A and -A for each amplitude, interleaved over the amplitudes (A1,
//          A1, A2, A2, A1, ...) until each amplitude has its count, so the heading comes back after every pair. The
//          sign order of each doublet is chosen at its first onset (DC_SteerTrainMath.DoubletFirstSign): a doublet
//          moves the car sideways by about v x its heading change x (hold + gap) to the side of its first step, so
//          the first step points away from the side the car is on (its lateral offset predicted one on + off ahead
//          from the heading). With every doublet +A first (the first 1.9.0.37 Play) each doublet moved the car the
//          same way: 0.7 m at 10 km/h x 1.0, 2.2 m at 30 km/h x 0.5, and every phase ended aborted_area at the
//          +-7 m corridor after 7..25 events. Each event is a gap of m_fStepOffS plus a jitter of 0 ..
//          m_fStepJitterS (random phase; 8 bits of a seeded order-10 LFSR, DC_ProbePrbs, per event, logged) at
//          steering 0, then m_fStepOnS at the amplitude (seg_flag 1), then the release to 0 (the next gap shows the
//          centering). event_idx marks the event from its onset to the next onset. An event starts only when the
//          site ahead holds its on and off time plus m_fTrainReserveS and the stop at m_fTrainDecelMs2; otherwise the
//          run stops (note area_end) and the next run (reset .. stop) continues with the next event, at most
//          m_iMaxRuns runs per phase;
//   trim   before each doublet (and before the PRBS) whose start finds the heading more than TRIM_MIN_DEG off the
//          strip (the doublets' small asymmetries add up, and a run resumed on a doublet's second event starts
//          turned): one pulse of +-TRIM_AMPLITUDE for the steps that turn the heading back
//          (DC_SteerTrainMath.TrimSteps), segment trim, event_idx -1, seg_flag 0, then the train's gap at steering 0
//          (its release window) before the next onset. A trim lies outside every event window (an event ends before
//          it, the next one's onset follows a full gap, longer than the analysis's D_MAX history);
//   prbs   after the events, the group's PRBS amplitude of m_sPrbsAmplitudes (0 = none): +-A from an order
//          m_iPrbsOrder LFSR (seed m_iPrbsSeed, the e01 generator), m_fPrbsBitS per bit, one period; it needs the same
//          room and gets its own run when the site ahead is too short.
// A spin (DC_ProbeSteerBase spin guard) or a run that leaves the area ends the run; the open event is written with
// that result and the next run continues; after m_iMaxSpins spins the phase ends with its remaining events unrun
// (note events_left). One row per event in e23_events.csv.
//
// BUDGET (defaults): 10 km/h 12 doublets of 0.5 and 8 of 1.0, 30 km/h 12 of 0.5 and 6 of 1.0, 60 km/h 10 of 0.5 and
// 6 of 1.0, about 3.5 min with m_fTimeoutS 270. This is PREVIEW coverage of CC-08 (the spec asks for 50 repetitions
// of each step); the analysis reports it as such. The full spec grid ('10:0.5x50,1.0x50;20:0.5x4;30:0.5x50,1.0x50;
// 40:0.5x4;60:0.5x50,1.0x50', m_iMaxRuns 10) takes about 13 min (10 km/h 4.5, 30 km/h 4, 60 km/h 4 min, 4 runs):
// a Play of its own with m_fTimeoutS about 1000, not the default plan. 80 km/h is a grid entry away ('80:0.3x6'). 20 and 40 km/h
// (4 doublets of 0.5 and a PRBS each) are outside the CC-08 grid: they make e21's speeds validated cells of the
// cmd -> steer_input model that CC-07 (a) may use to explain a readback deviation (no gap filling between speeds).
// HEADING AND LATERAL BUDGETS: every hold is cut to the HEADING_BUDGET_DEG budget (20 deg at 0.24 1/m per unit),
// under the base's 30 deg heading guard, and to the LATERAL_BUDGET_M budget (a doublet's sideways shift v x heading
// change x (hold + mean gap) <= 2 m, DC_SteerTrainMath.HoldSteps), inside the +-7 m corridor of the 10 m area:
// default holds 12 steps (10 km/h; 20 km/h x 0.5), 7 (20 km/h x 1.0; 30 km/h x 0.5), 4 (30 km/h x 1.0; 40 km/h x
// 0.5), 2 (40 km/h x 1.0; 60 km/h x 0.5), 1 (60 km/h x 1.0). PRBS bits are cut to the heading budget: 5 steps (4 at
// 60 km/h). The area (10 m, swept with the 60 km/h stopping envelope out to +-29 m) and the 30 deg heading guard are
// unchanged. tests/test_t0eng_steer.py drives the translated DC_SteerTrainMath on a car calibrated from that Play:
// the default grid ends every phase ok within about 7 m, the old schedule aborts.
// The analysis is tools/probes/t0eng/e23_steer_steps.py.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! The train's arithmetic, pure (tests/enforce_subset.py runs it on a vehicle model): hold steps under the heading
//! and lateral budgets, the doublet's first sign, the trim pulse
class DC_SteerTrainMath
{
	//------------------------------------------------------------------------------------------------
	//! Hold steps of a step of amplitude at kmh: at most onSteps, the heading change kappaPerSteer x |A| x v x hold
	//! within headingBudgetDeg, and the doublet's sideways shift v x that change x (hold + the mean gap of
	//! offSteps + jitterMaxSteps / 2) within lateralBudgetM; at least 1
	static int HoldSteps(float amplitude, float kmh, int onSteps, int offSteps, int jitterMaxSteps, float dt, float kappaPerSteer, float headingBudgetDeg, float lateralBudgetM)
	{
		float v = Math.Max(kmh, 1) / 3.6;
		float rate = kappaPerSteer * Math.AbsFloat(amplitude) * v;
		if (rate <= 0 || dt <= 0)
			return Math.MaxInt(1, onSteps);

		int heading = Math.Floor(headingBudgetDeg * Math.DEG2RAD / (rate * dt));
		float gap = (offSteps + 0.5 * jitterMaxSteps) * dt;
		float quota = lateralBudgetM / (v * rate);
		float holdS = 0.5 * (Math.Sqrt(gap * gap + 4 * quota) - gap);
		int lateral = Math.Floor(holdS / dt);
		return Math.MaxInt(1, Math.MinInt(onSteps, Math.MinInt(heading, lateral)));
	}

	//------------------------------------------------------------------------------------------------
	//! Steps per PRBS bit of amplitude at kmh: at most bitSteps, the heading within headingBudgetDeg at the
	//! sequence's largest |running sum| maxRun, and its sideways excursion v x rate x bit^2 x maxDrift (maxDrift: the
	//! largest |sum of the running sums| of the +-1 bits) within lateralBudgetM; at least 1
	static int PrbsBitSteps(float amplitude, float kmh, int bitSteps, int maxRun, int maxDrift, float dt, float kappaPerSteer, float headingBudgetDeg, float lateralBudgetM)
	{
		float v = Math.Max(kmh, 1) / 3.6;
		float rate = kappaPerSteer * Math.AbsFloat(amplitude) * v;
		if (rate <= 0 || dt <= 0)
			return Math.MaxInt(1, bitSteps);

		int heading = Math.Floor(headingBudgetDeg * Math.DEG2RAD / (rate * dt * Math.MaxInt(1, maxRun)));
		float bitS = Math.Sqrt(lateralBudgetM / (v * rate * Math.MaxInt(1, maxDrift)));
		int lateral = Math.Floor(bitS / dt);
		return Math.MaxInt(1, Math.MinInt(bitSteps, Math.MinInt(heading, lateral)));
	}

	//------------------------------------------------------------------------------------------------
	//! +1 or -1: the sign of a doublet's first step. A doublet shifts the car to the side of its first step's turn
	//! (steerSign: the turn direction of positive steering, +1 right), so the first step turns away from the side of
	//! the lateral offset predicted lookaheadS ahead from the heading off the strip (right positive)
	static int DoubletFirstSign(float lateralM, float headingOffDeg, float kmh, float lookaheadS, int steerSign)
	{
		float v = Math.Max(kmh, 0) / 3.6;
		float predicted = lateralM + v * Math.Sin(headingOffDeg * Math.DEG2RAD) * lookaheadS;
		int side = -1;
		if (predicted < 0)
			side = 1;

		if (steerSign < 0)
			return -side;

		return side;
	}

	//------------------------------------------------------------------------------------------------
	//! Steps of a trim pulse of amplitude that turns the heading back by |headingOffDeg| at kappaPerSteer, within
	//! 1 .. maxSteps
	static int TrimSteps(float headingOffDeg, float kmh, float dt, float kappaPerSteer, float amplitude, int maxSteps)
	{
		float v = Math.Max(kmh, 1) / 3.6;
		float perStep = kappaPerSteer * Math.AbsFloat(amplitude) * v * dt;
		if (perStep <= 0)
			return 1;

		int steps = Math.Round(Math.AbsFloat(headingOffDeg) * Math.DEG2RAD / perStep);
		return Math.MaxInt(1, Math.MinInt(maxSteps, steps));
	}

	//------------------------------------------------------------------------------------------------
	//! The trim command: amplitude against the heading off the strip (right positive) for the steer sign
	static float TrimSteer(float headingOffDeg, int steerSign, float amplitude)
	{
		float steer = Math.AbsFloat(amplitude);
		if (headingOffDeg > 0)
			steer = -steer;

		if (steerSign < 0)
			steer = -steer;

		return steer;
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "G19-1 probe e23: S2 steering steps 0 -> +-A with random phase and a held-out PRBS per speed (CC-08)")]
class DC_ProbeSteerStepsClass : DC_ProbeSteerBaseClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeSteerSteps : DC_ProbeSteerBase
{
	static const string PROBE_E23 = "e23";
	//! Caps checked before any event array is built
	static const int MAX_DOUBLETS = 100;		//!< per amplitude of one speed
	static const int MAX_EVENTS = 640;			//!< over the whole grid (the full CC-08 grid has 616)
	static const int MAX_SPEEDS = 6;
	//! Heading budget of one step or one PRBS excursion [deg]: under the base's 30 deg heading guard with margin.
	//! The hold of an event is cut to the steps that keep KAPPA_PER_STEER x |A| x v x hold under it; the PRBS bit to
	//! the steps that keep that x the sequence's largest running sum under it
	static const float HEADING_BUDGET_DEG = 20;
	//! Curvature per unit command assumed for the budget [1/m]: above the measured S105 steady value (e02a 1.8.0.13
	//! 0.229, 1.9.0.37 0.215 at 15..50 km/h)
	static const float KAPPA_PER_STEER = 0.24;
	//! A doublet's sideways shift budget [m] (DC_SteerTrainMath.HoldSteps), inside the +-7 m corridor
	static const float LATERAL_BUDGET_M = 2.0;
	//! Trim: a pulse when a doublet (or the PRBS) would start with the heading more than this off the strip [deg]
	static const float TRIM_MIN_DEG = 1;
	static const float TRIM_AMPLITUDE = 0.1;
	static const int TRIM_MAX_STEPS = 45;
	static const int JITTER_ORDER = 10;

	// === S2 ===
	[Attribute("10:0.5x12,1.0x8;20:0.5x4;30:0.5x12,1.0x6;40:0.5x4;60:0.5x10,1.0x6", UIWidgets.EditBox, "Step grid: groups '<speed km/h>:<amplitude>x<doublets>,...' separated by ';' (a doublet = 0 -> +A and 0 -> -A)", category: "S2")]
	protected string m_sStepGrid;

	[Attribute("0.4", UIWidgets.Slider, "Hold of each step at its amplitude [s]", "0.05 5 0.05", category: "S2")]
	protected float m_fStepOnS;

	[Attribute("0.6", UIWidgets.Slider, "Gap at steering 0 before each step (also the release / centering window) [s]", "0.1 10 0.05", category: "S2")]
	protected float m_fStepOffS;

	[Attribute("0.3", UIWidgets.Slider, "Random extra gap before each step, 0 .. this [s] (random phase)", "0 5 0.05", category: "S2")]
	protected float m_fStepJitterS;

	[Attribute("77", UIWidgets.Slider, "Seed of the jitter LFSR (order 10)", "1 1023 1", category: "S2")]
	protected int m_iStepSeed;

	[Attribute("1", UIWidgets.Slider, "Cruise at the speed before the train [s]", "0 10 0.5", category: "S2")]
	protected float m_fCruiseS;

	[Attribute("10:0.5;20:0.4;30:0.3;40:0.25;60:0.2", UIWidgets.EditBox, "Held-out PRBS amplitude per speed '<km/h>:<amplitude>;...' (missing or 0 = no PRBS at that speed)", category: "S2")]
	protected string m_sPrbsAmplitudes;

	[Attribute("5", UIWidgets.Slider, "PRBS LFSR order (5..10); one period = 2^order - 1 bits", "5 10 1", category: "S2")]
	protected int m_iPrbsOrder;

	[Attribute("0.15", UIWidgets.Slider, "PRBS bit length [s]", "0.03 2 0.01", category: "S2")]
	protected float m_fPrbsBitS;

	[Attribute("11", UIWidgets.Slider, "PRBS LFSR seed (non-zero modulo 2^order)", "1 1023 1", category: "S2")]
	protected int m_iPrbsSeed;

	[Attribute("2.5", UIWidgets.Slider, "Room kept ahead of an event or the PRBS, in seconds of travel at the current speed [s]", "0 20 0.5", category: "S2")]
	protected float m_fTrainReserveS;

	[Attribute("4", UIWidgets.Slider, "Deceleration assumed for the stop distance of the room check [m/s2]", "1 10 0.5", category: "S2")]
	protected float m_fTrainDecelMs2;

	[Attribute("4", UIWidgets.Slider, "Max runs (reset .. stop) per speed", "1 20 1", category: "S2")]
	protected int m_iMaxRuns;

	[Attribute("2", UIWidgets.Slider, "Spins after which a speed's remaining events are not run", "1 20 1", category: "S2")]
	protected int m_iMaxSpins;

	[Attribute("270", UIWidgets.Slider, "Probe timeout [world s]: 270 for the default (preview) grid, which takes about 210 s (e21 420 + e22 210 + e23 270 stay within 900 s); the full CC-08 grid needs about 800 s, in a Play of its own", "60 1200 10", category: "S2")]
	protected float m_fTimeoutS;

	// === STATE ===
	protected ref DC_ProbePrbs m_Jitter;
	protected ref DC_ProbePrbs m_Prbs;
	protected ref DC_ProbeCsv m_EventsCsv;
	protected int m_iOnSteps;
	protected int m_iOffSteps;
	protected int m_iJitterMaxSteps;
	protected int m_iCruiseSteps;
	protected int m_iPrbsBitSteps;
	protected int m_iPrbsPeriodBits;
	protected bool m_bOn;					//!< train: at the amplitude (else in the gap)
	protected int m_iStateEndStep;			//!< segment step at which the current gap / on state ends
	protected int m_iJitterSteps;			//!< of the event whose gap runs
	protected float m_fEventSteer;
	protected int m_iOpenEvent = -1;		//!< event with an onset whose row is not written yet
	protected int m_iOnsetStep = -1;
	protected int m_iReleaseStep = -1;
	protected bool m_bOnsetPending;
	protected bool m_bReleasePending;
	protected float m_fOnsetKmh;
	protected float m_fOnsetAlongM;
	protected int m_iOpenJitterSteps;
	protected int m_iPrbsBitsDrawn;
	protected bool m_bPrbsBitOpen;			//!< the drawn bit has not had its planned engine steps yet
	protected int m_iPrbsBitApplied;		//!< engine steps the open bit has been applied (covered steps)
	protected int m_iPrbsBitsApplied;		//!< bits applied for at least their planned engine steps
	protected int m_iPrbsExtended;			//!< bits applied for more steps than planned (a dispatch covered more)
	protected int m_iPrbsMaxRun;			//!< the largest |running sum| of +-1 over one PRBS period
	protected int m_iPrbsMaxDrift;			//!< the largest |sum of the running sums| (the PRBS's sideways excursion)
	protected int m_iHoldSteps;				//!< the open event's hold (HoldStepsFor)
	protected float m_fPrbsSteer;
	protected float m_fTrimSteer;			//!< the trim pulse's command
	protected int m_iTrimSteps;				//!< its engine steps
	protected bool m_bTrimDone;				//!< a trim ran in this gap (at most one before each doublet or the PRBS)
	protected int m_iTrims;					//!< trims over the probe (s2.trims)
	protected int m_iFlips;					//!< doublets started with -A first (s2.doublets_reordered)
	protected int m_iDoublets;				//!< doublet starts

	//------------------------------------------------------------------------------------------------
	override string GetSteerProbeId()
	{
		return PROBE_E23;
	}

	//------------------------------------------------------------------------------------------------
	override float GetSteerTimeoutS()
	{
		return Math.Max(m_fTimeoutS, 60);
	}

	//------------------------------------------------------------------------------------------------
	//! Straight runs with doublets: +-10 m of drift room
	override float GetFlatHalfWidthM()
	{
		return 10;
	}

	//------------------------------------------------------------------------------------------------
	//! The highest speed of the grid (60 by default)
	override float GetMaxSpeedKmh()
	{
		float top = 10;
		array<string> groups = {};
		m_sStepGrid.Split(";", groups, true);
		foreach (string groupText : groups)
		{
			array<string> parts = {};
			groupText.Split(":", parts, false);
			float speed;
			if (parts.Count() == 2 && ParseSteerFloat(parts[0], speed) && InRange(speed, 5, 100))
				top = Math.Max(top, speed);
		}

		return top;
	}

	//------------------------------------------------------------------------------------------------
	//! The timing attributes in range, 1..MAX_SPEEDS speed groups of 5..100 km/h, amplitudes in (0, 1], doublets
	//! 1..MAX_DOUBLETS per amplitude and at most MAX_EVENTS events in all, checked before any event is built
	override bool BuildSteerPlan(out string failure)
	{
		failure = string.Empty;
		bool timing = InRange(m_fStepOnS, 0.05, 5) && InRange(m_fStepOffS, 0.1, 10) && InRange(m_fStepJitterS, 0, 5) && InRange(m_fCruiseS, 0, 10) && InRange(m_fPrbsBitS, 0.03, 2) && InRange(m_fTrainReserveS, 0, 20) && InRange(m_fTrainDecelMs2, 1, 10) && m_iMaxRuns >= 1 && m_iMaxRuns <= 20 && m_iMaxSpins >= 1;
		if (!timing)
		{
			failure = "a step timing, PRBS, room or run-limit attribute is out of range";
			return false;
		}

		m_iOnSteps = Math.MaxInt(1, DC_ProbeSequences.StepsFor(m_fStepOnS, m_fDt));
		m_iOffSteps = Math.MaxInt(1, DC_ProbeSequences.StepsFor(m_fStepOffS, m_fDt));
		m_iJitterMaxSteps = DC_ProbeSequences.StepsFor(m_fStepJitterS, m_fDt);
		m_iCruiseSteps = DC_ProbeSequences.StepsFor(m_fCruiseS, m_fDt);
		m_iPrbsBitSteps = Math.MaxInt(1, DC_ProbeSequences.StepsFor(m_fPrbsBitS, m_fDt));
		m_Jitter = new DC_ProbePrbs();
		if (!m_Jitter.Init(JITTER_ORDER, m_iStepSeed))
		{
			failure = string.Format("jitter seed %1 not usable (order %2, non-zero modulo 2^order)", m_iStepSeed, JITTER_ORDER);
			return false;
		}

		m_Prbs = new DC_ProbePrbs();
		if (!m_Prbs.Init(m_iPrbsOrder, m_iPrbsSeed))
		{
			failure = string.Format("PRBS order %1 / seed %2 not supported (orders 5..10, seed non-zero modulo 2^order)", m_iPrbsOrder, m_iPrbsSeed);
			return false;
		}

		m_iPrbsPeriodBits = m_Prbs.GetPeriod();
		m_iPrbsMaxRun = 1;
		m_iPrbsMaxDrift = 1;
		int runningSum = 0;
		int drift = 0;
		for (int bit = 0; bit < m_iPrbsPeriodBits; bit++)
		{
			if (m_Prbs.NextBit() == 1)
				runningSum++;
			else
				runningSum--;

			m_iPrbsMaxRun = Math.MaxInt(m_iPrbsMaxRun, Math.AbsInt(runningSum));
			drift += runningSum;
			m_iPrbsMaxDrift = Math.MaxInt(m_iPrbsMaxDrift, Math.AbsInt(drift));
		}

		m_Prbs.Init(m_iPrbsOrder, m_iPrbsSeed);
		array<string> groups = {};
		m_sStepGrid.Split(";", groups, true);
		if (groups.IsEmpty() || groups.Count() > MAX_SPEEDS)
		{
			failure = string.Format("m_sStepGrid '%1' needs 1..%2 speed groups", m_sStepGrid, MAX_SPEEDS);
			return false;
		}

		int totalEvents = 0;
		foreach (string groupText : groups)
		{
			array<string> parts = {};
			groupText.Split(":", parts, false);
			float speed;
			if (parts.Count() != 2 || !ParseSteerFloat(parts[0], speed) || !InRange(speed, 5, 100))
			{
				failure = "m_sStepGrid '" + m_sStepGrid + "' is not '<speed>:<amplitude>x<doublets>,...;...'";
				return false;
			}

			array<float> amplitudes = {};
			array<int> counts = {};
			if (!ParseAmplitudes(parts[1], amplitudes, counts))
			{
				failure = string.Format("m_sStepGrid '%1': '%2' is not '<amplitude>x<doublets>,...' with amplitudes in (0, 1] and 1..%3 doublets", m_sStepGrid, parts[1], MAX_DOUBLETS);
				return false;
			}

			foreach (int doublets : counts)
			{
				totalEvents += 2 * doublets;
			}

			if (totalEvents > MAX_EVENTS)
			{
				failure = string.Format("m_sStepGrid '%1' has more than %2 events", m_sStepGrid, MAX_EVENTS);
				return false;
			}

			DC_ProbeSteerPhase phase = new DC_ProbeSteerPhase(DC_ESteerPhaseKind.STEPS, "steps_" + speed.ToString(-1, 0), SITE_FLAT, speed);
			phase.m_fStartAlongM = 2 * m_fAreaMarginM;
			phase.m_fLateralM = 0;
			BuildEvents(amplitudes, counts, phase.m_aEvents);
			float prbs = 0;
			if (!PrbsAmplitudeFor(speed, prbs))
			{
				failure = "m_sPrbsAmplitudes '" + m_sPrbsAmplitudes + "' is not '<km/h>:<amplitude>;...' with amplitudes in 0..1";
				return false;
			}

			phase.m_fPrbsAmplitude = prbs;
			phase.m_bPrbsDone = prbs <= 0;
			phase.m_iPrbsExpected = 0;
			if (prbs > 0)
				phase.m_iPrbsExpected = m_iPrbsPeriodBits;

			phase.m_iPrbsBits = 0;
			phase.m_iPrbsBitSteps = PrbsBitStepsFor(prbs, speed);
			m_aPhases.Insert(phase);
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! '<A>x<n>,...' -> amplitudes and doublet counts
	protected bool ParseAmplitudes(string text, notnull array<float> amplitudes, notnull array<int> counts)
	{
		array<string> tokens = {};
		text.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			array<string> pair = {};
			string trimmed = token.Trim();
			trimmed.Split("x", pair, false);
			float amplitude;
			float count;
			if (pair.Count() != 2 || !ParseSteerFloat(pair[0], amplitude) || !ParseSteerFloat(pair[1], count))
				return false;

			if (!InRange(amplitude, 0.01, 1) || !InRange(count, 1, MAX_DOUBLETS))
				return false;

			amplitudes.Insert(amplitude);
			int doublets = Math.Round(count);
			counts.Insert(doublets);
		}

		return !amplitudes.IsEmpty();
	}

	//------------------------------------------------------------------------------------------------
	//! Doublets interleaved over the amplitudes: A1+, A1-, A2+, A2-, A1+, ... until every count is used
	protected static void BuildEvents(notnull array<float> amplitudes, notnull array<int> counts, notnull array<float> outEvents)
	{
		array<int> left = {};
		foreach (int count : counts)
		{
			left.Insert(count);
		}

		bool any = true;
		while (any)
		{
			any = false;
			foreach (int i, float amplitude : amplitudes)
			{
				if (left[i] <= 0)
					continue;

				outEvents.Insert(amplitude);
				outEvents.Insert(-amplitude);
				left[i] = left[i] - 1;
				any = true;
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The PRBS amplitude of m_sPrbsAmplitudes for speed (0 when the speed is not listed); false when it does not parse
	protected bool PrbsAmplitudeFor(float speed, out float amplitude)
	{
		amplitude = 0;
		array<string> groups = {};
		m_sPrbsAmplitudes.Split(";", groups, true);
		foreach (string groupText : groups)
		{
			array<string> parts = {};
			groupText.Split(":", parts, false);
			float groupSpeed;
			float value;
			if (parts.Count() != 2 || !ParseSteerFloat(parts[0], groupSpeed) || !ParseSteerFloat(parts[1], value))
				return false;

			if (!InRange(value, 0, 1) || !InRange(groupSpeed, 5, 100))
				return false;

			if (Math.AbsFloat(groupSpeed - speed) < 0.01)
				amplitude = value;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	override bool OpenExtraCsvs(notnull DC_ProbeContext ctx)
	{
		array<string> columns = {"event_idx", "phase", "target_kmh", "amplitude", "run_idx", "jitter_steps", "onset_step", "release_step", "end_step", "onset_kmh", "onset_along_m", "hold_steps", "result"};
		m_EventsCsv = ctx.OpenCsv(PROBE_E23 + "_events", columns);
		return m_EventsCsv != null;
	}

	//------------------------------------------------------------------------------------------------
	override void CloseExtraCsvs()
	{
		if (m_EventsCsv)
			m_EventsCsv.Close();

		m_EventsCsv = null;
	}

	//------------------------------------------------------------------------------------------------
	override void OnBeginRun(notnull DC_ProbeSteerPhase phase)
	{
		m_bOn = false;
		m_bTrimDone = false;
		m_iOpenEvent = -1;
		m_bOnsetPending = false;
		m_bReleasePending = false;
		m_iPrbsBitsDrawn = 0;
		m_fPrbsSteer = 0;
	}

	//------------------------------------------------------------------------------------------------
	override DC_ESteerSegment AfterAccel(notnull DC_ProbeSteerPhase phase)
	{
		return DC_ESteerSegment.HOLD;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsSteeringSegment(DC_ESteerSegment segment)
	{
		return segment == DC_ESteerSegment.TRAIN || segment == DC_ESteerSegment.PRBS;
	}

	//------------------------------------------------------------------------------------------------
	//! hold: steering 0; train: the event's amplitude while on, else 0; prbs: +-A per bit. The onset and release steps
	//! of an event are the steps whose command is the first with the new value
	override void CommandSegment(notnull DC_ProbeSteerPhase phase)
	{
		m_Actuator.SetSteeringCall(true);
		m_fSegTarget = phase.m_fTargetKmh;
		if (m_eSegment == DC_ESteerSegment.TRAIN)
		{
			if (m_bOnsetPending)
			{
				m_iOnsetStep = m_iLastStep;
				m_bOnsetPending = false;
			}

			if (m_bReleasePending)
			{
				m_iReleaseStep = m_iLastStep;
				m_bReleasePending = false;
			}

			float steer = 0;
			if (m_bOn)
				steer = m_fEventSteer;

			m_bSegFlag = m_bOn;
			m_iEventIdx = m_iOpenEvent;
			m_Actuator.SetSpeedHold(phase.m_fTargetKmh, steer);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.PRBS)
		{
			// one bit at a time: the next bit is drawn only after the current one was applied for its planned engine
			// steps (AdvanceSegment counts the covered steps), so a dispatch covering several steps extends a bit and
			// never skips one
			if (!m_bPrbsBitOpen && m_iPrbsBitsDrawn < m_iPrbsPeriodBits)
			{
				if (m_Prbs.NextBit() == 1)
					m_fPrbsSteer = phase.m_fPrbsAmplitude;
				else
					m_fPrbsSteer = -phase.m_fPrbsAmplitude;

				m_iPrbsBitsDrawn++;
				m_bPrbsBitOpen = true;
				m_iPrbsBitApplied = 0;
			}

			m_bSegFlag = true;
			m_iEventIdx = -1;
			m_Actuator.SetSpeedHold(phase.m_fTargetKmh, m_fPrbsSteer);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.TRIM)
		{
			m_bSegFlag = false;
			m_iEventIdx = -1;
			m_Actuator.SetSpeedHold(phase.m_fTargetKmh, m_fTrimSteer);
			return;
		}

		m_bSegFlag = false;
		m_iEventIdx = -1;
		m_Actuator.SetSpeedHold(phase.m_fTargetKmh, 0);
	}

	//------------------------------------------------------------------------------------------------
	override void AdvanceSegment(notnull DC_ProbeSteerPhase phase)
	{
		if (m_eSegment == DC_ESteerSegment.HOLD)
		{
			if (m_iSegStep < m_iCruiseSteps)
				return;

			if (phase.m_iEventCursor < phase.m_aEvents.Count())
				StartTrain();
			else
				StartPrbsOrStop(phase);

			return;
		}

		if (m_eSegment == DC_ESteerSegment.TRAIN)
		{
			AdvanceTrain(phase);
			return;
		}

		// the trim pulse, then the train's gap at steering 0 (its release window) before the next onset or the PRBS
		if (m_eSegment == DC_ESteerSegment.TRIM)
		{
			if (m_iSegStep >= m_iTrimSteps)
				StartTrain();

			return;
		}

		if (m_eSegment == DC_ESteerSegment.PRBS)
		{
			// the dispatch just stepped applied the open bit for m_iLastCovered engine steps
			if (m_bPrbsBitOpen)
			{
				m_iPrbsBitApplied += m_iLastCovered;
				if (m_iPrbsBitApplied >= phase.m_iPrbsBitSteps)
				{
					if (m_iPrbsBitApplied > phase.m_iPrbsBitSteps)
						m_iPrbsExtended++;

					m_iPrbsBitsApplied++;
					m_bPrbsBitOpen = false;
				}
			}

			if (m_iPrbsBitsApplied >= m_iPrbsPeriodBits)
			{
				phase.m_bPrbsDone = true;
				phase.m_iPrbsBits = m_iPrbsBitsApplied;
				phase.m_iPrbsExtended = m_iPrbsExtended;
				phase.AddNote("prbs_run:" + phase.m_iRuns.ToString());
				EnterSegment(DC_ESteerSegment.STOP);
			}

			return;
		}

		EnterSegment(DC_ESteerSegment.STOP);
	}

	//------------------------------------------------------------------------------------------------
	//! Enters the train with the gap of the next event
	protected void StartTrain()
	{
		EnterSegment(DC_ESteerSegment.TRAIN);
		m_bOn = false;
		m_iJitterSteps = DrawJitterSteps();
		m_iStateEndStep = m_iOffSteps + m_iJitterSteps;
	}

	//------------------------------------------------------------------------------------------------
	//! The gap and on states of the train; event rows at the next onset, the train end or an aborted run
	protected void AdvanceTrain(notnull DC_ProbeSteerPhase phase)
	{
		if (m_iSegStep < m_iStateEndStep)
			return;

		if (m_bOn)
		{
			// on -> release: the gap of the next event starts now and is the centering window of this one
			m_bOn = false;
			m_bReleasePending = true;
			m_iJitterSteps = DrawJitterSteps();
			m_iStateEndStep = m_iSegStep + m_iOffSteps + m_iJitterSteps;
			return;
		}

		// gap over: close the open event; before a doublet (or the PRBS) a heading trim when it is due; then the next
		// onset if there is room
		CloseEvent(phase, "ok");
		if (StartTrimIfDue(phase))
			return;

		if (phase.m_iEventCursor >= phase.m_aEvents.Count())
		{
			StartPrbsOrStop(phase);
			return;
		}

		float v = Math.Max(m_Actuator.GetSpeedKmh() / 3.6, 0);
		float onOffS = (m_iOnSteps + m_iOffSteps + m_iJitterMaxSteps) * m_fDt;
		float need = v * (onOffS + m_fTrainReserveS) + v * v / (2 * m_fTrainDecelMs2);
		if (RemainingAlongM() < need)
		{
			phase.AddNote("area_end:run" + phase.m_iRuns.ToString() + "@event" + phase.m_iEventCursor.ToString());
			EnterSegment(DC_ESteerSegment.STOP);
			return;
		}

		OrderDoublet(phase);
		m_bTrimDone = false;
		m_iOpenEvent = phase.m_iEventCursor;
		m_fEventSteer = phase.m_aEvents[phase.m_iEventCursor];
		phase.m_iEventCursor++;
		m_iOpenJitterSteps = m_iJitterSteps;
		m_iOnsetStep = -1;
		m_iReleaseStep = -1;
		m_bOnsetPending = true;
		m_fOnsetKmh = m_Actuator.GetSpeedKmh();
		m_fOnsetAlongM = 0;
		if (m_Area)
			m_fOnsetAlongM = m_Area.AlongM(m_Actuator.GetPosition());

		m_bOn = true;
		m_iHoldSteps = HoldStepsFor(m_fEventSteer, Math.Max(m_Actuator.GetSpeedKmh(), phase.m_fTargetKmh));
		m_iStateEndStep = m_iSegStep + m_iHoldSteps;
	}

	//------------------------------------------------------------------------------------------------
	//! Hold steps of a step of amplitude at kmh: the configured on steps, cut so the heading change stays within
	//! HEADING_BUDGET_DEG and the doublet's sideways shift within LATERAL_BUDGET_M at KAPPA_PER_STEER (at least 1)
	protected int HoldStepsFor(float amplitude, float kmh)
	{
		return DC_SteerTrainMath.HoldSteps(amplitude, kmh, m_iOnSteps, m_iOffSteps, m_iJitterMaxSteps, m_fDt, KAPPA_PER_STEER, HEADING_BUDGET_DEG, LATERAL_BUDGET_M);
	}

	//------------------------------------------------------------------------------------------------
	//! At a doublet's first event (an even cursor with its partner of the opposite sign next): the two swapped when
	//! the first one would push the car further to the side it is on (DC_SteerTrainMath.DoubletFirstSign)
	protected void OrderDoublet(notnull DC_ProbeSteerPhase phase)
	{
		int cursor = phase.m_iEventCursor;
		if (cursor % 2 != 0 || cursor + 1 >= phase.m_aEvents.Count() || !m_Area)
			return;

		float first = phase.m_aEvents[cursor];
		float second = phase.m_aEvents[cursor + 1];
		if (first * second >= 0)
			return;

		m_iDoublets++;
		float lateral = m_Area.LateralM(m_Actuator.GetPosition());
		float headingOff = WrapDeg(m_Actuator.GetHeadingDeg() - m_Area.m_fHeadingDeg);
		float lookahead = (m_iOnSteps + m_iOffSteps) * m_fDt;
		int sign = DC_SteerTrainMath.DoubletFirstSign(lateral, headingOff, m_Actuator.GetSpeedKmh(), lookahead, GetSteerSign());
		if (first * sign > 0)
			return;

		phase.m_aEvents[cursor] = second;
		phase.m_aEvents[cursor + 1] = first;
		m_iFlips++;
	}

	//------------------------------------------------------------------------------------------------
	//! Before a doublet (an even cursor) or the PRBS (no event left), once per gap: a trim pulse when the heading is
	//! more than TRIM_MIN_DEG off the strip (segment trim; DC_SteerTrainMath.TrimSteps / TrimSteer). true when it
	//! started
	protected bool StartTrimIfDue(notnull DC_ProbeSteerPhase phase)
	{
		if (m_bTrimDone || !m_Area)
			return false;

		int left = phase.m_aEvents.Count() - phase.m_iEventCursor;
		if (left > 0 && phase.m_iEventCursor % 2 != 0)
			return false;

		if (left <= 0 && phase.m_bPrbsDone)
			return false;

		float headingOff = WrapDeg(m_Actuator.GetHeadingDeg() - m_Area.m_fHeadingDeg);
		if (Math.AbsFloat(headingOff) <= TRIM_MIN_DEG)
			return false;

		float kmh = Math.Max(m_Actuator.GetSpeedKmh(), phase.m_fTargetKmh);
		m_iTrimSteps = DC_SteerTrainMath.TrimSteps(headingOff, kmh, m_fDt, KAPPA_PER_STEER, TRIM_AMPLITUDE, TRIM_MAX_STEPS);
		m_fTrimSteer = DC_SteerTrainMath.TrimSteer(headingOff, GetSteerSign(), TRIM_AMPLITUDE);
		m_bTrimDone = true;
		m_iTrims++;
		EnterSegment(DC_ESteerSegment.TRIM);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! PRBS bit steps at amplitude and kmh: the configured bit, cut so the largest running sum of the sequence keeps
	//! the heading change within HEADING_BUDGET_DEG and its sideways excursion within LATERAL_BUDGET_M (at least 1)
	protected int PrbsBitStepsFor(float amplitude, float kmh)
	{
		return DC_SteerTrainMath.PrbsBitSteps(amplitude, kmh, m_iPrbsBitSteps, m_iPrbsMaxRun, m_iPrbsMaxDrift, m_fDt, KAPPA_PER_STEER, HEADING_BUDGET_DEG, LATERAL_BUDGET_M);
	}

	//------------------------------------------------------------------------------------------------
	//! The PRBS when it is due and there is room ahead, else the stop (the next run does it)
	protected void StartPrbsOrStop(notnull DC_ProbeSteerPhase phase)
	{
		if (phase.m_bPrbsDone)
		{
			EnterSegment(DC_ESteerSegment.STOP);
			return;
		}

		float v = Math.Max(m_Actuator.GetSpeedKmh() / 3.6, 0);
		float prbsS = m_iPrbsPeriodBits * phase.m_iPrbsBitSteps * m_fDt;
		float need = v * (prbsS + m_fTrainReserveS) + v * v / (2 * m_fTrainDecelMs2);
		if (RemainingAlongM() < need)
		{
			phase.AddNote("prbs_area_end:run" + phase.m_iRuns.ToString());
			EnterSegment(DC_ESteerSegment.STOP);
			return;
		}

		m_Prbs.Init(m_iPrbsOrder, m_iPrbsSeed);
		m_bTrimDone = false;
		m_iPrbsBitsDrawn = 0;
		m_bPrbsBitOpen = false;
		m_iPrbsBitApplied = 0;
		m_iPrbsBitsApplied = 0;
		m_iPrbsExtended = 0;
		EnterSegment(DC_ESteerSegment.PRBS);
	}

	//------------------------------------------------------------------------------------------------
	//! 0 .. m_iJitterMaxSteps from 8 LFSR bits
	protected int DrawJitterSteps()
	{
		if (m_iJitterMaxSteps <= 0)
			return 0;

		int value = 0;
		for (int bit = 0; bit < 8; bit++)
		{
			value = value * 2 + m_Jitter.NextBit();
		}

		float share = value / 255.0;
		int steps = Math.Round(share * m_iJitterMaxSteps);
		return steps;
	}

	//------------------------------------------------------------------------------------------------
	//! Writes the open event's row (if any) with result
	protected void CloseEvent(notnull DC_ProbeSteerPhase phase, string result)
	{
		if (m_iOpenEvent < 0)
			return;

		if (m_EventsCsv)
		{
			DC_ProbeRow row = new DC_ProbeRow();
			row.AddInt(m_iOpenEvent);
			row.AddString(phase.m_sName);
			row.AddFloat(phase.m_fTargetKmh, 3);
			row.AddFloat(phase.m_aEvents[m_iOpenEvent], 4);
			row.AddInt(phase.m_iRuns);
			row.AddInt(m_iOpenJitterSteps);
			AddIntOrEmpty(row, m_iOnsetStep);
			AddIntOrEmpty(row, m_iReleaseStep);
			row.AddInt(m_iLastStep);
			row.AddFloat(m_fOnsetKmh, 3);
			row.AddFloat(m_fOnsetAlongM, 3);
			row.AddInt(m_iHoldSteps);
			row.AddString(result);
			m_EventsCsv.WriteRow(row);
		}

		m_iOpenEvent = -1;
	}

	//------------------------------------------------------------------------------------------------
	override void OnRunEnded(notnull DC_ProbeSteerPhase phase, string result)
	{
		m_bOn = false;
		CloseEvent(phase, result);
		// an aborted PRBS: the bits it applied in full (the phases file shows it incomplete unless a later run
		// completes the whole period)
		if (m_eSegment == DC_ESteerSegment.PRBS && !phase.m_bPrbsDone)
		{
			phase.m_iPrbsBits = Math.MaxInt(phase.m_iPrbsBits, m_iPrbsBitsApplied);
			phase.m_iPrbsExtended = m_iPrbsExtended;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Another run while events or the PRBS remain, within m_iMaxRuns and m_iMaxSpins
	override bool WantsAnotherRun(notnull DC_ProbeSteerPhase phase)
	{
		CloseEvent(phase, "ok");
		int left = phase.m_aEvents.Count() - phase.m_iEventCursor;
		bool pending = left > 0 || !phase.m_bPrbsDone;
		if (!pending)
			return false;

		if (phase.m_iRuns >= m_iMaxRuns || phase.m_iSpins >= m_iMaxSpins)
		{
			phase.AddNote("events_left:" + left.ToString());
			if (!phase.m_bPrbsDone)
				phase.AddNote("prbs_not_run");

			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeParams(notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_sStepGrid=" + m_sStepGrid);
		outKeyValues.Insert("m_fStepOnS=" + m_fStepOnS.ToString(-1, 3));
		outKeyValues.Insert("m_fStepOffS=" + m_fStepOffS.ToString(-1, 3));
		outKeyValues.Insert("m_fStepJitterS=" + m_fStepJitterS.ToString(-1, 3));
		outKeyValues.Insert("m_iStepSeed=" + m_iStepSeed.ToString());
		outKeyValues.Insert("m_fCruiseS=" + m_fCruiseS.ToString(-1, 3));
		outKeyValues.Insert("m_sPrbsAmplitudes=" + m_sPrbsAmplitudes);
		outKeyValues.Insert("m_iPrbsOrder=" + m_iPrbsOrder.ToString());
		outKeyValues.Insert("m_fPrbsBitS=" + m_fPrbsBitS.ToString(-1, 3));
		outKeyValues.Insert("m_iPrbsSeed=" + m_iPrbsSeed.ToString());
		outKeyValues.Insert("m_fTrainReserveS=" + m_fTrainReserveS.ToString(-1, 3));
		outKeyValues.Insert("m_fTrainDecelMs2=" + m_fTrainDecelMs2.ToString(-1, 3));
		outKeyValues.Insert("m_iMaxRuns=" + m_iMaxRuns.ToString());
		outKeyValues.Insert("m_iMaxSpins=" + m_iMaxSpins.ToString());
		outKeyValues.Insert("m_fTimeoutS=" + m_fTimeoutS.ToString(-1, 1));
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeHeader(notnull array<string> lines)
	{
		lines.Insert("s2.on_steps=" + m_iOnSteps.ToString());
		lines.Insert("s2.off_steps=" + m_iOffSteps.ToString());
		lines.Insert("s2.jitter_max_steps=" + m_iJitterMaxSteps.ToString());
		lines.Insert("s2.prbs_bit_steps=" + m_iPrbsBitSteps.ToString());
		lines.Insert("s2.prbs_period_bits=" + m_iPrbsPeriodBits.ToString());
		lines.Insert("s2.prbs_max_running_sum=" + m_iPrbsMaxRun.ToString());
		lines.Insert("s2.prbs_max_drift=" + m_iPrbsMaxDrift.ToString());
		float budgetDeg = HEADING_BUDGET_DEG;
		float kappaPerSteer = KAPPA_PER_STEER;
		lines.Insert("s2.heading_budget_deg=" + budgetDeg.ToString(-1, 1) + " at kappa " + kappaPerSteer.ToString(-1, 3) + " 1/m per unit (holds and PRBS bits cut to it; the phases file has the bit steps per phase)");
		lines.Insert("s2.rule=train: per event a gap at 0 (off + jitter steps), then the amplitude for hold_steps (seg_flag 1), released into the next gap; event_idx from the onset to the next onset; e23_events.csv onset_step / release_step = the first step with the new command; prbs: seg_flag 1, event_idx -1, the e01 generator with this probe's order and seed, one period, every bit applied for at least prbs_bit_steps engine steps (a dispatch covering more extends the bit, prbs_extended counts them; no bit is skipped)");
		float lateralBudget = LATERAL_BUDGET_M;
		float trimMin = TRIM_MIN_DEG;
		float trimAmplitude = TRIM_AMPLITUDE;
		lines.Insert("s2.lateral_budget_m=" + lateralBudget.ToString(-1, 2) + " (a doublet's sideways shift v x heading change x (hold + mean gap); holds cut to it)");
		lines.Insert("s2.doublet_rule=events 2k and 2k + 1 are one doublet (opposite signs, one amplitude); its first step turns away from the side of the lateral offset predicted one on + off ahead (e23_events.csv amplitude is the applied sign)");
		lines.Insert("s2.trim_rule=segment trim, event_idx -1, seg_flag 0: before a doublet or the PRBS whose start finds the heading more than " + trimMin.ToString(-1, 2) + " deg off the strip, one pulse of +-" + trimAmplitude.ToString(-1, 2) + " turning it back, then a train gap at 0 before the next onset; outside every event window");
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeSummary(notnull array<string> lines)
	{
		lines.Insert("s2.trims=" + m_iTrims.ToString());
		lines.Insert("s2.doublets=" + m_iDoublets.ToString());
		lines.Insert("s2.doublets_reordered=" + m_iFlips.ToString());
	}
}
