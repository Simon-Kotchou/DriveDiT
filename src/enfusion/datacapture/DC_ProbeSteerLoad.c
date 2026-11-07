// ============================================================================
// DC_ProbeSteerLoad - probe e22: CC-07 step 0 (wheel-angle channel validation) and CC-07 (d), applied angle under
// load (DD-14b)
// ============================================================================
//
// MIGRATION_1_9.md 6.1 CC-07:
//   step 0  wheel_angle_l/r checked against the path curvature in the low-slip regime (|a_lat| <= 1 m/s2, 10-20
//           km/h): delta_eff predicts kappa_realised within 5% median, with no lag beyond one physics step;
//   (d)     at fixed commands, steady cornering at a_lat 2.0, 2.75 and 3.5 m/s2 and braking at 3 m/s2 in a turn:
//           |delta_eff - delta_eff,ref(command)| p95 <= max(0.25 deg, 5% of |delta_eff,ref|), delta_eff,ref = the
//           step-0 low-load value.
// Phases on the flat site (the runway; DC_ProbeSteerBase):
//   stair_<v>    one per group of m_sStairGrid ('<km/h>:<steer>,...;...'): reset on the centre line, settle, accel,
//                then the steering levels one after another, m_fStairHoldS each, with the speed hold (segment
//                stair; event_idx = the level index, seg_flag 1 after the first m_fStairSettleS of each level), stop.
//                The defaults keep a_lat below 1 m/s2 at kappa ~0.21 x steer (1.9.0.37 e02a): 12 km/h up to 0.3,
//                18 km/h up to 0.15. The levels alternate in sign, so the heading comes back after each pair;
//   load_<c>     one per fixed command of m_sLoadSteers: reset near the side of the area away from the turn
//                (m_fCircleEdgeM inside the area edge, m_fCircleStartAlongM along), settle, accel straight to
//                m_fRefKmh, ref (the command c at m_fRefKmh for m_fRefHoldS: the low-load reference of (d), seg_flag
//                1 in its second half), alat (for each target of m_sALatTargets: the speed-hold target is adapted
//                every m_fALatAdjustS by sqrt(target / measured) within +-15% until the window mean of |v r| is
//                within m_fALatToleranceMs2, then held for m_fALatHoldS with seg_flag 1; at most m_fALatMaxS per
//                level; seg_target = the a_lat target, event_idx = the level), brake (the command c, throttle 0, a
//                brake PI on the smoothed acc_l_z to -m_fBrakeDecelMs2 until m_fBrakeEndKmh; seg_flag 1 within
//                0.5 m/s2 of the target), stop.
// The turn direction of positive steering comes from the stair phases of this probe (sum of cmd_steer x heading
// rate), else m_iAssumedSteerSign (+1: e01 measured right on both builds).
// The analysis is tools/probes/t0eng/e21_selfsteer.py (--e22 <this play>).
// ============================================================================

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "G19-1 probe e22: CC-07 step 0 low-load staircase and CC-07 (d) loaded cornering and braking in a turn at fixed commands")]
class DC_ProbeSteerLoadClass : DC_ProbeSteerBaseClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeSteerLoad : DC_ProbeSteerBase
{
	static const string PROBE_E22 = "e22";
	static const float TIMEOUT_E22_S = 600;
	//! Cap of the brake segment [s]
	static const float BRAKE_CAP_S = 10;
	//! Smoothing of acc_l_z for the brake PI (weight of the new sample)
	static const float BRAKE_ACC_ALPHA = 0.3;
	//! Speed of the alat levels [km/h]: bounds of the adapted target
	static const float ALAT_MIN_KMH = 5;
	static const float ALAT_MAX_KMH = 90;

	// === STEP 0 ===
	[Attribute("12:0.1,-0.1,0.2,-0.2,0.3,-0.3;18:0.05,-0.05,0.1,-0.1,0.15,-0.15", UIWidgets.EditBox, "Step-0 staircase: groups '<speed km/h>:<steer>,...' separated by ';' (low-slip regime: 10-20 km/h, a_lat <= 1 m/s2)", category: "Step 0")]
	protected string m_sStairGrid;

	[Attribute("3", UIWidgets.Slider, "Hold of each staircase level [s]", "0.5 20 0.5", category: "Step 0")]
	protected float m_fStairHoldS;

	[Attribute("1", UIWidgets.Slider, "Transient at the start of each level, not flagged steady [s]", "0 10 0.1", category: "Step 0")]
	protected float m_fStairSettleS;

	// === LOAD ===
	[Attribute("0.3,-0.3", UIWidgets.EditBox, "Fixed commands of CC-07 (d), one phase each (0.3: kappa ~0.064 1/m, R ~16 m, so the circles fit the 50 m runway)", category: "Load")]
	protected string m_sLoadSteers;

	[Attribute("10", UIWidgets.Slider, "Low-load reference speed of each command [km/h] (a_lat ~0.5 m/s2 at 0.3)", "3 30 0.5", category: "Load")]
	protected float m_fRefKmh;

	[Attribute("6", UIWidgets.Slider, "Hold of the reference [s]; its second half is flagged steady", "1 30 0.5", category: "Load")]
	protected float m_fRefHoldS;

	[Attribute("2.0,2.75,3.5", UIWidgets.EditBox, "Steady-cornering a_lat targets [m/s2] (CC-07 d)", category: "Load")]
	protected string m_sALatTargets;

	[Attribute("0.1", UIWidgets.Slider, "a_lat tolerance of a converged window [m/s2]", "0.01 1 0.01", category: "Load")]
	protected float m_fALatToleranceMs2;

	[Attribute("0.5", UIWidgets.Slider, "a_lat adaptation window [s]", "0.1 5 0.1", category: "Load")]
	protected float m_fALatAdjustS;

	[Attribute("4", UIWidgets.Slider, "Steady hold of each a_lat level once converged [s]", "0.5 30 0.5", category: "Load")]
	protected float m_fALatHoldS;

	[Attribute("15", UIWidgets.Slider, "Max time of one a_lat level [s] (then the next level, note alat_unconverged_<i>)", "2 60 1", category: "Load")]
	protected float m_fALatMaxS;

	[Attribute("3", UIWidgets.Slider, "Braking in the turn: decel target [m/s2] (CC-07 d)", "0.5 8 0.25", category: "Load")]
	protected float m_fBrakeDecelMs2;

	[Attribute("0.2", UIWidgets.Slider, "Brake PI: feed-forward brake (1.9.0.37 e01: brake 0.3 gave -4.4 m/s2 at 40 km/h)", "0 1 0.01", category: "Load")]
	protected float m_fBrakeFeedForward;

	[Attribute("0.03", UIWidgets.Slider, "Brake PI: proportional gain [1/(m/s2)]", "0 1 0.005", category: "Load")]
	protected float m_fBrakeKp;

	[Attribute("0.1", UIWidgets.Slider, "Brake PI: integral gain [1/(m/s2 s)]", "0 2 0.01", category: "Load")]
	protected float m_fBrakeKi;

	[Attribute("5", UIWidgets.Slider, "Braking in the turn ends below this speed [km/h]", "1 30 0.5", category: "Load")]
	protected float m_fBrakeEndKmh;

	[Attribute("40", UIWidgets.Slider, "Circle runs start this far along the site [m]", "5 500 1", category: "Load")]
	protected float m_fCircleStartAlongM;

	[Attribute("5", UIWidgets.Slider, "Circle runs start this far inside the area edge away from the turn [m]", "1 50 0.5", category: "Load")]
	protected float m_fCircleEdgeM;

	// === STATE ===
	protected ref array<float> m_aLoadSteers;
	protected ref array<float> m_aALatTargets;
	protected int m_iStairHoldSteps;
	protected int m_iStairSettleSteps;
	protected int m_iRefHoldSteps;
	protected int m_iALatWindowSteps;
	protected int m_iALatHoldSteps;
	protected int m_iALatMaxSteps;
	protected int m_iBrakeCapSteps;
	protected float m_fKappaRefSum;
	protected int m_iKappaRefCount;
	protected float m_fKappaRef;
	protected int m_iLevel;
	protected int m_iLevelSteps;
	protected float m_fLevelKmh;
	protected float m_fALatSum;
	protected int m_iALatCount;
	protected int m_iWindowSteps;
	protected int m_iSteadySteps;
	protected float m_fBrakeCmd;
	protected float m_fBrakeIntegral;
	protected float m_fAccSmooth;
	protected bool m_bAccSmoothSet;

	//------------------------------------------------------------------------------------------------
	override string GetSteerProbeId()
	{
		return PROBE_E22;
	}

	//------------------------------------------------------------------------------------------------
	override float GetSteerTimeoutS()
	{
		return TIMEOUT_E22_S;
	}

	//------------------------------------------------------------------------------------------------
	override bool BuildSteerPlan(out string failure)
	{
		failure = string.Empty;
		m_iStairHoldSteps = Math.MaxInt(1, DC_ProbeSequences.StepsFor(m_fStairHoldS, m_fDt));
		m_iStairSettleSteps = DC_ProbeSequences.StepsFor(m_fStairSettleS, m_fDt);
		m_iRefHoldSteps = Math.MaxInt(2, DC_ProbeSequences.StepsFor(m_fRefHoldS, m_fDt));
		m_iALatWindowSteps = Math.MaxInt(1, DC_ProbeSequences.StepsFor(m_fALatAdjustS, m_fDt));
		m_iALatHoldSteps = DC_ProbeSequences.StepsFor(m_fALatHoldS, m_fDt);
		m_iALatMaxSteps = DC_ProbeSequences.StepsFor(m_fALatMaxS, m_fDt);
		m_iBrakeCapSteps = DC_ProbeSequences.StepsFor(BRAKE_CAP_S, m_fDt);

		array<string> groups = {};
		m_sStairGrid.Split(";", groups, true);
		foreach (string groupText : groups)
		{
			array<string> parts = {};
			groupText.Split(":", parts, false);
			float speed;
			if (parts.Count() != 2 || !ParseSteerFloat(parts[0], speed) || speed <= 0)
			{
				failure = "m_sStairGrid '" + m_sStairGrid + "' is not '<speed>:<steer>,...;...'";
				return false;
			}

			DC_ProbeSteerPhase stair = new DC_ProbeSteerPhase(DC_ESteerPhaseKind.STAIR, "stair_" + speed.ToString(-1, 0), SITE_FLAT, speed);
			if (!ParseSteerFloats(parts[1], stair.m_aLevels, -1, 1) || stair.m_aLevels.IsEmpty())
			{
				failure = "m_sStairGrid '" + m_sStairGrid + "': the levels of " + parts[0] + " are not steering values in -1..1";
				return false;
			}

			stair.m_fStartAlongM = 2 * m_fAreaMarginM;
			m_aPhases.Insert(stair);
		}

		m_aLoadSteers = {};
		m_aALatTargets = {};
		if (!ParseSteerFloats(m_sLoadSteers, m_aLoadSteers, -1, 1))
		{
			failure = "m_sLoadSteers '" + m_sLoadSteers + "' is not a list of steering values in -1..1";
			return false;
		}

		if (!ParseSteerFloats(m_sALatTargets, m_aALatTargets, 0.1, 10))
		{
			failure = "m_sALatTargets '" + m_sALatTargets + "' is not a list of a_lat targets in 0.1..10 m/s2";
			return false;
		}

		foreach (float steer : m_aLoadSteers)
		{
			if (steer == 0)
				continue;

			string name = "load_+" + steer.ToString(-1, 2);
			if (steer < 0)
				name = "load_" + steer.ToString(-1, 2);

			DC_ProbeSteerPhase load = new DC_ProbeSteerPhase(DC_ESteerPhaseKind.LOAD, name, SITE_FLAT, m_fRefKmh);
			load.m_fSteer = steer;
			load.m_fStartAlongM = m_fCircleStartAlongM;
			foreach (float target : m_aALatTargets)
			{
				load.m_aLevels.Insert(target);
			}

			m_aPhases.Insert(load);
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! A load run starts on the side of the area away from its turn (turn side = sign(c) x the steer sign)
	override void OnBeginRun(notnull DC_ProbeSteerPhase phase)
	{
		if (phase.m_eKind != DC_ESteerPhaseKind.LOAD)
			return;

		if (m_iMeasuredSteerSign == 0 && m_iSteerSignSamples > 0)
		{
			if (m_fSteerSignSum > 0)
				m_iMeasuredSteerSign = 1;
			else if (m_fSteerSignSum < 0)
				m_iMeasuredSteerSign = -1;
		}

		float turnSide = GetSteerSign();
		if (phase.m_fSteer < 0)
			turnSide = -turnSide;

		float offset = Math.Max(m_fFlatAreaHalfWidthM - m_fCircleEdgeM, 0);
		phase.m_fLateralM = -turnSide * offset;
		m_fKappaRefSum = 0;
		m_iKappaRefCount = 0;
		m_fKappaRef = 0;
		int steerSign = GetSteerSign();
		phase.AddNote("steer_sign:" + steerSign.ToString());
	}

	//------------------------------------------------------------------------------------------------
	override bool IsSteeringSegment(DC_ESteerSegment segment)
	{
		return segment == DC_ESteerSegment.STAIR;
	}

	//------------------------------------------------------------------------------------------------
	override DC_ESteerSegment AfterAccel(notnull DC_ProbeSteerPhase phase)
	{
		if (phase.m_eKind == DC_ESteerPhaseKind.STAIR)
			return DC_ESteerSegment.STAIR;

		return DC_ESteerSegment.REF;
	}

	//------------------------------------------------------------------------------------------------
	override void CommandSegment(notnull DC_ProbeSteerPhase phase)
	{
		m_Actuator.SetSteeringCall(true);
		if (m_eSegment == DC_ESteerSegment.STAIR)
		{
			int level = Math.MinInt(m_iSegStep / m_iStairHoldSteps, phase.m_aLevels.Count() - 1);
			m_iEventIdx = level;
			m_fSegTarget = phase.m_fTargetKmh;
			m_bSegFlag = (m_iSegStep % m_iStairHoldSteps) >= m_iStairSettleSteps;
			m_Actuator.SetSpeedHold(phase.m_fTargetKmh, phase.m_aLevels[level]);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.REF)
		{
			m_iEventIdx = -1;
			m_fSegTarget = m_fRefKmh;
			m_bSegFlag = m_iSegStep >= m_iRefHoldSteps / 2;
			m_Actuator.SetSpeedHold(m_fRefKmh, phase.m_fSteer);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.ALAT)
		{
			m_iEventIdx = m_iLevel;
			m_fSegTarget = phase.m_aLevels[m_iLevel];
			m_Actuator.SetSpeedHold(m_fLevelKmh, phase.m_fSteer);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.BRAKE)
		{
			m_iEventIdx = -1;
			m_fSegTarget = m_fBrakeDecelMs2;
			m_Actuator.SetCommand(phase.m_fSteer, 0, m_fBrakeCmd, 1, -1, false);
			return;
		}

		m_Actuator.SetSpeedHold(phase.m_fTargetKmh, 0);
	}

	//------------------------------------------------------------------------------------------------
	override void AdvanceSegment(notnull DC_ProbeSteerPhase phase)
	{
		if (m_eSegment == DC_ESteerSegment.STAIR)
		{
			if (m_iSegStep >= phase.m_aLevels.Count() * m_iStairHoldSteps)
				EnterSegment(DC_ESteerSegment.STOP);

			return;
		}

		vector velocity = m_Actuator.GetVelocityLocal();
		vector angular = m_Actuator.GetAngularVelocityLocal();
		float v = velocity[2];
		float r = angular[1];
		if (m_eSegment == DC_ESteerSegment.REF)
		{
			if (m_bSegFlag && v > 0.5)
			{
				m_fKappaRefSum += r / v;
				m_iKappaRefCount++;
			}

			if (m_iSegStep < m_iRefHoldSteps)
				return;

			if (m_iKappaRefCount > 0)
				m_fKappaRef = m_fKappaRefSum / m_iKappaRefCount;

			phase.AddNote("kappa_ref:" + m_fKappaRef.ToString(-1, 5));
			BeginLevel(phase, 0);
			EnterSegment(DC_ESteerSegment.ALAT);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.ALAT)
		{
			AdvanceLevel(phase, Math.AbsFloat(v * r));
			return;
		}

		if (m_eSegment == DC_ESteerSegment.BRAKE)
		{
			AdvanceBrake(phase);
			return;
		}

		EnterSegment(DC_ESteerSegment.STOP);
	}

	//------------------------------------------------------------------------------------------------
	//! Level i of the alat segment: its first speed from the reference curvature (v = sqrt(a / kappa_ref)), else the
	//! last level's speed
	protected void BeginLevel(notnull DC_ProbeSteerPhase phase, int level)
	{
		m_iLevel = level;
		m_iLevelSteps = 0;
		m_fALatSum = 0;
		m_iALatCount = 0;
		m_iWindowSteps = 0;
		m_iSteadySteps = 0;
		m_bSegFlag = false;
		float target = phase.m_aLevels[level];
		float kappa = Math.AbsFloat(m_fKappaRef);
		if (kappa > 0.001)
			m_fLevelKmh = Math.Sqrt(target / kappa) * 3.6;
		else if (level == 0)
			m_fLevelKmh = m_fRefKmh * 2;

		m_fLevelKmh = Math.Clamp(m_fLevelKmh, ALAT_MIN_KMH, ALAT_MAX_KMH);
	}

	//------------------------------------------------------------------------------------------------
	//! Window means of |v r|: adapt the speed target until converged, then hold; next level or the brake segment
	protected void AdvanceLevel(notnull DC_ProbeSteerPhase phase, float aLat)
	{
		m_iLevelSteps++;
		m_fALatSum += aLat;
		m_iALatCount++;
		m_iWindowSteps++;
		if (m_bSegFlag)
			m_iSteadySteps++;

		if (m_iWindowSteps >= m_iALatWindowSteps)
		{
			float target = phase.m_aLevels[m_iLevel];
			float mean = m_fALatSum / Math.Max(1, m_iALatCount);
			if (Math.AbsFloat(mean - target) <= m_fALatToleranceMs2)
			{
				m_bSegFlag = true;
			}
			else
			{
				m_bSegFlag = false;
				m_iSteadySteps = 0;
				float factor = 1.15;
				if (mean > 0.01)
					factor = Math.Clamp(Math.Sqrt(target / mean), 0.85, 1.15);

				m_fLevelKmh = Math.Clamp(m_fLevelKmh * factor, ALAT_MIN_KMH, ALAT_MAX_KMH);
			}

			m_fALatSum = 0;
			m_iALatCount = 0;
			m_iWindowSteps = 0;
		}

		bool held = m_iSteadySteps >= m_iALatHoldSteps;
		bool capped = m_iLevelSteps >= m_iALatMaxSteps;
		if (!held && !capped)
			return;

		string levelKmh = m_fLevelKmh.ToString(-1, 2);
		if (held)
			phase.AddNote("alat_" + m_iLevel.ToString() + "_kmh:" + levelKmh);
		else
			phase.AddNote("alat_unconverged_" + m_iLevel.ToString() + "_kmh:" + levelKmh);

		if (m_iLevel + 1 < phase.m_aLevels.Count())
		{
			BeginLevel(phase, m_iLevel + 1);
			return;
		}

		m_fBrakeCmd = m_fBrakeFeedForward;
		m_fBrakeIntegral = 0;
		m_bAccSmoothSet = false;
		EnterSegment(DC_ESteerSegment.BRAKE);
	}

	//------------------------------------------------------------------------------------------------
	//! Brake PI on the smoothed longitudinal acceleration (acc_l_z, decelerating negative) to -m_fBrakeDecelMs2
	protected void AdvanceBrake(notnull DC_ProbeSteerPhase phase)
	{
		vector acc = m_Actuator.GetAccelerationLocal();
		if (!m_bAccSmoothSet)
		{
			m_fAccSmooth = acc[2];
			m_bAccSmoothSet = true;
		}
		else
		{
			m_fAccSmooth += BRAKE_ACC_ALPHA * (acc[2] - m_fAccSmooth);
		}

		float error = m_fAccSmooth + m_fBrakeDecelMs2;
		m_fBrakeIntegral += error * m_fDt;
		m_fBrakeCmd = Math.Clamp(m_fBrakeFeedForward + m_fBrakeKp * error + m_fBrakeKi * m_fBrakeIntegral, 0, 1);
		m_bSegFlag = Math.AbsFloat(error) <= 0.5;
		if (m_Actuator.GetSpeedKmh() <= m_fBrakeEndKmh || m_iSegStep >= m_iBrakeCapSteps)
		{
			if (m_iSegStep >= m_iBrakeCapSteps)
				phase.AddNote("brake_cap");

			EnterSegment(DC_ESteerSegment.STOP);
		}
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeParams(notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_sStairGrid=" + m_sStairGrid);
		outKeyValues.Insert("m_fStairHoldS=" + m_fStairHoldS.ToString(-1, 3));
		outKeyValues.Insert("m_fStairSettleS=" + m_fStairSettleS.ToString(-1, 3));
		outKeyValues.Insert("m_sLoadSteers=" + m_sLoadSteers);
		outKeyValues.Insert("m_fRefKmh=" + m_fRefKmh.ToString(-1, 3));
		outKeyValues.Insert("m_fRefHoldS=" + m_fRefHoldS.ToString(-1, 3));
		outKeyValues.Insert("m_sALatTargets=" + m_sALatTargets);
		outKeyValues.Insert("m_fALatToleranceMs2=" + m_fALatToleranceMs2.ToString(-1, 3));
		outKeyValues.Insert("m_fALatAdjustS=" + m_fALatAdjustS.ToString(-1, 3));
		outKeyValues.Insert("m_fALatHoldS=" + m_fALatHoldS.ToString(-1, 3));
		outKeyValues.Insert("m_fALatMaxS=" + m_fALatMaxS.ToString(-1, 3));
		outKeyValues.Insert("m_fBrakeDecelMs2=" + m_fBrakeDecelMs2.ToString(-1, 3));
		outKeyValues.Insert("m_fBrakeFeedForward=" + m_fBrakeFeedForward.ToString(-1, 3));
		outKeyValues.Insert("m_fBrakeKp=" + m_fBrakeKp.ToString(-1, 4));
		outKeyValues.Insert("m_fBrakeKi=" + m_fBrakeKi.ToString(-1, 4));
		outKeyValues.Insert("m_fBrakeEndKmh=" + m_fBrakeEndKmh.ToString(-1, 3));
		outKeyValues.Insert("m_fCircleStartAlongM=" + m_fCircleStartAlongM.ToString(-1, 2));
		outKeyValues.Insert("m_fCircleEdgeM=" + m_fCircleEdgeM.ToString(-1, 2));
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeHeader(notnull array<string> lines)
	{
		lines.Insert("s1load.rule=stair: the levels of a group in order, event_idx = level, seg_flag 1 after the level's transient; ref: the load command at the reference speed, seg_flag 1 in its second half; alat: event_idx = level, seg_target = a_lat target [m/s2], seg_flag 1 once a window mean of |vel_l_z x angvel_l_y| is within the tolerance; brake: seg_target = decel target [m/s2], seg_flag 1 within 0.5 m/s2");
		lines.Insert("s1load.cc07=step 0: stair (and ref) rows in the low-slip regime; d: alat and brake rows against the ref rows of the same command");
	}
}
