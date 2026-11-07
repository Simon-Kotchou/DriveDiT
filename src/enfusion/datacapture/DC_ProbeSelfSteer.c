// ============================================================================
// DC_ProbeSelfSteer - probe e21: S1 Mode A self-steer hold at zero command (DD-14b; CC-07 a, b, c)
// ============================================================================
//
// MIGRATION_1_9.md 6.1 CC-07: does a driverless car that gets a zero steering command keep a straight line on 1.9?
// The first 1.9.0.37 Play (launch 20261106T194516Z-wb-ab6a, e01) read input_post_steer != cmd_steer on 714 of 3567
// steps, nonzero on 355 of 2218 driving steps at a zero command (max 0.037). This probe measures it where CC-07 asks:
//
//   flat_<v>            on the flat site (the runway, DC_ProbeSteerBase; area +-15 m), for each speed of m_sFlatSpeedsKmh
//                       (20, 40), at driving-assistance FULL (set explicitly): reset, settle, accel with
//                       SetSteering(0), hold (m_fSpeedSettleS at the speed, still held), measure (SetSteering(0) every
//                       step for m_fFlatHeldM, >= 200 m for CC-07 b), nocall (no SetSteering call at all for
//                       m_fFlatNocallM: the engine keeps its own input), stop;
//   flat_<v>_none       the same at assistance NONE (set explicitly; m_bAssistVariant; shorter segments,
//                       m_fAssistHeldM / m_fAssistNocallM): FULL against NONE whatever the original mode was. The
//                       original mode is set back after every phase and at every probe stop;
//   cross_<v>           on the cross-slope site (>= 5%, found at run time, below) for each speed of m_sCrossSpeedsKmh:
//                       the same segments with m_fCrossHeldM / m_fCrossNocallM (CC-07 c: heading drift per 100 m);
//                       the hold lasts until the car reaches the site's measure zone, so measure and nocall lie in it.
// Every row carries cmd_steer, input_pre/post_steer, steer_call, the yaw rate, position and heading, and the
// wheel-angle candidates, so the analysis (tools/probes/t0eng/e21_selfsteer.py) judges the readback (a), the
// curvature over >= 200 m (b) and the applied angle and drift on the slope (c) per variant.
//
// CROSS-SLOPE SITE: DC_SteerSiteSearch kind cross over the grid region of the base (Arland road extent, 50 m): the
// strip is laid along the contour through every grid point whose gradient is within 0.8 x m_fCrossMin ..
// 1.15 x m_fCrossMax, with the downhill on m_iCrossSide (+1 right), +-m_fCrossHeadingSpreadDeg; length = run-up +
// held + nocall + stop, the measure zone [run-up, run-up + held + nocall] must keep side x cross within
// m_fCrossMin..m_fCrossMax at every 5 m station, |mean grade| <= m_fCrossGradeMax, windowed grades and the whole
// strip's cross-slope bounded, every sample above the base's minimum surface y, and no obstacle hit within
// +-m_fCrossHalfWidthM. m_sCrossExplicit candidates ('x z heading;...') are tried first. If nothing passes at the full
// segment lengths, the search runs once more with m_fCrossFallbackM per variant and the cross phases use those
// lengths (header site.cross.fallback=1).
// ============================================================================

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "G19-1 probe e21: S1 Mode A self-steer at zero command, flat and cross-slope, held and no-call, assistance FULL and NONE (CC-07 a/b/c)")]
class DC_ProbeSelfSteerClass : DC_ProbeSteerBaseClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeSelfSteer : DC_ProbeSteerBase
{
	static const string PROBE_E21 = "e21";
	//! e21 + e22 + e23 stay within 15 min of Play: 420 + 240 + 240 s (a timeout ends the probe, footer stopped)
	static const float TIMEOUT_E21_S = 420;
	//! Assistance modes of the comparison (EVehicleDrivingAssistanceMode: 0 NONE, 2 FULL), set explicitly
	static const int ASSIST_FULL = 2;
	static const int ASSIST_NONE = 0;

	// === S1 FLAT ===
	[Attribute("20,40", UIWidgets.EditBox, "S1 flat: speeds [km/h], one phase each (CC-07 b: 20 and 40)", category: "S1 flat")]
	protected string m_sFlatSpeedsKmh;

	[Attribute("200", UIWidgets.Slider, "S1 flat: distance with SetSteering(0) every step [m] (CC-07 b: >= 200 m)", "20 1000 10", category: "S1 flat")]
	protected float m_fFlatHeldM;

	[Attribute("200", UIWidgets.Slider, "S1 flat: distance without any SetSteering call [m] (the no-call variant)", "0 1000 10", category: "S1 flat")]
	protected float m_fFlatNocallM;

	[Attribute("2", UIWidgets.Slider, "Speed settle at the target before the measurement, steering still held [s]", "0 20 0.5", category: "S1 flat")]
	protected float m_fSpeedSettleS;

	// === ASSISTANCE ===
	[Attribute("1", UIWidgets.CheckBox, "S1 flat once more at driving-assistance NONE (flat_<v>_none); the main flat phases run at FULL. Both modes are set explicitly for the phase and the original mode is set back after it (an engine global, no settings file)", category: "Assistance")]
	protected bool m_bAssistVariant;

	[Attribute("100", UIWidgets.Slider, "Variant: held distance [m]", "20 1000 10", category: "Assistance")]
	protected float m_fAssistHeldM;

	[Attribute("100", UIWidgets.Slider, "Variant: no-call distance [m]", "0 1000 10", category: "Assistance")]
	protected float m_fAssistNocallM;

	// === S1 CROSS-SLOPE ===
	[Attribute("1", UIWidgets.CheckBox, "S1 on a cross-slope site found at run time (CC-07 c)", category: "S1 cross")]
	protected bool m_bCross;

	[Attribute("20,40", UIWidgets.EditBox, "S1 cross: speeds [km/h] (CC-07 c: 20 and 40)", category: "S1 cross")]
	protected string m_sCrossSpeedsKmh;

	[Attribute("100", UIWidgets.Slider, "S1 cross: held distance [m]", "20 500 5", category: "S1 cross")]
	protected float m_fCrossHeldM;

	[Attribute("100", UIWidgets.Slider, "S1 cross: no-call distance [m]", "0 500 5", category: "S1 cross")]
	protected float m_fCrossNocallM;

	[Attribute("60", UIWidgets.Slider, "S1 cross: held and no-call distance of the shorter fallback search [m] (0 = no fallback)", "0 500 5", category: "S1 cross")]
	protected float m_fCrossFallbackM;

	[Attribute("75", UIWidgets.Slider, "S1 cross: run-up before the measure zone [m] (accel to 40 km/h and the speed settle)", "10 300 5", category: "S1 cross")]
	protected float m_fCrossRunupM;

	[Attribute("35", UIWidgets.Slider, "S1 cross: stop distance after the measure zone [m]", "10 200 5", category: "S1 cross")]
	protected float m_fCrossStopM;

	[Attribute("0.05", UIWidgets.EditBox, "S1 cross: min side x cross-slope at every measure-zone station (CC-07 c: >= 5%)", category: "S1 cross")]
	protected float m_fCrossMin;

	[Attribute("0.12", UIWidgets.EditBox, "S1 cross: max side x cross-slope at every measure-zone station", category: "S1 cross")]
	protected float m_fCrossMax;

	[Attribute("0.065", UIWidgets.EditBox, "S1 cross: preferred mean cross-slope (search score)", category: "S1 cross")]
	protected float m_fCrossTarget;

	[Attribute("1", UIWidgets.Slider, "S1 cross: downhill side of the heading, +1 right, -1 left", "-1 1 2", category: "S1 cross")]
	protected int m_iCrossSide;

	[Attribute("8", UIWidgets.Slider, "S1 cross: terrain, obstacle and area half width [m]", "2 50 0.5", category: "S1 cross")]
	protected float m_fCrossHalfWidthM;

	[Attribute("0.02", UIWidgets.EditBox, "S1 cross: max |mean grade| along the strip", category: "S1 cross")]
	protected float m_fCrossGradeMax;

	[Attribute("0.035", UIWidgets.EditBox, "S1 cross: max |grade| over any 20 m window of the measure zone", category: "S1 cross")]
	protected float m_fCrossGradeWindowMax;

	[Attribute("0.06", UIWidgets.EditBox, "S1 cross: max |grade| over any 20 m window of the whole strip", category: "S1 cross")]
	protected float m_fCrossGradeWindowMaxAll;

	[Attribute("0.15", UIWidgets.EditBox, "S1 cross: max |cross-slope| at any station of the whole strip", category: "S1 cross")]
	protected float m_fCrossAbsMaxAll;

	[Attribute("0.5", UIWidgets.EditBox, "S1 cross: max rms about the plane fit [m]", category: "S1 cross")]
	protected float m_fCrossRmsMax;

	[Attribute("10", UIWidgets.Slider, "S1 cross: the contour strip is also tried at +- this heading [deg]", "0 45 1", category: "S1 cross")]
	protected float m_fCrossHeadingSpreadDeg;

	[Attribute("", UIWidgets.EditBox, "S1 cross: explicit candidates 'x z heading;x z heading' tried before the grid (strip start, heading in deg)", category: "S1 cross")]
	protected string m_sCrossExplicit;

	// === STATE ===
	protected ref array<float> m_aFlatSpeeds;
	protected ref array<float> m_aCrossSpeeds;
	protected bool m_bCrossFallbackUsed;
	protected int m_iSettleAtSpeedSteps;

	//------------------------------------------------------------------------------------------------
	override string GetSteerProbeId()
	{
		return PROBE_E21;
	}

	//------------------------------------------------------------------------------------------------
	override float GetSteerTimeoutS()
	{
		return TIMEOUT_E21_S;
	}

	//------------------------------------------------------------------------------------------------
	//! The flat runs are straight: +-15 m leaves 12 m of drift room, enough for |kappa| up to 1.5e-4 over 400 m
	override float GetFlatHalfWidthM()
	{
		return 15;
	}

	//------------------------------------------------------------------------------------------------
	override float GetMaxSpeedKmh()
	{
		float top = 40;
		array<float> speeds = {};
		if (ParseSteerFloats(m_sFlatSpeedsKmh, speeds, 1, 120))
		{
			foreach (float speed : speeds)
			{
				top = Math.Max(top, speed);
			}
		}

		if (ParseSteerFloats(m_sCrossSpeedsKmh, speeds, 1, 120))
		{
			foreach (float crossSpeed : speeds)
			{
				top = Math.Max(top, crossSpeed);
			}
		}

		return top;
	}

	//------------------------------------------------------------------------------------------------
	//! Every list non-empty and every value in range (the comparisons also refuse a NaN); then the phases
	override bool BuildSteerPlan(out string failure)
	{
		failure = string.Empty;
		m_aFlatSpeeds = {};
		m_aCrossSpeeds = {};
		m_bCrossFallbackUsed = false;
		if (!ParseSteerFloats(m_sFlatSpeedsKmh, m_aFlatSpeeds, 5, 120) || m_aFlatSpeeds.IsEmpty())
		{
			failure = "m_sFlatSpeedsKmh '" + m_sFlatSpeedsKmh + "' is not a non-empty list of speeds in 5..120 km/h";
			return false;
		}

		if (m_bCross && (!ParseSteerFloats(m_sCrossSpeedsKmh, m_aCrossSpeeds, 5, 120) || m_aCrossSpeeds.IsEmpty()))
		{
			failure = "m_sCrossSpeedsKmh '" + m_sCrossSpeedsKmh + "' is not a non-empty list of speeds in 5..120 km/h";
			return false;
		}

		if (!InRange(m_fFlatHeldM, 20, 1000) || !InRange(m_fFlatNocallM, 0, 1000) || !InRange(m_fAssistHeldM, 20, 1000) || !InRange(m_fAssistNocallM, 0, 1000) || !InRange(m_fSpeedSettleS, 0, 20))
		{
			failure = "a flat or assistance distance / time attribute is out of range";
			return false;
		}

		if (m_bCross && (!InRange(m_fCrossMin, 0.01, 0.3) || !InRange(m_fCrossMax, m_fCrossMin, 0.3) || !InRange(m_fCrossHeldM, 20, 500) || !InRange(m_fCrossNocallM, 0, 500) || !InRange(m_fCrossHalfWidthM, 4, 50) || !InRange(m_fCrossRunupM, 10, 300) || !InRange(m_fCrossStopM, 10, 200)))
		{
			failure = "a cross-slope attribute is out of range (m_fCrossMin 0.01..0.3 <= m_fCrossMax, distances, half width 4..50 m)";
			return false;
		}

		m_iSettleAtSpeedSteps = DC_ProbeSequences.StepsFor(m_fSpeedSettleS, m_fDt);
		foreach (float flatKmh : m_aFlatSpeeds)
		{
			AddZeroPhase(DC_ESteerPhaseKind.FLAT, "flat_" + flatKmh.ToString(-1, 0), SITE_FLAT, flatKmh, ASSIST_FULL, m_fFlatHeldM, m_fFlatNocallM);
		}

		if (m_bAssistVariant)
		{
			foreach (float assistKmh : m_aFlatSpeeds)
			{
				AddZeroPhase(DC_ESteerPhaseKind.FLAT, "flat_" + assistKmh.ToString(-1, 0) + "_none", SITE_FLAT, assistKmh, ASSIST_NONE, m_fAssistHeldM, m_fAssistNocallM);
			}
		}

		if (m_bCross)
		{
			foreach (float crossKmh : m_aCrossSpeeds)
			{
				AddZeroPhase(DC_ESteerPhaseKind.CROSS, "cross_" + crossKmh.ToString(-1, 0), SITE_CROSS, crossKmh, ASSIST_FULL, m_fCrossHeldM, m_fCrossNocallM);
			}
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected void AddZeroPhase(DC_ESteerPhaseKind kind, string name, string site, float targetKmh, int assist, float heldM, float nocallM)
	{
		DC_ProbeSteerPhase phase = new DC_ProbeSteerPhase(kind, name, site, targetKmh);
		phase.m_iAssist = assist;
		phase.m_fHeldM = heldM;
		phase.m_fNocallM = nocallM;
		phase.m_fStartAlongM = 2 * m_fAreaMarginM;
		phase.m_fLateralM = 0;
		m_aPhases.Insert(phase);
	}

	//------------------------------------------------------------------------------------------------
	//! The cross-slope search (full segment lengths)
	override void AddExtraSearches()
	{
		if (!m_bCross || m_aCrossSpeeds.IsEmpty())
			return;

		AddSearch(SITE_CROSS, BuildCrossSearch(m_fCrossHeldM, m_fCrossNocallM));
	}

	//------------------------------------------------------------------------------------------------
	protected DC_SteerSiteSearch BuildCrossSearch(float heldM, float nocallM)
	{
		DC_SteerSiteCriteria c = new DC_SteerSiteCriteria();
		c.m_sKind = SITE_CROSS;
		c.m_fLengthM = m_fCrossRunupM + heldM + nocallM + m_fCrossStopM;
		SetSiteGeometry(c, m_fCrossHalfWidthM);
		c.m_fZoneFromM = m_fCrossRunupM;
		c.m_fZoneToM = m_fCrossRunupM + heldM + nocallM;
		c.m_fGradeMax = m_fCrossGradeMax;
		c.m_fGradeWindowMax = m_fCrossGradeWindowMax;
		c.m_fGradeWindowMaxAll = m_fCrossGradeWindowMaxAll;
		c.m_fCrossMin = m_fCrossMin;
		c.m_fCrossMax = m_fCrossMax;
		c.m_fCrossAbsMaxAll = m_fCrossAbsMaxAll;
		c.m_fCrossTarget = m_fCrossTarget;
		c.m_fRmsMax = m_fCrossRmsMax;
		c.m_fMinSurfaceY = m_fSiteMinSurfaceY;
		c.m_iSide = 1;
		if (m_iCrossSide < 0)
			c.m_iSide = -1;

		DC_SteerSiteSearch search = new DC_SteerSiteSearch();
		search.Configure(c, m_aExcludes, m_iSiteMaxStrips, m_iSiteMaxObstacleChecks);
		AddExplicitCandidates(search, m_sCrossExplicit);
		search.SetGrid(m_vGridMin[0], m_vGridMin[2], m_vGridMax[0], m_vGridMax[2], m_fGridSpacingM, 15, m_fCrossHeadingSpreadDeg);
		return search;
	}

	//------------------------------------------------------------------------------------------------
	//! 'x z heading;...' (an unparsable group is skipped with a WARNING)
	protected void AddExplicitCandidates(notnull DC_SteerSiteSearch search, string text)
	{
		array<string> candidates = {};
		text.Split(";", candidates, true);
		foreach (string candidate : candidates)
		{
			string trimmed = candidate.Trim();
			array<string> parts = {};
			trimmed.Split(" ", parts, true);
			float x;
			float z;
			float heading;
			if (parts.Count() != 3 || !ParseSteerFloat(parts[0], x) || !ParseSteerFloat(parts[1], z) || !ParseSteerFloat(parts[2], heading))
			{
				LogWarning("m_sCrossExplicit candidate '" + trimmed + "' is not 'x z heading' - skipped");
				continue;
			}

			search.AddExplicit(Vector(x, 0, z), heading);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! No cross site at the full lengths: one more search with m_fCrossFallbackM per variant
	override bool OnSearchesDone()
	{
		if (!m_bCross || m_bCrossFallbackUsed || m_fCrossFallbackM <= 0 || m_aCrossSpeeds.IsEmpty())
			return false;

		if (GetSite(SITE_CROSS))
			return false;

		m_bCrossFallbackUsed = true;
		LogWarning(string.Format("no cross-slope site for %1 m + %2 m - searching again with %3 m + %3 m", m_fCrossHeldM, m_fCrossNocallM, m_fCrossFallbackM));
		ReplaceSearch(SITE_CROSS, BuildCrossSearch(m_fCrossFallbackM, m_fCrossFallbackM));
		foreach (DC_ProbeSteerPhase phase : m_aPhases)
		{
			if (phase.m_eKind != DC_ESteerPhaseKind.CROSS)
				continue;

			phase.m_fHeldM = m_fCrossFallbackM;
			phase.m_fNocallM = m_fCrossFallbackM;
			phase.AddNote("cross_fallback");
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	override DC_ESteerSegment AfterAccel(notnull DC_ProbeSteerPhase phase)
	{
		return DC_ESteerSegment.HOLD;
	}

	//------------------------------------------------------------------------------------------------
	//! hold and measure: SetSteering(0) every step with the speed hold; nocall: the same without SetSteering
	override void CommandSegment(notnull DC_ProbeSteerPhase phase)
	{
		m_fSegTarget = phase.m_fTargetKmh;
		m_Actuator.SetSteeringCall(m_eSegment != DC_ESteerSegment.NOCALL);
		m_Actuator.SetSpeedHold(phase.m_fTargetKmh, 0);
		m_bSegFlag = m_eSegment == DC_ESteerSegment.MEASURE || m_eSegment == DC_ESteerSegment.NOCALL;
	}

	//------------------------------------------------------------------------------------------------
	override void AdvanceSegment(notnull DC_ProbeSteerPhase phase)
	{
		if (m_eSegment == DC_ESteerSegment.HOLD)
		{
			// at least the speed settle, and on the cross-slope site until the car reaches the measure zone the search
			// checked (its run-up and stop parts only meet the lax criteria)
			if (m_iSegStep < m_iSettleAtSpeedSteps)
				return;

			DC_SteerSite site = GetSite(phase.m_sSite);
			if (site && m_Area && m_Area.AlongM(m_Actuator.GetPosition()) < site.m_fZoneFromM)
				return;

			EnterSegment(DC_ESteerSegment.MEASURE);
			return;
		}

		if (m_eSegment == DC_ESteerSegment.MEASURE)
		{
			if (SegmentAlongM() < phase.m_fHeldM)
				return;

			if (phase.m_fNocallM > 0)
				EnterSegment(DC_ESteerSegment.NOCALL);
			else
				EnterSegment(DC_ESteerSegment.STOP);

			return;
		}

		if (m_eSegment == DC_ESteerSegment.NOCALL)
		{
			if (SegmentAlongM() >= phase.m_fNocallM)
			{
				m_Actuator.SetSteeringCall(true);
				EnterSegment(DC_ESteerSegment.STOP);
			}

			return;
		}

		EnterSegment(DC_ESteerSegment.STOP);
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeParams(notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_sFlatSpeedsKmh=" + m_sFlatSpeedsKmh);
		outKeyValues.Insert("m_fFlatHeldM=" + m_fFlatHeldM.ToString(-1, 2));
		outKeyValues.Insert("m_fFlatNocallM=" + m_fFlatNocallM.ToString(-1, 2));
		outKeyValues.Insert("m_fSpeedSettleS=" + m_fSpeedSettleS.ToString(-1, 3));
		outKeyValues.Insert("m_bAssistVariant=" + DC_LaunchGuard.BoolText(m_bAssistVariant));
		outKeyValues.Insert("m_fAssistHeldM=" + m_fAssistHeldM.ToString(-1, 2));
		outKeyValues.Insert("m_fAssistNocallM=" + m_fAssistNocallM.ToString(-1, 2));
		outKeyValues.Insert("m_bCross=" + DC_LaunchGuard.BoolText(m_bCross));
		outKeyValues.Insert("m_sCrossSpeedsKmh=" + m_sCrossSpeedsKmh);
		outKeyValues.Insert("m_fCrossHeldM=" + m_fCrossHeldM.ToString(-1, 2));
		outKeyValues.Insert("m_fCrossNocallM=" + m_fCrossNocallM.ToString(-1, 2));
		outKeyValues.Insert("m_fCrossFallbackM=" + m_fCrossFallbackM.ToString(-1, 2));
		outKeyValues.Insert("m_fCrossRunupM=" + m_fCrossRunupM.ToString(-1, 2));
		outKeyValues.Insert("m_fCrossStopM=" + m_fCrossStopM.ToString(-1, 2));
		outKeyValues.Insert("m_fCrossMin=" + m_fCrossMin.ToString(-1, 4));
		outKeyValues.Insert("m_fCrossMax=" + m_fCrossMax.ToString(-1, 4));
		outKeyValues.Insert("m_fCrossTarget=" + m_fCrossTarget.ToString(-1, 4));
		outKeyValues.Insert("m_iCrossSide=" + m_iCrossSide.ToString());
		outKeyValues.Insert("m_fCrossHalfWidthM=" + m_fCrossHalfWidthM.ToString(-1, 2));
		outKeyValues.Insert("m_fCrossGradeMax=" + m_fCrossGradeMax.ToString(-1, 4));
		outKeyValues.Insert("m_fCrossGradeWindowMax=" + m_fCrossGradeWindowMax.ToString(-1, 4));
		outKeyValues.Insert("m_fCrossGradeWindowMaxAll=" + m_fCrossGradeWindowMaxAll.ToString(-1, 4));
		outKeyValues.Insert("m_fCrossAbsMaxAll=" + m_fCrossAbsMaxAll.ToString(-1, 4));
		outKeyValues.Insert("m_fCrossRmsMax=" + m_fCrossRmsMax.ToString(-1, 3));
		outKeyValues.Insert("m_fCrossHeadingSpreadDeg=" + m_fCrossHeadingSpreadDeg.ToString(-1, 2));
		outKeyValues.Insert("m_sCrossExplicit=" + m_sCrossExplicit);
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeHeader(notnull array<string> lines)
	{
		lines.Insert("s1.rule=per phase: hold and measure call SetSteering(0) every step (steer_call 1), nocall calls no SetSteering (steer_call 0); the speed hold keeps throttle / brake; seg_flag 1 on measure and nocall; along_m from the site start");
		lines.Insert("s1.cc07=a: input_post_steer vs cmd_steer on steer_call 1 steps; b: kappa over the flat measure segments (>= 200 m); c: the cross-slope segments; d and step 0: e22");
	}

	//------------------------------------------------------------------------------------------------
	override void AppendProbeSummary(notnull array<string> lines)
	{
		lines.Insert("site.cross.fallback=" + DC_LaunchGuard.BoolText(m_bCrossFallbackUsed));
	}
}
