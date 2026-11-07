// ============================================================================
// DC_ProbeSteerBase - shared machinery of the G19-1 steering probes e21, e22, e23 (DD-14b; CC-07, CC-08)
// ============================================================================
//
// A DC_ProbeComponent base (never placed itself) for DC_ProbeSelfSteer (e21, S1 self-steer, CC-07 a/b/c),
// DC_ProbeSteerLoad (e22, CC-07 step 0 and d) and DC_ProbeSteerSteps (e23, S2 steering steps, CC-08). Each subclass
// implements one probe id and its plan; this class runs them the way DC_ProbeSequences runs e01/e02a:
//
//   - Mode A through DC_ProbeActuator (DC_ProbeSequences.c), set up as the image probes do (Init: the deployed launch
//     clutch in mode slip, DC_LaunchClutchConfig.Slip(), the S105 m/s per 1000 rpm of the 1.9.0.37 card; the default
//     shift policy) with this component's speed-hold gains. ForceEnableSimulation is latched only when e01 of the same
//     Play found it needed (DC_ProbeSequences.GetForceDecision() == 1); otherwise the stuck guard of the image probes
//     (DC_ProbeActuator.SetStuckForce) latches it after a drive start that does not move. SetSteeringCall(false) skips
//     the SetSteering call for the no-call variant of S1 (column steer_call);
//   - the wheel-angle candidates (DC_SteerChannels) every physics step and, for the rotator bones, every frame;
//   - a chase camera (DC_RigR1 behind and above the car) so the car is animated at full detail wherever it drives
//     (the bone and entity sources come from its animation); it is given back at the stop (not on world teardown);
//   - the test sites from a runtime search (DC_SteerSites.c) before the first phase: the flat site (explicit
//     candidate: the Arland main runway of e01, with the grid as fallback; no trust override, a blocking hit rejects
//     it like any candidate), and the subclass's other sites (e21: the cross-slope site). Each probe searches with its
//     own area half width (GetFlatHalfWidthM: e21 15 m, e22 22 m, e23 10 m) and the stopping envelope of its top speed
//     (GetMaxSpeedKmh; v^2 / 2 m_fEnvelopeDecelMs2 + v m_fEnvelopeReactS along, x sin GetEnvelopeAngleDeg sideways;
//     e22's circles may exit in any direction: the full stopping distance sideways and backwards). The motion is held
//     to those assumptions: a driving segment ends with overspeed above GetMaxSpeedKmh() + 3 km/h and, outside
//     circle probes, with aborted_heading beyond GetEnvelopeAngleDeg() off the strip direction; it ends with
//     aborted_area outside the area. The searches share one native-call budget per frame (DC_SteerBudget). Each found
//     site is written with site.<kind>.reuse, the attribute values that make the 1.8 cold baseline drive the same
//     strip. Phases on a site that was not found are skipped (result skipped, note no_site);
//   - the wheel-angle axle choice (DC_SteerChannels.ArmAxleSelection) is armed only in a settle segment, from half
//     the settle on, at a standstill with the steering readback within 1e-3 of 0, and disarmed when the settle ends;
//   - phases with segments: reset (teleport in OnProbeFrame, never inside a physics step), settle (brake 1 +
//     handbrake), accel (speed hold, steering 0), the subclass's segments, stop (brake 1). A driving segment ends early
//     with engine_off, aborted_area (outside the site area shrunk by the margin), overspeed, aborted_heading, no_speed
//     (accel timeout) or spin
//     (|yaw rate| above m_fSpinYawRateRadS for SPIN_STEPS steps);
//   - the driving-assistance mode (BaseVehicleControllerComponent.Get/SetDrivingAssistanceMode, a global static of
//     the engine, BaseVehicleControllerComponent.c:36-39 of 1.8.0.13 and 1.9.0.37; vanilla sets it only from the
//     gameplay settings, SCR_PlayerController.c:304-306): a phase may set another mode for its own duration. The
//     original mode is read at the start and set back at the end of that phase and at every probe stop, also on world
//     teardown: it is an in-memory engine global that touches no entity, and leaving it changed would carry into the
//     next Play of the same Workbench session. No settings or profile file is written (nothing here calls the user
//     settings API);
//   - files: <probe>_steps.csv (one row per physics step: the actuator columns, steer_call, assist_mode, site,
//     along_m / lateral_m in the site frame, in_area, seg_target, seg_flag, event_idx, the wheel-angle columns) and
//     <probe>_phases.csv (one row per phase); run header lines vehicle.*, gear_policy.*, launch_clutch.*, steer.*,
//     site.<kind>.*, assist.*, and at the stop the drive and channel summaries.
//
// Durations are seconds converted with the runner's step length (GetStepDtS), segment counters advance by the engine
// steps a dispatched step covers (CoveredSteps), as in DC_ProbeSequences. Every engine member used here exists with
// the same declaration in 1.8.0.13 and 1.9.0.37 (no overlay).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! Segments of a steering-probe phase (column segment)
enum DC_ESteerSegment
{
	SEARCH,		//!< site search before the first phase (no rows)
	RESET,		//!< waiting for the teleport (OnProbeFrame); brake 1 + handbrake
	SETTLE,		//!< m_fSettleS of brake 1 + handbrake
	ACCEL,		//!< speed hold to the phase's target, steering 0
	HOLD,		//!< e21: speed settle at the target; e23: cruise before the train
	MEASURE,	//!< e21: zero command, SetSteering(0) every step
	NOCALL,		//!< e21: zero command, no SetSteering call
	STAIR,		//!< e22: low-load steering levels (step 0)
	REF,		//!< e22: the load command at the low-load reference speed
	ALAT,		//!< e22: the load command with the speed adapted to an a_lat target
	BRAKE,		//!< e22: the load command, braking at the decel target
	TRAIN,		//!< e23: steering steps
	PRBS,		//!< e23: held-out PRBS
	STOP		//!< brake 1 until below m_fSpeedTolKmh
}

//------------------------------------------------------------------------------------------------
//! What a phase does
enum DC_ESteerPhaseKind
{
	FLAT,		//!< e21 zero-command runs on the flat site
	CROSS,		//!< e21 zero-command runs on the cross-slope site
	STAIR,		//!< e22 step-0 staircase
	LOAD,		//!< e22 loaded cornering and braking in a turn
	STEPS		//!< e23 step train and PRBS at one speed
}

//------------------------------------------------------------------------------------------------
//! One phase of a steering probe, in run order
class DC_ProbeSteerPhase
{
	DC_ESteerPhaseKind m_eKind;
	string m_sName;
	string m_sSite = "flat";				//!< flat | cross: the site the phase drives on
	int m_iIdx = -1;
	float m_fTargetKmh;
	float m_fSteer;							//!< e22 LOAD: the fixed command
	int m_iAssist = -1;						//!< -1 keep the mode, else 0 NONE, 1 PARTIAL, 2 FULL for this phase
	float m_fHeldM;							//!< e21: distance of MEASURE / NOCALL [m]
	float m_fNocallM;
	float m_fStartAlongM;
	float m_fLateralM;
	ref array<float> m_aLevels = {};		//!< e22 STAIR: steering levels; LOAD: a_lat targets
	ref array<float> m_aEvents = {};		//!< e23: signed step amplitudes in run order
	float m_fPrbsAmplitude;
	int m_iEventCursor;						//!< e23: next event
	bool m_bPrbsDone;
	int m_iRuns;							//!< runs (reset .. stop) started in this phase
	int m_iSpins;
	int m_iStartStep = -1;
	int m_iEndStep = -1;
	float m_fStartWorldMs = -1;
	float m_fEndWorldMs = -1;
	string m_sResult;						//!< ok aborted_area overspeed aborted_heading no_speed engine_off spin stop_cap skipped stopped error
	string m_sNote;
	float m_fMaxSpeedKmh;
	float m_fTurnDeg;
	int m_iEngineOffSteps;
	int m_iStallsAtStart;
	int m_iRestartsAtStart;
	int m_iSleepWakesAtStart;

	//------------------------------------------------------------------------------------------------
	void DC_ProbeSteerPhase(DC_ESteerPhaseKind kind, string name, string site, float targetKmh)
	{
		m_eKind = kind;
		m_sName = name;
		m_sSite = site;
		m_fTargetKmh = targetKmh;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends one note token (only [A-Za-z0-9_.:/+-] characters)
	void AddNote(string token)
	{
		if (m_sNote.IsEmpty())
			m_sNote = token;
		else
			m_sNote += "/" + token;
	}

	//------------------------------------------------------------------------------------------------
	//! The first abnormal run result is the phase result
	void SetRunResult(string result)
	{
		if (m_sResult.IsEmpty() || m_sResult == "ok")
			m_sResult = result;
	}
}

//------------------------------------------------------------------------------------------------
class DC_ProbeSteerBaseClass : DC_ProbeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeSteerBase : DC_ProbeComponent
{
	static const string SITE_FLAT = "flat";
	static const string SITE_CROSS = "cross";
	//! Safety cap of a stop segment [s]
	static const float STOP_CAP_S = 60;
	//! Steps with |yaw rate| above m_fSpinYawRateRadS that end a run with spin
	static const int SPIN_STEPS = 8;
	//! A driving segment ends with overspeed above GetMaxSpeedKmh() + this [km/h]
	static const float OVERSPEED_KMH = 3;

	// === SITE (flat; the explicit candidate is the e01 test area) ===
	[Attribute("659.6 0 2572.4", UIWidgets.Coords, "Flat site, explicit candidate: start (x, z used): the e01 test area on the Arland main runway (engine road 83, 60 m x 1217 m)", category: "Site")]
	protected vector m_vFlatStart;

	[Attribute("75.11", UIWidgets.Slider, "Flat site, explicit candidate: heading [deg], 0 = +Z, clockwise from above", "0 360 0.01", category: "Site")]
	protected float m_fFlatHeadingDeg;

	[Attribute("1000", UIWidgets.Slider, "Flat site length [m]: the longest run of e21 / e22 / e23 fits (S2 at 60 km/h ~750 m)", "100 3000 10", category: "Site")]
	protected float m_fFlatLengthM;

	[Attribute("5", UIWidgets.Slider, "Site terrain sample step along the strip [m]", "1 50 0.5", category: "Site")]
	protected float m_fSiteStepM;

	[Attribute("0.005", UIWidgets.EditBox, "Flat site: max |mean grade| (centre-line regression)", category: "Site")]
	protected float m_fFlatGradeMax;

	[Attribute("0.01", UIWidgets.EditBox, "Flat site: max |cross-slope| at every station, and max |grade| over any 20 m window", category: "Site")]
	protected float m_fFlatCrossMax;

	[Attribute("0.10", UIWidgets.EditBox, "Flat site: max rms of the surface about its plane fit [m]", category: "Site")]
	protected float m_fFlatRmsMax;

	[Attribute("", UIWidgets.EditBox, "Obstacle sweeps: class or prefab names (exact, ';' separated) that are never obstacles; a hit on one is excluded and the sweep traced again. Empty by default: every hit rejects the site", category: "Site")]
	protected string m_sBenignColliders;

	[Attribute("4", UIWidgets.Slider, "Stopping envelope: braking deceleration assumed after the car leaves the area [m/s2] (e01: brake 0.3 gave 4.4 m/s2)", "1 15 0.5", category: "Site")]
	protected float m_fEnvelopeDecelMs2;

	[Attribute("0.2", UIWidgets.Slider, "Stopping envelope: reaction time before the brake acts [s]", "0 2 0.05", category: "Site")]
	protected float m_fEnvelopeReactS;

	[Attribute("30", UIWidgets.Slider, "Stopping envelope: largest heading off the strip direction when the car leaves the area [deg] (lateral envelope = stopping distance x sin)", "0 90 1", category: "Site")]
	protected float m_fEnvelopeAngleDeg;

	[Attribute("3", UIWidgets.Slider, "Area margin [m]", "0 20 0.5", category: "Site")]
	protected float m_fAreaMarginM;

	// === SEARCH (grid fallback and budgets) ===
	[Attribute("600 0 780", UIWidgets.Coords, "Grid search region, min corner (x, z): the extent of the Arland road network (E-08a dump)", category: "Search")]
	protected vector m_vGridMin;

	[Attribute("3550 0 3520", UIWidgets.Coords, "Grid search region, max corner (x, z)", category: "Search")]
	protected vector m_vGridMax;

	[Attribute("50", UIWidgets.Slider, "Grid spacing [m]", "10 500 5", category: "Search")]
	protected float m_fGridSpacingM;

	[Attribute("6000", UIWidgets.Slider, "GetSurfaceY calls per render frame, shared by every search of the probe (terrain samples and sweep end points; a strip continues in the next frame)", "100 50000 100", category: "Search")]
	protected int m_iSiteSamplesPerFrame;

	[Attribute("120", UIWidgets.Slider, "TracePosition / TraceMove calls per render frame, shared by every search of the probe (benign retries included)", "1 5000 1", category: "Search")]
	protected int m_iSiteTracesPerFrame;

	[Attribute("3000", UIWidgets.Slider, "Max strips the grid generates per search", "10 50000 10", category: "Search")]
	protected int m_iSiteMaxStrips;

	[Attribute("48", UIWidgets.Slider, "Max candidates checked for obstacles per search", "1 1000 1", category: "Search")]
	protected int m_iSiteMaxObstacleChecks;

	[Attribute("3", UIWidgets.Slider, "Every surface sample of a site above this y [m] (sea level 0: no shore, no water)", "-100 500 0.5", category: "Search")]
	protected float m_fSiteMinSurfaceY;

	// === DRIVE ===
	[Attribute("0.3", UIWidgets.Slider, "Teleport height above GetSurfaceY [m]", "0 5 0.05", category: "Drive")]
	protected float m_fSpawnLiftM;

	[Attribute("3", UIWidgets.Slider, "Settle after each teleport, brake 1 + handbrake [s]", "0.5 30 0.5", category: "Drive")]
	protected float m_fSettleS;

	[Attribute("0.08", UIWidgets.Slider, "Probe speed hold: proportional gain [1/(km/h)] (probe-only PI, as e01)", "0 1 0.005", category: "Drive")]
	protected float m_fSpeedKp;

	[Attribute("0.02", UIWidgets.Slider, "Probe speed hold: integral gain [1/(km/h s)]", "0 1 0.005", category: "Drive")]
	protected float m_fSpeedKi;

	[Attribute("0.5", UIWidgets.Slider, "Probe speed hold: brake = -PI output x this", "0 2 0.05", category: "Drive")]
	protected float m_fBrakeScale;

	[Attribute("1", UIWidgets.Slider, "Speed tolerance [km/h]: accel ends at target - tol, stop below tol", "0.1 10 0.1", category: "Drive")]
	protected float m_fSpeedTolKmh;

	[Attribute("40", UIWidgets.Slider, "Time to reach the target speed before a run ends with no_speed [s]", "2 120 1", category: "Drive")]
	protected float m_fReachS;

	[Attribute("2", UIWidgets.Slider, "Spin guard: |yaw rate| above this for 8 steps ends a run with spin [rad/s] (full lock at 30 km/h turns at about 1 rad/s)", "0.2 10 0.1", category: "Drive")]
	protected float m_fSpinYawRateRadS;

	[Attribute("1", UIWidgets.Slider, "Turn direction of positive steering (+1 right) when the probe has not measured it (e01 and e02a measured +1 on 1.8.0.13 and 1.9.0.37)", "-1 1 2", category: "Drive")]
	protected int m_iAssumedSteerSign;

	// === WHEEL ANGLE (DC_SteerChannels) ===
	[Attribute("v_wheel_L01", UIWidgets.EditBox, "Front-left wheel pivot bone (S105_base.et WheelPosition Wheel_L01)", category: "Wheel angle")]
	protected string m_sWheelBoneL;

	[Attribute("v_wheel_R01", UIWidgets.EditBox, "Front-right wheel pivot bone", category: "Wheel angle")]
	protected string m_sWheelBoneR;

	[Attribute("_rotator", UIWidgets.EditBox, "Suffix of the kingpin pivot bones (1.9 KingpinAxes: v_wheel_L01_rotator)", category: "Wheel angle")]
	protected string m_sRotatorSuffix;

	[Attribute("Wheel_L01", UIWidgets.EditBox, "Front-left wheel slot (SlotManagerComponent)", category: "Wheel angle")]
	protected string m_sWheelSlotL;

	[Attribute("Wheel_R01", UIWidgets.EditBox, "Front-right wheel slot", category: "Wheel angle")]
	protected string m_sWheelSlotR;

	[Attribute("steering0", UIWidgets.EditBox, "Steering signal of the vehicle (S105 AnimVariables)", category: "Wheel angle")]
	protected string m_sSteerSignal;

	[Attribute("0", UIWidgets.Slider, "Simulation index of the front-left wheel (0 = Axle0L)", "0 15 1", category: "Wheel angle")]
	protected int m_iWheelIndexL;

	[Attribute("1", UIWidgets.Slider, "Simulation index of the front-right wheel (1 = Axle0R)", "0 15 1", category: "Wheel angle")]
	protected int m_iWheelIndexR;

	// === CAMERA ===
	[Attribute("1", UIWidgets.CheckBox, "Chase camera (DC_RigR1) on the car while the probe runs, so it is animated at full detail", category: "Camera")]
	protected bool m_bChaseCamera;

	[Attribute("0 2.8 -7.5", UIWidgets.Coords, "Chase camera mount, vehicle-local [m] (x right, y up, z forward)", category: "Camera")]
	protected vector m_vChaseMount;

	[Attribute("-14", UIWidgets.Slider, "Chase camera pitch [deg], negative = down", "-89 89 0.5", category: "Camera")]
	protected float m_fChasePitchDeg;

	[Attribute("60", UIWidgets.Slider, "Chase camera vertical FOV [deg]", "10 120 1", category: "Camera")]
	protected float m_fChaseFovDeg;

	// === PROBE STATE ===
	protected string m_sActiveProbe;
	protected ref DC_ProbeContext m_Ctx;
	protected ref DC_ProbeActuator m_Actuator;
	protected ref DC_SteerChannels m_Channels;
	protected ref DC_RigR1 m_Rig;
	protected ref DC_ProbeArea m_Area;
	protected ref DC_ProbeRow m_Row;
	protected ref DC_ProbeCsv m_StepsCsv;
	protected ref DC_ProbeCsv m_PhasesCsv;
	protected ref array<ref DC_ProbeSteerPhase> m_aPhases;
	protected ref array<ref DC_SteerSiteSearch> m_aSearches;
	protected ref array<string> m_aSearchKinds;
	protected ref array<IEntity> m_aExcludes;
	protected int m_iCursor;
	protected int m_iPhasesStarted;
	protected bool m_bProbeDone;
	protected DC_ESteerSegment m_eSegment;
	protected int m_iSegStep;
	protected int m_iPhaseStep;
	protected bool m_bTeleportDone;
	protected int m_iLastStep;
	protected bool m_bHasPrevHeading;
	protected float m_fPrevHeadingDeg;
	protected float m_fSegStartAlongM;
	protected float m_fSegTarget = -1;		//!< column seg_target
	protected bool m_bSegFlag;				//!< column seg_flag
	protected int m_iEventIdx = -1;			//!< column event_idx
	protected int m_iSpinSteps;
	protected int m_iLastCovered = 1;		//!< engine steps the last dispatched step covered (CoveredSteps)
	protected ref DC_SteerBudget m_Budget = new DC_SteerBudget();
	protected float m_fSteerSignSum;		//!< sum of cmd_steer x heading rate on the steering segments
	protected int m_iSteerSignSamples;
	protected int m_iMeasuredSteerSign;

	// Step counts from the attributes and the runner's step length at the start
	protected float m_fDt;
	protected int m_iSettleSteps;
	protected int m_iReachSteps;
	protected int m_iStopCapSteps;

	// Driving assistance (a global static of the engine): the original mode and the phase overrides
	protected EVehicleDrivingAssistanceMode m_eAssistOriginal;
	protected bool m_bAssistChanged;
	protected int m_iAssistSets;
	protected int m_iAssistRestores;
	protected int m_iAssistOverride = -1;		//!< the mode a phase set, -1 none

	//------------------------------------------------------------------------------------------------
	//! The one probe id of the subclass
	string GetSteerProbeId()
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	float GetSteerTimeoutS()
	{
		return 240;
	}

	//------------------------------------------------------------------------------------------------
	//! Half width of the flat site's area for this probe [m]: terrain-checked, the run aborts outside it (minus the
	//! margin), obstacle-swept with the stopping envelope added
	float GetFlatHalfWidthM()
	{
		return 10;
	}

	//------------------------------------------------------------------------------------------------
	//! The probe's top speed [km/h]: the stopping envelope, and a driving segment ends with overspeed above it + 3
	float GetMaxSpeedKmh()
	{
		return 40;
	}

	//------------------------------------------------------------------------------------------------
	//! The largest heading off the strip direction the probe drives at [deg]: the lateral envelope, and (unless
	//! AllowsAnyHeading) a driving segment ends with aborted_heading beyond it
	float GetEnvelopeAngleDeg()
	{
		if (AllowsAnyHeading())
			return 90;

		return m_fEnvelopeAngleDeg;
	}

	//------------------------------------------------------------------------------------------------
	//! true for probes that drive circles (e22): the envelope then covers every exit direction, no heading guard
	bool AllowsAnyHeading()
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Fills m_aPhases; false (with the reason in failure) on an attribute that does not parse
	bool BuildSteerPlan(out string failure)
	{
		failure = "no plan";
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Adds the subclass's searches besides the flat one (AddSearch); e21 adds the cross-slope site
	void AddExtraSearches()
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Called once every search has ended (before the first phase); a subclass may start another search here and
	//! return true to keep searching (e21: the shorter cross-slope fallback)
	bool OnSearchesDone()
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! The command of a subclass segment (anything after ACCEL except STOP)
	void CommandSegment(notnull DC_ProbeSteerPhase phase)
	{
		m_Actuator.SetSpeedHold(phase.m_fTargetKmh, 0);
	}

	//------------------------------------------------------------------------------------------------
	//! Transitions of a subclass segment (after the common engine / area / spin checks)
	void AdvanceSegment(notnull DC_ProbeSteerPhase phase)
	{
		EnterSegment(DC_ESteerSegment.STOP);
	}

	//------------------------------------------------------------------------------------------------
	//! The segment that follows a completed accel
	DC_ESteerSegment AfterAccel(notnull DC_ProbeSteerPhase phase)
	{
		return DC_ESteerSegment.STOP;
	}

	//------------------------------------------------------------------------------------------------
	//! true: the phase continues with another run (reset .. stop) after this stop (e23 resumes its train)
	bool WantsAnotherRun(notnull DC_ProbeSteerPhase phase)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Before the first run of a phase (after the phase counters are set)
	void OnBeginPhase(notnull DC_ProbeSteerPhase phase)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! At the start of each run of a phase (before the reset)
	void OnBeginRun(notnull DC_ProbeSteerPhase phase)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! true for segments whose steering counts for the steer-sign measurement
	bool IsSteeringSegment(DC_ESteerSegment segment)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! A subclass's extra CSVs (opened after the steps and phases CSVs); false refuses the start
	bool OpenExtraCsvs(notnull DC_ProbeContext ctx)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	void CloseExtraCsvs()
	{
	}

	//------------------------------------------------------------------------------------------------
	//! The subclass's attributes as key=value (params.*)
	void AppendProbeParams(notnull array<string> outKeyValues)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! The subclass's run header lines at the start
	void AppendProbeHeader(notnull array<string> lines)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! The subclass's run header lines at the stop
	void AppendProbeSummary(notnull array<string> lines)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! A run that ended in a segment of the subclass (aborted, spin): the subclass records it (e23: the open event)
	void OnRunEnded(notnull DC_ProbeSteerPhase phase, string result)
	{
	}

	//------------------------------------------------------------------------------------------------
	override void GetProbeIds(notnull array<string> outIds)
	{
		string id = GetSteerProbeId();
		if (!id.IsEmpty())
			outIds.Insert(id);
	}

	//------------------------------------------------------------------------------------------------
	override bool IsConcurrent(string probeId)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	override bool NeedsEgo(string probeId)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override bool NeedsSteps(string probeId)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override float GetProbeTimeoutS(string probeId)
	{
		return GetSteerTimeoutS();
	}

	//------------------------------------------------------------------------------------------------
	//! Every attribute of the base, then the subclass's (AppendProbeParams), dt_s
	override void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_vFlatStart=" + DC_ProbeArea.FormatVector(m_vFlatStart, 3));
		outKeyValues.Insert("m_fFlatHeadingDeg=" + m_fFlatHeadingDeg.ToString(-1, 4));
		outKeyValues.Insert("m_fFlatLengthM=" + m_fFlatLengthM.ToString(-1, 2));
		outKeyValues.Insert("m_fSiteStepM=" + m_fSiteStepM.ToString(-1, 2));
		float flatHalfWidth = GetFlatHalfWidthM();
		float maxKmh = GetMaxSpeedKmh();
		outKeyValues.Insert("flat_half_width_m=" + flatHalfWidth.ToString(-1, 2));
		outKeyValues.Insert("max_speed_kmh=" + maxKmh.ToString(-1, 1));
		outKeyValues.Insert("m_fFlatGradeMax=" + m_fFlatGradeMax.ToString(-1, 5));
		outKeyValues.Insert("m_fFlatCrossMax=" + m_fFlatCrossMax.ToString(-1, 5));
		outKeyValues.Insert("m_fFlatRmsMax=" + m_fFlatRmsMax.ToString(-1, 4));
		outKeyValues.Insert("m_sBenignColliders=" + m_sBenignColliders);
		outKeyValues.Insert("m_fEnvelopeDecelMs2=" + m_fEnvelopeDecelMs2.ToString(-1, 3));
		outKeyValues.Insert("m_fEnvelopeReactS=" + m_fEnvelopeReactS.ToString(-1, 3));
		outKeyValues.Insert("m_fEnvelopeAngleDeg=" + m_fEnvelopeAngleDeg.ToString(-1, 2));
		outKeyValues.Insert("m_fAreaMarginM=" + m_fAreaMarginM.ToString(-1, 3));
		outKeyValues.Insert("m_vGridMin=" + DC_ProbeArea.FormatVector(m_vGridMin, 1));
		outKeyValues.Insert("m_vGridMax=" + DC_ProbeArea.FormatVector(m_vGridMax, 1));
		outKeyValues.Insert("m_fGridSpacingM=" + m_fGridSpacingM.ToString(-1, 2));
		outKeyValues.Insert("m_iSiteSamplesPerFrame=" + m_iSiteSamplesPerFrame.ToString());
		outKeyValues.Insert("m_iSiteTracesPerFrame=" + m_iSiteTracesPerFrame.ToString());
		outKeyValues.Insert("m_iSiteMaxStrips=" + m_iSiteMaxStrips.ToString());
		outKeyValues.Insert("m_iSiteMaxObstacleChecks=" + m_iSiteMaxObstacleChecks.ToString());
		outKeyValues.Insert("m_fSiteMinSurfaceY=" + m_fSiteMinSurfaceY.ToString(-1, 2));
		outKeyValues.Insert("m_fSpawnLiftM=" + m_fSpawnLiftM.ToString(-1, 3));
		outKeyValues.Insert("m_fSettleS=" + m_fSettleS.ToString(-1, 3));
		outKeyValues.Insert("m_fSpeedKp=" + m_fSpeedKp.ToString(-1, 4));
		outKeyValues.Insert("m_fSpeedKi=" + m_fSpeedKi.ToString(-1, 4));
		outKeyValues.Insert("m_fBrakeScale=" + m_fBrakeScale.ToString(-1, 4));
		outKeyValues.Insert("m_fSpeedTolKmh=" + m_fSpeedTolKmh.ToString(-1, 3));
		outKeyValues.Insert("m_fReachS=" + m_fReachS.ToString(-1, 3));
		outKeyValues.Insert("m_fSpinYawRateRadS=" + m_fSpinYawRateRadS.ToString(-1, 3));
		outKeyValues.Insert("m_iAssumedSteerSign=" + m_iAssumedSteerSign.ToString());
		outKeyValues.Insert("m_sWheelBoneL=" + m_sWheelBoneL);
		outKeyValues.Insert("m_sWheelBoneR=" + m_sWheelBoneR);
		outKeyValues.Insert("m_sRotatorSuffix=" + m_sRotatorSuffix);
		outKeyValues.Insert("m_sWheelSlotL=" + m_sWheelSlotL);
		outKeyValues.Insert("m_sWheelSlotR=" + m_sWheelSlotR);
		outKeyValues.Insert("m_sSteerSignal=" + m_sSteerSignal);
		outKeyValues.Insert("m_iWheelIndexL=" + m_iWheelIndexL.ToString());
		outKeyValues.Insert("m_iWheelIndexR=" + m_iWheelIndexR.ToString());
		outKeyValues.Insert("m_bChaseCamera=" + DC_LaunchGuard.BoolText(m_bChaseCamera));
		outKeyValues.Insert("m_vChaseMount=" + DC_ProbeArea.FormatVector(m_vChaseMount, 3));
		outKeyValues.Insert("m_fChasePitchDeg=" + m_fChasePitchDeg.ToString(-1, 3));
		outKeyValues.Insert("m_fChaseFovDeg=" + m_fChaseFovDeg.ToString(-1, 3));
		AppendProbeParams(outKeyValues);
		float dt = GetStepDtS();
		outKeyValues.Insert("dt_s=" + dt.ToString(-1, 6));
	}

	//------------------------------------------------------------------------------------------------
	override bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		if (probeId != GetSteerProbeId())
			return false;

		ResetSteerState();
		m_Ctx = ctx;
		IEntity ego = ctx.GetEgo();
		if (!ego)
			return RefuseStart("no ego entity (DC_Ego) in the layer - probe not started");

		World world = GetGame().GetWorld();
		if (!world)
			return RefuseStart("no game world - probe not started");

		m_fDt = GetStepDtS();
		if (m_fDt <= 0)
			return RefuseStart("step length (GetStepDtS) is " + m_fDt.ToString(-1, 6) + " s - probe not started");

		m_iSettleSteps = DC_ProbeSequences.StepsFor(m_fSettleS, m_fDt);
		m_iReachSteps = DC_ProbeSequences.StepsFor(m_fReachS, m_fDt);
		m_iStopCapSteps = DC_ProbeSequences.StepsFor(STOP_CAP_S, m_fDt);

		// Mode A as the image probes drive it: the deployed launch clutch in mode slip (DC_ProbeActuator.Init)
		m_Actuator = new DC_ProbeActuator();
		if (!m_Actuator.Init(ego))
			return RefuseStart("actuator: " + m_Actuator.GetFailure() + " - probe not started");

		m_Actuator.SetSpeedGains(m_fSpeedKp, m_fSpeedKi, m_fBrakeScale);
		int decision = -1;
		DC_ProbeSequences sequences = DC_ProbeSequences.Cast(GetOwner().FindComponent(DC_ProbeSequences));
		if (sequences)
			decision = sequences.GetForceDecision();

		if (decision == 1)
			m_Actuator.ForceSimulation();

		m_Actuator.SetStuckForce(DC_ProbeActuator.STUCK_AFTER_S, DC_ProbeActuator.STUCK_BELOW_KMH);

		m_Channels = new DC_SteerChannels();
		m_Channels.InitChannels(ego, m_Actuator.GetSimulation(), m_sWheelBoneL, m_sWheelBoneR, m_sRotatorSuffix, m_sWheelSlotL, m_sWheelSlotR, m_sSteerSignal, m_iWheelIndexL, m_iWheelIndexR);

		m_aPhases = {};
		string planFailure;
		if (!BuildSteerPlan(planFailure))
			return RefuseStart(planFailure + " - probe not started");

		if (m_aPhases.IsEmpty())
			return RefuseStart("the plan has no phase - probe not started");

		// The obstacle traces ignore the car and everything attached to it
		m_aExcludes = {};
		CollectHierarchy(ego, m_aExcludes);
		m_aSearches = {};
		m_aSearchKinds = {};
		DC_SteerSiteCriteria flat = BuildFlatCriteria();
		DC_SteerSiteSearch flatSearch = new DC_SteerSiteSearch();
		flatSearch.Configure(flat, m_aExcludes, m_iSiteMaxStrips, m_iSiteMaxObstacleChecks);
		flatSearch.AddExplicit(m_vFlatStart, m_fFlatHeadingDeg);
		flatSearch.SetGrid(m_vGridMin[0], m_vGridMin[2], m_vGridMax[0], m_vGridMax[2], m_fGridSpacingM, 15, 0);
		AddSearch(SITE_FLAT, flatSearch);
		AddExtraSearches();

		m_Row = new DC_ProbeRow();
		array<string> stepColumns = {"physics_step", "world_time_ms", "timeslice_s", "phase", "segment", "phase_step", "run_idx"};
		DC_ProbeActuator.AddStepColumns(stepColumns);
		stepColumns.Insert("steer_call");
		stepColumns.Insert("assist_mode");
		stepColumns.Insert("site");
		stepColumns.Insert("along_m");
		stepColumns.Insert("lateral_m");
		stepColumns.Insert("in_area");
		stepColumns.Insert("seg_target");
		stepColumns.Insert("seg_flag");
		stepColumns.Insert("event_idx");
		DC_SteerChannels.AddColumns(stepColumns);
		m_StepsCsv = ctx.OpenCsv(probeId + "_steps", stepColumns);
		if (!m_StepsCsv)
			return RefuseStart(probeId + "_steps.csv could not be created");

		array<string> phaseColumns = {"phase_idx", "phase", "kind", "site", "start_step", "end_step", "start_world_ms", "end_world_ms", "target_kmh", "steer", "assist", "runs", "result", "note"};
		m_PhasesCsv = ctx.OpenCsv(probeId + "_phases", phaseColumns);
		if (!m_PhasesCsv)
			return RefuseStart(probeId + "_phases.csv could not be created");

		if (!OpenExtraCsvs(ctx))
			return RefuseStart("the probe's extra CSVs could not be created");

		m_eAssistOriginal = BaseVehicleControllerComponent.GetDrivingAssistanceMode();
		array<string> lines = {};
		AppendVehicleLines(lines, ego);
		m_Actuator.AppendGearboxLines(lines);
		m_Channels.AppendHeaderLines(lines);
		int assistOriginal = m_eAssistOriginal;
		lines.Insert("assist.original=" + assistOriginal.ToString());
		lines.Insert("assist.original_name=" + typename.EnumToString(EVehicleDrivingAssistanceMode, assistOriginal));
		lines.Insert("assist.rule=BaseVehicleControllerComponent.SetDrivingAssistanceMode (an engine global, in memory only, no settings file): a phase with assist >= 0 sets it for its own duration; the original is set back at the end of that phase and at every probe stop, world teardown included");
		lines.Insert("drive.force_sim_e01=" + DC_ProbeSequences.ForceDecisionName(decision));
		lines.Insert("drive.force_sim=" + DC_LaunchGuard.BoolText(m_Actuator.IsSimulationForced()));
		lines.Insert("drive.force_rule=ForceEnableSimulation latched only when e01 of this Play found it needed (decision 1); else the stuck guard (10 s below 3 km/h after a drive start)");
		lines.Insert("steer.steer_call_rule=steer_call 1: SetSteering(cmd_steer) on that step; 0: no SetSteering call (S1 no-call variant), the input keeps whatever the engine makes of it");
		lines.Insert("steer.frame_rule=along_m / lateral_m in the frame of the phase's site (start, heading), lateral right positive");
		AppendProbeHeader(lines);
		if (!ctx.AppendHeaderLines(lines))
			LogWarning("the vehicle / steer / assist lines could not be appended to the run header");

		if (m_bChaseCamera)
		{
			m_Rig = new DC_RigR1();
			if (!m_Rig.Attach(ego, m_vChaseMount, m_fChasePitchDeg, m_fChaseFovDeg, 0.1, 3000))
			{
				LogWarning("chase camera: " + m_Rig.GetFailure() + " - the probe runs without it (the bone and entity wheel-angle sources may not update at a distance)");
				m_Rig = null;
			}
		}

		m_sActiveProbe = probeId;
		m_eSegment = DC_ESteerSegment.SEARCH;
		ctx.Log(m_Channels.DescribeChannels());
		ctx.Log(m_Actuator.DescribeGearbox());
		ctx.Log(string.Format("dt %1 s, settle %2 steps, %3 phases planned, %4 site searches; assistance %5", m_fDt.ToString(-1, 6), m_iSettleSteps, m_aPhases.Count(), m_aSearches.Count(), typename.EnumToString(EVehicleDrivingAssistanceMode, assistOriginal)));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The attribute values that make a later run (the 1.8 cold baseline) drive exactly this site as its explicit
	//! candidate (site.<kind>.reuse)
	string SiteReuseText(string kind, notnull DC_SteerSite site)
	{
		float x = site.m_vStart[0];
		float z = site.m_vStart[2];
		if (kind == SITE_FLAT)
			return "m_vFlatStart=" + x.ToString(-1, 3) + " 0 " + z.ToString(-1, 3) + "; m_fFlatHeadingDeg=" + site.m_fHeadingDeg.ToString(-1, 4);

		return ExtraSiteReuseText(kind, site);
	}

	//------------------------------------------------------------------------------------------------
	//! The reuse text of the subclass's own site kinds (e21: the cross site)
	string ExtraSiteReuseText(string kind, notnull DC_SteerSite site)
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! The flat site's criteria from the Site attributes (zone = the whole strip; the probe's area half width)
	protected DC_SteerSiteCriteria BuildFlatCriteria()
	{
		DC_SteerSiteCriteria c = new DC_SteerSiteCriteria();
		c.m_sKind = SITE_FLAT;
		c.m_fLengthM = m_fFlatLengthM;
		SetSiteGeometry(c, GetFlatHalfWidthM());
		c.m_fZoneFromM = 0;
		c.m_fZoneToM = m_fFlatLengthM;
		c.m_fGradeMax = m_fFlatGradeMax;
		c.m_fGradeWindowMax = m_fFlatCrossMax;
		c.m_fGradeWindowMaxAll = m_fFlatCrossMax;
		c.m_fCrossMin = 0;
		c.m_fCrossMax = m_fFlatCrossMax;
		c.m_fCrossAbsMaxAll = m_fFlatCrossMax;
		c.m_fRmsMax = m_fFlatRmsMax;
		c.m_fMinSurfaceY = m_fSiteMinSurfaceY;
		return c;
	}

	//------------------------------------------------------------------------------------------------
	//! Area half width; the tyre-track corridor (the area minus the margin): an odd number of positions, the centre one
	//! on the centre line, at most 2 x track_half apart (the tyre samples then cover the corridor without gaps); the
	//! sample step; the stopping envelope of the probe's capped top speed (GetMaxSpeedKmh) along, x sin of the largest
	//! heading the probe allows (GetEnvelopeAngleDeg) sideways, and backwards too when any heading is allowed; the
	//! benign colliders
	protected void SetSiteGeometry(notnull DC_SteerSiteCriteria c, float halfWidthM)
	{
		c.m_fHalfWidthM = halfWidthM;
		c.m_fCorridorHalfM = Math.Max(halfWidthM - m_fAreaMarginM, 0);
		int half = Math.Ceil(c.m_fCorridorHalfM / (2 * c.m_fTrackHalfM));
		c.m_iCorridorPoints = 2 * half + 1;
		c.m_fStepM = m_fSiteStepM;
		float v = GetMaxSpeedKmh() / 3.6;
		float stopping = v * v / (2 * Math.Max(m_fEnvelopeDecelMs2, 0.5)) + v * m_fEnvelopeReactS;
		c.m_fEnvelopeLongM = stopping;
		c.m_fEnvelopeLatM = stopping * Math.Sin(GetEnvelopeAngleDeg() * Math.DEG2RAD);
		c.m_fEnvelopeBackM = 0;
		if (AllowsAnyHeading())
			c.m_fEnvelopeBackM = stopping;
		c.m_aBenign = {};
		array<string> names = {};
		m_sBenignColliders.Split(";", names, true);
		foreach (string name : names)
		{
			string trimmed = name.Trim();
			if (!trimmed.IsEmpty())
				c.m_aBenign.Insert(trimmed);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void AddSearch(string kind, notnull DC_SteerSiteSearch search)
	{
		m_aSearchKinds.Insert(kind);
		m_aSearches.Insert(search);
	}

	//------------------------------------------------------------------------------------------------
	//! The site found for kind, null when none (or the search has not ended)
	protected DC_SteerSite GetSite(string kind)
	{
		int index = m_aSearchKinds.Find(kind);
		if (index < 0)
			return null;

		DC_SteerSiteSearch search = m_aSearches[index];
		if (!search.IsDone())
			return null;

		return search.GetSelected();
	}

	//------------------------------------------------------------------------------------------------
	//! Replaces the search of kind (e21's shorter fallback)
	protected void ReplaceSearch(string kind, notnull DC_SteerSiteSearch search)
	{
		int index = m_aSearchKinds.Find(kind);
		if (index < 0)
		{
			AddSearch(kind, search);
			return;
		}

		m_aSearches[index] = search;
	}

	//------------------------------------------------------------------------------------------------
	//! root and all its descendants (GetChildren / GetSibling), for the obstacle traces' ExcludeArray
	protected static void CollectHierarchy(notnull IEntity root, notnull array<IEntity> outEntities)
	{
		array<IEntity> stack = {};
		stack.Insert(root);
		while (!stack.IsEmpty())
		{
			int last = stack.Count() - 1;
			IEntity entity = stack[last];
			stack.Remove(last);
			outEntities.Insert(entity);
			IEntity child = entity.GetChildren();
			while (child)
			{
				stack.Insert(child);
				child = child.GetSibling();
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	//! vehicle.* lines (sensitivities and assistance read, never set; gearbox facts), as DC_ProbeSequences writes them
	protected void AppendVehicleLines(notnull array<string> lines, notnull IEntity ego)
	{
		int assistance = BaseVehicleControllerComponent.GetDrivingAssistanceMode();
		lines.Insert("vehicle.driving_assistance_mode=" + assistance.ToString());
		lines.Insert("vehicle.driving_assistance_mode_name=" + typename.EnumToString(EVehicleDrivingAssistanceMode, assistance));
		CarControllerComponent controller = m_Actuator.GetController();
		VehicleWheeledSimulation simulation = m_Actuator.GetSimulation();
		if (controller)
		{
			lines.Insert("vehicle.has_automatic_gearbox=" + DC_LaunchGuard.BoolText(controller.HasAutomaticGearbox()));
			lines.Insert("vehicle.persistent_handbrake=" + DC_LaunchGuard.BoolText(controller.GetPersistentHandBrake()));
		}

		if (!simulation)
			return;

		float noise = simulation.GetNoiseSteerSensitivity();
		float roughness = simulation.GetRoughnessSensitivity();
		lines.Insert("vehicle.noise_steer_sensitivity=" + noise.ToString(-1, 4));
		lines.Insert("vehicle.roughness_sensitivity=" + roughness.ToString(-1, 4));
		int wheels = simulation.WheelCount();
		lines.Insert("vehicle.wheel_count=" + wheels.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! The search runs here (render frames), then the teleports of the reset segment
	override void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Actuator)
			return;

		if (m_eSegment == DC_ESteerSegment.SEARCH)
		{
			StepSearches();
			return;
		}

		if (m_eSegment != DC_ESteerSegment.RESET || m_bTeleportDone)
			return;

		DC_ProbeSteerPhase phase = GetPhase();
		if (!phase || !m_Area)
			return;

		vector target = m_Area.PointAt(phase.m_fStartAlongM, phase.m_fLateralM);
		if (!m_Actuator.Teleport(target, m_Area.m_fHeadingDeg, m_fSpawnLiftM))
		{
			LogError(string.Format("phase %1: teleport of DC_Ego failed (vehicle gone?) - probe ends", phase.m_sName));
			phase.m_sResult = "error";
			phase.AddNote("teleport_failed");
			FinishPhaseRecord(phase);
			m_bProbeDone = true;
			return;
		}

		m_Actuator.StartModeA();
		m_bTeleportDone = true;
		m_bHasPrevHeading = false;
	}

	//------------------------------------------------------------------------------------------------
	//! One budget slice of every unfinished search; when all have ended, the site lines and the first phase
	protected void StepSearches()
	{
		World world = GetGame().GetWorld();
		if (!world)
			return;

		// one budget for this frame, shared: each unfinished search takes what the ones before it left
		m_Budget.Reset(m_iSiteSamplesPerFrame, m_iSiteTracesPerFrame);
		bool allDone = true;
		foreach (DC_SteerSiteSearch search : m_aSearches)
		{
			if (search.IsDone())
				continue;

			if (!search.StepSearch(world, m_Budget))
				allDone = false;
		}

		if (!allDone)
			return;

		if (OnSearchesDone())
			return;

		array<string> lines = {};
		foreach (int i, DC_SteerSiteSearch done : m_aSearches)
		{
			string kind = m_aSearchKinds[i];
			done.AppendHeaderLines(lines, "site." + kind);
			LogInfo(done.DescribeSearch());
			DC_SteerSite selected = done.GetSelected();
			if (selected)
				lines.Insert("site." + kind + ".reuse=" + SiteReuseText(kind, selected));

			if (selected && selected.m_iBenignHits > 0)
				LogInfo(string.Format("the %1 site was swept with %2 benign hits (m_sBenignColliders)", kind, selected.m_iBenignHits));

			if (!selected)
				LogWarning(string.Format("no %1 site found - the phases on it are skipped (note no_site)", kind));
		}

		if (m_Ctx && !m_Ctx.AppendHeaderLines(lines))
			LogWarning("the site lines could not be appended to the run header");

		m_iCursor = 0;
		BeginPhase();
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbePostFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Channels || m_eSegment == DC_ESteerSegment.SEARCH)
			return;

		m_Channels.ReadPostFrame();
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Actuator || m_eSegment == DC_ESteerSegment.SEARCH)
			return;

		DC_ProbeSteerPhase phase = GetPhase();
		if (!phase)
			return;

		m_iLastStep = physicsStep;
		if (phase.m_iStartStep < 0)
		{
			phase.m_iStartStep = physicsStep;
			phase.m_fStartWorldMs = GetWorldMs();
		}

		CommandStep(phase);
		m_Actuator.ApplyStep(timeSlice);
		if (m_Actuator.ConsumeStuckWarning())
			LogWarning(m_Actuator.DescribeStuck());
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbePostSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Actuator || m_eSegment == DC_ESteerSegment.SEARCH)
			return;

		DC_ProbeSteerPhase phase = GetPhase();
		if (!phase || phase.m_iStartStep < 0)
			return;

		m_iLastStep = physicsStep;
		m_Actuator.ReadPost();
		m_Channels.ReadChannels();
		bool inArea = m_Area && m_Area.Contains(m_Actuator.GetPosition(), m_fAreaMarginM);
		WriteStepRow(physicsStep, timeSlice, phase, inArea);
		TrackStep(phase, timeSlice);
		int covered = CoveredSteps(timeSlice);
		m_iLastCovered = covered;
		m_iSegStep += covered;
		m_iPhaseStep += covered;
		Advance(phase, inArea);
	}

	//------------------------------------------------------------------------------------------------
	override bool IsProbeDone(string probeId)
	{
		if (probeId != m_sActiveProbe)
			return true;

		return m_bProbeDone;
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. Sets the driving-assistance mode back on every reason (an engine global, see the header); parks
	//! the car, records an interrupted phase and gives the camera back, except on reason delete (world teardown: the
	//! files are closed, nothing else is touched)
	override void OnProbeStop(string probeId, string reason)
	{
		if (probeId != m_sActiveProbe || m_sActiveProbe.IsEmpty())
			return;

		RestoreAssistMode();
		if (reason != DC_ProbeRunner.REASON_DELETE)
		{
			DC_ProbeSteerPhase phase = GetPhase();
			if (phase && !m_bProbeDone && m_eSegment != DC_ESteerSegment.SEARCH)
			{
				OnRunEnded(phase, "stopped");
				phase.m_sResult = "stopped";
				phase.AddNote("reason:" + reason);
				FinishPhaseRecord(phase);
			}

			array<string> summary = {};
			if (m_Actuator)
			{
				m_Actuator.SetSteeringCall(true);
				m_Actuator.Park();
				LogInfo(m_Actuator.DescribeDrive());
				m_Actuator.AppendDriveSummaryLines(summary);
			}

			if (m_Channels)
				m_Channels.AppendSummaryLines(summary);

			int restores = m_iAssistRestores;
			int sets = m_iAssistSets;
			summary.Insert("assist.sets=" + sets.ToString());
			summary.Insert("assist.restores=" + restores.ToString());
			int assistEnd = BaseVehicleControllerComponent.GetDrivingAssistanceMode();
			summary.Insert("assist.at_stop=" + assistEnd.ToString());
			int signUsed = GetSteerSign();
			summary.Insert("steer.sign_measured=" + m_iMeasuredSteerSign.ToString());
			summary.Insert("steer.sign_used=" + signUsed.ToString());
			AppendProbeSummary(summary);
			if (m_Ctx && !m_Ctx.AppendHeaderLines(summary))
				LogWarning("the drive / channel summary could not be appended to the run header");

			if (m_Rig)
				m_Rig.Detach();
		}

		if (m_StepsCsv)
			m_StepsCsv.Close();

		if (m_PhasesCsv)
			m_PhasesCsv.Close();

		CloseExtraCsvs();
		m_sActiveProbe = string.Empty;
		m_bProbeDone = true;
		m_Ctx = null;
		m_Actuator = null;
		m_Rig = null;
		m_StepsCsv = null;
		m_PhasesCsv = null;
	}

	//------------------------------------------------------------------------------------------------
	protected void ResetSteerState()
	{
		m_sActiveProbe = string.Empty;
		m_Ctx = null;
		m_Actuator = null;
		m_Channels = null;
		m_Rig = null;
		m_Area = null;
		m_Row = null;
		m_StepsCsv = null;
		m_PhasesCsv = null;
		m_aPhases = null;
		m_aSearches = null;
		m_aSearchKinds = null;
		m_aExcludes = null;
		m_iCursor = 0;
		m_iPhasesStarted = 0;
		m_bProbeDone = false;
		m_eSegment = DC_ESteerSegment.SEARCH;
		m_iSegStep = 0;
		m_iPhaseStep = 0;
		m_bTeleportDone = false;
		m_iLastStep = 0;
		m_bHasPrevHeading = false;
		m_fSegTarget = -1;
		m_bSegFlag = false;
		m_iEventIdx = -1;
		m_iSpinSteps = 0;
		m_fSteerSignSum = 0;
		m_iSteerSignSamples = 0;
		m_iMeasuredSteerSign = 0;
		m_bAssistChanged = false;
		m_iAssistSets = 0;
		m_iAssistRestores = 0;
		m_iAssistOverride = -1;
	}

	//------------------------------------------------------------------------------------------------
	protected bool RefuseStart(string reason)
	{
		LogError(reason);
		if (m_Rig)
			m_Rig.Detach();

		m_Rig = null;
		m_sActiveProbe = string.Empty;
		m_bProbeDone = true;
		m_Ctx = null;
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! +1 / -1: measured on this probe's steering segments, else m_iAssumedSteerSign
	protected int GetSteerSign()
	{
		if (m_iMeasuredSteerSign != 0)
			return m_iMeasuredSteerSign;

		if (m_iAssumedSteerSign >= 0)
			return 1;

		return -1;
	}

	//------------------------------------------------------------------------------------------------
	//! The driving-assistance mode of a phase (0 NONE, 1 PARTIAL, 2 FULL); the original is kept for the restore
	protected void SetAssistMode(int mode)
	{
		if (mode == 0)
			BaseVehicleControllerComponent.SetDrivingAssistanceMode(EVehicleDrivingAssistanceMode.NONE);
		else if (mode == 1)
			BaseVehicleControllerComponent.SetDrivingAssistanceMode(EVehicleDrivingAssistanceMode.PARTIAL);
		else
			BaseVehicleControllerComponent.SetDrivingAssistanceMode(EVehicleDrivingAssistanceMode.FULL);

		m_bAssistChanged = true;
		m_iAssistOverride = mode;
		m_iAssistSets++;
	}

	//------------------------------------------------------------------------------------------------
	//! Sets the original mode back after a phase override (idempotent)
	protected void RestoreAssistMode()
	{
		if (!m_bAssistChanged)
			return;

		BaseVehicleControllerComponent.SetDrivingAssistanceMode(m_eAssistOriginal);
		m_bAssistChanged = false;
		m_iAssistOverride = -1;
		m_iAssistRestores++;
	}

	//------------------------------------------------------------------------------------------------
	protected DC_ProbeSteerPhase GetPhase()
	{
		if (!m_aPhases || m_iCursor < 0 || m_iCursor >= m_aPhases.Count())
			return null;

		return m_aPhases[m_iCursor];
	}

	//------------------------------------------------------------------------------------------------
	//! Starts m_aPhases[m_iCursor] (phases on a missing site are skipped), then its first run
	protected void BeginPhase()
	{
		DC_ProbeSteerPhase phase = GetPhase();
		while (phase && !GetSite(phase.m_sSite))
		{
			phase.m_iIdx = m_iPhasesStarted;
			m_iPhasesStarted++;
			phase.m_sResult = "skipped";
			phase.AddNote("no_site");
			WritePhaseRow(phase);
			LogWarning(string.Format("phase %1 %2: skipped, no %3 site", phase.m_iIdx, phase.m_sName, phase.m_sSite));
			m_iCursor++;
			phase = GetPhase();
		}

		if (!phase)
		{
			m_bProbeDone = true;
			return;
		}

		phase.m_iIdx = m_iPhasesStarted;
		m_iPhasesStarted++;
		DC_SteerSite site = GetSite(phase.m_sSite);
		m_Area = site.MakeArea(site.m_fHalfWidthM);
		DC_LaunchClutch launchClutch = m_Actuator.GetLaunchClutch();
		phase.m_iStallsAtStart = launchClutch.GetStalls();
		phase.m_iRestartsAtStart = launchClutch.GetRestarts();
		phase.m_iSleepWakesAtStart = m_Actuator.GetSleepWakes();
		if (phase.m_iAssist >= 0)
			SetAssistMode(phase.m_iAssist);

		OnBeginPhase(phase);
		string target = phase.m_fTargetKmh.ToString(-1, 1) + " km/h";
		LogInfo(string.Format("phase %1 %2 (%3/%4) on the %5 site: target %6, steer %7, assist %8", phase.m_iIdx, phase.m_sName, m_iCursor + 1, m_aPhases.Count(), phase.m_sSite, target, phase.m_fSteer.ToString(-1, 3), phase.m_iAssist));
		BeginRun(phase);
	}

	//------------------------------------------------------------------------------------------------
	//! One run of the phase: reset (teleport), settle, accel, the subclass's segments, stop
	protected void BeginRun(notnull DC_ProbeSteerPhase phase)
	{
		phase.m_iRuns++;
		OnBeginRun(phase);
		m_Actuator.SetSteeringCall(true);
		m_eSegment = DC_ESteerSegment.RESET;
		m_iSegStep = 0;
		m_bTeleportDone = false;
		m_bHasPrevHeading = false;
		m_iSpinSteps = 0;
		m_fSegTarget = -1;
		m_bSegFlag = false;
		m_iEventIdx = -1;
	}

	//------------------------------------------------------------------------------------------------
	//! The command of this step: the common segments here, the rest from the subclass
	protected void CommandStep(notnull DC_ProbeSteerPhase phase)
	{
		switch (m_eSegment)
		{
			case DC_ESteerSegment.RESET:
			case DC_ESteerSegment.SETTLE:
			{
				m_Actuator.SetSteeringCall(true);
				m_Actuator.SetCommand(0, 0, 1, 1, -1, true);
				m_fSegTarget = -1;
				return;
			}

			case DC_ESteerSegment.STOP:
			{
				m_Actuator.SetSteeringCall(true);
				m_Actuator.SetCommand(0, 0, 1, 1, -1, false);
				m_fSegTarget = -1;
				return;
			}

			case DC_ESteerSegment.ACCEL:
			{
				m_Actuator.SetSteeringCall(true);
				m_Actuator.SetSpeedHold(phase.m_fTargetKmh, 0);
				m_fSegTarget = phase.m_fTargetKmh;
				return;
			}
		}

		CommandSegment(phase);
	}

	//------------------------------------------------------------------------------------------------
	protected void WriteStepRow(int physicsStep, float timeSlice, notnull DC_ProbeSteerPhase phase, bool inArea)
	{
		if (!m_StepsCsv || !m_Row)
			return;

		vector pos = m_Actuator.GetPosition();
		float along = 0;
		float lateral = 0;
		if (m_Area)
		{
			along = m_Area.AlongM(pos);
			lateral = m_Area.LateralM(pos);
		}

		int assist = BaseVehicleControllerComponent.GetDrivingAssistanceMode();
		m_Row.Clear();
		m_Row.AddInt(physicsStep);
		m_Row.AddFloat(GetWorldMs(), 3);
		m_Row.AddFloat(timeSlice, 6);
		m_Row.AddString(phase.m_sName);
		m_Row.AddString(SegmentName(m_eSegment));
		m_Row.AddInt(m_iPhaseStep);
		m_Row.AddInt(phase.m_iRuns);
		m_Actuator.AddStepValues(m_Row);
		m_Row.AddBool(m_Actuator.IsSteeringCalled());
		m_Row.AddInt(assist);
		m_Row.AddString(phase.m_sSite);
		m_Row.AddFloat(along, 3);
		m_Row.AddFloat(lateral, 3);
		m_Row.AddBool(inArea);
		m_Row.AddFloat(m_fSegTarget, 4);
		m_Row.AddBool(m_bSegFlag);
		m_Row.AddInt(m_iEventIdx);
		m_Channels.AddValues(m_Row);
		m_StepsCsv.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	//! Max speed, heading change, steer-sign sum, stall watch, spin watch
	protected void TrackStep(notnull DC_ProbeSteerPhase phase, float timeSlice)
	{
		float heading = m_Actuator.GetHeadingDeg();
		float delta = 0;
		if (m_bHasPrevHeading)
			delta = WrapDeg(heading - m_fPrevHeadingDeg);

		m_fPrevHeadingDeg = heading;
		m_bHasPrevHeading = true;
		if (!IsDrivingSegment(m_eSegment))
			return;

		float speed = m_Actuator.GetSpeedKmh();
		if (speed > phase.m_fMaxSpeedKmh)
			phase.m_fMaxSpeedKmh = speed;

		if (!m_Actuator.IsEngineOn())
			phase.m_iEngineOffSteps++;

		phase.m_fTurnDeg += delta;
		if (IsSteeringSegment(m_eSegment) && timeSlice > 0)
		{
			m_fSteerSignSum += m_Actuator.GetCommandSteer() * delta / timeSlice;
			m_iSteerSignSamples++;
		}

		vector angular = m_Actuator.GetAngularVelocityLocal();
		if (Math.AbsFloat(angular[1]) > m_fSpinYawRateRadS)
			m_iSpinSteps++;
		else
			m_iSpinSteps = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Every segment between the settle and the stop
	protected static bool IsDrivingSegment(DC_ESteerSegment segment)
	{
		switch (segment)
		{
			case DC_ESteerSegment.SEARCH:
			case DC_ESteerSegment.RESET:
			case DC_ESteerSegment.SETTLE:
			case DC_ESteerSegment.STOP:
				return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Segment transitions after the step (the state read by ReadPost decides)
	protected void Advance(notnull DC_ProbeSteerPhase phase, bool inArea)
	{
		switch (m_eSegment)
		{
			case DC_ESteerSegment.RESET:
			{
				if (m_bTeleportDone)
					EnterSegment(DC_ESteerSegment.SETTLE);

				return;
			}

			case DC_ESteerSegment.SETTLE:
			{
				// the wheel-angle axle choice: only at a confirmed zero-steer standstill, from half the settle on
				bool still = m_Actuator.GetSpeedKmh() < 0.5 && Math.AbsFloat(m_Actuator.GetPostSteer()) <= 0.001;
				if (still && m_iSegStep >= m_iSettleSteps / 2 && !m_Channels.AxlesSettled())
					m_Channels.ArmAxleSelection();

				if (m_iSegStep < m_iSettleSteps)
					return;

				m_Channels.DisarmAxleSelection();
				if (!m_Actuator.IsEngineOn())
				{
					m_Actuator.StartModeA();
					phase.AddNote("engine_restart_after_settle");
				}

				EnterSegment(DC_ESteerSegment.ACCEL);
				return;
			}

			case DC_ESteerSegment.STOP:
			{
				bool stopped = m_Actuator.GetSpeedKmh() < m_fSpeedTolKmh;
				if (!stopped && m_iSegStep < m_iStopCapSteps)
					return;

				if (!stopped)
					phase.AddNote("stop_cap");

				if (WantsAnotherRun(phase))
				{
					BeginRun(phase);
					return;
				}

				FinishPhase(phase);
				return;
			}
		}

		if (!m_Actuator.IsEngineOn())
		{
			EndRun(phase, "engine_off");
			return;
		}

		if (!inArea)
		{
			EndRun(phase, "aborted_area");
			return;
		}

		// the motion the obstacle envelope assumed: the capped speed (+3 km/h) and, outside circle probes, the heading
		float speedNow = m_Actuator.GetSpeedKmh();
		if (speedNow > GetMaxSpeedKmh() + OVERSPEED_KMH)
		{
			phase.AddNote("overspeed_kmh:" + speedNow.ToString(-1, 1));
			EndRun(phase, "overspeed");
			return;
		}

		float headingOff = Math.AbsFloat(WrapDeg(m_Actuator.GetHeadingDeg() - m_Area.m_fHeadingDeg));
		if (!AllowsAnyHeading() && headingOff > GetEnvelopeAngleDeg())
		{
			phase.AddNote("heading_off_deg:" + headingOff.ToString(-1, 1));
			EndRun(phase, "aborted_heading");
			return;
		}

		if (m_iSpinSteps >= SPIN_STEPS)
		{
			phase.m_iSpins++;
			EndRun(phase, "spin");
			return;
		}

		if (m_eSegment == DC_ESteerSegment.ACCEL)
		{
			if (m_Actuator.GetSpeedKmh() >= phase.m_fTargetKmh - m_fSpeedTolKmh)
			{
				EnterSegment(AfterAccel(phase));
				return;
			}

			if (m_iSegStep >= m_iReachSteps)
				EndRun(phase, "no_speed");

			return;
		}

		AdvanceSegment(phase);
	}

	//------------------------------------------------------------------------------------------------
	protected void EnterSegment(DC_ESteerSegment segment)
	{
		m_eSegment = segment;
		m_iSegStep = 0;
		m_bSegFlag = false;
		m_iSpinSteps = 0;
		m_fSegStartAlongM = 0;
		if (m_Area && m_Actuator)
			m_fSegStartAlongM = m_Area.AlongM(m_Actuator.GetPosition());
	}

	//------------------------------------------------------------------------------------------------
	//! A driving segment ended early: the run result, then brake to a stop
	protected void EndRun(notnull DC_ProbeSteerPhase phase, string result)
	{
		phase.SetRunResult(result);
		phase.AddNote("run" + phase.m_iRuns.ToString() + ":" + result + "@" + SegmentName(m_eSegment));
		OnRunEnded(phase, result);
		m_Actuator.SetSteeringCall(true);
		EnterSegment(DC_ESteerSegment.STOP);
	}

	//------------------------------------------------------------------------------------------------
	//! Writes the phase row, sets the assistance mode back and starts the next phase (or ends the probe)
	protected void FinishPhase(notnull DC_ProbeSteerPhase phase)
	{
		FinishPhaseRecord(phase);
		LogInfo(string.Format("phase %1 %2: %3 (steps %4, runs %5, max %6 km/h, turn %7 deg)", phase.m_iIdx, phase.m_sName, phase.m_sResult, m_iPhaseStep, phase.m_iRuns, phase.m_fMaxSpeedKmh.ToString(-1, 2), phase.m_fTurnDeg.ToString(-1, 1)));
		if (phase.m_sResult == "error")
		{
			m_bProbeDone = true;
			return;
		}

		m_iCursor++;
		m_iPhaseStep = 0;
		if (m_iCursor >= m_aPhases.Count())
		{
			m_bProbeDone = true;
			return;
		}

		BeginPhase();
	}

	//------------------------------------------------------------------------------------------------
	protected void FinishPhaseRecord(notnull DC_ProbeSteerPhase phase)
	{
		RestoreAssistMode();
		phase.m_iEndStep = m_iLastStep;
		phase.m_fEndWorldMs = GetWorldMs();
		if (phase.m_sResult.IsEmpty())
			phase.m_sResult = "ok";

		float maxSpeed = phase.m_fMaxSpeedKmh;
		float turn = phase.m_fTurnDeg;
		phase.AddNote("max_kmh:" + maxSpeed.ToString(-1, 3));
		phase.AddNote("turn_deg:" + turn.ToString(-1, 2));
		phase.AddNote("engine_off_steps:" + phase.m_iEngineOffSteps.ToString());
		phase.AddNote("spins:" + phase.m_iSpins.ToString());
		if (m_Actuator)
		{
			DC_LaunchClutch launchClutch = m_Actuator.GetLaunchClutch();
			int stalls = launchClutch.GetStalls() - phase.m_iStallsAtStart;
			int restarts = launchClutch.GetRestarts() - phase.m_iRestartsAtStart;
			int sleepWakes = m_Actuator.GetSleepWakes() - phase.m_iSleepWakesAtStart;
			phase.AddNote("stalls:" + stalls.ToString());
			phase.AddNote("restarts:" + restarts.ToString());
			phase.AddNote("sleep_wakes:" + sleepWakes.ToString());
		}

		WritePhaseRow(phase);
	}

	//------------------------------------------------------------------------------------------------
	protected void WritePhaseRow(notnull DC_ProbeSteerPhase phase)
	{
		if (!m_PhasesCsv)
			return;

		DC_ProbeRow row = new DC_ProbeRow();
		row.AddInt(phase.m_iIdx);
		row.AddString(phase.m_sName);
		row.AddString(PhaseKindName(phase.m_eKind));
		row.AddString(phase.m_sSite);
		AddIntOrEmpty(row, phase.m_iStartStep);
		AddIntOrEmpty(row, phase.m_iEndStep);
		AddMsOrEmpty(row, phase.m_fStartWorldMs);
		AddMsOrEmpty(row, phase.m_fEndWorldMs);
		row.AddFloat(phase.m_fTargetKmh, 3);
		row.AddFloat(phase.m_fSteer, 4);
		row.AddInt(phase.m_iAssist);
		row.AddInt(phase.m_iRuns);
		row.AddString(phase.m_sResult);
		row.AddString(phase.m_sNote);
		m_PhasesCsv.WriteRow(row);
	}

	//------------------------------------------------------------------------------------------------
	//! Engine steps one dispatched step covers: round(timeSlice / m_fDt), at least 1 (DC_ProbeSequences.CoveredSteps)
	protected int CoveredSteps(float timeSlice)
	{
		int steps = DC_ProbeSequences.StepsFor(timeSlice, m_fDt);
		if (steps < 1)
			steps = 1;

		return steps;
	}

	//------------------------------------------------------------------------------------------------
	//! Distance along the site's centre line since the current segment started [m]
	protected float SegmentAlongM()
	{
		if (!m_Area || !m_Actuator)
			return 0;

		return m_Area.AlongM(m_Actuator.GetPosition()) - m_fSegStartAlongM;
	}

	//------------------------------------------------------------------------------------------------
	//! Distance left to the end of the site area (minus the margin) along its centre line [m]
	protected float RemainingAlongM()
	{
		if (!m_Area || !m_Actuator)
			return 0;

		return m_Area.m_fLengthM - m_fAreaMarginM - m_Area.AlongM(m_Actuator.GetPosition());
	}

	//------------------------------------------------------------------------------------------------
	//! Parses '<f>,<f>,...' (empty allowed); every value in min..max
	static bool ParseSteerFloats(string text, notnull array<float> outValues, float min, float max)
	{
		outValues.Clear();
		array<string> tokens = {};
		text.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			float value;
			if (!ParseSteerFloat(token, value))
				return false;

			if (!InRange(value, min, max))
				return false;

			outValues.Insert(value);
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! min <= value <= max (false for a NaN: both comparisons fail)
	static bool InRange(float value, float min, float max)
	{
		return value >= min && value <= max;
	}

	//------------------------------------------------------------------------------------------------
	//! true when the whole trimmed token is one number
	static bool ParseSteerFloat(string token, out float value)
	{
		value = 0;
		string trimmed = token.Trim();
		if (trimmed.IsEmpty())
			return false;

		int parsed;
		value = trimmed.ToFloat(0, 0, parsed);
		return parsed == trimmed.Length();
	}

	//------------------------------------------------------------------------------------------------
	protected static void AddIntOrEmpty(notnull DC_ProbeRow row, int value)
	{
		if (value >= 0)
			row.AddInt(value);
		else
			row.AddEmpty();
	}

	//------------------------------------------------------------------------------------------------
	protected static void AddMsOrEmpty(notnull DC_ProbeRow row, float worldMs)
	{
		if (worldMs >= 0)
			row.AddFloat(worldMs, 3);
		else
			row.AddEmpty();
	}

	//------------------------------------------------------------------------------------------------
	static string SegmentName(DC_ESteerSegment segment)
	{
		switch (segment)
		{
			case DC_ESteerSegment.SEARCH: return "search";
			case DC_ESteerSegment.RESET: return "reset";
			case DC_ESteerSegment.SETTLE: return "settle";
			case DC_ESteerSegment.ACCEL: return "accel";
			case DC_ESteerSegment.HOLD: return "hold";
			case DC_ESteerSegment.MEASURE: return "measure";
			case DC_ESteerSegment.NOCALL: return "nocall";
			case DC_ESteerSegment.STAIR: return "stair";
			case DC_ESteerSegment.REF: return "ref";
			case DC_ESteerSegment.ALAT: return "alat";
			case DC_ESteerSegment.BRAKE: return "brake";
			case DC_ESteerSegment.TRAIN: return "train";
			case DC_ESteerSegment.PRBS: return "prbs";
			case DC_ESteerSegment.STOP: return "stop";
		}

		return "unknown";
	}

	//------------------------------------------------------------------------------------------------
	static string PhaseKindName(DC_ESteerPhaseKind kind)
	{
		switch (kind)
		{
			case DC_ESteerPhaseKind.FLAT: return "flat";
			case DC_ESteerPhaseKind.CROSS: return "cross";
			case DC_ESteerPhaseKind.STAIR: return "stair";
			case DC_ESteerPhaseKind.LOAD: return "load";
			case DC_ESteerPhaseKind.STEPS: return "steps";
		}

		return "unknown";
	}

	//------------------------------------------------------------------------------------------------
	//! Angle difference wrapped to -180..180 deg
	protected static float WrapDeg(float deltaDeg)
	{
		while (deltaDeg > 180)
		{
			deltaDeg -= 360;
		}

		while (deltaDeg < -180)
		{
			deltaDeg += 360;
		}

		return deltaDeg;
	}

	//------------------------------------------------------------------------------------------------
	protected static float GetWorldMs()
	{
		World world = GetGame().GetWorld();
		if (!world)
			return -1;

		return world.GetWorldTime();
	}

	//------------------------------------------------------------------------------------------------
	protected void LogInfo(string message)
	{
		if (m_Ctx)
			m_Ctx.Log(message);
	}

	//------------------------------------------------------------------------------------------------
	protected void LogWarning(string message)
	{
		if (m_Ctx)
			m_Ctx.Warn(message);
	}

	//------------------------------------------------------------------------------------------------
	protected void LogError(string message)
	{
		if (m_Ctx)
			m_Ctx.Error(message);
	}
}
