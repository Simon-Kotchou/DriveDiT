// ============================================================================
// DC_ProbeSequences - T0-ENG probes E-01 (Mode A actuation identification) and E-02a (first-pass steer -> kappa)
// ============================================================================
//
// A DC_ProbeComponent on the probe game mode (GameMode_DCProbes in DC_Probes_Arland), started, stepped and stopped by
// DC_ProbeRunner (PROBE_DESIGN.md 4.4, 5.3). It drives the placed, driverless S105 DC_Ego in Mode A (r2 3.2):
// SetPersistentHandBrake(false) and ForceStartEngine as in vanilla Game/Cinematics/VehicleCinematicTrack.c:127 and
// :140, then SetSteering/SetBreak/SetGear/SetClutch/SetThrottle on every physics step from OnProbeSimulate. The
// inputs are read back right after the setters (input_pre_*) and after the step in OnProbePostSimulate
// (input_post_*), next to the commanded values (cmd_*), so every step carries command, pre and post. The vanilla
// zeroing of the noise-steer and roughness sensitivities (VehicleCinematicTrack.c:125-126) is NOT copied (r2 3.2 X6):
// they are only read and echoed into the run header.
//
// GEARBOX (probe run 2, 55c77f8: the S105 has a manual gearbox, HasAutomaticGearbox 0, and sat in neutral, so the
// throttle only revved the engine). DC_ProbeActuator.Init reads HasAutomaticGearbox, GearboxGearsCount and
// GearboxForwardGearsCount once per probe start and derives the indices as vanilla does: neutral = gears - forward -
// 1, first = gears - forward (SCR_ToggleHandbrakeAction.c:99-108), top = gears - 1 (SCR_VehicleGearCondition.c LAST);
// below neutral is reverse (S105: 0 reverse, 1 neutral, 2..5 first..fourth; TBD-7). With a manual gearbox and no
// explicit gear (cmd gear -1) the actuator's shift policy picks the gear: neutral while the handbrake is commanded,
// first gear at its release (set before the throttle of that step), then first..top by rpm (attributes in category
// Gearbox), never neutral or reverse while driving; the chosen gear is the cmd_gear column. That park/drive choice
// is the probe's own: vanilla SCR_ToggleHandbrakeAction.c:93-108 releases into neutral and parks in first gear.
// gear_scripted keeps its explicit SetGear every step, and the policy continues from its last gear; gear_observe
// (automatic shifting) is skipped with note manual_gearbox on a manual gearbox; the move checks hold first gear; a
// probe stop sets neutral on a manual gearbox. At every release of the commanded handbrake the actuator also wakes
// the vehicle's physics as vanilla does (Physics.SetActive(ActiveState.ACTIVE), SCR_ToggleHandbrakeAction.c:97), and
// on every later step with the handbrake released it wakes a body found asleep again (Physics.IsActive false, the
// SCR_DestructionTireComponent.c:155-156 pattern), on top of the ForceEnableSimulation latch; the phase notes and the
// run header count both. e05 and e06a also latch ForceEnableSimulation when the car stays below 3 km/h for 10 s
// after a drive start (DC_ProbeActuator.SetStuckForce, WARNING in the log).
//
// WHAT IT DOES (authority, play mode, only while the runner has started the probe)
//   e01  phases in this order; each starts with a teleport to the area start (segment reset) and m_fSettleS of
//        brake 1 + handbrake (settle):
//          move_check          throttle m_fMoveCheckThrottle for m_fMoveCheckS in first gear (manual gearbox), no
//                              ForceEnableSimulation;
//          move_check_forced   only after a no_motion: ForceEnableSimulation, then the same (latched for the probe);
//          steer_prbs          speed hold m_fPrbsSpeedKmh, steering +-m_fPrbsAmplitude from DC_ProbePrbs, one period;
//          steer_reversal      speed hold, steering +-m_fReversalAmplitude reversed every m_fReversalPeriodS / 2,
//                              m_iReversalCount reversals, then one half period of zero steering (centering);
//          throttle_<v>        throttle v from standstill for m_fThrottleHoldS, then brake 1 to a stop;
//          brake_<v>           speed hold to m_fBrakeFromKmh, then brake v until below m_fSpeedTolKmh;
//          gear_observe        throttle m_fGearObserveThrottle for m_fGearObserveS, gear left to the gearbox
//                              (automatic gearbox only; skipped with note manual_gearbox otherwise);
//          gear_scripted       SetGear(g) every step for each g of m_sGearSequence, m_fGearHoldS each.
//        e01_steps.csv (one row per physics step) and e01_phases.csv (one row per phase).
//   e02a cells over m_sE02aGrid (speed km/h : steering levels): teleport to the side of the area away from the turn,
//        settle, accelerate straight with the speed hold (accel), hold the speed with the cell's steering for
//        m_fE02aHoldS, until m_fE02aMaxTurnDeg of heading change or until the area edge (excite), brake to a stop.
//        e02a_steps.csv and e02a_cells.csv. The turn direction of positive steering comes from the e01 steering
//        phases of the same Play (sign of the running sum of cmd_steer x heading rate), else m_iAssumedSteerSign.
//
// Segment transitions are decided in OnProbePostSimulate from the state after the step; the commands of the next
// step are set in OnProbeSimulate. Teleports run in OnProbeFrame (never inside a physics step). The step row of step
// N carries the phase and segment whose commands were applied in step N. The segment and phase counters advance by
// the engine steps a dispatched step covers (CoveredSteps: round(timeSlice / dt), n on a frame whose n fixed steps
// the runner's fixedframe fallback merges into one call, else 1), so every duration holds in world time at any frame
// rate; phase_step in the step rows counts covered steps, and a PRBS bit is drawn for every bit start reached.
//
// SHARED CLASSES (PROBE_DESIGN.md 4.8; DC_ProbeImages and DC_ProbeMarkers use them): DC_ProbeArea (test area
// geometry), DC_ProbeActuator (Mode A driver of one vehicle), DC_ProbePrbs (maximum-length LFSR).
//
// BUILD-AGNOSTIC (PROBE_DESIGN.md 8): every engine call exists with the same signature in 1.8.0.13 and 1.9.0.10
// (PROBE_APIS.md 4.3); every sequence, amplitude, speed, gear, area and tolerance is an [Attribute]; durations are in
// seconds and converted with the runner's step length (DC_ProbeComponent.GetStepDtS, not GetFixedTimeSlice). For the
// 1.9 re-run (MIGRATION_1_9.md 8.3, SURFACE.md A-01..B-08)
// the probe records instead of assuming 1.8 behaviour, and keeps going: self-steer and a steering rate limit or
// centering show up as input_* != cmd_* or as yaw with zero command; a stall (engine_on 0) ends the phase with
// engine_off and the next phase restarts the engine; cur_gear / fut_gear every step expose shift loops; a persistent
// handbrake that the controller sets again is released and counted (phase note phb_resets); ForceEnableSimulation is
// decided per Play by the move check; a clutch dip around scripted gear changes is an attribute (0 = off, 1.8).
// The 1.9-only members stay in the DD-14 overlay; they appear here only as text in the rerun19.overlay_only header
// value of PROBE_DESIGN.md 8.2.
//
// FAILURE HANDLING: OnProbeStart refuses (returns false, ERROR) without DC_Ego, its CarControllerComponent or its
// VehicleWheeledSimulation, with an unparsable list attribute or PRBS setting, without a fixed time slice, when the
// area relief exceeds m_fMaxAreaReliefM, or (e02a) when e01 of this Play found that the car does not move even with
// ForceEnableSimulation. A phase that cannot reach its speed, leaves the area or loses the engine ends with no_speed,
// aborted_area or engine_off and keeps its data; a failed teleport ends the probe (result error). OnProbeStop parks
// the car (brake 1 + handbrake), except on world teardown (reason delete: it only closes its files).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! A rectangular test area on the ground: centre line from m_vStart along m_fHeadingDeg (0 = +Z, clockwise from above).
class DC_ProbeArea
{
	vector m_vStart;			//!< x, z used; y ignored
	float m_fHeadingDeg;
	float m_fLengthM;
	float m_fHalfWidthM;

	//------------------------------------------------------------------------------------------------
	void Set(vector start, float headingDeg, float lengthM, float halfWidthM)
	{
		m_vStart = start;
		m_fHeadingDeg = headingDeg;
		m_fLengthM = lengthM;
		m_fHalfWidthM = halfWidthM;
	}

	//------------------------------------------------------------------------------------------------
	//! Unit vector along the centre line: (sin h, 0, cos h)
	vector GetDirection()
	{
		float rad = m_fHeadingDeg * Math.DEG2RAD;
		return Vector(Math.Sin(rad), 0, Math.Cos(rad));
	}

	//------------------------------------------------------------------------------------------------
	//! Unit vector to the right of the centre line seen from above: (cos h, 0, -sin h)
	vector GetRight()
	{
		float rad = m_fHeadingDeg * Math.DEG2RAD;
		return Vector(Math.Cos(rad), 0, -Math.Sin(rad));
	}

	//------------------------------------------------------------------------------------------------
	//! Distance along the centre line from m_vStart
	float AlongM(vector worldPos)
	{
		vector dir = GetDirection();
		return (worldPos[0] - m_vStart[0]) * dir[0] + (worldPos[2] - m_vStart[2]) * dir[2];
	}

	//------------------------------------------------------------------------------------------------
	//! Signed distance from the centre line, right of the heading positive
	float LateralM(vector worldPos)
	{
		vector right = GetRight();
		return (worldPos[0] - m_vStart[0]) * right[0] + (worldPos[2] - m_vStart[2]) * right[2];
	}

	//------------------------------------------------------------------------------------------------
	//! World x, z of a point given in area coordinates; y = 0
	vector PointAt(float alongM, float lateralM)
	{
		vector dir = GetDirection();
		vector right = GetRight();
		return Vector(m_vStart[0] + alongM * dir[0] + lateralM * right[0], 0, m_vStart[2] + alongM * dir[2] + lateralM * right[2]);
	}

	//------------------------------------------------------------------------------------------------
	//! true when worldPos lies inside the rectangle shrunk by marginM on every side
	bool Contains(vector worldPos, float marginM)
	{
		float along = AlongM(worldPos);
		if (along < marginM || along > m_fLengthM - marginM)
			return false;

		return Math.AbsFloat(LateralM(worldPos)) <= m_fHalfWidthM - marginM;
	}

	//------------------------------------------------------------------------------------------------
	//! max - min of BaseWorld.GetSurfaceY on a stepM grid over the area (edges included); minY / maxY out
	float MeasureRelief(notnull BaseWorld world, float stepM, out float minY, out float maxY)
	{
		minY = 0;
		maxY = 0;
		if (stepM <= 0)
			return -1;

		int alongCount = Math.Floor(m_fLengthM / stepM);
		int lateralCount = Math.Floor(2 * m_fHalfWidthM / stepM);
		bool first = true;
		for (int i = 0; i <= alongCount; i++)
		{
			for (int j = 0; j <= lateralCount; j++)
			{
				vector point = PointAt(i * stepM, -m_fHalfWidthM + j * stepM);
				float y = world.GetSurfaceY(point[0], point[2]);
				if (first || y < minY)
					minY = y;

				if (first || y > maxY)
					maxY = y;

				first = false;
			}
		}

		return maxY - minY;
	}

	//------------------------------------------------------------------------------------------------
	//! <prefix>.start, .heading_deg, .length_m, .half_width_m key=value lines
	void AppendHeaderLines(notnull array<string> lines, string prefix)
	{
		lines.Insert(prefix + ".start=" + FormatVector(m_vStart, 3));
		lines.Insert(prefix + ".heading_deg=" + m_fHeadingDeg.ToString(-1, 4));
		lines.Insert(prefix + ".length_m=" + m_fLengthM.ToString(-1, 3));
		lines.Insert(prefix + ".half_width_m=" + m_fHalfWidthM.ToString(-1, 3));
	}

	//------------------------------------------------------------------------------------------------
	//! "x y z" with the given decimals (header values; the layer's vector notation)
	static string FormatVector(vector value, int decimals)
	{
		float x = value[0];
		float y = value[1];
		float z = value[2];
		return x.ToString(-1, decimals) + " " + y.ToString(-1, decimals) + " " + z.ToString(-1, decimals);
	}
}

//------------------------------------------------------------------------------------------------
//! Maximum-length LFSR bit source for PRBS steering (orders 5..10, taps 5:(5,3) 6:(6,5) 7:(7,6) 8:(8,6,5,4) 9:(9,5) 10:(10,7))
//! Fibonacci form: the output is bit 0; the feedback (XOR of the bits at shift order - tap) enters at bit order - 1.
//! tools/probes/t0eng/e01_actuation.py prbs_bits() is the same generator.
class DC_ProbePrbs
{
	protected int m_iOrder;
	protected int m_iState;
	protected ref array<int> m_aShifts = {};

	//------------------------------------------------------------------------------------------------
	//! false for an unsupported order or a zero seed (seed is taken modulo 2^order)
	bool Init(int order, int seed)
	{
		m_iOrder = 0;
		m_iState = 0;
		m_aShifts.Clear();

		array<int> taps = {};
		switch (order)
		{
			case 5: taps.Insert(5); taps.Insert(3); break;
			case 6: taps.Insert(6); taps.Insert(5); break;
			case 7: taps.Insert(7); taps.Insert(6); break;
			case 8: taps.Insert(8); taps.Insert(6); taps.Insert(5); taps.Insert(4); break;
			case 9: taps.Insert(9); taps.Insert(5); break;
			case 10: taps.Insert(10); taps.Insert(7); break;
			default: return false;
		}

		int state = seed & ((1 << order) - 1);
		if (state == 0)
			return false;

		foreach (int tap : taps)
		{
			m_aShifts.Insert(order - tap);
		}

		m_iOrder = order;
		m_iState = state;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! 0 or 1 (0 after a failed Init)
	int NextBit()
	{
		if (m_iOrder <= 0)
			return 0;

		int bit = m_iState & 1;
		int feedback = 0;
		foreach (int shift : m_aShifts)
		{
			feedback = (feedback + ((m_iState >> shift) & 1)) & 1;	// XOR of single bits
		}

		m_iState = (m_iState >> 1) | (feedback << (m_iOrder - 1));
		return bit;
	}

	//------------------------------------------------------------------------------------------------
	//! 2^order - 1 (0 after a failed Init)
	int GetPeriod()
	{
		if (m_iOrder <= 0)
			return 0;

		return (1 << m_iOrder) - 1;
	}
}

//------------------------------------------------------------------------------------------------
//! Mode A actuation of one driverless vehicle (r2 3.2): a held command applied every physics step, inputs read back
//! after the setters (input_pre) and after the step (input_post). One instance per running probe; probes are
//! exclusive, so only one actuator drives at a time. Every probe that drives DC_Ego (e01, e02a, e05, e06a) drives it
//! through this class, so the gearbox handling below is the same for all of them.
class DC_ProbeActuator
{
	//! Speed-hold defaults (probe-only PI, not llc_v1); DC_ProbeSequences passes its attributes with SetSpeedGains
	static const float DEFAULT_SPEED_KP = 0.08;
	static const float DEFAULT_SPEED_KI = 0.02;
	static const float DEFAULT_BRAKE_SCALE = 0.5;

	//! Shift-policy defaults for a manual gearbox (probe-only, not llc_v1); DC_ProbeSequences passes its attributes with
	//! SetShiftPolicy, DC_ProbeImages and DC_ProbeMarkers drive with these. S105 (e04, run 2): rpm idle 800, peak power
	//! 4800, max 8000; in neutral throttle 0.5 revved it to 5800
	static const float DEFAULT_UPSHIFT_RPM = 4500;
	static const float DEFAULT_DOWNSHIFT_RPM = 1800;
	static const float DEFAULT_MIN_SHIFT_S = 1.5;
	static const float DEFAULT_FIRST_BELOW_KMH = 8;
	//! A policy gear that neither GetGear nor the controller's future gear shows this long after its SetGear is set again
	static const float GEAR_REASSERT_S = 1;
	//! Stuck guard of the image probes (SetStuckForce): a drive start after which the speed stays below STUCK_BELOW_KMH
	//! for STUCK_AFTER_S latches ForceEnableSimulation. Probe run 2: e05 drove its whole 10 min at 0 km/h
	static const float STUCK_AFTER_S = 10;
	static const float STUCK_BELOW_KMH = 3;

	protected IEntity m_Vehicle;
	protected CarControllerComponent m_Controller;
	protected VehicleWheeledSimulation m_Simulation;
	protected string m_sFailure = "not initialised";
	protected bool m_bForced;					//!< ForceSimulation was called (re-applied every step)
	protected int m_iPersistentHandbrakeResets;	//!< steps on which the controller had set the persistent handbrake again

	// Gearbox, read once by Init (DetectGearbox); indices derived as vanilla
	protected bool m_bAutomaticGearbox;
	protected int m_iGearsCount;
	protected int m_iForwardGearsCount;
	protected int m_iGearNeutral = -1;
	protected int m_iGearFirst = -1;
	protected int m_iGearTop = -1;
	protected bool m_bShiftPolicy;				//!< manual gearbox with usable indices: commands with gear -1 use the policy

	// Shift policy (SetShiftPolicy) and its state
	protected float m_fUpshiftRpm = DEFAULT_UPSHIFT_RPM;
	protected float m_fDownshiftRpm = DEFAULT_DOWNSHIFT_RPM;
	protected float m_fMinShiftS = DEFAULT_MIN_SHIFT_S;
	protected float m_fFirstBelowKmh = DEFAULT_FIRST_BELOW_KMH;
	protected float m_fShiftDipS;				//!< clutch 0 for this long after each policy change into a forward gear
	protected bool m_bFirstGearOnly;			//!< SetFirstGearOnly: the policy stays in first gear (move checks)
	protected int m_iPolicyGear = -1;			//!< gear chosen by the policy, -1 = none yet
	protected int m_iPolicySetGear = -1;		//!< gear of the policy's last SetGear call
	protected float m_fSinceShiftS;				//!< world time since the policy's last gear change [s]
	protected float m_fSinceSetGearS;			//!< world time since the policy's last SetGear call [s]
	protected float m_fDipLeftS;
	protected int m_iPolicyShifts;				//!< policy changes between two forward gears
	protected int m_iSetGearCalls;				//!< SetGear calls of the policy (changes and re-asserts)
	protected int m_iGearReasserts;

	// Physics wake at every release of the commanded handbrake (vanilla SCR_ToggleHandbrakeAction.c:96-97)
	protected bool m_bAppliedHandbrake = true;	//!< handbrake of the last ApplyStep (the placed car starts parked)
	protected int m_iWakeCalls;					//!< Physics.SetActive(ActiveState.ACTIVE) calls
	protected int m_iWakeAsleep;				//!< of those, with Physics.IsActive() false before the call
	protected int m_iWakeNoPhysics;				//!< releases without a Physics object (no call)
	protected int m_iSleepWakes;				//!< later released steps that found the body asleep and woke it

	// Stuck guard (SetStuckForce; e05, e06a): checked once per drive start
	protected float m_fStuckAfterS = -1;		//!< -1 = off
	protected float m_fStuckBelowKmh;
	protected float m_fSinceDriveStartS = -1;	//!< world time since the last release while its check is pending, else -1
	protected float m_fMaxKmhSinceDriveStart;
	protected int m_iStuckForces;				//!< drive starts that latched ForceEnableSimulation
	protected bool m_bStuckWarningDue;			//!< ConsumeStuckWarning: the caller logs the latch once

	// Applied in the last ApplyStep (columns cmd_gear, cmd_clutch): the explicit or the policy gear, the dip clutch
	protected int m_iAppliedGear = -1;			//!< -1 = no gear command
	protected float m_fAppliedClutch = 1;

	// Held command (cmd_*): applied by every ApplyStep until changed
	protected float m_fCmdSteer;
	protected float m_fCmdThrottle;
	protected float m_fCmdBrake = 1;
	protected float m_fCmdClutch = 1;
	protected int m_iCmdGear = -1;				//!< -1 = no SetGear call
	protected bool m_bCmdHandbrake = true;

	// Speed hold
	protected bool m_bSpeedHold;
	protected float m_fHoldKmh = -1;
	protected float m_fHoldIntegral;			//!< km/h * s
	protected float m_fKp = DEFAULT_SPEED_KP;
	protected float m_fKi = DEFAULT_SPEED_KI;
	protected float m_fBrakeScale = DEFAULT_BRAKE_SCALE;

	// Readback after the setters (input_pre_*) and after the step (input_post_*)
	protected float m_fPreSteer;
	protected float m_fPreThrottle;
	protected float m_fPreBrake;
	protected float m_fPreClutch;
	protected int m_iPreGear;
	protected bool m_bPreHandbrake;
	protected float m_fPostSteer;
	protected float m_fPostThrottle;
	protected float m_fPostBrake;
	protected float m_fPostClutch;
	protected int m_iPostGear;
	protected bool m_bPostHandbrake;

	// Vehicle state of the last ReadPost
	protected int m_iCurGear;
	protected int m_iFutGear;
	protected float m_fRpm;
	protected bool m_bEngineOn;
	protected float m_fSpeedKmh;
	protected vector m_vVelLocal;
	protected vector m_vAngVelLocal;
	protected vector m_vAccLocal;
	protected vector m_vPos;
	protected vector m_vFwd;
	protected float m_fHeadingDeg;
	protected bool m_bGroundContact;
	protected int m_iWheelsContact;

	//------------------------------------------------------------------------------------------------
	//! Finds CarControllerComponent and its VehicleWheeledSimulation (GetSimulation). false (GetFailure) if missing.
	bool Init(notnull IEntity vehicle)
	{
		m_Vehicle = vehicle;
		m_Controller = CarControllerComponent.Cast(vehicle.FindComponent(CarControllerComponent));
		if (!m_Controller)
		{
			m_sFailure = "no CarControllerComponent on " + vehicle.GetName();
			return false;
		}

		m_Simulation = m_Controller.GetSimulation();
		if (!m_Simulation)
			m_Simulation = VehicleWheeledSimulation.Cast(vehicle.FindComponent(VehicleWheeledSimulation));

		if (!m_Simulation)
		{
			m_sFailure = "no VehicleWheeledSimulation on " + vehicle.GetName();
			return false;
		}

		m_sFailure = string.Empty;
		DetectGearbox();
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Once per probe start (Init): HasAutomaticGearbox and the gear counts; neutral = gears - forward - 1 and first =
	//! gears - forward exactly as vanilla SCR_ToggleHandbrakeAction.c:99-108, top = gears - 1 as
	//! SCR_VehicleGearCondition.c (LAST), reverse below neutral. The shift policy is on for a manual gearbox with at
	//! least one forward gear and a neutral index >= 0.
	protected void DetectGearbox()
	{
		m_bAutomaticGearbox = m_Controller.HasAutomaticGearbox();
		m_iGearsCount = m_Simulation.GearboxGearsCount();
		m_iForwardGearsCount = m_Simulation.GearboxForwardGearsCount();
		m_iGearNeutral = m_iGearsCount - m_iForwardGearsCount - 1;
		m_iGearFirst = m_iGearsCount - m_iForwardGearsCount;
		m_iGearTop = m_iGearsCount - 1;
		m_bShiftPolicy = !m_bAutomaticGearbox && m_iForwardGearsCount > 0 && m_iGearNeutral >= 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Shift policy parameters (manual gearbox, commands with gear -1): up one gear at rpm >= upshiftRpm, down one at
	//! rpm <= downshiftRpm, at most one change per minShiftS, first gear below firstBelowKmh, clutch 0 for clutchDipS
	//! after each change into a forward gear (0 = no dip)
	void SetShiftPolicy(float upshiftRpm, float downshiftRpm, float minShiftS, float firstBelowKmh, float clutchDipS)
	{
		m_fUpshiftRpm = upshiftRpm;
		m_fDownshiftRpm = downshiftRpm;
		m_fMinShiftS = minShiftS;
		m_fFirstBelowKmh = firstBelowKmh;
		m_fShiftDipS = clutchDipS;
	}

	//------------------------------------------------------------------------------------------------
	//! true: the shift policy stays in first gear while driving (E-01 move checks)
	void SetFirstGearOnly(bool firstOnly)
	{
		m_bFirstGearOnly = firstOnly;
	}

	//------------------------------------------------------------------------------------------------
	//! HasAutomaticGearbox() was false at Init
	bool IsManualGearbox()
	{
		return !m_bAutomaticGearbox;
	}

	//------------------------------------------------------------------------------------------------
	bool IsShiftPolicyOn()
	{
		return m_bShiftPolicy;
	}

	//------------------------------------------------------------------------------------------------
	int GetPolicyShifts()
	{
		return m_iPolicyShifts;
	}

	//------------------------------------------------------------------------------------------------
	int GetWakeCalls()
	{
		return m_iWakeCalls;
	}

	//------------------------------------------------------------------------------------------------
	int GetWakeAsleep()
	{
		return m_iWakeAsleep;
	}

	//------------------------------------------------------------------------------------------------
	int GetSleepWakes()
	{
		return m_iSleepWakes;
	}

	//------------------------------------------------------------------------------------------------
	//! Stuck guard for the probes that only need motion (e05, e06a; never e01/e02a, whose move checks decide
	//! ForceEnableSimulation): afterS after each drive start, a car whose speed never reached belowKmh since the
	//! release latches ForceSimulation, unless it is already forced. afterS <= 0 turns it off.
	void SetStuckForce(float afterS, float belowKmh)
	{
		m_fStuckAfterS = afterS;
		m_fStuckBelowKmh = belowKmh;
	}

	//------------------------------------------------------------------------------------------------
	//! true once after the stuck guard latched ForceEnableSimulation; the caller logs DescribeStuck as a WARNING
	bool ConsumeStuckWarning()
	{
		if (!m_bStuckWarningDue)
			return false;

		m_bStuckWarningDue = false;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! One log line for the stuck guard's latch
	string DescribeStuck()
	{
		return string.Format("car stuck: below %1 km/h (max %2) for %3 s after a drive start, ForceEnableSimulation latched from now on (drive.stuck_forces %4, physics sleep re-wakes %5)", m_fStuckBelowKmh.ToString(-1, 1), m_fMaxKmhSinceDriveStart.ToString(-1, 2), m_fStuckAfterS.ToString(-1, 1), m_iStuckForces, m_iSleepWakes);
	}

	//------------------------------------------------------------------------------------------------
	//! Run header lines of the gearbox (vehicle.gearbox, vehicle.gear_index_*: TBD-7), the engine rpm marks, the shift
	//! policy (gear_policy.*) and the physics wake rule
	void AppendGearboxLines(notnull array<string> lines)
	{
		string kind = "manual";
		if (m_bAutomaticGearbox)
			kind = "automatic";

		lines.Insert("vehicle.gearbox=" + kind);
		lines.Insert("vehicle.gear_index_neutral=" + m_iGearNeutral.ToString());
		lines.Insert("vehicle.gear_index_first=" + m_iGearFirst.ToString());
		lines.Insert("vehicle.gear_index_top=" + m_iGearTop.ToString());
		lines.Insert("vehicle.gear_index_rule=neutral gears-forward-1, first gears-forward, top gears-1, reverse below neutral (vanilla SCR_ToggleHandbrakeAction, SCR_VehicleGearCondition)");
		if (m_Simulation)
		{
			float rpmIdle = m_Simulation.EngineGetRPMIdle();
			float rpmPeakPower = m_Simulation.EngineGetRPMPeakPower();
			float rpmMax = m_Simulation.EngineGetRPMMax();
			lines.Insert("vehicle.rpm_idle=" + rpmIdle.ToString(-1, 1));
			lines.Insert("vehicle.rpm_peak_power=" + rpmPeakPower.ToString(-1, 1));
			lines.Insert("vehicle.rpm_max=" + rpmMax.ToString(-1, 1));
		}

		float reassert = GEAR_REASSERT_S;
		lines.Insert("gear_policy.active=" + DC_LaunchGuard.BoolText(m_bShiftPolicy));
		lines.Insert("gear_policy.upshift_rpm=" + m_fUpshiftRpm.ToString(-1, 1));
		lines.Insert("gear_policy.downshift_rpm=" + m_fDownshiftRpm.ToString(-1, 1));
		lines.Insert("gear_policy.min_shift_s=" + m_fMinShiftS.ToString(-1, 3));
		lines.Insert("gear_policy.first_below_kmh=" + m_fFirstBelowKmh.ToString(-1, 3));
		lines.Insert("gear_policy.clutch_dip_s=" + m_fShiftDipS.ToString(-1, 3));
		lines.Insert("gear_policy.reassert_s=" + reassert.ToString(-1, 3));
		lines.Insert("gear_policy.rule=manual gearbox and cmd gear -1: neutral while the handbrake is commanded, first gear at its release (SetGear before SetThrottle), then first..top by rpm; SetGear on change, again after reassert_s without readback, and after every explicit gear (the policy continues from it); SetGear(neutral) at the probe stop; cmd_gear = the policy gear");
		lines.Insert("physics_wake.rule=Physics.SetActive(ActiveState.ACTIVE) at every release of the commanded handbrake (vanilla SCR_ToggleHandbrakeAction), and on every later step with the handbrake released whose body is asleep (Physics.IsActive false, SCR_DestructionTireComponent), on top of ForceEnableSimulation");
	}

	//------------------------------------------------------------------------------------------------
	//! Run header lines appended at the probe stop: what the shift policy and the physics wake did
	void AppendDriveSummaryLines(notnull array<string> lines)
	{
		lines.Insert("gear_policy.shifts=" + m_iPolicyShifts.ToString());
		lines.Insert("gear_policy.setgear_calls=" + m_iSetGearCalls.ToString());
		lines.Insert("gear_policy.reasserts=" + m_iGearReasserts.ToString());
		lines.Insert("physics_wake.calls=" + m_iWakeCalls.ToString());
		lines.Insert("physics_wake.asleep_before=" + m_iWakeAsleep.ToString());
		lines.Insert("physics_wake.no_physics=" + m_iWakeNoPhysics.ToString());
		lines.Insert("physics_wake.sleep_wakes=" + m_iSleepWakes.ToString());
		lines.Insert("drive.stuck_force_after_s=" + m_fStuckAfterS.ToString(-1, 3));
		lines.Insert("drive.stuck_force_below_kmh=" + m_fStuckBelowKmh.ToString(-1, 3));
		lines.Insert("drive.stuck_forces=" + m_iStuckForces.ToString());
		lines.Insert("drive.force_sim=" + DC_LaunchGuard.BoolText(m_bForced));
	}

	//------------------------------------------------------------------------------------------------
	//! One log line: gearbox kind, indices and the shift policy
	string DescribeGearbox()
	{
		string kind = "manual";
		if (m_bAutomaticGearbox)
			kind = "automatic";

		string policy = "off (automatic gearbox: no SetGear unless a phase sets one)";
		if (m_bShiftPolicy)
			policy = string.Format("on: neutral with the handbrake, first gear at its release, up at >= %1 rpm, down at <= %2 rpm, >= %3 s apart, first below %4 km/h, clutch dip %5 s", m_fUpshiftRpm.ToString(-1, 0), m_fDownshiftRpm.ToString(-1, 0), m_fMinShiftS.ToString(-1, 2), m_fFirstBelowKmh.ToString(-1, 1), m_fShiftDipS.ToString(-1, 2));
		else if (!m_bAutomaticGearbox)
			policy = "off (unusable gear counts)";

		return string.Format("gearbox %1: %2 gears, %3 forward -> neutral %4, first %5, top %6 (vanilla rule); shift policy %7", kind, m_iGearsCount, m_iForwardGearsCount, m_iGearNeutral, m_iGearFirst, m_iGearTop, policy);
	}

	//------------------------------------------------------------------------------------------------
	//! One log line: shifts, SetGear calls, physics wakes and the stuck guard so far
	string DescribeDrive()
	{
		string drive = string.Format("gear policy %1 shifts, %2 SetGear calls (%3 re-asserted); physics wake %4 calls, %5 with the body asleep before, %6 without physics; force_sim %7", m_iPolicyShifts, m_iSetGearCalls, m_iGearReasserts, m_iWakeCalls, m_iWakeAsleep, m_iWakeNoPhysics, DC_LaunchGuard.BoolText(m_bForced));
		return drive + string.Format("; %1 sleep re-wakes while driving; stuck force latches %2", m_iSleepWakes, m_iStuckForces);
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	IEntity GetVehicle()
	{
		return m_Vehicle;
	}

	//------------------------------------------------------------------------------------------------
	VehicleWheeledSimulation GetSimulation()
	{
		return m_Simulation;
	}

	//------------------------------------------------------------------------------------------------
	CarControllerComponent GetController()
	{
		return m_Controller;
	}

	//------------------------------------------------------------------------------------------------
	//! Mode A start: SetPersistentHandBrake(false); ForceStartEngine() when the engine is off. Sensitivities untouched.
	void StartModeA()
	{
		if (!m_Controller)
			return;

		m_Controller.SetPersistentHandBrake(false);
		if (!m_Controller.IsEngineOn())
			m_Controller.ForceStartEngine();
	}

	//------------------------------------------------------------------------------------------------
	//! VehicleWheeledSimulation.ForceEnableSimulation(), latched (column force_sim): ApplyStep repeats the call every
	//! step from now on, because the engine documents no duration for it
	void ForceSimulation()
	{
		if (!m_Simulation)
			return;

		m_Simulation.ForceEnableSimulation();
		m_bForced = true;
	}

	//------------------------------------------------------------------------------------------------
	bool IsSimulationForced()
	{
		return m_bForced;
	}

	//------------------------------------------------------------------------------------------------
	//! Open-loop command, held until changed. gear >= 0: SetGear(gear) every step; gear < 0: the shift policy picks
	//! the gear on a manual gearbox, no SetGear call on an automatic one. Clamped: steer [-1,1], others [0,1].
	//! Clears a speed hold.
	void SetCommand(float steer, float throttle, float brake, float clutch, int gear, bool handbrake)
	{
		m_fCmdSteer = Math.Clamp(steer, -1, 1);
		m_fCmdThrottle = Math.Clamp(throttle, 0, 1);
		m_fCmdBrake = Math.Clamp(brake, 0, 1);
		m_fCmdClutch = Math.Clamp(clutch, 0, 1);
		m_iCmdGear = gear;
		if (gear < 0)
			m_iCmdGear = -1;

		m_bCmdHandbrake = handbrake;
		m_bSpeedHold = false;
		m_fHoldKmh = -1;
		m_fHoldIntegral = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Closed-loop speed hold: PI on GetSpeedKmh -> throttle or brake (never both); steer held open loop; clutch 1.
	//! Calling it again with a new target or steering keeps the integrator.
	void SetSpeedHold(float targetKmh, float steer)
	{
		if (!m_bSpeedHold)
			m_fHoldIntegral = 0;

		m_bSpeedHold = true;
		m_fHoldKmh = Math.Max(targetKmh, 0);
		m_fCmdSteer = Math.Clamp(steer, -1, 1);
		m_fCmdClutch = 1;
		m_iCmdGear = -1;
		m_bCmdHandbrake = false;
	}

	//------------------------------------------------------------------------------------------------
	//! Probe-only PI, not llc_v1: throttle = kp * e + ki * integral(e) (e in km/h), brake = -that * brakeScale
	void SetSpeedGains(float kp, float ki, float brakeScale)
	{
		m_fKp = kp;
		m_fKi = ki;
		m_fBrakeScale = brakeScale;
	}

	//------------------------------------------------------------------------------------------------
	bool IsSpeedHeld()
	{
		return m_bSpeedHold;
	}

	//------------------------------------------------------------------------------------------------
	//! Call from OnProbeSimulate: runs the PI and the shift policy, wakes the physics at a release of the commanded
	//! handbrake and an asleep body on the later released steps, runs the stuck guard, calls
	//! SetSteering/SetBreak/SetGear/SetClutch/SetThrottle every step (gear before clutch and throttle, the order of
	//! VehicleCinematicTrack.c:149-162), then reads input_pre_*. SetGear: an explicit gear (SetCommand gear >= 0) every
	//! step, which the policy then continues from (AdoptExplicitGear); a policy gear on change and again after
	//! GEAR_REASSERT_S without readback (ApplyPolicyGear)
	void ApplyStep(float timeSlice)
	{
		if (!m_Simulation)
			return;

		if (m_bSpeedHold)
			RunSpeedHold(timeSlice);

		if (m_bForced)
			m_Simulation.ForceEnableSimulation();

		// 1.9 robustness: a controller that sets the persistent handbrake again is overridden and counted
		if (m_Controller && !m_bCmdHandbrake && m_Controller.GetPersistentHandBrake())
		{
			m_Controller.SetPersistentHandBrake(false);
			m_iPersistentHandbrakeResets++;
		}

		// Every drive start releases the commanded handbrake: wake the physics there, as vanilla does at a release.
		// While the handbrake stays released, a body found asleep is woken again and the stuck guard runs.
		if (!m_bCmdHandbrake && m_bAppliedHandbrake)
		{
			WakePhysics();
			m_fSinceDriveStartS = 0;
			m_fMaxKmhSinceDriveStart = 0;
		}
		else if (!m_bCmdHandbrake)
		{
			RewakeAsleepBody();
			CheckStuck(timeSlice);
		}

		m_bAppliedHandbrake = m_bCmdHandbrake;
		m_iAppliedGear = m_iCmdGear;
		m_fAppliedClutch = m_fCmdClutch;
		if (m_iCmdGear < 0 && m_bShiftPolicy)
		{
			m_iAppliedGear = RunShiftPolicy(timeSlice);
			if (m_fDipLeftS > 0)
			{
				m_fAppliedClutch = 0;
				m_fDipLeftS -= timeSlice;
			}
		}

		m_Simulation.SetSteering(m_fCmdSteer);
		m_Simulation.SetBreak(m_fCmdBrake, m_bCmdHandbrake);
		if (m_iCmdGear >= 0)
		{
			m_Simulation.SetGear(m_iCmdGear);
			AdoptExplicitGear(m_iCmdGear);
		}
		else if (m_iAppliedGear >= 0)
		{
			ApplyPolicyGear(m_iAppliedGear, timeSlice);
		}

		m_Simulation.SetClutch(m_fAppliedClutch);
		m_Simulation.SetThrottle(m_fCmdThrottle);

		m_fPreSteer = m_Simulation.GetSteering();
		m_fPreThrottle = m_Simulation.GetThrottle();
		m_fPreBrake = m_Simulation.GetBrake();
		m_fPreClutch = m_Simulation.GetClutch();
		m_iPreGear = m_Simulation.GetGear();
		m_bPreHandbrake = m_Simulation.IsHandbrakeOn();
	}

	//------------------------------------------------------------------------------------------------
	//! PI step with conditional integration (no wind-up while the output saturates in the error's direction)
	protected void RunSpeedHold(float timeSlice)
	{
		float error = m_fHoldKmh - m_Simulation.GetSpeedKmh();
		float output = m_fKp * error + m_fKi * m_fHoldIntegral;
		bool saturatedUp = output >= 1 && error > 0;
		bool saturatedDown = output <= -1 && error < 0;
		if (!saturatedUp && !saturatedDown && timeSlice > 0)
			m_fHoldIntegral += error * timeSlice;

		if (m_fKi > 0)
			m_fHoldIntegral = Math.Clamp(m_fHoldIntegral, -1 / m_fKi, 1 / m_fKi);

		output = m_fKp * error + m_fKi * m_fHoldIntegral;
		if (output >= 0)
		{
			m_fCmdThrottle = Math.Clamp(output, 0, 1);
			m_fCmdBrake = 0;
		}
		else
		{
			m_fCmdThrottle = 0;
			m_fCmdBrake = Math.Clamp(-output * m_fBrakeScale, 0, 1);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Gear of this step for a manual gearbox without an explicit gear. Neutral while the handbrake is commanded and
	//! first gear at its release, the indices derived as vanilla SCR_ToggleHandbrakeAction.c:99-108 does; the park/drive
	//! choice is the probe's own (vanilla PerformAction, :93-108, releases into neutral and parks in first gear, i.e.
	//! in gear for a parked player car; the probe needs first gear to drive off). While driving only first..top:
	//! first gear below m_fFirstBelowKmh or with SetFirstGearOnly, else up one gear at rpm >= m_fUpshiftRpm and down
	//! one at rpm <= m_fDownshiftRpm, at most one change per m_fMinShiftS. rpm and speed are those of the last
	//! ReadPost, so the policy is deterministic in the recorded state. A change into a forward gear starts the clutch
	//! dip (m_fShiftDipS; 0 = none).
	protected int RunShiftPolicy(float timeSlice)
	{
		m_fSinceShiftS += timeSlice;
		int gear = m_iPolicyGear;
		if (m_bCmdHandbrake)
		{
			gear = m_iGearNeutral;
			m_fDipLeftS = 0;
		}
		else if (gear < m_iGearFirst || gear > m_iGearTop)
		{
			gear = m_iGearFirst;
		}
		else if (m_bFirstGearOnly || m_fSpeedKmh < m_fFirstBelowKmh)
		{
			gear = m_iGearFirst;
		}
		else if (m_fSinceShiftS >= m_fMinShiftS)
		{
			if (m_fRpm >= m_fUpshiftRpm && gear < m_iGearTop)
				gear++;
			else if (m_fRpm <= m_fDownshiftRpm && gear > m_iGearFirst)
				gear--;
		}

		if (gear == m_iPolicyGear)
			return gear;

		if (m_iPolicyGear >= m_iGearFirst && gear >= m_iGearFirst)
			m_iPolicyShifts++;

		if (gear >= m_iGearFirst)
			m_fDipLeftS = m_fShiftDipS;

		m_iPolicyGear = gear;
		m_fSinceShiftS = 0;
		return gear;
	}

	//------------------------------------------------------------------------------------------------
	//! SetGear for the policy gear: when it changes (the vanilla pattern, VehicleCinematicTrack.c:152-156), and again
	//! when GEAR_REASSERT_S after the last call neither GetGear nor the controller's future gear (last ReadPost) shows it
	protected void ApplyPolicyGear(int gear, float timeSlice)
	{
		m_fSinceSetGearS += timeSlice;
		bool changed = gear != m_iPolicySetGear;
		bool shown = m_iPostGear == gear || m_iFutGear == gear;
		if (!changed && (shown || m_fSinceSetGearS < GEAR_REASSERT_S))
			return;

		if (!changed)
			m_iGearReasserts++;

		m_Simulation.SetGear(gear);
		m_iPolicySetGear = gear;
		m_fSinceSetGearS = 0;
		m_iSetGearCalls++;
	}

	//------------------------------------------------------------------------------------------------
	//! After an explicit SetGear (gear_scripted): the policy continues from that gear (a forward gear, else from first
	//! gear at the next release) and calls SetGear again at its next step, so neither the stop after an explicit phase
	//! nor Park keeps the gear cached before it; the explicit gear counts as the last change (m_fMinShiftS)
	protected void AdoptExplicitGear(int gear)
	{
		m_iPolicyGear = -1;
		if (gear >= m_iGearFirst && gear <= m_iGearTop)
			m_iPolicyGear = gear;

		m_iPolicySetGear = -1;
		m_fSinceShiftS = 0;
		m_fDipLeftS = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! vehicleEntity.GetPhysics().SetActive(ActiveState.ACTIVE) as vanilla SCR_ToggleHandbrakeAction.c:96-97 does when
	//! the parking brake is released; counts the calls and how often the body was asleep (IsActive false) before
	protected void WakePhysics()
	{
		Physics physics = m_Vehicle.GetPhysics();
		if (!physics)
		{
			m_iWakeNoPhysics++;
			return;
		}

		if (!physics.IsActive())
			m_iWakeAsleep++;

		physics.SetActive(ActiveState.ACTIVE);
		m_iWakeCalls++;
	}

	//------------------------------------------------------------------------------------------------
	//! On every step with the handbrake released after the release step: a body found asleep (Physics.IsActive false)
	//! is woken again, the vanilla SCR_DestructionTireComponent.c:155-156 pattern (DD-4-min Mode A does the same)
	protected void RewakeAsleepBody()
	{
		if (!m_Vehicle)
			return;

		Physics body = m_Vehicle.GetPhysics();
		if (!body || body.IsActive())
			return;

		body.SetActive(ActiveState.ACTIVE);
		m_iSleepWakes++;
	}

	//------------------------------------------------------------------------------------------------
	//! Stuck guard (SetStuckForce), once per drive start: m_fStuckAfterS after the release, a car whose speed (last
	//! ReadPost) never reached m_fStuckBelowKmh latches ForceSimulation unless it is already forced
	protected void CheckStuck(float timeSlice)
	{
		if (m_fStuckAfterS <= 0 || m_fSinceDriveStartS < 0)
			return;

		m_fSinceDriveStartS += timeSlice;
		m_fMaxKmhSinceDriveStart = Math.Max(m_fMaxKmhSinceDriveStart, Math.AbsFloat(m_fSpeedKmh));
		if (m_fSinceDriveStartS < m_fStuckAfterS)
			return;

		m_fSinceDriveStartS = -1;
		if (m_bForced || m_fMaxKmhSinceDriveStart >= m_fStuckBelowKmh)
			return;

		ForceSimulation();
		m_iStuckForces++;
		m_bStuckWarningDue = true;
	}

	//------------------------------------------------------------------------------------------------
	//! Call from OnProbePostSimulate: reads input_post_* and the vehicle state of this step
	void ReadPost()
	{
		if (!m_Simulation || !m_Vehicle)
			return;

		m_fPostSteer = m_Simulation.GetSteering();
		m_fPostThrottle = m_Simulation.GetThrottle();
		m_fPostBrake = m_Simulation.GetBrake();
		m_fPostClutch = m_Simulation.GetClutch();
		m_iPostGear = m_Simulation.GetGear();
		m_bPostHandbrake = m_Simulation.IsHandbrakeOn();

		if (m_Controller)
		{
			m_iCurGear = m_Controller.GetCurrentGear();
			m_iFutGear = m_Controller.GetFutureGear();
		}

		m_fRpm = m_Simulation.EngineGetRPM();
		m_bEngineOn = m_Simulation.EngineIsOn();
		m_fSpeedKmh = m_Simulation.GetSpeedKmh();
		m_vVelLocal = m_Simulation.GetVelocityLocal();
		m_vAngVelLocal = m_Simulation.GetAngularVelocityLocal();
		m_vAccLocal = m_Simulation.GetAccelerationLocal();

		vector transform[4];
		m_Vehicle.GetWorldTransform(transform);
		m_vPos = transform[3];
		m_vFwd = transform[2];
		m_fHeadingDeg = Math.Atan2(m_vFwd[0], m_vFwd[2]) * Math.RAD2DEG;

		m_bGroundContact = m_Simulation.HasAnyGroundContact();
		int contacts = 0;
		for (int i = 0, count = m_Simulation.WheelCount(); i < count; i++)
		{
			if (m_Simulation.WheelHasContact(i))
				contacts++;
		}

		m_iWheelsContact = contacts;
	}

	//------------------------------------------------------------------------------------------------
	//! BaseGameEntity.Teleport to (x, GetSurfaceY(x, z) + liftM, z) facing headingDeg, zero linear and angular
	//! velocity, command brake 1 + handbrake (the caller settles before driving). false when the vehicle is missing.
	//! Callers run it outside the physics step (OnProbeFrame or OnProbeStart), never from OnProbeSimulate or
	//! OnProbePostSimulate: every probe that drives keeps its reset segment waiting for the frame teleport.
	bool Teleport(vector posXZ, float headingDeg, float liftM)
	{
		BaseGameEntity entity = BaseGameEntity.Cast(m_Vehicle);
		World world = GetGame().GetWorld();
		if (!entity || !world)
			return false;

		float rad = headingDeg * Math.DEG2RAD;
		float sinH = Math.Sin(rad);
		float cosH = Math.Cos(rad);
		vector transform[4];
		transform[0] = Vector(cosH, 0, -sinH);
		transform[1] = Vector(0, 1, 0);
		transform[2] = Vector(sinH, 0, cosH);
		transform[3] = Vector(posXZ[0], world.GetSurfaceY(posXZ[0], posXZ[2]) + liftM, posXZ[2]);
		entity.Teleport(transform);

		Physics physics = entity.GetPhysics();
		if (physics)
		{
			physics.SetVelocity(vector.Zero);
			physics.SetAngularVelocity(vector.Zero);
		}

		SetCommand(0, 0, 1, 1, -1, true);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Brake 1 + handbrake, throttle 0, steering 0 and, on a manual gearbox with the shift policy, SetGear(neutral)
	//! always (also when the cached policy gear already reads neutral, e.g. after an interrupted gear_scripted), applied
	//! at once (no physics step needed). For probe stops: a driverless car keeps the last setter values when nobody
	//! calls the setters any more.
	void Park()
	{
		SetCommand(0, 0, 1, 1, -1, true);
		m_iPolicySetGear = -1;
		ApplyStep(0);
	}

	//------------------------------------------------------------------------------------------------
	//! Of the last ReadPost
	float GetSpeedKmh()
	{
		return m_fSpeedKmh;
	}

	//------------------------------------------------------------------------------------------------
	vector GetVelocityLocal()
	{
		return m_vVelLocal;
	}

	//------------------------------------------------------------------------------------------------
	vector GetAngularVelocityLocal()
	{
		return m_vAngVelLocal;
	}

	//------------------------------------------------------------------------------------------------
	bool IsEngineOn()
	{
		return m_bEngineOn;
	}

	//------------------------------------------------------------------------------------------------
	//! World position of the last ReadPost
	vector GetPosition()
	{
		return m_vPos;
	}

	//------------------------------------------------------------------------------------------------
	//! atan2(fwd_x, fwd_z) in degrees of the last ReadPost (0 = +Z, right turn = increasing)
	float GetHeadingDeg()
	{
		return m_fHeadingDeg;
	}

	//------------------------------------------------------------------------------------------------
	//! Commanded steering of the last ApplyStep (after the clamp)
	float GetCommandSteer()
	{
		return m_fCmdSteer;
	}

	//------------------------------------------------------------------------------------------------
	int GetPersistentHandbrakeResets()
	{
		return m_iPersistentHandbrakeResets;
	}

	//------------------------------------------------------------------------------------------------
	//! The actuator columns of section 5.3 (force_sim .. wheels_contact)
	static void AddStepColumns(notnull array<string> columns)
	{
		columns.Insert("force_sim");
		columns.Insert("cmd_steer");
		columns.Insert("cmd_throttle");
		columns.Insert("cmd_brake");
		columns.Insert("cmd_clutch");
		columns.Insert("cmd_gear");
		columns.Insert("cmd_handbrake");
		columns.Insert("speed_hold_kmh");
		columns.Insert("input_pre_steer");
		columns.Insert("input_pre_throttle");
		columns.Insert("input_pre_brake");
		columns.Insert("input_pre_clutch");
		columns.Insert("input_pre_gear");
		columns.Insert("input_pre_handbrake");
		columns.Insert("input_post_steer");
		columns.Insert("input_post_throttle");
		columns.Insert("input_post_brake");
		columns.Insert("input_post_clutch");
		columns.Insert("input_post_gear");
		columns.Insert("input_post_handbrake");
		columns.Insert("cur_gear");
		columns.Insert("fut_gear");
		columns.Insert("rpm");
		columns.Insert("engine_on");
		columns.Insert("speed_kmh");
		DC_ProbeCsv.AddVectorColumns(columns, "vel_l");
		DC_ProbeCsv.AddVectorColumns(columns, "angvel_l");
		DC_ProbeCsv.AddVectorColumns(columns, "acc_l");
		DC_ProbeCsv.AddVectorColumns(columns, "pos");
		DC_ProbeCsv.AddVectorColumns(columns, "fwd");
		columns.Insert("heading_deg");
		columns.Insert("ground_contact");
		columns.Insert("wheels_contact");
	}

	//------------------------------------------------------------------------------------------------
	//! The values of the last step, in AddStepColumns order. Decimals: inputs 4, speeds 3, angular velocity 6,
	//! acceleration 4, positions 3, unit axes 6, heading 5. cmd_clutch and cmd_gear are the applied values: the
	//! shift policy's gear (-1 only without a gear command) and clutch 0 during its dip
	void AddStepValues(notnull DC_ProbeRow row)
	{
		row.AddBool(m_bForced);
		row.AddFloat(m_fCmdSteer, 4);
		row.AddFloat(m_fCmdThrottle, 4);
		row.AddFloat(m_fCmdBrake, 4);
		row.AddFloat(m_fAppliedClutch, 4);
		row.AddInt(m_iAppliedGear);
		row.AddBool(m_bCmdHandbrake);
		row.AddFloat(m_fHoldKmh, 3);
		row.AddFloat(m_fPreSteer, 4);
		row.AddFloat(m_fPreThrottle, 4);
		row.AddFloat(m_fPreBrake, 4);
		row.AddFloat(m_fPreClutch, 4);
		row.AddInt(m_iPreGear);
		row.AddBool(m_bPreHandbrake);
		row.AddFloat(m_fPostSteer, 4);
		row.AddFloat(m_fPostThrottle, 4);
		row.AddFloat(m_fPostBrake, 4);
		row.AddFloat(m_fPostClutch, 4);
		row.AddInt(m_iPostGear);
		row.AddBool(m_bPostHandbrake);
		row.AddInt(m_iCurGear);
		row.AddInt(m_iFutGear);
		row.AddFloat(m_fRpm, 1);
		row.AddBool(m_bEngineOn);
		row.AddFloat(m_fSpeedKmh, 3);
		row.AddVector(m_vVelLocal, 3);
		row.AddVector(m_vAngVelLocal, 6);
		row.AddVector(m_vAccLocal, 4);
		row.AddVector(m_vPos, 3);
		row.AddVector(m_vFwd, 6);
		row.AddFloat(m_fHeadingDeg, 5);
		row.AddBool(m_bGroundContact);
		row.AddInt(m_iWheelsContact);
	}
}

//------------------------------------------------------------------------------------------------
//! What an E-01 phase or an E-02a cell excites
enum DC_EProbeSequenceKind
{
	MOVE_CHECK,
	MOVE_CHECK_FORCED,
	STEER_PRBS,
	STEER_REVERSAL,
	THROTTLE,
	BRAKE,
	GEAR_OBSERVE,
	GEAR_SCRIPTED,
	E02A_CELL
}

//------------------------------------------------------------------------------------------------
//! Segment of a phase (column segment)
enum DC_EProbeSequenceSegment
{
	RESET,		//!< waiting for the teleport (OnProbeFrame); brake 1 + handbrake
	SETTLE,		//!< m_fSettleS of brake 1 + handbrake
	ACCEL,		//!< speed hold to the phase's target, steering 0
	EXCITE,		//!< the phase's excitation
	STOP		//!< brake 1 until below m_fSpeedTolKmh
}

//------------------------------------------------------------------------------------------------
//! One E-01 phase or one E-02a cell of DC_ProbeSequences, in run order
class DC_ProbeSequencePhase
{
	DC_EProbeSequenceKind m_eKind;
	string m_sName;					//!< e01: move_check .. gear_scripted, throttle_0.30; e02a: cell_<idx>
	int m_iIdx = -1;				//!< phase_idx / cell_idx, assigned when the phase starts
	float m_fTargetKmh = -1;		//!< speed-hold target of accel / excite, -1 = none
	float m_fAmplitude;				//!< steering amplitude, throttle or brake of the excitation
	float m_fLateralM;				//!< start offset from the centre line (e02a)
	int m_iStartStep = -1;
	int m_iExciteStartStep = -1;
	int m_iEndStep = -1;
	float m_fStartWorldMs = -1;
	float m_fEndWorldMs = -1;
	bool m_bForceSim;
	string m_sResult;				//!< ok no_motion aborted_area no_speed engine_off segment_cap error stopped skipped
	string m_sNote;					//!< tokens key:value joined by '/', CSV-safe
	float m_fMaxSpeedKmh;
	float m_fTurnDeg;				//!< heading change during excite (right turn positive)
	int m_iEngineOffSteps;			//!< steps with engine_on 0 after the settle
	int m_iHandbrakeResetsAtStart;
	int m_iWakesAtStart;			//!< DC_ProbeActuator counters at the phase start (phase notes wakes, asleep_at_wake, shifts)
	int m_iWakeAsleepAtStart;
	int m_iShiftsAtStart;
	int m_iSleepWakesAtStart;		//!< phase note sleep_wakes

	//------------------------------------------------------------------------------------------------
	void DC_ProbeSequencePhase(DC_EProbeSequenceKind kind, string name, float targetKmh, float amplitude, float lateralM)
	{
		m_eKind = kind;
		m_sName = name;
		m_fTargetKmh = targetKmh;
		m_fAmplitude = amplitude;
		m_fLateralM = lateralM;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends one note token (only [A-Za-z0-9_.:/-] characters)
	void AddNote(string token)
	{
		if (m_sNote.IsEmpty())
			m_sNote = token;
		else
			m_sNote += "/" + token;
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "T0-ENG probes E-01 (Mode A actuation identification) and E-02a (first-pass steer to curvature circles) on DC_Ego")]
class DC_ProbeSequencesClass : DC_ProbeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeSequences : DC_ProbeComponent
{
	static const string PROBE_E01 = "e01";
	static const string PROBE_E02A = "e02a";
	static const float TIMEOUT_E01_S = 900;
	static const float TIMEOUT_E02A_S = 1500;
	//! Grid of MeasureRelief over the area [m]
	static const float RELIEF_GRID_M = 10;
	//! Safety cap of one brake excite or stop segment [s]; a valid run never reaches it
	static const float SEGMENT_CAP_S = 60;
	//! E-02a cells start this far inside the area edge on the side away from the turn [m] (PROBE_DESIGN.md 5.3)
	static const float E02A_EDGE_OFFSET_M = 5;

	// === AREA ===
	[Attribute("659.6 0 2572.4", UIWidgets.Coords, "Start of the test area (x, z used, y ignored): engine road dump E-08a 2026-10-22: Arland main runway (road 83, 60 m x 1217 m, y 38.9-39.2), 58 m in from its west end; the decoder runway (road 425) was misplaced", category: "Area")]
	protected vector m_vAreaStart;

	[Attribute("75.11", UIWidgets.Slider, "Heading of the area centre line [deg], 0 = +Z (north), clockwise seen from above", "0 360 0.01", category: "Area")]
	protected float m_fAreaHeadingDeg;

	[Attribute("1100", UIWidgets.Slider, "Length of the usable area along the centre line [m]", "10 5000 1", category: "Area")]
	protected float m_fAreaLengthM;

	[Attribute("25", UIWidgets.Slider, "Half width of the usable area [m]", "2 200 0.5", category: "Area")]
	protected float m_fAreaHalfWidthM;

	[Attribute("3", UIWidgets.Slider, "Margin inside the area edges; leaving the shrunk rectangle while driving ends the phase with aborted_area [m]", "0 50 0.5", category: "Area")]
	protected float m_fAreaMarginM;

	[Attribute("5", UIWidgets.Slider, "Refuse to drive when max - min of the terrain height over the area (10 m grid) exceeds this [m]", "0 100 0.5", category: "Area")]
	protected float m_fMaxAreaReliefM;

	[Attribute("0.3", UIWidgets.Slider, "Teleport height above GetSurfaceY [m]", "0 5 0.05", category: "Area")]
	protected float m_fSpawnLiftM;

	[Attribute("3", UIWidgets.Slider, "Settle time after each teleport, brake 1 + handbrake [s]", "0.5 30 0.5", category: "Area")]
	protected float m_fSettleS;

	// === CONTROL ===
	[Attribute("0.08", UIWidgets.Slider, "Probe speed hold: proportional gain [1/(km/h)] (probe-only PI, not llc_v1)", "0 1 0.005", category: "Control")]
	protected float m_fSpeedKp;

	[Attribute("0.02", UIWidgets.Slider, "Probe speed hold: integral gain [1/(km/h s)]", "0 1 0.005", category: "Control")]
	protected float m_fSpeedKi;

	[Attribute("0.5", UIWidgets.Slider, "Probe speed hold: brake = -PI output x this", "0 2 0.05", category: "Control")]
	protected float m_fBrakeScale;

	[Attribute("1", UIWidgets.Slider, "Speed tolerance [km/h]: accel ends at target - tol; brake and stop segments end below tol", "0.1 10 0.1", category: "Control")]
	protected float m_fSpeedTolKmh;

	// === E-01 ===
	[Attribute("0.5", UIWidgets.Slider, "Move check throttle", "0 1 0.05", category: "E-01")]
	protected float m_fMoveCheckThrottle;

	[Attribute("5", UIWidgets.Slider, "Move check duration [s]", "1 60 0.5", category: "E-01")]
	protected float m_fMoveCheckS;

	[Attribute("3", UIWidgets.Slider, "Move check: the car moves when its max speed reaches this [km/h]; else no_motion and a second check with ForceEnableSimulation", "0.5 50 0.5", category: "E-01")]
	protected float m_fMoveCheckMinKmh;

	[Attribute("0.3", UIWidgets.Slider, "PRBS steering amplitude (+-)", "0 1 0.01", category: "E-01")]
	protected float m_fPrbsAmplitude;

	[Attribute("20", UIWidgets.Slider, "Speed of the PRBS and reversal phases [km/h]", "1 120 1", category: "E-01")]
	protected float m_fPrbsSpeedKmh;

	[Attribute("0.2", UIWidgets.Slider, "PRBS bit length [s], converted to physics steps with the runner's step length (GetStepDtS)", "0.01 5 0.01", category: "E-01")]
	protected float m_fPrbsBitS;

	[Attribute("7", UIWidgets.Slider, "PRBS LFSR order (5..10); one period = 2^order - 1 bits", "5 10 1", category: "E-01")]
	protected int m_iPrbsOrder;

	[Attribute("1", UIWidgets.Slider, "PRBS LFSR seed (non-zero modulo 2^order)", "1 1023 1", category: "E-01")]
	protected int m_iPrbsSeed;

	[Attribute("0.3", UIWidgets.Slider, "Steering reversal amplitude (+-)", "0 1 0.01", category: "E-01")]
	protected float m_fReversalAmplitude;

	[Attribute("2", UIWidgets.Slider, "Steering reversal period [s]; the sign flips every half period", "0.1 20 0.1", category: "E-01")]
	protected float m_fReversalPeriodS;

	[Attribute("6", UIWidgets.Slider, "Number of steering reversals (then one half period of zero steering)", "1 50 1", category: "E-01")]
	protected int m_iReversalCount;

	[Attribute("0.3,0.6,1.0", UIWidgets.EditBox, "Throttle steps from standstill, comma-separated (one phase throttle_<v> each)", category: "E-01")]
	protected string m_sThrottleSteps;

	[Attribute("6", UIWidgets.Slider, "Hold time of each throttle step [s]", "0.5 60 0.5", category: "E-01")]
	protected float m_fThrottleHoldS;

	[Attribute("40", UIWidgets.Slider, "Speed from which the brake steps start [km/h]", "5 120 1", category: "E-01")]
	protected float m_fBrakeFromKmh;

	[Attribute("0.3,0.6,1.0", UIWidgets.EditBox, "Brake steps, comma-separated (one phase brake_<v> each, held until below the speed tolerance)", category: "E-01")]
	protected string m_sBrakeSteps;

	[Attribute("0.7", UIWidgets.Slider, "Throttle of the automatic gear observation (no SetGear; automatic gearbox only, skipped on a manual one)", "0 1 0.05", category: "E-01")]
	protected float m_fGearObserveThrottle;

	[Attribute("15", UIWidgets.Slider, "Duration of the automatic gear observation [s]", "1 120 1", category: "E-01")]
	protected float m_fGearObserveS;

	[Attribute("2,3,4,3,2", UIWidgets.EditBox, "Scripted SetGear sequence (gear indices, comma-separated); empty skips the phase. Indices as vanilla: neutral = gears - forward - 1, first = gears - forward (S105: 0 reverse, 1 neutral, 2..5 first..fourth; run header vehicle.gear_index_*)", category: "E-01")]
	protected string m_sGearSequence;

	[Attribute("3", UIWidgets.Slider, "Hold time of each scripted gear [s]", "0.5 30 0.5", category: "E-01")]
	protected float m_fGearHoldS;

	[Attribute("0.4", UIWidgets.Slider, "Throttle during the scripted gear phase", "0 1 0.05", category: "E-01")]
	protected float m_fGearScriptedThrottle;

	[Attribute("0", UIWidgets.Slider, "Clutch dip at each scripted gear change and at each shift-policy change into a forward gear [s]: clutch 0 for this long, then 1. 0 = no dip (the 1.8 sequence); a 1.9 re-run may need one (stall on clutch release, MIGRATION_1_9.md F8)", "0 2 0.05", category: "E-01")]
	protected float m_fGearClutchDipS;

	// === GEARBOX (shift policy of a manual gearbox, DC_ProbeActuator.SetShiftPolicy; e01 and e02a) ===
	[Attribute("4500", UIWidgets.Slider, "Shift policy (manual gearbox, phases without an explicit gear): up one gear at rpm >= this [S105 e04: idle 800, peak power 4800, max 8000]", "1000 10000 50", category: "Gearbox")]
	protected float m_fShiftUpRpm;

	[Attribute("1800", UIWidgets.Slider, "Shift policy: down one gear at rpm <= this (never below first while driving)", "500 8000 50", category: "Gearbox")]
	protected float m_fShiftDownRpm;

	[Attribute("1.5", UIWidgets.Slider, "Shift policy: minimum time between two gear changes [s]", "0.1 10 0.1", category: "Gearbox")]
	protected float m_fShiftMinS;

	[Attribute("8", UIWidgets.Slider, "Shift policy: first gear below this speed [km/h]", "0 50 0.5", category: "Gearbox")]
	protected float m_fShiftFirstBelowKmh;

	// === E-02a ===
	[Attribute("15:-0.5,-0.25,0.1,0.25,0.5;30:-0.25,-0.12,0.05,0.12,0.25;50:-0.12,-0.06,0.03,0.06,0.12", UIWidgets.EditBox, "Cells: groups '<speed km/h>:<steer>,<steer>,...' separated by ';' [design default: smaller steering at speed to stay near a_lat <= 4 m/s2]", category: "E-02a")]
	protected string m_sE02aGrid;

	[Attribute("25", UIWidgets.Slider, "Time to reach a speed-hold target before the phase ends with no_speed [s] (every accel segment, e01 and e02a)", "2 120 1", category: "E-02a")]
	protected float m_fE02aReachS;

	[Attribute("8", UIWidgets.Slider, "Hold time of a cell's steering at its speed [s]", "1 60 0.5", category: "E-02a")]
	protected float m_fE02aHoldS;

	[Attribute("330", UIWidgets.Slider, "A cell's excite ends once the heading changed by this much [deg]", "10 1080 5", category: "E-02a")]
	protected float m_fE02aMaxTurnDeg;

	[Attribute("-1", UIWidgets.Slider, "Turn direction of positive steering (+1 right, -1 left; r2 3.3: CarSteering +1 = left) when e01 did not measure it in this Play; only chooses the E-02a start side", "-1 1 2", category: "E-02a")]
	protected int m_iAssumedSteerSign;

	// === PLAY STATE (survives between e01 and e02a of one Play) ===
	protected float m_fSteerSignSum;			//!< sum of cmd_steer x heading rate [deg/s] over e01 steering excite steps
	protected int m_iSteerSignSamples;
	protected int m_iMeasuredSteerSign;			//!< +1 / -1 measured by e01 in this Play, 0 = not measured
	protected int m_iForceDecision = -1;		//!< -1 unknown, 0 not needed, 1 needed, 2 does not help
	protected string m_sForceDecisionSource;

	// === PROBE STATE (one running probe) ===
	protected string m_sActiveProbe;
	protected ref DC_ProbeContext m_Ctx;
	protected ref DC_ProbeArea m_Area;
	protected ref DC_ProbeActuator m_Actuator;
	protected ref DC_ProbePrbs m_Prbs;
	protected ref DC_ProbeRow m_Row;
	protected ref DC_ProbeCsv m_StepsCsv;
	protected ref DC_ProbeCsv m_PhasesCsv;		//!< e01_phases.csv or e02a_cells.csv
	protected ref array<ref DC_ProbeSequencePhase> m_aPhases;
	protected ref array<float> m_aThrottleSteps;
	protected ref array<float> m_aBrakeSteps;
	protected ref array<int> m_aGearSequence;
	protected int m_iCursor;					//!< index of the running phase in m_aPhases
	protected int m_iPhasesStarted;
	protected bool m_bProbeDone;
	protected DC_EProbeSequenceSegment m_eSegment;
	protected int m_iSegStep;					//!< engine steps covered in the current segment (CoveredSteps)
	protected int m_iPhaseStep;					//!< engine steps covered in the current phase
	protected int m_iPrbsBitsDrawn;				//!< PRBS bits drawn in the current segment
	protected bool m_bTeleportDone;
	protected bool m_bHasPrevHeading;
	protected float m_fPrevHeadingDeg;
	protected int m_iLastStep;
	protected float m_fPrbsSteer;
	protected bool m_bE02aRetried;
	protected int m_iSteerSignUsed;				//!< e02a
	protected string m_sSteerSignSource;		//!< e02a: measured_e01 or attribute

	// Step counts derived from the attributes and the runner's step length (GetStepDtS) at the probe start
	protected float m_fDt;
	protected int m_iSettleSteps;
	protected int m_iReachSteps;
	protected int m_iMoveCheckSteps;
	protected int m_iPrbsBitSteps;
	protected int m_iPrbsPeriodBits;
	protected int m_iReversalHalfSteps;
	protected int m_iThrottleHoldSteps;
	protected int m_iGearObserveSteps;
	protected int m_iGearHoldSteps;
	protected int m_iGearDipSteps;
	protected int m_iE02aHoldSteps;
	protected int m_iCapSteps;
	protected float m_fStartAlongM;				//!< every reset puts the car this far along the centre line

	//------------------------------------------------------------------------------------------------
	override void GetProbeIds(notnull array<string> outIds)
	{
		outIds.Insert(PROBE_E01);
		outIds.Insert(PROBE_E02A);
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
		if (probeId == PROBE_E02A)
			return TIMEOUT_E02A_S;

		return TIMEOUT_E01_S;
	}

	//------------------------------------------------------------------------------------------------
	//! Every attribute, dt and the derived PRBS step counts; e02a also steer_sign_used / steer_sign_source and the
	//! ForceEnableSimulation decision of this Play. The other run header extras of PROBE_DESIGN.md 5.3 (area.*,
	//! vehicle.*, rerun19.*) are appended by OnProbeStart (DC_ProbeContext.AppendHeaderLines)
	override void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_vAreaStart=" + DC_ProbeArea.FormatVector(m_vAreaStart, 3));
		outKeyValues.Insert("m_fAreaHeadingDeg=" + m_fAreaHeadingDeg.ToString(-1, 4));
		outKeyValues.Insert("m_fAreaLengthM=" + m_fAreaLengthM.ToString(-1, 3));
		outKeyValues.Insert("m_fAreaHalfWidthM=" + m_fAreaHalfWidthM.ToString(-1, 3));
		outKeyValues.Insert("m_fAreaMarginM=" + m_fAreaMarginM.ToString(-1, 3));
		outKeyValues.Insert("m_fMaxAreaReliefM=" + m_fMaxAreaReliefM.ToString(-1, 3));
		outKeyValues.Insert("m_fSpawnLiftM=" + m_fSpawnLiftM.ToString(-1, 3));
		outKeyValues.Insert("m_fSettleS=" + m_fSettleS.ToString(-1, 3));
		outKeyValues.Insert("m_fSpeedKp=" + m_fSpeedKp.ToString(-1, 4));
		outKeyValues.Insert("m_fSpeedKi=" + m_fSpeedKi.ToString(-1, 4));
		outKeyValues.Insert("m_fBrakeScale=" + m_fBrakeScale.ToString(-1, 4));
		outKeyValues.Insert("m_fSpeedTolKmh=" + m_fSpeedTolKmh.ToString(-1, 3));
		outKeyValues.Insert("m_fMoveCheckThrottle=" + m_fMoveCheckThrottle.ToString(-1, 4));
		outKeyValues.Insert("m_fMoveCheckS=" + m_fMoveCheckS.ToString(-1, 3));
		outKeyValues.Insert("m_fMoveCheckMinKmh=" + m_fMoveCheckMinKmh.ToString(-1, 3));
		outKeyValues.Insert("m_fPrbsAmplitude=" + m_fPrbsAmplitude.ToString(-1, 4));
		outKeyValues.Insert("m_fPrbsSpeedKmh=" + m_fPrbsSpeedKmh.ToString(-1, 3));
		outKeyValues.Insert("m_fPrbsBitS=" + m_fPrbsBitS.ToString(-1, 4));
		outKeyValues.Insert("m_iPrbsOrder=" + m_iPrbsOrder.ToString());
		outKeyValues.Insert("m_iPrbsSeed=" + m_iPrbsSeed.ToString());
		outKeyValues.Insert("m_fReversalAmplitude=" + m_fReversalAmplitude.ToString(-1, 4));
		outKeyValues.Insert("m_fReversalPeriodS=" + m_fReversalPeriodS.ToString(-1, 3));
		outKeyValues.Insert("m_iReversalCount=" + m_iReversalCount.ToString());
		outKeyValues.Insert("m_sThrottleSteps=" + m_sThrottleSteps);
		outKeyValues.Insert("m_fThrottleHoldS=" + m_fThrottleHoldS.ToString(-1, 3));
		outKeyValues.Insert("m_fBrakeFromKmh=" + m_fBrakeFromKmh.ToString(-1, 3));
		outKeyValues.Insert("m_sBrakeSteps=" + m_sBrakeSteps);
		outKeyValues.Insert("m_fGearObserveThrottle=" + m_fGearObserveThrottle.ToString(-1, 4));
		outKeyValues.Insert("m_fGearObserveS=" + m_fGearObserveS.ToString(-1, 3));
		outKeyValues.Insert("m_sGearSequence=" + m_sGearSequence);
		outKeyValues.Insert("m_fGearHoldS=" + m_fGearHoldS.ToString(-1, 3));
		outKeyValues.Insert("m_fGearScriptedThrottle=" + m_fGearScriptedThrottle.ToString(-1, 4));
		outKeyValues.Insert("m_fGearClutchDipS=" + m_fGearClutchDipS.ToString(-1, 3));
		outKeyValues.Insert("m_fShiftUpRpm=" + m_fShiftUpRpm.ToString(-1, 1));
		outKeyValues.Insert("m_fShiftDownRpm=" + m_fShiftDownRpm.ToString(-1, 1));
		outKeyValues.Insert("m_fShiftMinS=" + m_fShiftMinS.ToString(-1, 3));
		outKeyValues.Insert("m_fShiftFirstBelowKmh=" + m_fShiftFirstBelowKmh.ToString(-1, 3));
		outKeyValues.Insert("m_sE02aGrid=" + m_sE02aGrid);
		outKeyValues.Insert("m_fE02aReachS=" + m_fE02aReachS.ToString(-1, 3));
		outKeyValues.Insert("m_fE02aHoldS=" + m_fE02aHoldS.ToString(-1, 3));
		outKeyValues.Insert("m_fE02aMaxTurnDeg=" + m_fE02aMaxTurnDeg.ToString(-1, 3));
		outKeyValues.Insert("m_iAssumedSteerSign=" + m_iAssumedSteerSign.ToString());

		// dt_s: the runner's step length (DC_ProbeRunner STEP TIMING); engine_fixed_time_slice_s: GetFixedTimeSlice now
		World world = GetGame().GetWorld();
		float dt = GetStepDtS();
		float engineFixedSlice = 0;
		if (world)
			engineFixedSlice = world.GetFixedTimeSlice();

		outKeyValues.Insert("dt_s=" + dt.ToString(-1, 6));
		outKeyValues.Insert("engine_fixed_time_slice_s=" + engineFixedSlice.ToString(-1, 6));
		if (dt > 0)
		{
			int bitSteps = StepsFor(m_fPrbsBitS, dt);
			if (bitSteps < 1)
				bitSteps = 1;

			outKeyValues.Insert("prbs_bit_steps=" + bitSteps.ToString());
		}

		DC_ProbePrbs prbs = new DC_ProbePrbs();
		if (prbs.Init(m_iPrbsOrder, m_iPrbsSeed))
		{
			int periodBits = prbs.GetPeriod();
			outKeyValues.Insert("prbs_period_bits=" + periodBits.ToString());
		}

		if (probeId == PROBE_E02A)
		{
			string source;
			int steerSign = ChooseSteerSign(source);
			outKeyValues.Insert("steer_sign_used=" + steerSign.ToString());
			outKeyValues.Insert("steer_sign_source=" + source);
			outKeyValues.Insert("force_sim_decision=" + ForceDecisionText());
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Run header extras of PROBE_DESIGN.md 5.3 (area.*, vehicle.*, rerun19.* for e01), appended to run_header.txt;
	//! with an initialised actuator also its gearbox lines (vehicle.gearbox, vehicle.gear_index_*, gear_policy.*)
	protected void AppendRunHeaderExtras(string probeId, notnull DC_ProbeContext ctx, notnull IEntity ego, DC_ProbeActuator actuator, float relief, float minY, float maxY)
	{
		array<string> lines = {};
		m_Area.AppendHeaderLines(lines, "area");
		float grid = RELIEF_GRID_M;
		lines.Insert("area.margin_m=" + m_fAreaMarginM.ToString(-1, 3));
		lines.Insert("area.start_along_m=" + m_fStartAlongM.ToString(-1, 3));
		lines.Insert("area.relief_m=" + relief.ToString(-1, 3));
		lines.Insert("area.min_y=" + minY.ToString(-1, 3));
		lines.Insert("area.max_y=" + maxY.ToString(-1, 3));
		lines.Insert("area.relief_grid_m=" + grid.ToString(-1, 1));
		AppendVehicleLines(lines, ego);
		if (actuator)
			actuator.AppendGearboxLines(lines);

		if (probeId == PROBE_E01)
		{
			lines.Insert("rerun19.same_sequences=attributes of DC_ProbeSequences, no code change");
			lines.Insert("rerun19.compare=e01_actuation.py e01 on both play dirs (delay, steer_sign, echo, gearbox, force_sim, engine_on)");
			lines.Insert("rerun19.needs_post_go=S1 zero-steer hold flat and >=5% cross-slope at 20/40 km/h (held SetSteering(0) and a no-call variant); S2 steering steps 0->+-0.5 and 0->+1.0 at 10/30/60 km/h x50; S3 clutch/stall patterns; wheel-angle raw candidates (steering0 signal, wheel-bone yaw)");
			lines.Insert("rerun19.overlay_only=GetHandbrake, EngineGetRPMRedline, Update*Configuration (DC_ProbeSequences19.c, DD-14)");
			lines.Insert("rerun19.mode_b_fallback=CC-07 FAIL makes Mode B the production candidate (DD-4-rig)");
		}

		if (!ctx.AppendHeaderLines(lines))
			LogWarning("the area / vehicle / rerun19 lines could not be appended to the run header");
	}

	//------------------------------------------------------------------------------------------------
	//! vehicle.* lines of DC_Ego (sensitivities read, never set; gearbox facts; global assistance mode)
	protected void AppendVehicleLines(notnull array<string> lines, notnull IEntity ego)
	{
		int assistance = BaseVehicleControllerComponent.GetDrivingAssistanceMode();
		lines.Insert("vehicle.driving_assistance_mode=" + assistance.ToString());
		lines.Insert("vehicle.driving_assistance_mode_name=" + typename.EnumToString(EVehicleDrivingAssistanceMode, assistance));
		CarControllerComponent controller = CarControllerComponent.Cast(ego.FindComponent(CarControllerComponent));
		VehicleWheeledSimulation simulation;
		if (controller)
		{
			simulation = controller.GetSimulation();
			lines.Insert("vehicle.has_automatic_gearbox=" + BoolText(controller.HasAutomaticGearbox()));
			lines.Insert("vehicle.persistent_handbrake=" + BoolText(controller.GetPersistentHandBrake()));
		}

		if (!simulation)
			simulation = VehicleWheeledSimulation.Cast(ego.FindComponent(VehicleWheeledSimulation));

		if (!simulation)
			return;

		float noise = simulation.GetNoiseSteerSensitivity();
		float roughness = simulation.GetRoughnessSensitivity();
		lines.Insert("vehicle.noise_steer_sensitivity=" + noise.ToString(-1, 4));
		lines.Insert("vehicle.roughness_sensitivity=" + roughness.ToString(-1, 4));
		int gears = simulation.GearboxGearsCount();
		int forwardGears = simulation.GearboxForwardGearsCount();
		int wheels = simulation.WheelCount();
		lines.Insert("vehicle.gearbox_gears=" + gears.ToString());
		lines.Insert("vehicle.gearbox_forward_gears=" + forwardGears.ToString());
		lines.Insert("vehicle.wheel_count=" + wheels.ToString());
	}

	//------------------------------------------------------------------------------------------------
	override bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		if (probeId != PROBE_E01 && probeId != PROBE_E02A)
			return false;

		ResetProbeState();
		m_Ctx = ctx;

		IEntity ego = ctx.GetEgo();
		if (!ego)
			return RefuseStart("no ego entity (DC_Ego) in the layer - probe not started");

		World world = GetGame().GetWorld();
		if (!world)
			return RefuseStart("no game world - probe not started");

		// The runner's step length, not GetFixedTimeSlice: in the 1.8 fixedframe fallback one step is 1/30 s while
		// GetFixedTimeSlice returned 0.1 (DC_ProbeRunner STEP TIMING)
		m_fDt = GetStepDtS();
		if (m_fDt <= 0)
			return RefuseStart("step length (GetStepDtS) is " + m_fDt.ToString(-1, 6) + " s - probe not started");

		if (probeId == PROBE_E02A && m_iForceDecision == 2)
			return RefuseStart("e01 of this Play found that DC_Ego does not move, even with ForceEnableSimulation - e02a not started");

		m_aThrottleSteps = {};
		m_aBrakeSteps = {};
		m_aGearSequence = {};
		if (!ParseFloatList(m_sThrottleSteps, m_aThrottleSteps, 0, 1))
			return RefuseStart("m_sThrottleSteps '" + m_sThrottleSteps + "' is not a list of numbers in 0..1");

		if (!ParseFloatList(m_sBrakeSteps, m_aBrakeSteps, 0, 1))
			return RefuseStart("m_sBrakeSteps '" + m_sBrakeSteps + "' is not a list of numbers in 0..1");

		if (!ParseIntList(m_sGearSequence, m_aGearSequence))
			return RefuseStart("m_sGearSequence '" + m_sGearSequence + "' is not a list of gear indices >= 0");

		m_Prbs = new DC_ProbePrbs();
		if (!m_Prbs.Init(m_iPrbsOrder, m_iPrbsSeed))
			return RefuseStart(string.Format("PRBS order %1 / seed %2 not supported (orders 5..10, seed non-zero modulo 2^order)", m_iPrbsOrder, m_iPrbsSeed));

		m_Area = new DC_ProbeArea();
		m_Area.Set(m_vAreaStart, m_fAreaHeadingDeg, m_fAreaLengthM, m_fAreaHalfWidthM);
		m_fStartAlongM = 2 * m_fAreaMarginM;

		// The actuator reads the gearbox once here (DetectGearbox); its lines go into the run header extras
		m_Actuator = new DC_ProbeActuator();
		DC_ProbeActuator readyActuator;
		if (m_Actuator.Init(ego))
		{
			m_Actuator.SetSpeedGains(m_fSpeedKp, m_fSpeedKi, m_fBrakeScale);
			m_Actuator.SetShiftPolicy(m_fShiftUpRpm, m_fShiftDownRpm, m_fShiftMinS, m_fShiftFirstBelowKmh, m_fGearClutchDipS);
			readyActuator = m_Actuator;
		}

		float minY;
		float maxY;
		float relief = m_Area.MeasureRelief(world, RELIEF_GRID_M, minY, maxY);
		AppendRunHeaderExtras(probeId, ctx, ego, readyActuator, relief, minY, maxY);
		string reliefText = string.Format("area relief %1 m (surface y %2 .. %3, %4 m grid), limit %5 m", relief.ToString(-1, 2), minY.ToString(-1, 2), maxY.ToString(-1, 2), RELIEF_GRID_M, m_fMaxAreaReliefM.ToString(-1, 2));
		if (relief > m_fMaxAreaReliefM)
			return RefuseStart(reliefText + " - the area is not flat enough, probe not started (change m_vAreaStart / m_fAreaHeadingDeg)");

		ctx.Log(reliefText);
		if (!readyActuator)
			return RefuseStart(m_Actuator.GetFailure() + " - probe not started");

		ctx.Log(m_Actuator.DescribeGearbox());
		ComputeStepCounts();
		m_aPhases = {};
		if (probeId == PROBE_E01)
		{
			BuildE01Plan();
		}
		else
		{
			if (!BuildE02aPlan())
				return RefuseStart("m_sE02aGrid '" + m_sE02aGrid + "' is not '<speed>:<steer>,...;...' with speeds > 0 and steering in -1..1");

			if (m_iForceDecision == 1)
				m_Actuator.ForceSimulation();
		}

		if (m_aPhases.IsEmpty())
			return RefuseStart("the plan has no phase");

		m_Row = new DC_ProbeRow();
		array<string> stepColumns = {"physics_step", "world_time_ms", "timeslice_s", "phase", "segment", "phase_step"};
		DC_ProbeActuator.AddStepColumns(stepColumns);
		stepColumns.Insert("in_area");
		m_StepsCsv = ctx.OpenCsv(probeId + "_steps", stepColumns);
		if (!m_StepsCsv)
			return RefuseStart(probeId + "_steps.csv could not be created");

		if (probeId == PROBE_E01)
		{
			array<string> phaseColumns = {"phase_idx", "phase", "start_step", "end_step", "start_world_ms", "end_world_ms", "target_kmh", "amplitude", "force_sim", "result", "note"};
			m_PhasesCsv = ctx.OpenCsv("e01_phases", phaseColumns);
		}
		else
		{
			array<string> cellColumns = {"cell_idx", "target_kmh", "steer", "start_step", "excite_start_step", "end_step", "result", "note"};
			m_PhasesCsv = ctx.OpenCsv("e02a_cells", cellColumns);
		}

		if (!m_PhasesCsv)
			return RefuseStart("the phase / cell CSV could not be created");

		m_sActiveProbe = probeId;
		ctx.Log(string.Format("dt %1 s, settle %2 steps, PRBS %3 bits x %4 steps, %5 phases planned, force_sim %6, steer sign %7 (%8)", m_fDt.ToString(-1, 6), m_iSettleSteps, m_iPrbsPeriodBits, m_iPrbsBitSteps, m_aPhases.Count(), ForceDecisionText(), m_iSteerSignUsed, m_sSteerSignSource));
		BeginPhase();
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Actuator)
			return;

		DC_ProbeSequencePhase phase = GetPhase();
		if (!phase)
			return;

		m_iLastStep = physicsStep;
		if (phase.m_iStartStep < 0)
		{
			phase.m_iStartStep = physicsStep;
			phase.m_fStartWorldMs = GetWorldMs();
		}

		if (m_eSegment == DC_EProbeSequenceSegment.EXCITE && phase.m_iExciteStartStep < 0)
			phase.m_iExciteStartStep = physicsStep;

		SetStepCommand(phase);
		m_Actuator.ApplyStep(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbePostSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Actuator)
			return;

		DC_ProbeSequencePhase phase = GetPhase();
		if (!phase || phase.m_iStartStep < 0)
			return;

		m_iLastStep = physicsStep;
		m_Actuator.ReadPost();
		bool inArea = m_Area.Contains(m_Actuator.GetPosition(), m_fAreaMarginM);
		WriteStepRow(physicsStep, timeSlice, phase, inArea);
		TrackStep(phase, timeSlice);
		int covered = CoveredSteps(timeSlice);
		m_iSegStep += covered;
		m_iPhaseStep += covered;
		Advance(phase, inArea);
	}

	//------------------------------------------------------------------------------------------------
	//! Teleports for the reset segment run here, outside the physics step
	override void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (probeId != m_sActiveProbe || m_bProbeDone || !m_Actuator)
			return;

		if (m_eSegment != DC_EProbeSequenceSegment.RESET || m_bTeleportDone)
			return;

		DC_ProbeSequencePhase phase = GetPhase();
		if (!phase)
			return;

		vector target = m_Area.PointAt(m_fStartAlongM, phase.m_fLateralM);
		if (!m_Actuator.Teleport(target, m_fAreaHeadingDeg, m_fSpawnLiftM))
		{
			LogError(string.Format("phase %1: teleport of DC_Ego failed (vehicle gone?) - probe ends", phase.m_sName));
			phase.m_sResult = "error";
			phase.AddNote("teleport_failed");
			FinishPhase(phase);
			m_bProbeDone = true;
			return;
		}

		m_Actuator.StartModeA();
		m_bTeleportDone = true;
		m_bHasPrevHeading = false;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsProbeDone(string probeId)
	{
		if (probeId != m_sActiveProbe)
			return true;

		return m_bProbeDone;
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. Parks the car and records an interrupted phase, except on reason "delete" (world teardown: only
	//! the files are closed)
	override void OnProbeStop(string probeId, string reason)
	{
		if (probeId != m_sActiveProbe || m_sActiveProbe.IsEmpty())
			return;

		if (reason != DC_ProbeRunner.REASON_DELETE)
		{
			DC_ProbeSequencePhase phase = GetPhase();
			if (phase && !m_bProbeDone)
			{
				phase.m_sResult = "stopped";
				phase.AddNote("reason:" + reason);
				FinishPhaseRecord(phase);
			}

			if (m_Actuator)
			{
				m_Actuator.Park();
				LogInfo(m_Actuator.DescribeDrive());
				array<string> driveLines = {};
				m_Actuator.AppendDriveSummaryLines(driveLines);
				if (m_Ctx && !m_Ctx.AppendHeaderLines(driveLines))
					LogWarning("the gear policy / physics wake summary could not be appended to the run header");
			}

			if (probeId == PROBE_E01)
				LogInfo(string.Format("summary: move check %1, force_sim %2, steer_sign %3 (sum %4 over %5 steps; +1 = positive steer turns right)", ForceDecisionText(), m_sForceDecisionSource, m_iMeasuredSteerSign, m_fSteerSignSum.ToString(-1, 2), m_iSteerSignSamples));
			else
				LogInfo(string.Format("summary: %1 cells run, steer sign used %2 (%3), force_sim %4", m_iPhasesStarted, m_iSteerSignUsed, m_sSteerSignSource, ForceDecisionText()));
		}

		if (m_StepsCsv)
			m_StepsCsv.Close();

		if (m_PhasesCsv)
			m_PhasesCsv.Close();

		m_sActiveProbe = string.Empty;
		m_bProbeDone = true;
		m_Ctx = null;
		m_Actuator = null;
		m_StepsCsv = null;
		m_PhasesCsv = null;
	}

	//------------------------------------------------------------------------------------------------
	//! Clears everything of the previous probe run (the Play state m_fSteerSign*, m_iForceDecision is kept)
	protected void ResetProbeState()
	{
		m_sActiveProbe = string.Empty;
		m_Ctx = null;
		m_Area = null;
		m_Actuator = null;
		m_Prbs = null;
		m_Row = null;
		m_StepsCsv = null;
		m_PhasesCsv = null;
		m_aPhases = null;
		m_iCursor = 0;
		m_iPhasesStarted = 0;
		m_bProbeDone = false;
		m_eSegment = DC_EProbeSequenceSegment.RESET;
		m_iSegStep = 0;
		m_iPhaseStep = 0;
		m_iPrbsBitsDrawn = 0;
		m_bTeleportDone = false;
		m_bHasPrevHeading = false;
		m_iLastStep = 0;
		m_fPrbsSteer = 0;
		m_bE02aRetried = false;
		m_iSteerSignUsed = 0;
		m_sSteerSignSource = "not_used";
	}

	//------------------------------------------------------------------------------------------------
	//! Logs the refusal and returns false (OnProbeStart result)
	protected bool RefuseStart(string reason)
	{
		LogError(reason);
		m_sActiveProbe = string.Empty;
		m_bProbeDone = true;
		m_Ctx = null;
		return false;
	}

	//------------------------------------------------------------------------------------------------
	protected void ComputeStepCounts()
	{
		m_iSettleSteps = StepsFor(m_fSettleS, m_fDt);
		m_iReachSteps = StepsFor(m_fE02aReachS, m_fDt);
		m_iMoveCheckSteps = StepsFor(m_fMoveCheckS, m_fDt);
		m_iPrbsBitSteps = Math.MaxInt(1, StepsFor(m_fPrbsBitS, m_fDt));
		m_iPrbsPeriodBits = m_Prbs.GetPeriod();
		m_iReversalHalfSteps = Math.MaxInt(1, StepsFor(m_fReversalPeriodS * 0.5, m_fDt));
		m_iThrottleHoldSteps = StepsFor(m_fThrottleHoldS, m_fDt);
		m_iGearObserveSteps = StepsFor(m_fGearObserveS, m_fDt);
		m_iGearHoldSteps = Math.MaxInt(1, StepsFor(m_fGearHoldS, m_fDt));
		m_iGearDipSteps = Math.MinInt(StepsFor(m_fGearClutchDipS, m_fDt), m_iGearHoldSteps);
		m_iE02aHoldSteps = StepsFor(m_fE02aHoldS, m_fDt);
		m_iCapSteps = StepsFor(SEGMENT_CAP_S, m_fDt);
	}

	//------------------------------------------------------------------------------------------------
	//! round(seconds / dt), at least 0
	static int StepsFor(float seconds, float dt)
	{
		if (dt <= 0 || seconds <= 0)
			return 0;

		int steps = Math.Round(seconds / dt);
		return steps;
	}

	//------------------------------------------------------------------------------------------------
	//! Engine steps one dispatched step covers: round(timeSlice / m_fDt), at least 1. The runner's fixedframe
	//! fallback passes n x GetStepDtS for a call that covers n fixed steps (DC_ProbeRunner STEP TIMING); counting
	//! calls instead would stretch every phase by n in world time on such frames
	protected int CoveredSteps(float timeSlice)
	{
		int steps = StepsFor(timeSlice, m_fDt);
		if (steps < 1)
			steps = 1;

		return steps;
	}

	//------------------------------------------------------------------------------------------------
	protected void BuildE01Plan()
	{
		AddPhase(DC_EProbeSequenceKind.MOVE_CHECK, "move_check", -1, m_fMoveCheckThrottle, 0);
		AddPhase(DC_EProbeSequenceKind.STEER_PRBS, "steer_prbs", m_fPrbsSpeedKmh, m_fPrbsAmplitude, 0);
		AddPhase(DC_EProbeSequenceKind.STEER_REVERSAL, "steer_reversal", m_fPrbsSpeedKmh, m_fReversalAmplitude, 0);
		foreach (float throttle : m_aThrottleSteps)
		{
			AddPhase(DC_EProbeSequenceKind.THROTTLE, "throttle_" + throttle.ToString(-1, 2), -1, throttle, 0);
		}

		foreach (float brake : m_aBrakeSteps)
		{
			AddPhase(DC_EProbeSequenceKind.BRAKE, "brake_" + brake.ToString(-1, 2), m_fBrakeFromKmh, brake, 0);
		}

		AddPhase(DC_EProbeSequenceKind.GEAR_OBSERVE, "gear_observe", -1, m_fGearObserveThrottle, 0);
		if (!m_aGearSequence.IsEmpty())
			AddPhase(DC_EProbeSequenceKind.GEAR_SCRIPTED, "gear_scripted", -1, m_fGearScriptedThrottle, 0);
	}

	//------------------------------------------------------------------------------------------------
	//! One cell per (speed, steering level) of m_sE02aGrid, in grid order; false when the grid does not parse
	protected bool BuildE02aPlan()
	{
		string source;
		m_iSteerSignUsed = ChooseSteerSign(source);
		m_sSteerSignSource = source;
		float edgeOffset = Math.Max(m_fAreaHalfWidthM - E02A_EDGE_OFFSET_M, 0);

		array<string> groups = {};
		m_sE02aGrid.Split(";", groups, true);
		if (groups.IsEmpty())
			return false;

		foreach (string groupText : groups)
		{
			array<string> parts = {};
			groupText.Split(":", parts, false);
			if (parts.Count() != 2)
				return false;

			float speed;
			if (!ParseFloat(parts[0], speed) || speed <= 0)
				return false;

			array<float> levels = {};
			if (!ParseFloatList(parts[1], levels, -1, 1) || levels.IsEmpty())
				return false;

			foreach (float steer : levels)
			{
				// start on the side away from the turn: lateral is right positive, steerSign +1 = positive steer turns right
				float side = steer * m_iSteerSignUsed;
				float lateral = 0;
				if (side > 0)
					lateral = -edgeOffset;
				else if (side < 0)
					lateral = edgeOffset;

				AddPhase(DC_EProbeSequenceKind.E02A_CELL, string.Empty, speed, steer, lateral);
			}
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected void AddPhase(DC_EProbeSequenceKind kind, string name, float targetKmh, float amplitude, float lateralM)
	{
		m_aPhases.Insert(new DC_ProbeSequencePhase(kind, name, targetKmh, amplitude, lateralM));
	}

	//------------------------------------------------------------------------------------------------
	//! +1 / -1: measured by e01 in this Play, else m_iAssumedSteerSign (source: measured_e01 or attribute)
	protected int ChooseSteerSign(out string source)
	{
		if (m_iMeasuredSteerSign != 0)
		{
			source = "measured_e01";
			return m_iMeasuredSteerSign;
		}

		source = "attribute";
		if (m_iAssumedSteerSign >= 0)
			return 1;

		return -1;
	}

	//------------------------------------------------------------------------------------------------
	protected string ForceDecisionText()
	{
		return ForceDecisionName(m_iForceDecision);
	}

	//------------------------------------------------------------------------------------------------
	//! not_needed (0), needed (1), does_not_help (2), else unknown
	static string ForceDecisionName(int decision)
	{
		switch (decision)
		{
			case 0: return "not_needed";
			case 1: return "needed";
			case 2: return "does_not_help";
		}

		return "unknown";
	}

	//------------------------------------------------------------------------------------------------
	//! ForceEnableSimulation decision of this Play (e01 move checks or an e02a retry): -1 unknown, 0 not needed,
	//! 1 needed, 2 does not help
	int GetForceDecision()
	{
		return m_iForceDecision;
	}

	//------------------------------------------------------------------------------------------------
	//! For the image probes (e05, e06a), which drive DC_Ego but identify nothing: latches ForceEnableSimulation unless
	//! e01 of this Play found it not needed (decision 0). Probe run 2 (55c77f8) showed a driverless S105 whose engine
	//! ignored the throttle without it. owner: the probe game mode that carries DC_ProbeSequences. Returns the decision
	//! name used (run header drive.force_sim_e01).
	static string ApplyPlayForceDecision(IEntity owner, notnull DC_ProbeActuator actuator)
	{
		int decision = -1;
		if (owner)
		{
			DC_ProbeSequences sequences = DC_ProbeSequences.Cast(owner.FindComponent(DC_ProbeSequences));
			if (sequences)
				decision = sequences.GetForceDecision();
		}

		if (decision != 0)
			actuator.ForceSimulation();

		return ForceDecisionName(decision);
	}

	//------------------------------------------------------------------------------------------------
	protected DC_ProbeSequencePhase GetPhase()
	{
		if (!m_aPhases || m_iCursor < 0 || m_iCursor >= m_aPhases.Count())
			return null;

		return m_aPhases[m_iCursor];
	}

	//------------------------------------------------------------------------------------------------
	//! Starts m_aPhases[m_iCursor] with the reset segment (the teleport follows in OnProbeFrame)
	protected void BeginPhase()
	{
		DC_ProbeSequencePhase phase = GetPhase();

		// gear_observe watches automatic shifting (cmd_gear -1, no SetGear): a manual gearbox has none to watch
		while (phase && phase.m_eKind == DC_EProbeSequenceKind.GEAR_OBSERVE && m_Actuator.IsManualGearbox())
		{
			SkipPhase(phase, "manual_gearbox", "HasAutomaticGearbox() is false, there is no automatic shifting to observe; the shift policy drives every phase without an explicit gear");
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
		if (phase.m_eKind == DC_EProbeSequenceKind.E02A_CELL)
			phase.m_sName = "cell_" + phase.m_iIdx.ToString();

		if (phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK_FORCED)
			m_Actuator.ForceSimulation();

		if (phase.m_eKind == DC_EProbeSequenceKind.STEER_PRBS)
			m_Prbs.Init(m_iPrbsOrder, m_iPrbsSeed);

		phase.m_iHandbrakeResetsAtStart = m_Actuator.GetPersistentHandbrakeResets();
		phase.m_iWakesAtStart = m_Actuator.GetWakeCalls();
		phase.m_iWakeAsleepAtStart = m_Actuator.GetWakeAsleep();
		phase.m_iSleepWakesAtStart = m_Actuator.GetSleepWakes();
		phase.m_iShiftsAtStart = m_Actuator.GetPolicyShifts();
		m_eSegment = DC_EProbeSequenceSegment.RESET;
		m_iSegStep = 0;
		m_iPhaseStep = 0;
		m_iPrbsBitsDrawn = 0;
		m_bTeleportDone = false;
		m_bHasPrevHeading = false;
		m_fPrbsSteer = 0;

		string target = "-";
		if (phase.m_fTargetKmh >= 0)
			target = phase.m_fTargetKmh.ToString(-1, 1) + " km/h";

		LogInfo(string.Format("phase %1 %2 (%3/%4): target %5, amplitude %6, lateral %7 m", phase.m_iIdx, phase.m_sName, m_iCursor + 1, m_aPhases.Count(), target, phase.m_fAmplitude.ToString(-1, 3), phase.m_fLateralM.ToString(-1, 1)));
	}

	//------------------------------------------------------------------------------------------------
	//! Command of this step from the phase, its segment and the steps done in the segment. The move checks hold first
	//! gear on a manual gearbox (SetFirstGearOnly); every command with gear -1 goes through the actuator's shift policy
	protected void SetStepCommand(notnull DC_ProbeSequencePhase phase)
	{
		bool moveCheck = phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK || phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK_FORCED;
		m_Actuator.SetFirstGearOnly(moveCheck);
		switch (m_eSegment)
		{
			case DC_EProbeSequenceSegment.RESET:
			case DC_EProbeSequenceSegment.SETTLE:
			{
				m_Actuator.SetCommand(0, 0, 1, 1, -1, true);
				return;
			}

			case DC_EProbeSequenceSegment.STOP:
			{
				m_Actuator.SetCommand(0, 0, 1, 1, -1, false);
				return;
			}

			case DC_EProbeSequenceSegment.ACCEL:
			{
				m_Actuator.SetSpeedHold(phase.m_fTargetKmh, 0);
				return;
			}
		}

		// EXCITE
		switch (phase.m_eKind)
		{
			case DC_EProbeSequenceKind.MOVE_CHECK:
			case DC_EProbeSequenceKind.MOVE_CHECK_FORCED:
			case DC_EProbeSequenceKind.THROTTLE:
			case DC_EProbeSequenceKind.GEAR_OBSERVE:
			{
				m_Actuator.SetCommand(0, phase.m_fAmplitude, 0, 1, -1, false);
				return;
			}

			case DC_EProbeSequenceKind.BRAKE:
			{
				m_Actuator.SetCommand(0, 0, phase.m_fAmplitude, 1, -1, false);
				return;
			}

			case DC_EProbeSequenceKind.STEER_PRBS:
			{
				// Bit b starts at covered step b x m_iPrbsBitSteps: draw every bit whose start the segment reached, so
				// a step that covers several engine steps (CoveredSteps) neither skips a bit nor shifts the sequence
				int bitIdx = m_iSegStep / m_iPrbsBitSteps;
				while (m_iPrbsBitsDrawn <= bitIdx)
				{
					if (m_Prbs.NextBit() == 1)
						m_fPrbsSteer = phase.m_fAmplitude;
					else
						m_fPrbsSteer = -phase.m_fAmplitude;

					m_iPrbsBitsDrawn++;
				}

				m_Actuator.SetSpeedHold(phase.m_fTargetKmh, m_fPrbsSteer);
				return;
			}

			case DC_EProbeSequenceKind.STEER_REVERSAL:
			{
				int half = m_iSegStep / m_iReversalHalfSteps;
				float steer = 0;
				if (half <= m_iReversalCount)
				{
					if (half % 2 == 0)
						steer = phase.m_fAmplitude;
					else
						steer = -phase.m_fAmplitude;
				}

				m_Actuator.SetSpeedHold(phase.m_fTargetKmh, steer);
				return;
			}

			case DC_EProbeSequenceKind.GEAR_SCRIPTED:
			{
				int gearIdx = Math.MinInt(m_iSegStep / m_iGearHoldSteps, m_aGearSequence.Count() - 1);
				float clutch = 1;
				if (m_iGearDipSteps > 0 && m_iSegStep % m_iGearHoldSteps < m_iGearDipSteps)
					clutch = 0;	// dip at the start of every gear hold (the first one follows the gearbox's own gear)

				m_Actuator.SetCommand(0, phase.m_fAmplitude, 0, clutch, m_aGearSequence[gearIdx], false);
				return;
			}

			case DC_EProbeSequenceKind.E02A_CELL:
			{
				m_Actuator.SetSpeedHold(phase.m_fTargetKmh, phase.m_fAmplitude);
				return;
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void WriteStepRow(int physicsStep, float timeSlice, notnull DC_ProbeSequencePhase phase, bool inArea)
	{
		if (!m_StepsCsv || !m_Row)
			return;

		m_Row.Clear();
		m_Row.AddInt(physicsStep);
		m_Row.AddFloat(GetWorldMs(), 3);
		m_Row.AddFloat(timeSlice, 6);
		m_Row.AddString(phase.m_sName);
		m_Row.AddString(SegmentName(m_eSegment));
		m_Row.AddInt(m_iPhaseStep);
		m_Actuator.AddStepValues(m_Row);
		m_Row.AddBool(inArea);
		m_StepsCsv.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	//! Online bookkeeping after each step: max speed, heading change, steer-sign sum, stall watch. timeSlice: the world
	//! time the step covers (n x dt on a merged fallback step), for the heading rate
	protected void TrackStep(notnull DC_ProbeSequencePhase phase, float timeSlice)
	{
		float heading = m_Actuator.GetHeadingDeg();
		float delta = 0;
		if (m_bHasPrevHeading)
			delta = WrapDeg(heading - m_fPrevHeadingDeg);

		m_fPrevHeadingDeg = heading;
		m_bHasPrevHeading = true;

		bool driving = m_eSegment == DC_EProbeSequenceSegment.ACCEL || m_eSegment == DC_EProbeSequenceSegment.EXCITE;
		if (!driving)
			return;

		float speed = m_Actuator.GetSpeedKmh();
		if (speed > phase.m_fMaxSpeedKmh)
			phase.m_fMaxSpeedKmh = speed;

		if (!m_Actuator.IsEngineOn())
			phase.m_iEngineOffSteps++;

		if (m_eSegment != DC_EProbeSequenceSegment.EXCITE)
			return;

		phase.m_fTurnDeg += delta;
		bool steering = phase.m_eKind == DC_EProbeSequenceKind.STEER_PRBS || phase.m_eKind == DC_EProbeSequenceKind.STEER_REVERSAL;
		if (steering && timeSlice > 0)
		{
			m_fSteerSignSum += m_Actuator.GetCommandSteer() * delta / timeSlice;
			m_iSteerSignSamples++;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Segment transitions after the step (the state read by ReadPost decides)
	protected void Advance(notnull DC_ProbeSequencePhase phase, bool inArea)
	{
		switch (m_eSegment)
		{
			case DC_EProbeSequenceSegment.RESET:
			{
				if (m_bTeleportDone)
					EnterSegment(DC_EProbeSequenceSegment.SETTLE);

				return;
			}

			case DC_EProbeSequenceSegment.SETTLE:
			{
				if (m_iSegStep < m_iSettleSteps)
					return;

				if (!m_Actuator.IsEngineOn())
				{
					// a stalled or never-started engine gets one more Mode A start before the phase drives
					m_Actuator.StartModeA();
					phase.AddNote("engine_restart_after_settle");
				}

				EnterSegment(FirstDrivingSegment(phase));
				return;
			}

			case DC_EProbeSequenceSegment.STOP:
			{
				if (m_Actuator.GetSpeedKmh() < m_fSpeedTolKmh)
				{
					FinishPhase(phase);
					return;
				}

				if (m_iSegStep >= m_iCapSteps)
				{
					phase.AddNote("stop_cap");
					FinishPhase(phase);
				}

				return;
			}
		}

		// ACCEL and EXCITE. The move checks run their full time whatever the engine reports (a dormant car may
		// report it off), so that no_motion and the ForceEnableSimulation retry stay reachable
		bool moveCheck = phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK || phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK_FORCED;
		if (!moveCheck && !m_Actuator.IsEngineOn())
		{
			EndDriving(phase, "engine_off");
			return;
		}

		if (!inArea)
		{
			EndDriving(phase, "aborted_area");
			return;
		}

		if (m_eSegment == DC_EProbeSequenceSegment.ACCEL)
		{
			if (m_Actuator.GetSpeedKmh() >= phase.m_fTargetKmh - m_fSpeedTolKmh)
			{
				EnterSegment(DC_EProbeSequenceSegment.EXCITE);
				return;
			}

			if (m_iSegStep >= m_iReachSteps)
				EndDriving(phase, "no_speed");

			return;
		}

		AdvanceExcite(phase);
	}

	//------------------------------------------------------------------------------------------------
	//! End of the excite segment per phase kind
	protected void AdvanceExcite(notnull DC_ProbeSequencePhase phase)
	{
		switch (phase.m_eKind)
		{
			case DC_EProbeSequenceKind.MOVE_CHECK:
			case DC_EProbeSequenceKind.MOVE_CHECK_FORCED:
			{
				if (m_iSegStep < m_iMoveCheckSteps)
					return;

				if (phase.m_fMaxSpeedKmh >= m_fMoveCheckMinKmh)
					phase.m_sResult = "ok";
				else
					phase.m_sResult = "no_motion";

				FinishPhase(phase);
				return;
			}

			case DC_EProbeSequenceKind.BRAKE:
			{
				if (m_Actuator.GetSpeedKmh() < m_fSpeedTolKmh)
				{
					FinishPhase(phase);
					return;
				}

				if (m_iSegStep >= m_iCapSteps)
					EndDriving(phase, "segment_cap");

				return;
			}

			case DC_EProbeSequenceKind.STEER_PRBS:
			{
				if (m_iSegStep >= m_iPrbsPeriodBits * m_iPrbsBitSteps)
					EnterSegment(DC_EProbeSequenceSegment.STOP);

				return;
			}

			case DC_EProbeSequenceKind.STEER_REVERSAL:
			{
				if (m_iSegStep >= (m_iReversalCount + 2) * m_iReversalHalfSteps)
					EnterSegment(DC_EProbeSequenceSegment.STOP);

				return;
			}

			case DC_EProbeSequenceKind.THROTTLE:
			{
				if (m_iSegStep >= m_iThrottleHoldSteps)
					EnterSegment(DC_EProbeSequenceSegment.STOP);

				return;
			}

			case DC_EProbeSequenceKind.GEAR_OBSERVE:
			{
				if (m_iSegStep >= m_iGearObserveSteps)
					EnterSegment(DC_EProbeSequenceSegment.STOP);

				return;
			}

			case DC_EProbeSequenceKind.GEAR_SCRIPTED:
			{
				if (m_iSegStep >= m_aGearSequence.Count() * m_iGearHoldSteps)
					EnterSegment(DC_EProbeSequenceSegment.STOP);

				return;
			}

			case DC_EProbeSequenceKind.E02A_CELL:
			{
				if (m_iSegStep >= m_iE02aHoldSteps || Math.AbsFloat(phase.m_fTurnDeg) >= m_fE02aMaxTurnDeg)
					EnterSegment(DC_EProbeSequenceSegment.STOP);

				return;
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	protected DC_EProbeSequenceSegment FirstDrivingSegment(notnull DC_ProbeSequencePhase phase)
	{
		if (phase.m_fTargetKmh >= 0)
			return DC_EProbeSequenceSegment.ACCEL;

		return DC_EProbeSequenceSegment.EXCITE;
	}

	//------------------------------------------------------------------------------------------------
	protected void EnterSegment(DC_EProbeSequenceSegment segment)
	{
		m_eSegment = segment;
		m_iSegStep = 0;
		m_iPrbsBitsDrawn = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! A driving segment ended early: keep the result, brake to a stop
	protected void EndDriving(notnull DC_ProbeSequencePhase phase, string result)
	{
		phase.m_sResult = result;
		EnterSegment(DC_EProbeSequenceSegment.STOP);
	}

	//------------------------------------------------------------------------------------------------
	//! Writes the phase row, applies the per-Play decisions and starts the next phase (or ends the probe)
	protected void FinishPhase(notnull DC_ProbeSequencePhase phase)
	{
		FinishPhaseRecord(phase);
		LogInfo(string.Format("phase %1 %2: %3 (steps %4, max %5 km/h, turn %6 deg)", phase.m_iIdx, phase.m_sName, phase.m_sResult, m_iPhaseStep, phase.m_fMaxSpeedKmh.ToString(-1, 2), phase.m_fTurnDeg.ToString(-1, 1)));

		if (phase.m_sResult == "error")
		{
			m_bProbeDone = true;
			return;
		}

		if (m_sActiveProbe == PROBE_E01)
			ApplyE01Decisions(phase);
		else
			ApplyE02aDecisions(phase);

		if (m_bProbeDone)
			return;

		m_iCursor++;
		if (m_iCursor >= m_aPhases.Count())
		{
			m_bProbeDone = true;
			return;
		}

		BeginPhase();
	}

	//------------------------------------------------------------------------------------------------
	//! End step and time, result default ok, note tokens; one row in e01_phases.csv / e02a_cells.csv
	protected void FinishPhaseRecord(notnull DC_ProbeSequencePhase phase)
	{
		phase.m_iEndStep = m_iLastStep;
		phase.m_fEndWorldMs = GetWorldMs();
		phase.m_bForceSim = false;
		if (m_Actuator)
			phase.m_bForceSim = m_Actuator.IsSimulationForced();
		if (phase.m_sResult.IsEmpty())
			phase.m_sResult = "ok";

		float maxSpeed = phase.m_fMaxSpeedKmh;
		float turn = phase.m_fTurnDeg;
		phase.AddNote("max_kmh:" + maxSpeed.ToString(-1, 3));
		phase.AddNote("turn_deg:" + turn.ToString(-1, 2));
		phase.AddNote("engine_off_steps:" + phase.m_iEngineOffSteps.ToString());
		if (m_Actuator)
		{
			int resets = m_Actuator.GetPersistentHandbrakeResets() - phase.m_iHandbrakeResetsAtStart;
			int wakes = m_Actuator.GetWakeCalls() - phase.m_iWakesAtStart;
			int asleep = m_Actuator.GetWakeAsleep() - phase.m_iWakeAsleepAtStart;
			int shifts = m_Actuator.GetPolicyShifts() - phase.m_iShiftsAtStart;
			int sleepWakes = m_Actuator.GetSleepWakes() - phase.m_iSleepWakesAtStart;
			phase.AddNote("phb_resets:" + resets.ToString());
			phase.AddNote("wakes:" + wakes.ToString());
			phase.AddNote("asleep_at_wake:" + asleep.ToString());
			phase.AddNote("shifts:" + shifts.ToString());
			phase.AddNote("sleep_wakes:" + sleepWakes.ToString());
		}

		WritePhaseRow(phase);
	}

	//------------------------------------------------------------------------------------------------
	//! Writes a skipped row (note token) for a phase that does not apply and logs why; the caller moves the cursor on
	protected void SkipPhase(notnull DC_ProbeSequencePhase phase, string noteToken, string reason)
	{
		phase.m_iIdx = m_iPhasesStarted;
		m_iPhasesStarted++;
		phase.m_sResult = "skipped";
		phase.AddNote(noteToken);
		WritePhaseRow(phase);
		LogInfo(string.Format("phase %1 %2: skipped, %3 (%4)", phase.m_iIdx, phase.m_sName, noteToken, reason));
	}

	//------------------------------------------------------------------------------------------------
	protected void WritePhaseRow(notnull DC_ProbeSequencePhase phase)
	{
		if (!m_PhasesCsv)
			return;

		DC_ProbeRow row = new DC_ProbeRow();
		if (m_sActiveProbe == PROBE_E01)
		{
			row.AddInt(phase.m_iIdx);
			row.AddString(phase.m_sName);
			AddStepOrEmpty(row, phase.m_iStartStep);
			AddStepOrEmpty(row, phase.m_iEndStep);
			AddMsOrEmpty(row, phase.m_fStartWorldMs);
			AddMsOrEmpty(row, phase.m_fEndWorldMs);
			if (phase.m_fTargetKmh >= 0)
				row.AddFloat(phase.m_fTargetKmh, 3);
			else
				row.AddEmpty();

			row.AddFloat(phase.m_fAmplitude, 4);
			row.AddBool(phase.m_bForceSim);
			row.AddString(phase.m_sResult);
			row.AddString(phase.m_sNote);
		}
		else
		{
			row.AddInt(phase.m_iIdx);
			row.AddFloat(phase.m_fTargetKmh, 3);
			row.AddFloat(phase.m_fAmplitude, 4);
			AddStepOrEmpty(row, phase.m_iStartStep);
			AddStepOrEmpty(row, phase.m_iExciteStartStep);
			AddStepOrEmpty(row, phase.m_iEndStep);
			row.AddString(phase.m_sResult);
			phase.AddNote("lateral_start_m:" + phase.m_fLateralM.ToString(-1, 2));
			row.AddString(phase.m_sNote);
		}

		m_PhasesCsv.WriteRow(row);
	}

	//------------------------------------------------------------------------------------------------
	//! move_check -> force decision (and move_check_forced next on no_motion); steering phases -> steer sign
	protected void ApplyE01Decisions(notnull DC_ProbeSequencePhase phase)
	{
		if (phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK)
		{
			if (phase.m_sResult == "no_motion")
			{
				DC_ProbeSequencePhase forced = new DC_ProbeSequencePhase(DC_EProbeSequenceKind.MOVE_CHECK_FORCED, "move_check_forced", -1, m_fMoveCheckThrottle, 0);
				m_aPhases.InsertAt(forced, m_iCursor + 1);
				LogWarning("no motion without ForceEnableSimulation - repeating the move check with it");
			}
			else if (phase.m_fMaxSpeedKmh >= m_fMoveCheckMinKmh)
			{
				m_iForceDecision = 0;
				m_sForceDecisionSource = "e01_move_check";
			}

			return;
		}

		if (phase.m_eKind == DC_EProbeSequenceKind.MOVE_CHECK_FORCED)
		{
			if (phase.m_fMaxSpeedKmh >= m_fMoveCheckMinKmh)
			{
				m_iForceDecision = 1;
				m_sForceDecisionSource = "e01_move_check_forced";
				return;
			}

			m_iForceDecision = 2;
			m_sForceDecisionSource = "e01_move_check_forced";
			LogError("DC_Ego does not move even with ForceEnableSimulation - the remaining e01 phases are skipped");
			SkipRemainingPhases();
			return;
		}

		bool steering = phase.m_eKind == DC_EProbeSequenceKind.STEER_PRBS || phase.m_eKind == DC_EProbeSequenceKind.STEER_REVERSAL;
		if (!steering)
			return;

		if (m_fSteerSignSum > 0)
			m_iMeasuredSteerSign = 1;
		else if (m_fSteerSignSum < 0)
			m_iMeasuredSteerSign = -1;
	}

	//------------------------------------------------------------------------------------------------
	//! A cell that never moved (no_speed or engine_off below the move-check speed) forces the simulation and repeats
	//! once
	protected void ApplyE02aDecisions(notnull DC_ProbeSequencePhase phase)
	{
		bool stuck = phase.m_sResult == "no_speed" || phase.m_sResult == "engine_off";
		if (!stuck || m_bE02aRetried || m_Actuator.IsSimulationForced())
			return;

		if (phase.m_fMaxSpeedKmh >= m_fMoveCheckMinKmh)
			return;

		m_bE02aRetried = true;
		m_Actuator.ForceSimulation();
		m_iForceDecision = 1;
		m_sForceDecisionSource = "e02a_retry";
		DC_ProbeSequencePhase retry = new DC_ProbeSequencePhase(phase.m_eKind, string.Empty, phase.m_fTargetKmh, phase.m_fAmplitude, phase.m_fLateralM);
		retry.AddNote("retry_of:" + phase.m_iIdx.ToString());
		m_aPhases.InsertAt(retry, m_iCursor + 1);
		LogWarning(string.Format("cell %1 did not move (max %2 km/h) - ForceEnableSimulation latched, the cell is repeated", phase.m_iIdx, phase.m_fMaxSpeedKmh.ToString(-1, 2)));
	}

	//------------------------------------------------------------------------------------------------
	//! Writes a skipped row for every phase after the cursor and ends the probe
	protected void SkipRemainingPhases()
	{
		for (int i = m_iCursor + 1, count = m_aPhases.Count(); i < count; i++)
		{
			DC_ProbeSequencePhase skipped = m_aPhases[i];
			skipped.m_iIdx = m_iPhasesStarted;
			m_iPhasesStarted++;
			skipped.m_sResult = "skipped";
			skipped.AddNote("no_motion_with_force_sim");
			WritePhaseRow(skipped);
		}

		m_bProbeDone = true;
	}

	//------------------------------------------------------------------------------------------------
	//! Parses '<f>,<f>,...' (spaces allowed, empty list allowed); every value must lie in min..max
	protected bool ParseFloatList(string text, notnull array<float> outValues, float min, float max)
	{
		outValues.Clear();
		array<string> tokens = {};
		text.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			float value;
			if (!ParseFloat(token, value))
				return false;

			if (value < min || value > max)
				return false;

			outValues.Insert(value);
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Parses '<i>,<i>,...' of gear indices >= 0 (empty list allowed)
	protected bool ParseIntList(string text, notnull array<int> outValues)
	{
		outValues.Clear();
		array<string> tokens = {};
		text.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			string trimmed = token.Trim();
			if (trimmed.IsEmpty())
				continue;

			int parsed;
			int value = trimmed.ToInt(0, 0, parsed);
			if (parsed != trimmed.Length() || value < 0)
				return false;

			outValues.Insert(value);
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! true when the whole trimmed token is one number
	protected bool ParseFloat(string token, out float value)
	{
		string trimmed = token.Trim();
		if (trimmed.IsEmpty())
			return false;

		int parsed;
		value = trimmed.ToFloat(0, 0, parsed);
		return parsed == trimmed.Length();
	}

	//------------------------------------------------------------------------------------------------
	protected static void AddStepOrEmpty(notnull DC_ProbeRow row, int step)
	{
		if (step >= 0)
			row.AddInt(step);
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
	protected static string SegmentName(DC_EProbeSequenceSegment segment)
	{
		switch (segment)
		{
			case DC_EProbeSequenceSegment.RESET: return "reset";
			case DC_EProbeSequenceSegment.SETTLE: return "settle";
			case DC_EProbeSequenceSegment.ACCEL: return "accel";
			case DC_EProbeSequenceSegment.EXCITE: return "excite";
			case DC_EProbeSequenceSegment.STOP: return "stop";
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
	protected static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
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
