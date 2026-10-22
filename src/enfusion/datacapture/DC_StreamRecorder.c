// ============================================================================
// DC_StreamRecorder - ctl and telemetry on physics-step counters, the events stream and the driver interface
// (DD-4-min)
// ============================================================================
//
// A DC_SessionModule (dispatch order 20) on the capture game mode. It writes the dc-2.0 streams ctl (every physics
// step), telemetry (every k-th step, k = clocks.telemetry_every_k, TBD-1) and events (DC_Session.LogEvent), each
// through a DC_ChunkWriter that DC_Session owns. It also orchestrates the driver of every step.
//
// PER PHYSICS STEP P (docs/capture_contract/dc-2.0.md section 6, the R-01 decision boundary; no CallLater clock)
//   OnStepPre (EOnSimulate): a decision taken at P - 1 becomes the held command (hl_seq + 1, so hl_seq changes only
//     at P mod N == 1); inside an episode with a held command the driver's ApplyStep runs (llc_v1 calls the setters
//     through the DC_Actuator); then input_pre_* is read back through the actuator.
//   OnStepPost (EOnPostSimulate): input_post_* read back; inside an episode the driver's PostStep re-evaluates the
//     expert on the state after P (exp_*_now, lane errors, route progress, termination); the ctl row of P is
//     buffered, then the telemetry row when P mod k == 0.
//   OnStepDecide (after every module's OnStepPost; DC_EgoSession has begun or ended the episode by then): at a
//     decision step (P mod N == 0) inside an episode the driver's Decide gives the command applied over
//     (P, P + N]. A step without a held command writes mode I, driver none and empty hl_*; inside an episode the
//     episode's mode and driver are written (as the DD-5-min golden fixture).
//
// DRIVER INTERFACE (shared with DD-7-min, which implements llc_v1 and the expert; keep it stable)
//   A DC_DriverInterface component on the same game mode entity; without one the null driver applies: no episode,
//   no HL command, ctl hl_* / llc_* / exp_* empty, the ego stays parked. Exact signatures:
//
//     class DC_DriverInterface : ScriptComponent
//       bool GetSpawnPose(out vector position, out float headingDeg);       // false = DC_EgoSession's attribute pose
//       bool BeginDriver(notnull DC_DriverContext ctx);                     // readiness; false = no session
//       bool BeginEpisode(int episodeId, int firstPhysicsStep, notnull DC_EpisodeInfo info); // false = no episode
//       void ApplyStep(int physicsStep, float timeSlice, notnull DC_HLCommand held, notnull DC_LLCOutput llc);
//       void PostStep(int physicsStep, float timeSlice, notnull DC_DriverPost post);
//       bool Decide(int physicsStep, notnull DC_HLCommand command);         // false = keep the held command
//       void EndEpisode(int episodeId, int lastPhysicsStep, string terminationReason);
//       void EndDriver(string endReason);
//
//   Call order: GetSpawnPose (before the ego is spawned), BeginDriver once (the ego is spawned and its actuator
//   exists; the session directory does not exist yet), then per episode BeginEpisode at a decision step P_a (in
//   OnStepPre of P_a, so the rows of P_a carry the episode), ApplyStep in every OnStepPre with a held command,
//   PostStep in every OnStepPost of the episode, Decide in OnStepDecide of every decision step of the episode,
//   EndEpisode at the decision step P_b that ends it (after the rows of P_b), EndDriver once at the seal.
//   Units and signs: DC_HLCommand is the contract action_cmd (kappa right-positive [1/m], accel [m/s2], dir -1, 0 or
//   +1); DC_LLCOutput holds exactly what was passed to the setters (engine-native steering sign). A termination
//   reason in DC_DriverPost (an episodes.termination_reason value) ends the episode at the next decision step.
//   T0-MIN is forward-only: a command with dir -1 fails the session while m_bForwardOnly is set.
//
// DC_ACTUATOR (r3 P1 scope note): llc_v1 and the readbacks never call the simulation setters directly; they go
//   through one DC_Actuator, Mode A now (DC_ActuatorModeA: driverless, setters every step, as DD-4-probe's
//   DC_ProbeActuator), Mode B later (DD-4-rig DC_ModeB.c), so a Mode B promotion touches neither llc_v1 nor the
//   streams.
//
// Failure handling: a failed write reports DC_Session.FailWrite (script_error:write_fail); a missing ego simulation
// in a step fails the session with ego_lost; an invalid driver output fails it with script_error:other.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! One HL command (contract action_cmd): curvature target right-positive [1/m], acceleration target [m/s2],
//! direction -1 reverse, 0 stop, +1 forward
class DC_HLCommand
{
	float m_fKappa;
	float m_fAccel;
	int m_iDir = 1;

	//------------------------------------------------------------------------------------------------
	void Reset()
	{
		m_fKappa = 0;
		m_fAccel = 0;
		m_iDir = 1;
	}

	//------------------------------------------------------------------------------------------------
	void CopyFrom(notnull DC_HLCommand other)
	{
		m_fKappa = other.m_fKappa;
		m_fAccel = other.m_fAccel;
		m_iDir = other.m_iDir;
	}
}

//------------------------------------------------------------------------------------------------
//! What the driver passed to the setters in one physics step (ctl llc_*). m_bActive false: llc_v1 did not drive
//! and every llc_* field stays empty
class DC_LLCOutput
{
	bool m_bActive;
	float m_fSteerRaw;		//!< before the rate limiter and lag, engine-native sign
	float m_fSteerLim;		//!< as passed to SetSteering, engine-native sign
	float m_fThrottle;
	float m_fBrake;
	bool m_bHandbrake;
	int m_iGear = -1;		//!< passed to SetGear; -1 = no SetGear call (field empty)
	float m_fClutch = -1;	//!< passed to SetClutch by a scripted gearbox; < 0 = field empty

	//------------------------------------------------------------------------------------------------
	void Reset()
	{
		m_bActive = false;
		m_fSteerRaw = 0;
		m_fSteerLim = 0;
		m_fThrottle = 0;
		m_fBrake = 0;
		m_bHandbrake = false;
		m_iGear = -1;
		m_fClutch = -1;
	}
}

//------------------------------------------------------------------------------------------------
//! The driver's view after a physics step (filled by PostStep): the expert re-evaluated on the true state
//! (ctl exp_*_now), the lane-relative errors and route progress (telemetry), and a termination request
class DC_DriverPost
{
	bool m_bHasExpert;
	float m_fExpKappa;			//!< [1/m], right-positive
	float m_fExpAccel;			//!< [m/s2]
	int m_iExpDir = 1;
	bool m_bHasRoute;
	float m_fLaneLatErrM;		//!< right-positive
	float m_fLaneHeadErrRad;	//!< right-positive, wrapped to (-pi, pi]
	float m_fRouteProgressM;
	string m_sTermination;		//!< empty = continue; else an episodes.termination_reason value

	//------------------------------------------------------------------------------------------------
	void Reset()
	{
		m_bHasExpert = false;
		m_fExpKappa = 0;
		m_fExpAccel = 0;
		m_iExpDir = 1;
		m_bHasRoute = false;
		m_fLaneLatErrM = 0;
		m_fLaneHeadErrRad = 0;
		m_fRouteProgressM = 0;
		m_sTermination = string.Empty;
	}
}

//------------------------------------------------------------------------------------------------
//! Episode description given by the driver in BeginEpisode (the episodes row; the ctl mode and driver)
class DC_EpisodeInfo
{
	string m_sMode = "E";				//!< E, N, R, X, D, H or A
	string m_sDriver = "expert";		//!< expert, policy, human, ai or none
	string m_sVehicleCardId = "none";
	string m_sRouteId;
	string m_sRouteSha256;
	string m_sExpertSha;
	string m_sLlcSha;
	float m_fStyleSpeedFactor = 1;
	float m_fStyleLaneOffsetM = 0;
	float m_fStyleAlatMaxMps2 = 2.75;
	int m_iSeed;						//!< preset by the core to trial seed + episode id
}

//------------------------------------------------------------------------------------------------
//! The simulation inputs read back (GetSteering .. IsHandbrakeOn): inputs, never wheel angles or forces
class DC_InputReadback
{
	float m_fSteer;
	float m_fThrottle;
	float m_fBrake;
	float m_fClutch;
	int m_iGear;
	bool m_bHandbrake;
}

//------------------------------------------------------------------------------------------------
//! The one actuation path of the ego (r3 P1): Mode A now, Mode B later. llc_v1 calls Apply; the core calls Park,
//! ReadInputs and GetSimulation (telemetry)
class DC_Actuator
{
	//------------------------------------------------------------------------------------------------
	//! session.json control.actuation_mode: "A" or "B"
	string GetMode()
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! false (GetFailure) when the vehicle cannot be driven this way
	bool Init(notnull IEntity vehicle)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	VehicleWheeledSimulation GetSimulation()
	{
		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! Makes the vehicle drivable (engine on, persistent handbrake off)
	void Start()
	{
	}

	//------------------------------------------------------------------------------------------------
	//! One step's inputs: steering in [-1, 1] engine-native, throttle, brake and clutch in [0, 1] (clutch 1 =
	//! engaged), handbrake flag, gear >= 0 for a SetGear call (-1 = none)
	void Apply(float steer, float throttle, float brake, bool handbrake, float clutch, int gear)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Brake 1, handbrake, throttle 0, steering 0, applied at once
	void Park()
	{
	}

	//------------------------------------------------------------------------------------------------
	void ReadInputs(notnull DC_InputReadback readback)
	{
	}
}

//------------------------------------------------------------------------------------------------
//! Mode A (r2 3.2, as DD-4-probe DC_ProbeActuator): a driverless vehicle driven by the VehicleWheeledSimulation
//! setters every step. Start = SetPersistentHandBrake(false) + ForceStartEngine (vanilla VehicleCinematicTrack.c:127,
//! :140); the noise-steer and roughness sensitivities are left at their defaults (r2 X6). A controller that sets the
//! persistent handbrake again while the command releases it is overridden and counted (1.9 robustness).
class DC_ActuatorModeA : DC_Actuator
{
	protected IEntity m_Vehicle;
	protected CarControllerComponent m_Controller;
	protected VehicleWheeledSimulation m_Simulation;
	protected string m_sFailure = "not initialised";
	protected bool m_bForceSimulation;
	protected int m_iPersistentHandbrakeResets;

	//------------------------------------------------------------------------------------------------
	override string GetMode()
	{
		return "A";
	}

	//------------------------------------------------------------------------------------------------
	//! CarControllerComponent and its VehicleWheeledSimulation (GetSimulation; GetWheeledSimulation is obsolete)
	override bool Init(notnull IEntity vehicle)
	{
		m_Vehicle = vehicle;
		m_Controller = CarControllerComponent.Cast(vehicle.FindComponent(CarControllerComponent));
		if (!m_Controller)
		{
			m_sFailure = "no CarControllerComponent on the ego";
			return false;
		}

		m_Simulation = m_Controller.GetSimulation();
		if (!m_Simulation)
			m_Simulation = VehicleWheeledSimulation.Cast(vehicle.FindComponent(VehicleWheeledSimulation));

		if (!m_Simulation)
		{
			m_sFailure = "no VehicleWheeledSimulation on the ego";
			return false;
		}

		m_sFailure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	override VehicleWheeledSimulation GetSimulation()
	{
		return m_Simulation;
	}

	//------------------------------------------------------------------------------------------------
	//! ForceEnableSimulation on every Apply from now on (E-01 decides whether the S105 needs it; the engine documents
	//! no duration for it)
	void SetForceSimulation(bool force)
	{
		m_bForceSimulation = force;
	}

	//------------------------------------------------------------------------------------------------
	int GetPersistentHandbrakeResets()
	{
		return m_iPersistentHandbrakeResets;
	}

	//------------------------------------------------------------------------------------------------
	override void Start()
	{
		if (!m_Controller)
			return;

		m_Controller.SetPersistentHandBrake(false);
		if (!m_Controller.IsEngineOn())
			m_Controller.ForceStartEngine();
	}

	//------------------------------------------------------------------------------------------------
	override void Apply(float steer, float throttle, float brake, bool handbrake, float clutch, int gear)
	{
		if (!m_Simulation)
			return;

		if (m_bForceSimulation)
			m_Simulation.ForceEnableSimulation();

		if (m_Controller && !handbrake && m_Controller.GetPersistentHandBrake())
		{
			m_Controller.SetPersistentHandBrake(false);
			m_iPersistentHandbrakeResets++;
		}

		m_Simulation.SetSteering(Math.Clamp(steer, -1, 1));
		m_Simulation.SetThrottle(Math.Clamp(throttle, 0, 1));
		m_Simulation.SetBreak(Math.Clamp(brake, 0, 1), handbrake);
		m_Simulation.SetClutch(Math.Clamp(clutch, 0, 1));
		if (gear >= 0)
			m_Simulation.SetGear(gear);
	}

	//------------------------------------------------------------------------------------------------
	override void Park()
	{
		Apply(0, 0, 1, true, 1, -1);
	}

	//------------------------------------------------------------------------------------------------
	override void ReadInputs(notnull DC_InputReadback readback)
	{
		if (!m_Simulation)
			return;

		readback.m_fSteer = m_Simulation.GetSteering();
		readback.m_fThrottle = m_Simulation.GetThrottle();
		readback.m_fBrake = m_Simulation.GetBrake();
		readback.m_fClutch = m_Simulation.GetClutch();
		readback.m_iGear = m_Simulation.GetGear();
		readback.m_bHandbrake = m_Simulation.IsHandbrakeOn();
	}
}

//------------------------------------------------------------------------------------------------
//! What a driver gets in BeginDriver: the ego and its actuator, the clocks, the trial; and where it reports its
//! session.json control stamps. Built by DC_StreamRecorder; kept for the whole session
class DC_DriverContext
{
	protected DC_Session m_Session;
	protected IEntity m_Ego;
	protected ref DC_Actuator m_Actuator;
	protected string m_sLlcSha;
	protected string m_sExpertSha;
	protected string m_sVehicleCardId;
	protected string m_sVehicleCardSha;
	protected int m_iSteerSign;			//!< 0 = not set (DC_StreamRecorder's attribute applies)
	protected int m_iYawAxisSign;

	//------------------------------------------------------------------------------------------------
	void DC_DriverContext(DC_Session session, IEntity ego, DC_Actuator actuator)
	{
		m_Session = session;
		m_Ego = ego;
		m_Actuator = actuator;
	}

	//------------------------------------------------------------------------------------------------
	IEntity GetEgo()
	{
		return m_Ego;
	}

	//------------------------------------------------------------------------------------------------
	DC_Actuator GetActuator()
	{
		return m_Actuator;
	}

	//------------------------------------------------------------------------------------------------
	//! BaseWorld.GetFixedTimeSlice() [s]
	float GetFixedTimeSliceS()
	{
		if (!m_Session)
			return 0;

		return m_Session.GetFixedTimeSliceS();
	}

	//------------------------------------------------------------------------------------------------
	//! N: decision steps are the physics steps with step mod N == 0 (TBD-1)
	int GetStepsPerDecision()
	{
		if (!m_Session)
			return 1;

		return m_Session.GetStepsPerDecision();
	}

	//------------------------------------------------------------------------------------------------
	int GetTelemetryEveryK()
	{
		if (!m_Session)
			return 1;

		return m_Session.GetTelemetryEveryK();
	}

	//------------------------------------------------------------------------------------------------
	//! world_time_ms now (WorldTimestamp since the session open; 0 before the open)
	int GetWorldMs()
	{
		if (!m_Session)
			return 0;

		return m_Session.GetWorldMs();
	}

	//------------------------------------------------------------------------------------------------
	string GetTrialId()
	{
		if (!m_Session)
			return string.Empty;

		return m_Session.GetTrialId();
	}

	//------------------------------------------------------------------------------------------------
	//! The trial conf's "route" (empty without one)
	string GetTrialRoute()
	{
		if (!m_Session)
			return string.Empty;

		return m_Session.GetTrialRoute();
	}

	//------------------------------------------------------------------------------------------------
	int GetTrialSeed()
	{
		if (!m_Session)
			return 0;

		return m_Session.GetTrialSeed();
	}

	//------------------------------------------------------------------------------------------------
	string GetMapId()
	{
		if (!m_Session)
			return string.Empty;

		return m_Session.GetMapId();
	}

	//------------------------------------------------------------------------------------------------
	string GetPinId()
	{
		if (!m_Session)
			return string.Empty;

		return m_Session.GetPinId();
	}

	//------------------------------------------------------------------------------------------------
	//! pin.build_key of the launch record; empty with a dd-launch/1 record
	string GetBuildKey()
	{
		if (!m_Session)
			return string.Empty;

		return m_Session.GetBuildKey();
	}

	//------------------------------------------------------------------------------------------------
	//! session.json control.llc_sha, expert_sha, vehicle_card_id, vehicle_card_sha (empty = null)
	void SetStamps(string llcSha, string expertSha, string vehicleCardId, string vehicleCardSha)
	{
		m_sLlcSha = llcSha;
		m_sExpertSha = expertSha;
		m_sVehicleCardId = vehicleCardId;
		m_sVehicleCardSha = vehicleCardSha;
	}

	//------------------------------------------------------------------------------------------------
	//! control.steer_sign and control.yaw_axis_sign from the vehicle card (TBD-5); -1 or +1 each
	void SetSigns(int steerSign, int yawAxisSign)
	{
		m_iSteerSign = steerSign;
		m_iYawAxisSign = yawAxisSign;
	}

	//------------------------------------------------------------------------------------------------
	//! An events row (only while the session captures; kind must be a dc-2.0 event kind, e.g. nav_command)
	bool LogEvent(string kind, string payload)
	{
		if (!m_Session || !m_Session.IsCapturing())
			return false;

		return m_Session.LogEvent(kind, payload);
	}

	//------------------------------------------------------------------------------------------------
	void Log(string message)
	{
		Print("[DC_Driver] " + message, LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	void Warn(string message)
	{
		Print("[DC_Driver] WARNING: " + message, LogLevel.WARNING);
	}

	//------------------------------------------------------------------------------------------------
	void Error(string message)
	{
		Print("[DC_Driver] ERROR: " + message, LogLevel.ERROR);
	}

	//------------------------------------------------------------------------------------------------
	string GetLlcSha()
	{
		return m_sLlcSha;
	}

	//------------------------------------------------------------------------------------------------
	string GetExpertSha()
	{
		return m_sExpertSha;
	}

	//------------------------------------------------------------------------------------------------
	string GetVehicleCardId()
	{
		return m_sVehicleCardId;
	}

	//------------------------------------------------------------------------------------------------
	string GetVehicleCardSha()
	{
		return m_sVehicleCardSha;
	}

	//------------------------------------------------------------------------------------------------
	int GetSteerSign()
	{
		return m_iSteerSign;
	}

	//------------------------------------------------------------------------------------------------
	int GetYawAxisSign()
	{
		return m_iYawAxisSign;
	}
}

//------------------------------------------------------------------------------------------------
//! The driver interface (see the header): DD-7-min's llc_v1 + expert implement it as a component on the capture game
//! mode. This base class is the null driver: no spawn pose, no episode, no command
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 driver interface; this base class is the null driver (no episode, no HL command, the ego stays parked)")]
class DC_DriverInterfaceClass : ScriptComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_DriverInterface : ScriptComponent
{
	//------------------------------------------------------------------------------------------------
	//! Where the ego is spawned (world x, z; y from the surface) facing headingDeg (0 = +Z, clockwise from above).
	//! false = DC_EgoSession's attribute pose
	bool GetSpawnPose(out vector position, out float headingDeg)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Once, during readiness: the ego is spawned and its actuator exists; no directory yet. Load the route and the
	//! vehicle card here and report the stamps (ctx.SetStamps, ctx.SetSigns). false = no session (ERROR)
	bool BeginDriver(notnull DC_DriverContext ctx)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! At a decision step after the warm-up while no episode runs. Fill info (mode, driver, card, route, shas, style,
	//! seed). false = no episode now (asked again at the next decision step)
	bool BeginEpisode(int episodeId, int firstPhysicsStep, notnull DC_EpisodeInfo info)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! EOnSimulate of a step with a held command: compute and pass the inputs through ctx.GetActuator().Apply and
	//! record exactly what was passed in llc (m_bActive = true)
	void ApplyStep(int physicsStep, float timeSlice, notnull DC_HLCommand held, notnull DC_LLCOutput llc)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! EOnPostSimulate of every step of the episode, before its ctl row: the expert on the state after the step,
	//! lane errors and route progress, and a termination request
	void PostStep(int physicsStep, float timeSlice, notnull DC_DriverPost post)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! At every decision step of the episode, after its rows: the command applied over (P, P + N]. false = no new
	//! decision (the held command stays, hl_seq does not change)
	bool Decide(int physicsStep, notnull DC_HLCommand command)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! The episode ended at lastPhysicsStep (an episodes.termination_reason); the core parks the ego afterwards
	void EndEpisode(int episodeId, int lastPhysicsStep, string terminationReason)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Once at the seal or after a refused readiness (endReason = session_end.json end_reason or "refused")
	void EndDriver(string endReason)
	{
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 ctl and telemetry rows on physics-step counters, the events stream, and the per-step driver calls")]
class DC_StreamRecorderClass : DC_SessionModuleClass
{
}

//------------------------------------------------------------------------------------------------
class DC_StreamRecorder : DC_SessionModule
{
	// === SIGNS (TBD-5; recorded in session.json control.*, a vehicle card overrides through the driver) ===
	[Attribute("-1", UIWidgets.Slider, "Engine-to-contract steering sign (TBD-5): CarSteering is +1 for left, the contract is right-positive", "-1 1 2", category: "Streams")]
	protected int m_iSteerSign;

	[Attribute("1", UIWidgets.Slider, "Engine-to-contract yaw-rate sign (TBD-5): yaw_rate_rps = sign x angvel_local_y", "-1 1 2", category: "Streams")]
	protected int m_iYawAxisSign;

	[Attribute("unknown_on_1.8", UIWidgets.EditBox, "session.json control.steering_geometry (the draft value until CC-07 validates wheel angles)", category: "Streams")]
	protected string m_sSteeringGeometry;

	[Attribute("-1", UIWidgets.Slider, "telemetry.sun_elev_deg = sign x asin(sun_dir_y) (-1: GetCurrentSunMoonDirAndPhase gives the light direction, as vanilla SCR_AIDebugVisualization.c:173 reads it; checked at T0-MIN: noon is positive)", "-1 1 2", category: "Streams")]
	protected int m_iSunElevationSign;

	[Attribute("1", UIWidgets.CheckBox, "Road context per telemetry row (RoadNetworkManager.GetClosestRoad; road_key as DC_RoadDump, TBD-9)", category: "Streams")]
	protected bool m_bRoadContext;

	[Attribute("1", UIWidgets.CheckBox, "T0-MIN is forward-only until the reverse and neutral gear indices are measured: a command with dir -1 fails the session", category: "Streams")]
	protected bool m_bForwardOnly;

	// === CONSTANTS (dc-2.0.streams.json chunk_max_rows) ===
	static const int CTL_CHUNK_ROWS = 600;
	static const int TELEMETRY_CHUNK_ROWS = 200;
	static const int EVENTS_CHUNK_ROWS = 100;
	static const int WHEEL_COLUMNS = 4;
	static const string LOG = "[DC_Streams] ";

	//! episodes.termination_reason values (dc-2.0 9.6)
	protected static ref array<string> s_aTerminations = {"route_complete", "timeout", "offroad", "route_deviation", "stuck", "rollover", "collision", "engine_off", "script_error", "session_stop"};

	// === STATE ===
	protected DC_Session m_Session;
	protected DC_DriverInterface m_Driver;
	protected bool m_bDriverSearched;
	protected bool m_bDriverBegun;
	protected ref DC_Actuator m_Actuator;
	protected ref DC_DriverContext m_Context;
	protected DC_ChunkWriter m_Ctl;
	protected DC_ChunkWriter m_Telemetry;
	protected DC_ChunkWriter m_Events;
	protected TimeAndWeatherManagerEntity m_Weather;
	protected RoadNetworkManager m_Roads;
	protected string m_sEndReason = "refused";

	// HL command state
	protected ref DC_HLCommand m_Held = new DC_HLCommand();
	protected ref DC_HLCommand m_Pending = new DC_HLCommand();
	protected bool m_bHasHeld;
	protected bool m_bHasPending;
	protected int m_iHlSeq = -1;
	protected ref DC_EpisodeInfo m_EpisodeInfo;
	protected string m_sTermination;

	// per-step values
	protected ref DC_LLCOutput m_Llc = new DC_LLCOutput();
	protected ref DC_InputReadback m_Pre = new DC_InputReadback();
	protected ref DC_InputReadback m_PostInputs = new DC_InputReadback();
	protected ref DC_DriverPost m_DriverPost = new DC_DriverPost();
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();

	// road cache
	protected BaseRoad m_LastRoad;
	protected string m_sLastRoadKey;
	protected ref array<vector> m_aRoadPoints = {};

	//------------------------------------------------------------------------------------------------
	override int GetDispatchOrder()
	{
		return 20;
	}

	//------------------------------------------------------------------------------------------------
	override bool OnSessionPrepare(notnull DC_Session session)
	{
		if (SCR_Global.IsEditMode())
			return false;

		m_Session = session;
		DC_DriverInterface driver = GetDriver();
		if (driver)
			Print(LOG + "driver: " + driver.ClassName(), LogLevel.NORMAL);
		else
			Print(LOG + "no DC_DriverInterface on the game mode: the null driver applies (no episode, no HL command)", LogLevel.NORMAL);

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Ready once the ego module handed over the actuator and the driver accepted it (BeginDriver)
	override bool IsSessionReady(out string waitingFor)
	{
		if (!m_Actuator)
		{
			waitingFor = "the ego and its actuator";
			return false;
		}

		if (m_bDriverBegun)
			return true;

		m_Context = new DC_DriverContext(m_Session, m_Session.GetEgo(), m_Actuator);
		DC_DriverInterface driver = GetDriver();
		if (driver && !driver.BeginDriver(m_Context))
		{
			m_Session.RefuseReadiness("driver " + driver.ClassName() + " refused to begin (see its ERROR line)");
			waitingFor = "the driver";
			return false;
		}

		m_bDriverBegun = true;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void WriteSessionJson(string section, notnull DC_JsonWriter json)
	{
		if (section != "control")
			return;

		string mode = "A";
		if (m_Actuator)
			mode = m_Actuator.GetMode();

		json.AddString("actuation_mode", mode);
		if (m_Context)
		{
			json.AddStringOrNull("llc_sha", m_Context.GetLlcSha());
			json.AddStringOrNull("expert_sha", m_Context.GetExpertSha());
			json.AddStringOrNull("vehicle_card_id", m_Context.GetVehicleCardId());
			json.AddStringOrNull("vehicle_card_sha", m_Context.GetVehicleCardSha());
		}
		else
		{
			json.AddNull("llc_sha");
			json.AddNull("expert_sha");
			json.AddNull("vehicle_card_id");
			json.AddNull("vehicle_card_sha");
		}

		json.AddInt("steer_sign", GetSteerSign());
		json.AddInt("yaw_axis_sign", GetYawAxisSign());
		json.AddString("steering_geometry", m_sSteeringGeometry);
	}

	//------------------------------------------------------------------------------------------------
	//! ctl, telemetry and events writers (owned by the session); the weather manager and road network
	override bool OnSessionOpen(notnull DC_Session session)
	{
		ChimeraWorld world = ChimeraWorld.CastFrom(GetGame().GetWorld());
		if (world)
			m_Weather = world.GetTimeAndWeatherManager();

		if (!m_Weather)
		{
			Print(LOG + "ERROR: no TimeAndWeatherManagerEntity: the telemetry environment fields cannot be written", LogLevel.ERROR);
			return false;
		}

		SCR_AIWorld aiWorld = SCR_AIWorld.Cast(GetGame().GetAIWorld());
		if (m_bRoadContext && aiWorld)
			m_Roads = aiWorld.GetRoadNetworkManager();

		if (m_bRoadContext && !m_Roads)
			Print(LOG + "WARNING: no road network manager (SCR_AIWorld): the telemetry road fields stay empty", LogLevel.WARNING);

		array<string> columns = {};
		AddCtlColumns(columns);
		m_Ctl = OpenStream(session, "ctl", columns, CTL_CHUNK_ROWS);

		columns.Clear();
		AddTelemetryColumns(columns);
		m_Telemetry = OpenStream(session, "telemetry", columns, TELEMETRY_CHUNK_ROWS);

		columns.Clear();
		DC_Session.AddEventColumns(columns);
		m_Events = OpenStream(session, "events", columns, EVENTS_CHUNK_ROWS);
		if (!m_Ctl || !m_Telemetry || !m_Events)
			return false;

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! A DC_ChunkWriter registered with the session; null (and FailWrite) when it cannot be opened
	protected DC_ChunkWriter OpenStream(notnull DC_Session session, string name, notnull array<string> columns, int maxRows)
	{
		DC_ChunkWriter writer = new DC_ChunkWriter();
		bool opened = writer.Open(session.GetSessionDir(), name, columns, maxRows, session.GetChunkMaxMs(), session.GetKeepLogsOpen(), session.GetVerifyChunkLength());
		session.RegisterStream(writer);
		if (opened)
			return writer;

		session.FailWrite(name + ": " + writer.GetFailure());
		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! A decision of the previous step becomes the held command; the driver applies it; input_pre_* read back
	override void OnStepPre(int physicsStep, float timeSlice)
	{
		if (m_Session.GetEpisodeId() < 0)
		{
			m_bHasHeld = false;
			m_bHasPending = false;
		}
		else if (m_bHasPending)
		{
			m_Held.CopyFrom(m_Pending);
			m_bHasHeld = true;
			m_bHasPending = false;
			m_iHlSeq++;
		}

		m_Llc.Reset();
		DC_DriverInterface driver = GetDriver();
		if (m_bHasHeld && driver)
			driver.ApplyStep(physicsStep, timeSlice, m_Held, m_Llc);

		if (m_Actuator)
			m_Actuator.ReadInputs(m_Pre);
	}

	//------------------------------------------------------------------------------------------------
	//! input_post_*, the driver's PostStep, then the ctl row and (every k-th step) the telemetry row
	override void OnStepPost(int physicsStep, float timeSlice, int worldMs)
	{
		IEntity ego = m_Session.GetEgo();
		VehicleWheeledSimulation simulation;
		if (m_Actuator)
			simulation = m_Actuator.GetSimulation();

		if (!ego || !simulation)
		{
			m_Session.Fail(DC_Session.END_EGO_LOST, string.Format("no ego simulation in physics step %1", physicsStep));
			return;
		}

		m_Actuator.ReadInputs(m_PostInputs);
		m_DriverPost.Reset();
		DC_DriverInterface driver = GetDriver();
		if (m_Session.GetEpisodeId() >= 0 && driver)
		{
			driver.PostStep(physicsStep, timeSlice, m_DriverPost);
			if (!m_DriverPost.m_sTermination.IsEmpty() && m_sTermination.IsEmpty())
			{
				if (s_aTerminations.Find(m_DriverPost.m_sTermination) < 0)
				{
					m_Session.Fail(DC_Session.END_SCRIPT_ERROR, "driver termination '" + m_DriverPost.m_sTermination + "' is not an episodes.termination_reason");
					return;
				}

				m_sTermination = m_DriverPost.m_sTermination;
			}
		}

		if (m_Session.HasWriteFailure())
			return;

		WriteCtlRow(physicsStep, worldMs);
		if (!m_Session.HasWriteFailure() && physicsStep % m_Session.GetTelemetryEveryK() == 0)
			WriteTelemetryRow(physicsStep, worldMs, ego, simulation);
	}

	//------------------------------------------------------------------------------------------------
	//! At a decision step inside an episode: the driver's command for (P, P + N]
	override void OnStepDecide(int physicsStep)
	{
		if (physicsStep % m_Session.GetStepsPerDecision() != 0 || m_Session.GetEpisodeId() < 0)
			return;

		DC_DriverInterface driver = GetDriver();
		if (!driver)
			return;

		m_Pending.Reset();
		if (!driver.Decide(physicsStep, m_Pending))
			return;

		if (m_Pending.m_iDir < -1 || m_Pending.m_iDir > 1 || (m_bForwardOnly && m_Pending.m_iDir < 0))
		{
			m_Session.Fail(DC_Session.END_SCRIPT_ERROR, string.Format("driver command dir %1 at physics step %2 (allowed: 0, 1 while forward-only; -1, 0, 1 otherwise)", m_Pending.m_iDir, physicsStep));
			return;
		}

		m_bHasPending = true;
	}

	//------------------------------------------------------------------------------------------------
	override void OnSessionClosing(string endReason)
	{
		m_sEndReason = endReason;
	}

	//------------------------------------------------------------------------------------------------
	override void OnSessionRelease(bool teardown)
	{
		DC_DriverInterface driver = GetDriver();
		if (!teardown && driver && m_bDriverBegun)
			driver.EndDriver(m_sEndReason);

		m_bDriverBegun = false;
		m_Driver = null;
		m_Context = null;
		m_Actuator = null;
		m_Ctl = null;
		m_Telemetry = null;
		m_Events = null;
		m_LastRoad = null;
	}

	//------------------------------------------------------------------------------------------------
	//! Every column of ctl in dc-2.0 9.1 order
	static void AddCtlColumns(notnull array<string> columns)
	{
		array<string> head = {"physics_step", "world_time_ms", "episode_id", "mode", "driver", "hl_seq", "hl_kappa", "hl_accel", "hl_dir", "exp_kappa_now", "exp_accel_now", "exp_dir_now", "llc_steer_raw", "llc_steer_lim", "llc_throttle", "llc_brake", "llc_handbrake", "llc_gear", "llc_clutch"};
		foreach (string name : head)
		{
			columns.Insert(name);
		}

		AddInputColumns(columns, "input_pre_");
		AddInputColumns(columns, "input_post_");
		array<string> tail = {"noise_active", "noise_id", "steer_signal_raw", "wheel_angle_l_rad", "wheel_angle_r_rad", "react_veh_slot_id", "react_visible"};
		foreach (string tailName : tail)
		{
			columns.Insert(tailName);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected static void AddInputColumns(notnull array<string> columns, string prefix)
	{
		columns.Insert(prefix + "steer");
		columns.Insert(prefix + "throttle");
		columns.Insert(prefix + "brake");
		columns.Insert(prefix + "clutch");
		columns.Insert(prefix + "gear");
		columns.Insert(prefix + "handbrake");
	}

	//------------------------------------------------------------------------------------------------
	//! Every column of telemetry in dc-2.0 9.2 order
	static void AddTelemetryColumns(notnull array<string> columns)
	{
		columns.Insert("physics_step");
		columns.Insert("world_time_ms");
		columns.Insert("episode_id");
		columns.Insert("ego_slot_id");
		columns.Insert("ego_entity_id");
		DC_ProbeCsv.AddVectorColumns(columns, "pos");
		DC_ProbeCsv.AddVectorColumns(columns, "right");
		DC_ProbeCsv.AddVectorColumns(columns, "up");
		DC_ProbeCsv.AddVectorColumns(columns, "fwd");
		DC_ProbeCsv.AddVectorColumns(columns, "vel_local");
		DC_ProbeCsv.AddVectorColumns(columns, "angvel_local");
		DC_ProbeCsv.AddVectorColumns(columns, "acc_local");
		array<string> vehicle = {"speed_engine_mps", "gear", "rpm", "engine_on", "handbrake", "ground_contact", "wheel_count"};
		foreach (string name : vehicle)
		{
			columns.Insert(name);
		}

		for (int i = 0; i < WHEEL_COLUMNS; i++)
		{
			string prefix = "w" + i.ToString() + "_";
			columns.Insert(prefix + "contact");
			columns.Insert(prefix + "slip_long");
			columns.Insert(prefix + "slip_lat");
			columns.Insert(prefix + "material");
			columns.Insert(prefix + "temp_c");
		}

		array<string> context = {"road_key", "road_dist_m", "road_width_m", "lane_lat_err_m", "lane_head_err_rad", "route_progress_m", "env_weather_state", "env_rain", "env_fog", "env_fog_height_density", "env_wetness", "env_tod_h"};
		foreach (string contextName : context)
		{
			columns.Insert(contextName);
		}

		DC_ProbeCsv.AddVectorColumns(columns, "sun_dir");
		columns.Insert("moon_phase01");
		columns.Insert("sun_elev_deg");
		columns.Insert("traffic_n");
		columns.Insert("traffic_slots");
	}

	//------------------------------------------------------------------------------------------------
	//! The ctl row of a step (dc-2.0 9.1): hl_* of the held command, exp_*_now of PostStep, llc_* of ApplyStep, the
	//! input readbacks; N-mode, wheel-angle and reaction fields stay empty in T0-MIN
	protected void WriteCtlRow(int physicsStep, int worldMs)
	{
		int episodeId = m_Session.GetEpisodeId();
		string mode = "I";
		string driverKind = "none";
		if (episodeId >= 0 && m_EpisodeInfo)
		{
			mode = m_EpisodeInfo.m_sMode;
			driverKind = m_EpisodeInfo.m_sDriver;
		}

		m_Row.Clear();
		m_Row.AddInt(physicsStep);
		m_Row.AddInt(worldMs);
		m_Row.AddInt(episodeId);
		m_Row.AddString(mode);
		m_Row.AddString(driverKind);
		m_Row.AddInt(m_iHlSeq);
		if (m_bHasHeld)
		{
			m_Row.AddFloat(m_Held.m_fKappa, 6);
			m_Row.AddFloat(m_Held.m_fAccel, 4);
			m_Row.AddInt(m_Held.m_iDir);
		}
		else
		{
			AddEmpty(3);
		}

		if (m_DriverPost.m_bHasExpert)
		{
			m_Row.AddFloat(m_DriverPost.m_fExpKappa, 6);
			m_Row.AddFloat(m_DriverPost.m_fExpAccel, 4);
			m_Row.AddInt(m_DriverPost.m_iExpDir);
		}
		else
		{
			AddEmpty(3);
		}

		if (m_Llc.m_bActive)
		{
			m_Row.AddFloat(m_Llc.m_fSteerRaw, 4);
			m_Row.AddFloat(m_Llc.m_fSteerLim, 4);
			m_Row.AddFloat(m_Llc.m_fThrottle, 4);
			m_Row.AddFloat(m_Llc.m_fBrake, 4);
			m_Row.AddBool(m_Llc.m_bHandbrake);
			if (m_Llc.m_iGear >= 0)
				m_Row.AddInt(m_Llc.m_iGear);
			else
				m_Row.AddEmpty();

			if (m_Llc.m_fClutch >= 0)
				m_Row.AddFloat(m_Llc.m_fClutch, 4);
			else
				m_Row.AddEmpty();
		}
		else
		{
			AddEmpty(7);
		}

		AddInputValues(m_Pre);
		AddInputValues(m_PostInputs);
		m_Row.AddBool(false);		// noise_active (no N mode in T0-MIN)
		m_Row.AddInt(-1);			// noise_id
		AddEmpty(5);				// steer_signal_raw, wheel angles, react_*: reserved
		if (m_Ctl && !m_Ctl.AddRow(m_Row, physicsStep, worldMs))
			m_Session.FailWrite("ctl: " + m_Ctl.GetFailure());
	}

	//------------------------------------------------------------------------------------------------
	protected void AddInputValues(notnull DC_InputReadback readback)
	{
		m_Row.AddFloat(readback.m_fSteer, 4);
		m_Row.AddFloat(readback.m_fThrottle, 4);
		m_Row.AddFloat(readback.m_fBrake, 4);
		m_Row.AddFloat(readback.m_fClutch, 4);
		m_Row.AddInt(readback.m_iGear);
		m_Row.AddBool(readback.m_bHandbrake);
	}

	//------------------------------------------------------------------------------------------------
	protected void AddEmpty(int count)
	{
		for (int i = 0; i < count; i++)
		{
			m_Row.AddEmpty();
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The telemetry row of a k-th step (dc-2.0 9.2)
	protected void WriteTelemetryRow(int physicsStep, int worldMs, notnull IEntity ego, notnull VehicleWheeledSimulation simulation)
	{
		vector transform[4];
		ego.GetWorldTransform(transform);

		m_Row.Clear();
		m_Row.AddInt(physicsStep);
		m_Row.AddInt(worldMs);
		m_Row.AddInt(m_Session.GetEpisodeId());
		m_Row.AddInt(m_Session.GetEgoSlotId());
		m_Row.AddString(DC_ProbeCsv.CleanField(m_Session.GetEgoEntityId()));
		m_Row.AddVector(transform[3], 3);
		m_Row.AddVector(transform[0], 6);
		m_Row.AddVector(transform[1], 6);
		m_Row.AddVector(transform[2], 6);
		m_Row.AddVector(simulation.GetVelocityLocal(), 4);
		m_Row.AddVector(simulation.GetAngularVelocityLocal(), 6);
		m_Row.AddVector(simulation.GetAccelerationLocal(), 4);
		m_Row.AddFloat(simulation.GetSpeedKmh() / 3.6, 4);
		m_Row.AddInt(simulation.GetGear());
		m_Row.AddFloat(simulation.EngineGetRPM(), 1);
		m_Row.AddBool(simulation.EngineIsOn());
		m_Row.AddBool(simulation.IsHandbrakeOn());
		m_Row.AddBool(simulation.HasAnyGroundContact());
		int wheelCount = simulation.WheelCount();
		m_Row.AddInt(wheelCount);
		for (int i = 0; i < WHEEL_COLUMNS; i++)
		{
			if (i >= wheelCount)
			{
				AddEmpty(5);
				continue;
			}

			bool contact = simulation.WheelHasContact(i);
			m_Row.AddBool(contact);
			m_Row.AddFloat(simulation.WheelGetLongitudinalSlip(i), 4);
			m_Row.AddFloat(simulation.WheelGetLateralSlip(i), 4);
			GameMaterial material;
			if (contact)
				material = simulation.WheelGetContactMaterial(i);

			if (material)
				m_Row.AddString(DC_ProbeCsv.CleanField(material.GetResourceName()));
			else
				m_Row.AddEmpty();

			m_Row.AddEmpty();		// wN_temp_c: reserved
		}

		AddRoadValues(transform[3]);
		if (m_DriverPost.m_bHasRoute)
		{
			m_Row.AddFloat(m_DriverPost.m_fLaneLatErrM, 4);
			m_Row.AddFloat(m_DriverPost.m_fLaneHeadErrRad, 6);
			m_Row.AddFloat(m_DriverPost.m_fRouteProgressM, 3);
		}
		else
		{
			AddEmpty(3);
		}

		AddEnvironmentValues();
		m_Row.AddInt(0);			// traffic_n: no traffic before T2
		m_Row.AddEmpty();			// traffic_slots
		if (m_Telemetry && !m_Telemetry.AddRow(m_Row, physicsStep, worldMs))
			m_Session.FailWrite("telemetry: " + m_Telemetry.GetFailure());
	}

	//------------------------------------------------------------------------------------------------
	//! road_key, road_dist_m, road_width_m of the closest road; empty without one. The key is DC_RoadDump.MakeKey's
	//! text (TBD-9), cached per road object
	protected void AddRoadValues(vector position)
	{
		BaseRoad road;
		float distance;
		if (m_Roads)
			m_Roads.GetClosestRoad(position, road, distance);

		if (!road)
		{
			AddEmpty(3);
			return;
		}

		if (road != m_LastRoad)
		{
			m_aRoadPoints.Clear();
			road.GetPoints(m_aRoadPoints);
			m_sLastRoadKey = MakeRoadKey(m_aRoadPoints);
			m_LastRoad = road;
		}

		m_Row.AddString(m_sLastRoadKey);
		m_Row.AddFloat(distance, 3);
		m_Row.AddFloat(road.GetWidth(), 3);
	}

	//------------------------------------------------------------------------------------------------
	//! TBD-9 road key <n_pts>/<x0>:<z0>/<x1>:<z1>, 2 decimals, "0" without points: the text of DC_RoadDump.MakeKey
	//! (DD-4-probe), which is protected there; E-08b joins the two on this text
	static string MakeRoadKey(notnull array<vector> points)
	{
		int count = points.Count();
		if (count == 0)
			return "0";

		vector first = points[0];
		vector last = points[count - 1];
		float x0 = first[0];
		float z0 = first[2];
		float x1 = last[0];
		float z1 = last[2];
		return count.ToString() + "/" + x0.ToString(-1, 2) + ":" + z0.ToString(-1, 2) + "/" + x1.ToString(-1, 2) + ":" + z1.ToString(-1, 2);
	}

	//------------------------------------------------------------------------------------------------
	//! env_* and the sun and moon (GetCurrentSunMoonDirAndPhase)
	protected void AddEnvironmentValues()
	{
		string stateName = "unknown";
		WeatherState state = m_Weather.GetCurrentWeatherState();
		if (state)
			stateName = DC_ProbeCsv.CleanField(state.GetStateName());

		vector sunDir;
		vector moonDir;
		float moonPhase;
		m_Weather.GetCurrentSunMoonDirAndPhase(sunDir, moonDir, moonPhase);
		float sunElevation = m_iSunElevationSign * Math.Asin(Math.Clamp(sunDir[1], -1, 1)) * Math.RAD2DEG;

		m_Row.AddString(stateName);
		m_Row.AddFloat(m_Weather.GetRainIntensity(), 4);
		m_Row.AddFloat(m_Weather.GetFogAmount(), 4);
		m_Row.AddFloat(m_Weather.GetFogHeightDensity(), 4);
		m_Row.AddFloat(m_Weather.GetCurrentWetness(), 4);
		m_Row.AddFloat(m_Weather.GetTimeOfTheDay(), 4);
		m_Row.AddVector(sunDir, 6);
		m_Row.AddFloat(moonPhase, 4);
		m_Row.AddFloat(sunElevation, 3);
	}

	//------------------------------------------------------------------------------------------------
	// API of DC_EgoSession
	//------------------------------------------------------------------------------------------------
	//! The first DC_DriverInterface component on the game mode entity, null for the null driver
	DC_DriverInterface GetDriver()
	{
		if (m_bDriverSearched)
			return m_Driver;

		m_bDriverSearched = true;
		IEntity owner = GetOwner();
		if (!owner)
			return null;

		array<Managed> found = {};
		owner.FindComponents(DC_DriverInterface, found);
		foreach (Managed managed : found)
		{
			DC_DriverInterface driver = DC_DriverInterface.Cast(managed);
			if (!driver)
				continue;

			if (!m_Driver)
				m_Driver = driver;
			else
				Print(LOG + "WARNING: more than one DC_DriverInterface on the game mode - using " + m_Driver.ClassName(), LogLevel.WARNING);
		}

		return m_Driver;
	}

	//------------------------------------------------------------------------------------------------
	//! The ego module hands over the ego's actuator once the ego is spawned
	void SetActuator(DC_Actuator actuator)
	{
		m_Actuator = actuator;
	}

	//------------------------------------------------------------------------------------------------
	DC_Actuator GetActuator()
	{
		return m_Actuator;
	}

	//------------------------------------------------------------------------------------------------
	//! The episode that just began (its mode and driver go into the ctl rows); a new episode has no command yet
	void SetEpisode(notnull DC_EpisodeInfo info)
	{
		m_EpisodeInfo = info;
		m_bHasHeld = false;
		m_bHasPending = false;
		m_sTermination = string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! The episode ended: the held command is dropped at the next step
	void ClearEpisode()
	{
		m_EpisodeInfo = null;
		m_bHasPending = false;
		m_sTermination = string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! The first termination the driver requested in the running episode, empty when none
	string GetTermination()
	{
		return m_sTermination;
	}

	//------------------------------------------------------------------------------------------------
	//! control.steer_sign: the vehicle card's (through the driver) when set, else the attribute
	int GetSteerSign()
	{
		if (m_Context && m_Context.GetSteerSign() != 0)
			return m_Context.GetSteerSign();

		return m_iSteerSign;
	}

	//------------------------------------------------------------------------------------------------
	int GetYawAxisSign()
	{
		if (m_Context && m_Context.GetYawAxisSign() != 0)
			return m_Context.GetYawAxisSign();

		return m_iYawAxisSign;
	}

	//------------------------------------------------------------------------------------------------
	//! Class name of the driver component, "none" for the null driver
	string GetDriverName()
	{
		DC_DriverInterface driver = GetDriver();
		if (!driver)
			return "none";

		return driver.ClassName();
	}
}
