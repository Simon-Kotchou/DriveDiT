// ============================================================================
// DC_EgoSession - the ego of a dc-2.0 session: spawn, Mode A settle, liveness, slot identity, episodes (DD-4-min)
// ============================================================================
//
// A DC_SessionModule (dispatch order 10: it runs before the streams in every hook) on the capture game mode.
//
// READINESS (before the session directory exists)
//   - spawns the ego driverless from m_sEgoPrefab (GetGame().SpawnEntityPrefab with a world transform, the vanilla
//     pattern of game.c:93-104) at the driver's spawn pose (DC_DriverInterface.GetSpawnPose), else at the attribute
//     pose: the Arland main runway of the DD-4-probe test area. Nothing is looked up by name;
//   - registers it in the slot registry (DC_VehicleRegistry: slot id + IEntity.GetID() token; the ego slot is fixed
//     for the whole session and every episode);
//   - Mode A as in DD-4-probe: DC_ActuatorModeA (persistent handbrake off, engine started), parked (brake 1 and
//     handbrake) and parked again on every readiness poll of the settle, handed to DC_StreamRecorder (the one
//     actuation path, r3 P1), which re-applies the park on every physics step without a held command;
//   - ready after m_fSettleS of world time with the ego alive and slower than m_fSettleMaxKmh. A spawn failure or an
//     ego lost during the settle refuses readiness; DC_Session's timeout refuses it too (ERROR, no directory).
//
// CAPTURE
//   - liveness and identity at the top of every frame and before and after the simulation of every physics step
//     (before any row of it): the entity exists, is not deleted or destroyed, and still carries the registered GetID
//     token. A loss before the first ctl row ends the session with zero_ego (0 rows, 0 images), later with ego_lost;
//   - episodes: at a decision step (physics_step mod N == 0) after warmup_end, while no episode runs and the session
//     is not stopping, the driver's BeginEpisode is asked in OnStepPre (so the rows of that step carry the episode;
//     the null driver never begins one). A termination the driver reports, or the planned end, ends the episode at
//     the next decision step in OnStepDecide, after that step's rows: DC_DriverInterface.EndEpisode, one episodes
//     row (chunk_max_rows 1, so it is committed at once), events episode_start / episode_end, the ego parked. The
//     seal ends an episode still open at the last step (session_stop, or script_error after an error).
//
// Release: the ego is parked and kept (it dies with the world); on world teardown only references are dropped.
// ============================================================================

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 ego: driverless spawn, Mode A settle, liveness and slot identity, episode lifecycle and the episodes stream")]
class DC_EgoSessionClass : DC_SessionModuleClass
{
}

//------------------------------------------------------------------------------------------------
class DC_EgoSession : DC_SessionModule
{
	// === EGO ===
	[Attribute("{DA79E34823120087}Prefabs/Vehicles/Wheeled/S105/S105_base.et", UIWidgets.ResourcePickerThumbnail, "Ego vehicle prefab, spawned driverless (the S105 of DD-4-probe)", "et", category: "Ego")]
	protected ResourceName m_sEgoPrefab;

	[Attribute("1495.41 0 2626", UIWidgets.Coords, "Spawn position when the driver gives none (x, z used; y from GetSurfaceY): 6 m along the DD-4-probe runway test area [decoder-derived, unverified in the engine]", category: "Ego")]
	protected vector m_vSpawnPos;

	[Attribute("0.11", UIWidgets.Slider, "Spawn heading when the driver gives none [deg], 0 = +Z, clockwise seen from above (the runway centre line)", "-180 180 0.01", category: "Ego")]
	protected float m_fSpawnHeadingDeg;

	[Attribute("0.3", UIWidgets.Slider, "Spawn height above GetSurfaceY [m]", "0 5 0.05", category: "Ego")]
	protected float m_fSpawnLiftM;

	[Attribute("3", UIWidgets.Slider, "Settle after the spawn, brake and handbrake on [s of world time] (r2: capture starts only after the 3 s settle)", "0 30 0.5", category: "Ego")]
	protected float m_fSettleS;

	[Attribute("1", UIWidgets.Slider, "The ego counts as settled below this speed [km/h]", "0 20 0.1", category: "Ego")]
	protected float m_fSettleMaxKmh;

	[Attribute("0", UIWidgets.CheckBox, "Mode A: ForceEnableSimulation on every actuation (set only if E-01 shows the driverless S105 does not move without it)", category: "Ego")]
	protected bool m_bForceEnableSimulation;

	// === CONSTANTS ===
	static const int EPISODES_CHUNK_ROWS = 1;		//!< dc-2.0.streams.json episodes chunk_max_rows
	static const string TERMINATION_SESSION_STOP = "session_stop";
	static const string TERMINATION_SCRIPT_ERROR = "script_error";
	static const string LOG = "[DC_Ego] ";

	//! Episode modes the core can record: N, R and X need disturbance, recovery and exploration support (not in T0-MIN)
	protected static ref array<string> s_aEpisodeModes = {"E", "D", "H", "A"};
	protected static ref array<string> s_aDrivers = {"expert", "policy", "human", "ai", "none"};

	// === STATE ===
	protected DC_Session m_Session;
	protected DC_StreamRecorder m_Recorder;
	protected ref DC_VehicleRegistry m_Registry;
	protected ref DC_ActuatorModeA m_Actuator;
	protected IEntity m_Ego;
	protected DamageManagerComponent m_Damage;
	protected int m_iSlotId = -1;
	protected string m_sPrefab;
	protected float m_fSpawnWorldMs;
	protected bool m_bFaultEgoLost;
	protected DC_ChunkWriter m_Episodes;
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();

	// episode
	protected ref DC_EpisodeInfo m_EpisodeInfo;
	protected int m_iNextEpisodeId;
	protected int m_iEpisodeId = -1;
	protected int m_iEpisodeStartStep;
	protected int m_iEpisodeStartWorldMs;
	protected int m_iLastWorldMs;
	protected string m_sStopTermination;		//!< latched when the planned end begins during an episode
	protected string m_sEpisodeWeather;
	protected float m_fEpisodeTodH;
	protected float m_fEpisodeRain;
	protected float m_fEpisodeFog;
	protected float m_fEpisodeWetness;

	//------------------------------------------------------------------------------------------------
	override int GetDispatchOrder()
	{
		return 10;
	}

	//------------------------------------------------------------------------------------------------
	//! Spawn, register, Mode A start and park, hand the actuator to the streams
	override bool OnSessionPrepare(notnull DC_Session session)
	{
		if (SCR_Global.IsEditMode())
			return false;

		m_Session = session;
		m_Recorder = DC_StreamRecorder.Cast(GetOwner().FindComponent(DC_StreamRecorder));
		if (!m_Recorder)
		{
			session.RefuseReadiness("no DC_StreamRecorder on the game mode (the ego actuator has no user)");
			return false;
		}

		vector position = m_vSpawnPos;
		float headingDeg = m_fSpawnHeadingDeg;
		string poseSource = "attributes";
		DC_DriverInterface driver = m_Recorder.GetDriver();
		vector driverPosition;
		float driverHeadingDeg;
		if (driver && driver.GetSpawnPose(driverPosition, driverHeadingDeg))
		{
			position = driverPosition;
			headingDeg = driverHeadingDeg;
			poseSource = driver.ClassName();
		}

		string failure;
		if (!SpawnEgo(position, headingDeg, failure))
		{
			session.RefuseReadiness("ego: " + failure);
			return false;
		}

		m_Registry = new DC_VehicleRegistry();
		m_iSlotId = m_Registry.Register(m_Ego);
		m_Damage = DamageManagerComponent.Cast(m_Ego.FindComponent(DamageManagerComponent));

		m_Actuator = new DC_ActuatorModeA();
		if (!m_Actuator.Init(m_Ego))
		{
			session.RefuseReadiness("ego actuator: " + m_Actuator.GetFailure());
			return false;
		}

		m_Actuator.SetForceSimulation(m_bForceEnableSimulation);
		m_Actuator.Start();
		m_Actuator.Park();
		m_Recorder.SetActuator(m_Actuator);
		session.SetEgo(m_Ego, m_iSlotId, m_Registry.GetEntityId(m_iSlotId), m_sPrefab);
		m_fSpawnWorldMs = GetGame().GetWorld().GetWorldTime();

		Print(string.Format(LOG + "ego %1 spawned at %2 heading %3 deg (pose from %4): slot %5, entity %6; settling %7 s", m_sPrefab, DC_RigR1.FormatVector(m_Ego.GetOrigin(), 2), headingDeg.ToString(-1, 2), poseSource, m_iSlotId, m_Registry.GetEntityId(m_iSlotId), m_fSettleS.ToString(-1, 1)), LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! GetGame().SpawnEntityPrefab at (x, GetSurfaceY + lift, z) facing headingDeg (rows right, up, forward as
	//! DC_ProbeActuator.Teleport)
	protected bool SpawnEgo(vector position, float headingDeg, out string failure)
	{
		Resource resource = Resource.Load(m_sEgoPrefab);
		if (!resource || !resource.IsValid())
		{
			failure = "prefab " + m_sEgoPrefab + " cannot be loaded";
			return false;
		}

		BaseWorld world = GetGame().GetWorld();
		float rad = headingDeg * Math.DEG2RAD;
		float sinH = Math.Sin(rad);
		float cosH = Math.Cos(rad);
		EntitySpawnParams params = new EntitySpawnParams();
		params.TransformMode = ETransformMode.WORLD;
		params.Transform[0] = Vector(cosH, 0, -sinH);
		params.Transform[1] = Vector(0, 1, 0);
		params.Transform[2] = Vector(sinH, 0, cosH);
		params.Transform[3] = Vector(position[0], world.GetSurfaceY(position[0], position[2]) + m_fSpawnLiftM, position[2]);

		m_Ego = GetGame().SpawnEntityPrefab(resource, world, params);
		if (!m_Ego)
		{
			failure = "SpawnEntityPrefab(" + m_sEgoPrefab + ") returned null";
			return false;
		}

		m_sPrefab = DC_VehicleRegistry.PrefabOf(m_Ego);
		if (m_sPrefab.IsEmpty())
			m_sPrefab = m_sEgoPrefab;

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Ready after the settle with the ego alive and slow
	override bool IsSessionReady(out string waitingFor)
	{
		if (m_Session.GetFaultCase() == DC_Session.FAULT_READINESS_TIMEOUT)
		{
			waitingFor = "fault case readiness_timeout (never ready)";
			return false;
		}

		string failure;
		if (!IsEgoAlive(failure))
		{
			m_Session.RefuseReadiness("ego lost during the settle: " + failure);
			waitingFor = "the ego";
			return false;
		}

		// Mode A: the park is applied again on every poll (no physics step is dispatched before the open)
		m_Actuator.Park();
		float settledS = (GetGame().GetWorld().GetWorldTime() - m_fSpawnWorldMs) / 1000;
		if (settledS < m_fSettleS)
		{
			waitingFor = string.Format("the settle (%1 of %2 s)", settledS.ToString(-1, 1), m_fSettleS.ToString(-1, 1));
			return false;
		}

		float speedKmh = m_Actuator.GetSimulation().GetSpeedKmh();
		if (speedKmh > m_fSettleMaxKmh)
		{
			waitingFor = string.Format("the ego to stop (%1 km/h)", speedKmh.ToString(-1, 2));
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void WriteSessionJson(string section, notnull DC_JsonWriter json)
	{
		if (section == "rig")
			json.AddString("ego_prefab", m_sPrefab);
	}

	//------------------------------------------------------------------------------------------------
	//! The episodes stream (owned by the session)
	override bool OnSessionOpen(notnull DC_Session session)
	{
		if (session.GetFaultCase() == DC_Session.FAULT_ZERO_EGO)
		{
			m_bFaultEgoLost = true;
			Print(LOG + "WARNING: fault case zero_ego: the ego counts as lost from the first check on", LogLevel.WARNING);
		}

		array<string> columns = {};
		AddEpisodeColumns(columns);
		// The session keeps the only strong reference (RegisterStream); a new object assigned straight to the weak
		// member would be released at once
		DC_ChunkWriter writer = new DC_ChunkWriter();
		bool opened = writer.Open(session.GetSessionDir(), "episodes", columns, EPISODES_CHUNK_ROWS, session.GetChunkMaxMs(), session.GetKeepLogsOpen(), session.GetVerifyChunkLength());
		session.RegisterStream(writer);
		m_Episodes = writer;
		if (opened)
			return true;

		session.FailWrite("episodes: " + writer.GetFailure());
		return false;
	}

	//------------------------------------------------------------------------------------------------
	override void OnFrameStart(int renderFrame)
	{
		CheckEgo();
	}

	//------------------------------------------------------------------------------------------------
	//! Liveness and identity first; then an episode may begin at a decision step
	override void OnStepPre(int physicsStep, float timeSlice)
	{
		if (!CheckEgo())
			return;

		if (m_iEpisodeId >= 0 || physicsStep % m_Session.GetStepsPerDecision() != 0)
			return;

		if (m_Session.IsWarmupOver() && !m_Session.IsStopping())
			TryBeginEpisode(physicsStep);
	}

	//------------------------------------------------------------------------------------------------
	//! Liveness and identity again after the simulation, before any row of the step (order 10: before the streams)
	override void OnStepPost(int physicsStep, float timeSlice, int worldMs)
	{
		if (!CheckEgo())
			return;

		m_iLastWorldMs = worldMs;
		if (m_iEpisodeId >= 0 && physicsStep == m_iEpisodeStartStep)
			m_iEpisodeStartWorldMs = worldMs;
	}

	//------------------------------------------------------------------------------------------------
	//! At a decision step, after its rows: the planned end or the driver's termination ends the episode here
	override void OnStepDecide(int physicsStep)
	{
		if (m_iEpisodeId < 0 || physicsStep % m_Session.GetStepsPerDecision() != 0)
			return;

		// The first terminal event wins: the reason latched when the planned end began, else the driver's
		string reason = m_sStopTermination;
		if (reason.IsEmpty())
			reason = m_Recorder.GetTermination();

		if (!reason.IsEmpty())
			EndEpisode(physicsStep, reason);
	}

	//------------------------------------------------------------------------------------------------
	//! The planned end begins: the episode's terminal reason is latched now (a termination the driver already
	//! reported, else session_stop), so a later driver report cannot replace the earlier event
	override void OnSessionStopping(string endReason)
	{
		if (m_iEpisodeId < 0 || !m_sStopTermination.IsEmpty())
			return;

		m_sStopTermination = m_Recorder.GetTermination();
		if (m_sStopTermination.IsEmpty())
			m_sStopTermination = TERMINATION_SESSION_STOP;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsDrained()
	{
		return m_iEpisodeId < 0;
	}

	//------------------------------------------------------------------------------------------------
	//! An episode still open at the seal ends at the last step: session_stop, or script_error after an error
	override void OnSessionClosing(string endReason)
	{
		if (m_iEpisodeId < 0)
			return;

		// The first terminal event wins: the reason latched at the planned end, else a termination the driver already
		// reported, else the session's
		string reason = m_sStopTermination;
		if (reason.IsEmpty())
			reason = m_Recorder.GetTermination();

		if (reason.IsEmpty())
		{
			reason = TERMINATION_SESSION_STOP;
			if (DC_Session.IsErrorEndReason(endReason))
				reason = TERMINATION_SCRIPT_ERROR;
		}

		EndEpisode(m_Session.GetLastPostStep(), reason);
	}

	//------------------------------------------------------------------------------------------------
	//! The ego is parked and kept; on teardown only references are dropped
	override void OnSessionRelease(bool teardown)
	{
		string failure;
		if (!teardown && m_Actuator && IsEgoAlive(failure))
			m_Actuator.Park();

		m_Actuator = null;
		m_Recorder = null;
		m_Episodes = null;
		m_Damage = null;
		m_Ego = null;
	}

	//------------------------------------------------------------------------------------------------
	//! Fails the session when the ego is lost: zero_ego before the first ctl row (DC_StreamRecorder.HasCtlRow), ego_lost
	//! after it
	protected bool CheckEgo()
	{
		string failure;
		if (IsEgoAlive(failure))
			return true;

		string reason = DC_Session.END_EGO_LOST;
		if (!m_Recorder || !m_Recorder.HasCtlRow())
			reason = DC_Session.END_ZERO_EGO;

		m_Session.Fail(reason, failure);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! The ego exists, is not deleted or destroyed and still carries its registered GetID token (C35)
	protected bool IsEgoAlive(out string failure)
	{
		failure = string.Empty;
		if (m_bFaultEgoLost)
		{
			failure = "fault case zero_ego";
			return false;
		}

		if (!m_Ego || m_Ego.IsDeleted())
		{
			failure = "the ego entity is gone";
			return false;
		}

		if (m_Damage && m_Damage.IsDestroyed())
		{
			failure = "the ego is destroyed";
			return false;
		}

		return m_Registry.CheckIdentity(m_iSlotId, m_Ego, failure);
	}

	//------------------------------------------------------------------------------------------------
	//! Asks the driver for an episode starting at this decision step
	protected void TryBeginEpisode(int physicsStep)
	{
		DC_DriverInterface driver = m_Recorder.GetDriver();
		if (!driver)
			return;

		DC_EpisodeInfo info = new DC_EpisodeInfo();
		info.m_iSeed = m_Session.GetTrialSeed() + m_iNextEpisodeId;
		if (!driver.BeginEpisode(m_iNextEpisodeId, physicsStep, info))
			return;

		if (s_aEpisodeModes.Find(info.m_sMode) < 0 || s_aDrivers.Find(info.m_sDriver) < 0)
		{
			m_Session.Fail(DC_Session.END_SCRIPT_ERROR, "driver episode mode '" + info.m_sMode + "' / driver '" + info.m_sDriver + "' cannot be recorded (modes E, D, H, A; drivers expert, policy, human, ai, none)");
			return;
		}

		if (info.m_sVehicleCardId.IsEmpty())
			info.m_sVehicleCardId = "none";

		m_EpisodeInfo = info;
		m_iEpisodeId = m_iNextEpisodeId;
		m_iNextEpisodeId++;
		m_iEpisodeStartStep = physicsStep;
		m_iEpisodeStartWorldMs = m_Session.GetWorldMs();
		ReadEpisodeEnvironment();
		m_Session.SetEpisodeId(m_iEpisodeId);
		m_Recorder.SetEpisode(info);
		m_Session.LogEvent("episode_start", "route_id=" + DC_ProbeCsv.CleanField(info.m_sRouteId) + ";mode=" + info.m_sMode);
	}

	//------------------------------------------------------------------------------------------------
	//! Weather and time at the episode start (fixed within the episode, dc-2.0 9.6)
	protected void ReadEpisodeEnvironment()
	{
		m_sEpisodeWeather = "unknown";
		ChimeraWorld world = ChimeraWorld.CastFrom(GetGame().GetWorld());
		TimeAndWeatherManagerEntity weather;
		if (world)
			weather = world.GetTimeAndWeatherManager();

		if (!weather)
			return;

		WeatherState state = weather.GetCurrentWeatherState();
		if (state)
			m_sEpisodeWeather = DC_ProbeCsv.CleanField(state.GetStateName());

		m_fEpisodeTodH = weather.GetTimeOfTheDay();
		m_fEpisodeRain = weather.GetRainIntensity();
		m_fEpisodeFog = weather.GetFogAmount();
		m_fEpisodeWetness = weather.GetCurrentWetness();
	}

	//------------------------------------------------------------------------------------------------
	//! Driver notified, events episode_end, the episodes row, the episode cleared, the ego parked
	protected void EndEpisode(int lastPhysicsStep, string terminationReason)
	{
		DC_DriverInterface driver = m_Recorder.GetDriver();
		if (driver)
			driver.EndEpisode(m_iEpisodeId, lastPhysicsStep, terminationReason);

		m_Session.LogEvent("episode_end", "termination_reason=" + terminationReason);
		WriteEpisodeRow(lastPhysicsStep, terminationReason);
		Print(string.Format(LOG + "episode %1 ended at physics step %2: %3", m_iEpisodeId, lastPhysicsStep, terminationReason), LogLevel.NORMAL);

		m_iEpisodeId = -1;
		m_EpisodeInfo = null;
		m_sStopTermination = string.Empty;
		m_Session.SetEpisodeId(-1);
		m_Recorder.ClearEpisode();
		string failure;
		if (IsEgoAlive(failure))
			m_Actuator.Park();
	}

	//------------------------------------------------------------------------------------------------
	//! One episodes row (dc-2.0 9.6)
	protected void WriteEpisodeRow(int lastPhysicsStep, string terminationReason)
	{
		DC_EpisodeInfo info = m_EpisodeInfo;
		VehicleWheeledSimulation simulation;
		string failure;
		if (m_Actuator && IsEgoAlive(failure))
			simulation = m_Actuator.GetSimulation();

		m_Row.Clear();
		m_Row.AddInt(m_iEpisodeId);
		m_Row.AddInt(m_iEpisodeStartStep);
		m_Row.AddInt(lastPhysicsStep);
		m_Row.AddInt(m_iEpisodeStartWorldMs);
		m_Row.AddInt(m_iLastWorldMs);
		m_Row.AddInt(m_iSlotId);
		m_Row.AddString(DC_ProbeCsv.CleanField(m_Registry.GetEntityId(m_iSlotId)));
		m_Row.AddString(DC_ProbeCsv.CleanField(m_sPrefab));
		m_Row.AddString(DC_ProbeCsv.CleanField(info.m_sVehicleCardId));
		m_Row.AddString(DC_ProbeCsv.CleanField(info.m_sRouteId));
		m_Row.AddString(DC_ProbeCsv.CleanField(info.m_sRouteSha256));
		m_Row.AddString(info.m_sMode);
		m_Row.AddString(info.m_sDriver);
		m_Row.AddFloat(info.m_fStyleSpeedFactor, 4);
		m_Row.AddFloat(info.m_fStyleLaneOffsetM, 4);
		m_Row.AddFloat(info.m_fStyleAlatMaxMps2, 4);
		m_Row.AddString(m_sEpisodeWeather);
		m_Row.AddFloat(m_fEpisodeTodH, 6);
		m_Row.AddFloat(m_fEpisodeRain, 6);
		m_Row.AddFloat(m_fEpisodeFog, 6);
		m_Row.AddFloat(m_fEpisodeWetness, 6);
		m_Row.AddInt(info.m_iSeed);
		m_Row.AddString(terminationReason);
		m_Row.AddString(DC_ProbeCsv.CleanField(info.m_sExpertSha));
		m_Row.AddString(DC_ProbeCsv.CleanField(info.m_sLlcSha));
		m_Row.AddString(DC_Session.NAV_VERSION);
		if (simulation)
		{
			m_Row.AddFloat(simulation.GetNoiseSteerSensitivity(), 6);
			m_Row.AddFloat(simulation.GetRoughnessSensitivity(), 6);
		}
		else
		{
			m_Row.AddFloat(-1, 6);
			m_Row.AddFloat(-1, 6);
		}

		m_Row.AddEmpty();		// veh_dirt_body: reserved
		m_Row.AddEmpty();		// veh_dirt_glass: reserved
		if (m_Episodes && !m_Session.HasWriteFailure() && !m_Episodes.AddRow(m_Row, m_iEpisodeId, m_iLastWorldMs))
			m_Session.FailWrite("episodes: " + m_Episodes.GetFailure());
	}

	//------------------------------------------------------------------------------------------------
	//! Every column of episodes in dc-2.0 9.6 order
	static void AddEpisodeColumns(notnull array<string> columns)
	{
		array<string> names = {"episode_id", "start_physics_step", "end_physics_step", "start_world_ms", "end_world_ms", "ego_slot_id", "ego_entity_id", "ego_prefab", "vehicle_card_id", "route_id", "route_sha256", "mode", "driver", "style_speed_factor", "style_lane_offset_m", "style_alat_max_mps2", "env_weather_state", "env_tod_h_start", "env_rain", "env_fog", "env_wetness", "seed", "termination_reason", "expert_sha", "llc_sha", "nav_version", "noise_steer_sensitivity", "roughness_sensitivity", "veh_dirt_body", "veh_dirt_glass"};
		foreach (string name : names)
		{
			columns.Insert(name);
		}
	}
}
